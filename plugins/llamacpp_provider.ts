// plugins/llamacpp_provider.ts - production llama.cpp inference adapter.
//
// Demonstrates Category 2 (defineProvider): an industrial-strength local
// inference bridge connecting sofuu.ai.streamLocal() and agent loops to
// a high-throughput llama.cpp server instance with full hardware acceleration
// (Apple Metal, CUDA, Vulkan) and comprehensive quantization coverage (Q4_K_M,
// Q5_K_M, Q8_0, IQ3, etc.).
//
// Wire notes (llama.cpp server API):
//   POST /v1/chat/completions -> OpenAI-compatible SSE streaming or single JSON
//   POST /completion          -> Native raw prompt completion/streaming
//   GET  /health              -> Server status & readiness
//   GET  /v1/models           -> Loaded model & slot inspection
//   tool calls ride in choices[].delta.tool_calls[].function.{name,arguments}
//
// Endpoint: LLAMACPP_HOST (default http://127.0.0.1:8080).
//
// Standalone: no external dependencies beyond the Sofuu runtime SDK.

import { defineProvider } from 'sofuu';
import type { ProviderContext, ProviderEvent, ProviderHealth, ProviderRequest, ProviderResult, ProviderToolCall, ProviderToolDef, ProviderUsage } from 'sofuu';

// -- local types for single-identifier lexical stripping --------------------

interface LlamaFunctionCall {
  name: string;
  arguments: string;
}

interface LlamaToolCall {
  id?: string;
  type?: string;
  function: LlamaFunctionCall;
}

interface LlamaDelta {
  role?: string;
  content?: string;
  tool_calls?: LlamaToolCall[];
}

interface LlamaChoice {
  index?: number;
  delta?: LlamaDelta;
  message?: LlamaDelta;
  finish_reason?: string;
}

interface LlamaUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

interface LlamaChatChunk {
  id?: string;
  model?: string;
  choices?: LlamaChoice[];
  usage?: LlamaUsage;
}

interface LlamaModelEntry {
  id?: string;
  name?: string;
}

interface LlamaModelList {
  data?: LlamaModelEntry[];
}

interface DecoderLike {
  decode(bytes: Uint8Array, options?: DecodeOptions): string;
}

interface DecodeOptions {
  stream: boolean;
}

interface PluginError extends Error {
  code: string;
}

type WireRecord = Record<string, unknown>;
type WireList = WireRecord[];
type MaybeToolDefs = ProviderToolDef[] | undefined;
type MaybeChunk = LlamaChatChunk | undefined;
type MaybeUsage = ProviderUsage | undefined;
type MaybeDecoder = DecoderLike | null;
type MaybeChunkOrNull = LlamaChatChunk | null;
type LineSink = (line: string) => void;

// -- endpoint configuration ------------------------------------------------

var LCPP_DEFAULT_MODEL = 'default';

function lcppReadHost(): string {
  var fallback = 'http://127.0.0.1:8080';
  if (typeof process === 'undefined' || !process.env) return fallback;
  var host = process.env.LLAMACPP_HOST;
  if (!host) return fallback;
  return String(host).replace(/\/+$/, '');
}

var LCPP_BASE_URL = lcppReadHost();

function lcppEndpoint(suffix: string): string {
  return LCPP_BASE_URL + suffix;
}

function lcppFail(code: string, message: string): PluginError {
  var err = new Error(message) as PluginError;
  err.code = code;
  return err;
}

// -- request payload shaping -----------------------------------------------

function lcppToWireMessages(req: ProviderRequest): WireList {
  var out: WireList = [];
  if (req.system) {
    var sysMsg: WireRecord = {};
    sysMsg.role = 'system';
    sysMsg.content = req.system;
    out.push(sysMsg);
  }
  if (req.messages && req.messages.length) {
    for (var i = 0; i < req.messages.length; i++) {
      var m = req.messages[i];
      var wMsg: WireRecord = {};
      wMsg.role = m.role || 'user';
      wMsg.content = m.content || '';
      if (m.name) wMsg.name = m.name;
      out.push(wMsg);
    }
  } else if (req.prompt) {
    var userMsg: WireRecord = {};
    userMsg.role = 'user';
    userMsg.content = req.prompt;
    out.push(userMsg);
  }
  return out;
}

function lcppToTools(tools: MaybeToolDefs): WireList {
  var out: WireList = [];
  if (!tools || !tools.length) return out;
  for (var i = 0; i < tools.length; i++) {
    var def = tools[i];
    var fn: WireRecord = {};
    fn.name = def.name;
    if (def.description) fn.description = def.description;
    if (def.parameters) fn.parameters = def.parameters;
    var wrapper: WireRecord = {};
    wrapper.type = 'function';
    wrapper.function = fn;
    out.push(wrapper);
  }
  return out;
}

function lcppBuildPayload(req: ProviderRequest, stream: boolean): WireRecord {
  var payload: WireRecord = {};
  payload.model = req.model || LCPP_DEFAULT_MODEL;
  payload.messages = lcppToWireMessages(req);
  payload.stream = stream;
  if (typeof req.temperature === 'number') payload.temperature = req.temperature;
  if (typeof req.maxTokens === 'number') payload.max_tokens = req.maxTokens;
  if (typeof req.topP === 'number') payload.top_p = req.topP;
  var tools = lcppToTools(req.tools);
  if (tools.length) payload.tools = tools;
  return payload;
}

// -- response shaping ------------------------------------------------------

function lcppReadUsage(chunk: MaybeChunk): MaybeUsage {
  if (!chunk || !chunk.usage) return undefined;
  var u = chunk.usage;
  var promptTokens = (typeof u.prompt_tokens === 'number') ? u.prompt_tokens : 0;
  var completionTokens = (typeof u.completion_tokens === 'number') ? u.completion_tokens : 0;
  var totalTokens = (typeof u.total_tokens === 'number') ? u.total_tokens : (promptTokens + completionTokens);
  var usage: ProviderUsage = { promptTokens: 0, completionTokens: 0 };
  usage.promptTokens = promptTokens;
  usage.completionTokens = completionTokens;
  usage.totalTokens = totalTokens;
  return usage;
}

function lcppReadToolCalls(delta: LlamaDelta | undefined): ProviderToolCall[] {
  var out: ProviderToolCall[] = [];
  if (!delta || !delta.tool_calls) return out;
  var calls = delta.tool_calls;
  for (var i = 0; i < calls.length; i++) {
    var call = calls[i];
    if (!call || !call.function || !call.function.name) continue;
    var rawArgs = call.function.arguments || '{}';
    var parsedArgs = typeof rawArgs === 'string' ? rawArgs : JSON.stringify(rawArgs);
    var result: ProviderToolCall = { name: call.function.name, arguments: parsedArgs };
    if (call.id) result.id = call.id;
    out.push(result);
  }
  return out;
}

function lcppReadResult(chunk: MaybeChunk): ProviderResult {
  var result: ProviderResult = { text: '' };
  if (!chunk) return result;
  if (chunk.choices && chunk.choices.length) {
    var choice = chunk.choices[0];
    var msg = choice.message || choice.delta;
    if (msg && typeof msg.content === 'string') result.text = msg.content;
    var calls = lcppReadToolCalls(msg);
    if (calls.length) result.toolCalls = calls;
    if (choice.finish_reason) result.finishReason = choice.finish_reason;
  }
  var usage = lcppReadUsage(chunk);
  if (usage) result.usage = usage;
  if (chunk.model) result.model = chunk.model;
  return result;
}

// -- transport & SSE stream consumption ------------------------------------

function lcppPostChat(payload: WireRecord, ctx: ProviderContext): Promise<Response> {
  var init: RequestInit = {};
  init.method = 'POST';
  var headers: Record<string, string> = { 'content-type': 'application/json' };
  if (ctx.apiKey) headers['authorization'] = 'Bearer ' + ctx.apiKey;
  init.headers = headers;
  init.body = JSON.stringify(payload);
  if (ctx.signal) init.signal = ctx.signal;
  var base = ctx.baseUrl ? String(ctx.baseUrl).replace(/\/+$/, '') : LCPP_BASE_URL;
  var endpoint = base.endsWith('/v1/chat/completions')
    ? base
    : (base.endsWith('/v1') ? base + '/chat/completions' : base + '/v1/chat/completions');
  return ctx.fetch(endpoint, init).then(function (res) {
    if (!res || !res.ok) {
      throw lcppFail('E_TOOL', 'llama.cpp /v1/chat/completions returned HTTP ' + (res ? res.status : 0));
    }
    return res;
  });
}

function lcppDecodePiece(bytes: Uint8Array, decoder: MaybeDecoder): string {
  if (decoder) {
    try { return decoder.decode(bytes, { stream: true }); } catch (e) { /* fallback below */ }
  }
  var out = '';
  for (var i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
  return out;
}

function lcppMakeDecoder(): MaybeDecoder {
  if (typeof TextDecoder !== 'function') return null;
  var decoder = new TextDecoder('utf-8') as DecoderLike;
  return decoder;
}

function lcppConsumeWithReader(reader: ReadableStreamDefaultReader<Uint8Array>, onLine: LineSink): Promise<void> {
  var buffer = '';
  var decoder = lcppMakeDecoder();
  function pump(): Promise<void> {
    return reader.read().then(function (step) {
      if (!step || step.done) {
        var tail = buffer.trim();
        if (tail) onLine(tail);
        return;
      }
      var piece = step.value;
      buffer += (typeof piece === 'string') ? piece : lcppDecodePiece(piece, decoder);
      var cut = buffer.indexOf('\n');
      while (cut >= 0) {
        var line = buffer.slice(0, cut).trim();
        buffer = buffer.slice(cut + 1);
        if (line) onLine(line);
        cut = buffer.indexOf('\n');
      }
      return pump();
    });
  }
  return pump();
}

function lcppConsumeSse(res: Response, onLine: LineSink): Promise<void> {
  var body = res ? res.body : null;
  if (body && typeof body.getReader === 'function') {
    return lcppConsumeWithReader(body.getReader(), onLine);
  }
  return res.text().then(function (text) {
    var lines = String(text || '').split('\n');
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (line) onLine(line);
    }
  });
}

// -- health & model probe --------------------------------------------------

function lcppProbeHealth(): Promise<ProviderHealth> {
  var startedAt = Date.now();
  if (typeof fetch !== 'function') {
    var missing: ProviderHealth = { ok: false };
    missing.error = 'fetch is not available in this runtime';
    return Promise.resolve(missing);
  }
  return fetch(lcppEndpoint('/health')).then(function (res) {
    var isHealthy = res && res.ok;
    return fetch(lcppEndpoint('/v1/models')).then(function (mRes) {
      if (!mRes || !mRes.ok) {
        var basic: ProviderHealth = { ok: isHealthy };
        basic.latencyMs = Date.now() - startedAt;
        return basic;
      }
      return mRes.json().then(function (data) {
        var mList = data as LlamaModelList;
        var models: string[] = [];
        var list = (mList && mList.data) ? mList.data : [];
        for (var i = 0; i < list.length; i++) {
          var name = list[i].id || list[i].name;
          if (name) models.push(String(name));
        }
        var healthy: ProviderHealth = { ok: true };
        healthy.models = models;
        healthy.latencyMs = Date.now() - startedAt;
        return healthy;
      });
    }).catch(function () {
      var partial: ProviderHealth = { ok: isHealthy };
      partial.latencyMs = Date.now() - startedAt;
      return partial;
    });
  }).catch(function () {
    var offline: ProviderHealth = { ok: false };
    offline.error = 'llama.cpp server is not reachable at ' + LCPP_BASE_URL;
    return offline;
  });
}

// -- the provider export ---------------------------------------------------

export default defineProvider({
  name: 'llamacpp',
  kind: 'llamacpp',
  baseUrl: lcppReadHost(),
  model: 'default',
  description: 'Production llama.cpp inference bridge (/v1/chat/completions) with Metal/CUDA acceleration & comprehensive quant support',
  async complete(req, ctx): Promise<ProviderResult> {
    var payload = lcppBuildPayload(req, false);
    var res = await lcppPostChat(payload, ctx);
    var chunk = await res.json() as LlamaChatChunk;
    return lcppReadResult(chunk);
  },
  async stream(req, ctx, onEvent): Promise<ProviderResult> {
    var payload = lcppBuildPayload(req, true);
    var acc: ProviderResult = { text: '' };
    var toolCallMap: Record<number, ProviderToolCall> = {};
    var res = await lcppPostChat(payload, ctx);
    
    await lcppConsumeSse(res, function (line: string) {
      if (!line.startsWith('data:')) return;
      var raw = line.slice(5).trim();
      if (!raw || raw === '[DONE]') return;
      
      var chunk: MaybeChunkOrNull = null;
      try {
        chunk = JSON.parse(raw) as LlamaChatChunk;
      } catch (e) {
        return;
      }
      if (!chunk) return;
      
      if (chunk.choices && chunk.choices.length) {
        var choice = chunk.choices[0];
        var delta = choice.delta;
        if (delta && typeof delta.content === 'string' && delta.content) {
          acc.text = acc.text + delta.content;
          var textEvent: ProviderEvent = { type: 'delta' };
          textEvent.text = delta.content;
          onEvent(textEvent);
        }
        if (delta && delta.tool_calls) {
          for (var t = 0; t < delta.tool_calls.length; t++) {
            var tc = delta.tool_calls[t];
            var idx = typeof tc.id === 'string' ? t : 0;
            if (!toolCallMap[idx]) {
              var newCall: ProviderToolCall = {
                name: (tc.function && tc.function.name) || '',
                arguments: (tc.function && tc.function.arguments) || ''
              };
              if (tc.id) newCall.id = tc.id;
              toolCallMap[idx] = newCall;
            } else {
              if (tc.function && tc.function.name) {
                toolCallMap[idx].name = tc.function.name;
              }
              if (tc.function && tc.function.arguments) {
                toolCallMap[idx].arguments = toolCallMap[idx].arguments + tc.function.arguments;
              }
            }
          }
        }
        if (choice.finish_reason) acc.finishReason = choice.finish_reason;
      }
      
      var usage = lcppReadUsage(chunk);
      if (usage) {
        acc.usage = usage;
        var usageEvent: ProviderEvent = { type: 'usage' };
        usageEvent.usage = usage;
        onEvent(usageEvent);
      }
      if (chunk.model) acc.model = chunk.model;
    });

    // Finalize any accumulated tool calls
    var callKeys = Object.keys(toolCallMap);
    if (callKeys.length) {
      acc.toolCalls = [];
      for (var k = 0; k < callKeys.length; k++) {
        var call = toolCallMap[Number(callKeys[k])];
        acc.toolCalls.push(call);
        var callEv: ProviderEvent = { type: 'tool_call' };
        callEv.toolCall = call;
        onEvent(callEv);
      }
    }

    var doneEvent: ProviderEvent = { type: 'done' };
    onEvent(doneEvent);
    return acc;
  },
  health(): Promise<ProviderHealth> {
    return lcppProbeHealth();
  }
});
