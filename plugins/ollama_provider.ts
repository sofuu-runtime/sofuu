// plugins/ollama_provider.ts - local Ollama inference adapter (cookbook plugin).
//
// Demonstrates Category 2 (defineProvider): a streaming /api/chat bridge with
// tool-call passthrough and a health probe, written against ctx.fetch only.
//
// Wire notes (Ollama API):
//   POST /api/chat  -> one JSON object when stream is false, NDJSON when true
//   GET  /api/tags  -> installed models, used by the health probe
//   tool calls ride in message.tool_calls[].function.{name,arguments}
//
// Endpoint: OLLAMA_HOST (default http://127.0.0.1:11434). The CLI already
// ships a native local wire (sofuu -p ollama -m llama3.2); this file is the
// template you copy when you need custom wire handling or a different host.
//
// Standalone: no import beyond the SDK primitives.

import { defineProvider } from 'sofuu';
import type { ProviderContext, ProviderEvent, ProviderHealth, ProviderRequest, ProviderResult, ProviderToolCall, ProviderToolDef, ProviderUsage } from 'sofuu';

// -- local types (this block is stripped before the runtime sees the file) -

interface OllamaFunctionCall {
  name: string;
  arguments: Record<string, unknown>;
}

interface OllamaToolCall {
  function: OllamaFunctionCall;
}

interface OllamaMessage {
  role: string;
  content: string;
  tool_calls?: OllamaToolCall[];
}

interface OllamaChatChunk {
  model?: string;
  message?: OllamaMessage;
  done?: boolean;
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
}

interface OllamaTagEntry {
  name?: string;
  model?: string;
}

interface OllamaTags {
  models?: OllamaTagEntry[];
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

// Named aliases keep every annotation a single identifier: that is what the
// built-in TypeScript stripper can remove without a compiler.

type WireRecord = Record<string, unknown>;
type WireList = WireRecord[];
type RawToolArguments = string | Record<string, unknown>;
type MaybeToolCalls = ProviderToolCall[] | undefined;
type MaybeToolDefs = ProviderToolDef[] | undefined;
type MaybeChunk = OllamaChatChunk | undefined;
type MaybeMessage = OllamaMessage | undefined;
type MaybeUsage = ProviderUsage | undefined;
type MaybeDecoder = DecoderLike | null;
type MaybeChunkOrNull = OllamaChatChunk | null;
type LineSink = (line: string) => void;

// -- endpoint --------------------------------------------------------------

var DEFAULT_MODEL = 'llama3.2';

function readOllamaHost(): string {
  var fallback = 'http://127.0.0.1:11434';
  if (typeof process === 'undefined' || !process.env) return fallback;
  var host = process.env.OLLAMA_HOST;
  if (!host) return fallback;
  return String(host).replace(/\/+$/, '');
}

var BASE_URL = readOllamaHost();

function endpoint(suffix: string): string {
  return BASE_URL + suffix;
}

function fail(code: string, message: string): PluginError {
  var err = new Error(message) as PluginError;
  err.code = code;
  return err;
}

// -- request shaping -------------------------------------------------------

// Ollama wants arguments as an object; a model that emits malformed JSON
// gets an empty call rather than a crashed turn.
function parseArguments(raw: RawToolArguments): WireRecord {
  if (raw && typeof raw === 'object') return raw;
  var text = String(raw || '{}').trim();
  if (!text) return {};
  try {
    var parsed = JSON.parse(text);
    if (parsed && typeof parsed === 'object') return parsed;
  } catch (e) { /* empty call */ }
  return {};
}

function toWireToolCalls(calls: MaybeToolCalls): WireList {
  var out: WireList = [];
  if (!calls || !calls.length) return out;
  for (var i = 0; i < calls.length; i++) {
    var call = calls[i];
    var fn: WireRecord = {};
    fn.name = call.name;
    fn.arguments = parseArguments(call.arguments);
    var wrapper: WireRecord = {};
    wrapper.function = fn;
    out.push(wrapper);
  }
  return out;
}

function toWireMessages(req: ProviderRequest): WireList {
  var out: WireList = [];
  var messages = req.messages || [];
  for (var i = 0; i < messages.length; i++) {
    var message = messages[i];
    var wire: WireRecord = {};
    wire.role = message.role;
    wire.content = (typeof message.content === 'string') ? message.content : String(message.content || '');
    if (message.role === 'tool' && message.name) wire.tool_name = message.name;
    var calls = toWireToolCalls(message.toolCalls);
    if (calls.length) wire.tool_calls = calls;
    out.push(wire);
  }
  return out;
}

function toOllamaTools(tools: MaybeToolDefs): WireList {
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

function buildPayload(req: ProviderRequest, stream: boolean): WireRecord {
  var payload: WireRecord = {};
  payload.model = req.model || DEFAULT_MODEL;
  payload.messages = toWireMessages(req);
  payload.stream = stream;
  var tools = toOllamaTools(req.tools);
  if (tools.length) payload.tools = tools;
  var options: WireRecord = {};
  if (typeof req.temperature === 'number') options.temperature = req.temperature;
  if (typeof req.maxTokens === 'number') options.num_predict = req.maxTokens;
  if (Object.keys(options).length) payload.options = options;
  return payload;
}

// -- response shaping ------------------------------------------------------

function readUsage(chunk: MaybeChunk): MaybeUsage {
  if (!chunk) return undefined;
  var promptTokens = (typeof chunk.prompt_eval_count === 'number') ? chunk.prompt_eval_count : 0;
  var completionTokens = (typeof chunk.eval_count === 'number') ? chunk.eval_count : 0;
  if (!promptTokens && !completionTokens) return undefined;
  var usage: ProviderUsage = { promptTokens: 0, completionTokens: 0 };
  usage.promptTokens = promptTokens;
  usage.completionTokens = completionTokens;
  usage.totalTokens = promptTokens + completionTokens;
  return usage;
}

function readToolCalls(message: MaybeMessage): ProviderToolCall[] {
  var out: ProviderToolCall[] = [];
  if (!message || !message.tool_calls) return out;
  var calls = message.tool_calls;
  for (var i = 0; i < calls.length; i++) {
    var call = calls[i];
    if (!call || !call.function || !call.function.name) continue;
    var args = call.function.arguments || {};
    var result: ProviderToolCall = { name: call.function.name, arguments: JSON.stringify(args) };
    out.push(result);
  }
  return out;
}

function readResult(chunk: MaybeChunk): ProviderResult {
  var result: ProviderResult = { text: '' };
  if (!chunk) return result;
  var message = chunk.message;
  if (message && typeof message.content === 'string') result.text = message.content;
  var calls = readToolCalls(message);
  if (calls.length) result.toolCalls = calls;
  var usage = readUsage(chunk);
  if (usage) result.usage = usage;
  if (chunk.done_reason) result.finishReason = chunk.done_reason;
  if (chunk.model) result.model = chunk.model;
  return result;
}

// -- transport -------------------------------------------------------------

function postChat(payload: WireRecord, ctx: ProviderContext): Promise<Response> {
  var init: RequestInit = {};
  init.method = 'POST';
  init.headers = { 'content-type': 'application/json' };
  init.body = JSON.stringify(payload);
  if (ctx.signal) init.signal = ctx.signal;
  return ctx.fetch(endpoint('/api/chat'), init).then(function (res) {
    if (!res || !res.ok) {
      throw fail('E_TOOL', 'ollama /api/chat returned HTTP ' + (res ? res.status : 0));
    }
    return res;
  });
}

function decodePiece(bytes: Uint8Array, decoder: MaybeDecoder): string {
  if (decoder) {
    try { return decoder.decode(bytes, { stream: true }); } catch (e) { /* latin-1 below */ }
  }
  var out = '';
  for (var i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
  return out;
}

function makeDecoder(): MaybeDecoder {
  if (typeof TextDecoder !== 'function') return null;
  var decoder = new TextDecoder('utf-8') as DecoderLike;
  return decoder;
}

function consumeWithReader(reader: ReadableStreamDefaultReader<Uint8Array>, onLine: LineSink): Promise<void> {
  var buffer = '';
  var decoder = makeDecoder();
  function pump(): Promise<void> {
    return reader.read().then(function (step) {
      if (!step || step.done) {
        var tail = buffer.trim();
        if (tail) onLine(tail);
        return;
      }
      var piece = step.value;
      buffer += (typeof piece === 'string') ? piece : decodePiece(piece, decoder);
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

// NDJSON when the body is a stream, one-shot split otherwise. A runtime
// without ReadableStream still works: the whole body is read then split.
function consumeNdjson(res: Response, onLine: LineSink): Promise<void> {
  var body = res ? res.body : null;
  if (body && typeof body.getReader === 'function') {
    return consumeWithReader(body.getReader(), onLine);
  }
  return res.text().then(function (text) {
    var lines = String(text || '').split('\n');
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (line) onLine(line);
    }
  });
}

// -- health probe ----------------------------------------------------------

function probeTags(): Promise<ProviderHealth> {
  var startedAt = Date.now();
  if (typeof fetch !== 'function') {
    var missing: ProviderHealth = { ok: false };
    missing.error = 'fetch is not available in this runtime';
    return Promise.resolve(missing);
  }
  return fetch(endpoint('/api/tags')).then(function (res) {
    if (!res || !res.ok) {
      var unreachable: ProviderHealth = { ok: false };
      unreachable.error = 'HTTP ' + (res ? res.status : 0) + ' from /api/tags at ' + BASE_URL;
      return unreachable;
    }
    return res.json().then(function (data) {
      var tags = data as OllamaTags;
      var models: string[] = [];
      var list = (tags && tags.models) ? tags.models : [];
      for (var i = 0; i < list.length; i++) {
        var name = list[i].name || list[i].model;
        if (name) models.push(String(name));
      }
      var healthy: ProviderHealth = { ok: true };
      healthy.models = models;
      healthy.latencyMs = Date.now() - startedAt;
      return healthy;
    });
  }).then(function (health) {
    return health;
  }, function () {
    var offline: ProviderHealth = { ok: false };
    offline.error = 'ollama is not reachable at ' + BASE_URL;
    return offline;
  });
}

// -- the provider ----------------------------------------------------------

export default defineProvider({
  name: 'ollama',
  kind: 'ollama',
  /* A call and a literal, not the module constants: the runtime stripper
     rewrites ", key: IDENT" into ", key", which would leave an undefined
     shorthand binding here. */
  baseUrl: readOllamaHost(),
  model: 'llama3.2',
  description: 'Local Ollama bridge (/api/chat) with streaming and tool-call passthrough',
  async complete(req, ctx): Promise<ProviderResult> {
    var payload = buildPayload(req, false);
    var res = await postChat(payload, ctx);
    var chunk = await res.json() as OllamaChatChunk;
    return readResult(chunk);
  },
  async stream(req, ctx, onEvent): Promise<ProviderResult> {
    var payload = buildPayload(req, true);
    var acc: ProviderResult = { text: '' };
    var res = await postChat(payload, ctx);
    await consumeNdjson(res, function (line: string) {
      var chunk: MaybeChunkOrNull = null;
      try {
        chunk = JSON.parse(line) as OllamaChatChunk;
      } catch (e) {
        return;
      }
      if (!chunk) return;
      var message = chunk.message;
      if (message && typeof message.content === 'string' && message.content) {
        acc.text = acc.text + message.content;
        var textEvent: ProviderEvent = { type: 'text' };
        textEvent.text = message.content;
        onEvent(textEvent);
      }
      var calls = readToolCalls(message);
      if (calls.length) {
        if (!acc.toolCalls) acc.toolCalls = [];
        for (var i = 0; i < calls.length; i++) {
          acc.toolCalls.push(calls[i]);
          var callEvent: ProviderEvent = { type: 'tool_call' };
          callEvent.toolCall = calls[i];
          onEvent(callEvent);
        }
      }
      if (chunk.done) {
        var usage = readUsage(chunk);
        if (usage) {
          acc.usage = usage;
          var usageEvent: ProviderEvent = { type: 'usage' };
          usageEvent.usage = usage;
          onEvent(usageEvent);
        }
        if (chunk.model) acc.model = chunk.model;
        if (chunk.done_reason) acc.finishReason = chunk.done_reason;
      }
    });
    var doneEvent: ProviderEvent = { type: 'done' };
    onEvent(doneEvent);
    return acc;
  },
  health(): Promise<ProviderHealth> {
    return probeTags();
  }
});
