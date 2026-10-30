// providers.js — first-party inference provider adapters (PLUGIN_PLAN Phase 2).
//
// Phase 2 of the universal plugin plan ships the built-in adapters every
// Sofuu host wants without writing a plugin file. They register into the
// plugin driver's provider registry (src/js/plugins.js) so the same
// sofuu.plugins.PROVIDERS map, circuit breaker and capability surface apply:
//
//   ollama    — native /api/chat wire (NDJSON), tool calls, /api/tags, /api/version
//   llamacpp  — OpenAI-compatible /v1/chat/completions (SSE), GBNF grammar pass-through
//   openai    — OpenAI-compatible /v1/chat/completions (SSE), API key required
//   lmstudio  — thin alias of the openai adapter, http://127.0.0.1:1234
//   vllm      — thin alias of the openai adapter, http://127.0.0.1:8000
//
// Adapter contract (what every registered object exposes):
//
//   { name, kind, baseUrl, model, description, streaming: 'buffered',
//     async complete(req, ctx) -> { text, toolCalls, usage, finishReason, model },
//     async stream(req, ctx, onEvent) -> same object + { buffered: true },
//     usage: { promptTokens, completionTokens, totalTokens, raw }  (camelCase is
//            the contract; raw keeps the counters exactly as the wire sent them)
//     async listModels(ctx) -> [{ id, name, ... }],
//     async health(ctx) -> { ok, adapter, latencyMs, error? } }
//
//   req: { model, messages:[{role,content,tool_calls?,tool_call_id?,images?}],
//          prompt?, tools?, temperature?, max_tokens?, response_format?,
//          grammar? (llama.cpp extension) }
//   ctx: { signal?, apiKey?, baseUrl?, env? }   — env defaults to process.env
//   onEvent: called with { type:'delta', text } per chunk and exactly one
//            { type:'done', usage, toolCalls, finishReason } at the end.
//
// BUFFERED REPLAY, not incremental streaming. The Sofuu QuickJS fetch is a
// buffered client: it resolves once the whole body has arrived (rt/http_client.rs
// has no ReadableStream reader). So stream() issues the request with
// stream:true, reads the complete body with res.text(), then replays the
// NDJSON / SSE frames through onEvent in order and marks its result
// buffered:true. Incremental streaming for the native providers lives in Rust
// (sofuu.ai.stream) and stays the default there; this driver is the
// extensibility path and does not invent a streaming-body API.
//
// Errors follow the normative Sofuu taxonomy:
//   E_SCHEMA | E_TIMEOUT | E_DENIED | E_CIRCUIT_OPEN | E_TOOL | E_CAPABILITY
//   | E_BACKPRESSURE

(function () {
  'use strict';

  var g = (typeof globalThis !== 'undefined') ? globalThis :
          (typeof window !== 'undefined') ? window :
          (typeof global !== 'undefined') ? global : this;

  /* The adapters register INTO the plugin driver's registry. Without
   * sofuu.plugins there is nothing to extend, so define nothing at all: this
   * driver is eval'd unconditionally by the engine seam, and a load-order slip
   * must not throw there. */
  if (typeof g.sofuu === 'undefined' || !g.sofuu || !g.sofuu.plugins) return;
  var PLUGINS = g.sofuu.plugins;
  if (PLUGINS._nativeProvidersLoaded) return;
  PLUGINS._nativeProvidersLoaded = true;

  /* Response-body ceiling. Both wire formats are replayed from one buffered
   * body, so this is the only place a runaway server could grow the JS heap
   * without bound: fail with E_BACKPRESSURE instead of buffering it. */
  var MAX_BODY = 8 * 1024 * 1024;

  /* ── Normative error construction ────────────────────────────────── */

  function makeError(code, message, details) {
    var err = new Error(message || code);
    err.code = code;
    if (details) err.details = details;
    return err;
  }

  function abortError(name) {
    return makeError('E_TIMEOUT', name + ': request aborted by the caller (ctx.signal)',
                     { adapter: name });
  }

  /* ── Small helpers ───────────────────────────────────────────────── */

  function asString(v) {
    if (v === null || v === undefined) return '';
    return String(v);
  }

  function asInt(v) {
    var n = Number(v);
    if (!isFinite(n) || n < 0) return 0;
    return Math.floor(n);
  }

  function envGet(name) {
    if (typeof process !== 'undefined' && process && process.env) {
      var v = process.env[name];
      if (typeof v === 'string' && v) return v;
    }
    return null;
  }

  function envOr(name, dflt) {
    var v = envGet(name);
    return v ? v : dflt;
  }

  function nowMs() {
    return (typeof Date !== 'undefined' && Date.now) ? Date.now() : 0;
  }

  function shortText(text, n) {
    var t = asString(text).replace(/\s+/g, ' ').trim();
    return (t.length > n) ? (t.slice(0, n) + '...') : t;
  }

  /* A non-2xx body is diagnostic only — 400 chars is enough to see the
   * server's error without dragging a whole HTML page into the message. */
  function bodyExcerpt(text) {
    return shortText(text, 400);
  }

  /* ── Usage normalisation ────────────────────────────────────────── */

  /* Every adapter reports usage as { promptTokens, completionTokens,
   * totalTokens, raw } — camelCase is the provider contract the agent loop and
   * P.completeVia read; a snake_case object silently reports zero tokens and
   * mis-bills budget accounting. Each wire spells the counters differently
   * (Ollama: prompt_eval_count/eval_count, OpenAI: prompt_tokens/
   * completion_tokens, some proxies already send camelCase), so accept all
   * three shapes here, first match wins. */

  function firstInt(obj, keys) {
    if (!obj) return null;
    for (var i = 0; i < keys.length; i++) {
      var v = obj[keys[i]];
      if (v === undefined || v === null || v === '') continue;
      var n = Number(v);
      if (isFinite(n) && n >= 0) return Math.floor(n);
    }
    return null;
  }

  /* Copy only the counter keys off a wire object: usage.raw must never drag
   * the whole response body (Ollama's top-level json carries the message). */
  function usageCounters(src) {
    var out = {};
    if (!src || typeof src !== 'object') return out;
    var keys = ['promptTokens', 'completionTokens', 'totalTokens',
                'prompt_tokens', 'completion_tokens', 'total_tokens',
                'prompt_eval_count', 'eval_count'];
    for (var i = 0; i < keys.length; i++) {
      if (src[keys[i]] !== undefined && src[keys[i]] !== null) out[keys[i]] = src[keys[i]];
    }
    return out;
  }

  function countersOf(raw) {
    var c = usageCounters(raw);
    return {
      promptTokens: firstInt(c, ['promptTokens', 'prompt_tokens', 'prompt_eval_count']),
      completionTokens: firstInt(c, ['completionTokens', 'completion_tokens', 'eval_count']),
      totalTokens: firstInt(c, ['totalTokens', 'total_tokens'])
    };
  }

  function makeUsage(raw) {
    var c = countersOf(raw);
    var usage = {
      promptTokens: (c.promptTokens === null) ? 0 : c.promptTokens,
      completionTokens: (c.completionTokens === null) ? 0 : c.completionTokens,
      totalTokens: 0,
      raw: usageCounters(raw)
    };
    usage.totalTokens = (c.totalTokens === null)
      ? (usage.promptTokens + usage.completionTokens)
      : c.totalTokens;
    return usage;
  }

  /* Streaming frames carry partial counters: merge what this frame knows and
   * recompute the total from what the stream has produced so far. */
  function mergeUsage(target, raw) {
    var c = countersOf(raw);
    if (c.promptTokens !== null) target.promptTokens = c.promptTokens;
    if (c.completionTokens !== null) target.completionTokens = c.completionTokens;
    target.totalTokens = (c.totalTokens === null)
      ? (target.promptTokens + target.completionTokens)
      : c.totalTokens;
    target.raw = usageCounters(raw);
    return target;
  }

  function parseJson(name, text, where) {
    try {
      return JSON.parse(text);
    } catch (e) {
      throw makeError('E_TOOL',
        name + ': malformed JSON (' + where + ') — ' + shortText(text, 160),
        { adapter: name, where: where });
    }
  }

  /* OLLAMA_HOST / LLAMA_CPP_HOST are conventionally host:port, but the
   * published examples also use full URLs — accept both, and never emit a
   * scheme-less URL into fetch(). */
  function normaliseBase(url) {
    var s = asString(url).trim();
    if (!s) return '';
    if (s.indexOf('://') < 0) s = 'http://' + s;
    return s.replace(/\/+$/, '');
  }

  /* Per-call override beats the live environment, which beats the value
   * captured at load time: a host that changes OLLAMA_HOST at runtime should
   * not need to reload the driver. */
  function baseUrlOf(adapter, ctx) {
    var own = normaliseBase(adapter.baseUrl);
    if (ctx && typeof ctx.baseUrl === 'string' && ctx.baseUrl) {
      var given = normaliseBase(ctx.baseUrl);
      /* The plugin driver echoes spec.baseUrl into ctx.baseUrl, so an
       * identical value is not an override — otherwise an OLLAMA_HOST /
       * LLAMA_CPP_HOST change made after load would be ignored. */
      if (given && given !== own) return given;
    }
    var fromEnv = adapter.envVar ? envGet(adapter.envVar) : null;
    if (fromEnv) return normaliseBase(fromEnv);
    return own;
  }

  /* A caller that already pointed the host at .../v1 must not get /v1/v1/... */
  function joinUrl(base, path) {
    var b = normaliseBase(base);
    var p = asString(path);
    if (!p) return b;
    if (p.charAt(0) !== '/') p = '/' + p;
    if (b.slice(-3) === '/v1' && p.indexOf('/v1/') === 0) return b + p.slice(3);
    return b + p;
  }

  /* ── Transport ───────────────────────────────────────────────────── */

  /* The plugin driver hands every provider a Shield-2 egress-guarded fetch on
   * ctx.fetch (private hosts allowed only when the adapter's own baseUrl is
   * private, link-local/metadata always blocked, signal injected). Prefer it
   * when present; a bare host context still gets the global fetch. */
  function runtimeFetch(ctx) {
    if (ctx && typeof ctx.fetch === 'function') return ctx.fetch;
    return (typeof g.fetch === 'function') ? g.fetch : null;
  }

  function signalOf(ctx) {
    return (ctx && ctx.signal) ? ctx.signal : null;
  }

  function checkAbort(name, ctx) {
    var s = signalOf(ctx);
    if (s && s.aborted) throw abortError(name);
  }

  /* The host fetch may or may not plumb init.signal through to libuv, so race
   * the promise as well: an abort must reject promptly and the caller must
   * never wait on a request it already cancelled. */
  function raceAbort(name, ctx, promise) {
    var s = signalOf(ctx);
    if (!s || typeof s.addEventListener !== 'function') return promise;
    return new Promise(function (resolve, reject) {
      var settled = false;
      function detach() {
        if (typeof s.removeEventListener === 'function') s.removeEventListener('abort', onAbort);
      }
      function onAbort() {
        if (settled) return;
        settled = true;
        detach();
        reject(abortError(name));
      }
      try {
        s.addEventListener('abort', onAbort);
      } catch (e) { /* exotic signal shim: checkAbort above still covers it */ }
      promise.then(function (v) {
        if (settled) return;
        settled = true;
        detach();
        resolve(v);
      }, function (e) {
        if (settled) return;
        settled = true;
        detach();
        reject(e);
      });
    });
  }

  /* Classify a transport failure once. Typed errors raised by this driver (or
   * by a lower layer) pass through untouched; anything else becomes E_TOOL so
   * no raw host exception escapes the adapter surface. */
  function classifyError(name, ctx, err) {
    var s = signalOf(ctx);
    if (s && s.aborted) return abortError(name);
    var code = err && err.code;
    if (code === 'E_TIMEOUT' || code === 'E_DENIED' || code === 'E_TOOL' ||
        code === 'E_SCHEMA' || code === 'E_CAPABILITY' || code === 'E_BACKPRESSURE' ||
        code === 'E_CIRCUIT_OPEN') {
      return err;
    }
    if (err && (err.name === 'AbortError' || err.name === 'TimeoutError')) return abortError(name);
    return makeError('E_TOOL', name + ': request failed — ' + ((err && err.message) || asString(err)),
                     { adapter: name });
  }

  function authHeaders(key) {
    var h = { 'Content-Type': 'application/json' };
    if (key) h['Authorization'] = 'Bearer ' + key;
    return h;
  }

  /* Read a complete response body. cap = 0 disables the ceiling (error
   * excerpts and health probes, where the body is small by contract). */
  async function readText(name, ctx, res, cap) {
    if (!res || typeof res.text !== 'function') {
      throw makeError('E_TOOL', name + ': response has no text() body', { adapter: name });
    }
    var text;
    try {
      text = await raceAbort(name, ctx, Promise.resolve(res.text()));
    } catch (e) {
      throw classifyError(name, ctx, e);
    }
    checkAbort(name, ctx);
    text = asString(text);
    var limit = (typeof cap === 'number') ? cap : MAX_BODY;
    if (limit > 0 && text.length > limit) {
      throw makeError('E_BACKPRESSURE',
        name + ': response body exceeded ' + limit + ' bytes — refusing to buffer it',
        { adapter: name, limit: limit, bytes: text.length });
    }
    return text;
  }

  async function postJson(name, ctx, url, payload, headers) {
    var f = runtimeFetch(ctx);
    if (!f) throw makeError('E_CAPABILITY', name + ': this runtime has no fetch');
    checkAbort(name, ctx);
    var init = {
      method: 'POST',
      headers: headers || authHeaders(null),
      body: JSON.stringify(payload)
    };
    var s = signalOf(ctx);
    if (s) init.signal = s;
    var res;
    try {
      res = await raceAbort(name, ctx, Promise.resolve(f(url, init)));
    } catch (e) {
      throw classifyError(name, ctx, e);
    }
    checkAbort(name, ctx);
    if (!res) throw makeError('E_TOOL', name + ': fetch resolved without a response', { adapter: name });
    var status = asInt(res.status);
    if (status && (status < 200 || status >= 300)) {
      var raw = '';
      try {
        raw = await readText(name, ctx, res, 0);
      } catch (e2) {
        raw = '';
      }
      throw makeError('E_TOOL',
        name + ': HTTP ' + status + ' — ' + (bodyExcerpt(raw) || 'empty response body'),
        { adapter: name, status: status });
    }
    return res;
  }

  async function getJson(name, ctx, url, headers) {
    var f = runtimeFetch(ctx);
    if (!f) throw makeError('E_CAPABILITY', name + ': this runtime has no fetch');
    checkAbort(name, ctx);
    var init = { method: 'GET', headers: headers || {} };
    var s = signalOf(ctx);
    if (s) init.signal = s;
    var res;
    try {
      res = await raceAbort(name, ctx, Promise.resolve(f(url, init)));
    } catch (e) {
      throw classifyError(name, ctx, e);
    }
    checkAbort(name, ctx);
    var status = asInt(res && res.status);
    var text = '';
    try {
      text = await readText(name, ctx, res, 0);
    } catch (e) {
      text = '';
    }
    if (status && (status < 200 || status >= 300)) {
      throw makeError('E_TOOL',
        name + ': HTTP ' + status + ' — ' + (bodyExcerpt(text) || 'empty response body'),
        { adapter: name, status: status });
    }
    if (!text) return null;
    return parseJson(name, text, url);
  }

  /* A health probe reports state, never throws (abort included — it comes
   * back as ok:false / E_TIMEOUT rather than a rejected promise). Unknown
   * adapter names still throw E_CAPABILITY, from the namespace below. */
  async function probe(name, ctx, url, headers, pick) {
    var started = nowMs();
    try {
      var json = await getJson(name, ctx, url, headers);
      var out = { ok: true, adapter: name, latencyMs: Math.max(0, Math.round(nowMs() - started)) };
      var extra = pick ? (pick(json) || {}) : null;
      if (extra) {
        for (var k in extra) {
          if (Object.prototype.hasOwnProperty.call(extra, k)) out[k] = extra[k];
        }
      }
      return out;
    } catch (e) {
      return {
        ok: false,
        adapter: name,
        latencyMs: Math.max(0, Math.round(nowMs() - started)),
        error: { code: (e && e.code) || 'E_TOOL', message: (e && e.message) || asString(e) }
      };
    }
  }

  /* ── Buffered replay ─────────────────────────────────────────────── */

  async function emit(onEvent, ev) {
    if (typeof onEvent !== 'function') return;
    var r = onEvent(ev);
    if (r && typeof r.then === 'function') await r;
  }

  /* Walk a buffered body one line at a time. onLine(line, lineNo) returns
   * true to stop early (protocol terminator: "data: [DONE]", ollama
   * done:true). Abort is re-checked periodically so replaying a huge body is
   * still cancellable between chunks. */
  async function replayLines(name, ctx, text, onLine) {
    var len = text.length;
    if (!len) return 0;
    var i = 0;
    var lineNo = 0;
    while (i < len) {
      var nl = text.indexOf('\n', i);
      var end = (nl < 0) ? len : nl;
      var line = text.slice(i, end);
      if (line.length && line.charAt(line.length - 1) === '\r') line = line.slice(0, line.length - 1);
      lineNo += 1;
      /* One property read per line: an abort must land between replayed
       * chunks, not after the whole body. */
      checkAbort(name, ctx);
      if (await onLine(line, lineNo)) return lineNo;
      if (nl < 0) break;
      i = nl + 1;
    }
    return lineNo;
  }

  function newState() {
    return {
      text: '',
      usage: makeUsage(null),
      finishReason: null,
      toolAcc: [],    /* OpenAI SSE: arguments arrive as string fragments */
      toolCalls: [],  /* Ollama NDJSON: whole calls, the last frame wins */
      sawFrame: false
    };
  }

  async function adoptWhole(state, whole, onEvent) {
    if (whole.text) {
      state.text += whole.text;
      await emit(onEvent, { type: 'delta', text: whole.text });
    }
    if (whole.thinking) {
      state.thinking = (state.thinking || '') + whole.thinking;
      await emit(onEvent, { type: 'think', text: whole.thinking });
    }
    if (whole.toolCalls && whole.toolCalls.length) state.toolCalls = whole.toolCalls;
    state.usage = whole.usage;
    state.finishReason = whole.finishReason;
  }

  async function finishStream(state, onEvent, adapter, frames) {
    var usage = state.usage;
    var toolCalls = (state.toolCalls && state.toolCalls.length) ? state.toolCalls : drainToolAcc(state);
    await emit(onEvent, {
      type: 'done',
      usage: usage,
      toolCalls: toolCalls,
      finishReason: state.finishReason
    });
    return {
      text: state.text || (state.thinking || ''),
      thinking: state.thinking || '',
      toolCalls: toolCalls,
      usage: usage,
      finishReason: state.finishReason,
      model: null,
      buffered: true,
      adapter: adapter.name,
      kind: adapter.kind,
      frames: frames
    };
  }

  /* ── Tool-call normalisation ─────────────────────────────────────── */

  /* Both wires end up as [{ id, name, arguments: object }]. Ollama ships the
   * arguments as an object, OpenAI as a JSON string (fragmented across SSE
   * deltas), so parse once here. Unparseable argument text is kept on
   * argumentsText instead of throwing: a bad argument payload is the model's
   * mistake to repair, and E_TOOL here would discard the rest of a valid
   * response. */
  function normaliseToolCalls(list) {
    var out = [];
    if (!list || !list.length) return out;
    for (var i = 0; i < list.length; i++) {
      var tc = list[i] || {};
      var fn = tc.function || tc;
      var id = asString(tc.id) || ('call_' + i);
      var raw = fn.arguments;
      var args = {};
      var keepRaw = null;
      if (raw && typeof raw === 'object') {
        args = raw;
      } else if (typeof raw === 'string' && raw.replace(/\s+/g, '')) {
        try {
          var parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object') args = parsed;
          else keepRaw = raw;
        } catch (e) {
          keepRaw = raw;
        }
      }
      var entry = { id: id, name: asString(fn.name), arguments: args };
      if (keepRaw !== null) entry.argumentsText = keepRaw;
      out.push(entry);
    }
    return out;
  }

  function accumulateDeltas(state, list) {
    if (!list || !list.length) return;
    for (var i = 0; i < list.length; i++) {
      var tc = list[i] || {};
      var idx = (typeof tc.index === 'number') ? tc.index : state.toolAcc.length;
      var slot = state.toolAcc[idx];
      if (!slot) {
        slot = { id: '', name: '', arguments: '' };
        state.toolAcc[idx] = slot;
      }
      if (tc.id) slot.id = asString(tc.id);
      var fn = tc.function || {};
      if (fn.name) slot.name = asString(fn.name);
      if (typeof fn.arguments === 'string') slot.arguments += fn.arguments;
      else if (fn.arguments && typeof fn.arguments === 'object') slot.arguments += JSON.stringify(fn.arguments);
    }
  }

  function drainToolAcc(state) {
    if (!state.toolAcc.length) return [];
    var list = [];
    for (var i = 0; i < state.toolAcc.length; i++) {
      var slot = state.toolAcc[i];
      if (!slot) continue;
      list.push({ id: slot.id || ('call_' + i), function: { name: slot.name, arguments: slot.arguments } });
    }
    state.toolAcc = [];
    return normaliseToolCalls(list);
  }

  /* ── Shared request shaping ──────────────────────────────────────── */

  function pickOpt(req, key) {
    if (!req) return null;
    if (req[key] !== undefined && req[key] !== null) return req[key];
    if (req.opts && req.opts[key] !== undefined && req.opts[key] !== null) return req.opts[key];
    return null;
  }

  function numOpt(req, key) {
    var v = pickOpt(req, key);
    if (v === null) return null;
    var n = Number(v);
    return isFinite(n) ? n : null;
  }

  function pickModel(adapter, req) {
    var m = pickOpt(req, 'model') || adapter.model;
    if (!m) {
      throw makeError('E_SCHEMA',
        adapter.name + ': no model — pass req.model (an empty model is never guessed)',
        { adapter: adapter.name });
    }
    return asString(m);
  }

  function wireMessages(adapter, req) {
    var list = req && req.messages;
    if (!list && req && typeof req.prompt === 'string') list = [{ role: 'user', content: req.prompt }];
    if (!list || !list.length) {
      throw makeError('E_SCHEMA', adapter.name + ': req.messages must be a non-empty array',
                      { adapter: adapter.name });
    }
    var ollama = (adapter.kind === 'ollama');
    var out = [];
    for (var i = 0; i < list.length; i++) {
      var m = list[i] || {};
      var msg = { role: asString(m.role) || 'user', content: asString(m.content) };
      if (m.tool_calls) msg.tool_calls = m.tool_calls;
      if (m.tool_call_id) msg.tool_call_id = asString(m.tool_call_id);
      /* images is an Ollama extension — an OpenAI-compatible host rejects
       * unknown message fields, so only Ollama sees it. */
      if (ollama && m.images && m.images.length) msg.images = m.images;
      out.push(msg);
    }
    return out;
  }

  function maxTokensOf(req) {
    var v = numOpt(req, 'max_tokens');
    if (v === null) v = numOpt(req, 'maxTokens');
    return (v !== null && v > 0) ? Math.floor(v) : null;
  }

  /* Key policy: an explicit ctx.apiKey wins, then ctx.env, then the real
   * environment. Only adapters marked requiresKey fail when the key is
   * absent — llama.cpp and the local aliases are keyless by design. */
  function resolveKey(adapter, ctx) {
    var explicit = (ctx && typeof ctx.apiKey === 'string' && ctx.apiKey) ? ctx.apiKey : null;
    if (explicit) return explicit;
    var keyName = adapter.keyEnv;
    if (!keyName) return null;
    var ctxEnv = (ctx && ctx.env) ? ctx.env : null;
    if (ctxEnv && typeof ctxEnv[keyName] === 'string' && ctxEnv[keyName]) return ctxEnv[keyName];
    return envGet(keyName);
  }

  function requireKey(adapter, ctx) {
    var key = resolveKey(adapter, ctx);
    if (key) return key;
    if (!adapter.requiresKey) return null;
    var keyName = adapter.keyEnv || 'OPENAI_API_KEY';
    throw makeError('E_DENIED',
      adapter.name + ': no API key — pass ctx.apiKey, set ctx.env.' + keyName +
      ', or export ' + keyName + ' in the environment',
      { adapter: adapter.name, keyEnv: keyName });
  }

  /* ── OpenAI-compatible wire: llamacpp, openai, lmstudio, vllm ────── */

  function openAiBody(adapter, req, stream) {
    var body = {
      model: pickModel(adapter, req),
      messages: wireMessages(adapter, req),
      stream: !!stream
    };
    var temp = numOpt(req, 'temperature');
    if (temp !== null) body.temperature = temp;
    var maxTokens = maxTokensOf(req);
    if (maxTokens !== null) body.max_tokens = maxTokens;
    if (req && req.tools && req.tools.length) body.tools = req.tools;
    var rf = pickOpt(req, 'response_format');
    if (rf) body.response_format = rf;
    /* llama.cpp extension: GBNF grammar. Forwarded only when the caller
     * supplied one, so a plain OpenAI host never sees the field. */
    var grammar = pickOpt(req, 'grammar');
    if (grammar) body.grammar = grammar;
    return body;
  }

  function openAiErrorMessage(err) {
    if (typeof err === 'string') return err;
    if (err && typeof err.message === 'string') return err.message;
    return shortText(JSON.stringify(err), 160);
  }

  function parseOpenAiCompletion(name, text, where) {
    var json = parseJson(name, text, where);
    if (json && json.error) {
      throw makeError('E_TOOL', name + ': ' + openAiErrorMessage(json.error), { adapter: name });
    }
    var choices = (json && json.choices) || [];
    var choice = choices[0] || {};
    var msg = choice.message || {};
    /* Reasoning models (Qwen3, DeepSeek-R1, …) put the chain of thought in
     * reasoning_content and leave content EMPTY when they hit a token or
     * reasoning budget; observed live against llama-server, where the answer
     * arrived only as reasoning_content and the adapter returned "". Expose
     * it as thinking, and fall back to it so a turn is never silently blank. */
    var reasoning = asString(msg.reasoning_content) || asString(msg.reasoning);
    var content = asString(msg.content);
    return {
      text: content || reasoning,
      thinking: reasoning,
      toolCalls: normaliseToolCalls(msg.tool_calls),
      usage: makeUsage(json && json.usage),
      finishReason: choice.finish_reason ? asString(choice.finish_reason) : null,
      model: (json && json.model) ? asString(json.model) : null
    };
  }

  /* One SSE frame per line; "data: [DONE]" terminates. Returns true to stop. */
  async function handleSseLine(name, state, line, lineNo, onEvent) {
    if (!line) return false;
    if (line.charAt(0) === ':') return false; /* SSE comment / keep-alive */
    if (line.indexOf('data:') === 0) {
      var payload = line.slice(5).replace(/^\s+/, '');
      if (!payload) return false;
      if (payload === '[DONE]') {
        state.sawFrame = true;
        return true;
      }
      var json = parseJson(name, payload, 'SSE line ' + lineNo);
      state.sawFrame = true;
      if (json && json.error) {
        throw makeError('E_TOOL',
          name + ': ' + openAiErrorMessage(json.error) + ' (SSE line ' + lineNo + ')',
          { adapter: name, line: lineNo });
      }
      var choices = (json && json.choices) || [];
      var choice = choices[0] || {};
      var delta = choice.delta || {};
      if (typeof delta.content === 'string' && delta.content.length) {
        state.text += delta.content;
        await emit(onEvent, { type: 'delta', text: delta.content });
      }
      /* Reasoning deltas: same rule as the non-streaming body. */
      var thinkDelta = (typeof delta.reasoning_content === 'string' && delta.reasoning_content)
        ? delta.reasoning_content
        : ((typeof delta.reasoning === 'string' && delta.reasoning) ? delta.reasoning : '');
      if (thinkDelta) {
        state.thinking = (state.thinking || '') + thinkDelta;
        await emit(onEvent, { type: 'think', text: thinkDelta });
      }
      if (delta.tool_calls) accumulateDeltas(state, delta.tool_calls);
      if (choice.finish_reason) state.finishReason = asString(choice.finish_reason);
      if (json && json.usage) mergeUsage(state.usage, json.usage);
      return false;
    }
    /* event:, id: and retry: are legal SSE fields that carry no chat payload. */
    if (line.indexOf('event:') === 0 || line.indexOf('id:') === 0 || line.indexOf('retry:') === 0) {
      return false;
    }
    throw makeError('E_TOOL',
      name + ': malformed SSE line ' + lineNo + ' — ' + shortText(line, 120),
      { adapter: name, line: lineNo });
  }

  async function openAiComplete(adapter, req, ctx) {
    var name = adapter.name;
    var url = joinUrl(baseUrlOf(adapter, ctx), adapter.chatPath || '/chat/completions');
    var key = requireKey(adapter, ctx);
    checkAbort(name, ctx);
    var res = await postJson(name, ctx, url, openAiBody(adapter, req, false), authHeaders(key));
    var text = await readText(name, ctx, res);
    return parseOpenAiCompletion(name, text, 'response body');
  }

  async function openAiStream(adapter, req, ctx, onEvent) {
    var name = adapter.name;
    var url = joinUrl(baseUrlOf(adapter, ctx), adapter.chatPath || '/chat/completions');
    var key = requireKey(adapter, ctx);
    checkAbort(name, ctx);
    var res = await postJson(name, ctx, url, openAiBody(adapter, req, true), authHeaders(key));
    var text = await readText(name, ctx, res);
    var state = newState();
    var trimmed = text.replace(/^\s+/, '');
    var frames = 0;
    if (trimmed.charAt(0) === '{' && trimmed.indexOf('data:') < 0) {
      /* The host ignored stream:true and answered with one JSON completion:
       * adopt it through the same parser BEFORE the line walker can call the
       * body a malformed SSE frame. */
      await adoptWhole(state, parseOpenAiCompletion(name, text, 'buffered response'), onEvent);
    } else {
      frames = await replayLines(name, ctx, text, function (line, lineNo) {
        return handleSseLine(name, state, line, lineNo, onEvent);
      });
      if (!state.sawFrame && text.replace(/\s+/g, '')) {
        await adoptWhole(state, parseOpenAiCompletion(name, text, 'buffered response'), onEvent);
      }
    }
    return finishStream(state, onEvent, adapter, frames);
  }

  async function openAiHealth(adapter, ctx) {
    /* Fail CLOSED: a keyed adapter must not send an unauthenticated probe and
     * report the resulting 401 as a transport error (E_TOOL) — a missing key
     * is a configuration problem (E_DENIED), same as the chat path. */
    var key = requireKey(adapter, ctx);
    return probe(adapter.name, ctx, joinUrl(baseUrlOf(adapter, ctx), '/models'), authHeaders(key),
      function (json) {
        var list = (json && json.data) || [];
        return { models: list.length };
      });
  }

  async function openAiListModels(adapter, ctx) {
    var key = requireKey(adapter, ctx);
    var json = await getJson(adapter.name, ctx, joinUrl(baseUrlOf(adapter, ctx), '/models'),
                             authHeaders(key));
    var data = (json && json.data) || [];
    var out = [];
    for (var i = 0; i < data.length; i++) {
      var m = data[i] || {};
      var id = asString(m.id);
      if (id) out.push({ id: id, name: id });
    }
    return out;
  }

  function makeOpenAiAdapter(def) {
    return {
      name: def.name,
      kind: def.kind,
      envVar: def.envVar,
      baseUrl: normaliseBase(envOr(def.envVar, def.baseUrl)),
      model: def.modelEnv ? envOr(def.modelEnv, def.model || '') : (def.model || ''),
      description: def.description,
      streaming: 'buffered',
      requiresKey: !!def.requiresKey,
      keyEnv: def.keyEnv || null,
      /* A host whose base URL already ends in /v1 (openai, lmstudio, vllm)
       * takes /chat/completions; llama.cpp's default base has no /v1, so it
       * carries the full /v1/chat/completions (joinUrl dedupes a /v1 base). */
      chatPath: def.chatPath || '/chat/completions',

      async complete(req, ctx) {
        return await openAiComplete(this, req, ctx);
      },
      async stream(req, ctx, onEvent) {
        return await openAiStream(this, req, ctx, onEvent);
      },
      async health(ctx) {
        return await openAiHealth(this, ctx);
      },
      async listModels(ctx) {
        return await openAiListModels(this, ctx);
      }
    };
  }

  /* lmstudio and vllm speak the OpenAI wire, so they are the openai adapter
   * re-pointed at a local port: the methods DELEGATE with this bound to the
   * alias (which only carries name/kind/baseUrl/key policy). One parser, one
   * behaviour, one place to fix a wire bug. */
  function openAiAlias(source, def) {
    var alias = {
      name: def.name,
      /* kind is the WIRE kind: the plugin driver's registerProvider validates
       * it against ollama|llamacpp|openai|custom, and these two speak the
       * OpenAI wire. Identity lives in name + delegate. */
      kind: 'openai',
      envVar: def.envVar,
      baseUrl: normaliseBase(envOr(def.envVar, def.baseUrl)),
      model: def.modelEnv ? envOr(def.modelEnv, '') : '',
      description: def.description,
      streaming: 'buffered',
      requiresKey: false,
      keyEnv: null,
      chatPath: source.chatPath,
      delegate: source.name,

      complete: function (req, ctx) { return source.complete.call(alias, req, ctx); },
      stream: function (req, ctx, onEvent) { return source.stream.call(alias, req, ctx, onEvent); },
      health: function (ctx) { return source.health.call(alias, ctx); },
      listModels: function (ctx) { return source.listModels.call(alias, ctx); }
    };
    return alias;
  }

  /* ── Ollama wire ─────────────────────────────────────────────────── */

  function ollamaBody(adapter, req, stream) {
    var body = {
      model: pickModel(adapter, req),
      messages: wireMessages(adapter, req),
      stream: !!stream
    };
    if (req && req.tools && req.tools.length) body.tools = req.tools;
    /* Ollama takes sampling knobs under options; a temperature/max_tokens the
     * caller set is otherwise silently dropped. */
    var options = {};
    var temp = numOpt(req, 'temperature');
    if (temp !== null) options.temperature = temp;
    var maxTokens = maxTokensOf(req);
    if (maxTokens !== null) options.num_predict = maxTokens;
    for (var k in options) {
      if (Object.prototype.hasOwnProperty.call(options, k)) {
        body.options = options;
        break;
      }
    }
    var rf = pickOpt(req, 'response_format');
    if (rf) body.format = (typeof rf === 'string') ? rf : 'json';
    return body;
  }

  function parseOllamaChat(name, text, where) {
    var json = parseJson(name, text, where);
    if (json && typeof json.error === 'string') {
      throw makeError('E_TOOL', name + ': ' + json.error, { adapter: name });
    }
    var msg = (json && json.message) || {};
    /* DeepSeek-R1 style models served by Ollama stream the chain of thought
     * in message.thinking and may leave content empty. */
    var thinking = asString(msg.thinking);
    var content = asString(msg.content);
    return {
      text: content || thinking,
      thinking: thinking,
      toolCalls: normaliseToolCalls(msg.tool_calls),
      usage: makeUsage(json),
      finishReason: (json && json.done_reason) ? asString(json.done_reason) : null,
      model: (json && json.model) ? asString(json.model) : null
    };
  }

  /* One NDJSON frame per line; done:true terminates. Returns true to stop. */
  async function handleNdjsonLine(name, state, line, lineNo, onEvent) {
    if (!line) return false;
    var json = parseJson(name, line, 'NDJSON line ' + lineNo);
    state.sawFrame = true;
    if (json && typeof json.error === 'string') {
      throw makeError('E_TOOL',
        name + ': ' + json.error + ' (NDJSON line ' + lineNo + ')',
        { adapter: name, line: lineNo });
    }
    var msg = (json && json.message) || {};
    if (typeof msg.content === 'string' && msg.content.length) {
      state.text += msg.content;
      await emit(onEvent, { type: 'delta', text: msg.content });
    }
    if (typeof msg.thinking === 'string' && msg.thinking.length) {
      state.thinking = (state.thinking || '') + msg.thinking;
      await emit(onEvent, { type: 'think', text: msg.thinking });
    }
    if (msg.tool_calls && msg.tool_calls.length) state.toolCalls = normaliseToolCalls(msg.tool_calls);
    mergeUsage(state.usage, json);
    if (json && json.done === true) {
      state.finishReason = json.done_reason ? asString(json.done_reason) : 'stop';
      return true;
    }
    return false;
  }

  function makeOllamaAdapter() {
    return {
      name: 'ollama',
      kind: 'ollama',
      envVar: 'OLLAMA_HOST',
      baseUrl: normaliseBase(envOr('OLLAMA_HOST', 'http://127.0.0.1:11434')),
      model: envOr('OLLAMA_MODEL', 'llama3.2'),
      description: 'Native Ollama endpoint (/api/chat) with tool calls and NDJSON frames',
      streaming: 'buffered',
      requiresKey: false,
      keyEnv: null,
      chatPath: '/api/chat',

      async complete(req, ctx) {
        var name = this.name;
        var url = joinUrl(baseUrlOf(this, ctx), '/api/chat');
        checkAbort(name, ctx);
        var res = await postJson(name, ctx, url, ollamaBody(this, req, false));
        var text = await readText(name, ctx, res);
        return parseOllamaChat(name, text, 'response body');
      },

      async stream(req, ctx, onEvent) {
        var name = this.name;
        var url = joinUrl(baseUrlOf(this, ctx), '/api/chat');
        checkAbort(name, ctx);
        var res = await postJson(name, ctx, url, ollamaBody(this, req, true));
        var text = await readText(name, ctx, res);
        var state = newState();
        var frames = await replayLines(name, ctx, text, function (line, lineNo) {
          return handleNdjsonLine(name, state, line, lineNo, onEvent);
        });
        if (!state.sawFrame && text.replace(/\s+/g, '')) {
          await adoptWhole(state, parseOllamaChat(name, text, 'buffered response'), onEvent);
        }
        return finishStream(state, onEvent, this, frames);
      },

      async health(ctx) {
        return probe(this.name, ctx, joinUrl(baseUrlOf(this, ctx), '/api/version'), null,
          function (json) {
            return { version: (json && json.version) ? asString(json.version) : null };
          });
      },

      async listModels(ctx) {
        var json = await getJson(this.name, ctx, joinUrl(baseUrlOf(this, ctx), '/api/tags'), null);
        var list = (json && json.models) || [];
        var out = [];
        for (var i = 0; i < list.length; i++) {
          var m = list[i] || {};
          var id = asString(m.model) || asString(m.name);
          if (!id) continue;
          var entry = { id: id, name: asString(m.name) || id };
          if (m.size !== undefined) entry.size = asInt(m.size);
          if (m.modified_at) entry.modifiedAt = asString(m.modified_at);
          out.push(entry);
        }
        return out;
      }
    };
  }

  /* ── Registry ────────────────────────────────────────────────────── */

  /* registerProvider is the plugin driver's public entry point; older builds
   * (and hosts that trimmed the driver) only carry PROVIDERS, so fall back to
   * writing the map directly. */
  function register(spec) {
    if (typeof PLUGINS.registerProvider === 'function') {
      try {
        PLUGINS.registerProvider(spec);
      } catch (e) {
        /* A stricter/older registerProvider must not cost the whole driver:
         * the PROVIDERS fallback below still publishes this adapter. */
      }
    }
    if (!PLUGINS.PROVIDERS) PLUGINS.PROVIDERS = {};
    if (!PLUGINS.PROVIDERS[spec.name]) PLUGINS.PROVIDERS[spec.name] = spec;
    return spec;
  }

  var ollama = makeOllamaAdapter();
  var llamacpp = makeOpenAiAdapter({
    name: 'llamacpp',
    kind: 'llamacpp',
    envVar: 'LLAMA_CPP_HOST',
    baseUrl: 'http://127.0.0.1:8080',
    chatPath: '/v1/chat/completions',
    modelEnv: 'LLAMA_CPP_MODEL',
    model: '',
    description: 'llama.cpp llama-server endpoint (OpenAI-compatible) with GBNF grammar support',
    requiresKey: false,
    keyEnv: null
  });
  var openai = makeOpenAiAdapter({
    name: 'openai',
    kind: 'openai',
    envVar: 'OPENAI_BASE_URL',
    baseUrl: 'https://api.openai.com/v1',
    modelEnv: 'OPENAI_MODEL',
    model: '',
    description: 'OpenAI-compatible cloud endpoint (Bearer auth, SSE frames)',
    requiresKey: true,
    keyEnv: 'OPENAI_API_KEY'
  });
  var lmstudio = openAiAlias(openai, {
    name: 'lmstudio',
    envVar: 'LMSTUDIO_HOST',
    baseUrl: 'http://127.0.0.1:1234/v1',
    modelEnv: 'LMSTUDIO_MODEL',
    description: 'LM Studio local server (OpenAI-compatible, alias of the openai adapter)'
  });
  var vllm = openAiAlias(openai, {
    name: 'vllm',
    envVar: 'VLLM_HOST',
    baseUrl: 'http://127.0.0.1:8000/v1',
    modelEnv: 'VLLM_MODEL',
    description: 'vLLM local server (OpenAI-compatible, alias of the openai adapter)'
  });

  var REGISTRY = [];
  function add(adapter) {
    REGISTRY.push(register(adapter));
    return adapter;
  }
  add(ollama);
  add(llamacpp);
  add(openai);
  add(lmstudio);
  add(vllm);

  function find(name) {
    var key = asString(name);
    for (var i = 0; i < REGISTRY.length; i++) {
      if (REGISTRY[i].name === key) return REGISTRY[i];
    }
    return null;
  }

  /* Thin namespace over the registry. streaming:'buffered' is reported so the
   * replay limitation is visible in the API instead of being a surprise. */
  if (!g.sofuu.providers) g.sofuu.providers = {};
  var NS = g.sofuu.providers;

  NS.list = function () {
    var out = [];
    for (var i = 0; i < REGISTRY.length; i++) {
      var a = REGISTRY[i];
      out.push({
        name: a.name,
        kind: a.kind,
        baseUrl: baseUrlOf(a, null),
        model: a.model || '',
        description: a.description || '',
        streaming: 'buffered',
        requiresKey: !!a.requiresKey
      });
    }
    return out;
  };

  NS.get = function (name) {
    return find(name);
  };

  NS.health = async function (name, ctx) {
    var a = find(name);
    if (!a) {
      var names = [];
      for (var i = 0; i < REGISTRY.length; i++) names.push(REGISTRY[i].name);
      throw makeError('E_CAPABILITY',
        'sofuu.providers.health: unknown provider "' + asString(name) + '" (known: ' + names.join(', ') + ')',
        { adapter: asString(name), known: names });
    }
    if (typeof a.health !== 'function') {
      return { ok: false, adapter: a.name, error: { code: 'E_CAPABILITY', message: a.name + ': no health probe' } };
    }
    return await a.health(ctx || {});
  };

  NS.MAX_BODY = MAX_BODY;

})();
