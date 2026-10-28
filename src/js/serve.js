/* src/js/serve.js — the HTTP edge server, as shipped JS (workstream H1).
 *
 * Was a ~400-line inline string in main.rs: untestable, and its /v1/* routes
 * were a naive `role: content` concatenation with no usage, finish_reason,
 * stop sequences, tools or OpenAI error bodies. This module is the one
 * implementation; `sofuu serve` boots it with a config object and tests drive
 * `createHandler()` directly with a fake stream.
 *
 * API:
 *   sofuu.serve.start(opts)          → {ok, host, port, model}
 *     opts: { brainPath, modelPath, modelId, token, host, port, version }
 *   sofuu.serve.createHandler(opts)  → async function (req, res)
 *   sofuu.serve.state()              → {started, sessions, turns}
 *
 * Routes (bearer auth except where noted):
 *   GET  /health                     liveness only, NO auth (H2)
 *   GET  /v1/models                  configured ids + aliases
 *   POST /v1/chat/completions        OpenAI chat, stream or JSON (H1)
 *   POST /v1/completions, /infer     raw text completion (legacy)
 *   POST /v1/embeddings              OpenAI embeddings (H2)
 *   POST /v1/sessions                create a process-local session (H3)
 *   GET  /v1/sessions                list sessions
 *   GET  /v1/sessions/:id/turns      recorded turns
 *   POST /v1/agent/runs              agent.run, SSE events or JSON (H3)
 *   GET  /v1/tools                   tool inventory + policy verdict (H3)
 *   POST /v1/tools/:name/call        execute under policy (H3)
 *   POST /remember GET /recall GET /memories POST /forget POST /share
 *
 * Auth: bearer token, constant-time compare. `policy.http.auth === 'none'`
 * opts out explicitly (loud warning at start); anything else keeps bearer.
 * `policy.http.deny` path prefixes are evaluated per request (R7.6).
 *
 * Everything a route needs is resolved at call time (sofuu.ai.stream etc.),
 * so a test can substitute a fake stream before creating the handler.
 */
(function () {
  'use strict';
  if (!globalThis.sofuu) return;

  sofuu.serve = sofuu.serve || {};

  var S = {
    started: false,
    startedAt: 0,
    startedAtMs: 0,
    cfg: null,
    brain: null,
    server: null,
    sessions: Object.create(null),
  };

  /* ── config ──────────────────────────────────────────────────── */

  function str(v, dflt) { return typeof v === 'string' ? v : (dflt || ''); }
  function num(v, dflt) {
    var n = Number(v);
    return isFinite(n) ? n : dflt;
  }

  function normalize(opts) {
    opts = opts || {};
    var models = [];
    var i;
    if (Array.isArray(opts.models)) {
      for (i = 0; i < opts.models.length; i++) {
        var m = opts.models[i];
        if (m && m.path) models.push({ id: str(m.id) || baseName(str(m.path)), path: str(m.path) });
      }
    }
    var modelPath = str(opts.modelPath);
    if (models.length === 0 && modelPath) {
      models.push({ id: str(opts.modelId) || baseName(modelPath), path: modelPath });
    }
    return {
      brainPath: str(opts.brainPath),
      modelPath: modelPath || (models[0] && models[0].path) || '',
      models: models,
      token: str(opts.token),
      host: str(opts.host) || '127.0.0.1',
      port: num(opts.port, 8080),
      version: str(opts.version) || '0.1.0',
      project: str(opts.project) || str(opts.cwd),
    };
  }

  function baseName(p) {
    var parts = String(p || '').split(/[\\/]/);
    return parts[parts.length - 1] || 'model';
  }

  /* ── policy ──────────────────────────────────────────────────── */

  function policy() {
    try {
      if (sofuu.policy && typeof sofuu.policy.get === 'function') {
        var raw = sofuu.policy.get();
        return raw ? JSON.parse(raw) : {};
      }
    } catch (e) {}
    return {};
  }

  function authMode() {
    var p = policy();
    var a = p && p.http && p.http.auth;
    return (a === 'none') ? 'none' : 'bearer';
  }

  function policyVerdict(path, method) {
    try {
      if (sofuu.policy && typeof sofuu.policy.evaluate === 'function') {
        return JSON.parse(sofuu.policy.evaluate({ op: 'http.request', path: path, method: method }));
      }
    } catch (e) {}
    return { decision: 'allow', rule: '', reason: '' };
  }

  /* ── response helpers ────────────────────────────────────────── */

  function send(res, status, obj) {
    var body = JSON.stringify(obj);
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(body);
  }

  /* OpenAI error envelope (H1): {error:{message,type,param,code}}. */
  function fail(res, status, message, type, code, param) {
    send(res, status, {
      error: {
        message: String(message || 'error'),
        type: type || (status === 401 ? 'authentication_error' : 'invalid_request_error'),
        param: param === undefined ? null : param,
        code: code || null,
      },
    });
  }

  function failPolicy(res, verdict) {
    send(res, 403, {
      error: {
        message: verdict.reason || 'denied by policy',
        type: 'permission_error',
        param: null,
        code: 'policy_denied',
        rule: verdict.rule || '',
      },
    });
  }

  function sseHead(res) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });
  }

  function sseData(res, obj) { res.write('data: ' + JSON.stringify(obj) + '\n\n'); }

  /* ── request helpers ─────────────────────────────────────────── */

  function parseQuery(url) {
    var qs = {};
    var raw = (String(url || '').split('?')[1] || '');
    var parts = raw.split('&');
    for (var i = 0; i < parts.length; i++) {
      if (!parts[i]) continue;
      var eq = parts[i].indexOf('=');
      if (eq < 0) continue;
      try {
        qs[decodeURIComponent(parts[i].slice(0, eq))] =
          decodeURIComponent(parts[i].slice(eq + 1).replace(/\+/g, ' '));
      } catch (e) {}
    }
    return qs;
  }

  /* req.body is a fully-buffered string in this server. */
  function readJson(req) {
    var body = typeof req.body === 'string' ? req.body : '';
    if (!body) return { ok: true, data: {} };
    try {
      var d = JSON.parse(body);
      return { ok: true, data: (d && typeof d === 'object') ? d : {} };
    } catch (e) {
      return { ok: false, error: 'body is not valid JSON' };
    }
  }

  function ctEq(a, b) {
    if (typeof a !== 'string' || typeof b !== 'string' || a.length !== b.length) {
      /* Length still feeds the loop so the timing does not short-circuit
       * on length alone; mismatch is always false. */
      var n = typeof a === 'string' ? a.length : 0;
      for (var k = 0; k < n; k++) { /* burn */ }
      return false;
    }
    var d = 0;
    for (var i = 0; i < b.length; i++) d |= (a.charCodeAt(i) ^ b.charCodeAt(i));
    return d === 0;
  }

  /* ── models ──────────────────────────────────────────────────── */

  function modelList(cfg) {
    var out = [];
    for (var i = 0; i < cfg.models.length; i++) {
      out.push({
        id: cfg.models[i].id,
        object: 'model',
        created: S.startedAt || Math.floor(Date.now() / 1000),
        owned_by: 'sofuu',
      });
    }
    return out;
  }

  /* Resolve a request's `model` to a loadable path. An id from /v1/models
   * wins, then the configured path itself, then an explicit existing model
   * path. Anything else is 404 — never an engine error about a missing
   * file named like the id (H1 item 4). */
  async function resolveModel(cfg, requested) {
    var want = str(requested);
    if (!want) return cfg.modelPath || null;
    var i;
    for (i = 0; i < cfg.models.length; i++) {
      if (cfg.models[i].id === want) return cfg.models[i].path;
      if (cfg.models[i].path === want) return want;
    }
    if (cfg.modelPath && want === cfg.modelPath) return want;
    if (/\.(gguf|safetensors|pt|pth|npz|npy)$/i.test(want)) {
      try { if (sofuu.fs && await sofuu.fs.exists(want)) return want; } catch (e) {}
    }
    try {
      if (sofuu.fs && await sofuu.fs.exists(want + '/tokenizer.json')) return want;
    } catch (e2) {}
    return null;
  }

  /* ── embeddings ──────────────────────────────────────────────── */

  function brainEmbed() {
    if (S.embed) return S.embed;
    return function (text) { return sofuu.ai.embedLocal(String(text)); };
  }

  function brainOpen(cfg) {
    var embed = function (t) { return sofuu.ai.embedLocal(String(t)); };
    if (!cfg.brainPath) return null;
    if (!(sofuu.memory && typeof sofuu.memory.open === 'function')) return null;
    var dim = 768;
    var space = undefined;
    try {
      if (typeof sofuu.memory.inspect === 'function') {
        var info = JSON.parse(sofuu.memory.inspect(cfg.brainPath));
        if (info && info.ok) {
          if (info.dimension) dim = info.dimension;
          if (info.id) space = info.id;
        }
      }
    } catch (e0) {}
    var brain = null;
    try {
      brain = (space === undefined)
        ? sofuu.memory.open(cfg.brainPath, dim)
        : sofuu.memory.open(cfg.brainPath, dim, space);
    } catch (e) { console.error('brain open: ' + e); }
    if (space === 'semantic-projector-v1' && typeof sofuu.ai.embedLocalSemantic === 'function') {
      embed = function (t) { return sofuu.ai.embedLocalSemantic(String(t)); };
    } else if (space === 'semantic-table-v2' && typeof sofuu.ai.embedLocalSemanticV2 === 'function') {
      embed = function (t) { return sofuu.ai.embedLocalSemanticV2(String(t)); };
    }
    /* P2 (audit 2026-10-14): a manifest space this server cannot embed
     * LOCALLY must not fall through to the default hash embedder — hash
     * vectors stored under a foreign-space manifest are mislabeled. */
    if (brain && space && space !== 'hash-v1' &&
        space !== 'semantic-projector-v1' && space !== 'semantic-table-v2') {
      console.error("brain space '" + space + "' has no local embedder in this server — refusing to serve (hash vectors would be mislabeled in a '" + space + "' store)");
      brain = null;
    }
    S.embed = embed;
    return brain;
  }

  function embeddingOf(text) {
    var fn = brainEmbed();
    var vec = fn(text);
    var out = [];
    for (var i = 0; i < vec.length; i++) out.push(vec[i]);
    return out;
  }

  /* ── stop sequences ──────────────────────────────────────────── */

  /* Emits text that cannot be the start of a stop sequence; holds back at
   * most (longest stop - 1) code units. Surrogate pairs are never split. */
  function stopFilter(stops, emit) {
    if (!stops || stops.length === 0) {
      return {
        push: function (t) { if (t) emit(t); return false; },
        flush: function () {},
        hit: false,
      };
    }
    var maxLen = 0;
    for (var i = 0; i < stops.length; i++) maxLen = Math.max(maxLen, stops[i].length);
    var buf = '';
    var hit = false;
    return {
      get hit() { return hit; },
      push: function (text) {
        if (hit || !text) return hit;
        buf += text;
        var cut = -1;
        for (var s = 0; s < stops.length; s++) {
          var idx = buf.indexOf(stops[s]);
          if (idx >= 0 && (cut < 0 || idx < cut)) cut = idx;
        }
        if (cut >= 0) {
          if (cut > 0) emit(buf.slice(0, cut));
          buf = '';
          hit = true;
          return true;
        }
        var keep = Math.max(0, maxLen - 1);
        var safe = buf.length - keep;
        if (safe > 0) {
          /* Do not cut between a high and low surrogate. */
          var code = buf.charCodeAt(safe - 1);
          if (code >= 0xD800 && code <= 0xDBFF) safe -= 1;
          if (safe > 0) { emit(buf.slice(0, safe)); buf = buf.slice(safe); }
        }
        return false;
      },
      flush: function () {
        if (buf) { emit(buf); buf = ''; }
      },
    };
  }

  function normalizeStops(v) {
    var out = [];
    if (typeof v === 'string' && v) out.push(v);
    else if (Array.isArray(v)) {
      for (var i = 0; i < v.length && out.length < 4; i++) {
        if (typeof v[i] === 'string' && v[i]) out.push(v[i]);
      }
    }
    return out;
  }

  /* ── OpenAI tool shapes ──────────────────────────────────────── */

  function engineTools(list) {
    var out = [];
    if (!Array.isArray(list)) return out;
    for (var i = 0; i < list.length; i++) {
      var t = list[i];
      var fn = t && (t.function || t);
      if (!fn || !fn.name) continue;
      out.push({
        name: String(fn.name),
        description: str(fn.description),
        parameters: fn.parameters || { type: 'object', properties: {} },
      });
    }
    return out;
  }

  function openaiToolCalls(calls) {
    var out = [];
    if (!Array.isArray(calls)) return out;
    for (var i = 0; i < calls.length; i++) {
      var c = calls[i];
      if (!c) continue;
      var fn = c.function || c;
      out.push({
        index: i,
        id: str(c.id) || ('call_' + i),
        type: 'function',
        function: {
          name: str(fn.name),
          arguments: typeof fn.arguments === 'string'
            ? fn.arguments
            : JSON.stringify(fn.arguments || {}),
        },
      });
    }
    return out;
  }

  function usageOf(iter, promptFallback) {
    var u = (iter && iter.usage) || {};
    var pt = Number(u.promptTokens || 0) || promptFallback || 0;
    var ct = Number(u.completionTokens || 0) || 0;
    return { prompt_tokens: pt, completion_tokens: ct, total_tokens: pt + ct };
  }

  function finishReason(toolCalls, stopHit, usage, maxTokens) {
    if (toolCalls && toolCalls.length > 0) return 'tool_calls';
    if (stopHit) return 'stop';
    if (maxTokens > 0 && usage.completion_tokens >= maxTokens) return 'length';
    return 'stop';
  }

  /* ── chat completions ───────────────────────────────────────── */

  function chatPrompt(cfg, data) {
    var messages = Array.isArray(data.messages) ? data.messages : null;
    if (!messages) return { ok: false, error: 'messages[] is required' };
    var system = '';
    for (var i = 0; i < messages.length; i++) {
      if (messages[i] && messages[i].role === 'system' && typeof messages[i].content === 'string') {
        system = messages[i].content;
        break;
      }
    }
    /* One template source: the native ChatML formatter. With tools the
     * engine must render them (messages path); without tools we hand the
     * engine the already-templated prompt. */
    var templated = null;
    try {
      if (system) templated = sofuu.ai.applyChatTemplate(messages, system);
      else templated = sofuu.ai.applyChatTemplate({ messages: messages });
    } catch (eT) { templated = null; }
    return { ok: true, messages: messages, system: system, templated: templated };
  }

  async function streamChat(cfg, res, data, modelPath) {
    var prepared = chatPrompt(cfg, data);
    if (!prepared.ok) {
      fail(res, 400, prepared.error, 'invalid_request_error', 'invalid_messages', 'messages');
      return;
    }
    var maxTokens = Math.max(0, num(data.max_tokens, 512) | 0);
    var temperature = typeof data.temperature === 'number' ? data.temperature : 0.7;
    var stops = normalizeStops(data.stop);
    var tools = engineTools(data.tools);
    var includeUsage = !!(data.stream_options && data.stream_options.include_usage);
    var wantStream = data.stream === true;

    var reqCfg = {
      provider: 'local',
      model: modelPath,
      temperature: temperature,
      maxTokens: maxTokens,
    };
    if (tools.length > 0) {
      reqCfg.messages = prepared.messages;
      if (prepared.system) reqCfg.system = prepared.system;
      reqCfg.tools = tools;
    } else if (prepared.templated) {
      reqCfg.prompt = prepared.templated;
    } else {
      reqCfg.messages = prepared.messages;
      if (prepared.system) reqCfg.system = prepared.system;
    }

    var id = 'chatcmpl-' + Math.random().toString(36).slice(2, 12);
    var created = Math.floor(Date.now() / 1000);
    var modelName = str(data.model) || cfg.models[0] && cfg.models[0].id || baseName(modelPath);
    var iter = sofuu.ai.stream(reqCfg);

    if (wantStream) {
      sseHead(res);
      sseData(res, {
        id: id, object: 'chat.completion.chunk', created: created, model: modelName,
        choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      });
      var sf = stopFilter(stops, function (text) {
        sseData(res, {
          id: id, object: 'chat.completion.chunk', created: created, model: modelName,
          choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
        });
      });
      var thinkSeen = '';
      while (true) {
        var n = await iter.next();
        if (n.done) break;
        var chunk = n.value || {};
        if (chunk.think) {
          /* Thinking stays off the content channel (OpenAI has no field
           * for it); the final message does not include it either. */
          thinkSeen += chunk.think;
        }
        if (chunk.text && sf.push(chunk.text)) break;
      }
      sf.flush();
      var usage = usageOf(iter, sofuu.ai.estimateTokens(prepared.messages));
      var calls = openaiToolCalls(iter.toolCalls);
      if (calls.length > 0) {
        sseData(res, {
          id: id, object: 'chat.completion.chunk', created: created, model: modelName,
          choices: [{ index: 0, delta: { tool_calls: calls }, finish_reason: null }],
        });
      }
      var fr = finishReason(calls, sf.hit, usage, maxTokens);
      sseData(res, {
        id: id, object: 'chat.completion.chunk', created: created, model: modelName,
        choices: [{ index: 0, delta: {}, finish_reason: fr }],
      });
      if (includeUsage) {
        sseData(res, {
          id: id, object: 'chat.completion.chunk', created: created, model: modelName,
          choices: [], usage: usage,
        });
      }
      res.write('data: [DONE]\n\n');
      res.end();
      recordTurn(data, prepared, '', usage);
      return;
    }

    /* JSON response */
    var text = '';
    var sf2 = stopFilter(stops, function (t) { text += t; });
    while (true) {
      var n2 = await iter.next();
      if (n2.done) break;
      var ch = n2.value || {};
      if (ch.text && sf2.push(ch.text)) break;
    }
    sf2.flush();
    var usage2 = usageOf(iter, sofuu.ai.estimateTokens(prepared.messages));
    var calls2 = openaiToolCalls(iter.toolCalls);
    var fr2 = finishReason(calls2, sf2.hit, usage2, maxTokens);
    var message = { role: 'assistant', content: text };
    if (calls2.length > 0) {
      message.content = text || null;
      message.tool_calls = calls2.map(function (c) {
        return { id: c.id, type: 'function', function: c.function };
      });
    }
    send(res, 200, {
      id: id, object: 'chat.completion', created: created, model: modelName,
      choices: [{ index: 0, message: message, finish_reason: fr2 }],
      usage: usage2,
    });
    recordTurn(data, prepared, text, usage2);
  }

  /* Legacy /v1/completions: the prompt is raw text (no chat template). */
  async function streamCompletion(cfg, res, data, modelPath) {
    var prompt = str(data.prompt);
    if (!prompt && Array.isArray(data.prompt)) prompt = data.prompt.join('\n');
    var maxTokens = Math.max(0, num(data.max_tokens, 512) | 0);
    var temperature = typeof data.temperature === 'number' ? data.temperature : 0.7;
    var stops = normalizeStops(data.stop);
    var wantStream = data.stream === true;
    var id = 'cmpl-' + Math.random().toString(36).slice(2, 12);
    var created = Math.floor(Date.now() / 1000);
    var modelName = str(data.model) || cfg.models[0] && cfg.models[0].id || baseName(modelPath);

    var iter = sofuu.ai.streamLocal(modelPath, prompt, {
      maxTokens: maxTokens,
      temperature: temperature,
    });

    if (wantStream) {
      sseHead(res);
      var sf = stopFilter(stops, function (text) {
        sseData(res, {
          id: id, object: 'text_completion', created: created, model: modelName,
          choices: [{ text: text, index: 0, finish_reason: null }],
        });
      });
      while (true) {
        var n = await iter.next();
        if (n.done) break;
        if (n.value && n.value.text && sf.push(n.value.text)) break;
      }
      sf.flush();
      var usage = usageOf(iter, sofuu.ai.estimateTokens(prompt));
      sseData(res, {
        id: id, object: 'text_completion', created: created, model: modelName,
        choices: [{ text: '', index: 0, finish_reason: finishReason([], sf.hit, usage, maxTokens) }],
      });
      if (data.stream_options && data.stream_options.include_usage) {
        sseData(res, { id: id, object: 'text_completion', created: created, choices: [], usage: usage });
      }
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }

    var text = '';
    var sf2 = stopFilter(stops, function (t) { text += t; });
    while (true) {
      var n2 = await iter.next();
      if (n2.done) break;
      if (n2.value && n2.value.text && sf2.push(n2.value.text)) break;
    }
    sf2.flush();
    var usage2 = usageOf(iter, sofuu.ai.estimateTokens(prompt));
    send(res, 200, {
      id: id, object: 'text_completion', created: created, model: modelName,
      choices: [{ text: text, index: 0, finish_reason: finishReason([], sf2.hit, usage2, maxTokens) }],
      usage: usage2,
    });
  }

  /* ── sessions (process-local; H3's minimum surface) ─────────── */

  function sessionFor(req, data) {
    var sid = str(data.session_id) || str(data.sessionId);
    if (!sid) {
      var h = (req.headers && (req.headers['x-sofuu-session'] || req.headers['X-Sofuu-Session'])) || '';
      sid = str(h);
    }
    return sid || '';
  }

  function recordTurn(data, prepared, answer, usage) {
    var sid = str(data.session_id) || str(data.sessionId);
    if (!sid || !S.sessions[sid]) return;
    var promptText = '';
    var msgs = prepared.messages || [];
    for (var i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i] && msgs[i].role === 'user') {
        promptText = typeof msgs[i].content === 'string' ? msgs[i].content : '';
        break;
      }
    }
    S.sessions[sid].turns.push({
      index: S.sessions[sid].turns.length,
      prompt: promptText,
      answer: answer,
      usage: usage,
      created_at: Math.floor(Date.now() / 1000),
    });
  }

  /* ── agent runs ─────────────────────────────────────────────── */

  async function agentRun(cfg, req, res, data, modelPath) {
    var task = str(data.task) || str(data.input);
    if (!task) {
      fail(res, 400, 'task is required', 'invalid_request_error', 'invalid_task', 'task');
      return;
    }
    var def = data.def;
    if (!def) {
      def = {
        name: str(data.agent) || 'http',
        provider: 'local',
        model: modelPath,
      };
    } else if (typeof def === 'string') {
      def = { name: def, provider: 'local', model: modelPath };
    }
    var wantStream = data.stream !== false;
    var id = 'run-' + Math.random().toString(36).slice(2, 12);
    var events = [];
    var opts = {
      provider: 'local',
      model: modelPath,
      onStep: function (ev) {
        events.push(ev);
        if (wantStream) {
          try { sseData(res, { run_id: id, event: ev }); } catch (eW) {}
        }
      },
    };
    if (wantStream) sseHead(res);
    try {
      var result = await sofuu.agent.run(def, task, opts);
      if (wantStream) {
        sseData(res, {
          run_id: id,
          done: true,
          answer: result && result.answer,
          usage: (result && result.usage) || null,
          steps: (result && result.steps) || events.length,
          stopped: !!(result && result.stopped),
        });
        res.write('data: [DONE]\n\n');
        res.end();
      } else {
        send(res, 200, {
          run_id: id,
          answer: result && result.answer,
          usage: (result && result.usage) || null,
          steps: (result && result.steps) || events.length,
          stopped: !!(result && result.stopped),
        });
      }
    } catch (eR) {
      if (wantStream) {
        sseData(res, { run_id: id, error: { message: String(eR && eR.message || eR), type: 'server_error' } });
        res.write('data: [DONE]\n\n');
        res.end();
      } else {
        fail(res, 500, String(eR && eR.message || eR), 'server_error', 'agent_run_failed');
      }
    }
  }

  /* ── tool routes ────────────────────────────────────────────── */

  function toolEntry(name) {
    var n = String(name || '');
    if (sofuu.tools && sofuu.tools.TOOLS && sofuu.tools.TOOLS[n]) {
      return { source: 'builtin', def: sofuu.tools.TOOLS[n] };
    }
    if (sofuu.web && sofuu.web.TOOLS && sofuu.web.TOOLS[n]) {
      return { source: 'web', def: sofuu.web.TOOLS[n] };
    }
    var host = globalThis.__sofuu_host_tools;
    if (host && host[n]) return { source: 'host', def: host[n] };
    return null;
  }

  async function toolCall(res, name, data) {
    var entry = toolEntry(name);
    if (!entry) {
      fail(res, 404, "no tool named '" + name + "'", 'invalid_request_error', 'tool_not_found', 'name');
      return;
    }
    var verdict = policyVerdict('/v1/tools/' + name + '/call', 'POST');
    if (verdict.decision === 'deny') { failPolicy(res, verdict); return; }
    var v = {};
    try {
      v = sofuu.policy ? JSON.parse(sofuu.policy.evaluate({ op: 'tool.run', tool: String(name) })) : {};
    } catch (eP) { v = {}; }
    if (v.decision === 'deny') { failPolicy(res, v); return; }
    var args = (data && (data.arguments || data.args)) || {};
    try {
      var started = Date.now();
      var out = await entry.def.execute(args);
      var text = (typeof out === 'string') ? out : JSON.stringify(out);
      send(res, 200, {
        ok: true,
        tool: String(name),
        source: entry.source,
        text: text,
        duration_ms: Date.now() - started,
      });
    } catch (eT) {
      fail(res, 500, String(eT && eT.message || eT), 'server_error', 'tool_failed');
    }
  }

  /* ── memory routes ──────────────────────────────────────────── */

  function brainOr503(res) {
    if (S.brain) return true;
    fail(res, 503, 'brain not available', 'server_error', 'brain_unavailable');
    return false;
  }

  /* ── the handler ────────────────────────────────────────────── */

  function createHandler(opts) {
    var cfg = normalize(opts);

    return async function handler(req, res) {
      try {
        var path = String(req.url || '/').split('?')[0];
        var method = String(req.method || 'GET').toUpperCase();

        /* Liveness is unauthenticated and carries no data (H2). */
        if (method === 'GET' && path === '/health') {
          send(res, 200, {
            ok: true,
            status: 'ok',
            uptime_ms: S.startedAtMs ? (Date.now() - S.startedAtMs) : 0,
            version: cfg.version,
          });
          return;
        }

        /* Capabilities manifest (R1.3, T1.2). */
        if (method === 'GET' && (path === '/capabilities' || path === '/v1/capabilities')) {
          var caps = typeof sofuu.capabilities === 'function' ? sofuu.capabilities() : '{}';
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(typeof caps === 'string' ? caps : JSON.stringify(caps));
          return;
        }

        if (authMode() !== 'none') {
          var auth = (req.headers && (req.headers.authorization || req.headers.Authorization)) || '';
          if (!ctEq(auth, 'Bearer ' + cfg.token)) {
            fail(res, 401, 'missing or invalid bearer token', 'authentication_error', 'invalid_api_key');
            return;
          }
        }

        /* Policy sees every authenticated request; denies win (R7.6). */
        var verdict = policyVerdict(path, method);
        if (verdict.decision === 'deny') { failPolicy(res, verdict); return; }

        /* ── models ── */
        if (method === 'GET' && path === '/v1/models') {
          send(res, 200, { object: 'list', data: modelList(cfg) });
          return;
        }

        /* ── chat / completions ── */
        if (method === 'POST' && (path === '/v1/chat/completions' || path === '/chat/completions')) {
          var body = readJson(req);
          if (!body.ok) { fail(res, 400, body.error, 'invalid_request_error', 'invalid_json'); return; }
          var data = body.data;
          if (!Array.isArray(data.messages)) {
            fail(res, 400, 'messages[] is required', 'invalid_request_error', 'invalid_messages', 'messages');
            return;
          }
          var mp = await resolveModel(cfg, data.model);
          if (!mp) {
            fail(res, 404, "model '" + str(data.model) + "' is not configured on this server",
              'invalid_request_error', 'model_not_found', 'model');
            return;
          }
          await streamChat(cfg, res, data, mp);
          return;
        }
        if (method === 'POST' && (path === '/v1/completions' || path === '/completions' || path === '/infer')) {
          var cbody = readJson(req);
          if (!cbody.ok) { fail(res, 400, cbody.error, 'invalid_request_error', 'invalid_json'); return; }
          var cdata = cbody.data;
          var cmp = await resolveModel(cfg, cdata.model);
          if (!cmp) {
            fail(res, 404, "model '" + str(cdata.model) + "' is not configured on this server",
              'invalid_request_error', 'model_not_found', 'model');
            return;
          }
          await streamCompletion(cfg, res, cdata, cmp);
          return;
        }

        /* ── embeddings (H2) ── */
        if (method === 'POST' && path === '/v1/embeddings') {
          var ebody = readJson(req);
          if (!ebody.ok) { fail(res, 400, ebody.error, 'invalid_request_error', 'invalid_json'); return; }
          var edata = ebody.data;
          var inputs = [];
          if (typeof edata.input === 'string') inputs = [edata.input];
          else if (Array.isArray(edata.input)) {
            for (var ii = 0; ii < edata.input.length; ii++) {
              if (typeof edata.input[ii] !== 'string') {
                fail(res, 400, 'input must be a string or an array of strings (token arrays are not supported)',
                  'invalid_request_error', 'invalid_input', 'input');
                return;
              }
              inputs.push(edata.input[ii]);
            }
          } else {
            fail(res, 400, 'input is required', 'invalid_request_error', 'invalid_input', 'input');
            return;
          }
          if (inputs.length === 0 || inputs.length > 64) {
            fail(res, 400, 'input must contain 1..64 strings', 'invalid_request_error', 'invalid_input', 'input');
            return;
          }
          var totalTok = 0;
          for (var ti = 0; ti < inputs.length; ti++) {
            var est = sofuu.ai.estimateTokens(inputs[ti]);
            if (est > 8192) {
              fail(res, 400, 'one input exceeds the 8192-token cap', 'invalid_request_error', 'input_too_long', 'input');
              return;
            }
            totalTok += est;
          }
          var out = [];
          try {
            for (var ei = 0; ei < inputs.length; ei++) {
              out.push({ object: 'embedding', index: ei, embedding: embeddingOf(inputs[ei]) });
            }
          } catch (eEmb) {
            fail(res, 500, String(eEmb && eEmb.message || eEmb), 'server_error', 'embedding_failed');
            return;
          }
          send(res, 200, {
            object: 'list',
            data: out,
            model: str(edata.model) || 'sofuu-embed',
            usage: { prompt_tokens: totalTok, total_tokens: totalTok },
          });
          return;
        }

        /* ── sessions (H3) ── */
        if (method === 'POST' && path === '/v1/sessions') {
          var sbody = readJson(req);
          var sdata = sbody.ok ? sbody.data : {};
          var sid = 'sess_' + Math.random().toString(36).slice(2, 12);
          S.sessions[sid] = {
            id: sid,
            object: 'session',
            title: str(sdata.title),
            created_at: Math.floor(Date.now() / 1000),
            turns: [],
          };
          send(res, 200, {
            id: sid,
            object: 'session',
            title: S.sessions[sid].title,
            created_at: S.sessions[sid].created_at,
          });
          return;
        }
        if (method === 'GET' && path === '/v1/sessions') {
          var list = [];
          for (var sk in S.sessions) {
            if (!S.sessions[sk]) continue;
            list.push({
              id: S.sessions[sk].id,
              object: 'session',
              title: S.sessions[sk].title,
              created_at: S.sessions[sk].created_at,
              turns: S.sessions[sk].turns.length,
            });
          }
          send(res, 200, { object: 'list', data: list });
          return;
        }
        var turnsMatch = /^\/v1\/sessions\/([^/]+)\/turns$/.exec(path);
        if (method === 'GET' && turnsMatch) {
          var wantId = '';
          try { wantId = decodeURIComponent(turnsMatch[1]); } catch (eDec) { wantId = turnsMatch[1]; }
          var sess = S.sessions[wantId];
          if (!sess) {
            fail(res, 404, "no session '" + wantId + "'", 'invalid_request_error', 'session_not_found');
            return;
          }
          send(res, 200, { id: sess.id, object: 'list', data: sess.turns });
          return;
        }

        /* ── agent runs (H3) ── */
        if (method === 'POST' && path === '/v1/agent/runs') {
          var abody = readJson(req);
          if (!abody.ok) { fail(res, 400, abody.error, 'invalid_request_error', 'invalid_json'); return; }
          var adata = abody.data;
          var amp = await resolveModel(cfg, adata.model);
          if (!amp) {
            fail(res, 404, "model '" + str(adata.model) + "' is not configured on this server",
              'invalid_request_error', 'model_not_found', 'model');
            return;
          }
          await agentRun(cfg, req, res, adata, amp);
          return;
        }

        /* ── tools (H3) ── */
        if (method === 'GET' && path === '/v1/tools') {
          var queryCat = null;
          try {
            var qIdx = String(req.url || '').indexOf('?');
            if (qIdx >= 0) {
              var qStr = String(req.url).slice(qIdx + 1);
              var matchCat = /[?&](?:category|pack)=([^&]+)/.exec('?' + qStr);
              if (matchCat) queryCat = decodeURIComponent(matchCat[1]);
            }
          } catch (eQ) {}
          var inv = (sofuu.tools && typeof sofuu.tools.list === 'function')
            ? sofuu.tools.list(queryCat ? { category: queryCat } : {}) : [];
          send(res, 200, { object: 'list', data: inv });
          return;
        }
        var callMatch = /^\/v1\/tools\/([^/]+)\/call$/.exec(path);
        if (method === 'POST' && callMatch) {
          var tbody = readJson(req);
          if (!tbody.ok) { fail(res, 400, tbody.error, 'invalid_request_error', 'invalid_json'); return; }
          var tname = '';
          try { tname = decodeURIComponent(callMatch[1]); } catch (eTd) { tname = callMatch[1]; }
          await toolCall(res, tname, tbody.data);
          return;
        }

        /* ── memory ── */
        if (method === 'POST' && path === '/remember') {
          var mbody = readJson(req);
          if (!mbody.ok) { fail(res, 400, mbody.error, 'invalid_request_error', 'invalid_json'); return; }
          var mdata = mbody.data;
          if (!mdata.text) { fail(res, 400, 'text required', 'invalid_request_error', 'invalid_text', 'text'); return; }
          if (!brainOr503(res)) return;
          try {
            var vec = brainEmbed()(String(mdata.text));
            var role = str(mdata.role) || 'user';
            S.brain.remember(new Float32Array(vec), String(mdata.text), role, 0, str(mdata.source) || 'server');
            S.brain.flush();
            send(res, 200, { ok: true, memories: S.brain.count() });
          } catch (eRm) {
            fail(res, 500, String(eRm && eRm.message || eRm), 'server_error', 'remember_failed');
          }
          return;
        }
        if (method === 'GET' && path === '/recall') {
          var qs = parseQuery(req.url);
          if (!qs.q) { fail(res, 400, 'q parameter required', 'invalid_request_error', 'invalid_query', 'q'); return; }
          if (!brainOr503(res)) return;
          try {
            var rvec = brainEmbed()(qs.q);
            var k = parseInt(qs.k || '5', 10);
            send(res, 200, { ok: true, results: S.brain.recall(new Float32Array(rvec), k) });
          } catch (eRc) {
            fail(res, 500, String(eRc && eRc.message || eRc), 'server_error', 'recall_failed');
          }
          return;
        }
        if (method === 'GET' && path === '/memories') {
          if (!brainOr503(res)) return;
          var mqs = parseQuery(req.url);
          var lopts = {};
          if (mqs.tier !== undefined) lopts.tier = parseInt(mqs.tier, 10) || 0;
          if (mqs.role) lopts.role = mqs.role;
          if (mqs.entity) lopts.entity = mqs.entity;
          if (mqs.text) lopts.text = mqs.text;
          if (mqs.offset) lopts.offset = parseInt(mqs.offset, 10) || 0;
          lopts.limit = mqs.limit ? (parseInt(mqs.limit, 10) || 20) : 20;
          try {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(S.brain.list(lopts));
          } catch (eL) {
            fail(res, 500, String(eL && eL.message || eL), 'server_error', 'list_failed');
          }
          return;
        }
        if (method === 'POST' && path === '/forget') {
          var fbody = readJson(req);
          if (!fbody.ok) { fail(res, 400, fbody.error, 'invalid_request_error', 'invalid_json'); return; }
          if (!brainOr503(res)) return;
          try {
            var fid = parseInt(fbody.data.id, 10);
            if (!(fid >= 0)) { fail(res, 400, 'id required', 'invalid_request_error', 'invalid_id', 'id'); return; }
            var forgot = S.brain.forget(fid);
            S.brain.flush();
            send(res, 200, { ok: true, forgot: forgot, id: fid });
          } catch (eF) {
            fail(res, 500, String(eF && eF.message || eF), 'server_error', 'forget_failed');
          }
          return;
        }
        if (method === 'POST' && path === '/share') {
          if (!brainOr503(res)) return;
          try {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(S.brain.export());
          } catch (eS) {
            fail(res, 500, String(eS && eS.message || eS), 'server_error', 'share_failed');
          }
          return;
        }

        fail(res, 404, "unknown route '" + path + "'", 'invalid_request_error', 'not_found');
      } catch (e) {
        try {
          fail(res, 500, String(e && e.message || e), 'server_error', 'internal_error');
        } catch (e2) { /* headers already sent mid-stream — nothing to add */ }
      }
    };
  }

  /* ── start ──────────────────────────────────────────────────── */

  function start(opts) {
    var cfg = normalize(opts || {});
    /* R7.1: the HTTP surface answers to the SAME policy as every other one —
     * a project policy (.sofuu/policy.json) installs here at boot, so a deny
     * cannot be bypassed by entering through serve instead of chat/CLI. A
     * missing or broken policy file keeps whatever is already installed. */
    try {
      if (sofuu.policy && typeof sofuu.policy.set === 'function' &&
          typeof __session_read_file === 'function') {
        var praw = __session_read_file('.sofuu/policy.json');
        if (praw) { try { sofuu.policy.set(String(praw)); } catch (ePol) {} }
      }
    } catch (ePl) {}
    var auth = authMode();
    if (auth !== 'none' && !cfg.token) {
      throw new Error('serve: a bearer token is required (set --token / SOFUU_SERVE_TOKEN, or policy http.auth:"none" explicitly)');
    }
    if (auth === 'none') {
      console.error('\x1b[33m  ⚠ policy http.auth is "none": every /v1/* request is unauthenticated\x1b[0m');
    }
    S.cfg = cfg;
    S.startedAt = Math.floor(Date.now() / 1000);
    S.startedAtMs = Date.now();
    S.brain = brainOpen(cfg);
    var handler = createHandler(cfg);
    var server = sofuu.http.createServer(handler);
    server.listen(cfg.port, cfg.host);
    S.server = server;
    S.started = true;
    var primary = cfg.models.length ? cfg.models[0].id : null;
    console.log('brain server listening on http://' + cfg.host + ':' + cfg.port +
      (primary ? ' (model: ' + primary + ')' : ''));
    return { ok: true, host: cfg.host, port: cfg.port, model: primary };
  }

  sofuu.serve.start = start;
  sofuu.serve.createHandler = createHandler;
  sofuu.serve.state = function () {
    return { started: S.started, sessions: Object.keys(S.sessions).length };
  };
  sofuu.serve.VERSION = '1';
})();
