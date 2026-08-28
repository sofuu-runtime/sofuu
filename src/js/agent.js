/* src/js/agent.js — sofuu.agent: first-class agents + sub-agents
 * (PLAN-AGENTS A1–A5, A8 API, A9).
 *
 * Shipped JS, eval'd once per engine context by the Rust seam
 * (crates/sofuu-core/src/shipped.rs). Pure JS over the existing
 * primitives — sofuu.ai.complete/stream, sofuu.mcp, sofuu.memory,
 * sofuu.rlm, sofuu.web, fetch — zero new C, zero new host functions.
 *
 * Merges into the LIVE sofuu.agent object (memory.rs already registered
 * agent.create/prefetch there) — nothing is replaced.
 *
 * Design principles (PLAN-AGENTS, frozen):
 *   1. Agents-as-tools: delegation is a `delegate` tool call; no DSL.
 *   2. Runtime-level, headless-first: registered for every engine
 *      context; chat is just a renderer over this loop.
 *   3. One loop, single-threaded structured concurrency (Promise pools).
 *   4. Budgets + cancellation are part of the v1 API.
 *   5. Memory scopes shared/agent/off over one physical brain file.
 *
 * API:
 *   sofuu.agent.define(def) → handle
 *     def: { name*, system, tools:[spec|{mcp:[{name,command}]}|'web'],
 *            provider, model, effort, memory:'shared'|'agent'|'off',
 *            brainPath, budget:{maxSteps,maxDepth,maxTokens,maxWallMs},
 *            agents:[subAgentNames], identity:{role,expertise},
 *            rlm:'auto'|'on'|'off' }
 *   sofuu.agent.run(handleOrNameOrDef, task, opts) → Promise<AgentResult>
 *     opts: { context, history, signal(cancelId), onStep(evt), plain,
 *             provider/model/effort/api_key/base_url/profile/max_tokens,
 *             toolTimeoutMs, logs }
 *   AgentResult: { answer, steps, subRuns, usage, trace, stopped, runId, name }
 *   sofuu.agent.runMany([{agent,task,opts}], {concurrency:4})
 *   sofuu.agent.mapContext(context, task, {agent, chunkChars, concurrency, reduceAgent})
 *   sofuu.agent.cancel(id) · list() · loadDir(path?) · renderTrace(result)
 */
(function () {
  'use strict';
  var g = globalThis;
  if (typeof g.sofuu === 'undefined' || g.sofuu === null) g.sofuu = {};
  var sofuu = g.sofuu;
  if (!sofuu.agent) sofuu.agent = {};
  var A = sofuu.agent;
  if (A._agents_loaded) return;

  var VERSION = '0.1.0';
  /* P1 (PLAN-MEMORY-TOKENS): the ONE static core prompt — identity +
   * discipline only (~25 tokens). Byte-stable across turns so every
   * provider's prefix cache (OpenAI auto-cache, Anthropic markers, local
   * KV reuse) can anchor on it. def.system composes AFTER it; nothing
   * dynamic ever enters the system message. */
  var CORE_PROMPT = 'Sofuu coding agent. Be direct and concise. Verify with tools before asserting. If unsure, say so.';
  var BUDGET_DEFAULTS = { maxSteps: 12, maxDepth: 2, maxTokens: 200000, maxWallMs: 300000 };
  var TOOL_TIMEOUT_MS = 30000;
  var ANSWER_CAP_CHARS = 200000;
  /* Transient provider failures worth a bounded automatic retry: shared-pool
   * rate limits (HTTP 429 — as a status OR as an in-stream SSE error frame),
   * gateway 5xx, and connection-level hiccups — including the libcurl
   * transport-error wordings (verified against curl_easy_strerror on libcurl
   * 8.7.1; older wordings kept as alternates): HTTP/2 resets ("Stream error
   * in the HTTP/2 framing layer" code 92, "Error in the HTTP2 framing layer"
   * code 16, "HTTP/2 stream error" on older curl) and the connection getting
   * cut mid-transfer ("Transferred a partial file" / "Transfer closed with
   * outstanding read data remaining" code 18, "Server returned nothing (no
   * headers, no data)" / "Empty reply from server" code 52, "Failure when
   * receiving data from the peer" code 56, "Failed sending data to the peer"
   * code 55). Shared-pool models (OpenRouter free tier et al.) hit these
   * intermittently on long turns; without a retry the user just re-types the
   * same question. Anything else (auth, bad request, model not found) throws
   * immediately — retrying those only burns time. */
  var STREAM_RETRY_DELAYS = [1500, 4000];
  var STREAM_TRANSIENT_RE = /HTTP (429|5\d\d)|rate.?limit|temporarily|overloaded|timed? ?out|connection (reset|refused)|couldn'?t connect|ECONNRESET|ETIMEDOUT|http\/?2 (stream|framing)|transfer closed|partial file|server returned nothing|empty reply from server|(receiving|sending) data (from|to) the peer/i;
  /* Classifier for the retry policy above — used by streamWithRetry and
   * exported as sofuu.agent.isTransientProviderError for tests/consumers. */
  function isTransientProviderError(msg) {
    return STREAM_TRANSIENT_RE.test(String(msg == null ? '' : msg));
  }
  /* Tool results are the largest uncontrolled token source — cap each one
   * before it enters any message (head+tail kept, middle elided with a
   * recovery hint; per-def override budget.maxToolResultChars). The cap
   * SCALES with the model's context window (≈ win/8 chars, i.e. ~2% of
   * the window in tokens) between sane bounds — no flat constant. */
  function toolResultCapChars(d) {
    if (d && d.budget && d.budget.maxToolResultChars > 0) return d.budget.maxToolResultChars;
    var win = contextWindow(d && d.model);
    return Math.min(24576, Math.max(TOOL_RESULT_CAP_CHARS, Math.floor(win / 8)));
  }
  var TOOL_RESULT_CAP_CHARS = 4000; /* floor for tiny windows */
  var TOOL_RESULT_HEAD_RATIO = 0.7;
  var TOOL_RESULT_TAIL_RATIO = 0.2;
  /* Recall gating — similarity floor + block token budget (defaults;
   * per-def recallMin / recallBudget override). Calibrated for the bundled
   * TF-IDF embedder where unrelated text scores ~0.27 and paraphrases >0.35
   * on the composite score scale. The budget SCALES with the window
   * (≈ 2% of it) between sane bounds. */
  var RECALL_MIN_SCORE = 0.30;
  var RECALL_BUDGET_TOK = 1024; /* floor for tiny windows */
  function recallBudgetTok(d) {
    if (d && d.recallBudget > 0) return d.recallBudget;
    var win = contextWindow(d && d.model);
    return Math.min(16384, Math.max(RECALL_BUDGET_TOK, Math.floor(win * 0.02)));
  }
  /* P4: distilled memory writes — conclusions only, not raw dumps. */
  var STORE_CAP_CHARS = 1200;
  /* M4: trace events per run (RLM parity); ring keeps the first-16
   * skeleton + the freshest events beyond that. */
  var TRACE_CAP = 512;
  /* Module state is deliberately tiny (PLAN-AGENTS isolation rule): the
   * immutable registry, run/cancel tables, and infrastructure caches
   * (one brain handle per file, one MCP connection per command). Run
   * state lives in closures, never on the module. */
  var REGISTRY = {};   // name → normalized def (frozen)
  var NEXT_RUN = 1;
  var ACTIVE = {};     // runId → { cancelled, aborts:[] }
  var SIGNALS = {};    // external cancelId → runId
  var ACTIVE_COUNT = 0;
  /* M3: one open handle per brain file, each with its own decay clock —
   * Ebbinghaus decay runs at the turn boundary of the FIRST run after a
   * gap (the runtime owns recall/store, so it owns decay too). */
  var BRAINS = {};     // brainPath → { cma, lastDecay } | null
  var MCP_POOL = {};   // command → Promise<{client, tools}>
  /* M5: honest run counters — {n, day}; n resets when the UTC day rolls. */
  var RUN_COUNTS = {}; // name → {n, day}

  function env(name) {
    try { return (typeof process !== 'undefined' && process.env && process.env[name]) || ''; }
    catch (e) { return ''; }
  }
  /* PLAN-ML-GATES §7: local calendar date (YYYY-MM-DD) for the ephemeral
   * context message — the model is never told today's date anywhere else,
   * and no freshness judgment is possible without it. */
  function isoDate() {
    var dt = new Date();
    var p = function (n) { return (n < 10 ? '0' : '') + n; };
    return dt.getFullYear() + '-' + p(dt.getMonth() + 1) + '-' + p(dt.getDate());
  }
  /* Embedded hosts inject __sofuu_embed_config via the capi seam
   * (PLAN-HEADLESS H2.4): { embedded, configRoot, brainPath }. */
  function embedCfg() {
    return (g.__sofuu_embed_config && typeof g.__sofuu_embed_config === 'object')
      ? g.__sofuu_embed_config : null;
  }
  function sofuuDir() {
    var ec = embedCfg();
    return (ec && ec.configRoot) || ((env('HOME') || env('USERPROFILE') || '.') + '/.sofuu');
  }
  function clip(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n) + '…' : s;
  }
  function estTok(s) {
    if (sofuu.ai && typeof sofuu.ai.estimateTokens === 'function') {
      try { return sofuu.ai.estimateTokens(s) | 0; } catch (e) {}
    }
    return Math.ceil(String(s).length / 4);
  }
  function withTimeout(p, ms, msg) {
    return new Promise(function (resolve, reject) {
      var t = setTimeout(function () { reject(new Error(msg)); }, ms);
      p.then(function (v) { clearTimeout(t); resolve(v); },
             function (e) { clearTimeout(t); reject(e); });
    });
  }
  /* M5: day-stamped counter bump — the "last 24h" semantics are real now. */
  function bumpRunCount(name) {
    var day = Math.floor(Date.now() / 86400000);
    var e = RUN_COUNTS[name];
    if (!e || e.day !== day) { RUN_COUNTS[name] = { n: 1, day: day }; return; }
    e.n++;
  }

  /* ── Per-model context windows ────────────────────────────────────
   * The canonical source is the native capability registry
   * (`sofuu.ai.modelCaps`, rt/model_caps.rs). This prefix table remains
   * only as an offline fallback for hosts without the binding. Prefix
   * rules, first match wins; def.ctx_window (/ctx) still overrides. */
  var MODEL_CTX = [
    ['gpt-5', 400000], ['gpt-4.1', 1000000], ['gpt-4o', 128000], ['gpt-4-turbo', 128000], ['gpt-4', 128000],
    ['o3', 200000], ['o4', 200000], ['o1', 200000],
    ['claude-opus-4', 200000], ['claude-sonnet-4', 200000], ['claude-haiku-4', 200000],
    ['claude-3-7', 200000], ['claude-3-5', 200000], ['claude-3', 200000], ['claude', 200000],
    ['deepseek', 65536],
    ['qwen3', 131072], ['qwen2.5', 32768], ['qwen', 32768],
    ['llama4', 1000000], ['llama3.1', 131072], ['llama3', 8192], ['llama', 8192],
    ['mistral', 32768], ['mixtral', 32768],
    ['grok-4', 256000], ['grok', 131072],
  ];
  var DEFAULT_CTX_WINDOW = 32768;

  var _capsCache = {};
  function modelCaps(model) {
    var m = String(model == null ? '' : model);
    if (!m) return null;
    if (Object.prototype.hasOwnProperty.call(_capsCache, m)) return _capsCache[m];
    var caps = null;
    try {
      if (sofuu.ai && typeof sofuu.ai.modelCaps === 'function') {
        caps = JSON.parse(sofuu.ai.modelCaps(m));
        if (!caps || !caps.known) caps = null;
      }
    } catch (e) { caps = null; }
    _capsCache[m] = caps;
    return caps;
  }

  function contextWindow(model) {
    var m = String(model == null ? '' : model).toLowerCase();
    if (!m) return DEFAULT_CTX_WINDOW;
    var caps = modelCaps(m);
    if (caps && caps.ctxWindow > 0) return caps.ctxWindow;
    for (var i = 0; i < MODEL_CTX.length; i++) {
      if (m.indexOf(MODEL_CTX[i][0]) === 0 || m.indexOf('/' + MODEL_CTX[i][0]) >= 0) {
        return MODEL_CTX[i][1];
      }
    }
    return DEFAULT_CTX_WINDOW;
  }

  /* Effective per-response output cap for a def: explicit override wins,
   * then the model's published max output from the registry. 0 = unknown
   * → callers omit the field so the endpoint applies its own default. */
  function maxOutputFor(d) {
    if (d.maxTokensOut > 0) return d.maxTokensOut;
    var caps = modelCaps(d.model);
    return (caps && caps.maxOutput > 0) ? caps.maxOutput : 0;
  }

  /* ── Definitions ───────────────────────────────────────────────── */

  function normalizeDef(def) {
    if (!def || typeof def !== 'object') throw new Error('agent: definition object required');
    var name = String(def.name || '').trim();
    if (!name) throw new Error('agent.define: name is required');
    var d = {
      _sofuu_agent: true,
      name: name,
      /* P1: no persona padding by default — CORE_PROMPT only; the identity
       * line (below, run-time) carries the name when identity is set.
       * A custom def.system COMPOSES AFTER the core (never replaces it) so
       * the discipline lines survive user overrides; passing the core
       * itself (the chat def does) is a no-op, not a duplication. */
      system: (def.system && String(def.system) !== CORE_PROMPT)
        ? CORE_PROMPT + '\n' + String(def.system)
        : CORE_PROMPT,
      provider: def.provider || '',
      model: def.model || '',
      effort: def.effort || '',
      apiKey: def.api_key || def.apiKey || '',
      baseUrl: def.base_url || def.baseUrl || '',
      profile: def.profile || '',
      maxTokensOut: (def.max_tokens || def.maxTokensOut || 0) | 0,
      ctxWindow: (def.ctx_window || def.ctxWindow || 0) | 0,
      /* P2: recall gating knobs — 0 = scale with the model's window. */
      recallMin: (typeof def.recallMin === 'number') ? def.recallMin : RECALL_MIN_SCORE,
      recallBudget: (def.recallBudget | 0) || 0,
      /* Optional REMOTE embeddings for the brain (local-first without). */
      embedProvider: def.embed_provider || def.embedProvider || '',
      embedModel: def.embed_model || def.embedModel || '',
      memory: (def.memory === 'shared' || def.memory === 'off') ? def.memory : 'agent',
      brainPath: def.brainPath || '',
      rlm: (def.rlm === 'on' || def.rlm === 'off') ? def.rlm : 'auto',
      /* PLAN-ML-GATES: the context-economy gates (date fix, supervisor
       * rules, later the four models). 'off' disables every gate for this
       * def (chat /ml off); SOFUU_NO_ML=1 unregisters sofuu.ml entirely. */
      ml: def.ml === 'off' ? 'off' : 'on',
      identity: (def.identity && typeof def.identity === 'object')
        ? { role: String(def.identity.role || ''), expertise: String(def.identity.expertise || '') }
        : null,
      budget: {},
      agents: (Array.isArray(def.agents) ? def.agents : []).map(String),
      inlineTools: [],
      mcpSets: [],
      webTools: [],
      codeTools: [],
      toolTimeoutMs: (def.toolTimeoutMs | 0) || TOOL_TIMEOUT_MS,
    };
    var b = def.budget && typeof def.budget === 'object' ? def.budget : {};
    d.budget.maxSteps = (b.maxSteps | 0) || BUDGET_DEFAULTS.maxSteps;
    d.budget.maxDepth = (b.maxDepth | 0) || BUDGET_DEFAULTS.maxDepth;
    d.budget.maxTokens = (b.maxTokens | 0) || BUDGET_DEFAULTS.maxTokens;
    d.budget.maxWallMs = (b.maxWallMs | 0) || BUDGET_DEFAULTS.maxWallMs;
    /* Per-def tool-result context cap — 0 = scale with the model window. */
    d.budget.maxToolResultChars = (b.maxToolResultChars | 0) || 0;

    var tools = def.tools || [];
    if (!Array.isArray(tools)) throw new Error('agent.define: tools must be an array');
    for (var i = 0; i < tools.length; i++) {
      var t = tools[i];
      if (t === 'web') { d.webTools = ['web_search', 'web_open']; continue; }
      if (t === 'code' && sofuu.tools && sofuu.tools.GROUP) { d.codeTools = sofuu.tools.GROUP.slice(); continue; }
      if (t && typeof t === 'object' && Array.isArray(t.mcp)) {
        for (var j = 0; j < t.mcp.length; j++) {
          var s = t.mcp[j];
          if (!s || !s.command) throw new Error('agent.define: mcp set entry needs {name, command}');
          d.mcpSets.push({ name: String(s.name || ('mcp' + (j + 1))), command: String(s.command) });
        }
        continue;
      }
      if (t && t.builtin) {
        var bn = String(t.builtin);
        if (sofuu.web && sofuu.web.TOOLS && sofuu.web.TOOLS[bn]) {
          if (d.webTools.indexOf(bn) < 0) d.webTools.push(bn);
        } else if (sofuu.tools && sofuu.tools.TOOLS && sofuu.tools.TOOLS[bn]) {
          if (d.codeTools.indexOf(bn) < 0) d.codeTools.push(bn);
        } else {
          throw new Error("agent.define: unknown builtin tool '" + bn + "'");
        }
        continue;
      }
      if (t && t.name) {
        d.inlineTools.push({
          name: String(t.name),
          description: String(t.description || ''),
          parameters: t.parameters || { type: 'object', properties: {} },
          execute: (typeof t.execute === 'function') ? t.execute : null,
        });
        continue;
      }
      throw new Error('agent.define: bad tools[' + i + '] (need {name,…}, {mcp:[…]}, {builtin}, or "web")');
    }
    Object.freeze(d.budget);
    Object.freeze(d.identity || {});
    Object.freeze(d);
    return d;
  }

  function define(def) {
    var d = normalizeDef(def);
    if (REGISTRY[d.name]) throw new Error("agent.define: duplicate name '" + d.name + "'");
    REGISTRY[d.name] = d;
    return { name: d.name, _def: d };
  }

  function resolveDef(target) {
    if (typeof target === 'string') {
      var d = REGISTRY[target];
      if (!d) throw new Error("agent: unknown agent '" + target + "'");
      return d;
    }
    if (target && target._def) return target._def;
    if (target && target._sofuu_agent) return target;
    if (target && typeof target === 'object') return normalizeDef(target); // transient def (e.g. chat)
    throw new Error('agent: target must be a name, handle, or definition');
  }

  /* ── Memory scopes (A3) ──────────────────────────────────────────
   * One physical brain file; scope decides what a run sees/writes:
   *   shared → session brain, current behavior (all memories)
   *   agent  → records under entity "agent:<name>" (private namespace,
   *            rememberEntity + recall filtered to that entity)
   *   off    → no recall, no store, provably no writes                */

  var VEC_DIM = 0;
  function vecDim() {
    if (VEC_DIM > 0) return VEC_DIM;
    if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
      try { var v = sofuu.ai.embedLocal(''); if (v && v.length) { VEC_DIM = v.length; return VEC_DIM; } } catch (e) {}
    }
    VEC_DIM = 768;
    return VEC_DIM;
  }
  function toF32(obj) {
    if (obj instanceof Float32Array) return obj;
    if (Array.isArray(obj)) return new Float32Array(obj);
    var dim = vecDim();
    var arr = new Float32Array(dim);
    for (var i = 0; i < dim; i++) arr[i] = obj[i] || 0;
    return arr;
  }
  /* Local-first embeddings (chat parity): a REMOTE embeddings provider is
   * used only when explicitly configured on the definition, with a
   * dimension check — a mismatch falls back to the local embedder rather
   * than corrupting the brain. */
  async function embedTextFor(d, text) {
    if (d.embedProvider && d.embedModel && sofuu.ai && typeof sofuu.ai.embed === 'function') {
      try {
        var e = await sofuu.ai.embed(String(text), { provider: d.embedProvider, model: d.embedModel });
        if (e) {
          var vec = (e[0] && typeof e[0] !== 'number') ? e[0] : e;
          var n = (vec && vec.length) ? vec.length : 0;
          if (n === vecDim()) return toF32(vec);
        }
      } catch (e) {}
    }
    if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
      try { var v = sofuu.ai.embedLocal(String(text)); if (v) return v; } catch (e2) {}
    }
    return null;
  }
  function brainFor(d) {
    if (d.memory === 'off' || !sofuu.memory || !sofuu.memory.open) return null;
    var ec = embedCfg();
    var path = d.brainPath || (ec && ec.brainPath) ||
      ((env('HOME') || env('USERPROFILE') || '.') + '/.sofuu_brain.qtsq');
    if (BRAINS[path] !== undefined) return BRAINS[path];
    var entry = null;
    try {
      entry = { cma: sofuu.memory.open(path, vecDim()), lastDecay: Date.now(), lastConsolidate: 0 };
    } catch (e) { entry = null; }
    BRAINS[path] = entry;
    return entry;
  }
  /* M3: apply elapsed-time Ebbinghaus decay since the last turn on this
   * brain, then physically prune dead records (retain is a no-op when the
   * binding is missing on older builds). Called at the turn boundary —
   * survivors renumber ids, which is safe because no caller holds ids
   * across turns. Returns the number of clusters consolidated this tick
   * (0 = nothing due). */
  function brainDecayTick(entry) {
    if (!entry || !entry.cma || typeof entry.cma.decayTick !== 'function') return 0;
    var consolidated = 0;
    try {
      var nowMs = Date.now();
      var dt = Math.floor((nowMs - entry.lastDecay) / 1000);
      if (dt > 0) entry.cma.decayTick(Math.min(dt, 7 * 86400));
      entry.lastDecay = nowMs;
      if (typeof entry.cma.retain === 'function') entry.cma.retain();
      consolidated = brainConsolidateIfDue(entry);
    } catch (e) {}
    return consolidated;
  }
  /* Consolidation cadence: cluster weak-old episodic memories into
   * semantic records (the README's "k-means consolidation" — previously
   * callable only from scripts). Conservative by design: rate-limited to
   * once per brain per interval, and thresholded so small brains never
   * pay a k-means pass. A run that consolidates flushes immediately so
   * the merge survives a crash. */
  var CONSOLIDATE_INTERVAL_MS = 6 * 3600 * 1000;
  var CONSOLIDATE_MIN_RECORDS = 24;
  function brainConsolidateIfDue(entry) {
    if (!entry || !entry.cma || typeof entry.cma.consolidate !== 'function') return 0;
    try {
      var now = Date.now();
      if (entry.lastConsolidate && (now - entry.lastConsolidate) < CONSOLIDATE_INTERVAL_MS) return 0;
      if (entry.cma.count() < CONSOLIDATE_MIN_RECORDS) return 0;
      entry.lastConsolidate = now;
      var n = entry.cma.consolidate() || 0;
      if (n > 0) { try { entry.cma.flush(); } catch (e2) {} }
      return n;
    } catch (e) { return 0; }
  }
  function scopeRecall(d, cma, vec, topK) {
    if (!vec) return [];
    var hits = cma.recall(vec, topK || 15) || [];
    if (d.memory === 'agent') {
      var want = 'agent:' + d.name;
      hits = hits.filter(function (h) { return h && h.entity === want; });
    }
    return hits;
  }
  /* P4: distilled writes — clip to STORE_CAP_CHARS (the head of an answer
   * carries the conclusion; recall can't use the elaboration tail). */
  function scopeStore(d, cma, taskVec, ansVec, task, answer) {
    if (d.memory === 'agent') {
      var ent = 'agent:' + d.name;
      if (taskVec) cma.rememberEntity(taskVec, clip(task, STORE_CAP_CHARS), ent, 'agent');
      if (ansVec) cma.rememberEntity(ansVec, clip(answer, STORE_CAP_CHARS), ent, 'agent');
      try { cma.flush(); } catch (e) {}
      return;
    }
    if (taskVec) cma.remember(taskVec, clip(task, STORE_CAP_CHARS), 'user', 0);
    if (ansVec) cma.remember(ansVec, clip(answer, STORE_CAP_CHARS), 'assistant', 0);
    try { cma.flush(); } catch (e) {}
  }
  async function rememberIdentity(d, cma) {
    if (!d.identity || !cma || !cma.rememberEntity) return;
    try {
      var v = await embedTextFor(d, d.name + ' ' + d.identity.role + ' ' + d.identity.expertise);
      if (v) {
        cma.rememberEntity(v,
          'agent identity — ' + d.name +
          (d.identity.role ? ' · role: ' + d.identity.role : '') +
          (d.identity.expertise ? ' · expertise: ' + d.identity.expertise : ''),
          'agent:' + d.name, 'agent-identity');
        cma.flush();
      }
    } catch (e) {}
  }

  /* ── MCP pool (lazy, one connection per command; idle-disconnect so
   *    child processes never pin the event loop at exit) ─────────── */

  function mcpConnect(command) {
    if (MCP_POOL[command]) return MCP_POOL[command];
    if (!sofuu.mcp || !sofuu.mcp.connect) return Promise.reject(new Error('sofuu.mcp unavailable'));
    MCP_POOL[command] = (async function () {
      var client = await sofuu.mcp.connect(command);
      var res = await client.listTools();
      return { client: client, tools: (res && res.tools) || [] };
    })();
    MCP_POOL[command].catch(function () { delete MCP_POOL[command]; });
    return MCP_POOL[command];
  }
  function mcpIdleDisconnect() {
    if (ACTIVE_COUNT > 0) return;
    var pool = MCP_POOL; MCP_POOL = {};
    Object.keys(pool).forEach(function (cmd) {
      pool[cmd].then(function (c) { try { c.client.disconnect(); } catch (e) {} },
                      function () {});
    });
  }
  function runTick(fn) { try { setTimeout(fn, 0); } catch (e) { fn(); } }

  /* ── Tool resolution (A6 rules: inline > builtin > MCP; name→owner
   *    map replaces the old first-server-wins routing) ───────────── */

  async function resolveTools(d, emit) {
    var specs = [];
    var exec = {};
    var names = {};
    function warn(payload) { emit('warn', payload); }
    function add(spec, fn) {
      if (names[spec.name]) { warn({ tool: spec.name, note: 'duplicate tool name skipped' }); return; }
      names[spec.name] = 1;
      specs.push(spec);
      exec[spec.name] = fn;
    }
    for (var i = 0; i < d.inlineTools.length; i++) {
      var t = d.inlineTools[i];
      add({ name: t.name, description: t.description || ('inline tool ' + t.name),
            parameters: t.parameters },
          t.execute || null);
    }
    for (var w = 0; w < d.webTools.length; w++) {
      var bn = d.webTools[w];
      if (sofuu.web && sofuu.web.TOOLS && sofuu.web.TOOLS[bn]) {
        var bt = sofuu.web.TOOLS[bn];
        add({ name: bt.name, description: bt.description, parameters: bt.parameters }, bt.execute);
      }
    }
    for (var ct = 0; ct < d.codeTools.length; ct++) {
      var cn = d.codeTools[ct];
      if (sofuu.tools && sofuu.tools.TOOLS && sofuu.tools.TOOLS[cn]) {
        var cd = sofuu.tools.TOOLS[cn];
        add({ name: cd.name, description: cd.description, parameters: cd.parameters }, cd.execute);
      }
    }
    for (var m = 0; m < d.mcpSets.length; m++) {
      var set = d.mcpSets[m];
      var conn;
      try { conn = await mcpConnect(set.command); }
      catch (e) { warn({ mcp: set.name, note: 'unreachable: ' + clip(String(e.message || e), 120) }); continue; }
      for (var k = 0; k < conn.tools.length; k++) {
        var mt = conn.tools[k];
        var mname = String(mt.name || '');
        if (!mname) continue;
        if (names[mname]) {
          /* A6: inline/builtin wins on a clash; the MCP copy is shadowed. */
          warn({ tool: mname, mcp: set.name, note: 'name clash — inline tool wins, MCP tool shadowed' });
          continue;
        }
        names[mname] = 1;
        specs.push({ name: mname, description: String(mt.description || ('MCP tool from ' + set.name)),
                     parameters: mt.inputSchema || { type: 'object', properties: {} } });
        exec[mname] = (function (client, toolName) {
          return async function (args) {
            var res = await client.call('tools/call', { name: toolName, arguments: args || {} });
            var text = res && res.content && res.content[0] && res.content[0].text;
            return (typeof text === 'string') ? text : JSON.stringify(text);
          };
        })(conn.client, mname);
      }
    }
    return { specs: specs, exec: exec };
  }

  function delegateSpecFor(d, chain) {
    var allowed = d.agents.filter(function (n) { return chain.indexOf(n) < 0; });
    if (!allowed.length) return null;
    return {
      name: 'delegate',
      description: 'Hand a well-scoped subtask to a specialist agent. Write the subtask so it can be answered independently; the agent\'s answer comes back as the tool result.',
      parameters: {
        type: 'object',
        properties: {
          agent: { type: 'string', enum: allowed.slice(), description: 'Which sub-agent to run' },
          task: { type: 'string', description: 'The complete, self-contained subtask' },
          why: { type: 'string', description: 'One line on why you are delegating' },
        },
        required: ['agent', 'task'],
      },
    };
  }

  /* ── The loop (A1) ─────────────────────────────────────────────── */

  function aiOpts(d, opts, messages, withTools) {
    var o = { messages: messages };
    function pick(key, jsKey) {
      if (opts && opts[key]) { o[jsKey] = opts[key]; return; }
      if (d[key]) o[jsKey] = d[key];
    }
    if (opts && opts.provider) o.provider = opts.provider; else if (d.provider) o.provider = d.provider;
    if (opts && opts.model) o.model = opts.model; else if (d.model) o.model = d.model;
    if (opts && opts.effort) o.effort = opts.effort; else if (d.effort) o.effort = d.effort;
    if (opts && opts.api_key) o.api_key = opts.api_key; else if (d.apiKey) o.api_key = d.apiKey;
    if (opts && opts.base_url) o.base_url = opts.base_url; else if (d.baseUrl) o.base_url = d.baseUrl;
    if (opts && opts.profile) o.profile = opts.profile; else if (d.profile) o.profile = d.profile;
    /* Dynamic output cap: explicit def/opts value wins, else the model's
     * published max output (model_caps). 0 = unknown → omit the field so
     * the endpoint applies its own default. Never a flat constant. */
    var mo = (opts && opts.max_tokens) || d.maxTokensOut || 0;
    if (mo <= 0 && !(opts && opts.max_tokens)) mo = maxOutputFor(d);
    if (mo > 0) o.max_tokens = mo;
    if (withTools) o.tools = withTools;
    return o;
  }

  /* OpenAI streamed tool_calls arrive as delta fragments keyed by
   * `index`, with argument strings split across chunks — fold them into
   * complete {name, arguments(object)} entries. Tolerates already-merged
   * shapes ({name, arguments:{}}) too. */
  function mergeStreamToolCalls(tcs) {
    var byIndex = {};
    var order = [];
    for (var i = 0; i < tcs.length; i++) {
      var t = tcs[i];
      if (!t) continue;
      var idx = (t.index !== undefined && t.index !== null) ? t.index : order.length;
      if (byIndex[idx] === undefined) { byIndex[idx] = { id: '', name: '', args: '' }; order.push(idx); }
      var e = byIndex[idx];
      if (t.id) e.id = String(t.id);
      var fn = t.function || {};
      if (fn.name) e.name = e.name + String(fn.name);
      if (fn.arguments) e.args = e.args + String(fn.arguments);
      if (!fn.name && t.name) e.name = String(t.name);
      if (t.arguments && typeof t.arguments === 'object') {
        try { e.args = JSON.stringify(t.arguments); } catch (er) {}
      }
    }
    var out = [];
    for (var k = 0; k < order.length; k++) {
      var e2 = byIndex[order[k]];
      if (!e2.name) continue;
      var argsObj = {};
      if (e2.args) {
        try { argsObj = JSON.parse(e2.args); } catch (er) { argsObj = { _raw: e2.args }; }
      }
      out.push({ id: e2.id, name: e2.name, arguments: argsObj });
    }
    return out;
  }

  /* P2: threshold → dedupe → token budget. Returns kept ids/lines/hit
   * summaries plus honest accounting (dropped / budgetCut) for /why and
   * traces. */
  function gateRecall(d, hits, task, history) {
    var min = (typeof d.recallMin === 'number') ? d.recallMin : RECALL_MIN_SCORE;
    var dropped = 0;
    var passed = [];
    for (var i = 0; i < hits.length; i++) {
      var h = hits[i];
      if (!h || typeof h.score !== 'number' || !(h.score >= min)) { dropped++; continue; }
      passed.push(h);
    }
    /* Dedupe against the last few history turns + the task itself:
     * re-injecting text the model can already see is pure waste. A memory
     * counts as "already visible" when its head appears verbatim in recent
     * context (or its own head appears in the task — near-duplicate ask). */
    var seen = [];
    for (var m = Math.max(0, history.length - 4); m < history.length; m++) {
      seen.push(String((history[m] && history[m].content) || ''));
    }
    function alreadyVisible(text) {
      var head = clip(String(text == null ? '' : text), 120);
      if (head.length < 20) return false; /* too short to judge — keep it */
      if (String(task).indexOf(head.slice(0, Math.min(40, head.length))) >= 0) return true;
      for (var k = 0; k < seen.length; k++) {
        if (seen[k].indexOf(head) >= 0) return true;
      }
      return false;
    }
    var deduped = [];
    for (var j = 0; j < passed.length; j++) {
      if (alreadyVisible(passed[j].text)) { dropped++; continue; }
      deduped.push(passed[j]);
    }
    /* Highest-score-first fill until the block's token budget is spent. */
    var budget = d.recallBudget > 0 ? d.recallBudget : recallBudgetTok(d);
    var used = estTok('Context from the shared brain (may be relevant):');
    var kept = [], budgetCut = 0;
    for (var n2 = 0; n2 < deduped.length; n2++) {
      var line = '- ' + clip(deduped[n2].text, 300);
      if (kept.length > 0 && used + estTok(line) > budget) { budgetCut++; continue; }
      used += estTok(line);
      kept.push(deduped[n2]);
    }
    return {
      ids: kept.map(function (x) { return x.id; }),
      lines: kept.map(function (x) { return '- ' + clip(x.text, 300); }),
      hits: kept.map(function (x) { return { id: x.id, score: x.score,
                                              role: x.role || '', text: clip(x.text, 120) }; }),
      dropped: dropped, budgetCut: budgetCut,
    };
  }

  /* Cap one tool result before it enters any message. Head+tail are
   * kept (the middle is usually boilerplate) with an explicit marker that
   * teaches recovery: narrow the args and re-call. */
  function capToolResult(cap, s) {
    s = String(s == null ? '' : s);
    cap = (cap | 0) || TOOL_RESULT_CAP_CHARS;
    if (s.length <= cap) return { text: s, chars: s.length };
    var head = Math.floor(cap * TOOL_RESULT_HEAD_RATIO);
    var tail = Math.min(Math.floor(cap * TOOL_RESULT_TAIL_RATIO), Math.max(0, cap - head - 80));
    var omitted = s.length - head - tail;
    var text = s.slice(0, head) +
      '\n…[' + omitted + ' chars truncated — re-call the tool with narrower args if needed]…\n' +
      (tail > 0 ? s.slice(s.length - tail) : '');
    return { text: text, chars: s.length };
  }

  /* ── ML gates (PLAN-ML-GATES) ──────────────────────────────────────
   * Advisors, never filters (Principle 1): the gates observe the loop and
   * emit guidance; they never block, cap, reorder, or delete. Every call
   * is optional + try/catch-wrapped at the call site — a gate failure can
   * never break a turn. d.ml === 'off' (chat /ml off) or a missing
   * sofuu.ml (SOFUU_NO_ML=1) disables everything. */
  function mlEnabled(d) {
    return !!(d && d.ml !== 'off' && sofuu.ml && typeof sofuu.ml.track === 'function');
  }
  /* Canonical JSON (sorted keys, recursively) → a stable identity key for
   * exact-repeat-call detection. Computed here, not in Rust: key order in
   * the raw args object is JS-visible noise, and sorting it is trivial on
   * this side of the FFI. */
  function mlCanon(v) {
    if (v === null || v === undefined || typeof v === 'function') return 'null';
    if (typeof v !== 'object') return JSON.stringify(v);
    if (Array.isArray(v)) return '[' + v.map(mlCanon).join(',') + ']';
    var ks = Object.keys(v).sort();
    return '{' + ks.map(function (k) { return JSON.stringify(k) + ':' + mlCanon(v[k]); }).join(',') + '}';
  }
  function mlSig(name, args) { return name + ':' + mlCanon(args || {}); }
  /* Primary target of a call (path/pattern) — the re-read rule's key. */
  function mlTarget(args) {
    var a = args || {};
    return String(a.path || a.pattern || '');
  }
  /* Freshness gate (PLAN-ML-GATES §5): score COLLECTED MATERIAL against
   * the task; a stale verdict becomes one evidence-carrying notice on the
   * ephemeral context message — advise only, nothing is filtered, capped,
   * or dropped. Returns the parsed verdict when it fires, else null. */
  function mlFreshKind(toolName) {
    if (toolName === 'web_search' || toolName === 'web_fetch') return 'web';
    if (toolName === 'read_file' || toolName === 'grep' ||
        toolName === 'glob' || toolName === 'list_dir') return 'file';
    return 'tool';
  }
  function mlFreshScore(d, text, task, kind) {
    if (!mlEnabled(d) || !sofuu.ml.freshness ||
        typeof sofuu.ml.freshness.score !== 'function') return null;
    var t = String(text == null ? '' : text);
    if (t.length < 80) return null; /* too little material to judge */
    try {
      var v = JSON.parse(sofuu.ml.freshness.score(
        t.slice(0, 4000), String(task == null ? '' : task),
        JSON.stringify({ kind: kind })));
      if (v && v.stale) return v;
    } catch (eMl) {}
    return null;
  }
  /* Relevance gate (PLAN-ML-GATES §6): pre-retrieval advisor over the menu
   * of recalled memories. It scores each candidate against the task and
   * returns use/skip advice, rendered as ONE guidance notice on the
   * ephemeral context message. Advise only (Principle 1): nothing is
   * filtered, capped, or dropped from the recall block — the LLM decides
   * what to rely on. Returns the parsed plan {use,skip,scores} or null. */
  function mlRelevancePlan(d, task, recent, hits) {
    if (!mlEnabled(d) || !sofuu.ml.relevance ||
        typeof sofuu.ml.relevance.plan !== 'function') return null;
    if (!hits || !hits.length) return null;
    try {
      /* Recall hits map to memory candidates as trained (kind=memory, role=2);
       * strength carries the brain's retrieval confidence for this task. */
      var cands = hits.map(function (h) {
        return { text: String(h.text == null ? '' : h.text),
                 kind: 'memory',
                 strength: (typeof h.score === 'number') ? h.score : 0.5,
                 role: 2, path: '' };
      });
      var v = JSON.parse(sofuu.ml.relevance.plan(JSON.stringify({
        task: String(task == null ? '' : task).slice(0, 4000),
        recent: String(recent == null ? '' : recent).slice(0, 4000),
        candidates: cands })));
      if (!v || !Array.isArray(v.use) || !Array.isArray(v.skip)) return null;
      return v;
    } catch (eMl) {}
    return null;
  }
  /* Render one relevance guidance notice from a plan, or null when there is
   * nothing worth saying (every recalled item looks on-task). Items are
   * referenced by a short head snippet so the model can match them to the
   * recall block above. */
  function mlRelevanceNotice(plan, hits) {
    var useIds = plan.use, skipIds = plan.skip;
    if (!skipIds.length) return null;
    var snip = function (id) {
      var h = hits[id];
      return h ? '"' + clip(String(h.text == null ? '' : h.text), 48) + '"' : '';
    };
    var skipSnips = skipIds.slice(0, 3).map(snip).filter(Boolean).join(', ');
    if (useIds.length) {
      var useSnips = useIds.slice(0, 3).map(snip).filter(Boolean).join(', ');
      return '[relevance] Of the recalled context above, likely on-task: ' + useSnips +
        '; likely tangential to this task: ' + skipSnips +
        '. Advisory only — you decide what to rely on.';
    }
    return '[relevance] The recalled context above looks largely tangential to this task: ' +
      skipSnips + '. Advisory only — it is still shown if you need it.';
  }

  /**
   * sofuu.agent.run(handleOrNameOrDef, task, opts) → Promise<AgentResult>
   */
  async function run(target, task, opts) {
    opts = opts || {};
    var d = resolveDef(target);
    var depth = opts.depth || 0;
    var chain = opts.chain || [d.name];
    var started = Date.now();
    var runId = 'ar' + (NEXT_RUN++);
    var state = { cancelled: false, aborts: [] };
    ACTIVE[runId] = state;
    ACTIVE_COUNT++;
    var cancelId = opts.signal || runId;
    SIGNALS[cancelId] = runId;
    bumpRunCount(d.name); /* M5: day-stamped, resets on UTC rollover */

    var own = { promptTokens: 0, completionTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, llmCalls: 0, toolCalls: 0 };
    var tree = opts._treeUsage || { promptTokens: 0, completionTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, llmCalls: 0, toolCalls: 0 };
    var trace = [];
    var subRuns = [];
    var steps = 0;
    var stopped = null;

    function emit(kind, payload) {
      var e = { t: Date.now() - started, kind: kind };
      if (payload !== undefined) e.payload = payload;
      trace.push(e);
      /* M4: bound the HEAVY data, keep the shape — the first-16 skeleton
       * (start/plan) and the freshest (TRACE_CAP-16) events keep their
       * payloads; the event that just fell out of that window is demoted to
       * a {kind,t} stub so renderTrace still shows the run's full timeline
       * without retaining every tool-result/arg blob. */
      if (trace.length > TRACE_CAP) {
        var mid = trace[trace.length - (TRACE_CAP - 16) - 1];
        if (mid && mid.payload !== undefined) delete mid.payload;
      }
      if (opts.onStep) {
        try { opts.onStep({ runId: runId, name: d.name, depth: depth, t: e.t, kind: kind, payload: payload }); }
        catch (e2) {}
      }
    }
    function budgetBreach() {
      if (state.cancelled) return 'cancelled';
      if (Date.now() - started > d.budget.maxWallMs) return 'budget_wall';
      if (tree.promptTokens + tree.completionTokens > d.budget.maxTokens) return 'budget_tokens';
      if (steps >= d.budget.maxSteps) return 'budget_steps';
      return null;
    }
    function finish(answer) {
      if (answer && answer.length > ANSWER_CAP_CHARS) answer = answer.slice(0, ANSWER_CAP_CHARS);
      var wall = Date.now() - started;
      own.wallMs = wall;
      var res = {
        runId: runId, name: d.name, answer: answer || '',
        steps: steps, subRuns: subRuns,
        usage: own, trace: trace, stopped: stopped,
      };
      emit(stopped ? 'stop' : 'answer', { answer: clip(answer, 200), stopped: stopped });
      if (mlEnabled(d)) {
        try { sofuu.ml.track(JSON.stringify({ kind: 'run_end', run: String(runId) })); }
        catch (eMl) {}
      }
      if (opts.logs) logRun(res, task, wall);
      delete ACTIVE[runId];
      delete SIGNALS[cancelId];
      ACTIVE_COUNT--;
      if (ACTIVE_COUNT === 0) runTick(mcpIdleDisconnect);
      return res;
    }

    emit('start', { task: clip(task, 300), depth: depth, chain: chain.slice() });
    /* PLAN-ML-GATES §8: open this run's trajectory in the working set.
     * Sub-agents run under their own runId, so trajectories stay separate. */
    if (mlEnabled(d)) {
      try { sofuu.ml.track(JSON.stringify({ kind: 'run_start', run: String(runId),
                                            task: clip(String(task == null ? '' : task), 300) })); }
      catch (eMl) {}
    }
    /* Freshness gate state for this run (§5): stale verdicts collected as
     * material comes in, flushed as ONE notice at the next context
     * boundary; noticed keys dedupe repeat material (same tool+target). */
    var mlFreshPending = [];
    var mlFreshNoticed = {};
    function mlFreshNotice() {
      var ev = mlFreshPending.slice(0, 3).map(function (it) {
        return it.src + (it.reason ? ' (' + it.reason + ')' : '');
      }).join('; ');
      mlFreshPending = [];
      return '[freshness] Some collected material may be outdated for this task: ' +
        ev + '. Verify its currency before relying on it.';
    }
    /* Supervisor state for this run (§11): targets the relevance advice
     * flagged "probably not needed" (skip channel), one notice per loop
     * class per run at the boundary (credibility bar: no nagging). */
    var mlSkipTargets = [];
    var mlLoopNoticed = {};

    /* Stream one generation, with bounded backoff retry on transient
     * provider errors (see STREAM_TRANSIENT_RE). Retries happen ONLY while
     * nothing has been forwarded to the UI yet: HTTP-status failures
     * (429/5xx) always qualify — they die before any content byte — but
     * transport cuts (HTTP/2 stream resets, dropped connections) can arrive
     * AFTER chunks were shown, and re-streaming would duplicate text in the
     * live answer, so the error surfaces instead. Returns { st, parts } on
     * success; rethrows non-transient errors and cancellation. onThink /
     * onText observe chunks for live rendering. */
    async function streamWithRetry(ro, onThink, onText) {
      var attempt = 0;
      for (;;) {
        var st = sofuu.ai.stream(ro);
        state.aborts.push(function () { try { st.abort(); } catch (e) {} });
        var parts = [];
        var forwarded = false;
        try {
          for await (var chunk of st) {
            if (!chunk) continue;
            if (chunk.think) { forwarded = true; if (onThink) onThink(chunk.think); continue; }
            if (chunk.text) { forwarded = true; parts.push(chunk.text); if (onText) onText(chunk.text); }
          }
          return { st: st, parts: parts };
        } catch (e) {
          if (state.cancelled) throw e;
          var msg = String((e && e.message) || e);
          if (attempt < STREAM_RETRY_DELAYS.length && !forwarded && isTransientProviderError(msg)) {
            emit('tool_result', { name: 'system',
                                  result: 'provider error (' + clip(msg, 100) + ') — retrying in ' +
                                          (STREAM_RETRY_DELAYS[attempt] / 1000) + 's' });
            await new Promise(function (r) { setTimeout(r, STREAM_RETRY_DELAYS[attempt]); });
            attempt++;
            if (state.cancelled) throw new Error('cancelled');
            continue;
          }
          throw e;
        }
      }
    }

    try {
      /* Recall augmentation (A1.4, A3) — P2-gated: similarity floor →
       * dedupe vs what the model already sees → block token budget. Zero
       * relevant hits means zero injected bytes. */
      var brainEntry = brainFor(d);
      var cma = brainEntry ? brainEntry.cma : null;
      var recalledIds = [];
      var recalledHits = [];
      var recallBlock = '';
      if (cma) {
        await rememberIdentity(d, cma);
        var consolidatedN = brainDecayTick(brainEntry); /* M3: decay + prune + consolidate at the turn boundary */
        if (consolidatedN > 0) {
          emit('consolidated', { clusters: consolidatedN, scope: d.memory });
        }
        try {
          var qvec = await embedTextFor(d, String(task));
          var hits = scopeRecall(d, cma, qvec, 15);
          var gated = gateRecall(d, hits, String(task), opts.history || []);
          if (gated.lines.length) {
            recalledIds = gated.ids;
            recalledHits = gated.hits;
            recallBlock = 'Context from ' +
              (d.memory === 'agent' ? 'your private memory' : 'the shared brain') +
              ' (may be relevant):\n' + gated.lines.join('\n');
            emit('recall', { count: gated.ids.length,
                             dropped: gated.dropped, budgetCut: gated.budgetCut,
                             scope: d.memory,
                             hits: gated.hits });
          }
        } catch (e) {}
      }

      var identityLine = d.identity
        ? '\nYou are ' + d.name + (d.identity.role ? ', ' + d.identity.role : '') +
          (d.identity.expertise ? ' (expertise: ' + d.identity.expertise + ')' : '') + '.'
        : '';
      /* P1 composition — the system message is byte-stable for a given def
       * (core prompt + user override + identity, nothing dynamic). Recall /
       * shared mesh / watch changes ride a separate EPHEMERAL user-role
       * context message placed AFTER the system line: it never leaks into
       * the caller's stored history and never breaks prefix caching (P6). */
      var ctxParts = [];
      /* PLAN-ML-GATES §7: today's date rides the EPHEMERAL context message
       * (never the byte-stable system prompt, P1/P6) so the model can judge
       * staleness at all. ~8 tokens/turn; the freshness gate (§5) builds on
       * this, it does not substitute for it. */
      ctxParts.push('Current date: ' + isoDate() + '.');
      if (recallBlock) {
        ctxParts.push(recallBlock);
        /* Freshness gate on recalled memory (§5): the block is scored as
         * the model will see it; a firing verdict rides the SAME
         * ephemeral message as one notice with its evidence. */
        var rv = mlFreshScore(d, recallBlock, task, 'memory');
        if (rv) {
          mlFreshNoticed['memory:' + String(runId)] = true;
          mlFreshPending.push({ src: 'memory recall', reason: rv.reason });
          ctxParts.push(mlFreshNotice());
          emit('mlgate', { rule: 'freshness', tool: 'memory', step: 0,
                           nudge: clip(rv.reason || 'stale', 160) });
        }
        /* Relevance gate (§6): pre-retrieval advice over the recalled menu —
         * one guidance notice on the SAME ephemeral message, advise only. */
        var relRecent = (opts.history || []).slice(-4).map(function (m) {
          return String((m && m.content) || '');
        }).join('\n');
        var relPlan = mlRelevancePlan(d, task, relRecent, recalledHits);
        if (relPlan) {
          var relNotice = mlRelevanceNotice(relPlan, recalledHits);
          if (relNotice) {
            ctxParts.push(relNotice);
            emit('mlgate', { rule: 'relevance', tool: 'memory', step: 0,
                             use: relPlan.use.length, skip: relPlan.skip.length,
                             nudge: clip(relNotice, 160) });
          }
          /* The skip advice feeds the supervisor's skip channel (§11): if
           * the agent later reaches for a target matching a flagged item,
           * the pre-call checkpoint can name it. */
          if (relPlan.skip && relPlan.skip.length) {
            mlSkipTargets = relPlan.skip.map(function (id) {
              var h = recalledHits[id];
              if (!h) return '';
              return String(h.path || clip(String(h.text == null ? '' : h.text), 80));
            }).filter(Boolean);
          }
        }
      }
      if (opts.shared) ctxParts.push('Shared session notes:\n' + String(opts.shared));
      if (opts.watched) ctxParts.push('Filesystem changes since the last turn:\n' + String(opts.watched));
      var messages = [{ role: 'system', content: d.system + identityLine }]
        .concat(ctxParts.length ? [{ role: 'user', content: ctxParts.join('\n\n') }] : [])
        .concat(Array.isArray(opts.history) ? opts.history : [])
        .concat([{ role: 'user',
                   content: String(task == null ? '' : task) +
                            (opts.context ? '\n\n--- context ---\n' + opts.context : '') }]);

      /* Tool resolution + delegate injection (A2) — resolved BEFORE the
       * RLM gate so an RLM-routed turn can inject the agent's tool
       * whitelist into the sandbox (A4 full form). */
      var toolset = { specs: [], exec: {} };
      var hasDelegate = false;
      if (!opts.plain) {
        toolset = await resolveTools(d, emit);
        if (d.agents.length && depth < d.budget.maxDepth) {
          var ds = delegateSpecFor(d, chain);
          if (ds) { toolset.specs.push(ds); hasDelegate = true; }
        }
      }
      var tools = toolset.specs.length ? toolset.specs : null;

      var answer = '';

      /* RLM gate (A4.1): route oversized turns through sofuu.rlm.query,
       * with the agent's own tool whitelist injected into the sandbox —
       * sandbox snippets call tool(name, args); the host executes them
       * HERE through execOneTool (timeout, trace, budget accounting), so
       * tool code never runs inside the sandbox. `delegate` is excluded
       * (agents-inside-RLM goes through recurseVia, A4.2). Any RLM
       * failure falls back to the plain loop — never fatal. */
      if (depth === 0 && !opts.plain && d.rlm !== 'off' && sofuu.rlm && typeof sofuu.rlm.query === 'function') {
        try {
          var ctxText = String(task) + String(opts.context || '') +
            messages.map(function (m) { return m.content || ''; }).join('');
          var tokens = estTok(ctxText);
          var win = d.ctxWindow > 0 ? d.ctxWindow : contextWindow(d.model);
          var route = (d.rlm === 'on') ? 'rlm' : sofuu.rlm.route(tokens, win, String(task));
          if (route === 'rlm') {
            emit('rlm:route', { ctxTokens: tokens, windowTokens: win, route: 'rlm' });
            var ro = { provider: d.provider, model: d.model, effort: d.effort,
                       base_url: d.baseUrl, api_key: d.apiKey, profile: d.profile, trace: true };
            var mo = (opts && opts.max_tokens) || d.maxTokensOut || 0;
            if (mo <= 0) mo = maxOutputFor(d);
            if (mo > 0) ro.max_tokens = mo;
            var rlmTools = toolset.specs.filter(function (t) { return t.name !== 'delegate'; });
            if (rlmTools.length) {
              ro.tools = rlmTools;
              ro.execTool = function (name, args) { return execOneTool(name, args, 'rlm_' + runId); };
            }
            /* Cancel must also stop the episode between rounds (the
             * driver checks __rlm_aborted at every action). */
            state.aborts.push(function () { try { g.__rlm_aborted = true; } catch (eAb) {} });
            var res = await withCancelGuard(state, sofuu.rlm.query(ctxText, String(task), ro));
            if (res && res.stopped === 'aborted') { stopped = 'cancelled'; return finish('(cancelled)'); }
            var rlmTrace = (res && res.trace) || [];
            for (var ri = 0; ri < rlmTrace.length; ri++) {
              emit('rlm:' + rlmTrace[ri].kind, rlmTrace[ri].repr ? clip(String(rlmTrace[ri].repr), 200) : undefined);
            }
            own.llmCalls += (res && res.calls) || 0;
            tree.llmCalls += (res && res.calls) || 0;
            if (res && res.ms) { own.wallMs = res.ms; }
            try { sofuu.rlm.logRoute({ ctxTokens: tokens, windowTokens: win, question: String(task),
                                       route: 'rlm', latencyMs: (res && res.ms) || 0, rlmCalls: (res && res.calls) || 0 }); } catch (e) {}
            emit('rlm:done', { calls: (res && res.calls) || 0, toolCalls: (res && (res.toolCalls || res.tool_calls)) || 0,
                               rounds: (res && res.rounds) || 0,
                               ms: (res && res.ms) || 0, stopped: (res && res.stopped) || null });
            var rlmAnswer = String((res && res.answer) || '(no response)');
            await storeAndFinish(rlmAnswer);
            return finish(rlmAnswer);
          }
        } catch (e) {
          emit('rlm:fallback', { error: clip(String((e && e.message) || e), 160) });
        }
      }

      async function execOneTool(name, args, callId) {
        own.toolCalls++; tree.toolCalls++;
        emit('tool', { name: name, args: clip(JSON.stringify(args || {}), 200) });
        /* Supervisor checkpoint 1 — BEFORE the tool runs (PLAN-ML-GATES
         * §11): the trained net layered over the rule subset (exact repeat
         * calls, re-reads of unchanged files) checks the trajectory and
         * returns a nudge with its waste class. Advise only — the call
         * still runs; the nudge rides the result back in-band (~15 tokens)
         * so the LLM can adjust its next move. */
        var mlNudge = null;
        var mlVerdict = null;
        if (mlEnabled(d) && sofuu.ml.supervisor && typeof sofuu.ml.supervisor.check === 'function') {
          try {
            var verdict = JSON.parse(sofuu.ml.supervisor.check(JSON.stringify({
              run: String(runId), step: own.toolCalls, tool: name,
              sig: mlSig(name, args), target: mlTarget(args),
              argsText: mlCanon(args || {}),
              task: String(task == null ? '' : task).slice(0, 4000),
              skipTargets: mlSkipTargets, budget: d.budget.maxSteps,
            })));
            if (verdict) {
              mlVerdict = verdict;
              if (verdict.nudge) {
                mlNudge = String(verdict.nudge);
                emit('mlgate', { rule: String(verdict.reason || 'nudge'), tool: name,
                                 step: own.toolCalls, source: String(verdict.source || ''),
                                 score: (typeof verdict.score === 'number') ? verdict.score : 0,
                                 nudge: clip(mlNudge, 160) });
              }
            }
          } catch (eMl) {}
        }
        var result;
        var errored = false;
        try {
          if (name === 'delegate' && hasDelegate) {
            result = await execDelegate(args);
          } else {
            var fn = toolset.exec[name];
            if (!fn) throw new Error('no such tool: ' + name);
            result = await withTimeout(Promise.resolve(fn(args)), opts.toolTimeoutMs || d.toolTimeoutMs,
                                       'tool ' + name + ' timed out');
          }
        } catch (e) {
          result = 'tool error: ' + String((e && e.message) || e);
          errored = true;
          emit('tool_result', { name: name, error: clip(String((e && e.message) || e), 160) });
        }
        if (result !== undefined && (typeof result.then === 'function')) {
          result = await result;
        }
        /* P3: uniform cap at the boundary (delegate answers included — the
         * child already capped its own answer; no double truncation). */
        var cappedR = capToolResult(toolResultCapChars(d), result);
        if (mlNudge) cappedR.text += '\n[supervisor: ' + mlNudge + ']';
        emit('tool_result', { name: name, result: clip(cappedR.text, 200),
                              chars: cappedR.chars, kept: cappedR.text.length });
        /* Supervisor checkpoint 2 — AFTER the result returns: account the
         * outcome in the working set (trajectory + write tracking for the
         * re-read rule), then report a confidence-banded outcome label for
         * online learning (§13) — see the banding below. */
        if (mlEnabled(d)) {
          try { sofuu.ml.track(JSON.stringify({ kind: 'tool_result', run: String(runId),
                                                step: own.toolCalls, tool: name, target: mlTarget(args),
                                                chars: cappedR.chars, error: errored })); }
          catch (eMl2) {}
          if (mlVerdict && sofuu.ml.feedback && typeof sofuu.ml.feedback === 'function') {
            /* Outcome label for online learning (§13) — confidence-banded
             * proxies, with abstention in the ambiguous middle: a wrong
             * label teaches worse than none. errored = certain waste;
             * near-empty = probably waste, but a one-line result can be
             * THE answer, so down-weight; 80–319 chars = coin flip, no
             * label; a large clean result = probably useful (large ERROR
             * dumps took the errored branch already). */
            var mlWasted = null, mlConf = 0;
            if (errored) { mlWasted = true; mlConf = 1.0; }
            else if (cappedR.chars < 80) { mlWasted = true; mlConf = 0.6; }
            else if (cappedR.chars >= 320) { mlWasted = false; mlConf = 0.8; }
            if (mlWasted !== null) {
              try { sofuu.ml.feedback(JSON.stringify({ kind: 'outcome', model: 'supervisor',
                                                       run: String(runId), step: own.toolCalls,
                                                       wasted: mlWasted, conf: mlConf })); }
              catch (eMl3) {}
            }
          }
        }
        /* Freshness gate checkpoint 2 (§5): collected material is scored
         * as it arrives. A firing verdict is queued — the notice itself
         * rides the next context boundary (one per boundary, evidence
         * attached), never the tool result itself. */
        if (!errored) {
          var fv = mlFreshScore(d, cappedR.text, task, mlFreshKind(name));
          if (fv) {
            var fkey = name + ':' + mlTarget(args);
            if (!mlFreshNoticed[fkey]) {
              mlFreshNoticed[fkey] = true;
              mlFreshPending.push({ src: name + ' result', reason: fv.reason });
            }
          }
        }
        return cappedR.text;
      }

      async function execDelegate(args) {
        var childName = String((args && args.agent) || '');
        if (d.agents.indexOf(childName) < 0) {
          return 'tool error: unknown sub-agent \'' + childName + '\' (allowed: ' + d.agents.join(', ') + ')';
        }
        if (chain.indexOf(childName) >= 0) {
          return 'tool error: delegation cycle — \'' + childName + '\' is already running in this chain (' + chain.join(' → ') + '). Choose a different agent or answer directly.';
        }
        var childDef = REGISTRY[childName];
        if (!childDef) return 'tool error: sub-agent \'' + childName + '\' is not defined';
        emit('delegate', { agent: childName, task: clip(String((args && args.task) || ''), 200),
                           why: clip(String((args && args.why) || ''), 120) });
        var childRes = await run(childDef, String((args && args.task) || ''), {
          depth: depth + 1,
          chain: chain.concat([childName]),
          _treeUsage: tree,
          onStep: opts.onStep,
          toolTimeoutMs: opts.toolTimeoutMs,
        });
        subRuns.push(childRes);
        own.llmCalls += childRes.usage.llmCalls;
        own.toolCalls += childRes.usage.toolCalls;
        return childRes.answer +
          '\n[sub-agent ' + childName + ' · ' + childRes.usage.llmCalls + ' llm calls · ' +
          childRes.usage.toolCalls + ' tool calls' +
          (childRes.stopped ? ' · stopped: ' + childRes.stopped : '') + ']';
      }

      /* Plain path: no tools — one streamed (or completed) answer. */
      if (!tools) {
        answer = await finalAnswer(messages);
        await storeAndFinish(answer);
        return finish(answer);
      }

      /* Stream-first loop for EVERY provider (F4b landed): OpenAI-compatible
       * providers surface tool calls inside the stream as delta fragments,
       * Anthropic as content_block tool_use + input_json_delta fragments —
       * ai.rs folds both into stream.toolCalls — so a no-tool turn costs ONE
       * generation and the answer streams live. */
      var loopMsgs = [];
      for (;;) {
        var breach0 = budgetBreach();
        if (breach0) {
          stopped = breach0;
          /* Salvage (AUDIT-NO-RESPONSE-2026-08-24 b): a budget breach used
           * to exit with answer still '' — the chat printed a bare
           * "(no response)" indistinguishable from a dead provider. Spend
           * ONE final no-tool round asking the model to answer from what
           * it already has; partial progress beats silence. Never on user
           * cancel (Esc must stop instantly), and a failed salvage leaves
           * '' — the driver reports the breach either way. */
          if (breach0 !== 'cancelled' && !answer) {
            try {
              var salvageNote = 'Budget limit reached (' + breach0 + '). Do not call any tools. ' +
                'Summarize what you have done and found so far, and give your best answer now; ' +
                'say what could not be finished.';
              var salvaged = await finalAnswer(messages.concat(loopMsgs, [{ role: 'user', content: salvageNote }]));
              if (salvaged && salvaged !== '(no response)' && salvaged !== '(cancelled)') answer = salvaged;
            } catch (eSal) {
              if (state.cancelled) stopped = 'cancelled';
            }
          }
          break;
        }
        emit('plan', { step: steps + 1, tools: tools.map(function (t) { return t.name; }) });
        var so = aiOpts(d, opts, messages.concat(loopMsgs), tools);
        var sres;
        try {
          sres = await streamWithRetry(so,
            function (t) { emit('think', { text: clip(t, 200) }); },
            function (tx) {
              if (opts.onStep) {
                try { opts.onStep({ runId: runId, name: d.name, depth: depth,
                                    t: Date.now() - started, kind: 'answer_delta', payload: tx }); }
                catch (e2) {}
              }
            });
        } catch (eS) {
          if (state.cancelled) { stopped = 'cancelled'; break; }
          throw eS;
        }
        var st = sres.st;
        var sparts = sres.parts;
        var su = st.usage || {};
        own.promptTokens += su.promptTokens || 0;
        own.completionTokens += su.completionTokens || 0;
        own.cacheReadTokens += su.cacheReadTokens || 0;
        own.cacheWriteTokens += su.cacheWriteTokens || 0;
        tree.promptTokens += su.promptTokens || 0;
        tree.completionTokens += su.completionTokens || 0;
        tree.cacheReadTokens += su.cacheReadTokens || 0;
        tree.cacheWriteTokens += su.cacheWriteTokens || 0;
        own.llmCalls++; tree.llmCalls++;
        /* Per-request trace: the aggregates above re-count tool-result
         * growth across rounds, so consumers that need the TRUE context
         * size (the chat footer meter) take the FIRST request's prompt. */
        emit('llm', { round: own.llmCalls, usage: {
          promptTokens: su.promptTokens || 0, completionTokens: su.completionTokens || 0,
          cacheReadTokens: su.cacheReadTokens || 0, cacheWriteTokens: su.cacheWriteTokens || 0 } });
        var stcs = mergeStreamToolCalls(st.toolCalls || []);
        if (!stcs.length) {
          answer = sparts.join('').trim() || (state.cancelled ? '(cancelled)' : '(no response)');
          if (state.cancelled) stopped = stopped || 'cancelled';
          /* Empty-stream recovery (dynamic, no flat constants):
           * 200 with 0 text + 0 tool_calls is a provider quirk, not an
           * answer. su.finishReason (native, ai.rs) says why:
           *   • "length" → output budget exhausted — often an oversized
           *     explicit max_tokens (e.g. /maxout 384k) against an unknown
           *     model, or a tiny provider default → retry ONCE with
           *     max_tokens = 12.5% of the model's context window (pure
           *     ratio; unknown models fall back to their ctx default).
           *   • any other/absent → retry once WITHOUT effort (some
           *     gateways empty on thinking params). Never persisted — a
           *     transient empty must not disable thinking forever.
           * Fires regardless of d.effort: the empty is not necessarily
           * thinking-related. Still empty → THROW with the cause so the
           * chat shows ✗ + reason instead of a silent (no response). */
          if (answer === '(no response)' && !state.cancelled) {
            var fr = String(su.finishReason || '');
            if (!state.emptyRetried) {
              state.emptyRetried = true;
              emit('tool_result', { name: 'system',
                                    result: 'empty stream' + (fr ? ' (finish_reason: ' + fr + ')' : '') + ' → retrying once' });
              var retryOpts = Object.assign({}, opts, { effort: undefined });
              if (fr === 'length') {
                var winE = contextWindow(d.model);
                if (winE > 0) retryOpts.max_tokens = Math.floor(winE * 0.125);
              }
              var ro = aiOpts(Object.assign({}, d, { effort: '', max_tokens: 0, maxTokensOut: 0 }), retryOpts,
                              messages.concat(loopMsgs), tools);
              try {
                var rr = await streamWithRetry(ro,
                  function (t) { emit('think', { text: clip(t, 200) }); }, null);
                var rtxt = rr.parts.join('').trim();
                if (rtxt) { answer = rtxt; break; }
                var fr2 = String((rr.st && rr.st.usage && rr.st.usage.finishReason) || fr || '');
              } catch (eR) {
                if (state.cancelled) { stopped = 'cancelled'; break; }
                throw eR;
              }
            }
            throw new Error('provider returned an empty stream' +
              (fr ? ' (finish_reason: ' + fr + ')' : ' (no finish_reason reported)') +
              ' — check the model on this endpoint, or lower /maxout (an output cap above the model\'s real max empties some gateways)');
          }
          break;
        }
        steps++;
        for (var si = 0; si < stcs.length; si++) {
          if (state.cancelled) { stopped = 'cancelled'; break; }
          var stc = stcs[si];
          var sname = String(stc.name || '');
          var sargs = (stc.arguments && typeof stc.arguments === 'object') ? stc.arguments : {};
          var scid = 'call_' + runId + '_' + steps + '_' + si + '_' + sname;
          var sres = await execOneTool(sname, sargs, scid);
          loopMsgs.push({ role: 'assistant', content: null,
                          tool_calls: [{ id: scid, type: 'function',
                                         function: { name: sname, arguments: JSON.stringify(sargs) } }] });
          loopMsgs.push({ role: 'tool', tool_call_id: scid, content: sres });
        }
        /* Freshness gate boundary flush (§5/§16): stale verdicts queued
         * during this step's tool calls become ONE ephemeral user notice
         * before the next generation — advise only, data path untouched. */
        if (mlFreshPending.length) {
          var freshNotice = mlFreshNotice();
          loopMsgs.push({ role: 'user', content: freshNotice });
          emit('mlgate', { rule: 'freshness', tool: 'context', step: steps,
                           nudge: clip(freshNotice, 160) });
        }
        /* Supervisor checkpoint 3 — the loop boundary (§11): the "__loop__"
         * pseudo-action asks the net whether the RUN itself is spinning,
         * stalled, or over budget. One ephemeral notice per class per run
         * (credibility bar: no nagging) — advise only, data path untouched. */
        if (mlEnabled(d) && sofuu.ml.supervisor && typeof sofuu.ml.supervisor.loop === 'function') {
          try {
            var lv = JSON.parse(sofuu.ml.supervisor.loop(JSON.stringify({
              run: String(runId), step: own.toolCalls, budget: d.budget.maxSteps })));
            if (lv && lv.nudge && !mlLoopNoticed[String(lv.reason || 'loop')]) {
              mlLoopNoticed[String(lv.reason || 'loop')] = true;
              var loopNotice = '[supervisor] ' + String(lv.nudge);
              loopMsgs.push({ role: 'user', content: loopNotice });
              emit('mlgate', { rule: String(lv.reason || 'loop'), tool: 'context', step: steps,
                               source: String(lv.source || 'model'),
                               score: (typeof lv.score === 'number') ? lv.score : 0,
                               nudge: clip(loopNotice, 160) });
            }
          } catch (eMlL) {}
        }
        if (stopped) break;
      }

      await storeAndFinish(answer);
      return finish(answer);

      /* Final answer via ai.stream — emits answer_delta onStep events and
       * registers the stream's abort() so cancellation kills it mid-flight. */
      async function finalAnswer(msgs) {
        var o = aiOpts(d, opts, msgs);
        var onThink = function (t) { emit('think', { text: clip(t, 200) }); };
        var onDelta = function (tx) {
          if (opts.onStep) {
            try { opts.onStep({ runId: runId, name: d.name, depth: depth,
                                t: Date.now() - started, kind: 'answer_delta', payload: tx }); }
            catch (e2) {}
          }
        };
        var sres = await streamWithRetry(o, onThink, onDelta);
        var stream = sres.st;
        var parts = sres.parts;
        var u = stream.usage || {};
        own.promptTokens += u.promptTokens || 0;
        own.completionTokens += u.completionTokens || 0;
        own.cacheReadTokens += u.cacheReadTokens || 0;
        own.cacheWriteTokens += u.cacheWriteTokens || 0;
        tree.promptTokens += u.promptTokens || 0;
        tree.completionTokens += u.completionTokens || 0;
        tree.cacheReadTokens += u.cacheReadTokens || 0;
        tree.cacheWriteTokens += u.cacheWriteTokens || 0;
        own.llmCalls++; tree.llmCalls++;
        emit('llm', { round: own.llmCalls, usage: {
          promptTokens: u.promptTokens || 0, completionTokens: u.completionTokens || 0,
          cacheReadTokens: u.cacheReadTokens || 0, cacheWriteTokens: u.cacheWriteTokens || 0 } });
        var text = parts.join('').trim() || (state.cancelled ? '(cancelled)' : '(no response)');
        if (state.cancelled) stopped = stopped || 'cancelled';
        /* Same empty-stream recovery as the tool loop (see there): react to
         * finishReason, retry once (length → dynamic max_tokens; otherwise
         * without effort), then throw the cause instead of a bare fallback. */
        if (text === '(no response)' && !state.cancelled) {
          var frF = String(u.finishReason || '');
          if (!state.emptyRetried) {
            state.emptyRetried = true;
            var ro2opts = Object.assign({}, opts, { effort: undefined });
            if (frF === 'length') {
              var winF = contextWindow(d.model);
              if (winF > 0) ro2opts.max_tokens = Math.floor(winF * 0.125);
            }
            var ro2 = aiOpts(Object.assign({}, d, { effort: '', max_tokens: 0, maxTokensOut: 0 }), ro2opts, msgs);
            var r2 = await streamWithRetry(ro2, onThink, onDelta);
            var t2 = r2.parts.join('').trim();
            if (t2) return t2;
            frF = String((r2.st && r2.st.usage && r2.st.usage.finishReason) || frF || '');
          }
          throw new Error('provider returned an empty stream' +
            (frF ? ' (finish_reason: ' + frF + ')' : ' (no finish_reason reported)') +
            ' — check the model on this endpoint, or lower /maxout');
        }
        return text;
      }

      /* Store the turn in the scoped brain + reinforce used recalls. */
      async function storeAndFinish(ansText) {
        if (!cma || d.memory === 'off' || !ansText || ansText === '(no response)' || ansText === '(cancelled)') return;
        try {
          ansText = String(ansText);
          /* P4: skip trivial turns — a brain full of "ok"/"continue" noise
           * degrades recall for everything else. Whole turn skipped. */
          if (ansText.trim().length < 40) return;
          if (String(task).trim().length < 20) return;
          var meaningful = ansText.replace(/tool error:[^\n]*(\n|$)/g, '').trim();
          if (meaningful.length < 40) return; /* answers that are only tool errors */
          /* P4: reuse the recall-time task embedding — one embed per turn,
           * not two (a remote embedder would be an extra network round-trip). */
          var vu = (typeof qvec !== 'undefined' && qvec)
            ? qvec
            : await embedTextFor(d, String(task));
          var va = await embedTextFor(d, ansText);
          scopeStore(d, cma, vu, va, String(task), ansText);
          if (recalledIds.length && ansText.trim()) {
            try { cma.markPositive(recalledIds); } catch (e) {}
          }
          emit('stored', { scope: d.memory });
        } catch (e) {}
      }
    } catch (e) {
      delete ACTIVE[runId];
      delete SIGNALS[cancelId];
      ACTIVE_COUNT--;
      if (ACTIVE_COUNT === 0) runTick(mcpIdleDisconnect);
      throw e;
    }
  }

  /* Wrap a promise so a cancel() between awaits rejects promptly (the
   * underlying provider call still lands in the void; usage from it is
   * simply not counted — the run is dead anyway). */
  function withCancelGuard(state, p) {
    return new Promise(function (resolve, reject) {
      p.then(resolve, reject);
      var iv = setInterval(function () {
        if (state.cancelled) {
          clearInterval(iv);
          reject(new Error('cancelled'));
        }
      }, 25);
      p.then(function () { clearInterval(iv); }, function () { clearInterval(iv); });
    });
  }

  /* ── Parallel fan-out (A2.3) ───────────────────────────────────── */

  async function runMany(items, opts) {
    opts = opts || {};
    if (!Array.isArray(items)) throw new Error('agent.runMany: items array required');
    var conc = Math.max(1, (opts.concurrency | 0) || 4);
    var results = new Array(items.length);
    var next = 0;
    var workers = [];
    var n = Math.min(conc, items.length);
    for (var w = 0; w < n; w++) {
      workers.push((async function () {
        for (;;) {
          var i = next++;
          if (i >= items.length) return;
          var it = items[i] || {};
          try {
            results[i] = await run(it.agent, it.task, it.opts || {});
          } catch (e) {
            results[i] = { runId: '', name: String(it.agent || ''), answer: '',
                           steps: 0, subRuns: [], usage: { promptTokens: 0, completionTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, llmCalls: 0, toolCalls: 0 },
                           trace: [], stopped: 'error', error: String((e && e.message) || e) };
          }
        }
      })());
    }
    await Promise.all(workers);
    /* M2: fan-outs allocate N runs' worth of streams/traces at once — one
     * full GC after the batch bounds the garbage (mapContext rides through
     * here too). The bridge is host-provided; missing = no-op. */
    try { if (typeof __sofuu_gc === 'function') __sofuu_gc(); } catch (e) {}
    return results;
  }

  /* ── Map-reduce over long context (A4.3, the headliner) ────────── */

  function chunkText(text, chunkChars) {
    var overlap = 200;
    var paras = String(text).split(/\n\n+/);
    var chunks = [];
    var cur = '';
    for (var i = 0; i < paras.length; i++) {
      var p = paras[i];
      if (cur.length + p.length + 2 > chunkChars && cur.length > 0) {
        chunks.push(cur);
        cur = cur.slice(Math.max(0, cur.length - overlap));
      }
      cur = cur ? cur + '\n\n' + p : p;
      while (cur.length > chunkChars * 2) { /* one huge paragraph: hard-split */
        chunks.push(cur.slice(0, chunkChars));
        cur = cur.slice(chunkChars - overlap);
      }
    }
    if (cur.trim()) chunks.push(cur);
    return chunks;
  }

  async function mapContext(context, task, opts) {
    opts = opts || {};
    if (!opts.agent || !REGISTRY[opts.agent]) {
      throw new Error('agent.mapContext: opts.agent must name a defined agent');
    }
    var chunkChars = (opts.chunkChars | 0) || 16000;
    var chunks = chunkText(String(context == null ? '' : context), chunkChars);
    if (!chunks.length) throw new Error('agent.mapContext: empty context');
    var items = chunks.map(function (c, i) {
      return { agent: opts.agent, task: String(task), opts: { context: '[chunk ' + (i + 1) + '/' + chunks.length + ']\n' + c } };
    });
    var sub = await runMany(items, { concurrency: opts.concurrency || 4 });
    var parts = [];
    for (var i = 0; i < sub.length; i++) {
      parts.push('## chunk ' + (i + 1) + ' of ' + sub.length + '\n' + (sub[i].answer || '(no answer)'));
    }
    var reduceTarget = opts.reduceAgent || opts.agent;
    var merged = await run(reduceTarget,
      String(task) + '\n\nMerge these partial findings into one final answer. Keep the facts, drop duplicates:\n\n' + parts.join('\n\n'),
      { context: '' });
    merged.subRuns = sub.concat(merged.subRuns || []);
    return merged;
  }

  /* ── Cancellation (A5.2) ───────────────────────────────────────── */

  function cancel(id) {
    var runId = SIGNALS[id] || id;
    var st = ACTIVE[runId];
    if (!st) return false;
    st.cancelled = true;
    for (var i = 0; i < st.aborts.length; i++) {
      try { st.aborts[i](); } catch (e) {}
    }
    return true;
  }

  /* ── Registry / config dir (A8) ────────────────────────────────── */

  function list() {
    return Object.keys(REGISTRY).sort().map(function (name) {
      var d = REGISTRY[name];
      return {
        name: name,
        system: clip(d.system, 80),
        provider: d.provider, model: d.model,
        memory: d.memory,
        tools: d.inlineTools.length + d.webTools.length + d.codeTools.length,
        mcpServers: d.mcpSets.length,
        agents: d.agents.slice(),
        runs24h: (RUN_COUNTS[name] && RUN_COUNTS[name].n) || 0,
      };
    });
  }

  /* Load ~/.sofuu/agents/*.js (or an explicit dir). Each file is a plain
   * script that calls sofuu.agent.define({...}) — evaluated in global
   * scope with the runtime's own eval, i.e. trusted like any user script
   * (documented trust model). Invalid files are listed, never fatal. */
  async function loadDir(dir) {
    var out = { loaded: [], broken: [] };
    dir = dir || (sofuuDir() + '/agents');
    if (!sofuu.fs || !sofuu.fs.readdir) return out;
    var names;
    try { names = await sofuu.fs.readdir(dir); } catch (e) { return out; }
    var files = names.filter(function (f) { return /\.js$/.test(String(f)); }).sort();
    for (var i = 0; i < files.length; i++) {
      var before = Object.keys(REGISTRY).slice();
      try {
        var src = await sofuu.fs.readFile(dir + '/' + files[i]);
        if (typeof src !== 'string') src = String(src);
        (0, eval)(src); /* indirect eval → global scope */
        var after = Object.keys(REGISTRY);
        for (var k = 0; k < after.length; k++) {
          if (before.indexOf(after[k]) < 0) out.loaded.push(after[k]);
        }
      } catch (e) {
        out.broken.push({ file: files[i], error: clip(String((e && e.message) || e), 200) });
      }
    }
    return out;
  }

  /* ── Observability (A9) ────────────────────────────────────────── */

  function renderTrace(result, indent) {
    indent = indent || '';
    var lines = [];
    var r = result;
    lines.push(indent + r.name + ' · ' + (r.steps || 0) + ' steps · ' +
      ((r.usage && r.usage.llmCalls) || 0) + ' llm · ' +
      ((r.usage && r.usage.toolCalls) || 0) + ' tools · ' +
      (((r.usage && r.usage.promptTokens) || 0) + '→' + ((r.usage && r.usage.completionTokens) || 0)) + ' tk' +
      (r.stopped ? ' · stopped: ' + r.stopped : ''));
    var tr = r.trace || [];
    for (var i = 0; i < tr.length; i++) {
      var e = tr[i];
      var p = e.payload;
      var line = indent + '  · ' + e.kind + ' +' + e.t + 'ms';
      if (p && typeof p === 'object') {
        if (p.name) line += ' ' + p.name;
        if (p.agent) line += ' → ' + p.agent;
        if (p.task) line += ' (' + clip(p.task, 60) + ')';
        if (p.answer) line += ' ' + clip(p.answer, 60);
        if (p.error) line += ' ! ' + clip(p.error, 80);
      } else if (p !== undefined) {
        line += ' ' + clip(p, 80);
      }
      lines.push(line);
    }
    var subs = r.subRuns || [];
    for (var s = 0; s < subs.length; s++) {
      lines.push(indent + '  └ sub-run:');
      lines.push(renderTrace(subs[s], indent + '    '));
    }
    return lines.join('\n');
  }

  /* One JSON line per run to ~/.sofuu/logs/agent_runs.jsonl (5MB rotate).
   * Silent failure — logging must never break a run. */
  async function logRun(res, task, wall) {
    try {
      var dir = sofuuDir() + '/logs';
      try { await sofuu.fs.mkdir(dir); } catch (e2) {} /* exists = fine */
      var path = dir + '/agent_runs.jsonl';
      try {
        var prev = await sofuu.fs.readFile(path);
        if (prev && prev.length > 5 * 1024 * 1024) {
          await sofuu.fs.writeFile(path + '.1', prev);
          await sofuu.fs.writeFile(path, '');
        }
      } catch (e) {}
      var line = JSON.stringify({
        t: Date.now(), runId: res.runId, name: res.name,
        task: clip(task, 500), answer: clip(res.answer, 500),
        steps: res.steps, stopped: res.stopped,
        usage: res.usage, subRuns: (res.subRuns || []).length,
      });
      await sofuu.fs.appendFile(path, line + '\n');
    } catch (e) {}
  }

  /* ── Export (merge into the live sofuu.agent object) ───────────── */

  A._agents_loaded = true;
  A.VERSION = VERSION;
  /* P1: the shared static core prompt — chat's driver def and every
   * default compose from this one constant. */
  A.CORE_PROMPT = CORE_PROMPT;
  A.define = define;
  A.run = run;
  A.runMany = runMany;
  A.mapContext = mapContext;
  A.cancel = cancel;
  A.list = list;
  A.loadDir = loadDir;
  A.renderTrace = renderTrace;
  /* Read-only: the per-model context-window lookup used for RLM routing
   * (def.ctx_window / /ctx overrides still win at the gate). */
  A.contextWindow = contextWindow;
  /* Capability helpers for consumers (chat driver pickers/budgets). */
  A.modelCaps = modelCaps;
  A.maxOutputFor = function (defOrModel) {
    if (defOrModel && typeof defOrModel === 'object') return maxOutputFor(defOrModel);
    var caps = modelCaps(defOrModel);
    return (caps && caps.maxOutput) || 0;
  };
  /* Retry-policy classifier (see STREAM_TRANSIENT_RE) — exposed for tests
   * and consumers that need the same transient/permanent distinction. */
  A.isTransientProviderError = isTransientProviderError;
})();
