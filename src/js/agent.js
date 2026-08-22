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
  /* P3: tool results are the largest uncontrolled token source — cap each
   * one before it enters any message (head+tail kept, middle elided with a
   * recovery hint; per-def override budget.maxToolResultChars). */
  var TOOL_RESULT_CAP_CHARS = 4000;
  var TOOL_RESULT_HEAD = 2800;
  var TOOL_RESULT_TAIL = 800;
  /* P2: recall gating — similarity floor + block token budget (defaults;
   * per-def recallMin / recallBudget override). Calibrated for the bundled
   * TF-IDF embedder where unrelated text scores ~0.27 and paraphrases >0.35
   * on the composite score scale. */
  var RECALL_MIN_SCORE = 0.30;
  var RECALL_BUDGET_TOK = 1024;
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

  /* ── Per-model context windows (RLM routing) ──────────────────────
   * The router needs the model's context window to decide "does this turn
   * fit?". Before this table the fallback was a flat 32768 for every
   * model — a 200k claude turn routed to plain far too late. Prefix
   * rules, first match wins; def.ctx_window (/ctx) still overrides. */
  var MODEL_CTX = [
    ['gpt-5', 400000], ['gpt-4.1', 1000000], ['gpt-4o', 128000], ['gpt-4', 128000],
    ['o3', 200000], ['o4', 200000], ['o1', 200000],
    ['claude-opus-4', 200000], ['claude-sonnet-4', 200000], ['claude-haiku-4', 200000],
    ['claude-3-7', 200000], ['claude-3-5', 200000], ['claude-3', 200000], ['claude', 200000],
    ['gemini-1.5-pro', 2000000], ['gemini-2', 1000000], ['gemini', 128000],
    ['deepseek', 65536],
    ['qwen3', 131072], ['qwen2.5', 32768], ['qwen', 32768],
    ['llama4', 1000000], ['llama3.1', 131072], ['llama3', 8192], ['llama', 8192],
    ['mistral', 32768], ['mixtral', 32768],
    ['grok', 131072],
  ];
  var DEFAULT_CTX_WINDOW = 32768;

  function contextWindow(model) {
    var m = String(model == null ? '' : model).toLowerCase();
    if (!m) return DEFAULT_CTX_WINDOW;
    for (var i = 0; i < MODEL_CTX.length; i++) {
      if (m.indexOf(MODEL_CTX[i][0]) === 0 || m.indexOf('/' + MODEL_CTX[i][0]) >= 0) {
        return MODEL_CTX[i][1];
      }
    }
    return DEFAULT_CTX_WINDOW;
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
      /* P2: recall gating knobs (per-def overrides of the defaults). */
      recallMin: (typeof def.recallMin === 'number') ? def.recallMin : RECALL_MIN_SCORE,
      recallBudget: (def.recallBudget | 0) || RECALL_BUDGET_TOK,
      /* Optional REMOTE embeddings for the brain (local-first without). */
      embedProvider: def.embed_provider || def.embedProvider || '',
      embedModel: def.embed_model || def.embedModel || '',
      memory: (def.memory === 'shared' || def.memory === 'off') ? def.memory : 'agent',
      brainPath: def.brainPath || '',
      rlm: (def.rlm === 'on' || def.rlm === 'off') ? def.rlm : 'auto',
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
    /* P3: per-def tool-result context cap. */
    d.budget.maxToolResultChars = (b.maxToolResultChars | 0) || TOOL_RESULT_CAP_CHARS;

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
    var mo = (opts && opts.max_tokens) || d.maxTokensOut || 0;
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
    var budget = d.recallBudget > 0 ? d.recallBudget : RECALL_BUDGET_TOK;
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

  /* P3: cap one tool result before it enters any message. Head+tail are
   * kept (the middle is usually boilerplate) with an explicit marker that
   * teaches recovery: narrow the args and re-call. */
  function capToolResult(cap, s) {
    s = String(s == null ? '' : s);
    cap = (cap | 0) || TOOL_RESULT_CAP_CHARS;
    if (s.length <= cap) return { text: s, chars: s.length };
    var tailRoom = Math.min(TOOL_RESULT_TAIL, Math.max(0, cap - TOOL_RESULT_HEAD - 80));
    var omitted = s.length - TOOL_RESULT_HEAD - tailRoom;
    var text = s.slice(0, TOOL_RESULT_HEAD) +
      '\n…[' + omitted + ' chars truncated — re-call the tool with narrower args if needed]…\n' +
      (tailRoom > 0 ? s.slice(s.length - tailRoom) : '');
    return { text: text, chars: s.length };
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
      if (opts.logs) logRun(res, task, wall);
      delete ACTIVE[runId];
      delete SIGNALS[cancelId];
      ACTIVE_COUNT--;
      if (ACTIVE_COUNT === 0) runTick(mcpIdleDisconnect);
      return res;
    }

    emit('start', { task: clip(task, 300), depth: depth, chain: chain.slice() });

    try {
      /* Recall augmentation (A1.4, A3) — P2-gated: similarity floor →
       * dedupe vs what the model already sees → block token budget. Zero
       * relevant hits means zero injected bytes. */
      var brainEntry = brainFor(d);
      var cma = brainEntry ? brainEntry.cma : null;
      var recalledIds = [];
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
      if (recallBlock) ctxParts.push(recallBlock);
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
            if ((opts.max_tokens || d.maxTokensOut || 0) > 0) ro.max_tokens = opts.max_tokens || d.maxTokensOut;
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
        var result;
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
          emit('tool_result', { name: name, error: clip(String((e && e.message) || e), 160) });
        }
        if (result !== undefined && (typeof result.then === 'function')) {
          result = await result;
        }
        /* P3: uniform cap at the boundary (delegate answers included — the
         * child already capped its own answer; no double truncation). */
        var cappedR = capToolResult(d.budget.maxToolResultChars, result);
        emit('tool_result', { name: name, result: clip(cappedR.text, 200),
                              chars: cappedR.chars, kept: cappedR.text.length });
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
        if (breach0) { stopped = breach0; break; }
        emit('plan', { step: steps + 1, tools: tools.map(function (t) { return t.name; }) });
        var so = aiOpts(d, opts, messages.concat(loopMsgs), tools);
        var st = sofuu.ai.stream(so);
        state.aborts.push(function () { try { st.abort(); } catch (e) {} });
        var sparts = [];
        try {
          for await (var schunk of st) {
            if (schunk && schunk.think) { emit('think', { text: clip(schunk.think, 200) }); continue; }
            if (schunk && schunk.text) {
              sparts.push(schunk.text);
              if (opts.onStep) {
                try { opts.onStep({ runId: runId, name: d.name, depth: depth,
                                    t: Date.now() - started, kind: 'answer_delta', payload: schunk.text }); }
                catch (e2) {}
              }
            }
          }
        } catch (eS) {
          if (state.cancelled) { stopped = 'cancelled'; break; }
          throw eS;
        }
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
        var stcs = mergeStreamToolCalls(st.toolCalls || []);
        if (!stcs.length) {
          answer = sparts.join('').trim() || (state.cancelled ? '(cancelled)' : '(no response)');
          if (state.cancelled) stopped = stopped || 'cancelled';
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
        if (stopped) break;
      }

      await storeAndFinish(answer);
      return finish(answer);

      /* Final answer via ai.stream — emits answer_delta onStep events and
       * registers the stream's abort() so cancellation kills it mid-flight. */
      async function finalAnswer(msgs) {
        var o = aiOpts(d, opts, msgs);
        var stream = sofuu.ai.stream(o);
        var doAbort = function () { try { stream.abort(); } catch (e) {} };
        state.aborts.push(doAbort);
        var parts = [];
        for await (var chunk of stream) {
          if (chunk && chunk.think) { emit('think', { text: clip(chunk.think, 200) }); continue; }
          if (chunk && chunk.text) {
            parts.push(chunk.text);
            if (opts.onStep) {
              try { opts.onStep({ runId: runId, name: d.name, depth: depth,
                                  t: Date.now() - started, kind: 'answer_delta', payload: chunk.text }); }
              catch (e2) {}
            }
          }
        }
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
        var text = parts.join('').trim() || (state.cancelled ? '(cancelled)' : '(no response)');
        if (state.cancelled) stopped = stopped || 'cancelled';
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
})();
