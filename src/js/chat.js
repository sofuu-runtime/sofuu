/* src/js/chat.js — the shared chat turn engine (PLAN-DESKTOP workstream C).
 *
 * One turn engine for every host. The TUI grew its turn loop inside the
 * DRIVER string in chat.rs; the desktop app needs the SAME loop (history +
 * compaction, @file mentions, MCP + built-in tools, budgets, session
 * logging, no-think retry) without a terminal. This module is that loop,
 * extracted as shipped JS over the registered primitives (sofuu.agent,
 * sofuu.ai, sofuu.web, sofuu.tools, sofuu.mcp, sofuu.fs) plus the sync
 * session seams (rt/session_js.rs).
 *
 * API (all synchronous unless noted):
 *   sofuu.chat.init(opts)            → {ok, sessionId, project, model, provider}
 *   sofuu.chat.submit(text, opts)    → Promise<{ok, answer, usage, stopped}>
 *                                      opts.onEvent(ev) receives the stream;
 *                                      opts.noTools=true runs a pure chat
 *                                      turn with no tools attached
 *   sofuu.chat.cancel()              → aborts the running turn
 *   sofuu.chat.compact()             → Promise<{ok, savedTokens}>
 *   sofuu.chat.clear()               → {ok}
 *   sofuu.chat.resume(id)            → {ok, sessionId, turns}
 *   sofuu.chat.resolveApproval(id, allow, always) → {ok}
 *   sofuu.chat.state()               → {sessionId, project, model, …}
 *   sofuu.chat.sessionTurns(id)      → [{prompt, answer}] (sync)
 *   sofuu.chat.newSession()          → {ok, sessionId}
 *   sofuu.chat.setProject(path)      → {ok, sessionId, project}
 *
 * Event envelope on opts.onEvent: {kind, payload} — agent.js step events
 * pass through normalized (answer_delta payload becomes {text}), plus the
 * chat-level kinds: start, approval_request, approval_resolved, warn, done.
 *
 * Permission gate (opt-in via init({permissions:'prompt'}); the desktop sets
 * it): read-only tools auto-pass; everything else (write_file/edit_file/
 * bash, MCP tools) emits approval_request and awaits a host decision
 * delivered through the poke ({type:'approval', id, allow, always}) or the
 * direct resolveApproval call. Denial returns "denied by user" as the tool
 * result so the model can react. Cancel arrives as poke {type:'cancel'}.
 *
 * Session mesh: project-local sessions under <project>/.sofuu/sessions/.
 * The desktop writes sharded encrypted sessions (registry.qtsq + per-event
 * conv/*.qtsq, schema 2); the CLI also reads legacy plaintext registry.json
 * + flat <id>.qtsq (session.rs) as fallback so TUI and desktop sessions of
 * one project see each other. The fnv1a64 password is retained only for
 * reading legacy encrypted records.
 *
 * Deviation from the plan: ~/.sofuu/hooks.js middleware (F9) stays
 * TUI-only for now — the DRIVER's loader evals the file inline, a pattern
 * the shipped-JS security review rejects. A shared loader seam lands with
 * workstream B (chat.rs lib-ification).
 */
(function () {
  'use strict';
  if (!globalThis.sofuu) return; /* no engine namespace — nothing to do */

  /* ── Module state (threaded through every helper instead of the TUI
   * driver's closures, so the host can inspect/reset it). */
  var S = {
    initialized: false,
    host: 'embedded',
    project: '',
    permissions: 'auto',      /* 'auto' | 'prompt' */
    permissionProfile: 'prompt', /* 'full' | 'edit' | 'plan' | 'prompt' */
    toolTimeoutMs: 30000,     /* approval waits get init's human-scale value */
    pastTurns: 10,            /* resume depth */
    ctxAnnounced: false,      /* boot ctx event reached a sink (see submit) */
    cfg: null,                /* parsed ~/.sofuu/config.json (re-read per turn) */
    history: [],              /* [{role, content}] re-sent each turn */
    streaming: false,
    onEvent: null,            /* the live turn's event sink */
    info: null,               /* registry entry for this session */
    session: null,            /* SessionData mirror (persisted to the .qtsq) */
    pendingApprovals: {},     /* id → {resolve, tool} */
    alwaysAllowed: {},        /* tool name → true (session-scoped) */
    /* P2: bash gets command-scoped always-allow — approving "always" on a
     * shell command auto-runs THAT command again, never future arbitrary
     * ones. Keyed by the exact command string. */
    alwaysAllowedCommands: {},
    eventSeq: 0,
    recordSeq: {},
    approvalSeq: 0,
    noThink: [],              /* learned this process: model rejects thinking */
    totalTk: 0,               /* cumulative session tokens */
    sessionSpendUsd: 0,       /* cumulative session spend (pricing table) */
    autoCompactArmed: true,   /* one-shot auto-compaction per crossing */
    mcpClients: [],           /* [{name, client}] */
    mcpTools: [],             /* [{server, name, description, schema}] */
    mcpConnecting: null,      /* in-flight connect promise (dedupe) */
    agentsDirLoaded: false,
  };

  /* ── Small helpers ─────────────────────────────────────────────── */
  function env(n) { try { return process.env[n] || ''; } catch (e) { return ''; } }
  function home() { return env('HOME') || '.'; }
  function now() { return Math.floor(Date.now() / 1000); }
  function pid() { try { return process.pid | 0; } catch (e) { return 0; } }
  function clip(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n) + '…' : s;
  }
  function has(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }

  function emit(ev) {
    if (typeof S.onEvent === 'function') {
      try { S.onEvent(ev); } catch (e) {} /* a dead sink never kills a turn */
    }
  }

  /* ── Config (~/.sofuu/config.json, shared with the CLI) ──────────
   * Read synchronously through the session seam; re-read at the start of
   * every turn so settings changed in the host UI apply immediately. API
   * keys may be absent here (the desktop keeps them in the Keychain and
   * injects them via embed_config) — sofuu.ai.complete resolves the key
   * itself (explicit → embedded → env), so chat.js never has to see it. */
  function configPath() { return home() + '/.sofuu/config.json'; }
  /* A corrupt file must never break the turn, but it must not be SILENT
   * either — the user's providers/model silently vanished. Warn once per
   * process (console.warn reaches stderr; the warn event reaches the host
   * UI when a sink is attached). The file is left in place for manual
   * repair — mirrors ChatConfig::load in chat.rs. */
  var CONFIG_WARNED = false;
  function warnCorruptConfig() {
    if (CONFIG_WARNED) return;
    CONFIG_WARNED = true;
    var p = configPath();
    var msg = p + ' is corrupt — using defaults; providers/settings from this file are being ignored (left in place, not overwritten)';
    var line = 'Warning: ' + msg;
    try { console.warn(line); } catch (e) {}
    emit({ kind: 'warn', payload: { message: msg } });
  }
  function loadConfig() {
    try {
      var raw = (typeof __session_read_file === 'function') ? __session_read_file(configPath()) : null;
      var c = {};
      if (raw) { try { c = JSON.parse(raw); } catch (eParse) { warnCorruptConfig(); } }
      if (!c || typeof c !== 'object') c = {};
      /* Flat mirror from the ACTIVE provider entry (host UIs edit the
       * providers list; the flat fields are a cache of it). The sync is
       * UNCONDITIONAL for fields the entry defines — filling only empty
       * fields left a STALE flat mirror behind after a provider switch
       * (old endpoint + old key → 401s against the model the user
       * selected). Missing entry fields keep the flat value (a provider
       * entry may deliberately omit profile, e.g. openai-compat). */
      if (Array.isArray(c.providers) && c.providers.length) {
        var active = null;
        for (var i = 0; i < c.providers.length; i++) {
          if (c.providers[i] && c.providers[i].name === c.active) { active = c.providers[i]; break; }
        }
        if (!active) active = c.providers[0];
        if (active) {
          if (active.name) c.provider = active.name;
          if (active.model) c.model = active.model;
          if (active.endpoint) c.base_url = active.endpoint;
          if (active.profile) c.profile = active.profile;
          if (active.api_key) c.api_key = active.api_key;
        }
      }
      S.cfg = c;
    } catch (e) {
      if (!S.cfg) S.cfg = {};
    }
    return S.cfg;
  }
  function cfg() { return S.cfg || loadConfig(); }

  /* Known built-in providers (mirrors chat.rs usableConfig). */
  var KNOWN_PROVIDERS = ['openai', 'anthropic', 'local'];
  function usableConfig() {
    var c = cfg();
    if (!c.provider || !c.model) return false;
    if (KNOWN_PROVIDERS.indexOf(c.provider) < 0 && !c.base_url) return false;
    if (c.provider === 'local') return false; /* not shipped yet */
    return true;
  }

  /* ── Token budgets + history compaction (ported from the DRIVER) ── */
  var MAX_HISTORY_ENTRIES = 2000;
  var COMPACT_AT = 0.70;      /* auto-compact past 70% of the budget */

  function modelCaps() {
    try {
      if (sofuu.ai && typeof sofuu.ai.modelCaps === 'function') {
        var c = activeCfg();
        /* Second arg: the ENDPOINT this request will hit — when present,
         * caps the endpoint itself published for this exact model (the
         * discovered store) override the name-keyed registry. */
        var j = sofuu.ai.modelCaps(c.model || '', c.base_url || undefined);
        var caps = JSON.parse(j);
        return (caps && caps.known) ? caps : null;
      }
    } catch (e) {}
    return null;
  }
  /* The evidence-ladder resolve (Rust: ml::alloc::policy::resolve) —
   * learned-from-400s > endpoint-discovered > registry > conservative
   * default. A flat global ctx_window/max_output in config can only
   * SHRINK the resolved numbers, never inflate them: the strictest real
   * evidence clamps it down. This is what makes the ring denominator,
   * compaction budget and output cap track the SELECTED model instead of
   * one stale global number (the 1M-vs-262k bug). NOT cached across
   * turns: a mid-session 400 or a listing harvest can promote the
   * evidence and the next resolve must see it (the native is a HashMap
   * lookup — cheap). */
  /* Failover target for the CURRENT turn (null = active provider). Set
   * by submit()'s failover chain before turn() runs; every helper that
   * resolves caps or stream options reads it through activeCfg() so the
   * whole turn — wire options, caps ladder, def construction — runs
   * against the SAME provider. Serial turns make this safe. */
  function activeCfg() {
    var base = cfg();
    var ov = S.providerOverride;
    if (!ov) return base;
    return Object.assign({}, base, {
      provider: ov.name,
      base_url: ov.endpoint,
      api_key: ov.api_key,
      profile: ov.profile || '',
      model: ov.model,
    });
  }

  function resolveCaps() {
    var c = activeCfg();
    var v = null;
    try {
      if (sofuu.ai && typeof sofuu.ai.resolveCaps === 'function') {
        v = JSON.parse(sofuu.ai.resolveCaps(
          c.model || '', c.base_url || '',
          c.ctx_window > 0 ? c.ctx_window : 0,
          c.max_output > 0 ? c.max_output : 0,
          c.ctx_window_explicit === true));
      }
    } catch (e) { v = null; }
    if (!v || !(v.window > 0)) {
      /* No native (old runtime) or resolve failed: legacy ladder, still
       * per-model — config first but capped by any caps evidence we have.
       * An EXPLICIT value (`/ctx n` this session) keeps the caller's
       * number; an inherited global still shrinks to the evidence. */
      var capsL = modelCaps() || {};
      var explicit = c.ctx_window_explicit === true;
      var win = c.ctx_window > 0 ? c.ctx_window
        : (capsL.ctxWindow > 0 ? capsL.ctxWindow
        : ({ openai: 128000, anthropic: 128000, local: 32768 }[c.provider] || 32768));
      if (!explicit && capsL.ctxWindow > 0 && win > capsL.ctxWindow) win = capsL.ctxWindow;
      var mx = c.max_output > 0 ? c.max_output
        : (capsL.maxOutput > 0 ? capsL.maxOutput : 0);
      if (capsL.maxOutput > 0 && mx > capsL.maxOutput) mx = capsL.maxOutput;
      v = { window: win, maxOutput: mx, known: !!(capsL.known), source: 'legacy',
            clampedConfig: false, winSource: 'legacy', maxSource: 'legacy',
            thinking: (capsL.thinking || 'unknown') };
    }
    return v;
  }
  function ctxWindow() { return resolveCaps().window; }
  function ctxBudget() { return Math.floor(ctxWindow() * 0.85); }

  /* P2-5: the TUI's alloc-plan bridge, adapted for this driver. Resolves
   * inside Rust (registry → learned limits → conservative defaults); JS
   * only supplies usage state. Returns null when ML is off/unavailable —
   * every consumer keeps the legacy ratios exactly as before. */
  function allocPlanChat(taskTk, attachTk) {
    var c = cfg();
    if (!c || !c.ml) return null;
    try {
      if (!sofuu.ml || !sofuu.ml.alloc || typeof sofuu.ml.alloc.plan !== 'function') return null;
      var sizes = [];
      var histTk = 0;
      for (var i = 0; i < S.history.length; i++) {
        var t = estTok(String(S.history[i].content || ''));
        sizes.push(t); histTk += t;
      }
      var growthTk = 0, accel = 0;
      if (sizes.length >= 2) {
        var last3 = sizes.slice(-3);
        growthTk = last3.reduce(function (a, b) { return a + b; }, 0) / last3.length;
        if (sizes.length >= 4) {
          var half = Math.floor(sizes.length / 2);
          var m1 = sizes.slice(0, half).reduce(function (a, b) { return a + b; }, 0) / half;
          var m2 = sizes.slice(half).reduce(function (a, b) { return a + b; }, 0) / (sizes.length - half);
          if (m1 > 0) accel = m2 / m1;
        }
      }
      var p = JSON.parse(sofuu.ml.alloc.plan(JSON.stringify({
        model: c.model || '',
        baseUrl: c.base_url || '',
        cfgWindow: c.ctx_window > 0 ? c.ctx_window : 0,
        cfgMaxOutput: c.max_output > 0 ? c.max_output : 0,
        overheadTk: 0, calibrated: false,
        historyTk: histTk,
        turns: Math.floor(S.history.length / 2),
        toolFrac: 0,
        summary: S.history.length > 0 && S.history[0].role === 'system' &&
          String(S.history[0].content || '').indexOf('Prior conversation summary') === 0,
        growthTk: growthTk, growthAccel: accel,
        taskTk: taskTk || 0, attachTk: attachTk || 0,
        toolHeavy: true,
        writing: false,
      })));
      return (p && p.window > 0) ? p : null;
    } catch (e) { return null; }
  }
  function attachTokenBudget() {
    return Math.min(65536, Math.max(2048, Math.floor(ctxWindow() * 0.25)));
  }
  function estTok(s) {
    if (sofuu.ai && typeof sofuu.ai.estimateTokens === 'function') {
      try { return sofuu.ai.estimateTokens(s) | 0; } catch (e) {}
    }
    return Math.ceil(String(s).length / 4);
  }
  /* ── Turn blocks (Claude-style retention, 2026-09-03 e) ────────────
   * History is a list of TURN BLOCKS: one user message, then zero or more
   * retained tool_calls/tool messages (the turn's transcript), then the
   * assistant answer. Trimming/compaction must move whole blocks — an
   * orphaned tool message (its assistant tool_calls pair dropped) is a
   * hard 400 at OpenAI-compatible providers. turnStartAt(i) walks BACKWARD
   * from the message at i: the block starts at the user message that
   * precedes it (tool/tool_calls messages are never block starts; a
   * leading system summary is its own pseudo-block). */
  function isToolMsg(m) {
    return !!m && (m.role === 'tool' || (m.role === 'assistant' && m.tool_calls && m.tool_calls.length));
  }
  function turnStartAt(i) {
    var j = Math.max(0, Math.min(i, S.history.length - 1));
    while (j > 0 && S.history[j].role !== 'user') j--;
    return j;
  }
  function removeTurnBlock(i) {
    var s = turnStartAt(i);
    var e = s + 1;
    while (e < S.history.length && S.history[e].role !== 'user') e++;
    S.history.splice(s, e - s);  }
  /* A turn the model must never be allowed to delete on its own
   * authority (compaction is the one irreversible action in the chat).
   * Deliberately LEXICAL and model-free: a question the user asked, an
   * explicit instruction, or a recorded decision/approval. This is the
   * belt to the compaction gate's braces — if the gate is ever wrong
   * (and the held-out audit says a linear model is exactly as good as
   * the network on this data), the worst case is a few extra tokens,
   * not a silently deleted instruction. */
  var PROTECT_RE = /(\?|^\s*(please\s+)?(do\s+not|don't|never|always|must|make\s+sure|keep)\b)|(\b(approved|approve|accepted|rejected|decided|decision|agreed|sign\s*off)\b)/i;
  function isProtectedBlock(hist, idx) {
    var s = turnStartAt(idx);
    if (s < 0 || s >= hist.length) return false;
    var e = s + 1;
    while (e < hist.length && hist[e].role !== 'user') e++;
    for (var i = s; i < e; i++) {
      var m = hist[i];
      if (!m) continue;
      /* Only the USER's own words protect the turn: an assistant
       * message can contain anything, including boilerplate that trips
       * the pattern. */
      if (m.role === 'user' && PROTECT_RE.test(String(m.content || ''))) return true;
    }
    return false;
  }
  function historyTokens() {
    var n = 0;
    for (var i = 0; i < S.history.length; i++) {
      n += estTok(String(S.history[i].content || ''));
      /* Retained transcripts bill their tool_calls JSON too — the provider
       * counts the whole assistant message, not just its (null) content. */
      if (S.history[i].tool_calls) {
        try { n += estTok(JSON.stringify(S.history[i].tool_calls)); } catch (eTC) {}
      }
    }
    return n;
  }

  function streamOpts() {
    var c = activeCfg();
    var o = { messages: [], provider: c.provider, model: c.model };
    if (c.effort) o.effort = c.effort;
    if (c.api_key) o.api_key = c.api_key;
    if (c.base_url) o.base_url = c.base_url;
    if (c.profile) o.profile = c.profile;
    /* Output cap from the ladder: config honored only as far as the
     * strictest real evidence allows (a 384k global can't override an
     * 8192-max model anymore). A number the ladder DEFAULTED (model
     * unknown to every source) is never sent — the endpoint applies its
     * own default (Pass-31 rule). Layer 0 re-clamps on the wire. */
    var rc = resolveCaps();
    if (rc.maxOutput > 0 && rc.maxSource !== 'default') o.max_tokens = rc.maxOutput;
    return o;
  }
  async function complete(messages, opts) {
    var o = streamOpts(); o.messages = messages;
    if (opts && opts.tools) o.tools = opts.tools;
    var r = await sofuu.ai.complete(o);
    if (r && r.toolCalls) return r;
    return r && r.text ? r.text.trim() : '(no response)';
  }

  /* Summarize all but the last keepTurns turns into one system entry
   * (same shape the TUI's /compact produces). False = nothing to fold or
   * the summarizer failed — callers fall back to dropping, never block.
   * Turn blocks fold whole: a block's retained tool transcript goes into
   * the summary text (what happened), never into the kept context (the
   * orphaned-pairs problem). */
  async function summarizeHistory(keepTurns) {
    /* Block-aware split point: index where the (keepTurns+1)-th-from-last
     * turn block starts. Everything before it is folded. */
    var foldFrom = S.history.length;
    var seen = 0;
    for (var fi = S.history.length - 1; fi >= 0; fi--) {
      if (S.history[fi].role === 'user') {
        seen++;
        if (seen > keepTurns) { foldFrom = fi; break; }
      }
    }
    /* Nothing to fold only when EVERY block is kept (the scan never moved
     * foldFrom) — foldFrom === 0 is the normal two-block case and must fold. */
    if (foldFrom >= S.history.length) return false;
    var oldCount = foldFrom;
    var summary = await complete([
      { role: 'system', content: 'You are a conversation summarizer. Compress the following conversation into a compact summary that preserves key facts, decisions, and the user\'s intent. Output only the summary.' }
    ].concat(S.history.slice(0, oldCount)));
    if (!summary || typeof summary !== 'string' || !summary.trim() || summary.trim() === '(no response)') return false;
    S.history = [{ role: 'system', content: 'Prior conversation summary: ' + summary }].concat(S.history.slice(oldCount));
    return true;
  }
  async function trimHistory() {
    /* Hard safety net: the absolute entry cap — shed whole turn blocks from
     * the front so retained tool transcripts are never orphaned. */
    if (S.history.length > MAX_HISTORY_ENTRIES) {
      var over = S.history.length - MAX_HISTORY_ENTRIES;
      /* Fast path: the front is a system summary or user-started block —
       * removeTurnBlock clamps to the enclosing block. Multiple blocks
       * may be needed; each call is block-atomic. */
      while (over > 0 && S.history.length > 0) {
        var before0 = S.history.length;
        removeTurnBlock(0);
        over -= (before0 - S.history.length);
      }
    }
    var budget = ctxBudget();
    /* P2-5 (AUDIT-2026-09-01): ported the TUI's ML compaction machinery —
     * this driver used a fixed 0.70 cliff + no ML pass, so long embedded
     * sessions compacted earlier and lost strictly more context than the
     * TUI. allocPlanChat moves the cliff with context pressure (0.70
     * slack → 0.50 when the net sees overflow within ~3 turns); the ML
     * gate below frees mechanical junk (dup/boilerplate/re-fetchable turns)
     * BEFORE the cliff ever fires. Identical shape to chat.rs trimHistory.
     * With ML off the behaviour is exactly the old fixed cliff. */
    var compactAt = COMPACT_AT;
    try {
      var planC = allocPlanChat(0, 0);
      if (planC && planC.compactAt > 0) compactAt = planC.compactAt;
    } catch (ePC) {}
    var cMl = cfg();
    if (cMl && cMl.ml && S.history.length > 4 && historyTokens() > budget * 0.5) {
      try {
        if (sofuu.ml && sofuu.ml.compaction && typeof sofuu.ml.compaction.plan === 'function') {
          var segs = [], segHist = [];
          for (var i = 0; i < S.history.length; i++) {
            var m = S.history[i];
            if (m.role !== 'user' && m.role !== 'assistant') continue;
            var text = String(m.content || '');
            segs.push({ text: text, tokens: estTok(text), age: 0,
                        kind: m.role === 'user' ? 0 : 1,
                        retrievable: false, compacted: false });
            segHist.push(i);
          }
          for (var j = 0; j < segs.length; j++) segs[j].age = segs.length - 1 - j;
          if (segs.length > 4) {
            var taskTxt = '';
            for (var ti = S.history.length - 1; ti >= 0; ti--) {
              if (S.history[ti].role === 'user') { taskTxt = String(S.history[ti].content || ''); break; }
            }
            var recentTxt = S.history.slice(-4).map(function (mm) { return String(mm.content || ''); }).join('\n');
            var plan = JSON.parse(sofuu.ml.compaction.plan(JSON.stringify({
              task: taskTxt, summary: '', recent: recentTxt, segments: segs,
            }), JSON.stringify({ budget: historyTokens() })) || '{}');
            var compact = plan.compact || [], tiers = plan.tiers || [];
            var flaggedFree = {};
            for (var ci = 0; ci < compact.length; ci++) {
              var t = tiers[ci];
              if (t !== 'dup' && t !== 'boilerplate' && t !== 'retrievable') continue;
              var hi = segHist[compact[ci]];
              if (hi !== undefined) flaggedFree[hi] = 1;
            }
            /* Complete turn BLOCKS only (user+assistant both flagged),
             * never the newest block; oldest first, until usage drains
             * below half. Removal is block-atomic — a flagged block's
             * retained tool transcript leaves with it. */
            var blockStarts = [];
            for (var bi2 = 0; bi2 < S.history.length - 1; bi2++) {
              if (S.history[bi2].role !== 'user') continue;
              /* find this block's answer (the first plain assistant) */
              var bAns = -1;
              for (var bj2 = bi2 + 1; bj2 < S.history.length; bj2++) {
                if (S.history[bj2].role === 'user') break;
                if (S.history[bj2].role === 'assistant' && !S.history[bj2].tool_calls) { bAns = bj2; break; }
              }
              if (bAns >= 0 && flaggedFree[bi2] && flaggedFree[bAns]) {
                blockStarts.push(bi2);
              }
            }
            var freedTk = 0, droppedTurns = 0;
            /* blockObjs hold the block user-message OBJECTS — after each
             * removal the array shifts, so re-locate by identity. */
            var blockObjs = blockStarts.map(function (bx) { return S.history[bx]; });
            /* SAFETY CAP (2026-09-25): one pass may never gut the session.
             * The loop below only stops when usage drains below half the
             * budget, which on a long history is permission to delete
             * dozens of turns if the gate is wrong. Cap the blast radius
             * at a quarter of the complete blocks (min 1, and never the
             * two most recent blocks, which are the live context). */
            var completeBlocks = 0;
            for (var cb = 0; cb < S.history.length - 1; cb++) {
              if (S.history[cb].role === 'user') completeBlocks++;
            }
            var maxDrops = Math.max(1, Math.floor(completeBlocks * 0.25));
            if (completeBlocks - maxDrops < 2) maxDrops = Math.max(0, completeBlocks - 2);
            var skippedProtected = 0;
            for (var bN = 0; bN < blockObjs.length && droppedTurns < maxDrops &&
                   historyTokens() > budget * 0.5; bN++) {
              var blockIdx = S.history.indexOf(blockObjs[bN]);
              if (blockIdx < 0) continue; /* already removed */
              /* MODEL-INDEPENDENT GUARD: the gate is the component under
               * suspicion, so the last line of defence cannot be the
               * gate. A turn the user asked as a question, or one that
               * carries a decision/approval, is never removed on the
               * model's word alone. */
              if (isProtectedBlock(S.history, blockIdx)) { skippedProtected++; continue; }
              var beforeBlk = historyTokens();
              removeTurnBlock(blockIdx);
              freedTk += beforeBlk - historyTokens();
              droppedTurns++;
            }
            if (droppedTurns > 0) {
              emit({ kind: 'warn', payload: { message: 'ml-compaction freed ~' + freedTk +
                     ' tk (' + droppedTurns + ' of ' + completeBlocks + ' turn' +
                     (completeBlocks === 1 ? '' : 's') +
                     ': dup/boilerplate/re-fetchable' +
                     (skippedProtected > 0 ? '; ' + skippedProtected + ' protected turn' +
                       (skippedProtected === 1 ? '' : 's') + ' kept' : '') + ')' } });
            }
          }
        }
      } catch (eMlC) { /* gate advises — failure falls through to the cliff */ }
    }
    if (!S.autoCompactArmed && historyTokens() < budget * 0.5) S.autoCompactArmed = true;
    if (S.autoCompactArmed && historyTokens() > budget * compactAt && S.history.length > 2) {
      S.autoCompactArmed = false;
      var savedTk = historyTokens();
      /* Compact lifecycle for live UIs: start before the summarizer runs
       * (it's an LLM call — can take a while), the saved-tokens warn is
       * the completion payload, end settles or removes the row. */
      emit({ kind: 'compact', payload: { stage: 'start' } });
      var foldedCompact = false;
      try {
        foldedCompact = !!(await summarizeHistory(0));
      } catch (e) { /* summarizer failed — the drop loop below still guards */ }
      if (foldedCompact) {
        emit({ kind: 'warn', payload: { message: 'auto-compacted → summary (' +
               Math.max(0, savedTk - historyTokens()) + ' tk saved, window → ~0)' } });
      }
      emit({ kind: 'compact', payload: { stage: 'end', ok: foldedCompact } });
    }
    /* Drop-oldest guard: still over budget after compaction (or it failed).
     * Whole turn blocks only — an orphaned tool message 400s at the provider. */
    var dropped = 0;
    while (S.history.length > 0 && historyTokens() > budget) {
      /* never drop the LAST block (the newest turn) */
      var lastStart = turnStartAt(S.history.length - 1);
      if (lastStart === 0) break;
      var beforeDrop = historyTokens();
      removeTurnBlock(0);
      if (historyTokens() >= beforeDrop) break; /* safety: no progress */
      dropped++;
    }
    if (dropped > 0) {
      emit({ kind: 'warn', payload: { message: 'history trimmed by ' + dropped +
             ' turn' + (dropped === 1 ? '' : 's') + ' to fit the context budget' } });
    }
  }

  /* ── MCP tool wiring (ported from the DRIVER; lazy, never fatal) ── */
  /* Corrupt mcp.json must not break chat (empty list) but must not be
   * SILENT either — the user's MCP servers silently vanished. Warn once
   * per process; the file is left in place — mirrors ChatConfig::mcp_servers
   * in chat.rs. A raw non-array JSON body is malformed for this file but
   * stays quiet (server entries could not have parsed anyway). */
  var MCP_WARNED = false;
  function mcpServersList() {
    var raw = null;
    try {
      raw = (typeof __session_read_file === 'function')
        ? __session_read_file(home() + '/.sofuu/mcp.json') : null;
    } catch (e) { raw = null; }
    if (!raw) return [];
    var list = [];
    try {
      list = JSON.parse(raw);
    } catch (eParse) {
      if (!MCP_WARNED) {
        MCP_WARNED = true;
        var line = 'Warning: ' + home() + '/.sofuu/mcp.json is corrupt — MCP servers disabled until it is fixed (left in place, not overwritten)';
        try { console.warn(line); } catch (e2) {}
        emit({ kind: 'warn', payload: { message: home() + '/.sofuu/mcp.json is corrupt — MCP servers disabled until it is fixed (left in place, not overwritten)' } });
      }
      return [];
    }
    return Array.isArray(list) ? list : [];
  }
  async function connectMcpServers() {
    if (S.mcpConnecting) return S.mcpConnecting; /* one connect at a time */
    var list = mcpServersList();
    if (!list.length) return;
    if (!sofuu.mcp || !sofuu.mcp.connect) {
      emit({ kind: 'warn', payload: { message: 'sofuu.mcp unavailable — tools off' } });
      return;
    }
    S.mcpConnecting = (async function () {
      var results = await Promise.allSettled(list.map(async function (srv) {
        var client = await sofuu.mcp.connect(srv.command);
        var res = await client.listTools();
        var arr = (res && res.tools) || [];
        return { name: srv.name, client: client, tools: arr };
      }));
      var connected = 0, failed = 0;
      for (var i = 0; i < results.length; i++) {
        var r = results[i];
        if (r.status === 'fulfilled' && r.value) {
          S.mcpClients.push({ name: r.value.name, client: r.value.client });
          for (var j = 0; j < (r.value.tools || []).length; j++) {
            var t = r.value.tools[j];
            S.mcpTools.push({ server: r.value.name, name: t.name,
                              description: t.description || '', schema: t.inputSchema });
          }
          connected++;
        } else {
          failed++;
        }
      }
      if (connected > 0) {
        emit({ kind: 'warn', payload: { message: 'mcp: ' + connected + ' server' +
               (connected === 1 ? '' : 's') + ' · ' + S.mcpTools.length + ' tools' } });
      }
      if (failed > 0) {
        emit({ kind: 'warn', payload: { message: failed + ' MCP server' +
               (failed === 1 ? '' : 's') + ' unreachable (see ~/.sofuu/mcp.json)' } });
      }
    })();
    try {
      await S.mcpConnecting;
    } finally {
      S.mcpConnecting = null;
    }
  }
  async function callMcpTool(name, args) {
    /* Route by name→OWNER server (A6: first-server-wins was wrong under
     * multiple servers exposing the same tool name). */
    var owner = null;
    for (var i = 0; i < S.mcpTools.length; i++) {
      if (S.mcpTools[i].name === name) { owner = S.mcpTools[i].server; break; }
    }
    if (!owner) throw new Error('tool ' + name + ' not found on any MCP server');
    for (var j = 0; j < S.mcpClients.length; j++) {
      var c = S.mcpClients[j];
      if (c.name !== owner) continue;
      var res = await c.client.call('tools/call', { name: name, arguments: args || {} });
      var text = res && res.content && res.content[0] && res.content[0].text;
      return (typeof text === 'string') ? text : JSON.stringify(text);
    }
    throw new Error('tool ' + name + ': owning server "' + owner + '" is not connected');
  }

  /* ── Permission gate (PLAN-DESKTOP C) ──────────────────────────────
   * Profiles (setPermissions, persisted by the desktop):
   *   full   — every tool passes without asking
   *   edit   — read-only + write_file/edit_file pass; bash + MCP blocked
   *   plan   — read-only only; the model plans, changes nothing
   *   prompt — default desktop gate: non-read-only tools emit
   *            approval_request and await the host's decision.
   * Read-only tools always pass. Denied/blocked tools return a STRING as
   * the tool result so the model can react instead of the run dying. */
  var AUTO_PASS_TOOLS = {
    read_file: 1, grep: 1, glob: 1, list_dir: 1, todo_write: 1,
    web_search: 1, web_open: 1,
  };
  /* 'edit' profile: read-only plus jailed local edits pass; everything
   * else (bash, MCP) is blocked. In 'prompt' these still ask: they are
   * NOT in AUTO_PASS_TOOLS. */
  var EDIT_TOOLS = { write_file: 1, edit_file: 1 };
  function blockedTool(def, why) {
    return {
      name: def.name,
      description: def.description,
      parameters: def.parameters,
      execute: function () {
        return 'Blocked by permissions policy (' + why + '): ' + def.name;
      },
    };
  }
  function profileGates(name) {
    var p = S.permissionProfile;
    if (p === 'full') return null;                        /* pass everything */
    if (AUTO_PASS_TOOLS[name]) return null;               /* read-only */
    if (p === 'edit' && EDIT_TOOLS[name]) return null;    /* jailed edits */
    if (p === 'edit') return 'edit-only';                 /* bash + MCP */
    if (p === 'plan') return 'plan mode';                 /* no writes, no shell */
    return null;                                          /* prompt: ask below */
  }
  function requestApproval(tool, args) {
    S.approvalSeq++;
    // Unpredictable IDs: seq + time alone lets any WebView XSS guess the
    // next approval id and self-approve (resolve_approval). Mix in 64 bits
    // of randomness so only the holder of the approval_request event can
    // resolve it.
    var rand = '';
    try {
      if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
        var buf = new Uint8Array(8);
        crypto.getRandomValues(buf);
        for (var i = 0; i < buf.length; i++) rand += buf[i].toString(16).padStart(2, '0');
      } else {
        rand = Math.floor(Math.random() * 0xffffffff).toString(16) +
               Math.floor(Math.random() * 0xffffffff).toString(16);
      }
    } catch (e) {
      rand = Math.floor(Math.random() * 0xffffffff).toString(16) + Date.now().toString(36);
    }
    var id = 'ap-' + S.approvalSeq + '-' + Date.now().toString(36) + '-' + rand;
    return new Promise(function (resolve) {
      var entry = { resolve: resolve, tool: tool, args: args || {} };
      S.pendingApprovals[id] = entry;
      emit({ kind: 'approval_request',
             payload: { id: id, tool: tool, args: args || {}, risky: true } });
      /* Hard ceiling so a vanished host can't pin the turn forever —
       * toolTimeoutMs (default 30s; init may raise it for the desktop).
       * The handle is
       * stored on the entry so an early resolve can clear it: every
       * uncleared timer parks in the runtime's 1024-slot registry until
       * it fires, so one leak per approval eventually makes later
       * setTimeout calls throw (js-3). */
      entry.timer = setTimeout(function () {
        if (S.pendingApprovals[id]) {
          delete S.pendingApprovals[id];
          emit({ kind: 'approval_resolved', payload: { id: id } });
          resolve(false);
        }
      }, S.toolTimeoutMs);
    });
  }
  function gateTool(def) {
    var blocked = profileGates(def.name);
    if (blocked) return blockedTool(def, blocked);
    if (S.permissionProfile !== 'prompt') return def; /* full/edit auto-pass */
    if (S.permissions !== 'prompt') return def;       /* legacy 'auto' init */
    var name = def.name;
    if (AUTO_PASS_TOOLS[name]) return def;
    var execute = def.execute;
    return {
      name: def.name,
      description: def.description,
      parameters: def.parameters,
      execute: function (args) {
        if (S.alwaysAllowed[name]) return execute(args);
        /* Command-scoped always-allow (bash): the exact string the user
         * approved auto-runs; anything else still gates. */
        if (name === 'bash' && S.alwaysAllowedCommands[String(args && args.command)]) {
          return execute(args);
        }
        return requestApproval(name, args).then(function (allowed) {
          if (!allowed) return 'denied by user';
          return execute(args);
        });
      },
    };
  }
  function resolveApproval(id, allow, always) {
    var key = String(id);
    var p = S.pendingApprovals[key];
    if (!p) return { ok: false, error: 'no pending approval with that id' };
    delete S.pendingApprovals[key];
    /* The approval settled early — disarm its ceiling timer so the slot
     * returns to the runtime registry immediately (js-3). */
    try { if (p.timer) clearTimeout(p.timer); } catch (eT) {}
    if (always && allow) {
      /* bash never gets a wholesale session pass — 'always' scopes to the
       * exact command string (the whole-tool pass would auto-run any
       * future command, approval-free). */
      if (p.tool === 'bash' && p.args && typeof p.args.command === 'string' && p.args.command) {
        S.alwaysAllowedCommands[p.args.command] = true;
      } else {
        S.alwaysAllowed[p.tool] = true;
      }
    }
    emit({ kind: 'approval_resolved', payload: { id: key } });
    p.resolve(!!allow);
    return { ok: true };
  }

  /* Tool set handed to sofuu.agent.run each turn (ported from the
   * DRIVER's chatToolDefs + the permission gate): built-in web tools +
   * coding tools + MCP tools as inline specs. Inline/builtin wins a name
   * clash with a visible warn. */
  function chatToolDefs() {
    var defs = [];
    var names = {};
    if (sofuu.web && sofuu.web.TOOLS) {
      var webKeys = ['web_search', 'web_open'];
      for (var w = 0; w < webKeys.length; w++) {
        var wt = sofuu.web.TOOLS[webKeys[w]];
        if (!wt) continue;
        defs.push(gateTool({ name: wt.name, description: wt.description,
                             parameters: wt.parameters, execute: wt.execute }));
        names[wt.name] = 1;
      }
    }
    if (sofuu.tools && sofuu.tools.TOOLS && !env('SOFUU_NO_TOOLS')) {
      var group = sofuu.tools.GROUP || [];
      for (var g = 0; g < group.length; g++) {
        var t = sofuu.tools.TOOLS[group[g]];
        if (!t) continue;
        defs.push(gateTool({ name: t.name, description: t.description,
                             parameters: t.parameters, execute: t.execute }));
        names[t.name] = 1;
      }
    }
    for (var i = 0; i < S.mcpTools.length; i++) {
      var mt = S.mcpTools[i];
      if (names[mt.name]) {
        emit({ kind: 'warn', payload: { message: 'MCP tool ' + mt.name +
               ' (' + mt.server + ') shadowed by a built-in tool' } });
        continue;
      }
      defs.push(gateTool({
        name: mt.name,
        description: mt.description || ('MCP tool from ' + mt.server),
        parameters: mt.schema || { type: 'object', properties: {} },
        execute: (function (toolName) {
          return function (args) { return callMcpTool(toolName, args); };
        })(mt.name),
      }));
      names[mt.name] = 1;
    }
    return defs;
  }

  /* ── @file mentions (ported from the DRIVER, F3) ─────────────────── */
  async function expandMentions(text) {
    var mentions = [];
    var re = /@(?:("[^"]+")|([^\s]+(?:\:\d+-\d+)?))/g;
    text.replace(re, function (full, quoted, bare) {
      mentions.push(quoted ? quoted.slice(1, -1) : bare);
      return full;
    });
    if (mentions.length === 0) return { text: text, manifest: '' };
    var cwd = process.cwd();
    var totalTokens = 0;
    var blocks = [];
    var manifestParts = [];
    for (var i = 0; i < mentions.length; i++) {
      var mention = mentions[i];
      var pathPart = mention;
      var startLine, endLine;
      var rangeMatch = mention.match(/^(.+)\:(\d+)-(\d+)$/);
      if (rangeMatch) {
        pathPart = rangeMatch[1];
        startLine = parseInt(rangeMatch[2], 10);
        endLine = parseInt(rangeMatch[3], 10);
        /* P3-7 (twin of the TUI fix): "0-5" → slice(-1,5) wraps and
         * returns the LAST line. Line numbers are 1-based. */
        if (startLine < 1) startLine = 1;
        if (endLine < startLine) endLine = startLine;
      }
      var content = '';
      try {
        /* Reject paths escaping the project root (documented contract).
         * Symlink-aware: a link inside the project pointing outside (e.g.
         * ./link → /etc/passwd) passes the lexical prefix check, so also
         * canonicalize via sofuu.fs.realpath when available. */
        var normalized = pathPart.charAt(0) === '/' ? pathPart : (cwd + '/' + pathPart);
        var underRoot = normalized === cwd || normalized.indexOf(cwd + '/') === 0;
        if (pathPart.indexOf('..') >= 0 || !underRoot) {
          blocks.push('### ' + pathPart + '\n```\n(rejected: path escapes project root)\n```');
          manifestParts.push('@' + pathPart + ' (rejected)');
          continue;
        }
        if (sofuu.fs && typeof sofuu.fs.realpath === 'function') {
          try {
            var real = await sofuu.fs.realpath(pathPart.charAt(0) === '/' ? pathPart : (cwd + '/' + pathPart));
            var cwdReal = await sofuu.fs.realpath(cwd);
            if (real && cwdReal && !(real === cwdReal || real.indexOf(cwdReal + '/') === 0)) {
              blocks.push('### ' + pathPart + '\n```\n(rejected: symlink escapes project root)\n```');
              manifestParts.push('@' + pathPart + ' (rejected)');
              continue;
            }
          } catch (eReal) { /* missing file falls through to read error below */ }
        }
        if (sofuu.fs && typeof sofuu.fs.readFile === 'function') {
          content = await sofuu.fs.readFile(pathPart, 'utf8');
        } else {
          content = '';
        }
        if (!content) {
          blocks.push('### ' + pathPart + '\n```\n(file not found or empty)\n```');
          manifestParts.push('@' + pathPart + ' (missing)');
          continue;
        }
        if (startLine !== undefined) {
          var lines0 = content.split('\n');
          content = lines0.slice(startLine - 1, endLine).join('\n');
        }
        var tok = estTok(content);
        var attachBudget = attachTokenBudget();
        if (totalTokens + tok > attachBudget) {
          var remaining = attachBudget - totalTokens;
          if (remaining > 0) {
            var lines = content.split('\n');
            var kept = 0, keptTok = 0;
            for (var li = 0; li < lines.length; li++) {
              var lt = estTok(lines[li]);
              if (keptTok + lt > remaining) break;
              kept++; keptTok += lt;
            }
            content = lines.slice(0, kept).join('\n') +
              '\n…[trimmed ' + (lines.length - kept) + ' lines, ' + (tok - keptTok) + ' tk]';
            totalTokens += keptTok;
          } else {
            content = '…[budget exceeded]';
          }
        } else {
          totalTokens += tok;
        }
        var rangeLabel = startLine !== undefined ? (':' + startLine + '-' + endLine) : '';
        blocks.push('### ' + pathPart + rangeLabel + '\n```\n' + content + '\n```');
        manifestParts.push('@' + pathPart + rangeLabel + ' (' + estTok(content) + ' tk)');
      } catch (e) {
        blocks.push('### ' + pathPart + '\n```\n(error: ' + String(e.message || e) + ')\n```');
        manifestParts.push('@' + pathPart + ' (error)');
      }
    }
    var attached = blocks.length > 0 ? '\n\nAttached files:\n' + blocks.join('\n\n') : '';
    return { text: text + attached, manifest: manifestParts.join(', ') };
  }

  /* ── Session mesh (format-compatible with session.rs) ─────────────── */
  function utf8Bytes(s) {
    var out = [];
    for (var i = 0; i < s.length; i++) {
      var cp = s.codePointAt(i);
      if (cp > 0xffff) i++; /* surrogate pair consumed */
      if (cp < 0x80) out.push(cp);
      else if (cp < 0x800) out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
      else if (cp < 0x10000) out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
      else out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
    }
    return out;
  }
  /* fnv1a64 — must match session.rs::session_password byte for byte. */
  function fnv1a64(str) {
    var bytes = utf8Bytes(str);
    var h = 0xcbf29ce484222325n;
    for (var i = 0; i < bytes.length; i++) {
      h ^= BigInt(bytes[i]);
      h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
    }
    return h;
  }
  /* Legacy encrypted-session password; new QTSQ cache writes ignore it. */
  function sessionPassword(project) {
    return 'sofuu-session-' + fnv1a64(project).toString(16).padStart(16, '0');
  }
  /* s-<13hex millis>-<5hex pid>-<4hex rnd> (session.rs::gen_session_id). */
  var idCounter = 0;
  function genSessionId() {
    var millis = Date.now();
    var p = pid();
    idCounter++;
    var state = (BigInt(millis) ^
      (BigInt(p) * 0x9E3779B97F4A7C15n) ^
      (BigInt(idCounter) * 0xBF58476D1CE4E5B9n)) & 0xffffffffffffffffn;
    state = (state * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    var rnd = Number((state >> 32n) & 0xffffn);
    return 's-' + millis.toString(16).padStart(13, '0') +
           '-' + (p & 0xfffff).toString(16).padStart(5, '0') +
           '-' + rnd.toString(16).padStart(4, '0');
  }
  /* ── Project store ─────────────────────────────────────────────────
   * EVERYTHING lives under <project>/.sofuu/, qtsq files only:
   *   brain/brain.qtsq          PROJECT-LOCAL brain (never a global one)
   *   sessions/registry.qtsq    session registry (no plaintext registry)
   *   sessions/<sid>/session.qtsq   metadata + event INDEX (no texts)
   *   sessions/<sid>/conv/<seq>-<t>-<kind>.qtsq   ONE event per file,
 *                              written the moment it happens — the full
   *                             timestamped conversation accrues in
   *                             real time, never a bulk rewrite
   *   debug/ issues/ audits/    timestamped records, each carrying the
   *                             session id it happened in
   * Legacy layouts stay READABLE: flat <sid>.qtsq files and the old
   * plaintext registry.json (sessions written before this change, and
   * TUI-written ones until session.rs ports to the same layout). */

  function padN(n, w) {
    var str = String(n);
    while (str.length < w) str = '0' + str;
    return str;
  }
  /* The QTSQ codec does not mkdir — a touched file in the target dir is
   * the mkdir (its writer creates parent directories). */
  function ensureDir(dir) {
    try {
      if (typeof __session_write_file === 'function') __session_write_file(dir + '/.keep', '');
    } catch (e) {}
  }
  function sessionsDir(project) { return project + '/.sofuu/sessions'; }
  function sessionDir(project, id) { return sessionsDir(project) + '/' + id; }
  function sessionIndexPath(project, id) { return sessionDir(project, id) + '/session.qtsq'; }
  function convDir(project, id) { return sessionDir(project, id) + '/conv'; }
  function registryPath(project) { return sessionsDir(project) + '/registry.qtsq'; }
  function legacyRegistryPath(project) { return sessionsDir(project) + '/registry.json'; }
  function legacySessionFile(project, id) { return sessionsDir(project) + '/' + id + '.qtsq'; }

  function registryRead(project) {
    var reg = null;
    /* qtsq registry — the source of truth. */
    try {
      if (typeof __qtsq_session_load === 'function') {
        var raw = __qtsq_session_load(registryPath(project), sessionPassword(project));
        if (raw) {
          var j = JSON.parse(raw);
          if (j && Array.isArray(j.sessions)) reg = j;
        }
      }
    } catch (e) {}
    /* Legacy plaintext registry merged in (read-only): sessions written
     * before this layout, and TUI sessions. An entry is included ONLY
     * when its transcript file still exists — deleted sessions must not
     * resurrect from the old file (the folder/flat file is the truth). */
    try {
      var lraw = (typeof __session_read_file === 'function')
        ? __session_read_file(legacyRegistryPath(project)) : null;
      if (lraw) {
        var lj = JSON.parse(lraw);
        if (lj && Array.isArray(lj.sessions)) {
          if (!reg) reg = { sessions: [] };
          var have = {};
          for (var h = 0; h < reg.sessions.length; h++) {
            if (reg.sessions[h]) have[reg.sessions[h].id] = true;
          }
          for (var k = 0; k < lj.sessions.length; k++) {
            var si = lj.sessions[k];
            if (si && si.id && !have[si.id]) {
              try {
                if (!__session_read_file(legacySessionFile(project, si.id))) continue;
              } catch (eF) { continue; }
              reg.sessions.push(si);
            }
          }
        }
      }
    } catch (e) {}
    return reg || { sessions: [] };
  }
  function registryUpsert(project, info) {
    if (!project || !info) return;
    var reg = registryRead(project);
    var found = false;
    for (var i = 0; i < reg.sessions.length; i++) {
      if (reg.sessions[i] && reg.sessions[i].id === info.id) {
        reg.sessions[i] = info; found = true; break;
      }
    }
    if (!found) reg.sessions.push(info);
    try {
      ensureDir(sessionsDir(project));
      if (typeof __qtsq_session_save === 'function') {
        __qtsq_session_save(registryPath(project), JSON.stringify(reg), sessionPassword(project));
      }
    } catch (e) {}
  }
  /* The index: metadata + one line per event, never the texts. Small
   * enough to rewrite on every event. */
  function sessionPersist() {
    if (!S.session || !S.project) return;
    var d = S.session;
    if (d.events.length > 5000) d.events.splice(0, d.events.length - 5000);
    if (d.notes.length > 20) d.notes.length = 20;
    try {
      ensureDir(sessionDir(S.project, d.id));
      if (typeof __qtsq_session_save === 'function') {
        __qtsq_session_save(sessionIndexPath(S.project, d.id), JSON.stringify(d), sessionPassword(S.project));
      }
    } catch (e) {}
  }
  /* Real-time conversation: the event's text lands in its own tiny qtsq
   * file THIS moment — a crash mid-turn loses nothing already said. */
  function sessionLog(kind, text) {
    if (!S.session) return;
    var t = now();
    S.eventSeq = (S.eventSeq | 0) + 1;
    var ev = { seq: S.eventSeq, t: t, kind: kind };
    S.session.events.push(ev);
    if (S.info) S.info.last_seen = t;
    try {
      if (typeof __qtsq_session_save === 'function') {
        ensureDir(convDir(S.project, S.info.id));
        var f = convDir(S.project, S.info.id) + '/' + padN(ev.seq, 6) + '-' + t + '-' + kind + '.qtsq';
        __qtsq_session_save(f, JSON.stringify({
          seq: ev.seq, t: t, kind: kind, session: S.info.id,
          text: String(text == null ? '' : text),
        }), sessionPassword(S.project));
      }
    } catch (e) {}
    sessionPersist();
  }
  /* Per-workspace chat cap (user rule): the strip holds at most 7 — a new
   * chat requires deleting one first. Counted from the registry (what the
   * strip shows), not the filesystem. */
  var SESSIONS_MAX = 7;
  function sessionJoin(project, model, provider) {
    var regC = registryRead(project);
    if (regC.sessions.length >= SESSIONS_MAX) {
      throw new Error('chat limit reached (' + SESSIONS_MAX + ' per workspace) — delete a chat (the x on a pill) to add a new one');
    }
    var id = genSessionId();
    var cwd = '';
    try { cwd = process.cwd(); } catch (e) {}
    S.info = {
      id: id, pid: pid(), host: 'desktop', cwd: cwd,
      model: model || '', provider: provider || '',
      started_at: now(), last_seen: now(), task: null, ended: false,
    };
    S.session = {
      schema: 2, id: id, created: now(), host: 'desktop', cwd: cwd,
      model: model || '', provider: provider || '',
      task: null, notes: [], events: [],
    };
    S.eventSeq = 0;
    /* The project store skeleton — generated for every session, so the
     * structure exists even before the first record lands. */
    ensureDir(sessionDir(project, id));
    ensureDir(convDir(project, id));
    ensureDir(project + '/.sofuu/brain');
    ensureDir(project + '/.sofuu/debug');
    ensureDir(project + '/.sofuu/issues');
    ensureDir(project + '/.sofuu/audits');
    brainIndexTouch(project);
    registryUpsert(project, S.info);
    sessionLog('start', 'session started');
  }
  /* The brain folder carries its own manifest (unique id + description) —
   * the same index pattern as debug/issues/audits, so every store under
   * .sofuu/ is self-describing and pointable-to by the AI. The brain's
   * RECORDS already carry ids + text (the CMA store); this manifest
   * describes the FOLDER. */
  function brainIndexTouch(project) {
    try {
      if (typeof __qtsq_session_save !== 'function') return;
      ensureDir(project + '/.sofuu/brain');
      var id = 'brain-' + fnv1a64(project).toString(16).slice(0, 12);
      __qtsq_session_save(project + '/.sofuu/brain/index.qtsq', JSON.stringify({
        id: id,
        desc: 'Project-local memory store (brain) — per-memory ids + text live inside brain.qtsq; recall reads them',
        files: ['brain.qtsq'],
        updated: now(),
      }), sessionPassword(project));
    } catch (e) {}
  }
  /* ── debug / issues / audits records ───────────────────────────────
   * One qtsq file per record under <project>/.sofuu/<folder>/, each with
   *   • a UNIQUE record id (its own namespace, separate from the session
   *     id: dbg-/iss-/aud- + base36 time + counter) in the filename,
   *   • a short human description (`desc`) — the searchable line,
   *   • the session id it happened in (the link).
   * The folder's index.qtsq lists {id, session, t, kind, desc} — that
   * index is what the tiny models / the AI read FIRST (a few hundred
   * bytes) and only then fetch the specific record file. Failures here
   * are silent: a record must never break a turn. */
  function writeRecord(kind, desc, payload) {
    if (!S.project) return;
    var prefix = kind === 'issue' ? 'iss' : kind === 'audit' ? 'aud' : 'dbg';
    var folder = kind === 'issue' ? 'issues' : kind === 'audit' ? 'audits' : 'debug';
    try {
      if (typeof __qtsq_session_save !== 'function') return;
      var dir = S.project + '/.sofuu/' + folder;
      ensureDir(dir);
      var t = now();
      S.recordSeq[folder] = (S.recordSeq[folder] | 0) + 1;
      var id = prefix + '-' + t.toString(36) + '-' + padN(S.recordSeq[folder], 4);
      var f = dir + '/' + id + '.qtsq';
      var rec = {
        id: id, desc: String(desc || ''), t: t,
        session: S.info ? S.info.id : null, project: S.project,
      };
      for (var k in (payload || {})) rec[k] = payload[k];
      __qtsq_session_save(f, JSON.stringify(rec), sessionPassword(S.project));
      /* Folder index: the tiny-model entry point. */
      var idx = [];
      try {
        var iraw = __qtsq_session_load(dir + '/index.qtsq', sessionPassword(S.project));
        if (iraw) { var ij = JSON.parse(iraw); if (Array.isArray(ij.entries)) idx = ij.entries; }
      } catch (eI) {}
      idx.push({ id: id, session: rec.session, t: t,
                 kind: String(payload && payload.type || kind), desc: rec.desc });
      if (idx.length > 500) idx.splice(0, idx.length - 500);
      __qtsq_session_save(dir + '/index.qtsq', JSON.stringify({ id: folder + '-index', desc: folder + ' records, newest last', entries: idx }), sessionPassword(S.project));
    } catch (e) {}
  }

  function sessionEnd() {
    if (!S.session || !S.project) return;
    try {
      sessionLog('end', 'session ended');
      S.info.ended = true;
      registryUpsert(S.project, S.info);
    } catch (e) {}
    S.session = null;
    S.info = null;
  }
  function loadSessionData(project, id, tailTurns) {
    if (typeof __qtsq_session_load !== 'function') return null;
    var pw = sessionPassword(project);
    /* Current layout: session.qtsq is the INDEX (no texts) — the texts
     * live in conv/<seq>-<t>-<kind>.qtsq, one file per event.
     * P2-6 (AUDIT-2026-09-01): a months-old session is thousands of event
     * files; loading EVERY text then capping to the last 10 turns wasted
     * thousands of synchronous loads per resume. When the caller tells us
     * how many turns it will keep (tailTurns), fetch texts only for the
     * trailing events that can contribute: ~2 events/turn plus the
     * tools/served companions of the tail (they ride the same pair loop
     * in turnsFromData). */
    try {
      var raw = __qtsq_session_load(sessionIndexPath(project, id), pw);
      if (raw) {
        var d = JSON.parse(raw);
        if (d && d.schema === 2 && Array.isArray(d.events)) {
          var evs = d.events;
          var fetchFrom = 0;
          if (tailTurns > 0 && evs.length > tailTurns * 4) {
            /* Walk back tailTurns prompt events from the end: those open
             * the turns the caller keeps; everything before is droppable. */
            var seenPrompts = 0;
            for (var b = evs.length - 1; b >= 0; b--) {
              if (evs[b] && evs[b].kind === 'prompt') {
                seenPrompts++;
                if (seenPrompts >= tailTurns) { fetchFrom = b; break; }
              }
            }
            if (seenPrompts < tailTurns) fetchFrom = 0;
          }
          for (var i = fetchFrom; i < evs.length; i++) {
            var ev = evs[i];
            ev.text = '';
            if (i >= fetchFrom && (ev.file || (ev.seq && ev.kind))) {
              try {
                var f = ev.file || (convDir(project, id) + '/' +
                  padN(ev.seq, 6) + '-' + ev.t + '-' + ev.kind + '.qtsq');
                var eraw = __qtsq_session_load(f, pw);
                if (eraw) {
                  var eobj = JSON.parse(eraw);
                  ev.text = String((eobj && eobj.text) || '');
                  ev.file = f;
                }
              } catch (eE) {}
            }
          }
          return d;
        }
      }
    } catch (e) {}
    /* Legacy: one flat <sid>.qtsq with texts inline. */
    try {
      var lraw = __qtsq_session_load(legacySessionFile(project, id), pw);
      if (!lraw) return null;
      return JSON.parse(lraw);
    } catch (e) { return null; }
  }
  /* Pair prompt/answer events into turns, oldest first (session.rs
   * past_turns), capped at `cap` turns. */
  function turnsFromData(data, cap) {
    var out = [];
    if (!data || !Array.isArray(data.events)) return out;
    for (var i = 0; i < data.events.length; i++) {
      var ev = data.events[i];
      if (!ev || typeof ev.kind !== 'string') continue;
      if (ev.kind === 'prompt') out.push({ prompt: String(ev.text || ''), answer: '' });
      else if (ev.kind === 'answer' && out.length) out[out.length - 1].answer = String(ev.text || '');
      else if (ev.kind === 'tools' && out.length) {
        try { out[out.length - 1].tools = JSON.parse(ev.text || '[]'); } catch (eT) {}
      } else if (ev.kind === 'served' && out.length) {
        out[out.length - 1].servedBy = String(ev.text || '');
      }
    }
    if (cap > 0 && out.length > cap) out = out.slice(-cap);
    return out;
  }

  /* ── Step-event normalization for host UIs ──────────────────────────
   * agent.js emits answer_delta with a bare string payload (the TUI
   * renders it raw); host UIs get a uniform {text} payload plus small
   * display aids for the timeline (summary/text fields). The run-terminal
   * 'answer'/'stop' events are swallowed — chat.js emits its own 'done'
   * with full usage once the run returns. */
  function normalizeStep(e) {
    var p = (e.payload === undefined || e.payload === null) ? {} : e.payload;
    var out = { kind: e.kind, payload: p, runId: e.runId, name: e.name, depth: e.depth, t: e.t };
    if (e.kind === 'answer_delta') {
      out.payload = { text: typeof p === 'string' ? p : String(p.text || '') };
      return out;
    }
    if (e.kind === 'tool' && typeof p === 'object' && typeof p.args === 'string' && p.summary === undefined) {
      var np = {};
      for (var k in p) np[k] = p[k];
      np.summary = p.args;
      out.payload = np;
      return out;
    }
    if (e.kind === 'tool_result' && typeof p === 'object' && p.summary === undefined && p.text === undefined) {
      var np2 = {};
      for (var k2 in p) np2[k2] = p[k2];
      if (has(p, 'result')) np2.text = String(p.result);
      out.payload = np2;
      return out;
    }
    if (e.kind === 'recall' && typeof p === 'object' && p.summary === undefined) {
      var np3 = {};
      for (var k3 in p) np3[k3] = p[k3];
      var n3 = p.count || 0;
      np3.summary = n3 + ' memor' + (n3 === 1 ? 'y' : 'ies') + ' recalled';
      out.payload = np3;
      return out;
    }
    if (e.kind === 'plan' && typeof p === 'object' && p.text === undefined) {
      var np4 = {};
      for (var k4 in p) np4[k4] = p[k4];
      np4.text = 'step ' + (p.step || 0) + ((p.tools && p.tools.length) ? ': ' + p.tools.join(', ') : '');
      out.payload = np4;
      return out;
    }
    return out;
  }

  /* ── The turn (ported from the DRIVER's turn()) ──────────────────── */
  /* providerOverride (failover): a {name, endpoint, api_key, profile,
   * model} entry from the config's providers list. When present, the turn
   * runs against THAT provider instead of the active one — the caps
   * ladder resolves its model/endpoint pair exactly as it would the
   * active one. Null = the active provider (the normal path). */
  async function turn(text, forceNoThink, noTools, providerOverride, images) {
    S.providerOverride = providerOverride || null;
    var c = activeCfg();
    if (!usableConfig()) {
      throw new Error('No usable model configured — open Settings → Providers to set one up');
    }
    /* Sessionless submit (empty workspace): mint the chat now — the turn
     * itself is the user's explicit act. Folder switches never get here. */
    if (!S.session || !S.info) {
      var cMint = cfg();
      sessionJoin(S.project, cMint.model || '', cMint.provider || '');
    }
    /* Budget preflight (F6): block once session spend reaches the cap. */
    if (c.budget_usd > 0 && (c.spend_total_usd || 0) + S.sessionSpendUsd >= c.budget_usd) {
      throw new Error('Budget cap reached ($' +
        ((c.spend_total_usd || 0) + S.sessionSpendUsd).toFixed(4) +
        ' / $' + Number(c.budget_usd).toFixed(2) + ')');
    }
    if (!sofuu.agent || typeof sofuu.agent.run !== 'function') {
      throw new Error('agent runtime unavailable (src/js/agent.js failed to load)');
    }
    /* MCP is lazy: connect on demand only when a tool-using turn starts
     * (pure chat turns never invoke tools, so skip the handshake). */
    if (!noTools && S.mcpTools.length === 0 && mcpServersList().length > 0) {
      await connectMcpServers();
    }
    /* Sub-agents from ~/.sofuu/agents/*.js (loaded once per session). */
    if (!S.agentsDirLoaded) {
      S.agentsDirLoaded = true;
      try { if (typeof sofuu.agent.loadDir === 'function') await sofuu.agent.loadDir(); } catch (e) {}
    }
    var subAgentNames = [];
    try {
      if (typeof sofuu.agent.list === 'function') {
        subAgentNames = sofuu.agent.list().map(function (a) { return a.name; });
      }
    } catch (e) {}
    /* @agent mention: "@name task" or "@agent:name task" runs the named
     * agent directly. Chat history IS injected (runOpts.history below;
     * agent.js concatenates it into the request) — the named agent sees
     * the conversation so far. */
    var typed = text;
    var agentMention = null;
    var tt = text.trim();
    var mm = tt.match(/^@agent:([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/);
    if (!mm) mm = tt.match(/^@([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/);
    if (mm) {
      var aname = null;
      if (subAgentNames.indexOf(mm[1]) >= 0) aname = mm[1];
      else {
        var low = mm[1].toLowerCase();
        var ci = subAgentNames.filter(function (n) { return n.toLowerCase() === low; });
        if (ci.length === 1) aname = ci[0];
      }
      if (aname) agentMention = { name: aname, task: mm[2].trim() };
    }
    if (agentMention) text = agentMention.task;
    /* @file mentions on the remainder (the task itself may attach files). */
    var expanded = await expandMentions(text);
    var turnText = expanded.text;
    var manifest = expanded.manifest;
    /* session-2 (AUDIT-2026-09-07): one prompt event per submit — turn()
     * re-runs on THINK/CAPACITY retries, and duplicate prompt events
     * replay as extra user turns on resume. Reset per submit (S field:
     * turn() is module-scope, submit-scoped vars can't be captured). */
    if (!S.promptLoggedThisSubmit) {
      S.promptLoggedThisSubmit = true;
      sessionLog('prompt', String(typed));
    }
    /* Strict effort: only what the user selected; models that rejected
     * thinking (learned or persisted) never get an effort parameter. */
    var noThink = !!forceNoThink ||
      (c.no_think_models || []).indexOf(c.model) >= 0 ||
      S.noThink.indexOf(c.model) >= 0;
    var def = {
      name: 'chat',
      system: sofuu.agent.CORE_PROMPT,
      /* "Just chat" mode: an empty toolset makes the run a pure
       * conversation turn (no tool loop, no approvals). */
      tools: noTools ? [] : chatToolDefs(),
      agents: subAgentNames.length ? subAgentNames : undefined,
      provider: c.provider, model: c.model,
      effort: noThink ? undefined : (c.effort && c.effort !== 'off' ? c.effort : undefined),
      api_key: c.api_key || undefined,
      base_url: c.base_url || undefined,
      profile: c.profile || undefined,
      max_tokens: (function () {
        var r = resolveCaps();
        return (r.maxOutput > 0 && r.maxSource !== 'default') ? r.maxOutput : undefined;
      })(),
      embed_provider: c.embed_provider || undefined,
      embed_model: c.embed_model || undefined,
      memory: c.brain ? 'shared' : 'off',
      /* Project-local brain (P: no global) — every project carries its
       * own memories under .sofuu/brain/. */
      brainPath: S.project ? S.project + '/.sofuu/brain/brain.qtsq' : undefined,
      ml: c.ml ? 'on' : 'off',
      rlm: c.rlm === 'on' ? 'on' : (c.rlm === 'auto' ? 'auto' : 'off'),
      ctx_window: (function () { var r = resolveCaps(); return r.window > 0 ? r.window : undefined; })(),
      recallMin: c.recall_min > 0 ? c.recall_min : undefined,
      recallBudget: c.recall_budget > 0 ? c.recall_budget : undefined,
      /* Human-scale tool waits when the gate is on (approvals block inside
       * execute; agent.js's default 30s wrapper would kill them). */
      toolTimeoutMs: S.toolTimeoutMs,
      /* P1-3 (AUDIT-2026-09-01): this driver — the one every embedded/
       * headless host runs — still capped tool loops at 8 rounds while the
       * TUI driver allows 200 (chat.rs). A read→read→grep→edit→verify coding
       * turn died at 8. Port the TUI's env-tunable default; agent-mention
       * turns override via their own def budgets. */
      budget: { maxSteps: (parseInt(env('SOFUU_CHAT_MAX_STEPS'), 10) || 200),
                maxDepth: 1, maxTokens: 1e9, maxWallMs: 1e9 },
    };
    /* Live context meter (the desktop ring): the FIRST LLM request's
     * prompt is the true size of everything the model saw this turn —
     * system prompt + history + this task + tool schemas. Later rounds
     * re-count tool results and would balloon the number, so only the
     * first counts (same discipline as the TUI's footer meter). */
    var firstPromptTk = 0;
    /* Ring hover card: the first request's section breakdown + cache read,
     * captured off the same llm event that calibrates the total. */
    var firstSections = null;
    var firstCacheTk = 0;
    var toolLog = [];
    var onStep = function (e) {
      if (!e || typeof e.kind !== 'string') return;
      if (!firstPromptTk && e.kind === 'llm' && e.payload && e.payload.usage &&
          (e.payload.usage.promptTokens | 0) > 0) {
        firstPromptTk = (e.payload.usage.promptTokens | 0);
        if (e.payload.sections && typeof e.payload.sections === 'object') {
          firstSections = e.payload.sections;
        }
        if ((e.payload.usage.cacheReadTokens | 0) > 0) {
          firstCacheTk = (e.payload.usage.cacheReadTokens | 0);
        }
      }
      /* Structured resume (P1): collect the turn's tool activity so it can
       * be persisted with the session and replayed on resume — the old
       * transcript dropped everything except prompt/answer, so a resumed
       * session forgot which files were read and what commands ran. */
      if (e.kind === 'tool' && e.payload && e.payload.name) {
        toolLog.push({ n: String(e.payload.name), a: clip(String(e.payload.args || ''), 160) });
      } else if (e.kind === 'tool_result' && e.payload && e.payload.name) {
        /* System notes (stream continuation, empty-retry) are diagnostics —
         * they become debug records linked to this session. */
        if (e.payload.name === 'system') {
          writeRecord('debug', 'system note: ' + clip(String(e.payload.result || ''), 80),
                      { type: 'system_note', note: clip(String(e.payload.result || e.payload.text || ''), 300) });
        }
        for (var tl = toolLog.length - 1; tl >= 0; tl--) {
          if (toolLog[tl].n === e.payload.name && toolLog[tl].r === undefined) {
            toolLog[tl].r = clip(String(e.payload.result || e.payload.text || ''), 240);
            break;
          }
        }
      } else if (e.kind === 'allocgate' && e.payload && e.payload.action === 'learned_limit') {
        writeRecord('debug', 'learned ' + String(e.payload.kind || 'limit') + ' for ' + String(e.payload.model || 'model'),
                    { type: 'learned_limit', kind: String(e.payload.kind || ''),
                      model: String(e.payload.model || ''),
                      error: clip(String(e.payload.error || ''), 200) });
      }
      /* Run-terminal duplicates — chat.js emits its own 'done'. */
      if (e.kind === 'answer' || e.kind === 'stop') return;
      emit(normalizeStep(e));
    };
    /* Multimodal (P2): image attachments ride into the agent turn. */
    var runOpts = {
      history: S.history, signal: 'chat', onStep: onStep,
      shared: '', watched: '',
    };
    if (Array.isArray(images) && images.length) {
      runOpts.images = images;
    }
    var res = agentMention
      ? await sofuu.agent.run(agentMention.name, turnText, runOpts)
      : await sofuu.agent.run(def, turnText, runOpts);
    var answer = String((res && res.answer) || '(no response)');
    /* Persist the turn's tool activity + which model served it (structured
     * resume): the transcript keeps the prompt/answer lines for the TUI
     * and adds compact machine-readable rows the resume path replays. */
    if (toolLog.length) {
      var compactTools = toolLog.slice(-40).map(function (tl) {
        return { n: clip(tl.n, 60), a: tl.a || '', r: tl.r || '' };
      });
      sessionLog('tools', JSON.stringify(compactTools));
    }
    sessionLog('served', c.model || '');
    /* Usage + cost (pricing table from config; $ per 1M in/out tokens). */
    var u = (res && res.usage) || {};
    var ctxTk = u.promptTokens || 0, outTk = u.completionTokens || 0;
    S.totalTk += ctxTk + outTk;
    var costUsd = 0;
    if (ctxTk > 0 || outTk > 0) {
      var table = c.pricing || {};
      var price = table[(c.provider || '') + '/' + (c.model || '')];
      if (Array.isArray(price) && price.length >= 2) {
        costUsd = (ctxTk * price[0] + outTk * price[1]) / 1e6;
        S.sessionSpendUsd += costUsd;
      }
    }
    /* Per-turn usage record: one conv event carrying this turn's real
     * token counts + cost. sofuu.chat.usage() aggregates these across
     * every session in the workspace for the desktop Usage tab (and any
     * future /usage surfaces). */
    if (S.session && (ctxTk > 0 || outTk > 0)) {
      sessionLog('usage', JSON.stringify({
        model: c.model || '', provider: c.provider || '',
        pt: ctxTk | 0, ct: outTk | 0,
        cr: u.cacheReadTokens | 0, cw: u.cacheWriteTokens | 0,
        llm: u.llmCalls | 0, tools: u.toolCalls | 0,
        cost: costUsd,
      }));
    }
    /* Persist the turn to this session's .qtsq + heartbeat the registry. */
    sessionLog('answer', answer);
    registryUpsert(S.project, S.info);
    /* History stores the manifest only (not full file text). */
    var histText = manifest ? text + ' [' + manifest + ']' : text;
    /* Meter calibration inputs, captured BEFORE this turn lands in history:
     * overhead = firstPromptTk − histAtReq − turnTk (system + tool schemas +
     * ephemeral), so the post-turn ctx event = overhead + historyTokens()
     * — with retained transcripts (2026-09-03 e) the next request INCLUDES
     * the turn's transcript, and firstPromptTk alone would under-count it. */
    var histAtReq = historyTokens();
    var turnTk = estTok(histText);
    /* Claude-style retention (2026-09-03 e): the turn's tool transcript
     * (assistant tool_calls + tool results) persists into history between
     * the prompt and the answer — the model keeps its tool context across
     * turns and compaction sheds it, not a per-turn release. */
    S.history.push({ role: 'user', content: histText });
    var turnTranscript = (res && Array.isArray(res.transcript)) ? res.transcript : null;
    if (turnTranscript && turnTranscript.length) {
      for (var ti2 = 0; ti2 < turnTranscript.length; ti2++) {
        S.history.push(turnTranscript[ti2]);
      }
    }
    S.history.push({ role: 'assistant', content: answer });
    await trimHistory();
    /* Live ctx meter event: what the NEXT request will look like. With
     * retained transcripts (2026-09-03 e) that is calibrated-overhead +
     * the full history INCLUDING this turn's transcript — firstPromptTk
     * alone no longer represents the next request. Falls back to the
     * history estimate when the provider reported no usage. window = the
     * same evidence ladder the caps resolve uses (endpoint-discovered >
     * registry > config), so the ring never lies about the real window
     * the way a stale flat config number would. */
    var usedTk;
    if (firstPromptTk > 0) {
      var calOverhead = Math.max(0, firstPromptTk - histAtReq - turnTk);
      usedTk = historyTokens() + calOverhead;
    } else {
      usedTk = historyTokens();
    }
    var winTk = ctxWindow();
    var rSrc = resolveCaps();
    emit({ kind: 'ctx', payload: {
      used: usedTk,
      window: winTk,
      source: rSrc.source || 'default',
      clampedConfig: !!rSrc.clampedConfig,
      model: c.model || '',
      sections: firstSections || undefined,
      cacheHit: (firstCacheTk > 0 && firstPromptTk > 0)
        ? Math.min(1, firstCacheTk / firstPromptTk) : undefined,
      /* Session attribution: the desktop keys the ring per session, so a
       * late event (turn finishing after the view switched) can never
       * land on another session's meter. */
      sid: S.info ? S.info.id : null,
    } });
    /* Keep the brain manifest's `updated` fresh after a turn that may
     * have written memories (consolidation/remember). */
    brainIndexTouch(S.project);
    /* One full GC per completed turn (bounds JS garbage per turn). */
    try { if (typeof __sofuu_gc === 'function') __sofuu_gc(); } catch (e) {}
    S.providerOverride = null;
    return {
      answer: answer,
      usage: u,
      steps: (res && res.steps) || 0,
      stopped: (res && res.stopped) || null,
      costUsd: costUsd,
      servedBy: c.model || '',
    };
  }

  /* ── Public API ──────────────────────────────────────────────────── */
  function setProjectInternal(project, attach) {
    sessionEnd(); /* close the previous session cleanly, if any */
    /* Tool jail follows the project, then derive the mesh root EXACTLY as
     * session.rs::project_root does ($SOFUU_PROJECT → git toplevel → cwd,
     * literal paths — deliberately NOT realpath'd): the fnv1a64 session
     * password hashes that string, so any divergence (e.g. /tmp vs
     * /private/tmp symlinks) would make the TUI and desktop unable to read
     * each other's .qtsq files. One-sided JS-side canonicalization is
     * exactly the divergence this must avoid. */
    var chdirOk = false;
    try { process.chdir(project); chdirOk = true; } catch (e) {}
    var root = null;
    if (chdirOk) {
      try {
        if (typeof __session_project_root === 'function') root = __session_project_root();
      } catch (e) {}
    }
    if (!root || typeof root !== 'string') {
      /* chdir failed (dir missing) or no native — best effort: the literal
       * path (the mesh dir gets created under it on first write). */
      root = project;
    }
    S.project = root;
    if (!attach) {
      var c0 = cfg();
      sessionJoin(S.project, c0.model || '', c0.provider || '');
      return;
    }
    /* Project SWITCH (or boot): NEVER mint a chat — a new chat exists only
     * when the user asks for one (+ button) or actually sends a turn.
     * Attach the workspace's newest existing session instead; an empty
     * workspace stays sessionless (empty strip, no pill). */
    var reg = registryRead(S.project);
    /* Newest-first; if a container fails to load (corrupt), fall back to
     * the next one instead of going sessionless — the strip still has
     * pills, so the host must have a session to write into. */
    var bySeen = [];
    for (var bi = 0; bi < reg.sessions.length; bi++) {
      var bs = reg.sessions[bi];
      if (bs && bs.id) bySeen.push(bs);
    }
    bySeen.sort(function (a, b) { return (b.last_seen || 0) - (a.last_seen || 0); });
    for (var bj = 0; bj < bySeen.length; bj++) {
      if (resume(bySeen[bj].id) && S.info) return;
    }
    /* nothing to attach — stay sessionless */
    S.session = null;
    S.info = null;
    S.history = [];
  }

  /* ── Caps discovery (provider-agnostic) ──────────────────────────
   * The endpoint serving the request knows what its models accept — many
   * publish the numbers in their model listing. On boot (and whenever the
   * stored truth ages out), fetch the ACTIVE endpoint's listing and feed
   * it to the alloc gate's discovered store (keyed by API root + model,
   * so any provider works; field spellings are normalized in Rust).
   * Silent and best-effort by design: no listing, a fetch failure or a
   * 404 never blocks chat — the registry + error learning remain. Local
   * endpoints are skipped (the SSRF guard blocks them anyway; localhost
   * needs no discovery). */
  var DISCOVER_TTL_MS = 7 * 24 * 3600 * 1000;
  /* P3 (AUDIT-2026-09-07): failed discovery harvests retry on this short
   * clock instead of being cached for the full TTL. */
  var DISCOVER_RETRY_MS = 10 * 60 * 1000;
  var capsDiscoveredAt = 0;
  function isLocalHost(u) {
    return /^https?:\/\/(localhost|127\.|0\.0\.0\.0|\[::1\]|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)/i.test(String(u || ''));
  }
  function modelsRootUrl(endpoint) {
    var u = String(endpoint || '');
    /* Strip any completion-path suffixes down to the API root, then
     * append /models — every OpenAI-wire endpoint lists there. */
    u = u.split('#')[0].split('?')[0];
    while (/\/(chat\/completions|completions|messages)\/?$/.test(u)) {
      u = u.replace(/\/(chat\/completions|completions|messages)\/?$/, '');
    }
    if (u.charAt(u.length - 1) === '/') u = u.slice(0, -1);
    return u + '/models';
  }
  var capsHarvest = null; /* in-flight harvest promise (first-turn race fix) */
  function discoverCaps() {
    try {
      var c = cfg();
      if (!c.model || !c.base_url) return;
      if (isLocalHost(c.base_url)) return;
      if (Date.now() - capsDiscoveredAt < DISCOVER_TTL_MS) return capsHarvest;
      /* P3 (AUDIT-2026-09-07): the full TTL used to be stamped BEFORE the
       * harvest, so a fully failed run (network down, 401s) was cached for
       * 7 days like a success. Short retry clock up front (still throttles
       * re-entry while in flight); the full TTL lands only when ≥1 listing
       * actually succeeded. */
      capsDiscoveredAt = Date.now() - DISCOVER_TTL_MS + DISCOVER_RETRY_MS;
      /* Harvest the ACTIVE endpoint (its own numbers for THIS model are
       * the strongest discovered evidence) AND every other configured
       * provider with a distinct endpoint — cross-root fallback needs
       * their listings to know the same model's real caps when the
       * active endpoint publishes none. Provider-agnostic: whatever is
       * configured, nothing hardcoded. */
      var targets = [c.base_url];
      if (Array.isArray(c.providers)) {
        for (var pi = 0; pi < c.providers.length; pi++) {
          var pe = c.providers[pi];
          if (pe && pe.endpoint && pe.endpoint !== c.base_url &&
              targets.indexOf(pe.endpoint) < 0) {
            targets.push(pe.endpoint);
          }
        }
      }
      var harvests = [];
      for (var ti = 0; ti < targets.length; ti++) {
        (function (endpoint, apiKey) {
          var url = modelsRootUrl(endpoint);
          var headers = {};
          if (apiKey) headers['Authorization'] = 'Bearer ' + apiKey;
          var p = fetch(url, { headers: headers }).then(function (res) {
            if (!res.ok) return null;
            return res.json();
          }).then(function (j) {
            if (!j) return false;
            /* sofuu.ml.alloc.ingestListing(root, rawJson) — returns how many
             * entries carried caps; harmless (and silent) when nothing does. */
            if (sofuu.ml && sofuu.ml.alloc && typeof sofuu.ml.alloc.ingestListing === 'function') {
              try { sofuu.ml.alloc.ingestListing(endpoint, JSON.stringify(j)); } catch (eI) {}
            }
            return true;
          }).catch(function () { /* best-effort — never block chat */ return false; });
          harvests.push(p);
        })(targets[ti], ti === 0 ? c.api_key : ((c.providers || []).filter(function (p) {
          return p && p.endpoint === targets[ti];
        })[0] || {}).api_key);
      }
      /* P3 (AUDIT-2026-09-07): a harvest promise now resolves to whether
       * its listing actually landed — any success extends the stamp to
       * the full TTL; a total failure keeps the short retry clock. */
      capsHarvest = Promise.all(harvests).then(function (oks) {
        for (var oi = 0; oi < oks.length; oi++) {
          if (oks[oi]) { capsDiscoveredAt = Date.now(); break; }
        }
      }).catch(function () {});
    } catch (e) { /* best-effort */ }
    return capsHarvest;
  }

  /* ── F2: detect-on-select ──────────────────────────────────────────
   * Force a harvest for the ACTIVE endpoint (ignoring the TTL — the user
   * just switched to this model) and report what it found. Never silent:
   * the endpoint published caps, or it did not, and both are worth one
   * line. Hosts call this right after a model/provider switch; the CLI
   * does the same from its own slash loop. Resolves to a report object
   * so a host UI can render it however it likes. */
  function detectCaps() {
    var c = cfg();
    var res = { ok: false, endpoint: c.base_url || '', model: c.model || '',
                window: 0, source: '', ingested: 0, listing: false, reason: '' };
    try {
      if (!c.model || !c.base_url) { res.reason = 'no model/endpoint configured'; return Promise.resolve(res); }
      if (isLocalHost(c.base_url)) {
        res.window = resolveCaps().window;
        res.source = 'local-default';
        res.reason = 'local endpoint — detected from its first response';
        return Promise.resolve(res);
      }
      var endpoint = c.base_url;
      var headers = {};
      if (c.api_key) headers['Authorization'] = 'Bearer ' + c.api_key;
      var p = fetch(modelsRootUrl(endpoint), { headers: headers }).then(function (r2) {
        return (r2 && r2.ok) ? r2.json() : null;
      }).then(function (j) {
        res.listing = !!j;
        if (j && sofuu.ml && sofuu.ml.alloc && typeof sofuu.ml.alloc.ingestListing === 'function') {
          try { res.ingested = sofuu.ml.alloc.ingestListing(endpoint, JSON.stringify(j)) || 0; } catch (eI) { res.ingested = 0; }
        }
        /* Stamp the TTL so the per-turn harvest does not re-fetch what we
         * just pulled (a total failure keeps the short retry clock). */
        capsDiscoveredAt = Date.now() - DISCOVER_TTL_MS +
          (res.ingested > 0 ? DISCOVER_TTL_MS : DISCOVER_RETRY_MS);
        var r3 = resolveCaps();
        res.window = r3.window; res.source = r3.source || ''; res.known = !!r3.known;
        if (!j) res.reason = 'could not read the model list (auth, network, or no /models)';
        else if (res.ingested > 0 && r3.known) res.reason = 'detected from the endpoint listing';
        else res.reason = 'endpoint publishes no limits for this model';
        return res;
      }).catch(function (eH) {
        res.reason = 'could not reach the model list';
        return res;
      });
      capsHarvest = p;
      return p;
    } catch (e) { res.reason = 'detection failed'; return Promise.resolve(res); }
  }

  function init(opts) {
    opts = opts || {};
    S.host = String(opts.host || 'embedded');
    S.permissions = opts.permissions === 'prompt' ? 'prompt' : 'auto';
    /* A profile passed at init wins; otherwise the host can set one via
     * setPermissions after boot (the desktop persists it). */
    S.permissionProfile =
      (opts.permissionProfile === 'full' || opts.permissionProfile === 'edit' ||
       opts.permissionProfile === 'plan')
        ? opts.permissionProfile
        : 'prompt';
    S.toolTimeoutMs = (opts.toolTimeoutMs | 0) > 0 ? (opts.toolTimeoutMs | 0) : 30000;
    S.pastTurns = (opts.pastTurns | 0) > 0 ? (opts.pastTurns | 0) : 10;
    loadConfig();
    discoverCaps();
    var project = String(opts.project || '');
    if (!project) {
      try { project = process.cwd(); } catch (e) { project = '.'; }
    }
    /* Boot: attach the newest session of this workspace, or none. */
    setProjectInternal(project, true);
    /* Host poke: cancel + approval decisions arrive here from any thread
     * while a turn's blocking eval is running (rt/host_poke.rs). */
    globalThis.__host_poke = function (json) {
      var msg = null;
      try { msg = JSON.parse(String(json)); } catch (e) { return; }
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'cancel') { cancel(); return; }
      if (msg.type === 'approval') {
        resolveApproval(String(msg.id || ''), !!msg.allow, !!msg.always);
      }
    };
    S.initialized = true;
    var c = cfg();
    /* Seed the ctx meter at boot: empty history, real window (the same
     * ladder the caps resolve uses) — the ring has a denominator before
     * the first turn ever runs. */
    emit({ kind: 'ctx', payload: {
      used: 0,
      window: ctxWindow(),
      source: 'boot',
      sid: S.info ? S.info.id : null,
    } });
    return {
      ok: true,
      sessionId: S.info ? S.info.id : null,
      project: S.project,
      model: c.model || '', provider: c.provider || '',
      permissions: S.permissions,
      permissionProfile: S.permissionProfile,
    };
  }

  async function submit(text, opts) {
    opts = opts || {};
    if (S.streaming) throw new Error('a turn is already running');
    if (!S.initialized) init({});
    loadConfig(); /* pick up settings changed in the host UI */
    S.promptLoggedThisSubmit = false; /* session-2: dedupe the prompt log across failover/think retries */
    /* FIRST-TURN RACE FIX: the multi-root caps harvest is async, and a
     * model unknown to the ladder lets the flat config max_output ride the
     * wire — which is exactly what empties some gateways (200 + zero
     * text, finish_reason stop). When the model has NO evidence yet and a
     * harvest is in flight, wait for it (bounded) so the first turn goes
     * out with the same clamped caps every later turn gets. */
    discoverCaps(); /* start/refresh the harvest (TTL-gated) */
    if (!(resolveCaps().known)) {
      var t0 = Date.now();
      while (capsHarvest && Date.now() - t0 < 4000) {
        try { await Promise.race([capsHarvest, new Promise(function (r4) { setTimeout(r4, 250); })]); } catch (eH) {}
        if (resolveCaps().known) break;
        if (!(capsHarvest && Date.now() - t0 < 4000)) break;
      }
    }
    S.onEvent = typeof opts.onEvent === 'function' ? opts.onEvent : null;
    /* The boot ctx event fired inside init() — potentially before this
     * submit attached the event sink (the desktop inits at engine boot
     * with no sink, then submits). Re-emit once per sink attach so every
     * host sees the window denominator before the first turn. */
    if (!S.ctxAnnounced) {
      S.ctxAnnounced = true;
      emit({ kind: 'ctx', payload: { used: 0, window: ctxWindow(), source: 'boot',
                                     sid: S.info ? S.info.id : null } });
    }
    S.streaming = true;
    emit({ kind: 'start', payload: { sessionId: S.info ? S.info.id : null } });
    var THINK_ERR_RE = /thinking|reasoning_effort|extended.?thinking|reasoning/i;
    /* Failover-worthy failures: capacity (429/5xx/rate-limit) AND auth
     * (401/403) — a bad key is specific to that provider entry; the next
     * configured provider has its own key. Validation errors (400) and
     * model-not-found stay fatal: most providers would reject the same
     * request shape too, and silent switching would hide real bugs. */
    var CAPACITY_ERR_RE = /HTTP (40[13]|429|5\d\d)|rate.?limit|overloaded|temporarily|quota|invalid token|unauthorized|forbidden|api key/i;
    var noTools = !!opts.noTools;
    try {
      var res;
      var txt = String(text == null ? '' : text);
      /* The failover chain: the active provider, then every OTHER
       * configured provider entry that names a model (its endpoint + key
       * + model). Provider-agnostic — whatever is configured, nothing
       * hardcoded. A turn that dies with a CAPACITY-class error (429 /
       * 5xx / rate-limit — exhausted retry, not auth or validation,
       * which every provider would also fail) falls through to the next
       * entry; the user sees which provider took over. */
      var chain = [null];
      var cList = cfg();
      if (Array.isArray(cList.providers)) {
        for (var ci = 0; ci < cList.providers.length; ci++) {
          var pe = cList.providers[ci];
          if (pe && pe.name && pe.model && pe.endpoint &&
              pe.name !== cList.provider) {
            chain.push({ name: pe.name, endpoint: pe.endpoint,
                         api_key: pe.api_key || '', profile: pe.profile || '',
                         model: pe.model });
          }
        }
      }
      var chainIdx = 0;
      /* Image attachments: data URLs only, ≤8 — validated once, passed to
       * every attempt so a failover keeps the attachments. */
      var turnImages = Array.isArray(opts.images)
        ? opts.images.filter(function (u) {
            return typeof u === 'string' && u.indexOf('data:image/') === 0;
          }).slice(0, 8)
        : undefined;
      for (;;) {
        try {
          res = await turn(txt, false, noTools, chain[chainIdx], turnImages);
          break;
        } catch (e) {
          S.providerOverride = null;
          var msg = String((e && e.message) || e);
          /* Thinking-support runtime detection (same as the TUI): the
           * provider rejected a reasoning parameter → remember it and retry
           * once WITHOUT effort — on the SAME provider. P2-4: persist via
           * __chat_note_no_think (the config field the TUI uses) instead
           * of the process-local list that lost the learning on restart,
           * and key it to the model that will ACTUALLY serve the retry
           * (the failover target), not the config model. */
          var cA = cfg();
          if (THINK_ERR_RE.test(msg) && cA.effort && cA.model &&
              S.noThink.indexOf(cA.model) < 0 &&
              (cA.no_think_models || []).indexOf(cA.model) < 0) {
            S.noThink.push(cA.model);
            try { if (typeof __chat_note_no_think === 'function') __chat_note_no_think(cA.model); } catch (eNt) {}
            emit({ kind: 'warn', payload: { message: cA.model + ' rejected thinking (' +
                   clip(msg, 160) + ') — retrying without effort' } });
            /* js-2 (AUDIT-2026-09-07): the retry runs INSIDE this catch,
             * so a throw here skipped the classification below entirely
             * and escaped the turn — no failover, no issue record, no
             * done event. Classify the retry's own failure exactly like
             * the first: capacity → failover, anything else → issue
             * records + rethrow. The retry's error is the live truth
             * (the 400 is stale once the retry also failed), so `e` and
             * `msg` adopt it before falling through. */
            try {
              res = await turn(txt, true, noTools, chain[chainIdx], turnImages);
              break;
            } catch (eNt2) {
              e = eNt2;
              msg = String((e && e.message) || e);
            }
          }
          /* Capacity-class failure → fail over to the next provider. */
          if (CAPACITY_ERR_RE.test(msg) && chainIdx + 1 < chain.length) {
            chainIdx++;
            var next = chain[chainIdx];
            emit({ kind: 'warn', payload: { message:
              (chain[chainIdx - 1] ? chain[chainIdx - 1].name : cA.provider) +
              ' unavailable (' + clip(msg, 120) + ') — failing over to ' +
              next.name + ' / ' + next.model } });
            writeRecord('debug',
              'provider failover ' + (chain[chainIdx - 1] ? chain[chainIdx - 1].name : cA.provider) +
              ' -> ' + next.name,
              { type: 'provider_failover', from: (chain[chainIdx - 1] ? chain[chainIdx - 1].name : cA.provider),
                to: next.name, toModel: next.model, error: clip(msg, 300) });
            continue;
          }
          /* Project record: the failure is an ISSUE (actionable) with a
           * debug mirror — both linked to this session. */
          writeRecord('issue', 'turn failed: ' + clip(msg, 90),
                      { type: 'turn_error', error: clip(msg, 400),
                        model: cA.model || '', provider: cA.provider || '',
                        chainIndex: chainIdx });
          writeRecord('debug', 'turn error detail: ' + clip(msg, 90),
                      { type: 'turn_error_detail', error: clip(msg, 400) });
          throw e;
        }
      }
      var u = res.usage || {};
      emit({ kind: 'done', payload: {
        answer: res.answer,
        stopped: res.stopped,
        /* Which model actually served the turn — the truth after any
         * failover, so the UI never credits the wrong model. */
        servedBy: res.servedBy || cfg().model || '',
        usage: {
          promptTokens: u.promptTokens || 0,
          completionTokens: u.completionTokens || 0,
          cacheReadTokens: u.cacheReadTokens || 0,
          cacheWriteTokens: u.cacheWriteTokens || 0,
          llmCalls: u.llmCalls || 0,
          toolCalls: u.toolCalls || 0,
          costUsd: res.costUsd || 0,
          steps: res.steps || 0,
        },
      } });
      return { ok: true, answer: res.answer, usage: u, stopped: res.stopped };
    } finally {
      S.streaming = false;
      S.onEvent = null;
    }
  }

  function cancel() {
    /* Abort an in-flight RLM episode (rlm.js checks this between rounds). */
    try { globalThis.__rlm_aborted = true; } catch (e) {}
    if (sofuu.agent && typeof sofuu.agent.cancel === 'function') {
      try { sofuu.agent.cancel('chat'); } catch (e) {}
    }
    /* Unblock any pending approval waits — the run is being aborted. */
    for (var id in S.pendingApprovals) {
      var p = S.pendingApprovals[id];
      delete S.pendingApprovals[id];
      try { if (p.timer) clearTimeout(p.timer); } catch (eT) {}
      emit({ kind: 'approval_resolved', payload: { id: id } });
      try { p.resolve(false); } catch (e) {}
    }
    return { ok: true };
  }

  async function compact() {
    if (S.history.length === 0) return { ok: false, error: 'nothing to compact yet' };
    var before = historyTokens();
    var folded;
    try {
      folded = await summarizeHistory(1); /* fold all but the newest turn */
    } catch (e) {
      return { ok: false, error: 'compact failed: ' + String((e && e.message) || e) };
    }
    if (!folded) return { ok: false, error: 'nothing older than one turn to fold' };
    return { ok: true, savedTokens: Math.max(0, before - historyTokens()) };
  }

  function clearHistory() {
    S.history = [];
    S.autoCompactArmed = true;
    return { ok: true };
  }

  function resume(id) {
    if (!S.project) return { ok: false, error: 'no project set' };
    var data = loadSessionData(S.project, String(id || ''), S.pastTurns);
    if (!data) return { ok: false, error: 'session not found (or QTSQ unavailable)' };
    var turns = turnsFromData(data, S.pastTurns);
    /* A minted-but-unused session (zero turns) MUST still attach: refusing
     * it leaves the host sessionless next to an existing pill, and the next
     * turn then mints a duplicate (the desktop "session-2" bug). */
    S.history = [];
    for (var i = 0; i < turns.length; i++) {
      S.history.push({ role: 'user', content: turns[i].prompt });
      S.history.push({ role: 'assistant', content: turns[i].answer });
    }
    /* Structured resume (P1): the prior turns' tool activity rides along
     * as one readable context block — the model knows which files were
     * read, what ran and what came back, instead of inheriting only the
     * prose. (Full tool_use/tool_result wire pairs are NOT rebuilt: their
     * call ids are gone, and fabricated ids are how providers 400.) */
    var acts = [];
    for (var t2 = 0; t2 < turns.length; t2++) {
      var tls = turns[t2].tools || [];
      for (var t3 = 0; t3 < tls.length; t3++) {
        var row = '- ' + (tls[t3].n || 'tool');
        if (tls[t3].a) row += ' ' + tls[t3].a;
        if (tls[t3].r) row += ' -> ' + tls[t3].r;
        acts.push(row);
      }
    }
    if (acts.length) {
      S.history.push({ role: 'system', content:
        '[Resumed session — tool activity from the turns above, oldest first:\n' +
        acts.slice(-40).join('\n') + '\nUse this context; do not redo this work.]' });
    }
    /* Continue the resumed session: new turns append to its transcript
     * (and its conv/ event stream — numbering continues, never restarts). */
    S.session = data;
    S.eventSeq = (data.events && data.events.length)
      ? (data.events[data.events.length - 1].seq | 0) : 0;
    S.info = {
      id: data.id, pid: pid(), host: 'desktop',
      cwd: data.cwd || '', model: data.model || '', provider: data.provider || '',
      started_at: data.created || now(), last_seen: now(),
      task: data.task || null, ended: false,
    };
    registryUpsert(S.project, S.info);
    sessionLog('start', 'session resumed');
    return { ok: true, sessionId: data.id, turns: turns.length };
  }

  function sessionTurns(id) {
    if (!S.project) return [];
    var data = loadSessionData(S.project, String(id || ''), S.pastTurns);
    return data ? turnsFromData(data, S.pastTurns) : [];
  }

  function newSession() {
    S.history = [];
    S.alwaysAllowed = {};
    S.alwaysAllowedCommands = {};
    S.autoCompactArmed = true;
    var c = cfg();
    // Allow a session even without a project so the desktop topbar can
    // show a fresh "Untitled" pill on the home screen. Submission is still
    // gated by setProject (returns an error from the engine), but the
    // session exists in-memory and is visible until the user picks one.
    var p = S.project || '';
    sessionJoin(p, c.model || '', c.provider || '');
    return { ok: true, sessionId: S.info ? S.info.id : null };
  }

  function setProject(path) {
    if (!path || typeof path !== 'string') return { ok: false, error: 'path required' };
    loadConfig();
    S.history = [];
    S.alwaysAllowed = {};
    S.alwaysAllowedCommands = {};
    S.autoCompactArmed = true;
    /* attach=true: switching projects never creates a chat. */
    setProjectInternal(path, true);
    return { ok: true, sessionId: S.info ? S.info.id : null, project: S.project };
  }

  function setPermissions(profile) {
    var p = String(profile || '');
    if (p === 'full' || p === 'edit' || p === 'plan') {
      S.permissionProfile = p;
      S.permissions = 'auto'; /* profiles decide per-tool; no prompts */
      return { ok: true, permissions: p };
    }
    /* 'prompt' (or anything unknown) → the desktop approval gate */
    S.permissionProfile = 'prompt';
    S.permissions = 'prompt';
    return { ok: true, permissions: 'prompt' };
  }

  function state(opts) {
    var c = cfg();
    var pending = 0;
    for (var k in S.pendingApprovals) pending++;
    var st = {
      ok: true,
      host: S.host,
      project: S.project || null,
      sessionId: S.info ? S.info.id : null,
      model: c.model || '',
      provider: c.provider || '',
      effort: c.effort || '',
      streaming: S.streaming,
      historyTurns: Math.floor(S.history.length / 2),
      contextTokens: S.totalTk,
      sessionTokens: S.totalTk,
      sessionSpendUsd: S.sessionSpendUsd || 0,
      permissions: S.permissions,
      permissionProfile: S.permissionProfile,
      pendingApprovals: pending,
    };
    /* Retained history (retention forensics + hosts that surface the
     * transcript). OPT-IN — the desktop polls a settings card that never
     * reads it, and copying a 2000-entry history per call would be pure
     * serialization cost. Copies, never live references: tool_calls is
     * deep-copied too (js-4) — a host mutating the returned array must
     * not be able to rewrite the live request history that the next turn
     * sends to the provider. */
    if (opts && opts.history) {
      st.history = S.history.map(function (m) {
        var tcs;
        if (m.tool_calls && m.tool_calls.length) {
          try { tcs = JSON.parse(JSON.stringify(m.tool_calls)); }
          catch (eTC) { tcs = undefined; /* non-JSON debris: omit from the copy */ }
        }
        return { role: m.role, content: String(m.content == null ? '' : m.content),
                 tool_calls: tcs, tool_call_id: m.tool_call_id || undefined };
      });
    }
    return st;
  }

  /* Delete a session: detach if active, drop the registry entry. The
   * session's FOLDER (sessions/<sid>/ — and legacy flat files) is removed
   * by the host's Rust side; the engine never deletes files. */
  function deleteSession(id) {
    if (!S.project) return { ok: false, error: 'no project set' };
    id = String(id || '');
    if (!id) return { ok: false, error: 'id required' };
    var wasActive = !!(S.info && S.info.id === id);
    if (wasActive) {
      /* Detach silently — the transcript is about to be deleted, so no
       * 'end' event (that would write into a doomed conv stream). */
      S.session = null;
      S.info = null;
      S.history = [];
    }
    var reg = registryRead(S.project);
    var out = [];
    for (var i = 0; i < reg.sessions.length; i++) {
      var si = reg.sessions[i];
      if (si && si.id !== id) out.push(si);
    }
    try {
      ensureDir(sessionsDir(S.project));
      if (typeof __qtsq_session_save === 'function') {
        __qtsq_session_save(registryPath(S.project), JSON.stringify({ sessions: out }), sessionPassword(S.project));
      }
      /* Scrub the legacy plaintext registry too (old-era sessions live
       * there; the existence filter at read time also guards, but the
       * file should not keep listing the dead). */
      try {
        var lraw = (typeof __session_read_file === 'function')
          ? __session_read_file(legacyRegistryPath(S.project)) : null;
        if (lraw) {
          var lj = JSON.parse(lraw);
          if (lj && Array.isArray(lj.sessions)) {
            var lout = [];
            for (var lk = 0; lk < lj.sessions.length; lk++) {
              if (lj.sessions[lk] && lj.sessions[lk].id !== id) lout.push(lj.sessions[lk]);
            }
            if (lout.length !== lj.sessions.length && typeof __session_write_file === 'function') {
              __session_write_file(legacyRegistryPath(S.project), JSON.stringify({ sessions: lout }, null, 2));
            }
          }
        }
      } catch (eL) {}
    } catch (e) {}
    return { ok: true, wasActive: wasActive };
  }

  /* Delete EVERY chat in the current workspace: drops all registry
   * entries (qtsq + legacy plaintext) and returns the ids so the host's
   * Rust side can remove the session folders. Detaches first when the
   * active session is among them. */
  function clearSessions() {
    if (!S.project) return { ok: false, error: 'no project set' };
    var reg = registryRead(S.project);
    var ids = [];
    for (var i = 0; i < reg.sessions.length; i++) {
      if (reg.sessions[i] && reg.sessions[i].id) ids.push(reg.sessions[i].id);
    }
    var wasActive = !!(S.info && ids.indexOf(S.info.id) >= 0);
    if (wasActive) {
      /* Detach silently — the transcript is about to be deleted. */
      S.session = null;
      S.info = null;
      S.history = [];
    }
    try {
      ensureDir(sessionsDir(S.project));
      if (typeof __qtsq_session_save === 'function') {
        __qtsq_session_save(registryPath(S.project), JSON.stringify({ sessions: [] }), sessionPassword(S.project));
      }
    } catch (e) {}
    /* The legacy plaintext registry must go empty too, or its surviving
     * entries (whose flat files the host is about to remove) would
     * resurrect through the read-time merge. */
    try {
      if (typeof __session_read_file === 'function' && __session_read_file(legacyRegistryPath(S.project))) {
        if (typeof __session_write_file === 'function') {
          __session_write_file(legacyRegistryPath(S.project), JSON.stringify({ sessions: [] }, null, 2));
        }
      }
    } catch (eL) {}
    return { ok: true, ids: ids, cleared: ids.length, wasActive: wasActive };
  }

  /* Session listing for hosts (the desktop sidebar): the registry is
   * qtsq-encrypted, so the ENGINE reads it — the Rust mirror can't. */
  function sessions() {
    if (!S.project) return { sessions: [] };
    return registryRead(S.project);
  }

  /* Per-session usage loader: reads the session index (which already
   * lists every event as {seq, t, kind}, no texts) and decrypts ONLY the
   * tiny 'usage' conv files — never the prompts/answers. */
  function loadUsageRecords(project, id) {
    var out = [];
    if (typeof __qtsq_session_load !== 'function') return out;
    var pw = sessionPassword(project);
    try {
      var raw = __qtsq_session_load(sessionIndexPath(project, id), pw);
      if (!raw) return out;
      var d = JSON.parse(raw);
      if (!d || !Array.isArray(d.events)) return out;
      for (var i = 0; i < d.events.length; i++) {
        var ev = d.events[i];
        if (!ev || ev.kind !== 'usage') continue;
        try {
          var f = ev.file || (convDir(project, id) + '/' +
            padN(ev.seq, 6) + '-' + ev.t + '-' + ev.kind + '.qtsq');
          var eraw = __qtsq_session_load(f, pw);
          if (!eraw) continue;
          var eobj = JSON.parse(eraw);
          var rec = null;
          try { rec = JSON.parse(String((eobj && eobj.text) || '')); } catch (eP) { continue; }
          if (rec && typeof rec === 'object') {
            rec.t = ev.t | 0;
            out.push(rec);
          }
        } catch (eE) {}
      }
    } catch (e) {}
    return out;
  }

  /* ── Usage aggregation (desktop Settings › Usage) ───────────────────
   * Walks every session in the workspace registry, reads each session's
   * conv stream, and sums the per-turn 'usage' events logged by submit().
   * Returns per-model totals, a per-day series (LOCAL calendar day of the
   * event timestamp — matches the day labels the user actually sees; the
   * old "UTC" comment never matched the code) and per-session totals —
   * all real recorded tokens, no
   * estimates. Sessions whose events were logged before the usage event
   * existed simply contribute nothing. */
  function usage() {
    var empty = {
      ok: true, project: S.project || null,
      models: [], days: [], sessions: [],
      totals: { pt: 0, ct: 0, cr: 0, cw: 0, turns: 0, cost: 0, sessions: 0 },
    };
    if (!S.project) return empty;
    var reg = registryRead(S.project);
    var list = (reg && Array.isArray(reg.sessions)) ? reg.sessions : [];
    /* Null-prototype maps (P3): `model` comes straight off disk records —
     * a forged model:"__proto__" would make byModel[model] resolve to
     * Object.prototype and pollute every object in the runtime. */
    var byModel = Object.create(null);  /* model → {pt, ct, cr, cw, turns, cost} */
    var byDay = Object.create(null);    /* 'YYYY-MM-DD' → {pt, ct, turns} */
    var bySession = [];
    var tot = { pt: 0, ct: 0, cr: 0, cw: 0, turns: 0, cost: 0 };
    var pw = sessionPassword(S.project);
    for (var si = 0; si < list.length; si++) {
      var info = list[si] || {};
      var sid = String(info.id || '');
      if (!sid) continue;
      var usageRecs = loadUsageRecords(S.project, sid);
      var sPt = 0, sCt = 0, sTurns = 0, sCost = 0;
      for (var ri = 0; ri < usageRecs.length; ri++) {
        var rec = usageRecs[ri];
        var pt = rec.pt | 0, ct = rec.ct | 0;
        var cr = rec.cr | 0, cw = rec.cw | 0;
        var cost = Number(rec.cost) || 0;
        var model = String(rec.model || 'unknown');
        var m = byModel[model];
        if (!m) m = byModel[model] = { model: model, pt: 0, ct: 0, cr: 0, cw: 0, turns: 0, cost: 0 };
        m.pt += pt; m.ct += ct; m.cr += cr; m.cw += cw; m.turns++; m.cost += cost;
        var d = new Date(rec.t * 1000);
        var day = d.getFullYear() + '-' + padN(d.getMonth() + 1, 2) + '-' + padN(d.getDate(), 2);
        var dy = byDay[day];
        if (!dy) dy = byDay[day] = { day: day, pt: 0, ct: 0, turns: 0 };
        dy.pt += pt; dy.ct += ct; dy.turns++;
        tot.pt += pt; tot.ct += ct; tot.cr += cr; tot.cw += cw;
        tot.turns++; tot.cost += cost;
        sPt += pt; sCt += ct; sTurns++; sCost += cost;
      }
      if (sTurns > 0) {
        bySession.push({
          id: sid, model: info.model || '', task: info.task || null,
          started_at: info.started_at | 0, last_seen: info.last_seen | 0,
          pt: sPt, ct: sCt, turns: sTurns, cost: sCost,
        });
      }
    }
    var models = [];
    for (var k in byModel) models.push(byModel[k]);
    models.sort(function (a, b) { return (b.pt + b.ct) - (a.pt + a.ct); });
    var days = [];
    for (var kd in byDay) days.push(byDay[kd]);
    days.sort(function (a, b) { return a.day < b.day ? -1 : a.day > b.day ? 1 : 0; });
    bySession.sort(function (a, b) { return b.last_seen - a.last_seen; });
    var out = {
      ok: true, project: S.project || null,
      models: models, days: days, sessions: bySession,
      totals: {
        pt: tot.pt, ct: tot.ct, cr: tot.cr, cw: tot.cw,
        turns: tot.turns, cost: Math.round(tot.cost * 1e6) / 1e6,
        sessions: bySession.length,
      },
    };
    return out;
  }

  sofuu.chat = {
    init: init,
    submit: submit,
    cancel: cancel,
    compact: compact,
    clear: clearHistory,
    resume: resume,
    resolveApproval: resolveApproval,
    state: state,
    sessions: sessions,
    usage: usage,
    deleteSession: deleteSession,
    clearSessions: clearSessions,
    sessionTurns: sessionTurns,
    newSession: newSession,
    setProject: setProject,
    setPermissions: setPermissions,
    /* F2: force a caps harvest for the active endpoint and report the
     * detected window — call after any model/provider switch. */
    detectCaps: detectCaps,
    resolveCaps: resolveCaps,
  };
})();
