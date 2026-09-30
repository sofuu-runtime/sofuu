/* src/js/rlm.js — sofuu.rlm: the async RLM driver (PLAN-RLM R2).
 *
 * Shipped JS, eval'd once per engine context by the Rust seam
 * sofuu_rust_register_engine_js (crates/sofuu-core/src/rlm/js_api.rs),
 * which registers the synchronous __rlm_* host functions first. This file
 * wraps them into the async public API; model calls go through
 * sofuu.ai.complete, so every configured provider works and no API key
 * ever crosses into the Rust RLM code.
 *
 * Abort seam: by design there is no opts.signal — the chat's Esc handler
 * sets globalThis.__rlm_aborted = true; the loop checks the flag before
 * every round, frees the episode, and resolves with
 * { answer: '(aborted)', stopped: 'aborted', ... }.
 */
(function () {
  'use strict';
  var g = globalThis;
  var VERSION = '0.1.0';

  /* Headless first: never crash because the namespace object is missing. */
  if (typeof g.sofuu === 'undefined' || g.sofuu === null) g.sofuu = {};
  var sofuu = g.sofuu;

  function parseHostJson(raw, what) {
    var a;
    try { a = JSON.parse(raw); }
    catch (e) { throw new Error('rlm: bad ' + what + ' JSON from host: ' + e); }
    if (a && a.error) throw new Error('rlm: ' + a.error);
    return a;
  }

  /* Pass the caller's provider fields through to sofuu.ai.complete. */
  function completeOpts(opts, messages) {
    var o = { messages: messages };
    if (opts.provider) o.provider = opts.provider;
    if (opts.model) o.model = opts.model;
    if (opts.effort) o.effort = opts.effort;
    if (opts.base_url) o.base_url = opts.base_url;
    if (opts.api_key) o.api_key = opts.api_key;
    /* profile (wire format for custom providers) + max_tokens (/maxout)
     * were dropped here — RLM turns ignored both. */
    if (opts.profile) o.profile = opts.profile;
    if (opts.max_tokens) o.max_tokens = opts.max_tokens;
    return o;
  }

  /* alloc gate (PLAN-ML-GATES §14): RLM asks must fit the selected model
   * too. Resolves the plan from the ask's own shape (caps come from the
   * registry → caps the endpoint itself published for this model →
   * learned limits, all inside Rust; baseUrl tells it which endpoint),
   * takes the feasibility-checked output reserve, and truncates the
   * largest message when the prompt still does not fit. ML off → returns
   * the options untouched. */
  function allocFit(o) {
    if (!sofuu.ml || !sofuu.ml.alloc || typeof sofuu.ml.alloc.plan !== 'function') return o;
    try {
      var ms = o.messages || [];
      var tk = 0;
      for (var i = 0; i < ms.length; i++) tk += Math.ceil(String(ms[i].content || '').length / 4);
      var p = JSON.parse(sofuu.ml.alloc.plan(JSON.stringify({
        model: o.model || '', baseUrl: o.base_url || '',
        cfgMaxOutput: o.max_tokens || 0,
        historyTk: tk, turns: Math.floor(ms.length / 2), taskTk: tk,
      })));
      if (!p || !(p.window > 0)) return o;
      var reserve = o.max_tokens > 0 ? o.max_tokens : Math.max(512, Math.floor(p.window * 0.05));
      if (p.maxOutput > 0 && (o.max_tokens <= 0 || o.max_tokens > p.maxOutput)) {
        o.max_tokens = p.maxOutput;
        reserve = p.maxOutput;
      }
      if (tk + reserve <= p.window) return o;
      /* Does not fit: clamp the reserve to what is left; if even that is
       * gone, truncate the largest message so the ask goes out at all. */
      var left = p.window - tk - Math.max(256, Math.floor(p.window * 0.02));
      if (left >= 512) { o.max_tokens = Math.floor(left); return o; }
      var big = -1, bigLen = 0;
      for (var j = 0; j < ms.length; j++) {
        var l = String(ms[j].content || '').length;
        if (l > bigLen) { big = j; bigLen = l; }
      }
      if (big >= 0) {
        var keep = Math.max(200, (p.window - 512) * 4);
        if (bigLen > keep) {
          ms[big] = Object.assign({}, ms[big], {
            content: String(ms[big].content).slice(0, keep) +
              '\n…[truncated by the alloc gate to fit the model window]…',
          });
        }
      }
      o.max_tokens = 512;
      return o;
    } catch (e) { return o; }
  }
  /* Layer 1 error learning (same mechanism as agent.js): returns the
   * learned KIND ('context' | 'output') when the text parsed and the real
   * limit is cached, '' otherwise. */
  function allocNoteLimit(model, errText) {
    if (!sofuu.ml || !sofuu.ml.alloc || typeof sofuu.ml.alloc.noteLimit !== 'function') return '';
    try {
      var k = sofuu.ml.alloc.noteLimit(String(model || ''), String(errText == null ? '' : errText));
      return (k === 'context' || k === 'output') ? k : '';
    } catch (e) { return ''; }
  }

  /* ai.complete resolves to { text, usage, ... }; tolerate a bare string.
   * With opts.recurseVia = { agent: "name" } (PLAN-AGENTS A4.2), a
   * single-user-prompt sub-call (the resolve_llm recursion path) runs
   * through sofuu.agent.run instead of a raw completion — recursive RLM
   * steps gain tools and private memory. The top-level 2-message call is
   * always a plain completion.
   *
   * Asks run STREAM-FIRST (like every agent turn): ai.complete uses a
   * separate curl path with no cancellation hook, while streams expose
   * .abort() — that is what makes mid-request Esc work (the watcher in
   * query() kills in-flight asks the moment __rlm_aborted is set). */
  var ACTIVE_ASKS = [];   // { abort, episode } per in-flight ask stream
  /* P2-8 (AUDIT-2026-09-01): the abort flag used to be ONE module global —
   * a second concurrent sofuu.rlm.query reset it (g.__rlm_aborted = false)
   * and cleared the first episode's pending cancel; one cancel killed both
   * episodes. Now each query() owns an episode record; the host-set
   * __rlm_aborted remains the cancel SIGNAL (watchers copy it into their
   * own episode flags), but only the episode's own flag steers its loop. */
  var EPISODES = [];
  function episodeAborted(ep) {
    if (g.__rlm_aborted) ep.aborted = true;
    return !!ep.aborted;
  }

  async function ask(opts, messages, ep) {
    /* alloc gate Layer 1 — error learning: if the provider rejects the
     * ask over a limit, cache the real limit for this model and retry
     * ONCE PER KIND (allocFit re-resolves it). */
    var retriedKinds = {};
    for (;;) {
      try {
        return await askOnce(opts, messages, ep);
      } catch (e) {
        if (episodeAborted(ep)) return '';
        var kind = allocNoteLimit(opts && opts.model, (e && e.message) || e);
        if (kind && !retriedKinds[kind]) { retriedKinds[kind] = true; continue; }
        throw e;
      }
    }
  }

  async function askOnce(opts, messages, ep) {
    if (opts.recurseVia && messages.length === 1 && messages[0].role === 'user' &&
        g.sofuu && g.sofuu.agent && typeof g.sofuu.agent.run === 'function') {
      var r2 = await g.sofuu.agent.run(String(opts.recurseVia.agent), messages[0].content, { plain: true });
      return (r2 && r2.answer) ? r2.answer : '';
    }
    var o = allocFit(completeOpts(opts, messages));
    if (sofuu.ai && typeof sofuu.ai.stream === 'function') {
      var st = sofuu.ai.stream(o);
      var entry = { abort: function () { try { st.abort(); } catch (eA) {} }, episode: ep };
      ACTIVE_ASKS.push(entry);
      var text = '';
      try {
        for await (var ch of st) {
          if (ch && ch.text) text += ch.text;
        }
      } catch (eS) {
        /* Aborted mid-flight: swallow so the episode loop's top-of-loop
         * check reports stopped:'aborted' instead of rejecting the whole
         * query (agent.js would otherwise fall back to the plain loop). */
        if (episodeAborted(ep)) return '';
        throw eS;
      } finally {
        var ix = ACTIVE_ASKS.indexOf(entry);
        if (ix >= 0) ACTIVE_ASKS.splice(ix, 1);
      }
      return text;
    }
    var r = await sofuu.ai.complete(o);
    if (r && typeof r === 'object' && typeof r.text === 'string') return r.text;
    if (typeof r === 'string') return r;
    return r ? String(r) : '';
  }

  /**
   * sofuu.rlm.query(context, question, opts) → Promise resolving to
   * { answer, calls, toolCalls, rounds, ms, stopped, depthReached, trace? }
   * opts: { provider, model, effort, base_url, api_key,
   *         maxDepth, maxLlmCalls, maxWallMs, maxRounds, chunkChars, trace,
   *         tools: [{name, description, parameters}],   (A4: sandbox whitelist)
   *         maxToolCalls,
   *         execTool: async (name, args) => string }    (A4: executor — runs
   *           in the ENGINE context through the agent's normal tool path;
   *           never inside the sandbox, so tool code has no context access) }
   */
  async function query(context, question, opts) {
    opts = opts || {};
    if (typeof __rlm_new !== 'function')
      throw new Error('rlm: host functions not registered');
    if (!sofuu.ai || typeof sofuu.ai.complete !== 'function')
      throw new Error('rlm: sofuu.ai.complete unavailable');

    /* Tool specs are schema-only (G3: no executors cross into Rust — they
     * would be meaningless there; the host only needs names/schemas for
     * the sandbox system prompt and the budget). */
    var toolSpecs = Array.isArray(opts.tools)
      ? opts.tools.map(function (t) {
          return {
            name: String(t && t.name),
            description: String((t && t.description) || ''),
            parameters: (t && t.parameters) || null
          };
        })
      : [];

    var created = parseHostJson(__rlm_new(JSON.stringify({
      context: String(context == null ? '' : context),
      question: String(question == null ? '' : question),
      opts: {
        maxDepth: opts.maxDepth,
        maxLlmCalls: opts.maxLlmCalls,
        maxWallMs: opts.maxWallMs,
        maxRounds: opts.maxRounds,
        chunkChars: opts.chunkChars,
        maxToolCalls: opts.maxToolCalls,
        trace: !!opts.trace,
        tools: toolSpecs
      }
    })), 'new');
    var id = created.id;
    var started = Date.now();
    /* Per-episode abort record (P2-8): the host flag stays the SIGNAL, but
     * each episode latches it into its own token — episodes no longer
     * reset or clear each other's cancels. */
    var ep = { aborted: false };
    EPISODES.push(ep);
    /* Mid-request abort: poll the host-set flag and kill any in-flight ask
     * streams (Esc during a long LLM call — previously only noticed at the
     * next round boundary). 100ms poll ≈ the chat's Esc disambiguation
     * timer; the flag itself is set synchronously by agent.cancel. */
    var watcher = setInterval(function () {
      if (!g.__rlm_aborted) return;
      ep.aborted = true;
      var list = ACTIVE_ASKS.slice();
      for (var i = 0; i < list.length; i++) {
        if (list[i].episode === ep) list[i].abort();
      }
    }, 100);
    try {
      var action = parseHostJson(__rlm_start(id), 'action');
      for (;;) {
        if (episodeAborted(ep)) {
          return {
            answer: '(aborted)', calls: 0, toolCalls: 0, rounds: 0,
            ms: Date.now() - started, stopped: 'aborted', depthReached: 0
          };
        }
        if (action.kind === 'messages') {
          var reply = await ask(opts, action.messages, ep);
          action = parseHostJson(__rlm_step(id, reply), 'action');
        } else if (action.kind === 'resolve_llm') {
          /* v1: every sub-prompt is a PLAIN completion. Recursive nested
           * episodes (a sub-answer produced by sofuu.rlm.query itself at
           * depth + 1) are a documented later step — Episode already
           * threads the depth parameter through for it. */
          var answers = await Promise.all(action.prompts.map(function (p) {
            return ask(opts, [{ role: 'user', content: p }], ep);
          }));
          action = parseHostJson(__rlm_feed_llm(id, JSON.stringify(answers)), 'action');
        } else if (action.kind === 'resolve_tools') {
          /* A4 full form: run each sandbox tool() call through the
           * caller's executor (the agent's normal tool path — timeout,
           * trace, budgets live there). Errors become readable tool-error
           * strings; a missing executor feeds a visible marker so the
           * model can route around it instead of hanging the episode. */
          var results = await Promise.all(action.calls.map(function (c) {
            var p = (typeof opts.execTool === 'function')
              ? opts.execTool(String(c && c.name), (c && c.args) || {})
              : Promise.resolve(null);
            return p.then(
              function (r) { return (r === undefined || r === null) ? '' : String(r); },
              function (e) { return 'tool error: ' + String((e && e.message) || e); }
            );
          }));
          action = parseHostJson(__rlm_feed_tools(id, JSON.stringify(results)), 'action');
        } else if (action.kind === 'done') {
          var r = action.result || {};
          var out = {
            answer: r.answer,
            calls: r.calls,
            toolCalls: r.toolCalls !== undefined ? r.toolCalls : (r.tool_calls || 0),
            rounds: r.rounds,
            ms: r.ms,
            stopped: r.stopped,
            depthReached: r.depth_reached /* camelCase here, snake in trace */
          };
          if (r.trace !== undefined) out.trace = r.trace;
          return out;
        } else {
          throw new Error('rlm: unknown action kind ' + JSON.stringify(action.kind));
        }
      }
    } finally {
      clearInterval(watcher);
      /* Episodes never leak — free also on error/abort/throw paths. */
      var epIx = EPISODES.indexOf(ep);
      if (epIx >= 0) EPISODES.splice(epIx, 1);
      /* Re-arm the host cancel signal for the next episode — but only
       * when the LAST episode leaves (js-10, AUDIT-2026-09-07). The flag
       * is a SIGNAL, not per-episode state (each episode copies it into
       * ep.aborted via episodeAborted) — so leaving it set harms nobody
       * still running, but clearing it here used to steal the signal
       * from sibling episodes: whichever episode tore down first wiped
       * the Esc before a slower sibling's watcher/boundary check saw
       * it. Clearing on last-exit keeps the P3-13 guard too (the next
       * query is never stillborn-aborted by a stale flag). */
      if (g.__rlm_aborted && EPISODES.length === 0) g.__rlm_aborted = false;
      try { __rlm_free(id); } catch (e) {}
    }
  }

  function route(ctxTokens, windowTokens, question) {
    if (typeof __rlm_route !== 'function') return 'plain';
    return __rlm_route(ctxTokens, windowTokens, String(question == null ? '' : question));
  }

  /* Persist one routing decision ({ctxTokens, windowTokens, question,
   * route, latencyMs, rlmCalls}) to ~/.sofuu/routing_log.jsonl. */
  function logRoute(d) {
    if (typeof __rlm_log_route !== 'function') return;
    try { __rlm_log_route(JSON.stringify(d || {})); } catch (e) {}
  }

  sofuu.rlm = {
    VERSION: VERSION,
    query: query,
    route: route,
    logRoute: logRoute
  };
})();
