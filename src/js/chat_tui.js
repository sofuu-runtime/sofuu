/* chat_tui.js — the interactive chat TUI driver (sofuu chat).
 * @sofuu-source-only — eval'd as source from Rust; NOT a shipped .bc driver
 * (scripts/gen-driver-bc.sh and scripts/check-driver-bc.sh skip this marker).
 *
 * AUDIT-2026-10-31 P1-8: this text used to be a 198 KB raw string literal
 * inside crates/sofuu-core/src/chat.rs. It moved here so build.rs can embed
 * it gzipped (~59 KB) and chat.rs inflates it on first use; this file is the
 * single source of truth. Rust registers the __chat_* host functions it
 * calls (crates/sofuu-core/src/chat.rs).
 */

(function() {
  let cfg = JSON.parse(__chat_getcfg());
  /* Permission mode: cfg.permissions is the persisted source of truth
   * (/mode, /plan, /edit, /full write it via Rust). The runtime gate is
   * ONE — sofuu.agent.setPermissions drives agent.js's permissionBlocked.
   * Idempotent + guarded so a not-yet-loaded agent.js can't break boot. */
  function applyMode() {
    try {
      if (sofuu.agent && typeof sofuu.agent.setPermissions === 'function')
        sofuu.agent.setPermissions(cfg.permissions || 'full');
    } catch (e) {}
  }
  applyMode();
  let history = [];
  let lastShared = '';
  /* Output archive state is deliberately driver-local. Storage is additive
   * and best-effort: a QTSQ/index failure must never turn a usable model
   * response into a failed chat turn. */
  let archiveTurn = 0;
  let archiveAttempt = 1;
  let archivePendingRecovery = '';
  let archiveLastFailure = null;
  let archiveRecoveryUsed = false;
  /* session-2 (AUDIT-2026-09-07): the prompt event logs ONCE per logical
   * turn. turn() runs again on THINK/CAPACITY retries with the same
   * archiveTurn, so gate on it — duplicate prompt events would replay as
   * extra user turns on resume. */
  let promptLoggedTurn = -1;
  function archiveWrite(kind, status, source, content, metadata, links) {
    if (cfg.archive_enabled === false || content === undefined || content === null) return null;
    const text = String(content);
    if (!text.trim()) return null;
    const request = {
      turn: archiveTurn,
      attempt: archiveAttempt,
      kind: kind,
      status: status,
      source: source,
      provider: (applyFailover(cfg).provider || cfg.provider || ''),
      model: (applyFailover(cfg).model || cfg.model || ''),
      content: text,
      metadata: metadata || {},
      related_output_ids: (links && links.related_output_ids) || [],
      retry_of: (links && links.retry_of) || null,
      redact: true,
    };
    try {
      const result = JSON.parse(__chat_archive_write(JSON.stringify(request)) || '{}');
      return (result && result.stored && result.ref) ? result.ref : null;
    } catch (e) {
      return null;
    }
  }
  function archiveErrorCode(message) {
    const text = String(message || '').toLowerCase();
    if (/context|prompt.{0,12}too long|maximum context|token limit/.test(text)) return 'context_limit';
    if (/max.{0,12}output|length.{0,12}limit|too many output|finish.?reason.{0,8}length/.test(text)) return 'output_limit';
    if (/auth|unauthori[sz]ed|forbidden|api.?key|invalid token|credential/.test(text)) return 'auth';
    if (/timeout|timed out|connection|transport|http\/?2|empty reply|reset by peer|rate.?limit|overloaded|temporarily/.test(text)) return 'transport';
    if (/provider|model|endpoint|http \d+/.test(text)) return 'provider';
    return 'runtime';
  }
  function archiveContext(query) {
    const automatic = !!(query && query.automatic === true);
    if (cfg.archive_enabled === false || (automatic && cfg.archive_auto_recover === false)) return '';
    try {
      const requestedBudget = Number(query && query.budget_chars || cfg.archive_recovery_budget_chars || 12000);
      const q = Object.assign({}, query || {}, {
        automatic: automatic,
        budget_chars: Math.max(512, Math.min(131072, isFinite(requestedBudget) ? requestedBudget : 12000)),
        ml_relevance: cfg.ml !== false && !!(sofuu.ml && sofuu.ml.relevance &&
          typeof sofuu.ml.relevance.plan === 'function'),
      });
      const result = JSON.parse(__chat_archive_context(JSON.stringify(q)) || '{}');
      if (automatic && result && result.bundle && Array.isArray(result.bundle.items) && result.bundle.items.length) {
        const items = result.bundle.items;
        const latest = items.reduce(function (best, item) {
          return !best || String(item.created_at || '') > String(best.created_at || '') ? item : best;
        }, null);
        const kinds = items.slice(0, 3).map(function (item) { return String(item.kind || 'output'); }).join(' + ');
        try { out('\x1b[90m  ⏺ recovered ' + items.length + ' historical output' + (items.length === 1 ? '' : 's') +
          (latest && latest.created_at ? ' · latest ' + latest.created_at : '') +
          (kinds ? ' · ' + kinds : '') + '\x1b[0m\n'); } catch (eNotice) {}
      }
      return result && result.bundle && result.bundle.context ? String(result.bundle.context) : '';
    } catch (e) {
      return '';
    }
  }
  /* Public native-backed surface for scripts running inside the main chat
   * runtime. It returns parsed JSON envelopes so callers do not need to know
   * about the QuickJS string bridge. Historical context remains explicitly
   * bounded and labeled by the native implementation. */
  if (!sofuu.outputs) sofuu.outputs = {};
  function archiveApiResult(raw) {
    try { return JSON.parse(raw || '{}'); }
    catch (e) { return { ok: false, error: 'invalid output archive response' }; }
  }
  sofuu.outputs.info = function () {
    return archiveApiResult(__chat_archive_stats());
  };
  sofuu.outputs.list = function (query) {
    return archiveApiResult(__chat_archive_list(JSON.stringify(query || {})));
  };
  sofuu.outputs.get = function (id) {
    return archiveApiResult(__chat_archive_get(String(id || '')));
  };
  sofuu.outputs.search = function (query) {
    const q = typeof query === 'string' ? { text: query } : (query || {});
    return archiveApiResult(__chat_archive_search(JSON.stringify(q)));
  };
  sofuu.outputs.context = function (query) {
    const q = typeof query === 'string' ? { current_task: query } : Object.assign({}, query || {});
    q.automatic = false;
    return archiveApiResult(__chat_archive_context(JSON.stringify(q)));
  };
  sofuu.outputs.rebuildIndex = function () {
    return archiveApiResult(__chat_archive_rebuild());
  };
  sofuu.outputs.prune = function (policy) {
    const p = Object.assign({ dry_run: true }, policy || {});
    return archiveApiResult(__chat_archive_prune(JSON.stringify(p)));
  };
  /* Provider failover (P1): when a turn dies with a capacity/auth-class
   * error on the active provider, the main loop picks the next
   * configured provider entry and sets this override for the retry. All
   * per-turn consumers (streamOpts, resolveCaps, the agent def) read cfg
   * through applyFailover() so the whole turn runs against the target.
   * Cleared at turn end — the SAVED config never changes; failover is a
   * per-turn recovery, not a provider switch. */
  let failoverTo = null;
  function applyFailover(c) {
    if (!failoverTo) return c;
    return Object.assign({}, c, {
      provider: failoverTo.name,
      base_url: failoverTo.endpoint,
      api_key: failoverTo.api_key || '',
      profile: failoverTo.profile || '',
      model: failoverTo.model,
    });
  }
  function failoverChain() {
    const chain = [null];
    if (Array.isArray(cfg.providers)) {
      for (const pe of cfg.providers) {
        if (pe && pe.name && pe.model && pe.endpoint && pe.name !== cfg.provider) {
          chain.push({ name: pe.name, endpoint: pe.endpoint,
                       api_key: pe.api_key || '', profile: pe.profile || '',
                       model: pe.model });
        }
      }
    }
    return chain;
  }
  /* Footer meter = CURRENT context usage (what the next request's prompt
   * will be), not a lifetime total: it grows as history grows and drops
   * when /compact (or auto-compaction) folds history, exactly like the
   * context meters in other CLIs. Calibrated against the FIRST LLM request
   * of a turn — its prompt is system + history + tools + ephemeral, i.e.
   * the context size, before any tool results re-count it. Falls back to
   * the budget logic's own estimator when a provider reports no usage. */
  let usedCtx = 0;
  let ctxOverhead = -1;   /* −1 = not calibrated (history estimate only) */
  /* Live mid-turn growth (estimated tokens accumulated since the latest
   * in-flight request): the footer adds this WHILE a turn is running so
   * the meter climbs as the answer streams and tool results land — reset
   * to 0 when the turn ends (the end-of-turn calibration takes over). */
  let liveCtxExtra = 0;
  function ctxMeter() {
    const h = historyTokens();
    const base = ctxOverhead >= 0 ? h + ctxOverhead : h;
    return base + liveCtxExtra;
  }
  let lastChip = '';  /* last per-turn usage chip, so /compact can refresh the footer without clearing it */
  let rssWarned = false; /* M6: one-time RAM tripwire notice */
  /* ── MCP tool wiring (zero-config via ~/.sofuu/mcp.json) ────────
   * OPTIONAL and LAZY: servers are NOT connected at chat startup — a slow
   * or dead server must never block the prompt. Connections are made on
   * demand: the first /tools, or when the agent loop needs tools. Failed
   * servers are logged and skipped, never fatal. */
  let mcpClients = [];     /* [{name, client}] — live connections */
  let mcpTools = [];       /* [{server, name, description, schema}] */
  let mcpConnected = 0;
  let mcpFailed = 0;
  let mcpConnecting = null; /* in-flight connect promise (dedupe) */
  function mcpServersList() {
    try { return JSON.parse(__chat_mcpservers() || '[]'); } catch (e) { return []; }
  }
  async function connectMcpServers() {
    if (mcpConnecting) return mcpConnecting; /* one connect at a time */
    const list = mcpServersList();
    if (!list.length) return;
    if (!sofuu.mcp || !sofuu.mcp.connect) { out('\x1b[33m  ⚠ sofuu.mcp unavailable — tools off\x1b[0m'); return; }
    mcpConnecting = (async () => {
      const results = await Promise.allSettled(list.map(async (srv) => {
        const client = await sofuu.mcp.connect(srv.command);
        const res = await client.listTools();
        const arr = (res && res.tools) || [];
        return { name: srv.name, client, tools: arr };
      }));
      for (const r of results) {
        if (r.status === 'fulfilled' && r.value) {
          mcpClients.push({ name: r.value.name, client: r.value.client });
          for (const t of (r.value.tools || [])) {
            mcpTools.push({ server: r.value.name, name: t.name,
                            description: t.description || '', schema: t.inputSchema });
          }
          mcpConnected++;
        } else {
          mcpFailed++;
        }
      }
      if (mcpConnected > 0) {
        out('\x1b[90m  ⏺ mcp: ' + mcpConnected + ' server' + (mcpConnected === 1 ? '' : 's') + ' · ' + mcpTools.length + ' tools\x1b[0m');
        if (mcpFailed > 0) out('\x1b[90m  ⚠ ' + mcpFailed + ' MCP server' + (mcpFailed === 1 ? '' : 's') + ' unreachable (see ~/.sofuu/mcp.json)\x1b[0m');
      } else if (mcpFailed > 0) {
        out('\x1b[33m  ⚠ all ' + mcpFailed + ' MCP server' + (mcpFailed === 1 ? '' : 's') + ' unreachable — tools off\x1b[0m');
      }
    })();
    try {
      await mcpConnecting;
    } finally {
      mcpConnecting = null;
    }
  }
  function showTools() {
    if (mcpTools.length === 0) {
      out('\n\x1b[90m  No MCP servers connected. Add one to ~/.sofuu/mcp.json:\x1b[0m');
      out('  \x1b[90m  [{"name":"fs","command":"npx @modelcontextprotocol/server-filesystem /tmp"}]\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  MCP tools\x1b[0m');
    let lastServer = '';
    for (const t of mcpTools) {
      if (t.server !== lastServer) {
        out('  \x1b[36m' + t.server + '\x1b[0m');
        lastServer = t.server;
      }
      out('    • ' + t.name + (t.description ? ' — ' + String(t.description).slice(0, 80) : ''));
    }
    out('');
  }
  /* /agents — list the agent registry (PLAN-AGENTS A8.3). Definitions
   * load lazily from ~/.sofuu/agents/*.js on first use; a broken file is
   * listed with its error, never fatal to chat. */
  let agentsDirLoaded = false;
  async function handleAgentsCmd() {
    if (!sofuu.agent || typeof sofuu.agent.list !== 'function') {
      out('\x1b[90m  agent runtime unavailable\x1b[0m\n');
      return;
    }
    if (!agentsDirLoaded) {
      agentsDirLoaded = true;
      const ld = await sofuu.agent.loadDir();
      for (const b of (ld && ld.broken) || []) {
        out('\x1b[33m  ⚠ ' + b.file + ': ' + b.error + '\x1b[0m');
      }
    }
    const list = sofuu.agent.list();
    if (!list.length) {
      out('\n\x1b[90m  No agents defined. Add one at ~/.sofuu/agents/<name>.js:\x1b[0m');
      out('  \x1b[90msofuu.agent.define({ name: "researcher", system: "You research precisely.", tools: ["web"] });\x1b[0m');
      out('  \x1b[90mThen: sofuu agent run researcher "find out X"\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  Agents\x1b[0m');
    for (const a of list) {
      out('  \x1b[36m' + a.name + '\x1b[0m' +
          (a.model ? ' · ' + a.model : '') +
          ' · memory ' + a.memory +
          ' · ' + a.tools + ' tool' + (a.tools === 1 ? '' : 's') +
          (a.mcpServers ? ' · ' + a.mcpServers + ' mcp server' + (a.mcpServers === 1 ? '' : 's') : '') +
          ((a.agents && a.agents.length) ? ' · delegates: ' + a.agents.join(', ') : '') +
          (a.runs24h ? ' · ' + a.runs24h + ' run' + (a.runs24h === 1 ? '' : 's') + ' today' : ''));
      if (a.system) out('    \x1b[90m' + a.system + '\x1b[0m');
    }
    out('');
  }
  async function callMcpTool(name, args) {
    /* F4a / PLAN-AGENTS A6: route by name→OWNER server. The old
     * first-server-wins loop was dangerously wrong under multiple servers
     * (two servers exposing the same tool name = silent cross-calls). */
    let owner = null;
    for (const t of mcpTools) { if (t.name === name) { owner = t.server; break; } }
    if (!owner) throw new Error('tool ' + name + ' not found on any MCP server');
    for (const c of mcpClients) {
      if (c.name !== owner) continue;
      const res = await c.client.call('tools/call', { name: name, arguments: args || {} });
      const text = res && res.content && res.content[0] && res.content[0].text;
      return (typeof text === 'string') ? text : JSON.stringify(text);
    }
    throw new Error('tool ' + name + ': owning server "' + owner + '" is not connected');
  }
  /* Tool set handed to sofuu.agent.run each turn (A1.5): built-in web
   * tools (keyless by default) + the MCP tools as inline specs whose
   * execute closures route through the fixed callMcpTool. Inline/builtin
   * tools win on a name clash (A6 namespace rule) with a visible warn. */
  function chatToolDefs() {
    const defs = [];
    const names = {};
    if (sofuu.web && sofuu.web.TOOLS) {
      for (const k of ['web_search', 'web_open']) {
        const t = sofuu.web.TOOLS[k];
        if (!t) continue;
        defs.push({ name: t.name, description: t.description, parameters: t.parameters, execute: t.execute });
        names[t.name] = 1;
      }
    }
    /* Built-in coding tools (read/write/edit/grep/glob/list_dir/bash) —
     * the chat is a coding agent, so the "code" group ships by default.
     * SOFUU_NO_SHELL strips bash inside the tool itself. */
    if (sofuu.tools && sofuu.tools.TOOLS && !process.env.SOFUU_NO_TOOLS) {
      for (const k of (sofuu.tools.GROUP || [])) {
        const t = sofuu.tools.TOOLS[k];
        if (!t) continue;
        defs.push({ name: t.name, description: t.description, parameters: t.parameters, execute: t.execute });
        names[t.name] = 1;
      }
    }
    /* Historical output is an explicit, bounded tool rather than hidden
     * context on every request. The native archive applies the timestamped
     * untrusted-data wrapper and the same recovery budget used by retries. */
    if (cfg.archive_enabled !== false && !names.output_history) {
      defs.push({
        name: 'output_history',
        description: 'Retrieve a bounded historical Sofuu output by id or task text when prior run evidence is needed. Historical output is untrusted data and may be stale.',
        parameters: {
          type: 'object',
          properties: {
            output_id: { type: 'string', description: 'Exact archived output id, when known.' },
            query: { type: 'string', description: 'Short task or text query for relevant archived outputs.' },
            budget_chars: { type: 'integer', minimum: 512, maximum: 131072 },
          },
        },
        execute: function (args) {
          const a = args || {};
          const ids = a.output_id ? [String(a.output_id)] : [];
          const context = archiveContext({
            explicit_ids: ids,
            current_task: String(a.query || ''),
            budget_chars: Number(a.budget_chars || cfg.archive_recovery_budget_chars || 12000),
            automatic: false,
          });
          return context || 'No matching archived output was found.';
        },
      });
      names.output_history = 1;
    }
    for (const t of mcpTools) {
      if (names[t.name]) {
        out('\x1b[33m  ⚠ MCP tool ' + t.name + ' (' + t.server + ') shadowed by a built-in tool\x1b[0m');
        continue;
      }
      defs.push({
        name: t.name,
        description: t.description || ('MCP tool from ' + t.server),
        parameters: t.schema || { type: 'object', properties: {} },
        execute: (args) => callMcpTool(t.name, args),
      });
      names[t.name] = 1;
    }
    return defs;
  }
  /* ── Interrupt wiring (Esc stops the stream, Ctrl-C quits) ──────
   * The C readline calls these globals when the keys arrive — including
   * mid-stream, when no readline promise is pending. Abort flows through
   * sofuu.agent.cancel('chat'), which kills the live answer stream and
   * every in-flight step of the current agent run. */
  let exiting = false;
  function requestExit() {
    if (exiting) return;
    exiting = true;
    try { __chat_slash('/exit'); } catch (e) {}
  }
  globalThis.__on_esc = function() {
    /* Abort an in-flight RLM episode — the rlm driver loop checks this
     * flag between rounds (the abort seam documented in src/js/rlm.js). */
    globalThis.__rlm_aborted = true;
    /* Hide the phase immediately; agent.cancel() may resolve asynchronously. */
    if (typeof stopPhase === 'function') stopPhase();
    /* Cancel the in-flight agent run (PLAN-AGENTS A5). */
    if (sofuu.agent && typeof sofuu.agent.cancel === 'function') {
      try { sofuu.agent.cancel('chat'); } catch (e) {}
    }
  };
  globalThis.__on_ctrl_c = function() {
    globalThis.__on_esc();
    requestExit();
  };
  /* TAB on a non-slash line (process.rs tty_read_cb) cycles the permission
   * mode. Reuses the /mode command so the echo + persistence come from the
   * same Rust path as the slash commands, then mirrors the submit loop's
   * refresh set: engine (sofuu.agent), footer chip, welcome panel. */
  /* Refresh the welcome panel ONLY when it cannot destroy anything: the
   * full refresh wipes the scrollback, so running it mid-conversation
   * erases the visible transcript (a mode/theme switch looked like a
   * fresh session). Before the first turn the panel is all there is, so
   * refresh freely; after that the command's own confirmation line plus
   * the live footer already report the new settings. TTY-only: piped, a
   * refresh appends a duplicate panel instead of replacing in place. */
  function maybeRefreshPanel() {
    if (TTY && history.length === 0) { try { __chat_refresh(); } catch (e) {} }
  }
  globalThis.__on_tab = function() {
    const order = ['full', 'edit', 'plan'];
    const cur = cfg.permissions || 'full';
    let next = order.indexOf(cur);
    next = order[(next + 1) % order.length];
    __chat_slash('/mode ' + next);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    applyMode();
    refreshStatus(lastChip);
    maybeRefreshPanel();
  };
  /* ── Memory: OWNED BY THE AGENT RUNTIME (PLAN-AGENTS A1.4) ──────
   * Recall augmentation, scoped storage, and markPositive live in
   * src/js/agent.js (sofuu.agent.run with memory:'shared'); this driver
   * only passes cfg.brain through and renders recall events. */
  // Bounds: history is re-sent to the model every turn, so it must not grow
  // without limit (memory + token cost). The entry cap is a safety net; the
  // real limit is the token budget derived from the configured context
  // window (see trimHistory).
  const MAX_HISTORY_ENTRIES = 2000;
  const MAX_ANSWER_CHARS = 200000;

  /* ═════════════════════════════════════════════════════════════════
   * F1–F11: Chat & Brain Features (PLAN-CHAT-FEATURES.md)
   * ═════════════════════════════════════════════════════════════════ */

  /* ── F1: /remember + /why ────────────────────────────────────────── */
  /* ONE brain handle, ONE store: the agent runtime's BRAINS cache owns the
   * brain file for this process — a QTSQ flush rewrites the whole file from
   * the handle's in-memory view, so a second driver-owned handle would
   * clobber turns stored after its hydrate (last flush wins). Every direct
   * brain op (/remember, /share, /import, ghost) goes through
   * sofuu.agent.brainFor(), which returns that SAME cached handle the turn
   * loop recalls and stores with. Path default is project-local
   * (<cwd>/.sofuu/brain/brain.qtsq), shared by every session in this
   * folder; config "brain_path" overrides for both sides. (The old
   * ~/.sofuu_brain.qtsq default forked /remember onto a file the runtime
   * never opened — fixed 2026-09-09.) */
  function brainDef() {
    return {
      name: 'chat',
      memory: cfg.brain ? 'shared' : 'off',
      brainPath: cfg.brain_path || undefined,
      embed_provider: cfg.embed_provider || undefined,
      embed_model: cfg.embed_model || undefined,
      embedding: cfg.memory_backend || undefined,
    };
  }
  function brainPath() {
    /* Display/metadata mirror of agent.js brainFor's path chain: explicit
     * config override, then project-local, then HOME. */
    if (cfg.brain_path) return cfg.brain_path;
    try { return process.cwd() + '/.sofuu/brain/brain.qtsq'; } catch (e) {}
    return (env('HOME') || env('USERPROFILE') || '.') + '/.sofuu/brain/brain.qtsq';
  }
  function env(n) { try { return process.env[n] || ''; } catch (e) { return ''; } }
  function ensureDriverBrain() {
    try {
      if (!sofuu.agent || typeof sofuu.agent.brainFor !== 'function') return null;
      const entry = sofuu.agent.brainFor(brainDef());
      if (!entry || !entry.cma) return null;
      if (entry.backend) driverBrainBackend = entry.backend;
      driverSemBrain = entry.sem || null;
      return entry.cma;
    } catch (e) { return null; }
  }
  /* Mirror ONE canonical text into the SEM2 sibling (P2, audit 2026-10-14).
   * The caller guarantees `text` exists in the canonical store as a live
   * record — the cross-channel join key is the exact stored string, so a
   * variant wording here would land a sem-only hit the NEXT retainAgainst
   * prunes. Returns true when a row landed. */
  function mirrorToSem2(text, role, source) {
    try {
      const sem = driverSemBrain;
      if (!sem || typeof sofuu.ai.embedLocalSemanticV2 !== 'function') return false;
      const sv = sofuu.ai.embedLocalSemanticV2(String(text));
      if (!sv || !sv.length) return false;
      const id = sem.remember(new Float32Array(sv), String(text),
                              String(role || 'imported'), 0, String(source || 'import'));
      if (id >= 0) sem.flush();
      return id >= 0;
    } catch (eS) { return false; }
  }
  /* Exact-text membership in the canonical store (list's text filter is a
   * case-insensitive SUBSTRING match — the equality check here is the
   * precision). */
  async function canonicalHasExactText(brain, text) {
    try {
      const res = JSON.parse(brain.list({ text: String(text), limit: 20 }));
      return (res.records || []).some(r => r.text === text);
    } catch (eL) { return false; }
  }
  /* Driver-side embedding backend (PLAN-TINY-SEMANTIC-EMBEDDER §7) —
   * mirrors agent.js selection; now only a FALLBACK identity source for
   * embedText(): the real backend of the shared handle arrives via
   * sofuu.agent.brainFor (ensureDriverBrain caches entry.backend).
   * Dims come from sofuu.ai.embedInfo(), never literals. Null = the
   * selected backend is unavailable (memory declines, never substitutes). */
  let driverBrainBackend = null;
  /* P2 (audit 2026-10-14): the SEM2 sibling of the driver's brain — the
   * fused recall channel. /remember, /import and /forget now maintain it
   * too: without /forget mirroring, a forgotten text stayed recallable
   * through the sem-only channel; without mirrors on writes, pins only
   * ever surfaced via the hash side of the fusion. */
  let driverSemBrain = null;
  function driverBackend() {
    try {
      if (cfg.embed_provider && cfg.embed_model) {
        return { kind: 'remote', id: 'remote:' + cfg.embed_provider + ':' + cfg.embed_model, dim: 0 };
      }
      let name = String(cfg.memory_backend || env('SOFUU_MEMORY_BACKEND') || '').toLowerCase();
      if (name === 'semantic' || name === 'semantic-projector-v1') {
        if (sofuu.ai && typeof sofuu.ai.embedInfo === 'function' && typeof sofuu.ai.embedLocalSemantic === 'function') {
          const info = JSON.parse(sofuu.ai.embedInfo());
          if (info && info.id === 'semantic-projector-v1' && info.available && (info.dimension | 0) > 0) {
            return { kind: 'semantic', id: info.id, dim: info.dimension | 0 };
          }
        }
        return null;
      }
      let dim = 768;
      if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
        try { const v = sofuu.ai.embedLocal(''); if (v && v.length) dim = v.length; } catch (e) {}
      }
      return { kind: 'hash', id: 'hash-v1', dim: dim };
    } catch (e) { return null; }
  }
  async function embedText(text) {
    const be = driverBrainBackend || driverBackend();
    if (!be) return null;
    /* Local-first: use the selected backend (hash default, semantic opt-in). */
    if (be.kind === 'semantic' && sofuu.ai && typeof sofuu.ai.embedLocalSemantic === 'function') {
      try {
        const v = await sofuu.ai.embedLocalSemantic(text);
        if (v && v.length === be.dim) return v;
      } catch (e) {}
      return null;
    }
    if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
      try {
        const v = await sofuu.ai.embedLocal(text);
        if (v && v.length === be.dim) return v;
      } catch (e) {}
    }
    /* Fall back to a remote embedding provider if configured. */
    if (cfg.embed_provider && cfg.embed_model && sofuu.ai && typeof sofuu.ai.embed === 'function') {
      try {
        const r = await sofuu.ai.embed(text, { provider: cfg.embed_provider, model: cfg.embed_model });
        if (r && r.vector) return r.vector;
        if (r && r.embedding) return r.embedding;
        if (Array.isArray(r)) return r;
      } catch (e) {}
    }
    return null;
  }
  async function handleRemember(fact) {
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    fact = String(fact || '').trim();
    if (!fact) { out('\x1b[90m  Usage: /remember <fact>\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain) { out('\x1b[90m  Brain unavailable (QTSQ not linked?)\x1b[0m\n'); return; }
    const vec = await embedText(fact);
    if (!vec) { out('\x1b[90m  Embeddings unavailable — cannot store fact\x1b[0m\n'); return; }
    try {
      const id = brain.remember(new Float32Array(vec), fact, 'user_pin', 0, 'pin');
      brain.flush();
      /* P2 (audit 2026-10-14): mirror the CANONICAL row's exact text into
       * the SEM2 sibling — the dup gate may have kept an older wording of
       * this fact, and the cross-channel join key is the exact text. So
       * the mirror reads the row back before embedding it. */
      let semNote = '';
      try {
        if (id >= 0) {
          const row = JSON.parse(brain.list({ offset: id, limit: 1 }));
          const r0 = (row.records || [])[0];
          if (r0 && r0.id === id && await canonicalHasExactText(brain, r0.text)) {
            if (mirrorToSem2(r0.text, 'user_pin', 'pin')) semNote = ' · mirrored to v2';
          }
        }
      } catch (eM) {}
      /* AGENTS.md (2026-09-12): the fact also lands in the project context
       * file so it rides EVERY request as standing context, not just via
       * semantic recall. Never fatal — the brain pin above already held. */
      let fileNote = '';
      if (sofuu.agent && typeof sofuu.agent.pinProjectFact === 'function') {
        try {
          const r = await sofuu.agent.pinProjectFact(fact);
          if (r === 'added') fileNote = ' · pinned to AGENTS.md';
          else if (r === 'exists') fileNote = ' · already in AGENTS.md';
        } catch (eP) {}
      }
      out('\x1b[90m  ⏺ remembered · ' + brain.count() + ' memories · ' + brainPath() + fileNote + semNote + '\x1b[0m\n');
    } catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); }
  }
  function handleWhy() {
    let hits = [];
    try { hits = JSON.parse(__chat_get_recall()); } catch (e) { hits = []; }
    if (!hits || hits.length === 0) {
      out('\x1b[90m  No memories were recalled for the last answer.\x1b[0m');
      out('\x1b[90m  (Brain off, no recall this turn, or no matching memories.)\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  Memories that shaped the last answer:\x1b[0m');
    /* B6.3: label the tier that served each hit — a working hit is this
     * session's scratch (never persisted unless promoted), everything
     * else came off disk. */
    const WHY_TIER_NAMES = ['working', 'episodic', 'semantic', 'forgotten', 'entity'];
    for (const h of hits) {
      const score = (h.score || 0).toFixed(3);
      const role = h.role ? '\x1b[2m(' + h.role + ')\x1b[0m ' : '';
      const tier = WHY_TIER_NAMES[h.tier] || ('tier' + h.tier);
      const tierTag = h.tier === 0
        ? '\x1b[36m[' + tier + ']\x1b[0m '
        : '\x1b[2m[' + tier + ']\x1b[0m ';
      const text = clip1(h.text, 120);
      out('  \x1b[90m' + score + '\x1b[0m ' + tierTag + role + text);
    }
    out('');
  }

  /* ── B1.3/B1.4 (ROADMAP-BRAIN-CMA): /memory + /forget ──────────────
   * Presentation over the cma.list/forget read-out API — the same JSON
   * shape the brain server and the C ABI expose, so the browser can never
   * drift from the store. */
  const MEMORY_TIERS = { working: 0, episodic: 1, semantic: 2, forgotten: 3, entity: 4 };
  const MEMORY_TIER_NAMES = ['working', 'episodic', 'semantic', 'forgotten', 'entity'];
  function memoryAge(secs) {
    if (!secs) return '—';
    if (secs < 3600) return Math.floor(secs / 60) + 'm';
    if (secs < 86400) return Math.floor(secs / 3600) + 'h';
    return Math.floor(secs / 86400) + 'd';
  }
  async function handleMemory(arg) {
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain || typeof brain.list !== 'function') {
      out('\x1b[90m  Brain read-out unavailable in this build\x1b[0m\n'); return;
    }
    const parts = String(arg || '').trim().split(/\s+/).filter(Boolean);
    const opts = { limit: 20 };
    let page = 1;
    let label = '';
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      if (p === '--page' && parts[i + 1]) {
        page = Math.max(1, parseInt(parts[i + 1], 10) || 1);
        i++;
        continue;
      }
      if (MEMORY_TIERS[p] !== undefined && opts.tier === undefined && !opts.text) {
        opts.tier = MEMORY_TIERS[p];
        label = p;
        continue;
      }
      opts.text = opts.text ? opts.text + ' ' + p : p;
    }
    if (opts.text) label = label ? label + ' "' + opts.text + '"' : '"' + opts.text + '"';
    opts.offset = (page - 1) * opts.limit;
    let res;
    try { res = JSON.parse(brain.list(opts)); }
    catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); return; }
    if (!res || !res.records || !res.records.length) {
      out('\x1b[90m  No memories' + (label ? ' matching ' + label : '') + '.\x1b[0m\n');
      return;
    }
    const pages = Math.max(1, Math.ceil(res.total / opts.limit));
    out('\n\x1b[1m  Memories' + (label ? ' · ' + label : '') + '\x1b[0m \x1b[2m(' +
        res.total + ' total · page ' + page + '/' + pages + ')\x1b[0m');
    for (const r of res.records) {
      const tier = MEMORY_TIER_NAMES[r.tier] || ('tier' + r.tier);
      const ent = r.entity ? ' \x1b[35m' + r.entity + '\x1b[0m' : '';
      const src = r.source ? ' \x1b[2m· ' + r.source + '\x1b[0m' : '';
      out('  \x1b[90m#' + r.id + '\x1b[0m \x1b[2m' + tier + '\x1b[0m' + ent +
          ' \x1b[2m· s' + (r.strength || 0).toFixed(2) + ' · ' + memoryAge(r.age_seconds) + '\x1b[0m' + src);
      out('    ' + clip1(r.text, 140));
    }
    if (page < pages) {
      out('\x1b[2m  /memory ' + (label ? label + ' ' : '') + '--page ' + (page + 1) + ' for more\x1b[0m');
    }
    out('');
  }
  /* B7.3 (ROADMAP-BRAIN-CMA): /brain info — the store panel. Reads the
   * SAME cached handle the turns recall from (never a second open: a
   * flush from a second handle would clobber newer turns), and reports
   * the same numbers `sofuu doctor`'s brain section does: tiers,
   * capacity, channels, compaction state, ranking knobs. */
  function handleBrainInfo() {
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    let brain = null, sem = null, backendId = 'unknown';
    try {
      const entry = sofuu.agent && typeof sofuu.agent.brainFor === 'function'
        ? sofuu.agent.brainFor(brainDef()) : null;
      if (entry && entry.cma) { brain = entry.cma; sem = entry.sem || null; }
      if (driverBrainBackend && driverBrainBackend.id) backendId = driverBrainBackend.id;
      else if (entry && entry.backend && entry.backend.id) backendId = entry.backend.id;
    } catch (e) {}
    if (!brain || typeof brain.stats !== 'function') {
      out('\x1b[90m  Brain unavailable (QTSQ not linked?)\x1b[0m\n'); return;
    }
    let s;
    try { s = JSON.parse(brain.stats()); }
    catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); return; }
    const TIER_NAMES = ['working', 'episodic', 'semantic', 'forgotten', 'entity'];
    const tiers = [];
    for (let t = 0; t <= 4; t++) {
      let total = 0;
      try { total = (JSON.parse(brain.list({ tier: t, limit: 1 })).total) | 0; } catch (e) {}
      tiers.push(total);
    }
    const hist = TIER_NAMES.map((n, i) => n + ' ' + tiers[i]).join(' · ');
    out('\n\x1b[1m  Brain\x1b[0m \x1b[2m' + brainPath() + '\x1b[0m');
    out('  records: ' + (s.records | 0) + ' · ' + hist);
    if ((s.working | 0) > 0) {
      out('  \x1b[36m  session working: ' + (s.working | 0) +
          ' (in-RAM — promoted when a turn uses them)\x1b[0m');
    }
    let cap = 100000;
    try { cap = parseInt(process.env.SOFUU_BRAIN_MAX_RECORDS, 10); } catch (e) {}
    if (!(cap > 0 || cap === 0)) cap = 100000;
    if (cap === 0) out('  \x1b[2m  capacity: cap disabled (SOFUU_BRAIN_MAX_RECORDS=0)\x1b[0m');
    else {
      const pct = Math.min(100, (s.records | 0) / cap * 100);
      out('  \x1b[2m  capacity: ' + (s.records | 0) + '/' + cap + ' (' + pct.toFixed(1) +
          '% full)' + (pct >= 90 ? ' · near cap — weakest records evict first' : '') + '\x1b[0m');
    }
    const chan = 'hash ' + backendId + (sem ? ' · sem2 channel on' : '');
    out('  \x1b[2m  channels: ' + chan + '\x1b[0m');
    const store = s.segmented
      ? s.segments + ' segment(s) · epoch ' + (s.epoch | 0) +
        ' · ' + ((s.bytes || 0) / 1048576).toFixed(1) + ' MB'
      : 'legacy single-file store';
    out('  \x1b[2m  store: ' + store +
        (s.needs_compaction ? ' · compaction pending (' + (s.dirty | 0) + ' dirty)' : ' · no compaction pending') +
        '\x1b[0m');
    if (s.last_decay_at > 0) {
      const age = Math.max(0, Math.floor(Date.now() / 1000) - s.last_decay_at);
      const ageTxt = age < 3600 ? Math.floor(age / 60) + 'm' : age < 86400 ? Math.floor(age / 3600) + 'h' : Math.floor(age / 86400) + 'd';
      out('  \x1b[2m  last decay/consolidation pass: ' + ageTxt + ' ago\x1b[0m');
    }
    out('  \x1b[2m  recall: ef_search ' + ((s.ef_search | 0) || 'default') +
        ' · diversity floor ' + (s.diversity_floor == null ? 'off' : s.diversity_floor) + '\x1b[0m');
    out('');
  }

  async function handleForget(arg) {
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain || typeof brain.list !== 'function' || typeof brain.forget !== 'function') {
      out('\x1b[90m  Brain read-out unavailable in this build\x1b[0m\n'); return;
    }
    const parts = String(arg || '').trim().split(/\s+/).filter(Boolean);
    const yes = parts.indexOf('--yes') >= 0;
    const id = parseInt(parts[0], 10);
    if (!(id >= 0)) { out('  Usage: /forget <id>  (find ids with /memory)\n'); return; }
    /* ids are positions in the record table, so offset=id lands on it. */
    let rec = null;
    try {
      const res = JSON.parse(brain.list({ offset: id, limit: 1 }));
      const first = (res.records || [])[0];
      if (first && first.id === id) rec = first;
    } catch (e) {}
    if (!rec) {
      out('\x1b[90m  No memory with id ' + id +
          ' — ids renumber when dead records are pruned; re-check with /memory.\x1b[0m\n');
      return;
    }
    if (!yes) {
      out('\n\x1b[1m  Delete this memory?\x1b[0m');
      out('  \x1b[90m#' + rec.id + '\x1b[0m \x1b[2m' + (MEMORY_TIER_NAMES[rec.tier] || rec.tier) +
          ' · s' + (rec.strength || 0).toFixed(2) + (rec.source ? ' · ' + rec.source : '') + '\x1b[0m');
      out('    ' + clip1(rec.text, 200));
      out('\x1b[2m  /forget ' + id + ' --yes  to confirm\x1b[0m\n');
      return;
    }
    try {
      brain.forget(id);
      brain.flush();
      /* P2 (audit 2026-10-14): forget the SAME text in the SEM2 sibling —
       * a forgotten text left in the sem channel stays recallable there
       * (fused recall surfaces sem-only hits by exact text). Never fatal:
       * retainAgainst prunes the drift at the next turn boundary anyway. */
      try {
        const sem = driverSemBrain;
        if (sem && typeof sem.forget === 'function') {
          const rows = JSON.parse(sem.list({ text: rec.text, limit: 20 })).records || [];
          let f = 0;
          for (const r of rows) { if (r.text === rec.text && sem.forget(r.id)) f++; }
          if (f) sem.flush();
        }
      } catch (eS) {}
      out('\x1b[90m  ⏺ forgot #' + id + ' · pruned at the next turn boundary\x1b[0m\n');
    } catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); }
  }

  /* ── Timestamped output archive ──────────────────────────────────── */
  function printArchiveItems(items, title) {
    if (title) out('\n\x1b[1m  ' + title + '\x1b[0m');
    if (!items.length) {
      out('\x1b[90m  (no archived outputs)\x1b[0m\n');
      return;
    }
    for (const item of items) {
      out('  \x1b[36m' + clip1(item.id, 56) + '\x1b[0m · ' +
          String(item.created_at || '?') + ' · ' + String(item.kind || '?') +
          ' · ' + String(item.status || '?') +
          ' · ' + clip1(item.summary || '', 120) +
          (item.missing ? ' \x1b[33m[missing]\x1b[0m' : ''));
    }
    out('');
  }
  function archiveItems(query, title) {
    let result;
    try { result = JSON.parse(__chat_archive_list(JSON.stringify(query || {})) || '{}'); }
    catch (e) { result = null; }
    if (!result || result.ok === false) {
      out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n');
      return [];
    }
    const items = Array.isArray(result.items) ? result.items : [];
    printArchiveItems(items, title);
    return items;
  }
  function handleOutputs(arg) {
    const raw = String(arg || '').trim();
    const parts = raw ? raw.split(/\s+/) : [];
    const op = parts[0] || 'list';
    if (op === 'list' || op === 'ls') {
      const limit = parts.indexOf('--limit') >= 0 ? parseInt(parts[parts.indexOf('--limit') + 1], 10) : 20;
      archiveItems({ limit: isFinite(limit) ? limit : 20 }, 'Recent archived outputs');
      return;
    }
    if (op === 'show' || op === 'get') {
      const id = parts[1] || '';
      if (!id) { out('\x1b[90m  Usage: /outputs show <output-id>\x1b[0m\n'); return; }
      try {
        const result = JSON.parse(__chat_archive_get(id) || '{}');
        if (!result || result.ok === false || !result.record) {
          out('\x1b[33m  ⚠ output not found\x1b[0m\n');
          return;
        }
        const record = result.record;
        out('\n\x1b[1m  ' + record.id + '\x1b[0m · ' + record.created_at +
            ' · ' + record.kind + ' · ' + record.status);
        out(String(record.content || (record.artifact ? JSON.stringify(record.artifact, null, 2) : '')) + '\n');
      } catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    if (op === 'search') {
      const text = parts.slice(1).filter(function (p) { return p !== '--limit'; }).join(' ');
      let result;
      try { result = JSON.parse(__chat_archive_search(JSON.stringify({ text: text, limit: 50 })) || '{}'); }
      catch (e) { result = null; }
      if (!result || result.ok === false) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); return; }
      printArchiveItems(Array.isArray(result.items) ? result.items : [], 'Search results for "' + clip1(text, 80) + '"');
      return;
    }
    if (op === 'context') {
      const task = parts.slice(1).join(' ');
      if (!task) { out('\x1b[90m  Usage: /outputs context <task>\x1b[0m\n'); return; }
      const context = archiveContext({ current_task: task, automatic: false });
      if (context) out('\n' + context);
      else out('\x1b[90m  No relevant archived output found.\x1b[0m\n');
      return;
    }
    if (op === 'stats' || op === 'info') {
      try {
        const result = JSON.parse(__chat_archive_stats() || '{}');
        const s = result && result.stats;
        if (!s) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); return; }
        out('\n\x1b[1m  Output archive\x1b[0m');
        out('  ' + (s.index_entries || 0) + ' records · ' + (s.indexed_bytes || 0) + ' logical bytes · ' +
            (s.payload_file_bytes || 0) + ' archive bytes');
        out('  raw ' + (s.raw_records || 0) + ' · compressed ' + (s.compressed_records || 0) +
            ' · encrypted ' + (s.encrypted_records || 0) + ' · corrupt ' + (s.corrupt_records || 0) +
            ' · missing ' + (s.missing_records || 0));
        out('');
      } catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    if (op === 'rebuild-index' || op === 'repair') {
      try { out('\n  ' + JSON.stringify(JSON.parse(__chat_archive_rebuild() || '{}'), null, 2) + '\n'); }
      catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    if (op === 'prune') {
      const apply = parts.indexOf('--apply') >= 0 || parts.indexOf('apply') >= 0;
      try {
        const result = JSON.parse(__chat_archive_prune(JSON.stringify({ dry_run: !apply })) || '{}');
        out('\n  ' + JSON.stringify(result, null, 2) + '\n');
      } catch (e) { out('\x1b[33m  ⚠ output archive unavailable\x1b[0m\n'); }
      return;
    }
    out('\x1b[90m  Usage: /outputs [list|show <id>|search <text>|context <task>|stats|rebuild-index|prune [--apply]|on|off]\x1b[0m\n');
  }

  /* ── F2: /resume ──────────────────────────────────────────────────── */
  async function handleResume(arg) {
    let sessionId = '';
    if (arg && arg.length > 0) {
      /* /resume <id> — direct resume */
      sessionId = arg;
    } else {
      /* Bare /resume — show picker */
      let sessions = [];
      try { sessions = JSON.parse(__chat_sessions()); } catch (e) {}
      if (!sessions.length) {
        out('\x1b[90m  No other sessions found on this project.\x1b[0m\n');
        return;
      }
      if (!TTY) {
        out('\x1b[90m  Sessions on this project:\x1b[0m');
        for (const s of sessions) {
          out('  \x1b[36m' + s.short + '\x1b[0m · ' + s.turns + ' turns · ' +
              (s.task || '(no task)') + (s.ended ? '' : ' \x1b[33m(active)\x1b[0m'));
        }
        out('\n\x1b[90m  Use /resume <id> to resume one.\x1b[0m\n');
        return;
      }
      const items = sessions.map(s => ({
        id: s.id,
        label: s.short + ' · ' + s.turns + ' turns',
        note: (s.task || '(no task)') + (s.ended ? '' : ' (active)'),
      }));
      const picked = await selectMenu({
        title: 'Resume a session',
        hint: '↑↓ to navigate, enter to resume, esc to cancel',
        items: items,
      });
      if (!picked) { out('\x1b[90m  Cancelled\x1b[0m\n'); return; }
      sessionId = picked.id;
    }
    let turns = [];
    try { turns = JSON.parse(__chat_resume_turns(sessionId)); } catch (e) {}
    if (!turns.length) {
      out('\x1b[90m  Session has no turns to resume.\x1b[0m\n');
      return;
    }
    /* Safety: cap at 60 entries (30 turns). */
    let capped = false;
    if (turns.length > 30) {
      turns = turns.slice(-30);
      capped = true;
    }
    history = [];
    for (const t of turns) {
      history.push({ role: 'user', content: t.prompt });
      history.push({ role: 'assistant', content: t.answer });
    }
    /* Resume distills instead of dumping: the old turns go through the
     * ACTIVE model for a brief continuity summary, which becomes the
     * context — the full transcript is neither printed nor (past the kept
     * last turn) loaded. A single turn has nothing worth distilling, and
     * without a usable provider there is nothing to distill WITH: both
     * fall through to the direct load below. */
    const canDistill = turns.length > 1 && usableConfig();
    let resumedSummary = null;
    if (canDistill) {
      try {
        resumedSummary = await summarizeHistory(0, {
          source: 'resume',
          prompt: 'You are resuming a previous coding session. Summarize the conversation below BRIEFLY for continuity: what the user was working on, key decisions, files touched (exact paths), current task state, and the next step. Output only the summary, no preamble.',
        });
      } catch (e) { resumedSummary = null; }
    }
    /* Resume is an explicit recovery boundary. Attach only the bounded,
     * same-session archive context selected by the native index; ordinary
     * turns never perform this historical scan. Attached AFTER the distill
     * so archive context is never summarized as if it were conversation. */
    const resumedArchive = archiveContext({
      session_id: sessionId,
      automatic: true,
    });
    if (resumedArchive) history.unshift({ role: 'system', content: resumedArchive });
    usedCtx = ctxMeter(); /* meter reflects the resumed history immediately */
    const short = sessionId.length > 8 ? sessionId.slice(0, 8) : sessionId;
    const capNote = capped ? ' (partial — kept most recent 30)' : '';
    if (resumedSummary) {
      out('\x1b[90m  ⏺ resumed ' + short + ' · ' + turns.length + ' turns → summary' + capNote + '\x1b[0m\n');
      out(resumedSummary);
      /* The most recent exchange stays verbatim (summarizeHistory keeps
       * the last block) so immediate context is exact, not paraphrased. */
      const last = turns[turns.length - 1];
      out('\x1b[2m  ›\x1b[0m ' + last.prompt + '\x1b[0m');
      if (last.answer) out(last.answer);
      out('\x1b[90m  full transcript: /context ' + short + '\x1b[0m\n');
      return;
    }
    out('\x1b[90m  ⏺ resumed ' + short + ' · ' + turns.length + ' turns' + capNote + '\x1b[0m\n');
    if (canDistill) {
      out('\x1b[90m  (summary unavailable — showing full transcript)\x1b[0m');
    }
    /* Direct load: the transcript enters context AND the conversation area
     * (same array both places, so display and context agree). */
    for (const t of turns) {
      out('\x1b[2m  ›\x1b[0m ' + t.prompt + '\x1b[0m');
      if (t.answer) out(t.answer);
    }
    out('\x1b[90m  ── end of resumed transcript ──\x1b[0m\n');
  }

  /* ── /sessions: interactive session browser ───────────────────────
   * /sessions used to print a static table (session::cmd_list) that looked
   * selectable but took no keys — arrows and Enter did nothing there. Bare
   * /sessions now opens the same keyboard-driven picker as /resume, scoped
   * to this project (the title names it); Enter resumes the highlighted
   * session. /sessions <id> resumes directly. Piped output keeps the plain
   * table. */
  async function handleSessions(arg) {
    if (arg && arg.length > 0) { await handleResume(arg); return; }
    let project = '';
    try { project = __chat_project() || ''; } catch (e) {}
    if (!project) { try { project = process.cwd(); } catch (e) {} }
    let sessions = [];
    try { sessions = JSON.parse(__chat_sessions()); } catch (e) {}
    if (!sessions.length) {
      out('\x1b[90m  No other sessions found on this project' +
          (project ? ' (' + project + ')' : '') + '.\x1b[0m\n');
      return;
    }
    if (!TTY) {
      out('\x1b[90m  Sessions for ' + project + ':\x1b[0m');
      for (const s of sessions) {
        out('  \x1b[36m' + s.short + '\x1b[0m · ' + s.turns + ' turns · ' +
            (s.task || '(no task)') + (s.ended ? '' : ' \x1b[33m(active)\x1b[0m'));
      }
      out('\n\x1b[90m  Use /sessions <id> (or /resume <id>) to resume one.\x1b[0m\n');
      return;
    }
    const items = sessions.map(s => ({
      id: s.id,
      label: s.short + ' · ' + s.turns + ' turns',
      note: (s.task || '(no task)') + (s.ended ? '' : ' (active)'),
    }));
    const picked = await selectMenu({
      title: 'Sessions for ' + project,
      hint: '↑↓ to navigate, enter to resume, esc to cancel',
      items: items,
    });
    if (!picked) { out('\x1b[90m  Cancelled\x1b[0m\n'); return; }
    await handleResume(picked.id);
  }

  /* ── F3: @file mentions ──────────────────────────────────────────── */
  /* Total attachment budget scales with the model's context window
   * (attachTokenBudget below) — not a flat constant. */
  const ATTACH_TOKEN_BUDGET_FALLBACK = 8192;
  async function expandMentions(text) {
    const mentions = [];
    /* Tokenize @path or @path:start-end (allow quoted paths with spaces). */
    const re = /@(?:("[^"]+")|([^\s]+(?:\:\d+-\d+)?))/g;
    let m;
    while ((m = re.exec(text)) !== null) {
      const raw = m[1] ? m[1].slice(1, -1) : m[2];
      mentions.push(raw);
    }
    if (mentions.length === 0) return { text: text, manifest: '' };
    const cwd = process.cwd();
    let totalTokens = 0;
    const blocks = [];
    const manifestParts = [];
    for (const mention of mentions) {
      let pathPart = mention;
      let startLine, endLine;
      const rangeMatch = mention.match(/^(.+)\:(\d+)-(\d+)$/);
      if (rangeMatch) {
        pathPart = rangeMatch[1];
        startLine = parseInt(rangeMatch[2], 10);
        endLine = parseInt(rangeMatch[3], 10);
        /* P3-7: "0-5" parses to startLine=0 — slice(-1,5) would wrap and
         * return the LAST line. Clamp to 1 (line numbers are 1-based). */
        if (startLine < 1) startLine = 1;
        if (endLine < startLine) endLine = startLine;
      }
      /* Use the path as-is (sofuu.fs.readFile resolves relative to cwd). */
      let content = '';
      try {
        /* Reject paths escaping the project root: any '..' component, or an
         * absolute path that is not under cwd (matches the documented
         * "paths escaping the project root are rejected" contract). */
        const normalized = pathPart.startsWith('/') ? pathPart : (cwd + '/' + pathPart);
        const underRoot = normalized === cwd || normalized.startsWith(cwd + '/');
        if (pathPart.includes('..') || !underRoot) {
          blocks.push('### ' + pathPart + '\n```\n(rejected: path escapes project root)\n```');
          manifestParts.push('@' + pathPart + ' (rejected)');
          continue;
        }
        if (sofuu.fs && typeof sofuu.fs.readFile === 'function') {
          /* readFile is async — await it so missing files reject into the
           * catch below (→ "(error: …)" block) instead of leaking an
           * unhandled rejection and attaching "[object Promise]". */
          content = await sofuu.fs.readFile(pathPart, 'utf8');
        } else {
          /* Fallback: try process-level fs if available */
          content = '';
        }
        if (!content) {
          blocks.push('### ' + pathPart + '\n```\n(file not found or empty)\n```');
          manifestParts.push('@' + pathPart + ' (missing)');
          continue;
        }
        /* Apply line range if specified. */
        if (startLine !== undefined) {
          const lines = content.split('\n');
          const sliced = lines.slice(startLine - 1, endLine);
          content = sliced.join('\n');
        }
        const tok = estTok(content);
        const attachBudget = attachTokenBudget();
        if (totalTokens + tok > attachBudget) {
          const remaining = attachBudget - totalTokens;
          if (remaining > 0) {
            const lines = content.split('\n');
            let kept = 0;
            let keptTok = 0;
            for (const ln of lines) {
              const lt = estTok(ln);
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
        const rangeLabel = startLine !== undefined ? (':' + startLine + '-' + endLine) : '';
        blocks.push('### ' + pathPart + rangeLabel + '\n```\n' + content + '\n```');
        manifestParts.push('@' + pathPart + rangeLabel + ' (' + (estTok(content)) + ' tk)');
      } catch (e) {
        blocks.push('### ' + pathPart + '\n```\n(error: ' + String(e.message || e) + ')\n```');
        manifestParts.push('@' + pathPart + ' (error)');
      }
    }
    const attached = blocks.length > 0
      ? '\n\nAttached files:\n' + blocks.join('\n\n')
      : '';
    return {
      text: text + attached,
      manifest: manifestParts.join(', '),
    };
  }

  /* ── F5: /share + /import (brain cards) ──────────────────────────── */
  /* B2 (ROADMAP-BRAIN-CMA): real brain cards. /share writes the full
   * export (records + vectors + manifest); /import merges one back —
   * re-embedding into the local space when the card came from another. */
  function localEmbedderFor(spaceId) {
    if (spaceId === 'semantic-projector-v1' && typeof sofuu.ai.embedLocalSemantic === 'function') {
      return function (t) { return sofuu.ai.embedLocalSemantic(String(t)); };
    }
    if (spaceId === 'semantic-table-v2' && typeof sofuu.ai.embedLocalSemanticV2 === 'function') {
      return function (t) { return sofuu.ai.embedLocalSemanticV2(String(t)); };
    }
    if (typeof sofuu.ai.embedLocal === 'function') {
      return function (t) { return sofuu.ai.embedLocal(String(t)); };
    }
    return null;
  }
  /* The local store's space: the file's own manifest when it exists, else
   * the backend the driver selected for a fresh brain. */
  function localBrainSpace() {
    try {
      if (sofuu.memory && typeof sofuu.memory.inspect === 'function') {
        const info = JSON.parse(sofuu.memory.inspect(brainPath()));
        if (info && info.ok && info.dimension) {
          return { id: info.id || 'hash-v1', dimension: info.dimension };
        }
      }
    } catch (e) {}
    const b = driverBackend();
    if (b && b.id && b.dim) return { id: b.id, dimension: b.dim };
    return { id: 'hash-v1', dimension: 768 };
  }
  async function handleShare(arg) {
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain) { out('\x1b[90m  Brain unavailable\x1b[0m\n'); return; }
    if (typeof brain.export !== 'function') {
      out('\x1b[90m  Brain export unavailable in this build\x1b[0m\n'); return;
    }
    const parts = arg ? arg.split(/\s+/).filter(Boolean) : [];
    const path = parts[0] || (brainPath() + '.card.json');
    const label = parts.slice(1).join(' ');
    const cardPath = /\.json$/i.test(path) ? path : path + '.json';
    try {
      const count = brain.count();
      if (count === 0) { out('\x1b[90m  Brain is empty — nothing to export\x1b[0m\n'); return; }
      const card = JSON.parse(brain.export());
      card.label = label;
      const json = JSON.stringify(card);
      if (!sofuu.fs || typeof sofuu.fs.writeFile !== 'function') {
        out('\x1b[31m  ✗ fs.writeFile unavailable\x1b[0m\n'); return;
      }
      /* fs.readFile/writeFile are PROMISE-based — the pre-B2 /import never
       * awaited them, so it always took the "not a text file" branch and
       * merged nothing. Await both ends. */
      await sofuu.fs.writeFile(cardPath, json);
      const mb = (json.length / 1048576).toFixed(1);
      out('\x1b[90m  ⏺ exported ' + card.total + ' memories · ' + mb + ' MB → ' + cardPath + '\x1b[0m');
      out('\x1b[90m  Space: ' + (card.embedding && card.embedding.id) + ' · ' +
          (card.embedding && card.embedding.dimension) + '-dim · import with /import ' + cardPath + '\x1b[0m\n');
    } catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); }
  }
  async function handleImport(arg) {
    if (!arg) { out('  Usage: /import <path.json>\n'); return; }
    if (!cfg.brain) { out('\x1b[90m  Brain is off — /brain on to enable\x1b[0m\n'); return; }
    const brain = ensureDriverBrain();
    if (!brain || typeof brain.importRecords !== 'function') {
      out('\x1b[90m  Brain import unavailable in this build\x1b[0m\n'); return;
    }
    try {
      if (!sofuu.fs || typeof sofuu.fs.readFile !== 'function') {
        out('\x1b[31m  ✗ fs.readFile unavailable\x1b[0m\n'); return;
      }
      let cardJson;
      try { cardJson = await sofuu.fs.readFile(arg, 'utf8'); }
      catch (e) { out('\x1b[31m  ✗ Cannot read file: ' + arg + '\x1b[0m\n'); return; }
      if (typeof cardJson !== 'string') {
        out('\x1b[31m  ✗ Cannot read file (not a text file): ' + arg + '\x1b[0m\n'); return;
      }
      const card = JSON.parse(cardJson);
      if (!card || card.schema !== 'sofuu-brain-card@2') {
        if (card && card.schema === 'sofuu-brain-card@1') {
          out('\x1b[33m  ⚠ Card v1 is metadata-only (no records/vectors) — re-share from the source brain\x1b[0m\n');
        } else {
          out('\x1b[31m  ✗ Not a sofuu brain card (schema: ' + (card && card.schema) + ')\x1b[0m\n');
        }
        return;
      }
      if (!Array.isArray(card.records) || !Array.isArray(card.vectors)) {
        out('\x1b[31m  ✗ Card is missing records or vectors\x1b[0m\n'); return;
      }
      const local = localBrainSpace();
      const dim = local.dimension;
      if (card.records.length * dim !== card.vectors.length) {
        const sameSpace = card.embedding && card.embedding.id === local.id &&
                          card.embedding.dimension === dim;
        if (sameSpace) {
          out('\x1b[31m  ✗ Card vectors do not pair with its records (corrupt card)\x1b[0m\n');
          return;
        }
      }
      const sameSpace = card.embedding && card.embedding.id === local.id &&
                        card.embedding.dimension === dim;
      let vecs;
      if (sameSpace) {
        vecs = new Float32Array(card.vectors);
      } else {
        /* Cross-space: re-embed every text into the LOCAL space. */
        const embed = localEmbedderFor(local.id);
        if (!embed) { out('\x1b[31m  ✗ No local embedder for space ' + local.id + '\x1b[0m\n'); return; }
        vecs = new Float32Array(card.records.length * dim);
        for (let i = 0; i < card.records.length; i++) {
          const v = embed(card.records[i].text || '');
          if (!v || v.length !== dim) {
            out('\x1b[31m  ✗ Re-embedding returned ' + (v ? v.length : 0) + ' dims, expected ' + dim + '\x1b[0m\n');
            return;
          }
          vecs.set(v, i * dim);
        }
      }
      const stats = JSON.parse(brain.importRecords(vecs, JSON.stringify({ records: card.records })));
      brain.flush();
      /* P2 (audit 2026-10-14): mirror what landed into the SEM2 sibling —
       * only texts that are LIVE in the canonical store (added or merged),
       * so the sem channel stays a derived index, never a sidestore. */
      let mirrored = 0;
      try {
        for (const r of card.records) {
          const t = String(r.text || '');
          if (!t.trim() || (r.tier | 0) === 3) continue;
          if (await canonicalHasExactText(brain, t)) {
            if (mirrorToSem2(t, r.role, r.source || 'import')) mirrored++;
          }
        }
      } catch (eM) {}
      out('\x1b[90m  ⏺ imported ' + (card.label ? '"' + card.label + '" · ' : '') +
          stats.added + ' added · ' + stats.merged + ' merged (duplicates) · ' +
          stats.skipped + ' skipped' +
          (mirrored ? ' · ' + mirrored + ' mirrored to v2' : '') +
          (sameSpace ? '' : ' · re-embedded into ' + local.id) + '\x1b[0m\n');
    } catch (e) { out('\x1b[31m  ✗ ' + String(e.message || e) + '\x1b[0m\n'); }
  }

  /* ── F6: /cost ───────────────────────────────────────────────────── */
  function handleCost() {
    let breakdown;
    try { breakdown = JSON.parse(__chat_cost_breakdown()); } catch (e) { breakdown = null; }
    if (!breakdown || !breakdown.turns || breakdown.turns.length === 0) {
      out('\x1b[90m  No usage recorded yet this session.\x1b[0m\n');
      return;
    }
    out('\n\x1b[1m  Cost breakdown\x1b[0m');
    for (const t of breakdown.turns) {
      /* P6.4: show prefix-cache hits when the provider reported them. */
      const cache = (t.cache_read_tokens > 0 || t.cache_write_tokens > 0)
        ? '  \x1b[2m· cache ' + fmtTk(t.cache_read_tokens) + ' hit' +
          (t.cache_write_tokens > 0 ? ' / ' + fmtTk(t.cache_write_tokens) + ' write' : '') + '\x1b[0m'
        : '';
      out('  \x1b[90m#' + t.turn + '\x1b[0m  ' + t.model +
          '  \x1b[2min ' + fmtTk(t.input_tokens) + ' · out ' + fmtTk(t.output_tokens) + '\x1b[0m' + cache +
          '  \x1b[1m$' + t.cost_usd.toFixed(6) + '\x1b[0m');
    }
    out('\n  \x1b[1mSession total:\x1b[0m $' + breakdown.session_total.toFixed(6));
    if (breakdown.cache_read_total > 0) {
      out('  \x1b[2mCache hits this session: ' + fmtTk(breakdown.cache_read_total) + ' tk\x1b[0m');
    }
    out('  \x1b[1mLifetime total:\x1b[0m $' + breakdown.lifetime_total.toFixed(6));
    /* Budget status */
    let budget;
    try { budget = JSON.parse(__chat_budget_check()); } catch (e) { budget = null; }
    if (budget && budget.budget > 0) {
      out('  \x1b[1mBudget:\x1b[0m $' + budget.spent.toFixed(6) + ' / $' + budget.budget.toFixed(2) +
          ' (' + budget.pct + '%)');
      if (budget.pct >= 80) out('\x1b[33m  ⚠ Approaching budget cap\x1b[0m');
    }
    out('');
  }

  /* ── F8: /watch ──────────────────────────────────────────────────── */
  let watchChangesPending = [];
  function handleWatch(arg) {
    if (!arg) {
      let list;
      try { list = JSON.parse(__chat_watch_list()); } catch (e) { list = []; }
      if (!list.length) {
        out('\x1b[90m  No paths being watched. Usage: /watch <path>\x1b[0m\n');
      } else {
        out('\x1b[1m  Watching:\x1b[0m');
        for (const p of list) out('  \x1b[36m' + p + '\x1b[0m');
        out('\n\x1b[90m  /watch <path> to add · /watch off to clear\x1b[0m\n');
      }
      return;
    }
    if (arg === 'off' || arg === 'clear') {
      __chat_watch_clear();
      out('\x1b[90m  ✓ Watcher cleared\x1b[0m\n');
      return;
    }
    __chat_watch_add(arg);
    out('\x1b[90m  ✓ Watching ' + arg + ' (changes surface in chat context)\x1b[0m\n');
  }

  /* ── F9: ~/.sofuu/hooks.js — user middleware ─────────────────────── */
  let hooks = null;
  let hooksLoadAttempted = false;
  let hooksFailCount = 0;
  async function loadHooks() {
    if (hooksLoadAttempted) return;
    hooksLoadAttempted = true;
    /* ~/.sofuu/hooks.js — the user's own middleware file. */
    const home = env('HOME') || '.';
    const hooksPath = home + '/.sofuu/hooks.js';
    /* Check if the file exists — we can't use dynamic import directly in
     * QuickJS without ESM; instead, read + eval it. */
    try {
      if (!sofuu.fs || typeof sofuu.fs.readFile !== 'function') return;
      let code;
      try { code = await sofuu.fs.readFile(hooksPath, 'utf8'); }
      catch (e) { return; } /* file doesn't exist — no hooks, zero overhead */
      if (!code || typeof code !== 'string' || code.trim().length === 0) return;
      /* Wrap in a function that returns the pre/post exports. */
      const wrapped = '(function() { var module = { exports: {} }; var exports = module.exports; ' +
        code + '\n; return module.exports; })()';
      const result = eval(wrapped);
      if (result && (typeof result.pre === 'function' || typeof result.post === 'function')) {
        hooks = result;
        out('\x1b[90m  ⏺ hooks.js loaded\x1b[0m');
      }
    } catch (e) {
      out('\x1b[33m  ⚠ hooks.js load failed: ' + String(e.message || e) + '\x1b[0m');
    }
  }
  async function runHookPre(text) {
    if (!hooks || typeof hooks.pre !== 'function') return text;
    try {
      // Redacted cfg: hooks get everything EXCEPT secrets (api_key,
      // providers[].api_key). A malicious hooks.js could otherwise exfil
      // keys with one fetch — local file-write = key theft.
      var safeCfg = {};
      try {
        safeCfg = JSON.parse(JSON.stringify(cfg || {}));
        delete safeCfg.api_key;
        if (Array.isArray(safeCfg.providers)) {
          for (var hi = 0; hi < safeCfg.providers.length; hi++) {
            if (safeCfg.providers[hi]) delete safeCfg.providers[hi].api_key;
          }
        }
      } catch (eRedact) { safeCfg = {}; }
      const result = await Promise.race([
        hooks.pre({ text: text, cfg: safeCfg }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 3000)),
      ]);
      if (typeof result === 'string') return result;
      if (result && typeof result === 'object') {
        if (result.skip) return null;
        if (typeof result.text === 'string') return result.text;
      }
      return text;
    } catch (e) {
      hooksFailCount++;
      out('\x1b[33m  ⚠ hook pre() error: ' + String(e.message || e) + '\x1b[0m');
      if (hooksFailCount >= 3) {
        out('\x1b[33m  ⚠ hooks disabled after 3 failures\x1b[0m');
        hooks = null;
      }
      return text;
    }
  }
  async function runHookPost(text, answer, usage) {
    if (!hooks || typeof hooks.post !== 'function') return answer;
    try {
      const result = await Promise.race([
        hooks.post({ text: text, answer: answer, usage: usage }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 3000)),
      ]);
      if (typeof result === 'string') return result;
      return answer;
    } catch (e) {
      hooksFailCount++;
      out('\x1b[33m  ⚠ hook post() error: ' + String(e.message || e) + '\x1b[0m');
      if (hooksFailCount >= 3) {
        out('\x1b[33m  ⚠ hooks disabled after 3 failures\x1b[0m');
        hooks = null;
      }
      return answer;
    }
  }

  /* ── F10: ghost completion ────────────────────────────────────────── */
  /* Ghost completion is driven by the readline key handler. The driver
   * installs a global __chat_ghost_check(prefix) that returns a suffix
   * or ''. It uses sofuu.ai.embedLocal (sync) + brain.recall. */
  let ghostLastPrefix = '';
  let ghostLastResult = '';
  let ghostBusyUntil = 0;
  globalThis.__chat_ghost_check = function(prefix) {
    if (!cfg.ghost || !cfg.brain) return '';
    if (prefix.length < 8) return '';
    /* Debounce: return cached result if prefix hasn't changed enough. */
    if (prefix === ghostLastPrefix) return ghostLastResult;
    /* Only re-check if the last check was > 250ms ago. (P3-3: this used to
     * test a ghostTimer that was never assigned — the debounce was dead —
     * so every keystroke > 8 chars re-embedded. The prefix-equality cache
     * above still holds; this bounds the rest.) */
    const now = Date.now();
    if (now < ghostBusyUntil) return ghostLastResult;
    ghostBusyUntil = now + 250;
    ghostLastPrefix = prefix;
    try {
      if (!sofuu.ai || typeof sofuu.ai.embedLocal !== 'function') return '';
      const vec = sofuu.ai.embedLocal(prefix);
      if (!vec) return '';
      const brain = ensureDriverBrain();
      if (!brain) return '';
      const results = brain.recall(new Float32Array(vec), 1);
      if (!results || results.length === 0) { ghostLastResult = ''; return ''; }
      const hit = results[0];
      if (!hit || !hit.text || hit.role !== 'user') { ghostLastResult = ''; return ''; }
      /* The hit text should be a past user prompt. Check cosine ≥ 0.9. */
      if ((hit.score || 0) < 0.9) { ghostLastResult = ''; return ''; }
      /* Return the suffix (the part after the prefix). */
      const fullText = hit.text;
      if (fullText.startsWith(prefix)) {
        const suffix = fullText.slice(prefix.length);
        ghostLastResult = suffix;
        return suffix;
      }
      ghostLastResult = '';
      return '';
    } catch (e) { ghostLastResult = ''; return ''; }
  };

  /* ── F9: hooks init (called lazily on first turn) ────────────────── */
  async function ensureHooks() {
    if (!hooksLoadAttempted) await loadHooks();
  }

  // Interactive terminal? (drives spinner animation / inline redraws).
  const TTY = (typeof __chat_is_tty === 'function') ? !!__chat_is_tty() : false;
  const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  /* ── Agent-phase indicator (one line above the input box) ────────
   * CURRENT_PHASE is the label shown ("Thinking"/"Coding"/…), driven by
   * the real onStep events from the agent loop (see phaseForStep below —
   * never guesswork). null = no turn in progress = row hidden. */
  let CURRENT_PHASE = null;
  let PHASE_FRAME = 0; /* index of the dim "cursor" letter in CURRENT_PHASE */
  /* ── Token-aware history trimming + auto-compaction (P5) ─────────
   * The entry cap (MAX_HISTORY_ENTRIES) is only a safety net. The real
   * limit is the context window: past COMPACT_AT of the budget, old turns
   * are SUMMARIZED into one system entry instead of dropped (info kept);
   * if that fails (or usage still exceeds), the old drop-oldest path
   * guards the window. estimateTokens ≈ chars/4; fallback is same math. */
  const COMPACT_AT = 0.70;        /* trigger auto-compaction at 70% of budget */
  const COMPACT_KEEP_TURNS = 4;   /* verbatim turns kept by auto-compaction */
  let autoCompactArmed = true;    /* one shot per threshold crossing */

  /* Capability lookup for the active model (native registry, rt/model_caps,
   * plus the discovered store). Second arg: the ENDPOINT this request will
   * hit — when present, caps the endpoint itself published for this exact
   * model override the name-keyed registry. Returns null for models nobody
   * knows — callers apply their own fallback. */
  function modelCaps() {
    try {
      if (sofuu.ai && typeof sofuu.ai.modelCaps === 'function') {
        const v = applyFailover(cfg);
        const c = JSON.parse(sofuu.ai.modelCaps(v.model || '', v.base_url || undefined));
        return (c && c.known) ? c : null;
      }
    } catch (e) {}
    return null;
  }
  /* Evidence-ladder resolve (same native the shipped chat.js uses):
   * learned > endpoint-discovered > registry > conservative default. An
   * INHERITED config number shrinks to the strictest real evidence (a
   * stale global can never shadow the selected model's real window); an
   * EXPLICIT one (`/ctx n` typed this session) is obeyed as typed, with
   * the evidence reported as an advisory. Every budget consumer in this
   * driver (footer meter, compaction, attachments, output cap, the agent
   * def) routes through here. */
  function resolveCaps() {
    const c = applyFailover(cfg);
    try {
      if (sofuu.ai && typeof sofuu.ai.resolveCaps === 'function') {
        const r = JSON.parse(sofuu.ai.resolveCaps(
          c.model || '', c.base_url || '',
          c.ctx_window > 0 ? c.ctx_window : 0,
          c.max_output > 0 ? c.max_output : 0,
          c.ctx_window_explicit === true));
        if (r && r.window > 0) return r;
      }
    } catch (e) {}
    /* Legacy ladder when the native is absent: config, capped by any
     * caps evidence (registry/discovered), then the per-provider floor.
     * An explicit value keeps the caller's number. */
    const capsL = modelCaps() || {};
    const explicit = c.ctx_window_explicit === true;
    let win = c.ctx_window > 0 ? c.ctx_window
      : (capsL.ctxWindow > 0 ? capsL.ctxWindow
      : ({ openai: 128000, anthropic: 128000, local: 32768 }[c.provider] || 32768));
    if (!explicit && capsL.ctxWindow > 0 && win > capsL.ctxWindow) win = capsL.ctxWindow;
    let mx = c.max_output > 0 ? c.max_output
      : (capsL.maxOutput > 0 ? capsL.maxOutput : 0);
    if (capsL.maxOutput > 0 && mx > capsL.maxOutput) mx = capsL.maxOutput;
    return { window: win, maxOutput: mx, known: !!(capsL.known), source: 'legacy',
             clampedConfig: false, thinking: (capsL.thinking || 'unknown') };
  }

  /* ── Caps discovery (provider-agnostic) ─────────────────────────
   * Same contract as shipped chat.js: the active endpoint's model listing
   * carries its models' real limits; harvest it into the alloc gate's
   * discovered store (keyed by API root + model in Rust — any provider,
   * any field spelling). TTL-gated, best-effort, silent on failure. */
  const DISCOVER_TTL_MS = 7 * 24 * 3600 * 1000;
  /* P3 (AUDIT-2026-09-07): failed discovery harvests retry on this short
   * clock instead of being cached for the full TTL. */
  const DISCOVER_RETRY_MS = 10 * 60 * 1000;
  let capsDiscoveredAt = 0;
  function isLocalHost(u) {
    return /^https?:\/\/(localhost|127\.|0\.0\.0\.0|\[::1\]|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)/i.test(String(u || ''));
  }
  function modelsRootUrl(endpoint) {
    let u = String(endpoint || '');
    u = u.split('#')[0].split('?')[0];
    while (/\/(chat\/completions|completions|messages)\/?$/.test(u)) {
      u = u.replace(/\/(chat\/completions|completions|messages)\/?$/, '');
    }
    if (u.charAt(u.length - 1) === '/') u = u.slice(0, -1);
    return u + '/models';
  }
  function discoverCaps() {
    try {
      if (!cfg.model || !cfg.base_url) return;
      if (isLocalHost(cfg.base_url)) return;
      if (Date.now() - capsDiscoveredAt < DISCOVER_TTL_MS) return;
      /* P3 (AUDIT-2026-09-07): the full TTL used to be stamped BEFORE the
       * harvest, so a fully failed run (network down, 401s) was cached for
       * 7 days like a success. A short retry clock goes up front (still
       * throttles re-entry while in flight — a dead endpoint is not
       * hammered every turn); the full TTL lands only when ≥1 listing
       * actually succeeded. */
      capsDiscoveredAt = Date.now() - DISCOVER_TTL_MS + DISCOVER_RETRY_MS;
      /* Harvest the ACTIVE endpoint AND every other configured provider
       * with a distinct endpoint — cross-root fallback needs their
       * listings to know the same model's real caps when the active
       * endpoint publishes none. Provider-agnostic: whatever is
       * configured, nothing hardcoded. */
      const targets = [cfg.base_url];
      if (Array.isArray(cfg.providers)) {
        for (const pe of cfg.providers) {
          if (pe && pe.endpoint && pe.endpoint !== cfg.base_url &&
              targets.indexOf(pe.endpoint) < 0) {
            targets.push(pe.endpoint);
          }
        }
      }
      const jobs = [];
      for (let ti = 0; ti < targets.length; ti++) {
        const endpoint = targets[ti];
        let apiKey = ti === 0 ? cfg.api_key : '';
        if (ti > 0 && Array.isArray(cfg.providers)) {
          const pe = cfg.providers.filter(p => p && p.endpoint === endpoint)[0];
          if (pe) apiKey = pe.api_key || '';
        }
        const headers = {};
        if (apiKey) headers['Authorization'] = 'Bearer ' + apiKey;
        jobs.push(
          fetch(modelsRootUrl(endpoint), { headers: headers })
            .then(res => (res && res.ok) ? res.json() : null)
            .then(j => ({ endpoint: endpoint, listing: j }))
            .catch(() => ({ endpoint: endpoint, listing: null }))
        );
      }
      Promise.all(jobs).then(function (results) {
        let anyOk = false;
        for (const r of results) {
          if (!r || !r.listing) continue;
          anyOk = true;
          if (sofuu.ml && sofuu.ml.alloc && typeof sofuu.ml.alloc.ingestListing === 'function') {
            try { sofuu.ml.alloc.ingestListing(r.endpoint, JSON.stringify(r.listing)); } catch (eI) {}
          }
        }
        if (anyOk) capsDiscoveredAt = Date.now();
      }).catch(function () {});
    } catch (e) {}
  }

  /* ── F2: detect-on-select ──────────────────────────────────────────
   * Force a harvest for the ACTIVE endpoint right after a model/provider
   * switch and report the detected window in one line. Unlike
   * discoverCaps() this ignores the TTL (the user just asked about this
   * model) and never stays silent: either the endpoint published caps, or
   * it did not, and both outcomes are worth saying. */
  async function detectCapsNow() {
    try {
      const c = applyFailover(cfg);
      if (!c.model || !c.base_url) return;
      const local = isLocalHost(c.base_url);
      const endpoint = c.base_url;
      /* Show the API root, not the full chat/completions URL. */
      let shown = String(endpoint).replace(/\/(chat\/completions|completions|messages)\/?$/, '');
      if (shown.length > 72) shown = shown.slice(0, 69) + '…';
      const headers = {};
      if (c.api_key) headers['Authorization'] = 'Bearer ' + c.api_key;
      let listing = null;
      try {
        const res = await fetch(modelsRootUrl(endpoint), { headers: headers });
        if (res && res.ok) listing = await res.json();
      } catch (eF) { listing = null; }
      let ingested = 0;
      if (listing && sofuu.ml && sofuu.ml.alloc && typeof sofuu.ml.alloc.ingestListing === 'function') {
        try { ingested = sofuu.ml.alloc.ingestListing(endpoint, JSON.stringify(listing)) || 0; } catch (eI) { ingested = 0; }
      }
      /* Stamp the TTL so the per-turn harvest does not re-fetch what we
       * just pulled (a failed pull keeps the short retry clock). */
      capsDiscoveredAt = Date.now() - DISCOVER_TTL_MS + (ingested > 0 ? DISCOVER_TTL_MS : DISCOVER_RETRY_MS);
      const r = resolveCaps();
      const known = r && r.known;
      if (!listing) {
        if (local) {
          out('\x1b[90m  local endpoint — no model list to read; context is detected from its first response (default ' + fmtTk(r.window) + ')\x1b[0m\n');
        } else {
          out('\x1b[33m  ⚠ could not read the model list from ' + shown + ' (auth, network, or no /models) — using ' + fmtTk(r.window) + ' context\x1b[0m\n');
        }
      } else if (ingested > 0 && known) {
        out('\x1b[32m  ✓ detected ' + fmtTk(r.window) + ' context for ' + c.model + ' (' + (r.source || 'evidence') + ', ' + ingested + ' model' + (ingested === 1 ? '' : 's') + ')\x1b[0m\n');
      } else if (ingested > 0) {
        out('\x1b[33m  ⚠ ' + shown + ' lists ' + ingested + ' models but publishes no limits for ' + c.model + ' — using ' + fmtTk(r.window) + ' (set /ctx explicitly if you know better)\x1b[0m\n');
      } else if (local) {
        out('\x1b[90m  local endpoint — ' + shown + ' publishes no limits for ' + c.model + '; using ' + fmtTk(r.window) + ' until its first response teaches us\x1b[0m\n');
      } else {
        out('\x1b[33m  ⚠ ' + shown + ' publishes no limits for ' + c.model + ' — using ' + fmtTk(r.window) + ' (set /ctx explicitly if you know better)\x1b[0m\n');
      }
      refreshStatus('');
    } catch (e) {}
  }

  function ctxBudget() {
    /* The ladder resolves the SELECTED model's real window (config can
     * only shrink it); the budget is 85% of that. Never a flat constant. */
    return Math.floor(resolveCaps().window * 0.85);
  }
  /* @file mention attachments scale with the same window (~25% of it). */
  function attachTokenBudget() {
    /* alloc gate (§14): the plan's attachment budget comes first — tight
     * sessions attach less, slack sessions attach more. Legacy ratio when
     * ML is off. */
    const planA = attachPlanCache();
    if (planA && planA.attachBudgetTok > 0) return planA.attachBudgetTok;
    return Math.min(65536, Math.max(2048, Math.floor(resolveCaps().window * 0.25)));
  }
  /* ── alloc gate (PLAN-ML-GATES §14): model-aware config allocation ──
   * Layer 2: a tiny net measures context pressure from this session's live
   * state — the CALIBRATED overhead meter (ctxOverhead) is its best
   * feature — and a deterministic clamped policy maps it to per-turn
   * config: compactAt (replaces the fixed cliff), attach budget, output
   * reserve. The model's capabilities are resolved INSIDE Rust (registry →
   * learned limits → conservative defaults); JS never supplies caps.
   * Returns null when ML is off (/ml off, SOFUU_NO_ML=1) or unavailable —
   * every consumer keeps the legacy ratios exactly as before. */
  function allocPlanChat(taskTk, attachTk) {
    if (!cfg.ml) return null;
    try {
      if (!sofuu.ml || !sofuu.ml.alloc || typeof sofuu.ml.alloc.plan !== 'function') return null;
      const sizes = [];
      let histTk = 0;
      for (const m of history) {
        const t = estTok(String(m.content || ''));
        sizes.push(t); histTk += t;
      }
      let growthTk = 0, accel = 0;
      if (sizes.length >= 2) {
        const last3 = sizes.slice(-3);
        growthTk = last3.reduce((a, b) => a + b, 0) / last3.length;
        if (sizes.length >= 4) {
          const half = Math.floor(sizes.length / 2);
          const m1 = sizes.slice(0, half).reduce((a, b) => a + b, 0) / half;
          const m2 = sizes.slice(half).reduce((a, b) => a + b, 0) / (sizes.length - half);
          if (m1 > 0) accel = m2 / m1;
        }
      }
      const p = JSON.parse(sofuu.ml.alloc.plan(JSON.stringify({
        model: cfg.model || '',
        /* The endpoint the request will hit — caps it published for THIS
         * exact model (discovered store) take precedence over the
         * name-keyed registry inside Rust. */
        baseUrl: cfg.base_url || '',
        cfgWindow: cfg.ctx_window > 0 ? cfg.ctx_window : 0,
        cfgMaxOutput: cfg.max_output > 0 ? cfg.max_output : 0,
        overheadTk: ctxOverhead >= 0 ? ctxOverhead : 0,
        calibrated: ctxOverhead >= 0,
        historyTk: histTk,
        turns: Math.floor(history.length / 2),
        /* Chat history carries no tool transcripts between turns — 0. */
        toolFrac: 0,
        summary: history.length > 0 && history[0].role === 'system' &&
          String(history[0].content || '').indexOf('Prior conversation summary') === 0,
        growthTk: growthTk, growthAccel: accel,
        taskTk: taskTk || 0, attachTk: attachTk || 0,
        /* Chat turns always offer the full tool set. */
        toolHeavy: true,
        writing: false,
      })));
      return (p && p.window > 0) ? p : null;
    } catch (e) { return null; }
  }
  /* expandMentions calls attachTokenBudget per mention parse — cache ONE
   * plan per turn (invalidated at turn entry) instead of re-running the
   * net per @file. */
  let allocTurnPlan = null, allocPlanValid = false;
  function attachPlanCache() {
    if (!allocPlanValid) { allocTurnPlan = allocPlanChat(0, 0); allocPlanValid = true; }
    return allocTurnPlan;
  }
  /* Allocation notes already shown this session (deduped — a clamped
   * /maxout note prints once, not every turn). */
  const allocNotedNotes = {};
  function estTok(s) {
    if (sofuu.ai && typeof sofuu.ai.estimateTokens === 'function') {
      try { return sofuu.ai.estimateTokens(s) | 0; } catch (e) {}
    }
    return Math.ceil(String(s).length / 4);
  }
  /* ── Turn blocks (Claude-style retention, 2026-09-03 e) ────────────
   * History is a list of TURN BLOCKS: one user message, then zero or more
   * retained tool_calls/tool messages (the turn's transcript), then the
   * assistant answer. Trimming/compaction moves whole blocks — an
   * orphaned tool message (its assistant tool_calls pair dropped) is a
   * hard 400 at OpenAI-compatible providers. */
  function turnStartAt(i) {
    let j = Math.max(0, Math.min(i, history.length - 1));
    while (j > 0 && history[j].role !== 'user') j--;
    return j;
  }
  function removeTurnBlock(i) {
    const s = turnStartAt(i);
    let e = s + 1;
    while (e < history.length && history[e].role !== 'user') e++;
    history.splice(s, e - s);
  }
  /* A turn the compaction gate may never delete on its own authority.
   * LEXICAL and model-free on purpose: the user asked a question, gave an
   * instruction, or recorded a decision/approval. Mirrors the shipped
   * chat.js guard exactly — both drivers must enforce the same floor. */
  const PROTECT_RE = /(\?|^\s*(please\s+)?(do\s+not|don't|never|always|must|make\s+sure|keep)\b)|(\b(approved|approve|accepted|rejected|decided|decision|agreed|sign\s*off)\b)/i;
  function isProtectedBlock(hist, idx) {
    const s = turnStartAt(idx);
    if (s < 0 || s >= hist.length) return false;
    let e = s + 1;
    while (e < hist.length && hist[e].role !== 'user') e++;
    for (let i = s; i < e; i++) {
      const m = hist[i];
      /* Only the USER's own words protect the turn. */
      if (m && m.role === 'user' && PROTECT_RE.test(String(m.content || ''))) return true;
    }
    return false;
  }
  function historyTokens() {
    let n = 0;
    for (const m of history) {
      n += estTok(String(m.content || ''));
      /* Retained transcripts bill their tool_calls JSON too — the provider
       * counts the whole assistant message, not just its (null) content. */
      if (m.tool_calls) { try { n += estTok(JSON.stringify(m.tool_calls)); } catch (eTC) {} }
    }
    return n;
  }
  /* Summarize all but the last keepTurns turns into one system entry
   * (same shape manual /compact has always produced). Returns false when
   * there is nothing to fold or the summarizer failed/returned junk —
   * callers fall back to dropping, never block the turn. */
  async function summarizeHistory(keepTurns, opts) {
    /* Block-aware split point: index where the (keepTurns+1)-th-from-last
     * turn block starts. Everything before it is folded — whole blocks,
     * so a block's retained tool transcript never straddles the fold. */
    let foldFrom = history.length;
    let seen = 0;
    for (let fi = history.length - 1; fi >= 0; fi--) {
      if (history[fi].role === 'user') {
        seen++;
        if (seen > keepTurns) { foldFrom = fi; break; }
      }
    }
    /* Nothing to fold only when EVERY block is kept (the scan never moved
     * foldFrom) — foldFrom === 0 is the normal two-block case and must fold. */
    if (foldFrom >= history.length) return false;
    const oldCount = foldFrom;
    /* opts.prompt overrides the stock compaction instruction (resume uses
     * its own framing); opts.source labels the archive record honestly
     * instead of stamping everything 'compaction'. Returns the summary
     * text on success (truthy — existing `if (await …)` callers keep
     * working), false when there is nothing usable. */
    const sysPrompt = (opts && opts.prompt) ||
      'You are a conversation summarizer. Compress the following conversation into a compact summary that preserves key facts, decisions, and the user\'s intent. Output only the summary.';
    /* Strict gateways reject a request whose last message is not
     * role=user (2026-10-09: every auto + manual compaction on tokenrouter
     * died with "The last message must have role=user" — the folded prefix
     * ends with an assistant message). The instruction rides LAST as a user
     * message: system-first stays for lenient providers, and
     * instruction-last is the better summary shape anyway. */
    const tailAsk = { role: 'user', content: (opts && opts.tailPrompt) ||
      'Summarize the conversation above into a compact summary that preserves key facts, decisions, and the user\'s intent. Output only the summary.' };
    const summary = await complete([
      { role: 'system', content: sysPrompt },
      ...history.slice(0, oldCount),
      tailAsk,
    ]);
    if (!summary || typeof summary !== 'string' || !summary.trim() || summary.trim() === '(no response)') return false;
    archiveWrite('summary', 'complete', (opts && opts.source) || 'compaction', summary, {
      compacted_messages: oldCount,
      kept_turns: keepTurns,
    });
    history = [{ role: 'system', content: 'Prior conversation summary: ' + summary },
               ...history.slice(oldCount)];
    return summary.trim();
  }
  async function trimHistory() {
    /* Hard safety net: the absolute entry cap — shed whole turn blocks from
     * the front so retained tool transcripts are never orphaned. */
    if (history.length > MAX_HISTORY_ENTRIES) {
      let over = history.length - MAX_HISTORY_ENTRIES;
      while (over > 0 && history.length > 0) {
        const before0 = history.length;
        removeTurnBlock(0);
        over -= (before0 - history.length);
      }
    }
    const budget = ctxBudget();
    /* alloc gate (§14): the compaction cliff moves with context pressure —
     * 0.70 of budget when the session is slack, down to 0.50 when the net
     * sees overflow coming within ~3 turns. Fixed cliff when ML is off. */
    const planC = allocPlanChat(0, 0);
    const compactAt = (planC && planC.compactAt > 0) ? planC.compactAt : COMPACT_AT;
    /* Re-arm one-shot compaction once usage drains back below half. */
    if (!autoCompactArmed && historyTokens() < budget * 0.5) autoCompactArmed = true;
    /* PLAN-ML-GATES §12: ML compaction gate — opportunistic passes that
     * free MECHANICAL junk (duplicates, boilerplate, re-fetchable reads)
     * before the one-shot cliff ever fires. Runs while usage is past
     * half the budget. The net SELECTS — only free-tier segments are
     * dropped here, no LLM calls. History stays strictly alternating,
     * so a turn goes only when BOTH its user prompt and assistant
     * answer are flagged; one-sided junk waits for the cliff. Turns go
     * oldest-first until usage drains below half the budget (the pass
     * budget is the whole flagged set — the drain target is what keeps
     * passes small). System summaries are never candidates. The cliff
     * below + the drop-oldest guard remain the safety nets; with the
     * gate off (/ml off or SOFUU_NO_ML=1) behaviour is exactly as
     * before. */
    if (cfg.ml && history.length > 4 && historyTokens() > budget * 0.5) {
      try {
        if (sofuu.ml && sofuu.ml.compaction && typeof sofuu.ml.compaction.plan === 'function') {
          const segs = [], segHist = [];
          for (let i = 0; i < history.length; i++) {
            const m = history[i];
            if (m.role !== 'user' && m.role !== 'assistant') continue;
            const text = String(m.content || '');
            segs.push({ text: text, tokens: estTok(text), age: 0,
                        kind: m.role === 'user' ? 0 : 1,
                        retrievable: false, compacted: false });
            segHist.push(i);
          }
          for (let j = 0; j < segs.length; j++) segs[j].age = segs.length - 1 - j;
          if (segs.length > 4) {
            let task = '';
            for (let i = history.length - 1; i >= 0; i--) {
              if (history[i].role === 'user') { task = String(history[i].content || ''); break; }
            }
            const recentTxt = history.slice(-4).map(m => String(m.content || '')).join('\n');
            const plan = JSON.parse(sofuu.ml.compaction.plan(JSON.stringify({
              task: task, summary: '', recent: recentTxt, segments: segs,
            }), JSON.stringify({ budget: historyTokens() })), '{}');
            const compact = plan.compact || [], tiers = plan.tiers || [];
            const flaggedFree = new Set();
            for (let c = 0; c < compact.length; c++) {
              const t = tiers[c];
              if (t !== 'dup' && t !== 'boilerplate' && t !== 'retrievable') continue;
              const hi = segHist[compact[c]];
              if (hi !== undefined) flaggedFree.add(hi);
            }
            /* Complete turn BLOCKS only (user+assistant both flagged),
             * never the newest block; oldest first, until usage drains
             * below half. Removal is block-atomic — a flagged block's
             * retained tool transcript leaves with it. */
            const blockStarts = [];
            for (let bi = 0; bi < history.length - 1; bi++) {
              if (history[bi].role !== 'user') continue;
              let bAns = -1;
              for (let bj = bi + 1; bj < history.length; bj++) {
                if (history[bj].role === 'user') break;
                if (history[bj].role === 'assistant' && !history[bj].tool_calls) { bAns = bj; break; }
              }
              if (bAns >= 0 && flaggedFree.has(bi) && flaggedFree.has(bAns)) blockStarts.push(bi);
            }
            let freed = 0, droppedTurns = 0;
            /* blockObjs hold the block user-message OBJECTS — after each
             * removal the array shifts, so re-locate by identity. */
            const blockObjs = blockStarts.map(bx => history[bx]);
            /* SAFETY CAP (2026-09-25): one pass may never gut the session.
             * Without it the only stop condition is "usage below half the
             * budget", which on a long history authorises deleting dozens
             * of turns whenever the gate is wrong. A quarter of the
             * complete blocks, never the two most recent. */
            /* count complete turns (a user message + its answer) */
            let completeBlocks = 0;
            for (let cb = 0; cb < history.length - 1; cb++) {
              if (history[cb].role === 'user') completeBlocks++;
            }
            const cap0 = Math.max(1, Math.floor(completeBlocks * 0.25));
            const maxDrops = (completeBlocks - cap0 < 2)
              ? Math.max(0, completeBlocks - 2) : cap0;
            let skippedProtected = 0;
            for (let b = 0; b < blockObjs.length && droppedTurns < maxDrops &&
                   historyTokens() > budget * 0.5; b++) {
              const blockIdx = history.indexOf(blockObjs[b]);
              if (blockIdx < 0) continue; /* already removed */
              /* Model-independent guard — the gate is the component under
               * suspicion, so the last line of defence cannot be the gate. */
              if (isProtectedBlock(history, blockIdx)) { skippedProtected++; continue; }
              const beforeBlk = historyTokens();
              removeTurnBlock(blockIdx);
              freed += beforeBlk - historyTokens();
              droppedTurns++;
            }
            if (droppedTurns > 0) {
              out('\x1b[90m  ml-compaction freed ' + fmtTk(freed) + ' tk (' +
                  droppedTurns + ' of ' + completeBlocks + ' turn' +
                  (completeBlocks === 1 ? '' : 's') +
                  ': dup/boilerplate/re-fetchable' +
                  (skippedProtected > 0 ? '; ' + skippedProtected + ' protected kept' : '') +
                  ')\x1b[90m\x1b[0m');
            }
          }
        }
      } catch (e) { /* the gate advises — any failure falls through to the cliff */ }
    }
    /* P5: auto-compaction — when the 1M (or model) window is nearly full,
     * summarize the whole context and reset usage to near-zero so the
     * session can continue indefinitely. Per-answer 32k thinking / 64k
     * output are *not* session caps — total thinking+output across many
     * turns can exceed them, only the per-answer limit is enforced. */
    if (autoCompactArmed && historyTokens() > budget * compactAt && history.length > 2) {
      autoCompactArmed = false;
      const savedTk = historyTokens();
      try {
        // Summarize *all* turns and keep only the summary → window → ~0
        if (await summarizeHistory(0)) {
          const m0 = Math.max(0, savedTk + (ctxOverhead >= 0 ? ctxOverhead : 0));
          usedCtx = ctxMeter();
          out('\x1b[90m  ⚙ auto-compacted → summary (' +
              fmtTk(Math.max(0, savedTk - historyTokens())) + ' tk saved, meter ' +
              fmtTk(m0) + '→' + fmtTk(usedCtx) + ')\x1b[0m');
        }
      } catch (e) {
        /* A failed auto-compact must not be silent: the drop-oldest guard
         * below will eat turns with no summary kept, which reads as
         * context loss. One dim line names the cause. */
        out('\x1b[90m  [auto-compact failed: ' + String(e.message || e) + ' — history will trim instead]\x1b[0m\n');
      }
    }
    /* Drop-oldest guard: still over budget after compaction (or compaction
     * failed)? Shed oldest whole turn blocks — never the newest block, and
     * never an orphaned tool message (hard 400 at the provider). */
    let droppedTurns = 0;
    while (history.length > 0 && historyTokens() > budget) {
      const lastStart = turnStartAt(history.length - 1);
      if (lastStart === 0) break; /* only the newest block left */
      const beforeDrop = historyTokens();
      removeTurnBlock(0);
      if (historyTokens() >= beforeDrop) break; /* safety: no progress */
      droppedTurns++;
    }
    if (droppedTurns > 0) {
      out('\x1b[33m  ⚠ history trimmed by ' + droppedTurns + ' turn' + (droppedTurns === 1 ? '' : 's') +
          ' to fit the ' + Math.floor(ctxBudget() / 100) / 10 + 'k-token context budget\x1b[0m');
    }
  }
  function streamOpts() {
    const c = applyFailover(cfg);
    const o = { messages: [], provider: c.provider, model: c.model };
    if (c.effort) o.effort = c.effort;
    if (c.api_key) o.api_key = c.api_key;
    if (c.base_url) o.base_url = c.base_url;
    if (c.profile) o.profile = c.profile;
    if (c.threads) o.threads = c.threads;
    /* Output cap from the ladder: config honored only as far as the
     * strictest real evidence for THIS model allows. A number the ladder
     * DEFAULTED (model unknown everywhere) is never sent — the endpoint
     * applies its own default (Pass-31 rule). Layer 0 re-clamps on the
     * wire — this keeps the driver consistent with it. */
    const capR = resolveCaps();
    if (capR.maxOutput > 0 && capR.maxSource !== 'default') o.max_tokens = capR.maxOutput;
    return o;
  }
  async function complete(messages, opts) {
    const o = streamOpts(); o.messages = messages;
    if (opts && opts.tools) o.tools = opts.tools;
    const r = await sofuu.ai.complete(o);
    if (r && r.toolCalls) return r;
    return r && r.text ? r.text.trim() : '(no response)';
  }
  /* Known built-in providers. An unknown provider name is treated as
   * "custom" by the AI layer and needs a base URL. */
  const KNOWN = ['openai', 'anthropic', 'local'];
  function usableConfig(view) {
    const v = view || cfg;
    if (!v.provider || !v.model) return false;
    if (v.provider === 'local') return true;
    return true;
  }
  /* Pretty tool-call rendering: show the interesting argument as a quoted
   * string (query/text/path/…) instead of a raw JSON blob; fall back to
   * compact JSON. Clipped so the line stays one readable row. */
  function prettyToolArgs(argsStr) {
    let o = null;
    try { o = JSON.parse(argsStr); } catch (e) { o = null; }
    if (o && typeof o === 'object') {
      const keys = ['query', 'text', 'message', 'msg', 'path', 'file', 'url', 'city', 'pattern', 'task', 'code'];
      for (const k of keys) {
        if (typeof o[k] === 'string' && o[k]) {
          let s = o[k];
          if (s.length > 60) s = s.slice(0, 60) + '…';
          return JSON.stringify(s);
        }
      }
      const s = JSON.stringify(o);
      return s.length > 80 ? s.slice(0, 80) + '…' : s;
    }
    const s = String(argsStr || '');
    return s.length > 60 ? s.slice(0, 60) + '…' : s;
  }
  function clip1(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n) + '…' : s;
  }

  /* ── TUI theme palette ────────────────────────────────────────────
   * All transcript colors resolve through T(role) into the active
   * theme's 256-palette params (see theme.rs — fixed colors, never the
   * terminal-themed 30-37 range, so a theme looks the same under any
   * terminal theme). THEME_FALLBACK reproduces the pre-theme look so a
   * missing bridge degrades to exactly today's colors, never unstyled.
   * PAL resets after every slash command (theme switches apply to the
   * very next paint) and is otherwise cached for the session. */
  const THEME_FALLBACK = { accent: '1;35', tool: '36', delegate: '35',
    heading: '1;36', code: '33', warn: '33', error: '1;31',
    add: '1;32', del: '1;31', hunk: '36' };
  let PAL = null;
  function pal() {
    if (!PAL) {
      try { PAL = JSON.parse(__chat_theme()); } catch (e) { PAL = null; }
      if (!PAL || typeof PAL !== 'object') PAL = {};
    }
    return PAL;
  }
  function T(role) {
    const p = pal();
    return (p && typeof p[role] === 'string' && p[role]) ? p[role] : THEME_FALLBACK[role];
  }

  /* Inline spans: code yellow, bold otherwise. Single pass split on
   * backticks — even segments are prose (bold applies), odd closed ones
   * are code (literal, so ** inside code never bolds). An unterminated
   * backtick keeps its mark and stays prose; unpaired ** stays literal.
   * Bold never spans segments, so markers can't merge across a span. */
  function fmtInline(t) {
    const segs = String(t).split('`');
    let r = '';
    for (let i = 0; i < segs.length; i++) {
      if (i % 2 === 1 && i !== segs.length - 1) {
        r += '\x1b[' + T('code') + 'm' + segs[i] + '\x1b[0m';
      } else {
        if (i % 2 === 1) r += '`';
        r += segs[i].replace(/\*\*([^*]+)\*\*/g, '\x1b[1m$1\x1b[0m');
      }
    }
    return r;
  }

  /* Markdown hierarchy for the final answer paint (TTY only). Model
   * answers arrive as undifferentiated white text — headings, emphasis
   * and code all read at the same level, so long answers are a wall.
   * Applied ONCE to the complete answer at turn end (never to the
   * streaming deltas, where markers can straddle chunk boundaries, and
   * never piped output, which stays raw). Newlines are never added or
   * removed, so soft-wrap row counts stay close to what streamed. */
  function formatMarkdown(s) {
    const lines = String(s == null ? '' : s).split('\n');
    const out = [];
    let inFence = false;
    for (const line of lines) {
      /* Fence lines dim whole; content inside fences is left alone
       * (code blocks are already visually distinct by indentation). */
      if (/^\s*```/.test(line)) { inFence = !inFence; out.push('\x1b[2m' + line + '\x1b[0m'); continue; }
      if (inFence) { out.push(line); continue; }
      /* Setext/HR rules and quotes recede. */
      if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { out.push('\x1b[2m' + line + '\x1b[0m'); continue; }
      if (/^\s>/.test(line)) { out.push('\x1b[2m' + line + '\x1b[0m'); continue; }
      let t = line;
      /* ATX headings: bold cyan, markers stripped. The body stays literal
       * (no inline spans inside) so the whole title reads as one level —
       * a reset mid-title would split it into two visual weights. */
      const hm = t.match(/^(#{1,6})\s+(.*)$/);
      if (hm) { out.push('\x1b[' + T('heading') + 'm' + hm[2] + '\x1b[0m'); continue; }
      out.push(fmtInline(t));
    }
    return out.join('\n');
  }

  /* Live todo checklist (2026-10-03): the agent maintains its plan via
   * todo_write on long tasks, and the checklist rides the tool_result
   * event — but the TUI never rendered it, so users never SAW a todo
   * list. Painted here, every update: done dim ✓ (finished work
   * recedes), doing cyan ▸ at full weight (the current step pops),
   * queued dim ○. Bounded at 15 rows with a "+N more" tail so a 50-item
   * list cannot flood the transcript. Replaces the one-row "checklist
   * updated (N steps)" summary — the list IS the result. */
  function paintChecklist(todos) {
    const list = Array.isArray(todos) ? todos : [];
    let done = 0;
    for (const t of list) if (t && t.status === 'done') done++;
    out('\x1b[1m  ☑ ' + done + '/' + list.length + '\x1b[0m');
    const shown = Math.min(list.length, 15);
    for (let i = 0; i < shown; i++) {
      const t = list[i] || {};
      const c = String(t.content || '').replace(/\s+/g, ' ').trim();
      if (t.status === 'done') out('\x1b[2m    ✓ ' + c + '\x1b[0m');
      else if (t.status === 'doing') out('\x1b[' + T('tool') + 'm    ▸ ' + c + '\x1b[0m');
      else out('\x1b[2m    ○ ' + c + '\x1b[0m');
    }
    if (list.length > shown) out('\x1b[2m    +' + (list.length - shown) + ' more\x1b[0m');
  }

  /* Paint a unified diff from an edit_file/write_file tool_result event,
   * one row per diff line so the TUI shows WHAT changed, not just that
   * something did. Rows arrive as `NNNN mark content` (4-wide gutter,
   * mark in {'-','+',' '}, ANSI-free — the payload also travels to the
   * model). The gutter paints dim; +/- rows get a full-row background
   * tint (dark green / dark red) under bold bright text, difftool-style;
   * hunk headers cyan; context, file markers and notes dim. Text-span
   * tints only (no padding to terminal width — a padded row that wraps
   * would tear the gutter on the continuation line). */
  function paintDiff(diff) {
    const rows = String(diff == null ? '' : diff).split('\n');
    for (const row of rows) {
      const t = row.replace(/^\s+/, '');
      if (t.indexOf('+++') === 0 || t.indexOf('---') === 0 || t.indexOf('...') === 0) {
        out('\x1b[2m    ' + t + '\x1b[0m');
      } else if (t.indexOf('@@') === 0) {
        out('\x1b[' + T('hunk') + 'm    ' + t + '\x1b[0m');
      } else {
        const m = row.match(/^\s*(\d+)\s+([-+ ])\s?([\s\S]*)$/);
        if (!m) { out('\x1b[2m    ' + row + '\x1b[0m'); continue; }
        let g = m[1];
        while (g.length < 4) g = ' ' + g;
        if (m[2] === '+') out('    \x1b[2m' + g + '\x1b[0m \x1b[' + T('add') + 'm+ ' + m[3] + '\x1b[0m');
        else if (m[2] === '-') out('    \x1b[2m' + g + '\x1b[0m \x1b[' + T('del') + 'm- ' + m[3] + '\x1b[0m');
        else out('\x1b[2m    ' + g + '   ' + m[3] + '\x1b[0m');
      }
    }
  }

  /* ── Agent-phase label mapping ────────────────────────────────────
   * Pure function: onStep's {kind, payload} → the phase label, or null to
   * leave CURRENT_PHASE unchanged (tool_result: the tool/delegate phase
   * that started the work keeps showing until the next step arrives). */
  function phaseForStep(kind, payload) {
    const p = (payload === undefined || payload === null) ? {} : payload;
    if (kind === 'plan' || kind === 'think' || kind === 'recall') return 'Thinking';
    if (kind === 'tool' || kind === 'delegate') {
      const name = String((p.name || p.agent || p.tool || p.action || '') || '');
      if (/web_search|web_open|search/i.test(name)) return 'Web searching';
      if (/debug|trace|error|fix|diagnos/i.test(name)) return 'Debugging';
      if (/write|create|edit|implement|refactor|generate/i.test(name) &&
          !/debug|test/i.test(name)) return 'Coding';
      if (/read|list|scan|grep|inspect|review|audit|test|lint|check|verify/i.test(name)) return 'Auditing';
      return 'Working';
    }
    if (kind === 'tool_result') return null; /* keep the preceding phase */
    if (kind === 'answer_delta') return 'Working';
    if (typeof kind === 'string' && kind.indexOf('rlm:') === 0) return 'Working';
    return null;
  }
  /* Renders `label` with one dim "cursor" letter at PHASE_FRAME, advances
   * the cursor, and overlays the result above the input box. A single dim
   * position sweeps left→right and loops — not a fading trail. */
  function tickPhase() {
    if (!CURRENT_PHASE) { try { __chat_phase(''); } catch (e) {} return; }
    const label = CURRENT_PHASE;
    let line = '';
    for (let i = 0; i < label.length; i++) {
      line += (i === PHASE_FRAME ? '\x1b[2m' + label[i] + '\x1b[0m' : label[i]);
    }
    PHASE_FRAME = (PHASE_FRAME + 1) % label.length;
    try { __chat_phase('  ' + line); } catch (e) {}
  }
  let phaseTimer = null;
  const startPhase = () => {
    if (!TTY || phaseTimer) return;
    PHASE_FRAME = 0;
    tickPhase();
    phaseTimer = setInterval(tickPhase, 120);
  };
  function stopPhase() {
    if (phaseTimer) { clearInterval(phaseTimer); phaseTimer = null; }
    CURRENT_PHASE = null;
    if (TTY) { try { __chat_phase(''); } catch (e) {} }
  }

  async function turn(text, ov, turnNo, attemptNo) {
    failoverTo = ov || null;
    archiveTurn = turnNo || archiveTurn;
    archiveAttempt = attemptNo || archiveAttempt;
    const recoveryForTurn = archivePendingRecovery;
    archivePendingRecovery = '';
    const c = applyFailover(cfg);
    /* Unusable config = never call the API: one clean line, nothing else.
     * Judged on the MERGED view so a failover target is usable even when
     * the active config is not. */
    if (!usableConfig(c)) {
      failoverTo = null;
      out('\x1b[90m  No usable model configured — run /provider to set one up\x1b[0m\n');
      return;
    }
    /* Caps discovery (TTL-gated, best-effort): the endpoint's model
     * listing carries what its models really accept — harvest it into
     * the alloc gate's discovered store. Silent on any failure; local
     * endpoints skipped. */
    discoverCaps();
    /* F9: load hooks.js lazily on first turn. */
    await ensureHooks();
    /* F9: pre-hook — may modify or skip the prompt. */
    const hookedText = await runHookPre(text);
    if (hookedText === null) { out('\x1b[90m  (skipped by pre-hook)\x1b[0m\n'); return; }
    text = hookedText;
    /* F6: budget preflight — block if session spend ≥ budget. */
    let budget;
    try { budget = JSON.parse(__chat_budget_check()); } catch (e) { budget = null; }
    if (budget && !budget.ok) {
      out('\x1b[33m  ⚠ Budget cap reached ($' + budget.spent.toFixed(4) + ' / $' + budget.budget.toFixed(2) + ')\x1b[0m\n');
      return;
    }
    /* ── The agent runtime owns the loop (PLAN-AGENTS A1.5) ──────────
     * Recall augmentation, RLM routing, tool planning/execution, brain
     * storage, budgets and cancellation all live in src/js/agent.js —
     * ONE implementation for chat and headless. This driver renders
     * onStep events and owns input/history/session logging. */
    if (!sofuu.agent || typeof sofuu.agent.run !== 'function') {
      out('\x1b[31m  ✗ agent runtime unavailable (src/js/agent.js failed to load)\x1b[0m\n');
      stopPhase();
      return;
    }
    /* MCP is lazy: connect on demand only when a tool-using turn starts. */
    if (mcpTools.length === 0 && mcpServersList().length > 0) {
      await connectMcpServers();
    }
    /* Sub-agents (PLAN-AGENTS A8.3): definitions from ~/.sofuu/agents/*.js
     * are offered to the model via the `delegate` tool — loaded once per
     * session (same lazy load /agents uses; maxDepth:1 means children
     * can't delegate further). Loaded BEFORE mention parsing so `@name`
     * can resolve on the first turn too. */
    if (!agentsDirLoaded) {
      agentsDirLoaded = true;
      try { if (sofuu.agent && typeof sofuu.agent.loadDir === 'function') await sofuu.agent.loadDir(); } catch (e) {}
    }
    var subAgentNames = [];
    try {
      if (sofuu.agent && typeof sofuu.agent.list === 'function')
        subAgentNames = sofuu.agent.list().map(function (a) { return a.name; });
    } catch (e) {}
    /* ── @agent mentions (both forms) ────────────────────────────────
     * "@name task" or "@agent:name task" runs the named agent DIRECTLY
     * with its own definition (system/tools/memory/model) as a focused
     * run — no chat history injected. Agents win over same-named files;
     * unknown names fall through to the @file path below. Esc cancels
     * via signal:'chat' like any turn. */
    const typed = text;
    let agentMention = null;
    {
      const t = text.trim();
      let m = /^@agent:([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/.exec(t);
      if (!m) m = /^@([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/.exec(t);
      if (m) {
        let name = null;
        if (subAgentNames.indexOf(m[1]) >= 0) name = m[1];
        else {
          const low = m[1].toLowerCase();
          const ci = subAgentNames.filter(function (n) { return n.toLowerCase() === low; });
          if (ci.length === 1) name = ci[0];
        }
        if (name) agentMention = { name: name, task: m[2].trim() };
      }
    }
    if (agentMention) text = agentMention.task;
    /* F3: expand @file mentions (on the remainder — the task itself may
     * attach files). Fresh alloc plan for this turn (attachment budget). */
    allocPlanValid = false;
    const expanded = await expandMentions(text);
    const turnText = expanded.text;
    const manifest = expanded.manifest;
    if (promptLoggedTurn !== archiveTurn) {
      promptLoggedTurn = archiveTurn;
      try { __chat_log('prompt', String(typed)); } catch (e) {}
    }
    /* The turn starts in Thinking, including while lazy MCP/agent setup runs. */
    CURRENT_PHASE = 'Thinking';
    startPhase();
    if (TTY) out('\x1b[' + T('accent') + 'm  ⏺ you\x1b[0m \x1b[2m·\x1b[0m ' + typed);
    /* M1: /watch changes surface in chat context ONCE — drained into this
     * turn's ephemeral context note and cleared (hard-capped at the poll
     * site too, so an idle session can't accumulate). */
    const watchedNote = watchChangesPending.length
      ? watchChangesPending.map(c => '- ' + c.path + ' (' + c.kind + ')').join('\n')
      : '';
    watchChangesPending.length = 0;
    /* Strict effort: the CLI follows EXACTLY what the user picked.
     * `off`/empty → no thinking on ANY endpoint (never auto-injects).
     * The thinking block is only sent when c.effort is low/medium/high/max.
     * Models detected as thinking-incapable at runtime (no_think_models)
     * also never get an effort parameter. */
    let noThink = (c.no_think_models || []).indexOf(c.model) >= 0 || (c.effort === 'off') || (resolveCaps().thinking === 'none');
    /* Local inference engines (ollama, llama.cpp) get the same lean shape as
     * the in-process local provider: no coding preset, no tool defs, a short
     * system prompt, so a CPU-bound model is not buried in prompt tokens. */
    const isLocal = c.provider === 'local' || c.provider === 'ollama' || c.provider === 'llamacpp';
    const def = {
      name: 'chat',
      preset: isLocal ? undefined : 'coding',
      /* P1: the byte-stable shared core prompt. For local models, keep
       * it lean so CPU prefill latency matches direct `sofuu infer`. */
      system: isLocal ? 'You are a helpful AI assistant.' : sofuu.agent.CORE_PROMPT,
      tools: isLocal ? [] : chatToolDefs(),
      agents: (isLocal || !subAgentNames.length) ? undefined : subAgentNames,
      provider: c.provider, model: c.model,
      /* strict: only what the user selected, never more; off → undefined */
      effort: noThink ? undefined : (c.effort && c.effort !== 'off' ? c.effort : undefined),
      api_key: c.api_key || undefined,
      base_url: c.base_url || undefined,
      profile: c.profile || undefined,
      threads: c.threads || undefined,
      max_tokens: (function () { const r = resolveCaps(); return (r.maxOutput > 0 && r.maxSource !== 'default') ? r.maxOutput : undefined; })(),
      embed_provider: c.embed_provider || undefined,
      embed_model: c.embed_model || undefined,
      /* Memory embedding backend (PLAN-TINY-SEMANTIC-EMBEDDER §7):
       * ''/hash default; 'semantic' opts the brain into the 64-dim
       * projector (agent.js resolves + enforces, SOFUU_MEMORY_BACKEND
       * env also honored there). */
      embedding: c.memory_backend || undefined,
      memory: c.brain ? 'shared' : 'off',
      /* Config "brain_path" override — an empty value falls through to
       * agent.js's project-local <cwd>/.sofuu/brain/brain.qtsq default,
       * so this only bites when a host deliberately points the brain
       * elsewhere. The driver's /remember etc. share whatever this
       * resolves to (sofuu.agent.brainFor). */
      brainPath: c.brain_path || undefined,
      /* PLAN-ML-GATES: context-economy gates for this turn ('off' skips
       * every gate in agent.js; SOFUU_NO_ML=1 unregisters sofuu.ml too). */
      ml: c.ml ? 'on' : 'off',
      rlm: c.rlm === 'on' ? 'on' : (c.rlm === 'auto' ? 'auto' : 'off'),
      ctx_window: (function () { const r = resolveCaps(); return r.window > 0 ? r.window : undefined; })(),
      /* P2: config-level recall gating knobs (0 = agent.js defaults). */
      recallMin: c.recall_min > 0 ? c.recall_min : undefined,
      recallBudget: c.recall_budget > 0 ? c.recall_budget : undefined,
      /* Chat historically has no budgets beyond the step cap and the 200k
       * answer cap (both enforced inside agent.js). The step cap is
       * deliberately generous: real coding turns routinely need 10-30
       * rounds (read → read sections → grep → edit → verify), and the old
       * flat 8 made long tool-using turns die in a bare "(no response)"
       * (AUDIT-NO-RESPONSE-2026-08-24 cause b). On the rare breach the
       * agent spends one salvage round summarizing progress (agent.js) and
       * the driver prints why it stopped. SOFUU_CHAT_MAX_STEPS is a test
       * seam, not a user knob. */
      /* maxContinuations: how many times the step allowance renews before
       * the run finally stops (2026-10-01). 0 = hard stop at maxSteps,
       * which is what a test that only wants to exercise the breach
       * path sets — otherwise every breach test silently runs 3x the
       * windows it thinks it is running. */
      budget: { maxSteps: (parseInt(env('SOFUU_CHAT_MAX_STEPS'), 10) || 200), maxDepth: 1, maxTokens: 1e9, maxWallMs: 1e9,
                maxContinuations: env('SOFUU_CHAT_MAX_CONTINUATIONS') === ''
                  ? 2 : Math.max(0, parseInt(env('SOFUU_CHAT_MAX_CONTINUATIONS'), 10) || 0) },
    };
    /* alloc gate (§14): surface allocation notes BEFORE the request goes
     * out — config clamped to the model's caps, a learned limit applied,
     * unknown-model conservative defaults. One dim ASCII line each,
     * deduped per session; silent when ML is off. */
    try {
      const planN = allocPlanChat(estTok(turnText), manifest ? estTok(String(manifest)) : 0);
      if (planN && Array.isArray(planN.notes)) {
        for (const n of planN.notes) {
          if (allocNotedNotes[n]) continue;
          allocNotedNotes[n] = true;
          out('\x1b[90m  ~ alloc · ' + n + '\x1b[0m');
        }
      }
    } catch (e) {}
    /* Streaming render state — same UX as the pre-migration driver:
     * spinner + growing line in the TUI, plain writes when piped. */
    let acc = '', anim = null, si2 = 0, sawThink = false, capped = false;
    /* Whitespace rhythm: a blank row opens each tool-call group so tools
     * don't run directly into the prose above. Set on tool/delegate,
     * cleared by answer text — consecutive calls in one burst share one
     * blank row instead of each taking one. TTY only (pipes stay clean). */
    let lastRowWasTool = false;
    const thinks = [];
    const stopAnim = () => { if (anim) { clearTimeout(anim); anim = null; } };
    const tick2 = () => { outLast('\x1b[2m' + SPINNER[si2++ % SPINNER.length] + '\x1b[0m ' + acc); };
    const startAnim = () => { if (TTY && !anim) anim = setTimeout(function loop() { tick2(); anim = setTimeout(loop, 120); }, 120); };
    let rlmSummary = null;
    let answer = '(no response)';
    let res = null;
    /* The phase indicator is started at turn entry and always torn down in
     * the `finally` below, mirroring stopAnim(). One onStep renderer for
     * both paths (chat loop + direct @agent mention run). */
    let firstPromptTk = 0;  /* first LLM request's prompt = current context size */
    /* ── Live context meter (real-time during a running turn) ─────────
     * The footer's usedCtx used to freeze until the turn ENDED: the
     * numbers only calibrated from the final result. Mid-turn the real
     * signal exists though — EVERY 'llm' event carries that request's
     * promptTokens (each tool round is bigger than the last), and the
     * streaming deltas/tool results estimate the growth BETWEEN requests.
     * liveReqTk = last real request size; liveGrownTk = estimated growth
     * since that request (this turn's deltas + fresh tool results). The
     * 2s metricsTimer paints refreshStatus, which folds them in. */
    let liveReqTk = 0;       /* latest request's real promptTokens */
    let liveGrownTk = 0;     /* estimated growth since that request */
    let liveTurnTk = 0;      /* est. tokens of THIS turn's prompt text */
    let lastLivePaint = 0;   /* throttle: painted footer at most 1/s */
    liveCtxExtra = 0;        /* fresh turn: no mid-turn growth yet */
    const paintLive = function () {
      liveCtxExtra = liveGrownTk;
      const now = Date.now();
      if (now - lastLivePaint > 1000) { lastLivePaint = now; refreshStatus(''); }
    };
    const onStep = function (e) {
          const p = (e.payload === undefined || e.payload === null) ? {} : e.payload;
          /* Meter capture: the first LLM request's prompt size. */
          if (!firstPromptTk && e.kind === 'llm' && p.usage &&
              p.usage.promptTokens > 0) {
            firstPromptTk = p.usage.promptTokens;
          }
          /* Live meter: each round's real request size replaces the
           * estimate; growth since it starts from zero again. The first
           * request also calibrates ctxOverhead EARLY (same formula as the
           * turn-end calibration) so the +overhead part is real mid-turn. */
          if (e.kind === 'llm' && p.usage && p.usage.promptTokens > 0) {
            liveReqTk = e.payload.usage.promptTokens;
            liveGrownTk = 0;
            if (!liveTurnTk) liveTurnTk = estTok(text);
            if (ctxOverhead < 0) {
              const hNow = historyTokens();
              ctxOverhead = Math.max(0, liveReqTk - hNow - liveTurnTk);
            }
            usedCtx = liveReqTk;
            paintLive();
          }
          /* Between requests: the streamed answer + fresh tool results
           * grow the context the NEXT request will carry. Estimate only —
           * replaced by the real number on the next 'llm' event. */
          if (e.kind === 'answer_delta') {
            liveGrownTk += estTok(String(p));
            paintLive();
          } else if (e.kind === 'tool_result' && p && p.result !== undefined) {
            liveGrownTk += estTok(String(p.result));
            paintLive();
          }
          const ph = phaseForStep(e.kind, p);
          if (ph !== null && ph !== CURRENT_PHASE) {
            CURRENT_PHASE = ph;
            PHASE_FRAME = 0;
          }
          if (e.kind === 'think') {
            stopAnim();
            thinks.push(String(p.text || ''));
            /* Effort OFF: thinking runs under the hood — collected (it
             * still feeds the Thinking phase) but not painted into the
             * transcript, and sawThink stays false so no seal line is
             * emitted for a line that was never drawn. */
            if (c.effort || c.provider === 'local') {
              sawThink = true;
              outLast('  ▸ thinking \x1b[3m' + thinks.join('') + '\x1b[0m');
            }
          } else if (e.kind === 'answer_delta') {
            if (sawThink && TTY) { out(''); sawThink = false; thinks.length = 0; }
            acc += String(p);
            if (acc.length > MAX_ANSWER_CHARS && !capped) {
              capped = true;
              out('  \x1b[' + T('warn') + 'm⚠ answer capped at ' + MAX_ANSWER_CHARS + ' chars\x1b[0m');
            }
            if (TTY) { startAnim(); tick2(); }
            else { process.stdout.write(String(p)); }
            lastRowWasTool = false;
          } else if (e.kind === 'plan') {
            /* A new planning round discards any preliminary streamed text. */
            acc = '';
          } else if (e.kind === 'tool') {
            stopAnim();
            if (TTY && !lastRowWasTool) out(' '); /* air before a tool group */
            lastRowWasTool = true;
            const a = prettyToolArgs(p.args);
            out('\x1b[90m  ⏺\x1b[0m \x1b[' + T('tool') + 'm' + p.name + '\x1b[0m\x1b[2m(' + a + ')\x1b[0m');
          } else if (e.kind === 'delegate') {
            stopAnim();
            if (TTY && !lastRowWasTool) out(' '); /* air before a tool group */
            lastRowWasTool = true;
            out('\x1b[90m  ⏺\x1b[0m \x1b[' + T('delegate') + 'm' + p.agent + '\x1b[0m\x1b[2m ← ' + clip1(p.task, 60) + '\x1b[0m');
          } else if (e.kind === 'recall') {
            /* F1/P2: /why reports exactly what gating let through this turn
             * (hits already thresholded, deduped, budget-cut by agent.js). */
            try { __chat_set_recall(JSON.stringify(Array.isArray(p.hits) ? p.hits : [])); } catch (e2) {}
            /* Brain feedback: one dim line so a recalled context is visible
             * (and the brain being ON is observable), never the contents. */
            if (p && p.count > 0) {
              out('\x1b[90m  ⏺ brain · ' + p.count + ' memor' + (p.count === 1 ? 'y' : 'ies') +
                  ' recalled (/why to inspect)\x1b[0m');
            }
          } else if (e.kind === 'consolidated') {
            /* Brain housekeeping (rare, rate-limited): weak-old episodic
             * memories were merged into semantic clusters. */
            out('\x1b[90m  ⏳ brain · consolidated ' + (p.clusters || 1) +
                ' cluster' + ((p.clusters || 1) === 1 ? '' : 's') +
                ' — old memories merged\x1b[0m');
          } else if (e.kind === 'mlgate') {
            /* PLAN-ML-GATES: a supervisor rule gate flagged waste (repeat
             * call / re-read of an unchanged file). One dim ASCII-only line;
             * the advisory nudge itself rides the tool result in-band. */
            stopAnim();
            out('\x1b[90m  ~ ml · ' + p.rule + ' · ' + p.tool + ' (step ' + p.step + ')\x1b[0m');
          } else if (e.kind === 'allocgate') {
            /* alloc gate (§14): the pre-flight fit check corrected the
             * request (re-capped tool results / clamped output / dropped
             * old messages), or error learning picked up a provider limit.
             * One dim ASCII line — allocation stays observable. */
            stopAnim();
            out('\x1b[90m  ~ alloc · ' +
                (p.action ? p.action + (p.model ? ' · ' + p.model : '') +
                            (p.error ? ' · ' + clip1(p.error, 80) : '')
                          : 'fit ' + String(p.fit || '') + ' · ' + String(p.actions || '')) +
                '\x1b[0m');
          } else if (e.kind === 'tool_result') {
            /* Keep tool output sparse: normal results stay in the live
             * transcript, while failures (or an explicitly marked result)
             * become independently recoverable archive records. */
            if (p && (p.error || p.archive === true)) {
              const material = p.error || p.result;
              if (material !== undefined && material !== null) {
                archiveWrite('tool_result', p.error ? 'failed' : 'complete', 'tool', material, {
                  tool_name: String(p.name || ''),
                  error_code: p.error ? archiveErrorCode(p.error) : undefined,
                  chars: p.chars || 0,
                });
              }
            }
            if (TTY && p.result) {
              /* todo_write carries its checklist on the event: paint the
               * list itself instead of the "checklist updated" summary. */
              if (p.name === 'todo_write' && Array.isArray(p.todos) && p.todos.length) {
                paintChecklist(p.todos);
              } else {
                /* Compact one-row result: embedded newlines become " · "
                 * so multi-line tool output (search results, file dumps)
                 * no longer splats unindented rows into the transcript.
                 * The ↳ marker stays dim but the CONTENT renders at full
                 * weight — the whole row used to be dim and read as a
                 * whisper next to the white answer text. */
                const r = String(p.result).replace(/\s*\n+\s*/g, ' · ').replace(/\s+/g, ' ').trim();
                out('\x1b[2m  ↳\x1b[0m ' + clip1(r, 140) + '\x1b[0m');
                /* edit_file/write_file carry their unified diff on p.diff
                 * (the 200-char event result could never hold it): paint it
                 * multi-line below the summary row so the change itself is
                 * visible, not just its byte count. */
                if ((p.name === 'edit_file' || p.name === 'write_file') && p.diff) {
                  paintDiff(p.diff);
                }
              }
            }
            /* Tool ERRORS used to print nothing at all: the error emit
             * carries {name, error} with no `result`, so the branch above
             * skipped it and a failed edit_file died silently on screen
             * while only the model saw the message in-band. Red ✗, always
             * visible, so a failed write is unmistakable. */
            else if (TTY && p.error) {
              /* The error text itself usually starts with "<tool>:" (the
               * tools name their own failures) — don't print the name
               * twice ("✗ edit_file: edit_file: …"). */
              const em = clip1(String(p.error), 200);
              const nm = String(p.name || 'tool');
              out('\x1b[' + T('error') + 'm  ✗ ' + (em.indexOf(nm + ':') === 0 ? em : nm + ': ' + em) + '\x1b[0m');
            }
          } else if (e.kind === 'rlm:route') {
            out('\x1b[90m  ⏺ rlm · working…\x1b[0m');
          } else if (e.kind === 'rlm:done') {
            rlmSummary = p;
          }
    };
    let runError = null;
    /* Re-apply every turn: covers the mode being set (or persisted from a
     * previous session) after agent.js finished loading past boot. */
    applyMode();
    try {
      res = agentMention
        ? await sofuu.agent.run(agentMention.name, turnText, {
            signal: 'chat', onStep, context: recoveryForTurn || undefined,
          })
        : await sofuu.agent.run(def, turnText, {
            history: history, signal: 'chat', onStep,
            /* P1/P6: volatile context rides the ephemeral context message —
             * never inside the byte-stable system prompt. */
            shared: lastShared || '',
            watched: watchedNote || '',
            context: recoveryForTurn || undefined,
          });
      answer = String((res && res.answer) || '(no response)');
    } catch (e) {
      runError = e;
    } finally {
      stopAnim();
      stopPhase(); /* turn end (G): hide the indicator, never leave it stale */
      /* Turn over — the mid-turn growth estimate is superseded by the
       * end-of-turn calibration below; zero it so ctxMeter()/refreshStatus
       * paint the settled value, not estimate + old growth. */
      liveCtxExtra = 0;
    }
    if (runError) {
      const message = String((runError && runError.message) || runError);
      const errorCode = archiveErrorCode(message);
      const partialRef = acc.trim()
        ? archiveWrite('partial', 'partial', 'assistant', acc, {
            error_code: errorCode,
            turn_complete: false,
          }, { retry_of: archiveLastFailure && archiveLastFailure.errorId })
        : null;
      const related = partialRef ? [partialRef.id] : [];
      const errorRef = archiveWrite('error', 'failed',
        errorCode === 'runtime' ? 'runtime' : 'provider', message, {
          error_code: errorCode,
          turn_complete: false,
        }, { related_output_ids: related, retry_of: archiveLastFailure && archiveLastFailure.errorId });
      archiveLastFailure = {
        errorId: errorRef && errorRef.id,
        partialId: partialRef && partialRef.id,
        errorCode: errorCode,
      };
      throw runError;
    }
    if (rlmSummary) {
      /* RLM turn: the answer arrived complete — print it plus the
       * one-line trace summary (same format as the pre-migration path). */
      out(TTY ? formatMarkdown(answer) : answer);
      out('\n\x1b[90m  ⏺ rlm · ' + (rlmSummary.calls || 0) + ' calls · ' + (rlmSummary.rounds || 0) + ' rounds · '
          + ((rlmSummary.ms || 0) / 1000).toFixed(1) + 's' + (rlmSummary.stopped ? ' · ' + rlmSummary.stopped : '') + '\x1b[0m\n');
    } else {
      if (TTY) {
        if (sawThink) out('');          /* seal a think-only answer */
        /* Full-text repaint with markdown hierarchy (headings/bold/code
         * get their levels — the streamed rows above showed raw markers).
         * Same newlines in and out, so the replaceable unit just swaps. */
        outLast(formatMarkdown(answer)); /* final line, no spinner */
      } else {
        /* Piped mode renders only answer_delta chunks — a turn with zero
         * deltas would print nothing at all, so emit the final answer
         * (e.g. "(no response)") when nothing streamed. */
        if (!acc && answer) out(answer);
        out('\n');
      }
    }
    if (res && res.stopped === 'cancelled') out('\x1b[90m  ⏹ stopped (esc)\x1b[0m');
    else if (res && res.stopped && String(res.stopped).indexOf('budget_') === 0) {
      /* Budget breach (steps/wall/tokens): the agent attempted a salvage
       * summary before stopping; say WHY it stopped so a breach is never
       * mistaken for a dead provider (AUDIT-NO-RESPONSE-2026-08-24 b). */
      /* With renewals, naming only maxSteps reads as a lie: the run
       * genuinely executed several windows' worth of rounds. Report the
       * renewals in the headline and let the detail line carry the
       * totals. */
      const renewed = !!(res.breach && res.breach.continuations);
      const why = res.stopped === 'budget_steps'
        ? 'step budget' + (renewed
            ? ' after ' + res.breach.continuations + ' renewal' +
              (res.breach.continuations === 1 ? '' : 's') +
              ' (' + (res.breach.maxSteps || '?') + ' rounds/window)'
            : (agentMention ? '' : ' (' + def.budget.maxSteps + ' rounds)'))
        : (res.stopped === 'budget_wall' ? 'wall-clock budget' : 'token budget');
      out('\x1b[90m  ⏹ stopped: ' + why + ' reached\x1b[0m');
      /* Say WHY, not just that it stopped. A bare "step budget reached"
       * cannot be acted on: 200 rounds on a genuinely large task needs a
       * bigger budget or a /compact, while 200 rounds re-reading one file
       * is a different bug entirely. agent.js collects this at the moment
       * of the breach, which is the only time the numbers exist. */
      try {
        const b = res.breach;
        if (b && b.kind === 'budget_steps') {
          const tk = (b.promptTokens || 0) + (b.completionTokens || 0);
          const win = b.ctxWindow || 0;
          const pct = win > 0 ? Math.round((tk / win) * 100) : null;
          out('\x1b[90m    ↳ ' + (b.steps || 0) + '/' + (b.maxSteps || 0) + ' rounds · ' +
              (b.llmCalls || 0) + ' llm calls · ' + (b.toolCalls || 0) + ' tool calls' +
              (b.loopMsgs ? ' · ' + b.loopMsgs + ' msgs in turn' : '') +
              (tk ? ' · ctx ' + fmtTk(tk) + (pct !== null ? ' (' + pct + '% of ' + fmtTk(win) + ')' : '') : '') +
              (b.mostCalledTool ? ' · most: ' + b.mostCalledTool : '') +
              '\x1b[0m');
          /* Renewals already spent — the difference between "never got a
           * second window" and "renewed and still not done". */
          if (b.continuations) {
            out('\x1b[90m    ↳ ' + b.continuations + '/' + (b.maxContinuations || '?') +
                ' renewals spent before it stopped\x1b[0m');
          }
          /* If the window was the real constraint, say so — /compact is
           * the immediate action and the user cannot infer it. */
          if (pct !== null && pct >= 85) {
            out('\x1b[90m    ↳ context was at ' + pct + '% — /compact now, or lower /ctx\x1b[0m');
          }
        }
      } catch (eB) {}
    } else if (res && res.stopped) {
      /* Unknown stop cause (future agent.js values): still say SOMETHING —
       * a silent stop reads as a dead provider (P3-2). */
      out('\x1b[90m  ⏹ stopped: ' + String(res.stopped) + '\x1b[0m');
    }
    if (agentMention && res) {
      /* @agent mention: compact run summary under the answer. */
      const u = res.usage || {};
      out('\x1b[90m  ⏺ via ' + agentMention.name + ' · ' + (u.llmCalls || 0) + ' llm calls · '
          + (u.toolCalls || 0) + ' tool calls' + (res.stopped ? ' · ' + res.stopped : '') + '\x1b[0m');
    }
    const ctxTk = (res && res.usage && res.usage.promptTokens) || 0;
    const outTk = (res && res.usage && res.usage.completionTokens) || 0;
    /* Meter calibration on a REAL request size: prefer the first LLM
     * request of this turn; a single-LLM-call turn's aggregate is that
     * same number. Multi-round tool turns are never used as-is — their
     * aggregate re-counts tool results and would balloon the meter.
     * Overhead = system prompt + tool schemas + ephemeral context: the
     * request's prompt minus the history it carries and minus this
     * turn's own text, so historyTokens() + overhead stays equal to the
     * request size and never double-counts the turn. */
    const histAtReq = historyTokens();
    const turnTk = estTok(text);
    if (firstPromptTk > 0) {
      usedCtx = firstPromptTk;
      ctxOverhead = Math.max(0, firstPromptTk - histAtReq - turnTk);
    } else if (ctxTk > 0 && ((res && res.usage && res.usage.llmCalls) || 0) <= 1) {
      usedCtx = ctxTk;
      ctxOverhead = Math.max(0, ctxTk - histAtReq - turnTk);
    }
    /* Chip names the two numbers (user request 2026-09-03 f): "in 30k ·
     * out 800" — the bare "30k→800" made the sides ambiguous once the
     * footer also shows ctx. Multi-round turns: the usage aggregate SUMS
     * every round's prompt — it is the turn's total INPUT bill, not one
     * request's context size (each tool round re-sends the whole context);
     * the call count says so. */
    const nCalls = ((res && res.usage && res.usage.llmCalls) || 0);
    let tk = (ctxTk > 0 || outTk > 0)
      ? ('in ' + fmtTk(ctxTk) + ' · out ' + fmtTk(outTk)) : '';
    if (nCalls > 1) tk = tk + ' · ' + nCalls + ' calls';
    /* F6: report usage → compute cost + persist. Cache token slots (P6.4)
     * ride along so /cost can show prefix-cache hits when non-zero. */
    if (ctxTk > 0 || outTk > 0) {
      const cacheR = (res && res.usage && res.usage.cacheReadTokens) || 0;
      const cacheW = (res && res.usage && res.usage.cacheWriteTokens) || 0;
      try { __chat_report_usage(c.model || '', ctxTk, outTk, cacheR, cacheW); } catch (e) {}
      /* Show running spend in the footer next to tk. */
      let cost;
      try { cost = JSON.parse(__chat_budget_check()); } catch (e) { cost = null; }
      if (cost && cost.spent > 0) {
        tk = tk + ' · $' + cost.spent.toFixed(4);
      }
    }
    out('\n');
    refreshStatus(tk);
    /* Persist the answer to this session's .qtsq (the prompt was already
     * logged at the start of the turn) so a resumed session replays the
     * full transcript. A failed-salvage placeholder is NEVER persisted nor
     * pushed into history (P2-14): a resumed session would replay
     * "(no response)" as if it were the answer. */
    const noResponse = answer === '(no response)' && res && res.stopped;
    if (!noResponse) {
      try { __chat_log('answer', String(answer)); } catch (e) {}
      /* F3: history stores the manifest only (not full file text). */
      const histText = manifest ? text + ' [' + manifest + ']' : text;
      history.push({ role: 'user', content: histText });
      /* Claude-style retention (2026-09-03 e): the turn's tool transcript
       * (assistant tool_calls + tool results) persists into history between
       * the prompt and the answer — the model keeps its tool context across
       * turns; compaction sheds it, not a per-turn release. */
      const turnTranscript = (res && Array.isArray(res.transcript)) ? res.transcript : null;
      if (turnTranscript) {
        for (const tm of turnTranscript) history.push(tm);
      }
      history.push({ role: 'assistant', content: answer });
      await trimHistory();
    }
    /* Meter: re-estimate for the NEXT request. With transcripts retained
     * (2026-09-03 e) the settled value stays close to the live one — it
     * moves only when compaction/drop-oldest just shrank history. A big
     * drop is compaction (its own ⚙ line explains it), so the settle note
     * stays generic; fire only on a real unexplained shift. */
    const liveShown = usedCtx + liveCtxExtra;
    usedCtx = ctxMeter();
    if (TTY && liveShown > usedCtx * 1.2 && liveShown > 1000) {
      out('\x1b[90m  · ctx meter settled ' + fmtTk(Math.round(liveShown)) + '→' +
          fmtTk(usedCtx) + '\x1b[0m');
    }
    /* M2: one full GC per completed turn — bounds JS garbage to a single
     * turn's worth instead of accumulating against the heap cap. */
    try { __chat_gc(); } catch (e) {}
    /* F9: post-hook — may modify the answer. */
    const finalAnswer = await runHookPost(text, answer, res && res.usage);
    if (finalAnswer !== answer) {
      /* Update the last history entry if the post-hook changed the answer. */
      if (history.length >= 1) history[history.length - 1].content = finalAnswer;
    }
    /* Materialize the user-visible result after post-processing. A cancelled
     * or budget-stopped run is partial evidence, never a successful final. */
    const usage = (res && res.usage) || {};
    const archiveMeta = {
      prompt_tokens: usage.promptTokens || 0,
      completion_tokens: usage.completionTokens || 0,
      llm_calls: usage.llmCalls || 0,
      tool_calls: usage.toolCalls || 0,
      stopped: res && res.stopped ? String(res.stopped) : undefined,
    };
    const partialText = (acc && acc.trim()) ? acc :
      ((finalAnswer && finalAnswer !== '(no response)' && finalAnswer !== '(cancelled)') ? finalAnswer : '');
    if (res && res.stopped === 'cancelled') {
      const partialRef = partialText
        ? archiveWrite('partial', 'cancelled', 'assistant', partialText, archiveMeta,
            { retry_of: archiveLastFailure && archiveLastFailure.errorId })
        : null;
      archiveWrite('error', 'cancelled', 'runtime', 'turn cancelled by user', {
        error_code: 'cancelled',
        stopped: 'cancelled',
      }, { related_output_ids: partialRef ? [partialRef.id] : [] });
    } else if (res && res.stopped) {
      const errorCode = String(res.stopped) === 'budget_tokens' ? 'output_limit' : 'runtime';
      const partialRef = partialText
        ? archiveWrite('partial', 'partial', 'assistant', partialText,
            Object.assign({}, archiveMeta, { error_code: errorCode }),
            { retry_of: archiveLastFailure && archiveLastFailure.errorId })
        : null;
      archiveWrite('error', 'failed', 'runtime', 'turn stopped: ' + String(res.stopped), {
        error_code: errorCode,
        stopped: String(res.stopped),
      }, { related_output_ids: partialRef ? [partialRef.id] : [] });
    } else if (finalAnswer && finalAnswer !== '(no response)') {
      archiveWrite('final', 'complete', 'assistant', finalAnswer, archiveMeta, {
        retry_of: archiveLastFailure && archiveLastFailure.errorId,
      });
    } else {
      archiveWrite('error', 'failed', 'runtime', 'model returned no response', {
        error_code: 'empty_response',
      }, { retry_of: archiveLastFailure && archiveLastFailure.errorId });
    }
  }
  // ── Provider setup wizard: /provider (no args) ──────────────────
  // Three wire formats only: openai-compatible, anthropic, local (ollama).
  // The endpoint is AUTO-DETECTED from the URL; the profile question is
  // only asked when detection can't decide.
  const WIZARD_PROVIDERS = ['openai', 'anthropic', 'local'];
  const WIZARD_BUILTIN_BASE = {
    openai: 'https://api.openai.com/v1',
    anthropic: 'https://api.anthropic.com/v1',
    local: 'http://127.0.0.1:11434',
  };
  function detectProfile(base) {
    const u = String(base || '').toLowerCase();
    /* Host-based detection — the reliable, unambiguous signals. */
    if (u.includes('anthropic') || u.includes('claude')) return 'anthropic';
    if (u.includes('127.0.0.1') || u.includes('localhost') || u.includes('0.0.0.0')) return 'local';
    /* Path-based: /v1 with /chat/completions or /models → openai-compat;
     * /v1/messages → anthropic. */
    if (u.includes('/messages')) return 'anthropic';
    if (u.includes('ollama') || u.includes(':11434')) return 'local';
    /* Default: anything with /v1 (or nothing recognizable) is treated as
     * OpenAI-compatible — the most common server format. */
    return 'openai';
  }
  function sleepMs(ms) { return new Promise(r => setTimeout(r, ms)); }
  async function fetchWithTimeout(url, opts, ms) {
    let timer = null;
    const to = new Promise((_, rej) => { timer = setTimeout(() => rej(new Error('timeout')), ms || 8000); });
    try {
      return await Promise.race([fetch(url, opts), to]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
  /* Stored endpoint → API root for the models listing. The base may be a
   * bare host, a /v1 root, or the FULL completion URL — including corrupted
   * forms like /v1/chat/completions/chat/completions (saved by old desktop
   * builds). Collapse doubles, loop-strip every completion suffix, then
   * drop a leftover /chat: the listing lives at the version root
   * (…/v1/models — /chat/models is not a wire endpoint). stripV1 is for the
   * local branch (ollama's /api/tags sits at the server ROOT). */
  function wizardModelsRoot(base, stripV1) {
    let u = String(base || '').trim();
    u = u.split('#')[0].split('?')[0].replace(/\/+$/, '');
    u = u.replace(/\/chat\/completions\/chat\/completions$/i, '/chat/completions');
    let again = true;
    while (again) {
      again = false;
      const sufs = ['/chat/completions', '/completions', '/messages', '/api/chat'];
      for (const s of sufs) {
        if (u.toLowerCase().endsWith(s)) { u = u.slice(0, u.length - s.length); again = true; }
      }
    }
    while (u.charAt(u.length - 1) === '/') u = u.slice(0, -1);
    if (stripV1) u = u.replace(/\/v1$/i, '');
    else if (/\/chat$/i.test(u)) u = u.slice(0, -'/chat'.length);
    return u;
  }
  async function wizardFetchModels(prov, base, key) {
    try {
      if (prov === 'local') {
        /* /api/tags lives at the server ROOT (P3-5: appending it to a
         * completion endpoint 404'd every local fetch). */
        const root = wizardModelsRoot(base || 'http://127.0.0.1:11434', true);
        const r = await fetchWithTimeout(root + '/api/tags', {}, 4000);
        const j = JSON.parse(await r.text());
        return (j.models || []).map(m => m.name);
      }
      if (prov === 'anthropic') return []; // no public model list
      /* openai-compatible: listing at the API root + /models. */
      const root = wizardModelsRoot(base || WIZARD_BUILTIN_BASE.openai, false);
      const h = key ? { Authorization: 'Bearer ' + key } : {};
      const r = await fetchWithTimeout(root + '/models', { headers: h }, 6000);
      /* text() + JSON.parse — Response.json() truncated some chunked
       * bodies ("unexpected data at the end", e.g. lightning.ai's 53-model
       * listing), so parse the full body ourselves. */
      const j = JSON.parse(await r.text());
      return (j.data || []).map(m => m.id).filter(Boolean);
    } catch (e) {
      return []; // unreachable / no key → fall back to manual entry
    }
  }
  async function providerWizard(prefill) {
    out('\n\x1b[1m  ⚙ Provider setup\x1b[0m  \x1b[90m(Ctrl-D cancels)\x1b[0m');
    const pre = prefill || {};
    // 1. Provider NAME — the real name the user knows (e.g. fireworks-ai,
    // groq, my-local-llm). "openai/anthropic/local" are endpoint FORMATS,
    // not names — never stored as the provider name.
    let name = '';
    while (!name) {
      const hint = pre.name ? ' [' + pre.name + ']' : '';
      const inp = await __readline('  Provider name (e.g. fireworks-ai, groq, deepseek)' + hint + ': ');
      if (inp === null || inp === undefined) { out(''); return false; }
      name = String(inp).trim() || (pre.name || '');
      if (!name) out('\x1b[31m  A provider name is required.\x1b[0m');
    }
    // 2. Base URL — the REAL endpoint (required; the format is detected
    // from it, so e.g. https://api.fireworks.ai/inference/v1 works).
    let base = '';
    while (!base) {
      const hint = pre.base_url ? ' [' + pre.base_url + ']' : '';
      const inp = await __readline('  Base URL (e.g. https://api.fireworks.ai/inference/v1)' + hint + ': ');
      base = (inp === null || inp === undefined ? '' : String(inp).trim()) || (pre.base_url || '');
      if (!base) out('\x1b[31m  A base URL is required.\x1b[0m');
    }
    // 2b. AUTO-DETECT the wire format from the URL. Ask only when the
    // detection is ambiguous (a custom host that looks like neither).
    let profile = detectProfile(base);
    if (profile === 'local') {
      profile = 'local';
    }
    if (profile === 'openai' && !/api\.openai|openai|anthropic|claude|ollama|11434|localhost|127\.0\.0\.1/.test(base.toLowerCase())) {
      // Custom host that didn't match any known pattern — confirm the format.
      const q = await __readline('  Can\'t auto-detect the format — 1) OpenAI-compatible 2) Anthropic: ');
      const qt = String(q || '').trim();
      if (qt === '2') profile = 'anthropic';
      else profile = 'openai';
    }
    // 3. API key
    const keyInp = await __readline('  API key (Enter = skip): ');
    const key = (keyInp === null || keyInp === undefined ? '' : String(keyInp).trim()) || '';
    // 4. fetch models (the detected format decides the list endpoint)
    outLast('  \x1b[90m⏳ Fetching models…\x1b[0m');
    const models = await wizardFetchModels(profile, base, key);
    if (models.length > 0) {
      outLast('  \x1b[90m✓ ' + models.length + ' models found\x1b[0m');
    } else {
      outLast('  \x1b[90m⚠ could not fetch models — type one manually\x1b[0m');
    }
    let model = '';
    if (models.length > 0) {
      out('\n  \x1b[1mAvailable models\x1b[0m');
      const page = models.slice(0, 30);
      page.forEach((m, i) => out('    ' + (i + 1) + ') ' + m));
      if (models.length > page.length) out('    \x1b[90m… ' + (models.length - page.length) + ' more\x1b[0m');
      while (!model) {
        const inp = await __readline('  Pick 1-' + page.length + ', 0 = type manually: ');
        if (inp === null || inp === undefined) { out(''); return false; }
        const t = String(inp).trim();
        const n = parseInt(t, 10);
        if (n >= 1 && n <= page.length) model = page[n - 1];
        else if (t === '0' || t === '') {
          const c = await __readline('  Model name: ');
          model = (c === null || c === undefined ? '' : String(c).trim());
        }
      }
    } else {
      const inp = await __readline('  (Could not fetch models) Type a model name: ');
      model = (inp === null || inp === undefined ? '' : String(inp).trim());
    }
    // 5. persist — provider NAME is the real name the user gave; the
    // profile (wire format) is stored separately for the AI layer.
    // Deduplicate any re-appended /chat/completions (wizard guard: editing an
    // existing endpoint that already contains the suffix would 404).
    let bare = base.replace(/\/+$/, '').replace(/\/chat\/completions\/chat\/completions/g, '/chat/completions');
    let hasChatSuffix = bare.endsWith('/chat/completions') || bare.endsWith('/completions') || bare.endsWith('/api/chat');
    let endpoint = hasChatSuffix ? bare : (profile === 'local' ? bare + '/api/chat' : (profile === 'openai' ? bare + '/chat/completions' : bare));
    try { __chat_apply_provider(name, endpoint, key, model, profile); } catch (e) {}
    cfg = JSON.parse(__chat_getcfg());
    out('\n  \x1b[32m✓ Provider configured:\x1b[0m ' +
      name +
      (model ? ' · model ' + model : '') +
      ' · ' + endpoint +
      (key ? ' · key set' : '') + '\n');
    refreshStatus('');
    maybeRefreshPanel();
    return true;
  }

  async function compact() {
    if (history.length === 0) { out('\x1b[90m  [Nothing to compact yet.]\x1b[0m\n'); return; }
    out('\x1b[90m  [Compacting ' + Math.floor(history.length / 2) + ' turns…]\x1b[0m');
    const before = historyTokens();
    try {
      /* Manual /compact folds everything but the most recent turn. */
      if (!await summarizeHistory(1)) {
        out('\x1b[90m  [Nothing older than one turn to fold — history unchanged]\x1b[0m\n');
        return;
      }
      const m0 = Math.max(0, before + (ctxOverhead >= 0 ? ctxOverhead : 0));
      usedCtx = ctxMeter();
      out('\x1b[90m  [Compacted → context (' + fmtTk(Math.max(0, before - historyTokens())) +
          ' tk saved, meter ' + fmtTk(m0) + '→' + fmtTk(usedCtx) + ')]\x1b[0m\n');
      /* M3 (Brain Core II): the folded summary is the session's distilled
       * gist — land it in the brain (source 'compact') so it stays
       * recallable after the turn it compacted. Compaction never blocks
       * on the brain. */
      try {
        const gistText = String((history[0] && history[0].content) || '')
          .replace(/^Prior conversation summary: /, '');
        if (sofuu.agent && typeof sofuu.agent.rememberGist === 'function' && gistText) {
          await sofuu.agent.rememberGist(brainDef(), gistText);
        }
      } catch (eG) {}
      refreshStatus(lastChip); /* meter drops visibly right away */
    } catch (e) { out('\x1b[31m  [Compact failed: ' + String(e.message || e) + ']\x1b[0m\n'); }
  }
  // Output routing: in the full-screen TUI everything lands in the
  // conversation area; otherwise plain stdout (piped/REPL stays clean).
  function out(s) {
    if (TTY) { try { __tui_log(s); } catch (e) {} }
    else { console.log(s); }
  }
  function outLast(s) {
    if (TTY) { try { __tui_log_last(s); } catch (e) {} }
    else { console.log(s); }
  }
  function fmtTk(n) {
    return n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : String(n);
  }
  function refreshStatus(tk) {
    lastChip = tk;
    if (typeof __chat_status !== 'function') return;
    const dim = (s) => '\x1b[2m' + s + '\x1b[0m';
    let s = 'sofuu ' + dim('·') + ' \x1b[' + T('accent') + 'm' + (cfg.model || '(no model)') + '\x1b[0m'
      /* Mode chip rides the status row only when RESTRICTIVE (plan/edit):
       * the default 'full' stays clean, but a gated turn must never be
       * confused for an unguarded one — amber so it reads at a glance.
       * Placed RIGHT AFTER the model: the footer truncates from the tail
       * (≈47 cells on an 80-col pty once hints are subtracted), and the
       * active restriction must survive where provider/effort may not. */
      + (cfg.permissions && cfg.permissions !== 'full'
        ? ' ' + dim('·') + ' \x1b[33m' + 'mode ' + cfg.permissions + '\x1b[0m'
        : '');
    if (cfg.provider) {
      const prov = cfg.provider === 'custom'
        ? ((cfg.profile || 'openai') + '-compat')
        : cfg.provider;
      s += ' ' + dim('·') + ' ' + dim(prov);
    }
    if (cfg.effort) s += ' ' + dim('·') + ' ' + dim('effort ' + cfg.effort);
    /* The per-turn usage chip ("in 96 · out 1 · $") goes to the METRIC row,
     * not here: the status row shares its width with the hints column
     * (W-gutter-1-hints ≈ 32 cells on an 80-col terminal) and the chip
     * was silently truncated away there — "no token display" was really
     * "no room". Row 2 has the space. */
    const chip = tk || '';
    /* Footer row 1 right: dim hints. Row 2: left = context used,
     * right = real-time RAM of the sofuu process. The window is the
     * ladder-resolved one for the SELECTED model — the footer can't
     * show a stale global number the model doesn't have. */
    const win = resolveCaps().window;
    /* Mid-turn: the meter shows the live estimate (last real request +
     * growth since) so it climbs while the answer streams/tools run. */
    const used = usedCtx > 0 ? usedCtx + liveCtxExtra : liveCtxExtra;
    const pct = used > 0 ? Math.round(used / win * 100) : 0;
    /* Percent only once it is meaningful (≥1%) — "ctx 0.2k/128k (0%)"
     * reads like a bug. Turns amber past 80% of the window. */
    let ctxMet;
    if (used > 0) {
      ctxMet = 'ctx ' + fmtTk(used) + '/' + fmtTk(win) + (pct >= 1 ? ' · ' + pct + '%' : '');
      if (used / win > 0.8) ctxMet = '\x1b[33m' + ctxMet + '\x1b[0m';
    } else {
      ctxMet = 'ctx 0/' + fmtTk(win);
    }
    /* Brain state rides the ctx metric — the footer stays clean when the
     * brain is off. The per-turn usage chip rides the SAME row (left of
     * RAM): "ctx 2.4k/128k · brain · in 30k · out 800 · $0.0121". */
    if (cfg.brain) { ctxMet = ctxMet + dim(' · brain'); }
    if (chip) { ctxMet = ctxMet + dim(' · ' + chip); }
    let ramMet = '';
    try {
      if (typeof __chat_rss === 'function') {
        const b = Number(__chat_rss()) || 0;
        if (b > 0) {
          const mb = b / 1048576;
          ramMet = 'ram: ' + mb.toFixed(1) + ' MB';
          /* M6: soft tripwire — amber past the warn threshold + a one-time
           * notice. Inform, never kill; if this fires persistently, some
           * structure is leaking. */
          const warnMb = cfg.rss_warn_mb > 0 ? cfg.rss_warn_mb : 1024;
          if (mb >= warnMb) {
            ramMet = '\x1b[33m' + ramMet + '\x1b[0m';
            if (!rssWarned) {
              rssWarned = true;
              out('\x1b[33m  ⚠ RAM usage crossed ' + warnMb + ' MB — watch the footer; persistent growth means something is leaking\x1b[0m');
            }
          }
        }
      }
    } catch (e) {}
    try { __chat_status(s, '/help · ⏎ · esc · ctrl-k copy', ramMet, ctxMet); } catch (e) {}
  }
  // ── Interactive selector overlay (pickers for /model /provider /effort)
  // C routes keys + draws the overlay rows; ALL state lives here.
  let selectorCtl = null;
  globalThis.__selector_key = function(name, ch) {
    if (selectorCtl) { try { selectorCtl(name, ch); } catch (e) {} }
  };
  async function selectMenu(opts) {
    if (typeof __selector_open !== 'function') return null;
    const title = opts.title || 'Select';
    const hint  = opts.hint  || '';
    const tabs  = opts.tabs  || [];
    const items   = opts.items || [];
    const currentId = opts.currentId || '';
    let filter = '', tabIdx = 0, sel = 0, off = 0, result = null;
    const H = (typeof __tui_height === 'function') ? __tui_height() : 24;
    const maxV = Math.max(3, Math.min(10, H - 12));
    if (opts.initialTab) {
      const at = tabs.indexOf(opts.initialTab);
      if (at >= 0) tabIdx = at;
    }

    function filteredItems() {
      const t = tabs[tabIdx];
      const f = filter.toLowerCase();
      return items.filter(it =>
        (!t || t === 'all' || it.tab === t) &&
        (!f || it.label.toLowerCase().includes(f) || (it.note || '').toLowerCase().includes(f)));
    }
    function clampSel(list) {
      if (list.length === 0) { sel = 0; off = 0; return; }
      if (sel >= list.length) sel = list.length - 1;
      if (sel < 0) sel = 0;
      if (sel < off) off = sel;
      if (sel >= off + maxV) off = sel - maxV + 1;
    }
    function draw() {
      const list = filteredItems();
      clampSel(list);
      const rows = [];
      /* Spacing rule: section gaps are multiples of 2 (2 blank rows). */
      rows.push('');
      rows.push('');
      rows.push('  \x1b[1;35m' + title + '\x1b[0m  \x1b[90m(type to search · esc cancels)\x1b[0m');
      if (hint) rows.push('  \x1b[90m' + hint + '\x1b[0m');
      if (filter)
        rows.push('  \x1b[90mfilter:\x1b[0m ' + filter + ' \x1b[90m· ' + list.length + ' match' + (list.length === 1 ? '' : 'es') + '\x1b[0m');
      if (tabs.length > 1) {
        rows.push('');
        rows.push('');
        rows.push('  ' + tabs.map((t, i) =>
          i === tabIdx ? '\x1b[7;35m ' + t + ' \x1b[0m' : '\x1b[90m ' + t + ' \x1b[0m').join('  '));
      }
      rows.push('');
      rows.push('');
      const vis = list.slice(off, off + maxV);
      for (let i = 0; i < vis.length; i++) {
        const gi = off + i, it = vis[i], curRow = gi === sel;
        rows.push((curRow ? '\x1b[35m❯\x1b[0m \x1b[1m' : '  ') + it.label + (curRow ? '\x1b[0m' : '') +
                  (it.note ? '  \x1b[90m' + it.note + '\x1b[0m' : '') +
                  (it.id === currentId ? ' \x1b[32m← current\x1b[0m' : ''));
      }
      if (list.length === 0) rows.push('  \x1b[90m(no matches)\x1b[0m');
      if (off + maxV < list.length) rows.push('  \x1b[90m▼ ' + (list.length - off - maxV) + ' more\x1b[0m');
      if (off > 0) rows.push('  \x1b[90m▲ ' + off + ' above\x1b[0m');
      __selector_draw(rows.join('\n'));
    }

    const opened = __selector_open();
    draw();
    selectorCtl = function(name, ch) {
      const list = filteredItems();
      if (name === 'char') { filter += ch; sel = 0; off = 0; draw(); return; }
      if (name === 'backspace') { filter = filter.slice(0, -1); sel = 0; off = 0; draw(); return; }
      if (name === 'redraw') { draw(); return; }
      if (name === 'tab' || name === 'shift-tab' ||
          ((name === 'left' || name === 'right') && tabs.length > 1)) {
        const d = (name === 'shift-tab' || name === 'left') ? tabs.length - 1 : 1;
        tabIdx = (tabIdx + d) % tabs.length;
        sel = 0; off = 0; draw(); return;
      }
      if (name === 'up'   || (name === 'left'  && tabs.length <= 1)) {
        if (list.length) { sel = (sel - 1 + list.length) % list.length; draw(); } return;
      }
      if (name === 'down' || (name === 'right' && tabs.length <= 1)) {
        if (list.length) { sel = (sel + 1) % list.length; draw(); } return;
      }
      if (name === 'pageup')   { if (list.length) { sel = Math.max(0, sel - maxV); draw(); } return; }
      if (name === 'pagedown') { if (list.length) { sel = Math.min(list.length - 1, sel + maxV); draw(); } return; }
      if (name === 'enter') {
        result = list[sel] || null;
        selectorCtl = null;
        __selector_close(result ? 'select' : 'cancel');
        return;
      }
      if (name === 'esc') {
        selectorCtl = null;
        __selector_close('cancel');
        return;
      }
    };
    await opened; /* resolves when __selector_close runs */
    return result;
  }
  async function pickModel() {
    /* One tab per USER-ADDED provider (Tab / Shift-Tab / ←→ switch). Each
     * tab lists that provider's REAL live models fetched from its own
     * endpoint; an unreachable endpoint falls back to manual entry instead
     * of fake suggestions. Picking from another provider's tab switches
     * the active provider first, so the model applies to the right
     * endpoint/key. */
    if (!cfg.provider || !cfg.model) {
      out('\x1b[90m  No provider configured yet — run /provider first\x1b[0m\n');
      await providerWizard();
      try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
      return;
    }
    let provs = (cfg.providers && cfg.providers.length) ? cfg.providers.slice() : [];
    if (!provs.length) {
      /* Legacy flat config: synthesize the single active provider. */
      provs = [{ name: cfg.provider, endpoint: cfg.base_url, api_key: cfg.api_key || '', model: cfg.model, profile: '' }];
    }
    const tabs = provs.map(p => p.name);

    outLast('\x1b[90m  ⏳ Fetching models from ' + tabs.length + ' provider' + (tabs.length === 1 ? '' : 's') + '…\x1b[0m');
    /* All providers in PARALLEL — each fetch races its own timeout, so one
     * slow/unreachable endpoint can't stall the others. */
    const lists = await Promise.all(provs.map(async p => {
      try {
        const profile = p.profile || detectProfile(p.endpoint);
        const got = await Promise.race([
          wizardFetchModels(profile, p.endpoint, p.api_key || ''),
          new Promise(r => setTimeout(() => r(null), 6000)),
        ]);
        if (Array.isArray(got) && got.length) {
          /* Remember the live listing on the provider entry — an endpoint
           * that is unreachable NEXT time still offers these models. */
          try { __chat_models_cache(p.name, JSON.stringify(got)); } catch (e) {}
        }
        return Array.isArray(got) ? got : [];
      } catch (e) { return []; }
    }));
    const okCount = lists.filter(l => l.length > 0).length;
    if (okCount > 0) {
      outLast('\x1b[90m  ✓ live models from ' + okCount + '/' + tabs.length + ' providers\x1b[0m');
    } else {
      outLast('\x1b[90m  ⚠ no endpoint reachable — type a model name manually\x1b[0m');
    }

    const items = [], seen = {};
    const activeName = cfg.active || cfg.provider;
    for (let ti = 0; ti < provs.length; ti++) {
      const p = provs[ti], live = lists[ti] || [], isCur = p.name === activeName;
      if (isCur) {
        /* Current model first on its own tab (sel starts at 0). */
        const id = p.name + '|' + cfg.model;
        if (!seen[id]) { seen[id] = 1; items.push({ id: id, label: cfg.model, tab: p.name, note: 'current' }); }
      }
      for (const m of live) {
        const id = p.name + '|' + m;
        if (seen[id]) continue; seen[id] = 1;
        items.push({ id: id, label: m, tab: p.name, note: '' });
      }
      /* Models remembered for this provider (every model the user set here
       * + the last successful listing) — shown even when the live fetch
       * came back empty, so an unreachable endpoint no longer hides them. */
      const stored = Array.isArray(p.models) ? p.models : [];
      for (const m of stored) {
        const id = p.name + '|' + m;
        if (seen[id]) continue; seen[id] = 1;
        items.push({ id: id, label: m, tab: p.name, note: 'saved' });
      }
      /* The provider's stored default model, when not in the live list. */
      if (p.model && !seen[p.name + '|' + p.model]) {
        seen[p.name + '|' + p.model] = 1;
        items.push({ id: p.name + '|' + p.model, label: p.model, tab: p.name, note: 'saved' });
      }
      if (isCur && live.length === 0) {
        items.push({ id: '\x00manual', label: '⌨ Type a model name…', tab: p.name,
                     note: 'endpoint unreachable — enter it manually' });
      }
    }
    const it = await selectMenu({
      title: 'Select a model',
      hint: '↑↓ move · tab switch provider · enter apply · esc cancel',
      tabs: tabs, items: items, initialTab: activeName,
      currentId: activeName + '|' + cfg.model,
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    if (it.id === '\x00manual') {
      const m = await __readline('  Model name: ');
      if (m && String(m).trim()) __chat_slash('/model ' + String(m).trim());
    } else {
      const bar = it.id.indexOf('|');
      const provName = it.id.slice(0, bar);
      const m = it.id.slice(bar + 1);
      /* Model picked on ANOTHER provider's tab → switch provider first. */
      if (provName && provName !== activeName && tabs.indexOf(provName) >= 0) {
        try { __chat_select_provider(provName); } catch (e) {}
      }
      __chat_slash('/model ' + m);
    }
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickProvider() {
    const provs = (cfg.providers && cfg.providers.length) ? cfg.providers : [];
    const items = [];
    for (const p of provs) {
      const active = p.name === cfg.active;
      const note = (p.model || '(no model)') + ' · ' + (p.endpoint || '(no endpoint)') + (p.api_key ? ' · key set' : '') + (active ? ' ← active' : '');
      items.push({ id: '\x00select:' + p.name, label: p.name + (active ? ' ✓ active' : ''), tab: '', note: note });
      items.push({ id: '\x00edit:' + p.name, label: '  ✎ Edit ' + p.name, tab: '', note: 'fix name / endpoint / key / model' });
      items.push({ id: '\x00remove:' + p.name, label: '  🗑 Remove ' + p.name, tab: '', note: 'remove from list' });
    }
    if (!provs.length) {
      items.push({ id: '\x00none', label: '(no providers yet)', tab: '', note: 'add one below' });
    }
    items.push({ id: '\x00add', label: '＋ Add a provider', tab: '', note: 'OpenAI-compatible · Anthropic · local (guided setup)' });
    const currentId = cfg.active ? ('\x00select:' + cfg.active) : '\x00add';
    const it = await selectMenu({ title: 'Providers', hint: '↑↓ move · enter apply · esc cancel', tabs: [], items: items, currentId: currentId });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    if (it.id === '\x00add' || it.id === '\x00none') {
      await providerWizard();
    } else if (it.id.indexOf('\x00select:') === 0) {
      const name = it.id.slice('\x00select:'.length);
      try { __chat_select_provider(name); } catch (e) {}
      try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
      out('\x1b[32m  ✓ Active provider → ' + name + '\x1b[0m');
    } else if (it.id.indexOf('\x00edit:') === 0) {
      const name = it.id.slice('\x00edit:'.length);
      const entry = provs.find(function(p){ return p.name === name; });
      if (entry) await providerWizard({ name: entry.name, base_url: entry.endpoint });
    } else if (it.id.indexOf('\x00remove:') === 0) {
      const name = it.id.slice('\x00remove:'.length);
      const ans = await __readline('  Remove provider "' + name + '"? y/N: ');
      if (ans && String(ans).trim().toLowerCase() === 'y') {
        try { __chat_remove_provider(name); } catch (e) {}
        out('\x1b[90m  Removed ' + name + '\x1b[0m');
      } else { out('\x1b[90m  (unchanged)\x1b[0m'); try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {} refreshStatus(''); maybeRefreshPanel(); return; }
    }
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickEffort() {
    /* Capability-aware picker (rt/model_caps via sofuu.ai.modelCaps):
     * notes show the REAL budgets this model will get; models that
     * cannot think say so instead of offering a fake ladder. */
    const caps = modelCaps();
    const fmt = n => n >= 1000 ? Math.round(n / 1000) + 'k' : String(n);
    const runtimeNoThink = (cfg.no_think_models || []).indexOf(cfg.model) >= 0;
    const isAnthropicWire = (cfg.profile === 'anthropic' || cfg.provider === 'anthropic');
    if (runtimeNoThink) {
      out('\x1b[33m  ⓘ ' + cfg.model + ' does not support thinking — effort is not applicable.\x1b[0m\n');
      return;
    }
    // Anthropic endpoint: 64k thinking is endpoint property, any model on it can use it
    // OpenAI endpoint: per-model ladder, generic for unknown models, none only for known non-reasoning
    if (!isAnthropicWire && caps && caps.thinking === 'none') {
      out('\x1b[33m  ⓘ ' + cfg.model + ' does not support thinking — effort is not applicable.\x1b[0m\n');
      return;
    }
    let order, notes;
    if (!isAnthropicWire && caps && caps.thinking === 'effort') {
      /* Discrete ladder model (OpenAI style): only its real levels.
       * Selecting an unsupported level falls back to the highest one.
       * OpenAI endpoint is generic — any provider's models can appear here. */
      order = ['off'].concat(caps.efforts || []);
      notes = { off: 'strict off — no thinking' };
      for (const l of (caps.efforts || [])) {
        notes[l] = l === caps.efforts[caps.efforts.length - 1]
          ? 'highest this model supports'
          : l + ' reasoning';
      }
      if (cfg.effort === 'max') notes.currentNote = "max → falls back to '" + caps.efforts[caps.efforts.length - 1] + "'";
    } else if (isAnthropicWire) {
      /* Anthropic wire: 64k is thinking budget (not context window), endpoint-driven, any model */
      order = ['off', 'low', 'medium', 'high', 'max'];
      const cap = 64000;
      notes = { off: 'strict off — no thinking',
                low: '≈' + fmt(Math.ceil(cap * 0.0625)) + ' tk',
                medium: '≈' + fmt(Math.ceil(cap * 0.25)) + ' tk',
                high: '≈' + fmt(Math.ceil(cap * 0.5)) + ' tk',
                max: '≈' + fmt(Math.ceil(cap * 0.9)) + ' tk (of ' + fmt(cap) + ' thinking)' };
    } else {
      /* Budget-style unknown on OpenAI generic endpoint or fallback */
      order = ['off', 'low', 'medium', 'high', 'max'];
      notes = { off: 'strict off — no thinking',
                low: '~6% thinking budget', medium: '~25%', high: '~50%', max: '~90% (uses the full ceiling)' };
      if (caps && caps.maxThinkingBudget > 0) {
        const cap = caps.maxThinkingBudget;
        notes.low = '≈' + fmt(Math.ceil(cap * 0.0625)) + ' tk';
        notes.medium = '≈' + fmt(Math.ceil(cap * 0.25)) + ' tk';
        notes.high = '≈' + fmt(Math.ceil(cap * 0.5)) + ' tk';
        notes.max = '≈' + fmt(Math.ceil(cap * 0.9)) + ' tk (of ' + fmt(cap) + ')';
      }
    }
    const it = await selectMenu({
      title: 'Thinking effort' + (cfg.model ? ' — ' + cfg.model : ''),
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: order.map(l => ({ id: l, label: l, tab: '', note: notes[l] || '' })),
      currentId: cfg.effort || 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/effort ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickTheme() {
    /* 13 dark-blend themes, independent of the terminal's own theme.
     * Applies through /theme <name> (the Rust side validates + persists)
     * so the picker and the typed command share one code path. */
    let themes = [];
    try { themes = JSON.parse(__chat_themes()); } catch (e) {}
    if (!themes.length) { out('\x1b[90m  No themes available.\x1b[0m\n'); return; }
    const it = await selectMenu({
      title: 'Theme',
      hint: '↑↓ move · enter apply · esc cancel',
      tabs: [],
      items: themes.map(t => ({ id: t.name, label: t.name })),
      currentId: cfg.theme || '',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/theme ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    PAL = null;
    maybeRefreshPanel();
  }
  async function pickRlm() {
    const notes = { off: 'provider default routing', on: 'every long turn goes through the RLM loop', auto: 'heuristic routing for long-context turns' };
    const order = ['off', 'on', 'auto'];
    const it = await selectMenu({
      title: 'RLM routing',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: order.map(l => ({ id: l, label: l, tab: '', note: notes[l] })),
      currentId: cfg.rlm || 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/rlm ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickCtx() {
    const mc = modelCaps();
    const eff = resolveCaps().window;
    const presets = [
      { id: '0', label: 'default', note: (mc ? 'model default (' + fmtTk(mc.ctxWindow) + ')' : 'provider default (' + fmtTk(eff) + ')') },
      { id: '32768', label: '32k', note: '32,768 tokens' },
      { id: '131072', label: '128k', note: '131,072 tokens' },
      { id: '1048576', label: '1M', note: '1,048,576 tokens' },
    ];
    const it = await selectMenu({
      title: 'Context window',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: presets.map(p => ({ id: p.id, label: p.label, tab: '', note: p.note })),
      currentId: String(cfg.ctx_window || 0),
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/ctx ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickMaxout() {
    const mc = modelCaps();
    const presets = [
      { id: '0', label: 'default', note: mc && mc.maxOutput > 0
          ? "model's max output (" + fmtTk(mc.maxOutput) + ')'
          : 'provider default' },
      { id: '4096', label: '4k', note: '4,096 tokens' },
      { id: '32768', label: '32k', note: '32,768 tokens' },
      { id: '131072', label: '128k', note: '131,072 tokens' },
      { id: '384000', label: '384k', note: '384,000 tokens (max)' },
    ];
    const it = await selectMenu({
      title: 'Max output tokens',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: presets.map(p => ({ id: p.id, label: p.label, tab: '', note: p.note })),
      currentId: String(cfg.max_output || 0),
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/maxout ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickBrain() {
    const notes = { on: 'memory recall + agent loop enabled', off: 'memory integration disabled' };
    const it = await selectMenu({
      title: 'Brain (memory)',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: ['on', 'off'].map(l => ({ id: l, label: l, tab: '', note: notes[l] })),
      currentId: cfg.brain ? 'on' : 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/brain ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function pickSync() {
    const notes = { on: 'session-mesh polling enabled', off: 'this session stops polling peers' };
    const it = await selectMenu({
      title: 'Session sync',
      hint: '↑↓/←→ move · enter apply · esc cancel',
      tabs: [],
      items: ['on', 'off'].map(l => ({ id: l, label: l, tab: '', note: notes[l] })),
      currentId: cfg.sync ? 'on' : 'off',
    });
    if (!it) { outLast('\x1b[90m  (unchanged)\x1b[0m'); return; }
    __chat_slash('/sync ' + it.id);
    try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
    refreshStatus('');
    maybeRefreshPanel();
  }
  async function main() {
    let me = '';
    try {
      const p0 = JSON.parse(__chat_poll());
      me = p0.me || '';
      if (p0.shared) lastShared = p0.shared;
      for (const n of (p0.notices || [])) out('\x1b[90m  ⇄ ' + n.id + ' · ' + n.kind + ': ' + n.text + '\x1b[0m');
    } catch (e) {}
    /* MCP is OPTIONAL and LAZY: no connect at startup — a slow/dead server
     * must never delay the prompt. The first /tools (or a tool-using agent
     * turn) connects on demand. */
    refreshStatus('');
    /* Real-time metrics: refresh the footer (ctx used + RAM) every 2s so
     * the numbers track the live process without waiting for input. The
     * repaint carries lastChip — the per-turn usage chip ("in→out tk · $")
     * must SURVIVE the timer, not vanish 2s after the turn ends. */
    const metricsTimer = setInterval(function() {
      try { refreshStatus(lastChip); } catch (e) {}
    }, 2000);
    const stopMetrics = () => { try { clearInterval(metricsTimer); } catch (e) {} };
    globalThis.__metrics_stop = stopMetrics;
    if (TTY) {
      /* Full-screen interface: take over the terminal. */
      try { __tui_on(); } catch (e) {}
      /* Bordered welcome panel (title + session facts) — built in Rust. */
      try { __chat_welcome(); } catch (e) {}
      /* Re-open the previous transcript (persisted in the session .qtsq). */
      try {
        const past = JSON.parse(__chat_past_turns());
        for (const t of past) {
          out('\x1b[' + T('accent') + 'm  ⏺ you\x1b[0m · ' + t.prompt);
          if (t.answer) out(t.answer);
        }
        if (past.length) out('');
      } catch (e) {}
    } else {
      /* Compact two-line banner (built in Rust, same content as the panel). */
      try { __chat_welcome(); } catch (e) {}
    }
    /* ── First-run onboarding: no usable model → force the guided setup
     * wizard (provider → URL → key → model list → pick). The wizard
     * persists the choice to ~/.sofuu/config.json, so it survives restarts
     * until the user changes it with /provider or /model. */
    if (!usableConfig()) {
      out('\x1b[90m  No usable model configured yet — set one up to start chatting.\x1b[0m');
      await providerWizard();
      try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
      refreshStatus('');
      if (usableConfig()) {
        out('\x1b[32m  ✓ Ready — ' + cfg.provider + ' · ' + cfg.model + '\x1b[0m\n');
      } else {
        out('\x1b[90m  Setup skipped — run /provider any time to configure a model.\x1b[0m\n');
      }
      maybeRefreshPanel();
    }
    while (!__chat_exit_check()) {
      let notices = [];
      try {
        const poll = JSON.parse(__chat_poll());
        notices = poll.notices || [];
        if (poll.shared) lastShared = poll.shared;
      } catch (e) {}
      for (const n of notices) out('\x1b[90m  ⇄ ' + n.id + ' · ' + n.kind + ': ' + n.text + '\x1b[0m');
      /* F8: filesystem watcher — poll for changes each tick. */
      try {
        const watchList = JSON.parse(__chat_watch_list());
        if (watchList.length > 0) {
          const changes = JSON.parse(__chat_watch_poll ? __chat_watch_poll() : '[]');
          for (const c of changes) {
            out('\x1b[90m  📂 ' + c.path + ' (' + c.kind + ')\x1b[0m');
            watchChangesPending.push(c);
          }
          /* M1: hard cap — the pending queue must never grow unbounded in
           * a long /watch session (turn() drains it each turn anyway). */
          if (watchChangesPending.length > 100) {
            watchChangesPending.splice(0, watchChangesPending.length - 100);
          }
        }
      } catch (e) {}
      const inp = await __readline('\x1b[' + T('accent') + 'm❯\x1b[0m ');
      if (inp === null || inp === undefined) { requestExit(); break; }
      const t = inp.trim();
      if (t === '') continue;
      if (t[0] === '/') {
        /* Snapshot the settings this command could touch, so the panel
         * refresh below can be gated on an ACTUAL change rather than on the
         * command's name. */
        const before = { p: cfg.provider, m: cfg.model, e: cfg.effort,
                         x: cfg.ctx_window, o: cfg.max_output, b: cfg.brain,
                         ml: cfg.ml, rl: cfg.rlm, sy: cfg.sync, th: cfg.theme,
                         pm: cfg.permissions, api: cfg.api_key, bu: cfg.base_url };
        const r = __chat_slash(t);
        // Slash commands may have changed provider/model/api_key — re-read
        // the Rust-side config so the live session uses the new values.
        try { cfg = JSON.parse(__chat_getcfg()); } catch (e) {}
        /* A theme switch must recolor the very next paint, not next
         * session: drop the cached palette on every command (re-read is
         * one JSON parse per command — noise). */
        PAL = null;
        /* Any command that can change settings re-renders the welcome panel
         * so the header (model/provider/effort) updates instantly. */
        const cfgCmds = ['/model', '/provider', '/effort', '/ctx', '/maxout', '/brain', '/ml', '/rlm', '/sync', '/outputs', '/mode', '/plan', '/edit', '/full', '/theme'];
        const isCfgCmd = cfgCmds.some(c => t === c || t.indexOf(c + ' ') === 0);
        /* ...but only repaint when something really moved. Gating on the
         * name alone re-printed the whole banner for a bare `/ctx`, which
         * changes nothing and can never open its picker without a TTY — so
         * piping a session showed the header twice. Under a TTY the repaint
         * is in place (tui_reset) and harmless; in a pipe it APPENDS, and
         * that duplicate is the bug. */
        const changed = isCfgCmd && (
             before.p !== cfg.provider  || before.m  !== cfg.model
          || before.e !== cfg.effort   || before.x  !== cfg.ctx_window
          || before.o !== cfg.max_output || before.b !== cfg.brain
          || before.ml !== cfg.ml      || before.rl !== cfg.rlm
          || before.sy !== cfg.sync    || before.pm !== cfg.permissions
          || before.th !== cfg.theme
          || before.api !== cfg.api_key || before.bu !== cfg.base_url);
        if (isCfgCmd) {
          /* A mode change must reach the engine runtime, not just the panel:
           * applyMode() re-pushes cfg.permissions into sofuu.agent so the
           * next turn is gated immediately, and refreshStatus re-renders
           * the amber mode chip. */
          applyMode();
          refreshStatus(lastChip);
        }
        if (changed) maybeRefreshPanel();
        /* F2 detect-on-select: a model/provider switch is exactly when the
         * user needs the NEW model's real window, and the 7-day discovery
         * TTL usually has nothing cached for a root just selected. Force
         * the harvest for the active endpoint and report what it found —
         * a silent 32k for a 1M model (or vice versa) is the bug this
         * removes. */
        const reselect = t === '/model' || t.indexOf('/model ') === 0 ||
                         t === '/provider' || t.indexOf('/provider ') === 0;
        if (reselect && (r === 'ok' || r === 'pick_model' || r === 'pick_provider')) {
          await detectCapsNow();
        }
        if (r === 'pick_model') {
          if (TTY) await pickModel();
          else out('\x1b[90m  Usage: /model <name>  (the picker needs a TTY)\x1b[0m\n');
        } else if (r === 'pick_provider') {
          if (TTY) await pickProvider();
          else out('\x1b[90m  Usage: /provider <name> [url]  (the picker needs a TTY)\x1b[0m\n');
        } else if (r === 'pick_effort') {
          if (TTY) await pickEffort();
          else out('\x1b[90m  Usage: /effort <off|low|medium|high|max>  (current: ' + (cfg.effort || 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_rlm') {
          if (TTY) await pickRlm();
          else out('\x1b[90m  Usage: /rlm <on|off|auto>  (current: ' + (cfg.rlm || 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_ctx') {
          if (TTY) await pickCtx();
          else out('\x1b[90m  Usage: /ctx <tokens|default>  (0-' + (cfg.ctx_cap || 0) + '; current: ' + (cfg.ctx_window || 'provider default') + ')\x1b[0m\n');
        } else if (r === 'pick_maxout') {
          if (TTY) await pickMaxout();
          else out('\x1b[90m  Usage: /maxout <tokens|default>  (0-' + (cfg.max_output_cap || 0) + '; current: ' + (cfg.max_output || 'provider default') + ')\x1b[0m\n');
        } else if (r === 'pick_brain') {
          if (TTY) await pickBrain();
          else out('\x1b[90m  Usage: /brain <on|off|info>  (current: ' + (cfg.brain ? 'on' : 'off') + ')\x1b[0m\n');
        } else if (r === 'brain_info') {
          handleBrainInfo();
        } else if (r === 'pick_sync') {
          if (TTY) await pickSync();
          else out('\x1b[90m  Usage: /sync <on|off>  (current: ' + (cfg.sync ? 'on' : 'off') + ')\x1b[0m\n');
        } else if (r === 'pick_theme') {
          if (TTY) { await pickTheme(); }
          else {
            let themes = [];
            try { themes = JSON.parse(__chat_themes()); } catch (e) {}
            out('\x1b[90m  Usage: /theme <name>  (current: ' + (cfg.theme || 'sofuu') + ')\x1b[0m');
            for (const th of themes) {
              const cur = th.name === cfg.theme ? ' \x1b[32m← current\x1b[0m' : '';
              out('  \x1b[36m' + th.name + '\x1b[0m' + cur);
            }
            out('');
          }
        } else if (r === 'clear') {
          history = [];
          /* /compact resets the meter (below) — /clear must too, or the
           * footer shows stale tokens until the next turn re-syncs (P2-9).
           * The calibration stays: the per-request overhead is a property
           * of the provider, not of the (now empty) history. */
          usedCtx = ctxMeter();
          out('\x1b[90m  ✓ session cleared\x1b[0m\n');
        }
        else if (r === 'compact') { await compact(); }
        else if (r === 'tools') { await connectMcpServers(); showTools(); }
        else if (r === 'agents') { await handleAgentsCmd(); }
        // F1–F11 feature dispatch
        else if (r === 'remember') { await handleRemember(t.replace(/^\/remember\s*/, '').trim()); }
        else if (r === 'why') { handleWhy(); }
        else if (r === 'memory') { await handleMemory(t.replace(/^\/memory\s*/, '').trim()); }
        else if (r === 'forget') { await handleForget(t.replace(/^\/forget\s*/, '').trim()); }
        else if (r === 'resume') { await handleResume(t.replace(/^\/resume\s*/, '').trim()); }
        else if (r === 'sessions') { await handleSessions(t.replace(/^\/sessions\s*/, '').trim()); }
        else if (r === 'share') { await handleShare(t.replace(/^\/share\s*/, '').trim()); }
        else if (r === 'import') { await handleImport(t.replace(/^\/import\s*/, '').trim()); }
        else if (r === 'cost') { handleCost(); }
        else if (r === 'outputs') { handleOutputs(t.replace(/^\/outputs\s*/, '').trim()); }
        else if (r === 'watch') {
          var watchArg = t.replace(/^\/watch\s*/, '').trim();
          handleWatch(watchArg);
        }
        // 'unknown' prints its message (with did-you-mean) in Rust
        // handle_slash — nothing more to show here.
        continue;
      }
      /* One logical user turn may have several provider attempts. Keep one
       * stable turn number, but give every retry its own archive attempt and
       * allow at most one narrowly selected recovery lookup. */
      archiveTurn++;
      archiveAttempt = 1;
      archivePendingRecovery = '';
      archiveLastFailure = null;
      archiveRecoveryUsed = false;
      /* A failed turn must never kill the chat: print the error and loop
       * back to the prompt. (Without this, an API/config exception rejects
       * the driver's main() promise and the CLI exits immediately.)
       *
       * Thinking-support runtime detection: when the provider rejects a
       * reasoning parameter for this model, remember it (persisted), tell
       * the user, and retry the turn once WITHOUT effort. */
      const THINK_ERR_RE = /thinking|reasoning_effort|extended.?thinking|reasoning/i;
      /* Provider failover (P1): capacity/auth-class failures (429 / 5xx /
       * rate-limit / 401/403 — a bad key is specific to that provider
       * entry) fall through to the next configured provider; validation
       * errors (400) stay fatal. One no-effort retry per model — the
       * reasoning-parameter rejection is per-model, persisted via
       * __chat_no_think. */
      const CAPACITY_ERR_RE = /HTTP (40[13]|429|5\d\d)|rate.?limit|overloaded|temporarily|quota|invalid token|unauthorized|forbidden|api key/i;
      const chain = failoverChain();
      const noThinkTried = {};
      let chainIdx = 0;
      let turnErr = null;
      function prepareArchiveRecovery(message) {
        if (!archiveLastFailure || archiveRecoveryUsed) return;
        archiveRecoveryUsed = true;
        const ids = [];
        if (archiveLastFailure.errorId) ids.push(archiveLastFailure.errorId);
        if (archiveLastFailure.partialId) ids.push(archiveLastFailure.partialId);
        const recovered = archiveContext({
          turn: archiveTurn,
          attempt: archiveAttempt,
          error_code: archiveLastFailure.errorCode || archiveErrorCode(message),
          related_output_ids: ids,
          current_task: t,
          automatic: true,
        });
        if (recovered) archivePendingRecovery = recovered;
      }
      for (;;) {
        const cur = chain[chainIdx];
        const curModel = cur ? cur.model : cfg.model;
        try {
          await turn(t, cur, archiveTurn, archiveAttempt);
          turnErr = null;
          break;
        } catch (e) {
          failoverTo = null;
          turnErr = e;
          const msg = String((e && e.message) || e);
          /* Setup/MCP/hook failures can happen before the agent's run
           * boundary. Preserve those too, while keeping the archive path
           * non-fatal to the normal retry loop. */
          if (!archiveLastFailure) {
            const errorCode = archiveErrorCode(msg);
            const errorRef = archiveWrite('error', 'failed',
              errorCode === 'runtime' ? 'runtime' : 'provider', msg, {
                error_code: errorCode,
                turn_complete: false,
              });
            archiveLastFailure = { errorId: errorRef && errorRef.id, partialId: null, errorCode: errorCode };
          }
          if (THINK_ERR_RE.test(msg) && cfg.effort && curModel &&
              !noThinkTried[curModel] &&
              !(cfg.no_think_models || []).includes(curModel)) {
            noThinkTried[curModel] = true;
            try { __chat_no_think(curModel); } catch (eNT) {}
            /* session-2 e2e catch: __chat_no_think only updates the Rust
             * config — this JS mirror still showed an empty
             * no_think_models, so the "no-effort retry" re-sent
             * reasoning_effort and hit the same rejection (the retry was a
             * no-op on the wire; every later turn repeated it too).
             * Reload the mirror, exactly like the provider wizard does. */
            try { cfg = JSON.parse(__chat_getcfg()); } catch (eCFG) {}
            out('\x1b[33m  ⓘ ' + curModel + ' rejected thinking (' + msg.slice(0, 160) +
                ') — disabling effort for this model and retrying.\x1b[0m\n');
            prepareArchiveRecovery(msg);
            archiveAttempt++;
            continue;
          }
          if (CAPACITY_ERR_RE.test(msg) && chainIdx + 1 < chain.length) {
            chainIdx++;
            const next = chain[chainIdx];
            out('\x1b[33m  ⓘ ' + (chain[chainIdx - 1] ? chain[chainIdx - 1].name : cfg.provider) +
                ' unavailable (' + msg.slice(0, 120) + ') — failing over to ' +
                next.name + ' / ' + next.model + '\x1b[0m\n');
            prepareArchiveRecovery(msg);
            archiveAttempt++;
            continue;
          }
          break;
        }
      }
      failoverTo = null;
      archiveLastFailure = null;
      if (turnErr) out('\x1b[31m  ✗ ' + String((turnErr && turnErr.message) || turnErr) + '\x1b[0m\n');
    }
    /* Brain flush happens per-store inside sofuu.agent.run (agent.js). */
    /* Disconnect every MCP server: their child processes hold ref'd libuv
     * handles, which would pin the event loop and hang the exit. */
    for (const c of mcpClients) { try { c.client.disconnect(); } catch (e) {} }
    /* Stop the metrics timer (it would pin the loop). */
    try { if (globalThis.__metrics_stop) globalThis.__metrics_stop(); } catch (e) {}
    /* Leave the full-screen interface: restore the previous terminal. */
    if (TTY) {
      try { __tui_off(); } catch (e) {}
      process.stdout.write('\n');
    }
  }
  return main();
})()
