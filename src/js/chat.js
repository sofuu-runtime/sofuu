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
 * Session mesh: format-compatible with crates/sofuu-core/src/session.rs —
 * plaintext registry.json + per-session .qtsq (same fnv1a64 password), so
 * TUI and desktop sessions of one project see each other.
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
    cfg: null,                /* parsed ~/.sofuu/config.json (re-read per turn) */
    history: [],              /* [{role, content}] re-sent each turn */
    streaming: false,
    onEvent: null,            /* the live turn's event sink */
    info: null,               /* registry entry for this session */
    session: null,            /* SessionData mirror (persisted to the .qtsq) */
    pendingApprovals: {},     /* id → {resolve, tool} */
    alwaysAllowed: {},        /* tool name → true (session-scoped) */
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
  function loadConfig() {
    try {
      var raw = (typeof __session_read_file === 'function') ? __session_read_file(configPath()) : null;
      var c = raw ? JSON.parse(raw) : {};
      if (!c || typeof c !== 'object') c = {};
      /* Flat mirror from the active provider entry when the flat fields
       * are empty (same discipline as the Rust config load) — host UIs
       * edit the providers list. */
      if (Array.isArray(c.providers) && c.providers.length) {
        var active = null;
        for (var i = 0; i < c.providers.length; i++) {
          if (c.providers[i] && c.providers[i].name === c.active) { active = c.providers[i]; break; }
        }
        if (!active) active = c.providers[0];
        if (active) {
          if (!c.provider) c.provider = active.name || '';
          if (!c.model) c.model = active.model || '';
          if (!c.base_url) c.base_url = active.endpoint || '';
          if (!c.profile) c.profile = active.profile || '';
          if (!c.api_key) c.api_key = active.api_key || '';
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
        var c = JSON.parse(sofuu.ai.modelCaps(cfg().model || ''));
        return (c && c.known) ? c : null;
      }
    } catch (e) {}
    return null;
  }
  function ctxWindow() {
    var c = cfg();
    return c.ctx_window > 0
      ? c.ctx_window
      : ((modelCaps() || {}).ctxWindow > 0
        ? modelCaps().ctxWindow
        : ({ openai: 128000, anthropic: 128000, local: 32768 }[c.provider] || 32768));
  }
  function ctxBudget() { return Math.floor(ctxWindow() * 0.85); }
  function attachTokenBudget() {
    return Math.min(65536, Math.max(2048, Math.floor(ctxWindow() * 0.25)));
  }
  function estTok(s) {
    if (sofuu.ai && typeof sofuu.ai.estimateTokens === 'function') {
      try { return sofuu.ai.estimateTokens(s) | 0; } catch (e) {}
    }
    return Math.ceil(String(s).length / 4);
  }
  function historyTokens() {
    var n = 0;
    for (var i = 0; i < S.history.length; i++) n += estTok(String(S.history[i].content || ''));
    return n;
  }

  function streamOpts() {
    var c = cfg();
    var o = { messages: [], provider: c.provider, model: c.model };
    if (c.effort) o.effort = c.effort;
    if (c.api_key) o.api_key = c.api_key;
    if (c.base_url) o.base_url = c.base_url;
    if (c.profile) o.profile = c.profile;
    if (c.max_output > 0) o.max_tokens = c.max_output;
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
   * the summarizer failed — callers fall back to dropping, never block. */
  async function summarizeHistory(keepTurns) {
    var keepMsgs = Math.min(keepTurns * 2, S.history.length);
    var oldCount = S.history.length - keepMsgs;
    if (oldCount <= 0) return false;
    var summary = await complete([
      { role: 'system', content: 'You are a conversation summarizer. Compress the following conversation into a compact summary that preserves key facts, decisions, and the user\'s intent. Output only the summary.' }
    ].concat(S.history.slice(0, oldCount)));
    if (!summary || typeof summary !== 'string' || !summary.trim() || summary.trim() === '(no response)') return false;
    S.history = [{ role: 'system', content: 'Prior conversation summary: ' + summary }].concat(S.history.slice(oldCount));
    return true;
  }
  async function trimHistory() {
    if (S.history.length > MAX_HISTORY_ENTRIES) {
      S.history.splice(0, S.history.length - MAX_HISTORY_ENTRIES);
    }
    var budget = ctxBudget();
    if (!S.autoCompactArmed && historyTokens() < budget * 0.5) S.autoCompactArmed = true;
    if (S.autoCompactArmed && historyTokens() > budget * COMPACT_AT && S.history.length > 2) {
      S.autoCompactArmed = false;
      var savedTk = historyTokens();
      try {
        if (await summarizeHistory(0)) {
          emit({ kind: 'warn', payload: { message: 'auto-compacted → summary (' +
                 Math.max(0, savedTk - historyTokens()) + ' tk saved, window → ~0)' } });
        }
      } catch (e) { /* summarizer failed — the drop loop below still guards */ }
    }
    /* Drop-oldest guard: still over budget after compaction (or it failed). */
    var dropped = 0;
    while (S.history.length > 2 && historyTokens() > budget) {
      S.history.splice(0, 2);
      dropped++;
    }
    if (dropped > 0) {
      emit({ kind: 'warn', payload: { message: 'history trimmed by ' + dropped +
             ' turn' + (dropped === 1 ? '' : 's') + ' to fit the context budget' } });
    }
  }

  /* ── MCP tool wiring (ported from the DRIVER; lazy, never fatal) ── */
  function mcpServersList() {
    try {
      var raw = (typeof __session_read_file === 'function')
        ? __session_read_file(home() + '/.sofuu/mcp.json') : null;
      if (!raw) return [];
      var list = JSON.parse(raw);
      return Array.isArray(list) ? list : [];
    } catch (e) { return []; }
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
    read_file: 1, grep: 1, glob: 1, list_dir: 1,
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
    var id = 'ap-' + S.approvalSeq + '-' + Date.now().toString(36);
    return new Promise(function (resolve) {
      S.pendingApprovals[id] = { resolve: resolve, tool: tool };
      emit({ kind: 'approval_request',
             payload: { id: id, tool: tool, args: args || {}, risky: true } });
      /* Hard ceiling so a vanished host can't pin the turn forever —
       * init gives the desktop a human-scale value (1h). */
      setTimeout(function () {
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
    if (always && allow) S.alwaysAllowed[p.tool] = true;
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
    var m;
    while ((m = re.exec(text)) !== null) {
      mentions.push(m[1] ? m[1].slice(1, -1) : m[2]);
    }
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
      }
      var content = '';
      try {
        /* Reject paths escaping the project root (documented contract). */
        var normalized = pathPart.charAt(0) === '/' ? pathPart : (cwd + '/' + pathPart);
        var underRoot = normalized === cwd || normalized.indexOf(cwd + '/') === 0;
        if (pathPart.indexOf('..') >= 0 || !underRoot) {
          blocks.push('### ' + pathPart + '\n```\n(rejected: path escapes project root)\n```');
          manifestParts.push('@' + pathPart + ' (rejected)');
          continue;
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
  function sessionsDir(project) { return project + '/.sofuu/sessions'; }
  function sessionFile(project, id) { return sessionsDir(project) + '/' + id + '.qtsq'; }
  function registryRead(project) {
    try {
      var raw = (typeof __session_read_file === 'function')
        ? __session_read_file(sessionsDir(project) + '/registry.json') : null;
      if (!raw) return { sessions: [] };
      var reg = JSON.parse(raw);
      if (!reg || !Array.isArray(reg.sessions)) return { sessions: [] };
      return reg;
    } catch (e) { return { sessions: [] }; }
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
      if (typeof __session_write_file === 'function') {
        __session_write_file(sessionsDir(project) + '/registry.json', JSON.stringify(reg, null, 2));
      }
    } catch (e) {}
  }
  function sessionPersist() {
    if (!S.session || !S.project) return;
    var d = S.session;
    if (d.events.length > 300) d.events.splice(0, d.events.length - 300);
    if (d.notes.length > 20) d.notes.length = 20;
    try {
      if (typeof __qtsq_session_save === 'function') {
        __qtsq_session_save(sessionFile(S.project, d.id), JSON.stringify(d), sessionPassword(S.project));
      }
    } catch (e) {}
  }
  function sessionLog(kind, text) {
    if (!S.session) return;
    S.session.events.push({ t: now(), kind: kind, text: String(text == null ? '' : text) });
    if (S.info) S.info.last_seen = now();
    sessionPersist();
  }
  function sessionJoin(project, model, provider) {
    var id = genSessionId();
    var cwd = '';
    try { cwd = process.cwd(); } catch (e) {}
    S.info = {
      id: id, pid: pid(), host: 'desktop', cwd: cwd,
      model: model || '', provider: provider || '',
      started_at: now(), last_seen: now(), task: null, ended: false,
    };
    S.session = {
      schema: 1, id: id, created: now(), host: 'desktop', cwd: cwd,
      model: model || '', provider: provider || '',
      task: null, notes: [], events: [],
    };
    /* Registry first: its __session_write_file mkdirs .sofuu/sessions/,
     * which the QTSQ codec (no mkdir of its own) needs for the .qtsq. */
    registryUpsert(project, S.info);
    sessionLog('start', 'session started');
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
  function loadSessionData(project, id) {
    try {
      if (typeof __qtsq_session_load !== 'function') return null;
      var raw = __qtsq_session_load(sessionFile(project, id), sessionPassword(project));
      if (!raw) return null;
      return JSON.parse(raw);
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
  async function turn(text, forceNoThink, noTools) {
    var c = cfg();
    if (!usableConfig()) {
      throw new Error('No usable model configured — open Settings → Providers to set one up');
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
     * agent directly (no chat history injected). */
    var typed = text;
    var agentMention = null;
    var tt = text.trim();
    var mm = /^@agent:([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/.exec(tt);
    if (!mm) mm = /^@([A-Za-z0-9_.-]+)(?::\s|\s+)([\s\S]+)$/.exec(tt);
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
    sessionLog('prompt', String(typed));
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
      max_tokens: c.max_output > 0 ? c.max_output : undefined,
      embed_provider: c.embed_provider || undefined,
      embed_model: c.embed_model || undefined,
      memory: c.brain ? 'shared' : 'off',
      ml: c.ml ? 'on' : 'off',
      rlm: c.rlm === 'on' ? 'on' : (c.rlm === 'auto' ? 'auto' : 'off'),
      ctx_window: c.ctx_window > 0 ? c.ctx_window : undefined,
      recallMin: c.recall_min > 0 ? c.recall_min : undefined,
      recallBudget: c.recall_budget > 0 ? c.recall_budget : undefined,
      /* Human-scale tool waits when the gate is on (approvals block inside
       * execute; agent.js's default 30s wrapper would kill them). */
      toolTimeoutMs: S.toolTimeoutMs,
      budget: { maxSteps: 8, maxDepth: 1, maxTokens: 1e9, maxWallMs: 1e9 },
    };
    var onStep = function (e) {
      if (!e || typeof e.kind !== 'string') return;
      /* Run-terminal duplicates — chat.js emits its own 'done'. */
      if (e.kind === 'answer' || e.kind === 'stop') return;
      emit(normalizeStep(e));
    };
    var res = agentMention
      ? await sofuu.agent.run(agentMention.name, turnText, { signal: 'chat', onStep: onStep })
      : await sofuu.agent.run(def, turnText, {
          history: S.history, signal: 'chat', onStep: onStep,
          shared: '', watched: '',
        });
    var answer = String((res && res.answer) || '(no response)');
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
    /* Persist the turn to this session's .qtsq + heartbeat the registry. */
    sessionLog('answer', answer);
    registryUpsert(S.project, S.info);
    /* History stores the manifest only (not full file text). */
    var histText = manifest ? text + ' [' + manifest + ']' : text;
    S.history.push({ role: 'user', content: histText }, { role: 'assistant', content: answer });
    await trimHistory();
    /* One full GC per completed turn (bounds JS garbage per turn). */
    try { if (typeof __sofuu_gc === 'function') __sofuu_gc(); } catch (e) {}
    return {
      answer: answer,
      usage: u,
      steps: (res && res.steps) || 0,
      stopped: (res && res.stopped) || null,
      costUsd: costUsd,
    };
  }

  /* ── Public API ──────────────────────────────────────────────────── */
  function setProjectInternal(project) {
    sessionEnd(); /* close the previous session cleanly, if any */
    /* Tool jail follows the project, then derive the mesh root EXACTLY as
     * session.rs::project_root does ($SOFUU_PROJECT → git toplevel → cwd,
     * canonicalized): the fnv1a64 session password hashes that string, so
     * any divergence (e.g. /tmp vs /private/tmp symlinks) would make the
     * TUI and desktop unable to read each other's .qtsq files. */
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
    var c = cfg();
    sessionJoin(S.project, c.model || '', c.provider || '');
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
    var project = String(opts.project || '');
    if (!project) {
      try { project = process.cwd(); } catch (e) { project = '.'; }
    }
    setProjectInternal(project);
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
    S.onEvent = typeof opts.onEvent === 'function' ? opts.onEvent : null;
    S.streaming = true;
    emit({ kind: 'start', payload: { sessionId: S.info ? S.info.id : null } });
    var THINK_ERR_RE = /thinking|reasoning_effort|extended.?thinking|reasoning/i;
    var noTools = !!opts.noTools;
    try {
      var res;
      var txt = String(text == null ? '' : text);
      try {
        res = await turn(txt, false, noTools);
      } catch (e) {
        /* Thinking-support runtime detection (same as the TUI): the
         * provider rejected a reasoning parameter → remember it and retry
         * once WITHOUT effort. */
        var msg = String((e && e.message) || e);
        var c = cfg();
        if (THINK_ERR_RE.test(msg) && c.effort && c.model &&
            S.noThink.indexOf(c.model) < 0 &&
            (c.no_think_models || []).indexOf(c.model) < 0) {
          S.noThink.push(c.model);
          emit({ kind: 'warn', payload: { message: c.model + ' rejected thinking (' +
                 clip(msg, 160) + ') — retrying without effort' } });
          res = await turn(txt, true, noTools);
        } else {
          throw e;
        }
      }
      var u = res.usage || {};
      emit({ kind: 'done', payload: {
        answer: res.answer,
        stopped: res.stopped,
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
    var data = loadSessionData(S.project, String(id || ''));
    if (!data) return { ok: false, error: 'session not found (or QTSQ unavailable)' };
    var turns = turnsFromData(data, S.pastTurns);
    if (!turns.length) return { ok: false, error: 'session has no turns to resume' };
    S.history = [];
    for (var i = 0; i < turns.length; i++) {
      S.history.push({ role: 'user', content: turns[i].prompt });
      S.history.push({ role: 'assistant', content: turns[i].answer });
    }
    /* Continue the resumed session: new turns append to its transcript. */
    S.session = data;
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
    var data = loadSessionData(S.project, String(id || ''));
    return data ? turnsFromData(data, S.pastTurns) : [];
  }

  function newSession() {
    S.history = [];
    S.alwaysAllowed = {};
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
    S.autoCompactArmed = true;
    setProjectInternal(path);
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

  function state() {
    var c = cfg();
    var pending = 0;
    for (var k in S.pendingApprovals) pending++;
    return {
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
      permissions: S.permissions,
      permissionProfile: S.permissionProfile,
      pendingApprovals: pending,
    };
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
    sessionTurns: sessionTurns,
    newSession: newSession,
    setProject: setProject,
    setPermissions: setPermissions,
  };
})();
