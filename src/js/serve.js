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

  function makeError(code, msg, details) {
    var err = new Error(msg);
    err.name = 'SofuuError';
    err.code = code || 'ERR_SERVE';
    err.rule = 'RULES.md §3.5';
    if (details) err.details = details;
    return err;
  }

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

  /* Host mode (`sofuu host`): a chat-backed session service for remote
   * clients. This is deliberately separate from the OpenAI-shaped H3
   * sessions in S.sessions — host mode is gated by cfg.hostMode, so a plain
   * `sofuu serve` keeps its H3 contract untouched.
   *
   *   H.ring[sid]  per-session replay ring (bounded, in-memory)
   *   H.subs[sid]  live SSE subscribers
   *   H.run        the single active chat turn (chat.js is one global turn)
   *
   * The ring exists so a client that loses its socket can resume from its
   * cursor without replaying the whole transcript. It is intentionally NOT
   * the on-disk events.ndjson: chat.js already owns that file's seq space. */
  var H = {
    ready: false,
    /* Identifies this process's ring sequence space. A client stores the
     * epoch with its cursor and sends it back on reconnect; a host restart
     * resets `seq` to 1, and without the epoch a stale cursor that happens
     * to land inside the new ring would silently skip the events it never
     * saw. */
    epoch: Date.now().toString(36) + '-' + Math.floor(Math.random() * 1679616).toString(36),
    ring: Object.create(null),
    subs: Object.create(null),
    run: null,
    runSeq: 0,
    pairPin: '',
    pairExpiresAt: 0,
    pairAttempts: 0,
    pairLockedUntil: 0,
    devices: null,
    heartbeat: null,
  };
  var HOST_RING_MAX = 2000;
  var HOST_HEARTBEAT_MS = 15000;
  var HOST_PAIR_TTL_MS = 10 * 60 * 1000;
  var HOST_PAIR_MAX_ATTEMPTS = 5;
  var HOST_PAIR_LOCK_MS = 60 * 1000;

  /* Stabilization Freeze Pass 2.3 — bounded streaming text buffer.
   * SSE responses write each delta straight to the socket, but the
   * non-streaming JSON path must assemble the whole answer. Appending each
   * token to one growing string allocates a larger string per token; this
   * coalesces into ~64 KiB chunks and joins once. */
  function textChunkBuffer(capBytes) {
    var cap = capBytes > 0 ? capBytes : 64 * 1024;
    var chunks = [];
    var current = '';
    var size = 0;
    return {
      push: function (s) {
        if (s === undefined || s === null || s === '') return;
        s = String(s);
        size += s.length;
        if (current && current.length + s.length >= cap) {
          chunks.push(current);
          current = s;
        } else {
          current += s;
        }
      },
      length: function () { return size; },
      text: function () {
        if (current) {
          chunks.push(current);
          current = '';
        }
        return chunks.join('');
      },
    };
  }

  /* Pass 2.4 — one collection at a completed request boundary. The response
   * has already been written, so this can only release transient buffers. */
  function reclaimRequestMemory() {
    try { if (typeof __sofuu_gc === 'function') __sofuu_gc(); }
    catch (eGc) { /* a host without GC still serves; the bound is a courtesy. */ }
  }

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
      /* P4: server-wide default for agent runs that do not name a mode. */
      approvalMode: str(opts.approvalMode) || 'deny',
      /* SV-02: see the /health route — set by the Rust boot call from
       * --public-health. */
      publicHealth: opts.publicHealth === true,
      /* Host mode (`sofuu host`). Off for plain `sofuu serve`, so the H3
       * route contract does not move under existing clients. */
      hostMode: opts.hostMode === true,
      hostName: str(opts.hostName) || 'sofuu-host',
      workspace: str(opts.workspace) || str(opts.project) || '',
      stateRoot: str(opts.stateRoot),
      pairPin: str(opts.pairPin),
      approvalTimeoutMs: num(opts.approvalTimeoutMs, 15 * 60 * 1000),
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
    } catch (e) { /* a policy document that does not parse falls back to '{}', which policyVerdict reads as no rule. */ }
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
    } catch (e) { /* an unreadable verdict is treated as no verdict below; the caller decides, not this parse. */ }
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
      } catch (e) { /* a query parameter that will not decode is skipped; the rest of the query still parses. */ }
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

  /* ══════════════════════════════════════════════════════════════════
   * Host mode (`sofuu host`) — chat-backed sessions for remote clients.
   * The chat engine itself is sofUU.chat (src/js/chat.js); this layer owns
   * the replay ring, detached turns, SSE subscribers, pairing and device
   * tokens. Every route below is gated on cfg.hostMode so `sofuu serve`
   * keeps its H3 contract.
   * ══════════════════════════════════════════════════════════════════ */

  function hostHome() {
    try { return String(process.env.HOME || '.') + '/.sofuu/state'; }
    catch (e) { return '.sofuu/state'; }
  }

  /* The paired-device store is a 0600 atomic sofuu.state file. Device
   * tokens are stored ONLY as SHA-256 hashes — a leaked state file cannot
   * be replayed as a bearer credential. */
  function hostDevices(cfg) {
    if (H.devices) return H.devices;
    try {
      if (typeof sofuu === 'undefined' || !sofuu.state ||
          typeof sofuu.state.open !== 'function') return null;
      H.devices = sofuu.state.open('paired_devices', {
        root: cfg.stateRoot || hostHome(),
      });
    } catch (eDev) {
      H.devices = null;
    }
    return H.devices;
  }

  function hostTokenHash(token) {
    try { return 'sha256:' + sofuu.crypto.sha256('sofuu-device:' + String(token)); }
    catch (e) { return ''; }
  }

  function hostDeviceList(cfg) {
    var store = hostDevices(cfg);
    if (!store) return [];
    var out = [];
    try {
      var keys = store.keys('dev:') || [];
      for (var i = 0; i < keys.length; i++) {
        var rec = store.get(keys[i]);
        if (!rec) continue;
        out.push({
          id: rec.id, name: rec.name || 'device', platform: rec.platform || '',
          created_at: rec.created_at || 0, last_seen: rec.last_seen || 0,
        });
      }
    } catch (eList) { /* an unreadable device entry is omitted, never fatal */ }
    out.sort(function (a, b) { return (b.last_seen || 0) - (a.last_seen || 0); });
    return out;
  }

  function hostDeviceLookup(cfg, token) {
    if (!cfg.hostMode || String(token).indexOf('dev_') !== 0) return null;
    var store = hostDevices(cfg);
    if (!store) return null;
    var hash = hostTokenHash(token);
    if (!hash) return null;
    try {
      var keys = store.keys('dev:') || [];
      for (var i = 0; i < keys.length; i++) {
        var rec = store.get(keys[i]);
        if (rec && ctEq(String(rec.token_hash || ''), hash)) {
          rec.last_seen = Math.floor(Date.now() / 1000);
          try { store.set(keys[i], rec); } catch (eTouch) { /* last_seen is advisory */ }
          return rec;
        }
      }
    } catch (eFind) { /* store failure fails closed: no device match */ }
    return null;
  }

  function hostMaster(req, cfg) {
    var auth = (req.headers && (req.headers.authorization || req.headers.Authorization)) || '';
    return (authMode() === 'none') || ctEq(auth, 'Bearer ' + cfg.token);
  }

  /* Master token OR a paired device token. */
  function hostAuthorized(req, cfg) {
    if (!cfg.hostMode) {
      if (authMode() === 'none') return true;
      var a0 = (req.headers && (req.headers.authorization || req.headers.Authorization)) || '';
      return ctEq(a0, 'Bearer ' + cfg.token);
    }
    if (hostMaster(req, cfg)) return true;
    var auth = (req.headers && (req.headers.authorization || req.headers.Authorization)) || '';
    if (auth.indexOf('Bearer dev_') !== 0) return false;
    return !!hostDeviceLookup(cfg, auth.slice(7));
  }

  function hostMintToken() {
    var bytes = new Uint8Array(32);
    var hex = '';
    try {
      sofuu.crypto.randomFill(bytes);
      for (var i = 0; i < bytes.length; i++) hex += (bytes[i] < 16 ? '0' : '') + bytes[i].toString(16);
    } catch (e) {
      for (var j = 0; j < 32; j++) hex += Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
    }
    return 'dev_' + hex;
  }

  function hostPair(cfg, data) {
    var nowMs = Date.now();
    if (!cfg.hostMode) return { status: 404, body: { error: { message: 'not a host', code: 'not_found' } } };
    if (nowMs < H.pairLockedUntil) {
      return { status: 429, body: { error: { message: 'too many pairing attempts — wait a minute', code: 'pair_locked' } } };
    }
    if (!H.pairPin || nowMs > H.pairExpiresAt) {
      return { status: 410, body: { error: { message: 'pairing PIN expired (restart `sofuu host --pair`)', code: 'pair_expired' } } };
    }
    var pin = String(data.pin || data.code || '');
    if (!ctEq(pin, H.pairPin)) {
      H.pairAttempts++;
      if (H.pairAttempts >= HOST_PAIR_MAX_ATTEMPTS) {
        H.pairAttempts = 0;
        H.pairLockedUntil = nowMs + HOST_PAIR_LOCK_MS;
      }
      return { status: 401, body: { error: { message: 'invalid pairing PIN', code: 'pair_invalid' } } };
    }
    var store = hostDevices(cfg);
    if (!store) {
      return { status: 503, body: { error: { message: 'paired-device store unavailable in this build', code: 'pair_store_unavailable' } } };
    }
    H.pairAttempts = 0;
    var token = hostMintToken();
    var id = 'dev-' + (typeof sofuu.crypto.randomUUID === 'function'
      ? sofuu.crypto.randomUUID() : String(nowMs) + '-' + Math.random().toString(36).slice(2, 10));
    var devName = String(data.name || data.device || 'Android').slice(0, 80);
    var platform = String(data.platform || 'android').slice(0, 40);
    var rec = {
      id: id, name: devName, platform: platform,
      token_hash: hostTokenHash(token),
      created_at: Math.floor(nowMs / 1000),
      last_seen: Math.floor(nowMs / 1000),
    };
    try { store.set('dev:' + id, rec); }
    catch (eSet) {
      return { status: 500, body: { error: { message: 'could not persist the paired device', code: 'pair_store_failed' } } };
    }
    var st = hostChatState();
    return {
      status: 200,
      body: {
        ok: true, token: token,
        device: { id: id, name: devName, platform: platform },
        host: {
          name: cfg.hostName, version: cfg.version,
          workspace: cfg.workspace || (st && st.project) || '',
          model: (st && st.model) || '', provider: (st && st.provider) || '',
        },
      },
    };
  }

  /* ── replay ring + subscribers ─────────────────────────────────── */

  function hostRing(sid) {
    var r = H.ring[sid];
    if (!r) r = H.ring[sid] = { buf: [], next: 1 };
    return r;
  }

  function hostRingLatest(sid) {
    var r = H.ring[sid];
    return r ? r.next - 1 : 0;
  }

  function hostRingOldest(sid) {
    var r = H.ring[sid];
    return (r && r.buf.length) ? r.buf[0].seq : (r ? r.next : 1);
  }

  function hostDropSub(sid, sub) {
    var list = H.subs[sid];
    if (!list) return;
    for (var i = list.length - 1; i >= 0; i--) {
      if (list[i] === sub) list.splice(i, 1);
    }
    if (list.length === 0) delete H.subs[sid];
  }

  function hostRingAppend(sid, kind, payload, runId) {
    var r = hostRing(sid);
    /* Snapshot the payload: a replay hours later must not show a shape the
     * engine later mutated in place. Falls back to the live reference if
     * the payload is not JSON-clean (a ring event is still better than a
     * dropped one). */
    var body;
    try { body = JSON.parse(JSON.stringify(payload || {})); }
    catch (eCopy) { body = payload || {}; }
    var ev = {
      kind: String(kind || 'message'),
      payload: body,
      seq: r.next++,
      sid: sid,
      ts: Date.now(),
    };
    if (runId) ev.run_id = runId;
    r.buf.push(ev);
    if (r.buf.length > HOST_RING_MAX) r.buf.splice(0, r.buf.length - HOST_RING_MAX);
    var subs = H.subs[sid];
    if (subs) {
      for (var i = subs.length - 1; i >= 0; i--) {
        var sub = subs[i];
        try {
          if (typeof sub.closed === 'function' && sub.closed()) {
            subs.splice(i, 1);
            continue;
          }
          sseData(sub, ev);
        } catch (eWrite) {
          /* A dead socket must never stall the turn. The heartbeat would
           * prune it too, but doing it here keeps the fan-out small. */
          subs.splice(i, 1);
        }
      }
      if (subs.length === 0) delete H.subs[sid];
    }
    return ev;
  }

  function hostHeartbeat() {
    for (var sid in H.subs) {
      var subs = H.subs[sid];
      if (!subs) continue;
      for (var i = subs.length - 1; i >= 0; i--) {
        var sub = subs[i];
        try {
          if (typeof sub.closed === 'function' && sub.closed()) {
            subs.splice(i, 1);
            continue;
          }
          sub.write(': ping ' + Date.now() + '\n\n');
        } catch (e) {
          subs.splice(i, 1);
        }
      }
      if (subs.length === 0) delete H.subs[sid];
    }
  }

  function hostStartHeartbeat() {
    if (H.heartbeat) return;
    H.heartbeat = setInterval(hostHeartbeat, HOST_HEARTBEAT_MS);
    /* The heartbeat is a server-lifetime timer; it must not be the reason
     * a finished script cannot exit. */
    try { if (H.heartbeat && typeof H.heartbeat.unref === 'function') H.heartbeat.unref(); }
    catch (eUnref) { /* a host without unref keeps the timer; the server owns the loop anyway. */ }
  }

  /* ── chat engine bridge ────────────────────────────────────────── */

  function hostChatState() {
    try { return sofuu.chat.state() || {}; }
    catch (e) { return {}; }
  }

  function hostChatInit(cfg) {
    if (!cfg.hostMode || H.ready) return H.ready;
    var project = cfg.workspace || cfg.project || '';
    try {
      sofuu.chat.init({
        host: cfg.hostName || 'sofuu-host',
        project: project,
        permissions: 'prompt',
        /* Desktop-style approvals: the phone may be asleep when a tool
         * asks, so the ceiling is measured in minutes, not seconds. */
        permissionProfile: 'prompt',
        toolTimeoutMs: Math.max(60000, cfg.approvalTimeoutMs | 0),
      });
      H.ready = true;
    } catch (eInit) {
      H.ready = false;
      console.error('host: chat.init failed: ' + String((eInit && eInit.message) || eInit));
    }
    return H.ready;
  }

  function hostResume(sid) {
    var st = hostChatState();
    if (st.sessionId && String(st.sessionId) === sid) return { ok: true, sessionId: sid, active: true };
    try {
      var r = sofuu.chat.resume(sid) || { ok: false, error: 'session not found' };
      return r;
    } catch (e) {
      return { ok: false, error: String((e && e.message) || e) };
    }
  }

  function hostSessions() {
    var out = [];
    try {
      var r = sofuu.chat.sessions();
      var list = (r && r.sessions) || [];
      for (var i = 0; i < list.length; i++) {
        var s = list[i];
        if (!s || !s.id) continue;
        out.push({
          id: s.id,
          host: 'host',
          cwd: s.cwd || '',
          model: s.model || '',
          provider: s.provider || '',
          started_at: s.started_at | 0,
          last_seen: s.last_seen | 0,
          task: s.task || null,
          ended: s.ended === true,
        });
      }
    } catch (e) { /* an unreadable registry lists nothing; the engine stays usable */ }
    out.sort(function (a, b) { return (b.last_seen || 0) - (a.last_seen || 0); });
    return out;
  }

  function hostSessionExists(sid) {
    var st = hostChatState();
    if (st.sessionId && String(st.sessionId) === sid) return true;
    var list = hostSessions();
    for (var i = 0; i < list.length; i++) if (list[i].id === sid) return true;
    return false;
  }

  function hostSnapshot(sid) {
    var turns = [];
    try { turns = sofuu.chat.sessionTurns(sid) || []; } catch (e) { turns = []; }
    var st = hostChatState();
    var active = !!(st.sessionId && String(st.sessionId) === sid);
    var runDone = !!(H.run && H.run.sid === sid && H.run.sawDone);
    var pending = [];
    if (active && st.streaming && typeof sofuu.chat.pendingApprovals === 'function') {
      try { pending = sofuu.chat.pendingApprovals() || []; } catch (eP) { pending = []; }
    }
    var run = null;
    if (H.run && H.run.sid === sid) {
      run = { runId: H.run.runId, startedAt: H.run.startedAt };
    }
    return {
      ok: true,
      sessionId: sid,
      epoch: H.epoch,
      turns: turns,
      latestSeq: hostRingLatest(sid),
      oldestSeq: hostRingOldest(sid),
      streaming: !!(active && st.streaming && !runDone),
      activeRun: run,
      pendingApprovals: pending,
      /* Seed for a client that reconnects mid-turn: the deltas it missed
       * are not in the persisted turns yet. A turn whose `done` is already
       * in the ring must not seed a second bubble. */
      partialAnswer: (H.run && H.run.sid === sid && !runDone) ? H.run.partial : '',
      config: {
        model: st.model || '', provider: st.provider || '',
        effort: st.effort || '',
        permissions: st.permissionProfile || st.permissions || 'prompt',
        contextTokens: st.contextTokens | 0,
        contextWindow: st.contextWindow | 0,
        sessionTokens: st.sessionTokens | 0,
        sessionSpendUsd: Number(st.sessionSpendUsd) || 0,
      },
    };
  }

  function hostInfo(cfg) {
    var st = hostChatState();
    return {
      ok: true,
      host: true,
      name: cfg.hostName,
      version: cfg.version,
      workspace: cfg.workspace || st.project || '',
      model: st.model || '',
      provider: st.provider || '',
      permissions: st.permissionProfile || st.permissions || 'prompt',
      streaming: !!st.streaming,
      activeSessionId: st.sessionId || null,
      activeRun: H.run ? { runId: H.run.runId, sessionId: H.run.sid, startedAt: H.run.startedAt } : null,
      approvalTimeoutMs: cfg.approvalTimeoutMs,
      uptimeMs: S.startedAtMs ? (Date.now() - S.startedAtMs) : 0,
      pairedDevices: hostDeviceList(cfg).length,
    };
  }

  /* One turn per host process: sofuu.chat is a single global engine. The
   * HTTP response returns 202 immediately; the turn runs detached and its
   * events land in the ring whether or not a socket is attached. */
  function hostStartRun(cfg, sid, data) {
    if (H.run) {
      return { status: 409, body: { error: { message: 'a turn is already running', code: 'turn_in_progress', runId: H.run.runId } } };
    }
    var st = hostChatState();
    if (st.streaming) {
      return { status: 409, body: { error: { message: 'a turn is already running', code: 'turn_in_progress' } } };
    }
    var resumed = hostResume(sid);
    if (!resumed || resumed.ok !== true) {
      return { status: 404, body: { error: { message: String((resumed && resumed.error) || "no session '" + sid + "'"), code: 'session_not_found' } } };
    }
    var prompt = String(data.prompt === undefined ? (data.text === undefined ? '' : data.text) : data.prompt);
    if (!prompt.trim()) {
      return { status: 400, body: { error: { message: 'prompt is required', code: 'invalid_prompt' } } };
    }
    var images = Array.isArray(data.images)
      ? data.images.filter(function (u) { return typeof u === 'string' && u.indexOf('data:image/') === 0; }).slice(0, 8)
      : [];
    H.runSeq++;
    var runId = 'run_' + Date.now().toString(36) + '_' + H.runSeq;
    var run = H.run = { runId: runId, sid: sid, startedAt: Date.now(), sawDone: false, partial: '' };
    var sink = function (ev) {
      if (!ev || typeof ev !== 'object') return;
      if (H.run !== run) return;
      /* The engine's own `sid` is its project id, not the chat session id;
       * overwrite it so a multiplexing client can route the event. */
      if (ev.kind === 'answer_delta' && ev.payload && typeof ev.payload.text === 'string') {
        run.partial += ev.payload.text;
        /* Bounded: a runaway stream must not pin an unbounded string that
         * also rides every snapshot. 256 KiB is far past any UI need. */
        if (run.partial.length > 262144) run.partial = run.partial.slice(-262144);
      }
      var rec = hostRingAppend(sid, ev.kind, ev.payload, runId);
      if (ev.kind === 'done') run.sawDone = true;
      return rec;
    };
    var promise;
    try {
      promise = sofuu.chat.submit(prompt, {
        noTools: data.noTools === true || data.mode === 'chat',
        images: images,
        onEvent: sink,
      });
    } catch (eStart) {
      var em = String((eStart && eStart.message) || eStart);
      H.run = null;
      hostRingAppend(sid, 'done', { error: { message: em }, usage: null }, runId);
      var code = (eStart && eStart.code) || '';
      return { status: code === 'ERR_CONCURRENT_TURN' ? 409 : 500, body: { error: { message: em, code: code || 'submit_failed' } } };
    }
    Promise.resolve(promise).then(function (r) {
      if (H.run !== run) return;
      if (!run.sawDone) {
        hostRingAppend(sid, 'done', {
          answer: (r && r.answer) || '',
          stopped: !!(r && r.stopped),
          usage: (r && r.usage) || null,
        }, runId);
      }
      H.run = null;
    }, function (err) {
      if (H.run !== run) return;
      if (!run.sawDone) {
        hostRingAppend(sid, 'done', {
          error: { message: String((err && err.message) || err) },
          usage: null,
        }, runId);
      }
      H.run = null;
    });
    return {
      status: 202,
      body: { ok: true, sessionId: sid, runId: runId, seq: hostRingLatest(sid) },
    };
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
      try { if (sofuu.fs && await sofuu.fs.exists(want)) return want; } catch (e) { /* exists() throwing is the same answer as false here: fall through to the tokenizer probe. */ }
    }
    try {
      if (sofuu.fs && await sofuu.fs.exists(want + '/tokenizer.json')) return want;
    } catch (e2) { /* same as above: a probe that throws resolves to not-a-model-directory, and the caller reports its own error. */ }
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
    } catch (e0) { /* the vector-space probe is optional; dimension and id keep their defaults. */ }
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
      while (true) {
        var n = await iter.next();
        if (n.done) break;
        var chunk = n.value || {};
        /* Thinking stays off the content channel (OpenAI has no field for
         * it) and the final message does not include it either. It used to
         * accumulate into a growing string for no reader; not accumulating
         * it is the Pass 2.3 bound. */
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
    var textBuf = textChunkBuffer(64 * 1024);
    var sf2 = stopFilter(stops, function (t) { textBuf.push(t); });
    while (true) {
      var n2 = await iter.next();
      if (n2.done) break;
      var ch = n2.value || {};
      if (ch.text && sf2.push(ch.text)) break;
    }
    sf2.flush();
    var text = textBuf.text();
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

    var textBuf = textChunkBuffer(64 * 1024);
    var sf2 = stopFilter(stops, function (t) { textBuf.push(t); });
    while (true) {
      var n2 = await iter.next();
      if (n2.done) break;
      if (n2.value && n2.value.text && sf2.push(n2.value.text)) break;
    }
    sf2.flush();
    var text = textBuf.text();
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
    var approval = null;
    var opts = {
      provider: 'local',
      model: modelPath,
      /* P4 async HITL: opt in explicitly with {"approvalMode":"async"} (or a
       * server-wide config). The token rides this authenticated response /
       * SSE stream only; it is never written to the event log. */
      approvalMode: str(data.approvalMode) || cfg.approvalMode || 'deny',
      onApprovalRequest: function (req) {
        approval = req;
        if (wantStream) {
          try {
            sseData(res, {
              run_id: req.runId, awaiting_approval: {
                run_id: req.runId, approval_id: req.approvalId,
                approval_token: req.approvalToken, tool: req.tool,
                args: req.args, expires_at: new Date(req.expiresAt).toISOString()
              }
            });
          } catch (eW) { /* the checkpoint is durable; the client can poll the run id. */ }
        }
      },
      onStep: function (ev) {
        events.push(ev);
        if (wantStream) {
          try { sseData(res, { run_id: id, event: ev }); } catch (eW) { /* the client hung up mid-run: the step is already recorded in events[], so the run result is unaffected. */ }
        }
      },
    };
    if (wantStream) sseHead(res);
    try {
      var result = await sofuu.agent.run(def, task, opts);
      var realId = (result && result.runId) || id;
      var awaiting = null;
      if (result && result.stopped === 'awaiting_approval' && approval) {
        awaiting = {
          run_id: realId, approval_id: approval.approvalId,
          approval_token: approval.approvalToken, tool: approval.tool,
          args: approval.args,
          expires_at: new Date(approval.expiresAt).toISOString()
        };
      }
      if (wantStream) {
        sseData(res, {
          run_id: realId,
          done: true,
          answer: result && result.answer,
          usage: (result && result.usage) || null,
          steps: (result && result.steps) || events.length,
          stopped: !!(result && result.stopped),
          awaiting_approval: awaiting || undefined,
        });
        res.write('data: [DONE]\n\n');
        res.end();
      } else {
        send(res, 200, {
          run_id: realId,
          answer: result && result.answer,
          usage: (result && result.usage) || null,
          steps: (result && result.steps) || events.length,
          stopped: !!(result && result.stopped),
          awaiting_approval: awaiting || undefined,
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

  /* ── host routes ─────────────────────────────────────────────── */

  function hostDecode(s) {
    try { return decodeURIComponent(s); } catch (e) { return s; }
  }

  /* A paired-device token is a chat credential, not a second master key.
   * Everything it may call is a host-mode route below; the model,
   * completion, embedding, agent and device-admin surfaces stay
   * master-only. Kept as a path allowlist (not a prefix test) so a new
   * route is closed to devices until it is deliberately opened here. */
  function hostDeviceRouteAllowed(method, path) {
    if (method === 'GET' && path === '/v1/host') return true;
    if (path === '/v1/sessions') return method === 'GET' || method === 'POST';
    var m = /^\/v1\/sessions\/([^/]+)(?:\/([^/]+))?(?:\/([^/]+))?(?:\/([^/]+))?$/.exec(path);
    if (!m) return false;
    var a = m[2] || '';
    var b = m[3] || '';
    var c = m[4] || '';
    if (a === '' && (method === 'GET' || method === 'DELETE')) return true;
    if (a === 'snapshot' || a === 'turns') return method === 'GET' && b === '';
    if (a === 'events') return method === 'GET' && b === '';
    if (a === 'runs') {
      if (b === '') return method === 'POST';
      return method === 'POST' && b !== '' && c === 'cancel';
    }
    if (a === 'approve' || a === 'permissions' || a === 'compact') {
      return method === 'POST' && b === '';
    }
    return false;
  }

  async function hostRoute(cfg, req, res, method, path) {
    if (!hostChatInit(cfg)) {
      fail(res, 503, 'the chat engine failed to initialize on this host', 'server_error', 'chat_unavailable');
      return true;
    }

    /* GET /v1/host — identity + live posture (the phone's "connected" card). */
    if (method === 'GET' && path === '/v1/host') {
      send(res, 200, hostInfo(cfg));
      return true;
    }

    /* Paired devices: master token only. A device token must not be able to
     * list or revoke its siblings. */
    if (method === 'GET' && path === '/v1/host/devices') {
      if (!hostMaster(req, cfg)) { fail(res, 403, 'the master token is required', 'permission_error', 'master_required'); return true; }
      send(res, 200, { ok: true, devices: hostDeviceList(cfg) });
      return true;
    }
    var hostDevMatch = /^\/v1\/host\/devices\/([^/]+)$/.exec(path);
    if (method === 'DELETE' && hostDevMatch) {
      if (!hostMaster(req, cfg)) { fail(res, 403, 'the master token is required', 'permission_error', 'master_required'); return true; }
      var devId = hostDecode(hostDevMatch[1]);
      var devStore = hostDevices(cfg);
      var removed = false;
      try { removed = !!(devStore && devStore.delete('dev:' + devId)); } catch (eDel) { removed = false; }
      if (!removed) { fail(res, 404, "no paired device '" + devId + "'", 'invalid_request_error', 'device_not_found'); return true; }
      send(res, 200, { ok: true, id: devId, revoked: true });
      return true;
    }

    /* Rotate the pairing PIN without restarting the host. */
    if (method === 'POST' && path === '/v1/host/pin') {
      if (!hostMaster(req, cfg)) { fail(res, 403, 'the master token is required', 'permission_error', 'master_required'); return true; }
      var pb = readJson(req);
      if (!pb.ok) { fail(res, 400, pb.error, 'invalid_request_error', 'invalid_json'); return true; }
      var newPin = String((pb.data && pb.data.pin) || '');
      if (!/^\d{4,8}$/.test(newPin)) newPin = String(Math.floor(100000 + Math.random() * 900000));
      H.pairPin = newPin;
      H.pairExpiresAt = Date.now() + HOST_PAIR_TTL_MS;
      H.pairAttempts = 0;
      H.pairLockedUntil = 0;
      send(res, 200, { ok: true, pin: newPin, expiresAt: H.pairExpiresAt });
      return true;
    }

    /* Chat sessions (NOT the H3 process-local sessions). */
    if (method === 'GET' && path === '/v1/sessions') {
      send(res, 200, { ok: true, sessions: hostSessions() });
      return true;
    }
    if (method === 'POST' && path === '/v1/sessions') {
      /* chat.newSession() resets the engine's history/session pointers.
       * Doing that under a live turn would write the turn's completion
       * into the wrong transcript, so a host exposes it only when idle. */
      if (H.run) {
        fail(res, 409, 'a turn is running — create the session after it finishes',
          'invalid_request_error', 'turn_in_progress');
        return true;
      }
      try {
        var created = sofuu.chat.newSession() || {};
        send(res, 200, {
          ok: true,
          sessionId: created.sessionId || null,
          session: created.sessionId ? (hostSessions().filter(function (s) { return s.id === created.sessionId; })[0] || null) : null,
        });
      } catch (eNew) {
        var nCode = String((eNew && eNew.code) || '');
        fail(res, nCode === 'ERR_CHAT_LIMIT' ? 409 : 500,
          String((eNew && eNew.message) || eNew), 'invalid_request_error', nCode || 'new_session_failed');
      }
      return true;
    }

    var hostSidMatch = /^\/v1\/sessions\/([^/]+)$/.exec(path);
    if (hostSidMatch) {
      var sid0 = hostDecode(hostSidMatch[1]);
      if (method === 'GET') {
        if (!hostSessionExists(sid0)) { fail(res, 404, "no session '" + sid0 + "'", 'invalid_request_error', 'session_not_found'); return true; }
        var infoList = hostSessions();
        var info = null;
        for (var ii = 0; ii < infoList.length; ii++) if (infoList[ii].id === sid0) { info = infoList[ii]; break; }
        send(res, 200, {
          ok: true, session: info,
          latestSeq: hostRingLatest(sid0), oldestSeq: hostRingOldest(sid0),
        });
        return true;
      }
      if (method === 'DELETE') {
        if (!hostSessionExists(sid0)) { fail(res, 404, "no session '" + sid0 + "'", 'invalid_request_error', 'session_not_found'); return true; }
        if (H.run && H.run.sid === sid0) {
          fail(res, 409, 'a turn is running — cancel it before deleting the session',
            'invalid_request_error', 'turn_in_progress');
          return true;
        }
        var delRes = {};
        try { delRes = sofuu.chat.deleteSession(sid0) || {}; }
        catch (eDel2) {
          fail(res, 500, String((eDel2 && eDel2.message) || eDel2), 'server_error', 'delete_failed');
          return true;
        }
        delete H.ring[sid0];
        delete H.subs[sid0];
        send(res, 200, { ok: true, sessionId: sid0, deleted: true, wasActive: !!delRes.wasActive });
        return true;
      }
    }

    var hostSnapMatch = /^\/v1\/sessions\/([^/]+)\/snapshot$/.exec(path);
    if (method === 'GET' && hostSnapMatch) {
      var snapSid = hostDecode(hostSnapMatch[1]);
      if (!hostSessionExists(snapSid)) { fail(res, 404, "no session '" + snapSid + "'", 'invalid_request_error', 'session_not_found'); return true; }
      send(res, 200, hostSnapshot(snapSid));
      return true;
    }

    var hostTurnsMatch = /^\/v1\/sessions\/([^/]+)\/turns$/.exec(path);
    if (method === 'GET' && hostTurnsMatch) {
      var turnsSid = hostDecode(hostTurnsMatch[1]);
      if (!hostSessionExists(turnsSid)) { fail(res, 404, "no session '" + turnsSid + "'", 'invalid_request_error', 'session_not_found'); return true; }
      var turnsOut = [];
      try { turnsOut = sofuu.chat.sessionTurns(turnsSid) || []; } catch (eT) { turnsOut = []; }
      send(res, 200, { ok: true, sessionId: turnsSid, turns: turnsOut });
      return true;
    }

    /* GET /v1/sessions/:id/events?after_seq=N&follow=1
     * SSE: a sync header, an optional replay from the ring, then live
     * events. A cursor outside the ring gets `sync/type=reset` and the
     * client refetches /snapshot. */
    var hostEvMatch = /^\/v1\/sessions\/([^/]+)\/events$/.exec(path);
    if (method === 'GET' && hostEvMatch) {
      var evSid = hostDecode(hostEvMatch[1]);
      if (!hostSessionExists(evSid)) { fail(res, 404, "no session '" + evSid + "'", 'invalid_request_error', 'session_not_found'); return true; }
      var q = parseQuery(req.url);
      var afterRaw = q.after_seq !== undefined ? q.after_seq : q.afterSeq;
      var afterSeq = null;
      if (afterRaw !== undefined && String(afterRaw).length > 0) {
        var parsedSeq = parseInt(String(afterRaw), 10);
        if (!isFinite(parsedSeq) || parsedSeq < 0) {
          fail(res, 400, 'after_seq must be a non-negative integer', 'invalid_request_error', 'invalid_after_seq', 'after_seq');
          return true;
        }
        afterSeq = parsedSeq;
      }
      var follow = !(String(q.follow) === '0' || String(q.follow) === 'false');
      /* A cursor is only meaningful inside the ring that issued it. The
       * client echoes the epoch it stored with its cursor; a mismatch
       * means the host restarted (or another process owns the port now)
       * and the seq must not be trusted, even if it happens to be in
       * range. */
      var clientEpoch = q.epoch !== undefined ? String(q.epoch) : '';
      var epochMismatch = clientEpoch !== '' && clientEpoch !== H.epoch;
      var ring = hostRing(evSid);
      var latest = ring.next - 1;
      var oldest = ring.buf.length ? ring.buf[0].seq : ring.next;
      sseHead(res);
      if (afterSeq === null) {
        sseData(res, {
          kind: 'sync', sid: evSid, seq: latest,
          payload: { type: 'ready', epoch: H.epoch, latestSeq: latest, oldestSeq: oldest },
        });
      } else if (epochMismatch || afterSeq < oldest - 1 || afterSeq > latest) {
        sseData(res, {
          kind: 'sync', sid: evSid, seq: latest,
          payload: {
            type: 'reset', epoch: H.epoch, latestSeq: latest, oldestSeq: oldest,
            afterSeq: afterSeq, staleEpoch: epochMismatch,
          },
        });
      } else {
        sseData(res, {
          kind: 'sync', sid: evSid, seq: latest,
          payload: { type: 'replay', epoch: H.epoch, latestSeq: latest, oldestSeq: oldest, afterSeq: afterSeq },
        });
        for (var ri = 0; ri < ring.buf.length; ri++) {
          if (ring.buf[ri].seq > afterSeq) sseData(res, ring.buf[ri]);
        }
      }
      if (!follow) { try { res.end(); } catch (eEnd) { /* already closed */ } return true; }
      var subList = H.subs[evSid];
      if (!subList) subList = H.subs[evSid] = [];
      subList.push(res);
      hostStartHeartbeat();
      return true;
    }

    var hostRunMatch = /^\/v1\/sessions\/([^/]+)\/runs$/.exec(path);
    if (method === 'POST' && hostRunMatch) {
      var runSid = hostDecode(hostRunMatch[1]);
      if (!hostSessionExists(runSid)) { fail(res, 404, "no session '" + runSid + "'", 'invalid_request_error', 'session_not_found'); return true; }
      var rb = readJson(req);
      if (!rb.ok) { fail(res, 400, rb.error, 'invalid_request_error', 'invalid_json'); return true; }
      var started = hostStartRun(cfg, runSid, rb.data || {});
      if (started.status >= 400) {
        var rerr = started.body.error || {};
        fail(res, started.status, rerr.message || 'could not start the turn',
          started.status === 409 ? 'invalid_request_error' : 'server_error',
          rerr.code || 'run_failed');
      } else {
        send(res, 202, started.body);
      }
      return true;
    }

    var hostCancelMatch = /^\/v1\/sessions\/([^/]+)\/runs\/([^/]+)\/cancel$/.exec(path);
    if (method === 'POST' && hostCancelMatch) {
      var cSid = hostDecode(hostCancelMatch[1]);
      var cRun = hostDecode(hostCancelMatch[2]);
      if (H.run && H.run.sid === cSid && (cRun === H.run.runId || cRun === 'current')) {
        try { sofuu.chat.cancel(); } catch (eC) { /* nothing running is the common case */ }
        send(res, 200, { ok: true, sessionId: cSid, runId: H.run.runId, cancelled: true });
      } else if (!H.run || H.run.sid !== cSid) {
        send(res, 200, { ok: true, sessionId: cSid, runId: cRun, cancelled: false, reason: 'no_active_run' });
      } else {
        fail(res, 404, "run '" + cRun + "' is not active on this session", 'invalid_request_error', 'run_not_found');
      }
      return true;
    }

    var hostApproveMatch = /^\/v1\/sessions\/([^/]+)\/approve$/.exec(path);
    if (method === 'POST' && hostApproveMatch) {
      var aSid = hostDecode(hostApproveMatch[1]);
      if (!hostSessionExists(aSid)) { fail(res, 404, "no session '" + aSid + "'", 'invalid_request_error', 'session_not_found'); return true; }
      var ab = readJson(req);
      if (!ab.ok) { fail(res, 400, ab.error, 'invalid_request_error', 'invalid_json'); return true; }
      var adata = ab.data || {};
      var aid = String(adata.id || adata.approvalId || '');
      var allow = adata.allow === true || adata.decision === 'allow';
      if (!aid) { fail(res, 400, 'approval id is required', 'invalid_request_error', 'invalid_approval_id', 'id'); return true; }
      var ar = {};
      try { ar = sofuu.chat.resolveApproval(aid, allow, adata.always === true) || {}; }
      catch (eA) { ar = { ok: false, error: String((eA && eA.message) || eA) }; }
      if (ar.ok === false) {
        fail(res, 404, String(ar.error || 'no pending approval with that id'), 'invalid_request_error', 'approval_not_found');
      } else {
        send(res, 200, { ok: true, id: aid, allow: allow, always: adata.always === true });
      }
      return true;
    }

    var hostPermMatch = /^\/v1\/sessions\/([^/]+)\/permissions$/.exec(path);
    if (method === 'POST' && hostPermMatch) {
      var pSid = hostDecode(hostPermMatch[1]);
      if (!hostSessionExists(pSid)) { fail(res, 404, "no session '" + pSid + "'", 'invalid_request_error', 'session_not_found'); return true; }
      var permBody = readJson(req);
      if (!permBody.ok) { fail(res, 400, permBody.error, 'invalid_request_error', 'invalid_json'); return true; }
      var perm = String((permBody.data && (permBody.data.profile || permBody.data.permissions)) || 'prompt');
      var permRes = sofuu.chat.setPermissions(perm) || {};
      send(res, 200, { ok: true, sessionId: pSid, permissions: permRes.permissions || perm });
      return true;
    }

    var hostCompactMatch = /^\/v1\/sessions\/([^/]+)\/compact$/.exec(path);
    if (method === 'POST' && hostCompactMatch) {
      var coSid = hostDecode(hostCompactMatch[1]);
      if (!hostSessionExists(coSid)) { fail(res, 404, "no session '" + coSid + "'", 'invalid_request_error', 'session_not_found'); return true; }
      if (H.run) { fail(res, 409, 'a turn is running — compact after it finishes', 'invalid_request_error', 'turn_in_progress'); return true; }
      var cRes = {};
      try { cRes = await sofuu.chat.compact() || {}; }
      catch (eComp) { cRes = { ok: false, error: String((eComp && eComp.message) || eComp) }; }
      if (cRes.ok === false) { fail(res, 400, String(cRes.error || 'compact failed'), 'invalid_request_error', 'compact_failed'); }
      else send(res, 200, { ok: true, sessionId: coSid, savedTokens: cRes.savedTokens || 0 });
      return true;
    }

    return false;
  }

  /* ── the handler ────────────────────────────────────────────── */

  function createHandler(opts) {
    var cfg = normalize(opts);

    return async function handler(req, res) {
      /* Pass 2.4: only routes that can allocate turn-sized state reclaim at
       * their boundary. Health probes must stay allocation-free and fast. */
      var heavy = false;
      try {
        var path = String(req.url || '/').split('?')[0];
        var method = String(req.method || 'GET').toUpperCase();
        heavy = path.indexOf('/v1/') === 0
          || path.indexOf('/webhooks/') === 0
          || path === '/chat/completions' || path === '/completions'
          || path === '/infer' || path === '/remember' || path === '/recall'
          || path === '/memories' || path === '/forget' || path === '/share';

        /* Liveness. /healthz and /livez are the probe endpoints: public and
         * data-free (no version, no uptime, no counts).
         * /health used to be public too, so an authenticated server answered
         * one route without the token and two shipped tests disagreed about
         * the contract (serve_test H2 vs brain_auth_e2e). The boundary is now
         * uniform: with a token configured, /health needs it. */
        if (method === 'GET' && (path === '/healthz' || path === '/livez')) {
          send(res, 200, { ok: true, status: 'ok' });
          return;
        }

        if (method === 'GET' && path === '/health') {
          /* SV-02 (AUDIT-2026-11-08): container probes only know /health, so
           * an operator running behind k8s/compose passes --public-health and
           * gets the same data-free answer the /healthz probe already gives.
           * Everything else stays behind the token. */
          if (authMode() !== 'none' && !cfg.publicHealth) {
            var hAuth = (req.headers && (req.headers.authorization || req.headers.Authorization)) || '';
            if (!ctEq(hAuth, 'Bearer ' + cfg.token)) {
              fail(res, 401, 'missing or invalid bearer token', 'authentication_error', 'invalid_api_key');
              return;
            }
          }
          var scouterStatus = null;
          try {
            if (typeof sofuu !== 'undefined' && sofuu.scouter) {
              scouterStatus = JSON.parse(sofuu.scouter.status());
            }
          } catch (eScouter) { /* advisory */ }
          send(res, 200, {
            ok: true,
            status: 'ok',
            uptime_ms: S.startedAtMs ? (Date.now() - S.startedAtMs) : 0,
            version: cfg.version,
            scouter: scouterStatus,
          });
          return;
        }

        /* Scouter Health Report endpoint */
        if (method === 'GET' && (path === '/scouter' || path === '/v1/scouter')) {
          /* No Scouter substrate means NO health verdict. The previous
           * fallback answered with a perfect synthetic report, which read
           * as a measured healthy context on a build that had never
           * measured anything. */
          var sc = null;
          if (typeof sofuu !== 'undefined' && sofuu.scouter && typeof sofuu.scouter.status === 'function') {
            try { sc = sofuu.scouter.status(); } catch (eScouter) { sc = null; }
          }
          if (typeof sc !== 'string' || !sc.trim()) {
            fail(res, 503, 'the Scouter substrate is unavailable in this build',
              'server_error', 'scouter_unavailable');
            return;
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(sc);
          return;
        }

        /* Capabilities manifest (R1.3, T1.2). */
        if (method === 'GET' && (path === '/capabilities' || path === '/v1/capabilities')) {
          var caps = typeof sofuu.capabilities === 'function' ? sofuu.capabilities() : '{}';
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(typeof caps === 'string' ? caps : JSON.stringify(caps));
          return;
        }

        /* Generalized messaging channels: WhatsApp Cloud API verification and
         * signed event delivery, generic webhook ingress, Telegram webhook
         * mode. These endpoints authenticate their own caller (HMAC over the
         * raw body, verify-token handshake, X-Telegram-Bot-Api-Secret-Token),
         * so an external provider — which cannot present the serve bearer —
         * can reach them. The driver fails closed when a secret is missing and
         * answers its own HTTP status; anything it does not own falls through
         * to the normal 404 after the bearer gate below. */
        if (path.indexOf('/webhooks/') === 0 &&
            typeof sofuu !== 'undefined' && sofuu.channels &&
            typeof sofuu.channels.handleWebhook === 'function') {
          var chHandled = await sofuu.channels.handleWebhook(req, res, path);
          if (chHandled) return;
        }

        /* Pairing is the ONE host route that cannot present a token yet
         * (the PIN is the credential); it is rate-limited and expiring. */
        if (cfg.hostMode && method === 'POST' && (path === '/v1/pair' || path === '/v1/host/pair')) {
          var pairBody = readJson(req);
          if (!pairBody.ok) { fail(res, 400, pairBody.error, 'invalid_request_error', 'invalid_json'); return; }
          var paired = hostPair(cfg, pairBody.data);
          if (paired.status >= 400) {
            var perf = paired.body.error || {};
            fail(res, paired.status, perf.message || 'pairing failed',
              paired.status === 401 ? 'authentication_error' : 'invalid_request_error',
              perf.code || 'pair_failed');
          } else {
            send(res, 200, paired.body);
          }
          return;
        }

        if (!hostAuthorized(req, cfg)) {
          fail(res, 401, 'missing or invalid bearer token', 'authentication_error', 'invalid_api_key');
          return;
        }

        /* Paired-device tokens are scoped to the chat surface. Without
         * this, a phone token could drive /v1/agent/runs or the OpenAI
         * completion routes with the Mac's credentials. */
        if (cfg.hostMode && !hostMaster(req, cfg) && !hostDeviceRouteAllowed(method, path)) {
          fail(res, 403, 'this token is scoped to host chat sessions',
            'permission_error', 'device_scope', null);
          return;
        }

        /* Policy sees every authenticated request; denies win (R7.6). */
        var verdict = policyVerdict(path, method);
        if (verdict.decision === 'deny') { failPolicy(res, verdict); return; }

        /* ── host mode (sofuu host) ──────────────────────────────── */
        if (cfg.hostMode) {
          var hostHandled = await hostRoute(cfg, req, res, method, path);
          if (hostHandled) return;
        }

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

        /* ── sessions (H3; host mode has its own chat-backed routes) ── */
        if (!cfg.hostMode && method === 'POST' && path === '/v1/sessions') {
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
        if (!cfg.hostMode && method === 'GET' && path === '/v1/sessions') {
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

        /* P4 async HITL: POST /v1/agent/runs/:id/approve
         * Body: { approvalId, approvalToken, decision: 'allow' | 'deny' }.
         * The token is validated in constant time by the agent driver; an
         * expired or consumed checkpoint fails closed. */
        var approveMatch = /^\/v1\/agent\/runs\/([^/]+)\/approve$/.exec(path);
        if (method === 'POST' && approveMatch) {
          var pbody = readJson(req);
          if (!pbody.ok) { fail(res, 400, pbody.error, 'invalid_request_error', 'invalid_json'); return; }
          var pdata = pbody.data || {};
          var prun = '';
          try { prun = decodeURIComponent(approveMatch[1]); }
          catch (eDec) { prun = approveMatch[1]; }
          try {
            var pr = await sofuu.agent.resume(prun, {
              approvalId: str(pdata.approvalId || pdata.approval_id),
              approvalToken: str(pdata.approvalToken || pdata.approval_token),
              decision: str(pdata.decision || (pdata.allow === true ? 'allow' : 'deny')),
            });
            send(res, 200, {
              ok: true, run_id: prun, resumed: true,
              answer: pr && pr.answer, stopped: pr && pr.stopped,
              usage: (pr && pr.usage) || null, steps: (pr && pr.steps) || 0,
            });
          } catch (eAp) {
            var apCode = String((eAp && eAp.code) || '');
            var apStatus = apCode === 'E_APPROVAL_EXPIRED' ? 410
              : (apCode === 'E_APPROVAL_TOKEN_INVALID' || apCode === 'E_DENIED') ? 403
              : apCode === 'E_CAPABILITY' ? 501 : 400;
            fail(res, apStatus, String((eAp && eAp.message) || eAp),
              apStatus === 403 ? 'authentication_error' : 'invalid_request_error',
              apCode || 'approval_failed');
          }
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
          } catch (eQ) { /* a malformed percent-escape in ?category= leaves queryCat empty, which means unfiltered. */ }
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
      } finally {
        if (heavy) reclaimRequestMemory();
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
        if (praw) { try { sofuu.policy.set(String(praw)); } catch (ePol) { /* an invalid policy file leaves the previous posture in force. */ } }
      }
    } catch (ePl) { /* loading the session policy is best effort: without it the default policy stays in force. */ }
    var auth = authMode();
    if (auth !== 'none' && !cfg.token) {
      throw makeError('ERR_AUTH_REQUIRED', 'serve: a bearer token is required (set --token / SOFUU_SERVE_TOKEN, or policy http.auth:"none" explicitly)');
    }
    if (auth === 'none') {
      console.error('\x1b[33m  ⚠ policy http.auth is "none": every /v1/* request is unauthenticated\x1b[0m');
    }
    S.cfg = cfg;
    S.startedAt = Math.floor(Date.now() / 1000);
    S.startedAtMs = Date.now();
    S.brain = brainOpen(cfg);
    if (cfg.hostMode) {
      /* Pairing PIN lives for ten minutes from boot (or the last rotation).
       * `--pair` generates it in Rust; an empty value means no pairing. */
      if (cfg.pairPin) {
        H.pairPin = cfg.pairPin;
        H.pairExpiresAt = Date.now() + HOST_PAIR_TTL_MS;
      }
      hostChatInit(cfg);
      hostStartHeartbeat();
    }
    var handler = createHandler(cfg);
    var server = sofuu.http.createServer(handler);
    server.listen(cfg.port, cfg.host);
    // listen() starts unref'd so a finished script can exit; the CLI must stay up.
    if (typeof server.ref === 'function') server.ref();
    S.server = server;
    S.started = true;
    var primary = cfg.models.length ? cfg.models[0].id : null;
    if (cfg.hostMode) {
      var hostSt = hostChatState();
      console.log('sofuu host "' + cfg.hostName + '" listening on http://' + cfg.host + ':' + cfg.port +
        ' (workspace: ' + (cfg.workspace || hostSt.project || '.') + ')');
      /* The PIN line is printed by the Rust CLI before boot (with the
       * "enter it in the app" hint); echoing it twice reads as two
       * different codes at a glance. */
    } else {
      console.log('brain server listening on http://' + cfg.host + ':' + cfg.port +
        (primary ? ' (model: ' + primary + ')' : ''));
    }
    return { ok: true, host: cfg.host, port: cfg.port, model: primary,
             hostMode: cfg.hostMode === true };
  }

  sofuu.serve.start = start;
  sofuu.serve.createHandler = createHandler;
  sofuu.serve.state = function () {
    return {
      started: S.started,
      sessions: Object.keys(S.sessions).length,
      hostMode: !!(S.cfg && S.cfg.hostMode),
      hostSessions: (S.cfg && S.cfg.hostMode) ? hostSessions().length : 0,
      activeRun: H.run ? { runId: H.run.runId, sessionId: H.run.sid } : null,
      subscribers: Object.keys(H.subs).reduce(function (n, k) { return n + (H.subs[k] ? H.subs[k].length : 0); }, 0),
    };
  };
  sofuu.serve.VERSION = '1';
})();
