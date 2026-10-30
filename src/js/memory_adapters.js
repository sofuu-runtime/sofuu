/* memory_adapters.js — the pluggable memory backend layer.
 * @sofuu-source-only — compiled ON FIRST USE, not at boot; NOT a shipped
 * .bc driver (scripts/gen-driver-bc.sh and scripts/check-driver-bc.sh skip
 * this marker). Loader: rt/memory_adapters.rs, `sofuu.memory.__loadAdapters`.
 *
 * PLAN-PLUGGABLE-MEMORY-ADAPTERS. Gives `sofuu.memory.open()` three handle
 * kinds behind one contract:
 *
 *   1. native QTSQ  — untouched, synchronous (the default, memory_api.js);
 *   2. a custom adapter object — any {remember, recall} plus optional
 *      open/delete/count/flush/close;
 *   3. a built-in REST adapter — qdrant | pinecone | pgvector, built on
 *      `fetch` alone (no native client, no binary growth).
 *
 * Remote handles are asynchronous: every method returns a promise, so
 * `await brain.remember(...)` reads identically on all three and the native
 * path needs no change. Both flat call forms keep working —
 * remember({vec, text}) and remember(vec, text, role, page, source).
 *
 * This file is ~30 KB of source. It used to live in memory_api.js and cost
 * 1.5 ms of parse on EVERY boot (`sofuu eval "1+1"` included); moving it
 * here and compiling it on first adapter use took boot back to baseline.
 * It builds against the tiny internal bridge memory_api.js exposes as
 * `sofuu.memory.__flat` (handle registry, typed-refusal helper, view test).
 */
(function () {
  'use strict';
  var S = globalThis.sofuu;
  if (!S) return;
  var M = S.memory;
  if (!M || typeof M.__installAdapters !== 'function') return;
  var F = M.__flat;
  if (!F) return;
  /* Bridge locals: the router's handle registry and its two guards, so the
   * flat API's handle numbering and typed refusals are the same here. */
  var live = F.live, refuse = F.refuse, isView = F.isView;

  /* ═══ Pluggable memory adapters ══════════════════════════════════════
   *
   * PLAN-PLUGGABLE-MEMORY-ADAPTERS. `open()` now returns one of three handle
   * kinds, all driven by the flat API above:
   *
   *   1. native QTSQ  — unchanged and synchronous (the default);
   *   2. a custom adapter object — any {remember, recall} the developer
   *      passes, plus optional open/delete/count/flush/close;
   *   3. a built-in REST adapter — qdrant | pinecone | pgvector, built on
   *      `fetch` alone (no native client, no binary growth).
   *
   * Remote handles are asynchronous: every method returns a promise, so
   * `await brain.remember(...)` reads identically on all three and the
   * native path needs no change. Both flat call forms keep working —
   * remember({vec, text}) and remember(vec, text, role, page, source).
   *
   * Nothing here runs at boot: `open()` is the only entry point, so an
   * engine that never opens an adapter pays parse time only.
   */

  var ADAPTERS = {};   /* id -> factory(uri, opts) */

  /* Instance methods the native brain has and the adapter contract does not.
   * They stay callable (the flat dispatcher addresses methods by name) but
   * answer a typed refusal instead of a bare TypeError. */
  var ADAPTER_UNSUPPORTED = [
    'list', 'export', 'importRecords', 'stats', 'rememberWorking',
    'promoteWorking', 'clearWorking', 'rememberEntity', 'rememberEntityTurn',
    'markPositive', 'decayTick', 'retain', 'retainAgainst', 'consolidate',
    'compact', 'kvHints'
  ];

  function memErr(code, message) { return refuse(Error, code, message); }
  function textOf(v) { return v === undefined || v === null ? '' : String(v); }
  function clipText(v) {
    var s = textOf(v);
    return s.length > 180 ? s.slice(0, 180) + '...' : s;
  }
  function listOf(v) {
    if (v === undefined || v === null) return [];
    if (Array.isArray(v)) return v;
    if (isView(v)) return Array.prototype.slice.call(v);
    return [v];
  }
  function asPlain(v) { return Array.prototype.slice.call(v); }
  function copyInto(dst, src) {
    for (var k in (src || {})) {
      if (Object.prototype.hasOwnProperty.call(src, k)) dst[k] = src[k];
    }
    return dst;
  }

  function hostOf(url) {
    var m = /^[a-z][a-z0-9+.\-]*:\/\/(?:[^\/?#@]*@)?(\[[^\]]+\]|[^\/?#:]+)/i.exec(textOf(url));
    return m ? m[1].replace(/^\[/, '').replace(/\]$/, '').toLowerCase() : '';
  }

  /* ── R7 firewall inheritance (plan §5) ──────────────────────────────
   * Remote adapters go through `fetch`, which already refuses denied and
   * internal addresses by RESOLVED ip. The host check here is the same
   * policy asked one step earlier, so a denied endpoint is a typed
   * refusal with the rule id instead of a generic transport failure. */
  function netGate(url) {
    var host = hostOf(url);
    if (!host || !S.policy || typeof S.policy.evaluate !== 'function') return;
    var verdict = null;
    try {
      verdict = JSON.parse(S.policy.evaluate(JSON.stringify({ op: 'net.fetch', host: host })));
    } catch (eV) { return; }   /* an evaluator fault is not a denial */
    if (verdict && verdict.decision === 'deny') {
      throw memErr('E_PERMISSION_DENIED',
        'memory adapter: ' + host + ' refused by policy (' +
        textOf(verdict.rule || 'net') + ')' +
        (verdict.reason ? ': ' + textOf(verdict.reason) : ''));
    }
  }

  /* Mutations fail closed under a read-only policy. `--read-only` narrows
   * the effective policy to fs.write:"off", which is the signal here; a
   * deployment that wants the filesystem read-only but adapter writes on
   * can say memory.write:"on" (memory.write:"off" refuses explicitly). */
  function writeGate() {
    if (!S.policy || typeof S.policy.get !== 'function') return;
    var doc = null;
    try { doc = JSON.parse(S.policy.get() || 'null'); } catch (eP) { return; }
    if (!doc) return;
    var mem = (doc.memory && typeof doc.memory === 'object') ? doc.memory : {};
    var fs = (doc.fs && typeof doc.fs === 'object') ? doc.fs : {};
    if (textOf(mem.write) === 'off') {
      throw memErr('E_PERMISSION_DENIED',
        'memory adapter: adapter writes are disabled by policy (R7.1/memory.write)');
    }
    if (textOf(fs.write) === 'off' && textOf(mem.write) !== 'on') {
      throw memErr('E_PERMISSION_DENIED',
        'memory adapter: running read-only (R7.1/fs.write; set memory.write:"on" to allow adapter writes)');
    }
  }

  /* Secrets come from the policy's configured source, never from a literal
   * in the script. `apiKey` is accepted for local harnesses; the value is
   * only ever placed in a request header and never echoed in an error. */
  function apiKeyOf(o) {
    if (o.apiKey !== undefined && o.apiKey !== null && o.apiKey !== '') return textOf(o.apiKey);
    var name = textOf(o.apiKeySecret || o.apiKeyEnv);
    if (!name) return '';
    try {
      if (S.policy && typeof S.policy.secret === 'function') {
        var v = S.policy.secret(name);
        if (v) return textOf(v);
      }
    } catch (eS) { }
    try {
      if (typeof process !== 'undefined' && process && process.env) {
        return textOf(process.env[name] || '');
      }
    } catch (eE) { }
    return '';
  }

  /* ── HTTP: one JSON round trip with the contract's error taxonomy ── */
  function httpJson(url, req) {
    req = req || {};
    netGate(url);
    var init = { method: req.method || 'GET', timeoutMs: req.timeoutMs || 15000 };
    var headers = copyInto({}, req.headers);
    if (req.body !== undefined) {
      init.body = JSON.stringify(req.body);
      if (!headers['content-type']) headers['content-type'] = 'application/json';
    }
    init.headers = headers;
    return fetch(url, init).then(function (res) {
      return res.text().then(function (raw) {
        var body = textOf(raw);
        var status = res.status | 0;
        if (!res.ok) {
          var code = (status === 401 || status === 403) ? 'E_AUTH'
            : (status === 404 ? 'E_NOT_FOUND' : 'E_HTTP');
          throw memErr(code, 'memory adapter: HTTP ' + status + ' from ' +
            hostOf(url) + ' — ' + clipText(body));
        }
        if (!body) return null;
        try { return JSON.parse(body); } catch (eJson) {
          throw memErr('E_HTTP', 'memory adapter: ' + hostOf(url) +
            ' answered non-JSON — ' + clipText(body));
        }
      });
    }, function (eNet) {
      throw memErr('E_NETWORK', 'memory adapter: ' + hostOf(url) +
        ' is unreachable — ' + clipText(eNet && eNet.message ? eNet.message : eNet));
    });
  }

  /* ── Vectors & scores ─────────────────────────────────────────────────
   * `dimensions` is the adapter's space. A mismatched vector is refused —
   * padding/truncating would corrupt the index it lands in (same rule the
   * agent's memory path already enforces). Scores are clamped cosine, not
   * rescaled: (s+1)/2 would silently move every caller's threshold. */
  function asVec(v, dims, what) {
    if (v === undefined || v === null) return null;
    var out;
    if (isView(v)) out = v;
    else if (Array.isArray(v)) out = new Float32Array(v);
    else throw memErr('E_TYPE', 'memory adapter: ' + what + ' must be a Float32Array or an array of numbers');
    if (dims > 0 && out.length !== dims) {
      throw memErr('E_DIM', 'memory adapter: ' + what + ' has ' + out.length +
        ' dims, the adapter is configured for ' + dims);
    }
    return out;
  }
  function normScore(s) {
    s = +s;
    if (!isFinite(s)) return 0;
    return s < 0 ? 0 : (s > 1 ? 1 : s);
  }
  function uuid() {
    var hex = '0123456789abcdef', out = '';
    for (var i = 0; i < 36; i++) {
      if (i === 8 || i === 13 || i === 18 || i === 23) { out += '-'; continue; }
      if (i === 14) { out += '4'; continue; }
      if (i === 19) { out += hex[((Math.random() * 4) | 0) + 8]; continue; }
      out += hex[(Math.random() * 16) | 0];
    }
    return out;
  }

  /* ── Auto-embedding ───────────────────────────────────────────────────
   * Default is the in-process deterministic embedder the brain itself uses
   * (hash trigrams, `dimensions` wide), so identical text always yields an
   * identical vector and no API key is needed. `embed` (a function) wins,
   * then `embedSpace`, then a configured provider. */
  function embedderOf(opts) {
    opts = opts || {};
    var ai = S.ai;
    if (typeof opts.embed === 'function') {
      return function (text) { return Promise.resolve(opts.embed(String(text))); };
    }
    var space = textOf(opts.embedSpace || opts.embed_space).toLowerCase();
    if (space === 'semantic' && ai && typeof ai.embedLocalSemantic === 'function') {
      return function (text) { return Promise.resolve(ai.embedLocalSemantic(String(text))); };
    }
    if ((space === 'semantic-v2' || space === 'semantic_v2') &&
        ai && typeof ai.embedLocalSemanticV2 === 'function') {
      return function (text) { return Promise.resolve(ai.embedLocalSemanticV2(String(text))); };
    }
    if ((opts.embedProvider || opts.embedModel) && ai && typeof ai.embed === 'function') {
      return function (text) {
        return Promise.resolve(ai.embed(String(text), {
          provider: opts.embedProvider, model: opts.embedModel
        })).then(function (e) {
          var raw = (e && e[0] && typeof e[0] !== 'number') ? e[0] : e;
          return raw ? new Float32Array(raw) : null;
        });
      };
    }
    var dims = opts.dimensions | 0;
    if (ai && typeof ai.embedLocal === 'function') {
      return function (text) {
        return Promise.resolve(ai.embedLocal(String(text), dims >= 16 ? dims : undefined));
      };
    }
    return function () { return Promise.resolve(null); };
  }

  /* ── Input normalization: the flat API's two call forms ────────────── */
  function recordOf(a, b, c, d, e) {
    var rec = { text: '', metadata: {} }, meta = rec.metadata, k;
    if (a !== null && a !== undefined && typeof a !== 'string' &&
        (isView(a) || Array.isArray(a))) {
      /* positional: (vector, text, role, pageId, source) */
      rec.vector = a;
      rec.text = textOf(b);
      if (c) meta.role = textOf(c);
      if (d) meta.kv_page_id = d | 0;
      if (e !== undefined) meta.source = textOf(e);
      return rec;
    }
    rec.text = textOf(a);
    if (b && typeof b === 'object' && !isView(b) && !Array.isArray(b)) {
      /* plan form: (text, {vector, metadata, tags, pinned, role, source}) */
      if (b.text !== undefined) rec.text = textOf(b.text);
      if (b.vector !== undefined) rec.vector = b.vector;
      else if (b.vec !== undefined) rec.vector = b.vec;
      if (b.metadata && typeof b.metadata === 'object') copyInto(meta, b.metadata);
      if (b.tags !== undefined) meta.tags = listOf(b.tags);
      if (b.pinned) meta.pinned = true;
      if (b.role) meta.role = textOf(b.role);
      if (b.source !== undefined) meta.source = textOf(b.source);
      if (b.kv_page_id !== undefined) meta.kv_page_id = b.kv_page_id | 0;
      return rec;
    }
    /* flat mapper's text-only form: (text, role, pageId, source) */
    if (typeof b === 'string' && b) meta.role = b;
    if (c) meta.kv_page_id = c | 0;
    if (d !== undefined) meta.source = textOf(d);
    return rec;
  }

  function queryOf(a, k) {
    var q = { limit: 5, text: '' };
    var isObj = a !== null && a !== undefined && typeof a === 'object' &&
      !isView(a) && !Array.isArray(a);
    if (isObj) {
      if (a.vector !== undefined) q.vector = a.vector;
      else if (a.vec !== undefined) q.vector = a.vec;
      if (a.text !== undefined) q.text = textOf(a.text);
      else if (a.query !== undefined) q.text = textOf(a.query);
      if (a.limit !== undefined) q.limit = a.limit | 0;
      else if (a.k !== undefined) q.limit = a.k | 0;
      else if (a.top_k !== undefined) q.limit = a.top_k | 0;
      if (a.threshold !== undefined) q.threshold = +a.threshold;
      if (a.filter !== undefined && a.filter !== null) q.filter = a.filter;
    } else if (isView(a) || Array.isArray(a)) {
      q.vector = a;
      if (k !== undefined && typeof k === 'object' && k !== null) {
        applyQueryOptions(q, k);
      } else if (k !== undefined) {
        q.limit = k | 0;
      }
    } else {
      q.text = textOf(a);
      /* recall("text", 5) and recall("text", {limit, filter}) are both
       * natural; the second argument is an overload, not just a count. */
      if (k !== undefined && typeof k === 'object' && k !== null) {
        applyQueryOptions(q, k);
      } else if (k !== undefined) {
        q.limit = k | 0;
      }
    }
    if (!(q.limit > 0)) q.limit = 5;
    if (q.limit > 1000) q.limit = 1000;
    return q;
  }

  function applyQueryOptions(q, o) {
    if (o.limit !== undefined) q.limit = o.limit | 0;
    else if (o.k !== undefined) q.limit = o.k | 0;
    else if (o.top_k !== undefined) q.limit = o.top_k | 0;
    if (o.threshold !== undefined) q.threshold = +o.threshold;
    if (o.filter !== undefined && o.filter !== null) q.filter = o.filter;
    if (o.vector !== undefined) q.vector = o.vector;
    else if (o.vec !== undefined) q.vector = o.vec;
  }

  /* ── The uniform remote handle ─────────────────────────────────────────
   * Same method names and the same two call forms as a native brain, so the
   * flat dispatcher above reaches it without a special case. Every method
   * awaits `ready`, which is the adapter's open(); a broken endpoint
   * therefore surfaces on the first call and on `brain.ready`, never as a
   * bare TypeError. */
  function adapterHandle(id, impl, dims, opts, autoEmbed, requireVector) {
    var embed = embedderOf(opts);
    var closed = false;
    var h = {
      __adapter: id,
      __async: true,
      ready: Promise.resolve().then(function () {
        return typeof impl.open === 'function' ? impl.open(opts && opts.uri, opts || {}) : null;
      })
    };
    function call(fn) {
      if (closed) return Promise.reject(memErr('E_TYPE', 'memory adapter: this brain is closed'));
      return h.ready.then(fn);
    }
    function vectorFor(rec) {
      var v = asVec(rec.vector, dims, 'vector');
      if (v) return Promise.resolve(v);
      if (!rec.text) {
        return requireVector
          ? Promise.reject(memErr('E_RECORD', 'memory adapter: remember needs text or an explicit vector'))
          : Promise.resolve(null);
      }
      /* A custom adapter that did not ask for vectors gets the raw text —
       * embedding behind its back would both cost work and hide the fact
       * that the router owns the embedding space. */
      if (!autoEmbed) return Promise.resolve(null);
      return embed(rec.text).then(function (e) {
        var ev = asVec(e, dims, 'embedding');
        if (!ev && requireVector) {
          throw memErr('E_EMBED',
            'memory adapter: this adapter stores vectors and no embedder is available — ' +
            'pass {vector}, {embed: fn}, or {embedProvider}');
        }
        return ev;
      });
    }
    h.remember = function (a, b, c, d, e) {
      return call(function () {
        writeGate();
        var rec = recordOf(a, b, c, d, e);
        return vectorFor(rec).then(function (v) {
          return impl.remember({
            text: rec.text,
            vector: v ? asPlain(v) : undefined,
            metadata: rec.metadata
          });
        }).then(function (out) {
          out = out || {};
          return { id: out.id === undefined ? null : out.id, success: out.success !== false };
        });
      });
    };
    h.learn = function (text, role, source) {
      return h.remember(text, { role: textOf(role || 'user'), source: source });
    };
    h.recall = function (a, k) {
      return call(function () {
        var q = queryOf(a, k);
        var given = asVec(q.vector, dims, 'query vector');
        var pending = given ? Promise.resolve(given)
          : (q.text ? embed(q.text) : Promise.reject(memErr('E_QUERY',
            'memory adapter: recall needs text or a vector')));
        return pending.then(function (v) {
          var qv = asVec(v, dims, 'embedding');
          if (!qv) {
            throw memErr('E_EMBED',
              'memory adapter: recall by text needs an embedder — pass {vector} or {embed: fn}');
          }
          return impl.recall({
            vector: asPlain(qv), text: q.text, limit: q.limit,
            threshold: q.threshold, filter: q.filter
          });
        }).then(function (hits) {
          var out = [], i, hit, score;
          for (i = 0; i < (hits ? hits.length : 0); i++) {
            hit = hits[i] || {};
            score = normScore(hit.score);
            if (q.threshold !== undefined && !(score >= q.threshold)) continue;
            out.push({
              id: hit.id === undefined ? null : hit.id,
              text: textOf(hit.text),
              score: score,
              metadata: (hit.metadata && typeof hit.metadata === 'object') ? hit.metadata : {},
              timestamp: hit.timestamp === undefined ? null : hit.timestamp
            });
            if (out.length >= q.limit) break;
          }
          return out;
        });
      });
    };
    h.search = h.recall;
    h.count = function () {
      return call(function () { return impl.count ? impl.count() : 0; });
    };
    h.flush = function () {
      return call(function () {
        return impl.flush ? impl.flush() : { ok: true, recordsWritten: 0 };
      });
    };
    h.forget = function (idValue) {
      return call(function () {
        writeGate();
        if (!impl.delete) throw memErr('E_ADAPTER_UNSUPPORTED',
          'memory adapter: this backend does not implement delete()');
        return impl.delete(idValue);
      });
    };
    h.delete = h.forget;
    h.close = function () {
      if (closed) return Promise.resolve(true);
      closed = true;
      return h.ready.then(function () {
        return typeof impl.close === 'function' ? impl.close() : null;
      }, function () { return null; }).then(function () { return true; });
    };
    for (var u = 0; u < ADAPTER_UNSUPPORTED.length; u++) {
      (function (name) {
        h[name] = function () {
          return Promise.reject(memErr('E_ADAPTER_UNSUPPORTED',
            'memory adapter: ' + name + ' is not part of the MemoryAdapter contract ' +
            '(remember / recall / delete / count / flush / close)'));
        };
      })(ADAPTER_UNSUPPORTED[u]);
    }
    return h;
  }

  function registerHandle(h, meta) {
    var id = live.indexOf(h), slot;
    if (id < 0) {
      slot = live.indexOf(null);
      if (slot >= 0) { id = slot; live[slot] = h; }
      else { id = live.length; live.push(h); }
    }
    try {
      Object.defineProperty(h, '__handle', { value: id, enumerable: false });
      h.toJSON = function () {
        return { handle: id, adapter: meta.adapter, dim: meta.dim | 0 };
      };
    } catch (e) { }
    return h;
  }

  function openAdapterBackend(adapter, uri, opts) {
    opts = copyInto({}, opts || {});
    opts.uri = uri;
    var id, impl;
    if (typeof adapter === 'function') {
      id = 'custom';
      impl = adapter(uri, opts);
    } else {
      id = textOf(adapter).toLowerCase();
      if (!Object.prototype.hasOwnProperty.call(ADAPTERS, id)) {
        var known = [];
        for (var k in ADAPTERS) {
          if (Object.prototype.hasOwnProperty.call(ADAPTERS, k)) known.push(k);
        }
        throw memErr('E_ADAPTER', 'memory adapter: unknown adapter "' + id +
          '" (known: ' + known.join(', ') + ')');
      }
      impl = ADAPTERS[id](uri, opts);
    }
    if (!impl || typeof impl.remember !== 'function' || typeof impl.recall !== 'function') {
      throw memErr('E_ADAPTER',
        'memory adapter: a backend must implement remember() and recall()');
    }
    var dims = opts.dimensions | 0;
    return registerHandle(
      adapterHandle(id, impl, dims, opts, true, true),
      { adapter: id, dim: dims });
  }

  function openCustomBackend(obj) {
    var id = textOf(obj.id || 'custom');
    var opts = copyInto({}, obj);
    return registerHandle(
      adapterHandle(id, wrapBackend(id, obj), obj.dimensions | 0, opts,
        typeof obj.embed === 'function', false),
      { adapter: id, dim: obj.dimensions | 0 });
  }

  /* A developer's adapter is the backend itself; this only pins the shape so
   * optional methods are checked once, not on every call. */
  function wrapBackend(id, obj) {
    return {
      open: function (uri, opts) {
        return typeof obj.open === 'function' ? obj.open(uri, opts) : null;
      },
      remember: function (rec) { return obj.remember(rec); },
      recall: function (q) { return obj.recall(q); },
      count: typeof obj.count === 'function' ? function () { return obj.count(); } : null,
      flush: typeof obj.flush === 'function' ? function () { return obj.flush(); } : null,
      delete: typeof obj.delete === 'function' ? function (v) { return obj.delete(v); } : null,
      close: typeof obj.close === 'function' ? function () { return obj.close(); } : null
    };
  }

  /* ── Shared payload helpers ────────────────────────────────────────── */
  function payloadOf(rec) {
    var p = copyInto({}, rec.metadata);
    p.text = rec.text;
    p.ts = (typeof Date !== 'undefined' && Date.now) ? Date.now() : 0;
    return p;
  }
  function hitOf(id, payload, score) {
    payload = payload || {};
    var meta = {}, k;
    for (k in payload) {
      if (Object.prototype.hasOwnProperty.call(payload, k) &&
          k !== 'text' && k !== 'ts') meta[k] = payload[k];
    }
    return {
      id: id, text: textOf(payload.text), score: score, metadata: meta,
      timestamp: payload.ts === undefined ? null : payload.ts
    };
  }
  function headersWith(base, extra) { return copyInto(copyInto({}, base), extra); }

  /* ── Adapter: qdrant (REST) ────────────────────────────────────────────
   * PUT /collections/{c}/points?wait=true, POST …/points/search. A missing
   * collection is created on first open (Cosine, `dimensions` wide) unless
   * {create:false} says the table is managed elsewhere. */
  ADAPTERS.qdrant = function (uri, o) {
    o = o || {};
    var base = textOf(uri).replace(/\/+$/, '');
    if (!base) throw memErr('E_ADAPTER', 'memory adapter: qdrant needs the cluster URL');
    var dims = o.dimensions | 0;
    if (dims <= 0) throw memErr('E_ADAPTER', 'memory adapter: qdrant needs {dimensions}');
    var col = textOf(o.collection || 'sofuu_memory');
    var headers = {};
    var key = apiKeyOf(o);
    if (key) headers['api-key'] = key;
    var cbase = base + '/collections/' + encodeURIComponent(col);

    function mustOf(filter) {
      var must = [], k;
      for (k in (filter || {})) {
        if (Object.prototype.hasOwnProperty.call(filter, k) && filter[k] !== undefined) {
          must.push({ key: k, match: { value: filter[k] } });
        }
      }
      return must.length ? must : null;
    }
    return {
      open: function () {
        return httpJson(cbase, { headers: headers }).then(function (r) {
          return !!r;
        }, function (e) {
          if (!e || e.code !== 'E_NOT_FOUND' || o.create === false) throw e;
          return httpJson(cbase, {
            method: 'PUT', headers: headers,
            body: { vectors: { size: dims, distance: textOf(o.distance || 'Cosine') } }
          }).then(function () { return true; });
        });
      },
      remember: function (rec) {
        var id = uuid();
        return httpJson(cbase + '/points?wait=true', {
          method: 'PUT', headers: headers,
          body: { points: [{ id: id, vector: rec.vector, payload: payloadOf(rec) }] }
        }).then(function () { return { id: id, success: true }; });
      },
      recall: function (q) {
        var body = { vector: q.vector, limit: q.limit, with_payload: true };
        var must = mustOf(q.filter);
        if (must) body.filter = { must: must };
        return httpJson(cbase + '/points/search', {
          method: 'POST', headers: headers, body: body
        }).then(function (r) {
          var rows = (r && r.result) || [], out = [], i;
          for (i = 0; i < rows.length; i++) {
            out.push(hitOf(rows[i].id, rows[i].payload, rows[i].score));
          }
          return out;
        });
      },
      count: function () {
        return httpJson(cbase + '/points/count', {
          method: 'POST', headers: headers, body: { exact: true }
        }).then(function (r) {
          return (r && r.result && (r.result.count | 0)) || 0;
        });
      },
      delete: function (id) {
        return httpJson(cbase + '/points/delete?wait=true', {
          method: 'POST', headers: headers, body: { points: [id] }
        }).then(function () { return true; });
      },
      flush: function () { return { ok: true, recordsWritten: 0 }; },
      close: function () { }
    };
  };

  /* ── Adapter: pinecone (REST) ──────────────────────────────────────────
   * POST /vectors/upsert, POST /query. The index (dimension + metric) is
   * created outside Sofuu; `uri` is the index host. */
  ADAPTERS.pinecone = function (uri, o) {
    o = o || {};
    var base = textOf(uri).replace(/\/+$/, '');
    if (!base) throw memErr('E_ADAPTER', 'memory adapter: pinecone needs the index host URL');
    var dims = o.dimensions | 0;
    if (dims <= 0) throw memErr('E_ADAPTER', 'memory adapter: pinecone needs {dimensions}');
    var headers = {};
    var key = apiKeyOf(o);
    if (key) headers['Api-Key'] = key;
    var ns = textOf(o.namespace);
    function withNs(path) {
      return ns ? path + (path.indexOf('?') >= 0 ? '&' : '?') +
        'namespace=' + encodeURIComponent(ns) : path;
    }
    return {
      open: function () {
        return httpJson(withNs(base + '/describe_index_stats'), {
          method: 'POST', headers: headers, body: {}
        }).then(function () { return true; });
      },
      remember: function (rec) {
        var id = uuid().replace(/-/g, '');
        return httpJson(withNs(base + '/vectors/upsert'), {
          method: 'POST', headers: headers,
          body: { vectors: [{ id: id, values: rec.vector, metadata: payloadOf(rec) }] }
        }).then(function () { return { id: id, success: true }; });
      },
      recall: function (q) {
        var body = { vector: q.vector, topK: q.limit, includeMetadata: true };
        if (q.filter) body.filter = q.filter;
        return httpJson(withNs(base + '/query'), {
          method: 'POST', headers: headers, body: body
        }).then(function (r) {
          var rows = (r && r.matches) || [], out = [], i;
          for (i = 0; i < rows.length; i++) {
            out.push(hitOf(rows[i].id, rows[i].metadata, rows[i].score));
          }
          return out;
        });
      },
      count: function () {
        return httpJson(withNs(base + '/describe_index_stats'), {
          method: 'POST', headers: headers, body: {}
        }).then(function (r) { return (r && (r.totalVectorCount | 0)) || 0; });
      },
      delete: function (id) {
        return httpJson(withNs(base + '/vectors/delete'), {
          method: 'POST', headers: headers, body: { ids: [id] }
        }).then(function () { return true; });
      },
      flush: function () { return { ok: true, recordsWritten: 0 }; },
      close: function () { }
    };
  };

  /* ── Adapter: pgvector over PostgREST ──────────────────────────────────
   * INSERT into a table, search through an RPC (`match_memories` by
   * default) that takes query_embedding/match_threshold/match_count. Column
   * names are configurable because every deployment names them its own way;
   * the defaults match the common Supabase pgvector tutorial shape. */
  ADAPTERS.pgvector = function (uri, o) {
    o = o || {};
    var base = textOf(uri).replace(/\/+$/, '');
    if (!base) throw memErr('E_ADAPTER', 'memory adapter: pgvector needs the project URL');
    var rest = textOf(o.restPath || '/rest/v1').replace(/\/+$/, '');
    var table = textOf(o.table || 'memories');
    var rpc = textOf(o.rpc || 'match_memories');
    var cols = o.columns || {};
    var cText = textOf(cols.text || 'content');
    var cVec = textOf(cols.embedding || 'embedding');
    var cMeta = textOf(cols.metadata || 'metadata');
    var cId = textOf(cols.id || 'id');
    var res = o.result || {};
    var rText = textOf(res.text || cText);
    var rScore = textOf(res.score || 'similarity');
    var rMeta = textOf(res.metadata || cMeta);
    var headers = {};
    var key = apiKeyOf(o);
    if (key) { headers['apikey'] = key; headers['Authorization'] = 'Bearer ' + key; }
    var tableUrl = base + rest + '/' + encodeURIComponent(table);
    var rpcUrl = base + rest + '/rpc/' + encodeURIComponent(rpc);

    return {
      open: function () {
        return httpJson(tableUrl + '?select=' + encodeURIComponent(cId) + '&limit=1',
          { headers: headers }).then(function () { return true; });
      },
      remember: function (rec) {
        var body = {};
        body[cText] = rec.text;
        body[cVec] = rec.vector;
        body[cMeta] = rec.metadata;
        return httpJson(tableUrl, {
          method: 'POST', headers: headersWith(headers, { Prefer: 'return=representation' }),
          body: body
        }).then(function (rows) {
          var row = (rows && rows[0]) || {};
          return { id: row[cId] === undefined ? null : row[cId], success: true };
        });
      },
      recall: function (q) {
        var body = { query_embedding: q.vector, match_count: q.limit, match_threshold: 0 };
        if (q.filter) body.filter = q.filter;
        return httpJson(rpcUrl, { method: 'POST', headers: headers, body: body })
          .then(function (rows) {
            var out = [], i, row;
            for (i = 0; i < (rows ? rows.length : 0); i++) {
              row = rows[i] || {};
              out.push({
                id: row[cId] === undefined ? null : row[cId],
                text: textOf(row[rText]),
                score: row[rScore],
                metadata: (row[rMeta] && typeof row[rMeta] === 'object') ? row[rMeta] : {},
                timestamp: row.ts === undefined ? null : row.ts
              });
            }
            return out;
          });
      },
      count: function () {
        /* GET + Prefer: count=exact, not HEAD: the fetch bridge drives curl
         * through CUSTOMREQUEST, so a HEAD answers CURLE_GOT_NOTHING (52)
         * instead of headers. PostgREST reports the total in Content-Range
         * on a normal GET, and limit=1 keeps the payload to one row. */
        netGate(tableUrl);
        return fetch(tableUrl + '?select=' + encodeURIComponent(cId) + '&limit=1', {
          timeoutMs: 15000,
          headers: headersWith(headers, { Prefer: 'count=exact' })
        }).then(function (r) {
          var status = (r && r.status) | 0;
          if (!r || !r.ok) {
            throw memErr((status === 401 || status === 403) ? 'E_AUTH' : 'E_HTTP',
              'memory adapter: HTTP ' + status + ' from ' + hostOf(tableUrl));
          }
          var cr = (r.headers && typeof r.headers.get === 'function')
            ? textOf(r.headers.get('content-range')) : '';
          return r.text().then(function () {
            var m = /\/(\d+)\s*$/.exec(cr);
            return m ? (m[1] | 0) : 0;
          });
        });
      },
      delete: function (id) {
        return httpJson(tableUrl + '?' + encodeURIComponent(cId) + '=eq.' +
          encodeURIComponent(textOf(id)), { method: 'DELETE', headers: headers })
          .then(function () { return true; });
      },
      flush: function () { return { ok: true, recordsWritten: 0 }; },
      close: function () { }
    };
  };


  M.__installAdapters({ remote: openAdapterBackend, custom: openCustomBackend });
})();
