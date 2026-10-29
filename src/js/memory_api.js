/* memory_api.js — the flat memory/brain API over the CMA instance methods.
 * @sofuu-source-only — eval'd at registration; NOT a shipped .bc driver
 * (scripts/gen-driver-bc.sh and scripts/check-driver-bc.sh skip this marker).
 *
 * The engine's CMA module (rt/memory.rs) exposes `open` on the module and the
 * dataset methods on the object `open` returns:
 *
 *     var brain = sofuu.memory.open(path, dim);
 *     brain.remember(vec, 'text');
 *
 * The runtime contract (R1 / T1.3) documents the FLAT shape instead —
 * `sofuu.memory.remember({vec, text})`, `recall({query})`, ... — and the
 * QTSQ-free build already presents it (typed-refusal stubs with the same
 * names). A release (QTSQ) build used to present only the instance shape, so
 * the documented API existed on the degraded build and NOT on the real one
 * (contract/matrix.json: memory x js = partial). This file installs the flat
 * methods once on the module object — `sofuu.brain` is an alias of the same
 * object — delegating to the most recently opened brain or to `{handle}`.
 *
 * Both call forms are accepted, so nothing that worked stops working:
 *
 *     sofuu.memory.remember({ vec: v, text: 'x' })   // contract / funnel form
 *     sofuu.memory.remember(v, 'x', 'user', 0)       // instance-argument form
 *
 * Evaluated by rt/memory.rs at registration (QTSQ builds only; capability-
 * gated builds keep their typed refusals). Included as source, not as a
 * shipped .bc driver: it must run before any host JS, capi used to inline
 * the same code, and it is small.
 */
(function () {
  'use strict';
  var S = globalThis.sofuu;
  if (!S) return;
  var M = S.memory;
  if (!M || typeof M.open !== 'function') return;
  if (M.__flat_memory_api) return;

  /* Belt and braces: the Rust caller evaluates this only in the has_qtsq
   * build, but if the manifest is already registered and says the codec is
   * missing, leave the typed-refusal stubs untouched. */
  var capped = false;
  try {
    var caps = JSON.parse(S.capabilities());
    capped = ((caps.build && caps.build.missing) || []).indexOf('qtsq') >= 0;
  } catch (eCaps) { /* manifest not registered yet — the Rust cfg gates it */ }
  if (capped) return;

  var live = [];
  var realOpen = M.open;

  M.open = function (a, dim, spaceId) {
    // Object form {"path","dim","embed_id"|"space"} and the in-process
    // positional form (path, dim, spaceId) both work.
    var path, d, sid;
    if (a && typeof a === 'object') {
      path = a.path;
      d = a.dim;
      sid = a.embed_id !== undefined ? a.embed_id : a.space;
    } else {
      path = a;
      d = dim;
      sid = spaceId;
    }
    var inst = (sid === undefined || sid === null) ? realOpen(path, d) : realOpen(path, d, sid);
    var id = live.indexOf(inst);
    if (id < 0) {
      var emptyIdx = live.indexOf(null);
      if (emptyIdx >= 0) {
        id = emptyIdx;
        live[id] = inst;
      } else {
        id = live.length;
        live.push(inst);
      }
    }
    try {
      Object.defineProperty(inst, '__handle', { value: id, enumerable: false });
      inst.toJSON = function () {
        return { handle: id, dim: (function () { try { return d; } catch (e) { return null; } })() };
      };
    } catch (e) { /* sealed object — the handle registry still works */ }
    return inst;
  };

  /* A target is: an opened brain itself, {handle: N}, or nothing (most
   * recently opened). */
  function pick(a) {
    if (a && a.__handle !== undefined) return a;
    if (a && a.handle !== undefined && a.handle !== null) {
      var got = live[a.handle | 0];
      if (!got) throw new TypeError('memory: no open brain with handle ' + a.handle);
      return got;
    }
    for (var i = live.length - 1; i >= 0; i--) {
      if (live[i]) return live[i];
    }
    throw new TypeError('memory: no open brain — call memory.open first');
  }

  function f32(v) {
    if (v instanceof Float32Array) return v;
    return new Float32Array(v || []);
  }

  /* An options object (flat/contract form) vs a positional argument. Opened
   * brains and typed arrays are positional; plain objects are options. */
  function isOpts(a) {
    return !!a && typeof a === 'object' && !Array.isArray(a) &&
      a.__handle === undefined &&
      !(typeof Float32Array !== 'undefined' && a instanceof Float32Array) &&
      !(typeof Uint8Array !== 'undefined' && a instanceof Uint8Array);
  }

  /* Typed refusal: a thrown Error carrying `.code`, the shape the rest of the
   * runtime's bridges use (plugins.js / providers.js `makeError`). Used where
   * the alternative would be a silent no-op — the native layer keeps its own
   * defensive clamp, but a clamp that answers `true` is a lie. */
  function refuse(ctor, code, message) {
    var err = new ctor(message);
    err.code = code;
    return err;
  }

  /* Dual-form dispatch: options object -> mapper(a); otherwise forward the
   * raw arguments to the current brain (instance-argument form). */
  function invoke(name, mapper) {
    return function () {
      var a = arguments[0];
      if (isOpts(a)) {
        var inst = pick(a);
        return inst[name].apply(inst, mapper ? mapper(a) : []);
      }
      var cur = pick({});
      return cur[name].apply(cur, Array.prototype.slice.call(arguments));
    };
  }

  M.remember = invoke('remember', function (a) {
    if (!a.vec && a.text) {
      return [String(a.text), a.role || 'user', a.kv_page_id | 0,
              a.source === undefined ? undefined : String(a.source)];
    }
    return [f32(a.vec), a.text, a.role || 'user', a.kv_page_id | 0,
            a.source === undefined ? undefined : String(a.source)];
  });
  M.learn = invoke('learn', function (a) {
    var text = typeof a === 'string' ? a : (a.text !== undefined ? a.text : '');
    var role = a.role || 'user';
    var source = a.source !== undefined ? String(a.source) : undefined;
    return [String(text), role, source];
  });
  /* B1.5 / W6.4: the read-out surface, same shape as the in-process API.
   *
   * Contract: `m.list({tier?, role?, entity?, text?, min_strength?, offset?,
   * limit?})` returns an ARRAY of real record objects. The JSON envelope
   * `{total, offset, limit, records}` is the native instance/C-ABI contract
   * (`js_cma_list` still answers it in rt/memory.rs for the chat driver), and
   * this bridge is where it becomes JS values. `offset`/`limit` are applied
   * natively, in store insertion order, so a page is stable while the store is
   * unchanged; `total` (the count BEFORE pagination), `offset` and `limit`
   * ride along as non-enumerable properties so a pager can render "N of M"
   * without a second call, while `JSON.stringify(rows)` stays the records. */
  M.list = function (a) {
    var inst = pick(isOpts(a) ? a : {});
    var opts = {};
    if (isOpts(a)) {
      if (a.tier !== undefined) opts.tier = a.tier | 0;
      if (a.role !== undefined) opts.role = String(a.role);
      if (a.entity !== undefined) opts.entity = String(a.entity);
      if (a.text !== undefined) opts.text = String(a.text);
      if (a.min_strength !== undefined) opts.minStrength = +a.min_strength;
      if (a.offset !== undefined) opts.offset = a.offset | 0;
      if (a.limit !== undefined) opts.limit = a.limit | 0;
    }
    var out = isOpts(a) ? inst.list(opts) : inst.list(a);
    var env = typeof out === 'string' ? JSON.parse(out) : out;
    var recs = (env && env.records) || [];
    try {
      Object.defineProperty(recs, 'total', { value: (env && env.total) | 0, enumerable: false });
      Object.defineProperty(recs, 'offset', { value: (env && env.offset) | 0, enumerable: false });
      Object.defineProperty(recs, 'limit', { value: (env && env.limit) | 0, enumerable: false });
    } catch (eMeta) { /* frozen array — the records still work */ }
    return recs;
  };
  M.export = invoke('export', function () { return []; });
  M.recall = invoke('recall', function (a) {
    var q = a.text !== undefined ? a.text : a.query;
    var k = a.k === undefined ? (a.top_k !== undefined ? (a.top_k | 0) : 5) : (a.k | 0);
    if (!a.vec && q !== undefined && q !== null) return [String(q), k];
    var args = [f32(a.vec), k];
    if (q !== undefined && q !== null) args.push(String(q));
    return args;
  });
  M.search = invoke('search', function (a) {
    var q = typeof a === 'string' ? a : (a.text !== undefined ? a.text : (a.query !== undefined ? a.query : ''));
    var k = a.k === undefined ? (a.top_k !== undefined ? (a.top_k | 0) : 5) : (a.k | 0);
    return [String(q), k];
  });
  M.count = invoke('count', function () { return []; });
  M.flush = invoke('flush', function () { return []; });
  M.forget = invoke('forget', function (a) { return [a.index | 0]; });
  /* B6: the session working set, same shapes as the in-process API — a
   * headless embedder promotes exactly like the chat driver does. */
  M.rememberWorking = invoke('rememberWorking', function (a) {
    return [f32(a.vec), a.text || '', a.role || 'user', a.kv_page_id | 0,
            a.source === undefined ? undefined : String(a.source)];
  });
  M.promoteWorking = invoke('promoteWorking', function (a) {
    return [JSON.stringify(Array.isArray(a.ids) ? a.ids : (a.id !== undefined ? [a.id] : []))];
  });
  M.clearWorking = invoke('clearWorking', function () { return []; });
  /* B7.1: stats is part of the read-out parity (doctor and /brain info read
   * the same numbers a headless host does). */
  M.stats = invoke('stats', function () { return []; });

  /* CMA parity additions. */
  M.rememberEntity = invoke('rememberEntity', function (a) {
    return [f32(a.vec), a.text || '',
            a.entity_name || a.entityName || '',
            a.entity_type || a.entityType || 'unknown'];
  });
  M.rememberEntityTurn = invoke('rememberEntityTurn', function (a) {
    return [f32(a.vec), a.text || '',
            a.entity_name || a.entityName || '',
            a.entity_type || a.entityType || 'unknown'];
  });
  M.markPositive = invoke('markPositive', function (a) {
    return [a.ids !== undefined ? a.ids : a.id];
  });
  /* W6.2 — object parameter parsing. `{seconds}`, `{hours}` (×3600) or `{dt}`
   * seconds; a bare number is already seconds. Validated HERE, not silently
   * coerced: the native `decayTick` deliberately turns a negative/non-finite
   * dt into 0 (a no-op) so a ToUint32 wrap can never erase the whole brain
   * (P3, AUDIT-2026-09-07), but a no-op that returns `true` is a silent lie,
   * and a missing unit used to decay a silent 1 second. Both are now typed
   * refusals (E_DT) that leave every strength untouched. */
  M.decayTick = function (a) {
    var inst = pick(isOpts(a) ? a : {});
    var dt;
    if (isOpts(a)) {
      if (a.dt !== undefined) dt = +a.dt;
      else if (a.seconds !== undefined) dt = +a.seconds;
      else if (a.hours !== undefined) dt = +a.hours * 3600;
      else {
        throw refuse(TypeError, 'E_DT',
          'memory.decayTick: pass {seconds} or {hours} (or a bare number of seconds)');
      }
    } else {
      dt = +a;
    }
    if (!isFinite(dt) || dt < 0) {
      throw refuse(RangeError, 'E_DT',
        'memory.decayTick: dt must be a finite, non-negative number of seconds (got ' +
        String(dt) + ')');
    }
    inst.decayTick(dt);
    return true;
  };
  M.retain = invoke('retain', function () { return []; });
  M.retainAgainst = function (a) {
    var inst, other;
    if (isOpts(a)) {
      inst = pick(a);
      var otherHandle = a.other !== undefined ? a.other
        : (a.canonical !== undefined ? a.canonical : a.target);
      other = pick({ handle: otherHandle });
    } else {
      inst = pick({});
      other = pick(a);
    }
    return inst.retainAgainst(other);
  };
  M.consolidate = invoke('consolidate', function () { return []; });
  M.compact = invoke('compact', function () { return []; });
  /* W6.3 — ONE serialization contract, stated once: the native
   * `importRecords(vecs, recordsJson)` parses `recordsJson` as a JSON OBJECT
   * `{"records":[...]}` (`Cma::import_records` reads `root["records"]`; a bare
   * array matches nothing and imports zero rows — exactly the mismatch this
   * wrapper used to send, which made every flat import a silent no-op). So
   * every call from here sends that envelope, `records` may be an array or a
   * JSON string in either call form, and the native reply
   * `{"added","merged","skipped"}` is parsed back into a real object. The
   * instance method keeps its JSON-string reply for the chat driver and the C
   * ABI; this flat surface is the JS-shaped one. */
  function importEnvelope(recs) {
    var arr = recs;
    if (typeof recs === 'string') {
      try { arr = JSON.parse(recs); }
      catch (eJson) {
        throw refuse(TypeError, 'E_RECORDS', 'memory.importRecords: records is not valid JSON');
      }
    }
    if (arr && !Array.isArray(arr) && typeof arr === 'object' &&
        Array.isArray(arr.records)) {
      arr = arr.records;
    }
    if (!Array.isArray(arr)) {
      throw refuse(TypeError, 'E_RECORDS',
        'memory.importRecords: records must be an array of record objects (or {"records":[...]})');
    }
    return JSON.stringify({ records: arr });
  }

  function importStats(out) {
    var s = typeof out === 'string' ? JSON.parse(out) : out;
    if (!s || typeof s !== 'object') {
      throw refuse(TypeError, 'E_IMPORT', 'memory.importRecords: the engine returned no stats');
    }
    return { added: s.added | 0, merged: s.merged | 0, skipped: s.skipped | 0 };
  }

  M.importRecords = function (a) {
    var inst = pick(isOpts(a) ? a : {});
    if (!isOpts(a)) {
      /* positional: (vecsFloat32, recordsJson) */
      return importStats(inst.importRecords(a, importEnvelope(arguments[1])));
    }
    return importStats(inst.importRecords(f32(a.vecs || a.vec), importEnvelope(a.records)));
  };
  M.kvHints = invoke('kvHints', function (a) {
    var n = a.n_hints !== undefined ? (a.n_hints | 0) : (a.n === undefined ? 5 : (a.n | 0));
    return [f32(a.vec), n];
  });

  /* close releases the handle. Flush first so a host that never calls
   * flush() explicitly still leaves the brain consistent on disk (the
   * native object itself is freed by GC). */
  M.close = function (a) {
    var inst = pick(isOpts(a) ? a : (a || {}));
    try { if (typeof inst.flush === 'function') inst.flush(); } catch (eFlush) {}
    var id = live.indexOf(inst);
    if (id >= 0) live[id] = null;
    return true;
  };
  M.handles = function () {
    var count = 0;
    for (var i = 0; i < live.length; i++) if (live[i]) count++;
    return count;
  };

  try {
    Object.defineProperty(M, '__flat_memory_api', { value: 'full', enumerable: false });
  } catch (eMark) { M.__flat_memory_api = 'full'; }
  /* `sofuu.brain` is normally the same object already (memory_register
   * aliases it); restore the alias if a host replaced it with something
   * non-namespace-like. */
  try {
    if (!S.brain || (typeof S.brain === 'object' && S.brain.__handle === undefined &&
        typeof S.brain.remember !== 'function')) {
      S.brain = M;
    }
  } catch (eBrain) {}
})();
