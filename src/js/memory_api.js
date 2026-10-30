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
   * build, but if the build facts say the codec is missing, leave the
   * typed-refusal stubs untouched.
   *
   * Read `sofuu.build` — the manifest's build block, a pure function of the
   * compilation — NOT `sofuu.capabilities()`. The full manifest probes the
   * GPU, so asking it here created a Metal device (~18 ms) on every boot,
   * before a single line of user code ran. */
  var capped = false;
  try {
    capped = !!(S.build && S.build.missing && S.build.missing.indexOf('qtsq') >= 0);
  } catch (eBuild) { /* not registered yet — the Rust cfg gates it */ }
  if (capped) return;

  var live = [];
  var realOpen = M.open;

  M.open = function (a, dim, spaceId) {
    // Object form {"path","dim","embed_id"|"space"} and the in-process
    // positional form (path, dim, spaceId) both work.
    var path, d, sid, opts;
    if (a && typeof a === 'object') {
      path = a.path;
      d = a.dim;
      sid = a.embed_id !== undefined ? a.embed_id : a.space;
      /* W2.2 / W3.3: open modes. `readonly` takes a SHARED lock and writes
       * nothing at all (no repair, no backup, no lock file); `verifyFull`
       * keeps the paranoid open that reads and digests every segment
       * instead of deferring the load to first row access. */
      opts = {
        readonly: !!a.readonly,
        verify_full: !!(a.verifyFull || a.verify_full)
      };
    } else {
      path = a;
      d = dim;
      /* M.open(path, dim, spaceId) and M.open(path, dim, {readonly, verifyFull})
       * are both natural calls, so the third positional argument is accepted
       * as either a space id or the open-modes object. */
      if (spaceId && typeof spaceId === 'object') {
        sid = spaceId.embed_id !== undefined ? spaceId.embed_id : spaceId.space;
        opts = {
          readonly: !!spaceId.readonly,
          verify_full: !!(spaceId.verifyFull || spaceId.verify_full)
        };
      } else {
        sid = spaceId;
        opts = { readonly: false, verify_full: false };
      }
    }
    var inst = (sid === undefined || sid === null)
      ? realOpen(path, d, undefined, opts)
      : realOpen(path, d, sid, opts);
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
  /* W1.7: "you gave me no brain / a stale handle" is a typed refusal, not an
   * anonymous TypeError — a host driving the C-ABI used to receive the
   * useless `js_exception` for a double-closed handle. */
  function pick(a) {
    if (a && a.__handle !== undefined) return a;
    if (a && a.handle !== undefined && a.handle !== null) {
      var got = live[a.handle | 0];
      if (!got) {
        throw refuse(TypeError, 'E_TYPE',
          'memory: no open brain with handle ' + a.handle);
      }
      return got;
    }
    for (var i = live.length - 1; i >= 0; i--) {
      if (live[i]) return live[i];
    }
    throw refuse(TypeError, 'E_TYPE', 'memory: no open brain — call memory.open first');
  }

  /* W1.3 (AUDIT-2026-10-29): the vector gate, JS side.
   *
   * A Float32Array passes through untouched; a plain Array (or any other
   * value the `new Float32Array(...)` coercion accepted) keeps the old
   * ergonomics — `new Float32Array([0.1, 0.2])` and `[0.1, 0.2]` both work.
   * A WRONG TypedArray (Float64Array, Int32Array, Uint8Array, …) is a typed
   * refusal instead of a silent conversion: re-interpreting its bytes is not
   * just lossy, the native bridge used to read a 768-element Float64Array as
   * 1536 floats of double bits. The Rust bridge enforces the same rule (it is
   * the real boundary); this is what keeps the diagnosis local and cheap. */
  var AB_ISVIEW = (typeof ArrayBuffer !== 'undefined' &&
    typeof ArrayBuffer.isView === 'function');
  function isView(a) {
    return !!a && typeof a === 'object' && AB_ISVIEW && ArrayBuffer.isView(a);
  }
  function ctorName(v) {
    try {
      return (v && v.constructor && v.constructor.name) || 'ArrayBuffer view';
    } catch (e) { return 'ArrayBuffer view'; }
  }
  function f32(v, what) {
    if (typeof Float32Array !== 'undefined' && v instanceof Float32Array) return v;
    if (isView(v)) {
      var name = ctorName(v);
      throw refuse(TypeError, 'E_TYPE',
        'memory' + (what ? '.' + what : '') + ': expected a Float32Array but received a ' +
        name + ' — a ' + name + "'s bytes cannot be re-interpreted as float32; pass a " +
        'Float32Array (or a plain Array of numbers)');
    }
    return new Float32Array(v || []);
  }

  /* An options object (flat/contract form) vs a positional argument. Opened
   * brains and typed arrays are positional; plain objects are options. */
  function isOpts(a) {
    /* W1.3: ANY ArrayBuffer view is a positional vector argument. Before
     * this, a Float64Array/Int32Array was treated as an OPTIONS object (only
     * Float32Array and Uint8Array were excluded), so the positional form
     * silently took the options path and never reached the vector gate. */
    return !!a && typeof a === 'object' && !Array.isArray(a) &&
      a.__handle === undefined && !isView(a);
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

  /* ── W1.8 (AUDIT-2026-10-29): damage events ──────────────────────
   *
   * Damage used to be visible ONLY as stderr ("[sofuu/cma] SEGMENT DAMAGE:
   * …"). `M.onEvent(cb)` delivers a real event object for each damage fact
   * the store reports; the native side calls the hook below (by name, from
   * rt/memory.rs) at the moment the damage is found or a flush fails — it
   * never synthesizes an event from a counter it does not track.
   *
   * The registry lives HERE, in JS, so the callbacks are ordinary JS values
   * the GC owns; Rust only invokes the hook with a JSON payload.
   * Returns an unsubscribe function.
   */
  var eventHandlers = [];
  var EMIT_KEY = '__sofuu_memory_event';
  function emitMemoryEvent(payload) {
    var ev;
    try { ev = typeof payload === 'string' ? JSON.parse(payload) : payload; }
    catch (eParse) { return; }
    if (!ev || !ev.type) return;
    /* Snapshot: a handler that unsubscribes must not skip the next one. */
    var list = eventHandlers.slice();
    for (var i = 0; i < list.length; i++) {
      try { list[i](ev); } catch (eHandler) { /* a handler must never break the store */ }
    }
  }
  try {
    Object.defineProperty(globalThis, EMIT_KEY, {
      value: emitMemoryEvent, enumerable: false, configurable: true, writable: true
    });
  } catch (eDef) {
    try { globalThis[EMIT_KEY] = emitMemoryEvent; } catch (eSet) { /* no hooks then */ }
  }

  M.onEvent = function (cb) {
    if (typeof cb !== 'function') {
      throw refuse(TypeError, 'E_TYPE', 'memory.onEvent: the callback must be a function');
    }
    eventHandlers.push(cb);
    var active = true;
    return function off() {
      if (!active) return false;
      active = false;
      var i = eventHandlers.indexOf(cb);
      if (i >= 0) eventHandlers.splice(i, 1);
      return true;
    };
  };

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
  /* W6.1: the close-flush must REPORT a failure. It used to swallow the
   * flush exception and answer `true` regardless, so a host that only ever
   * calls close() (the documented "flush on the way out" path) never learned
   * its last writes were not on disk. The handle is released either way —
   * then the CODED flush error is rethrown. */
  M.close = function (a) {
    var inst = pick(isOpts(a) ? a : (a || {}));
    var flushErr = null;
    try { if (typeof inst.flush === 'function') inst.flush(); } catch (eFlush) { flushErr = eFlush; }
    var id = live.indexOf(inst);
    if (id >= 0) live[id] = null;
    if (flushErr) throw flushErr;
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
