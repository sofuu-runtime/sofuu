// triggers.js — `sofuu.triggers`: reactive loops with pre-LLM gates
// (PLAN-AUTONOMOUS-REACTIVE-RUNTIME Phase 2).
//
// The shape every 24/7 agent hand-rolls:
//
//   const runner = sofuu.triggers.register({
//     id: 'monitor-loop',
//     kind: 'interval', every: '60s', nonOverlapping: true,
//     backoff: { initialMs: 2000, maxMs: 60000, multiplier: 2 },
//     budget: { maxTriggersPerHour: 20, maxAiCallsPerDay: 200 },
//     async gate(ctx) { return payloadOrNull; },   // 0 tokens when null
//     async onTrigger(payload, ctx) { /* call the model here */ },
//     onError(err, ctx) { /* log; backoff already applied */ }
//   });
//   await runner.start();
//
// Design:
//   * `gate()` is pure deterministic JS with no implicit model access. A
//     null/false/undefined result is a clean skip: onTrigger is NOT called.
//   * `nonOverlapping` (default true) is a hard lock: a tick that arrives
//     while the previous onTrigger is still running is counted and skipped.
//   * Failures feed an exponential backoff; a success resets it.
//   * Budgets are persisted through `sofuu.state` when available, so a
//     process restart does not reset a daily AI-call cap. Without the native
//     store the counters fall back to process memory.
//   * Kinds are domain-agnostic: interval, cron, watch, stream/event. A
//     stream trigger is fed by runner.push(payload) (or
//     sofuu.triggers.emit(id, payload)) — webhooks/channels can call it.
//   * `SOFUU_DAEMON_INTERVAL_MS` (set by sofuu-agent --daemon/--every) is
//     the default interval when a spec omits `every`, so the same agent runs
//     natively under launchd/systemd instead of being re-spawned every 5s.
//
// Errors (`RULES.md §3.5`): E_TRIGGER_INVALID for a bad spec/kind/expression;
// E_BUDGET_EXCEEDED only when a spec opts into throwing.

(function () {
  'use strict';
  var global = globalThis;
  if (!global.sofuu) global.sofuu = {};

  function makeError(code, msg, details) {
    var err = new Error(msg);
    err.name = 'SofuuError';
    err.code = code || 'E_TRIGGER_INVALID';
    err.rule = 'RULES.md §3.5';
    if (details) err.details = details;
    return err;
  }

  function nowMs() { return Date.now(); }

  function clamp(n, lo, hi) { return n < lo ? lo : (n > hi ? hi : n); }

  function finite(v, name) {
    var n = Number(v);
    if (!Number.isFinite(n)) {
      throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: ' + name + ' must be a finite number');
    }
    return n;
  }

  /* ── durations ────────────────────────────────────────────────────── */

  function parseDuration(v, name) {
    if (typeof v === 'number') {
      if (!Number.isFinite(v) || v < 0) {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: ' + name + ' must be a number >= 0');
      }
      return Math.floor(v);
    }
    if (typeof v !== 'string' || !v.trim()) {
      throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: ' + name + ' is required (e.g. "30s", "5m" or 30000)');
    }
    var s = v.trim().toLowerCase();
    var m = /^([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h|d)?$/.exec(s);
    if (!m) {
      throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: cannot parse ' + name + ' \'' + v + '\' (use ms/s/m/h/d)');
    }
    var n = parseFloat(m[1]);
    var unit = m[2] || 'ms';
    var mul = unit === 'ms' ? 1 : unit === 's' ? 1000 : unit === 'm' ? 60000 : unit === 'h' ? 3600000 : 86400000;
    return Math.floor(n * mul);
  }

  /* ── cron (5-field, UTC or local) ─────────────────────────────────── */

  var MONTH_NAMES = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6,
                      jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };
  var DOW_NAMES = { sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6 };

  function parseField(raw, min, max, names, fieldName) {
    var out = Object.create(null);
    var parts = String(raw).split(',');
    for (var p = 0; p < parts.length; p++) {
      var part = parts[p].trim();
      if (!part) {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: empty ' + fieldName + ' field in cron expression');
      }
      var step = 1;
      var slash = part.indexOf('/');
      if (slash >= 0) {
        step = parseInt(part.slice(slash + 1), 10);
        part = part.slice(0, slash);
        if (!(step > 0)) {
          throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: invalid step in cron ' + fieldName + ' field');
        }
      }
      var lo, hi;
      if (part === '*' || part === '?') {
        lo = min; hi = max;
      } else {
        var dash = part.indexOf('-');
        if (dash > 0) {
          lo = parseCronValue(part.slice(0, dash), names, fieldName);
          hi = parseCronValue(part.slice(dash + 1), names, fieldName);
        } else {
          lo = parseCronValue(part, names, fieldName);
          hi = (slash >= 0) ? max : lo;
        }
      }
      if (lo < min || hi > max || lo > hi) {
        throw makeError('E_TRIGGER_INVALID',
          'sofuu.triggers: cron ' + fieldName + ' value out of range (' + min + '-' + max + ')');
      }
      for (var v = lo; v <= hi; v += step) out[v] = true;
    }
    var list = Object.keys(out).map(Number).sort(function (a, b) { return a - b; });
    return list;
  }

  function parseCronValue(tok, names, fieldName) {
    var s = String(tok).trim().toLowerCase();
    if (names && Object.prototype.hasOwnProperty.call(names, s)) return names[s];
    var n = parseInt(s, 10);
    if (!Number.isFinite(n) || String(n) !== s) {
      throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: invalid cron ' + fieldName + ' value \'' + tok + '\'');
    }
    return n;
  }

  function parseCron(expr) {
    var parts = String(expr == null ? '' : expr).trim().split(/\s+/);
    if (parts.length !== 5) {
      throw makeError('E_TRIGGER_INVALID',
        'sofuu.triggers: cron needs 5 fields (minute hour day-of-month month day-of-week), got ' +
        (parts.length || 0));
    }
    var minute = parseField(parts[0], 0, 59, null, 'minute');
    var hour = parseField(parts[1], 0, 23, null, 'hour');
    var dom = parseField(parts[2], 1, 31, null, 'day-of-month');
    var month = parseField(parts[3], 1, 12, MONTH_NAMES, 'month');
    // Cron allows both 0 and 7 for Sunday; fold 7 onto 0 after parsing so
    // ranges like `5-7` (Fri–Sun) stay valid.
    var dowRaw = parseField(parts[4], 0, 7, DOW_NAMES, 'day-of-week');
    var dowSet = Object.create(null);
    for (var di = 0; di < dowRaw.length; di++) dowSet[dowRaw[di] === 7 ? 0 : dowRaw[di]] = true;
    var dow = Object.keys(dowSet).map(Number).sort(function (a, b) { return a - b; });
    return {
      minute: minute, hour: hour, dom: dom, month: month, dow: dow,
      domRestricted: parts[2] !== '*' && parts[2] !== '?',
      dowRestricted: parts[4] !== '*' && parts[4] !== '?'
    };
  }

  function cronMatch(c, d, utc) {
    var minute = utc ? d.getUTCMinutes() : d.getMinutes();
    var hour = utc ? d.getUTCHours() : d.getHours();
    var dom = utc ? d.getUTCDate() : d.getDate();
    var month = (utc ? d.getUTCMonth() : d.getMonth()) + 1;
    var dow = utc ? d.getUTCDay() : d.getDay();
    if (c.minute.indexOf(minute) < 0 || c.hour.indexOf(hour) < 0 || c.month.indexOf(month) < 0) {
      return false;
    }
    var domOk = c.dom.indexOf(dom) >= 0;
    var dowOk = c.dow.indexOf(dow) >= 0;
    if (!c.domRestricted && !c.dowRestricted) return true;
    if (c.domRestricted && c.dowRestricted) return domOk || dowOk;
    return c.domRestricted ? domOk : dowOk;
  }

  function nextCronMs(c, fromMs, utc) {
    var d = new Date(fromMs + 60000);
    d.setSeconds(0, 0);
    // 366 days of minutes; a valid 5-field expression always matches inside
    // that horizon (the worst case is Feb 29, which is < 8 years — so allow
    // 8 years of minutes, still cheap as a bounded loop).
    var limit = 8 * 366 * 24 * 60;
    for (var i = 0; i < limit; i++) {
      if (cronMatch(c, d, utc)) return d.getTime();
      d = new Date(d.getTime() + 60000);
    }
    throw makeError('E_TRIGGER_INVALID',
      'sofuu.triggers: cron expression has no match in the next 8 years');
  }

  /* ── watcher snapshot (poll-based; works on every platform) ───────── */

  function statOf(p) {
    if (!global.sofuu.fs || typeof global.sofuu.fs.statSync !== 'function') {
      throw makeError('E_TRIGGER_INVALID',
        'sofuu.triggers: kind \'watch\' requires sofuu.fs.statSync');
    }
    try {
      var st = global.sofuu.fs.statSync(p);
      return st || null;
    } catch (e) {
      return null;
    }
  }

  function signature(st) {
    if (!st) return 'missing';
    return String(st.is_dir ? 'd' : 'f') + ':' + String(st.mtimeMs || 0) + ':' + String(st.size || 0);
  }

  function listDir(p) {
    if (!global.sofuu.fs || typeof global.sofuu.fs.readdirSync !== 'function') return [];
    try {
      var names = global.sofuu.fs.readdirSync(p);
      return Array.isArray(names) ? names.map(String) : [];
    } catch (e) {
      return [];
    }
  }

  /* ── Runner ───────────────────────────────────────────────────────── */

  var RUNNERS = Object.create(null);
  var SEQ = 0;
  var SIGNALS_INSTALLED = false;

  function openState() {
    try {
      if (global.sofuu.state && typeof global.sofuu.state.open === 'function') {
        return global.sofuu.state.open('triggers');
      }
    } catch (e) { /* a host without the native store falls back to memory. */ }
    return null;
  }

  function installSignalHandlers() {
    if (SIGNALS_INSTALLED) return;
    SIGNALS_INSTALLED = true;
    if (typeof process === 'undefined' || !process || typeof process.on !== 'function') return;
    try {
      process.on('SIGINT', function () { stopAll('SIGINT'); });
      process.on('SIGTERM', function () { stopAll('SIGTERM'); });
    } catch (e) { /* a host without signals simply stops via runner.stop(). */ }
  }

  function validateBudget(b) {
    if (!b || typeof b !== 'object') return {};
    var out = {};
    var keys = ['maxTriggersPerHour', 'maxRunsPerHour', 'maxAiCallsPerDay', 'maxAiCallsPerHour'];
    for (var i = 0; i < keys.length; i++) {
      if (b[keys[i]] !== undefined && b[keys[i]] !== null) {
        var n = Number(b[keys[i]]);
        if (!Number.isFinite(n) || n < 0) {
          throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: budget.' + keys[i] + ' must be >= 0');
        }
        out[keys[i]] = Math.floor(n);
      }
    }
    if (b.throwOnExceed === true) out.throwOnExceed = true;
    return out;
  }

  class TriggerRunner {
    constructor(spec) {
      if (!spec || typeof spec !== 'object') {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers.register: a spec object is required');
      }
      var kind = String(spec.kind || '').toLowerCase();
      if (kind === 'event') kind = 'stream';
      if (['interval', 'cron', 'watch', 'stream'].indexOf(kind) < 0) {
        throw makeError('E_TRIGGER_INVALID',
          'sofuu.triggers.register: kind must be interval | cron | watch | stream/event');
      }
      if (typeof spec.onTrigger !== 'function') {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers.register: onTrigger(payload, ctx) is required');
      }
      if (spec.gate !== undefined && spec.gate !== null && typeof spec.gate !== 'function') {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers.register: gate must be a function');
      }
      if (spec.onError !== undefined && spec.onError !== null && typeof spec.onError !== 'function') {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers.register: onError must be a function');
      }

      this.id = String(spec.id || ('trigger-' + (++SEQ)));
      if (RUNNERS[this.id]) {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers.register: duplicate id \'' + this.id + '\'');
      }
      this.kind = kind;
      this.nonOverlapping = spec.nonOverlapping !== false;
      this.gate = typeof spec.gate === 'function' ? spec.gate : function () { return true; };
      this.onTrigger = spec.onTrigger;
      this.onError = typeof spec.onError === 'function' ? spec.onError : null;
      this.onSkip = typeof spec.onSkip === 'function' ? spec.onSkip : null;
      this.state = (spec.state === null) ? null : (spec.state || openState());
      this.budget = validateBudget(spec.budget);

      var jitter = Number(spec.jitterPct == null ? 0 : spec.jitterPct);
      if (!Number.isFinite(jitter) || jitter < 0 || jitter > 100) {
        throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: jitterPct must be 0..100');
      }
      this.jitterPct = jitter;

      var bo = spec.backoff && typeof spec.backoff === 'object' ? spec.backoff : {};
      this.backoff = {
        initialMs: bo.initialMs == null ? 2000 : finite(bo.initialMs, 'backoff.initialMs'),
        maxMs: bo.maxMs == null ? 60000 : finite(bo.maxMs, 'backoff.maxMs'),
        multiplier: bo.multiplier == null ? 2 : finite(bo.multiplier, 'backoff.multiplier')
      };
      if (this.backoff.initialMs < 0 || this.backoff.maxMs < 1 || this.backoff.multiplier < 1) {
        throw makeError('E_TRIGGER_INVALID',
          'sofuu.triggers: backoff needs initialMs >= 0, maxMs >= 1, multiplier >= 1');
      }

      /* kind-specific configuration */
      if (kind === 'interval') {
        var every = spec.every;
        if (every === undefined || every === null || every === '') {
          var envMs = 0;
          try {
            if (typeof process !== 'undefined' && process.env &&
                process.env.SOFUU_DAEMON_INTERVAL_MS) {
              envMs = Number(process.env.SOFUU_DAEMON_INTERVAL_MS);
            }
          } catch (e) { /* no env in this host */ }
          if (Number.isFinite(envMs) && envMs > 0) every = envMs;
        }
        this.everyMs = parseDuration(every, 'every');
        if (this.everyMs <= 0) {
          throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: interval every must be > 0');
        }
        this.immediate = spec.immediate !== false;
      } else if (kind === 'cron') {
        this.cron = parseCron(spec.cron || spec.expression);
        var tz = String(spec.timezone || spec.tz || 'local').toLowerCase();
        this.cronUtc = tz === 'utc' || tz === 'z';
        this.immediate = spec.immediate === true;
      } else if (kind === 'watch') {
        var paths = spec.paths || spec.watch;
        if (typeof paths === 'string') paths = [paths];
        if (!Array.isArray(paths) || paths.length === 0) {
          throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: watch needs paths (string or array)');
        }
        this.paths = paths.map(String);
        this.debounceMs = parseDuration(spec.debounceMs == null ? 500 : spec.debounceMs, 'debounceMs');
        this.pollMs = parseDuration(spec.pollMs == null ? 1000 : spec.pollMs, 'pollMs');
        if (this.pollMs <= 0) {
          throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers: watch pollMs must be > 0');
        }
        this.depth = Math.max(0, Math.min(8, (spec.depth | 0) || 1));
        this.emitInitial = spec.emitInitial === true;
        this._snap = Object.create(null);
        this._changed = Object.create(null);
        this._debounceTimer = null;
        this._primed = false;
      }

      this._tick = 0;
      this._failures = 0;
      this._timer = null;
      this._busy = false;
      this._stopped = false;
      this._started = false;
      this._donePromise = null;
      this._resolveDone = null;
      this._lastRunAt = 0;
      this._nextRunAt = 0;
      this._healthAt = 0;
      this._stats = {
        ticks: 0, gateSkips: 0, gateHits: 0, aiCalls: 0,
        overlapSkips: 0, budgetSkips: 0, errors: 0,
        lastError: null
      };
      this._memBudget = Object.create(null);

      RUNNERS[this.id] = this;
    }

    /* ── budgets ─────────────────────────────────────────────────── */

    _budgetIncrement(bucket, windowMs, max) {
      if (!(max > 0)) return true;
      if (this.state) {
        try {
          return this.state.incrementWindow('budget:' + this.id + ':' + bucket, windowMs, max).allowed;
        } catch (e) {
          // A broken state store must not silently disable the budget: fall
          // through to the in-memory counter, which at least bounds one run.
        }
      }
      var now = nowMs();
      var w = this._memBudget[bucket];
      if (!w || now - w.start >= windowMs) w = this._memBudget[bucket] = { start: now, count: 0 };
      if (w.count >= max) return false;
      w.count++;
      return true;
    }

    _budgetOk(stage) {
      var b = this.budget;
      var triggers = b.maxTriggersPerHour !== undefined ? b.maxTriggersPerHour : b.maxRunsPerHour;
      var ai = b.maxAiCallsPerDay !== undefined ? b.maxAiCallsPerDay : b.maxAiCallsPerHour;
      var aiWindow = b.maxAiCallsPerDay !== undefined ? 86400000 : 3600000;
      var ok = true;
      if (stage === 'gate') {
        ok = triggers === undefined || this._budgetIncrement('triggers:hour', 3600000, triggers);
      } else if (stage === 'ai') {
        ok = ai === undefined || this._budgetIncrement('ai:window', aiWindow, ai);
      }
      if (!ok && b.throwOnExceed) {
        throw makeError('E_BUDGET_EXCEEDED',
          'sofuu.triggers: budget exceeded for \'' + this.id + '\' at the ' + stage + ' stage',
          { triggerId: this.id, stage: stage });
      }
      return ok;
    }

    /* ── lifecycle ───────────────────────────────────────────────── */

    _done() {
      if (!this._donePromise) {
        var self = this;
        this._donePromise = new Promise(function (resolve) { self._resolveDone = resolve; });
      }
      return this._donePromise;
    }

    _clearTimer() {
      if (this._timer !== null) {
        clearTimeout(this._timer);
        this._timer = null;
      }
      if (this._debounceTimer !== null) {
        clearTimeout(this._debounceTimer);
        this._debounceTimer = null;
      }
    }

    stop(reason) {
      if (this._stopped) return this._done();
      this._stopped = true;
      this.stopReason = String(reason || 'stopped');
      this._clearTimer();
      this._nextRunAt = 0;
      if (this._resolveDone) this._resolveDone({ id: this.id, stopped: this.stopReason, stats: this.stats() });
      this._publishHealth(true);
      return this._done();
    }

    async start() {
      if (this._started) return this._done();
      this._started = true;
      installSignalHandlers();
      if (this.kind === 'interval') {
        this._arm(this.immediate ? 0 : this.everyMs);
      } else if (this.kind === 'cron') {
        this._armCron();
      } else if (this.kind === 'watch') {
        this._pollWatch();
      }
      // stream: no timer — push() drives it.
      this._publishHealth(true);
      return this._done();
    }

    _arm(delay) {
      if (this._stopped) return;
      var d = Math.max(1, Math.floor(delay));
      if (this.jitterPct > 0 && this._failures === 0) {
        d = Math.max(1, Math.floor(d * (1 + (Math.random() * 2 - 1) * (this.jitterPct / 100))));
      }
      this._nextRunAt = nowMs() + d;
      var self = this;
      this._timer = setTimeout(async function () {
        self._timer = null;
        if (self._stopped) return;
        try { await self._fire({ kind: 'interval', triggerId: self.id, tick: self._tick + 1, nowMs: nowMs() }); }
        catch (e) { /* _fire reports failures through onError; never break the loop. */ }
        if (self._stopped) return;
        var next = self._failures > 0 ? self._backoffMs() : self.everyMs;
        self._arm(next);
      }, d);
    }

    _backoffMs() {
      var b = this.backoff;
      var n = Math.max(0, this._failures - 1);
      var ms = b.initialMs * Math.pow(b.multiplier, n);
      return Math.max(1, Math.min(b.maxMs, Math.floor(ms)));
    }

    _armCron() {
      if (this._stopped) return;
      var self = this;
      var next;
      try {
        next = nextCronMs(this.cron, nowMs(), this.cronUtc);
      } catch (e) {
        this._stopped = true;
        if (this._resolveDone) this._resolveDone({ id: this.id, stopped: 'error', error: String((e && e.message) || e) });
        return;
      }
      this._nextRunAt = next;
      this._timer = setTimeout(async function () {
        self._timer = null;
        if (self._stopped) return;
        try { await self._fire({ kind: 'cron', triggerId: self.id, tick: self._tick + 1, nowMs: nowMs() }); }
        catch (e) { /* same contract as the interval loop. */ }
        if (self._stopped) return;
        if (self._failures > 0) {
          // Back off without losing the cron schedule: the next regular
          // occurrence is still the anchor; the error delay is a floor.
          var retry = self._backoffMs();
          self._timer = setTimeout(function () {
            self._timer = null;
            if (!self._stopped) self._armCron();
          }, retry);
        } else {
          self._armCron();
        }
      }, Math.max(1, next - nowMs()));
    }

    /* ── watch ───────────────────────────────────────────────────── */

    _walk(root, depth, out) {
      var st = statOf(root);
      if (!st) return;
      out.push(root);
      if (!st.is_dir || depth <= 0) return;
      var names = listDir(root);
      for (var i = 0; i < names.length && i < 512; i++) {
        this._walk(root.replace(/[\\/]+$/, '') + '/' + names[i], depth - 1, out);
      }
    }

    _pollWatch() {
      if (this._stopped) return;
      var self = this;
      var changed = Object.create(null);
      for (var i = 0; i < this.paths.length; i++) {
        var roots = [];
        this._walk(this.paths[i], this.depth, roots);
        for (var r = 0; r < roots.length; r++) {
          var p = roots[r];
          var sig = signature(statOf(p));
          if (this._snap[p] !== sig) {
            changed[p] = true;
            this._snap[p] = sig;
          }
        }
      }
      var keys = Object.keys(changed);
      if (keys.length) {
        if (this._primed || this.emitInitial) {
          for (var k = 0; k < keys.length; k++) this._changed[keys[k]] = true;
          if (this._debounceTimer !== null) clearTimeout(this._debounceTimer);
          this._debounceTimer = setTimeout(function () {
            self._debounceTimer = null;
            var events = Object.keys(self._changed).map(function (path) {
              var st = statOf(path);
              return { path: path, mtimeMs: st ? Number(st.mtimeMs || 0) : 0,
                       size: st ? Number(st.size || 0) : 0,
                       isDir: !!(st && st.is_dir) };
            });
            self._changed = Object.create(null);
            if (self._stopped) return;
            self._fire({ kind: 'watch', triggerId: self.id, tick: self._tick + 1,
                         nowMs: nowMs(), paths: self.paths.slice(), events: events });
          }, this.debounceMs);
        }
      }
      this._primed = true;
      this._nextRunAt = nowMs() + this.pollMs;
      this._timer = setTimeout(function () {
        self._timer = null;
        self._pollWatch();
      }, this.pollMs);
    }

    /* ── one tick ────────────────────────────────────────────────── */

    async _fire(signal) {
      if (this._stopped) return { skipped: 'stopped' };
      if (this._busy && this.nonOverlapping) {
        this._stats.overlapSkips++;
        return { skipped: 'overlap' };
      }
      this._busy = true;
      var tick = ++this._tick;
      var startedAt = nowMs();
      var ctx = {
        triggerId: this.id,
        tick: tick,
        nowMs: startedAt,
        state: this.state,
        signal: signal,
        runner: this
      };
      this._stats.ticks++;
      var payload;
      try {
        payload = await this.gate(ctx);
      } catch (eGate) {
        this._busy = false;
        this._failed(eGate, ctx, 'gate');
        return { skipped: 'error', error: String((eGate && eGate.message) || eGate) };
      }
      if (this._stopped) {
        // A stop that arrived while the gate was fetching must not start the
        // AI stage afterwards: the operator asked for the loop to be quiet.
        this._busy = false;
        return { skipped: 'stopped' };
      }
      if (payload === null || payload === undefined || payload === false) {
        this._stats.gateSkips++;
        this._succeeded(startedAt);
        this._busy = false;
        return { skipped: 'gate' };
      }
      this._stats.gateHits++;
      var withinBudget;
      try {
        withinBudget = this._budgetOk('gate');
      } catch (eBud) {
        this._busy = false;
        this._failed(eBud, ctx, 'budget');
        return { skipped: 'budget', error: String((eBud && eBud.message) || eBud) };
      }
      if (!withinBudget) {
        this._stats.budgetSkips++;
        this._skip('budget', payload, ctx);
        this._succeeded(startedAt);
        this._busy = false;
        return { skipped: 'budget', payload: payload };
      }
      try {
        withinBudget = this._budgetOk('ai');
      } catch (eBud2) {
        this._busy = false;
        this._failed(eBud2, ctx, 'budget');
        return { skipped: 'budget', error: String((eBud2 && eBud2.message) || eBud2) };
      }
      if (!withinBudget) {
        this._stats.budgetSkips++;
        this._skip('budget', payload, ctx);
        this._succeeded(startedAt);
        this._busy = false;
        return { skipped: 'budget', payload: payload };
      }
      try {
        this._stats.aiCalls++;
        await this.onTrigger(payload, ctx);
        this._succeeded(startedAt);
        this._busy = false;
        return { fired: true, payload: payload, tick: tick };
      } catch (eTrigger) {
        this._busy = false;
        this._failed(eTrigger, ctx, 'onTrigger');
        return { skipped: 'error', error: String((eTrigger && eTrigger.message) || eTrigger) };
      }
    }

    _skip(reason, payload, ctx) {
      if (!this.onSkip) return;
      try {
        var r = this.onSkip({ reason: reason, triggerId: this.id, payload: payload }, ctx);
        if (r && typeof r.catch === 'function') r.catch(function () {});
      } catch (e) { /* a skip hook is diagnostic only. */ }
    }

    _succeeded(startedAt) {
      this._failures = 0;
      this._lastRunAt = nowMs();
      this._stats.lastDurationMs = nowMs() - startedAt;
      this._publishHealth(false);
    }

    _failed(err, ctx, stage) {
      this._failures++;
      this._stats.errors++;
      this._stats.lastError = {
        stage: stage,
        code: (err && err.code) || 'E_TRIGGER_INVALID',
        message: String((err && err.message) || err || 'unknown trigger failure'),
        at: nowMs()
      };
      this._publishHealth(true);
      if (this.onError) {
        try {
          var r = this.onError(err, ctx);
          if (r && typeof r.catch === 'function') r.catch(function () {});
        } catch (e) { /* the error hook is diagnostic; backoff is the policy. */ }
      }
    }

    /* Live health for `sofuu daemon status`: a compact snapshot under
     * .sofuu/state/triggers.json. Throttled on the success path (a 1s loop
     * must not fsync twice a second for a metric), immediate on failure. */
    _publishHealth(force) {
      if (global.sofuu && global.sofuu.ui && typeof global.sofuu.ui._onTriggerTick === 'function') {
        try {
          global.sofuu.ui._onTriggerTick(Object.keys(RUNNERS).map(function (k) { return RUNNERS[k].stats(); }));
        } catch (eUiT) { /* UI telemetry never blocks trigger scheduling. */ }
      }
      if (!this.state) return;
      var t = nowMs();
      if (!force && t - this._healthAt < 5000) return;
      this._healthAt = t;
      try {
        this.state.set('health:' + this.id, this.stats(), { ttlMs: 7 * 86400000 });
      } catch (e) { /* health is observability; it never changes the loop. */ }
    }

    /* ── stream feed ─────────────────────────────────────────────── */

    push(payload) {
      if (this.kind !== 'stream') {
        throw makeError('E_TRIGGER_INVALID',
          'sofuu.triggers: runner \'' + this.id + '\' is ' + this.kind + ', not stream');
      }
      var signal = { kind: 'stream', triggerId: this.id, tick: this._tick + 1,
                     nowMs: nowMs(), event: payload };
      if (this._failures > 0) {
        // Backoff is a throttle for feeds too: a failing consumer must not
        // turn a high-rate webhook into a hot loop.
        var self = this;
        return new Promise(function (resolve) {
          setTimeout(function () { resolve(self._fire(signal)); }, self._backoffMs());
        });
      }
      return this._fire(signal);
    }

    emit(payload) { return this.push(payload); }

    stats() {
      return {
        id: this.id,
        kind: this.kind,
        running: this._started && !this._stopped,
        ticks: this._stats.ticks,
        gateSkips: this._stats.gateSkips,
        gateHits: this._stats.gateHits,
        aiCalls: this._stats.aiCalls,
        overlapSkips: this._stats.overlapSkips,
        budgetSkips: this._stats.budgetSkips,
        errors: this._stats.errors,
        lastError: this._stats.lastError,
        lastRunAt: this._lastRunAt,
        nextRunAt: this._nextRunAt,
        lastDurationMs: this._stats.lastDurationMs || 0,
        failures: this._failures,
        stopped: this._stopped ? this.stopReason || 'stopped' : ''
      };
    }
  }

  function stopAll(reason) {
    var ids = Object.keys(RUNNERS);
    for (var i = 0; i < ids.length; i++) {
      try { RUNNERS[ids[i]].stop(reason); } catch (e) { /* one bad runner must not leak the rest. */ }
    }
  }

  /**
   * Run one gate/trigger pass without starting a loop. Useful for `--once`
   * modes and tests; it uses the same runner semantics (non-overlap is
   * irrelevant for a single call, budgets still apply).
   */
  async function runOnce(spec) {
    var runner = new TriggerRunner(Object.assign({}, spec, { kind: spec.kind || 'stream' }));
    try {
      var out = await runner._fire({ kind: 'once', triggerId: runner.id, tick: 1, nowMs: nowMs() });
      return out;
    } finally {
      runner._stopped = true;
      runner._clearTimer();
      delete RUNNERS[runner.id];
    }
  }

  var api = {
    register: function (spec) { return new TriggerRunner(spec); },
    get: function (id) { return RUNNERS[String(id)] || null; },
    stats: function (id) {
      if (id !== undefined && id !== null) {
        var r = RUNNERS[String(id)];
        return r ? r.stats() : null;
      }
      return Object.keys(RUNNERS).map(function (k) { return RUNNERS[k].stats(); });
    },
    stopAll: stopAll,
    emit: function (id, payload) {
      var r = RUNNERS[String(id)];
      if (!r) throw makeError('E_TRIGGER_INVALID', 'sofuu.triggers.emit: unknown trigger \'' + id + '\'');
      return r.push(payload);
    },
    runOnce: runOnce,
    parseCron: parseCron,
    nextCronMs: function (expr, fromMs, tz) {
      var c = parseCron(expr);
      var utc = String(tz || 'local').toLowerCase() === 'utc';
      return nextCronMs(c, Number.isFinite(Number(fromMs)) ? Number(fromMs) : nowMs(), utc);
    },
    parseDuration: parseDuration,
    TriggerRunner: TriggerRunner
  };

  global.sofuu.triggers = api;
  if (!global.Sofuu) global.Sofuu = {};
  global.Sofuu.triggers = api;
})();
