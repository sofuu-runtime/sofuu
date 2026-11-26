// state.js — `sofuu.state`: operational KV, atomic cooldowns, rate windows.
//
// PLAN-AUTONOMOUS-REACTIVE-RUNTIME Phase 1. This is the explicit, exact
// counterpart to semantic memory: cooldown stamps, cursors, watermarks and
// counters that must survive a restart and must never decay. The native half
// lives in crates/sofuu-core/src/rt/state.rs; this driver only shapes the
// JS surface and turns host failures into structured SofuuErrors.
//
//   const store = sofuu.state.open('alerts');
//   if (store.allowAfterCooldown('BTC-USD:TARGET_ABOVE:65000', 30 * 60_000)) {
//     store.incrementWindow('alerts:hour', 3_600_000, 20);
//     // … dispatch
//   }
//
// Isolation: a namespace is a FILE NAME inside <project>/.sofuu/state/
// (or SOFUU_STATE_ROOT). Path separators and '..' are refused by the host,
// never sanitised into a different file.
//
// Errors (`RULES.md §3.5`): E_STATE_IO for every storage/validation failure.

(function () {
  'use strict';
  var global = globalThis;
  if (!global.sofuu) global.sofuu = {};

  function makeError(code, msg, details) {
    var err = new Error(msg);
    err.name = 'SofuuError';
    err.code = code || 'E_STATE_IO';
    err.rule = 'RULES.md §3.5';
    if (details) err.details = details;
    return err;
  }

  function host(name) {
    var fn = global[name];
    if (typeof fn !== 'function') {
      throw makeError('E_STATE_IO', 'sofuu.state: the native state store is unavailable in this build (missing ' + name + ')');
    }
    return function () {
      try {
        return fn.apply(null, arguments);
      } catch (e) {
        if (e && e.code) throw e;
        /* The native half throws a TypeError whose message carries the
         * structured code (`E_STATE_IO: ...`). Promote it so callers see one
         * error shape across both halves. */
        var m = String((e && e.message) || e || '');
        var mm = /^(E_[A-Z0-9_]+)\s*:/.exec(m);
        if (mm) throw makeError(mm[1], m);
        throw e;
      }
    };
  }

  function parseJson(raw, what) {
    try {
      return JSON.parse(String(raw));
    } catch (e) {
      throw makeError('E_STATE_IO', 'sofuu.state: ' + what + ' returned malformed JSON: ' + String((e && e.message) || e));
    }
  }

  class StateStore {
    constructor(namespace, opts) {
      opts = opts || {};
      var ns = (namespace === undefined || namespace === null || namespace === '')
        ? 'default' : String(namespace);
      var root = (opts.root === undefined || opts.root === null) ? null : String(opts.root);
      this.namespace = ns;
      this.handle = host('__sofuu_state_open')(ns, root);
      this._closed = false;
    }

    _ensure() {
      if (this._closed || !this.handle) {
        throw makeError('E_STATE_IO', 'sofuu.state: this StateStore is closed');
      }
    }

    get(key) {
      this._ensure();
      if (key === undefined || key === null) {
        throw makeError('E_STATE_IO', 'sofuu.state.get: key is required');
      }
      var raw = host('__sofuu_state_get')(this.handle, String(key));
      if (raw === null || raw === undefined) return null;
      return parseJson(raw, 'get');
    }

    set(key, value, opts) {
      this._ensure();
      if (key === undefined || key === null) {
        throw makeError('E_STATE_IO', 'sofuu.state.set: key is required');
      }
      var json;
      try {
        json = JSON.stringify(value);
      } catch (e) {
        throw makeError('E_STATE_IO', 'sofuu.state.set: value is not JSON-serializable: ' + String((e && e.message) || e));
      }
      if (json === undefined) {
        throw makeError('E_STATE_IO', 'sofuu.state.set: value is not JSON-serializable (undefined)');
      }
      var ttl = 0;
      if (opts && opts.ttlMs !== undefined && opts.ttlMs !== null) {
        ttl = Number(opts.ttlMs);
        if (!Number.isFinite(ttl) || ttl < 0) {
          throw makeError('E_STATE_IO', 'sofuu.state.set: ttlMs must be a finite number >= 0');
        }
        ttl = Math.floor(ttl);
      }
      return !!host('__sofuu_state_set')(this.handle, String(key), json, ttl);
    }

    delete(key) {
      this._ensure();
      return !!host('__sofuu_state_delete')(this.handle, String(key));
    }

    keys(prefix) {
      this._ensure();
      var raw = host('__sofuu_state_keys')(this.handle,
        (prefix === undefined || prefix === null) ? '' : String(prefix));
      var out = parseJson(raw, 'keys');
      return Array.isArray(out) ? out : [];
    }

    /**
     * Atomic check-and-stamp. Returns true when `cooldownMs` has elapsed
     * since the last allowed call (and, unless dryRun, stamps NOW in the
     * same operation). Two racing callers cannot both receive true.
     */
    allowAfterCooldown(key, cooldownMs, opts) {
      this._ensure();
      if (key === undefined || key === null) {
        throw makeError('E_STATE_IO', 'sofuu.state.allowAfterCooldown: key is required');
      }
      var ms = Number(cooldownMs);
      if (!Number.isFinite(ms) || ms < 0) {
        throw makeError('E_STATE_IO', 'sofuu.state.allowAfterCooldown: cooldownMs must be a finite number >= 0');
      }
      var dry = !!(opts && opts.dryRun);
      return !!host('__sofuu_state_cooldown')(this.handle, String(key), Math.floor(ms), dry);
    }

    resetCooldown(key) {
      this._ensure();
      return !!host('__sofuu_state_reset_cooldown')(this.handle, String(key));
    }

    /**
     * Rolling window counter. With maxAllowed, a call is allowed only while
     * the window holds fewer than that many successful increments; the count
     * saturates at the cap, so a hot loop cannot grow the document forever.
     */
    incrementWindow(key, windowMs, maxAllowed) {
      this._ensure();
      var win = Number(windowMs);
      if (!Number.isFinite(win) || win <= 0) {
        throw makeError('E_STATE_IO', 'sofuu.state.incrementWindow: windowMs must be a finite number > 0');
      }
      var max = -1;
      if (maxAllowed !== undefined && maxAllowed !== null) {
        max = Number(maxAllowed);
        if (!Number.isFinite(max) || max < 0) {
          throw makeError('E_STATE_IO', 'sofuu.state.incrementWindow: maxAllowed must be a finite number >= 0');
        }
      }
      var raw = host('__sofuu_state_window')(this.handle, String(key), Math.floor(win), max);
      var out = parseJson(raw, 'incrementWindow');
      return { count: Number(out.count) || 0, allowed: !!out.allowed };
    }

    /** Flush is implicit (every mutation commits); close just drops the handle. */
    close() {
      if (this._closed) return false;
      this._closed = true;
      var h = this.handle;
      this.handle = null;
      return !!host('__sofuu_state_close')(h);
    }
  }

  var state = {
    open: function (namespace, opts) { return new StateStore(namespace, opts); },
    StateStore: StateStore
  };

  global.sofuu.state = state;
  if (!global.Sofuu) global.Sofuu = {};
  global.Sofuu.state = state;
})();
