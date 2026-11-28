/*
 * v2_ui/bridge.js — the one JS -> Android channel.
 *
 * The Kotlin side registers `globalThis.__sofuu_android_invoke(action, argsJson)`
 * once the host-function ABI is wired (see PLAN-SOFUU-ANDROID-V2 §6). When it
 * is absent — inside sofuu-emu, on desktop, or in this first shell — calls
 * return a typed `E_HOST_UNAVAILABLE` envelope. It never fabricates success
 * (RULES.md §1.9): an unwired OS action is an error the UI can render.
 */
(function () {
  'use strict';
  var g = globalThis;

  function hostFn() {
    if (typeof g.__sofuu_android_invoke === 'function') return g.__sofuu_android_invoke;
    return null;
  }

  var api = {
    get available() {
      return hostFn() !== null;
    },

    /**
     * Invoke a native action. Synchronous by design: JNI callbacks into JS
     * are not part of the ABI yet, so callers get an envelope now and async
     * results later arrive as `sofuu.ui` agent events.
     */
    invoke: function (action, args) {
      var fn = hostFn();
      if (!fn) {
        return {
          ok: false,
          code: 'E_HOST_UNAVAILABLE',
          message: 'Android host bridge is not attached (' + String(action) + ')'
        };
      }
      try {
        var raw = fn(String(action), JSON.stringify(args || {}));
        if (raw === null || raw === undefined) {
          return { ok: false, code: 'E_HOST_NO_RESULT', message: String(action) };
        }
        return JSON.parse(String(raw));
      } catch (e) {
        return {
          ok: false,
          code: 'E_HOST_BRIDGE',
          message: String((e && e.message) || e)
        };
      }
    }
  };

  g.SofuuAndroid = api;
})();
