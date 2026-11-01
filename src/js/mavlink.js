// src/js/mavlink.js — the `sofuu.mavlink` surface (PLAN-DRONES-MAVLINK §4.4).
//
// @sofuu-source-only
//
// This file is a thin wrapper: every byte that can reach a wire is written by
// the Rust layer (`rt/mavlink_link.rs` → `rt/mavlink_gate.rs`), and this script
// has no way to bypass it. It exists to give the API its documented shape:
//
//   const fc = sofuu.mavlink.connect("udp:192.168.1.20:14550");
//   fc.on("GLOBAL_POSITION_INT", (e) => telemetry.push(e.data));
//   fc.on("ack", (e) => console.log(e.command_name, e.result_name));
//   fc.send({ type: "COMMAND_LONG", command: "MAV_CMD_NAV_TAKEOFF", alt: 10 });
//   fc.close();
//
// Capability gate: `__sofuu_mavlink_enabled` is registered by the engine ONLY
// when the policy's `hardware.enabled` is true, and it re-reads the policy.
// If either says no, this script returns without defining the namespace —
// `sofuu.mavlink` does not exist rather than existing and refusing.

(function () {
  'use strict';

  const g = globalThis;
  if (typeof g.__sofuu_mavlink_enabled !== 'function') return;

  let armed = false;
  try {
    armed = !!g.__sofuu_mavlink_enabled();
  } catch (_) {
    armed = false;
  }
  if (!armed) return;

  const root = g.sofuu || (g.sofuu = {});
  if (root.mavlink) return; // never replace an existing surface

  /* The pump interval. 20 ms is fast enough for 10 Hz decision loops and slow
   * enough that an idle link costs nothing measurable. */
  const POLL_MS = 20;

  function errFrom(r) {
    const e = new Error(r.error || 'mavlink failure');
    if (r.code) e.code = r.code;
    if (r.rule) e.rule = r.rule;
    /* Present on a rate-limit refusal: how long until the same frame would be
     * allowed. The skill layer paces with this rather than guessing. */
    if (typeof r.retry_after_ms === 'number') e.retry_after_ms = r.retry_after_ms;
    return e;
  }

  class Connection {
    constructor(id, device) {
      this.id = id;
      this.device = device;
      this._listeners = new Map();
      this._closed = false;
      this._errors = 0;
      /* Start pumping immediately: the runtime's dead-man heartbeat (§4.5)
       * runs inside the pump, so an idle-but-open link must still tick. */
      this._timer = setInterval(() => this._pump(), POLL_MS);
      if (this._timer && typeof this._timer.unref === 'function') this._timer.unref();
    }

    /** Subscribe to a message name ("GLOBAL_POSITION_INT"), an event kind
     *  ("ack" / "statustext" / "resync" / "error"), or "*" for everything. */
    on(type, fn) {
      if (typeof fn !== 'function') throw new TypeError('mavlink.on(type, fn) needs a function');
      const t = String(type);
      const set = this._listeners.get(t) || new Set();
      set.add(fn);
      this._listeners.set(t, set);
      return this;
    }

    off(type, fn) {
      const set = this._listeners.get(String(type));
      if (!set) return this;
      if (fn) set.delete(fn);
      else set.clear();
      if (set.size === 0) this._listeners.delete(String(type));
      return this;
    }

    once(type, fn) {
      const self = this;
      const wrap = function (ev, state) {
        self.off(type, wrap);
        return fn(ev, state);
      };
      return this.on(type, wrap);
    }

    /** Send one frame. Throws a typed Error carrying `.code` and `.rule` when
     *  the gate refuses it — nothing was written. */
    send(obj) {
      if (this._closed) throw new Error('mavlink connection is closed');
      const r = JSON.parse(g.__sofuu_mavlink_send(this.id, JSON.stringify(obj || {})));
      if (!r.ok) throw errFrom(r);
      return r;
    }

    /** Non-sending: what the gate would decide, plus live GPS state. */
    why(obj) {
      const r = this.status();
      return { policy: r, would_send: JSON.parse(JSON.stringify(obj || {})) };
    }

    status() {
      const r = JSON.parse(g.__sofuu_mavlink_status(this.id));
      if (!r.ok) throw errFrom(r);
      return r;
    }

    close() {
      if (this._closed) return;
      this._closed = true;
      if (this._timer) {
        clearInterval(this._timer);
        this._timer = null;
      }
      this._listeners.clear();
      g.__sofuu_mavlink_close(this.id);
    }

    _pump() {
      if (this._closed) return;
      const types = Array.from(this._listeners.keys());
      let r;
      try {
        r = JSON.parse(g.__sofuu_mavlink_poll(this.id, JSON.stringify(types)));
      } catch (e) {
        /* A throwing pump would kill the interval and silently stop the
         * heartbeat. Count and keep the link alive; a listener that throws is
         * likewise isolated below. */
        this._errors++;
        return;
      }
      const events = r.events || [];
      for (let i = 0; i < events.length; i++) this._dispatch(events[i], r.state);
    }

    _dispatch(ev, state) {
      const fire = (name) => {
        const set = this._listeners.get(name);
        if (!set) return;
        const fns = Array.from(set);
        for (let i = 0; i < fns.length; i++) {
          try {
            fns[i](ev, state);
          } catch (_) {
            this._errors++;
          }
        }
      };
      fire('*');
      if (!ev || !ev.event) return;
      if (ev.event === 'telemetry') fire(ev.message);
      else if (ev.event === 'ack') {
        fire('ack');
        fire('COMMAND_ACK');
      } else if (ev.event === 'statustext') fire('STATUSTEXT');
      else if (ev.event === 'arming_denied') fire('E_ARMING_DENIED');
      else if (ev.event === 'resync') fire('resync');
      else if (ev.event === 'error') fire('error');
    }
  }

  root.mavlink = {
    connect(device) {
      if (device === undefined || device === null) {
        throw new TypeError('sofuu.mavlink.connect(device) needs a device, e.g. "udp:127.0.0.1:14550"');
      }
      const r = JSON.parse(g.__sofuu_mavlink_connect(String(device)));
      if (!r.ok) throw errFrom(r);
      return new Connection(r.id, r.device);
    },
  };
})();
