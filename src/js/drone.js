// src/js/drone.js — `sofuu.drone`: profiles, skills and missions (§4.6 L2/L3).
//
// @sofuu-source-only
//
// Every skill here expands to frames that cross the SAME gate as a raw
// `sofuu.mavlink` send — a prebuilt skill is a pre-vetted policy shape, never a
// bypass. A mission is validated in full (by the Rust `preflight`) before the
// first frame exists, and every frame is still re-checked as it is sent.
//
//   const drone = sofuu.drone.profile("udp:127.0.0.1:14550");
//   await drone.takeoff(20);
//   await drone.goto(52.1234, 4.1235, 20);
//   await drone.rtl();
//
//   await sofuu.drone.mission(drone, {
//     steps: [{type:'takeoff',alt:20},{type:'survey',...},{type:'rtl'}],
//     safety: { maxAlt: 40, geofenceRadius: 300 }
//   });

(function () {
  'use strict';

  const g = globalThis;
  const sofuu = g.sofuu;
  // Inert unless the capability-gated MAVLink surface exists.
  if (!sofuu || !sofuu.mavlink) return;
  if (sofuu.drone) return;

  const ACK_TIMEOUT_MS = 5000;

  /* MAV_CMD ids a skill needs. Kept in step with rt/mavlink.rs's `cmd` module;
   * a mismatch shows up as an "unknown MAV_CMD" error from the frame builder
   * rather than a wrong command, because the builder resolves BY NAME. */
  const CMD = {
    ARM_DISARM: 'MAV_CMD_COMPONENT_ARM_DISARM',
    TAKEOFF: 'MAV_CMD_NAV_TAKEOFF',
    RTL: 'MAV_CMD_NAV_RETURN_TO_LAUNCH',
    LAND: 'MAV_CMD_NAV_LAND',
    LOITER: 'MAV_CMD_NAV_LOITER_UNLIM',
    SPEED: 'MAV_CMD_DO_CHANGE_SPEED',
    ORBIT: 'MAV_CMD_DO_ORBIT',
    WAYPOINT: 'MAV_CMD_NAV_WAYPOINT',
    SET_MODE: 'MAV_CMD_DO_SET_MODE',
  };

  /* ArduCopter custom-mode numbers. `mode('GUIDED')` or `mode(4)`. Selecting a
   * mode is not optional: ArduCopter boots disarmed in STABILIZE and refuses
   * NAV_TAKEOFF there, so without this the whole skill layer is unusable until
   * an operator sets the mode from a GCS. */
  const COPTER_MODES = {
    STABILIZE: 0, ACRO: 1, ALT_HOLD: 2, AUTO: 3, GUIDED: 4, LOITER: 5,
    RTL: 6, LAND: 9, BRAKE: 17, GUIDED_NOGPS: 20,
  };

  function commandLong(name, params, extra) {
    const o = { type: 'COMMAND_LONG', command: name };
    for (const k in params || {}) o[k] = params[k];
    return Object.assign(o, extra || {});
  }

  /** A connection with skills attached. */
  class Drone {
    constructor(conn, device) {
      this.fc = conn;
      this.device = device;
      this._pending = [];
      // One listener for the whole drone: skills share the ACK stream.
      conn.on('ack', (ev) => {
        for (let i = this._pending.length - 1; i >= 0; i--) {
          const p = this._pending[i];
          if (p.command === ev.command) {
            this._pending.splice(i, 1);
            if (ev.result === 0) p.resolve(ev);
            else {
              const e = new Error(ev.text ? ev.text : 'vehicle refused: ' + ev.result_name);
              e.code = ev.code || 'E_COMMAND_FAILED';
              e.result = ev.result_name;
              p.reject(e);
            }
          }
        }
      });
    }

    /** Wait until the endpoint has heard from the vehicle. With the `udp:`
     *  bind-and-learn model there is no address to send to until the FC (or
     *  SITL) speaks, so firing immediately fails on the transport rather than
     *  on anything real. `udpout:` endpoints are connected from the start. */
    _waitLink(timeoutMs) {
      if (this._linked) return Promise.resolve(true);
      const deadline = Date.now() + (timeoutMs || 3000);
      const self = this;
      return new Promise((resolve) => {
        const tick = () => {
          let st = null;
          try { st = self.fc.status(); } catch (_) { st = null; }
          if (st && st.connected) { self._linked = true; resolve(true); return; }
          if (Date.now() >= deadline) { resolve(false); return; }
          setTimeout(tick, 50);
        };
        tick();
      });
    }

    /** Send one command and resolve on its ACK (rejects with the gate's typed
     *  error when the frame never left, or with the FC's reason when it did). */
    _command(obj, commandId) {
      return this._waitLink(3000).then((linked) => {
        if (!linked) {
          const e = new Error('E_NO_LINK: no telemetry from ' + this.device +
            ' within 3 s — nothing to send to (use udpout: if this side must speak first)');
          e.code = 'E_NO_LINK';
          throw e;
        }
        return this._sendCommand(obj, commandId);
      });
    }

    _sendCommand(obj, commandId) {
      return new Promise((resolve, reject) => {
        const entry = { command: commandId, resolve, reject };
        let attempts = 0;
        let ackTimer = null;
        const fail = (e) => {
          if (ackTimer) clearTimeout(ackTimer);
          const i = this._pending.indexOf(entry);
          if (i >= 0) this._pending.splice(i, 1);
          reject(e);
        };
        const attempt = () => {
          try {
            this.fc.send(obj);
          } catch (e) {
            /* The gate's rate limit is a FLOOR, not a suggestion: instead of
             * retrying in a tight loop (which is exactly what the limit exists
             * to stop) the skill layer waits the interval the gate reports, a
             * bounded number of times, and then surfaces the refusal. */
            if (e.rule === 'R-HW/rateLimits' && attempts++ < 3) {
              const wait = typeof e.retry_after_ms === 'number' ? e.retry_after_ms + 10 : 130;
              setTimeout(attempt, wait);
              return;
            }
            fail(e);
            return;
          }
          this._pending.push(entry);
          ackTimer = setTimeout(() => {
            const err = new Error('E_TIMEOUT: no COMMAND_ACK for ' + obj.command +
              ' within ' + ACK_TIMEOUT_MS + ' ms');
            err.code = 'E_TIMEOUT';
            fail(err);
          }, ACK_TIMEOUT_MS);
          if (ackTimer && typeof ackTimer.unref === 'function') ackTimer.unref();
        };
        attempt();
      });
    }

    /** Select a flight mode by name ('GUIDED') or custom-mode number. */
    mode(name) {
      let custom;
      if (typeof name === 'number') custom = name;
      else custom = COPTER_MODES[String(name || '').trim().toUpperCase()];
      if (custom === undefined) {
        const e = new Error("E_BAD_MODE: unknown flight mode '" + name + "'");
        e.code = 'E_BAD_MODE';
        return Promise.reject(e);
      }
      /* param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (1), param2 = custom mode. */
      return this._command(commandLong(CMD.SET_MODE, { param1: 1, param2: custom }), 176);
    }

    arm() {
      return this._command(commandLong(CMD.ARM_DISARM, { param1: 1 }), 400);
    }

    disarm() {
      return this._command(commandLong(CMD.ARM_DISARM, { param1: 0 }), 400);
    }

    /** Take off and resolve when the vehicle is actually at altitude.
     *
     *  The ACK means the command was accepted, not that the climb happened.
     *  Resolving on the ACK let a following `goto` stream its position targets
     *  while the vehicle was still climbing — and targets are ignored during
     *  the takeoff phase, so the goto silently burned its whole window. */
    async takeoff(altM) {
      const r = await this._command(commandLong(CMD.TAKEOFF, { alt: altM }), 22);
      await this._waitAlt(altM, 30000);
      return r;
    }

    /** Wait until the reported altitude is within `tolM` of `targetM`. */
    async _waitAlt(targetM, timeoutMs, tolM) {
      const tol = tolM === undefined ? 1.0 : tolM;
      const deadline = Date.now() + (timeoutMs || 30000);
      while (Date.now() < deadline) {
        let a = null;
        try { a = this.fc.status().relative_alt_m; } catch (_) { a = null; }
        if (typeof a === 'number' && Math.abs(a - targetM) <= tol) return a;
        await new Promise((r) => setTimeout(r, 100));
      }
      const e = new Error(
        'E_NO_CLIMB: altitude never reached ' + targetM + ' m ± ' + tol + ' m'
      );
      e.code = 'E_NO_CLIMB';
      throw e;
    }

    rtl() {
      return this._command(commandLong(CMD.RTL), 20);
    }

    land() {
      return this._command(commandLong(CMD.LAND), 21);
    }

    hold() {
      return this._command(commandLong(CMD.LOITER), 17);
    }

    setSpeed(ms) {
      return this._command(commandLong(CMD.SPEED, { speed: ms }), 178);
    }

    /** Fly to a point. Resolves when the vehicle has actually arrived.
     *
     *  One frame is not enough: the autopilot's guided target expires after
     *  GUID_TIMEOUT (~3 s), so a single send moves the vehicle for three
     *  seconds and then it stops in mid-air. The target is therefore streamed
     *  at 5 Hz until the distance closes, and if it never closes this rejects
     *  with E_NO_PROGRESS instead of reporting a frame as a completed flight. */
    async goto(lat, lon, altM, opts) {
      const o = opts || {};
      if (!(await this._waitLink(3000))) {
        const e = new Error('E_NO_LINK: no telemetry from ' + this.device + ' within 3 s');
        e.code = 'E_NO_LINK';
        throw e;
      }
      if (o.speed !== undefined) await this.setSpeed(o.speed);

      const dist = (p) => {
        const dLat = (lat - p.lat) * 111320;
        const dLon = (lon - p.lon) * 111320 * Math.cos((lat * Math.PI) / 180);
        return Math.sqrt(dLat * dLat + dLon * dLon);
      };
      const start = this._position();
      const d0 = start ? dist(start) : NaN;
      let best = d0;
      const deadline = Date.now() + (o.timeoutMs || 30000);

      while (Date.now() < deadline) {
        this.fc.send({
          type: 'SET_POSITION_TARGET_GLOBAL_INT',
          lat: lat,
          lon: lon,
          alt: altM,
          /* POS_ONLY (3576). Do NOT send vx/vy/vz: a velocity target of zero
           * is *commanded* zero velocity, and the autopilot then holds station
           * while the position target is ignored — a silent no-op that looks
           * exactly like a working link. */
          typeMask: 3576,
        });
        const p = this._position();
        if (p) {
          const d = dist(p);
          if (d < best || isNaN(best)) best = d;
          if (d <= 5) return { reached: true, remainingM: d, closedM: d0 - d };
        }
        await new Promise((r) => setTimeout(r, 200));
      }
      const e = new Error(
        'E_NO_PROGRESS: never reached the waypoint (closest ' +
          (isNaN(best) ? '?' : best.toFixed(1)) + ' m of ' +
          (isNaN(d0) ? '?' : d0.toFixed(1)) + ' m)'
      );
      e.code = 'E_NO_PROGRESS';
      throw e;
    }

    _position() {
      try {
        const st = this.fc.status();
        return (st && st.position) || null;
      } catch (_) {
        return null;
      }
    }

    /** Circle a point. */
    async orbit(lat, lon, radiusM, altM, turns) {
      await this.goto(lat, lon, altM);
      return this._command(commandLong(CMD.ORBIT, { param1: turns || 1, param3: radiusM }), 34);
    }

    /** Visit a list of points in order. */
    async patrol(points, opts) {
      const o = opts || {};
      let i = 0;
      for (;;) {
        for (; i < points.length; i++) {
          await this.goto(points[i].lat, points[i].lon, points[i].alt !== undefined ? points[i].alt : o.alt, o);
        }
        if (!o.loop) break;
        i = 0;
      }
      return { points: points.length };
    }

    /** Boustrophedon survey: the waypoints come from the Rust generator, so the
     *  same code path is exercised in tests and in flight. */
    async survey(area, altM) {
      const pts = area.points || [];
      let n = 0;
      for (const p of pts) {
        await this.goto(p.lat, p.lon, altM);
        n++;
      }
      return { waypoints: n };
    }

    telemetry() {
      return this.fc.status();
    }

    close() {
      this.fc.close();
    }
  }

  sofuu.drone = {
    /** Connect and return a skill-carrying handle. */
    profile(device) {
      const conn = sofuu.mavlink.connect(device);
      const d = new Drone(conn, device);
      sofuu.drone.current = d;
      return d;
    },

    /** The skill catalogue (single source of truth: rt/drone_ops.rs). */
    catalog() {
      try {
        return JSON.parse(g.__sofuu_drone_catalog());
      } catch (_) {
        return [];
      }
    },

    /** Pre-flight a mission object without sending anything. */
    validate(device, mission) {
      return JSON.parse(g.__sofuu_drone_validate(String(device), JSON.stringify(mission || {})));
    },

    /** Validate the whole mission, then fly it step by step. If any step would
     *  be refused, NOTHING is sent. */
    async mission(drone, mission) {
      const device = drone && drone.device ? drone.device : mission.device;
      const pre = sofuu.drone.validate(device, mission);
      if (!pre || pre.ok !== true) {
        const e = new Error('E_MISSION_REFUSED: pre-flight refused the mission');
        e.code = 'E_MISSION_REFUSED';
        e.preflight = pre;
        throw e;
      }
      const steps = mission.steps || [];
      for (let i = 0; i < steps.length; i++) {
        const s = steps[i];
        const type = String(s.type || '').toLowerCase();
        if (type === 'takeoff') await drone.takeoff(s.alt);
        else if (type === 'goto') await drone.goto(s.lat, s.lon, s.alt, s);
        else if (type === 'orbit') await drone.orbit(s.lat, s.lon, s.radius, s.alt, s.turns);
        else if (type === 'rtl' || type === 'return_to_launch') await drone.rtl();
        else if (type === 'land') await drone.land();
        else if (type === 'hold' || type === 'loiter') await drone.hold();
        else if (type === 'speed' || type === 'set_speed') await drone.setSpeed(s.speed);
        else if (type === 'arm') await (s.on === false ? drone.disarm() : drone.arm());
        else if (type === 'mode') await drone.mode(s.mode !== undefined ? s.mode : s.custom);
        else if (type === 'delay' || type === 'wait') {
          const ms = s.ms !== undefined ? s.ms : (s.s || 0) * 1000;
          await new Promise((r) => setTimeout(r, ms));
        } else throw new Error('E_BAD_MISSION: unknown step type ' + type);
      }
      return { ok: true, steps: steps.length };
    },

    /** The agent toolpack (§4.6 L1): tool definitions that call the skills. */
    tools() {
      const defs = [];
      const add = (name, description, schema, run) => defs.push({ name, description, schema, run });
      add('drone_takeoff', 'Take off to an altitude in metres', { type: 'object', properties: { alt: { type: 'number' } }, required: ['alt'] },
        (a, ctx) => ctx.drone.takeoff(a.alt));
      add('drone_goto', 'Fly to a latitude/longitude at an altitude',
        { type: 'object', properties: { lat: { type: 'number' }, lon: { type: 'number' }, alt: { type: 'number' } }, required: ['lat', 'lon', 'alt'] },
        (a, ctx) => ctx.drone.goto(a.lat, a.lon, a.alt));
      add('drone_rtl', 'Return to the launch point', { type: 'object', properties: {} },
        (a, ctx) => ctx.drone.rtl());
      add('drone_land', 'Land now', { type: 'object', properties: {} },
        (a, ctx) => ctx.drone.land());
      add('drone_hold', 'Hold position', { type: 'object', properties: {} },
        (a, ctx) => ctx.drone.hold());
      add('drone_telemetry', 'Read the current telemetry and GPS state', { type: 'object', properties: {} },
        (a, ctx) => ctx.drone.telemetry());
      return defs;
    },
  };

  /* ── L1: the agent toolpack ────────────────────────────────────────
   * Tools are registered through the public `plugins.defineTool` API (no edits
   * to the shared tools.js), and they act on the handle the operator opened
   * with `sofuu.drone.profile(...)`. An unset handle fails loudly instead of
   * silently connecting to whatever the policy happens to allow first. */
  function activeDrone() {
    const d = sofuu.drone.current;
    if (!d) {
      const e = new Error('E_NO_PROFILE: call sofuu.drone.profile(device) before using drone tools');
      e.code = 'E_NO_PROFILE';
      throw e;
    }
    return d;
  }

  const TOOL_DEFS = [
    ['drone_takeoff', 'Take off to an altitude in metres',
      { type: 'object', properties: { alt: { type: 'number' } }, required: ['alt'] },
      async (a) => activeDrone().takeoff(a.alt)],
    ['drone_goto', 'Fly to a latitude/longitude at an altitude',
      { type: 'object', properties: { lat: { type: 'number' }, lon: { type: 'number' }, alt: { type: 'number' } }, required: ['lat', 'lon', 'alt'] },
      async (a) => activeDrone().goto(a.lat, a.lon, a.alt)],
    ['drone_rtl', 'Return to the launch point', { type: 'object', properties: {} },
      async () => activeDrone().rtl()],
    ['drone_land', 'Land now', { type: 'object', properties: {} },
      async () => activeDrone().land()],
    ['drone_hold', 'Hold position (loiter)', { type: 'object', properties: {} },
      async () => activeDrone().hold()],
    ['drone_telemetry', 'Read the current telemetry and GPS state', { type: 'object', properties: {} },
      async () => activeDrone().telemetry()],
  ];

  /** Register the tools and the pack name. Returns how many tools landed, so a
   *  caller can SEE a partial/no-op registration instead of guessing.
   *
   *  This has to be deferred: `sofuu.tools` / `sofuu.plugins` are other shipped
   *  drivers, and this one is evaluated during engine boot — possibly before
   *  them. A boot-order dependency that silently no-ops is exactly the
   *  "function exists, feature dead" class, so the retry is explicit and the
   *  outcome is reported on the object. */
  sofuu.drone.registerTools = function registerTools() {
    let count = 0;
    try {
      const P = sofuu.tools && sofuu.tools.PACKS;
      if (P && !P.drone) P.drone = TOOL_DEFS.map(function (t) { return t[0]; });
      if (sofuu.plugins && typeof sofuu.plugins.defineTool === 'function') {
        const have = typeof sofuu.plugins.list === 'function' ? sofuu.plugins.list() : [];
        for (let i = 0; i < TOOL_DEFS.length; i++) {
          const t = TOOL_DEFS[i];
          if (have.some(function (x) { return x && x.name === t[0]; })) { count++; continue; }
          sofuu.plugins.defineTool({
            name: t[0], description: t[1], schema: t[2], category: 'drone', run: t[3],
          });
          count++;
        }
      }
    } catch (_) {
      count = 0;
    }
    sofuu.drone.toolpackReady = count === TOOL_DEFS.length;
    sofuu.drone.toolpackCount = count;
    return count;
  };

  sofuu.drone.toolpackReady = false;
  sofuu.drone.toolpackCount = 0;
  if (sofuu.drone.registerTools() === 0) {
    // One deferred retry, after the other drivers have loaded.
    setTimeout(function () { sofuu.drone.registerTools(); }, 0);
  }
})();
