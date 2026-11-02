//! `sofuu.mavlink` — the JS surface's native half (PLAN-DRONES-MAVLINK §4.4).
//!
//! The driver (`src/js/mavlink.js`) is a thin wrapper; everything that can
//! write a byte lives here, and every write goes through
//! `rt/mavlink_link.rs` → `rt/mavlink_gate.rs`.
//!
//! **Capability gating:** `mod_mavlink_register` is called by the engine only
//! when the policy's `hardware.enabled` is true, so an ungated build has no
//! `__sofuu_mavlink_*` globals at all and `mavlink.js` refuses to define the
//! namespace. Two independent gates, the same posture as QTSQ (§4.3: no door
//! beats a locked door).
//!
//! Everything crosses the FFI as JSON strings: the surface returns verdicts and
//! event batches, and JS parses them. No JS-owned pointers, no lifetime rules
//! for a caller to get wrong.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde_json::{json, Value};

use super::mavlink_gate::{self as gate, HardwarePolicy, Verdict};
use super::mavlink_link::{self as link, LinkCore};

struct Slot {
    core: LinkCore,
    io: Box<dyn link::Io>,
    device: String,
}

#[derive(Default)]
struct Registry {
    next_id: u32,
    last_error: String,
    links: HashMap<u32, Slot>,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Registry::default()))
}

/// Monotonic process clock — ONE clock for the gate, the heartbeat and GPS
/// freshness, so the JS side can never skew a safety decision by passing time.
pub fn process_now_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Resolve the policy and decide whether the surface exists at all.
pub fn hardware_enabled() -> bool {
    crate::rt::policy::layered_policy_from_disk()
        .map(|p| HardwarePolicy::from_policy(&p).enabled)
        .unwrap_or(false)
}

/// Every rule decision is appended to `.sofuu/logs/hardware.jsonl` (§4.3).
pub fn append_audit(records: &[Value]) {
    if records.is_empty() {
        return;
    }
    let dir = std::path::Path::new(".sofuu/logs");
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("hardware.jsonl"))
    else {
        return;
    };
    use std::io::Write;
    for r in records {
        let _ = writeln!(f, "{r}");
    }
}

/* ── the operations the driver calls ──────────────────────────────────── */

/// Connect. Returns `{"ok":true,"id":N,"device":…}` or `{"ok":false,"error":…}`.
pub fn connect(device: &str) -> Value {
    let policy = match crate::rt::policy::layered_policy_from_disk() {
        Ok(p) => HardwarePolicy::from_policy(&p),
        Err(e) => return json!({ "ok": false, "error": e }),
    };
    if !policy.enabled {
        return json!({ "ok": false, "error": "hardware.enabled is false; the drone surface does not exist" });
    }
    let core = LinkCore::new(device, policy);
    let verdict = core.check_device();
    if !verdict.allowed {
        return json!({ "ok": false, "error": verdict.reason, "code": verdict.code, "rule": verdict.rule });
    }
    let io = match link::open_io(device) {
        Ok(io) => io,
        Err(e) => return json!({ "ok": false, "error": format!("cannot open {device}: {e}") }),
    };
    let mut r = match registry().lock() {
        Ok(r) => r,
        Err(_) => return json!({ "ok": false, "error": "link registry poisoned" }),
    };
    r.next_id = r.next_id.saturating_add(1).max(1);
    let id = r.next_id;
    r.links.insert(id, Slot { core, io, device: device.to_string() });
    json!({ "ok": true, "id": id, "device": device })
}

fn verdict_json(v: Verdict) -> Value {
    if v.allowed {
        json!({ "ok": true, "rule": v.rule })
    } else {
        let mut out = json!({ "ok": false, "code": v.code, "rule": v.rule, "error": v.reason });
        /* A rate-limit refusal tells the caller how long to wait; the skill
         * layer paces itself with it instead of guessing 130 ms. */
        if let Some(ms) = v.retry_after_ms {
            out["retry_after_ms"] = json!(ms);
        }
        out
    }
}

/// Send one frame described by the §4.4 object shape.
pub fn send(id: u32, obj: &Value) -> Value {
    let mut r = match registry().lock() {
        Ok(r) => r,
        Err(_) => return json!({ "ok": false, "error": "link registry poisoned" }),
    };
    let Some(slot) = r.links.get_mut(&id) else {
        return json!({ "ok": false, "error": format!("no such link {id}") });
    };
    let frame = match link::frame_from_json(&slot.core, obj) {
        Ok(f) => f,
        Err(e) => return json!({ "ok": false, "code": "E_BAD_REQUEST", "error": e }),
    };
    let now = process_now_ms();
    let approved = obj.get("approved").and_then(|v| v.as_bool()).unwrap_or(false);
    match link::send_frame(&mut slot.core, slot.io.as_mut(), &frame, now, approved) {
        Ok(()) => {
            let audit = slot.core.take_audit();
            drop(r);
            append_audit(&audit);
            json!({ "ok": true, "message": frame.name() })
        }
        Err(v) => {
            let audit = slot.core.take_audit();
            drop(r);
            append_audit(&audit);
            verdict_json(v)
        }
    }
}

/// Drain telemetry + emit the runtime's heartbeat. This is the runtime's job —
/// an agent never schedules it (§4.5).
pub fn poll(id: u32, listener_types: &[String]) -> Value {
    let mut r = match registry().lock() {
        Ok(r) => r,
        Err(_) => return json!({ "events": [], "error": "link registry poisoned" }),
    };
    let Some(slot) = r.links.get_mut(&id) else {
        return json!({ "events": [], "error": format!("no such link {id}") });
    };
    let now = process_now_ms();
    let mut out: Vec<Value> = Vec::new();
    match link::pump(&mut slot.core, slot.io.as_mut(), now) {
        Ok(events) => {
            for ev in events {
                out.push(event_json(&ev, listener_types));
            }
        }
        Err(e) => out.push(json!({ "event": "error", "error": e.to_string() })),
    }
    // The dead-man heartbeat.
    if let Some(bytes) = slot.core.tick(now) {
        if let Err(e) = slot.io.write(&bytes) {
            out.push(json!({ "event": "error", "error": format!("heartbeat write: {e}") }));
        }
    }
    let audit = slot.core.take_audit();
    let state = json!({
        "gps_fix_type": slot.core.state.gps_fix_type,
        "gps_hdop": slot.core.state.gps_hdop,
        "home": slot.core.state.home,
    });
    drop(r);
    append_audit(&audit);
    json!({ "events": out, "state": state })
}

fn event_json(ev: &link::Event, types: &[String]) -> Value {
    match ev {
        link::Event::Telemetry { message, data } => {
            json!({ "event": "telemetry", "message": message, "data": data, "subscribed": types.iter().any(|t| t == message) })
        }
        link::Event::CommandAck { command, command_name, result, result_name, text } => json!({
            "event": "ack", "command": command, "command_name": command_name,
            "result": result, "result_name": result_name, "text": text
        }),
        link::Event::ArmingDenied { command, text } => json!({
            "event": "arming_denied", "code": link::E_ARMING_DENIED, "command": command, "text": text
        }),
        link::Event::StatusText { severity, text } => {
            json!({ "event": "statustext", "severity": severity, "text": text })
        }
        link::Event::Resync { discarded_bytes, crc_errors } => {
            json!({ "event": "resync", "discarded_bytes": discarded_bytes, "crc_errors": crc_errors })
        }
    }
}

pub fn status(id: u32) -> Value {
    let r = match registry().lock() {
        Ok(r) => r,
        Err(_) => return json!({ "ok": false, "error": "link registry poisoned" }),
    };
    let Some(slot) = r.links.get(&id) else {
        return json!({ "ok": false, "error": format!("no such link {id}") });
    };
    json!({
        "ok": true,
        "device": slot.device,
        /* "connected" means the endpoint has heard SOMETHING: with the udp:
         * bind-and-learn model that is when a peer address exists at all, so
         * a caller can wait for it instead of failing on the transport. */
        "connected": slot.core.state.last_telemetry_ms.is_some(),
        "mode": format!("{:?}", slot.core.policy.mavlink_mode),
        "gps_fix_type": slot.core.state.gps_fix_type,
        "gps_hdop": slot.core.state.gps_hdop,
        "home": slot.core.state.home,
        /* Where the vehicle is now (home is the first fix and never moves).
         * The JS `goto` needs this to tell "flew to the waypoint" from
         * "frame accepted, nothing happened". */
        "position": slot.core.state.last_lat_lon.map(|(lat, lon)| json!({"lat": lat, "lon": lon})),
        /* Altitude above home. The JS skill layer waits on this the way the CLI's
         * climb monitor does: `takeoff` resolving on the ACK alone let a
         * following `goto` stream its targets while the vehicle was still
         * climbing, where the autopilot ignores them. */
        "relative_alt_m": slot.core.state.last_relative_alt_m,
        "pending": slot.core.pending.keys().copied().collect::<Vec<u16>>(),
        "stats": {
            "discarded_bytes": slot.core.parser.stats().discarded_bytes,
            "crc_errors": slot.core.parser.stats().crc_errors,
            "skipped_messages": slot.core.parser.stats().skipped_messages,
        },
        /* Transport facts: a second sender on this endpoint, or a reply
         * address that moved, explains a resync storm far faster than a
         * packet capture does. */
        "transport": slot.io.transport_stats(),
        "bounds": {
            "maxSpeedMs": slot.core.policy.bounds().max_speed_ms,
            "maxAltRelM": slot.core.policy.bounds().max_alt_rel_m,
            "maxYawRateDs": slot.core.policy.bounds().max_yaw_rate_ds,
            "geofenceRadiusM": slot.core.policy.bounds().geofence_radius_m,
        },
    })
}

pub fn close(id: u32) -> Value {
    let mut r = match registry().lock() {
        Ok(r) => r,
        Err(_) => return json!({ "ok": false, "error": "link registry poisoned" }),
    };
    match r.links.remove(&id) {
        Some(mut slot) => {
            let audit = slot.core.take_audit();
            drop(r);
            append_audit(&audit);
            json!({ "ok": true })
        }
        None => json!({ "ok": false, "error": format!("no such link {id}") }),
    }
}

/// The driver source. Embedded as text and compiled once at boot, but ONLY on
/// a context whose policy enables `hardware` — a build that never uses drones
/// never pays for it. `@sofuu-source-only` keeps it out of the `.bc` snapshot
/// lane (`scripts/check-driver-bc.sh`), same as `memory_api.js`.
pub const DRIVER_JS_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mavlink.js.gz"));

/// The skill/mission layer (§4.6 L2/L3). Loaded right after the MAVLink
/// driver, and inert unless `sofuu.mavlink` exists.
pub const DRONE_JS_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/drone.js.gz"));

/// Inflate at the one point that needs it (a policy-enabled context), the same
/// single-path mechanism the compat/RLM/memory drivers use: the cap is 5.5 MiB
/// and these two sources are ~12 KB of `.rodata` raw.
pub fn driver_js() -> (Option<String>, Option<String>) {
    (
        crate::shipped::inflate_gz(DRIVER_JS_GZ, "mavlink.js"),
        crate::shipped::inflate_gz(DRONE_JS_GZ, "drone.js"),
    )
}

/* ── the FFI globals the driver calls ─────────────────────────────────── */

use std::os::raw::c_int;

use sofuu_ffi::bridge::{js_new_string, js_to_string, register_global_fn};
use sofuu_ffi::qjs::{JSCFunction, JSContext, JSValue, JSValueConst};

fn arg(ctx: *mut JSContext, argv: *const JSValueConst, i: usize, argc: c_int) -> Option<String> {
    if (i as c_int) >= argc || argv.is_null() {
        return None;
    }
    // SAFETY: argv has argc entries per the JSCFunction contract.
    unsafe { js_to_string(ctx, *argv.add(i)) }
}

fn arg_id(ctx: *mut JSContext, argv: *const JSValueConst, argc: c_int) -> Option<u32> {
    if argc < 1 || argv.is_null() {
        return None;
    }
    // The driver passes a JS number, and `js_to_string` deliberately refuses
    // non-strings — so read the number directly and fall back to a string form
    // for callers that stringify ids.
    let v = unsafe { *argv };
    let mut n: u32 = 0;
    // SAFETY: v is this call's first argument; n is a live u32.
    if unsafe { sofuu_ffi::qjs::sofuu_js_to_uint32(ctx, &mut n, v) } == 0 && n > 0 {
        return Some(n);
    }
    arg(ctx, argv, 0, argc)?.trim().parse::<u32>().ok()
}

unsafe extern "C" fn js_mavlink_enabled(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    unsafe { js_new_string(ctx, if hardware_enabled() { "1" } else { "" }) }
}

unsafe extern "C" fn js_mavlink_connect(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let out = match arg(ctx, argv, 0, argc) {
        Some(device) => connect(&device),
        None => json!({ "ok": false, "error": "connect(device) needs a device string" }),
    };
    unsafe { js_new_string(ctx, &out.to_string()) }
}

unsafe extern "C" fn js_mavlink_send(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let out = (|| {
        let id = arg_id(ctx, argv, argc)?;
        let body = arg(ctx, argv, 1, argc)?;
        let v: Value = serde_json::from_str(&body).ok()?;
        Some(send(id, &v))
    })()
    .unwrap_or_else(|| json!({ "ok": false, "error": "send(id, object) needs a link id and a JSON object" }));
    unsafe { js_new_string(ctx, &out.to_string()) }
}

unsafe extern "C" fn js_mavlink_poll(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let out = (|| {
        let id = arg_id(ctx, argv, argc)?;
        // The types list is advisory: it lets the driver skip JSON work for
        // messages nobody subscribed to.
        let types: Vec<String> = arg(ctx, argv, 1, argc)
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();
        Some(poll(id, &types))
    })()
    .unwrap_or_else(|| json!({ "events": [], "error": "poll(id) needs a link id" }));
    unsafe { js_new_string(ctx, &out.to_string()) }
}

unsafe extern "C" fn js_mavlink_status(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let out = arg_id(ctx, argv, argc)
        .map(status)
        .unwrap_or_else(|| json!({ "ok": false, "error": "status(id) needs a link id" }));
    unsafe { js_new_string(ctx, &out.to_string()) }
}

unsafe extern "C" fn js_mavlink_close(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let out = arg_id(ctx, argv, argc)
        .map(close)
        .unwrap_or_else(|| json!({ "ok": false, "error": "close(id) needs a link id" }));
    unsafe { js_new_string(ctx, &out.to_string()) }
}

/// Pre-flight a mission object against the policy (PLAN-DRONES §4.6 L3).
pub fn validate_mission(device: &str, doc: &Value) -> Value {
    let policy = match crate::rt::policy::layered_policy_from_disk() {
        Ok(p) => HardwarePolicy::from_policy(&p),
        Err(e) => return json!({ "ok": false, "error": e }),
    };
    let mission = match crate::rt::drone_ops::parse_mission(doc) {
        Ok(m) => m,
        Err(e) => return json!({ "ok": false, "error": e }),
    };
    crate::rt::drone_ops::preflight(&policy, device, &mission).to_json()
}

unsafe extern "C" fn js_drone_validate(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let out = (|| {
        let device = arg(ctx, argv, 0, argc)?;
        let body = arg(ctx, argv, 1, argc)?;
        let doc: Value = serde_json::from_str(&body).ok()?;
        Some(validate_mission(&device, &doc))
    })()
    .unwrap_or_else(|| json!({ "ok": false, "error": "validate(device, missionObject) needs both arguments" }));
    unsafe { js_new_string(ctx, &out.to_string()) }
}

unsafe extern "C" fn js_drone_catalog(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let out = crate::rt::drone_ops::skill_catalog();
    unsafe { js_new_string(ctx, &out.to_string()) }
}

/// Install the native globals. Called ONLY when `hardware.enabled` is true, so
/// a build without the section has no drone surface whatsoever.
///
/// # Safety
/// `ctx` must be a live QuickJS context on the engine thread.
pub unsafe fn mod_mavlink_register(ctx: *mut JSContext) {
    register_global_fn(ctx, "__sofuu_mavlink_enabled", js_mavlink_enabled as JSCFunction);
    register_global_fn(ctx, "__sofuu_mavlink_connect", js_mavlink_connect as JSCFunction);
    register_global_fn(ctx, "__sofuu_mavlink_send", js_mavlink_send as JSCFunction);
    register_global_fn(ctx, "__sofuu_mavlink_poll", js_mavlink_poll as JSCFunction);
    register_global_fn(ctx, "__sofuu_mavlink_status", js_mavlink_status as JSCFunction);
    register_global_fn(ctx, "__sofuu_mavlink_close", js_mavlink_close as JSCFunction);
    register_global_fn(ctx, "__sofuu_drone_validate", js_drone_validate as JSCFunction);
    register_global_fn(ctx, "__sofuu_drone_catalog", js_drone_catalog as JSCFunction);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy_json(mode: &str) -> Value {
        json!({
            "hardware": {
                "enabled": true,
                "devices": { "mode": "allowlist",
                             "allow": [{ "path": "udp:127.0.0.1:0", "class": "mavlink-udp" }] },
                "mavlink": { "mode": mode, "send": ["HEARTBEAT", "COMMAND_*"] }
            }
        })
    }

    #[test]
    fn frame_from_json_builds_a_takeoff_with_the_alt_alias() {
        let core = LinkCore::new("udp:127.0.0.1:0", HardwarePolicy::from_policy(&policy_json("allowlist")));
        let v = json!({ "type": "COMMAND_LONG", "command": "MAV_CMD_NAV_TAKEOFF", "alt": 12.5 });
        let f = link::frame_from_json(&core, &v).expect("builds");
        assert_eq!(f.msgid, 76);
        assert_eq!(f.u16_at(28), 20 + 2); // NAV_TAKEOFF
        assert_eq!(f.f32_at(24), 12.5, "alt maps to param7");
    }

    #[test]
    fn frame_from_json_accepts_a_numeric_command() {
        let core = LinkCore::new("udp:127.0.0.1:0", HardwarePolicy::from_policy(&policy_json("allowlist")));
        let f = link::frame_from_json(&core, &json!({"type":"COMMAND_LONG","command":20})).unwrap();
        assert_eq!(f.u16_at(28), 20);
    }

    #[test]
    fn frame_from_json_rejects_an_unknown_type_with_a_useful_message() {
        let core = LinkCore::new("udp:127.0.0.1:0", HardwarePolicy::from_policy(&policy_json("allowlist")));
        let e = link::frame_from_json(&core, &json!({"type":"PARAM_SET"})).unwrap_err();
        assert!(e.contains("unknown message type"), "{e}");
        let e = link::frame_from_json(&core, &json!({"type":"HEARTBEAT","junk":1})).unwrap();
        assert_eq!(e.msgid, 0, "extra keys are ignored, not fatal");
    }

    #[test]
    fn frame_from_json_rejects_a_message_with_no_builder() {
        let core = LinkCore::new("udp:127.0.0.1:0", HardwarePolicy::from_policy(&policy_json("allowlist")));
        let e = link::frame_from_json(&core, &json!({"type":"ATTITUDE"})).unwrap_err();
        assert!(e.contains("no JSON builder yet"), "{e}");
    }

    #[test]
    fn process_clock_is_monotonic() {
        let a = process_now_ms();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = process_now_ms();
        assert!(b >= a, "{b} >= {a}");
    }
}
