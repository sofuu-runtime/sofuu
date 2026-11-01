//! The minimal-code layer — skills, profiles, missions (PLAN-DRONES §4.6).
//!
//! Four layers of height over the raw `sofuu.mavlink` API, and **one rule that
//! governs all of them**: every layer expands to the same gated frames. A
//! prebuilt skill is a *pre-vetted policy shape*, not a bypass.
//!
//! ```text
//! L1  the agent composes skills           (PACKS.drone)
//! L2  sofuu.drone.profile(...) + skills   (takeoff/goto/patrol/orbit/…)
//! L3  a mission object                    (static, shareable, pre-flight checkable)
//! L4  a composer UI emitting L3           (later phase)
//! ```
//!
//! **Why the bodies are Rust here, not Kaze.** §4.6 authors skill bodies in
//! Kaze; the Kaze VM is a concurrent lane whose API is still moving, and
//! coupling this lane to it would make both unbuildable at the same time. The
//! *frame path is identical either way* — a skill body here produces exactly
//! the frames a Kaze body would, and both cross `rt/mavlink_gate.rs`. Swapping
//! a body to Kaze later is a local change, not an architectural one.
//!
//! The property that earns this layer its keep is **pre-flight validation**:
//! a mission is a static object, so the whole thing can be judged against the
//! policy's bounds, geofence and send-whitelist *before the first frame
//! exists* — while still being re-checked per frame at runtime (defence in
//! depth: skill → mission check → frame veto).

use serde_json::{json, Value};

use super::mavlink::{self, Frame};
use super::mavlink_gate::{self as gate, GateState, HardwarePolicy, SendRequest, Verdict};
use super::mavlink_link::LinkCore;

/// The frames a mission item expands to, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Takeoff { alt_m: f32 },
    Goto { lat: f64, lon: f64, alt_m: f32, speed_ms: Option<f32> },
    Orbit { lat: f64, lon: f64, radius_m: f32, alt_m: f32, turns: u16 },
    Survey { lat: f64, lon: f64, width_m: f32, length_m: f32, alt_m: f32, spacing_m: f32 },
    Rtl,
    Land,
    Hold,
    SetSpeed { speed_ms: f32 },
    Delay { ms: u64 },
    Arm { arm: bool },
    /// Switch the autopilot's flight mode (`DO_SET_MODE`). A mission that
    /// cannot select GUIDED cannot take off: ArduCopter starts disarmed in
    /// STABILIZE and `NAV_TAKEOFF` is refused there.
    Mode { custom: u8, name: &'static str },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mission {
    pub vehicle: String,
    pub steps: Vec<Step>,
    pub safety: MissionSafety,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MissionSafety {
    pub max_alt_m: Option<f32>,
    pub geofence_radius_m: Option<f32>,
    pub rtl_battery_pct: Option<u8>,
}

/// A pre-flight verdict: every step judged, plus the reason a mission may not
/// fly. `ok` is false when ANY step would be refused.
#[derive(Clone, Debug, PartialEq)]
pub struct Preflight {
    pub ok: bool,
    pub steps: Vec<StepVerdict>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepVerdict {
    pub index: usize,
    pub kind: String,
    pub allowed: bool,
    pub code: &'static str,
    pub rule: &'static str,
    pub reason: String,
}

/// Parse the §4.6 mission object:
///
/// ```json
/// { "vehicle": "copter-sim",
///   "steps": [ {"type":"takeoff","alt":20},
///              {"type":"survey","lat":..,"lon":..,"width":..,"length":..,"alt":25},
///              {"type":"rtl"} ],
///   "safety": { "maxAlt": 40, "geofenceRadius": 300, "rtlBattery": 20 } }
/// ```
pub fn parse_mission(v: &Value) -> Result<Mission, String> {
    let vehicle = v.get("vehicle").and_then(|x| x.as_str()).unwrap_or("default").to_string();
    let steps_raw = v.get("steps").and_then(|x| x.as_array()).ok_or("mission needs a `steps` array")?;
    let mut steps = Vec::with_capacity(steps_raw.len());
    for (i, s) in steps_raw.iter().enumerate() {
        steps.push(parse_step(s).map_err(|e| format!("step {i}: {e}"))?);
    }
    if steps.is_empty() {
        return Err("mission has no steps".into());
    }
    let safety = v.get("safety").cloned().unwrap_or(Value::Null);
    let f = |k: &str| safety.get(k).and_then(|x| x.as_f64());
    let sf = MissionSafety {
        max_alt_m: f("maxAlt").map(|x| x as f32),
        geofence_radius_m: f("geofenceRadius").map(|x| x as f32),
        rtl_battery_pct: safety.get("rtlBattery").and_then(|x| x.as_u64()).map(|x| x as u8),
    };
    Ok(Mission { vehicle, steps, safety: sf })
}

/// Horizontal distance between two (lat, lon) points, in metres (equirectangular
/// over the small distances a mission step covers). Used to verify that a
/// waypoint actually had an effect, not merely that a frame was accepted.
pub fn ground_distance_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dlat = (b.0 - a.0) * 111_320.0;
    let dlon = (b.1 - a.1) * 111_320.0 * (a.0 * std::f64::consts::PI / 180.0).cos();
    (dlat * dlat + dlon * dlon).sqrt()
}

fn num(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

/// ArduCopter custom-mode numbers. Only the vetted subset the mission DSL is
/// expected to select; `"custom": <n>` is the escape hatch for anything else.
pub fn copter_mode(name: &str) -> Option<u8> {
    Some(match name.trim().to_ascii_uppercase().as_str() {
        "STABILIZE" => 0,
        "ACRO" => 1,
        "ALT_HOLD" => 2,
        "AUTO" => 3,
        "GUIDED" => 4,
        "LOITER" => 5,
        "RTL" => 6,
        "LAND" => 9,
        "BRAKE" => 17,
        "GUIDED_NOGPS" => 20,
        _ => return None,
    })
}

pub fn copter_mode_name(custom: u8) -> &'static str {
    match custom {
        0 => "STABILIZE",
        1 => "ACRO",
        2 => "ALT_HOLD",
        3 => "AUTO",
        4 => "GUIDED",
        5 => "LOITER",
        6 => "RTL",
        9 => "LAND",
        17 => "BRAKE",
        20 => "GUIDED_NOGPS",
        _ => "CUSTOM",
    }
}

fn req(v: &Value, k: &str) -> Result<f64, String> {
    num(v, k).ok_or_else(|| format!("needs `{k}`"))
}

fn parse_step(v: &Value) -> Result<Step, String> {
    let ty = v.get("type").and_then(|x| x.as_str()).ok_or("missing `type`")?;
    Ok(match ty.to_ascii_lowercase().as_str() {
        "takeoff" => Step::Takeoff { alt_m: req(v, "alt")? as f32 },
        "goto" | "goto_point" => Step::Goto {
            lat: req(v, "lat")?,
            lon: req(v, "lon")?,
            alt_m: req(v, "alt")? as f32,
            speed_ms: num(v, "speed").map(|x| x as f32),
        },
        "orbit" => Step::Orbit {
            lat: req(v, "lat")?,
            lon: req(v, "lon")?,
            radius_m: req(v, "radius")? as f32,
            alt_m: req(v, "alt")? as f32,
            turns: num(v, "turns").unwrap_or(1.0) as u16,
        },
        "survey" | "survey_grid" => Step::Survey {
            lat: req(v, "lat")?,
            lon: req(v, "lon")?,
            width_m: req(v, "width")? as f32,
            length_m: req(v, "length")? as f32,
            alt_m: req(v, "alt")? as f32,
            spacing_m: num(v, "spacing").unwrap_or(20.0) as f32,
        },
        "rtl" | "return_to_launch" => Step::Rtl,
        "land" => Step::Land,
        "hold" | "loiter" => Step::Hold,
        "speed" | "set_speed" => Step::SetSpeed { speed_ms: req(v, "speed")? as f32 },
        "delay" | "wait" => Step::Delay { ms: (num(v, "ms").or_else(|| num(v, "s").map(|s| s * 1000.0)).ok_or("needs `ms` or `s`")?) as u64 },
        "arm" => Step::Arm { arm: v.get("on").and_then(|x| x.as_bool()).unwrap_or(true) },
        "mode" => {
            let name = v.get("mode").and_then(|x| x.as_str()).unwrap_or("");
            let custom = match num(v, "custom") {
                Some(n) => n as u8,
                None => copter_mode(name).ok_or_else(|| {
                    format!("unknown flight mode '{name}' — use a known name or `\"custom\": <n>`")
                })?,
            };
            Step::Mode { custom, name: copter_mode_name(custom) }
        }
        other => return Err(format!("unknown step type '{other}'")),
    })
}

impl Step {
    pub fn kind(&self) -> &'static str {
        match self {
            Step::Takeoff { .. } => "takeoff",
            Step::Goto { .. } => "goto",
            Step::Orbit { .. } => "orbit",
            Step::Survey { .. } => "survey",
            Step::Rtl => "rtl",
            Step::Land => "land",
            Step::Hold => "hold",
            Step::SetSpeed { .. } => "speed",
            Step::Delay { .. } => "delay",
            Step::Arm { .. } => "arm",
            Step::Mode { .. } => "mode",
        }
    }

    /// Altitude this step would command, when it commands one.
    pub fn altitude(&self) -> Option<f32> {
        match self {
            Step::Takeoff { alt_m } | Step::Goto { alt_m, .. } | Step::Orbit { alt_m, .. } | Step::Survey { alt_m, .. } => Some(*alt_m),
            _ => None,
        }
    }
}

/* ── L2: the skills — each expands to gated frames ────────────────────── */

/// MAV_CMD for a skill, in the order the vehicle needs them.
pub fn skill_frames(core: &LinkCore, step: &Step) -> Vec<Frame> {
    let mut out = Vec::new();
    match step {
        Step::Arm { arm } => {
            let mut p = [0.0f32; 7];
            p[0] = if *arm { 1.0 } else { 0.0 };
            out.push(mavlink::msg_command_long(
                core.seq,
                core.sysid,
                core.compid,
                1,
                1,
                mavlink::cmd::COMPONENT_ARM_DISARM,
                p,
                0,
            ));
        }
        Step::Takeoff { alt_m } => {
            let mut p = [0.0f32; 7];
            p[6] = *alt_m;
            out.push(mavlink::msg_command_long(
                core.seq,
                core.sysid,
                core.compid,
                1,
                1,
                mavlink::cmd::NAV_TAKEOFF,
                p,
                0,
            ));
        }
        Step::Goto { lat, lon, alt_m, speed_ms } => {
            if let Some(s) = speed_ms {
                let mut p = [0.0f32; 7];
                p[1] = *s;
                out.push(mavlink::msg_command_long(
                    core.seq,
                    core.sysid,
                    core.compid,
                    1,
                    1,
                    mavlink::cmd::DO_CHANGE_SPEED,
                    p,
                    0,
                ));
            }
            out.push(position_frame(core, *lat, *lon, *alt_m));
        }
        Step::Orbit { lat, lon, radius_m, alt_m, turns } => {
            /* A circular waypoint generator: the FC's DO_ORBIT is not in every
             * firmware's guided mode, so v1 emits the orbit centre plus the
             * radius as a waypoint radius command. Each frame is gated. */
            out.push(position_frame(core, *lat, *lon, *alt_m));
            let mut p = [0.0f32; 7];
            p[0] = *turns as f32;
            p[2] = *radius_m;
            out.push(mavlink::msg_command_long(
                core.seq,
                core.sysid,
                core.compid,
                1,
                1,
                mavlink::cmd::DO_ORBIT,
                p,
                0,
            ));
        }
        Step::Survey { lat, lon, width_m, length_m, alt_m, spacing_m } => {
            for (la, lo) in survey_grid(*lat, *lon, *width_m, *length_m, *spacing_m) {
                out.push(position_frame(core, la, lo, *alt_m));
            }
        }
        Step::Rtl => {
            out.push(mavlink::msg_command_long(core.seq, core.sysid, core.compid, 1, 1, mavlink::cmd::NAV_RETURN_TO_LAUNCH, [0.0; 7], 0));
        }
        Step::Land => {
            out.push(mavlink::msg_command_long(core.seq, core.sysid, core.compid, 1, 1, mavlink::cmd::NAV_LAND, [0.0; 7], 0));
        }
        Step::Hold => {
            out.push(mavlink::msg_command_long(core.seq, core.sysid, core.compid, 1, 1, mavlink::cmd::NAV_LOITER_UNLIM, [0.0; 7], 0));
        }
        Step::SetSpeed { speed_ms } => {
            let mut p = [0.0f32; 7];
            p[1] = *speed_ms;
            out.push(mavlink::msg_command_long(core.seq, core.sysid, core.compid, 1, 1, mavlink::cmd::DO_CHANGE_SPEED, p, 0));
        }
        Step::Mode { custom, .. } => {
            /* MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (1) in param1 tells the FC that
               param2 is the firmware-specific custom mode number, which is the
               only portable way to ask ArduCopter/PX4 for GUIDED. */
            let mut p = [0.0f32; 7];
            p[0] = 1.0;
            p[1] = *custom as f32;
            out.push(mavlink::msg_command_long(core.seq, core.sysid, core.compid, 1, 1, mavlink::cmd::DO_SET_MODE, p, 0));
        }
        Step::Delay { .. } => {}
    }
    out
}

pub fn position_frame(core: &LinkCore, lat: f64, lon: f64, alt_m: f32) -> Frame {
    mavlink::msg_set_position_target_global_int(
        core.seq,
        core.sysid,
        core.compid,
        1,
        1,
        (lat * 1e7) as i32,
        (lon * 1e7) as i32,
        alt_m,
        [0.0; 3],
        0.0,
        /* Position only. The per-axis mask bits are easy to get wrong and the
           failure is silent: a wrong mask commands zero acceleration and the
           vehicle holds station while ignoring the waypoint. */
        mavlink::type_mask::POS_ONLY,
        mavlink::frame_id::GLOBAL_RELATIVE_ALT_INT,
    )
}

/// Boustrophedon ("mow the lawn") waypoint generator: `width × length` metres
/// starting at (lat, lon), lanes `spacing` metres apart.
pub fn survey_grid(lat: f64, lon: f64, width_m: f32, length_m: f32, spacing_m: f32) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let spacing = spacing_m.max(1.0);
    let lanes = ((width_m / spacing).ceil() as usize).max(1);
    let lat_per_m = 1.0 / 111_320.0;
    let lon_per_m = 1.0 / (111_320.0 * lat.to_radians().cos().abs().max(1e-6));
    for i in 0..lanes {
        let d_lat = i as f64 * spacing as f64 * lat_per_m;
        let la = lat + d_lat;
        if i % 2 == 0 {
            out.push((la, lon));
            out.push((la, lon + length_m as f64 * lon_per_m));
        } else {
            out.push((la, lon + length_m as f64 * lon_per_m));
            out.push((la, lon));
        }
    }
    out
}

/* ── L3: pre-flight validation ────────────────────────────────────────── */

/// Judge EVERY frame of EVERY step against the policy before anything flies.
///
/// Each step is judged with a fresh gate state, so the answer is "would this
/// frame be allowed on a healthy link" rather than a rate-limit artefact. The
/// runtime still re-checks every frame as it is sent — this is the layer above
/// the veto, never a substitute for it.
pub fn preflight(policy: &HardwarePolicy, device: &str, mission: &Mission) -> Preflight {
    let probe = LinkCore::new(device, policy.clone());
    let mut steps = Vec::new();
    let mut ok = true;
    for (index, step) in mission.steps.iter().enumerate() {
        let mut verdict = Verdict::allow("mission/none");
        /* Mission-level safety first: a step above the mission's own ceiling is
         * refused even if the policy would allow it. The stricter of the two
         * wins, which is why this runs before the frame check. */
        if let Some(alt) = step.altitude() {
            if let Some(max) = mission.safety.max_alt_m {
                if alt > max {
                    verdict = Verdict::deny(
                        gate::E_SAFETY_VIOLATION,
                        "R-HW/mission.maxAlt",
                        format!("{} commands {alt:.0} m, above the mission's {max:.0} m ceiling", step.kind()),
                    );
                }
            }
        }
        if verdict.allowed {
            for frame in skill_frames(&probe, step) {
                let mut state = GateState::default();
                // A pre-flight answer assumes a healthy link: 3D fix, good HDOP,
                // and a home at the mission's first coordinate. Whether that is
                // TRUE is the runtime's call, per frame, at send time.
                state.gps_fix_type = mavlink::GPS_FIX_3D;
                state.gps_hdop = Some(0.8);
                state.last_gps_ms = Some(0);
                state.home = mission.steps.iter().find_map(|s| match s {
                    Step::Goto { lat, lon, .. } | Step::Orbit { lat, lon, .. } | Step::Survey { lat, lon, .. } => Some((*lat, *lon)),
                    _ => None,
                });
                let mut p = policy.clone();
                if let Some(r) = mission.safety.geofence_radius_m {
                    p.bounds.geofence_radius_m = p.bounds.geofence_radius_m.min(r);
                }
                if let Some(a) = mission.safety.max_alt_m {
                    p.bounds.max_alt_rel_m = p.bounds.max_alt_rel_m.min(a);
                }
                let v = gate::gate_send(
                    &p,
                    &mut state,
                    &SendRequest { device, frame: &frame },
                    0,
                    false,
                );
                if !v.allowed {
                    verdict = v;
                    break;
                }
            }
        }
        if !verdict.allowed {
            ok = false;
        }
        steps.push(StepVerdict {
            index,
            kind: step.kind().to_string(),
            allowed: verdict.allowed,
            code: verdict.code,
            rule: verdict.rule,
            reason: verdict.reason,
        });
    }
    Preflight { ok, steps }
}

impl Preflight {
    pub fn to_json(&self) -> Value {
        json!({
            "ok": self.ok,
            "steps": self.steps.iter().map(|s| json!({
                "index": s.index, "kind": s.kind, "allowed": s.allowed,
                "code": s.code, "rule": s.rule, "reason": s.reason,
            })).collect::<Vec<_>>(),
        })
    }
}

/// The agent-facing toolpack shape (§4.6 L1): each skill's name, description
/// and the MAV_CMD it expands to, so `PACKS.drone` can be generated from one
/// source of truth instead of a hand-maintained list.
pub fn skill_catalog() -> Value {
    json!([
        {"name": "arm", "description": "Arm or disarm the vehicle", "expandsTo": "MAV_CMD_COMPONENT_ARM_DISARM"},
        {"name": "takeoff", "description": "Take off to an altitude in metres", "expandsTo": "MAV_CMD_NAV_TAKEOFF"},
        {"name": "goto", "description": "Fly to a latitude/longitude at an altitude", "expandsTo": "SET_POSITION_TARGET_GLOBAL_INT"},
        {"name": "orbit", "description": "Circle a point at a radius", "expandsTo": "MAV_CMD_DO_ORBIT"},
        {"name": "survey", "description": "Boustrophedon survey grid over an area", "expandsTo": "SET_POSITION_TARGET_GLOBAL_INT"},
        {"name": "returnToLaunch", "description": "Return to the launch point", "expandsTo": "MAV_CMD_NAV_RETURN_TO_LAUNCH"},
        {"name": "land", "description": "Land now", "expandsTo": "MAV_CMD_NAV_LAND"},
        {"name": "hold", "description": "Hold position (loiter)", "expandsTo": "MAV_CMD_NAV_LOITER_UNLIM"},
        {"name": "setSpeed", "description": "Change the commanded speed in m/s", "expandsTo": "MAV_CMD_DO_CHANGE_SPEED"},
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DEV: &str = "udp:127.0.0.1:14550";

    fn policy() -> HardwarePolicy {
        HardwarePolicy::from_policy(&json!({
            "hardware": {
                "enabled": true,
                "devices": { "mode": "allowlist", "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                "mavlink": { "mode": "allowlist",
                             "send": ["HEARTBEAT", "COMMAND_*", "SET_POSITION_TARGET_*"] },
                "bounds": { "maxSpeedMs": 10, "maxAltRelM": 50, "maxYawRateDs": 90, "geofenceRadiusM": 300 }
            }
        }))
    }

    #[test]
    fn a_mission_parses_into_steps() {
        let m = parse_mission(&json!({
            "vehicle": "copter-sim",
            "steps": [{"type":"takeoff","alt":20},
                      {"type":"goto","lat":52.1,"lon":4.1,"alt":25},
                      {"type":"rtl"}],
            "safety": {"maxAlt": 40, "geofenceRadius": 300, "rtlBattery": 20}
        }))
        .expect("parses");
        assert_eq!(m.vehicle, "copter-sim");
        assert_eq!(m.steps.len(), 3);
        assert_eq!(m.steps[0], Step::Takeoff { alt_m: 20.0 });
        assert_eq!(m.safety.max_alt_m, Some(40.0));
        assert_eq!(m.safety.rtl_battery_pct, Some(20));
    }

    #[test]
    fn an_unknown_step_is_refused_with_its_index() {
        let e = parse_mission(&json!({"steps":[{"type":"takeoff","alt":10},{"type":"teleport"}]})).unwrap_err();
        assert!(e.contains("step 1"), "{e}");
        assert!(e.contains("teleport"), "{e}");
    }

    #[test]
    fn a_mission_without_steps_is_refused() {
        assert!(parse_mission(&json!({"steps":[]})).is_err());
        assert!(parse_mission(&json!({})).is_err());
    }

    #[test]
    fn a_step_missing_a_required_number_is_refused() {
        let e = parse_mission(&json!({"steps":[{"type":"takeoff"}]})).unwrap_err();
        assert!(e.contains("alt"), "{e}");
    }

    #[test]
    fn takeoff_expands_to_one_gated_command_long() {
        let core = LinkCore::new(DEV, policy());
        let frames = skill_frames(&core, &Step::Takeoff { alt_m: 20.0 });
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].msgid, 76);
        assert_eq!(frames[0].u16_at(28), mavlink::cmd::NAV_TAKEOFF);
        assert_eq!(frames[0].f32_at(24), 20.0);
    }

    #[test]
    fn ground_distance_is_metres_over_a_short_leg() {
        /* 0.0007 deg of latitude ≈ 78 m — the leg the SITL goto demo uses. */
        let d = ground_distance_m((-35.3632621, 149.1652375), (-35.3625622, 149.1652375));
        assert!((d - 77.9).abs() < 1.0, "{d}");
        assert_eq!(ground_distance_m((-35.0, 149.0), (-35.0, 149.0)), 0.0);
    }

    #[test]
    fn a_mode_step_requests_the_custom_mode_number() {
        /* ArduCopter boots disarmed in STABILIZE and refuses NAV_TAKEOFF
         * outside GUIDED, so a mission needs a way to name a mode. */
        let m = parse_mission(&json!({"steps":[{"type":"mode","mode":"guided"}]})).unwrap();
        assert!(matches!(m.steps[0], Step::Mode { custom: 4, name: "GUIDED" }), "{:?}", m.steps[0]);
        let core = LinkCore::new(DEV, policy());
        let frames = skill_frames(&core, &m.steps[0]);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].u16_at(28), mavlink::cmd::DO_SET_MODE);
        assert_eq!(frames[0].f32_at(0), 1.0, "param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED");
        assert_eq!(frames[0].f32_at(4), 4.0, "param2 = custom mode number");

        // An unknown name is refused rather than guessed at.
        let e = parse_mission(&json!({"steps":[{"type":"mode","mode":"HYPERDRIVE"}]})).unwrap_err();
        assert!(e.contains("HYPERDRIVE"), "{e}");
        // The numeric escape hatch works for firmware-specific modes.
        let m = parse_mission(&json!({"steps":[{"type":"mode","custom":21}]})).unwrap();
        assert!(matches!(m.steps[0], Step::Mode { custom: 21, .. }));
    }

    #[test]
    fn goto_with_a_speed_emits_speed_then_position() {
        let core = LinkCore::new(DEV, policy());
        let frames = skill_frames(&core, &Step::Goto { lat: 52.1, lon: 4.1, alt_m: 25.0, speed_ms: Some(5.0) });
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].u16_at(28), mavlink::cmd::DO_CHANGE_SPEED);
        assert_eq!(frames[1].msgid, 86);
        assert_eq!(frames[1].f32_at(12), 25.0);
    }

    #[test]
    fn the_survey_grid_mows_the_lawn() {
        // 100 m wide, 200 m long, 25 m lanes → 4 lanes, each 2 points.
        let grid = survey_grid(52.0, 4.0, 100.0, 200.0, 25.0);
        assert_eq!(grid.len(), 8);
        // Alternate lanes run in opposite directions (boustrophedon).
        let lon_first_lane_end = grid[1].1;
        let lon_second_lane_start = grid[2].1;
        assert!((lon_first_lane_end - lon_second_lane_start).abs() < 1e-9, "lanes must connect");
        // Latitude advances by one lane spacing (25 m ≈ 2.246e-4 degrees).
        let dlat = grid[2].0 - grid[0].0;
        assert!((dlat - 25.0 / 111_320.0).abs() < 1e-9, "lane spacing, got {dlat}");
    }

    #[test]
    fn preflight_allows_a_mission_inside_every_limit() {
        let m = parse_mission(&json!({
            "steps": [{"type":"takeoff","alt":20},{"type":"rtl"}],
            "safety": {"maxAlt": 40}
        }))
        .unwrap();
        let pf = preflight(&policy(), DEV, &m);
        assert!(pf.ok, "{:?}", pf.steps);
        assert!(pf.steps.iter().all(|s| s.allowed));
    }

    #[test]
    fn preflight_refuses_a_step_above_the_policy_ceiling() {
        let m = parse_mission(&json!({"steps":[{"type":"takeoff","alt":999}]})).unwrap();
        let pf = preflight(&policy(), DEV, &m);
        assert!(!pf.ok);
        assert_eq!(pf.steps[0].rule, "R-HW/bounds/maxAltRelM");
    }

    #[test]
    fn preflight_refuses_a_step_above_the_mission_ceiling_even_when_the_policy_allows_it() {
        // The policy ceiling is 50 m; the mission says 30 m; the step wants 40.
        let m = parse_mission(&json!({"steps":[{"type":"takeoff","alt":40}],"safety":{"maxAlt":30}})).unwrap();
        let pf = preflight(&policy(), DEV, &m);
        assert!(!pf.ok);
        assert_eq!(pf.steps[0].rule, "R-HW/mission.maxAlt", "the stricter ceiling wins");
    }

    #[test]
    fn preflight_refuses_a_target_beyond_the_geofence() {
        let m = parse_mission(&json!({
            "steps": [{"type":"goto","lat":52.1,"lon":4.1,"alt":20},
                      {"type":"goto","lat":53.0,"lon":4.1,"alt":20}]
        }))
        .unwrap();
        let pf = preflight(&policy(), DEV, &m);
        assert!(!pf.ok, "the second point is ~100 km away");
        assert_eq!(pf.steps[1].rule, "R-HW/bounds/geofenceRadiusM");
    }

    #[test]
    fn preflight_refuses_a_command_outside_the_send_whitelist() {
        let mut p = policy();
        p.send = vec!["HEARTBEAT".to_string()];
        let m = parse_mission(&json!({"steps":[{"type":"rtl"}]})).unwrap();
        let pf = preflight(&p, DEV, &m);
        assert!(!pf.ok);
        assert_eq!(pf.steps[0].rule, "R-HW/mavlink.send");
    }

    #[test]
    fn preflight_refuses_an_unauthorised_device() {
        let m = parse_mission(&json!({"steps":[{"type":"rtl"}]})).unwrap();
        let pf = preflight(&policy(), "udp:10.0.0.9:14550", &m);
        assert!(!pf.ok);
        assert_eq!(pf.steps[0].rule, "R-HW/devices.allow");
    }

    #[test]
    fn preflight_reports_every_step_not_just_the_first_failure() {
        let m = parse_mission(&json!({
            "steps": [{"type":"takeoff","alt":999},{"type":"rtl"},{"type":"takeoff","alt":999}]
        }))
        .unwrap();
        let pf = preflight(&policy(), DEV, &m);
        assert_eq!(pf.steps.len(), 3);
        assert!(!pf.steps[0].allowed);
        assert!(pf.steps[1].allowed, "rtl is fine");
        assert!(!pf.steps[2].allowed);
    }

    #[test]
    fn the_skill_catalog_lists_every_skill() {
        let c = skill_catalog();
        let names: Vec<&str> = c.as_array().unwrap().iter().filter_map(|s| s["name"].as_str()).collect();
        for want in ["arm", "takeoff", "goto", "orbit", "survey", "returnToLaunch", "land", "hold", "setSpeed"] {
            assert!(names.contains(&want), "missing {want} in {names:?}");
        }
    }
}
