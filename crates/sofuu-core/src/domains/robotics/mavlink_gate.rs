//! The frame veto chain — PLAN-DRONES-MAVLINK §4.3, wire half.
//!
//! One choke point between a decision and a wire:
//!
//! ```text
//! classify → deny-list → send-whitelist → command whitelist → bounds
//!          → rate limits → approval
//! ```
//!
//! Ordering is the correctness argument: the deny-list is consulted before any
//! allow rule, so a broad allow can never re-enable something explicitly
//! denied. Every rule is a small auditable constant or a policy line — never a
//! model output, never a network call.
//!
//! Two rules carry the doctrine:
//!
//! * **Reject, never clamp.** An out-of-bounds command returns a typed error
//!   naming the rule. The frame is never written and never silently rescaled —
//!   what the agent asked for and what flew must not diverge.
//! * **Fail closed.** An unparseable rate limit denies; an unknown MAV_CMD is
//!   refused; absent bounds fall back to conservative defaults; a device path
//!   that cannot be canonicalised is refused.
//!
//! ## Why this is not `rt/hardware.rs`
//!
//! `rt/hardware.rs` is the **capability** firewall (PLAN-KAZE §3.3): it judges
//! a Kaze `needs:` family and the physical range of an argument. This module
//! is the **frame** firewall: it judges a MAVLink frame about to be written,
//! after the capability call has already been authorised. Both plans name
//! `rt/hardware.rs`, so the split is deliberate — a scripted capability and a
//! wire frame are different objects with different rules, and defence in depth
//! means both gates run (capability → mission pre-flight → frame veto).
//!
//! ```
//! use sofuu_core::rt::mavlink;
//! use sofuu_core::rt::mavlink_gate::{HardwarePolicy, GateState, SendRequest, gate_send};
//!
//! let policy = HardwarePolicy::from_policy(&serde_json::json!({
//!     "hardware": {
//!         "enabled": true,
//!         "devices": { "mode": "allowlist",
//!                      "allow": [{ "path": "udp:127.0.0.1:14550", "class": "mavlink-udp" }] },
//!         "mavlink": { "mode": "observe" }
//!     }
//! }));
//! let mut state = GateState::default();
//! let hb = mavlink::msg_heartbeat(1, 255, 191);
//! let v = gate_send(&policy, &mut state,
//!                   &SendRequest { device: "udp:127.0.0.1:14550", frame: &hb },
//!                   0, false);
//! assert!(v.allowed, "heartbeat is allowed in observe mode");
//! ```

use std::collections::HashMap;

use serde_json::Value;

use super::mavlink::{self, Class, Frame};

/// The §1.4 code every refused physical action carries.
pub const E_SAFETY_VIOLATION: &str = "E_SAFETY_VIOLATION";
/// Fail-closed on telemetry loss / GPS dropout — a refusal the agent can act on.
pub const E_NO_GPS_FIX: &str = "E_NO_GPS_FIX";
/// The hardware section is absent or switched off — no door at all.
pub const E_HARDWARE_DISABLED: &str = "E_HARDWARE_DISABLED";
/// An opt-in approval is pending; the caller asks the operator and re-calls.
pub const E_APPROVAL_REQUIRED: &str = "E_APPROVAL_REQUIRED";

/// Telemetry older than this vetoes coordinate-dependent commands (§4.3).
pub const TELEMETRY_MAX_AGE_MS: u64 = 1_500;
/// HDOP floor for accepting a position as usable.
pub const GPS_MAX_HDOP: f32 = 2.0;
/* Conservative bounds used when the policy omits them. A missing bound must
 * never mean "unbounded": the defaults are the shipped profile's limits. */
pub const DEFAULT_MAX_SPEED_MS: f32 = 15.0;
pub const DEFAULT_MAX_ALT_REL_M: f32 = 120.0;
pub const DEFAULT_MAX_YAW_RATE_DS: f32 = 180.0;
pub const DEFAULT_GEOFENCE_RADIUS_M: f32 = 500.0;

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// A gate decision. `allowed == false` always carries a code and the rule that
/// produced it (the R7.3 house style) so `sofuu policy why` can explain it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub allowed: bool,
    pub code: &'static str,
    pub rule: &'static str,
    pub reason: String,
    /// For a rate-limit refusal: how long until the same frame would be
    /// allowed. A mission executor paces itself with this instead of guessing
    /// (and instead of aborting the mission — a rate limit is a timing fact,
    /// not a safety violation, and the only correct response is to wait).
    pub retry_after_ms: Option<u64>,
}

impl Verdict {
    pub fn allow(rule: &'static str) -> Verdict {
        Verdict { allowed: true, code: "", rule, reason: String::new(), retry_after_ms: None }
    }

    pub fn deny(code: &'static str, rule: &'static str, reason: impl Into<String>) -> Verdict {
        Verdict {
            allowed: false,
            code,
            rule,
            reason: reason.into(),
            retry_after_ms: None,
        }
    }

    /// A refusal the caller can retry once the interval elapses.
    pub fn pace(rule: &'static str, reason: impl Into<String>, wait_ms: u64) -> Verdict {
        Verdict {
            allowed: false,
            code: E_SAFETY_VIOLATION,
            rule,
            reason: reason.into(),
            retry_after_ms: Some(wait_ms),
        }
    }

    /// The audit record shape for the observability WAL. Every decision is
    /// logged — allow and deny alike — so an after-action review is a
    /// `sofuu logs show` away.
    pub fn audit_record(&self, device: &str, message: &str, now_ms: u64) -> Value {
        serde_json::json!({
            "kind": "hardware.gate",
            "at_ms": now_ms,
            "device": device,
            "message": message,
            "decision": if self.allowed { "allow" } else { "deny" },
            "code": self.code,
            "rule": self.rule,
            "reason": self.reason,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DevicesMode {
    /// No device may be opened.
    Off,
    /// Only exact, canonicalised `devices.allow` entries.
    Allowlist,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MavlinkMode {
    /// Nothing at all.
    Off,
    /// Receive telemetry; the only frame Sofuu may write is HEARTBEAT.
    Observe,
    /// `send` decides. Deny still wins.
    Allowlist,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Approval {
    Off,
    OncePerRun,
    Every,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRule {
    pub path: String,
    pub class: String,
    pub baud: Option<u32>,
}

/// A minimum interval between two allowed sends of a message class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Period {
    /// At least this many milliseconds between sends.
    Every(u64),
    /// Unparseable rule — fail closed, deny every matching send.
    Never,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub max_speed_ms: f32,
    pub max_alt_rel_m: f32,
    pub max_yaw_rate_ds: f32,
    pub geofence_radius_m: f32,
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            max_speed_ms: DEFAULT_MAX_SPEED_MS,
            max_alt_rel_m: DEFAULT_MAX_ALT_REL_M,
            max_yaw_rate_ds: DEFAULT_MAX_YAW_RATE_DS,
            geofence_radius_m: DEFAULT_GEOFENCE_RADIUS_M,
        }
    }
}

/// The `hardware` policy section, resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct HardwarePolicy {
    pub enabled: bool,
    pub devices_mode: DevicesMode,
    pub devices: Vec<DeviceRule>,
    pub mavlink_mode: MavlinkMode,
    pub send: Vec<String>,
    pub deny: Vec<String>,
    /// (message-name glob, period). Every matching pattern applies and the
    /// strictest (largest interval) wins, so object key ordering can never
    /// loosen a limit.
    pub rate_limits: Vec<(String, Period)>,
    pub bounds: Bounds,
    pub approval: Approval,
}

/// MAV_CMD values that may ever leave the process. An id outside this list is
/// refused without consulting the policy — the whitelist is design-time and
/// auditable, and the agent cannot widen it.
pub const ALLOWED_COMMANDS: &[u16] = &[
    mavlink::cmd::NAV_WAYPOINT,
    mavlink::cmd::NAV_LOITER_UNLIM,
    mavlink::cmd::NAV_LOITER_TURNS,
    mavlink::cmd::NAV_LOITER_TIME,
    mavlink::cmd::NAV_LOITER_TO_ALT,
    mavlink::cmd::NAV_SPLINE_WAYPOINT,
    mavlink::cmd::NAV_RETURN_TO_LAUNCH,
    mavlink::cmd::NAV_LAND,
    mavlink::cmd::NAV_TAKEOFF,
    mavlink::cmd::NAV_GUIDED_ENABLE,
    mavlink::cmd::NAV_DELAY,
    mavlink::cmd::DO_ORBIT,
    mavlink::cmd::DO_SET_MODE,
    mavlink::cmd::DO_CHANGE_SPEED,
    mavlink::cmd::DO_SET_HOME,
    mavlink::cmd::DO_PAUSE_CONTINUE,
    mavlink::cmd::MISSION_START,
    mavlink::cmd::COMPONENT_ARM_DISARM,
    // Telemetry configuration: it changes what the link receives, never what
    // the vehicle does, so it is allowlisted like a request, not an actuator.
    mavlink::cmd::SET_MESSAGE_INTERVAL,
];

/// Commands no profile may allow. `PARAM_SET` keeps the agent away from the
/// flight controller's tuning; `DO_MOTOR_TEST` would spin motors. Both are
/// checked before the allowlist and before policy.
pub const ALWAYS_DENIED_COMMANDS: &[(u16, &str)] =
    &[(mavlink::cmd::PARAM_SET, "PARAM_SET"), (mavlink::cmd::DO_MOTOR_TEST, "MAV_CMD_DO_MOTOR_TEST")];

/// Name lookup for the commands the gate reasons about. Policy `deny` entries
/// may use either spelling (`PARAM_SET` or `MAV_CMD_PARAM_SET`).
pub fn command_name(cmd: u16) -> Option<&'static str> {
    Some(match cmd {
        mavlink::cmd::NAV_WAYPOINT => "MAV_CMD_NAV_WAYPOINT",
        mavlink::cmd::NAV_LOITER_UNLIM => "MAV_CMD_NAV_LOITER_UNLIM",
        mavlink::cmd::NAV_LOITER_TURNS => "MAV_CMD_NAV_LOITER_TURNS",
        mavlink::cmd::NAV_LOITER_TIME => "MAV_CMD_NAV_LOITER_TIME",
        mavlink::cmd::NAV_LOITER_TO_ALT => "MAV_CMD_NAV_LOITER_TO_ALT",
        mavlink::cmd::NAV_RETURN_TO_LAUNCH => "MAV_CMD_NAV_RETURN_TO_LAUNCH",
        mavlink::cmd::NAV_LAND => "MAV_CMD_NAV_LAND",
        mavlink::cmd::NAV_TAKEOFF => "MAV_CMD_NAV_TAKEOFF",
        mavlink::cmd::NAV_SPLINE_WAYPOINT => "MAV_CMD_NAV_SPLINE_WAYPOINT",
        mavlink::cmd::NAV_GUIDED_ENABLE => "MAV_CMD_NAV_GUIDED_ENABLE",
        mavlink::cmd::NAV_DELAY => "MAV_CMD_NAV_DELAY",
        mavlink::cmd::DO_ORBIT => "MAV_CMD_DO_ORBIT",
        mavlink::cmd::DO_SET_MODE => "MAV_CMD_DO_SET_MODE",
        mavlink::cmd::DO_CHANGE_SPEED => "MAV_CMD_DO_CHANGE_SPEED",
        mavlink::cmd::DO_SET_HOME => "MAV_CMD_DO_SET_HOME",
        mavlink::cmd::DO_PAUSE_CONTINUE => "MAV_CMD_DO_PAUSE_CONTINUE",
        mavlink::cmd::DO_MOTOR_TEST => "MAV_CMD_DO_MOTOR_TEST",
        mavlink::cmd::MISSION_START => "MAV_CMD_MISSION_START",
        mavlink::cmd::PARAM_SET => "PARAM_SET",
        mavlink::cmd::COMPONENT_ARM_DISARM => "MAV_CMD_COMPONENT_ARM_DISARM",
        _ => return None,
    })
}

/* ── policy resolution ────────────────────────────────────────────────── */

fn get<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cur = v;
    for k in path {
        cur = cur.get(*k)?;
    }
    Some(cur)
}

fn str_at<'a>(v: &'a Value, path: &[&str]) -> Option<&'a str> {
    get(v, path).and_then(|x| x.as_str())
}

fn bool_at(v: &Value, path: &[&str]) -> Option<bool> {
    get(v, path).and_then(|x| x.as_bool())
}

fn f32_at(v: &Value, path: &[&str]) -> Option<f32> {
    get(v, path).and_then(|x| x.as_f64()).map(|f| f as f32)
}

fn list_at(v: &Value, path: &[&str]) -> Vec<String> {
    get(v, path)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|e| e.as_str()).map(str::to_string).collect())
        .unwrap_or_default()
}

/// `"10Hz"` / `"1Hz"` / `"0.5Hz"` / `"100ms"` / `"2s"` → a minimum interval.
/// `None` means "unparseable" and the caller fails closed.
pub fn parse_rate(raw: &str) -> Option<Period> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    if let Some(num) = s.strip_suffix("hz") {
        let hz: f64 = num.trim().parse().ok()?;
        if !(hz.is_finite() && hz > 0.0) {
            return None;
        }
        return Some(Period::Every((1000.0 / hz).ceil() as u64));
    }
    if let Some(num) = s.strip_suffix("ms") {
        let ms: f64 = num.trim().parse().ok()?;
        if !(ms.is_finite() && ms > 0.0) {
            return None;
        }
        return Some(Period::Every(ms.ceil() as u64));
    }
    if let Some(num) = s.strip_suffix('s') {
        let secs: f64 = num.trim().parse().ok()?;
        if !(secs.is_finite() && secs > 0.0) {
            return None;
        }
        return Some(Period::Every((secs * 1000.0).ceil() as u64));
    }
    None
}

impl HardwarePolicy {
    /// Resolve the `hardware` section of a policy document. An absent or
    /// non-object section yields a disabled gate (no door), never an enabled
    /// one — the capability-gated posture of §4.3.
    pub fn from_policy(policy: &Value) -> HardwarePolicy {
        let Some(hw) = get(policy, &["hardware"]).filter(|v| v.is_object()) else {
            return HardwarePolicy::disabled();
        };

        let devices_mode =
            match str_at(hw, &["devices", "mode"]).map(|s| s.trim().to_ascii_lowercase()) {
                Some(ref m) if m == "allowlist" || m == "allow-list" || m == "allow" => {
                    DevicesMode::Allowlist
                }
                _ => DevicesMode::Off,
            };
        let mut devices = Vec::new();
        if let Some(list) = get(hw, &["devices", "allow"]).and_then(|v| v.as_array()) {
            for entry in list {
                let Some(path) = entry.get("path").and_then(|v| v.as_str()) else { continue };
                devices.push(DeviceRule {
                    path: path.to_string(),
                    class: entry
                        .get("class")
                        .and_then(|v| v.as_str())
                        .unwrap_or("uart")
                        .to_string(),
                    baud: entry.get("baud").and_then(|v| v.as_u64()).map(|b| b as u32),
                });
            }
        }

        let mavlink_mode = match str_at(hw, &["mavlink", "mode"]).map(|s| s.trim().to_ascii_lowercase())
        {
            Some(ref m) if m == "allowlist" || m == "allow-list" || m == "allow" => {
                MavlinkMode::Allowlist
            }
            Some(ref m) if m == "observe" => MavlinkMode::Observe,
            _ => MavlinkMode::Off,
        };

        let mut rate_limits = Vec::new();
        if let Some(map) = get(hw, &["mavlink", "rateLimits"]).and_then(|v| v.as_object()) {
            for (glob, raw) in map {
                let period = raw.as_str().and_then(parse_rate).unwrap_or(Period::Never); // fail closed on a typo
                rate_limits.push((glob.clone(), period));
            }
        }

        let d = Bounds::default();
        let bounds = Bounds {
            max_speed_ms: f32_at(hw, &["bounds", "maxSpeedMs"]).unwrap_or(d.max_speed_ms),
            max_alt_rel_m: f32_at(hw, &["bounds", "maxAltRelM"]).unwrap_or(d.max_alt_rel_m),
            max_yaw_rate_ds: f32_at(hw, &["bounds", "maxYawRateDs"]).unwrap_or(d.max_yaw_rate_ds),
            geofence_radius_m: f32_at(hw, &["bounds", "geofenceRadiusM"])
                .unwrap_or(d.geofence_radius_m),
        };

        let approval = match str_at(hw, &["approval"]).map(|s| s.trim().to_ascii_lowercase()) {
            Some(ref m) if m == "every" => Approval::Every,
            Some(ref m) if m == "once-per-run" || m == "onceperrun" => Approval::OncePerRun,
            _ => Approval::Off,
        };

        HardwarePolicy {
            enabled: bool_at(hw, &["enabled"]).unwrap_or(false),
            devices_mode,
            devices,
            mavlink_mode,
            send: list_at(hw, &["mavlink", "send"]),
            deny: list_at(hw, &["mavlink", "deny"]),
            rate_limits,
            bounds,
            approval,
        }
    }

    fn disabled() -> HardwarePolicy {
        HardwarePolicy {
            enabled: false,
            devices_mode: DevicesMode::Off,
            devices: Vec::new(),
            mavlink_mode: MavlinkMode::Off,
            send: Vec::new(),
            deny: Vec::new(),
            rate_limits: Vec::new(),
            bounds: Bounds::default(),
            approval: Approval::Off,
        }
    }

    /// Is `device` an allowed endpoint? Exact, canonicalised match — never a
    /// suffix or prefix match (a device allowlist that matched loosely would
    /// let `/dev/ttyAMA0` be satisfied by `/dev/ttyAMA0-evil`).
    pub fn gate_device(&self, device: &str) -> Verdict {
        if !self.enabled {
            return Verdict::deny(
                E_HARDWARE_DISABLED,
                "R-HW/enabled",
                "the hardware section is absent or disabled",
            );
        }
        match self.devices_mode {
            DevicesMode::Off => Verdict::deny(
                E_SAFETY_VIOLATION,
                "R-HW/devices.mode",
                "devices.mode is off; no device may be opened",
            ),
            DevicesMode::Allowlist => {
                let Some(canonical) = canonical_device(device) else {
                    return Verdict::deny(
                        E_SAFETY_VIOLATION,
                        "R-HW/devices.allow",
                        format!("'{device}' is not a resolvable device endpoint"),
                    );
                };
                let hit = self
                    .devices
                    .iter()
                    .any(|rule| canonical_device(&rule.path).as_deref() == Some(canonical.as_str()));
                if hit {
                    Verdict::allow("R-HW/devices.allow")
                } else {
                    Verdict::deny(
                        E_SAFETY_VIOLATION,
                        "R-HW/devices.allow",
                        format!("'{canonical}' is not in devices.allow"),
                    )
                }
            }
        }
    }

    /// The resolved bounds (for `sofuu policy why` / the audit trail).
    pub fn bounds(&self) -> Bounds {
        self.bounds
    }
}

/// Canonicalise a device endpoint for allowlist matching.
///
/// * `udp:host:port` — an endpoint we BIND and receive on (SITL's `--out`).
/// * `udpout:host:port` — an endpoint we SEND to, binding an ephemeral local
///   port (a companion that must speak first).
/// * `uart:path` / filesystem paths — `realpath`, so a symlink resolves to its
///   target; a path that cannot be resolved (does not exist) is refused. The
///   fs-jail TOCTOU lesson applied to devices.
pub fn canonical_device(raw: &str) -> Option<String> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    /* Match the scheme case-insensitively (a policy written `UDP:` is the same
     * endpoint), but slice the payload from the ORIGINAL string: a filesystem
     * path is case-sensitive and must never be lowercased. */
    let lower = s.to_ascii_lowercase();
    for (scheme, off) in [("udp:", 4usize), ("udpout:", 7usize)] {
        if lower.starts_with(scheme) {
            let rest = &s[off..];
            let (host, port) = rest.rsplit_once(':')?;
            let host = host.trim().trim_start_matches('[').trim_end_matches(']');
            let port: u16 = port.trim().parse().ok()?;
            if host.is_empty() {
                return None;
            }
            let name = scheme.trim_end_matches(':');
            return Some(format!("{name}:{}:{}", host.to_ascii_lowercase(), port));
        }
    }
    if lower.starts_with("uart:") {
        let inner = &s[5..];
        return Some(format!("uart:{}", std::fs::canonicalize(inner).ok()?.display()));
    }
    Some(std::fs::canonicalize(s).ok()?.display().to_string())
}

/* ── gate state ───────────────────────────────────────────────────────── */

/// Mutable gate state: rate-limit clocks, the latest GPS picture, and the
/// once-per-run approval cache. All time comes from the caller, so every rule
/// is testable without sleeping.
#[derive(Debug, Default)]
pub struct GateState {
    last_sent_ms: HashMap<&'static str, u64>,
    pub gps_fix_type: u8,
    pub gps_hdop: Option<f32>,
    pub last_gps_ms: Option<u64>,
    pub last_telemetry_ms: Option<u64>,
    /// First valid fix — the geofence centre (§4.3).
    pub home: Option<(f64, f64)>,
    /// Last reported altitude above home, in metres. Completion-aware actions
    /// ("takeoff to 15 m" waits until it is actually there) read this.
    pub last_relative_alt_m: Option<f32>,
    /// Last reported position. `home` is the first fix and never moves; this is
    /// where the vehicle is now, which is what a "did the waypoint take effect"
    /// check needs.
    pub last_lat_lon: Option<(f64, f64)>,
    approved_once: HashMap<&'static str, bool>,
}

impl GateState {
    /// Feed the receive path. Never gated (read-only), but it is what makes the
    /// fail-closed rules possible.
    pub fn observe(&mut self, frame: &Frame, now_ms: u64) {
        self.last_telemetry_ms = Some(now_ms);
        match frame.msgid {
            // GPS_RAW_INT: eph = HDOP in centimetres (offset 20), fix_type at 28.
            24 => {
                self.gps_fix_type = frame.u8_at(28);
                self.gps_hdop = Some(frame.u16_at(20) as f32 / 100.0);
                self.last_gps_ms = Some(now_ms);
            }
            // GLOBAL_POSITION_INT: lat/lon in 1e-7 degrees. A zero pair is the
            // "no fix" placeholder ArduPilot emits, not a real position.
            33 => {
                let lat = frame.i32_at(4);
                let lon = frame.i32_at(8);
                if lat != 0 || lon != 0 {
                    self.last_gps_ms = Some(now_ms);
                    self.last_lat_lon = Some((lat as f64 / 1e7, lon as f64 / 1e7));
                    if self.home.is_none() {
                        self.home = Some((lat as f64 / 1e7, lon as f64 / 1e7));
                    }
                }
                /* relative_alt is mm above home — the number a climb monitor
                 * needs, and the one "takeoff to 15 m" is judged against. */
                self.last_relative_alt_m = Some(frame.i32_at(16) as f32 / 1000.0);
            }
            _ => {}
        }
    }

    fn gps_usable(&self, now_ms: u64) -> Result<(), Verdict> {
        match self.last_gps_ms {
            Some(t) if now_ms.saturating_sub(t) <= TELEMETRY_MAX_AGE_MS => {}
            _ => {
                return Err(Verdict::deny(
                    E_NO_GPS_FIX,
                    "R-HW/failClosed",
                    "no GPS telemetry within the last 1.5 s",
                ))
            }
        }
        if self.gps_fix_type < mavlink::GPS_FIX_3D {
            return Err(Verdict::deny(
                E_NO_GPS_FIX,
                "R-HW/failClosed",
                format!("GPS fix type {} is below 3D", self.gps_fix_type),
            ));
        }
        if let Some(hdop) = self.gps_hdop {
            if hdop > GPS_MAX_HDOP {
                return Err(Verdict::deny(
                    E_NO_GPS_FIX,
                    "R-HW/failClosed",
                    format!("GPS HDOP {hdop:.2} exceeds {GPS_MAX_HDOP:.2}"),
                ));
            }
        }
        Ok(())
    }
}

/// One send being judged.
pub struct SendRequest<'a> {
    pub device: &'a str,
    pub frame: &'a Frame,
}

/* ── the pipeline ─────────────────────────────────────────────────────── */

/// Judge one outgoing frame. `approved` is the caller's answer to a previous
/// `E_APPROVAL_REQUIRED`; it is ignored unless `hardware.approval` is opt-in.
///
/// This is the ONLY write path — there is no raw frame write anywhere in the
/// runtime, which is the view-guard lesson (`rt/view_guard.rs`) applied to a
/// wire instead of memory.
pub fn gate_send(
    policy: &HardwarePolicy,
    state: &mut GateState,
    req: &SendRequest<'_>,
    now_ms: u64,
    approved: bool,
) -> Verdict {
    if !policy.enabled {
        return Verdict::deny(
            E_HARDWARE_DISABLED,
            "R-HW/enabled",
            "the hardware section is absent or disabled",
        );
    }
    let device = policy.gate_device(req.device);
    if !device.allowed {
        return device;
    }
    let frame = req.frame;
    let Some(spec) = frame.spec() else {
        return Verdict::deny(
            E_SAFETY_VIOLATION,
            "R-HW/classify",
            format!("message id {} is outside the supported subset", frame.msgid),
        );
    };

    let command = command_of(frame);

    /* Builder-denied commands first: "no policy can re-enable this" is a
     * stronger and more useful statement than any mode rule that also happens
     * to refuse the frame, so it is what `policy why` should name. It denies in
     * every mode either way — this only changes the explanation, never an
     * outcome. */
    if let Some(cmd) = command {
        if let Some((_, name)) = ALWAYS_DENIED_COMMANDS.iter().find(|(id, _)| *id == cmd) {
            return Verdict::deny(
                E_SAFETY_VIOLATION,
                "R-HW/deny/always",
                format!("{name} ({cmd}) is denied by every profile"),
            );
        }
    }

    /* Mode gate: in observe mode the only writable frame is HEARTBEAT, so a
     * mis-specified send list cannot become a command path. */
    let mode_rule = match policy.mavlink_mode {
        MavlinkMode::Off => {
            return Verdict::deny(
                E_SAFETY_VIOLATION,
                "R-HW/mavlink.mode",
                "mavlink.mode is off; nothing may be sent",
            )
        }
        MavlinkMode::Observe => {
            if spec.class == Class::Actuator {
                return Verdict::deny(
                    E_SAFETY_VIOLATION,
                    "R-HW/mavlink.mode",
                    format!("{} is actuator-bearing and observe mode is receive-only", spec.name),
                );
            }
            "R-HW/mavlink.mode/observe"
        }
        MavlinkMode::Allowlist => "R-HW/mavlink.send",
    };

    /* Deny-list: always wins, before any allow rule. */
    for pattern in &policy.deny {
        if pattern_matches(pattern, spec.name, command) {
            return Verdict::deny(
                E_SAFETY_VIOLATION,
                "R-HW/deny",
                format!("{} matches a hardware deny rule: {pattern}", spec.name),
            );
        }
    }

    /* Send whitelist, allowlist mode only. */
    if policy.mavlink_mode == MavlinkMode::Allowlist {
        let ok = policy.send.iter().any(|pattern| pattern_matches(pattern, spec.name, command));
        if !ok {
            return Verdict::deny(
                E_SAFETY_VIOLATION,
                "R-HW/mavlink.send",
                format!("{} is not in mavlink.send", spec.name),
            );
        }
    }

    /* MAV_CMD-level whitelist: a command id nobody vetted never leaves. */
    if let Some(cmd) = command {
        if !ALLOWED_COMMANDS.contains(&cmd) {
            let name = command_name(cmd).unwrap_or("unknown");
            return Verdict::deny(
                E_SAFETY_VIOLATION,
                "R-HW/command",
                format!("MAV_CMD {cmd} ({name}) is not in the command whitelist"),
            );
        }
    }

    /* Payload bounds — reject, never clamp. */
    if let Err(v) = check_bounds(policy, state, frame, now_ms) {
        return v;
    }

    /* Rate limits: the strictest matching period wins. */
    let mut min_interval: Option<u64> = None;
    for (pattern, period) in &policy.rate_limits {
        if !pattern_matches(pattern, spec.name, command) {
            continue;
        }
        match period {
            Period::Never => {
                return Verdict::deny(
                    E_SAFETY_VIOLATION,
                    "R-HW/rateLimits",
                    format!("{} matches an unparseable rate limit (fail closed)", spec.name),
                )
            }
            Period::Every(ms) => min_interval = Some(min_interval.map_or(*ms, |cur| cur.max(*ms))),
        }
    }
    if let Some(every) = min_interval {
        if let Some(last) = state.last_sent_ms.get(spec.name) {
            let elapsed = now_ms.saturating_sub(*last);
            if elapsed < every {
                return Verdict::pace(
                    "R-HW/rateLimits",
                    format!(
                        "{} rate-limited: {elapsed} ms since last send, {every} ms required",
                        spec.name
                    ),
                    every - elapsed,
                );
            }
        }
    }

    /* Approval is opt-in by policy (§8: the deterministic gate is the only
     * Sofuu-side layer unless the operator turns this on). */
    match policy.approval {
        Approval::Off => {}
        Approval::Every => {
            if !approved {
                return Verdict::deny(
                    E_APPROVAL_REQUIRED,
                    "R-HW/approval",
                    format!("{} requires per-call approval", spec.name),
                );
            }
        }
        Approval::OncePerRun => {
            if !approved && !state.approved_once.get(spec.name).copied().unwrap_or(false) {
                return Verdict::deny(
                    E_APPROVAL_REQUIRED,
                    "R-HW/approval",
                    format!("{} requires approval once per run", spec.name),
                );
            }
            state.approved_once.insert(spec.name, true);
        }
    }

    /* Allowed: record the clock so the next call is judged against this one. */
    state.last_sent_ms.insert(spec.name, now_ms);
    Verdict::allow(mode_rule)
}

/// The MAV_CMD a COMMAND_LONG (76) / COMMAND_INT (75) frame carries.
pub fn command_of(frame: &Frame) -> Option<u16> {
    match frame.msgid {
        75 | 76 => Some(frame.u16_at(28)),
        _ => None,
    }
}

/// Glob match against the message name, or against the command name when the
/// frame carries one. `MAV_CMD_` is optional in a policy spelling
/// (`PARAM_SET` and `MAV_CMD_PARAM_SET` both match).
fn pattern_matches(pattern: &str, message: &str, command: Option<u16>) -> bool {
    let pat = pattern.trim();
    if crate::rt::policy::glob_matches(pat, message) {
        return true;
    }
    let Some(cmd) = command else { return false };
    match command_name(cmd) {
        Some(name) => {
            crate::rt::policy::glob_matches(pat, name)
                || name
                    .strip_prefix("MAV_CMD_")
                    .map(|bare| crate::rt::policy::glob_matches(pat, bare))
                    .unwrap_or(false)
                || pat
                    .strip_prefix("MAV_CMD_")
                    .map(|bare| crate::rt::policy::glob_matches(bare, name))
                    .unwrap_or(false)
        }
        // An id with no name can still be named numerically.
        None => pat.trim_start_matches("MAV_CMD_").parse::<u16>().map(|want| want == cmd).unwrap_or(false),
    }
}

/// Coordinate- and rate-bearing checks. Each arm either passes or returns the
/// reason it refused — nothing is rescaled.
fn check_bounds(
    policy: &HardwarePolicy,
    state: &GateState,
    frame: &Frame,
    now_ms: u64,
) -> Result<(), Verdict> {
    let b = policy.bounds;
    match frame.msgid {
        // SET_POSITION_TARGET_GLOBAL_INT (86) / LOCAL_NED (84): same offsets.
        84 | 86 => {
            let mask = frame.u16_at(48);
            let global = frame.msgid == 86;

            if mask & mavlink::type_mask::IGNORE_VELOCITY == 0 {
                let (vx, vy, vz) = (frame.f32_at(16), frame.f32_at(20), frame.f32_at(24));
                let speed = (vx * vx + vy * vy + vz * vz).sqrt();
                if !speed.is_finite() {
                    return Err(Verdict::deny(
                        E_SAFETY_VIOLATION,
                        "R-HW/bounds/maxSpeedMs",
                        "velocity is not a finite number",
                    ));
                }
                if speed > b.max_speed_ms {
                    return Err(Verdict::deny(
                        E_SAFETY_VIOLATION,
                        "R-HW/bounds/maxSpeedMs",
                        format!("commanded speed {speed:.1} m/s exceeds {:.1} m/s", b.max_speed_ms),
                    ));
                }
            }
            if mask & mavlink::type_mask::IGNORE_YAW_RATE == 0 {
                // MAVLink carries yaw rate in rad/s; the policy speaks deg/s.
                let rate_ds = frame.f32_at(44) * (180.0 / std::f32::consts::PI);
                if !rate_ds.is_finite() {
                    return Err(Verdict::deny(
                        E_SAFETY_VIOLATION,
                        "R-HW/bounds/maxYawRateDs",
                        "yaw rate is not a finite number",
                    ));
                }
                if rate_ds.abs() > b.max_yaw_rate_ds {
                    return Err(Verdict::deny(
                        E_SAFETY_VIOLATION,
                        "R-HW/bounds/maxYawRateDs",
                        format!(
                            "commanded yaw rate {rate_ds:.0} deg/s exceeds {:.0} deg/s",
                            b.max_yaw_rate_ds
                        ),
                    ));
                }
            }
            if mask & mavlink::type_mask::IGNORE_POSITION == 0 && global {
                // alt is metres in the *_INT altitude frames.
                let fr = frame.u8_at(52);
                let alt_is_m = matches!(
                    fr,
                    mavlink::frame_id::GLOBAL_RELATIVE_ALT_INT | mavlink::frame_id::GLOBAL_RELATIVE_ALT
                );
                if alt_is_m {
                    let alt = frame.f32_at(12);
                    if !alt.is_finite() || alt > b.max_alt_rel_m {
                        return Err(Verdict::deny(
                            E_SAFETY_VIOLATION,
                            "R-HW/bounds/maxAltRelM",
                            format!(
                                "commanded altitude {alt:.0} m exceeds {:.0} m",
                                b.max_alt_rel_m
                            ),
                        ));
                    }
                }
                let lat = frame.i32_at(4) as f64 / 1e7;
                let lon = frame.i32_at(8) as f64 / 1e7;
                state.gps_usable(now_ms)?;
                check_fence(b, state, lat, lon)?;
            }
            // A LOCAL_NED target is relative to the vehicle, so the fence is
            // enforced on global targets; the vehicle's own containment is the
            // FC's geofence, not ours.
            Ok(())
        }
        // COMMAND_LONG (76) / COMMAND_INT (75).
        75 | 76 => {
            match frame.u16_at(28) {
                mavlink::cmd::NAV_TAKEOFF => {
                    // param7 = altitude for NAV_TAKEOFF.
                    let alt = frame.f32_at(24);
                    if !alt.is_finite() || alt > b.max_alt_rel_m {
                        return Err(Verdict::deny(
                            E_SAFETY_VIOLATION,
                            "R-HW/bounds/maxAltRelM",
                            format!(
                                "takeoff altitude {alt:.0} m exceeds {:.0} m",
                                b.max_alt_rel_m
                            ),
                        ));
                    }
                }
                mavlink::cmd::DO_CHANGE_SPEED => {
                    // param2 = speed (m/s).
                    let speed = frame.f32_at(4);
                    if !speed.is_finite() || speed > b.max_speed_ms {
                        return Err(Verdict::deny(
                            E_SAFETY_VIOLATION,
                            "R-HW/bounds/maxSpeedMs",
                            format!(
                                "commanded speed {speed:.1} m/s exceeds {:.1} m/s",
                                b.max_speed_ms
                            ),
                        ));
                    }
                }
                mavlink::cmd::NAV_WAYPOINT
                | mavlink::cmd::NAV_SPLINE_WAYPOINT
                | mavlink::cmd::NAV_LOITER_UNLIM
                | mavlink::cmd::NAV_LOITER_TURNS
                | mavlink::cmd::NAV_LOITER_TIME
                | mavlink::cmd::NAV_LOITER_TO_ALT => {
                    // COMMAND_INT carries lat/lon in x/y as 1e-7 degrees.
                    if frame.msgid == 75 {
                        let frame_kind = frame.u8_at(32);
                        let lat = frame.i32_at(16) as f64 / 1e7;
                        let lon = frame.i32_at(20) as f64 / 1e7;
                        if lat != 0.0 || lon != 0.0 {
                            state.gps_usable(now_ms)?;
                            check_fence(b, state, lat, lon)?;
                        } else if frame_kind == mavlink::frame_id::GLOBAL_INT
                            || frame_kind == mavlink::frame_id::GLOBAL_RELATIVE_ALT_INT
                        {
                            // A global target of 0,0 is a sentinel, not a place.
                            return Err(Verdict::deny(
                                E_SAFETY_VIOLATION,
                                "R-HW/bounds/geofenceRadiusM",
                                "global waypoint has a zero position",
                            ));
                        }
                    }
                }
                _ => {}
            }
            Ok(())
        }
        // MISSION_ITEM_INT (73): uploading a mission writes waypoints.
        73 => {
            let cmd = frame.u16_at(30);
            // Only navigation items carry coordinates; mission plumbing items
            // (delays, conditions) do not.
            if matches!(
                cmd,
                mavlink::cmd::NAV_WAYPOINT
                    | mavlink::cmd::NAV_SPLINE_WAYPOINT
                    | mavlink::cmd::NAV_LOITER_UNLIM
                    | mavlink::cmd::NAV_LOITER_TURNS
                    | mavlink::cmd::NAV_LOITER_TIME
                    | mavlink::cmd::NAV_LOITER_TO_ALT
                    | mavlink::cmd::NAV_LAND
                    | mavlink::cmd::NAV_TAKEOFF
            ) {
                let lat = frame.i32_at(16) as f64 / 1e7;
                let lon = frame.i32_at(20) as f64 / 1e7;
                state.gps_usable(now_ms)?;
                check_fence(b, state, lat, lon)?;
                let alt = frame.f32_at(24);
                if alt.is_finite() && alt > b.max_alt_rel_m {
                    return Err(Verdict::deny(
                        E_SAFETY_VIOLATION,
                        "R-HW/bounds/maxAltRelM",
                        format!(
                            "mission item altitude {alt:.0} m exceeds {:.0} m",
                            b.max_alt_rel_m
                        ),
                    ));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check_fence(b: Bounds, state: &GateState, lat: f64, lon: f64) -> Result<(), Verdict> {
    let Some((hlat, hlon)) = state.home else {
        return Err(Verdict::deny(
            E_NO_GPS_FIX,
            "R-HW/failClosed",
            "no home position yet; a geofenced target cannot be verified",
        ));
    };
    let d = haversine_m(hlat, hlon, lat, lon);
    if d > b.geofence_radius_m as f64 {
        return Err(Verdict::deny(
            E_SAFETY_VIOLATION,
            "R-HW/bounds/geofenceRadiusM",
            format!("target is {d:.0} m from home, beyond the {:.0} m fence", b.geofence_radius_m),
        ));
    }
    Ok(())
}

/// Great-circle distance in metres.
pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    // asinh form is stable for small a (avoids acos domain edge).
    2.0 * EARTH_RADIUS_M * a.sqrt().asinh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::mavlink::{frame_id, msg_command_long, msg_heartbeat, msg_set_position_target_global_int, type_mask};
    use serde_json::json;

    const DEV: &str = "udp:127.0.0.1:14550";

    fn policy(v: serde_json::Value) -> HardwarePolicy {
        HardwarePolicy::from_policy(&v)
    }

    /// A fully-enabled allowlist policy with a generous fence, for testing the
    /// rule under examination rather than the ones around it.
    fn open_policy() -> HardwarePolicy {
        policy(json!({
            "hardware": {
                "enabled": true,
                "devices": { "mode": "allowlist",
                             "allow": [{ "path": "udp:127.0.0.1:14550", "class": "mavlink-udp" }] },
                "mavlink": { "mode": "allowlist",
                             "send": ["HEARTBEAT", "COMMAND_*", "SET_POSITION_TARGET_*", "MISSION_*"] },
                "bounds": { "maxSpeedMs": 10, "maxAltRelM": 50, "maxYawRateDs": 90, "geofenceRadiusM": 300 }
            }
        }))
    }

    fn state_at_home(now: u64) -> GateState {
        let mut s = GateState::default();
        s.gps_fix_type = 3;
        s.gps_hdop = Some(0.8);
        s.last_gps_ms = Some(now);
        s.last_telemetry_ms = Some(now);
        s.home = Some((52.1234, 4.1234));
        s
    }

    fn takeoff(alt: f32) -> Frame {
        let mut params = [0.0f32; 7];
        params[6] = alt;
        msg_command_long(1, 255, 191, 1, 1, mavlink::cmd::NAV_TAKEOFF, params, 0)
    }

    fn goto_frame(lat_e7: i32, lon_e7: i32, alt: f32, vel: [f32; 3], yaw_rate: f32, mask: u16) -> Frame {
        msg_set_position_target_global_int(
            1, 255, 191, 1, 1, lat_e7, lon_e7, alt, vel, yaw_rate, mask,
            frame_id::GLOBAL_RELATIVE_ALT_INT,
        )
    }

    const POS_ONLY_MASK: u16 = type_mask::IGNORE_VELOCITY
        | type_mask::IGNORE_YAW
        | type_mask::IGNORE_YAW_RATE;

    /* ── device allowlist ─────────────────────────────────────────── */

    #[test]
    fn absent_hardware_section_is_no_door_at_all() {
        let p = policy(json!({}));
        assert!(!p.enabled);
        let mut st = GateState::default();
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &msg_heartbeat(0, 255, 191) }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.code, E_HARDWARE_DISABLED);
    }

    #[test]
    fn disabled_flag_wins_over_a_fully_specified_policy() {
        let p = policy(json!({ "hardware": { "enabled": false, "devices": { "mode": "allowlist" } } }));
        assert!(!p.enabled);
    }

    #[test]
    fn devices_mode_off_refuses_every_endpoint() {
        let p = policy(json!({ "hardware": { "enabled": true, "devices": { "mode": "off" } } }));
        assert_eq!(p.gate_device(DEV).code, E_SAFETY_VIOLATION);
    }

    #[test]
    fn device_allowlist_is_exact_not_suffix() {
        let p = open_policy();
        assert!(p.gate_device(DEV).allowed);
        // A near-miss must not match: the allowlist is exact after
        // canonicalisation, never a suffix/prefix rule.
        assert!(!p.gate_device("udp:127.0.0.1:14551").allowed);
        assert!(!p.gate_device("udp:evil.example.com:14550").allowed);
    }

    #[test]
    fn udp_endpoints_canonicalise_case_and_brackets() {
        assert_eq!(canonical_device("UDP:Example.COM:14550").unwrap(), "udp:example.com:14550");
        assert_eq!(canonical_device("udp:[::1]:14550").unwrap(), "udp:::1:14550");
        assert!(canonical_device("udp::14550").is_none());
        assert!(canonical_device("udp:host:notaport").is_none());
    }

    #[test]
    fn filesystem_paths_resolve_symlinks_and_refuse_missing() {
        let dir = std::env::temp_dir().join(format!("sofuu-hw-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("ttyAMA0");
        std::fs::write(&real, b"").unwrap();
        let link = dir.join("link");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let canonical = canonical_device(link.to_str().unwrap()).unwrap();
        assert_eq!(canonical, std::fs::canonicalize(&real).unwrap().display().to_string());
        assert!(canonical_device("/dev/definitely-not-a-device-sofuu").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /* ── mode gates ───────────────────────────────────────────────── */

    #[test]
    fn observe_mode_allows_heartbeat_and_refuses_commands() {
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "observe" } }
        }));
        let mut st = GateState::default();
        let hb = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &msg_heartbeat(0, 255, 191) }, 0, false);
        assert!(hb.allowed, "{hb:?}");

        let cmd = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(10.0) }, 0, false);
        assert!(!cmd.allowed);
        assert_eq!(cmd.code, E_SAFETY_VIOLATION);
        assert_eq!(cmd.rule, "R-HW/mavlink.mode");
    }

    #[test]
    fn mavlink_mode_off_refuses_even_heartbeat() {
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "off" } }
        }));
        let mut st = GateState::default();
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &msg_heartbeat(0, 255, 191) }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.rule, "R-HW/mavlink.mode");
    }

    #[test]
    fn allowlist_mode_refuses_a_message_that_is_not_listed() {
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "allowlist", "send": ["HEARTBEAT"] } }
        }));
        let mut st = state_at_home(0);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(10.0) }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.rule, "R-HW/mavlink.send");
    }

    /* ── deny list ────────────────────────────────────────────────── */

    #[test]
    fn deny_list_beats_a_broad_allow() {
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "allowlist", "send": ["COMMAND_*"],
                                       "deny": ["COMMAND_*"] } }
        }));
        let mut st = state_at_home(0);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(10.0) }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.rule, "R-HW/deny");
    }

    #[test]
    fn motor_test_and_param_set_are_denied_by_every_profile() {
        let p = open_policy();
        let mut st = state_at_home(0);
        for cmd in [mavlink::cmd::DO_MOTOR_TEST, mavlink::cmd::PARAM_SET] {
            let f = msg_command_long(1, 255, 191, 1, 1, cmd, [0.0; 7], 0);
            let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
            assert!(!v.allowed, "cmd {cmd} must never leave");
            assert_eq!(v.rule, "R-HW/deny/always");
        }
    }

    #[test]
    fn observe_mode_still_names_the_permanent_deny_for_motor_test() {
        // `policy why` must say the strongest true thing: DO_MOTOR_TEST is
        // denied by every profile, not merely by observe mode.
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "observe" } }
        }));
        let mut st = GateState::default();
        let f = msg_command_long(1, 255, 191, 1, 1, mavlink::cmd::DO_MOTOR_TEST, [0.0; 7], 0);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
        assert_eq!(v.rule, "R-HW/deny/always");
    }

    #[test]
    fn policy_may_name_a_command_with_or_without_the_mav_cmd_prefix() {
        for spelling in ["MAV_CMD_NAV_RETURN_TO_LAUNCH", "NAV_RETURN_TO_LAUNCH"] {
            let p = policy(json!({
                "hardware": { "enabled": true,
                              "devices": { "mode": "allowlist",
                                           "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                              "mavlink": { "mode": "allowlist", "send": ["COMMAND_*"],
                                           "deny": [spelling] } }
            }));
            let mut st = state_at_home(0);
            let f = msg_command_long(1, 255, 191, 1, 1, mavlink::cmd::NAV_RETURN_TO_LAUNCH, [0.0; 7], 0);
            let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
            assert!(!v.allowed, "{spelling} must deny");
            assert_eq!(v.rule, "R-HW/deny");
        }
    }

    /* ── command whitelist ────────────────────────────────────────── */

    #[test]
    fn an_unvetted_mav_cmd_is_refused_even_when_the_message_is_allowed() {
        let p = open_policy();
        let mut st = state_at_home(0);
        // 190 = MAV_CMD_DO_SET_ACTUATOR: real, dangerous, not whitelisted.
        let f = msg_command_long(1, 255, 191, 1, 1, 190, [0.0; 7], 0);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.rule, "R-HW/command");
    }

    /* ── bounds: reject, never clamp ──────────────────────────────── */

    #[test]
    fn takeoff_above_the_ceiling_is_rejected_not_clamped() {
        let p = open_policy();
        let mut st = state_at_home(0);
        let f = takeoff(80.0);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.rule, "R-HW/bounds/maxAltRelM");
        // The frame the caller holds is untouched: nothing rescaled it.
        assert_eq!(f.f32_at(24), 80.0);
    }

    #[test]
    fn takeoff_within_the_ceiling_is_allowed() {
        let p = open_policy();
        let mut st = state_at_home(0);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(20.0) }, 0, false);
        assert!(v.allowed, "{v:?}");
    }

    #[test]
    fn speed_bound_uses_the_velocity_magnitude() {
        let p = open_policy();
        let mut st = state_at_home(0);
        let vel_mask = type_mask::IGNORE_POSITION | type_mask::IGNORE_YAW | type_mask::IGNORE_YAW_RATE;
        // 3-4-5: magnitude exactly 10 m/s at the bound → allowed.
        let at = goto_frame(521_234_000, 41_234_000, 20.0, [3.0, 4.0, 0.0], 0.0, vel_mask);
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &at }, 0, false).allowed);
        // 6-8-10 → magnitude 10 too; use a genuinely faster one.
        let over = goto_frame(521_234_000, 41_234_000, 20.0, [9.0, 9.0, 0.0], 0.0, vel_mask);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &over }, 0, false);
        assert_eq!(v.rule, "R-HW/bounds/maxSpeedMs");
    }

    #[test]
    fn masked_velocity_fields_are_not_checked() {
        let p = open_policy();
        let mut st = state_at_home(0);
        let f = goto_frame(
            521_234_000, 41_234_000, 20.0, [99.0, 99.0, 99.0], 0.0,
            type_mask::IGNORE_VELOCITY | type_mask::IGNORE_YAW | type_mask::IGNORE_YAW_RATE,
        );
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false).allowed);
    }

    #[test]
    fn yaw_rate_bound_converts_radians_to_degrees() {
        let p = open_policy(); // 90 deg/s
        let mut st = state_at_home(0);
        let mask = type_mask::IGNORE_POSITION | type_mask::IGNORE_VELOCITY | type_mask::IGNORE_YAW;
        let rad = 90.0f32.to_radians();
        let ok = goto_frame(0, 0, 0.0, [0.0; 3], rad, mask);
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &ok }, 0, false).allowed);

        let over = goto_frame(0, 0, 0.0, [0.0; 3], 3.0, mask); // ~172 deg/s
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &over }, 0, false);
        assert_eq!(v.rule, "R-HW/bounds/maxYawRateDs");
    }

    #[test]
    fn a_target_beyond_the_fence_is_rejected() {
        let p = open_policy();
        let mut st = state_at_home(0);
        // ~1.1 km north of home.
        let f = goto_frame(521_234_000 + 100_000, 41_234_000, 20.0, [0.0; 3], 0.0, POS_ONLY_MASK);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.rule, "R-HW/bounds/geofenceRadiusM");
    }

    #[test]
    fn haversine_is_sane() {
        let d = haversine_m(0.0, 0.0, 1.0, 0.0);
        assert!((d - 111_195.0).abs() < 100.0, "one degree of latitude ≈ 111.2 km, got {d}");
        assert_eq!(haversine_m(52.1, 4.1, 52.1, 4.1), 0.0);
    }

    /* ── fail-closed on GPS ───────────────────────────────────────── */

    #[test]
    fn stale_gps_vetoes_a_coordinate_command() {
        let p = open_policy();
        let mut st = state_at_home(0);
        let f = goto_frame(521_234_000, 41_234_000, 20.0, [0.0; 3], 0.0, POS_ONLY_MASK);
        // 2 s later, with no telemetry in between.
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 2_000, false);
        assert!(!v.allowed);
        assert_eq!(v.code, E_NO_GPS_FIX);
    }

    #[test]
    fn a_2d_fix_vetoes_a_coordinate_command() {
        let p = open_policy();
        let mut st = state_at_home(0);
        st.gps_fix_type = 2;
        let f = goto_frame(521_234_000, 41_234_000, 20.0, [0.0; 3], 0.0, POS_ONLY_MASK);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
        assert_eq!(v.code, E_NO_GPS_FIX);
    }

    #[test]
    fn a_high_hdop_vetoes_a_coordinate_command() {
        let p = open_policy();
        let mut st = state_at_home(0);
        st.gps_hdop = Some(2.6);
        let f = goto_frame(521_234_000, 41_234_000, 20.0, [0.0; 3], 0.0, POS_ONLY_MASK);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
        assert_eq!(v.code, E_NO_GPS_FIX);
    }

    #[test]
    fn gps_telemetry_refreshes_the_state_the_gate_reads() {
        let mut st = GateState::default();
        let mut gps = Frame::new(24, 0, 1, 1).unwrap();
        gps.set_u16_at(20, 260); // eph = 2.6 m → HDOP 2.6
        gps.set_u8_at(28, 4);
        st.observe(&gps, 100);
        assert_eq!(st.gps_fix_type, 4);
        assert_eq!(st.gps_hdop, Some(2.6));
        assert_eq!(st.last_gps_ms, Some(100));

        // A parameter-only message must not be mistaken for a GPS fix.
        let sys = Frame::new(1, 0, 1, 1).unwrap();
        st.observe(&sys, 500);
        assert_eq!(st.last_gps_ms, Some(100), "SYS_STATUS is not a fix");
        assert_eq!(st.last_telemetry_ms, Some(500));
    }

    #[test]
    fn home_is_the_first_valid_global_position_and_a_zero_placeholder_is_ignored() {
        let mut st = GateState::default();
        let none = Frame::new(33, 0, 1, 1).unwrap(); // all zeros
        st.observe(&none, 10);
        assert!(st.home.is_none(), "0,0 is ArduPilot's no-fix placeholder");

        let mut pos = Frame::new(33, 0, 1, 1).unwrap();
        pos.set_u32_at(4, 521_234_000);
        pos.set_u32_at(8, 41_234_000);
        st.observe(&pos, 20);
        assert_eq!(st.home, Some((52.1234, 4.1234)));

        let mut elsewhere = Frame::new(33, 0, 1, 1).unwrap();
        elsewhere.set_u32_at(4, 520_000_000);
        elsewhere.set_u32_at(8, 40_000_000);
        st.observe(&elsewhere, 30);
        assert_eq!(st.home, Some((52.1234, 4.1234)), "home is set once");
    }

    /* ── rate limits ──────────────────────────────────────────────── */

    #[test]
    fn rate_limit_parsing() {
        assert_eq!(parse_rate("10Hz"), Some(Period::Every(100)));
        assert_eq!(parse_rate("1Hz"), Some(Period::Every(1000)));
        assert_eq!(parse_rate("0.5Hz"), Some(Period::Every(2000)));
        assert_eq!(parse_rate("100ms"), Some(Period::Every(100)));
        assert_eq!(parse_rate("2s"), Some(Period::Every(2000)));
        assert_eq!(parse_rate(""), None);
        assert_eq!(parse_rate("soon"), None);
        assert_eq!(parse_rate("0Hz"), None);
    }

    #[test]
    fn an_unparseable_rate_limit_denies_instead_of_allowing() {
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "allowlist", "send": ["HEARTBEAT"],
                                       "rateLimits": { "HEARTBEAT": "as fast as you like" } } }
        }));
        let mut st = GateState::default();
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &msg_heartbeat(0, 255, 191) }, 0, false);
        assert!(!v.allowed);
        assert_eq!(v.rule, "R-HW/rateLimits");
    }

    #[test]
    fn rate_limit_blocks_a_second_send_too_soon_and_allows_one_later() {
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "allowlist", "send": ["HEARTBEAT"],
                                       "rateLimits": { "HEARTBEAT": "1Hz" } } }
        }));
        let mut st = GateState::default();
        let hb = msg_heartbeat(0, 255, 191);
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &hb }, 0, false).allowed);
        let second = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &hb }, 500, false);
        assert!(!second.allowed);
        assert_eq!(second.rule, "R-HW/rateLimits");
        /* The refusal carries the remaining wait, so a mission executor paces
         * itself rather than guessing or aborting a multi-command step. */
        assert_eq!(second.retry_after_ms, Some(500));
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &hb }, 1000, false).allowed);
    }

    #[test]
    fn the_strictest_matching_rate_limit_wins_regardless_of_order() {
        let p = policy(json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "allowlist", "send": ["COMMAND_*"],
                                       "rateLimits": { "*": "1Hz", "COMMAND_*": "10Hz" } } }
        }));
        let mut st = state_at_home(0);
        let f = takeoff(10.0);
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false).allowed);
        // 200 ms later the 10 Hz rule would allow; the 1 Hz '*' rule does not.
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 200, false);
        assert!(!v.allowed, "the 1 Hz '*' rule must dominate");
    }

    #[test]
    fn a_denied_send_does_not_consume_the_rate_limit_clock() {
        let p = open_policy();
        let mut st = state_at_home(0);
        assert!(!gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(999.0) }, 0, false).allowed);
        // The clock is untouched, so a legal takeoff right after is allowed.
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(10.0) }, 0, false).allowed);
    }

    /* ── approval ─────────────────────────────────────────────────── */

    #[test]
    fn approval_off_never_asks() {
        let p = open_policy();
        let mut st = state_at_home(0);
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(10.0) }, 0, false).allowed);
    }

    #[test]
    fn approval_every_asks_every_time() {
        let mut p = open_policy();
        p.approval = Approval::Every;
        let mut st = state_at_home(0);
        let f = takeoff(10.0);
        assert_eq!(
            gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false).code,
            E_APPROVAL_REQUIRED
        );
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 1, true).allowed);
        assert_eq!(
            gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 2, false).code,
            E_APPROVAL_REQUIRED
        );
    }

    #[test]
    fn approval_once_per_run_caches_the_answer() {
        let mut p = open_policy();
        p.approval = Approval::OncePerRun;
        let mut st = state_at_home(0);
        let f = takeoff(10.0);
        assert_eq!(
            gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false).code,
            E_APPROVAL_REQUIRED
        );
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 1, true).allowed);
        assert!(gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 2, false).allowed);
    }

    /* ── audit + fail-closed parsing ──────────────────────────────── */

    #[test]
    fn every_decision_carries_a_rule_and_an_audit_record() {
        let p = open_policy();
        let mut st = state_at_home(0);
        let v = gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &takeoff(80.0) }, 5, false);
        assert!(!v.rule.is_empty());
        let rec = v.audit_record(DEV, "COMMAND_LONG", 5);
        assert_eq!(rec["kind"], "hardware.gate");
        assert_eq!(rec["decision"], "deny");
        assert_eq!(rec["code"], E_SAFETY_VIOLATION);
        assert_eq!(rec["message"], "COMMAND_LONG");
    }

    #[test]
    fn bounds_defaults_are_conservative_when_the_policy_omits_them() {
        let p = policy(json!({ "hardware": { "enabled": true } }));
        assert_eq!(p.bounds.max_speed_ms, DEFAULT_MAX_SPEED_MS);
        assert_eq!(p.bounds.max_alt_rel_m, DEFAULT_MAX_ALT_REL_M);
        assert_eq!(p.bounds.max_yaw_rate_ds, DEFAULT_MAX_YAW_RATE_DS);
        assert_eq!(p.bounds.geofence_radius_m, DEFAULT_GEOFENCE_RADIUS_M);
    }

    #[test]
    fn an_unknown_mode_string_fails_closed() {
        for mode in ["allowalist", "yes", ""] {
            let p = policy(json!({
                "hardware": { "enabled": true, "devices": { "mode": mode },
                              "mavlink": { "mode": mode } }
            }));
            assert_eq!(p.devices_mode, DevicesMode::Off, "devices.mode {mode:?}");
            assert_eq!(p.mavlink_mode, MavlinkMode::Off, "mavlink.mode {mode:?}");
        }
    }
}
