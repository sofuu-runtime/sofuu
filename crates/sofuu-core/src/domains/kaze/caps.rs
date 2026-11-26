// domains/kaze/caps.rs — capability registry, action trace, and host adapters.
//
// Kaze scripts name what they need (`needs:`); the host supplies configured
// backends through `CapHost`. All host arguments and results cross `libsofuu_kaze.a`
// (`kaze_vm.c`) as bounded JSON values (`serde_json::Value`), with zero duplicated
// VM value arenas on the Rust side.

use serde_json::{json, Value as Json};

/// Capability families the language knows. Anything else written in `needs:`
/// or before a dot is a **device instance name** (`fan`, `mister`, `lidar`).
pub const FAMILIES: &[&str] = &[
    "drone", "rover", "sensor", "device", "memory", "policy", "log", "alert", "ai", "tool",
];

/// The families an explicitly selected physical test simulator may answer.
pub const PHYSICAL_TEST_FAMILIES: &[&str] = &["drone", "rover"];

/// Instance names resolved to a capability family instead of falling back to `device`.
pub const ALIASES: &[(&str, &str)] = &[
    ("tools", "tool"),
    ("agent", "ai"),
    ("model", "ai"),
    ("llm", "ai"),
    ("lidar", "sensor"),
    ("camera", "sensor"),
    ("thermostat", "sensor"),
    ("probe", "sensor"),
    ("fan", "device"),
    ("mister", "device"),
    ("pump", "device"),
    ("relay", "device"),
    ("valve", "device"),
    ("motor", "device"),
    ("light", "device"),
];

/// Bare telemetry names (`when battery < 20%`) read through the script's sole hardware family.
pub const TELEMETRY: &[&str] = &[
    "battery", "altitude", "heading", "speed", "position", "temperature", "humidity", "distance",
    "armed",
];

pub fn is_telemetry(name: &str) -> bool {
    TELEMETRY.contains(&name)
}

/// Resolve a declared resource or a dotted receiver to its family.
pub fn family_of(resource: &str) -> &'static str {
    if let Some(f) = FAMILIES.iter().find(|f| **f == resource) {
        return f;
    }
    if let Some((_, f)) = ALIASES.iter().find(|(a, _)| *a == resource) {
        return f;
    }
    "device"
}

/// The hardware families a bare telemetry read could come from.
pub fn hardware_families() -> &'static [&'static str] {
    &["drone", "rover", "sensor"]
}

/// Allowed argument counts for one operation, or `None` when unknown.
pub fn op_arity(family: &str, op: &str) -> Option<(usize, usize)> {
    match (family, op) {
        ("drone", "arm" | "land" | "rtl" | "hold" | "stop") => Some((0, 0)),
        ("drone", "takeoff") => Some((1, 1)),
        ("drone", "goto") => Some((1, 3)),
        ("drone", "telemetry") => Some((1, 1)),
        ("rover", "forward" | "backward") => Some((0, 1)),
        ("rover", "turn_left" | "turn_right") => Some((1, 1)),
        ("rover", "stop") => Some((0, 0)),
        ("rover", "telemetry") => Some((1, 1)),
        ("sensor", "temperature" | "humidity" | "distance" | "front_distance") => Some((0, 0)),
        ("sensor", "read") => Some((1, 1)),
        ("sensor", "telemetry") => Some((1, 1)),
        ("device", "set_relay") => Some((2, 2)),
        ("device", "set_duty") => Some((2, 2)),
        ("device", "pulse") => Some((2, 2)),
        ("device", "read_gpio") => Some((1, 1)),
        ("memory", "save") => Some((2, 2)),
        ("memory", "recall") => Some((1, 1)),
        ("policy", "detects" | "detect") => Some((2, 2)),
        ("policy", "allow") => Some((0, 1)),
        ("policy", "reject") => Some((1, 1)),
        ("log", "info" | "warn" | "error" | "debug") => Some((1, 1)),
        ("alert", "send") => Some((1, 1)),
        ("ai", "prompt" | "complete") => Some((1, 1)),
        ("ai", "classify") => Some((2, 2)),
        ("tool", "call" | "run") => Some((1, 2)),
        _ => None,
    }
}

pub fn known_op(family: &str, op: &str) -> bool {
    op_arity(family, op).is_some()
}

/// Where an action's effect came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionMode {
    Real,
    Simulated,
}

impl ActionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ActionMode::Real => "real",
            ActionMode::Simulated => "simulated",
        }
    }
}

/// One line of the execution trace.
#[derive(Clone, Debug)]
pub struct Action {
    pub seq: u32,
    pub family: String,
    pub op: String,
    pub args: Vec<String>,
    pub result: String,
    pub mode: ActionMode,
    pub backend: String,
}

impl Action {
    pub fn render(&self) -> String {
        let args = if self.args.is_empty() {
            String::new()
        } else {
            format!("({})", self.args.join(", "))
        };
        let backend = if self.backend.is_empty() { "none" } else { self.backend.as_str() };
        format!(
            "{}.{}{} -> {} [{}: {}]",
            self.family,
            self.op,
            args,
            self.result,
            self.mode.as_str(),
            backend
        )
    }
}

/// The modelled machine behind a physical test simulation.
#[derive(Clone, Debug)]
pub struct SimState {
    pub battery: f64,
    pub armed: bool,
    pub altitude_m: f64,
    pub on_ground: bool,
    pub lat: f64,
    pub lon: f64,
    pub heading: f64,
    pub speed: f64,
    pub distance: f64,
}

impl Default for SimState {
    fn default() -> Self {
        SimState {
            battery: 100.0,
            armed: false,
            altitude_m: 0.0,
            on_ground: true,
            lat: 0.0,
            lon: 0.0,
            heading: 0.0,
            speed: 0.0,
            distance: 2.0,
        }
    }
}

impl SimState {
    /// `key=value,key=value` overrides for reproducible simulation scenarios.
    pub fn apply_overrides(&mut self, spec: &str) -> Result<(), String> {
        for pair in spec.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let (k, v) = pair
                .split_once('=')
                .ok_or_else(|| format!("E_CAP_INVALID: --sim expects key=value, got '{}'", pair))?;
            let key = k.trim();
            let raw = v.trim();
            let numeric = |raw: &str| -> Result<f64, String> {
                raw.parse::<f64>()
                    .ok()
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| format!("E_CAP_INVALID: --sim {}: '{}' is not a finite number", key, raw))
            };
            let boolean = |raw: &str| -> Result<bool, String> {
                match raw {
                    "true" | "1" | "yes" => Ok(true),
                    "false" | "0" | "no" => Ok(false),
                    _ => Err(format!("E_CAP_INVALID: --sim {}: '{}' is not a boolean", key, raw)),
                }
            };
            match key {
                "battery" => self.battery = numeric(raw)?,
                "altitude" => self.altitude_m = numeric(raw)?,
                "heading" => self.heading = numeric(raw)?,
                "speed" => self.speed = numeric(raw)?,
                "distance" | "front_distance" => self.distance = numeric(raw)?,
                "lat" => self.lat = numeric(raw)?,
                "lon" => self.lon = numeric(raw)?,
                "armed" => self.armed = boolean(raw)?,
                "on_ground" => self.on_ground = boolean(raw)?,
                other => {
                    return Err(format!(
                        "E_CAP_INVALID: --sim: unknown key '{}'; the test simulator models a vehicle \
                         (battery, altitude, heading, speed, distance, lat, lon, armed, on_ground)",
                        other
                    ))
                }
            }
        }
        Ok(())
    }

    fn drain(&mut self, amount: f64) {
        self.battery = (self.battery - amount).max(0.0);
    }
}

/// Format a finite float cleanly (integers without `.0`).
pub fn fmt_num(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

/// Render any JSON-encoded Kaze value in Kaze's human display form.
pub fn render_value(v: &Json) -> String {
    match v {
        Json::Null => "nothing".to_string(),
        Json::Number(n) => n.as_f64().map(fmt_num).unwrap_or_else(|| n.to_string()),
        Json::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Json::String(s) => format!("\"{}\"", s),
        Json::Array(items) => {
            let parts: Vec<String> = items.iter().map(render_value).collect();
            format!("[{}]", parts.join(", "))
        }
        Json::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, val)| format!("{}: {}", k, render_value(val)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

fn str_at(args: &[Json], idx: usize) -> String {
    match args.get(idx) {
        Some(Json::String(s)) => s.clone(),
        Some(other) => render_value(other),
        None => String::new(),
    }
}

fn num_at(args: &[Json], idx: usize, fallback: f64) -> f64 {
    args.get(idx).and_then(Json::as_f64).unwrap_or(fallback)
}

/// Host capability backend invoked by `kaze_vm.c` after R7 checks pass.
pub trait CapHost {
    fn call(&mut self, family: &str, op: &str, args: &[Json]) -> Result<Json, String>;
    fn actions(&self) -> &[Action];
    fn logs(&self) -> &[(String, String)];
    fn check_async_dispatch(&self) -> Result<(), String> { Ok(()) }
    fn note_async_completion(&mut self, _family: &str, _op: &str, _succeeded: bool) {}
    fn note_async_unknown(&mut self, _family: &str, _op: &str) {}
}

/// Ordinary host dispatcher (`log`, `policy`, `alert`, optional `SimHost`).
pub struct RuntimeHost {
    physical: Option<SimHost>,
    blocked_live: Vec<String>,
    alert_sink: Option<String>,
    pub(crate) actions: Vec<Action>,
    pub(crate) logs: Vec<(String, String)>,
    log_bytes: usize,
    seq: u32,
}

const MAX_LOG_RECORDS: usize = 1024;
const MAX_LOG_BYTES: usize = 256 * 1024;
const MAX_ACTION_RECORDS: usize = 4096;

const SIM_BACKEND: &str = "sim";
const LOG_BACKEND: &str = "local-log";
const POLICY_DETECTOR_BACKEND: &str = "policy-detector";

impl RuntimeHost {
    pub fn new(state: SimState, physical_sim: Vec<String>, blocked_live: Vec<String>) -> Self {
        Self::with_alert(state, physical_sim, blocked_live, None)
    }

    pub fn with_alert(
        state: SimState,
        physical_sim: Vec<String>,
        blocked_live: Vec<String>,
        alert_sink: Option<String>,
    ) -> Self {
        let physical = if physical_sim.is_empty() {
            None
        } else {
            Some(SimHost::with_simulated(state, physical_sim))
        };
        Self { physical, blocked_live, alert_sink, actions: Vec::new(), logs: Vec::new(), log_bytes: 0, seq: 0 }
    }

    pub(crate) fn note(&mut self, family: &str, op: &str, args: &[String], result: String) {
        self.note_with_backend(family, op, args, result, ActionMode::Real, "local");
    }

    pub(crate) fn note_with_backend(
        &mut self,
        family: &str,
        op: &str,
        args: &[String],
        result: String,
        mode: ActionMode,
        backend: &str,
    ) {
        self.seq += 1;
        self.actions.push(Action {
            seq: self.seq,
            family: family.to_string(),
            op: op.to_string(),
            args: args.to_vec(),
            result,
            mode,
            backend: backend.to_string(),
        });
    }

    pub(crate) fn check_trace_capacity(&self) -> Result<(), String> {
        if self.actions.len() >= MAX_ACTION_RECORDS {
            Err("E_KAZE_MEMORY: per-run action trace limit reached".to_string())
        } else {
            Ok(())
        }
    }
}

impl CapHost for RuntimeHost {
    fn actions(&self) -> &[Action] { &self.actions }
    fn logs(&self) -> &[(String, String)] { &self.logs }

    fn check_async_dispatch(&self) -> Result<(), String> {
        self.check_trace_capacity()
    }

    fn note_async_completion(&mut self, family: &str, op: &str, succeeded: bool) {
        self.note(
            family,
            op,
            &["<redacted>".to_string()],
            if succeeded { "<redacted result>" } else { "<redacted error>" }.to_string(),
        );
    }

    fn note_async_unknown(&mut self, family: &str, op: &str) {
        self.note(
            family,
            op,
            &["<redacted>".to_string()],
            "<unknown outcome>".to_string(),
        );
    }

    fn call(&mut self, family: &str, op: &str, args: &[Json]) -> Result<Json, String> {
        self.check_trace_capacity()?;
        if self.blocked_live.iter().any(|selected| selected == family) {
            return Err(format!(
                "E_CAP_DENIED: live {}.{} is disabled until hardware policy and safety checks are verified",
                family, op
            ));
        }
        let is_physical = matches!(family, "drone" | "rover");
        let shown: Vec<String> = if is_physical {
            args.iter().map(|v| match v {
                Json::Null | Json::Number(_) | Json::Bool(_) => render_value(v),
                _ => "<redacted>".to_string(),
            }).collect()
        } else {
            args.iter().map(|_| "<redacted>".to_string()).collect()
        };
        let (result, mode, backend) = match (family, op) {
            ("drone" | "rover", _) => match &mut self.physical {
                Some(host) if host.serves(family) => {
                    let result = host.call(family, op, args);
                    (result, ActionMode::Simulated, SIM_BACKEND.to_string())
                }
                _ => (
                    Err(format!("E_CAP_MISSING: no backend selected for {}.{}; drone/rover simulation is available for tests with an explicit sim grant", family, op)),
                    ActionMode::Real,
                    String::new(),
                ),
            },
            ("policy", "detects" | "detect") => {
                let kind = str_at(args, 0).to_lowercase();
                let text = str_at(args, 1);
                (
                    detect(&kind, &text).map(Json::Bool),
                    ActionMode::Real,
                    POLICY_DETECTOR_BACKEND.to_string(),
                )
            }
            ("policy", "allow") | ("policy", "reject") => (
                Err("E_CAP_INVALID: policy verdicts are the VM's own control flow; no backend serves them".to_string()),
                ActionMode::Real,
                String::new(),
            ),
            ("log", "info" | "warn" | "error" | "debug") => {
                let msg = str_at(args, 0);
                let level = match op { "warn" => "warning", "error" => "error", "debug" => "debug", _ => "info" };
                let result = if self.logs.len() >= MAX_LOG_RECORDS || self.log_bytes.saturating_add(msg.len()) > MAX_LOG_BYTES {
                    Err("E_KAZE_MEMORY: per-run log limit reached".to_string())
                } else {
                    self.log_bytes += msg.len();
                    self.logs.push((level.to_string(), msg));
                    Ok(Json::Null)
                };
                (result, ActionMode::Real, LOG_BACKEND.to_string())
            }
            ("alert", "send") => {
                let Some(sink) = self.alert_sink.clone().filter(|s| !s.is_empty()) else {
                    return Err("E_CAP_MISSING: embedded Kaze has no available backend for alert.send".to_string());
                };
                let msg = str_at(args, 0);
                if msg.is_empty() || msg.len() > 1024 {
                    return Err("E_KAZE_RANGE: alert message must contain 1 to 1024 bytes".to_string());
                }
                (deliver_alert_msg(&sink, &msg), ActionMode::Real, sink)
            }
            _ => (
                Err(format!("E_CAP_MISSING: embedded Kaze has no available backend for {}.{}", family, op)),
                ActionMode::Real,
                String::new(),
            ),
        };
        let rendered = match &result {
            Ok(Json::Null) => "nothing".to_string(),
            Ok(v) if is_physical => render_value(v),
            Ok(_) => "<redacted>".to_string(),
            Err(e) if is_physical => format!("error: {}", e),
            Err(_) => "error".to_string(),
        };
        self.note_with_backend(family, op, &shown, rendered, mode, &backend);
        result
    }
}

fn deliver_alert_msg(sink: &str, msg: &str) -> Result<Json, String> {
    use std::io::Write;
    let line = format!("[ALERT] {}\n", msg);
    let bytes = line.as_bytes();
    let res = match sink {
        "stdout" => std::io::stdout().write_all(bytes),
        "stderr" => std::io::stderr().write_all(bytes),
        s if s.starts_with("file:") => {
            let path = &s[5..];
            if path.is_empty() {
                return Err("E_CONFIG: empty file path for alert sink".to_string());
            }
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut f| f.write_all(bytes))
        }
        unsupported => return Err(format!("E_CONFIG: unsupported alert sink '{}'", unsupported)),
    };
    res.map_err(|e| format!("E_IO: failed to deliver alert: {}", e))?;
    Ok(Json::Null)
}

/// Physical test simulation only (`drone` and `rover` when explicitly selected).
pub struct SimHost {
    pub state: SimState,
    selected: Vec<String>,
    blocked_live: Vec<String>,
}

impl SimHost {
    pub fn new(state: SimState) -> Self {
        Self::with_simulated(state, Vec::new())
    }

    pub fn with_simulated(state: SimState, physical_sim: Vec<String>) -> Self {
        let selected = physical_sim
            .into_iter()
            .filter(|family| PHYSICAL_TEST_FAMILIES.contains(&family.as_str()))
            .collect();
        SimHost { state, selected, blocked_live: Vec::new() }
    }

    pub fn with_configuration(state: SimState, physical_sim: Vec<String>, blocked_live: Vec<String>) -> Self {
        let mut host = Self::with_simulated(state, physical_sim);
        host.blocked_live = blocked_live;
        host
    }

    pub fn serves(&self, family: &str) -> bool {
        self.selected.iter().any(|selected| selected == family)
    }

    fn telemetry(&self, family: &str, field: &str) -> Result<Json, String> {
        match (family, field) {
            ("drone", "battery") => Ok(json!(self.state.battery)),
            ("drone", "altitude") => Ok(json!(self.state.altitude_m)),
            ("drone", "heading") => Ok(json!(self.state.heading)),
            ("drone", "speed") => Ok(json!(self.state.speed)),
            ("drone", "armed") => Ok(Json::Bool(self.state.armed)),
            ("drone", "latitude" | "lat") => Ok(json!(self.state.lat)),
            ("drone", "longitude" | "lon") => Ok(json!(self.state.lon)),
            ("rover", "battery") => Ok(json!(self.state.battery)),
            ("rover", "heading") => Ok(json!(self.state.heading)),
            ("rover", "speed") => Ok(json!(self.state.speed)),
            ("rover", "armed") => Ok(Json::Bool(self.state.armed)),
            ("rover", "distance") => Ok(json!(self.state.distance)),
            _ => Err(format!(
                "E_CAP_MISSING: the simulated {} has no '{}' reading",
                family, field
            )),
        }
    }
}

impl CapHost for SimHost {
    fn actions(&self) -> &[Action] { &[] }
    fn logs(&self) -> &[(String, String)] { &[] }

    fn call(&mut self, family: &str, op: &str, args: &[Json]) -> Result<Json, String> {
        if !PHYSICAL_TEST_FAMILIES.contains(&family) {
            return Err(format!(
                "E_CAP_MISSING: the physical test simulator has no backend for {}.{}",
                family, op
            ));
        }
        if self.blocked_live.iter().any(|selected| selected == family) {
            return Err(format!(
                "E_CAP_DENIED: live {}.{} is disabled until hardware policy and safety checks are verified",
                family, op
            ));
        }
        if !self.serves(family) {
            return Err(format!(
                "E_CAP_MISSING: no physical test simulator was selected for {}; \
                 select it explicitly for a test, e.g. --cap {}=sim",
                family, family
            ));
        }
        match (family, op) {
            ("drone", "arm") => {
                self.state.armed = true;
                self.state.drain(1.0);
                Ok(Json::Null)
            }
            ("drone", "takeoff") => {
                if !self.state.armed {
                    return Err("E_CAP_DENIED: the drone is not armed; write 'arm drone' first".to_string());
                }
                let altitude = args.first().and_then(Json::as_f64).filter(|n| n.is_finite())
                    .ok_or_else(|| "E_KAZE_RANGE: drone.takeoff needs a finite altitude in meters".to_string())?;
                self.state.altitude_m = altitude;
                self.state.on_ground = false;
                self.state.drain(1.0);
                Ok(Json::Null)
            }
            ("drone", "goto") => {
                if !self.state.on_ground {
                    let number = |idx: usize| args.get(idx).and_then(Json::as_f64)
                        .filter(|n| n.is_finite())
                        .ok_or_else(|| format!("E_KAZE_RANGE: drone.goto requires finite numeric argument {}", idx + 1));
                    let lat = number(0)?;
                    let (lon, alt) = if args.len() >= 3 {
                        (number(1)?, number(2)?)
                    } else {
                        (self.state.lon, number(1)?)
                    };
                    self.state.lat = lat;
                    self.state.lon = lon;
                    self.state.altitude_m = alt;
                    self.state.drain(1.0);
                    Ok(Json::Null)
                } else {
                    Err("E_CAP_DENIED: the drone is on the ground; take off before flying to a point".to_string())
                }
            }
            ("drone", "rtl") | ("drone", "land") => {
                self.state.altitude_m = 0.0;
                self.state.on_ground = true;
                self.state.speed = 0.0;
                if op == "rtl" {
                    self.state.lat = 0.0;
                    self.state.lon = 0.0;
                }
                Ok(Json::Null)
            }
            ("drone", "hold") | ("drone", "stop") | ("rover", "stop") => {
                self.state.speed = 0.0;
                Ok(Json::Null)
            }
            ("drone", "telemetry") | ("rover", "telemetry") => {
                self.telemetry(family, &str_at(args, 0))
            }
            ("rover", "forward") => {
                self.state.speed = num_at(args, 0, 1.0);
                Ok(Json::Null)
            }
            ("rover", "backward") => {
                self.state.speed = -num_at(args, 0, 1.0).abs();
                Ok(Json::Null)
            }
            ("rover", "turn_left") => {
                let degrees = args.first().and_then(Json::as_f64).filter(|n| n.is_finite())
                    .ok_or_else(|| "E_KAZE_RANGE: rover.turn_left needs a finite angle in degrees".to_string())?;
                self.state.heading = (self.state.heading + degrees) % 360.0;
                Ok(Json::Null)
            }
            ("rover", "turn_right") => {
                let degrees = args.first().and_then(Json::as_f64).filter(|n| n.is_finite())
                    .ok_or_else(|| "E_KAZE_RANGE: rover.turn_right needs a finite angle in degrees".to_string())?;
                self.state.heading = (self.state.heading - degrees) % 360.0;
                Ok(Json::Null)
            }
            _ => Err(format!("E_CAP_MISSING: embedded Kaze has no available backend for {}.{}", family, op)),
        }
    }
}

/// Deterministic secret / prompt-injection detector behind `policy detects <kind> in <text>`.
pub fn detect(kind: &str, text: &str) -> Result<bool, String> {
    match kind {
        "secrets" => {
            let t = text.to_lowercase();
            Ok(t.contains("sk-")
                || t.contains("ghp_")
                || t.contains("xoxb-")
                || t.contains("akia")
                || t.contains("begin private key")
                || t.contains("api_key")
                || t.contains("api key")
                || t.contains("password=")
                || t.contains("password: ")
                || t.contains("secret=")
                || t.contains("token=")
                || t.contains("bearer "))
        }
        "prompt_injection" => {
            let t = text.to_lowercase();
            Ok((t.contains("ignore") && t.contains("instruction"))
                || (t.contains("disregard")
                    && (t.contains("prior") || t.contains("previous") || t.contains("above")))
                || t.contains("system prompt")
                || t.contains("you are now")
                || t.contains("reveal your")
                || t.contains("jailbreak"))
        }
        other => Err(format!(
            "E_CAP_INVALID: unsupported policy detector '{}'; this host implements secrets and prompt_injection",
            other
        )),
    }
}
