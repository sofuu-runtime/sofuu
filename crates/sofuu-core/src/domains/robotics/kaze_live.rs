//! Kaze v0.2 (P-K1) — the live capability backends behind `CapHost`.
//!
//! v0.1 answered every capability call in-process (`sfs::caps::SimHost`). P-K1
//! puts real hardware behind the same trait:
//!
//! * [`DroneLink`] — a live MAVLink link (`udp:host:port`, or `uart:/dev/ttyAMA0@921600`)
//!   with a receiver thread polling at 20 Hz, a 1 Hz HEARTBEAT, and every outgoing
//!   command judged by `mavlink_gate` before it reaches the wire. Commands are
//!   ACK-correlated: a Kaze action returns when the autopilot answered, not when
//!   the bytes left.
//! * [`LiveHost`] — a `CapHost` that routes live hardware to its backend and
//!   other operations to the ordinary fail-closed runtime host. It does not
//!   implicitly select a physical simulator.
//!
//! The invariant is unchanged: `rt::hardware` still judges H-1..H-4 *above* this
//! trait, and `LinkCore::send` still refuses to encode a frame the MAVLink policy
//! has not allowed. Two doors, both locked, in series.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value as J};

use super::mavlink;
use super::mavlink_gate::{self as gate, HardwarePolicy};
use super::mavlink_link::{self as link, Event, Io, LinkCore};
use crate::sfs::arena::Arena;
use crate::sfs::caps::{render_value, CapHost, RuntimeHost, SimState};
use crate::sfs::value::{Heap, Value};

/// How often the receiver thread pumps the transport (20 Hz, inside the plan's
/// 10–50 Hz band).
pub const DEFAULT_POLL_MS: u64 = 50;
/// How long `connect` waits for the vehicle's first datagram over UDP.
pub const PEER_WAIT_MS: u64 = 1_500;
/// How long `connect` waits for the first position report after saying hello.
pub const WARMUP_MS: u64 = 30_000;
/// How long a command waits for its COMMAND_ACK before the script is told the
/// link did not answer.
pub const DEFAULT_ACK_TIMEOUT_MS: u64 = 3_000;
/// A cached reading is not evidence of the vehicle's current state forever.
/// This is a conservative local limit, not a claim of physical safety.
const TELEMETRY_MAX_AGE: Duration = Duration::from_secs(3);

/// What the autopilot last told us. Updated by the receiver thread; read by
/// `drone.telemetry(...)`, never invented.
#[derive(Clone, Debug, Default)]
pub struct DroneTelemetry {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: f64,
    pub rel_alt_m: f64,
    pub heading_deg: f64,
    pub vx_ms: f64,
    pub vy_ms: f64,
    pub battery_pct: f64,
    pub satellites: f64,
    pub armed: bool,
    pub mode: String,
    /// True once a HEARTBEAT has arrived: before that, every reading is a
    /// guess, and a script that acts on a guess is a bug.
    pub heard: bool,
    observed: std::collections::HashMap<&'static str, Instant>,
}

impl DroneTelemetry {
    /// The value `drone.telemetry("<field>")` returns for `field`.
    pub fn read(&self, field: &str) -> Result<Value, String> {
        let field = match field {
            "latitude" => "lat",
            "longitude" => "lon",
            "battery" | "altitude" | "altitude_msl" | "heading" | "lat" | "lon" | "speed" | "satellites" | "armed" => field,
            _ => return Err("E_CAP_MISSING: unknown drone telemetry field".into()),
        };
        self.require_fresh("heartbeat")?;
        self.require_fresh(field)?;
        Ok(match field {
            "battery" => Value::Num(self.battery_pct),
            "altitude" => Value::Num(self.rel_alt_m),
            "altitude_msl" => Value::Num(self.alt_m),
            "heading" => Value::Num(self.heading_deg),
            "lat" => Value::Num(self.lat),
            "lon" => Value::Num(self.lon),
            "speed" => Value::Num(self.vx_ms.hypot(self.vy_ms)),
            "satellites" => Value::Num(self.satellites),
            "armed" => Value::Bool(self.armed),
            _ => unreachable!("fields were validated above"),
        })
    }

    fn require_fresh(&self, field: &str) -> Result<(), String> {
        if self.observed.get(field).is_some_and(|at| at.elapsed() <= TELEMETRY_MAX_AGE) {
            Ok(())
        } else {
            Err(format!("E_KAZE_LINK: drone {} is missing or stale", field))
        }
    }

    fn fresh_mode(&self) -> Result<&str, String> {
        self.require_fresh("heartbeat")?;
        self.require_fresh("mode")?;
        Ok(&self.mode)
    }

    fn apply(&mut self, event: &Event) {
        let now = Instant::now();
        match event {
            Event::Telemetry { message, data } => match *message {
                "HEARTBEAT" => {
                    if let Some(mode) = data["base_mode"].as_u64().filter(|m| *m <= u8::MAX as u64) {
                        self.armed = mode & 128 != 0;
                        self.heard = true;
                        self.observed.insert("heartbeat", now);
                        self.observed.insert("armed", now);
                    } else {
                        self.heard = false;
                        self.observed.remove("heartbeat");
                        self.observed.remove("armed");
                    }
                    if let Some(m) = data["mode"].as_str().filter(|m| !m.is_empty()) {
                        self.mode = m.to_string();
                        self.observed.insert("mode", now);
                    } else {
                        self.observed.remove("mode");
                    }
                }
                "SYS_STATUS" => {
                    let value = data["battery_remaining_pct"].as_f64().filter(|v| (0.0..=100.0).contains(v));
                    record_reading(&mut self.observed, "battery", &mut self.battery_pct, value, now);
                }
                "GLOBAL_POSITION_INT" => {
                    record_reading(&mut self.observed, "lat", &mut self.lat, data["lat"].as_f64().filter(|v| (-90.0..=90.0).contains(v)), now);
                    record_reading(&mut self.observed, "lon", &mut self.lon, data["lon"].as_f64().filter(|v| (-180.0..=180.0).contains(v)), now);
                    record_reading(&mut self.observed, "altitude_msl", &mut self.alt_m, data["alt_m"].as_f64(), now);
                    record_reading(&mut self.observed, "altitude", &mut self.rel_alt_m, data["relative_alt_m"].as_f64(), now);
                    // MAVLink's 65535 sentinel is decoded as 655.35 degrees.
                    record_reading(&mut self.observed, "heading", &mut self.heading_deg, data["heading_deg"].as_f64().filter(|v| (0.0..360.0).contains(v)), now);
                    let velocity = data["vx_ms"].as_f64().zip(data["vy_ms"].as_f64())
                        .filter(|(x, y)| x.is_finite() && y.is_finite() && x.hypot(*y).is_finite());
                    if let Some((x, y)) = velocity {
                        self.vx_ms = x;
                        self.vy_ms = y;
                        self.observed.insert("speed", now);
                    } else {
                        self.observed.remove("speed");
                    }
                }
                "GPS_RAW_INT" => {
                    // MAVLink uses 255 for an unknown satellite count.
                    let value = data["satellites"].as_f64().filter(|v| (0.0..255.0).contains(v));
                    record_reading(&mut self.observed, "satellites", &mut self.satellites, value, now);
                }
                _ => {}
            },
            _ => {}
        }
    }
}

fn record_reading(
    observed: &mut std::collections::HashMap<&'static str, Instant>,
    field: &'static str,
    destination: &mut f64,
    value: Option<f64>,
    now: Instant,
) {
    if let Some(value) = value.filter(|n| n.is_finite()) {
        *destination = value;
        observed.insert(field, now);
    } else {
        // An explicit unknown or malformed update invalidates an older value.
        observed.remove(field);
    }
}

/// One live MAVLink link, owned by its receiver thread.
pub struct DroneLink {
    tx: Sender<Request>,
    tele: Arc<Mutex<DroneTelemetry>>,
    notes: Arc<Mutex<Vec<(String, String)>>>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    ack_timeout_ms: u64,
    pub device: String,
}

struct Request {
    command: u16,
    params: [f32; 7],
    /// A frame that takes no ACK (`SET_POSITION_TARGET_GLOBAL_INT`).
    frame: Option<mavlink::Frame>,
    reply: Sender<Result<(), String>>,
}

impl DroneLink {
    /// Open the transport and start the receiver thread.
    pub fn connect(device: &str) -> Result<DroneLink, String> {
        DroneLink::with_timing(device, DEFAULT_POLL_MS, DEFAULT_ACK_TIMEOUT_MS)
    }

    pub fn with_timing(device: &str, poll_ms: u64, ack_timeout_ms: u64) -> Result<DroneLink, String> {
        let policy = HardwarePolicy::from_policy(&policy_for_device(device));
        let mut core = LinkCore::new(device, policy);
        let verdict = core.check_device();
        if !verdict.allowed {
            return Err(format!("{}: {} ({})", verdict.code, verdict.reason, verdict.rule));
        }
        let mut io = open_device(device)?;
        if device.trim().to_ascii_lowercase().starts_with("udp") {
            // 1. Pin the reply address: a UDP link has no peer until the
            //    vehicle speaks.
            wait_for_first_bytes(io.as_mut(), PEER_WAIT_MS).map_err(|_| {
                format!("E_KAZE_LINK: nothing is streaming to {} — is the autopilot sending there?", device)
            })?;
            // 2. Say hello, so the autopilot treats this as an active GCS and
            //    starts its telemetry streams (SITL stays quiet until it does).
            if let Some(bytes) = core.tick(0) {
                io.write(&bytes)
                    .map_err(|e| format!("E_KAZE_LINK: cannot write to {}: {}", device, e))?;
            }
            // 3. Ask for the messages a flight needs. ArduPilot 4.6 has no
            //    `SRx_*` stream parameters any more: silence until asked is the
            //    default, and a GCS that never asks flies blind.
            request_streams(&mut core, io.as_mut())?;
            // 4. Wait for the first position, so the gate's fail-closed GPS rule
            //    is not tripped by a startup race.
            warm_up(&mut core, io.as_mut(), WARMUP_MS)
                .map_err(|why| format!("E_KAZE_LINK: {} at {}", why, device))?;
        }
        DroneLink::with_io(core, io, device.to_string(), poll_ms, ack_timeout_ms)
    }

    /// The seam the tests use: any `Io`, any clock-speed.
    pub fn with_io(
        mut core: LinkCore,
        mut io: Box<dyn Io>,
        device: String,
        poll_ms: u64,
        ack_timeout_ms: u64,
    ) -> Result<DroneLink, String> {

        // The first heartbeat goes out before the thread starts, so a script's
        // very first command already talks to a vehicle that has heard us.
        if let Some(bytes) = core.tick(0) {
            io.write(&bytes).map_err(|e| format!("E_KAZE_LINK: cannot write to {}: {}", device, e))?;
        }
        let (tx, rx) = channel::<Request>();
        let tele = Arc::new(Mutex::new(DroneTelemetry::default()));
        let notes = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = {
            let tele = Arc::clone(&tele);
            let notes = Arc::clone(&notes);
            let stop = Arc::clone(&stop);
            thread::Builder::new()
                .name(format!("kaze-mavlink-{}", device))
                .spawn(move || run_link(core, io, rx, tele, notes, stop, poll_ms, ack_timeout_ms))
                .map_err(|e| format!("E_KAZE_LINK: cannot start the receiver thread: {}", e))?
        };
        Ok(DroneLink { tx, tele, notes, stop, handle: Some(handle), ack_timeout_ms, device })
    }

    /// Gate and send a frame the vehicle does not acknowledge (a position
    /// target). The gate still judges it: this is not a bypass.
    pub fn send_target(&self, frame: mavlink::Frame) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .send(Request { command: 0, params: [0.0; 7], frame: Some(frame), reply: reply_tx })
            .map_err(|_| "E_KAZE_LINK: the link thread is gone".to_string())?;
        match reply_rx.recv_timeout(Duration::from_millis(self.ack_timeout_ms)) {
            Ok(result) => result,
            Err(_) => Err("E_KAZE_LINK_TIMEOUT: target completion is unknown; the link did not return a result".to_string()),
        }
    }

    /// The frame the link would use for a position target.
    pub fn position_frame(&self, lat: f64, lon: f64, alt: f32) -> mavlink::Frame {
        // The link rewrites `seq` on the way out (`LinkCore::send`), so any
        // value works here; sysid/compid match the LinkCore defaults.
        mavlink::msg_set_position_target_global_int(
            0,
            255,
            191,
            1,
            1,
            (lat * 1e7) as i32,
            (lon * 1e7) as i32,
            alt,
            [0.0, 0.0, 0.0],
            0.0,
            0x0DF8, // position + yaw only (ignore velocity, accel, yaw rate)
            6,      // MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
        )
    }

    /// Gate, send, and wait for the autopilot's COMMAND_ACK.
    pub fn command(&self, command: u16, params: [f32; 7]) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .send(Request { command, params, frame: None, reply: reply_tx })
            .map_err(|_| "E_KAZE_LINK: the link thread is gone".to_string())?;
        let wait = Duration::from_millis(self.ack_timeout_ms.saturating_add(1_000));
        match reply_rx.recv_timeout(wait) {
            Ok(result) => result,
            Err(_) => Err(format!(
                "E_KAZE_LINK_TIMEOUT: completion of {} is unknown; the link did not return a result",
                gate::command_name(command).unwrap_or("the command")
            )),
        }
    }

    pub fn telemetry(&self) -> DroneTelemetry {
        self.tele.clone().lock().map(|t| t.clone()).unwrap_or_default()
    }

    /// Drain what the autopilot said (STATUSTEXT, arming refusals).
    pub fn take_notes(&self) -> Vec<(String, String)> {
        self.notes.lock().map(|mut n| std::mem::take(&mut *n)).unwrap_or_default()
    }

    pub fn is_alive(&self) -> bool {
        self.handle.as_ref().map(|h| !h.is_finished()).unwrap_or(false)
    }
}

impl Drop for DroneLink {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

type Tele = Arc<Mutex<DroneTelemetry>>;

fn run_link(
    mut core: LinkCore,
    mut io: Box<dyn Io>,
    rx: Receiver<Request>,
    tele: Tele,
    notes: Arc<Mutex<Vec<(String, String)>>>,
    stop: Arc<AtomicBool>,
    poll_ms: u64,
    ack_timeout_ms: u64,
) {
    let start = Instant::now();
    let now_ms = || start.elapsed().as_millis() as u64;
    let mut pending: Option<(u16, Sender<Result<(), String>>, u64)> = None;

    while !stop.load(Ordering::SeqCst) {
        let now = now_ms();

        // 1 Hz dead-man heartbeat, emitted by the runtime, never the script.
        if let Some(bytes) = core.tick(now) {
            if io.write(&bytes).is_err() {
                break;
            }
        }

        // Drain the transport.
        match link::pump(&mut core, io.as_mut(), now) {
            Ok(events) => {
                for event in events {
                    match &event {
                        Event::CommandAck { command, result, result_name, .. } => {
                            if let Some((want, _reply, _)) = &pending {
                                if want == command {
                                    let (_, reply, _) = pending.take().expect("checked above");
                                    let verdict = if *result == 0 {
                                        Ok(())
                                    } else {
                                        Err(format!(
                                            "E_COMMAND_REJECTED: the autopilot answered {} for {}",
                                            result_name,
                                            gate::command_name(*command).unwrap_or("the command")
                                        ))
                                    };
                                    let _ = reply.send(verdict);
                                }
                            }
                        }
                        Event::ArmingDenied { text, .. } => {
                            push_note(&notes, "error", text.clone());
                        }
                        Event::StatusText { severity, text } => {
                            let level = if *severity <= 3 { "error" } else if *severity <= 4 { "warning" } else { "info" };
                            push_note(&notes, level, text.clone());
                        }
                        other => {},
                    }
                    if let Ok(mut t) = tele.lock() {
                        t.apply(&event);
                    }
                }
            }
            Err(_) => break,
        }

        // One command at a time: a link is a serial resource.
        if pending.is_none() {
            if let Ok(req) = rx.try_recv() {
                // A position target has no ACK: gate, write, and answer at once.
                if let Some(frame) = req.frame {
                    let sent = link::send_frame(&mut core, io.as_mut(), &frame, now, false);
                    let _ = req.reply.send(match sent {
                        Ok(()) => Ok(()),
                        Err(v) => Err(format!("{}: {} ({})", v.code, v.reason, v.rule)),
                    });
                    thread::sleep(Duration::from_millis(poll_ms));
                    continue;
                }
                let frame = link::command_long(&core, 1, 1, req.command, req.params);
                match link::send_frame(&mut core, io.as_mut(), &frame, now, true) {
                    Ok(()) => pending = Some((req.command, req.reply, now)),
                    Err(v) => {
                        let _ = req.reply.send(Err(format!("{}: {} ({})", v.code, v.reason, v.rule)));
                    }
                }
            }
        }

        // An unanswered command must not hang the script.
        if let Some((_, _, sent)) = &pending {
            if now.saturating_sub(*sent) > ack_timeout_ms {
                let (command, reply, _) = pending.take().expect("checked above");
                let _ = reply.send(Err(format!(
                    "E_KAZE_LINK_TIMEOUT: completion is unknown; no COMMAND_ACK for {} within {} ms",
                    gate::command_name(command).unwrap_or("the command"),
                    ack_timeout_ms
                )));
            }
        }

        thread::sleep(Duration::from_millis(poll_ms));
    }

    if let Some((_, reply, _)) = pending {
        let _ = reply.send(Err("E_KAZE_LINK: the link closed before the autopilot answered".to_string()));
    }
}

fn push_note(notes: &Arc<Mutex<Vec<(String, String)>>>, level: &str, text: String) {
    if let Ok(mut n) = notes.lock() {
        n.push((level.to_string(), text));
    }
}

/// The policy a live Kaze run is allowed to use: exactly the device the operator
/// named, MAVLink in allowlist mode. Fail-closed everywhere else — an unnamed
/// device is still refused by the gate.
pub fn policy_for_device(device: &str) -> J {
    let class = if device.trim().to_ascii_lowercase().starts_with("udp") { "udp" } else { "uart" };
    json!({
        "hardware": {
            "enabled": true,
            "devices": { "mode": "allowlist", "allow": [ { "path": device, "class": class } ] },
            // The only message a Kaze script can make this link emit: the
            // gate checks this list before the policy does, so naming one
            // message is the narrowest door the language needs.
            "mavlink": {
                "mode": "allowlist",
                "send": ["COMMAND_LONG", "SET_POSITION_TARGET_GLOBAL_INT"]
            }
        }
    })
}

/// What a bare live grant resolves to: `--cap drone=real` and
/// `--cap drone=mavlink` both mean the MAVLink link on the endpoint this repo
/// documents for SITL and companion computers.
pub const DEFAULT_MAVLINK_DEVICE: &str = "udp:127.0.0.1:14550";

/// The device a granted backend string names. `sim` never reaches here (the
/// simulator stays in place), and a `udp:`/`uart:` string is its own device.
fn backend_device(device: &str) -> &str {
    match device.trim() {
        "real" | "mavlink" => DEFAULT_MAVLINK_DEVICE,
        other => other,
    }
}

/// `udp:host:port`, `uart:/dev/ttyAMA0@921600`, or a bare device path.
fn open_device(device: &str) -> Result<Box<dyn Io>, String> {
    let trimmed = device.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("udp:") {
        return link::open_io(trimmed).map_err(|e| format!("E_KAZE_LINK: cannot open {}: {}", trimmed, e));
    }
    let rest = trimmed.strip_prefix("uart:").unwrap_or(trimmed);
    let (path, baud) = match rest.rsplit_once('@') {
        Some((p, b)) => (p, b.parse::<u32>().unwrap_or(57_600)),
        None => (rest, 57_600),
    };
    match link::UartIo::open(path, baud) {
        Ok(io) => Ok(Box::new(io)),
        Err(e) => Err(format!("E_KAZE_LINK: cannot open {} at {} baud: {}", path, baud, e)),
    }
}

/// A `CapHost` with selected live hardware and a fail-closed runtime host.
pub struct LiveHost {
    sim: RuntimeHost,
    live: Option<DroneLink>,
    gpio: Option<crate::rt::gpio::GpioBus>,
}

impl LiveHost {
    /// `backends` is the `family -> device` map the host was granted. A missing
    /// live backend does not select a simulator.
    pub fn new(
        sim: SimState,
        backends: &[(String, String)],
        pins: &[(String, u16)],
    ) -> Result<LiveHost, String> {
        let mut live = None;
        for (family, device) in backends {
            if family != "drone" || device.is_empty() || device == "sim" {
                continue;
            }
            live = Some(DroneLink::connect(backend_device(device))?);
        }
        let gpio = if pins.is_empty() {
            None
        } else {
            let root = std::env::var_os("KAZE_GPIO_ROOT").map(std::path::PathBuf::from);
            Some(crate::rt::gpio::GpioBus::new(pins, root.as_deref()))
        };
        Ok(LiveHost { sim: RuntimeHost::new(sim, Vec::new(), Vec::new()), live, gpio })
    }

    /// A host around an already-open link. Tests use it to drive the real
    /// command path without a socket.
    #[cfg(test)]
    pub fn with_link(sim: SimState, link: DroneLink) -> LiveHost {
        LiveHost { sim: RuntimeHost::new(sim, Vec::new(), Vec::new()), live: Some(link), gpio: None }
    }

    /// The GPIO wiring, for `--json` reports and tests.
    pub fn gpio(&self) -> Option<&crate::rt::gpio::GpioBus> {
        self.gpio.as_ref()
    }

    pub fn is_live(&self) -> bool {
        self.live.is_some()
    }

    pub fn device(&self) -> Option<&str> {
        self.live.as_ref().map(|l| l.device.as_str())
    }

    fn drain_notes(&mut self) {
        if let Some(link) = &self.live {
            for (level, text) in link.take_notes() {
                self.sim.logs.push((level, text));
            }
        }
    }
}

impl CapHost for LiveHost {
    fn call(
        &mut self,
        family: &str,
        op: &str,
        args: &[Value],
        arena: &mut Arena,
        heap: &mut Heap,
    ) -> Result<Value, String> {
        if family == "drone" && self.live.is_some() {
            self.sim.check_trace_capacity()?;
            let shown: Vec<String> = args.iter().map(|v| render_value(*v, arena, heap)).collect();
            let mut link = self.live.take().expect("checked above");
            let result = live_drone(&mut link, op, args, arena);
            self.live = Some(link);
            let rendered = match &result {
                Ok(v) => render_value(*v, arena, heap),
                Err(e) => format!("error: {}", e),
            };
            self.sim.note("drone", op, &shown, rendered);
            self.drain_notes();
            return result;
        }
        if family == "device" && self.gpio.is_some() {
            self.sim.check_trace_capacity()?;
            let shown: Vec<String> = args.iter().map(|v| render_value(*v, arena, heap)).collect();
            let mut bus = self.gpio.take().expect("checked above");
            let result = live_device(&mut bus, op, args, arena, heap);
            self.gpio = Some(bus);
            let rendered = match &result {
                Ok(v) => render_value(*v, arena, heap),
                Err(e) => format!("error: {}", e),
            };
            self.sim.note("device", op, &shown, rendered);
            return result;
        }
        self.sim.call(family, op, args, arena, heap)
    }

    fn actions(&self) -> &[crate::sfs::caps::Action] {
        self.sim.actions()
    }

    fn logs(&self) -> &[(String, String)] {
        self.sim.logs()
    }
}

/// Ask the vehicle for the five messages a flight needs. ArduPilot 4.6 removed
/// the `SRx_*` stream parameters, so `MAV_CMD_SET_MESSAGE_INTERVAL` is how a GCS
/// says what it wants to receive. It configures telemetry, never actuation.
fn request_streams(core: &mut LinkCore, io: &mut dyn Io) -> Result<(), String> {
    for (msgid, hz) in [(0u16, 1.0f32), (1, 2.0), (24, 4.0), (30, 4.0), (33, 4.0)] {
        let frame = link::command_long(
            core,
            1,
            1,
            mavlink::cmd::SET_MESSAGE_INTERVAL,
            [msgid as f32, 1_000_000.0 / hz, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let bytes = core.send(&frame, 0, false).map_err(|v| {
            format!("E_KAZE_LINK: cannot request message {}: {} ({})", msgid, v.reason, v.rule)
        })?;
        io.write(&bytes)
            .map_err(|e| format!("E_KAZE_LINK: cannot write to the link: {}", e))?;
    }
    Ok(())
}

/// Block until the transport produces its first bytes (or the deadline passes).
fn wait_for_first_bytes(io: &mut dyn Io, timeout_ms: u64) -> Result<(), ()> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut buf = [0u8; 2048];
    while Instant::now() < deadline {
        match io.read(&mut buf) {
            Ok(0) => thread::sleep(Duration::from_millis(20)),
            Ok(_) => return Ok((),),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(20))
            }
            Err(_) => return Err(()),
        }
    }
    Err(())
}

/// `UdpIo` pins its reply address to the source of the first datagram, so a
/// "connected" link is one the vehicle is already talking to.
/// Wait until the vehicle is talking and has reported a position at least once.
/// The reply address pins on the first datagram, and the gate refuses
/// coordinate-dependent commands until a `GPS_RAW_INT` has arrived inside the
/// last 1.5 s. Warm-up keeps a startup artifact from being reported as a flight
/// problem.
fn warm_up(core: &mut LinkCore, io: &mut dyn Io, timeout_ms: u64) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut buf = [0u8; 2048];
    let mut heard = false;
    let mut positioned = false;
    while Instant::now() < deadline {
        match io.read(&mut buf) {
            Ok(0) => thread::sleep(Duration::from_millis(20)),
            Ok(n) => {
                heard = true;
                for event in core.on_bytes(&buf[..n], 0) {
                    if let Event::Telemetry { message, .. } = event {
                        if message == "GPS_RAW_INT" {
                            positioned = true;
                        }
                    }
                }
                if positioned {
                    return Ok(());
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(20))
            }
            Err(_) => break,
        }
    }
    if !heard {
        Err("nothing is streaming".to_string())
    } else {
        Err("the vehicle is talking but has not reported a position yet".to_string())
    }
}

/// The device map: one Kaze verb, one GPIO line, no guessing.
fn live_device(
    bus: &mut crate::rt::gpio::GpioBus,
    op: &str,
    args: &[Value],
    arena: &mut Arena,
    heap: &mut Heap,
) -> Result<Value, String> {
    let name_of = |args: &[Value]| -> String {
        args.first()
            .map(|v| render_value(*v, arena, heap))
            .unwrap_or_default()
            .trim_matches('"')
            .to_string()
    };
    match op {
        "set_relay" => {
            let name = name_of(args);
            let on = match args.get(1) {
                Some(Value::Bool(b)) => *b,
                Some(v) => v.as_num().map(|n| n != 0.0).unwrap_or(false),
                None => false,
            };
            bus.set(&name, on)?;
            Ok(Value::Unit)
        }
        "pulse" => {
            let name = name_of(args);
            let seconds = args.get(1).and_then(|v| v.as_num()).unwrap_or(1.0);
            bus.pulse(&name, seconds)?;
            Ok(Value::Unit)
        }
        "read_gpio" => {
            let name = name_of(args);
            Ok(Value::Bool(bus.read(&name)?))
        }
        // The GPIO bus is on/off only: there is no pwmchip backend in this build,
        // so a duty cycle cannot be written. Refusing is the honest answer — a
        // script that believes it set 40% would be acting on a lie.
        "set_duty" => Err(format!(
            "E_KAZE_LINK: the GPIO bus has no PWM channel for '{}' in this build; \
             set_duty is simulated, not actuated",
            name_of(args)
        )),
        other => Err(format!("E_KAZE_LINK: the live device bus has no '{}'", other)),
    }
}

/// Guided mode is a precondition, not a nicety: ArduCopter refuses NAV_TAKEOFF
/// and NAV_WAYPOINT anywhere else. Switching is a gated command like any other,
/// and it costs one frame only when the vehicle is somewhere else.
fn ensure_guided(link: &mut DroneLink) -> Result<(), String> {
    if link.telemetry().fresh_mode()?.eq_ignore_ascii_case("guided") {
        return Ok(());
    }
    let guided = crate::rt::drone_ops::copter_mode("guided")
        .ok_or_else(|| "E_KAZE_LINK: this build has no GUIDED mode for the vehicle".to_string())?;
    link.command(
        mavlink::cmd::DO_SET_MODE,
        // MAV_CMD_DO_SET_MODE: param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        // param2 = the autopilot's own mode number (GUIDED is 4 in ArduCopter).
        [1.0, guided as f32, 0.0, 0.0, 0.0, 0.0, 0.0],
    )?;
    // ACCEPTED means "queued", not "in effect": ArduCopter applies the mode on
    // its next loop, and a takeoff one tick early is refused. Wait for the
    // vehicle to say it is in GUIDED before the flight command goes out.
    for _ in 0..40 {
        if link.telemetry().fresh_mode().is_ok_and(|mode| mode.eq_ignore_ascii_case("guided")) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err("E_KAZE_LINK: the vehicle accepted GUIDED but is not reporting it".to_string())
}

/// The v0.2 command map: one Kaze verb, one MAV_CMD, through the gate.
fn live_drone(link: &mut DroneLink, op: &str, args: &[Value], arena: &mut Arena) -> Result<Value, String> {
    let num = |i: usize| args.get(i).and_then(|v| v.as_num()).filter(|n| n.is_finite())
        .ok_or_else(|| "E_KAZE_RANGE: drone command requires finite numeric arguments".to_string());
    match op {
        "arm" => link
            .command(mavlink::cmd::COMPONENT_ARM_DISARM, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .map(|_| Value::Unit),
        "takeoff" => {
            let alt = num(0)?;
            ensure_guided(link)?;
            link.command(mavlink::cmd::NAV_TAKEOFF, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, alt as f32])
                .map(|_| Value::Unit)
        }
        // A live position target requires three explicit coordinates.
        "goto" => {
            // There is no verified mapping from a Kaze point/index to a GPS
            // position. Never fill in missing coordinates from cached data.
            if args.len() != 3 {
                return Err("E_CAP_MISSING: live drone.goto requires explicit latitude, longitude and altitude".into());
            }
            let (lat, lon, alt) = (num(0)?, num(1)?, num(2)?);
            ensure_guided(link)?;
            // ArduCopter answers UNSUPPORTED to NAV_WAYPOINT as a command (it
            // is a mission item, not a command), so guided flight is a position
            // target — the same frame `sofuu.drone` uses.
            let frame = link.position_frame(lat, lon, alt as f32);
            link.send_target(frame).map(|_| Value::Unit)
        }
        "rtl" => link
            .command(mavlink::cmd::NAV_RETURN_TO_LAUNCH, [0.0; 7])
            .map(|_| Value::Unit),
        "land" => link.command(mavlink::cmd::NAV_LAND, [0.0; 7]).map(|_| Value::Unit),
        "hold" | "stop" => link
            .command(mavlink::cmd::NAV_LOITER_UNLIM, [0.0; 7])
            .map(|_| Value::Unit),
        "telemetry" => {
            let field = args.first().map(|v| render_value(*v, arena, &Heap::empty())).unwrap_or_default();
            let field = field.trim_matches('"').to_string();
            link.telemetry().read(&field)
        }
        other => Err(format!("E_KAZE_LINK: the live drone link has no '{}'", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io;

    const DEV: &str = "udp:127.0.0.1:14550";

    /// A flight controller in two buffers: it answers every COMMAND_LONG with a
    /// COMMAND_ACK, and streams whatever telemetry the test queues for it.
    /// Everything goes through the real encoder and parser, so the test covers
    /// the wire format, not a mock of it.
    struct AutoFc {
        written: Arc<Mutex<Vec<u8>>>,
        incoming: Arc<Mutex<VecDeque<u8>>>,
        parser: mavlink::Parser,
        result: u8,
    }

    impl AutoFc {
        fn new(result: u8) -> (AutoFc, Arc<Mutex<Vec<u8>>>, Arc<Mutex<VecDeque<u8>>>) {
            let written = Arc::new(Mutex::new(Vec::new()));
            let incoming = Arc::new(Mutex::new(VecDeque::new()));
            (
                AutoFc {
                    written: Arc::clone(&written),
                    incoming: Arc::clone(&incoming),
                    parser: mavlink::Parser::new(),
                    result,
                },
                written,
                incoming,
            )
        }
    }

    impl Io for AutoFc {
        fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.written.lock().unwrap().extend_from_slice(bytes);
            self.parser.push(bytes);
            while let Some(frame) = self.parser.next() {
                if frame.msgid == 76 {
                    // COMMAND_LONG: param1..7 (f32) first, then command (u16).
                    let command = frame.u16_at(28);
                    let mut ack = mavlink::Frame::new(77, 0, 1, 1).expect("COMMAND_ACK is known");
                    ack.set_u16_at(0, command);
                    ack.set_u8_at(2, self.result);
                    let wire = mavlink::encode(&ack).expect("a valid ack encodes");
                    self.incoming.lock().unwrap().extend(wire);
                }
            }
            Ok(())
        }

        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let mut queue = self.incoming.lock().unwrap();
            let n = queue.len().min(buf.len());
            for slot in buf.iter_mut().take(n) {
                *slot = queue.pop_front().expect("n bytes were counted");
            }
            Ok(n)
        }
    }

    /// A transport that never answers.
    struct Silent {
        written: Arc<Mutex<Vec<u8>>>,
    }

    impl Io for Silent {
        fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.written.lock().unwrap().extend_from_slice(bytes);
            Ok(())
        }
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Ok(0)
        }
    }

    fn core() -> LinkCore {
        LinkCore::new(DEV, HardwarePolicy::from_policy(&policy_for_device(DEV)))
    }

    fn commands_on_the_wire(bytes: &[u8]) -> Vec<u16> {
        let mut parser = mavlink::Parser::new();
        parser.push(bytes);
        let mut seen = Vec::new();
        while let Some(frame) = parser.next() {
            if frame.msgid == 76 {
                seen.push(frame.u16_at(28));
            }
        }
        seen
    }

    #[test]
    fn an_arm_command_reaches_the_wire_and_waits_for_the_ack() {
        let (fc, written, _incoming) = AutoFc::new(0);
        let link = DroneLink::with_io(core(), Box::new(fc), DEV.to_string(), 1, 500).expect("link");
        link.command(mavlink::cmd::COMPONENT_ARM_DISARM, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .expect("the autopilot accepted the arm");
        let bytes = written.lock().unwrap().clone();
        assert!(
            commands_on_the_wire(&bytes).contains(&mavlink::cmd::COMPONENT_ARM_DISARM),
            "the wire carried: {:?}",
            commands_on_the_wire(&bytes)
        );
    }

    #[test]
    fn a_rejected_command_surfaces_the_autopilots_answer() {
        let (fc, _written, _incoming) = AutoFc::new(2); // MAV_RESULT::DENIED
        let link = DroneLink::with_io(core(), Box::new(fc), DEV.to_string(), 1, 500).expect("link");
        let err = link
            .command(mavlink::cmd::NAV_TAKEOFF, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0])
            .expect_err("a DENIED ack is a failure");
        assert!(err.starts_with("E_COMMAND_REJECTED:"), "{err}");
        assert!(err.contains("DENIED"), "{err}");
    }

    #[test]
    fn an_unanswered_command_times_out_instead_of_hanging() {
        let written = Arc::new(Mutex::new(Vec::new()));
        let silent = Silent { written: Arc::clone(&written) };
        let link = DroneLink::with_io(core(), Box::new(silent), DEV.to_string(), 1, 120).expect("link");
        let started = Instant::now();
        let err = link
            .command(mavlink::cmd::NAV_LAND, [0.0; 7])
            .expect_err("nothing answered");
        assert!(err.starts_with("E_KAZE_LINK_TIMEOUT:"), "{err}");
        assert!(err.contains("unknown"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(3), "the wait was bounded");
    }

    #[test]
    fn telemetry_comes_from_the_vehicle_not_from_the_script() {
        let (fc, _written, incoming) = AutoFc::new(0);
        // One GLOBAL_POSITION_INT: 12.5 m above home, 47.6 N / 8.5 E.
        let mut frame = mavlink::Frame::new(33, 0, 1, 1).expect("GLOBAL_POSITION_INT is known");
        frame.set_i32_at(4, 476_000_000);
        frame.set_i32_at(8, 85_000_000);
        frame.set_i32_at(12, 512_500);
        frame.set_i32_at(16, 12_500);
        let wire = mavlink::encode(&frame).expect("encodes");
        let heartbeat = mavlink::encode(&mavlink::msg_heartbeat(0, 1, 1)).expect("heartbeat encodes");
        incoming.lock().unwrap().extend(heartbeat.into_iter().chain(wire));

        let link = DroneLink::with_io(core(), Box::new(fc), DEV.to_string(), 1, 500).expect("link");
        let mut tele = DroneTelemetry::default();
        for _ in 0..50 {
            tele = link.telemetry();
            if tele.rel_alt_m > 0.0 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(tele.rel_alt_m, 12.5);
        assert_eq!(tele.lat, 47.6);
        assert_eq!(tele.lon, 8.5);
        assert_eq!(tele.read("altitude").unwrap(), Value::Num(12.5));
    }

    #[test]
    fn telemetry_rejects_unknown_missing_stale_and_sentinel_values() {
        // Static protocol data exercises the decoder/cache rules; it is not a
        // receipt for a real hardware connection or a physical simulation.
        let mut tele = DroneTelemetry::default();
        assert!(tele.read("not_a_field").unwrap_err().starts_with("E_CAP_MISSING:"));
        assert!(tele.read("battery").unwrap_err().contains("missing or stale"));
        tele.apply(&Event::Telemetry { message: "HEARTBEAT", data: serde_json::json!({"base_mode": 128, "mode": "guided"}) });
        assert_eq!(tele.read("armed").unwrap(), Value::Bool(true));
        assert!(tele.read("battery").unwrap_err().contains("missing or stale"));
        tele.apply(&Event::Telemetry { message: "SYS_STATUS", data: serde_json::json!({"battery_remaining_pct": 42}) });
        assert_eq!(tele.read("battery").unwrap(), Value::Num(42.0));
        tele.apply(&Event::Telemetry { message: "SYS_STATUS", data: serde_json::json!({"battery_remaining_pct": -1}) });
        assert!(tele.read("battery").is_err());
        tele.apply(&Event::Telemetry { message: "GPS_RAW_INT", data: serde_json::json!({"satellites": 255}) });
        assert!(tele.read("satellites").is_err());
        tele.apply(&Event::Telemetry { message: "GLOBAL_POSITION_INT", data: serde_json::json!({"lat": 47.6, "lon": 8.5, "heading_deg": 655.35}) });
        assert_eq!(tele.read("latitude").unwrap(), Value::Num(47.6));
        assert!(tele.read("heading").is_err());
        assert!(tele.read("altitude").is_err());
        tele.observed.insert("lat", Instant::now() - TELEMETRY_MAX_AGE - Duration::from_secs(1));
        assert!(tele.read("lat").unwrap_err().contains("missing or stale"));
        tele.observed.insert("heartbeat", Instant::now() - TELEMETRY_MAX_AGE - Duration::from_secs(1));
        assert!(tele.read("armed").unwrap_err().contains("heartbeat"));
        assert!(tele.fresh_mode().is_err());
    }

    #[test]
    fn a_live_host_routes_the_drone_family_and_keeps_the_simulator_for_the_rest() {
        let (fc, _written, _incoming) = AutoFc::new(0);
        let link = DroneLink::with_io(core(), Box::new(fc), DEV.to_string(), 1, 500).expect("link");
        let mut host = LiveHost::with_link(SimState::default(), link);
        let mut arena = Arena::new(256 * 1024);
        let mut heap = Heap::default();
        host.call("drone", "arm", &[], &mut arena, &mut heap)
            .expect("arm is answered by the live link");
        host.call("log", "info", &[Value::Num(1.0)], &mut arena, &mut heap)
            .expect("log stays in-process");
        assert!(host.is_live());
        assert_eq!(host.actions().len(), 2, "both calls are in one trace");
    }
}

