//! The MAVLink link — protocol logic, the one gated write path, and the
//! dead-man heartbeat (PLAN-DRONES-MAVLINK §4.2, §4.3, §4.5).
//!
//! The split here is deliberate:
//!
//! * [`LinkCore`] is **pure protocol**. It takes bytes and a clock and returns
//!   events; it opens no sockets and spawns no threads. Every rule — the gate,
//!   sequence numbers, heartbeat cadence, ACK correlation, PreArm diagnosis —
//!   is unit-testable without hardware, which is §6's testing strategy.
//! * The `Io` backends below move bytes. `UdpIo` is a non-blocking socket;
//!   `UartIo` is a termios raw device. A driver (the CLI, or the engine's
//!   uv_udp handle) calls `pump()`.
//!
//! Two invariants that the doctine requires and the code enforces:
//!
//! 1. **No frame exists before the gate has judged it.** `LinkCore::send` is
//!    the only function that can produce outgoing bytes, and it calls
//!    `mavlink_gate::gate_send` first. There is no `send_raw`.
//! 2. **The heartbeat is runtime-scheduled, never agent-driven** (§4.5). Agents
//!    ask for `HEARTBEAT` at most; `tick()` is what actually emits it at 1 Hz.

use std::collections::{HashMap, VecDeque};

use serde_json::{json, Value};

use super::mavlink::{self, Class, Frame, MsgSpec, Parser};
use super::mavlink_gate::{self as gate, GateState, HardwarePolicy, SendRequest, Verdict};

/// MAV_RESULT values carried by COMMAND_ACK.
pub mod mav_result {
    pub const ACCEPTED: u8 = 0;
    pub const TEMPORARILY_REJECTED: u8 = 1;
    pub const DENIED: u8 = 2;
    pub const UNSUPPORTED: u8 = 3;
    pub const FAILED: u8 = 4;
    pub const IN_PROGRESS: u8 = 5;
    pub const CANCELLED: u8 = 6;

    pub fn name(v: u8) -> &'static str {
        match v {
            ACCEPTED => "ACCEPTED",
            TEMPORARILY_REJECTED => "TEMPORARILY_REJECTED",
            DENIED => "DENIED",
            UNSUPPORTED => "UNSUPPORTED",
            FAILED => "FAILED",
            IN_PROGRESS => "IN_PROGRESS",
            CANCELLED => "CANCELLED",
            _ => "UNKNOWN",
        }
    }
}

/// The §1.4 code for an arming refusal, with ArduPilot's own PreArm text.
pub const E_ARMING_DENIED: &str = "E_ARMING_DENIED";
/// How long after a failed arm a STATUSTEXT still explains it (§4.1).
pub const PREARM_WINDOW_MS: u64 = 3_000;
/// Default heartbeat cadence — the dead-man contract (§4.5).
pub const HEARTBEAT_PERIOD_MS: u64 = 1_000;
/// How long the pinned UDP peer may stay silent before a different source that
/// is actively sending may take over the reply address.
pub const PEER_MIGRATE_AFTER_MS: u64 = 5_000;

/// Something the link produced. Drained by the driver and forwarded to JS /
/// the event log / `.qtsq`.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A received telemetry frame, decoded to a JSON object.
    Telemetry { message: &'static str, data: Value },
    /// The FC answered a gated command.
    CommandAck { command: u16, command_name: &'static str, result: u8, result_name: &'static str, text: Option<String> },
    /// An arming refusal, with the PreArm reason attached when one arrived.
    ArmingDenied { command: u16, text: String },
    /// STATUSTEXT, verbatim (severity + text).
    StatusText { severity: u8, text: String },
    /// The link had to discard bytes (EMI / corruption / unsupported message).
    Resync { discarded_bytes: u64, crc_errors: u64 },
}

/// A command waiting for its COMMAND_ACK.
#[derive(Clone, Debug)]
pub struct PendingCommand {
    pub command: u16,
    pub sent_at_ms: u64,
}

/// The pure protocol core.
#[derive(Debug)]
pub struct LinkCore {
    pub device: String,
    pub policy: HardwarePolicy,
    pub state: GateState,
    pub parser: Parser,
    pub seq: u8,
    pub sysid: u8,
    pub compid: u8,
    pub heartbeat_period_ms: u64,
    pub last_heartbeat_ms: Option<u64>,
    /// Commands awaiting an ACK, keyed by MAV_CMD.
    pub pending: HashMap<u16, PendingCommand>,
    /// Recent STATUSTEXTs, for the PreArm correlation (§4.1).
    pub statustexts: VecDeque<(u64, u8, String)>,
    /// Every gate decision, as WAL-ready records. The driver drains this.
    pub audit: Vec<Value>,
    /// Discards already reported, so routine skipping of message types this
    /// link does not decode does not emit one Resync per pump (a real
    /// autopilot streams ~40 of them a second).
    pub reported_discards: u64,
}

impl LinkCore {
    pub fn new(device: impl Into<String>, policy: HardwarePolicy) -> LinkCore {
        LinkCore {
            device: device.into(),
            policy,
            state: GateState::default(),
            parser: Parser::new(),
            seq: 0,
            sysid: 255,
            compid: 191, // MAV_COMP_ID_ONBOARD_COMPUTER
            heartbeat_period_ms: HEARTBEAT_PERIOD_MS,
            last_heartbeat_ms: None,
            pending: HashMap::new(),
            statustexts: VecDeque::new(),
            audit: Vec::new(),
            reported_discards: 0,
        }
    }

    fn next_seq(&mut self) -> u8 {
        let s = self.seq;
        self.seq = self.seq.wrapping_add(1);
        s
    }

    /// Can this device be opened at all? (The same resolver the send path uses.)
    pub fn check_device(&self) -> Verdict {
        self.policy.gate_device(&self.device)
    }

    /// Feed received bytes. Returns the events they produced, in order.
    ///
    /// The receive path is **not** gated (read-only sensors, §4.3) but it is
    /// what feeds the GPS/fail-closed state and the audit record of the
    /// vehicle's own reports.
    pub fn on_bytes(&mut self, bytes: &[u8], now_ms: u64) -> Vec<Event> {
        self.parser.push(bytes);
        let before = self.parser.stats();
        let mut out = Vec::new();
        while let Some(frame) = self.parser.next() {
            self.state.observe(&frame, now_ms);
            if let Some(ev) = self.event_for(&frame, now_ms) {
                out.push(ev);
            }
        }
        let after = self.parser.stats();
        /* Report a resync for the two things that mean something went wrong: a
         * CRC failure (corruption), or a burst of discards larger than any
         * single frame (a hole in the stream). Skipping the message types this
         * link does not decode is normal and stays silent — a real autopilot
         * streams dozens of them per second, and one Resync per pump buried
         * every telemetry line. */
        let new_crc_errors = after.crc_errors > before.crc_errors;
        let burst = after.discarded_bytes.saturating_sub(self.reported_discards) > 256;
        if new_crc_errors || burst {
            self.reported_discards = after.discarded_bytes;
            out.push(Event::Resync {
                discarded_bytes: after.discarded_bytes,
                crc_errors: after.crc_errors,
            });
        }
        out
    }

    fn event_for(&mut self, frame: &Frame, now_ms: u64) -> Option<Event> {
        let spec: &'static MsgSpec = frame.spec()?;
        match spec.id {
            253 => {
                let text = statustext_of(frame);
                self.statustexts.push_back((now_ms, frame.u8_at(0), text.clone()));
                while self.statustexts.len() > 32 {
                    self.statustexts.pop_front();
                }
                Some(Event::StatusText { severity: frame.u8_at(0), text })
            }
            77 => {
                let command = frame.u16_at(0);
                let result = frame.u8_at(2);
                self.pending.remove(&command);
                let command_name = gate::command_name(command).unwrap_or("MAV_CMD_UNKNOWN");
                let text = if result == mav_result::FAILED || result == mav_result::DENIED {
                    self.prearm_reason(now_ms)
                } else {
                    None
                };
                Some(Event::CommandAck {
                    command,
                    command_name,
                    result,
                    result_name: mav_result::name(result),
                    text,
                })
            }
            _ => Some(Event::Telemetry { message: spec.name, data: telemetry_json(frame) }),
        }
    }

    /// The ArduPilot PreArm text that explains a failed arm, if one arrived
    /// close enough (§4.1: `E_ARMING_DENIED: PreArm: High GPS HDOP (…)`).
    fn prearm_reason(&self, now_ms: u64) -> Option<String> {
        self.statustexts
            .iter()
            .rev()
            .find(|(t, sev, _)| *sev <= 4 && now_ms.saturating_sub(*t) <= PREARM_WINDOW_MS)
            .map(|(_, _, text)| text.clone())
    }

    /// Emit the heartbeat if it is due. Called from the driver's timer — never
    /// from agent code (§4.5).
    ///
    /// A heartbeat refused by the gate is *not* an error worth raising: if the
    /// policy denies it, the FC's GCS failsafe firing is the correct,
    /// documented consequence (§4.5).
    pub fn tick(&mut self, now_ms: u64) -> Option<Vec<u8>> {
        let due = match self.last_heartbeat_ms {
            Some(t) => now_ms.saturating_sub(t) >= self.heartbeat_period_ms,
            None => true,
        };
        if !due {
            return None;
        }
        self.last_heartbeat_ms = Some(now_ms);
        let seq = self.next_seq();
        let hb = mavlink::msg_heartbeat(seq, self.sysid, self.compid);
        // HEARTBEAT is not actuator-bearing, so no approval/ack bookkeeping.
        match gate::gate_send(
            &self.policy,
            &mut self.state,
            &SendRequest { device: &self.device, frame: &hb },
            now_ms,
            false,
        ) {
            v if v.allowed => {
                self.audit.push(v.audit_record(&self.device, "HEARTBEAT", now_ms));
                mavlink::encode(&hb).ok()
            }
            v => {
                self.audit.push(v.audit_record(&self.device, "HEARTBEAT", now_ms));
                None
            }
        }
    }

    /// Judge and encode one outgoing frame. `Ok(bytes)` is the ONLY way bytes
    /// reach a transport; `Err(verdict)` means nothing was written.
    pub fn send(&mut self, frame: &Frame, now_ms: u64, approved: bool) -> Result<Vec<u8>, Verdict> {
        let verdict = gate::gate_send(
            &self.policy,
            &mut self.state,
            &SendRequest { device: &self.device, frame },
            now_ms,
            approved,
        );
        let name = frame.name();
        self.audit.push(verdict.audit_record(&self.device, name, now_ms));
        if !verdict.allowed {
            return Err(verdict);
        }
        if let Some(cmd) = gate::command_of(frame) {
            self.pending.insert(cmd, PendingCommand { command: cmd, sent_at_ms: now_ms });
        }
        let mut out = frame.clone();
        out.seq = self.next_seq();
        mavlink::encode(&out).map_err(|e| {
            Verdict::deny(gate::E_SAFETY_VIOLATION, "R-HW/encode", format!("frame encode failed: {e}"))
        })
    }

    /// Drain the audit records for the driver to append to the WAL.
    pub fn take_audit(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.audit)
    }
}

/// STATUSTEXT payload: severity(0), text char[50](1..51), id(51), chunk_seq(53).
/// Newer ArduPilot sends chunked long text; reassembly is out of v1 scope and
/// documented as such.
pub fn statustext_of(frame: &Frame) -> String {
    let raw = &frame.payload[1..51.min(frame.payload.len())];
    let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).trim().to_string()
}

/// Decode a telemetry payload into the JSON shape JS and `.qtsq` consume.
/// Units are the ones a pilot reads (metres, degrees, volts), not raw ints.
pub fn telemetry_json(frame: &Frame) -> Value {
    match frame.msgid {
        0 => json!({
            "type": frame.u8_at(4),
            "autopilot": frame.u8_at(5),
            "base_mode": frame.u8_at(6),
            "system_status": frame.u8_at(7),
            "mavlink_version": frame.u8_at(8),
            /* `custom_mode` is the ONLY field that says which flight mode the
             * vehicle is really in: ArduPilot sets several base_mode capability
             * bits at once, so GUIDED and STABILIZE share their bit pattern.
             * Without this, "the mode switch was accepted but the vehicle
             * ignored my position target" is undiagnosable from telemetry. */
            "custom_mode": frame.u32_at(0),
            "mode": crate::rt::drone_ops::copter_mode_name((frame.u32_at(0) & 0xFF) as u8),
        }),
        /* Wire order is the type-size sort in the definition, NOT the XML order:
           three u32 flags, then nine u16 fields (load, voltage, current_battery,
           drop_rate_comm, errors_comm, errors_count1..4), and only then the u8s.
           `battery_remaining` read straight after `current_battery` lands on
           byte 18 = drop_rate_comm's low byte → a permanently-zero battery. */
        1 => json!({
            "voltage_v": frame.u16_at(14) as f64 / 1000.0,
            "current_a": frame.i16_at(16) as f64 / 100.0,
            "battery_remaining_pct": frame.u8_at(30) as i8,
            "load_pct": frame.u16_at(12) as f64 / 10.0,
        }),
        24 => json!({
            "fix_type": frame.u8_at(28),
            "satellites": frame.u8_at(29),
            "hdop": frame.u16_at(20) as f64 / 100.0,
            "lat": frame.i32_at(8) as f64 / 1e7,
            "lon": frame.i32_at(12) as f64 / 1e7,
            "alt_m": frame.i32_at(16) as f64 / 1000.0,
        }),
        30 => json!({
            "roll_deg": (frame.f32_at(4) as f64).to_degrees(),
            "pitch_deg": (frame.f32_at(8) as f64).to_degrees(),
            "yaw_deg": (frame.f32_at(12) as f64).to_degrees(),
            "roll_rate_ds": (frame.f32_at(16) as f64).to_degrees(),
            "pitch_rate_ds": (frame.f32_at(20) as f64).to_degrees(),
            "yaw_rate_ds": (frame.f32_at(24) as f64).to_degrees(),
        }),
        33 => json!({
            "lat": frame.i32_at(4) as f64 / 1e7,
            "lon": frame.i32_at(8) as f64 / 1e7,
            "alt_m": frame.i32_at(12) as f64 / 1000.0,
            "relative_alt_m": frame.i32_at(16) as f64 / 1000.0,
            "vx_ms": frame.i16_at(20) as f64 / 100.0,
            "vy_ms": frame.i16_at(22) as f64 / 100.0,
            "vz_ms": frame.i16_at(24) as f64 / 100.0,
            "heading_deg": frame.u16_at(26) as f64 / 100.0,
        }),
        /* Same reordering trap, worse: the 18 u16 channels come first (bytes
           4..40), so `chancount` is at 40 and `rssi` at 41. Read at 4 it
           reports the low byte of chan1_raw — SITL's 1500 gave chancount 220. */
        65 => {
            let count = frame.u8_at(40) as usize;
            let mut ch = Vec::with_capacity(count.min(18));
            for i in 0..count.min(18) {
                ch.push(frame.u16_at(4 + i * 2));
            }
            json!({ "chancount": count, "channels": ch, "rssi": frame.u8_at(41) })
        }
        77 => json!({
            "command": frame.u16_at(0),
            "result": frame.u8_at(2),
            "result_name": mav_result::name(frame.u8_at(2)),
        }),
        _ => json!({}),
    }
}

/* ── IO backends ──────────────────────────────────────────────────────── */

/// A byte pipe. Implementations are deliberately dumb.
///
/// `Send` because the JS surface keeps links in a process-wide registry behind
/// a mutex (`rt/mavlink_js.rs`); the sockets and fds are plain owned handles.
pub trait Io: Send {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<()>;
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize>;
    /// Transport-level counters for `status`. A second sender on the endpoint,
    /// or a reply address that migrated, is a fact the operator should be able
    /// to see without reaching for tcpdump.
    fn transport_stats(&self) -> Value {
        Value::Null
    }
}

/// A MAVLink UDP endpoint.
///
/// Semantics match ArduPilot's `--out=udp:<HOST>:<PORT>` topology the plan
/// describes (§5: *SITL on the Mac, Sofuu on the Pi 5, `--out=udp:<PI5_IP>:14540`*):
/// the device string names **Sofuu's own endpoint**, so we BIND it and receive
/// there. Replies go to the pinned peer — the source of the first datagram.
///
/// The pin has two escape hatches, both learned the hard way on the first
/// live SITL run:
///
/// * **Every datagram is parsed, whatever its source.** Dropping packets from
///   a second source meant the first router misconfiguration cost half the
///   telemetry with no diagnostic at all; the counters below make it visible
///   instead. Only the *reply* address is pinned, and transmitting is still
///   gated frame by frame.
/// * **A silent peer yields its pin.** MAVProxy binds a fresh ephemeral port
///   when it restarts, so a permanently-sticky peer left the companion deaf
///   forever. After [`PEER_MIGRATE_AFTER_MS`] of silence, an active source
///   takes over the reply address and `peer_changes` records it.
pub struct UdpIo {
    sock: std::net::UdpSocket,
    peer: Option<std::net::SocketAddr>,
    peer_seen: Option<std::time::Instant>,
    /// Datagrams received from a source other than the pinned peer.
    pub foreign_datagrams: u64,
    /// Times the reply address migrated.
    pub peer_changes: u64,
}

/// Parse `udp:host:port` / `udpout:host:port` into (scheme, host, port).
pub fn parse_endpoint(device: &str) -> Option<(&'static str, String, u16)> {
    let s = device.trim();
    let lower = s.to_ascii_lowercase();
    let (scheme, rest) = if lower.starts_with("udpout:") {
        ("udpout", &s[7..])
    } else if lower.starts_with("udp:") {
        ("udp", &s[4..])
    } else {
        return None;
    };
    let (host, port) = rest.rsplit_once(':')?;
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    let port: u16 = port.trim().parse().ok()?;
    if host.is_empty() {
        return None;
    }
    Some((scheme, host.to_string(), port))
}

/// Parse a `udp:` endpoint (host, port). Kept for callers that only care about
/// the bind form.
pub fn parse_udp(device: &str) -> Option<(String, u16)> {
    let (_, host, port) = parse_endpoint(device)?;
    Some((host, port))
}

impl UdpIo {
    /// Bind the endpoint named by `device`.
    ///
    /// `udp:host:port`    — bind that endpoint and learn the peer from the
    ///                      first datagram (SITL's `--out` stream; the FC
    ///                      speaks first).
    /// `udpout:host:port` — bind an ephemeral local port and send to that fixed
    ///                      address immediately (a companion that must speak
    ///                      first; without this the link deadlocks, because a
    ///                      silent FC never teaches us a peer).
    pub fn connect(device: &str) -> std::io::Result<UdpIo> {
        let (scheme, host, port) = parse_endpoint(device)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "not a udp endpoint"))?;
        if scheme == "udpout" {
            use std::net::ToSocketAddrs;
            let peer = format!("{host}:{port}").to_socket_addrs()?.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "no address resolved")
            })?;
            let bind = if peer.is_ipv4() { "0.0.0.0" } else { "[::]" };
            let sock = std::net::UdpSocket::bind((bind, 0))?;
            sock.set_nonblocking(true)?;
            return Ok(UdpIo {
                sock,
                peer: Some(peer),
                peer_seen: Some(std::time::Instant::now()),
                foreign_datagrams: 0,
                peer_changes: 0,
            });
        }
        // A literal IP binds that interface; a name binds wildcard and is
        // enforced by the first-sender pin instead.
        let bind_host = if host.parse::<std::net::IpAddr>().is_ok() {
            host.clone()
        } else {
            "0.0.0.0".to_string()
        };
        let sock = std::net::UdpSocket::bind((bind_host.as_str(), port))?;
        sock.set_nonblocking(true)?;
        Ok(UdpIo {
            sock,
            peer: None,
            peer_seen: None,
            foreign_datagrams: 0,
            peer_changes: 0,
        })
    }

    pub fn local_addr(&self) -> Option<std::net::SocketAddr> {
        self.sock.local_addr().ok()
    }

    pub fn peer(&self) -> Option<std::net::SocketAddr> {
        self.peer
    }
}

impl Io for UdpIo {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let Some(peer) = self.peer else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "no peer yet — no datagram has been received on this endpoint",
            ));
        };
        self.sock.send_to(bytes, peer)?;
        Ok(())
    }
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.sock.recv_from(buf) {
            Ok((n, from)) => {
                let now = std::time::Instant::now();
                match self.peer {
                    None => {
                        self.peer = Some(from);
                        self.peer_seen = Some(now);
                    }
                    Some(peer) if peer == from => self.peer_seen = Some(now),
                    Some(_) => {
                        /* Two sources on one port is a misconfiguration (two
                         * routers, or a GCS forwarding to us). The bytes are
                         * real telemetry, so they are returned — the old code
                         * threw them away and the only symptom was a resync
                         * storm that hid half the stream. Only the reply
                         * address stays pinned, and it yields once the pinned
                         * peer has gone quiet for long enough that it is gone
                         * (MAVProxy rebinds an ephemeral port on restart). */
                        self.foreign_datagrams += 1;
                        let stale = self
                            .peer_seen
                            .map(|t| t.elapsed().as_millis() as u64 >= PEER_MIGRATE_AFTER_MS)
                            .unwrap_or(true);
                        if stale {
                            self.peer = Some(from);
                            self.peer_changes += 1;
                        }
                    }
                }
                Ok(n)
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(0),
            Err(e) => Err(e),
        }
    }
    fn transport_stats(&self) -> Value {
        json!({
            "peer": self.peer.map(|p| p.to_string()),
            "foreign_datagrams": self.foreign_datagrams,
            "peer_changes": self.peer_changes,
        })
    }
}

/// UART in raw mode at a fixed baud, via `libc` termios (no serialport crate).
pub struct UartIo {
    fd: std::os::unix::io::RawFd,
}

impl UartIo {
    pub fn open(path: &str, baud: u32) -> std::io::Result<UartIo> {
        use std::os::unix::io::IntoRawFd;
        let f = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
        let fd = f.into_raw_fd();
        Self::adopt(fd, baud)
    }

    /// Configure an already-open descriptor. Used by `open`, and by the pty-pair
    /// test that stands in for `/dev/ttyAMA0` without hardware (§6.2).
    pub fn adopt(fd: std::os::unix::io::RawFd, baud: u32) -> std::io::Result<UartIo> {
        // SAFETY: fd is an open r+w descriptor owned by the caller (moved here).
        let rc = unsafe { configure_termios(fd, baud) };
        if rc != 0 {
            // SAFETY: we own fd and are abandoning it.
            unsafe { libc::close(fd) };
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "termios configuration failed",
            ));
        }
        Ok(UartIo { fd })
    }
}

/// Put `fd` into 8N1 raw mode at `baud`, non-blocking, no flow control.
/// Returns 0 on success.
///
/// SAFETY: `fd` must be an open, readable+writable tty descriptor.
unsafe fn configure_termios(fd: std::os::unix::io::RawFd, baud: u32) -> i32 {
    use std::mem::MaybeUninit;
    let mut t = MaybeUninit::<libc::termios>::uninit();
    if unsafe { libc::tcgetattr(fd, t.as_mut_ptr()) } != 0 {
        return -1;
    }
    let mut t = unsafe { t.assume_init() };

    let speed = match baud {
        9600 => libc::B9600,
        19200 => libc::B19200,
        38400 => libc::B38400,
        57600 => libc::B57600,
        115200 => libc::B115200,
        230400 => libc::B230400,
        /* 460800/921600 have no termios constant on macOS (they need
         * IOSSIOSPEED); Linux carries B460800/B921600. Falling back to 57600
         * rather than silently running at the wrong rate is the honest choice —
         * a mis-set baud on a flight controller link reads as random noise. */
        _ => libc::B57600,
    };
    if unsafe { libc::cfsetispeed(&mut t, speed) } != 0
        || unsafe { libc::cfsetospeed(&mut t, speed) } != 0
    {
        return -1;
    }
    // raw: no echo, no line discipline, no software/hardware flow control.
    unsafe { libc::cfmakeraw(&mut t) };
    t.c_cflag &= !libc::CSTOPB; // 1 stop bit
    t.c_cflag &= !libc::PARENB; // no parity
    t.c_cflag &= !libc::CRTSCTS;
    t.c_cflag |= libc::CLOCAL | libc::CREAD;
    t.c_cc[libc::VMIN] = 0;
    t.c_cc[libc::VTIME] = 0;
    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &t) } != 0 {
        return -1;
    }
    // Non-blocking reads so a pump() never parks the loop thread.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return -1;
    }
    0
}

impl Io for UartIo {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let mut off = 0;
        while off < bytes.len() {
            // SAFETY: fd is a live tty descriptor; the slice is valid.
            let n = unsafe {
                libc::write(
                    self.fd,
                    bytes[off..].as_ptr() as *const libc::c_void,
                    bytes.len() - off,
                )
            };
            if n < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::WouldBlock {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                return Err(err);
            }
            off += n as usize;
        }
        Ok(())
    }
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // SAFETY: fd is a live tty descriptor; the buffer is valid for buf.len().
        let n = unsafe { libc::read(self.fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(0);
            }
            return Err(err);
        }
        Ok(n as usize)
    }
}

impl Drop for UartIo {
    fn drop(&mut self) {
        // SAFETY: fd is owned by this struct.
        unsafe { libc::close(self.fd) };
    }
}

/// Open the device named in a `devices.allow` entry.
pub fn open_io(device: &str) -> std::io::Result<Box<dyn Io>> {
    let lower = device.trim().to_ascii_lowercase();
    if lower.starts_with("udp:") || lower.starts_with("udpout:") {
        Ok(Box::new(UdpIo::connect(device)?))
    } else {
        let path = device.trim().strip_prefix("uart:").unwrap_or(device.trim());
        Ok(Box::new(UartIo::open(path, 57_600)?))
    }
}

/// One pump: read what is available, decode, and hand back events.
/// Non-blocking by construction, so a 10 Hz driver never parks the loop.
/// Drain what the transport has and decode it.
///
/// This reads the socket to exhaustion (bounded, so a flood cannot starve the
/// heartbeat): the old version stopped after the first datagram shorter than
/// the read buffer, which capped the link at one datagram per call. An
/// autopilot streams faster than that, so any pause built a backlog the link
/// could never work off — a 5 s mission delay was enough to push a
/// COMMAND_ACK behind seconds of queued telemetry and time out a healthy
/// vehicle.
pub fn pump(core: &mut LinkCore, io: &mut dyn Io, now_ms: u64) -> std::io::Result<Vec<Event>> {
    let mut buf = [0u8; 2048];
    let mut events = Vec::new();
    const MAX_DATAGRAMS: usize = 256;
    for _ in 0..MAX_DATAGRAMS {
        let n = io.read(&mut buf)?;
        if n == 0 {
            break;
        }
        events.extend(core.on_bytes(&buf[..n], now_ms));
    }
    Ok(events)
}

/// Encode + gate + write in one call — the driver's send helper.
pub fn send_frame(
    core: &mut LinkCore,
    io: &mut dyn Io,
    frame: &Frame,
    now_ms: u64,
    approved: bool,
) -> Result<(), Verdict> {
    let bytes = core.send(frame, now_ms, approved)?;
    io.write(&bytes).map_err(|e| {
        Verdict::deny(
            gate::E_SAFETY_VIOLATION,
            "R-HW/transport",
            format!("transport write failed: {e}"),
        )
    })
}

/// Convenience: a COMMAND_LONG for a MAV_CMD and its parameters.
#[allow(clippy::too_many_arguments)]
pub fn command_long(
    core: &LinkCore,
    target_system: u8,
    target_component: u8,
    command: u16,
    params: [f32; 7],
) -> Frame {
    mavlink::msg_command_long(
        core.seq,
        core.sysid,
        core.compid,
        target_system,
        target_component,
        command,
        params,
        0,
    )
}

/// True when a message class may be written in the policy's current mode.
pub fn writable(policy: &HardwarePolicy, spec: &MsgSpec) -> bool {
    match policy.mavlink_mode {
        gate::MavlinkMode::Off => false,
        gate::MavlinkMode::Observe => spec.class != Class::Actuator,
        gate::MavlinkMode::Allowlist => true,
    }
}

/// Build a frame from the `sofuu.mavlink` object shape (§4.4):
///
/// ```text
/// { type: "HEARTBEAT" }
/// { type: "COMMAND_LONG", command: "MAV_CMD_NAV_TAKEOFF" | 22,
///   param1..param7, alt?: <param7>, targetSystem?, targetComponent? }
/// { type: "SET_POSITION_TARGET_GLOBAL_INT",
///   lat, lon, alt, vx, vy, vz, yawRate, frame?, typeMask? }
/// ```
///
/// Anything else is an error naming the supported set — never a silently
/// zeroed frame.
pub fn frame_from_json(core: &LinkCore, v: &Value) -> Result<Frame, String> {
    let ty = v.get("type").and_then(|t| t.as_str()).ok_or("missing `type`")?;
    let spec = mavlink::spec_by_name(ty).ok_or_else(|| format!("unknown message type '{ty}'"))?;
    let num = |k: &str| v.get(k).and_then(|x| x.as_f64());
    let target_system = num("targetSystem").unwrap_or(1.0) as u8;
    let target_component = num("targetComponent").unwrap_or(1.0) as u8;

    match spec.id {
        0 => Ok(mavlink::msg_heartbeat(core.seq, core.sysid, core.compid)),
        76 | 75 => {
            let cmd_raw = v
                .get("command")
                .ok_or("COMMAND_* needs a `command` (name or MAV_CMD id)")?;
            let cmd = match cmd_raw {
                Value::Number(n) => n.as_u64().ok_or("command id must be an integer")? as u16,
                Value::String(s) => {
                    let want = s.trim().trim_start_matches("MAV_CMD_").to_ascii_uppercase();
                    *gate::ALLOWED_COMMANDS
                        .iter()
                        .chain(gate::ALWAYS_DENIED_COMMANDS.iter().map(|(c, _)| c))
                        .find(|c| {
                            gate::command_name(**c)
                                .map(|n| n.trim_start_matches("MAV_CMD_") == want)
                                .unwrap_or(false)
                        })
                        .ok_or_else(|| format!("unknown MAV_CMD '{s}'"))?
                }
                _ => return Err("command must be a name or a number".into()),
            };
            let mut params = [0.0f32; 7];
            for (i, p) in params.iter_mut().enumerate() {
                if let Some(n) = num(&format!("param{}", i + 1)) {
                    *p = n as f32;
                }
            }
            // Ergonomic aliases: `alt` is param7 for NAV_TAKEOFF, `speed` is
            // param2 for DO_CHANGE_SPEED — the two the agent tools set by name.
            if let Some(alt) = num("alt") {
                if cmd == mavlink::cmd::NAV_TAKEOFF {
                    params[6] = alt as f32;
                }
            }
            if let Some(speed) = num("speed") {
                if cmd == mavlink::cmd::DO_CHANGE_SPEED {
                    params[1] = speed as f32;
                }
            }
            if spec.id == 76 {
                Ok(mavlink::msg_command_long(
                    core.seq,
                    core.sysid,
                    core.compid,
                    target_system,
                    target_component,
                    cmd,
                    params,
                    0,
                ))
            } else {
                // COMMAND_INT carries a frame + integer lat/lon; keep the f32
                // parameters and place x/y/z from `lat`/`lon`/`alt` when given.
                let mut f = Frame::new(75, core.seq, core.sysid, core.compid)
                    .map_err(|e| e.to_string())?;
                for (i, p) in params.iter().enumerate() {
                    f.set_f32_at(i * 4, *p);
                }
                if let Some(lat) = num("lat") {
                    f.set_i32_at(16, (lat * 1e7) as i32);
                }
                if let Some(lon) = num("lon") {
                    f.set_i32_at(20, (lon * 1e7) as i32);
                }
                if let Some(alt) = num("alt") {
                    f.set_f32_at(24, alt as f32);
                }
                f.set_u16_at(28, cmd);
                f.set_u8_at(30, target_system);
                f.set_u8_at(31, target_component);
                f.set_u8_at(32, num("frame").unwrap_or(mavlink::frame_id::GLOBAL_RELATIVE_ALT_INT as f64) as u8);
                Ok(f)
            }
        }
        86 => {
            let lat = num("lat").ok_or("SET_POSITION_TARGET_GLOBAL_INT needs `lat`")?;
            let lon = num("lon").ok_or("SET_POSITION_TARGET_GLOBAL_INT needs `lon`")?;
            let alt = num("alt").unwrap_or(0.0);
            let vel = [num("vx").unwrap_or(0.0) as f32, num("vy").unwrap_or(0.0) as f32, num("vz").unwrap_or(0.0) as f32];
            let yaw_rate = num("yawRate").unwrap_or(0.0) as f32;
            /* Default mask depends on what the caller actually supplied.
             *
             * If it gave no velocity, it is asking for a POSITION — send
             * POS_ONLY. Defaulting to "position + velocity" would command a
             * velocity of zero, which the autopilot obeys by holding station
             * while ignoring the waypoint: a silent no-op that looks exactly
             * like a working link. If it did give a velocity, use both.
             *
             * FORCE_SET is never OR-ed in — it is not an ignore bit (512 means
             * "force is set"), so including it would *enable* force. */
            let has_velocity = num("vx").is_some() || num("vy").is_some() || num("vz").is_some();
            let default_mask = if has_velocity {
                mavlink::type_mask::IGNORE_ACCELERATION
                    | mavlink::type_mask::IGNORE_YAW
                    | mavlink::type_mask::IGNORE_YAW_RATE
            } else {
                mavlink::type_mask::POS_ONLY
            };
            let mask = num("typeMask").unwrap_or(default_mask as f64) as u16;
            Ok(mavlink::msg_set_position_target_global_int(
                core.seq,
                core.sysid,
                core.compid,
                target_system,
                target_component,
                (lat * 1e7) as i32,
                (lon * 1e7) as i32,
                alt as f32,
                vel,
                yaw_rate,
                mask,
                num("frame").unwrap_or(mavlink::frame_id::GLOBAL_RELATIVE_ALT_INT as f64) as u8,
            ))
        }
        44 => {
            let count = num("count").ok_or("MISSION_COUNT needs `count`")? as u16;
            let mut f = Frame::new(44, core.seq, core.sysid, core.compid).map_err(|e| e.to_string())?;
            f.set_u16_at(0, count);
            f.set_u8_at(2, target_system);
            f.set_u8_at(3, target_component);
            Ok(f)
        }
        other => Err(format!(
            "'{ty}' (msgid {other}) has no JSON builder yet — supported: HEARTBEAT, COMMAND_LONG, \
             COMMAND_INT, SET_POSITION_TARGET_GLOBAL_INT, MISSION_COUNT"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::mavlink::{frame_id, type_mask};
    use serde_json::json;

    const DEV: &str = "udp:127.0.0.1:14550";

    fn policy(mode: &str) -> HardwarePolicy {
        HardwarePolicy::from_policy(&json!({
            "hardware": {
                "enabled": true,
                "devices": { "mode": "allowlist",
                             "allow": [{ "path": DEV, "class": "mavlink-udp" }] },
                "mavlink": { "mode": mode,
                             "send": ["HEARTBEAT", "COMMAND_*", "SET_POSITION_TARGET_*"],
                             "rateLimits": { "COMMAND_*": "10Hz" } },
                "bounds": { "maxSpeedMs": 10, "maxAltRelM": 50, "maxYawRateDs": 90, "geofenceRadiusM": 300 }
            }
        }))
    }

    fn core_with_gps(mode: &str) -> LinkCore {
        let mut c = LinkCore::new(DEV, policy(mode));
        let mut gps = Frame::new(24, 0, 1, 1).unwrap();
        gps.set_u16_at(20, 80); // hdop 0.8
        gps.set_u8_at(28, 3);
        gps.set_u32_at(8, 521_234_000);
        gps.set_u32_at(12, 41_234_000);
        c.on_bytes(&mavlink::encode(&gps).unwrap(), 0);
        let mut pos = Frame::new(33, 0, 1, 1).unwrap();
        pos.set_u32_at(4, 521_234_000);
        pos.set_u32_at(8, 41_234_000);
        pos.set_i32_at(16, 0);
        c.on_bytes(&mavlink::encode(&pos).unwrap(), 0);
        c
    }

    fn takeoff(alt: f32) -> Frame {
        let mut p = [0.0f32; 7];
        p[6] = alt;
        mavlink::msg_command_long(0, 255, 191, 1, 1, mavlink::cmd::NAV_TAKEOFF, p, 0)
    }

    /* ── the write path ───────────────────────────────────────────── */

    #[test]
    fn send_refuses_a_disallowed_device_before_any_bytes_exist() {
        let mut c = LinkCore::new("udp:10.0.0.9:14550", policy("allowlist"));
        let err = c.send(&takeoff(10.0), 0, false).unwrap_err();
        assert_eq!(err.rule, "R-HW/devices.allow");
        assert!(c.audit.iter().any(|r| r["decision"] == "deny"));
    }

    #[test]
    fn send_in_observe_mode_refuses_actuator_frames() {
        let mut c = core_with_gps("observe");
        let err = c.send(&takeoff(10.0), 10, false).unwrap_err();
        assert_eq!(err.rule, "R-HW/mavlink.mode");
    }

    #[test]
    fn send_allows_a_whitelisted_in_bounds_command_and_assigns_a_seq() {
        let mut c = core_with_gps("allowlist");
        let bytes = c.send(&takeoff(15.0), 10, false).expect("allowed");
        let decoded = mavlink::decode(&bytes).unwrap();
        assert_eq!(decoded.msgid, 76);
        assert_eq!(decoded.seq, c.seq.wrapping_sub(1), "seq assigned from the link");
        assert_eq!(decoded.f32_at(24), 15.0);
        assert!(c.pending.contains_key(&mavlink::cmd::NAV_TAKEOFF), "a command waits for its ACK");
    }

    #[test]
    fn send_seq_increments_across_frames() {
        let mut c = core_with_gps("allowlist");
        let a = mavlink::decode(&c.send(&takeoff(10.0), 0, false).unwrap()).unwrap().seq;
        let b = mavlink::decode(&c.send(&takeoff(11.0), 200, false).unwrap()).unwrap().seq;
        assert_eq!(b, a.wrapping_add(1));
    }

    #[test]
    fn a_refused_send_leaves_no_pending_ack_and_no_seq_change() {
        let mut c = core_with_gps("allowlist");
        let before = c.seq;
        assert!(c.send(&takeoff(999.0), 0, false).is_err());
        assert_eq!(c.seq, before, "a refused frame must not consume a sequence number");
        assert!(c.pending.is_empty());
    }

    /* ── heartbeat / dead-man ─────────────────────────────────────── */

    #[test]
    fn heartbeat_is_emitted_at_1hz_by_the_runtime_not_by_callers() {
        let mut c = core_with_gps("allowlist");
        assert!(c.tick(0).is_some(), "first tick emits immediately");
        assert!(c.tick(500).is_none(), "not due yet");
        assert!(c.tick(999).is_none());
        assert!(c.tick(1_000).is_some(), "1 Hz cadence");
        let hb = mavlink::decode(&c.tick(2_000).unwrap()).unwrap();
        assert_eq!(hb.msgid, 0);
        assert_eq!(hb.sysid, 255);
        assert_eq!(hb.compid, 191);
    }

    #[test]
    fn heartbeat_stops_when_the_policy_forbids_it() {
        // observe mode allows HEARTBEAT, so use an off-mode policy instead.
        let mut c = LinkCore::new(DEV, policy("off"));
        assert!(c.tick(0).is_none(), "denied by mode");
        assert_eq!(c.audit[0]["decision"], "deny");
    }

    #[test]
    fn heartbeat_respects_its_rate_limit() {
        let mut p = policy("allowlist");
        p.rate_limits.push(("HEARTBEAT".to_string(), gate::Period::Every(5_000)));
        let mut c = LinkCore::new(DEV, p);
        assert!(c.tick(0).is_some());
        assert!(c.tick(1_000).is_none(), "the 5 s policy limit wins over the 1 s cadence");
        assert!(c.tick(5_000).is_some());
    }

    /* ── receive path + events ────────────────────────────────────── */

    #[test]
    fn telemetry_decodes_to_pilot_units() {
        let mut c = LinkCore::new(DEV, policy("observe"));
        let mut f = Frame::new(33, 0, 1, 1).unwrap();
        f.set_i32_at(4, 521_234_000);
        f.set_i32_at(8, 41_234_000);
        f.set_i32_at(12, 1_500_000); // 1500 m MSL
        f.set_i32_at(16, 25_000); // 25 m AGL
        f.set_i16_at(20, 150); // 1.5 m/s
        f.set_u16_at(26, 9_000); // 90 deg
        let ev = c.on_bytes(&mavlink::encode(&f).unwrap(), 0);
        match &ev[0] {
            Event::Telemetry { message, data } => {
                assert_eq!(*message, "GLOBAL_POSITION_INT");
                assert_eq!(data["lat"], 52.1234);
                assert_eq!(data["alt_m"], 1500.0);
                assert_eq!(data["relative_alt_m"], 25.0);
                assert_eq!(data["vx_ms"], 1.5);
                assert_eq!(data["heading_deg"], 90.0);
            }
            other => panic!("expected telemetry, got {other:?}"),
        }
    }

    #[test]
    fn receive_path_updates_the_gate_state_the_write_path_reads() {
        let mut c = LinkCore::new(DEV, policy("allowlist"));
        // No GPS yet: a coordinate command is refused.
        let f = mavlink::msg_set_position_target_global_int(
            0, 255, 191, 1, 1, 521_234_000, 41_234_000, 20.0, [0.0; 3], 0.0,
            type_mask::IGNORE_VELOCITY | type_mask::IGNORE_YAW | type_mask::IGNORE_YAW_RATE,
            frame_id::GLOBAL_RELATIVE_ALT_INT,
        );
        assert_eq!(c.send(&f, 0, false).unwrap_err().code, gate::E_NO_GPS_FIX);

        // The FC reports a fix; now the same command is allowed.
        let mut gps = Frame::new(24, 0, 1, 1).unwrap();
        gps.set_u16_at(20, 80);
        gps.set_u8_at(28, 3);
        c.on_bytes(&mavlink::encode(&gps).unwrap(), 100);
        let mut pos = Frame::new(33, 0, 1, 1).unwrap();
        pos.set_u32_at(4, 521_234_000);
        pos.set_u32_at(8, 41_234_000);
        c.on_bytes(&mavlink::encode(&pos).unwrap(), 100);

        assert!(c.send(&f, 150, false).is_ok());
    }

    #[test]
    fn resync_reports_corruption_and_holes_but_not_routine_skipping() {
        /* Skipping message types this link does not decode is normal traffic,
         * not a fault: a real autopilot streams ~40 of them per second, and one
         * Resync per pump drowned the telemetry. */
        let mut c = LinkCore::new(DEV, policy("observe"));
        let silent = c.on_bytes(&[0x00, 0xFF, 0x11, 0x22], 0);
        assert!(
            !silent.iter().any(|e| matches!(e, Event::Resync { .. })),
            "routine skipping must not report: {silent:?}"
        );

        /* A CRC failure is corruption and is always reported. */
        let mut bad = mavlink::encode(&Frame::new(0, 0, 1, 1).unwrap()).unwrap();
        let n = bad.len();
        bad[n - 1] ^= 0xFF; // flip a CRC byte
        let ev = c.on_bytes(&bad, 10);
        assert!(
            ev.iter().any(|e| matches!(e, Event::Resync { crc_errors: 1, .. })),
            "a CRC failure must be reported: {ev:?}"
        );

        /* A hole bigger than any frame (lost UDP datagram) is reported too. */
        let mut c = LinkCore::new(DEV, policy("observe"));
        let hole = vec![0x00u8; 512];
        let ev = c.on_bytes(&hole, 20);
        assert!(
            ev.iter().any(|e| matches!(e, Event::Resync { discarded_bytes: 512, .. })),
            "a stream hole must be reported: {ev:?}"
        );
    }

    /* ── Field offsets: MAVLink sorts wire fields by type size ────── */

    #[test]
    fn sys_status_battery_remaining_is_after_the_u16_block() {
        /* SYS_STATUS sorts u32 flags, then NINE u16 fields, then the u8s — so
         * battery_remaining is byte 30, not the byte after current_battery. */
        let mut f = Frame::new(1, 0, 1, 1).unwrap();
        f.set_u16_at(12, 415); // load 41.5 %
        f.set_u16_at(14, 12_600); // 12.6 V
        f.set_u16_at(16, 340); // 3.40 A
        f.payload[18] = 0x7F; // drop_rate_comm — must NOT be read as battery
        f.set_u8_at(30, 87);
        let t = telemetry_json(&f);
        assert_eq!(t["battery_remaining_pct"], 87, "{t}");
        assert_eq!(t["voltage_v"], 12.6, "{t}");
        assert_eq!(t["current_a"], 3.4, "{t}");
        assert_eq!(t["load_pct"], 41.5, "{t}");
    }

    #[test]
    fn rc_channels_puts_chancount_after_the_eighteen_channels() {
        /* chan1..chan18 occupy bytes 4..40, so chancount is 40 and rssi 41.
         * Read at 4 the count came back as chan1_raw's low byte (SITL's 1500
         * reported chancount 220). */
        let mut f = Frame::new(65, 0, 1, 1).unwrap();
        for i in 0..8u16 {
            f.set_u16_at(4 + i as usize * 2, 1500 + i);
        }
        f.set_u8_at(40, 8);
        f.set_u8_at(41, 200);
        let t = telemetry_json(&f);
        assert_eq!(t["chancount"], 8, "{t}");
        assert_eq!(t["rssi"], 200, "{t}");
        assert_eq!(
            t["channels"],
            serde_json::json!([1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507]),
            "{t}"
        );
    }

    /* ── ACK + PreArm correlation ─────────────────────────────────── */

    #[test]
    fn a_command_ack_clears_the_pending_entry() {
        let mut c = core_with_gps("allowlist");
        c.send(&takeoff(10.0), 0, false).unwrap();
        assert!(c.pending.contains_key(&mavlink::cmd::NAV_TAKEOFF));

        let mut ack = Frame::new(77, 0, 1, 1).unwrap();
        ack.set_u16_at(0, mavlink::cmd::NAV_TAKEOFF);
        ack.set_u8_at(2, mav_result::ACCEPTED);
        let ev = c.on_bytes(&mavlink::encode(&ack).unwrap(), 50);
        match &ev[0] {
            Event::CommandAck { result_name, .. } => assert_eq!(*result_name, "ACCEPTED"),
            other => panic!("expected ack, got {other:?}"),
        }
        assert!(c.pending.is_empty());
    }

    #[test]
    fn a_failed_arm_surfaces_ardupilot_prearm_text() {
        let mut c = core_with_gps("allowlist");
        // ArduPilot broadcasts the reason just before/around the ACK.
        let mut st = Frame::new(253, 0, 1, 1).unwrap();
        st.set_u8_at(0, 3); // MAV_SEVERITY_ERROR
        let msg = b"PreArm: High GPS HDOP (HDOP=2.6, required < 2.0)";
        for (i, b) in msg.iter().enumerate() {
            st.set_u8_at(1 + i, *b);
        }
        c.on_bytes(&mavlink::encode(&st).unwrap(), 1_000);

        let mut arm = [0.0f32; 7];
        arm[0] = 1.0;
        let f = mavlink::msg_command_long(0, 255, 191, 1, 1, mavlink::cmd::COMPONENT_ARM_DISARM, arm, 0);
        c.send(&f, 1_010, false).unwrap();

        let mut ack = Frame::new(77, 0, 1, 1).unwrap();
        ack.set_u16_at(0, mavlink::cmd::COMPONENT_ARM_DISARM);
        ack.set_u8_at(2, mav_result::FAILED);
        let ev = c.on_bytes(&mavlink::encode(&ack).unwrap(), 1_020);
        match &ev[0] {
            Event::CommandAck { text, result_name, .. } => {
                assert_eq!(*result_name, "FAILED");
                let t = text.as_deref().expect("PreArm text must be attached");
                assert!(t.contains("High GPS HDOP"), "{t}");
            }
            other => panic!("expected ack with text, got {other:?}"),
        }
    }

    #[test]
    fn a_successful_ack_does_not_carry_a_prearm_text() {
        let mut c = core_with_gps("allowlist");
        let mut ack = Frame::new(77, 0, 1, 1).unwrap();
        ack.set_u16_at(0, mavlink::cmd::NAV_TAKEOFF);
        ack.set_u8_at(2, mav_result::ACCEPTED);
        match &c.on_bytes(&mavlink::encode(&ack).unwrap(), 0)[0] {
            Event::CommandAck { text, .. } => assert!(text.is_none()),
            other => panic!("expected ack, got {other:?}"),
        }
    }

    #[test]
    fn a_stale_statustext_is_not_attributed_to_a_later_failure() {
        let mut c = core_with_gps("allowlist");
        let mut st = Frame::new(253, 0, 1, 1).unwrap();
        st.set_u8_at(0, 3);
        for (i, b) in b"PreArm: old reason".iter().enumerate() {
            st.set_u8_at(1 + i, *b);
        }
        c.on_bytes(&mavlink::encode(&st).unwrap(), 0);

        let mut ack = Frame::new(77, 0, 1, 1).unwrap();
        ack.set_u16_at(0, mavlink::cmd::NAV_TAKEOFF);
        ack.set_u8_at(2, mav_result::FAILED);
        // 3 s window: a much later failure must not get the old text.
        match &c.on_bytes(&mavlink::encode(&ack).unwrap(), PREARM_WINDOW_MS + 1)[0] {
            Event::CommandAck { text, .. } => assert!(text.is_none(), "stale text must not attach"),
            other => panic!("expected ack, got {other:?}"),
        }
    }

    #[test]
    fn statustext_decoding_stops_at_the_nul() {
        let mut st = Frame::new(253, 0, 1, 1).unwrap();
        st.set_u8_at(0, 6);
        for (i, b) in b"hello".iter().enumerate() {
            st.set_u8_at(1 + i, *b);
        }
        assert_eq!(statustext_of(&st), "hello");
    }

    /* ── audit ────────────────────────────────────────────────────── */

    #[test]
    fn every_send_decision_is_auditable_and_drainable() {
        let mut c = core_with_gps("allowlist");
        c.send(&takeoff(10.0), 0, false).unwrap();
        let _ = c.send(&takeoff(999.0), 10, false);
        let records = c.take_audit();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["decision"], "allow");
        assert_eq!(records[1]["decision"], "deny");
        assert!(records[0]["at_ms"].as_u64() == Some(0));
        assert!(c.take_audit().is_empty(), "draining is destructive");
    }

    /* ── transport parsing ────────────────────────────────────────── */

    #[test]
    fn udp_endpoint_parsing() {
        assert_eq!(parse_udp("udp:127.0.0.1:14550"), Some(("127.0.0.1".into(), 14550)));
        assert_eq!(parse_udp("udp:example.com:14540"), Some(("example.com".into(), 14540)));
        assert_eq!(parse_udp("udp:[::1]:14550"), Some(("::1".into(), 14550)));
        assert_eq!(parse_udp("/dev/ttyAMA0"), None);
        assert_eq!(parse_udp("udp::14550"), None);
    }

    #[test]
    fn writable_reflects_the_policy_mode() {
        let spec = mavlink::spec_by_name("COMMAND_LONG").unwrap();
        assert!(!writable(&policy("off"), spec));
        assert!(!writable(&policy("observe"), spec));
        assert!(writable(&policy("allowlist"), spec));
        let hb = mavlink::spec_by_name("HEARTBEAT").unwrap();
        assert!(writable(&policy("observe"), hb));
    }

    /* ── §6.5: the fuzz properties, deterministically in the suite ── */

    /// `crates/fuzz/fuzz_targets/mavlink.rs` needs the cargo-fuzz layout and a
    /// working ASan host; neither exists here. These are the same two
    /// properties, driven by a deterministic LCG so CI covers them every run:
    ///
    /// 1. arbitrary bytes never panic the parser, and
    /// 2. a refused frame never produces bytes.
    #[test]
    fn fuzz_style_no_panic_and_no_bypass() {
        let p = policy("allowlist");
        let mut seed: u64 = 0x5EED_1234_ABCD_0001;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };

        let mut core = LinkCore::new(DEV, p.clone());
        for _ in 0..2_000 {
            let len = (next() % 64) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| (next() & 0xFF) as u8).collect();
            // 1. the parser is total over arbitrary input
            let _ = core.on_bytes(&bytes, 0);
            let _ = core.parser.next();
            let st = core.parser.stats();
            let _ = st.discarded_bytes + st.crc_errors;
        }

        // 2. whatever the fuzzer-shaped frames are, a denial means no bytes.
        let ids = [0u32, 75, 76, 84, 86, 73, 44, 77];
        for _ in 0..2_000 {
            let msgid = ids[(next() as usize) % ids.len()];
            let Ok(mut frame) = Frame::new(msgid, 0, 255, 191) else { continue };
            for i in 0..frame.payload.len() {
                frame.payload[i] = (next() & 0xFF) as u8;
            }
            let mut state = GateState::default();
            let device = if next() % 2 == 0 { DEV } else { "udp:10.0.0.9:14550" };
            let verdict = gate::gate_send(
                &p,
                &mut state,
                &SendRequest { device, frame: &frame },
                0,
                false,
            );
            if !verdict.allowed {
                assert!(!verdict.code.is_empty(), "a refusal must carry a code");
                assert!(!verdict.rule.is_empty(), "a refusal must name the rule");
                let mut c = LinkCore::new(device, p.clone());
                assert!(
                    c.send(&frame, 0, false).is_err(),
                    "gate denied msgid {msgid} but the send path produced bytes"
                );
            }
        }

        // Permanently denied commands are denied whatever the payload says.
        for cmd in [mavlink::cmd::PARAM_SET, mavlink::cmd::DO_MOTOR_TEST] {
            let f = mavlink::msg_command_long(0, 255, 191, 1, 1, cmd, [1.0; 7], 0);
            let mut st = GateState::default();
            let v = gate::gate_send(&p, &mut st, &SendRequest { device: DEV, frame: &f }, 0, false);
            assert!(!v.allowed);
            assert_eq!(v.rule, "R-HW/deny/always");
        }
    }

    /* ── real transports (§6.2: no hardware needed) ───────────────── */

    #[test]
    fn one_pump_drains_every_queued_datagram() {
        /* A pump used to stop after the first datagram shorter than the read
         * buffer. An autopilot streams faster than 100 datagrams/s, so the link
         * fell permanently behind and a COMMAND_ACK could sit buried behind
         * seconds of telemetry — a healthy vehicle reported as
         * `E_TIMEOUT: no COMMAND_ACK`. One pump must now take everything that
         * is queued. */
        let device = "udp:127.0.0.1:0";
        let p = HardwarePolicy::from_policy(&json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": device, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "observe" } }
        }));
        let mut core = LinkCore::new(device, p);
        let mut io = UdpIo::connect(device).expect("bind");
        let local = io.local_addr().expect("local addr");
        let fc = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();

        /* 40 one-frame datagrams, the shape of a 20 Hz telemetry stream that
         * arrived while nobody was reading. */
        for i in 0..40u8 {
            let mut att = Frame::new(30, i, 1, 1).unwrap();
            att.set_f32_at(4, 0.1);
            fc.send_to(&mavlink::encode(&att).unwrap(), local).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));

        let ev = pump(&mut core, &mut io, 0).expect("pump");
        let telemetry = ev
            .iter()
            .filter(|e| matches!(e, Event::Telemetry { message, .. } if *message == "ATTITUDE"))
            .count();
        assert_eq!(telemetry, 40, "one pump must drain the whole queue; got {telemetry}");
    }

    #[test]
    fn udp_transport_round_trips_against_a_plain_socket() {
        // Port 0 → the OS picks one; the FC is a plain std socket, exactly the
        // shape of SITL's `--out=udp:…` stream.
        let device = "udp:127.0.0.1:0";
        let p = HardwarePolicy::from_policy(&json!({
            "hardware": { "enabled": true,
                          "devices": { "mode": "allowlist",
                                       "allow": [{ "path": device, "class": "mavlink-udp" }] },
                          "mavlink": { "mode": "observe" } }
        }));
        let mut core = LinkCore::new(device, p);
        let mut io = UdpIo::connect(device).expect("bind an ephemeral udp endpoint");
        let local = io.local_addr().expect("local addr");
        let fc = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        fc.set_read_timeout(Some(std::time::Duration::from_millis(500))).unwrap();

        // FC → Sofuu: one ATTITUDE frame.
        let mut att = Frame::new(30, 0, 1, 1).unwrap();
        att.set_f32_at(4, 0.25);
        fc.send_to(&mavlink::encode(&att).unwrap(), local).unwrap();

        let mut got = Vec::new();
        for _ in 0..100 {
            got.extend(pump(&mut core, &mut io, 0).expect("pump"));
            if !got.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(
            matches!(&got[0], Event::Telemetry { message, .. } if *message == "ATTITUDE"),
            "expected ATTITUDE, got {got:?}"
        );
        assert!(io.peer().is_some(), "the sender is pinned as the peer");

        // Sofuu → FC: the heartbeat goes back to the address we learned.
        let hb = core.tick(0).expect("heartbeat is due immediately");
        io.write(&hb).expect("write to the pinned peer");
        let mut buf = [0u8; 256];
        let (n, _) = fc.recv_from(&mut buf).expect("heartbeat must reach the FC");
        assert_eq!(mavlink::decode(&buf[..n]).unwrap().msgid, 0);
    }

    #[test]
    fn udp_write_before_any_datagram_fails_loudly() {
        // No peer learned yet → an error, never a silent drop.
        let mut io = UdpIo::connect("udp:127.0.0.1:0").unwrap();
        let err = io.write(&[1, 2, 3]).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotConnected);
    }

    #[test]
    fn uart_transport_round_trips_over_a_pty_pair() {
        // §6.2: a pty pair stands in for /dev/ttyAMA0 — the gate's device-path
        // logic and the termios setup are exercised with no hardware at all.
        // SAFETY: openpty allocates a fresh pty; both fds are owned below
        // (slave moves into UartIo, master is closed explicitly).
        unsafe {
            let mut master: libc::c_int = -1;
            let mut slave: libc::c_int = -1;
            let rc = libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert_eq!(rc, 0, "openpty");
            let mut io = UartIo::adopt(slave, 57_600).expect("raw mode on the pty slave");

            let frame = mavlink::encode(&mavlink::msg_heartbeat(7, 1, 1)).unwrap();
            let written = libc::write(master, frame.as_ptr() as *const libc::c_void, frame.len());
            assert_eq!(written as usize, frame.len(), "write to the pty master");

            let mut buf = [0u8; 256];
            let mut got = 0usize;
            for _ in 0..100 {
                let n = io.read(&mut buf[got..]).expect("read from the pty slave");
                if n > 0 {
                    got += n;
                    if got >= frame.len() {
                        break;
                    }
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
            assert!(got >= frame.len(), "frame must cross the pty, got {got} bytes");
            let decoded = mavlink::decode(&buf[..frame.len()]).expect("decode across the pty");
            assert_eq!(decoded.msgid, 0);
            assert_eq!(decoded.seq, 7);
            libc::close(master);
        }
    }
}
