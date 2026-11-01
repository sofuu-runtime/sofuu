//! MAVLink v2 codec — hand-rolled common-dialect subset.
//!
//! PLAN-DRONES-MAVLINK §4.1. No generated dialect, no dialect crate: the wire
//! format is small enough to audit line-by-line, and the gate has to be able
//! to explain every byte it judges.
//!
//! Scope is exactly the messages in `MESSAGES`. A well-formed frame carrying
//! any other message id is skipped (its payload is never interpreted) and
//! counted — telemetry from a newer flight controller must not break the link,
//! and a message we cannot CRC-validate must never be acted on.
//!
//! ```
//! use sofuu_core::rt::mavlink::{Frame, encode, decode, msg_heartbeat};
//!
//! let f = msg_heartbeat(7, 255, 191);
//! let bytes = encode(&f).expect("encodes");
//! assert_eq!(decode(&bytes).expect("decodes").msgid, 0);
//! ```

/// Start-of-frame magic for MAVLink v2.
pub const STX: u8 = 0xFD;
/// Largest payload a MAVLink v2 frame may carry.
pub const MAX_PAYLOAD: usize = 255;
/// STX + header (len, incompat, compat, seq, sysid, compid, msgid×3) — 10 bytes.
pub const HEADER_LEN: usize = 10;
/// Bytes appended to the header + payload.
pub const TRAILER_LEN: usize = 2;
/// MAVLink v2 incompatibility flag: the frame carries a 13-byte signature.
pub const INCOMPAT_SIGNED: u8 = 0x01;
/// Length of the signature block at the end of a signed frame.
pub const SIGNATURE_LEN: usize = 13;

/// Wire class. The gate's whole pipeline is keyed on it: telemetry flows,
/// actuator-bearing frames get the full veto chain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    /// The dead-man contract (§4.5) — the one non-actuator message Sofuu sends.
    Heartbeat,
    /// Sensor/state data. Receive path is not gated (read-only).
    Telemetry,
    /// Every message that can move a vehicle or change its state. Gated.
    Actuator,
}

/// Documented direction. Not enforced by the codec — the gate decides what may
/// be written — but it is what makes the table auditable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Send,
    Recv,
    Both,
}

/// One message of the supported subset.
#[derive(Clone, Copy, Debug)]
pub struct MsgSpec {
    pub id: u32,
    pub name: &'static str,
    /// Full payload length in bytes (the dialect's max_len). Receivers
    /// zero-extend a truncated payload to this length.
    pub len: usize,
    pub crc_extra: u8,
    pub class: Class,
    pub dir: Dir,
}

/// The supported subset. `id`/`len`/`crc_extra` are pinned from the upstream
/// `common.xml` dialect (verified against `mavlink/c_library_v2`'s generated
/// headers — see `tests::crc_extra_table_is_pinned`).
///
/// Two corrections against the plan's first draft of this table:
/// `MISSION_ITEM_INT` is id **73** (38 is MISSION_WRITE_PARTIAL_LIST) and
/// `MISSION_REQUEST` is id **40** (43 is MISSION_REQUEST_LIST).
/// `GPS_RAW_INT` is an addition: the §4.3 fail-closed rule needs `fix_type`
/// and `hdop`, and no other message in the subset carries them.
pub const MESSAGES: &[MsgSpec] = &[
    MsgSpec { id: 0, name: "HEARTBEAT", len: 9, crc_extra: 50, class: Class::Heartbeat, dir: Dir::Both },
    MsgSpec { id: 1, name: "SYS_STATUS", len: 43, crc_extra: 124, class: Class::Telemetry, dir: Dir::Recv },
    MsgSpec { id: 24, name: "GPS_RAW_INT", len: 52, crc_extra: 24, class: Class::Telemetry, dir: Dir::Recv },
    MsgSpec { id: 30, name: "ATTITUDE", len: 28, crc_extra: 39, class: Class::Telemetry, dir: Dir::Recv },
    MsgSpec { id: 33, name: "GLOBAL_POSITION_INT", len: 28, crc_extra: 104, class: Class::Telemetry, dir: Dir::Recv },
    MsgSpec { id: 40, name: "MISSION_REQUEST", len: 5, crc_extra: 230, class: Class::Actuator, dir: Dir::Both },
    MsgSpec { id: 44, name: "MISSION_COUNT", len: 9, crc_extra: 221, class: Class::Actuator, dir: Dir::Both },
    MsgSpec { id: 65, name: "RC_CHANNELS", len: 42, crc_extra: 118, class: Class::Telemetry, dir: Dir::Recv },
    MsgSpec { id: 73, name: "MISSION_ITEM_INT", len: 38, crc_extra: 38, class: Class::Actuator, dir: Dir::Both },
    MsgSpec { id: 75, name: "COMMAND_INT", len: 35, crc_extra: 158, class: Class::Actuator, dir: Dir::Send },
    MsgSpec { id: 76, name: "COMMAND_LONG", len: 33, crc_extra: 152, class: Class::Actuator, dir: Dir::Send },
    MsgSpec { id: 77, name: "COMMAND_ACK", len: 10, crc_extra: 143, class: Class::Telemetry, dir: Dir::Recv },
    MsgSpec { id: 84, name: "SET_POSITION_TARGET_LOCAL_NED", len: 53, crc_extra: 143, class: Class::Actuator, dir: Dir::Send },
    MsgSpec { id: 86, name: "SET_POSITION_TARGET_GLOBAL_INT", len: 53, crc_extra: 5, class: Class::Actuator, dir: Dir::Send },
    MsgSpec { id: 253, name: "STATUSTEXT", len: 54, crc_extra: 83, class: Class::Telemetry, dir: Dir::Recv },
];

/// MAV_CMD values the gate names. Pinned from `common.xml`'s enum; the gate's
/// command rule is an allowlist, so an unrecognised value is refused rather
/// than guessed at.
pub mod cmd {
    pub const NAV_WAYPOINT: u16 = 16;
    pub const NAV_LOITER_UNLIM: u16 = 17;
    pub const NAV_LOITER_TURNS: u16 = 18;
    pub const NAV_LOITER_TIME: u16 = 19;
    pub const NAV_RETURN_TO_LAUNCH: u16 = 20;
    pub const NAV_LAND: u16 = 21;
    pub const NAV_TAKEOFF: u16 = 22;
    pub const NAV_LOITER_TO_ALT: u16 = 31;
    pub const NAV_SPLINE_WAYPOINT: u16 = 82;
    pub const NAV_GUIDED_ENABLE: u16 = 92;
    pub const NAV_DELAY: u16 = 93;
    pub const DO_ORBIT: u16 = 34;
    pub const DO_SET_MODE: u16 = 176;
    pub const DO_CHANGE_SPEED: u16 = 178;
    pub const DO_SET_HOME: u16 = 179;
    pub const DO_PAUSE_CONTINUE: u16 = 193;
    /// Never allowed by any shipped profile — the agent must not spin motors.
    pub const DO_MOTOR_TEST: u16 = 209;
    pub const MISSION_START: u16 = 300;
    /// Never allowed by any shipped profile — the agent must not re-tune the FC.
    pub const PARAM_SET: u16 = 23;
    pub const COMPONENT_ARM_DISARM: u16 = 400;
    /// Ask the autopilot to stream one message at an interval. Telemetry
    /// configuration, not actuation: ArduPilot 4.6 removed the `SRx_*` stream
    /// parameters, so a GCS must request what it wants to receive.
    pub const SET_MESSAGE_INTERVAL: u16 = 511;
}

/// MAV_FRAME values used by the bounds checks.
pub mod frame_id {
    pub const GLOBAL: u8 = 0;
    pub const LOCAL_NED: u8 = 1;
    pub const GLOBAL_RELATIVE_ALT: u8 = 3;
    pub const GLOBAL_INT: u8 = 5;
    pub const GLOBAL_RELATIVE_ALT_INT: u8 = 6;
    pub const LOCAL_OFFSET_NED: u8 = 7;
    pub const BODY_NED: u8 = 8;
    pub const BODY_OFFSET_NED: u8 = 9;
}

/// SET_POSITION_TARGET `type_mask` bits. A SET bit means "ignore this field",
/// which is why the bounds checks only look at unmasked fields.
///
/// The bits are **per axis**, not one bit per quantity:
///
/// ```text
///    1 X_IGNORE    8 VX_IGNORE   64 AX_IGNORE   512 FORCE_SET
///    2 Y_IGNORE   16 VY_IGNORE  128 AY_IGNORE  1024 YAW_IGNORE
///    4 Z_IGNORE   32 VZ_IGNORE  256 AZ_IGNORE  2048 YAW_RATE_IGNORE
/// ```
///
/// Getting this wrong is silent and total. The first version of this module
/// defined `IGNORE_POSITION = 1, IGNORE_VELOCITY = 2, IGNORE_ACCELERATION = 4,
/// …`, so the "position only" frame it built came to **62** — ignore Y, Z and
/// all velocity while *using* acceleration AX/AY/AZ, which it sent as zero.
/// ArduCopter then commanded zero acceleration and the vehicle held position
/// perfectly while ignoring every waypoint: `goto` was inert for an entire
/// session. ArduPilot's own autotest uses [`POS_ONLY`] (3576) for the same job.
pub mod type_mask {
    pub const X_IGNORE: u16 = 1;
    pub const Y_IGNORE: u16 = 2;
    pub const Z_IGNORE: u16 = 4;
    pub const VX_IGNORE: u16 = 8;
    pub const VY_IGNORE: u16 = 16;
    pub const VZ_IGNORE: u16 = 32;
    pub const AX_IGNORE: u16 = 64;
    pub const AY_IGNORE: u16 = 128;
    pub const AZ_IGNORE: u16 = 256;
    /// Not an ignore bit: set means "force is set" (LOCAL_NED multirotor only).
    pub const FORCE_SET: u16 = 512;
    pub const YAW_IGNORE: u16 = 1024;
    pub const YAW_RATE_IGNORE: u16 = 2048;

    pub const IGNORE_POSITION: u16 = X_IGNORE | Y_IGNORE | Z_IGNORE;
    pub const IGNORE_VELOCITY: u16 = VX_IGNORE | VY_IGNORE | VZ_IGNORE;
    pub const IGNORE_ACCELERATION: u16 = AX_IGNORE | AY_IGNORE | AZ_IGNORE;
    pub const IGNORE_YAW: u16 = YAW_IGNORE;
    pub const IGNORE_YAW_RATE: u16 = YAW_RATE_IGNORE;

    /// Use position only — ignore velocity, acceleration, yaw and yaw rate.
    /// 3576, the mask ArduPilot's own autotest sends for `fly_guided_move_to`.
    pub const POS_ONLY: u16 = IGNORE_VELOCITY | IGNORE_ACCELERATION | IGNORE_YAW | IGNORE_YAW_RATE;
    /// Use velocity only — ignore position, acceleration, yaw and yaw rate.
    pub const VEL_ONLY: u16 = IGNORE_POSITION | IGNORE_ACCELERATION | IGNORE_YAW | IGNORE_YAW_RATE;
}

/// GPS fix types (`GPS_FIX_TYPE`). 3 = 3D fix, which is the gate's floor.
pub const GPS_FIX_3D: u8 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MavError {
    /// Fewer bytes than the header declares.
    ShortBuffer,
    /// The buffer does not start with `STX`.
    BadStx,
    /// CRC-16/X.25 mismatch — a corrupted frame (EMI, sag, truncation).
    BadCrc,
    /// A message id outside `MESSAGES`. Its payload cannot be CRC-validated,
    /// so it is never interpreted.
    UnknownMessage,
    /// Payload longer than `MAX_PAYLOAD`.
    PayloadTooLong,
}

impl std::fmt::Display for MavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MavError::ShortBuffer => write!(f, "buffer is shorter than the frame header"),
            MavError::BadStx => write!(f, "frame does not start with 0xFD"),
            MavError::BadCrc => write!(f, "frame CRC-16/X.25 mismatch"),
            MavError::UnknownMessage => write!(f, "message id is outside the supported subset"),
            MavError::PayloadTooLong => write!(f, "payload exceeds 255 bytes"),
        }
    }
}

impl std::error::Error for MavError {}

/// Look up a message by id.
pub fn spec_by_id(id: u32) -> Option<&'static MsgSpec> {
    MESSAGES.iter().find(|m| m.id == id)
}

/// Look up a message by name (case-insensitive). Policy files spell messages
/// by name; the wire spells them by id.
pub fn spec_by_name(name: &str) -> Option<&'static MsgSpec> {
    MESSAGES
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(name.trim()))
}

/// A decoded frame plus the spec that validated it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub seq: u8,
    pub sysid: u8,
    pub compid: u8,
    pub msgid: u32,
    /// Payload zero-extended to the message's full length, so field accessors
    /// never need bounds arithmetic.
    pub payload: Vec<u8>,
}

impl Frame {
    /// A zero-filled frame for `msgid`.
    pub fn new(msgid: u32, seq: u8, sysid: u8, compid: u8) -> Result<Frame, MavError> {
        let spec = spec_by_id(msgid).ok_or(MavError::UnknownMessage)?;
        Ok(Frame { seq, sysid, compid, msgid, payload: vec![0u8; spec.len] })
    }

    pub fn spec(&self) -> Option<&'static MsgSpec> {
        spec_by_id(self.msgid)
    }

    pub fn name(&self) -> &'static str {
        spec_by_id(self.msgid).map(|s| s.name).unwrap_or("UNKNOWN")
    }

    pub fn class(&self) -> Option<Class> {
        spec_by_id(self.msgid).map(|s| s.class)
    }

    /* ── little-endian field accessors (zero past the end, never panic) ── */

    pub fn u8_at(&self, off: usize) -> u8 {
        self.payload.get(off).copied().unwrap_or(0)
    }

    pub fn u16_at(&self, off: usize) -> u16 {
        u16::from_le_bytes([self.u8_at(off), self.u8_at(off + 1)])
    }

    pub fn i16_at(&self, off: usize) -> i16 {
        self.u16_at(off) as i16
    }

    pub fn u32_at(&self, off: usize) -> u32 {
        u32::from_le_bytes([
            self.u8_at(off),
            self.u8_at(off + 1),
            self.u8_at(off + 2),
            self.u8_at(off + 3),
        ])
    }

    pub fn i32_at(&self, off: usize) -> i32 {
        self.u32_at(off) as i32
    }

    pub fn f32_at(&self, off: usize) -> f32 {
        f32::from_bits(self.u32_at(off))
    }

    pub fn set_u8_at(&mut self, off: usize, v: u8) {
        if off < self.payload.len() {
            self.payload[off] = v;
        }
    }

    pub fn set_u16_at(&mut self, off: usize, v: u16) {
        for (i, b) in v.to_le_bytes().iter().enumerate() {
            self.set_u8_at(off + i, *b);
        }
    }

    pub fn set_u32_at(&mut self, off: usize, v: u32) {
        for (i, b) in v.to_le_bytes().iter().enumerate() {
            self.set_u8_at(off + i, *b);
        }
    }

    pub fn set_f32_at(&mut self, off: usize, v: f32) {
        self.set_u32_at(off, v.to_bits());
    }

    pub fn set_i16_at(&mut self, off: usize, v: i16) {
        self.set_u16_at(off, v as u16);
    }

    pub fn set_i32_at(&mut self, off: usize, v: i32) {
        self.set_u32_at(off, v as u32);
    }
}

/* ── CRC-16/X.25 (MAVLink's checksum; the X.25 cache algorithm) ────────── */

/// One byte through the MAVLink CRC accumulator. The seed is `0xFFFF`; the
/// per-message CRC_EXTRA byte is fed in last.
pub fn crc_accumulate(mut crc: u16, byte: u8) -> u16 {
    let tmp = byte ^ (crc as u8);
    let tmp = tmp ^ (tmp << 4);
    let tmp = tmp as u16;
    (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
}

/// CRC over the header (from `len`, not including `STX`), the payload as it
/// appears on the wire, and the message's CRC_EXTRA byte.
fn frame_crc(spec: &MsgSpec, seq: u8, sysid: u8, compid: u8, payload: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for b in [
        payload.len() as u8,
        0u8, // incompat
        0u8, // compat
        seq,
        sysid,
        compid,
    ] {
        crc = crc_accumulate(crc, b);
    }
    for b in (spec.id & 0xFF_FFFF).to_le_bytes().iter().take(3) {
        crc = crc_accumulate(crc, *b);
    }
    for b in payload {
        crc = crc_accumulate(crc, *b);
    }
    crc_accumulate(crc, spec.crc_extra)
}

/// Encode a frame. Trailing zero payload bytes are trimmed per the v2 spec and
/// `len` reflects the trimmed payload. Never signs (§4.1: v1 does not sign).
pub fn encode(frame: &Frame) -> Result<Vec<u8>, MavError> {
    let spec = spec_by_id(frame.msgid).ok_or(MavError::UnknownMessage)?;
    if frame.payload.len() > MAX_PAYLOAD {
        return Err(MavError::PayloadTooLong);
    }
    let mut payload = frame.payload.clone();
    payload.resize(spec.len.max(payload.len()), 0);
    while payload.last() == Some(&0) && !payload.is_empty() {
        payload.pop();
    }
    if payload.len() > MAX_PAYLOAD {
        return Err(MavError::PayloadTooLong);
    }

    let mut out = Vec::with_capacity(HEADER_LEN + payload.len() + TRAILER_LEN);
    out.push(STX);
    out.push(payload.len() as u8);
    out.push(0); // incompat
    out.push(0); // compat
    out.push(frame.seq);
    out.push(frame.sysid);
    out.push(frame.compid);
    out.extend_from_slice(&(spec.id & 0xFF_FFFF).to_le_bytes()[..3]);
    out.extend_from_slice(&payload);
    let crc = frame_crc(spec, frame.seq, frame.sysid, frame.compid, &payload);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// Decode exactly one frame from the front of `buf`. `buf` may be longer than
/// the frame (the remainder is ignored — use `Parser` for a stream).
pub fn decode(buf: &[u8]) -> Result<Frame, MavError> {
    if buf.len() < HEADER_LEN {
        return Err(MavError::ShortBuffer);
    }
    if buf[0] != STX {
        return Err(MavError::BadStx);
    }
    let len = buf[1] as usize;
    let incompat = buf[2];
    let seq = buf[4];
    let sysid = buf[5];
    let compid = buf[6];
    let msgid = u32::from_le_bytes([buf[7], buf[8], buf[9], 0]);

    let sig = if incompat & INCOMPAT_SIGNED != 0 { SIGNATURE_LEN } else { 0 };
    let total = HEADER_LEN + len + TRAILER_LEN + sig;
    if buf.len() < total {
        return Err(MavError::ShortBuffer);
    }

    let spec = spec_by_id(msgid).ok_or(MavError::UnknownMessage)?;
    let payload = &buf[HEADER_LEN..HEADER_LEN + len];

    let mut crc: u16 = 0xFFFF;
    for b in &buf[1..HEADER_LEN] {
        crc = crc_accumulate(crc, *b);
    }
    for b in payload {
        crc = crc_accumulate(crc, *b);
    }
    crc = crc_accumulate(crc, spec.crc_extra);
    let claimed = u16::from_le_bytes([buf[HEADER_LEN + len], buf[HEADER_LEN + len + 1]]);
    if crc != claimed {
        return Err(MavError::BadCrc);
    }

    let mut full = payload.to_vec();
    full.resize(spec.len, 0); // v2 zero-extension
    Ok(Frame { seq, sysid, compid, msgid, payload: full })
}

/// What the stream scanner had to throw away — the EMI/telemetry-loss receipt.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct ParserStats {
    /// Bytes thrown away outside any recognised frame — noise, or the bytes of
    /// a frame whose CRC failed. **This is the loss metric.**
    pub discarded_bytes: u64,
    /// Frames whose CRC did not match — corrupted in transit.
    pub crc_errors: u64,
    /// Well-formed frames of message types this link does not decode, skipped
    /// whole. Routine traffic from any real autopilot (~40 types streamed),
    /// not a fault — kept apart from `discarded_bytes` so that a real hole in
    /// the stream is still visible.
    pub skipped_messages: u64,
}

/// Sliding-window resynchronising stream parser (§4.1).
///
/// A corrupted byte on a UART under motor EMI must not tear down the link: on
/// any failure the scanner advances one byte and looks for the next `0xFD`,
/// so the connection recovers as soon as the wire does.
#[derive(Default, Debug)]
pub struct Parser {
    buf: Vec<u8>,
    stats: ParserStats,
}

impl Parser {
    pub fn new() -> Self {
        Self { buf: Vec::new(), stats: ParserStats::default() }
    }

    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    pub fn stats(&self) -> ParserStats {
        self.stats
    }

    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Drop everything buffered (e.g. on reconnect).
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Yield the next CRC-valid frame, advancing past anything unusable.
    /// Returns `None` when the buffer holds no complete frame.
    pub fn next(&mut self) -> Option<Frame> {
        loop {
            /* Find the first plausible start byte. */
            let Some(start) = self.buf.iter().position(|b| *b == STX) else {
                self.stats.discarded_bytes += self.buf.len() as u64;
                self.buf.clear();
                return None;
            };
            if start > 0 {
                self.stats.discarded_bytes += start as u64;
                self.buf.drain(..start);
            }

            if self.buf.len() < HEADER_LEN {
                return None; // need more header bytes
            }
            let len = self.buf[1] as usize;
            let incompat = self.buf[2];
            let sig = if incompat & INCOMPAT_SIGNED != 0 { SIGNATURE_LEN } else { 0 };
            let total = HEADER_LEN + len + TRAILER_LEN + sig;
            if self.buf.len() < total {
                return None; // need more payload bytes
            }

            let msgid = u32::from_le_bytes([self.buf[7], self.buf[8], self.buf[9], 0]);
            if spec_by_id(msgid).is_none() {
                /* We cannot CRC-validate a message we have no CRC_EXTRA for, so
                 * the header's `len` is unverified — and under EMI a garbage
                 * header claiming a long payload would swallow the real frame
                 * behind it.
                 *
                 * The compromise: trust the length only when the byte after the
                 * claimed frame is an STX (so the next frame really does start
                 * there), or when the buffer ends exactly there. That is what an
                 * unsupported message from a real autopilot looks like, and it
                 * turns a byte-wise walk through ArduPilot's ~40 streamed types
                 * into one skip each. A garbage header can still only mislead us
                 * onto an STX, where parsing resumes — it can no longer swallow
                 * a frame. Otherwise fall back to dropping a single byte. */
                let resumes_on_frame = self.buf.get(total).map(|b| *b == STX).unwrap_or(true);
                if resumes_on_frame {
                    /* A whole frame of a type we do not decode: routine, and
                     * counted apart from real loss. */
                    self.stats.skipped_messages += 1;
                    self.buf.drain(..total);
                } else {
                    /* The header does not describe a frame that ends where a
                     * frame should — this is noise, so drop one byte. */
                    self.stats.discarded_bytes += 1;
                    self.buf.drain(..1);
                }
                continue;
            }

            match decode(&self.buf[..total]) {
                Ok(frame) => {
                    self.buf.drain(..total);
                    return Some(frame);
                }
                Err(_) => {
                    /* Corrupt: drop the magic byte and rescan, so the next
                     * valid frame behind it is not lost. */
                    self.stats.crc_errors += 1;
                    self.stats.discarded_bytes += 1;
                    self.buf.drain(..1);
                    continue;
                }
            }
        }
    }
}

/* ── frame builders for the messages Sofuu actually sends ─────────────── */

/// HEARTBEAT: type=MAV_TYPE_ONBOARD_CONTROLLER(18), autopilot=MAV_AUTOPILOT_INVALID(8),
/// base_mode=0, system_status=MAV_STATE_ACTIVE(4), mavlink_version=3.
pub fn msg_heartbeat(seq: u8, sysid: u8, compid: u8) -> Frame {
    let mut f = Frame::new(0, seq, sysid, compid).expect("HEARTBEAT is in the table");
    f.set_u8_at(4, 18);
    f.set_u8_at(5, 8);
    f.set_u8_at(6, 0);
    f.set_u8_at(7, 4);
    f.set_u8_at(8, 3);
    f
}

/// COMMAND_LONG with `param1..param7` supplied positionally (the MAVLink order).
#[allow(clippy::too_many_arguments)]
pub fn msg_command_long(
    seq: u8,
    sysid: u8,
    compid: u8,
    target_system: u8,
    target_component: u8,
    command: u16,
    params: [f32; 7],
    confirmation: u8,
) -> Frame {
    let mut f = Frame::new(76, seq, sysid, compid).expect("COMMAND_LONG is in the table");
    for (i, p) in params.iter().enumerate() {
        f.set_f32_at(i * 4, *p);
    }
    f.set_u16_at(28, command);
    f.set_u8_at(30, target_system);
    f.set_u8_at(31, target_component);
    f.set_u8_at(32, confirmation);
    f
}

/// SET_POSITION_TARGET_GLOBAL_INT — the `goto` message.
#[allow(clippy::too_many_arguments)]
pub fn msg_set_position_target_global_int(
    seq: u8,
    sysid: u8,
    compid: u8,
    target_system: u8,
    target_component: u8,
    lat_e7: i32,
    lon_e7: i32,
    alt_m: f32,
    velocity_ms: [f32; 3],
    yaw_rate_ds: f32,
    type_mask: u16,
    coordinate_frame: u8,
) -> Frame {
    let mut f = Frame::new(86, seq, sysid, compid).expect("msg 86 is in the table");
    f.set_u32_at(0, 0); // time_boot_ms — 0 = "now"
    f.set_u32_at(4, lat_e7 as u32);
    f.set_u32_at(8, lon_e7 as u32);
    f.set_f32_at(12, alt_m);
    for (i, v) in velocity_ms.iter().enumerate() {
        f.set_f32_at(16 + i * 4, *v);
    }
    f.set_f32_at(28, 0.0);
    f.set_f32_at(32, 0.0);
    f.set_f32_at(36, 0.0);
    f.set_f32_at(40, 0.0); // yaw
    f.set_f32_at(44, yaw_rate_ds);
    f.set_u16_at(48, type_mask);
    f.set_u8_at(50, target_system);
    f.set_u8_at(51, target_component);
    f.set_u8_at(52, coordinate_frame);
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_published_check_value() {
        // MAVLink's checksum is the CRC-16/MCRF4XX variant of X.25: poly
        // 0x1021, init 0xFFFF, reflected, and — unlike classic X.25 — NO final
        // xor. Its published check value for "123456789" is 0x6F91 (X.25's
        // 0x906E is the xorout=0xFFFF variant and would be the wrong pin).
        let mut crc: u16 = 0xFFFF;
        for b in b"123456789" {
            crc = crc_accumulate(crc, *b);
        }
        assert_eq!(crc, 0x6F91);
    }

    #[test]
    fn crc_extra_table_is_pinned() {
        // (id, name, len, crc_extra) — from mavlink/c_library_v2 common.h and
        // the per-message headers. A drift here silently makes every frame
        // unreadable by real firmware, so the table is asserted, not trusted.
        let expected: &[(u32, &str, usize, u8)] = &[
            (0, "HEARTBEAT", 9, 50),
            (1, "SYS_STATUS", 43, 124),
            (24, "GPS_RAW_INT", 52, 24),
            (30, "ATTITUDE", 28, 39),
            (33, "GLOBAL_POSITION_INT", 28, 104),
            (40, "MISSION_REQUEST", 5, 230),
            (44, "MISSION_COUNT", 9, 221),
            (65, "RC_CHANNELS", 42, 118),
            (73, "MISSION_ITEM_INT", 38, 38),
            (75, "COMMAND_INT", 35, 158),
            (76, "COMMAND_LONG", 33, 152),
            (77, "COMMAND_ACK", 10, 143),
            (84, "SET_POSITION_TARGET_LOCAL_NED", 53, 143),
            (86, "SET_POSITION_TARGET_GLOBAL_INT", 53, 5),
            (253, "STATUSTEXT", 54, 83),
        ];
        assert_eq!(MESSAGES.len(), expected.len(), "table size drifted");
        for (id, name, len, crc) in expected {
            let spec = spec_by_id(*id).unwrap_or_else(|| panic!("missing msgid {id}"));
            assert_eq!(spec.name, *name);
            assert_eq!(spec.len, *len, "{name} payload len");
            assert_eq!(spec.crc_extra, *crc, "{name} CRC_EXTRA");
        }
    }

    #[test]
    fn heartbeat_round_trips() {
        let f = msg_heartbeat(7, 255, 191);
        let bytes = encode(&f).unwrap();
        assert_eq!(bytes[0], STX);
        assert_eq!(bytes[1], 9); // v2 keeps a 9-byte payload (tail is non-zero)
        let back = decode(&bytes).unwrap();
        assert_eq!(back.msgid, 0);
        assert_eq!(back.seq, 7);
        assert_eq!(back.sysid, 255);
        assert_eq!(back.compid, 191);
        assert_eq!(back.payload.len(), 9);
        assert_eq!(back.u8_at(4), 18, "MAV_TYPE_ONBOARD_CONTROLLER");
        assert_eq!(back.u8_at(8), 3, "mavlink_version");
    }

    #[test]
    fn trailing_zeros_are_trimmed_and_restored() {
        // COMMAND_LONG with only a command number set: everything after byte 30
        // is zero, so the wire payload is shorter than the 33-byte spec len.
        let mut f = Frame::new(76, 0, 255, 191).unwrap();
        f.set_u16_at(28, cmd::NAV_RETURN_TO_LAUNCH);
        let bytes = encode(&f).unwrap();
        assert!(bytes[1] < 33, "payload must be truncated, got {}", bytes[1]);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.payload.len(), 33, "receiver zero-extends to spec len");
        assert_eq!(back.u16_at(28), cmd::NAV_RETURN_TO_LAUNCH);
    }

    #[test]
    fn all_zero_payload_trims_to_nothing_and_still_decodes() {
        let f = Frame::new(76, 0, 255, 191).unwrap();
        let bytes = encode(&f).unwrap();
        assert_eq!(bytes[1], 0);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.payload.len(), 33);
        assert_eq!(back.u16_at(28), 0);
    }

    #[test]
    fn mission_ids_are_the_dialect_ids_not_the_plan_draft() {
        // The plan's first table listed MISSION_ITEM_INT as 38 and
        // MISSION_REQUEST as 43; those are MISSION_WRITE_PARTIAL_LIST and
        // MISSION_REQUEST_LIST. Pinned so the correction cannot regress.
        assert_eq!(spec_by_name("MISSION_ITEM_INT").unwrap().id, 73);
        assert_eq!(spec_by_name("MISSION_REQUEST").unwrap().id, 40);
        assert!(spec_by_id(38).is_none());
        assert!(spec_by_id(43).is_none());
    }

    #[test]
    fn name_lookup_is_case_insensitive() {
        assert_eq!(spec_by_name("heartbeat").unwrap().id, 0);
        assert_eq!(spec_by_name(" AtTiTuDe ").unwrap().id, 30);
        assert!(spec_by_name("PARAM_SET").is_none());
    }

    #[test]
    fn bad_crc_is_refused() {
        let mut bytes = encode(&msg_heartbeat(1, 255, 191)).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        assert_eq!(decode(&bytes), Err(MavError::BadCrc));
    }

    #[test]
    fn payload_corruption_is_refused() {
        let mut bytes = encode(&msg_command_long(1, 255, 191, 1, 1, cmd::NAV_TAKEOFF, [0.0; 7], 0)).unwrap();
        let n = bytes.len();
        bytes[n - 5] ^= 0x01; // flip a payload bit, leave the CRC
        assert_eq!(decode(&bytes), Err(MavError::BadCrc));
    }

    #[test]
    fn unknown_message_never_decodes() {
        // Hand-build a frame for msgid 999: structurally valid, but its
        // payload cannot be CRC-validated, so it must not become a Frame.
        let mut bytes = vec![STX, 0, 0, 0, 0, 255, 191, 0xE7, 0x03, 0x00];
        bytes.extend_from_slice(&[0xAA, 0xBB]);
        assert_eq!(decode(&bytes), Err(MavError::UnknownMessage));
    }

    #[test]
    fn signed_frames_are_parsed_and_accepted() {
        // An ArduPilot FC with signing enabled appends a 13-byte signature.
        // Sofuu does not sign, but must not drop that telemetry (§4.1).
        let f = msg_heartbeat(3, 1, 1);
        let plain = encode(&f).unwrap();
        let mut signed = plain.clone();
        signed[2] = INCOMPAT_SIGNED;
        // Recompute the CRC over the bytes as they now stand — the incompat
        // byte is inside the checksummed region, so reusing the unsigned
        // helper would produce a frame firmware rejects.
        let len = signed[1] as usize;
        let mut crc: u16 = 0xFFFF;
        for b in &signed[1..HEADER_LEN + len] {
            crc = crc_accumulate(crc, *b);
        }
        crc = crc_accumulate(crc, spec_by_id(0).unwrap().crc_extra);
        signed.truncate(HEADER_LEN + len);
        signed.extend_from_slice(&crc.to_le_bytes());
        signed.extend_from_slice(&[0u8; SIGNATURE_LEN]);
        let back = decode(&signed).unwrap();
        assert_eq!(back.msgid, 0);
        assert_eq!(back.sysid, 1);
    }

    #[test]
    fn position_target_type_mask_bits_match_the_published_values() {
        /* These are the numbers from MAVLink's POSITION_TARGET_TYPEMASK_* and
         * from ArduPilot's own autotest (`MAV_POS_TARGET_TYPE_MASK`). Getting
         * them wrong is silent: a bad "position only" mask commanded zero
         * acceleration and made every waypoint a no-op. */
        use super::type_mask::*;
        assert_eq!((X_IGNORE, Y_IGNORE, Z_IGNORE), (1, 2, 4));
        assert_eq!((VX_IGNORE, VY_IGNORE, VZ_IGNORE), (8, 16, 32));
        assert_eq!((AX_IGNORE, AY_IGNORE, AZ_IGNORE), (64, 128, 256));
        assert_eq!((FORCE_SET, YAW_IGNORE, YAW_RATE_IGNORE), (512, 1024, 2048));
        assert_eq!(IGNORE_POSITION, 7);
        assert_eq!(IGNORE_VELOCITY, 56);
        assert_eq!(IGNORE_ACCELERATION, 448);
        /* 3576 == MAV_POS_TARGET_TYPE_MASK.POS_ONLY in ArduPilot's
         * Tools/autotest/vehicle_test_suite.py. */
        assert_eq!(POS_ONLY, 3576);
        assert_eq!(VEL_ONLY, 3527);
        /* POS_ONLY must not ask for force, and must leave position unmasked. */
        assert_eq!(POS_ONLY & FORCE_SET, 0);
        assert_eq!(POS_ONLY & IGNORE_POSITION, 0);
    }

    #[test]
    fn parser_resyncs_through_noise() {
        let mut p = Parser::new();
        let good = encode(&msg_heartbeat(9, 255, 191)).unwrap();
        // Noise, a good frame, more noise with a fake magic byte, a second good frame.
        p.push(&[0x00, 0xFF, 0x12]);
        p.push(&good);
        p.push(&[0xFD, 0x01, 0x02]); // truncated/garbage "header"
        p.push(&good);

        let first = p.next().expect("first frame");
        assert_eq!(first.seq, 9);
        let second = p.next().expect("second frame survives the noise");
        assert_eq!(second.seq, 9);
        assert!(p.next().is_none());
        let st = p.stats();
        assert!(st.discarded_bytes >= 3, "noise counted: {st:?}");
    }

    #[test]
    fn an_unsupported_message_is_skipped_in_one_step_and_does_not_eat_the_next_frame() {
        /* ArduPilot streams ~40 message types this link does not decode. Each
         * one used to cost a byte-wise walk (a resync storm on the first live
         * SITL run); each one must now be skipped in a single step. */
        let m1 = encode(&msg_heartbeat(1, 255, 191)).unwrap();
        let m2 = encode(&msg_heartbeat(2, 255, 191)).unwrap();
        let mut unknown = vec![STX, 0x0C, 0x00, 0x00, 0x07, 0x01, 0x01];
        unknown.extend_from_slice(&[0x88, 0x88, 0x88]); // msgid 0x888888 — not in the table
        unknown.extend_from_slice(&[0u8; 12]);
        unknown.extend_from_slice(&[0xAB, 0xCD]); // its (unverifiable) CRC

        let mut p = Parser::new();
        p.push(&m1);
        p.push(&unknown);
        p.push(&m2);

        assert_eq!(p.next().map(|f| f.seq), Some(1));
        assert_eq!(p.next().map(|f| f.seq), Some(2), "the frame behind the unknown one survived");
        assert!(p.next().is_none());
        /* One skip, and it is not counted as loss. */
        assert_eq!(p.stats().skipped_messages, 1, "{:?}", p.stats());
        assert_eq!(p.stats().discarded_bytes, 0, "{:?}", p.stats());
    }

    #[test]
    fn a_garbage_length_cannot_desynchronise_the_stream() {
        /* A noise header claiming a long payload. Two outcomes are acceptable
         * and both are safe: the byte after the claimed frame is an STX, so the
         * skip lands on a frame; or it is not, and we fall back to the
         * byte-wise walk. What must never happen is that the stream ends up
         * wedged — so the frame behind the noise is still found, and nothing is
         * ever accepted without a valid CRC. */
        let good = encode(&msg_heartbeat(3, 255, 191)).unwrap();
        let mut junk = vec![STX, 30, 0, 0, 1, 2, 3, 0xEE, 0xEE, 0xEE]; // claims 30
        junk.extend_from_slice(&[0x11; 32]); // lands at offset 42, which is not STX
        junk.extend_from_slice(&good);

        let mut p = Parser::new();
        p.push(&junk);
        let f = p.next().expect("the frame behind the noise is still found");
        assert_eq!(f.seq, 3);
        assert!(p.next().is_none());
    }

    #[test]
    fn parser_drops_one_byte_and_recovers_from_a_corrupt_frame() {
        let good = encode(&msg_heartbeat(1, 255, 191)).unwrap();
        let mut corrupt = encode(&msg_heartbeat(2, 255, 191)).unwrap();
        let n = corrupt.len();
        corrupt[n - 1] ^= 0xFF;

        let mut p = Parser::new();
        p.push(&corrupt);
        p.push(&good);

        let f = p.next().expect("the good frame behind a corrupt one");
        assert_eq!(f.seq, 1);
        assert_eq!(p.stats().crc_errors, 1);
    }

    #[test]
    fn parser_skips_unknown_messages_without_losing_the_next_frame() {
        // A well-formed frame for msgid 999 (outside the subset).
        let mut unknown = vec![STX, 2, 0, 0, 0, 255, 191, 0xE7, 0x03, 0x00, 0x11, 0x22, 0x33, 0x44];
        unknown.truncate(HEADER_LEN + 2 + TRAILER_LEN);
        let good = encode(&msg_heartbeat(5, 255, 191)).unwrap();
        let mut p = Parser::new();
        p.push(&unknown);
        p.push(&good);
        let f = p.next().expect("frame after the unknown message");
        assert_eq!(f.seq, 5);
        // Skipped whole, in one step, and counted as a skip rather than as
        // loss — the header is unverifiable, but the byte after the frame is an
        // STX, which is what an unsupported message from a real autopilot looks
        // like (ArduPilot streams ~40 such types).
        assert_eq!(p.stats().skipped_messages, 1);
        assert_eq!(p.stats().discarded_bytes, 0);
    }

    #[test]
    fn parser_needs_more_bytes_before_emitting() {
        let bytes = encode(&msg_heartbeat(0, 255, 191)).unwrap();
        let mut p = Parser::new();
        p.push(&bytes[..4]);
        assert!(p.next().is_none());
        p.push(&bytes[4..]);
        assert!(p.next().is_some());
    }

    #[test]
    fn parser_handles_byte_at_a_time_feeding() {
        let bytes = encode(&msg_heartbeat(4, 255, 191)).unwrap();
        let mut p = Parser::new();
        let mut got = None;
        for b in &bytes {
            p.push(&[*b]);
            if let Some(f) = p.next() {
                got = Some(f);
            }
        }
        assert_eq!(got.expect("frame from byte-at-a-time stream").seq, 4);
        assert_eq!(p.stats().discarded_bytes, 0);
    }

    #[test]
    fn set_position_target_global_int_round_trips_fields() {
        let f = msg_set_position_target_global_int(
            11, 255, 191, 1, 1, 521_234_567, 41_234_567, 25.0, [1.0, 2.0, -0.5], 0.0,
            0b1100_1111_1000, frame_id::GLOBAL_RELATIVE_ALT_INT,
        );
        let back = decode(&encode(&f).unwrap()).unwrap();
        assert_eq!(back.i32_at(4), 521_234_567);
        assert_eq!(back.i32_at(8), 41_234_567);
        assert_eq!(back.f32_at(12), 25.0);
        assert_eq!(back.f32_at(16), 1.0);
        assert_eq!(back.f32_at(24), -0.5);
        assert_eq!(back.u16_at(48), 0b1100_1111_1000);
        assert_eq!(back.u8_at(52), frame_id::GLOBAL_RELATIVE_ALT_INT);
    }

    #[test]
    fn field_accessors_never_panic_past_the_end() {
        let f = Frame::new(40, 0, 1, 1).unwrap(); // MISSION_REQUEST, 5 bytes
        assert_eq!(f.u8_at(4), 0);
        assert_eq!(f.u8_at(999), 0);
        assert_eq!(f.u16_at(999), 0);
        assert_eq!(f.f32_at(999), 0.0);
    }

    #[test]
    fn encoder_refuses_an_unsupported_message() {
        let f = Frame { seq: 0, sysid: 1, compid: 1, msgid: 999, payload: vec![] };
        assert_eq!(encode(&f), Err(MavError::UnknownMessage));
    }

    #[test]
    fn classify_puts_everything_that_can_move_a_vehicle_in_actuator() {
        for spec in MESSAGES {
            let expected = match spec.name {
                "HEARTBEAT" => Class::Heartbeat,
                "SYS_STATUS" | "GPS_RAW_INT" | "ATTITUDE" | "GLOBAL_POSITION_INT"
                | "RC_CHANNELS" | "STATUSTEXT" | "COMMAND_ACK" => Class::Telemetry,
                _ => Class::Actuator,
            };
            assert_eq!(spec.class, expected, "{} class", spec.name);
        }
    }
}
