//! cargo-fuzz target for the MAVLink codec + hardware gate (PLAN-DRONES §6.5).
//!
//!   cd crates/fuzz && cargo fuzz run mavlink -- -max_len=512 -timeout=5
//!
//! Two properties, and the second is the one that matters:
//!
//! 1. **Never panic.** Arbitrary bytes from an EMI-corrupted UART must be
//!    scanned, discarded and resynchronised — never unwrapped, indexed past,
//!    or allowed to abort the companion computer.
//! 2. **Never let a denied frame through.** Whatever the fuzzer builds, if the
//!    gate refuses it the send path must produce no bytes. A fuzz find here
//!    would be a safety hole, not a crash.

#![no_main]

use libfuzzer_sys::fuzz_target;
use sofuu_core::rt::mavlink::{self, Frame};
use sofuu_core::rt::mavlink_gate::{
    self as gate, GateState, HardwarePolicy, SendRequest,
};

fuzz_target!(|data: &[u8]| {
    /* ── 1. the stream parser survives arbitrary noise ─────────────── */
    let mut parser = mavlink::Parser::new();
    parser.push(data);
    let mut decoded = 0;
    while let Some(frame) = parser.next() {
        decoded += 1;
        // Field accessors are total: they must not panic past the payload.
        let _ = frame.u8_at(0);
        let _ = frame.u32_at(60);
        let _ = frame.f32_at(999);
        let _ = frame.name();
        let _ = frame.class();
        // Re-encoding a decoded frame must round-trip through the codec.
        if let Ok(bytes) = mavlink::encode(&frame) {
            let _ = mavlink::decode(&bytes);
        }
        if decoded > 64 {
            break; // a pathological input must not spin forever
        }
    }
    let _ = parser.stats();

    /* ── 2. the gate refuses rather than letting anything through ──── */
    let policy = HardwarePolicy::from_policy(&serde_json::json!({
        "hardware": {
            "enabled": true,
            "devices": { "mode": "allowlist",
                         "allow": [{ "path": "udp:127.0.0.1:14550", "class": "mavlink-udp" }] },
            "mavlink": { "mode": "allowlist",
                         "send": ["HEARTBEAT", "COMMAND_*", "SET_POSITION_TARGET_*"],
                         "deny": ["PARAM_SET", "MAV_CMD_DO_MOTOR_TEST"],
                         "rateLimits": { "HEARTBEAT": "1Hz" } },
            "bounds": { "maxSpeedMs": 5, "maxAltRelM": 30, "maxYawRateDs": 45, "geofenceRadiusM": 200 }
        }
    }));

    // Build a frame out of the fuzz bytes: a real msgid, a payload shaped from
    // the input. This exercises the bounds/command paths with adversarial values.
    let ids = [0u32, 75, 76, 84, 86, 73, 44, 77];
    let msgid = ids[(data.first().copied().unwrap_or(0) as usize) % ids.len()];
    let Ok(mut frame) = Frame::new(msgid, 0, 255, 191) else { return };
    for (i, b) in data.iter().take(frame.payload.len()).enumerate() {
        frame.payload[i] = *b;
    }

    let mut state = GateState::default();
    let device = if data.len() % 2 == 0 { "udp:127.0.0.1:14550" } else { "udp:10.0.0.1:14550" };
    let verdict = gate::gate_send(
        &policy,
        &mut state,
        &SendRequest { device, frame: &frame },
        0,
        false,
    );

    // The invariant: a refusal means the send path would write nothing.
    if !verdict.allowed {
        assert!(!verdict.code.is_empty(), "a refusal must carry a §1.4 code");
        assert!(!verdict.rule.is_empty(), "a refusal must name the rule");
        // A denied frame must never be encodable through LinkCore::send.
        let mut core = sofuu_core::rt::mavlink_link::LinkCore::new(device, policy.clone());
        assert!(
            core.send(&frame, 0, false).is_err(),
            "gate denied {msgid} but the send path produced bytes"
        );
    }

    // Commands no profile may ever allow must be denied even when the send
    // list would match them.
    if let Some(raw) = data.get(1) {
        let mut f = Frame::new(76, 0, 255, 191).unwrap();
        f.set_u16_at(28, if *raw % 2 == 0 { mavlink::cmd::PARAM_SET } else { mavlink::cmd::DO_MOTOR_TEST });
        let mut st = GateState::default();
        let v = gate::gate_send(
            &policy,
            &mut st,
            &SendRequest { device: "udp:127.0.0.1:14550", frame: &f },
            0,
            false,
        );
        assert!(!v.allowed, "a permanently denied command was allowed");
        assert_eq!(v.rule, "R-HW/deny/always");
    }
});
