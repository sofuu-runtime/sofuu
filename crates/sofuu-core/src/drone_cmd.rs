// drone_cmd.rs — `sofuu drone`: the MAVLink link from the command line.
//
// PLAN-DRONES-MAVLINK §4.6 (L1: the operator's first contact) and §5 (the $0
// development loop). No QuickJS boot, no model, no network beyond the link:
//
//   sofuu drone observe [--device udp:host:port] [--seconds N] [--json]
//   sofuu drone status  [--device …]          # one-shot telemetry snapshot
//   sofuu drone send    --command NAV_RETURN_TO_LAUNCH [--device …] [--arg 1=…]
//   sofuu drone why     <MESSAGE|MAV_CMD_NAME> [--device …]
//   sofuu drone init    <name> [--dir <path>] # scaffold policy + example
//
// Every outgoing frame crosses `rt/mavlink_gate.rs`; there is no flag that
// bypasses it. A refusal prints the typed verdict (code + rule + reason) and
// exits non-zero, which is what makes `sofuu drone send` usable in a script.

use std::io::Write;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::rt::mavlink::{self, Frame};
use crate::rt::drone_ops;
use crate::rt::mavlink_gate::{self as gate, HardwarePolicy, Verdict};
use crate::rt::mavlink_link::{self as link, Event, LinkCore};

pub fn print_help() {
    println!(
        "Usage: sofuu drone <command> [options]\n\n\
         Fly a vehicle through the R7 hardware firewall. Every frame is judged\n\
         by the policy's `hardware` section before a byte reaches a wire.\n\n\
         Commands:\n\
         \x20 observe [--device D] [--seconds N] [--json]\n\
         \x20             Receive telemetry and emit the 1 Hz heartbeat.\n\
         \x20 status  [--device D]\n\
         \x20             One-shot telemetry snapshot, then exit.\n\
         \x20 send    --command <MAV_CMD> [--device D] [--arg N=V]…\n\
         \x20             Send one gated command; prints the typed verdict.\n\
         \x20 mission <file.json> [--dry-run]\n         \x20             Pre-flight the whole mission, then fly it step by step.\n         \x20 why     <MESSAGE|MAV_CMD> [--device D]\n\
         \x20             Explain what the gate would decide, without sending.\n\
         \x20 init    <name> [--dir <path>]\n\
         \x20             Scaffold a policy template + a working example.\n\n\
         Options shared by every verb:\n\
         \x20 --device D   udp:host:port or /dev/ttyAMA0 (default: first devices.allow entry)\n\
         \x20 --policy P   explicit policy file (default: ~/.sofuu/policy.json + ./.sofuu/policy.json)\n"
    );
}

/// One parsed command line: `--flag value` pairs plus positionals.
struct Args {
    positional: Vec<String>,
    flags: std::collections::HashMap<String, String>,
    multi: std::collections::HashMap<String, Vec<String>>,
}

impl Args {
    fn parse(raw: &[String]) -> Args {
        let mut a = Args { positional: Vec::new(), flags: Default::default(), multi: Default::default() };
        let mut i = 0;
        while i < raw.len() {
            let s = &raw[i];
            if let Some(name) = s.strip_prefix("--") {
                let value = raw.get(i + 1).filter(|v| !v.starts_with("--")).cloned();
                match value {
                    Some(v) => {
                        a.multi.entry(name.to_string()).or_default().push(v.clone());
                        a.flags.insert(name.to_string(), v);
                        i += 2;
                    }
                    None => {
                        a.flags.insert(name.to_string(), "true".into());
                        i += 1;
                    }
                }
            } else {
                a.positional.push(s.clone());
                i += 1;
            }
        }
        a
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(String::as_str)
    }

    fn all(&self, k: &str) -> Vec<String> {
        self.multi.get(k).cloned().unwrap_or_default()
    }

    fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }
}

/// Load the policy. Fail closed: an unreadable or invalid policy is an error
/// with the reason, never an empty policy that would disable the gate.
fn load_policy(args: &Args) -> Result<(HardwarePolicy, Value), String> {
    let doc = match args.get("policy") {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read policy {path}: {e}"))?;
            serde_json::from_str::<Value>(&text).map_err(|e| format!("policy {path} is not JSON: {e}"))?
        }
        None => crate::rt::policy::layered_policy_from_disk()?,
    };
    let resolved = HardwarePolicy::from_policy(&doc);
    if !resolved.enabled {
        return Err(
            "no `hardware` section is enabled in the policy — the drone surface does not exist. \
             Run `sofuu drone init <name>` for a template."
                .into(),
        );
    }
    Ok((resolved, doc))
}

fn pick_device(args: &Args, policy: &HardwarePolicy) -> Result<String, String> {
    if let Some(d) = args.get("device") {
        return Ok(d.to_string());
    }
    policy
        .devices
        .first()
        .map(|r| r.path.clone())
        .ok_or_else(|| "no --device given and the policy's devices.allow is empty".to_string())
}

/// Append gate decisions to `.sofuu/logs/hardware.jsonl` (§4.3 audit trail).
fn flush_audit(core: &mut LinkCore) {
    let records = core.take_audit();
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
    for r in records {
        let _ = writeln!(f, "{r}");
    }
}

fn now_ms(since: Instant) -> u64 {
    since.elapsed().as_millis() as u64
}

fn print_event(ev: &Event, as_json: bool) {
    if as_json {
        let v = match ev {
            Event::Telemetry { message, data } => json!({"event":"telemetry","message":message,"data":data}),
            Event::CommandAck { command, command_name, result, result_name, text } => json!({
                "event":"ack","command":command,"command_name":command_name,
                "result":result,"result_name":result_name,"text":text
            }),
            Event::ArmingDenied { command, text } => json!({
                "event":"arming_denied","code":"E_ARMING_DENIED","command":command,"text":text
            }),
            Event::StatusText { severity, text } => json!({"event":"statustext","severity":severity,"text":text}),
            Event::Resync { discarded_bytes, crc_errors } => json!({
                "event":"resync","discarded_bytes":discarded_bytes,"crc_errors":crc_errors
            }),
        };
        println!("{v}");
        return;
    }
    match ev {
        Event::Telemetry { message, data } => println!("{message} {data}"),
        Event::CommandAck { command_name, result_name, text, .. } => match text {
            Some(t) => println!("ACK {command_name} {result_name}: {t}"),
            None => println!("ACK {command_name} {result_name}"),
        },
        Event::ArmingDenied { text, .. } => println!("E_ARMING_DENIED {text}"),
        Event::StatusText { severity, text } => println!("STATUSTEXT[{severity}] {text}"),
        Event::Resync { discarded_bytes, crc_errors } => {
            println!("resync (discarded {discarded_bytes} bytes, {crc_errors} crc errors)")
        }
    }
}

fn open_link(device: &str, policy: HardwarePolicy) -> Result<(LinkCore, Box<dyn link::Io>), String> {
    let core = LinkCore::new(device, policy);
    let verdict = core.check_device();
    if !verdict.allowed {
        return Err(format!(
            "{}: {} ({})",
            verdict.code, verdict.reason, verdict.rule
        ));
    }
    let io = link::open_io(device).map_err(|e| format!("cannot open {device}: {e}"))?;
    Ok((core, io))
}

pub fn run(args: &[String]) -> i32 {
    if args.is_empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
        print_help();
        return 0;
    }
    let verb = args[0].as_str();
    let a = Args::parse(&args[1..]);
    match verb {
        "observe" => cmd_observe(&a),
        "status" => cmd_status(&a),
        "send" => cmd_send(&a),
        "why" => cmd_why(&a),
        "mission" => cmd_mission(&a),
        "init" => cmd_init(&a),
        other => {
            eprintln!("sofuu drone: unknown command '{other}' (try `sofuu drone help`)");
            1
        }
    }
}

/// Stream telemetry (and heartbeat) until `--seconds` elapses or Ctrl-C.
fn cmd_observe(a: &Args) -> i32 {
    let (policy, _doc) = match load_policy(a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone observe: {e}");
            return 1;
        }
    };
    let device = match pick_device(a, &policy) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("sofuu drone observe: {e}");
            return 1;
        }
    };
    let (mut core, mut io) = match open_link(&device, policy) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone observe: {e}");
            return 1;
        }
    };
    let seconds: u64 = a.get("seconds").and_then(|s| s.parse().ok()).unwrap_or(30);
    let as_json = a.has("json");
    eprintln!(
        "sofuu drone: {} ({}) for {}s — mode {}",
        device,
        if as_json { "json" } else { "text" },
        seconds,
        match core.policy.mavlink_mode {
            gate::MavlinkMode::Off => "off",
            gate::MavlinkMode::Observe => "observe",
            gate::MavlinkMode::Allowlist => "allowlist",
        }
    );

    let started = Instant::now();
    let mut last_tick = Instant::now();
    let mut saw_telemetry = false;
    while started.elapsed() < Duration::from_secs(seconds) {
        let t = now_ms(started);
        match link::pump(&mut core, io.as_mut(), t) {
            Ok(events) => {
                for ev in &events {
                    if matches!(ev, Event::Telemetry { .. }) {
                        saw_telemetry = true;
                    }
                    print_event(ev, as_json);
                }
            }
            Err(e) => {
                eprintln!("sofuu drone observe: read failed: {e}");
                break;
            }
        }
        // Heartbeat cadence is wall-clock, independent of the pump rate.
        if last_tick.elapsed() >= Duration::from_millis(50) {
            last_tick = Instant::now();
            if let Some(bytes) = core.tick(t) {
                let _ = io.write(&bytes);
            }
            flush_audit(&mut core);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    flush_audit(&mut core);
    if !saw_telemetry {
        eprintln!("sofuu drone observe: no telemetry arrived — is the FC/SITL sending?");
        return 2;
    }
    0
}

/// Read for a couple of seconds and print what the vehicle is doing.
fn cmd_status(a: &Args) -> i32 {
    let (policy, _) = match load_policy(a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone status: {e}");
            return 1;
        }
    };
    let device = match pick_device(a, &policy) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("sofuu drone status: {e}");
            return 1;
        }
    };
    let (mut core, mut io) = match open_link(&device, policy) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone status: {e}");
            return 1;
        }
    };
    let started = Instant::now();
    let window = Duration::from_millis(a.get("seconds").and_then(|s| s.parse::<u64>().ok()).unwrap_or(2) * 1000);
    let mut latest: std::collections::HashMap<String, Value> = Default::default();
    while started.elapsed() < window {
        let t = now_ms(started);
        if let Ok(events) = link::pump(&mut core, io.as_mut(), t) {
            for ev in events {
                if let Event::Telemetry { message, data } = ev {
                    latest.insert(message.to_string(), data);
                }
            }
        }
        if let Some(bytes) = core.tick(t) {
            let _ = io.write(&bytes);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    flush_audit(&mut core);

    let mut out = json!({
        "device": device,
        "connected": !latest.is_empty(),
        "mode": format!("{:?}", core.policy.mavlink_mode),
        "gps": {
            "fix_type": core.state.gps_fix_type,
            "hdop": core.state.gps_hdop,
            "home": core.state.home,
        },
        "telemetry": Value::Object(latest.into_iter().collect()),
        "link": io.transport_stats(),
    });
    if let Some(batt) = out["telemetry"].get("SYS_STATUS").cloned() {
        out["battery"] = batt;
    }
    println!("{out}");
    if out["connected"] == false {
        2
    } else {
        0
    }
}

/// Resolve `NAV_RETURN_TO_LAUNCH` / `MAV_CMD_NAV_RETURN_TO_LAUNCH` / `20` to an id.
fn parse_command(raw: &str) -> Option<u16> {
    let s = raw.trim();
    if let Ok(n) = s.parse::<u16>() {
        return Some(n);
    }
    let want = s.trim_start_matches("MAV_CMD_");
    let want = format!("MAV_CMD_{}", want.to_ascii_uppercase());
    gate::ALLOWED_COMMANDS.iter().copied().chain(gate::ALWAYS_DENIED_COMMANDS.iter().map(|(c, _)| *c))
        .find(|c| gate::command_name(*c).map(|n| n.eq_ignore_ascii_case(&want)).unwrap_or(false))
}

fn parse_message(raw: &str) -> Option<&'static mavlink::MsgSpec> {
    mavlink::spec_by_name(raw)
}

fn cmd_send(a: &Args) -> i32 {
    let (policy, _) = match load_policy(a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone send: {e}");
            return 1;
        }
    };
    let device = match pick_device(a, &policy) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("sofuu drone send: {e}");
            return 1;
        }
    };
    let (mut core, mut io) = match open_link(&device, policy) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone send: {e}");
            return 1;
        }
    };

    // `param N=V` (1-based, COMMAND_LONG order) fills the command parameters.
    let mut params = [0.0f32; 7];
    for raw in a.all("param") {
        let Some((k, v)) = raw.split_once('=') else {
            eprintln!("sofuu drone send: --param wants N=V, got {raw}");
            return 1;
        };
        let Ok(idx) = k.trim().parse::<usize>() else {
            eprintln!("sofuu drone send: --param index must be a number, got {k}");
            return 1;
        };
        if !(1..=7).contains(&idx) {
            eprintln!("sofuu drone send: --param index must be 1..7");
            return 1;
        }
        match v.trim().parse::<f32>() {
            Ok(f) => params[idx - 1] = f,
            Err(_) => {
                eprintln!("sofuu drone send: --param value must be a number, got {v}");
                return 1;
            }
        }
    }

    let frame: Frame = if let Some(cmd_raw) = a.get("command") {
        let Some(cmd) = parse_command(cmd_raw) else {
            eprintln!("sofuu drone send: unknown MAV_CMD '{cmd_raw}'");
            return 1;
        };
        link::command_long(&core, 1, 1, cmd, params)
    } else if let Some(msg_raw) = a.get("message") {
        let Some(spec) = parse_message(msg_raw) else {
            eprintln!("sofuu drone send: unknown message '{msg_raw}'");
            return 1;
        };
        match Frame::new(spec.id, core.seq, core.sysid, core.compid) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("sofuu drone send: {e}");
                return 1;
            }
        }
    } else if a.get("command").is_none() && a.get("message").is_none() {
        eprintln!("sofuu drone send: give --command <MAV_CMD> or --message <NAME>");
        return 1;
    } else {
        unreachable!()
    };

    let started = Instant::now();
    if !wait_for_link(&mut core, io.as_mut(), started, 3_000) {
        eprintln!("sofuu drone send: no telemetry from {device} within 3 s — no peer to send to (try udpout:)");
        return 2;
    }
    match link::send_frame(&mut core, io.as_mut(), &frame, now_ms(started), false) {
        Ok(()) => {
            // Wait briefly for the ACK so the operator sees the FC's answer.
            let wait = Duration::from_millis(1_500);
            while started.elapsed() < wait {
                let t = now_ms(started);
                if let Ok(events) = link::pump(&mut core, io.as_mut(), t) {
                    let mut done = false;
                    for ev in &events {
                        print_event(ev, false);
                        if matches!(ev, Event::CommandAck { .. }) {
                            done = true;
                        }
                    }
                    if done {
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            flush_audit(&mut core);
            0
        }
        Err(v) => {
            flush_audit(&mut core);
            eprintln!("{}: {} ({})", v.code, v.reason, v.rule);
            1
        }
    }
}

/// Explain a decision without sending anything — the `policy why` parity the
/// plan asks for, for the frame path.
fn cmd_why(a: &Args) -> i32 {
    let (policy, _) = match load_policy(a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone why: {e}");
            return 1;
        }
    };
    let device = match pick_device(a, &policy) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("sofuu drone why: {e}");
            return 1;
        }
    };
    let Some(what) = a.positional.first() else {
        eprintln!("sofuu drone why: give a message name or MAV_CMD");
        return 1;
    };

    let mut core = LinkCore::new(&device, policy);
    let frame = if let Some(spec) = parse_message(what) {
        Frame::new(spec.id, 0, core.sysid, core.compid).unwrap_or_else(|_| {
            let mut f = Frame::new(76, 0, core.sysid, core.compid).unwrap();
            f.set_u16_at(28, 0);
            f
        })
    } else if let Some(cmd) = parse_command(what) {
        let mut f = Frame::new(76, 0, core.sysid, core.compid).unwrap();
        f.set_u16_at(28, cmd);
        f
    } else {
        eprintln!("sofuu drone why: '{what}' is neither a message nor a MAV_CMD");
        return 1;
    };

    /* A dry run: judge at t=0 with the state the link would have. Note this is
     * the STATIC half — a live link may additionally refuse on GPS freshness,
     * which is why the runtime's answer is authoritative and this is an
     * explanation. */
    let verdict: Verdict = gate::gate_send(
        &core.policy,
        &mut core.state,
        &gate::SendRequest { device: &device, frame: &frame },
        0,
        false,
    );
    println!(
        "{}",
        json!({
            "device": device,
            "message": frame.name(),
            "command": gate::command_of(&frame).and_then(gate::command_name),
            "decision": if verdict.allowed { "allow" } else { "deny" },
            "code": verdict.code,
            "rule": verdict.rule,
            "reason": verdict.reason,
            "bounds": {
                "maxSpeedMs": core.policy.bounds().max_speed_ms,
                "maxAltRelM": core.policy.bounds().max_alt_rel_m,
                "maxYawRateDs": core.policy.bounds().max_yaw_rate_ds,
                "geofenceRadiusM": core.policy.bounds().geofence_radius_m,
            },
        })
    );
    if verdict.allowed {
        0
    } else {
        1
    }
}

/// Wait for the link to come up. With the `udp:` bind-and-learn model we have
/// no peer address until the FC's first datagram, so a mission that fired
/// immediately would fail on the transport rather than on anything real.
/// `udpout:` endpoints are already addressed and return at once.
fn wait_for_link(core: &mut LinkCore, io: &mut dyn link::Io, started: Instant, max_ms: u64) -> bool {
    if core.state.last_telemetry_ms.is_some() {
        return true;
    }
    let deadline = Duration::from_millis(max_ms);
    while started.elapsed() < deadline {
        let t = now_ms(started);
        let _ = link::pump(core, io, t);
        if core.state.last_telemetry_ms.is_some() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

/// Pump until `gone()` is true, an error arrives, or the deadline passes.
/// A mission that aborts while the aircraft is airborne must not simply exit.
/// Left hovering in GUIDED, control passes to the battery failsafe with no
/// operator in the loop — the first live SITL run ended with the vehicle at
/// 9.6 m and the mission process gone. Command RTL **through the gate** before
/// returning, and say so.
fn abort_in_flight(core: &mut LinkCore, io: &mut dyn link::Io, started: Instant) {
    let alt = core.state.last_relative_alt_m.unwrap_or(0.0);
    if alt < 1.0 {
        return;
    }
    eprintln!("abort: vehicle airborne at {alt:.1} m — commanding RTL before exit");
    let frame = mavlink::msg_command_long(
        core.seq,
        core.sysid,
        core.compid,
        1,
        1,
        mavlink::cmd::NAV_RETURN_TO_LAUNCH,
        [0.0; 7],
        0,
    );
    for attempt in 0..4 {
        match link::send_frame(core, io, &frame, now_ms(started), false) {
            Ok(()) => {
                eprintln!("abort: NAV_RETURN_TO_LAUNCH sent");
                return;
            }
            Err(v) => match v.retry_after_ms {
                Some(wait) if attempt < 3 => std::thread::sleep(Duration::from_millis(wait + 10)),
                _ => {
                    eprintln!("abort: RTL REFUSED: {} ({}) — the operator must take over", v.reason, v.rule);
                    return;
                }
            },
        }
    }
}

/// Wait up to `budget` **from now** until `gone` is satisfied.
///
/// The budget is relative to this call, not to the mission's start: the old
/// version compared `started.elapsed()` against the budget, so every ACK wait
/// shorter than the elapsed mission time returned immediately and a healthy
/// vehicle reported `E_TIMEOUT: no COMMAND_ACK` from step three onward.
fn pump_until(
    core: &mut LinkCore,
    io: &mut dyn link::Io,
    started: Instant,
    budget: Duration,
    mut gone: impl FnMut(&LinkCore) -> bool,
) -> bool {
    let until = Instant::now() + budget;
    while Instant::now() < until {
        let t = now_ms(started);
        if let Ok(events) = link::pump(core, io, t) {
            for ev in &events {
                print_event(ev, false);
            }
        }
        if gone(core) {
            return true;
        }
        if let Some(bytes) = core.tick(t) {
            let _ = io.write(&bytes);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

/// `sofuu drone mission <file.json> [--dry-run]` — L3 (§4.6).
///
/// The mission is a static object, so it is judged **in full before the first
/// frame exists**. `--dry-run` prints that verdict and stops; execution still
/// re-checks every frame at the gate (defence in depth).
fn cmd_mission(a: &Args) -> i32 {
    let (policy, _) = match load_policy(a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone mission: {e}");
            return 1;
        }
    };
    let device = match pick_device(a, &policy) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("sofuu drone mission: {e}");
            return 1;
        }
    };
    let Some(path) = a.positional.first() else {
        eprintln!("sofuu drone mission: give a mission .json file");
        return 1;
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("sofuu drone mission: cannot read {path}: {e}");
            return 1;
        }
    };
    let doc: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone mission: {path} is not JSON: {e}");
            return 1;
        }
    };
    let mission = match drone_ops::parse_mission(&doc) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("sofuu drone mission: {e}");
            return 1;
        }
    };

    // Pre-flight: every step, every frame, before anything flies.
    let pf = drone_ops::preflight(&policy, &device, &mission);
    println!("{}", pf.to_json());
    if !pf.ok {
        eprintln!("sofuu drone mission: pre-flight refused the mission — nothing was sent");
        return 1;
    }
    if a.has("dry-run") {
        eprintln!("sofuu drone mission: pre-flight passed (dry run, nothing sent)");
        return 0;
    }

    let (mut core, mut io) = match open_link(&device, policy) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("sofuu drone mission: {e}");
            return 1;
        }
    };
    let started = Instant::now();
    if !wait_for_link(&mut core, io.as_mut(), started, 3_000) {
        flush_audit(&mut core);
        eprintln!(
            "sofuu drone mission: no telemetry from {device} within 3 s — the FC never spoke, \
             so there is no address to command. Check the SITL --out endpoint, or use udpout: \
             if this side must speak first."
        );
        return 2;
    }

    for (i, step) in mission.steps.iter().enumerate() {
        if let drone_ops::Step::Delay { ms } = step {
            eprintln!("[{i}] {} {}ms", step.kind(), ms);
            /* Sleep *with the link live*: a blind sleep stops draining
             * telemetry and stops the 1 Hz heartbeat, and the heartbeat is a
             * dead-man contract (§4.5) — a mission pause is not a reason to
             * let go of the stick. It also stops a backlog building up behind
             * the next step's ACK wait. */
            let until = Instant::now() + Duration::from_millis(*ms);
            while Instant::now() < until {
                let t = now_ms(started);
                if let Ok(events) = link::pump(&mut core, io.as_mut(), t) {
                    for ev in &events {
                        print_event(ev, false);
                    }
                }
                if let Some(b) = core.tick(t) {
                    let _ = io.write(&b);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            continue;
        }
        eprintln!("[{i}] {}", step.kind());
        for frame in drone_ops::skill_frames(&core, step) {
            let cmd = gate::command_of(&frame);
            /* A back-to-back step (mode → arm → takeoff) can land inside the
             * policy's command rate limit. That is a timing fact, not a safety
             * refusal: wait the interval the gate reports and re-send once.
             * Aborting here would make any mission with two commands in the
             * same second unflyable. */
            let mut attempt = 0;
            let sent = loop {
                match link::send_frame(&mut core, io.as_mut(), &frame, now_ms(started), false) {
                    Ok(()) => break true,
                    Err(v) => {
                        attempt += 1;
                        match (v.retry_after_ms, attempt) {
                            (Some(wait), 1) => {
                                eprintln!(
                                    "[{i}] pacing {}: waiting {wait} ms ({})",
                                    frame.name(),
                                    v.rule
                                );
                                let until = Instant::now() + Duration::from_millis(wait + 10);
                                while Instant::now() < until {
                                    let t = now_ms(started);
                                    let _ = link::pump(&mut core, io.as_mut(), t);
                                    std::thread::sleep(Duration::from_millis(5));
                                }
                                let t = now_ms(started);
                                if let Some(b) = core.tick(t) {
                                    let _ = io.write(&b);
                                }
                            }
                            _ => {
                                eprintln!("[{i}] REFUSED {}: {} ({})", frame.name(), v.reason, v.rule);
                                abort_in_flight(&mut core, io.as_mut(), started);
                                flush_audit(&mut core);
                                return 1;
                            }
                        }
                    }
                }
            };
            debug_assert!(sent);
            // Completion-aware: wait for the FC's ACK, then for the physical
            // result the step implies.
            if let Some(cmd) = cmd {
                let acked = pump_until(&mut core, io.as_mut(), started, Duration::from_secs(5), |c| {
                    !c.pending.contains_key(&cmd)
                });
                if !acked {
                    eprintln!("[{i}] E_TIMEOUT: no COMMAND_ACK for {} within 5 s", frame.name());
                    abort_in_flight(&mut core, io.as_mut(), started);
                    flush_audit(&mut core);
                    return 2;
                }
            }
        }
        // "takeoff to 15 m" must actually be at 15 m before the next waypoint,
        // or horizontal commands execute from the grass (§4.6).
        if let drone_ops::Step::Takeoff { alt_m } = step {
            let target = *alt_m;
            let reached = pump_until(&mut core, io.as_mut(), started, Duration::from_secs(30), |c| {
                c.state.last_relative_alt_m.map(|a| (a - target).abs() <= 0.5).unwrap_or(false)
            });
            if !reached {
                eprintln!("[{i}] E_TIMEOUT: altitude never reached {target:.1} m ± 0.5 within 30 s");
                abort_in_flight(&mut core, io.as_mut(), started);
                flush_audit(&mut core);
                return 2;
            }
        }
        /* "go to this point" is not done when the frame is sent. A position
         * target can be accepted by the transport and silently ignored by the
         * autopilot — GUIDED drops them while `position_ok()` is false — and a
         * mission that reports success without the vehicle moving is worse than
         * one that fails loudly. Verify the distance to the target actually
         * closes; a vehicle that never starts moving is E_NO_PROGRESS. */
        /* "go to this point" is not done when the frame is sent. Two things
         * make a single frame insufficient, and both were found in SITL:
         *
         *  - `GUID_TIMEOUT` (3 s) expires a guided target, so one frame moves
         *    the vehicle for three seconds and then it stops in mid-air;
         *  - a position target can be accepted and silently ignored (GUIDED
         *    drops them when it has no usable position), and reporting success
         *    without the vehicle moving is worse than failing loudly.
         *
         * So stream the target until it is actually reached, and fail with
         * E_NO_PROGRESS if the distance never closes. */
        if let drone_ops::Step::Goto { lat, lon, alt_m, .. } = step {
            let target = (*lat, *lon);
            let Some(start_pos) = core.state.last_lat_lon else {
                eprintln!("[{i}] E_NO_POSITION: no position fix to measure goto progress against");
                abort_in_flight(&mut core, io.as_mut(), started);
                flush_audit(&mut core);
                return 2;
            };
            let d0 = drone_ops::ground_distance_m(start_pos, target);
            let mut best = d0;
            let mut reached = false;
            let deadline = Instant::now() + Duration::from_secs(30);
            while Instant::now() < deadline {
                let frame = drone_ops::position_frame(&core, *lat, *lon, *alt_m);
                if let Err(v) = link::send_frame(&mut core, io.as_mut(), &frame, now_ms(started), false) {
                    eprintln!("[{i}] REFUSED {}: {} ({})", frame.name(), v.reason, v.rule);
                    abort_in_flight(&mut core, io.as_mut(), started);
                    flush_audit(&mut core);
                    return 1;
                }
                let t = now_ms(started);
                if let Ok(events) = link::pump(&mut core, io.as_mut(), t) {
                    for ev in &events {
                        print_event(ev, false);
                    }
                }
                if let Some(b) = core.tick(t) {
                    let _ = io.write(&b);
                }
                if let Some(p) = core.state.last_lat_lon {
                    let d = drone_ops::ground_distance_m(p, target);
                    if d < best {
                        best = d;
                    }
                    if d <= 5.0 {
                        reached = true;
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(200)); // 5 Hz
            }
            if !reached {
                let moved = d0 - best;
                eprintln!(
                    "[{i}] E_NO_PROGRESS: {:.1} m of the {:.1} m to the waypoint in 30 s                      (closest {:.1} m){}{}",
                    moved,
                    d0,
                    best,
                    if moved < 5.0 {
                        " — the autopilot is ignoring the position target"
                    } else {
                        ""
                    },
                    if moved < 5.0 { " (wrong type_mask or no usable position)" } else { "" }
                );
                abort_in_flight(&mut core, io.as_mut(), started);
                flush_audit(&mut core);
                return 2;
            }
            eprintln!("[{i}] goto reached ({:.1} m of {:.1} m closed)", d0 - best, d0);
        }
    }
    flush_audit(&mut core);
    println!("{}", json!({"ok": true, "steps": mission.steps.len(), "device": device}));
    0
}

/// `sofuu drone init <name>` — the `sofuu plugin init` pattern pointed at
/// vehicles (§4.6): a policy template, a connect check, and a working example.
fn cmd_init(a: &Args) -> i32 {
    let Some(name) = a.positional.first() else {
        eprintln!("sofuu drone init: give a name, e.g. `sofuu drone init copter-sim`");
        return 1;
    };
    let dir = std::path::PathBuf::from(a.get("dir").unwrap_or(".")).join(name);
    if dir.exists() {
        eprintln!("sofuu drone init: {} already exists", dir.display());
        return 1;
    }
    let policy_dir = dir.join(".sofuu");
    if let Err(e) = std::fs::create_dir_all(&policy_dir) {
        eprintln!("sofuu drone init: {e}");
        return 1;
    }
    let policy = json!({
        "_notes": [
            "PLAN-DRONES-MAVLINK §4.3. This template is SITL-first: the endpoint",
            "is the simulator's UDP output and the mode is observe (receive only).",
            "Change devices.mode/mavlink.mode to allowlist only after reading §4.3."
        ],
        "hardware": {
            "enabled": true,
            "devices": {
                "mode": "allowlist",
                "allow": [{ "path": "udp:127.0.0.1:14550", "class": "mavlink-udp" }]
            },
            "mavlink": {
                "mode": "observe",
                "send": ["HEARTBEAT"],
                "deny": ["PARAM_SET", "MAV_CMD_DO_MOTOR_TEST"],
                "rateLimits": { "COMMAND_*": "10Hz", "SET_POSITION_TARGET_*": "20Hz", "HEARTBEAT": "1Hz" }
            },
            "bounds": { "maxSpeedMs": 15, "maxAltRelM": 120, "maxYawRateDs": 180, "geofenceRadiusM": 500 },
            "approval": "off"
        }
    });
    if let Err(e) = std::fs::write(
        policy_dir.join("policy.json"),
        serde_json::to_string_pretty(&policy).unwrap_or_default() + "\n",
    ) {
        eprintln!("sofuu drone init: {e}");
        return 1;
    }
    let example = r#"// Drone example — PLAN-DRONES-MAVLINK §4.6 (L2).
// Run:  sofuu drone observe --device udp:127.0.0.1:14550
//       sofuu drone status
//
// The JS surface (sofuu.mavlink) and the skill layer land in M-D2/M-D3x;
// until then the CLI above is the working path against SITL.
"#;
    if let Err(e) = std::fs::write(dir.join("drone.js"), example) {
        eprintln!("sofuu drone init: {e}");
        return 1;
    }
    println!("created {}", dir.display());
    println!("  .sofuu/policy.json  — SITL-first hardware policy (observe mode)");
    println!("  drone.js            — the example's entry point");
    println!();
    println!("Next:");
    println!("  1. start SITL:  sim_vehicle.py -v ArduCopter --out=udp:0.0.0.0:14550");
    println!("  2. observe:     sofuu drone observe --policy {}/.sofuu/policy.json", dir.display());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        Args::parse(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn flags_and_positionals_parse() {
        let a = args(&["observe", "--device", "udp:1.2.3.4:14550", "--seconds", "5", "--json"]);
        assert_eq!(a.positional, vec!["observe"]);
        assert_eq!(a.get("device"), Some("udp:1.2.3.4:14550"));
        assert_eq!(a.get("seconds"), Some("5"));
        assert!(a.has("json"));
    }

    #[test]
    fn repeated_params_are_collected() {
        let a = args(&["send", "--param", "1=1", "--param", "7=20"]);
        assert_eq!(a.all("param"), vec!["1=1", "7=20"]);
        // `get` is last-wins (the conventional flag behaviour); only `param`
        // is read through `all`.
        assert_eq!(a.get("param"), Some("7=20"));
    }

    #[test]
    fn command_names_resolve_with_and_without_the_prefix() {
        assert_eq!(parse_command("MAV_CMD_NAV_RETURN_TO_LAUNCH"), Some(mavlink::cmd::NAV_RETURN_TO_LAUNCH));
        assert_eq!(parse_command("NAV_RETURN_TO_LAUNCH"), Some(mavlink::cmd::NAV_RETURN_TO_LAUNCH));
        assert_eq!(parse_command("nav_return_to_launch"), Some(mavlink::cmd::NAV_RETURN_TO_LAUNCH));
        assert_eq!(parse_command("20"), Some(20));
        // A denied command still resolves — the gate is what refuses it, and the
        // operator must be able to ask why.
        assert_eq!(parse_command("MAV_CMD_DO_MOTOR_TEST"), Some(mavlink::cmd::DO_MOTOR_TEST));
        assert_eq!(parse_command("NOPE"), None);
    }

    #[test]
    fn message_names_resolve() {
        assert_eq!(parse_message("COMMAND_LONG").map(|s| s.id), Some(76));
        assert_eq!(parse_message("heartbeat").map(|s| s.id), Some(0));
        assert!(parse_message("PARAM_SET").is_none(), "not in the supported subset");
    }
}
