// script_cmd.rs — `sofuu script`: the Kaze CLI.
//
// PLAN-KAZE-LANGUAGE §3.4. Four verbs, no engine boot: the Kaze pipeline has
// no QuickJS dependency, so `sofuu script check` answers in a few milliseconds
// and can run on a machine that has no model, no config and no network.
//
//   sofuu script run <file> [--cap fam=sim] [--fuel N] [--ticks N] …
//   sofuu script check <file>
//   sofuu script compile <file> [--out f.kazebc] [--disasm]
//   sofuu script repl
//
// Exit codes follow the run: 0 when the script ran, 1 when it did not, and the
// parse/check/compile stages use 1 as well — a script that cannot be checked is
// not a different kind of failure.

use std::io::{BufRead, Write};

use crate::sfs::caps::SimState;
use crate::sfs::vm::Verdict;
use crate::sfs::{self, isa, RunOptions, RunReport};

pub fn print_help() {
    println!(
        "Usage: sofuu script <command> [options]\n\n\
         Kaze ({}) — the capability-gated scripting language for robots, drones\n\
         and edge devices. A Kaze script declares what it needs (needs: drone, log)\n\
         and the host grants a backend for each device family.\n\n\
         Commands:\n\
         \x20 run <file.kaze>      Check, compile and run a script\n\
         \x20 check <file.kaze>    Front end only: parse + check, no execution\n\
         \x20 compile <file.kaze>  Emit the .kazebc module (deterministic bytes)\n\
         \x20 repl                 Interactive session: define routines, then .run\n\
         \x20 version              Print the Kaze language version\n\n\
         run options:\n\
         \x20 --cap <family>[=<device>]  Grant a backend: sim (default),\n\
         \x20                        udp:127.0.0.1:14550, or uart:/dev/ttyAMA0@921600\n\
         \x20 --entry <name>         Routine to run (default: main, or the only one)\n\
         \x20 --arg <name>=<json>    Bind a parameter (repeatable)\n\
         \x20 --fuel <n>             Instruction budget (default 200000)\n\
         \x20 --heap-kb <n>          Budget for lists and records in KiB (default 4096)\n\
         \x20 --ticks <n>            Supervisor ticks after the entry (default 0)\n\
         \x20 --tick-ms <n>          Virtual tick period (default 100)\n\
         \x20 --sim <k=v,k=v>        Initial machine state (battery=15,armed=true)\n\
         \x20 --no-policy            Skip the on-disk R7 policy layer\n\
         \x20 --json                 Machine-readable report\n",
        sfs::VERSION
    );
}

/// Parse `--flag value` pairs into a map; the runner owns validation.
struct Flags {
    positional: Vec<String>,
    caps: Vec<String>,
    /// `family -> device` from `--cap family=device`; empty device means sim.
    backend: Vec<(String, String)>,
    /// `device -> pin` from `--pin` (Kaze v0.2 GPIO).
    pins: Vec<(String, u16)>,
    entry: Option<String>,
    args: serde_json::Map<String, serde_json::Value>,
    fuel: Option<u64>,
    heap_kb: Option<u64>,
    ticks: Option<u32>,
    tick_ms: Option<u64>,
    sim: String,
    out: Option<String>,
    disasm: bool,
    json: bool,
    /// The canonical text trace, uncoloured: what the golden corpus stores.
    plain: bool,
    no_policy: bool,
    want_help: bool,
}

fn parse_flags(argv: &[String]) -> Result<Flags, String> {
    let mut f = Flags {
        positional: Vec::new(),
        caps: Vec::new(),
        backend: Vec::new(),
        pins: Vec::new(),
        entry: None,
        args: serde_json::Map::new(),
        fuel: None,
        heap_kb: None,
        ticks: None,
        tick_ms: None,
        sim: String::new(),
        out: None,
        disasm: false,
        json: false,
        plain: false,
        no_policy: false,
        want_help: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut next = |what: &str| -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", what))
        };
        match a {
            "--cap" => {
                let v = next("--cap")?;
                let (family, device) = match v.split_once('=') {
                    Some((f, d)) => (f.trim().to_string(), d.trim().to_string()),
                    None => (v.trim().to_string(), "sim".to_string()),
                };
                if family.is_empty() {
                    return Err("--cap needs a family (drone=udp:127.0.0.1:14550)".to_string());
                }
                if !f.caps.contains(&family) {
                    f.caps.push(family.clone());
                }
                f.backend.retain(|(f, _)| f != &family);
                f.backend.push((family, device));
            }
            "--pin" => {
                let v = next("--pin")?;
                let (name, raw) = v
                    .split_once('=')
                    .ok_or_else(|| "--pin expects name=<gpio number>".to_string())?;
                let pin: u16 = raw
                    .trim()
                    .parse()
                    .map_err(|_| format!("--pin '{}' is not a GPIO number", raw))?;
                f.pins.push((name.trim().to_string(), pin));
            }
            "--entry" => f.entry = Some(next("--entry")?),
            "--arg" => {
                let v = next("--arg")?;
                let (k, raw) = v
                    .split_once('=')
                    .ok_or_else(|| "--arg expects name=value".to_string())?;
                let parsed = serde_json::from_str::<serde_json::Value>(raw)
                    .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
                f.args.insert(k.to_string(), parsed);
            }
            "--fuel" => {
                let v = next("--fuel")?;
                f.fuel = Some(v.parse().map_err(|_| format!("--fuel '{}' is not a number", v))?);
            }
            "--heap-kb" => {
                let v = next("--heap-kb")?;
                f.heap_kb = Some(v.parse().map_err(|_| format!("--heap-kb '{}' is not a number", v))?);
            }
            "--ticks" => {
                let v = next("--ticks")?;
                f.ticks = Some(v.parse().map_err(|_| format!("--ticks '{}' is not a number", v))?);
            }
            "--tick-ms" => {
                let v = next("--tick-ms")?;
                f.tick_ms = Some(v.parse().map_err(|_| format!("--tick-ms '{}' is not a number", v))?);
            }
            "--sim" => f.sim = next("--sim")?,
            "--out" | "-o" => f.out = Some(next("--out")?),
            "--disasm" => f.disasm = true,
            "--json" => f.json = true,
            "--plain" => f.plain = true,
            "--no-policy" => f.no_policy = true,
            "-h" | "--help" | "help" => f.want_help = true,
            other => f.positional.push(other.to_string()),
        }
        i += 1;
    }
    Ok(f)
}

fn policy_layer(no_policy: bool) -> Option<serde_json::Value> {
    if no_policy {
        return None;
    }
    crate::rt::policy::layered_policy_from_disk().ok()
}

fn options_from(f: &Flags) -> Result<RunOptions, String> {
    let mut sim = SimState::default();
    if !f.sim.is_empty() {
        sim.apply_overrides(&f.sim)?;
    }
    Ok(RunOptions {
        entry: f.entry.clone(),
        args: f.args.clone(),
        grants: f.caps.clone(),
        fuel: f.fuel.unwrap_or(sfs::vm::Limits::default().fuel),
        heap_bytes: f
            .heap_kb
            .map(|kb| (kb as usize).saturating_mul(1024))
            .unwrap_or(sfs::vm::Limits::default().heap_bytes),
        ticks: f.ticks.unwrap_or(0),
        tick_ms: f.tick_ms.unwrap_or(100),
        sim,
        policy: policy_layer(f.no_policy),
        backend: f.backend.clone(),
        pins: f.pins.clone(),
    })
}

/// `sofuu script …`; returns the process exit code.
pub fn run(argv: &[String]) -> i32 {
    let Some(command) = argv.first().map(String::as_str) else {
        print_help();
        return 2;
    };
    let rest = &argv[1..];
    match command {
        "-h" | "--help" | "help" => {
            print_help();
            0
        }
        "version" | "--version" => {
            println!("kaze {}", sfs::VERSION);
            0
        }
        "run" => cmd_run(rest),
        "check" => cmd_check(rest),
        "compile" => cmd_compile(rest),
        "repl" => cmd_repl(rest),
        other => {
            eprintln!("\x1b[31mError:\x1b[0m unknown script command '{}'", other);
            print_help();
            2
        }
    }
}

/// A compiled module ends in `.kazebc`, or starts with the `KAZEBC1` magic
/// (a renamed artifact still runs).
fn is_compiled_module(path: &str) -> bool {
    if path.ends_with(".kazebc") {
        return true;
    }
    let mut head = [0u8; 8];
    let opened = std::fs::File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut head));
    opened.is_ok() && &head == b"KAZEBC1\0"
}

fn cmd_run(argv: &[String]) -> i32 {
    let flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {}", e);
            return 2;
        }
    };
    if flags.want_help {
        print_help();
        return 0;
    }
    let Some(path) = flags.positional.first() else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu script run' needs a .kaze file");
        return 2;
    };
    let opts = match options_from(&flags) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {}", e);
            return 2;
        }
    };
    // A `.kazebc` artifact runs without the front end: deserialize, execute.
    let outcome = if is_compiled_module(path) {
        std::fs::read(path)
            .map_err(|e| sfs::KazeError::new("E_KAZE_IO", format!("cannot read '{}': {}", path, e)))
            .and_then(|bytes| {
                isa::deserialize(&bytes).map_err(|e| sfs::KazeError::new("E_KAZE_BADCODE", e))
            })
            .and_then(|module| sfs::run_module(module, path, &opts))
    } else {
        sfs::run_file(path, &opts)
    };
    match outcome {
        Ok(report) => {
            if flags.json {
                println!("{}", report_json(&report));
            } else if flags.plain {
                print!("{}", report.render_plain());
            } else {
                print_report(&report);
            }
            0
        }
        Err(err) => {
            if flags.json {
                let json = serde_json::json!({
                    "ok": false,
                    "code": err.code,
                    "error": err.message,
                });
                println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
            } else {
                eprintln!("\x1b[31m{}\x1b[0m: {}", err.code, err.message);
            }
            1
        }
    }
}

fn cmd_check(argv: &[String]) -> i32 {
    let flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {}", e);
            return 2;
        }
    };
    if flags.want_help {
        print_help();
        return 0;
    }
    let Some(path) = flags.positional.first() else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu script check' needs a .kaze file");
        return 2;
    };
    match sfs::prepare_file(path) {
        Ok(prepared) => {
            if flags.json {
                let json = serde_json::json!({
                    "ok": true,
                    "items": prepared.module.items.len(),
                    "caps": prepared.module.caps.len(),
                    "consts": prepared.module.consts.len(),
                    "code_bytes": prepared.module.code_bytes(),
                });
                println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
            } else {
                println!(
                    "\x1b[32m✓\x1b[0m {} — {} routines, {} capabilities, {} constants, {} bytes of code",
                    path,
                    prepared.module.items.len(),
                    prepared.module.caps.len(),
                    prepared.module.consts.len(),
                    prepared.module.code_bytes()
                );
            }
            0
        }
        Err(err) => {
            eprintln!("\x1b[31m{}\x1b[0m: {}", err.code, err.message);
            1
        }
    }
}

fn cmd_compile(argv: &[String]) -> i32 {
    let flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {}", e);
            return 2;
        }
    };
    if flags.want_help {
        print_help();
        return 0;
    }
    let Some(path) = flags.positional.first() else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu script compile' needs a .kaze file");
        return 2;
    };
    let prepared = match sfs::prepare_file(path) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("\x1b[31m{}\x1b[0m: {}", err.code, err.message);
            return 1;
        }
    };
    let bytes = isa::serialize(&prepared.module);
    if flags.disasm {
        print!("{}", isa::disassemble(&prepared.module));
    }
    match &flags.out {
        Some(out) => match std::fs::write(out, &bytes) {
            Ok(()) => {
                println!(
                    "\x1b[32m✓\x1b[0m {} → {} ({} bytes, revision-stable)",
                    path,
                    out,
                    bytes.len()
                );
                0
            }
            Err(e) => {
                eprintln!("\x1b[31mE_KAZE_IO\x1b[0m: cannot write '{}': {}", out, e);
                1
            }
        },
        None => {
            println!("{} bytes ({} words)", bytes.len(), prepared.module.code_bytes() / 4);
            0
        }
    }
}

fn print_report(report: &RunReport) {
    for (level, msg) in &report.logs {
        match level.as_str() {
            "error" => eprintln!("\x1b[31merror\x1b[0m: {}", msg),
            "warning" => eprintln!("\x1b[33mwarning\x1b[0m: {}", msg),
            _ => eprintln!("info: {}", msg),
        }
    }
    if !report.entry.is_empty() {
        println!("\x1b[1m{}\x1b[0m → {}", report.entry, report.value);
    }
    if !report.actions.is_empty() {
        println!("actions:");
        for (i, a) in report.actions.iter().enumerate() {
            println!("  {:>2}. {}", i + 1, a);
        }
    }
    if let Some(v) = &report.verdict {
        match v {
            Verdict::Allow(text) => println!("verdict: allow {}", text),
            Verdict::Reject(reason) => println!("verdict: reject — {}", reason),
        }
    }
    println!(
        "{} instructions · {} fuel · {} ticks",
        report.instructions, report.fuel_used, report.ticks
    );
}

fn report_json(report: &RunReport) -> String {
    let verdict = match &report.verdict {
        Some(Verdict::Allow(t)) => serde_json::json!({ "allow": true, "reason": t }),
        Some(Verdict::Reject(r)) => serde_json::json!({ "allow": false, "reason": r }),
        None => serde_json::Value::Null,
    };
    let json = serde_json::json!({
        "ok": true,
        "entry": report.entry,
        "result": report.value,
        "actions": report.actions,
        "logs": report.logs.iter().map(|(l, m)| serde_json::json!({"level": l, "message": m})).collect::<Vec<_>>(),
        "fuel": report.fuel_used,
        "instructions": report.instructions,
        "ticks": report.ticks,
        "verdict": verdict,
    });
    serde_json::to_string_pretty(&json).unwrap_or_default()
}

// ── repl ────────────────────────────────────────────────────────────
//
// A block language has no useful one-line expression prompt, so the REPL is a
// session, not a calculator: you paste a block, it is checked and kept, and
// `.run` executes a routine from the kept source. Everything you define stays
// visible in `.list`, which makes the REPL a usable scratchpad for a mission.

fn cmd_repl(argv: &[String]) -> i32 {
    let flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {}", e);
            return 2;
        }
    };
    let mut needs: Vec<String> = flags.caps.clone();
    let mut grants: Vec<String> = flags.caps.clone();
    let mut body = String::new();
    let stdin = std::io::stdin();
    let mut pending = String::new();

    println!(
        "kaze {} REPL — paste a block, blank line to add it. .help for commands.",
        sfs::VERSION
    );
    loop {
        if pending.is_empty() {
            print!("kaze> ");
        } else {
            print!("  ...> ");
        }
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let trimmed = line.trim_end();
        if pending.is_empty() && trimmed.starts_with('.') {
            match handle_repl_command(trimmed, &mut needs, &mut grants, &body, &flags) {
                ReplFlow::Continue => continue,
                ReplFlow::Exit => break,
            }
        }
        if trimmed.trim().is_empty() {
            if pending.trim().is_empty() {
                continue;
            }
            let candidate = format!("{}{}\n", needs_line(&needs), pending);
            match sfs::prepare(&candidate, "<repl>") {
                Ok(prepared) => {
                    let added = prepared.module.items.len();
                    body.push_str(&pending);
                    println!("\x1b[32m✓\x1b[0m added {} routine(s)", added);
                }
                Err(err) => eprintln!("\x1b[31m{}\x1b[0m: {}", err.code, err.message),
            }
            pending.clear();
            continue;
        }
        pending.push_str(trimmed);
        pending.push('\n');
    }
    0
}

fn needs_line(needs: &[String]) -> String {
    if needs.is_empty() {
        String::new()
    } else {
        format!("needs: {}\n", needs.join(", "))
    }
}

enum ReplFlow {
    Continue,
    Exit,
}

fn handle_repl_command(
    line: &str,
    needs: &mut Vec<String>,
    grants: &mut Vec<String>,
    body: &str,
    flags: &Flags,
) -> ReplFlow {
    let mut parts = line.splitn(2, ' ');
    let cmd = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();
    match cmd {
        ".exit" | ".quit" => return ReplFlow::Exit,
        ".help" => {
            println!(
                ".list                 show the source added so far\n\
                 .needs <a, b>         declare capabilities for this session\n\
                 .cap <family>         grant the sim backend for a family\n\
                 .run <name> [json]    run a routine with optional arguments\n\
                 .clear                forget everything defined so far\n\
                 .exit                 leave"
            );
        }
        ".list" => {
            if body.trim().is_empty() {
                println!("(nothing defined yet)");
            } else {
                print!("{}{}", needs_line(needs), body);
            }
        }
        ".clear" => {
            // `body` is immutable here; the caller owns it. Tell the user the
            // honest behaviour instead of pretending.
            println!("(restart the REPL to clear; the session source is append-only)");
        }
        ".needs" => {
            for n in rest.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                if !needs.iter().any(|x| x == n) {
                    needs.push(n.to_string());
                }
            }
            println!("needs: {}", needs.join(", "));
        }
        ".cap" => {
            for n in rest.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                if !grants.iter().any(|x| x == n) {
                    grants.push(n.to_string());
                }
                if !needs.iter().any(|x| x == n) {
                    needs.push(n.to_string());
                }
            }
            println!("granted: {}", grants.join(", "));
        }
        ".run" => {
            let mut it = rest.splitn(2, ' ');
            let name = it.next().unwrap_or("");
            if name.is_empty() {
                println!("usage: .run <name> [json-args]");
                return ReplFlow::Continue;
            }
            let args = it
                .next()
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
                .and_then(|v| match v {
                    serde_json::Value::Object(m) => Some(m),
                    _ => None,
                })
                .unwrap_or_default();
            let src = format!("{}{}", needs_line(needs), body);
            let opts = RunOptions {
                entry: Some(name.to_string()),
                args,
                grants: grants.clone(),
                fuel: flags.fuel.unwrap_or(sfs::vm::Limits::default().fuel),
                heap_bytes: flags
                    .heap_kb
                    .map(|kb| (kb as usize).saturating_mul(1024))
                    .unwrap_or(sfs::vm::Limits::default().heap_bytes),
                ticks: flags.ticks.unwrap_or(0),
                tick_ms: flags.tick_ms.unwrap_or(100),
                sim: SimState::default(),
                policy: policy_layer(flags.no_policy),
                backend: flags.backend.clone(),
                pins: flags.pins.clone(),
            };
            match sfs::run_source(&src, "<repl>", &opts) {
                Ok(report) => print_report(&report),
                Err(err) => eprintln!("\x1b[31m{}\x1b[0m: {}", err.code, err.message),
            }
        }
        other => println!("unknown command '{}'; .help lists them", other),
    }
    ReplFlow::Continue
}
