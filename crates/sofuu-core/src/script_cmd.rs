// script_cmd.rs — `sofuu script`: the Kaze CLI.
//
// PLAN-KAZE-LANGUAGE §3.4. Four verbs, no engine boot for the front end: the
// Kaze pipeline has no QuickJS dependency, so `sofuu script check` answers in a
// few milliseconds and can run on a machine that has no model, no config and
// no network.
//
// `sofuu script run` boots the JavaScript runtime only for a script (or
// `.kazebc` artifact) that declares `ai` or `tool`: those calls are served by
// the real provider and MCP adapters through the asynchronous Kaze bridge, the
// same coordinator a compiled agent uses. Every other script keeps the
// engine-free synchronous path.
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

/// A CLI failure from either execution path.
///
/// The synchronous runner reports a `KazeError`; the asynchronous bridge runs
/// inside the JavaScript runtime and answers with that runtime's own versioned
/// diagnostic object, which a `KazeError` has no field for. Both keep their
/// real detail instead of being flattened into a string.
enum CliFailure {
    Kaze(sfs::KazeError),
    Bridge {
        code: String,
        message: String,
        diagnostic: Option<serde_json::Value>,
    },
}

impl CliFailure {
    fn kaze(err: sfs::KazeError) -> Self {
        CliFailure::Kaze(err)
    }

    fn bridge(code: &str, message: impl Into<String>) -> Self {
        CliFailure::Bridge {
            code: code.to_string(),
            message: message.into(),
            diagnostic: None,
        }
    }

    fn json(&self) -> serde_json::Value {
        match self {
            CliFailure::Kaze(err) => kaze_error_json(err, ""),
            CliFailure::Bridge { code, message, diagnostic } => serde_json::json!({
                "ok": false,
                "code": code,
                "error": crate::rt::diag_contract::redact_event_text(
                    &crate::rt::diag_contract::truncate_chars(message, 64 * 1024),
                ),
                "diagnostic": diagnostic.clone().unwrap_or(serde_json::Value::Null),
            }),
        }
    }

    fn print(&self) {
        match self {
            CliFailure::Kaze(err) => print_kaze_error(err, ""),
            CliFailure::Bridge { code, message, diagnostic } => match diagnostic {
                Some(view) => eprint!(
                    "{}",
                    crate::rt::diag_contract::Diagnostic::from_json(view).render_human(
                        crate::rt::diag_contract::full_detail_requested(),
                        crate::rt::engine::stderr_wants_color(),
                    )
                ),
                None => eprintln!("\x1b[31m{}\x1b[0m: {}", code, message),
            },
        }
    }
}

/// A script whose first line declares `ai` or `tool` needs the JavaScript
/// runtime: the real provider and MCP adapters live there, and the
/// asynchronous bridge is what suspends the routine for a real call instead of
/// blocking the loop. Every other script keeps the engine-free path.
fn declares_async_capability(needs: &[String]) -> bool {
    needs.iter().any(|name| matches!(sfs::caps::family_of(name), "ai" | "tool"))
}

pub fn print_help() {
    println!(
        "Usage: sofuu script <command> [options]\n\n\
         Kaze ({}) — the capability-gated scripting language for AI agents,\n\
         guardrails, robots, drones, and edge devices. A Kaze script declares\n\
         what it needs (needs: policy, log) and the host grants a backend.\n\n\
         Commands:\n\
         \x20 run <file.kaze>      Check, compile and run a script\n\
         \x20 check <file.kaze>    Front end only: parse + check, no execution\n\
         \x20 compile <file.kaze>  Emit the .kazebc module (deterministic bytes)\n\
         \x20 repl                 Interactive session: define routines, then .run\n\
         \x20 version              Print the Kaze language version\n\n\
         run options:\n\
         \x20 --cap <family>[=<device>]  Grant a family; drone=sim/rover=sim are tests\n\
         \x20                       memory=ephemeral selects per-run CMA storage\n\
         \x20 --policy <file>        Operator R7 policy with tools.allow or tools.deny\n\
         \x20 --pin <name>=<line>  Configure a device pin (live devices disabled)\n\
         \x20 --entry <name>         Routine to run (default: main, or the only one)\n\
         \x20 --arg <name>=<json>    Bind a parameter (repeatable)\n\
         \x20 --config <file>        Load routine arguments from a JSON file\n\
         \x20 --fuel <n>             Instruction budget (default 200000)\n\
         \x20 --heap-kb <n>          Budget for lists and records in KiB (default 4096)\n\
         \x20 --ticks <n>            Supervisor ticks after the entry (default 0)\n\
         \x20 --tick-ms <n>          Virtual tick period (default 100)\n\
         \x20 --sim <k=v,k=v>        Initial machine state (battery=15,armed=true)\n\
         \x20 --no-policy            Skip loading; capability calls are denied\n\
         \x20 --json                 Machine-readable report\n",
        sfs::VERSION
    );
}

/// KZ-011 (AUDIT-2026-11-08): `--help` AFTER a subcommand used to repeat the
/// whole surface, so `sofuu script run --help` never told you what run does
/// or which flags it takes. Each subcommand now answers for itself.
fn print_subcommand_help(sub: &str) {
    match sub {
        "run" => println!(
            "Usage: sofuu script run <file.kaze> [options]\n\n\
             Check, compile and run one script in the shared VM.\n\n\
             Options:\n\
             \x20 --cap <family>[=<device>]  Grant a family; drone=sim/rover=sim are tests\n\
             \x20                            memory=ephemeral selects per-run CMA\n\
             \x20 --entry <name>             Routine to run (default: main)\n\
             \x20 --arg <name>=<json>        Bind a parameter (repeatable)\n\
             \x20 --config <file>            Load routine arguments from JSON\n\
             \x20 --fuel <n>                 Instruction budget (default 200000)\n\
             \x20 --heap-kb <n>              List/record budget in KiB (default 4096)\n\
             \x20 --ticks <n> --tick-ms <n>  Supervisor ticks after the entry\n\
             \x20 --sim <k=v,k=v>            Initial machine state\n\
             \x20 --policy <file>            Operator policy layer\n\
             \x20 --no-policy                Skip loading; capability calls are denied\n\
             \x20 --json                     Machine-readable report\n"
        ),
        "check" => println!(
            "Usage: sofuu script check <file.kaze> [options]\n\n\
             Parse and check only — no execution, no capabilities granted.\n\n\
             Options:\n\
             \x20 --json     Machine-readable report (items, caps, consts, bytes)\n"
        ),
        "compile" => println!(
            "Usage: sofuu script compile <file.kaze> [-o <out.kazebc>] [--disasm]\n\n\
             Emit the deterministic .kazebc module.\n\n\
             Options:\n\
             \x20 -o, --out <path>  Output path (default <file>.kazebc)\n\
             \x20 --disasm          Print the instruction listing\n\
             \x20 --plain           Uncoloured canonical text trace\n"
        ),
        "repl" => println!(
            "Usage: sofuu script repl [options]\n\n\
             Interactive session: define routines, inspect capabilities, then .run.\n\n\
             Commands inside the session: .help  .items  .run <name>  .quit\n"
        ),
        _ => print_help(),
    }
}

/// Parse `--flag value` pairs into a map; the runner owns validation.
struct Flags {
    positional: Vec<String>,
    caps: Vec<String>,
    /// `family -> device` from `--cap family=device`; empty means no backend.
    backend: Vec<(String, String)>,
    /// `device -> pin` from `--pin` (Kaze v0.2 GPIO).
    pins: Vec<(String, u16)>,
    entry: Option<String>,
    config_file: Option<String>,
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
    /// KZ-051/KZ-052: an explicit operator policy file. `sofuu script`
    /// advertised the flag but never parsed it, so `--policy bad.json` became
    /// a positional and the run used the default-allow document.
    policy_file: Option<String>,
    want_help: bool,
}

fn parse_flags(argv: &[String]) -> Result<Flags, String> {
    let mut f = Flags {
        positional: Vec::new(),
        caps: Vec::new(),
        backend: Vec::new(),
        pins: Vec::new(),
        entry: None,
        config_file: None,
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
        policy_file: None,
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
                    Some((f, d)) if !d.trim().is_empty() => (f.trim().to_string(), d.trim().to_string()),
                    Some(_) => return Err("--cap family= needs a backend name".to_string()),
                    None => (v.trim().to_string(), String::new()),
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
            "--config" | "-c" => {
                let path = next("--config")?;
                if path.is_empty() { return Err("--config needs a file path".into()); }
                f.config_file = Some(path);
            }
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
            "--policy" => f.policy_file = Some(next("--policy")?),
            "-h" | "--help" | "help" => f.want_help = true,
            other if other.starts_with("--policy=") => {
                let v = other.trim_start_matches("--policy=").to_string();
                if v.is_empty() {
                    return Err("--policy= needs a file path".to_string());
                }
                f.policy_file = Some(v);
            }
            other if other.starts_with("--config=") => {
                let path = other.trim_start_matches("--config=");
                if path.is_empty() { return Err("--config= needs a file path".into()); }
                f.config_file = Some(path.to_string());
            }
            other => f.positional.push(other.to_string()),
        }
        i += 1;
    }
    Ok(f)
}

/// Why an options build failed, so the caller can tell a usage mistake from a
/// policy that refused the run.
///
/// AUDIT-2026-11-08 KZ-051/KZ-052 (RULES 3.5.4): a policy file that exists but
/// cannot be parsed is FATAL. Dropping it and running with the default-allow
/// document is the failure the audit found: the operator wrote a guard, the
/// runtime ignored it, and the script armed a drone with exit 0.
enum OptErr {
    /// Bad flags or arguments: exit 2, "Error:" prefix.
    Usage(String),
    /// A policy layer exists but does not apply: exit 1, E_CAP_DENIED.
    Policy(String),
}

impl OptErr {
    fn exit_code(&self) -> i32 {
        match self {
            OptErr::Usage(_) => 2,
            OptErr::Policy(_) => 1,
        }
    }
    fn code(&self) -> &'static str {
        match self {
            OptErr::Usage(_) => "E_USAGE",
            OptErr::Policy(_) => "E_CAP_DENIED",
        }
    }
    fn message(&self) -> &str {
        match self {
            OptErr::Usage(msg) | OptErr::Policy(msg) => msg,
        }
    }
}

fn policy_layer(f: &Flags) -> Result<Option<serde_json::Value>, OptErr> {
    if f.no_policy {
        if f.policy_file.is_some() {
            return Err(OptErr::Usage(
                "--policy and --no-policy are mutually exclusive".to_string(),
            ));
        }
        return Ok(None);
    }
    /* KZ-051/KZ-052: an operator policy file is parsed here and a document
     * that cannot be understood is FATAL (exit 1, E_CAP_DENIED). Skipping it
     * is the audited failure: the guard existed, the runtime ran anyway. */
    if let Some(path) = &f.policy_file {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| OptErr::Policy(format!("--policy {path}: {e}")))?;
        let v = crate::rt::policy::parse_policy_strict(&raw)
            .map_err(|e| OptErr::Policy(format!("--policy {path}: not valid JSON: {e}")))?;
        if !v.is_object() {
            return Err(OptErr::Policy(format!(
                "--policy {path}: must be a JSON object"
            )));
        }
        crate::rt::policy::validate_policy(&v)
            .map_err(|e| OptErr::Policy(format!("--policy {path}: {e}")))?;
        return Ok(Some(v));
    }
    crate::rt::policy::layered_kaze_policy_from_disk()
        .map(Some)
        .map_err(OptErr::Policy)
}

fn options_from(f: &Flags) -> Result<RunOptions, OptErr> {
    let mut sim = SimState::default();
    if !f.sim.is_empty() {
        if !f.backend.iter().any(|(family, backend)| backend == "sim"
            && matches!(sfs::caps::family_of(family), "drone" | "rover")) {
            return Err(OptErr::Usage("--sim requires an explicit drone=sim or rover=sim testing backend".into()));
        }
        sim.apply_overrides(&f.sim).map_err(OptErr::Usage)?;
    }
    /* KZ-035 (AUDIT-2026-11-08): the guided E_KAZE_PARAM message offers
     * "or positional argument", so the runner honours it: after the script
     * path, each positional binds to the next routine parameter (by index, the
     * same keys bind_args consults for --arg-less calls). A named --arg always
     * wins over the positional in the same slot. */
    let mut args = f.args.clone();
    for (i, raw) in f.positional.iter().skip(1).enumerate() {
        let key = i.to_string();
        if args.contains_key(&key) {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(raw)
            .unwrap_or_else(|_| serde_json::Value::String(raw.clone()));
        args.insert(key, value);
    }
    Ok(RunOptions {
        entry: f.entry.clone(),
        args,
        ambient_args: true,
        config_path: f.config_file.clone(),
        grants: f.caps.clone(),
        fuel: f.fuel.unwrap_or(sfs::vm::Limits::default().fuel),
        timeout_ms: sfs::RunOptions::default().timeout_ms,
        heap_bytes: f
            .heap_kb
            .map(|kb| (kb as usize).saturating_mul(1024))
            .unwrap_or(sfs::vm::Limits::default().heap_bytes),
        ticks: f.ticks.unwrap_or(0),
        tick_ms: f.tick_ms.unwrap_or(100),
        sim,
        policy: policy_layer(f)?,
        runtime_policy: None,
        backend: f.backend.clone(),
        pins: f.pins.clone(),
        source_name: None,
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

/// Render a Kaze failure through the shared diagnostic contract.
///
/// The reporter never reopens source paths; the caller owns the exit status.
fn print_kaze_error(err: &sfs::KazeError, _path: &str) {
    let diag = err.diagnostic(None);
    let full = crate::rt::diag_contract::full_detail_requested();
    eprint!(
        "{}",
        diag.render_human(full, crate::rt::engine::stderr_wants_color())
    );
}

/// Inspect flag positions without treating an option's value as `--json`.
fn requested_json(argv: &[String]) -> bool {
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--json" { return true; }
        if matches!(argv[i].as_str(), "--cap" | "--pin" | "--entry" | "--arg" |
            "--fuel" | "--heap-kb" | "--ticks" | "--tick-ms" | "--sim" |
            "--out" | "-o" | "--policy" | "--config" | "-c") { i += 1; }
        i += 1;
    }
    false
}

fn report_cli_failure(code: &str, message: &str, json: bool) {
    let err = sfs::KazeError::new(code, message);
    if json {
        println!("{}", serde_json::to_string_pretty(&kaze_error_json(&err, "")).unwrap_or_default());
    } else {
        print_kaze_error(&err, "");
    }
}

/// The machine-readable half of the same failure. `code` and `error` keep
/// their published meaning; `diagnostic` is the additive versioned view.
fn kaze_error_json(err: &sfs::KazeError, _path: &str) -> serde_json::Value {
    let error = crate::rt::diag_contract::redact_event_text(
        &crate::rt::diag_contract::truncate_chars(&err.message, 64 * 1024),
    );
    serde_json::json!({
        "ok": false,
        "code": &err.code,
        "error": error,
        "diagnostic": err.diagnostic(None).to_json(),
    })
}

fn cmd_run(argv: &[String]) -> i32 {
    let flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            report_cli_failure("E_USAGE", &e, requested_json(argv));
            return 2;
        }
    };
    if flags.want_help {
        print_subcommand_help("run");
        return 0;
    }
    let Some(path) = flags.positional.first() else {
        report_cli_failure("E_USAGE", "'sofuu script run' needs a .kaze file", flags.json);
        return 2;
    };
    let opts = match options_from(&flags) {
        Ok(o) => o,
        Err(e) => {
            report_cli_failure(e.code(), e.message(), flags.json);
            return e.exit_code();
        }
    };
    // Read the input once: probing a pipe and reopening it can consume input
    // and hang before either execution or diagnostic reporting begins.
    let outcome: Result<RunReport, CliFailure> = match std::fs::read(path) {
        Err(e) => Err(CliFailure::kaze(sfs::KazeError::new(
            "E_KAZE_IO",
            format!("cannot read '{}': {}", path, e),
        ))),
        Ok(bytes) => {
            if path.ends_with(".kazebc") || bytes.starts_with(b"KAZEBC1\0") {
                isa::deserialize(&bytes)
                    .map_err(|e| CliFailure::kaze(sfs::KazeError::new("E_KAZE_BADCODE", e)))
                    .and_then(|module| {
                        if declares_async_capability(&module.needs) {
                            // A compiled artifact that declares ai or tool runs
                            // through the same bridge; the bridge resolves the
                            // path itself, so the artifact the CLI accepted is
                            // the artifact that runs.
                            run_async_bridge(path, path, &opts, &flags)
                        } else {
                            sfs::run_module(module, path, &opts).map_err(CliFailure::kaze)
                        }
                    })
            } else {
                sfs::decode_source(bytes, path)
                    .and_then(|src| sfs::prepare(&src, path).map(|prepared| (src, prepared)))
                    .map_err(CliFailure::kaze)
                    .and_then(|(src, prepared)| {
                        if declares_async_capability(&prepared.module.needs) {
                            run_async_bridge(path, &src, &opts, &flags)
                        } else {
                            sfs::run_prepared_source(prepared, &opts).map_err(CliFailure::kaze)
                        }
                    })
            }
        }
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&err.json()).unwrap_or_default()
                );
            } else {
                err.print();
            }
            1
        }
    }
}

/// Run a declared-`ai`/`tool` script through the asynchronous Kaze bridge in
/// the JavaScript runtime — the same coordinator a compiled agent uses.
///
/// `input` is the source text or, for a `.kazebc` artifact, its path. The
/// bridge performs the real provider and MCP calls and returns either the
/// runtime's `RunReport` shape or its versioned diagnostic. Nothing is
/// synthesized here: a missing provider, a denied grant or a closed MCP
/// connection comes back as a typed failure.
fn run_async_bridge(
    path: &str,
    input: &str,
    opts: &RunOptions,
    flags: &Flags,
) -> Result<RunReport, CliFailure> {
    let options = bridge_options(path, opts, flags);
    let runtime = sofuu_ffi::SofuuRuntime::init().ok_or_else(|| {
        CliFailure::bridge(
            "E_KAZE_RUNTIME",
            "the JavaScript runtime could not be initialised for this ai/tool script",
        )
    })?;
    let script = bridge_script(input, &options);
    let rc = runtime.eval_string(&script, "<kaze-cli-bridge>");
    if rc != 0 {
        return Err(CliFailure::bridge(
            "E_KAZE_HOST",
            "the runtime failed while starting the asynchronous Kaze bridge for this script",
        ));
    }
    let raw = read_global_string(&runtime, "__kazeCliOutcome").ok_or_else(|| {
        CliFailure::bridge(
            "E_KAZE_HOST",
            "the asynchronous Kaze bridge left no outcome for this script",
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        CliFailure::bridge(
            "E_KAZE_HOST",
            format!("the asynchronous Kaze bridge returned an unreadable outcome: {}", e),
        )
    })?;
    if value.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
        let report = value.get("report").cloned().unwrap_or(serde_json::Value::Null);
        return Ok(report_from_bridge(&report));
    }
    let code = value.get("code").and_then(serde_json::Value::as_str)
        .unwrap_or("E_KAZE_HOST").to_string();
    let message = value.get("error").and_then(serde_json::Value::as_str)
        .unwrap_or("the script did not run").to_string();
    let diagnostic = match value.get("diagnostic") {
        Some(serde_json::Value::Null) | None => None,
        Some(view) => Some(view.clone()),
    };
    Err(CliFailure::Bridge { code, message, diagnostic })
}

/// The options object the bridge accepts, built from the same parsed CLI flags
/// the synchronous path uses, so both paths grant, deny and bind identically.
fn bridge_options(
    path: &str,
    opts: &RunOptions,
    flags: &Flags,
) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    out.insert("sourceName".into(), serde_json::json!(path));
    out.insert("ambientArgs".into(), serde_json::json!(opts.ambient_args));
    if let Some(entry) = &opts.entry {
        out.insert("entry".into(), serde_json::json!(entry));
    }
    if let Some(config) = &opts.config_path {
        out.insert("configPath".into(), serde_json::json!(config));
    }
    out.insert("fuel".into(), serde_json::json!(opts.fuel));
    out.insert("heapBytes".into(), serde_json::json!(opts.heap_bytes));
    out.insert("ticks".into(), serde_json::json!(opts.ticks));
    out.insert("tickMs".into(), serde_json::json!(opts.tick_ms));
    out.insert("timeoutMs".into(), serde_json::json!(opts.timeout_ms));
    if !opts.grants.is_empty() {
        out.insert("grants".into(), serde_json::json!(opts.grants));
    }
    if !opts.backend.is_empty() {
        let mut caps = serde_json::Map::new();
        for (family, backend) in &opts.backend {
            caps.insert(
                family.clone(),
                if backend.is_empty() { serde_json::json!(true) } else { serde_json::json!(backend) },
            );
        }
        out.insert("caps".into(), serde_json::Value::Object(caps));
    }
    if !opts.pins.is_empty() {
        let mut pins = serde_json::Map::new();
        for (name, line) in &opts.pins {
            pins.insert(name.clone(), serde_json::json!(line));
        }
        out.insert("pins".into(), serde_json::Value::Object(pins));
    }
    if !flags.sim.is_empty() {
        out.insert("sim".into(), bridge_sim(&flags.sim));
    }
    // `None` is a deliberate `--no-policy`: the bridge then denies every
    // capability call, which is the same refusal the synchronous path gives.
    if let Some(policy) = &opts.policy {
        out.insert("policy".into(), policy.clone());
    }
    out
}

/// `--sim battery=15,armed=true` as an object, so the bridge applies the same
/// overrides the synchronous path already parsed.
fn bridge_sim(spec: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for pair in spec.split(',').filter(|p| !p.trim().is_empty()) {
        let (key, raw) = match pair.split_once('=') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => (pair.trim(), ""),
        };
        let value = serde_json::from_str::<serde_json::Value>(raw)
            .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
        map.insert(key.to_string(), value);
    }
    serde_json::Value::Object(map)
}

/// The bootstrap evaluated in the runtime. It only forwards arguments and
/// returns what `runAsync` really produced.
fn bridge_script(input: &str, options: &serde_json::Map<String, serde_json::Value>) -> String {
    format!(
        "(function () {{\n\
         \x20 globalThis.__kazeCliOutcome = null;\n\
         \x20 if (!sofuu.kaze || typeof sofuu.kaze.runAsync !== 'function') {{\n\
         \x20   globalThis.__kazeCliOutcome = JSON.stringify({{ ok: false, code: 'E_KAZE_RUNTIME',\n\
         \x20     error: 'this Sofuu runtime has no asynchronous Kaze bridge for ai/tool capabilities' }});\n\
         \x20   return;\n\
         \x20 }}\n\
         \x20 sofuu.kaze.runAsync({source}, {{}}, {options}).then(function (report) {{\n\
         \x20   globalThis.__kazeCliOutcome = JSON.stringify({{ ok: true, report: report }});\n\
         \x20 }}, function (error) {{\n\
         \x20   globalThis.__kazeCliOutcome = JSON.stringify({{\n\
         \x20     ok: false,\n\
         \x20     code: (error && typeof error.code === 'string') ? error.code : 'E_KAZE_HOST',\n\
         \x20     error: (error && error.message) ? String(error.message) : String(error),\n\
         \x20     diagnostic: (error && error.diagnostic) ? error.diagnostic : null\n\
         \x20   }});\n\
         \x20 }});\n\
         }})();",
        source = js_string_literal(input),
        options = serde_json::Value::Object(options.clone()),
    )
}

/// A JS string literal. `serde_json` already escapes quotes, backslashes and
/// control characters; U+2028/U+2029 are legal inside a JSON string but only
/// legal in a JS string literal from ES2019, so escape them explicitly.
fn js_string_literal(text: &str) -> String {
    serde_json::to_string(text)
        .unwrap_or_else(|_| "\"\"".to_string())
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

/// Read a global string the bootstrap wrote. Only used on the owner thread,
/// right after `eval_string` pumped the runtime to completion.
fn read_global_string(runtime: &sofuu_ffi::SofuuRuntime, name: &str) -> Option<String> {
    let ctx = runtime.engine_ctx() as *mut sofuu_ffi::qjs::JSContext;
    if ctx.is_null() {
        return None;
    }
    let key = std::ffi::CString::new(name).ok()?;
    unsafe {
        let global = sofuu_ffi::qjs::sofuu_js_get_global_object(ctx);
        let value = sofuu_ffi::qjs::sofuu_js_get_property_str(ctx, global, key.as_ptr());
        let text = if sofuu_ffi::qjs::is_undefined(value) || sofuu_ffi::qjs::is_null(value) {
            None
        } else {
            let raw = sofuu_ffi::qjs::sofuu_js_to_cstring(ctx, value);
            let out = if raw.is_null() {
                None
            } else {
                let owned = std::ffi::CStr::from_ptr(raw).to_string_lossy().into_owned();
                sofuu_ffi::qjs::sofuu_js_free_cstring(ctx, raw);
                Some(owned)
            };
            out
        };
        sofuu_ffi::qjs::sofuu_js_free_value(ctx, value);
        sofuu_ffi::qjs::sofuu_js_free_value(ctx, global);
        text
    }
}

/// The bridge reports the same fields the synchronous runner does.
fn report_from_bridge(report: &serde_json::Value) -> RunReport {
    let text = |key: &str| report.get(key).and_then(serde_json::Value::as_str).unwrap_or("").to_string();
    let number = |key: &str| report.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let actions = report.get("actions").and_then(serde_json::Value::as_array)
        .map(|items| items.iter().filter_map(serde_json::Value::as_str).map(str::to_string).collect())
        .unwrap_or_default();
    let logs = report.get("logs").and_then(serde_json::Value::as_array)
        .map(|items| {
            items.iter().map(|entry| {
                (
                    entry.get("level").and_then(serde_json::Value::as_str).unwrap_or("info").to_string(),
                    entry.get("message").and_then(serde_json::Value::as_str).unwrap_or("").to_string(),
                )
            }).collect()
        })
        .unwrap_or_default();
    let simulated_families = report.get("simulated_families").and_then(serde_json::Value::as_array)
        .map(|items| items.iter().filter_map(serde_json::Value::as_str).map(str::to_string).collect())
        .unwrap_or_default();
    let verdict = report.get("verdict").and_then(serde_json::Value::as_object).map(|v| {
        let reason = v.get("reason").and_then(serde_json::Value::as_str).unwrap_or("").to_string();
        if v.get("allow").and_then(serde_json::Value::as_bool).unwrap_or(false) {
            Verdict::Allow(reason)
        } else {
            Verdict::Reject(reason)
        }
    });
    RunReport {
        entry: text("entry"),
        value: text("result"),
        actions,
        logs,
        fuel_used: number("fuel"),
        instructions: number("instructions"),
        ticks: number("ticks") as u32,
        verdict,
        simulated_families,
    }
}

fn cmd_check(argv: &[String]) -> i32 {
    let flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            report_cli_failure("E_USAGE", &e, requested_json(argv));
            return 2;
        }
    };
    if flags.want_help {
        print_subcommand_help("check");
        return 0;
    }
    let Some(path) = flags.positional.first() else {
        report_cli_failure("E_USAGE", "'sofuu script check' needs a .kaze file", flags.json);
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
            if flags.json {
                println!("{}", serde_json::to_string_pretty(&kaze_error_json(&err, path)).unwrap_or_default());
            } else {
                print_kaze_error(&err, path);
            }
            1
        }
    }
}

fn cmd_compile(argv: &[String]) -> i32 {
    let flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            report_cli_failure("E_USAGE", &e, requested_json(argv));
            return 2;
        }
    };
    if flags.want_help {
        print_subcommand_help("compile");
        return 0;
    }
    let Some(path) = flags.positional.first() else {
        report_cli_failure("E_USAGE", "'sofuu script compile' needs a .kaze file", flags.json);
        return 2;
    };
    let prepared = match sfs::prepare_file(path) {
        Ok(p) => p,
        Err(err) => {
            if flags.json {
                println!("{}", serde_json::to_string_pretty(&kaze_error_json(&err, path)).unwrap_or_default());
            } else {
                print_kaze_error(&err, path);
            }
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
                report_cli_failure("E_KAZE_IO", &format!("cannot write '{}': {}", out, e), flags.json);
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
    if !report.simulated_families.is_empty() {
        println!("simulation (testing): {}", report.simulated_families.join(", "));
    }
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
        "simulated_families": report.simulated_families,
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
    let mut flags = match parse_flags(argv) {
        Ok(f) => f,
        Err(e) => {
            report_cli_failure("E_USAGE", &e, false);
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
            match handle_repl_command(trimmed, &mut needs, &mut grants, &body, &mut flags) {
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
    flags: &mut Flags,
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
                 .cap <family>[=sim]   grant a family; drone/rover sim is testing only\n\
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
                let (family, backend) = match n.split_once('=') {
                    Some((family, backend)) if !family.trim().is_empty() && !backend.trim().is_empty() => (family.trim(), backend.trim()),
                    Some(_) => { eprintln!("E_CAP_INVALID: .cap needs a family and backend"); continue; }
                    None => (n, ""),
                };
                if let Err(d) = crate::rt::hardware::validate_grant(family, backend) {
                    eprintln!("{}: {}", d.code, d.reason);
                    continue;
                }
                flags.backend.retain(|(f, _)| f != family);
                flags.backend.push((family.to_string(), backend.to_string()));
                if !grants.iter().any(|x| x == family) {
                    grants.push(family.to_string());
                }
                if !needs.iter().any(|x| x == family) {
                    needs.push(family.to_string());
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
            let mut opts = match options_from(flags) {
                Ok(opts) => opts,
                Err(e) => {
                    /* REPL: report and stay open — a bad guard file must not
                     * take the session down, but it must also never be
                     * silently skipped (KZ-051/KZ-052). */
                    let tag = if matches!(e, OptErr::Usage(_)) {
                        "E_USAGE"
                    } else {
                        "E_CAP_DENIED"
                    };
                    eprintln!("\x1b[31m{tag}\x1b[0m: {}", e.message());
                    return ReplFlow::Continue;
                }
            };
            opts.entry = Some(name.to_string());
            opts.args = args;
            opts.grants = grants.clone();
            match sfs::run_source(&src, "<repl>", &opts) {
                Ok(report) => print_report(&report),
                Err(err) => eprintln!("\x1b[31m{}\x1b[0m: {}", err.code, err.message),
            }
        }
        other => println!("unknown command '{}'; .help lists them", other),
    }
    ReplFlow::Continue
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags() -> Flags {
        parse_flags(&[]).expect("empty flag list")
    }

    #[test]
    fn a_script_declaring_ai_or_tool_routes_to_the_async_bridge() {
        // `ai` and `tool` need the JavaScript runtime; every other family is
        // served in-process, so routing stays engine-free for those scripts.
        assert!(declares_async_capability(&["ai".to_string()]));
        assert!(declares_async_capability(&["tool".to_string()]));
        assert!(declares_async_capability(&["log".into(), "ai".into()]));
        for family in ["log", "policy", "memory", "alert", "drone", "sensor", "device"] {
            assert!(!declares_async_capability(&[family.to_string()]), "{family}");
        }
        // A declaration of a device that resolves to another family must not
        // be mistaken for an async capability.
        assert!(!declares_async_capability(&["lidar".to_string()]));
    }

    #[test]
    fn bridge_options_carry_the_same_grants_policy_and_configuration() {
        let mut opts = match options_from(&flags()) {
            Ok(opts) => opts,
            Err(e) => panic!("default options: {}", e.message()),
        };
        opts.grants = vec!["ai".into(), "tool".into()];
        opts.backend = vec![("ai".into(), String::new()), ("tool".into(), "mcp".into())];
        opts.entry = Some("main".into());
        opts.config_path = Some("agent.config.json".into());
        opts.policy = Some(serde_json::json!({"tools": {"allow": ["kaze.*"]}}));

        let built = bridge_options("agent.kaze", &opts, &flags());
        assert_eq!(built["sourceName"], serde_json::json!("agent.kaze"));
        assert_eq!(built["grants"], serde_json::json!(["ai", "tool"]));
        // An empty backend means "granted, no device"; it must not become an
        // empty-string backend, which the bridge rejects.
        assert_eq!(built["caps"], serde_json::json!({"ai": true, "tool": "mcp"}));
        assert_eq!(built["entry"], serde_json::json!("main"));
        assert_eq!(built["configPath"], serde_json::json!("agent.config.json"));
        assert_eq!(built["policy"], serde_json::json!({"tools": {"allow": ["kaze.*"]}}));
        assert_eq!(built["fuel"], serde_json::json!(opts.fuel));
        assert_eq!(built["heapBytes"], serde_json::json!(opts.heap_bytes));

        // `--no-policy` is a deliberate refusal: the bridge must not be handed
        // a policy document it never received.
        opts.policy = None;
        assert!(bridge_options("agent.kaze", &opts, &flags()).get("policy").is_none());
    }

    #[test]
    fn bridge_sim_keeps_the_overrides_the_synchronous_path_parsed() {
        assert_eq!(
            bridge_sim("battery=15,armed=true"),
            serde_json::json!({"battery": 15, "armed": true})
        );
        assert_eq!(bridge_sim("mode=auto"), serde_json::json!({"mode": "auto"}));
        assert_eq!(bridge_sim(""), serde_json::json!({}));
    }

    #[test]
    fn a_js_string_literal_cannot_break_out_of_the_bootstrap() {
        let quoted = js_string_literal("needs: ai\naction main():\n    return \"x\"\n");
        assert!(quoted.starts_with('"') && quoted.ends_with('"'));
        assert!(!quoted.contains('\n'), "a raw newline would end the literal");
        assert!(quoted.contains("\\n"));
        // A JSON string literal is also a JS string literal, so the round
        // trip is the property that matters: the text cannot close the
        // literal early and start executing.
        for raw in ["\" ] } ); run(); //", "\\", "line\nbreak", "tab\tx"] {
            let literal = js_string_literal(raw);
            let parsed: String = serde_json::from_str(&literal).expect("valid JSON literal");
            assert_eq!(parsed, raw);
            assert!(!literal.contains('\n'), "{literal}");
        }
        // U+2028/U+2029 are legal in JSON but must be escaped for older JS.
        let separators = js_string_literal("a\u{2028}b\u{2029}c");
        assert_eq!(separators, "\"a\\u2028b\\u2029c\"");
    }

    #[test]
    fn the_bridge_report_matches_the_synchronous_shape() {
        let report = report_from_bridge(&serde_json::json!({
            "ok": true,
            "entry": "main",
            "result": "\"done\"",
            "fuel": 12,
            "instructions": 9,
            "ticks": 2,
            "simulated_families": ["drone"],
            "actions": ["ai.prompt(<redacted>) -> <redacted result>"],
            "logs": [{"level": "info", "message": "hello"}],
            "verdict": {"allow": true, "reason": "operator"}
        }));
        assert_eq!(report.entry, "main");
        assert_eq!(report.value, "\"done\"");
        assert_eq!(report.fuel_used, 12);
        assert_eq!(report.instructions, 9);
        assert_eq!(report.ticks, 2);
        assert_eq!(report.simulated_families, vec!["drone"]);
        assert_eq!(report.logs, vec![("info".to_string(), "hello".to_string())]);
        assert!(matches!(report.verdict, Some(Verdict::Allow(ref t)) if t == "operator"));
        assert!(report.render_plain().contains("simulation (testing): drone"));

        // An empty report is a real empty report, not a fabricated one.
        let empty = report_from_bridge(&serde_json::Value::Null);
        assert_eq!(empty.entry, "");
        assert_eq!(empty.value, "");
        assert!(empty.verdict.is_none());
    }
}
