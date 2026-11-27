//! sofuu-agent — lean standalone agent runtime.
//!
//! Provides the execution environment for autonomous agents, toolpacks,
//! MCP servers/clients, memory, and Kaze guardrails with zero TUI/chat
//! overhead and no monolithic CLI subcommand baggage.
//!
//! Used as the default lean base binary by `sofuu compile`.

use std::env;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::process::ExitCode;
use sofuu_ffi::SofuuRuntime;

fn js_exit_code_i32(rt: &SofuuRuntime, rc: i32) -> i32 {
    let st = sofuu_core::rt::engine::run_status_raw(rt.engine_ctx());
    match (st.exit_code, rc, st.has_uncaught_error) {
        (Some(chosen), _, _) => chosen,
        (None, rc, _) if rc != 0 => 1,
        (None, _, true) => 1,
        _ => 0,
    }
}

fn install_compiled_prelude(
    rt: &SofuuRuntime,
    brain: Option<&sofuu_core::compiler::MountedBrain>,
    resolved_models: &sofuu_core::model_config::ResolvedAgentModel,
    resolved_channels: &sofuu_core::channel_config::ResolvedChannels,
) {
    let argv: Vec<String> = env::args().skip(1).collect();
    let argv_json = serde_json::to_string(&argv).unwrap_or_else(|_| "[]".to_string());
    let models_prelude = sofuu_core::model_config::js_prelude_for(resolved_models);
    let channels_prelude = sofuu_core::channel_config::js_prelude_for(resolved_channels);
    let embedded = match brain {
        Some(b) => {
            let path = serde_json::to_string(&b.path).unwrap_or_else(|_| "\"\"".to_string());
            format!(
                "try{{S.memory.embedded={{path:{path},dim:{},open:function(d){{return S.memory.open(this.path,d||this.dim);}}}};}}catch(e){{}}",
                b.dim
            )
        }
        None => String::new(),
    };
    let js = format!(
        "{models_prelude}{channels_prelude}(function(){{var S=globalThis.sofuu;if(!S)return;try{{S.args={argv_json};}}catch(e){{}}if(!S.memory)return;{embedded}}})();"
    );
    let _ = rt.eval_string(&js, "<compiled-prelude>");
}

/// Parse a `--every` value (`500ms`, `2s`, `1m`, `1h`, `1d`, or a bare
/// millisecond count) into milliseconds. Mirrors the daemon CLI's parser and
/// the JS `sofuu.triggers.parseDuration` contract.
fn parse_every_ms(raw: &str) -> Option<u64> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    let (num, mul) = if let Some(v) = s.strip_suffix("ms") {
        (v, 1.0)
    } else if let Some(v) = s.strip_suffix('s') {
        (v, 1000.0)
    } else if let Some(v) = s.strip_suffix('m') {
        (v, 60_000.0)
    } else if let Some(v) = s.strip_suffix('h') {
        (v, 3_600_000.0)
    } else if let Some(v) = s.strip_suffix('d') {
        (v, 86_400_000.0)
    } else {
        (s.as_str(), 1.0)
    };
    let n: f64 = num.trim().parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    Some((n * mul) as u64)
}

/// PLAN-AUTONOMOUS-REACTIVE-RUNTIME P2: make `--daemon` / `--every` reach
/// `sofuu.triggers` in the embedded program.
///
/// A reactive agent is the same code under `sofuu run`, a compiled artifact
/// and `sofuu daemon install`. The trigger driver reads
/// `SOFUU_DAEMON_INTERVAL_MS` when a spec omits `every`, so the OS service
/// manager's cadence is honoured without re-spawning a QuickJS process every
/// few seconds. `SOFUU_DAEMON=1` lets an app detect the supervised mode.
fn apply_daemon_env(args: &[String]) {
    let mut daemon = false;
    let mut every: Option<String> = None;
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--" {
            break;
        }
        if a == "--daemon" || a == "-d" {
            daemon = true;
        } else if (a == "--every" || a == "--interval") && i + 1 < args.len() {
            every = Some(args[i + 1].clone());
            i += 1;
        } else if let Some(v) = a.strip_prefix("--every=") {
            every = Some(v.to_string());
        } else if let Some(v) = a.strip_prefix("--interval=") {
            every = Some(v.to_string());
        }
        i += 1;
    }
    if daemon {
        env::set_var("SOFUU_DAEMON", "1");
    }
    if let Some(raw) = every {
        if let Some(ms) = parse_every_ms(&raw) {
            env::set_var("SOFUU_DAEMON_INTERVAL_MS", ms.to_string());
        } else {
            eprintln!("Warning: --every '{raw}' is not a duration (use 500ms, 2s, 1m, 1h)");
        }
    }
}


/// CP-046 / CP-047 (AUDIT-2026-11-08): a standalone artifact used to forward
/// EVERY argument to the embedded program, so `artifact --version` ran the
/// user code (usually starting a real workload) and a typo like `--confg` was
/// silently swallowed. The artifact now answers for itself first: version and
/// help are intercepted, and an unknown leading flag is refused with the
/// list of what is accepted.
fn artifact_arg_gate(args: &[String], is_kaze: bool) -> Option<ExitCode> {
    let model_value_flags = [
        "--models-config",
        "--provider",
        "-p",
        "--model",
        "-m",
        "--api-key",
        "--base-url",
        "--profile",
    ];
    let channel_value_flags = [
        "--channels-config",
        "--channels",
        "--channel",
        "--type",
        "--channel-type",
        "--channel-name",
        "--token",
        "--access-token",
        "--phone-number-id",
        "--app-secret",
        "--verify-token",
        "--url",
        "--default-recipient",
        "--allowed-senders",
        "--allowed-sender",
        "--network-mode",
        "--api-base",
    ];
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--" {
            return None;
        }
        match a {
            "--version" | "-V" => {
                println!("sofuu-artifact {}", env!("CARGO_PKG_VERSION"));
                return Some(ExitCode::SUCCESS);
            }
            "--help" | "-h" if is_kaze => return None,
            "--help" | "-h" => {
                println!(
                    "Usage: {} [options] [-- <program args>]\n\n\
                     A compiled Sofuu artifact: the program is embedded, so there is\n\
                     nothing to install. Arguments reach the program as sofuu.args and\n\
                     appear in process.argv.\n\n\
                     Model & Provider Options (per-agent models.json):\n\
                     \x20 --setup-model            Interactive wizard to add/configure provider & model\n\
                     \x20 --edit-model [name]      Edit, switch, add, or remove configured models\n\
                     \x20 --list-models            Show providers and models configured for this binary\n\
                     \x20 --models-config <path>   Use a specific models.json file\n\
                     \x20 --provider, -p <name>    Override active provider or saved profile name\n\
                     \x20 --model, -m <name>       Override model name\n\
                     \x20 --api-key <key>          Override API key for this run\n\
                     \x20 --base-url <url>         Override endpoint base URL\n\n\
                     Messaging Options (per-agent channels.json):\n\
                     \x20 --setup-channel          Interactive wizard to add/configure a channel\n\
                     \x20 --edit-channel [name]    Edit, switch, add, or remove configured channels\n\
                     \x20 --list-channels          Show channels configured for this binary\n\
                     \x20 --channels-config <path> Use a specific channels.json file\n\
                     \x20 --channel <name>         Override the active channel for this run\n\n\
                     Runtime Options:\n\
                     \x20 --ui                     Open native 2D GPU UI window (requires libsofuu_gpu)\n\
                     \x20 --interactive, -i        Launch interactive session (when supported)\n\
                     \x20 --arg name=value         Bind one routine parameter (repeatable)\n\
                     \x20 --config <path>          JSON configuration file\n\
                     \x20 --json                   Machine-readable output where the program supports it\n\
                     \x20 --daemon, -d             Run the program's reactive loop under a supervisor\n\
                     \x20 --every <duration>       Loop cadence (500ms, 2s, 1m, 1h; SOFUU_DAEMON_INTERVAL_MS)\n\
                     \x20 --version, -V            Print the artifact version and exit\n\
                     \x20 --help, -h               This text\n\
                     \x20 --                       Pass everything after this through untouched\n",
                    args.first().map(|s| s.as_str()).unwrap_or("artifact")
                );
                return Some(ExitCode::SUCCESS);
            }
            _ => {}
        }
        if model_value_flags.contains(&a) || channel_value_flags.contains(&a) {
            if i + 1 >= args.len() || args[i + 1].starts_with("--") {
                eprintln!("Error: {} needs a value", a);
                return Some(ExitCode::from(2));
            }
            i += 2;
            continue;
        }
        if a == "--edit-model" || a == "--edit-channel" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if a == "--setup-model"
            || a == "--list-models"
            || a == "--setup-channel"
            || a == "--list-channels"
            || a == "--allow-anonymous"
            || a == "--test-send"
            || a == "--interactive"
            || a == "-i"
            || a.starts_with("--edit-model=")
            || a.starts_with("--edit-channel=")
            || model_value_flags
                .iter()
                .any(|f| a.strip_prefix(f).is_some_and(|rest| rest.starts_with('=')))
            || channel_value_flags
                .iter()
                .any(|f| a.strip_prefix(f).is_some_and(|rest| rest.starts_with('=')))
        {
            i += 1;
            continue;
        }
        let takes_value = ["--arg", "--config", "-c", "--entry", "--cap", "--policy",
            "--pin", "--sim", "--fuel", "--every", "--interval"];
        let artifact_every = a == "--every" || a == "--interval";
        if (is_kaze || artifact_every) && takes_value.contains(&a) {
            if i + 1 >= args.len() || args[i + 1].starts_with("--") {
                eprintln!("Error: {} needs a value", a);
                return Some(ExitCode::from(2));
            }
            i += 2;
            continue;
        }
        let known_kaze = is_kaze && (takes_value.iter().any(|flag| a.strip_prefix(flag)
            .is_some_and(|rest| rest.starts_with('=')))
            || matches!(a, "--interactive" | "-i" | "--daemon" | "-d"));
        let known = known_kaze
            || matches!(a, "--daemon" | "-d" | "--ui")
            || a.strip_prefix("--every=").is_some_and(|rest| !rest.is_empty())
            || a.strip_prefix("--interval=").is_some_and(|rest| !rest.is_empty())
            || a.starts_with("--arg") || a.starts_with("--config") || a == "--json";
        if a.starts_with('-') && !known {
            eprintln!("Error: Unrecognized flag '{}'", a);
            if is_kaze {
                eprintln!("  use --help for the Kaze agent options");
            } else {
                eprintln!("  accepted: --setup-model, --edit-model, --list-models, --setup-channel, --edit-channel, --list-channels, --provider, --model, --channel, --config, --arg, --ui, --json, --version, --help, --");
            }
            return Some(ExitCode::from(2));
        }
        i += 1;
    }
    None
}
fn run_compiled_payload_rc(
    payload: &sofuu_core::compiler::EmbeddedPayload,
    args: &[String],
    target: &sofuu_core::model_config::AgentTarget,
) -> i32 {
    apply_daemon_env(args);
    let resolved_models = sofuu_core::model_config::apply_runtime_config(args, target);
    let resolved_channels = sofuu_core::channel_config::apply_runtime_config(args, target);
    let rt = match SofuuRuntime::init() {
        Some(rt) => rt,
        None => {
            eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
            return 1;
        }
    };
    let brain = match sofuu_core::compiler::materialize_brain(payload) {
        Ok(brain) => brain,
        Err(e) => {
            eprintln!("[sofuu-agent] embedded brain not mounted: {e}");
            None
        }
    };
    install_compiled_prelude(&rt, brain.as_ref(), &resolved_models, &resolved_channels);

    // SAFETY: engine_ctx() is this runtime's live QuickJS context.
    let rc = unsafe {
        sofuu_core::rt::engine::eval_compiled_and_run(
            rt.engine_ctx() as *mut sofuu_ffi::qjs::JSContext,
            &payload.bytecode,
            &payload.entry,
        )
    };
    if rc == 0 {
        0
    } else {
        let code = unsafe {
            let ctx = rt.engine_ctx() as *mut sofuu_ffi::qjs::JSContext;
            let global = sofuu_ffi::qjs::sofuu_js_get_global_object(ctx);
            let proc_obj = sofuu_ffi::qjs::sofuu_js_get_property_str(ctx, global, c"process".as_ptr());
            let ec = sofuu_ffi::qjs::sofuu_js_get_property_str(ctx, proc_obj, c"exitCode".as_ptr());
            let mut val: i32 = rc;
            let _ = sofuu_ffi::qjs::JS_ToInt32(ctx, &mut val, ec);
            sofuu_ffi::qjs::sofuu_js_free_value(ctx, ec);
            sofuu_ffi::qjs::sofuu_js_free_value(ctx, proc_obj);
            sofuu_ffi::qjs::sofuu_js_free_value(ctx, global);
            if val == 0 { 1 } else { val }
        };
        code
    }
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = env::args().collect();
    let (use_ui, args) = sofuu_core::ui_bridge::take_ui_arg(&raw_args);

    // 1. Embedded Payload (Standalone compiled agent binary mode)
    match sofuu_core::compiler::detect_embedded_payload() {
        Ok(Some(payload)) => {
            /* CP-046/CP-047: answer for the artifact before the embedded
             * program ever runs. */
            let is_kaze = std::path::Path::new(&payload.entry).extension().is_some_and(|ext| ext == "kaze");
            if let Some(code) = artifact_arg_gate(&args, is_kaze) {
                return code;
            }
            let target = sofuu_core::model_config::AgentTarget::binary(&args[0]);
            if let Some(code) = sofuu_core::model_config::handle_cli_command(&args, &target) {
                return code;
            }
            if let Some(code) = sofuu_core::channel_config::handle_cli_command(&args, &target) {
                return code;
            }
            let mut payload_argv: Vec<String> = Vec::with_capacity(args.len() + 1);
            payload_argv.push(args[0].clone());
            payload_argv.push(args[0].clone());
            payload_argv.extend_from_slice(&args[1..]);
            unsafe {
                let owned: Vec<CString> = payload_argv
                    .iter()
                    .map(|a| CString::new(a.as_str()).unwrap_or_default())
                    .collect();
                let mut ptrs: Vec<*mut c_char> =
                    owned.iter().map(|c| c.as_ptr() as *mut c_char).collect();
                sofuu_core::modules::process::mod_process_set_args(
                    owned.len() as c_int,
                    ptrs.as_mut_ptr(),
                );
            }
            if use_ui {
                let ui_title = std::path::Path::new(&args[0])
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Sofuu Agent")
                    .to_string();
                let args_c = args.clone();
                return sofuu_core::ui_bridge::run_with_ui(&ui_title, move || {
                    run_compiled_payload_rc(&payload, &args_c, &target)
                });
            }
            let rc = run_compiled_payload_rc(&payload, &args, &target);
            return if rc == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(rc.clamp(1, 255) as u8)
            };
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return ExitCode::FAILURE;
        }
    }

    // 2. Direct script execution fallback
    if args.len() < 2 {
        println!("\x1b[1msofuu-agent\x1b[0m v{} — lean autonomous agent runtime", env!("CARGO_PKG_VERSION"));
        println!("Usage: sofuu-agent <entry.js|ts> [args...]");
        return ExitCode::SUCCESS;
    }
    /* A flag is not a filename: before this arm, sofuu-agent --help tried to
     * open a file called "--help" and printed "Cannot open file". Any other
     * unknown flag still fails, but usage is one word away. */
    if matches!(args[1].as_str(), "-h" | "--help" | "help") {
        println!("\x1b[1msofuu-agent\x1b[0m v{} — lean autonomous agent runtime", env!("CARGO_PKG_VERSION"));
        println!("Usage: sofuu-agent <entry.js|ts> [args...]");
        println!();
        println!("  The entry script gets the agent runtime: sofuu.agent, sofuu.ai,");
        println!("  sofuu.tools and sofuu.memory, plus SOFUU_* / AGENT_* configuration.");
        println!("  A Kaze agent runs through: sofuu script run <agent.kaze>");
        println!("  Reactive agents: --daemon --every 60s (or SOFUU_DAEMON_INTERVAL_MS).");
        return ExitCode::SUCCESS;
    }

    let script = args[1].clone();
    let args_daemon = args.clone();
    let exec_fallback = move || -> i32 {
        apply_daemon_env(&args_daemon);
        let rt = match SofuuRuntime::init() {
            Some(rt) => rt,
            None => {
                eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                return 1;
            }
        };
        let rc = rt.eval_file(&script);
        js_exit_code_i32(&rt, rc)
    };
    if use_ui {
        let ui_title = std::path::Path::new(&args[1])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Sofuu Agent")
            .to_string();
        return sofuu_core::ui_bridge::run_with_ui(&ui_title, exec_fallback);
    }
    let code = exec_fallback();
    ExitCode::from(code.clamp(0, 255) as u8)
}
