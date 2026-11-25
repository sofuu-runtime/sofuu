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

fn js_exit_code(rt: &SofuuRuntime, rc: i32) -> ExitCode {
    let st = sofuu_core::rt::engine::run_status_raw(rt.engine_ctx());
    let code = match (st.exit_code, rc, st.has_uncaught_error) {
        (Some(chosen), _, _) => chosen,
        (None, rc, _) if rc != 0 => 1,
        (None, _, true) => 1,
        _ => 0,
    };
    ExitCode::from(code.clamp(0, 255) as u8)
}

fn install_compiled_prelude(
    rt: &SofuuRuntime,
    brain: Option<&sofuu_core::compiler::MountedBrain>,
    resolved_models: &sofuu_core::model_config::ResolvedAgentModel,
) {
    let argv: Vec<String> = env::args().skip(1).collect();
    let argv_json = serde_json::to_string(&argv).unwrap_or_else(|_| "[]".to_string());
    let models_prelude = sofuu_core::model_config::js_prelude_for(resolved_models);
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
        "{models_prelude}(function(){{var S=globalThis.sofuu;if(!S)return;try{{S.args={argv_json};}}catch(e){{}}if(!S.memory)return;{embedded}}})();"
    );
    let _ = rt.eval_string(&js, "<compiled-prelude>");
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
                     Runtime Options:\n\
                     \x20 --interactive, -i        Launch interactive session (when supported)\n\
                     \x20 --arg name=value         Bind one routine parameter (repeatable)\n\
                     \x20 --config <path>          JSON configuration file\n\
                     \x20 --json                   Machine-readable output where the program supports it\n\
                     \x20 --version, -V            Print the artifact version and exit\n\
                     \x20 --help, -h               This text\n\
                     \x20 --                       Pass everything after this through untouched\n",
                    args.first().map(|s| s.as_str()).unwrap_or("artifact")
                );
                return Some(ExitCode::SUCCESS);
            }
            _ => {}
        }
        if model_value_flags.contains(&a) {
            if i + 1 >= args.len() || args[i + 1].starts_with("--") {
                eprintln!("Error: {} needs a value", a);
                return Some(ExitCode::from(2));
            }
            i += 2;
            continue;
        }
        if a == "--edit-model" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if a == "--setup-model"
            || a == "--list-models"
            || a == "--interactive"
            || a == "-i"
            || a.starts_with("--edit-model=")
            || model_value_flags
                .iter()
                .any(|f| a.strip_prefix(f).is_some_and(|rest| rest.starts_with('=')))
        {
            i += 1;
            continue;
        }
        let takes_value = ["--arg", "--config", "-c", "--entry", "--cap", "--policy",
            "--pin", "--sim", "--fuel", "--every", "--interval"];
        if is_kaze && takes_value.contains(&a) {
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
        let known = known_kaze || a.starts_with("--arg") || a.starts_with("--config") || a == "--json";
        if a.starts_with('-') && !known {
            eprintln!("Error: Unrecognized flag '{}'", a);
            if is_kaze {
                eprintln!("  use --help for the Kaze agent options");
            } else {
                eprintln!("  accepted: --setup-model, --edit-model, --list-models, --provider, --model, --config, --arg, --json, --version, --help, --");
            }
            return Some(ExitCode::from(2));
        }
        i += 1;
    }
    None
}
fn run_compiled_payload(
    payload: &sofuu_core::compiler::EmbeddedPayload,
    args: &[String],
    target: &sofuu_core::model_config::AgentTarget,
) -> ExitCode {
    let resolved_models = sofuu_core::model_config::apply_runtime_config(args, target);
    let rt = match SofuuRuntime::init() {
        Some(rt) => rt,
        None => {
            eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
            return ExitCode::FAILURE;
        }
    };
    let brain = match sofuu_core::compiler::materialize_brain(payload) {
        Ok(brain) => brain,
        Err(e) => {
            eprintln!("[sofuu-agent] embedded brain not mounted: {e}");
            None
        }
    };
    install_compiled_prelude(&rt, brain.as_ref(), &resolved_models);

    // SAFETY: engine_ctx() is this runtime's live QuickJS context.
    let rc = unsafe {
        sofuu_core::rt::engine::eval_compiled_and_run(
            rt.engine_ctx() as *mut sofuu_ffi::qjs::JSContext,
            &payload.bytecode,
            &payload.entry,
        )
    };
    js_exit_code(&rt, rc)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

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
            return run_compiled_payload(&payload, &args, &target);
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
        return ExitCode::SUCCESS;
    }

    let script = &args[1];
    let rt = match SofuuRuntime::init() {
        Some(rt) => rt,
        None => {
            eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
            return ExitCode::FAILURE;
        }
    };
    let rc = rt.eval_file(script);
    js_exit_code(&rt, rc)
}
