//! sofuu-micro — ultra-lean MCP server & tool micro-runtime.
//!
//! Provides the minimal execution environment for compiled MCP servers and
//! standalone tools with zero TUI/chat overhead, zero monolithic CLI baggage,
//! and no unused cognitive memory or heavy inference machinery.
//!
//! Used as the default micro-profile base binary by `sofuu mcp build` and
//! `sofuu compile --profile mcp`.

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

fn install_micro_prelude(rt: &SofuuRuntime) {
    let argv: Vec<String> = env::args().skip(1).collect();
    let argv_json = serde_json::to_string(&argv).unwrap_or_else(|_| "[]".to_string());
    let js = format!(
        "(function(){{var S=globalThis.sofuu;if(!S)return;try{{S.args={argv_json};S.isMicro=true;}}catch(e){{}}}})();"
    );
    let _ = rt.eval_string(&js, "<micro-prelude>");
}

fn run_compiled_payload(payload: &sofuu_core::compiler::EmbeddedPayload) -> ExitCode {
    let rt = match SofuuRuntime::init() {
        Some(rt) => rt,
        None => {
            eprintln!("\x1b[31mFatal:\x1b[0m sofuu-micro runtime init failed");
            return ExitCode::FAILURE;
        }
    };
    install_micro_prelude(&rt);

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

    // 1. Embedded Payload (Standalone compiled MCP binary mode)
    match sofuu_core::compiler::detect_embedded_payload() {
        Ok(Some(payload)) => {
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
            return run_compiled_payload(&payload);
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return ExitCode::FAILURE;
        }
    }

    // 2. Direct script execution fallback
    if args.len() < 2 {
        println!("\x1b[1msofuu-micro\x1b[0m v{} — ultra-lean MCP server & tool micro-runtime", env!("CARGO_PKG_VERSION"));
        println!("Usage: sofuu-micro <server.js|ts> [args...]");
        println!("  Build with: sofuu mcp build <server.ts> -o <binary>");
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
    install_micro_prelude(&rt);
    let rc = rt.eval_file(script);
    js_exit_code(&rt, rc)
}
