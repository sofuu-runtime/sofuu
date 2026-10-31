// shipped.rs — shipped runtime JS drivers beyond RLM (PLAN-AGENTS).
//
// The engine seam sofuu_rust_register_engine_js (rlm/js_api.rs, called at
// the end of engine_register_builtins) registers the __rlm_* host functions,
// evals src/js/rlm.js, and then evals every driver here. These modules are
// pure JS over the already-registered primitives (sofuu.ai, sofuu.mcp,
// sofuu.memory, sofuu.rlm, fetch, sofuu.fs) — they need no host functions
// of their own, so wiring them in is just an eval:
//
//   src/js/agent.js — sofuu.agent.define/run/runMany/mapContext/cancel/...
//                     (first-class agents + sub-agents, PLAN-AGENTS A1-A5,
//                      A8, A9). Merges into the existing sofuu.agent object
//                     (memory.rs's agent.create/prefetch survive).
//   src/js/web.js   — sofuu.web.search/open: provider-neutral web search
//                     (keyless DuckDuckGo default; Brave/Tavily with keys).
//
// Ordering guarantee: the seam runs AFTER rt/memory.rs registered
// sofuu.agent.{create,prefetch}, and both drivers merge properties into the
// live namespace objects instead of replacing them, so nothing clobbers
// anything regardless of registration order.

// Binary budget (Phase 6, 2026-10-23): release embeds the .bc snapshots and
// NOT the sources. Both together cost 725,326 B of a 5 MiB binary, and the
// sources are dead weight at runtime - eval_shipped_with_bytecode only fell
// back to them when a snapshot failed to load. Dropping them frees 496,262 B;
// the Phase 5 work had already eaten the budget down to 12 KB of headroom.
//
// What keeps this honest:
//   * debug builds still embed every source (`cfg(debug_assertions)`), so
//     development and `cargo test` exercise the readable path;
//   * scripts/check-driver-bc.sh (`make bc-check`) recompiles all seven
//     drivers and fails if a committed snapshot is not byte-identical, so a
//     stale .bc cannot ship - it was the one failure mode with no guard
//     (eval_shipped_with_bytecode prefers the snapshot and returns on it).
use std::os::raw::c_void;

use sofuu_ffi::bridge::JSContext;
use sofuu_ffi::qjs;

/// The agent runtime driver — single source of truth is src/js/agent.js.
#[cfg(debug_assertions)]
pub const AGENT_JS: &str = include_str!("../../../src/js/agent.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const AGENT_JS: &str = "";
pub const AGENT_BC: &[u8] = include_bytes!("../../../src/js/agent.bc.gz");
/// The web-search driver — single source of truth is src/js/web.js.
#[cfg(debug_assertions)]
pub const WEB_JS: &str = include_str!("../../../src/js/web.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const WEB_JS: &str = "";
pub const WEB_BC: &[u8] = include_bytes!("../../../src/js/web.bc.gz");
/// The coding-tools driver — single source of truth is src/js/tools.js.
#[cfg(debug_assertions)]
pub const TOOLS_JS: &str = include_str!("../../../src/js/tools.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const TOOLS_JS: &str = "";
pub const TOOLS_BC: &[u8] = include_bytes!("../../../src/js/tools.bc.gz");
/// The automated fact extractor — single source of truth is src/js/extractor.js.
#[cfg(debug_assertions)]
pub const EXTRACTOR_JS: &str = include_str!("../../../src/js/extractor.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const EXTRACTOR_JS: &str = "";
pub const EXTRACTOR_BC: &[u8] = include_bytes!("../../../src/js/extractor.bc.gz");
/// The native structured storage driver — single source of truth is src/js/storage.js.
#[cfg(debug_assertions)]
pub const STORAGE_JS: &str = include_str!("../../../src/js/storage.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const STORAGE_JS: &str = "";
pub const STORAGE_BC: &[u8] = include_bytes!("../../../src/js/storage.bc.gz");
/// The chat turn engine — single source of truth is src/js/chat.js
/// (PLAN-DESKTOP C). Eval'd LAST: it builds on sofuu.agent/web/tools and
/// exposes sofuu.chat.* for embedded hosts (the desktop app). Hosts that
/// never call sofuu.chat.init pay nothing — the module only defines state.
#[cfg(debug_assertions)]
pub const CHAT_JS: &str = include_str!("../../../src/js/chat.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const CHAT_JS: &str = "";
pub const CHAT_BC: &[u8] = include_bytes!("../../../src/js/chat.bc.gz");
/// The HTTP edge server — single source of truth is src/js/serve.js
/// (workstream H1). `sofuu serve` boots it; the module itself only defines
/// state and functions, so a process that never serves pays nothing.
/// Eval'd after chat: its agent/tool routes use the driver namespaces.
#[cfg(debug_assertions)]
pub const SERVE_JS: &str = include_str!("../../../src/js/serve.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const SERVE_JS: &str = "";
pub const SERVE_BC: &[u8] = include_bytes!("../../../src/js/serve.bc.gz");
/// The recursive language model driver — single source of truth is src/js/rlm.js.
#[cfg(debug_assertions)]
pub const RLM_JS: &str = include_str!("../../../src/js/rlm.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const RLM_JS: &str = "";
pub const RLM_BC: &[u8] = include_bytes!("../../../src/js/rlm.bc.gz");
/// The plugins and extensibility standard driver — single source of truth is src/js/plugins.js.
#[cfg(debug_assertions)]
pub const PLUGINS_JS: &str = include_str!("../../../src/js/plugins.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const PLUGINS_JS: &str = "";
pub const PLUGINS_BC: &[u8] = include_bytes!("../../../src/js/plugins.bc.gz");
/// The first-party inference provider adapters — single source of truth is src/js/providers.js.
#[cfg(debug_assertions)]
pub const PROVIDERS_JS: &str = include_str!("../../../src/js/providers.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const PROVIDERS_JS: &str = "";
pub const PROVIDERS_BC: &[u8] = include_bytes!("../../../src/js/providers.bc.gz");
/// The observability CLI renderer — single source of truth is src/js/observability.js
/// (OBSERVABILITY-SPEC §4: the "sofuu logs" surface). Rust collects and
/// validates the event log (rt/events.rs, rt/observability.rs); this driver
/// only renders it, because rendering is the part that changes and the binary
/// is size-capped (§0). It MERGES into the sofuu.observability object the Rust
/// seam installed - adding cli/renderRun - and leaves runs/lastCheckpoint/
/// postmortem/crashCard untouched, so eval it AFTER those host functions exist.
#[cfg(debug_assertions)]
pub const OBSERVABILITY_JS: &str = include_str!("../../../src/js/observability.js");
/// Release builds carry bytecode only - see the file header.
#[cfg(not(debug_assertions))]
pub const OBSERVABILITY_JS: &str = "";
pub const OBSERVABILITY_BC: &[u8] = include_bytes!("../../../src/js/observability.bc.gz");

/// Inflate one gzipped embedded asset (AUDIT-2026-10-31 P1-8).
///
/// The shipped JS that only one code path needs is embedded gzipped, so its
/// text stops costing binary budget: `build.rs` compresses the committed .js
/// into OUT_DIR and the site that evals it inflates once into a
/// process-lifetime cache. Nothing here is on the startup path — a caller
/// that never reaches the feature never inflates.
///
/// `what` names the asset so a torn payload fails with the file it came from
/// rather than a bare zlib error. Returns None (after saying so) instead of an
/// empty string, so a caller can fail loudly rather than silently run without
/// the feature.
pub fn inflate_gz(gz: &[u8], what: &str) -> Option<String> {
    use std::io::Read;
    if gz.len() < 2 || gz[0] != 0x1f || gz[1] != 0x8b {
        eprintln!("[sofuu] {what}: embedded payload is not gzip (build.rs staging?)");
        return None;
    }
    let mut s = String::new();
    match flate2::read::GzDecoder::new(gz).read_to_string(&mut s) {
        Ok(_) if s.is_empty() => {
            eprintln!("[sofuu] {what}: embedded payload inflated to nothing");
            None
        }
        Ok(_) => Some(s),
        Err(e) => {
            eprintln!("[sofuu] {what}: embedded payload failed to inflate: {e}");
            None
        }
    }
}

/// Eval one shipped JS driver into the engine context. A driver failure
/// must never take the engine down (the drivers guard their own deps) —
/// log the exception and let startup continue, matching the rlm.js seam's
/// error policy.
///
/// # Safety
/// `ctx` must be a live QuickJS context on the engine thread.
pub unsafe fn eval_shipped(ctx: *mut c_void, src: &str, name: &str) {
    if ctx.is_null() {
        return;
    }
    let ctx = ctx as *mut JSContext;
    let Ok(c_src) = std::ffi::CString::new(src) else {
        return;
    };
    let Ok(c_name) = std::ffi::CString::new(name) else {
        return;
    };
    // SAFETY: ctx is the engine's live context; buffers outlive the call.
    unsafe {
        let r = qjs::JS_Eval(
            ctx,
            c_src.as_ptr(),
            src.len(),
            c_name.as_ptr(),
            qjs::JS_EVAL_TYPE_GLOBAL,
        );
        if qjs::is_exception(r) {
            let exc = qjs::JS_GetException(ctx);
            let p = qjs::sofuu_js_to_cstring(ctx, exc);
            if !p.is_null() {
                let msg = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
                qjs::sofuu_js_free_cstring(ctx, p);
                eprintln!("[sofuu] {name} driver eval failed: {msg}");
            }
            qjs::sofuu_js_free_value(ctx, exc);
        }
        qjs::sofuu_js_free_value(ctx, r);
    }
}

/// Load and evaluate precompiled QuickJS bytecode into the engine context.
/// Returns true if execution succeeded, false if bytecode read or eval threw.
///
/// # Safety
/// `ctx` must be a live QuickJS context on the engine thread.
pub unsafe fn eval_bytecode(ctx: *mut c_void, bytecode: &[u8], name: &str) -> bool {
    if ctx.is_null() || bytecode.is_empty() {
        return false;
    }
    let mut decompressed = Vec::new();
    let bc = if bytecode.len() >= 2 && bytecode[0] == 0x1f && bytecode[1] == 0x8b {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(bytecode);
        if decoder.read_to_end(&mut decompressed).is_ok() {
            &decompressed[..]
        } else {
            bytecode
        }
    } else {
        bytecode
    };
    let ctx = ctx as *mut JSContext;

    let fun_obj = qjs::JS_ReadObject(
        ctx,
        bc.as_ptr(),
        bc.len(),
        qjs::JS_READ_OBJ_BYTECODE | qjs::JS_READ_OBJ_ROM_DATA,
    );

    if qjs::is_exception(fun_obj) {
        let exc = qjs::JS_GetException(ctx);
        let p = qjs::sofuu_js_to_cstring(ctx, exc);
        if !p.is_null() {
            let msg = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, p);
            eprintln!("[sofuu] {name} bytecode read failed: {msg}");
        }
        qjs::sofuu_js_free_value(ctx, exc);
        return false;
    }

    // JS_EvalFunction executes the bytecode function and takes ownership of fun_obj
    let res = qjs::JS_EvalFunction(ctx, fun_obj);
    if qjs::is_exception(res) {
        let exc = qjs::JS_GetException(ctx);
        let p = qjs::sofuu_js_to_cstring(ctx, exc);
        if !p.is_null() {
            let msg = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, p);
            eprintln!("[sofuu] {name} bytecode eval failed: {msg}");
        }
        qjs::sofuu_js_free_value(ctx, exc);
        qjs::sofuu_js_free_value(ctx, res);
        return false;
    }

    qjs::sofuu_js_free_value(ctx, res);
    true
}

/// Compile a JavaScript source string into QuickJS bytecode.
///
/// # Safety
/// `ctx` must be a live QuickJS context.
pub unsafe fn compile_to_bytecode(ctx: *mut JSContext, src: &str, name: &str) -> Option<Vec<u8>> {
    if ctx.is_null() || src.is_empty() {
        return None;
    }
    let Ok(c_src) = std::ffi::CString::new(src) else {
        return None;
    };
    let Ok(c_name) = std::ffi::CString::new(name) else {
        return None;
    };

    let fun_obj = qjs::JS_Eval(
        ctx,
        c_src.as_ptr(),
        src.len(),
        c_name.as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL | qjs::JS_EVAL_FLAG_COMPILE_ONLY,
    );

    if qjs::is_exception(fun_obj) {
        let exc = qjs::JS_GetException(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
        qjs::sofuu_js_free_value(ctx, fun_obj);
        return None;
    }

    let mut out_len: usize = 0;
    let bc_ptr = qjs::JS_WriteObject(ctx, &mut out_len, fun_obj, qjs::JS_WRITE_OBJ_BYTECODE);
    qjs::sofuu_js_free_value(ctx, fun_obj);

    if bc_ptr.is_null() {
        return None;
    }

    let slice = std::slice::from_raw_parts(bc_ptr, out_len);
    let vec = slice.to_vec();
    qjs::js_free(ctx, bc_ptr as *mut std::ffi::c_void);
    Some(vec)
}

/// Evaluates a shipped driver: tries the fast precompiled bytecode path first;
/// if absent or fails, falls back silently to source evaluation.
///
/// # Safety
/// `ctx` must be a live QuickJS context on the engine thread.
pub unsafe fn eval_shipped_with_bytecode(
    ctx: *mut c_void,
    src: &str,
    bytecode: Option<&[u8]>,
    name: &str,
) {
    if let Some(bc) = bytecode {
        if !bc.is_empty() && eval_bytecode(ctx, bc, name) {
            return;
        }
    }
    if src.is_empty() {
        // Release build: bytecode only. Say so loudly instead of silently
        // starting without the driver (the old failure mode was a log line
        // from eval_shipped that looked like a user-script error).
        eprintln!("[sofuu] {name} unavailable: bytecode load failed and this build carries no source fallback (run `make bc-check`)");
        return;
    }
    eval_shipped(ctx, src, name);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AUDIT-2026-10-31 P1-8: the gzipped assets build.rs stages must
    /// inflate to exactly the committed sources. Staging in build.rs instead
    /// of committing a .gz is what makes drift impossible; this is the test
    /// that holds it to the files the binary actually embeds.
    #[test]
    fn gzipped_shipped_assets_match_their_sources() {
        for (asset, src) in [
            ("compat.js", include_str!("../../../src/js/compat.js")),
            ("rlm.js", include_str!("../../../src/js/rlm.js")),
            ("chat_tui.js", include_str!("../../../src/js/chat_tui.js")),
        ] {
            let path = std::path::Path::new(env!("OUT_DIR")).join(format!("{asset}.gz"));
            let gz = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("{asset}.gz not staged by build.rs: {e}"));
            let got = inflate_gz(&gz, asset).unwrap_or_else(|| panic!("{asset}.gz: inflate failed"));
            assert_eq!(got.len(), src.len(), "{asset}: inflated length differs from source");
            assert!(got == src, "{asset}: staged gzip does not match src/js/{asset}");
        }
    }

    /// Source text is embedded in debug builds only, so this runs there.
    #[cfg(debug_assertions)]
    #[test]
    fn shipped_sources_are_nonempty_and_parseable_names() {
        assert!(AGENT_JS.contains("sofuu.agent"));
        // the exported API surface the CLI + chat rely on
        assert!(AGENT_JS.contains("A.run = run"));
        assert!(AGENT_JS.contains("A.mapContext = mapContext"));
        assert!(WEB_JS.contains("sofuu.web"));
        assert!(WEB_JS.contains("web_search"));
        // the chat turn engine's exported API surface (desktop consumes it)
        assert!(CHAT_JS.contains("sofuu.chat"));
        assert!(CHAT_JS.contains("chat.submit"));
        assert!(CHAT_JS.contains("chat.init"));
        // the native provider adapters register into sofuu.plugins + expose
        // sofuu.providers (PLUGIN_PLAN Phase 2)
        assert!(PROVIDERS_JS.contains("sofuu.providers"));
        assert!(PROVIDERS_JS.contains("sofuu.plugins"));
        let rt = unsafe { qjs::JS_NewRuntime() };
        assert!(!rt.is_null());
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        assert!(!ctx.is_null());

        // Verify each shipped driver compiles cleanly to QuickJS bytecode
        for (name, src) in [
            ("agent.js", AGENT_JS),
            ("web.js", WEB_JS),
            ("tools.js", TOOLS_JS),
            ("extractor.js", EXTRACTOR_JS),
            ("storage.js", STORAGE_JS),
            ("chat.js", CHAT_JS),
            ("plugins.js", PLUGINS_JS),
            ("providers.js", PROVIDERS_JS),
            ("observability.js", OBSERVABILITY_JS),
        ] {
            let bc = unsafe { compile_to_bytecode(ctx, src, name) };
            assert!(bc.is_some(), "{name}: failed to compile to bytecode");
            let bytes = bc.unwrap();
            assert!(!bytes.is_empty(), "{name}: bytecode was empty");
        }

        unsafe {
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }

    #[test]
    fn test_bytecode_compilation_and_execution() {
        let rt = unsafe { qjs::JS_NewRuntime() };
        assert!(!rt.is_null());
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        assert!(!ctx.is_null());

        let sample_js = "globalThis.__bc_test_val = 42 * 2;";
        let bc = unsafe { compile_to_bytecode(ctx, sample_js, "<test>") }.expect("bytecode compile");
        assert!(!bc.is_empty(), "compiled bytecode must not be empty");

        // Fresh context to evaluate bytecode
        let ctx2 = unsafe { qjs::JS_NewContext(rt) };
        let ok = unsafe { eval_bytecode(ctx2 as *mut c_void, &bc, "<test>") };
        assert!(ok, "eval_bytecode must succeed");

        let global = unsafe { qjs::sofuu_js_get_global_object(ctx2) };
        let val = unsafe { qjs::sofuu_js_get_property_str(ctx2, global, c"__bc_test_val".as_ptr()) };
        let mut i: i32 = 0;
        unsafe { qjs::JS_ToInt32(ctx2, &mut i, val) };
        assert_eq!(i, 84, "bytecode execution must yield exact result");

        unsafe {
            qjs::sofuu_js_free_value(ctx2, val);
            qjs::sofuu_js_free_value(ctx2, global);
            qjs::JS_FreeContext(ctx2);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
        }
    }
}
