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

use std::os::raw::c_void;

use sofuu_ffi::bridge::JSContext;
use sofuu_ffi::qjs;

/// The agent runtime driver — single source of truth is src/js/agent.js.
pub const AGENT_JS: &str = include_str!("../../../src/js/agent.js");
pub const AGENT_BC: &[u8] = include_bytes!("../../../src/js/agent.bc");
/// The web-search driver — single source of truth is src/js/web.js.
pub const WEB_JS: &str = include_str!("../../../src/js/web.js");
pub const WEB_BC: &[u8] = include_bytes!("../../../src/js/web.bc");
/// The coding-tools driver — single source of truth is src/js/tools.js.
pub const TOOLS_JS: &str = include_str!("../../../src/js/tools.js");
pub const TOOLS_BC: &[u8] = include_bytes!("../../../src/js/tools.bc");
/// The automated fact extractor — single source of truth is src/js/extractor.js.
pub const EXTRACTOR_JS: &str = include_str!("../../../src/js/extractor.js");
pub const EXTRACTOR_BC: &[u8] = include_bytes!("../../../src/js/extractor.bc");
/// The native structured storage driver — single source of truth is src/js/storage.js.
pub const STORAGE_JS: &str = include_str!("../../../src/js/storage.js");
pub const STORAGE_BC: &[u8] = include_bytes!("../../../src/js/storage.bc");
/// The chat turn engine — single source of truth is src/js/chat.js
/// (PLAN-DESKTOP C). Eval'd LAST: it builds on sofuu.agent/web/tools and
/// exposes sofuu.chat.* for embedded hosts (the desktop app). Hosts that
/// never call sofuu.chat.init pay nothing — the module only defines state.
pub const CHAT_JS: &str = include_str!("../../../src/js/chat.js");
pub const CHAT_BC: &[u8] = include_bytes!("../../../src/js/chat.bc");
/// The recursive language model driver — single source of truth is src/js/rlm.js.
pub const RLM_JS: &str = include_str!("../../../src/js/rlm.js");
pub const RLM_BC: &[u8] = include_bytes!("../../../src/js/rlm.bc");

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
    let ctx = ctx as *mut JSContext;

    let fun_obj = qjs::JS_ReadObject(
        ctx,
        bytecode.as_ptr(),
        bytecode.len(),
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
    eval_shipped(ctx, src, name);
}

#[cfg(test)]
mod tests {
    use super::*;

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
