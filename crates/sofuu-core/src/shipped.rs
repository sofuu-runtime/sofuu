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
/// The web-search driver — single source of truth is src/js/web.js.
pub const WEB_JS: &str = include_str!("../../../src/js/web.js");
/// The coding-tools driver — single source of truth is src/js/tools.js.
pub const TOOLS_JS: &str = include_str!("../../../src/js/tools.js");

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
        // Balanced braces is a weak syntax smoke check — the real gate is
        // the engine evaluating both drivers on every boot (ffi_runtime_test
        // boots a full runtime, so a syntax error there fails the suite).
        for (name, src) in [("agent.js", AGENT_JS), ("web.js", WEB_JS)] {
            let open = src.matches('{').count();
            let close = src.matches('}').count();
            assert_eq!(open, close, "{name}: unbalanced braces");
        }
    }
}
