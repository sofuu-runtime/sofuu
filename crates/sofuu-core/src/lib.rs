// sofuu-core — Sofuu Rust core library.
//
// Exposes the Rust-ported subsystems as a reusable API:
//   - memory: CMA cognitive memory (HNSW + decay + consolidation + dream)
//   - http::sse: streaming SSE parser
//   - mcp::jsonrpc: MCP JSON-RPC 2.0 (safe parsing)
//   - ts: TypeScript type stripper
//   - bundler: ESM bundler (graph + transform + emit)
//   - npm: npm resolution + safe tarball extraction + SHA-1 integrity
//   - rlm: Recursive-Language-Model scaffolding — hermetic QuickJS sandbox,
//     episode loop, routing heuristics (PLAN-RLM Part 1)
//
// The binary (main.rs) uses sofuu-ffi for the C runtime; these modules are
// pure Rust logic with no C dependency.

pub mod bundler;
pub mod embed_config; // H2: process-global embedded-mode config (is_embedded, config_root, signals)
pub mod ffi_exports; // C-ABI exports consumed by the remaining C core (Track D3)

// M10: the runtime + bridge end-to-end test lives HERE (not in sofuu-ffi)
// to keep sofuu-ffi free of a dev-dependency on this crate — with the
// engine ported to Rust, sofuu-ffi defines no_mangle symbols, and the
// dev-dep cycle would link sofuu-ffi twice in its own test binary (fatal
// under fat LTO). This module drives SofuuRuntime through the ffi wrapper.
#[cfg(test)]
mod ffi_runtime_test {
    use crate::rt::TEST_LOOP_LOCK;

    #[test]
    fn runtime_and_bridge_work() {
        // The runtime owns the process-global libuv loop — serialize with
        // the rt tests that pump it, like every other loop-touching test.
        let _guard = TEST_LOOP_LOCK.lock().unwrap();

        let rt = sofuu_ffi::SofuuRuntime::init().expect("runtime init");

        let rc = rt.eval_string("1 + 1", "<test>");
        assert_eq!(rc, 0);

        let rc = rt.eval_file("/nonexistent/does-not-exist.js");
        assert_ne!(rc, 0);

        // Register a Rust callback and call it from JS.
        let ctx = rt.engine_ctx() as *mut sofuu_ffi::qjs::JSContext;
        assert!(!ctx.is_null());
        // SAFETY: ctx is the valid QuickJS context; test_fn is 'static.
        unsafe {
            sofuu_ffi::bridge::register_global_fn(
                ctx,
                "__test_fn",
                test_fn as sofuu_ffi::bridge::JSCFunction,
            )
        };
        let rc = rt.eval_string(
            "if (__test_fn() !== 'hello-from-rust') throw new Error('bad: ' + __test_fn());",
            "<bridge-test>",
        );
        assert_eq!(rc, 0, "Rust bridge fn should be callable");
    }

    unsafe extern "C" fn test_fn(
        ctx: *mut sofuu_ffi::qjs::JSContext,
        _this: sofuu_ffi::qjs::JSValueConst,
        _argc: std::os::raw::c_int,
        _argv: *const sofuu_ffi::qjs::JSValueConst,
    ) -> sofuu_ffi::qjs::JSValue {
        sofuu_ffi::bridge::js_new_string(ctx, "hello-from-rust")
    }
}
pub mod http;
pub mod mcp;
pub mod memory;
pub mod modules; // M2: console shell (more land in later phases)
pub mod npm;
pub mod rlm;
pub mod rt; // M1: event loop + promise bridge (replaces src/io/loop.c + promises.c)
pub mod shipped; // shipped JS drivers beyond RLM: agent.js + web.js (PLAN-AGENTS)
pub mod ts;
