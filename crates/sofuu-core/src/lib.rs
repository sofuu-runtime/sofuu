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
pub mod brain_cmd; // W7: `sofuu brain verify|repair|stats|backup|restore|migrate`
pub mod capabilities; // R1: the capability manifest (sofuu.capabilities)
pub mod compiler; // `sofuu compile`: append-trailer standalone agent binaries
pub mod embedding; // compact learned semantic projector for memory recall
pub mod embed_config; // H2: process-global embedded-mode config (is_embedded, config_root, signals)
pub mod ffi_exports; // C-ABI exports consumed by the remaining C core (Track D3)
pub mod frontend; // AI-Native Frontend Engine (JSX/TSX transpiler + micro-runtime + dev server)

// M10: the runtime + bridge end-to-end test lives HERE (not in sofuu-ffi)
// to keep sofuu-ffi free of a dev-dependency on this crate — with the
// engine ported to Rust, sofuu-ffi defines no_mangle symbols, and the
// dev-dep cycle would link sofuu-ffi twice in its own test binary (fatal
// under fat LTO). This module drives SofuuRuntime through the ffi wrapper.
#[cfg(test)]
mod ffi_runtime_test {
    use crate::rt::test_loop_lock;

    #[test]
    fn runtime_and_bridge_work() {
        // The runtime owns the process-global libuv loop — serialize with
        // the rt tests that pump it, like every other loop-touching test.
        let _guard = test_loop_lock();

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

    /// AUDIT-2026-10-31 P2-2: `sofuu eval --code-file m.js` on a file using ESM
    /// syntax died with "SyntaxError: unexpected token in expression: 'export'"
    /// because the string path only ever compiled as a GLOBAL script. A source
    /// that will not parse as a script but does parse as a module must run as
    /// one; a plain script must stay a script.
    #[test]
    fn eval_string_accepts_module_source() {
        let _guard = test_loop_lock();
        let rt = sofuu_ffi::SofuuRuntime::init().expect("runtime init");

        // Module-only syntax: this is the failing case from the audit.
        let rc = rt.eval_string(
            "export const answer = 41;\nglobalThis.__p22 = answer + 1;\n",
            "<p2-2-module>",
        );
        assert_eq!(rc, 0, "module-source eval must succeed");
        assert_eq!(
            rt.eval_string(
                "if (globalThis.__p22 !== 42) throw new Error('module body did not run: ' + globalThis.__p22);",
                "<p2-2-check>",
            ),
            0,
            "the module body must actually have executed"
        );

        // Plain scripts keep working (and still see the script-only globals).
        assert_eq!(rt.eval_string("var plainVar = 7;", "<p2-2-script>"), 0);
        assert_eq!(
            rt.eval_string("if (plainVar !== 7) throw new Error('script binding lost');", "<p2-2-check2>"),
            0
        );

        // Import syntax needs a real module graph: the entry is a module, so
        // its failed import must surface as an error, not as a syntax error.
        assert_ne!(
            rt.eval_string("import './definitely-missing.js';", "<p2-2-import>"),
            0,
            "a broken import must fail the eval"
        );

        // Broken source still fails, and the syntax error is the script's.
        assert_ne!(rt.eval_string("this is not javascript", "<p2-2-broken>"), 0);
    }

    /// AUDIT-2026-10-31 P2-10: `sofuu run a.js` evaluated the ENTRY
    /// module twice when the entry was reached by a different spelling than the
    /// importer used ("./a.js" vs the absolute path), because the engine keyed
    /// loaded_modules by the filename string it was handed. The entry's own
    /// name must be the canonical path so a cycle closes on one identity.
    #[test]
    fn eval_file_cycle_loads_entry_once() {
        let _guard = test_loop_lock();
        static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sofuu_p210_{}_{}", std::process::id(), seq));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.js"),
            "import './b.js';\nglobalThis.__order = (globalThis.__order || '') + 'A';\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("b.js"),
            "import './a.js';\nglobalThis.__order = (globalThis.__order || '') + 'B';\n",
        )
        .unwrap();

        let rt = sofuu_ffi::SofuuRuntime::init().expect("runtime init");
        let entry = dir.join("a.js");
        let rc = rt.eval_file(entry.to_str().unwrap());
        assert_eq!(rc, 0, "the cyclic graph must evaluate");
        let check = format!(
            "if (globalThis.__order !== 'BA') throw new Error('order=' + globalThis.__order);"
        );
        assert_eq!(
            rt.eval_string(&check, "<p2-10-check>"),
            0,
            "the entry must be evaluated exactly once (a second visit shows up as a third letter)"
        );
        let _ = std::fs::remove_dir_all(&dir);
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
pub mod infer; // CPU-first GGUF inference engine (C++ SIMD GEMV kernels + Rust)
pub mod mcp;
pub mod memory;
pub mod ml; // PLAN-ML-GATES: context-economy models + the in-memory working set
pub mod modules; // M2: console shell (more land in later phases)
pub mod npm;
pub mod rlm;
pub mod rt; // M1: event loop + promise bridge (replaces src/io/loop.c + promises.c)
pub mod shipped; // shipped JS drivers beyond RLM: agent.js + web.js (PLAN-AGENTS)
pub mod ts;

/// Whether THIS build linked the QTSQ brain/KV codec (sofuu-core/build.rs).
///
/// A QTSQ-free build is not a degraded `sofuu.memory` — the whole memory
/// surface is compiled out (rt/memory.rs gates the register bodies on
/// `#[cfg(has_qtsq)]`), so JS sees no `sofuu.memory` at all and the
/// funnel answers `unknown_method`. Callers that must behave correctly in
/// BOTH configurations (the libsofuu test suite, `sofuu doctor`, docs)
/// branch on this rather than assuming a brain exists.
pub const HAS_QTSQ: bool = cfg!(has_qtsq);
