// sofuu-ffi — safe Rust bindings to the Sofuu C core.
//
// ALL `unsafe` in the codebase lives here (or in the C itself). The rest of
// Sofuu is safe Rust. Each binding wraps the C API in an ownership-safe Rust
// type: `SofuuRuntime` owns the `SofuuRuntime*` and frees it on Drop.
//
// Phase 0: we bind just enough to make the Rust binary a drop-in for the C
// main() — init/eval/run-jobs/destroy + the REPL eval path. More bindings
// (chat bridge, HTTP, MCP, memory) land as those subsystems migrate.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

pub mod bridge;
pub mod curl; // M4: fetch() over libcurl (multi-handle ↔ libuv bridge)
pub mod http_parser; // M5: vendored http-parser (server req parsing)
pub mod qjs;
pub mod qjs_rt;
#[cfg(has_qtsq)]
pub mod qtsq; // M9: QTSQ codec bindings (brain/KV persistence; crates/sofuu-core/src/rt/memory.rs)
pub mod uv;

// ── C API (mirrors src/sofuu.h) ─────────────────────────────────

// Opaque handle to the C SofuuRuntime. Distinct name to avoid clashing
// with the Rust wrapper type below.
#[repr(C)]
struct CSofuuRuntime {
    engine: *mut c_void,
}

extern "C" {
    fn sofuu_init() -> *mut CSofuuRuntime;
    fn sofuu_eval_file(rt: *mut CSofuuRuntime, path: *const c_char) -> c_int;
    fn sofuu_eval_string(rt: *mut CSofuuRuntime, source: *const c_char, filename: *const c_char) -> c_int;
    fn sofuu_run_jobs(rt: *mut CSofuuRuntime);
    fn sofuu_destroy(rt: *mut CSofuuRuntime);
    fn sofuu_get_engine(rt: *mut CSofuuRuntime) -> *mut c_void;
}

// engine.h
extern "C" {
    fn engine_eval_repl(eng: *mut c_void, source: *const c_char, is_error: *mut c_int) -> *mut c_char;
}

// engine.h — the C engine dies in M10; these symbols are exported from
// crates/sofuu-core/src/rt/engine.rs (sofuu_init/eval_*/run_jobs/destroy/
// get_engine + sofuu_engine_ctx + engine_eval_repl).
extern "C" {
    fn sofuu_engine_ctx(eng: *mut c_void) -> *mut c_void;
    // src/io/tui.c also retired in M10 — same bridges now exported
    // from rt/tui.rs (sofuu_tui_active/log/width/reset/clear_gap/discard_last/overlay_row aliases).
    fn sofuu_tui_active() -> c_int;
    fn sofuu_tui_log(line: *const c_char);
    fn sofuu_tui_width() -> c_int;
    fn sofuu_tui_reset();
    fn sofuu_tui_clear_gap();
    fn sofuu_tui_discard_last();
    fn sofuu_tui_overlay_row(row: c_int, content: *const c_char);
    fn sofuu_tui_phase_row() -> c_int;
}

// ── TUI routing (Rust shell prints → alt-screen conversation) ──────

/// True while the chat's full-screen UI owns the terminal.
pub fn tui_active() -> bool {
    // SAFETY: trivial C getter.
    unsafe { sofuu_tui_active() != 0 }
}

/// Append a line to the TUI conversation area (no-op outside the TUI).
pub fn tui_log(line: &str) {
    let c = CString::new(line).unwrap_or_default();
    // SAFETY: c is valid for the call; C copies it.
    unsafe { sofuu_tui_log(c.as_ptr()) };
}

/// Terminal width in cells (for composing full-width panels). Falls back
/// to 80 when the size can't be queried.
pub fn tui_width() -> usize {
    // SAFETY: trivial C getter.
    let w = unsafe { sofuu_tui_width() };
    if w > 0 { w as usize } else { 80 }
}

/// Clear the TUI conversation buffer (used when config changes so the
/// welcome panel re-logs with the new model/provider).
pub fn tui_reset() {
    // SAFETY: trivial C call.
    unsafe { sofuu_tui_reset() };
}

pub fn tui_clear_gap() {
    unsafe { sofuu_tui_clear_gap() };
}

pub fn tui_discard_last() {
    unsafe { sofuu_tui_discard_last() };
}

/// Overwrite one absolute screen row (used by the animated logo).
pub fn tui_overlay_row(row: i32, content: &str) {
    let c = CString::new(content).unwrap_or_default();
    // SAFETY: c is valid for the call; C writes it to the terminal.
    unsafe { sofuu_tui_overlay_row(row, c.as_ptr()) };
}

/// The gap row above the input box top border (h-6) — where the animated
/// agent-phase indicator ("Thinking"/"Coding"/…) is painted via
/// `tui_overlay_row`. Same row `tui_clear_gap()` blanks.
pub fn tui_phase_row() -> i32 {
    // SAFETY: trivial C getter.
    unsafe { sofuu_tui_phase_row() }
}

// resolver.h
extern "C" {
    fn npm_install(pkg_spec: *const c_char, dest_dir: *const c_char) -> c_int;
    fn npm_install_local_package_json(dest_dir: *const c_char) -> c_int;
}

// ── Safe wrapper: the runtime ────────────────────────────────────

/// Owns a Sofuu C runtime. Freed on drop.
pub struct SofuuRuntime {
    engine: *mut CSofuuRuntime,
}

// The C runtime is used from the single main thread (like the C binary).
unsafe impl Send for SofuuRuntime {}

impl SofuuRuntime {
    /// Initialize the runtime (QuickJS context + builtins).
    pub fn init() -> Option<Self> {
        // SAFETY: sofuu_init returns a heap-allocated runtime or NULL.
        let ptr = unsafe { sofuu_init() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { engine: ptr })
        }
    }

    /// Evaluate a JS file. Returns 0 on success, non-zero on failure.
    pub fn eval_file(&self, path: &str) -> i32 {
        let c_path = CString::new(path).unwrap_or_default();
        // SAFETY: c_path is valid for the call; self.engine is valid.
        unsafe { sofuu_eval_file(self.engine, c_path.as_ptr()) }
    }

    /// Evaluate a JS string. Returns 0 on success.
    pub fn eval_string(&self, source: &str, filename: &str) -> i32 {
        let c_src = CString::new(source).unwrap_or_default();
        let c_name = CString::new(filename).unwrap_or_default();
        // SAFETY: both CStrings are valid for the duration of the call.
        unsafe { sofuu_eval_string(self.engine, c_src.as_ptr(), c_name.as_ptr()) }
    }

    /// Pump the microtask/job queue (async completion).
    pub fn run_jobs(&self) {
        // SAFETY: self.engine is valid.
        unsafe { sofuu_run_jobs(self.engine) }
    }

    /// Access the underlying engine pointer (for REPL eval).
    fn engine(&self) -> *mut c_void {
        // SAFETY: self.engine is valid; returns the engine handle.
        unsafe { sofuu_get_engine(self.engine) }
    }

    /// Access the QuickJS context (for registering native callbacks).
    pub fn engine_ctx(&self) -> *mut c_void {
        // SAFETY: shim reads the ctx from the C SofuuEngine (no layout guess).
        unsafe { sofuu_engine_ctx(self.engine()) }
    }

    // NOTE: `bundle` was removed in the D3 wiring — `sofuu bundle` is now
    // pure Rust (sofuu_core::bundler). The C `sofuu_bundle` remains only in
    // the c-only build.

    /// Install an npm package into dest_dir. 0 on success.
    pub fn npm_install(&self, spec: &str, dest_dir: &str) -> i32 {
        let c_spec = CString::new(spec).unwrap_or_default();
        let c_dir = CString::new(dest_dir).unwrap_or_default();
        // SAFETY: both CStrings valid.
        unsafe { npm_install(c_spec.as_ptr(), c_dir.as_ptr()) }
    }

    /// Install all deps from package.json in dest_dir. 0 on success.
    pub fn npm_install_local(&self, dest_dir: &str) -> i32 {
        let c_dir = CString::new(dest_dir).unwrap_or_default();
        // SAFETY: c_dir valid.
        unsafe { npm_install_local_package_json(c_dir.as_ptr()) }
    }

    /// REPL-style eval: returns the pretty-printed result as a Rust String
    /// (error text included), or None if the result was undefined.
    pub fn eval_repl(&self, source: &str) -> Option<String> {
        let c_src = CString::new(source).unwrap_or_default();
        let mut is_error: c_int = 0;
        let eng = self.engine();
        if eng.is_null() {
            return None;
        }
        // SAFETY: eng is a valid SofuuEngine*; engine_eval_repl returns a
        // malloc'd string we must free. We convert to Rust String, then free.
        let out = unsafe { engine_eval_repl(eng, c_src.as_ptr(), &mut is_error) };
        if out.is_null() {
            return None;
        }
        // SAFETY: out is a valid C string (malloc'd by engine_eval_repl).
        let s = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        // SAFETY: out was allocated by malloc (via strdup in C) — libc::free.
        unsafe { libc::free(out as *mut c_void) };
        Some(s)
    }
}

impl Drop for SofuuRuntime {
    fn drop(&mut self) {
        // SAFETY: self.engine is a valid runtime from sofuu_init.
        unsafe { sofuu_destroy(self.engine) };
    }
}

// ── QTSQ session store (free functions — no runtime state) ────────
// Session data persists as one `.qtsq` file per session (sanitize +
// password-vault encrypted; fail-closed codec). The password is derived
// per project by the caller (crates/sofuu-core/src/session.rs).

/// Persist a session-data payload as a `.qtsq` file. 0 on success (QTSQ_OK).
/// Returns -1 without QTSQ (fail-closed, same as the retired C shims).
pub fn qtsq_session_save(path: &str, data: &[u8], password: &str) -> i32 {
    let c_path = CString::new(path).unwrap_or_default();
    let c_pw = CString::new(password).unwrap_or_default();
    #[cfg(has_qtsq)]
    {
        // SAFETY: buffers are valid for the duration of the call; the codec
        // (qtsq.rs, M10 — formerly src/ffi_shim.c) copies the payload in.
        unsafe { qtsq::qtsq_session_save(c_path.as_ptr(), data.as_ptr(), data.len(), c_pw.as_ptr()) }
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = (&c_path, data, &c_pw);
        -1 // no QTSQ — sessions can't persist
    }
}

/// Load a session-data `.qtsq` file, decrypting with the same per-project
/// password. Returns the decompressed JSON payload or None (also None
/// without QTSQ — fail-closed).
pub fn qtsq_session_load(path: &str, password: &str) -> Option<Vec<u8>> {
    let c_path = CString::new(path).unwrap_or_default();
    let c_pw = CString::new(password).unwrap_or_default();
    #[cfg(has_qtsq)]
    {
        let mut out_size: usize = 0;
        // SAFETY: the codec returns a malloc'd buffer we copy out and free.
        let ptr = unsafe { qtsq::qtsq_session_load(c_path.as_ptr(), c_pw.as_ptr(), &mut out_size) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: ptr is valid for out_size bytes (malloc'd by the codec).
        let bytes = unsafe { std::slice::from_raw_parts(ptr, out_size) }.to_vec();
        // SAFETY: ptr was allocated with malloc — plain libc::free.
        unsafe { libc::free(ptr as *mut c_void) };
        Some(bytes)
    }
    #[cfg(not(has_qtsq))]
    {
        let _ = (&c_path, &c_pw);
        None
    }
}

// ── Bindings tests ───────────────────────────────────────────────
//
// M10: the runtime+bridge end-to-end test moved to sofuu-core (lib.rs
// ffi_runtime_test) so this crate has NO dev-dependency on sofuu-core —
// the M10 no_mangle shims make a dev-dep cycle unmixable with fat LTO.
// These tests exercise the bindings directly (qjs/uv/qtsq) and never touch
// the process-global runtime, so they are safe under parallel execution.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qjs;

    // ── M0 keystone tests: the new qjs bindings boot and drive QuickJS ──

    /// Boot a standalone QuickJS context through the M0 bindings, eval
    /// `1 + 1`, and read the result `2` back — no C glue involved.
    #[test]
    fn m0_qjs_boots_context_and_evals() {
        use crate::qjs;
        // SAFETY: JS_NewRuntime/JS_NewContext are the canonical init path.
        let rt = unsafe { qjs::JS_NewRuntime() };
        assert!(!rt.is_null());
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        assert!(!ctx.is_null());
        // SAFETY: ctx is a fresh valid context on this thread.
        let ctp = unsafe { qjs::CtxPtr::new(ctx) };

        let src = c"1 + 1";
        // SAFETY: src is a valid C string; global eval, len from cstr.
        let result = unsafe {
            qjs::JS_Eval(
                ctx,
                src.as_ptr(),
                5,
                c"<m0-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(result) }, "1+1 must not throw");

        // The value should be the int 2. Read via JS_ToInt32 (exported).
        let mut out: i32 = -1;
        // SAFETY: result is a live value of ctx.
        assert!(unsafe { qjs::to_int32(ctp, result, &mut out) });
        assert_eq!(out, 2, "eval(1+1) must be 2");

        // SAFETY: result was created by this ctx and must be freed.
        unsafe { qjs::sofuu_js_free_value(ctx, result) };

        // SAFETY: teardown of the context/runtime we created.
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// Register a Rust native function into the global object using ONLY
    /// the M0 bindings (no C glue), then call it from JS.
    #[test]
    fn m0_rust_native_module_callable_from_js() {
        use crate::qjs;

        // SAFETY: standalone runtime + context.
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { qjs::CtxPtr::new(ctx) };

        // SAFETY: m0_hello is 'static; name is a static C string.
        let func = unsafe { qjs::new_cfunction(ctp, "m0_hello", m0_hello_fn, 0) };
        // SAFETY: global object of ctx.
        let global = unsafe { qjs::global_object(ctp) };
        // SAFETY: global is live; func is live (property now owns it — do
        // NOT free func afterward; JS_SetPropertyStr takes ownership here).
        let rc = unsafe { qjs::sofuu_js_set_property_str(ctx, global, c"m0_hello".as_ptr(), func) };
        assert!(rc >= 0, "set property must succeed");
        // SAFETY: global must be freed (we created it).
        unsafe { qjs::sofuu_js_free_value(ctx, global) };

        // Call it from JS and check the return value through the bindings.
        let src = c"m0_hello()";
        let result = unsafe {
            qjs::JS_Eval(
                ctx,
                src.as_ptr(),
                10,
                c"<m0-call>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            )
        };
        assert!(!unsafe { qjs::is_exception(result) });
        let s = unsafe { qjs::value_to_string(ctp, result) };
        assert_eq!(s.as_deref(), Some("hello-from-m0"));
        // SAFETY: result is live; free it.
        unsafe { qjs::sofuu_js_free_value(ctx, result) };

        // SAFETY: teardown.
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    /// The M0 GC guard: Rooted<T> takes ownership and frees on drop.
    /// (Rooted must be dropped BEFORE the context is freed — the test scopes
    /// it in a block so drop order is explicit.)
    #[test]
    fn m0_rooted_keeps_value_alive() {
        use crate::qjs;
        // SAFETY: standalone context.
        let rt = unsafe { qjs::JS_NewRuntime() };
        let ctx = unsafe { qjs::JS_NewContext(rt) };
        let ctp = unsafe { qjs::CtxPtr::new(ctx) };

        let s = {
            // SAFETY: string value created here (refcount 1); Rooted takes it.
            let v = unsafe { qjs::new_string(ctp, "rooted-string") };
            let rooted = unsafe { qjs::Rooted::new(ctp, v) };
            // Read through the rooted handle — value is alive (refcount 1).
            // SAFETY: rooted.as_value() is live.
            let s = unsafe { qjs::value_to_string(ctp, rooted.as_value()) };
            // rooted drops HERE (block end) → frees exactly once (1 → 0),
            // while the context is still alive.
            s
        };
        assert_eq!(s.as_deref(), Some("rooted-string"));

        // SAFETY: teardown.
        unsafe { qjs::JS_FreeContext(ctx) };
        unsafe { qjs::JS_FreeRuntime(rt) };
    }

    // The M0 native-module callback: returns "hello-from-m0" as a string.
    unsafe extern "C" fn m0_hello_fn(
        ctx: *mut qjs::JSContext,
        _this: qjs::JSValueConst,
        _argc: c_int,
        _argv: *const qjs::JSValueConst,
    ) -> qjs::JSValue {
        // SAFETY: ctx is the calling context; new_string creates a value.
        unsafe { qjs::new_string(qjs::CtxPtr::new(ctx), "hello-from-m0") }
    }

}
