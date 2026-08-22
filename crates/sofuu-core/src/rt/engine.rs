// rt/engine.rs — QuickJS engine lifecycle + file evaluation (M10).
//
// Port of the retired `src/engine/engine.c` + `src/sofuu.c` +
// `src/repl/repl.c` (the REPL loop itself has lived in Rust main.rs since
// M0 — only `engine_eval_repl` + `repl_inspect` moved here, verbatim).
// The C ABI is preserved exactly: `sofuu_init` / `sofuu_eval_file` /
// `sofuu_eval_string` / `sofuu_run_jobs` / `sofuu_destroy` /
// `sofuu_get_engine` (sofuu.h), `engine_eval_repl` + `sofuu_engine_ctx`
// (engine.h, consumed by sofuu-ffi's SofuuRuntime), and the ESM module
// loader symbol registered with QuickJS (`sofuu_module_loader`).
//
// Boot order (byte-identical to engine.c):
//   JS_NewRuntime → memory limit (512MB) → GC threshold (64MB) →
//   JS_NewContext → js_init_module_std/os → JS_SetModuleLoaderFunc →
//   rejection tracker (sofuu_rt_install_rejection_tracker) → sofuu_loop_init.
// `engine_register_builtins` calls every mod_*_register in the C order
// (property insertion order is JS-visible), then builds the capital-S
// `Sofuu` convenience aliases with the same dup/ownership semantics.
//
// The ESM loader is the one piece with real logic: resolve_module_path's
// canonicalize/fallback ladder (realpath semantics), the TS strip hook
// (sofuu_ts_strip_rs), the CJS wrapper (is_cjs/cjs_to_esm), and the
// loader-base handoff for transitive imports — all behavior-for-behavior
// with the C original, including its "return a best guess so the error
// message names a useful file" fallbacks.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use crate::ffi_exports::sofuu_ts_strip_rs;
use crate::modules::{
    console::mod_console_register,
    process::{mod_process_cleanup, mod_process_register, process_dispatch_uncaught},
};
use crate::rlm::js_api::sofuu_rust_register_engine_js;
use crate::rt::{
    ai::mod_ai_register,
    cjs::{cjs_to_esm, is_cjs, mod_cjs_register},
    event_loop::{sofuu_loop_close, sofuu_loop_init, sofuu_loop_run},
    fs::mod_fs_register,
    http_client::mod_http_client_register,
    http_sse::mod_http_sse_register,
    http_server::mod_http_server_register,
    mcp::mod_mcp_register,
    npm::npm_resolve,
    process_spawn::mod_subprocess_register,
    promise::{sofuu_rt_install_rejection_tracker, sofuu_rt_report_pending_rejections},
    timer::mod_timer_register,
};
#[cfg(has_qtsq)]
use crate::rt::memory::{mod_agent_register, mod_kv_register, mod_memory_register};
use sofuu_ffi::qjs;

/// The public runtime handle — mirrors `struct SofuuRuntime { SofuuEngine
/// *engine; }` (src/sofuu.c). sofuu-ffi's `CSofuuRuntime` has the same
/// layout; only the final binary links the two sides together.
#[repr(C)]
pub struct SofuuRuntime {
    engine: *mut c_void,
}

/// `struct SofuuEngine { JSRuntime *rt; JSContext *ctx; }` (engine.h).
#[repr(C)]
pub struct SofuuEngine {
    rt: *mut qjs::JSRuntime,
    ctx: *mut qjs::JSContext,
}

// The ESM loader's base path (a stable heap string; QuickJS stores an
// opaque pointer to it — C kept a `static char *g_loader_base`; the same
// lifetime contract holds, per-thread for the same reason).
thread_local! {
    static G_LOADER_BASE: RefCell<Option<CString>> = const { RefCell::new(None) };
}

// ── sofuu.h ABI ───────────────────────────────────────────────────

/// Create a runtime: engine_create + engine_register_builtins. NULL on
/// failure (sofuu_init's caller sees the same failure mode as C).
#[no_mangle]
pub unsafe extern "C" fn sofuu_init() -> *mut SofuuRuntime {
    let eng = unsafe { engine_create() };
    if eng.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: engine_create returned a live engine; builtins registered
    // before the runtime escapes (C order: register after create).
    unsafe { engine_register_builtins(eng) };
    let rt = Box::new(SofuuRuntime { engine: eng as *mut c_void });
    Box::into_raw(rt) as *mut SofuuRuntime
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_eval_file(rt: *mut SofuuRuntime, path: *const c_char) -> c_int {
    if rt.is_null() || path.is_null() {
        return 1;
    }
    // SAFETY: rt is a live runtime from sofuu_init; path a C string.
    unsafe { engine_eval_file((*rt).engine as *mut SofuuEngine, path) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_eval_string(
    rt: *mut SofuuRuntime,
    source: *const c_char,
    filename: *const c_char,
) -> c_int {
    if rt.is_null() || source.is_null() {
        return 1;
    }
    // SAFETY: rt is live; source/filename are NUL-terminated C strings.
    unsafe { engine_eval_string((*rt).engine as *mut SofuuEngine, source, filename) }
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_run_jobs(rt: *mut SofuuRuntime) {
    if rt.is_null() {
        return;
    }
    // SAFETY: rt is live.
    unsafe { engine_run_jobs((*rt).engine as *mut SofuuEngine) };
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_destroy(rt: *mut SofuuRuntime) {
    if rt.is_null() {
        return;
    }
    // SAFETY: rt is live; the engine is torn down in the C order
    // (rejections → loop close → process cleanup → GC → contexts).
    unsafe { engine_destroy((*rt).engine as *mut SofuuEngine) };
    drop(unsafe { Box::from_raw(rt) });
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_get_engine(rt: *mut SofuuRuntime) -> *mut c_void {
    if rt.is_null() {
        return ptr::null_mut();
    }
    (*rt).engine
}

/// ffi_shim.sofuu_engine_ctx — read JSContext from a SofuuEngine* (no
/// layout guessing anywhere else).
#[no_mangle]
pub unsafe extern "C" fn sofuu_engine_ctx(eng: *mut c_void) -> *mut c_void {
    if eng.is_null() {
        return ptr::null_mut();
    }
    let e = eng as *mut SofuuEngine;
    // SAFETY: eng is a live SofuuEngine*.
    unsafe { (*e).ctx as *mut c_void }
}

// ── engine.h ABI ──────────────────────────────────────────────────

/// REPL eval: returns a malloc'd pretty-printed string (free with
/// libc::free); `is_error` is set to 1 on exception. The C contract.
#[no_mangle]
pub unsafe extern "C" fn engine_eval_repl(
    eng: *mut c_void,
    source: *const c_char,
    is_error: *mut c_int,
) -> *mut c_char {
    if eng.is_null() || source.is_null() {
        return ptr::null_mut();
    }
    let e = eng as *mut SofuuEngine;
    // SAFETY: eng is a live SofuuEngine*; source a C string; is_error a
    // valid out-ptr (caller's, may be NULL out of caution).
    unsafe { engine_eval_repl_inner(e, source, is_error) }
}

// ── Engine construction / teardown ───────────────────────────────

unsafe fn engine_create() -> *mut SofuuEngine {
    let eng = Box::new(SofuuEngine { rt: ptr::null_mut(), ctx: ptr::null_mut() });
    // SAFETY: JS_NewRuntime/JS_NewContext are the canonical init path.
    let rt = unsafe { qjs::JS_NewRuntime() };
    if rt.is_null() {
        return ptr::null_mut();
    }
    // Generous memory limit (512MB) + GC threshold (64MB) — engine.c.
    unsafe { qjs::JS_SetMemoryLimit(rt, 512 * 1024 * 1024) };
    unsafe { qjs::JS_SetGCThreshold(rt, 64 * 1024 * 1024) };
    let ctx = unsafe { qjs::JS_NewContext(rt) };
    if ctx.is_null() {
        // SAFETY: rt is orphaned and must be released.
        unsafe { qjs::JS_FreeRuntime(rt) };
        return ptr::null_mut();
    }
    // Standard helpers (eval, parseInt, …).
    unsafe { qjs::js_init_module_std(ctx, c"std".as_ptr()) };
    unsafe { qjs::js_init_module_os(ctx, c"os".as_ptr()) };
    // ESM module loader.
    unsafe { qjs::JS_SetModuleLoaderFunc(rt, None, Some(sofuu_module_loader), ptr::null_mut()) };
    // Unhandled-rejection tracker (M1 — Rust port; C's tracker is gone).
    unsafe { sofuu_rt_install_rejection_tracker(rt) };
    // libuv event loop.
    unsafe { sofuu_loop_init() };
    let eng = Box::into_raw(eng);
    // SAFETY: eng owns exactly these rt/ctx (no leaks on any path above).
    unsafe {
        (*eng).rt = rt;
        (*eng).ctx = ctx;
    }
    eng
}

/// Register every built-in native module, then build the `Sofuu` global.
/// The order is the C registration order — property order is JS-visible.
unsafe fn engine_register_builtins(eng: *mut SofuuEngine) {
    // SAFETY: eng is live.
    let ctx = unsafe { (*eng).ctx };
    unsafe {
        mod_console_register(ctx);
        mod_process_register(ctx);
        mod_timer_register(ctx);
        mod_fs_register(ctx);
        mod_subprocess_register(ctx);
        mod_http_client_register(ctx);
        mod_http_server_register(ctx);
        mod_http_sse_register(ctx);
        mod_ai_register(ctx);
        mod_mcp_register(ctx);
        mod_cjs_register(ctx);
    }
    /* M2 (PLAN-MEMORY-TOKENS): a global GC bridge so JS drivers can bound
     * garbage after heavy fan-outs (agent.js runMany/mapContext) — the
     * chat driver has its own __chat_gc alias; this one serves every host
     * (sofuu run, headless, chat). */
    unsafe extern "C" fn js_sofuu_gc(
        ctx: *mut qjs::JSContext,
        _this: qjs::JSValueConst,
        _argc: c_int,
        _argv: *const qjs::JSValueConst,
    ) -> qjs::JSValue {
        let rt = qjs::JS_GetRuntime(ctx);
        if !rt.is_null() {
            qjs::JS_RunGC(rt);
        }
        qjs::sofuu_js_undefined()
    }
    unsafe { sofuu_ffi::bridge::register_global_fn(ctx, "__sofuu_gc", js_sofuu_gc) };
    #[cfg(has_qtsq)]
    unsafe {
        // SOFUU_MEMORY && SOFUU_QTSQ_PRESENT — the memory/kv/agent shells
        // register only when the QTSQ codec is linked (same gate as C).
        mod_memory_register(ctx);
        mod_kv_register(ctx);
        mod_agent_register(ctx);
    }

    // ── Build capital-S 'Sofuu' convenience global ────────────────
    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    let sofuu_obj = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr()) };

    // COPY_FN(ctx, target, src_obj, key): copy a property onto the Sofuu
    // object (ownership moves; macro hygiene — every live value is passed
    // in explicitly).
    macro_rules! copy_fn {
        ($c:expr, $t:expr, $src:expr, $key:literal) => {{
            let key_c = CString::new($key).unwrap_or_default();
            let v = unsafe { qjs::sofuu_js_get_property_str($c, $src, key_c.as_ptr()) };
            if unsafe { qjs::JS_IsFunction($c, v) } != 0 {
                unsafe { qjs::sofuu_js_set_property_str($c, $t, key_c.as_ptr(), v) };
            } else {
                unsafe { qjs::sofuu_js_free_value($c, v) };
            }
        }};
    }

    let sofuu = unsafe { qjs::sofuu_js_new_object(ctx) };

    // sofuu.fs.* → Sofuu.*
    let fs_obj = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"fs".as_ptr()) };
    copy_fn!(ctx, sofuu, fs_obj, "readFile");
    copy_fn!(ctx, sofuu, fs_obj, "writeFile");
    copy_fn!(ctx, sofuu, fs_obj, "appendFile");
    copy_fn!(ctx, sofuu, fs_obj, "exists");
    copy_fn!(ctx, sofuu, fs_obj, "readdir");
    unsafe { qjs::sofuu_js_free_value(ctx, fs_obj) };

    copy_fn!(ctx, sofuu, sofuu_obj, "sleep");
    copy_fn!(ctx, sofuu, sofuu_obj, "exec");

    // Sofuu.createMCPServer → sofuu.mcp.serve (the real implementation)
    let mcp_obj = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"mcp".as_ptr()) };
    let srv = unsafe { qjs::sofuu_js_get_property_str(ctx, mcp_obj, c"serve".as_ptr()) };
    if unsafe { qjs::JS_IsFunction(ctx, srv) } != 0 {
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                sofuu,
                c"createMCPServer".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, srv),
            )
        };
    }
    unsafe { qjs::sofuu_js_free_value(ctx, srv) };
    unsafe { qjs::sofuu_js_free_value(ctx, mcp_obj) };

    // sofuu.http namespace (sofuu.http.createServer / fetch)
    let http_obj = unsafe { qjs::sofuu_js_new_object(ctx) };
    let cs = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"createServer".as_ptr()) };
    if unsafe { qjs::JS_IsFunction(ctx, cs) } != 0 {
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                http_obj,
                c"createServer".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, cs),
            )
        };
    }
    unsafe { qjs::sofuu_js_free_value(ctx, cs) };
    let ft = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"fetch".as_ptr()) };
    if unsafe { qjs::JS_IsFunction(ctx, ft) } != 0 {
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                http_obj,
                c"fetch".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, ft),
            )
        };
    }
    unsafe { qjs::sofuu_js_free_value(ctx, ft) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"http".as_ptr(), http_obj) };

    // global.sleep alias
    let sl = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"sleep".as_ptr()) };
    if unsafe { qjs::JS_IsFunction(ctx, sl) } != 0 {
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                global,
                c"sleep".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, sl),
            )
        };
    }
    unsafe { qjs::sofuu_js_free_value(ctx, sl) };

    let ai = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"ai".as_ptr()) };
    if !unsafe { qjs::is_undefined(ai) } && !unsafe { qjs::is_null(ai) } {
        unsafe { qjs::sofuu_js_set_property_str(ctx, sofuu, c"ai".as_ptr(), ai) };
    } else {
        unsafe { qjs::sofuu_js_free_value(ctx, ai) };
    }

    let mem = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"memory".as_ptr()) };
    if !unsafe { qjs::is_undefined(mem) } && !unsafe { qjs::is_null(mem) } {
        unsafe { qjs::sofuu_js_set_property_str(ctx, sofuu, c"memory".as_ptr(), mem) };
    } else {
        unsafe { qjs::sofuu_js_free_value(ctx, mem) };
    }

    let kv = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"kv".as_ptr()) };
    if !unsafe { qjs::is_undefined(kv) } && !unsafe { qjs::is_null(kv) } {
        unsafe { qjs::sofuu_js_set_property_str(ctx, sofuu, c"kv".as_ptr(), kv) };
    } else {
        unsafe { qjs::sofuu_js_free_value(ctx, kv) };
    }

    let agent_mod = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"agent".as_ptr()) };
    if !unsafe { qjs::is_undefined(agent_mod) } && !unsafe { qjs::is_null(agent_mod) } {
        unsafe { qjs::sofuu_js_set_property_str(ctx, sofuu, c"agent".as_ptr(), agent_mod) };
    } else {
        unsafe { qjs::sofuu_js_free_value(ctx, agent_mod) };
    }

    unsafe { qjs::sofuu_js_free_value(ctx, sofuu_obj) };
    copy_fn!(ctx, sofuu, global, "createServer");
    unsafe { qjs::sofuu_js_set_property_str(ctx, global, c"Sofuu".as_ptr(), sofuu) };
    unsafe { qjs::sofuu_js_free_value(ctx, global) };

    // Rust-side JS surface: sofuu.rlm host functions + shipped driver.
    unsafe { sofuu_rust_register_engine_js(ctx as *mut c_void) };
}

/// C `dirname()` — strips the final component; "." for a bare name,
/// "/" for the root. Rust's Path::parent differs on root/empty, so this
/// normalizes back to C semantics for the loader's join ladder.
fn c_dirname(p: &str) -> String {
    match Path::new(p).parent() {
        Some(par) => {
            let s = par.to_string_lossy();
            if s.is_empty() {
                if p.starts_with('/') {
                    "/".to_string()
                } else {
                    ".".to_string()
                }
            } else {
                s.into_owned()
            }
        }
        None => {
            if p.starts_with('/') {
                "/".to_string()
            } else {
                ".".to_string()
            }
        }
    }
}

/// engine_read_file — reads a whole file into a heap buffer with a
/// trailing NUL (JS_Eval reads past input_len), like the C helper.
fn engine_read_file(path: &str) -> Option<(Vec<u8>, usize)> {
    match std::fs::read(path) {
        Ok(mut buf) => {
            let n = buf.len();
            buf.push(0); // NUL sentinel — QuickJS reads past input_len
            Some((buf, n))
        }
        Err(_) => None,
    }
}

/// resolve_module_path — the loader's resolution ladder (engine.c):
/// relative/absolute → realpath (canonicalize) → +.js → /index.js →
/// best-guess; bare specifiers → npm walk-up from the importer's dir,
/// then cwd, then pass-through. Returns the path as a Rust String (the
/// C returned strdup'd buffers and callers freed them).
fn resolve_module_path(base_name: &str, module_name: &str) -> String {
    if module_name.starts_with('.') || module_name.starts_with('/') {
        let full = if module_name.starts_with('.') {
            format!("{}/{}", c_dirname(base_name), module_name)
        } else {
            module_name.to_string()
        };

        if let Ok(r) = std::fs::canonicalize(&full) {
            return r.to_string_lossy().into_owned();
        }
        let with_ext = format!("{}.js", full);
        if let Ok(r) = std::fs::canonicalize(&with_ext) {
            return r.to_string_lossy().into_owned();
        }
        let as_index = format!("{}/index.js", full);
        if let Ok(r) = std::fs::canonicalize(&as_index) {
            return r.to_string_lossy().into_owned();
        }
        // Could not resolve — return the best guess anyway so the error
        // message includes a useful filename.
        if module_name.contains(".js") || module_name.contains(".mjs") {
            full
        } else {
            with_ext
        }
    } else {
        // Non-relative, non-absolute: npm walk-up from the importer's dir.
        let dir = c_dirname(base_name);
        let dir_c = CString::new(dir.as_str()).unwrap_or_default();
        let name_c = CString::new(module_name).unwrap_or_default();
        // SAFETY: dir_c/name_c are NUL-terminated; npm_resolve returns a
        // malloc'd string or NULL; copy out then free.
        let p = unsafe { npm_resolve(dir_c.as_ptr(), name_c.as_ptr()) };
        if !p.is_null() {
            let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
            unsafe { libc::free(p as *mut c_void) };
            return s;
        }
        // Also try from cwd in case base is not set correctly.
        if let Ok(cwd) = std::env::current_dir() {
            let cwd_c = CString::new(cwd.to_string_lossy().as_bytes()).unwrap_or_default();
            let p = unsafe { npm_resolve(cwd_c.as_ptr(), name_c.as_ptr()) };
            if !p.is_null() {
                let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
                unsafe { libc::free(p as *mut c_void) };
                return s;
            }
        }
        // Fallback: pass through unchanged (produces a useful error).
        module_name.to_string()
    }
}

/// update_loader_base — repoint the loader's opaque base at `new_base`
/// (strdup semantics; the thread_local owns the previous buffer until the
/// next update, mirroring C's g_loader_base lifetime).
unsafe fn update_loader_base(rt: *mut qjs::JSRuntime, new_base: &str) {
    let c = CString::new(new_base).unwrap_or_default();
    let ptr = c.as_ptr() as *mut c_void;
    // SAFETY: rt live; sofuu_module_loader 'static; ptr stays alive (it is
    // moved into G_LOADER_BASE below — CString moves do not reallocate).
    unsafe { qjs::JS_SetModuleLoaderFunc(rt, None, Some(sofuu_module_loader), ptr) };
    G_LOADER_BASE.with(|g| {
        *g.borrow_mut() = Some(c);
    });
}

/// sofuu_module_loader — the ESM loader registered with QuickJS. Resolves
/// the specifier, reads + strips/wraps the source, compiles the module.
#[no_mangle]
pub unsafe extern "C" fn sofuu_module_loader(
    ctx: *mut qjs::JSContext,
    module_name: *const c_char,
    opaque: *mut c_void,
) -> *mut c_void {
    if module_name.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: module_name is a NUL-terminated C string.
    let name = unsafe { CStr::from_ptr(module_name) }.to_string_lossy().into_owned();
    // SAFETY: opaque is our g_loader_base CString or NULL.
    let base: String = if opaque.is_null() {
        "./".to_string()
    } else {
        unsafe { CStr::from_ptr(opaque as *const c_char) }
            .to_string_lossy()
            .into_owned()
    };

    let path = resolve_module_path(&base, &name);
    let Some((source, mut len)) = engine_read_file(&path) else {
        // JS_ThrowReferenceError(ctx, "Cannot find module '%s' (resolved: '%s')", ...)
        let msg = CString::new(format!(
            "Cannot find module '{}' (resolved: '{}')",
            name, path
        ))
        .unwrap_or_default();
        // SAFETY: msg is a NUL-terminated format string with matching args.
        unsafe { qjs::JS_ThrowReferenceError(ctx, c"%s".as_ptr(), msg.as_ptr()) };
        return ptr::null_mut();
    };

    let mut src_ptr: *const c_char = source.as_ptr() as *const c_char;
    let mut need_free: *mut c_char = ptr::null_mut();

    if path.contains(".ts") || path.contains(".mts") {
        // Rust stripper (sofuu_core::ts); on failure keep the original text.
        let stripped = unsafe { sofuu_ts_strip_rs(src_ptr, len, &mut len) };
        if !stripped.is_null() {
            need_free = stripped;
            src_ptr = stripped;
        }
    } else if unsafe { is_cjs(src_ptr) } != 0 {
        let wrapped = unsafe { cjs_to_esm(src_ptr, len, &mut len) };
        if !wrapped.is_null() {
            need_free = wrapped;
            src_ptr = wrapped;
        }
    }

    // Compile module — pass the resolved path as filename for stack traces.
    let path_c = CString::new(path.as_str()).unwrap_or_default();
    // SAFETY: src_ptr is NUL-terminated (read_file/strip/cjs_to_esm all
    // guarantee the sentinel), len is the payload length; path_c valid.
    let func = unsafe {
        qjs::JS_Eval(
            ctx,
            src_ptr,
            len,
            path_c.as_ptr(),
            qjs::JS_EVAL_TYPE_MODULE | qjs::JS_EVAL_FLAG_COMPILE_ONLY,
        )
    };
    if !need_free.is_null() {
        // SAFETY: need_free was malloc'd by the stripper/cjs wrapper.
        unsafe { libc::free(need_free as *mut c_void) };
    }

    if unsafe { qjs::is_exception(func) } {
        // SAFETY: ctx is live; the exception is in flight.
        unsafe { qjs::js_std_dump_error(ctx) };
        return ptr::null_mut();
    }

    let mod_def = unsafe { qjs::sofuu_js_value_get_ptr(func) };

    // Update the opaque base to THIS module's resolved path so transitive
    // imports resolve relative to its directory (update_loader_base owns
    // the storage; path is still owned by this frame's String).
    let rt = unsafe { qjs::JS_GetRuntime(ctx) };
    unsafe { update_loader_base(rt, path.as_str()) };

    // SAFETY: func is a live module value; the module def outlives it.
    unsafe { qjs::sofuu_js_free_value(ctx, func) };
    mod_def
}

/// dump_error — print an exception + stack to stderr (the eval-file path).
unsafe fn dump_error(ctx: *mut qjs::JSContext) {
    // SAFETY: ctx is live; every value below is freed exactly once.
    let exception = unsafe { qjs::JS_GetException(ctx) };
    let str_v = unsafe { qjs::JS_ToString(ctx, exception) };
    let msg = unsafe { qjs::sofuu_js_to_cstring(ctx, str_v) };
    let msg_s: std::borrow::Cow<str> = if msg.is_null() {
        "(unknown error)".into()
    } else {
        unsafe { CStr::from_ptr(msg) }.to_string_lossy()
    };
    eprintln!("\x1b[31mError:\x1b[0m {}", msg_s);
    let stack = unsafe { qjs::sofuu_js_get_property_str(ctx, exception, c"stack".as_ptr()) };
    if !unsafe { qjs::is_undefined(stack) } && !unsafe { qjs::is_null(stack) } {
        let s = unsafe { qjs::sofuu_js_to_cstring(ctx, stack) };
        if !s.is_null() {
            eprintln!("{}", unsafe { CStr::from_ptr(s) }.to_string_lossy());
            // SAFETY: s from to_cstring on this ctx.
            unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
        }
    }
    unsafe { qjs::sofuu_js_free_value(ctx, stack) };
    if !msg.is_null() {
        // SAFETY: msg from to_cstring on this ctx.
        unsafe { qjs::sofuu_js_free_cstring(ctx, msg) };
    }
    unsafe { qjs::sofuu_js_free_value(ctx, str_v) };
    unsafe { qjs::sofuu_js_free_value(ctx, exception) };
}

/// engine_eval_file — load + (TS-strip) + eval a file as a module, then
/// drain the event loop. Returns 0 on success (1 on any error).
unsafe fn engine_eval_file(eng: *mut SofuuEngine, path_c: *const c_char) -> c_int {
    // SAFETY: path_c is a NUL-terminated C string.
    let path = unsafe { CStr::from_ptr(path_c) }.to_string_lossy().into_owned();
    let Some((source, len)) = engine_read_file(&path) else {
        eprintln!("\x1b[31mError:\x1b[0m Cannot open file '{}'", path);
        return 1;
    };

    // TypeScript: strip types before evaluating (.ts suffix check).
    let mut eval_src: *const c_char = source.as_ptr() as *const c_char;
    let mut eval_len = len;
    let mut stripped: *mut c_char = ptr::null_mut();
    if path.ends_with(".ts") && path.len() >= 3 {
        stripped = unsafe { sofuu_ts_strip_rs(eval_src, eval_len, &mut eval_len) };
        if !stripped.is_null() {
            eval_src = stripped;
        }
    }

    // Update the module loader opaque: this file is the base for its
    // relative imports (update_loader_base owns the storage).
    let rt = unsafe { (*eng).rt };
    unsafe { update_loader_base(rt, path.as_str()) };

    // SAFETY: eval_src is NUL-terminated (read_file/strip guarantee it);
    // eval_len is the payload length; path_c is stable for the call.
    let result = unsafe {
        qjs::JS_Eval(
            (*eng).ctx,
            eval_src,
            eval_len,
            path_c,
            qjs::JS_EVAL_TYPE_MODULE,
        )
    };
    drop(source);
    if !stripped.is_null() {
        // SAFETY: stripped was malloc'd by the stripper.
        unsafe { libc::free(stripped as *mut c_void) };
    }

    if unsafe { qjs::is_exception(result) } {
        unsafe { dump_error((*eng).ctx) };
        // SAFETY: result is a live exception value.
        unsafe { qjs::sofuu_js_free_value((*eng).ctx, result) };
        return 1;
    }

    // SAFETY: result is a live value.
    unsafe { qjs::sofuu_js_free_value((*eng).ctx, result) };
    // Run the libuv event loop — drains timers, I/O, and Promise chains.
    unsafe { sofuu_loop_run((*eng).ctx) };
    0
}

/// engine_eval_string — global eval; exceptions go through
/// process_dispatch_uncaught (uncaughtException handler) first, then the
/// stderr dump + non-zero exit (the C order).
unsafe fn engine_eval_string(
    eng: *mut SofuuEngine,
    source: *const c_char,
    filename: *const c_char,
) -> c_int {
    let ctx = unsafe { (*eng).ctx };
    let name = if filename.is_null() { c"<eval>".as_ptr() } else { filename };
    // SAFETY: source is a NUL-terminated C string; strlen from CStr.
    let src_len = unsafe { CStr::from_ptr(source) }.to_bytes().len();
    let result = unsafe { qjs::JS_Eval(ctx, source, src_len, name, qjs::JS_EVAL_TYPE_GLOBAL) };

    if unsafe { qjs::is_exception(result) } {
        // SAFETY: exception value is live; uncaught dispatch takes it.
        let exc = unsafe { qjs::JS_GetException(ctx) };
        if unsafe { process_dispatch_uncaught(ctx, exc) } == 0 {
            let str_v = unsafe { qjs::JS_ToString(ctx, exc) };
            let msg = unsafe { qjs::sofuu_js_to_cstring(ctx, str_v) };
            if msg.is_null() {
                eprintln!("\x1b[31mError:\x1b[0m (unknown error)");
            } else {
                eprintln!(
                    "\x1b[31mError:\x1b[0m {}",
                    unsafe { CStr::from_ptr(msg) }.to_string_lossy()
                );
                // SAFETY: msg from to_cstring on this ctx.
                unsafe { qjs::sofuu_js_free_cstring(ctx, msg) };
            }
            let stack = unsafe { qjs::sofuu_js_get_property_str(ctx, exc, c"stack".as_ptr()) };
            if !unsafe { qjs::is_undefined(stack) } {
                let ss = unsafe { qjs::sofuu_js_to_cstring(ctx, stack) };
                if !ss.is_null() {
                    eprintln!("{}", unsafe { CStr::from_ptr(ss) }.to_string_lossy());
                    // SAFETY: ss from to_cstring on this ctx.
                    unsafe { qjs::sofuu_js_free_cstring(ctx, ss) };
                }
            }
            unsafe { qjs::sofuu_js_free_value(ctx, stack) };
            unsafe { qjs::sofuu_js_free_value(ctx, str_v) };
        }
        // SAFETY: exc/result are live values.
        unsafe { qjs::sofuu_js_free_value(ctx, exc) };
        unsafe { qjs::sofuu_js_free_value(ctx, result) };
        return 1;
    }

    // SAFETY: result is a live value.
    unsafe { qjs::sofuu_js_free_value(ctx, result) };
    unsafe { sofuu_loop_run(ctx) };
    0
}

/// engine_run_jobs — pump the microtask/job queue until empty.
unsafe fn engine_run_jobs(eng: *mut SofuuEngine) {
    loop {
        let mut ctx1: *mut qjs::JSContext = ptr::null_mut();
        let err = unsafe { qjs::JS_ExecutePendingJob((*eng).rt, &mut ctx1) };
        if err <= 0 {
            if err < 0 {
                unsafe { dump_error(ctx1) };
            }
            break;
        }
    }
}

/// engine_destroy — the C teardown order: surface pending rejections,
/// close the loop, process cleanup, GC, free context + runtime.
unsafe fn engine_destroy(eng: *mut SofuuEngine) {
    if eng.is_null() {
        return;
    }
    // SAFETY: eng is live for the whole teardown sequence.
    unsafe { sofuu_rt_report_pending_rejections((*eng).ctx) };
    unsafe { sofuu_loop_close() };
    unsafe { mod_process_cleanup((*eng).ctx) };
    unsafe { qjs::JS_RunGC((*eng).rt) };
    unsafe { qjs::JS_FreeContext((*eng).ctx) };
    unsafe { qjs::JS_FreeRuntime((*eng).rt) };
    drop(unsafe { Box::from_raw(eng) });
}

// ── REPL display (engine.c's repl_inspect + engine_eval_repl) ─────

/// `printf("%g")` — 6 significant digits, trailing zeros trimmed,
/// scientific when the rounded exponent is < -4 or >= 6 (C's rule).
/// The integer path in repl_inspect handles the common cases; this
/// preserves the C display for everything else.
fn g_format(d: f64) -> String {
    if d.is_nan() {
        return "nan".to_string();
    }
    if d.is_infinite() {
        return if d.is_sign_negative() { "-inf" } else { "inf" }.to_string();
    }
    if d == 0.0 {
        return if d.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let neg = d.is_sign_negative();
    let ad = d.abs();
    // 6 significant digits, rounded — "d.ddddde±n".
    let sci = format!("{:.5e}", ad);
    let epos = sci.find('e').unwrap();
    let exp: i32 = sci[epos + 1..].parse().unwrap();
    let out;
    if exp < -4 || exp >= 6 {
        // Scientific: trim trailing zeros from the mantissa.
        let mut mant = sci[..epos].to_string();
        if mant.contains('.') {
            while mant.ends_with('0') {
                mant.pop();
            }
            if mant.ends_with('.') {
                mant.pop();
            }
        }
        let sign = if exp < 0 { "-" } else { "+" };
        out = format!("{}e{}{:02}", mant, sign, exp.abs());
    } else {
        // Fixed notation, rounded to (5 - exp) decimals, trailing zeros
        // trimmed (%.6g semantics).
        let decimals = (5 - exp).max(0) as usize;
        let mut fs = format!("{:.*}", decimals, ad);
        if fs.contains('.') {
            while fs.ends_with('0') {
                fs.pop();
            }
            if fs.ends_with('.') {
                fs.pop();
            }
        }
        out = fs;
    }
    if neg {
        format!("-{}", out)
    } else {
        out
    }
}

/// repl_inspect — the REPL value renderer (engine.c: colors, %lld for
/// integral doubles in ±1e15, %g otherwise, JSON.stringify for objects).
unsafe fn repl_inspect(ctx: *mut qjs::JSContext, val: qjs::JSValueConst) -> String {
    if unsafe { qjs::is_undefined(val) } {
        return "\x1b[90mundefined\x1b[0m".to_string();
    }
    if unsafe { qjs::is_null(val) } {
        return "\x1b[1mnull\x1b[0m".to_string();
    }

    if (val.tag as i32) == qjs::JS_TAG_BOOL {
        return if unsafe { qjs::JS_ToBool(ctx, val) } != 0 {
            "\x1b[33mtrue\x1b[0m".to_string()
        } else {
            "\x1b[33mfalse\x1b[0m".to_string()
        };
    }

    if unsafe { qjs::is_number(val) } {
        let mut d: f64 = 0.0;
        // SAFETY: val is a number; JS_ToFloat64 writes d on success.
        unsafe { qjs::JS_ToFloat64(ctx, &mut d, val) };
        let ll = d as i64;
        if d == ll as f64 && d >= -1e15 && d <= 1e15 {
            // C: snprintf(buf, "%lld", (long long)d) — the integral path.
            return format!("\x1b[33m{}\x1b[0m", ll);
        }
        return format!("\x1b[33m{}\x1b[0m", g_format(d));
    }

    if unsafe { qjs::sofuu_js_is_string(val) } != 0 {
        // SAFETY: to_cstring returns a ptr we free.
        let s = unsafe { qjs::sofuu_js_to_cstring(ctx, val) };
        let body = if s.is_null() {
            String::new()
        } else {
            let out = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
            unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
            out
        };
        return format!("\x1b[32m'{}'\x1b[0m", body);
    }

    if unsafe { qjs::JS_IsFunction(ctx, val) } != 0 {
        let name = unsafe { qjs::sofuu_js_get_property_str(ctx, val, c"name".as_ptr()) };
        let n = unsafe { qjs::sofuu_js_to_cstring(ctx, name) };
        let name_s = if n.is_null() || unsafe { *n } == 0 {
            "(anonymous)".to_string()
        } else {
            unsafe { CStr::from_ptr(n) }.to_string_lossy().into_owned()
        };
        if !n.is_null() {
            // SAFETY: n from to_cstring on this ctx.
            unsafe { qjs::sofuu_js_free_cstring(ctx, n) };
        }
        unsafe { qjs::sofuu_js_free_value(ctx, name) };
        return format!("\x1b[36m[Function: {}]\x1b[0m", name_s);
    }

    // Array / Object — JSON.stringify.
    if unsafe { qjs::is_object(val) } {
        let undef = unsafe { qjs::sofuu_js_undefined() };
        let json = unsafe { qjs::JS_JSONStringify(ctx, val, undef, undef) };
        if !unsafe { qjs::is_exception(json) } {
            // SAFETY: to_cstring returns a ptr we free.
            let s = unsafe { qjs::sofuu_js_to_cstring(ctx, json) };
            let body = if s.is_null() {
                "{}".to_string()
            } else {
                let out = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
                unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
                out
            };
            unsafe { qjs::sofuu_js_free_value(ctx, json) };
            return format!("\x1b[0m{}\x1b[0m", body);
        }
        unsafe { qjs::sofuu_js_free_value(ctx, json) };
        return "\x1b[0m[Object]\x1b[0m".to_string();
    }

    let str_v = unsafe { qjs::JS_ToString(ctx, val) };
    let s = unsafe { qjs::sofuu_js_to_cstring(ctx, str_v) };
    let body = if s.is_null() {
        String::new()
    } else {
        let out = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
        unsafe { qjs::sofuu_js_free_cstring(ctx, s) };
        out
    };
    unsafe { qjs::sofuu_js_free_value(ctx, str_v) };
    body
}

/// engine_eval_repl_inner — evaluate one REPL line, drain microtasks,
/// and return the display string (malloc'd, C contract).
unsafe fn engine_eval_repl_inner(
    eng: *mut SofuuEngine,
    source: *const c_char,
    is_error: *mut c_int,
) -> *mut c_char {
    let ctx = unsafe { (*eng).ctx };
    let rt = unsafe { (*eng).rt };
    let src_len = unsafe { CStr::from_ptr(source) }.to_bytes().len();

    let result = unsafe {
        qjs::JS_Eval(ctx, source, src_len, c"<repl>".as_ptr(), qjs::JS_EVAL_TYPE_GLOBAL)
    };

    // Drain microtasks (engine.c: the for(;;) job loop).
    let mut ctx1: *mut qjs::JSContext = ptr::null_mut();
    loop {
        let e = unsafe { qjs::JS_ExecutePendingJob(rt, &mut ctx1) };
        if e <= 0 {
            break;
        }
    }

    if unsafe { qjs::is_exception(result) } {
        if !is_error.is_null() {
            // SAFETY: is_error is a valid out-ptr.
            unsafe { *is_error = 1 };
        }
        // SAFETY: exc/estr are live values.
        let exc = unsafe { qjs::JS_GetException(ctx) };
        let estr = unsafe { qjs::JS_ToString(ctx, exc) };
        let msg = unsafe { qjs::sofuu_js_to_cstring(ctx, estr) };
        let msg_s = if msg.is_null() {
            "Unknown error".to_string()
        } else {
            let out = unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned();
            unsafe { qjs::sofuu_js_free_cstring(ctx, msg) };
            out
        };
        let mut buf = format!("\x1b[31m{}\x1b[0m", msg_s);
        let stack = unsafe { qjs::sofuu_js_get_property_str(ctx, exc, c"stack".as_ptr()) };
        if !unsafe { qjs::is_undefined(stack) } {
            let st = unsafe { qjs::sofuu_js_to_cstring(ctx, stack) };
            if !st.is_null() {
                let st_s = unsafe { CStr::from_ptr(st) }.to_string_lossy();
                buf.push_str(&format!("\n\x1b[90m{}\x1b[0m", st_s));
                // SAFETY: st from to_cstring on this ctx.
                unsafe { qjs::sofuu_js_free_cstring(ctx, st) };
            }
        }
        unsafe { qjs::sofuu_js_free_value(ctx, stack) };
        unsafe { qjs::sofuu_js_free_value(ctx, estr) };
        unsafe { qjs::sofuu_js_free_value(ctx, exc) };
        unsafe { qjs::sofuu_js_free_value(ctx, result) };
        let out = buf.into_bytes();
        return c_bytes_dup(&out);
    }

    let display = unsafe { repl_inspect(ctx, result) };
    unsafe { qjs::sofuu_js_free_value(ctx, result) };
    let out = display.into_bytes();
    c_bytes_dup(&out)
}

/// libc::malloc + copy + NUL — the C "strdup" contract for strings that
/// cross back into sofuu-ffi (which frees them with libc::free).
fn c_bytes_dup(s: &[u8]) -> *mut c_char {
    // SAFETY: malloc'd block is exactly the right size + NUL.
    let p = unsafe { libc::malloc(s.len() + 1) } as *mut u8;
    if p.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        ptr::copy_nonoverlapping(s.as_ptr(), p, s.len());
        *p.add(s.len()) = 0;
    }
    p as *mut c_char
}
