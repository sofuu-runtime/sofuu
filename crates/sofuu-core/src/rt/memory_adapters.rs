//! Lazy loader for the pluggable memory adapter source.
//!
//! PLAN-PLUGGABLE-MEMORY-ADAPTERS puts three REST backends and the custom
//! adapter contract in `src/js/memory_adapters.js`. That file is ~30 KB of
//! source; evaluating it at boot cost 1.5 ms on `sofuu eval "1+1"` — every
//! script paid it whether or not it ever opened a memory backend. So the
//! source is embedded (a standalone binary must carry it) but compiled only
//! when `sofuu.memory.open()` first asks for a non-native backend:
//!
//! ```js
//! sofuu.memory.__loadAdapters();   // installs the router's adapter surface
//! ```
//!
//! The JS side (`memory_api.js`) calls this at most once per engine and keeps
//! the resulting surface in its own closure, so a second engine in the same
//! process gets its own install and this loader stays stateless.

use std::ffi::{CStr, CString};

use sofuu_ffi::qjs;

const ADAPTERS_JS: &str = include_str!("../../../../src/js/memory_adapters.js");

/// `sofuu.memory.__loadAdapters() → boolean`.
///
/// Evaluates the adapter source in this context. Returns true when it ran;
/// a compile/runtime error is reported on stderr and answers false, which
/// the caller turns into a typed `E_ADAPTER` refusal (a broken adapter file
/// must never take the engine down at open time).
unsafe extern "C" fn js_load_adapters(
    ctx: *mut qjs::JSContext,
    _this: qjs::JSValueConst,
    _argc: std::ffi::c_int,
    _argv: *const qjs::JSValueConst,
) -> qjs::JSValue {
    let src = ADAPTERS_JS;
    let Ok(c_src) = CString::new(src) else {
        return unsafe { qjs::sofuu_js_new_bool(ctx, 0) };
    };
    let Ok(c_name) = CString::new("<memory-adapters>") else {
        return unsafe { qjs::sofuu_js_new_bool(ctx, 0) };
    };
    // SAFETY: ctx is the engine's live context; the buffers outlive the call.
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
                let msg = CStr::from_ptr(p).to_string_lossy().into_owned();
                qjs::sofuu_js_free_cstring(ctx, p);
                eprintln!("[sofuu] memory_adapters eval failed: {msg}");
            }
            qjs::sofuu_js_free_value(ctx, exc);
            return qjs::sofuu_js_new_bool(ctx, 0);
        }
        qjs::sofuu_js_free_value(ctx, r);
        qjs::sofuu_js_new_bool(ctx, 1)
    }
}

/// Install `sofuu.memory.__loadAdapters`. Called from `mod_memory_register`
/// (QTSQ builds), right after the flat API source is evaluated.
///
/// # Safety
/// `ctx` must be a live QuickJS context on the engine thread.
pub unsafe fn mod_memory_adapters_register(ctx: *mut qjs::JSContext) {
    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    let mut sofuu_obj = unsafe { qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr()) };
    if unsafe { qjs::is_undefined(sofuu_obj) } {
        sofuu_obj = unsafe { qjs::sofuu_js_new_object(ctx) };
        unsafe {
            qjs::sofuu_js_set_property_str(
                ctx,
                global,
                c"sofuu".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, sofuu_obj),
            )
        };
    }
    let mem = unsafe { qjs::sofuu_js_get_property_str(ctx, sofuu_obj, c"memory".as_ptr()) };
    if !unsafe { qjs::is_undefined(mem) } {
        let f = unsafe {
            qjs::sofuu_js_new_cfunction(ctx, js_load_adapters, c"__loadAdapters".as_ptr(), 0)
        };
        unsafe { qjs::sofuu_js_set_property_str(ctx, mem, c"__loadAdapters".as_ptr(), f) };
    }
    unsafe {
        qjs::sofuu_js_free_value(ctx, mem);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);
        qjs::sofuu_js_free_value(ctx, global);
    }
}
