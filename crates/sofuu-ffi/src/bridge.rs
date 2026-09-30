// sofuu-ffi — QuickJS bridge bindings.
//
// These let Rust register native callback functions into the QuickJS global
// object (so a JS driver can call back into Rust), and read/write simple
// values. All `unsafe` is confined here; the safe wrappers below guarantee
// the C strings and values are handled correctly.
//
// M10: src/ffi_shim.c is DELETED — the sofuu_js_* helpers are now real Rust
// functions in crate::qjs (tag checks + value construction + refcount ops,
// all reimplemented from the public quickjs.h inline bodies). This module is
// now a thin convenience layer over qjs: the same helpers, same names.

pub use crate::qjs::{
    JSCFunction, JSContext, JSRuntime, JSValue, JSValueConst, JSValueUnion,
    sofuu_js_free_cstring, sofuu_js_to_cstring,
};

use crate::qjs;
use std::ffi::{CStr, CString, c_char};

// ── Helpers ─────────────────────────────────────────────────────
// These take raw QuickJS context pointers, so they are `unsafe fn` by
// design. They are only called from the FFI crate / chat bridge, where the
// context is guaranteed valid.

/// Register a Rust `JSCFunction` into the global object under `name`.
/// The function must be `'static` (no captured references).
///
/// # Safety
/// `ctx` must be a valid QuickJS context; `func` must be `'static`.
pub unsafe fn register_global_fn(ctx: *mut JSContext, name: &str, func: JSCFunction) {
    // P3 (AUDIT-2026-09-07): a NUL in `name` used to unwrap_or_default into
    // an EMPTY registered name. The C API cannot express it — escape so the
    // failure is visible, never silent.
    let cname = CString::new(name.replace('\0', "\\0")).unwrap_or_default();
    // SAFETY: ctx is a valid QuickJS context; cname valid for the call.
    let global = unsafe { qjs::sofuu_js_get_global_object(ctx) };
    let v = unsafe { qjs::sofuu_js_new_cfunction(ctx, func, cname.as_ptr(), 1) };
    let rc = unsafe { qjs::sofuu_js_set_property_str(ctx, global, cname.as_ptr(), v) };
    if rc < 0 {
        eprintln!("[sofuu-ffi] register {} failed (rc={})", name, rc);
    }
    // NOTE: JS_SetPropertyStr does NOT consume the value's reference —
    // the property now owns it. Do NOT free v (double-free corrupts it).
    unsafe { qjs::sofuu_js_free_value(ctx, global) };
}

/// Convert a JS string value to a Rust String. Returns None for non-strings.
///
/// # Safety
/// `ctx` must be a valid QuickJS context; `val` a value from that context.
pub unsafe fn js_to_string(ctx: *mut JSContext, val: JSValueConst) -> Option<String> {
    // SAFETY: val is a JSValueConst from the engine.
    if unsafe { qjs::sofuu_js_is_string(val) } == 0 {
        return None;
    }
    // SAFETY: sofuu_js_to_cstring returns a pointer we must free.
    let p = unsafe { qjs::sofuu_js_to_cstring(ctx, val) };
    if p.is_null() {
        return None;
    }
    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
    // SAFETY: p came from sofuu_js_to_cstring — free with sofuu_js_free_cstring.
    unsafe { qjs::sofuu_js_free_cstring(ctx, p) };
    Some(s)
}

/// Create a JS string value.
///
/// # Safety
/// `ctx` must be a valid QuickJS context.
pub unsafe fn js_new_string(ctx: *mut JSContext, s: &str) -> JSValue {
    // P3 (AUDIT-2026-09-07): NULs used to be silently dropped (CString +
    // unwrap_or_default → empty string). JS_NewStringLen takes the raw
    // bytes, so the value matches the Rust string exactly.
    unsafe { qjs::JS_NewStringLen(ctx, s.as_ptr() as *const c_char, s.len()) }
}

/// Create a JS boolean value.
///
/// # Safety
/// `ctx` must be a valid QuickJS context.
pub unsafe fn js_new_bool(ctx: *mut JSContext, b: bool) -> JSValue {
    // SAFETY: trivial.
    unsafe { qjs::sofuu_js_new_bool(ctx, if b { 1 } else { 0 }) }
}

/// Free a JS value (caller must not use it afterward).
///
/// # Safety
/// `ctx` must be a valid QuickJS context; `v` a value owned by it.
pub unsafe fn js_free_value(ctx: *mut JSContext, v: JSValue) {
    // SAFETY: v is a value owned by this context.
    unsafe { qjs::sofuu_js_free_value(ctx, v) };
}
