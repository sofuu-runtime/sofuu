// rt/cjs.rs — CommonJS compatibility shim (PLAN-RUST-MIGRATION M7).
//
// Port of the deleted `src/npm/cjs.c`, semantics verbatim:
//   is_cjs / cjs_to_esm — the engine's module loader calls these directly.
//   js_require — the global synchronous require() (wraps the resolved file
//                in a (exports, require, module, __filename, __dirname)
//                function and returns module.exports).
//   mod_cjs_register — registers the global `require`.
//
// C symbols replaced: `is_cjs`, `cjs_to_esm`, `mod_cjs_register` (engine.c
// calls them unchanged); `js_require` is internal but keeps its C name for
// parity of the registered global.

use std::ffi::{CStr, CString, c_char, c_int};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

/// Simple heuristic: CJS keywords but NO ES module exports/imports.
///
/// # Safety
/// `source` must be a NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn is_cjs(source: *const c_char) -> c_int {
    if source.is_null() {
        return 0;
    }
    let s = CStr::from_ptr(source).to_string_lossy();
    let has_cjs = s.contains("module.exports")
        || s.contains("exports.")
        || s.contains("require(");
    if !has_cjs {
        return 0;
    }
    let has_esm = s.contains("export default ")
        || s.contains("export {")
        || s.contains("import {")
        || s.contains("import \"")
        || s.contains("import '");
    if has_esm {
        0
    } else {
        1
    }
}

/// Wrap CJS source into an ESM that exports default module.exports.
/// Returns malloc'd NUL-terminated buffer (C `free()` contract), or NULL.
///
/// # Safety
/// C-string/memory contract: source valid for len bytes; out_len written.
#[no_mangle]
pub unsafe extern "C" fn cjs_to_esm(
    source: *const c_char,
    len: usize,
    out_len: *mut usize,
) -> *mut c_char {
    if source.is_null() {
        return std::ptr::null_mut();
    }
    let header = b"var module = { exports: {} }; var exports = module.exports;\n";
    let footer = b"\nexport default module.exports;\n";
    let total = header.len() + len + footer.len();
    let buf = libc::malloc(total + 1) as *mut c_char;
    if buf.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::copy_nonoverlapping(header.as_ptr(), buf as *mut u8, header.len());
    std::ptr::copy_nonoverlapping(
        source as *const u8,
        buf.add(header.len()) as *mut u8,
        len,
    );
    std::ptr::copy_nonoverlapping(
        footer.as_ptr(),
        buf.add(header.len() + len) as *mut u8,
        footer.len(),
    );
    *buf.add(total) = 0;
    if !out_len.is_null() {
        *out_len = total;
    }
    buf
}

unsafe fn read_file(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

unsafe extern "C" fn js_require(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }
    let spec_ptr = qjs::sofuu_js_to_cstring(ctx, *argv);
    if spec_ptr.is_null() {
        return qjs::sofuu_js_undefined();
    }
    let spec = CStr::from_ptr(spec_ptr).to_string_lossy().into_owned();

    let Ok(cwd) = std::env::current_dir() else {
        qjs::sofuu_js_free_cstring(ctx, spec_ptr);
        return qjs::JS_ThrowReferenceError(ctx, c"Cannot determine cwd".as_ptr());
    };

    let resolved = crate::npm::npm_resolve(&cwd, &spec).map(|p| p.to_string_lossy().into_owned());

    let Some(resolved) = resolved else {
        let msg = format!("Cannot find module '{}'", spec);
        qjs::sofuu_js_free_cstring(ctx, spec_ptr);
        let mc = CString::new(msg).unwrap_or_default();
        return qjs::JS_ThrowReferenceError(ctx, mc.as_ptr());
    };
    qjs::sofuu_js_free_cstring(ctx, spec_ptr);

    let Some(source) = read_file(&resolved) else {
        return qjs::JS_ThrowReferenceError(
            ctx,
            c"Failed to read module".as_ptr(),
        );
    };

    let wrap_head = "(function(exports, require, module, __filename, __dirname) { ";
    let wrap_tail = "\n})";
    let mut wrapped = Vec::with_capacity(wrap_head.len() + source.len() + wrap_tail.len() + 1);
    wrapped.extend_from_slice(wrap_head.as_bytes());
    wrapped.extend_from_slice(&source);
    wrapped.extend_from_slice(wrap_tail.as_bytes());
    wrapped.push(0);

    let resolved_c = CString::new(resolved.clone()).unwrap_or_default();
    let func = qjs::JS_Eval(
        ctx,
        wrapped.as_ptr() as *const c_char,
        wrapped.len() - 1,
        resolved_c.as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );

    if qjs::is_exception(func) {
        return func;
    }

    let module_obj = qjs::sofuu_js_new_object(ctx);
    let exports_obj = qjs::sofuu_js_new_object(ctx);
    qjs::sofuu_js_set_property_str(ctx, module_obj, c"exports".as_ptr(), qjs::sofuu_js_dup_value(ctx, exports_obj));

    let require_func = qjs::sofuu_js_new_cfunction(ctx, js_require, c"require".as_ptr(), 1);
    let filename_val = qjs::sofuu_js_new_string(ctx, resolved_c.as_ptr());
    let dirname_val = qjs::sofuu_js_new_string(ctx, resolved_c.as_ptr()); // simplistic (parity)

    let mut call_args: [JSValueConst; 5] = [
        exports_obj,
        require_func,
        module_obj,
        filename_val,
        dirname_val,
    ];
    let ret = qjs::JS_Call(ctx, func, qjs::sofuu_js_undefined(), 5, call_args.as_mut_ptr());

    qjs::sofuu_js_free_value(ctx, func);
    qjs::sofuu_js_free_value(ctx, exports_obj);
    qjs::sofuu_js_free_value(ctx, require_func);
    qjs::sofuu_js_free_value(ctx, filename_val);
    qjs::sofuu_js_free_value(ctx, dirname_val);

    if qjs::is_exception(ret) {
        qjs::sofuu_js_free_value(ctx, module_obj);
        return ret;
    }
    qjs::sofuu_js_free_value(ctx, ret);

    let final_exports = qjs::sofuu_js_get_property_str(ctx, module_obj, c"exports".as_ptr());
    qjs::sofuu_js_free_value(ctx, module_obj);
    final_exports
}

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_cjs_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    qjs::sofuu_js_set_property_str(
        ctx,
        global,
        c"require".as_ptr(),
        qjs::sofuu_js_new_cfunction(ctx, js_require, c"require".as_ptr(), 1),
    );
    qjs::sofuu_js_free_value(ctx, global);
}
