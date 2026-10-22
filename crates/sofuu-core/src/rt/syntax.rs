// rt/syntax.rs - Parse-only syntax gate for agent file edits.
//
// PLAN-AUTONOMOUS-AGENT-LOOP-AND-DRIVERS Phase 5. The edit tool must be
// able to validate a replacement candidate BEFORE it reaches disk:
//
//   sofuu.parse(code)                              -> classic script
//   sofuu.parse(code, { filename, module, ts })    -> same, with knobs
//
//   { ok: true,  mode: 'script' | 'module' }
//   { ok: false, mode, type: 'SyntaxError',
//     error: 'SyntaxError: unexpected token ...', line: 42 }
//
// This is QuickJS's own parser via JS_EVAL_FLAG_COMPILE_ONLY, so the
// verdict is exactly the one the runtime would reach if it later loaded
// the file - there is no second, drifting notion of "valid".
//
// Why not the obvious alternatives:
//   * eval / sofuu.sandbox.run - both EXECUTE the candidate. A gate that
//     runs the file it is validating can create files, open sockets or
//     hang on an infinite loop before it ever says no.
//   * new Function(src) - rejects legal ESM (import/export, top-level
//     await) and accepts top-level "return", i.e. wrong in both
//     directions.
//
// Modes. "module: true" compiles as an ES module; module compilation
// RESOLVES imports, so a module whose imports cannot be found fails here
// with a non-SyntaxError type. Callers MUST read "type" and treat that
// as "unverified", never as "broken" - a missing dependency is not a
// syntax error. The default classic-script mode resolves nothing and does
// no file IO. Module compilation uses a synthetic module name
// ("<sofuu-parse:...>") so a probe can never register itself in
// ctx->loaded_modules under a real specifier and shadow the file's own
// later import.
//
// "ts: true" strips TypeScript type syntax first (crate::ts::strip keeps
// byte offsets and newlines, so reported line numbers stay accurate).
//
// Failure is reported as a value, never thrown: the call site is a tool
// that has to decide between "reject the edit" and "write anyway", and a
// throwing parser would make the no-verdict case indistinguishable from
// the broken-code case.

use std::ffi::{c_char, c_int, CString};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

/// One compile-only attempt's outcome.
struct Attempt {
    ok: bool,
    /// Error class name (SyntaxError, ReferenceError, ...), empty on ok.
    name: String,
    /// "SyntaxError: unexpected token ..." (the value's own toString), empty on ok.
    message: String,
    /// 1-based line of the error, 0 when the parser did not report one.
    line: i32,
}

/// Read a JS value as a Rust string (empty when it is not convertible).
unsafe fn js_string(ctx: *mut JSContext, v: JSValueConst) -> String {
    let p = unsafe { qjs::sofuu_js_to_cstring(ctx, v) };
    if p.is_null() {
        return String::new();
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned();
    unsafe { qjs::sofuu_js_free_cstring(ctx, p) };
    s
}

/// Read a boolean-ish property; missing/undefined reads as false.
unsafe fn opt_bool(ctx: *mut JSContext, obj: JSValueConst, key: *const c_char) -> bool {
    let v = unsafe { qjs::sofuu_js_get_property_str(ctx, obj, key) };
    let out = unsafe { qjs::JS_ToBool(ctx, v) } != 0;
    unsafe { qjs::sofuu_js_free_value(ctx, v) };
    out
}

/// Read a string property; None when absent or not a string.
unsafe fn opt_string(ctx: *mut JSContext, obj: JSValueConst, key: *const c_char) -> Option<String> {
    let v = unsafe { qjs::sofuu_js_get_property_str(ctx, obj, key) };
    let out = if unsafe { qjs::sofuu_js_is_string(v) } != 0 {
        Some(unsafe { js_string(ctx, v) })
    } else {
        None
    };
    unsafe { qjs::sofuu_js_free_value(ctx, v) };
    out
}

/// Set a string property, skipping values a C string cannot carry.
unsafe fn set_str(ctx: *mut JSContext, obj: JSValueConst, key: *const c_char, val: &str) {
    if let Ok(c_val) = CString::new(val) {
        let v = unsafe { qjs::sofuu_js_new_string(ctx, c_val.as_ptr()) };
        unsafe { qjs::sofuu_js_set_property_str(ctx, obj, key, v) };
    }
}

/// Compile "src" without executing it. Never runs a byte of user code.
unsafe fn compile_only(ctx: *mut JSContext, src: &str, filename: &str, module: bool) -> Attempt {
    let mut out = Attempt {
        ok: false,
        name: String::new(),
        message: String::new(),
        line: 0,
    };

    // The parser reports fileName/lineNumber only when it has a name, and
    // module probes are registered under it - hence the synthetic prefix.
    let label = if module {
        format!("<sofuu-parse:{}>", filename)
    } else {
        filename.to_string()
    };

    let (Ok(c_src), Ok(c_name)) = (CString::new(src), CString::new(label)) else {
        out.name = "Error".to_string();
        out.message = "Error: source or filename contains a NUL byte".to_string();
        return out;
    };

    let flags = if module {
        qjs::JS_EVAL_TYPE_MODULE
    } else {
        qjs::JS_EVAL_TYPE_GLOBAL
    } | qjs::JS_EVAL_FLAG_COMPILE_ONLY;

    // SAFETY: ctx is live on this thread; both CStrings outlive the call.
    let v = unsafe { qjs::JS_Eval(ctx, c_src.as_ptr(), src.len(), c_name.as_ptr(), flags) };
    if unsafe { qjs::sofuu_js_is_exception(v) } == 0 {
        // A compiled function (or module) object - drop it, we only wanted
        // the parser's verdict.
        unsafe { qjs::sofuu_js_free_value(ctx, v) };
        out.ok = true;
        return out;
    }
    unsafe { qjs::sofuu_js_free_value(ctx, v) };

    let exc = unsafe { qjs::JS_GetException(ctx) };
    out.message = unsafe { js_string(ctx, exc) };
    if let Some(n) = unsafe { opt_string(ctx, exc, c"name".as_ptr()) } {
        out.name = n;
    }
    let lv = unsafe { qjs::sofuu_js_get_property_str(ctx, exc, c"lineNumber".as_ptr()) };
    if unsafe { qjs::sofuu_js_is_number(lv) } != 0 {
        let mut line: i32 = 0;
        if unsafe { qjs::JS_ToInt32(ctx, &mut line, lv) } == 0 && line > 0 {
            out.line = line;
        }
    }
    unsafe { qjs::sofuu_js_free_value(ctx, lv) };
    unsafe { qjs::sofuu_js_free_value(ctx, exc) };

    if out.message.is_empty() {
        out.message = if out.name.is_empty() {
            "Error: parse failed".to_string()
        } else {
            format!("{}: parse failed", out.name)
        };
    }
    out
}

/// sofuu.parse(code[, opts]) - see the module header.
unsafe extern "C" fn js_parse(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let code = if argc >= 1 && !argv.is_null() {
        unsafe { js_string(ctx, *argv) }
    } else {
        String::new()
    };

    let mut module = false;
    let mut ts = false;
    let mut filename = String::from("<parse>");
    if argc >= 2 && !argv.is_null() {
        let opts = unsafe { *argv.add(1) };
        if unsafe { qjs::sofuu_js_is_object(opts) } != 0 {
            module = unsafe { opt_bool(ctx, opts, c"module".as_ptr()) };
            ts = unsafe { opt_bool(ctx, opts, c"ts".as_ptr()) };
            if let Some(f) = unsafe { opt_string(ctx, opts, c"filename".as_ptr()) } {
                if !f.is_empty() {
                    filename = f;
                }
            }
        }
    }

    let src = if ts { crate::ts::strip(&code) } else { code };
    let verdict = unsafe { compile_only(ctx, &src, &filename, module) };

    let out = unsafe { qjs::sofuu_js_new_object(ctx) };
    unsafe {
        qjs::sofuu_js_set_property_str(
            ctx,
            out,
            c"ok".as_ptr(),
            qjs::sofuu_js_new_bool(ctx, verdict.ok as c_int),
        );
        set_str(
            ctx,
            out,
            c"mode".as_ptr(),
            if module { "module" } else { "script" },
        );
    }
    if !verdict.ok {
        unsafe {
            set_str(ctx, out, c"error".as_ptr(), &verdict.message);
            set_str(ctx, out, c"type".as_ptr(), &verdict.name);
            qjs::sofuu_js_set_property_str(
                ctx,
                out,
                c"line".as_ptr(),
                qjs::sofuu_js_new_int32(ctx, verdict.line),
            );
        }
    }
    out
}

// -- Module Registration ------------------------------------------

/// Install sofuu.parse on the engine's sofuu namespace.
///
/// # Safety
/// ctx must be a live QuickJS context on the engine thread.
pub unsafe fn mod_syntax_register(ctx: *mut JSContext) {
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

    let parse_fn = unsafe { qjs::sofuu_js_new_cfunction(ctx, js_parse, c"parse".as_ptr(), 2) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"parse".as_ptr(), parse_fn) };

    unsafe { qjs::sofuu_js_free_value(ctx, sofuu_obj) };
    unsafe { qjs::sofuu_js_free_value(ctx, global) };
}
