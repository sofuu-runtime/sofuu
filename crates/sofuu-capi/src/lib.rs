// sofuu-capi — the embeddable Sofuu C ABI (PLAN-HEADLESS H1).
//
// `libsofuu` lets any host (iOS, Android, desktop, server, edge) embed
// Sofuu with all features and zero TUI. This crate wraps sofuu-core +
// sofuu-ffi into a small stable C ABI whose surface is a JSON funnel
// (`sofuu_rt_call`) plus an eval escape hatch (`sofuu_rt_eval`).
//
// Memory contract (mirrors ffi_exports.rs): every buffer the library hands
// out through a `char **out_json` is allocated with libc::malloc; the host
// frees it with `sofuu_free()`. Never free() these with the host's own
// allocator. NULL inputs are tolerated everywhere by returning NULL/error.
//
// Error model: every fallible call returns an int (0 = SOFUU_OK, negative
// = SOFUU_ERR_*) AND, when out_json is provided, a JSON envelope:
//   {"ok":true,"result":...}
//   {"ok":false,"error":{"code":"<stable_code>","message":"<human text>"}}
//
// Threading: one SofuuRuntime is owned by one thread. Multiple runtimes
// per process are supported, each on its own thread; they share the
// centralized libuv loop, so concurrent loop-driven work is serialized.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value as JsonValue;
use sofuu_ffi::qjs;

// ── Opaque handle ─────────────────────────────────────────────────

/// The host's runtime handle. Created by sofuu_rt_new, freed by sofuu_rt_free.
/// One per thread; never shared. Opaque to C hosts (see sofuu_embed.h).
pub struct CapiRuntime {
    /// The underlying FFI runtime (QuickJS engine + builtins).
    ffi_rt: sofuu_ffi::SofuuRuntime,
}

// ── Helpers (mirrors ffi_exports.rs conventions) ──────────────────

#[inline]
unsafe fn cstr_or_null<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        None
    } else {
        CStr::from_ptr(p).to_str().ok()
    }
}

#[inline]
unsafe fn malloc_cstr(s: &str) -> *mut c_char {
    let b = s.as_bytes();
    let buf = libc::malloc(b.len() + 1) as *mut c_char;
    if buf.is_null() {
        return ptr::null_mut();
    }
    ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, b.len());
    *buf.add(b.len()) = 0;
    buf
}

/// Build the standard error envelope JSON and return it as a malloc'd C string.
unsafe fn error_envelope(code: &str, message: &str) -> *mut c_char {
    let json = format!(
        r#"{{"ok":false,"error":{{"code":{},"message":{}}}}}"#,
        serde_json::to_string(code).unwrap_or_else(|_| "\"internal\"".into()),
        serde_json::to_string(message).unwrap_or_else(|_| "\"\"".into()),
    );
    malloc_cstr(&json)
}

/// Read a JSValue as a JSON string via JS_JSONStringify. Returns "null" for
/// undefined/null, or the JSON string representation. Frees the input value.
unsafe fn value_to_json(ctx: *mut qjs::JSContext, val: qjs::JSValue) -> String {
    // undefined / null → literal "null"
    if qjs::is_undefined(val) || qjs::is_null(val) {
        qjs::sofuu_js_free_value(ctx, val);
        return "null".to_string();
    }

    // Use JS_JSONStringify(ctx, val, undefined, undefined) for objects.
    // For primitives (bool, number, string), JSON.stringify works but we
    // can also just convert directly. JS_JSONStringify handles all of them.
    let undef = qjs::sofuu_js_undefined();
    let json_val = qjs::JS_JSONStringify(ctx, val, undef, undef);
    qjs::sofuu_js_free_value(ctx, val);

    if qjs::is_exception(json_val) || qjs::is_undefined(json_val) {
        qjs::sofuu_js_free_value(ctx, json_val);
        return "null".to_string();
    }

    let s = qjs::sofuu_js_to_cstring(ctx, json_val);
    let out = if s.is_null() {
        "null".to_string()
    } else {
        let r = CStr::from_ptr(s).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, s);
        r
    };
    qjs::sofuu_js_free_value(ctx, json_val);
    out
}

// ── ABI version ───────────────────────────────────────────────────

/// Returns the ABI version this library implements (v2 = 2).
#[no_mangle]
pub unsafe extern "C" fn sofuu_embed_abi_version() -> u32 {
    2
}

/// Install (or clear, with NULL) the host log callback. Process-global:
/// applies to every runtime in this process. When set, console.log/info/
/// warn/error (and friends) are routed to `cb` instead of stdout/stderr.
/// Strings are only valid for the callback's duration; copy what you keep.
#[no_mangle]
pub unsafe extern "C" fn sofuu_embed_set_log_cb(
    cb: Option<sofuu_core::embed_config::LogCallback>,
    opaque: *mut c_void,
) {
    sofuu_core::embed_config::set_log_callback(cb, opaque);
}

// ── Lifecycle ─────────────────────────────────────────────────────

/// Create a runtime. config_json is a NUL-terminated JSON object or NULL
/// for defaults. Recognized keys (all optional):
///   { "embedded": true,           // hosted-library mode (default true)
///     "config_root": "/path",     // replaces $HOME/.sofuu derivations
///     "brain_path": "/path.qtsq", // explicit brain file
///     "api_keys": {"openai": …},  // provider keys, used ahead of env vars
///     "qtsq": true,               // QTSQ brain/KV (build-gated)
///     "enable_signals": false }   // signal handlers (default off)
/// Returns NULL on failure.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_new(config_json: *const c_char) -> *mut CapiRuntime {
    // Parse config (NULL → empty object → all defaults).
    let config_str = cstr_or_null(config_json).unwrap_or("{}");
    let config_val: JsonValue = match serde_json::from_str(config_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(), // bad JSON → NULL (caller sees SOFUU_ERR_BAD_CONFIG)
    };

    let embedded = config_val
        .get("embedded")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let config_root = config_val
        .get("config_root")
        .and_then(|v| v.as_str())
        .map(String::from);
    let enable_signals = config_val
        .get("enable_signals")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Set the process-global embedded config so core modules (process,
    // console, ai) can read it.
    sofuu_core::embed_config::configure(embedded, config_root.clone(), enable_signals);
    if let Some(keys) = config_val.get("api_keys").and_then(|v| v.as_object()) {
        for (name, key) in keys {
            if let Some(k) = key.as_str() {
                sofuu_core::embed_config::set_api_key(name, k.to_string());
            }
        }
    }
    let brain_path = config_val
        .get("brain_path")
        .and_then(|v| v.as_str())
        .map(String::from);
    sofuu_core::embed_config::set_brain_path(brain_path.clone());

    // Create the underlying runtime (QuickJS + builtins).
    let Some(ffi_rt) = sofuu_ffi::SofuuRuntime::init() else {
        return ptr::null_mut();
    };

    // Expose the embed config to user JS (agent.js reads it for brain/log
    // paths; hosts can also inspect it). Values are JSON-serialized, so no
    // manual escaping is needed.
    let root_json = match &config_root {
        Some(p) => serde_json::to_string(p).unwrap_or_else(|_| "null".into()),
        None => "null".into(),
    };
    let brain_json = match &brain_path {
        Some(p) => serde_json::to_string(p).unwrap_or_else(|_| "null".into()),
        None => "null".into(),
    };
    let src = format!(
        "globalThis.__sofuu_embed_config = {{ embedded: {}, configRoot: {}, brainPath: {} }};",
        embedded, root_json, brain_json
    );
    if let Ok(csrc) = CString::new(src) {
        unsafe {
            let ctx = ffi_rt.engine_ctx() as *mut qjs::JSContext;
            if !ctx.is_null() {
                let val = qjs::JS_Eval(
                    ctx,
                    csrc.as_ptr(),
                    csrc.as_bytes().len(),
                    c"<embed-config>".as_ptr(),
                    qjs::JS_EVAL_TYPE_GLOBAL,
                );
                if qjs::is_exception(val) {
                    // Shouldn't happen for this fixed snippet; clear and move on.
                    let exc = qjs::JS_GetException(ctx);
                    qjs::sofuu_js_free_value(ctx, exc);
                }
                qjs::sofuu_js_free_value(ctx, val);
            }
        }
    }

    // A7: per-process agent definitions. If the config includes "agents",
    // inject each definition via sofuu.agent.define(). This lets hosts
    // supply agents without filesystem access.
    if let Some(agents) = config_val.get("agents").and_then(|v| v.as_array()) {
        for agent_def in agents {
            let def_json = serde_json::to_string(agent_def).unwrap_or_default();
            let src = format!(
                "try {{ sofuu.agent.define({}); }} catch(e) {{ /* duplicate/bad def — non-fatal */ }}",
                def_json
            );
            if let Ok(csrc) = CString::new(src) {
                unsafe {
                    let ctx = ffi_rt.engine_ctx() as *mut qjs::JSContext;
                    if !ctx.is_null() {
                        let val = qjs::JS_Eval(
                            ctx,
                            csrc.as_ptr(),
                            csrc.as_bytes().len(),
                            c"<agent-def>".as_ptr(),
                            qjs::JS_EVAL_TYPE_GLOBAL,
                        );
                        qjs::sofuu_js_free_value(ctx, val);
                    }
                }
            }
        }
    }

    let handle = Box::new(CapiRuntime {
        ffi_rt,
    });

    Box::into_raw(handle)
}

/// Destroy a runtime and release its engine. NULL-tolerant.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_free(rt: *mut CapiRuntime) {
    if !rt.is_null() {
        drop(Box::from_raw(rt));
    }
}

/// Free a buffer handed out by this library (out_json strings). NULL-safe.
#[no_mangle]
pub unsafe extern "C" fn sofuu_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        libc::free(ptr);
    }
}

// ── The generic funnel ────────────────────────────────────────────

/// Call a feature by name. `method` is a dot-path under the `sofuu` JS
/// namespace (e.g. "ai.complete", "memory.open", "rlm.query").
///
/// `args_json` is either a JSON object (passed as a single options arg) or
/// a JSON array (spread as positional args); NULL/"" means no arguments.
///
/// Blocks until the feature's promise settles (the event loop is drained).
/// On return, *out_json (when non-NULL) is a malloc'd JSON envelope the
/// host frees with sofuu_free(). Returns SOFUU_OK or a SOFUU_ERR_*.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_call(
    rt: *mut CapiRuntime,
    method: *const c_char,
    args_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    if rt.is_null() || method.is_null() {
        if !out_json.is_null() {
            *out_json = error_envelope("invalid_arg", "NULL runtime or method");
        }
        return -1; // SOFUU_ERR_INVALID_ARG
    }

    let rt = &*rt;
    let method_str = match cstr_or_null(method) {
        Some(s) => s,
        None => {
            if !out_json.is_null() {
                *out_json = error_envelope("invalid_arg", "method is not valid UTF-8");
            }
            return -1;
        }
    };

    // Parse args: NULL/"" → no args, JSON object → single arg, JSON array → spread.
    let args_part = match cstr_or_null(args_json) {
        None | Some("") => "undefined".to_string(),
        Some(s) => {
            // Validate it's valid JSON.
            match serde_json::from_str::<JsonValue>(s) {
                Ok(_) => format!("JSON.parse({})", serde_json::to_string(s).unwrap_or_default()),
                Err(_) => {
                    if !out_json.is_null() {
                        *out_json = error_envelope("bad_args", "args_json is not valid JSON");
                    }
                    return -1;
                }
            }
        }
    };

    // Generate the JS expression: an async IIFE that resolves the method
    // path on `sofuu`, calls it, and stores the envelope in __capi_r.
    //
    // We use an async IIFE so the call is awaited, then drain the event
    // loop (sofuu_loop_run) to resolve the promise before reading the result.
    let js = format!(
        r#"var __capi_r=null;(async function(){{
  try {{
    var p={method_json}.split("."),o=sofuu;
    for(var i=0;i<p.length;i++){{if(o==null)throw new TypeError("sofuu."+p.slice(0,i+1).join(".")+" is not available");o=o[p[i]]}}
    if(typeof o!=="function")throw new TypeError("sofuu."+{method_json}+" is not a function");
    var a={args_part};
    var r=a===undefined?await o():(Array.isArray(a)?await o.apply(null,a):await o(a));
    __capi_r={{ok:true,result:r}}
  }}catch(e){{
    __capi_r={{ok:false,error:{{code:"js_exception",message:e.message||String(e)}}}}
  }}
}})()"#,
        method_json = serde_json::to_string(method_str).unwrap_or_default(),
        args_part = args_part,
    );

    rt_call_and_capture(rt, &js, out_json)
}

/// Evaluate arbitrary JS (the eval escape hatch). The source is evaluated
/// as a global script. For async work, the result is the Promise object
/// itself — use sofuu_rt_call for features that return promises.
/// The event loop is drained after evaluation.
/// Returns SOFUU_OK or a SOFUU_ERR_*.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_eval(
    rt: *mut CapiRuntime,
    source: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    if rt.is_null() || source.is_null() {
        if !out_json.is_null() {
            *out_json = error_envelope("invalid_arg", "NULL runtime or source");
        }
        return -1;
    }

    let rt = &*rt;
    let src = match cstr_or_null(source) {
        Some(s) => s,
        None => {
            if !out_json.is_null() {
                *out_json = error_envelope("invalid_arg", "source is not valid UTF-8");
            }
            return -1;
        }
    };

    let ctx = rt.ffi_rt.engine_ctx() as *mut qjs::JSContext;
    if ctx.is_null() {
        if !out_json.is_null() {
            *out_json = error_envelope("internal", "engine context is null");
        }
        return -6; // SOFUU_ERR_INTERNAL
    }

    // Wrap the source so we always get a JSON envelope:
    //   (function(){ try { return {ok:true, result: eval(SRC) }; }
    //                catch(e) { return {ok:false, error:{...}}; } })()
    // This captures sync results. For async, the result will be a Promise
    // object — the caller should use sofuu_rt_call for async features.
    let wrapper = format!(
        r#"(function(){{try{{return{{ok:true,result:eval({src_json})}}}}catch(e){{return{{ok:false,error:{{code:"js_exception",message:e.message||String(e)}}}}}}}})()"#,
        src_json = serde_json::to_string(src).unwrap_or_default(),
    );

    // Eval the wrapper.
    let c_wrapper = CString::new(wrapper).unwrap_or_default();
    let result = qjs::JS_Eval(
        ctx,
        c_wrapper.as_ptr(),
        c_wrapper.as_bytes().len(),
        c"<capi-eval>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );

    if qjs::is_exception(result) {
        // Exception during eval itself (shouldn't happen with the try/catch
        // wrapper, but handle it defensively).
        let exc = qjs::JS_GetException(ctx);
        let msg = qjs::sofuu_js_to_cstring(ctx, exc);
        let err_msg = if msg.is_null() {
            "unknown eval error".to_string()
        } else {
            let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, msg);
            s
        };
        qjs::sofuu_js_free_value(ctx, exc);
        qjs::sofuu_js_free_value(ctx, result);
        if !out_json.is_null() {
            *out_json = error_envelope("js_exception", &err_msg);
        }
        return -4; // SOFUU_ERR_JS_EXCEPTION
    }

    // Drain microtasks + event loop.
    drain_loop(ctx);

    // The result is the envelope object {ok:true, result:...} or
    // {ok:false, error:{...}}. JSON.stringify it and return.
    let json_str = value_to_json(ctx, result);

    if !out_json.is_null() {
        *out_json = malloc_cstr(&json_str);
    }

    0 // SOFUU_OK
}

// ── Streaming / cancellable variant ───────────────────────────────

/// Process-global counter for cancel IDs.
static NEXT_CANCEL_ID: AtomicU64 = AtomicU64::new(1);

// Store the host callback in a thread-local so js_stream_callback can read it.
// (QuickJS + libuv are single-threaded, so this is safe.)
thread_local! {
    static STREAM_CB: std::cell::Cell<Option<(
        unsafe extern "C" fn(*const c_char, *mut c_void),
        *mut c_void,
    )>> = std::cell::Cell::new(None);
}

/// The C-callable callback that JS invokes for each streaming event.
/// It reads the event JSON string and forwards it to the host callback.
///
/// # Safety
/// `ctx` must be a live QuickJS context; `this_val`/`argc`/`argv` per QuickJS calling convention.
unsafe extern "C" fn js_stream_callback(
    ctx: *mut qjs::JSContext,
    _this_val: qjs::JSValueConst,
    argc: c_int,
    argv: *const qjs::JSValueConst,
) -> qjs::JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }

    // Read the event JSON string from argv[0].
    let event_val = *argv;
    let event_str = if qjs::sofuu_js_is_string(event_val) != 0 {
        let s = qjs::sofuu_js_to_cstring(ctx, event_val);
        if s.is_null() {
            None
        } else {
            let r = CStr::from_ptr(s).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, s);
            Some(r)
        }
    } else {
        // Not a string — try JSON stringify.
        let json_val = qjs::JS_JSONStringify(ctx, event_val, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
        if qjs::is_exception(json_val) || qjs::is_undefined(json_val) || qjs::is_null(json_val) {
            qjs::sofuu_js_free_value(ctx, json_val);
            None
        } else {
            let s = qjs::sofuu_js_to_cstring(ctx, json_val);
            qjs::sofuu_js_free_value(ctx, json_val);
            if s.is_null() {
                None
            } else {
                let r = CStr::from_ptr(s).to_string_lossy().into_owned();
                qjs::sofuu_js_free_cstring(ctx, s);
                Some(r)
            }
        }
    };

    // Forward to the host callback.
    STREAM_CB.with(|cell| {
        if let Some((cb, opaque)) = cell.get() {
            if let Some(ref json) = event_str {
                let c_json = CString::new(json.as_str()).unwrap_or_default();
                cb(c_json.as_ptr(), opaque);
            }
        }
    });

    qjs::sofuu_js_undefined()
}

/// Streaming variant: calls a method that accepts an `onStep` callback option
/// (e.g. `agent.run`) and forwards each event to the host's `on_event`
/// callback on the calling thread.
///
/// The method is called with `args_json` merged with `{ onStep: <internal cb> }`.
/// *out_cancel_id receives a token usable with `sofuu_rt_cancel`.
/// Returns SOFUU_OK or a SOFUU_ERR_*.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_call_stream(
    rt: *mut CapiRuntime,
    method: *const c_char,
    args_json: *const c_char,
    on_event: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    opaque: *mut c_void,
    out_cancel_id: *mut u64,
) -> c_int {
    if rt.is_null() || method.is_null() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }

    // If no callback is provided, just fall back to a regular call.
    if on_event.is_none() {
        let mut out: *mut c_char = ptr::null_mut();
        return sofuu_rt_call(rt, method, args_json, &mut out);
    }

    let rt_ref = &*rt;
    let ctx = rt_ref.ffi_rt.engine_ctx() as *mut qjs::JSContext;
    if ctx.is_null() {
        return -6; // SOFUU_ERR_INTERNAL
    }

    // Generate a cancel ID.
    let cancel_id = NEXT_CANCEL_ID.fetch_add(1, Ordering::Relaxed);
    if !out_cancel_id.is_null() {
        *out_cancel_id = cancel_id;
    }

    // Store the host callback in thread-local so js_stream_callback can read it.
    STREAM_CB.with(|cell| {
        cell.set(Some((on_event.unwrap(), opaque)));
    });

    // Register the C callback as a global JS function so we can pass it as onStep.
    let cb_name = "__sofuu_stream_cb";
    let c_name = CString::new(cb_name).unwrap();
    let global = qjs::sofuu_js_get_global_object(ctx);
    let fn_val = qjs::sofuu_js_new_cfunction(
        ctx,
        js_stream_callback,
        c_name.as_ptr(),
        1,
    );
    qjs::sofuu_js_set_property_str(ctx, global, c_name.as_ptr(), fn_val);
    qjs::sofuu_js_free_value(ctx, global);

    // Build the JS: merge args_json with onStep callback, then call the method.
    let method_str = match cstr_or_null(method) {
        Some(s) => s.to_string(),
        None => return -1,
    };

    let args_part = match cstr_or_null(args_json) {
        None | Some("") => "{}".to_string(),
        Some(s) => {
            match serde_json::from_str::<JsonValue>(s) {
                Ok(_) => s.to_string(),
                Err(_) => return -1, // bad_args
            }
        }
    };

    // For agent.run, the funnel args are {target, task, ...opts}.
    // agent.run takes (target, task, opts) — we spread them.
    let args_json_lit = serde_json::to_string(&args_part).unwrap_or_default();
    let call_expr = if method_str == "agent.run" {
        format!(
            r#"var a=JSON.parse({args_json_lit});
            if(typeof a!=="object"||a===null)a={{}};
            a.onStep=function(evt){{try{{globalThis.__sofuu_stream_cb(JSON.stringify(evt));}}catch(e){{}}}};
            a.signal={cancel_id};
            var r=await o(a.target, a.task, a);"#
        )
    } else {
        format!(
            r#"var a=JSON.parse({args_json_lit});
            if(typeof a!=="object"||a===null)a={{}};
            a.onStep=function(evt){{try{{globalThis.__sofuu_stream_cb(JSON.stringify(evt));}}catch(e){{}}}};
            a.signal={cancel_id};
            var r=await o(a);"#
        )
    };

    let js = format!(
        r#"var __capi_r=null;(async function(){{
  try {{
    var p={method_json}.split("."),o=sofuu;
    for(var i=0;i<p.length;i++){{if(o==null)throw new TypeError("sofuu."+p.slice(0,i+1).join(".")+" is not available");o=o[p[i]]}}
    if(typeof o!=="function")throw new TypeError("sofuu."+{method_json}+" is not a function");
    {call_expr}
    __capi_r={{ok:true,result:r}}
  }}catch(e){{
    __capi_r={{ok:false,error:{{code:"js_exception",message:e.message||String(e)}}}}
  }}
}})()"#,
        method_json = serde_json::to_string(&method_str).unwrap_or_default(),
        call_expr = call_expr,
    );

    // Streaming contract: results are delivered incrementally through the
    // host's on_event callback; the final envelope is intentionally not
    // returned (the API has no out_json parameter — see sofuu_embed.h).
    let rc = rt_call_and_capture(rt_ref, &js, ptr::null_mut());

    // Clean up: clear the thread-local callback + the global JS function.
    STREAM_CB.with(|cell| {
        cell.set(None);
    });
    let global = qjs::sofuu_js_get_global_object(ctx);
    qjs::sofuu_js_set_property_str(ctx, global, c_name.as_ptr(), qjs::sofuu_js_undefined());
    qjs::sofuu_js_free_value(ctx, global);

    rc
}

/// Request cancellation of a streaming call. Calls `sofuu.agent.cancel(id)`.
/// Returns SOFUU_OK if the cancel ID was recognized, -1 if not found.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_cancel(
    rt: *mut CapiRuntime,
    cancel_id: u64,
) -> c_int {
    if rt.is_null() || cancel_id == 0 {
        return -1;
    }

    let rt_ref = &*rt;
    let ctx = rt_ref.ffi_rt.engine_ctx() as *mut qjs::JSContext;
    if ctx.is_null() {
        return -6;
    }

    // Call sofuu.agent.cancel(cancelId) — it returns true/false.
    let js = format!(
        r#"(function(){{try{{return{{ok:true,result:sofuu.agent.cancel({})}}}}catch(e){{return{{ok:false,error:{{code:"js_exception",message:e.message||String(e)}}}}}}}})()"#,
        cancel_id,
    );

    let mut out: *mut c_char = ptr::null_mut();
    let _ = rt_call_and_capture(rt_ref, &js, &mut out);

    // Check if the cancel was recognized (result:true).
    let recognized = if !out.is_null() {
        let s = CStr::from_ptr(out).to_string_lossy();
        let ok = s.contains(r#""ok":true"#) && s.contains(r#""result":true"#);
        sofuu_free(out as *mut c_void);
        ok
    } else {
        false
    };

    if recognized {
        0
    } else {
        -1
    }
}

// ── Internal helpers ──────────────────────────────────────────────

/// Core eval-and-capture: eval a JS expression that sets `__capi_r` to the
/// result envelope, drain the event loop, then read and return the envelope.
unsafe fn rt_call_and_capture(
    rt: &CapiRuntime,
    js: &str,
    out_json: *mut *mut c_char,
) -> c_int {
    let ctx = rt.ffi_rt.engine_ctx() as *mut qjs::JSContext;
    if ctx.is_null() {
        if !out_json.is_null() {
            *out_json = error_envelope("internal", "engine context is null");
        }
        return -6;
    }

    // Eval the JS (sets __capi_r via an async IIFE + .then chain).
    let c_js = CString::new(js).unwrap_or_default();
    let result = qjs::JS_Eval(
        ctx,
        c_js.as_ptr(),
        c_js.as_bytes().len(),
        c"<capi-call>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );

    if qjs::is_exception(result) {
        let exc = qjs::JS_GetException(ctx);
        let msg = qjs::sofuu_js_to_cstring(ctx, exc);
        let err_msg = if msg.is_null() {
            "unknown eval error".to_string()
        } else {
            let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, msg);
            s
        };
        qjs::sofuu_js_free_value(ctx, exc);
        qjs::sofuu_js_free_value(ctx, result);
        if !out_json.is_null() {
            *out_json = error_envelope("js_exception", &err_msg);
        }
        return -4;
    }
    // The result of the async IIFE is a Promise — we don't need it.
    qjs::sofuu_js_free_value(ctx, result);

    // Drain the event loop (libuv + JS microtasks) until the promise settles.
    drain_loop(ctx);

    // Read __capi_r — the envelope set by the .then/.catch handler.
    let read_expr = c"JSON.stringify(__capi_r)";
    let envelope = qjs::JS_Eval(
        ctx,
        read_expr.as_ptr(),
        read_expr.to_bytes().len(),
        c"<capi-read>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );

    if qjs::is_exception(envelope) {
        // Shouldn't happen — clear exception state.
        let exc = qjs::JS_GetException(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
        qjs::sofuu_js_free_value(ctx, envelope);
        if !out_json.is_null() {
            *out_json = error_envelope("internal", "failed to read __capi_r");
        }
        return -6;
    }

    // Read the JSON string.
    let s = qjs::sofuu_js_to_cstring(ctx, envelope);
    let json = if s.is_null() {
        r#"{"ok":false,"error":{"code":"internal","message":"null stringify result"}}"#.to_string()
    } else {
        let r = CStr::from_ptr(s).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, s);
        r
    };
    qjs::sofuu_js_free_value(ctx, envelope);

    // Clean up __capi_r.
    let undef = qjs::sofuu_js_undefined();
    let global = qjs::sofuu_js_get_global_object(ctx);
    qjs::sofuu_js_set_property_str(ctx, global, c"__capi_r".as_ptr(), undef);
    qjs::sofuu_js_free_value(ctx, global);

    if !out_json.is_null() {
        *out_json = malloc_cstr(&json);
    }

    0 // SOFUU_OK
}

/// Drain both QuickJS microtasks and the libuv event loop until idle.
/// This blocks until all promises settle and all I/O completes.
///
/// # Safety
/// `ctx` must be the live QuickJS context (from the runtime handle).
unsafe fn drain_loop(ctx: *mut qjs::JSContext) {
    // Get the JSRuntime for JS_ExecutePendingJob.
    let rt = qjs::JS_GetRuntime(ctx);

    // Pump microtasks first (immediate completions).
    loop {
        let mut ctx1: *mut qjs::JSContext = ptr::null_mut();
        let e = qjs::JS_ExecutePendingJob(rt, &mut ctx1);
        if e <= 0 {
            break;
        }
    }

    // Now drain the libuv loop (I/O callbacks fire promises, which
    // schedule microtasks, which we drain after each UV_RUN_ONCE).
    // Call the Rust function directly (not via extern "C") to ensure
    // the linker includes sofuu-core, which provides the no_mangle
    // engine symbols (sofuu_init, sofuu_destroy, sofuu_engine_ctx, etc.).
    sofuu_core::rt::event_loop::sofuu_loop_run(ctx);
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Serialize with any other test that touches the process-global libuv loop.
    // (sofuu-core's rt::TEST_LOOP_LOCK is not pub, so we use our own.)
    use std::sync::Mutex;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn abi_version_is_2() {
        let _g = TEST_LOCK.lock().unwrap();
        assert_eq!(unsafe { sofuu_embed_abi_version() }, 2);
    }

    #[test]
    fn rt_new_and_free_with_null_config() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null(), "rt_new with NULL config should succeed");
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn rt_new_with_empty_config() {
        let _g = TEST_LOCK.lock().unwrap();
        let cfg = CString::new("{}").unwrap();
        let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
        assert!(!rt.is_null(), "rt_new with {{}} config should succeed");
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn rt_new_with_bad_json_returns_null() {
        let _g = TEST_LOCK.lock().unwrap();
        let bad = CString::new("{invalid json").unwrap();
        let rt = unsafe { sofuu_rt_new(bad.as_ptr()) };
        assert!(rt.is_null(), "rt_new with bad JSON should return NULL");
    }

    #[test]
    fn sofuu_free_null_is_noop() {
        // Just verify it doesn't crash.
        unsafe { sofuu_free(ptr::null_mut()) };
    }

    #[test]
    fn eval_sync_arithmetic() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let src = CString::new("1 + 1").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0, "eval 1+1 should succeed");
        assert!(!out.is_null(), "out_json should not be NULL");

        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        // The envelope wraps the result: {"ok":true,"result":...}
        assert!(json.contains(r#""ok":true"#), "envelope ok=true, got: {json}");
        assert!(json.contains(r#""result":2"#), "result should be 2, got: {json}");

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn eval_sync_string() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let src = CString::new(r#""hello""#).unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0);
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains(r#""hello""#), "result should contain hello, got: {json}");

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn eval_throws_returns_error_envelope() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let src = CString::new("throw new Error('boom')").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0, "eval with try/catch wrapper should succeed (rc=0)");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains(r#""ok":false"#), "envelope ok=false, got: {json}");
        assert!(json.contains("boom"), "error message should contain 'boom', got: {json}");

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn call_null_runtime_returns_error() {
        let _g = TEST_LOCK.lock().unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let method = CString::new("ai.complete").unwrap();
        let rc = unsafe { sofuu_rt_call(ptr::null_mut(), method.as_ptr(), ptr::null(), &mut out) };
        assert_eq!(rc, -1); // SOFUU_ERR_INVALID_ARG
        assert!(!out.is_null());
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains("invalid_arg"));
        unsafe { sofuu_free(out as *mut c_void) };
    }

    #[test]
    fn call_unknown_method_returns_error_envelope() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let method = CString::new("nonexistent.method").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_call(rt, method.as_ptr(), ptr::null(), &mut out) };
        assert_eq!(rc, 0, "call should return rc=0 (error is in the envelope)");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains(r#""ok":false"#), "envelope ok=false for unknown method, got: {json}");

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn call_bad_args_json_returns_error() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let method = CString::new("ai.complete").unwrap();
        let bad_args = CString::new("not-json").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_call(rt, method.as_ptr(), bad_args.as_ptr(), &mut out) };
        assert_eq!(rc, -1, "bad args_json should be SOFUU_ERR_INVALID_ARG");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains("bad_args"));
        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn stream_calls_agent_run_with_events() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // Define a minimal agent that just returns a final answer via a mock.
        let setup = CString::new(
            r#"sofuu.agent.define({
              name: "stream-test",
              system: "You are a test agent.",
              provider: "mock",
              model: "mock-stream-test",
              budget: { maxSteps: 1 }
            });
            // Intercept ai.complete: return a final answer on the first call.
            var origComplete = sofuu.ai.complete;
            sofuu.ai.complete = function(opts) {
              if (opts.model === 'mock-stream-test') {
                return Promise.resolve({ text: 'hello from mock' });
              }
              return origComplete.call(sofuu.ai, opts);
            };"#,
        ).unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) };
        assert_eq!(rc, 0, "setup should succeed");
        unsafe { sofuu_free(out as *mut c_void) };

        // Now stream agent.run — collect events.
        static STREAM_EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());
        STREAM_EVENTS.lock().unwrap().clear();

        unsafe extern "C" fn collect_events(event_json: *const c_char, _opaque: *mut c_void) {
            let s = unsafe { CStr::from_ptr(event_json) }.to_string_lossy().into_owned();
            STREAM_EVENTS.lock().unwrap().push(s);
        }

        let method = CString::new("agent.run").unwrap();
        let args = CString::new(r#"{"target":"stream-test","task":"say hi"}"#).unwrap();
        let mut cancel_id: u64 = 0;
        let _rc = unsafe {
            sofuu_rt_call_stream(
                rt,
                method.as_ptr(),
                args.as_ptr(),
                Some(collect_events),
                ptr::null_mut(),
                &mut cancel_id,
            )
        };

        let events = STREAM_EVENTS.lock().unwrap().clone();
        // We should have received at least a "start" event from the agent loop.
        assert!(!events.is_empty(), "should have received at least one event, got: {events:?}");

        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn multi_instance_two_runtimes_independent() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt1 = unsafe { sofuu_rt_new(ptr::null()) };
        let rt2 = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt1.is_null());
        assert!(!rt2.is_null());

        // Set a variable in rt1.
        let set_src = CString::new("var __test_var = 42").unwrap();
        let mut out1: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt1, set_src.as_ptr(), &mut out1) };
        assert_eq!(rc, 0);
        unsafe { sofuu_free(out1 as *mut c_void) };

        // rt2 should NOT see __test_var (separate QuickJS contexts).
        let get_src = CString::new("typeof __test_var").unwrap();
        let mut out2: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt2, get_src.as_ptr(), &mut out2) };
        assert_eq!(rc, 0);
        let json = unsafe { CStr::from_ptr(out2) }.to_string_lossy();
        assert!(
            json.contains(r#""undefined""#),
            "rt2 should not see rt1's variable, got: {json}"
        );

        unsafe { sofuu_free(out2 as *mut c_void) };
        unsafe { sofuu_rt_free(rt1) };
        unsafe { sofuu_rt_free(rt2) };
    }

    #[test]
    fn embedded_process_exit_throws_exit_error() {
        let _g = TEST_LOCK.lock().unwrap();
        // NULL config → embedded=true by default.
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let src = CString::new(
            "try { process.exit(3); 'no-throw' } catch (e) { e.name + ':' + e.exitCode }",
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0);
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains(r#""ok":true"#), "envelope ok=true, got: {json}");
        assert!(
            json.contains("ExitError:3"),
            "process.exit should throw ExitError with exitCode, got: {json}"
        );

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    // Capture sink for the log-callback test.
    static LOG_CAPTURED: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

    unsafe extern "C" fn test_log_cb(level: *const c_char, msg: *const c_char, _opaque: *mut c_void) {
        let l = CStr::from_ptr(level).to_string_lossy().into_owned();
        let m = CStr::from_ptr(msg).to_string_lossy().into_owned();
        if let Ok(mut v) = LOG_CAPTURED.lock() {
            v.push((l, m));
        }
    }

    #[test]
    fn log_callback_routes_console_output() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        LOG_CAPTURED.lock().unwrap().clear();
        unsafe { sofuu_embed_set_log_cb(Some(test_log_cb), ptr::null_mut()) };

        let src = CString::new(r#"console.log("hi from cb"); console.warn("careful")"#).unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0);

        // Always restore the global callback before asserting.
        unsafe { sofuu_embed_set_log_cb(None, ptr::null_mut()) };

        let captured = LOG_CAPTURED.lock().unwrap().clone();
        assert!(
            captured.iter().any(|(l, m)| l == "log" && m == "hi from cb"),
            "console.log should reach the callback, got: {captured:?}"
        );
        assert!(
            captured.iter().any(|(l, m)| l == "warn" && m == "careful"),
            "console.warn should reach the callback, got: {captured:?}"
        );

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn embed_config_visible_to_js() {
        let _g = TEST_LOCK.lock().unwrap();
        let cfg = CString::new(
            r#"{"config_root":"/tmp/sofuu-capi-test","brain_path":"/tmp/sofuu-capi-brain.qtsq"}"#,
        )
        .unwrap();
        let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
        assert!(!rt.is_null());

        let src = CString::new(
            "__sofuu_embed_config.configRoot + '|' + __sofuu_embed_config.brainPath",
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0);
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(
            json.contains("/tmp/sofuu-capi-test|/tmp/sofuu-capi-brain.qtsq"),
            "embed config should be visible to JS, got: {json}"
        );

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };

        // Reset the process-global config so later tests see defaults.
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn agent_funnel_list_and_define() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // Define an agent via the eval escape hatch.
        let define_src = CString::new(
            r#"sofuu.agent.define({ name: "capi-agent", system: "test", provider: "mock", model: "mock" });"true""#,
        ).unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, define_src.as_ptr(), &mut out) };
        assert_eq!(rc, 0);
        unsafe { sofuu_free(out as *mut c_void) };

        // List agents via the funnel.
        let method = CString::new("agent.list").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_call(rt, method.as_ptr(), ptr::null(), &mut out) };
        assert_eq!(rc, 0, "agent.list should succeed");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains(r#""ok":true"#), "agent.list envelope ok=true, got: {json}");
        assert!(json.contains("capi-agent"), "agent.list should contain capi-agent, got: {json}");
        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn per_process_agent_definitions_in_config() {
        let _g = TEST_LOCK.lock().unwrap();
        let cfg = CString::new(
            r#"{"agents":[{"name":"config-agent","system":"from config","provider":"mock","model":"mock"}]}"#,
        ).unwrap();
        let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
        assert!(!rt.is_null());

        // Verify the agent was defined.
        let method = CString::new("agent.list").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_call(rt, method.as_ptr(), ptr::null(), &mut out) };
        assert_eq!(rc, 0);
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        assert!(json.contains("config-agent"), "config-defined agent should be in list, got: {json}");
        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn cancel_unknown_id_returns_error() {
        let _g = TEST_LOCK.lock().unwrap();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // Cancel a non-existent ID should return -1 (not found).
        let rc = unsafe { sofuu_rt_cancel(rt, 99999) };
        assert_eq!(rc, -1, "cancel of unknown ID should return -1");

        unsafe { sofuu_rt_free(rt) };
    }
}
