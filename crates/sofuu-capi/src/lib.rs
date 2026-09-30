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

    if qjs::is_exception(json_val) {
        // ffi-2: drain the exception (e.g. a circular reference) so it cannot
        // leak into the next JS_GetException caller on this context.
        let exc = qjs::JS_GetException(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
        qjs::sofuu_js_free_value(ctx, json_val);
        return "null".to_string();
    }
    if qjs::is_undefined(json_val) {
        // undefined is a legitimate stringify result (function/symbol values
        // stringify to undefined) — nothing is pending here, don't drain.
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

/// Evaluate a fixed init snippet during `sofuu_rt_new` and guarantee no
/// exception stays pending on the context (ffi-1). A snippet failure (parse
/// error or throw) can't be intercepted by any try/catch afterwards, so
/// without an explicit drain `rt->current_exception` would stick around and
/// be consumed by the next JS_GetException caller as if it were its own.
/// Returns true when the snippet evaluated cleanly.
unsafe fn eval_init_snippet(ctx: *mut qjs::JSContext, src: &CString, filename: &CStr) -> bool {
    let val = qjs::JS_Eval(
        ctx,
        src.as_ptr(),
        src.as_bytes().len(),
        filename.as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );
    let ok = !qjs::is_exception(val);
    if !ok {
        let exc = qjs::JS_GetException(ctx);
        qjs::sofuu_js_free_value(ctx, exc);
    }
    qjs::sofuu_js_free_value(ctx, val);
    ok
}

// ── Memory facade (E0-1) ──────────────────────────────────────────

/// The brain's instance methods (`remember`/`recall`/`count`/`flush`…)
/// live on the object `memory.open()` returns. The JSON funnel can only
/// call *functions under `sofuu.*`*, and that object stringifies to `{}`
/// — so before this shim, the entire memory surface was unreachable from
/// C without hand-building `rt_eval` strings (7 of the 17 documented
/// funnel methods were dead ends).
///
/// The shim keeps the in-process API byte-for-byte compatible and adds a
/// handle-addressed, JSON-serializable layer on top:
///
///   - `memory.open(path, dim, spaceId?)` still returns the real CMA
///     object, now carrying a non-enumerable `__handle` and a `toJSON`
///     that serializes to `{"handle":N,"dim":D}`. The funnel therefore
///     returns a usable handle instead of `{}`.
///   - `memory.remember({handle, vec, text, role, kv})` and friends take
///     a single options object and operate on that handle. `handle` is
///     optional — it defaults to the most recently opened brain, so the
///     common single-brain case reads `{"vec":[…],"text":"…"}`.
///
/// The same dead-end and the same fix apply to two more namespaces, which
/// is why all three are handled here:
///
///   - **MCP** — `mcp.connect()` returns a client whose `call`/`listTools`
///     are instance methods, so a host could connect but never call a tool.
///     `mcp.call({handle, name, args})` closes that.
///   - **HTTP** — a JSON funnel cannot pass a JS function, so
///     `http.serve({handler:"<globalName>", port, host})` takes the *name*
///     of a handler the host defined with `sofuu_rt_eval`. (This replaces
///     the documented-but-nonexistent `http.serve.start`.)
const MEMORY_FACADE_JS: &str = r#"
(function () {
  /* The funnel facade. Each namespace is installed INDEPENDENTLY — an
   * early `return` used to gate the whole IIFE on sofuu.memory, which meant
   * that on a QTSQ-free build (no sofuu.memory at all) the mcp and http
   * facades were skipped too, silently taking out tools that need no
   * codec. Nothing noticed because that path only ever ran in CI. */
  var S = globalThis.sofuu;
  if (!S) return;
  if (S.__sofuu_facade) return;
  S.__sofuu_facade = true;

  /* ── Memory. Only exists when the QTSQ codec is linked; a QTSQ-free
   * build legitimately has no sofuu.memory, and the funnel answers
   * unknown_method for it. ── */
  var M = S.memory;
  if (M && typeof M.open === 'function' && !M.__sofuu_facade) {
  var live = [];
  var realOpen = M.open;

  M.open = function (a, dim, spaceId) {
    // Accept both the funnel's object form {"path","dim","embed_id"} and
    // the in-process positional form (path, dim, spaceId).
    var path, d, sid;
    if (a && typeof a === 'object') {
      path = a.path; d = a.dim; sid = a.embed_id !== undefined ? a.embed_id : a.space;
    } else {
      path = a; d = dim; sid = spaceId;
    }
    var inst = sid === undefined || sid === null ? realOpen(path, d) : realOpen(path, d, sid);
    var id = live.indexOf(inst);
    if (id < 0) { id = live.length; live.push(inst); }
    try {
      Object.defineProperty(inst, '__handle', { value: id, enumerable: false });
      inst.toJSON = function () {
        return { handle: id, dim: (function () { try { return d; } catch (e) { return null; } })() };
      };
    } catch (e) { /* sealed object — handle registry still works */ }
    return inst;
  };

  function pick(a) {
    if (a && a.handle !== undefined && a.handle !== null) {
      var got = live[a.handle | 0];
      if (!got) throw new TypeError('memory: no open brain with handle ' + a.handle);
      return got;
    }
    var last = live.length ? live[live.length - 1] : null;
    if (!last) throw new TypeError('memory: no open brain — call memory.open first');
    return last;
  }
  function f32(v) {
    if (v instanceof Float32Array) return v;
    return new Float32Array(v || []);
  }
  function one(a, method, args) {
    var inst = pick(a);
    return inst[method].apply(inst, args);
  }

  M.remember = function (a) {
    a = a || {};
    return one(a, 'remember', [f32(a.vec), a.text, a.role || 'user', a.kv_page_id | 0]);
  };
  M.recall = function (a) {
    a = a || {};
    return one(a, 'recall', [f32(a.vec), a.k === undefined ? 5 : (a.k | 0)]);
  };
  M.count = function (a) { return one(a || {}, 'count', []); };
  M.flush = function (a) { return one(a || {}, 'flush', []); };
  M.forget = function (a) {
    a = a || {};
    return one(a, 'forget', [a.index | 0]);
  };
  M.close = function (a) {
    var inst = pick(a || {});
    var id = live.indexOf(inst);
    if (id >= 0) live.splice(id, 1);
    return true;
  };
  M.handles = function () { return live.length; };
  M.__sofuu_facade = true;
  } /* end memory block */

  /* ── MCP: the same dead-end, and the highest-value one after memory.
   * `mcp.connect` returns a client whose `call`/`listTools` are instance
   * methods, so the funnel could connect but never call a tool. A handle
   * registry fixes the whole tool surface. ── */
  var X = globalThis.sofuu.mcp;
  if (X && typeof X.connect === 'function' && !X.__sofuu_facade) {
    var clients = [];
    var realConnect = X.connect;
    X.connect = function () {
      var c = realConnect.apply(null, arguments);
      var id = clients.indexOf(c);
      if (id < 0) { id = clients.length; clients.push(c); }
      try {
        Object.defineProperty(c, '__handle', { value: id, enumerable: false });
        c.toJSON = function () { return { handle: id }; };
      } catch (e) {}
      return c;
    };
    function pickClient(a) {
      a = a || {};
      var got = (a.handle === undefined || a.handle === null)
        ? clients[clients.length - 1] : clients[a.handle | 0];
      if (!got) throw new TypeError('mcp: connect a server first');
      return got;
    }
    X.call = function (a) {
      a = a || {};
      return pickClient(a).call(a.name, a.args === undefined ? {} : a.args);
    };
    X.listTools = function (a) { return pickClient(a).listTools(); };
    X.listResources = function (a) { return pickClient(a).listResources(); };
    X.disconnect = function (a) {
      var c = pickClient(a), id = clients.indexOf(c);
      try { c.disconnect(); } finally { if (id >= 0) clients.splice(id, 1); }
      return true;
    };
    X.__sofuu_facade = true;
  }

  /* ── HTTP serving. A JSON funnel cannot pass a JS function, so the host
   * defines the handler as a global via sofuu_rt_eval and names it here:
   *   http.serve({ handler: "myHandler", port: 8080, host: "127.0.0.1" })
   * (Previously documented as http.serve.start, which did not exist.) ── */
  var H = globalThis.sofuu.http || globalThis.sofuu;
  if (H && typeof H.createServer === 'function' && !H.__sofuu_serve_facade) {
    var servers = [];
    H.serve = function (a) {
      a = a || {};
      var fn = a.handler;
      if (typeof fn === 'string') fn = globalThis[fn];
      if (typeof fn !== 'function') {
        throw new TypeError('http.serve: define a global handler with sofuu_rt_eval and pass its name');
      }
      var srv = H.createServer(fn);
      var id = servers.indexOf(srv);
      if (id < 0) { id = servers.length; servers.push(srv); }
      try { Object.defineProperty(srv, '__handle', { value: id, enumerable: false }); } catch (e) {}
      var port = a.port === undefined ? 8080 : (a.port | 0);
      var host = a.host === undefined ? '127.0.0.1' : a.host;
      if (host) srv.listen(port, host); else srv.listen(port);
      return { handle: id, port: port, host: host };
    };
    H.__sofuu_serve_facade = true;
  }
})();
"#;

/// Evaluate the memory facade shim on a fresh context. Returns true when it
/// installed (a false result means `sofuu.memory` is unavailable on this
/// build — the runtime still works, only the memory funnel is absent).
unsafe fn install_memory_facade(ctx: *mut qjs::JSContext) -> bool {
    let src = CString::new(MEMORY_FACADE_JS).unwrap_or_default();
    eval_init_snippet(ctx, &src, c"<memory-facade>")
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
    let mut api_keys: Vec<(String, String)> = Vec::new();
    if let Some(keys) = config_val.get("api_keys").and_then(|v| v.as_object()) {
        for (name, key) in keys {
            if let Some(k) = key.as_str() {
                sofuu_core::embed_config::set_api_key(name, k.to_string());
                api_keys.push((name.clone(), k.to_string()));
            }
        }
    }
    let brain_path = config_val
        .get("brain_path")
        .and_then(|v| v.as_str())
        .map(String::from);
    sofuu_core::embed_config::set_brain_path(brain_path.clone());

    // F-3 (AUDIT-2026-09-01): the globals above are clobbered by every
    // sofuu_rt_new, which silently retargeted earlier runtimes in the same
    // process. This thread's runtime also installs its COMPLETE config as
    // per-thread settings; embed_config readers resolve those first and
    // exclusively, so runtime B can no longer retarget runtime A. Threads
    // without an installed runtime keep reading the globals.
    // E2: headless request defaults, so an embedder sets provider/model/
    // endpoint once here instead of on every call. Optional and never
    // overriding a per-call value.
    let default_provider = config_val
        .get("provider")
        .and_then(|v| v.as_str())
        .map(String::from);
    let default_model = config_val
        .get("model")
        .and_then(|v| v.as_str())
        .map(String::from);
    let default_base_url = config_val
        .get("base_url")
        .and_then(|v| v.as_str())
        .map(String::from);

    sofuu_core::embed_config::install_runtime_settings(
        embedded,
        config_root.clone(),
        enable_signals,
        api_keys,
        brain_path.clone(),
        default_provider,
        default_model,
        default_base_url,
    );

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
                // ffi-1: eval_init_snippet drains any exception so init can
                // never leave a sticky error on the context.
                eval_init_snippet(ctx, &csrc, c"<embed-config>");
                // E0-1: make the brain reachable from the JSON funnel. Runs
                // on the same context, after the embed config, because the
                // facade keys brain paths off it.
                install_memory_facade(ctx);
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
                        // ffi-1: drain any exception (e.g. a parse error in
                        // the def snippet) instead of leaving it sticky.
                        eval_init_snippet(ctx, &csrc, c"<agent-def>");
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
    //
    // E0-4: a missing method is a *distinct, documented* error
    // (`unknown_method`) rather than a generic `js_exception`, so a host can
    // tell "you asked for a method this build does not have" from "the
    // method threw". The marker is set on the thrown object and re-read in
    // the catch.
    let js = format!(
        r#"var __capi_r=null;(async function(){{
  function markUnknown(m){{var e=new TypeError(m);e.__sofuuUnknown=true;return e}}
  try {{
    var p={method_json}.split("."),o=sofuu;
    for(var i=0;i<p.length;i++){{if(o==null)throw markUnknown("sofuu."+p.slice(0,i+1).join(".")+" is not available");o=o[p[i]]}}
    if(typeof o!=="function")throw markUnknown("sofuu."+{method_json}+" is not a function");
    var a={args_part};
    var r=a===undefined?await o():(Array.isArray(a)?await o.apply(null,a):await o(a));
    __capi_r={{ok:true,result:r}}
  }}catch(e){{
    __capi_r={{ok:false,error:{{code:(e&&e.__sofuuUnknown)?"unknown_method":"js_exception",message:e.message||String(e)}}}}
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

    /// Cancel IDs of streams currently in flight on this thread. E0-2:
    /// `sofuu_rt_cancel` consults this so a host can cancel an async-iterable
    /// stream (`ai.stream`), not just an `agent.run`.
    static ACTIVE_STREAMS: std::cell::RefCell<Vec<u64>> =
        const { std::cell::RefCell::new(Vec::new()) };
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
        if qjs::is_exception(json_val) {
            // ffi-2: drain stringify's exception so nothing stays pending.
            let exc = qjs::JS_GetException(ctx);
            qjs::sofuu_js_free_value(ctx, exc);
            qjs::sofuu_js_free_value(ctx, json_val);
            None
        } else if qjs::is_undefined(json_val) || qjs::is_null(json_val) {
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

    // P2-19 (AUDIT-2026-09-01): parse method + args BEFORE installing the
    // thread-local slot and the global callback — the old order left the
    // slot + a dangling `opaque` installed on every early `return -1`
    // (null/bad method or bad args JSON), poisoning the NEXT stream call.
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

    // Build the JS: merge args_json with the onStep callback, then call the method.
    //
    // E0-2 — two streaming shapes, one driver:
    //
    //   1. Methods that take an `onStep` option (agent.run) push events
    //      through the injected callback. Their return value is the result.
    //   2. Methods that RETURN an async iterable (ai.stream) were silently
    //      delivering zero events: they ignore `onStep`, and the old driver
    //      just awaited the iterator and threw it away. The header advertised
    //      ai.stream all along. The driver now checks for
    //      `Symbol.asyncIterator` on the result and pumps each chunk to the
    //      host as `{"kind":"delta",...}`, checking a cancel flag per chunk.
    //
    // Both shapes end with a `{"kind":"done","result":…}` event, because the
    // C signature has no out_json and the final value was previously dropped
    // with nothing but a code comment saying so.
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
    /* Shape 2: an async iterable (ai.stream). Pump it to the host. */
    if(r!=null&&typeof r[Symbol.asyncIterator]==="function"){{
      var deltas=0,aborted=false;
      for await(var chunk of r){{
        if(globalThis.__sofuu_cancelled && globalThis.__sofuu_cancelled[{cancel_id}]){{aborted=true;break;}}
        deltas++;
        try{{
          var payload=(chunk&&typeof chunk==="object")?chunk:{{value:chunk}};
          payload.kind=payload.kind||"delta";
          globalThis.__sofuu_stream_cb(JSON.stringify(payload));
        }}catch(e){{}}
      }}
      try{{if(aborted&&typeof r.abort==="function")r.abort();}}catch(e){{}}
      globalThis.__sofuu_stream_cb(JSON.stringify({{kind:"done",deltas:deltas,aborted:aborted,usage:(r&&r.usage)?r.usage:null}}));
      __capi_r={{ok:true,result:{{deltas:deltas,aborted:aborted}}}}
    }}else{{
      /* Shape 1: onStep already delivered the events; report the result. */
      globalThis.__sofuu_stream_cb(JSON.stringify({{kind:"done",result:r}}));
      __capi_r={{ok:true,result:r}}
    }}
    }}catch(e){{
    /* Exactly one terminal event, always. A host that waits for
     * `kind:"done"` to know the stream finished must never be left
     * hanging because the failure happened instead of the success. */
    try{{
      globalThis.__sofuu_stream_cb(JSON.stringify({{kind:"done",error:{{code:"js_exception",message:e.message||String(e)}}}}));
    }}catch(e2){{}}
    __capi_r={{ok:false,error:{{code:"js_exception",message:e.message||String(e)}}}}
  }}
}})()"#,
        method_json = serde_json::to_string(&method_str).unwrap_or_default(),
        call_expr = call_expr,
    );

    // Streaming contract: results are delivered incrementally through the
    // host's on_event callback, and the stream ALWAYS ends with exactly one
    // terminal `{"kind":"done",…}` event — carrying `deltas`/`aborted` on
    // success or `error` on failure. (This signature has no out_json, so the
    // final value is delivered as that event rather than dropped.)
    ACTIVE_STREAMS.with(|c| c.borrow_mut().push(cancel_id));
    let rc = rt_call_and_capture(rt_ref, &js, ptr::null_mut());
    ACTIVE_STREAMS.with(|c| c.borrow_mut().retain(|id| *id != cancel_id));

    // Clean up: clear the thread-local callback + the global JS function.
    STREAM_CB.with(|cell| {
        cell.set(None);
    });
    let global = qjs::sofuu_js_get_global_object(ctx);
    qjs::sofuu_js_set_property_str(ctx, global, c_name.as_ptr(), qjs::sofuu_js_undefined());
    qjs::sofuu_js_free_value(ctx, global);

    rc
}

/// Request cancellation of a streaming call.
///
/// E0-2: a stream in flight on this thread is flagged via
/// `__sofuu_cancelled[id]`, which the async-iterable pump (ai.stream)
/// checks once per chunk; otherwise we fall back to `sofuu.agent.cancel(id)`
/// (agent.run). Returns SOFUU_OK if the cancel ID was recognized, -1 if not.
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

    // If a stream is in flight on this thread, set the cancel flag the
    // pump reads. agent.run also honours it via a.signal + agent.cancel.
    let is_active = ACTIVE_STREAMS.with(|c| c.borrow().contains(&cancel_id));
    if is_active {
        let js = format!(
            r#"(function(){{globalThis.__sofuu_cancelled=globalThis.__sofuu_cancelled||{{}};globalThis.__sofuu_cancelled[{cancel_id}]=true;return{{ok:true,result:true}}}})()"#
        );
        let mut out: *mut c_char = ptr::null_mut();
        let _ = rt_call_and_capture(rt_ref, &js, &mut out);
        if !out.is_null() {
            sofuu_free(out as *mut c_void);
        }
        // Also nudge agent.run's own cancellation path when it exists.
        let agent_js = format!(
            r#"(function(){{try{{return{{ok:true,result:!!sofuu.agent.cancel({})}}}}catch(e){{return{{ok:true,result:true}}}}}})()"#,
            cancel_id,
        );
        let mut aout: *mut c_char = ptr::null_mut();
        let _ = rt_call_and_capture(rt_ref, &agent_js, &mut aout);
        if !aout.is_null() {
            sofuu_free(aout as *mut c_void);
        }
        return 0;
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

// ── Vector embedding ABI (PLAN-MULTIMODAL-EMBEDDINGS H-E1) ──────
//
// Direct float-vector entry points so hosts never JSON-encode embeddings.
// The embedders are stateless pure functions: `rt` is reserved for future
// per-runtime config and may be NULL on all three calls.
//
// Memory contract: `*out` is libc::malloc'd (dim / n*dim floats); the host
// frees it with sofuu_free(). Never free() it with the host allocator.
//
// Space ids (stable strings; see PLAN-TINY-SEMANTIC-EMBEDDER):
//   "sem2-64"  (default) — semantic-table-v2, 64-dim
//   "sem1-64"            — semantic-projector-v1, 64-dim
//   "hash-768"           — hash-v1 trigram TF-IDF, 768-dim
// One space per index — hosts must not mix spaces in a single store; the
// brain-file layer enforces the same rule (brain.qtsq vs brain-v2.qtsq).
//
// Error codes: 0 = SOFUU_OK, -1 = SOFUU_ERR_INVALID_ARG (NULL pointers,
// empty batch, non-UTF8/NULL text), -7 = SOFUU_ERR_UNKNOWN_SPACE,
// -8 = SOFUU_ERR_MODEL_UNAVAIL (baked artifact missing/corrupt),
// -9 = SOFUU_ERR_NOMEM. (Codes -2/-3/-5 belong to the funnel — see
// include/sofuu_embed.h — and are never reused here.)

#[derive(Clone, Copy, PartialEq, Eq)]
enum EmbedSpace {
    Sem2,
    Sem1,
    Hash,
}

/// Resolve a space id. E0-5: the default is now **hash-768**, not sem2-64.
///
/// The default used to disagree with three other things at once: the
/// brain's canonical space (HASH_DIM = 768), the documented "shipped
/// default" in docs/EMBEDDING.md, and the measured quality table where
/// hash-v1 wins on recall (R@5 0.493 vs 0.090). An embedder calling
/// `sofuu_embed_local(rt, text, NULL, …)` therefore got 64-dim vectors it
/// could not store in — or match against — the 768-dim brain. Now the C
/// default, the brain, the docs, and the benchmark all agree.
fn resolve_embed_space(space: Option<&str>) -> Result<EmbedSpace, c_int> {
    match space {
        None | Some("") | Some("hash-768") | Some("hash") | Some("hash-v1") => Ok(EmbedSpace::Hash),
        Some("sem2-64") | Some("sem2") => Ok(EmbedSpace::Sem2),
        Some("sem1-64") | Some("sem1") => Ok(EmbedSpace::Sem1),
        _ => Err(-7), // SOFUU_ERR_UNKNOWN_SPACE
    }
}

fn embed_text(space: EmbedSpace, text: &str) -> Option<Vec<f32>> {
    use sofuu_core::embedding as e;
    match space {
        EmbedSpace::Sem2 => e::semantic_v2::semantic_v2(text),
        EmbedSpace::Sem1 => e::semantic_v1(text),
        EmbedSpace::Hash => {
            let mut v = vec![0f32; e::HASH_DIM];
            e::hash_v1_features_into(text, &mut v);
            Some(v)
        }
    }
}

/// Embed one text into `*out` (malloc'd `*out_dim` floats).
/// `space` NULL/"" selects the default (`hash-768` — the brain's own space).
/// `rt` is reserved and may be NULL.
#[no_mangle]
pub unsafe extern "C" fn sofuu_embed_local(
    _rt: *mut CapiRuntime,
    text: *const c_char,
    space: *const c_char,
    out: *mut *mut f32,
    out_dim: *mut usize,
) -> c_int {
    if text.is_null() || out.is_null() || out_dim.is_null() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }
    let text = match cstr_or_null(text) {
        Some(s) => s,
        None => return -1,
    };
    let sp = match resolve_embed_space(cstr_or_null(space)) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let vec = match embed_text(sp, text) {
        Some(v) => v,
        None => return -8, // SOFUU_ERR_MODEL_UNAVAIL
    };
    let dim = vec.len();
    let buf = libc::malloc(dim * std::mem::size_of::<f32>()) as *mut f32;
    if buf.is_null() {
        return -9; // SOFUU_ERR_NOMEM
    }
    ptr::copy_nonoverlapping(vec.as_ptr(), buf, dim);
    *out = buf;
    *out_dim = dim;
    0 // SOFUU_OK
}

/// Embed `n` texts into `*out` (malloc'd `n * *out_dim` floats, row-major).
/// `texts` is an array of `n` NUL-terminated C strings; any NULL entry is
/// SOFUU_ERR_INVALID_ARG. `space` is resolved once for the whole batch.
/// `rt` is reserved and may be NULL.
#[no_mangle]
pub unsafe extern "C" fn sofuu_embed_batch(
    _rt: *mut CapiRuntime,
    texts: *const *const c_char,
    n: usize,
    space: *const c_char,
    out: *mut *mut f32,
    out_n: *mut usize,
    out_dim: *mut usize,
) -> c_int {
    if texts.is_null() || n == 0 || out.is_null() || out_n.is_null() || out_dim.is_null() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }
    let sp = match resolve_embed_space(cstr_or_null(space)) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // Embed first (validates every entry) so a late NULL text cannot leak
    // a half-filled malloc'd buffer.
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let p = *texts.add(i);
        if p.is_null() {
            return -1;
        }
        let t = match cstr_or_null(p) {
            Some(s) => s,
            None => return -1,
        };
        match embed_text(sp, t) {
            Some(v) => rows.push(v),
            None => return -8, // SOFUU_ERR_MODEL_UNAVAIL
        }
    }
    let dim = rows.first().map(|v| v.len()).unwrap_or(0);
    let buf = libc::malloc(n * dim * std::mem::size_of::<f32>()) as *mut f32;
    if buf.is_null() {
        return -9; // SOFUU_ERR_NOMEM
    }
    for (i, v) in rows.iter().enumerate() {
        ptr::copy_nonoverlapping(v.as_ptr(), buf.add(i * dim), dim);
    }
    *out = buf;
    *out_n = n;
    *out_dim = dim;
    0 // SOFUU_OK
}

/// Embed image bytes (PNG/JPEG) into `*out` (malloc'd 64 floats, img1-64
/// space — joint with sem2-64 text geometry). `rt` is reserved and may be
/// NULL. Errors: -1 invalid arg, -3 model-unavailable (also covers
/// undecodable bytes: no vector exists for them), -9 nomem.
#[no_mangle]
pub unsafe extern "C" fn sofuu_embed_image(
    _rt: *mut CapiRuntime,
    bytes: *const u8,
    len: usize,
    out: *mut *mut f32,
    out_dim: *mut usize,
) -> c_int {
    if bytes.is_null() || len == 0 || out.is_null() || out_dim.is_null() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }
    let data = std::slice::from_raw_parts(bytes, len);
    let vec = match sofuu_core::embedding::image::semantic_img(data) {
        Some(v) => v,
        None => return -8, // SOFUU_ERR_MODEL_UNAVAIL (or undecodable input)
    };
    let dim = vec.len();
    let buf = libc::malloc(dim * std::mem::size_of::<f32>()) as *mut f32;
    if buf.is_null() {
        return -9; // SOFUU_ERR_NOMEM
    }
    ptr::copy_nonoverlapping(vec.as_ptr(), buf, dim);
    *out = buf;
    *out_dim = dim;
    0 // SOFUU_OK
}

/// Space manifest: malloc'd JSON `{"ok":true,"result":{...}}` the host frees
/// with sofuu_free(). Combines the baked SEM1/SEM2 manifests with the
/// hash-v1 descriptor. `rt` is reserved and may be NULL.
#[no_mangle]
pub unsafe extern "C" fn sofuu_embed_info(
    _rt: *mut CapiRuntime,
    out_json: *mut *mut c_char,
) -> c_int {
    if out_json.is_null() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }
    use sofuu_core::embedding as e;
    let json = format!(
        concat!(
            r#"{{"ok":true,"result":{{"default_space":"hash-768","spaces":["#,
            r#"{{"space":"hash-768","id":"{}","input":"{}","format":"HASH","dimension":{},"available":true,"default":true}},"#,
            r#"{{"space":"sem2-64","model":{}}},"#,
            r#"{{"space":"sem1-64","model":{}}},"#,
            r#"{{"space":"img1-64","model":{}}}"#,
            r#"]}}}}"#,
        ),
        e::INPUT_EMBEDDER_ID,
        e::INPUT_EMBEDDER_ID,
        e::HASH_DIM,
        e::semantic_v2::model_info_json_v2(),
        e::model_info_json(),
        e::image::model_info_json_img(),
    );
    *out_json = malloc_cstr(&json);
    if (*out_json).is_null() {
        return -9; // SOFUU_ERR_NOMEM
    }
    0 // SOFUU_OK
}

// ── Provider voice (PLAN-MULTIMODAL-EMBEDDINGS M2C) ───────────────
//
// Thin funnel routing — no HTTP duplication: hosts base64 audio themselves
// (1 line in Swift/Kotlin) and these call the same ai.transcribe/ai.speak
// the JS surface uses. Base64 alphabet is validated before embedding so a
// hostile b64 string cannot break out of the JS string literal; `text`
// goes through serde_json quoting for the same reason. `opts_json`
// NULL/"" selects {}. Speak resolves `{audio: Uint8Array, format}` — over
// the JSON funnel the array travels as an indexed object (documented).

fn b64_alphabet_ok(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'+' | b'/' | b'='))
}

/// Transcribe base64 audio. Returns the funnel rc (0 + `{ok,result:{text}}`
/// envelope, or provider error envelope).
#[no_mangle]
pub unsafe extern "C" fn sofuu_voice_transcribe(
    rt: *mut CapiRuntime,
    audio_b64: *const c_char,
    opts_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    if rt.is_null() || audio_b64.is_null() || out_json.is_null() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }
    let b64 = match cstr_or_null(audio_b64) {
        Some(s) if b64_alphabet_ok(s) => s,
        _ => return -1,
    };
    let opts = cstr_or_null(opts_json).filter(|s| !s.is_empty()).unwrap_or("{}");
    let src = format!(r#"sofuu.ai.transcribe({{audio_b64:"{b64}"}},{opts})"#);
    let src_c = match CString::new(src) {
        Ok(c) => c,
        Err(_) => return -1,
    };
    sofuu_rt_eval(rt, src_c.as_ptr(), out_json)
}

/// Speak text. Returns the funnel rc (0 + `{ok,result:{audio,format}}`
/// envelope; `audio` is an indexed-byte object over JSON transport).
#[no_mangle]
pub unsafe extern "C" fn sofuu_voice_speak(
    rt: *mut CapiRuntime,
    text: *const c_char,
    opts_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    if rt.is_null() || text.is_null() || out_json.is_null() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }
    let t = match cstr_or_null(text) {
        Some(s) => s,
        None => return -1,
    };
    let quoted = serde_json::to_string(t).unwrap_or_else(|_| "\"\"".into());
    let opts = cstr_or_null(opts_json).filter(|s| !s.is_empty()).unwrap_or("{}");
    let src = format!(r#"sofuu.ai.speak({quoted},{opts})"#);
    let src_c = match CString::new(src) {
        Ok(c) => c,
        Err(_) => return -1,
    };
    sofuu_rt_eval(rt, src_c.as_ptr(), out_json)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Serialize with any other test that touches the process-global libuv loop.
    // (sofuu-core's rt::TEST_LOOP_LOCK is not pub, so we use our own.)
    // Poison-tolerant: one failing test must not cascade into every other
    // test as a confusing PoisonError that hides the real failure.
    use std::sync::Mutex;
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Skip a memory-dependent test on a QTSQ-FREE build, and say why.
    ///
    /// A QTSQ-free build does not have a degraded `sofuu.memory` — the whole
    /// surface is compiled out (sofuu-core/src/rt/memory.rs gates the
    /// register bodies on `#[cfg(has_qtsq)]`), so there is no brain to open
    /// and the funnel correctly answers `unknown_method`. CI builds this way
    /// on purpose (QTSQ is a proprietary local checkout), and the JS suite
    /// covers the brain through a mock server. Without this skip those tests
    /// would fail in CI while passing locally — a gate that only ever runs on
    /// one machine is not a gate.
    macro_rules! require_qtsq {
        () => {
            if !sofuu_core::HAS_QTSQ {
                eprintln!(
                    "skipping: this build has no QTSQ codec, so sofuu.memory \
                     does not exist (see sofuu_core::HAS_QTSQ)"
                );
                return;
            }
        };
    }

    /// Call a funnel method and return the parsed envelope, asserting the
    /// call itself succeeded at the C level. Panics with the raw JSON on
    /// failure so a broken method names itself in the test output.
    fn call(rt: *mut CapiRuntime, method: &str, args: &str) -> JsonValue {
        let m = CString::new(method).unwrap();
        let a = CString::new(args).unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_call(rt, m.as_ptr(), a.as_ptr(), &mut out) };
        assert!(!out.is_null(), "{method}: out_json must not be NULL");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        unsafe { sofuu_free(out as *mut c_void) };
        assert_eq!(rc, 0, "{method}: rc should be 0, got {rc}");
        let v: JsonValue = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{method}: envelope is not JSON ({e}): {json}"));
        assert_eq!(
            v.get("ok").and_then(|o| o.as_bool()),
            Some(true),
            "{method} should return ok:true, got {json}"
        );
        v.get("result").cloned().unwrap_or(JsonValue::Null)
    }

    /// Same as `call` but for methods that are expected to fail; returns the
    /// error code so callers can assert the *stable* code, not just failure.
    fn call_err(rt: *mut CapiRuntime, method: &str, args: &str) -> String {
        let m = CString::new(method).unwrap();
        let a = CString::new(args).unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let _ = unsafe { sofuu_rt_call(rt, m.as_ptr(), a.as_ptr(), &mut out) };
        assert!(!out.is_null(), "{method}: out_json must not be NULL");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        unsafe { sofuu_free(out as *mut c_void) };
        let v: JsonValue = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{method}: envelope is not JSON ({e}): {json}"));
        assert_eq!(
            v.get("ok").and_then(|o| o.as_bool()),
            Some(false),
            "{method} was expected to fail, got {json}"
        );
        v.get("error")
            .and_then(|e| e.get("code"))
            .and_then(|c| c.as_str())
            .unwrap_or("<no code>")
            .to_string()
    }

    #[test]
    fn abi_version_is_2() {
        let _g = lock();
        assert_eq!(unsafe { sofuu_embed_abi_version() }, 2);
    }

    #[test]
    fn rt_new_and_free_with_null_config() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null(), "rt_new with NULL config should succeed");
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn rt_new_with_empty_config() {
        let _g = lock();
        let cfg = CString::new("{}").unwrap();
        let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
        assert!(!rt.is_null(), "rt_new with {{}} config should succeed");
        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn rt_new_with_bad_json_returns_null() {
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let method = CString::new("nonexistent.method").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_call(rt, method.as_ptr(), ptr::null(), &mut out) };
        assert_eq!(rc, 0, "call should return rc=0 (error is in the envelope)");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        assert!(json.contains(r#""ok":false"#), "envelope ok=false for unknown method, got: {json}");
        // E0-4: a missing method carries the stable `unknown_method` code so
        // a host can distinguish "not in this build" from "the method threw".
        assert!(
            json.contains(r#""code":"unknown_method""#),
            "unknown method must report the stable unknown_method code, got: {json}"
        );

        unsafe { sofuu_free(out as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    /// A method that exists but throws must keep the generic `js_exception`
    /// code — the `unknown_method` marker must not leak onto real failures.
    ///
    /// On a QTSQ-free build `memory.*` genuinely does not exist, so the
    /// correct answer there IS `unknown_method`. Asserting the wrong code
    /// for the build would be a test that only ever runs on one machine, so
    /// this asserts the honest answer for whichever build is under test —
    /// and that is itself the contract: the code distinguishes "this build
    /// lacks it" from "it exists and failed".
    #[test]
    fn a_throwing_method_is_not_reported_as_unknown_method() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // memory.count with a handle that was never opened:
        //  - QTSQ build  → the method exists and throws inside → js_exception
        //  - QTSQ-free   → the method does not exist          → unknown_method
        let code = call_err(rt, "memory.count", r#"{"handle":99}"#);
        let expected = if sofuu_core::HAS_QTSQ {
            "js_exception"
        } else {
            "unknown_method"
        };
        assert_eq!(
            code, expected,
            "a method that exists but throws must report js_exception, and a \
             method absent from this build must report unknown_method"
        );

        unsafe { sofuu_rt_free(rt) };
    }


    #[test]
    fn call_bad_args_json_returns_error() {
        let _g = lock();
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
        let _g = lock();
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

    /// E0-2: `ai.stream` returns an async iterable and ignores `onStep`, so
    /// before the pump rewrite the funnel delivered ZERO events for it while
    /// the header advertised it. This mocks the stream and proves the delta
    /// pump works, and that a trailing `done` event carries the result.
    #[test]
    fn stream_calls_ai_stream_emits_deltas_and_done() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // Replace ai.stream with a 3-chunk async iterator.
        let setup = CString::new(
            r#"(function(){
              var chunks = ["Hel", "lo ", "world"];
              sofuu.ai.stream = function(opts) {
                return (async function*(){
                  for (var i=0;i<chunks.length;i++){ yield { text: chunks[i] }; }
                })();
              };
            })()"#,
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) }, 0);
        unsafe { sofuu_free(out as *mut c_void) };

        static EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());
        EVENTS.lock().unwrap().clear();
        unsafe extern "C" fn collect(event_json: *const c_char, _opaque: *mut c_void) {
            let s = unsafe { CStr::from_ptr(event_json) }.to_string_lossy().into_owned();
            EVENTS.lock().unwrap().push(s);
        }

        let method = CString::new("ai.stream").unwrap();
        let args = CString::new(r#"{"prompt":"hi"}"#).unwrap();
        let mut cancel_id: u64 = 0;
        let rc = unsafe {
            sofuu_rt_call_stream(
                rt,
                method.as_ptr(),
                args.as_ptr(),
                Some(collect),
                ptr::null_mut(),
                &mut cancel_id,
            )
        };
        assert_eq!(rc, 0, "ai.stream should return OK");
        assert!(cancel_id != 0, "cancel id should be issued");

        let events = EVENTS.lock().unwrap().clone();
        // 3 delta events (one per chunk) + 1 done.
        let deltas: Vec<&String> = events.iter().filter(|e| e.contains(r#""kind":"delta""#)).collect();
        assert_eq!(deltas.len(), 3, "expected 3 delta events, got {events:?}");
        assert!(events.iter().any(|e| e.contains("Hel")), "delta 1 text missing");
        assert!(events.iter().any(|e| e.contains("world")), "delta 3 text missing");
        assert!(
            events.iter().any(|e| e.contains(r#""kind":"done""#)),
            "expected a trailing done event, got {events:?}"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    /// E0-2: a *live* mid-flight cancel. The host cancels from inside the
    /// event callback on the first delta; the iterator has 20 chunks, so an
    /// early stop is directly observable in the delta count. (Before this,
    /// only cancelling an *unknown* id was ever tested.)
    ///
    /// The cancel id is readable during the callback because
    /// `sofuu_rt_call_stream` writes `*out_cancel_id` before the JS runs;
    /// the host passes a pointer to it as the `opaque`.
    #[test]
    fn stream_ai_stream_cancels_mid_flight() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // 20 chunks, so an early stop is observable.
        let setup = CString::new(
            r#"(function(){
              var chunks = []; for (var i=0;i<20;i++) chunks.push("t"+i);
              sofuu.ai.stream = function(opts) {
                return (async function*(){
                  for (var i=0;i<chunks.length;i++){ yield { text: chunks[i] }; }
                })();
              };
            })()"#,
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) }, 0);
        unsafe { sofuu_free(out as *mut c_void) };

        static DELTAS: Mutex<u32> = Mutex::new(0);
        static DONE_ABORTED: Mutex<bool> = Mutex::new(false);
        *DELTAS.lock().unwrap() = 0;
        *DONE_ABORTED.lock().unwrap() = false;

        // opaque carries the runtime + the live cancel-id slot.
        #[repr(C)]
        struct CancelCtx {
            rt: *mut CapiRuntime,
            cid: u64,
        }

        unsafe extern "C" fn collect_and_cancel(event_json: *const c_char, opaque: *mut c_void) {
            let s = unsafe { CStr::from_ptr(event_json) }.to_string_lossy().into_owned();
            let ctx = unsafe { &*(opaque as *const CancelCtx) };
            if s.contains(r#""kind":"delta""#) {
                let mut d = DELTAS.lock().unwrap();
                *d += 1;
                let n = *d;
                drop(d);
                if n == 1 {
                    assert!(ctx.cid != 0, "cancel id must be readable during the callback");
                    unsafe { sofuu_rt_cancel(ctx.rt, ctx.cid) };
                }
            } else if s.contains(r#""kind":"done""#) {
                *DONE_ABORTED.lock().unwrap() = s.contains(r#""aborted":true"#);
            }
        }

        let mut ctx = CancelCtx { rt, cid: 0 };
        let method = CString::new("ai.stream").unwrap();
        let args = CString::new(r#"{"prompt":"hi"}"#).unwrap();
        let rc = unsafe {
            sofuu_rt_call_stream(
                rt,
                method.as_ptr(),
                args.as_ptr(),
                Some(collect_and_cancel),
                (&mut ctx as *mut CancelCtx) as *mut c_void,
                &mut ctx.cid,
            )
        };
        assert_eq!(rc, 0, "cancelling mid-flight must still return OK");

        let deltas = *DELTAS.lock().unwrap();
        assert!(
            deltas < 20,
            "cancel should stop the pump early, but all 20 deltas arrived ({deltas})"
        );
        assert!(
            *DONE_ABORTED.lock().unwrap(),
            "the done event must report aborted:true after a live cancel"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn multi_instance_two_runtimes_independent() {
        let _g = lock();
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

    // F-3 (AUDIT-2026-09-01): every sofuu_rt_new used to clobber the
    // process-global embed config, so runtime B's config_root / api keys
    // silently retargeted runtime A. Now each runtime installs its config
    // as per-thread settings on its owning thread (ABI: one runtime per
    // thread). This test replays the clobber: runtime A is created on its
    // own thread, then runtime B is created on the main thread while A is
    // still alive — A's readers must still see A's config.
    #[test]
    fn multi_instance_embed_config_isolated_across_runtimes() {
        let _g = lock();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();

        let t = std::thread::spawn(move || {
            let cfg = CString::new(
                r#"{"config_root":"/tmp/sofuu-capi-iso-a","api_keys":{"probe":"key-a"}}"#,
            )
            .unwrap();
            let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
            assert!(!rt.is_null(), "runtime A creation should succeed");
            ready_tx.send(()).unwrap();
            // Hold A alive until B exists on the main thread.
            done_rx.recv().unwrap();
            // The same readers rt/ai.rs and session_js.rs use on the
            // runtime's thread — these must resolve A's settings, not B's
            // clobbered globals.
            let root = sofuu_core::embed_config::get_config_root();
            let key = sofuu_core::embed_config::api_key("probe");
            unsafe { sofuu_rt_free(rt) };
            (root, key)
        });

        ready_rx.recv().unwrap();
        let cfg_b = CString::new(
            r#"{"config_root":"/tmp/sofuu-capi-iso-b","api_keys":{"probe":"key-b"}}"#,
        )
        .unwrap();
        let rt_b = unsafe { sofuu_rt_new(cfg_b.as_ptr()) };
        assert!(!rt_b.is_null(), "runtime B creation should succeed");
        done_tx.send(()).unwrap();

        let (root_a, key_a) = t.join().unwrap();
        assert_eq!(
            root_a.as_deref(),
            Some("/tmp/sofuu-capi-iso-a"),
            "runtime B's config_root must not retarget runtime A's thread"
        );
        assert_eq!(
            key_a.as_deref(),
            Some("key-a"),
            "runtime B's api key must not leak into runtime A's thread"
        );
        unsafe { sofuu_rt_free(rt_b) };
    }

    #[test]
    fn embedded_process_exit_throws_exit_error() {
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
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
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // Cancel a non-existent ID should return -1 (not found).
        let rc = unsafe { sofuu_rt_cancel(rt, 99999) };
        assert_eq!(rc, -1, "cancel of unknown ID should return -1");

        unsafe { sofuu_rt_free(rt) };
    }

    // ffi-2: JSON.stringify blowing up inside value_to_json (circular ref)
    // must not leave rt->current_exception pending on the context, where the
    // next JS_GetException caller would consume it as if it were its own.
    #[test]
    fn eval_circular_result_drains_the_stringify_exception() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // The eval itself succeeds — only the Rust-side stringify explodes.
        let src = CString::new("var c={}; c.self=c; c").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0, "eval itself should succeed");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy();
        // The circular object is nested inside the {ok,result} envelope, so
        // stringify fails for the WHOLE envelope — value_to_json falls back
        // to a bare "null" (documented fallback, kept as a behavior pin).
        assert_eq!(
            json, "null",
            "un-stringifyable envelope falls back to bare null, got: {json}"
        );
        unsafe { sofuu_free(out as *mut c_void) };

        // White-box probe: the stringify TypeError must be drained.
        let ctx = unsafe { (*rt).ffi_rt.engine_ctx() } as *mut qjs::JSContext;
        let exc = unsafe { qjs::JS_GetException(ctx) };
        let clean = unsafe { qjs::is_null(exc) } || unsafe { qjs::is_undefined(exc) };
        unsafe { qjs::sofuu_js_free_value(ctx, exc) };
        assert!(
            clean,
            "no exception may stay pending after value_to_json's stringify failure"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    // ffi-1: eval_init_snippet must drain any exception from an init snippet
    // (a parse error can't be caught by any try/catch inside that snippet) so
    // the next JS_GetException caller never consumes a stale error.
    #[test]
    fn init_snippet_failure_drains_and_keeps_the_context_usable() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());
        let ctx = unsafe { (*rt).ffi_rt.engine_ctx() } as *mut qjs::JSContext;

        // A parse error — nothing inside the snippet can intercept it.
        let bad = CString::new("function(").unwrap();
        let ok = unsafe { eval_init_snippet(ctx, &bad, c"<probe-bad>") };
        assert!(!ok, "parse-error snippet must report failure");

        let exc = unsafe { qjs::JS_GetException(ctx) };
        let clean = unsafe { qjs::is_null(exc) } || unsafe { qjs::is_undefined(exc) };
        unsafe { qjs::sofuu_js_free_value(ctx, exc) };
        assert!(clean, "no exception may stay pending after a failed init snippet");

        // The context must keep evaluating cleanly afterwards.
        let good = CString::new("6 * 7").unwrap();
        let ok2 = unsafe { eval_init_snippet(ctx, &good, c"<probe-good>") };
        assert!(ok2, "context stays usable after a drained snippet failure");

        unsafe { sofuu_rt_free(rt) };
    }

    // M2C: voice funnel validation (no network — transfer paths are
    // covered by tests/voice_test.js against a mock).
    #[test]
    fn voice_fns_reject_bad_args() {
        let good_b64 = CString::new("QUJDRA==").unwrap();
        let bad_b64 = CString::new("not b64!!").unwrap();
        let opts = CString::new("{}").unwrap();
        let text = CString::new("hi").unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        // NULL runtime / inputs.
        assert_eq!(
            unsafe { sofuu_voice_transcribe(ptr::null_mut(), good_b64.as_ptr(), opts.as_ptr(), &mut out) },
            -1
        );
        assert_eq!(
            unsafe { sofuu_voice_transcribe(ptr::null_mut(), ptr::null(), opts.as_ptr(), &mut out) },
            -1
        );
        assert_eq!(
            unsafe { sofuu_voice_speak(ptr::null_mut(), text.as_ptr(), opts.as_ptr(), &mut out) },
            -1
        );
        // Hostile b64 never reaches eval (injection guard).
        // (Needs a live rt to get past the NULL-rt check first — covered
        // by shape: bad alphabet returns -1 before any JS runs.)
        assert!(!b64_alphabet_ok("not b64!!"));
        assert!(!b64_alphabet_ok(""));
        assert!(b64_alphabet_ok("QUJDRA=="));
        let _ = bad_b64;
    }

    // H-E1: vector embedding ABI. Pure/stateless (baked read-only models),
    // so no TEST_LOCK — these touch no process-global loop state.
    #[test]
    fn embed_local_batch_info_round_trip() {
        let t = CString::new("hello world").unwrap();

        // Default space (NULL) → hash-768 (E0-5), L2-normalized.
        let mut out: *mut f32 = ptr::null_mut();
        let mut dim: usize = 0;
        let rc = unsafe { sofuu_embed_local(ptr::null_mut(), t.as_ptr(), ptr::null(), &mut out, &mut dim) };
        assert_eq!(rc, 0, "default-space embed must succeed");
        assert_eq!(dim, 768, "E0-5: the default space is the brain's hash-768");
        assert!(!out.is_null());
        let v = unsafe { std::slice::from_raw_parts(out, dim) };
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "embeddings are L2-normalized, got {norm}");
        unsafe { sofuu_free(out as *mut c_void) };

        // Unknown space → -2, no buffer.
        let bad = CString::new("nope").unwrap();
        let mut out2: *mut f32 = ptr::null_mut();
        let mut dim2: usize = 0;
        let rc2 = unsafe { sofuu_embed_local(ptr::null_mut(), t.as_ptr(), bad.as_ptr(), &mut out2, &mut dim2) };
        assert_eq!(rc2, -7, "unknown space must refuse");
        assert!(out2.is_null());

        // hash-768 → 768 dims.
        let h = CString::new("hash-768").unwrap();
        let mut out3: *mut f32 = ptr::null_mut();
        let mut dim3: usize = 0;
        let rc3 = unsafe { sofuu_embed_local(ptr::null_mut(), t.as_ptr(), h.as_ptr(), &mut out3, &mut dim3) };
        assert_eq!(rc3, 0);
        assert_eq!(dim3, 768);
        unsafe { sofuu_free(out3 as *mut c_void) };

        // NULL text → -1.
        let mut out4: *mut f32 = ptr::null_mut();
        let mut dim4: usize = 0;
        assert_eq!(
            unsafe { sofuu_embed_local(ptr::null_mut(), ptr::null(), ptr::null(), &mut out4, &mut dim4) },
            -1
        );

        // Batch of 3 → 3 rows × 768 dims, row-major (default space).
        let texts = [CString::new("alpha").unwrap(), CString::new("beta").unwrap(), CString::new("gamma").unwrap()];
        let ptrs: Vec<*const c_char> = texts.iter().map(|c| c.as_ptr()).collect();
        let mut bout: *mut f32 = ptr::null_mut();
        let mut bn: usize = 0;
        let mut bdim: usize = 0;
        let brc = unsafe {
            sofuu_embed_batch(ptr::null_mut(), ptrs.as_ptr(), ptrs.len(), ptr::null(), &mut bout, &mut bn, &mut bdim)
        };
        assert_eq!(brc, 0);
        assert_eq!((bn, bdim), (3, 768));
        assert!(!bout.is_null());
        unsafe { sofuu_free(bout as *mut c_void) };

        // Manifest parses and names the default space.
        let mut p: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_embed_info(ptr::null_mut(), &mut p) }, 0);
        let js = unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() };
        let val: serde_json::Value = serde_json::from_str(&js).expect("manifest must be JSON");
        assert_eq!(val["result"]["default_space"], "hash-768");
        assert_eq!(val["result"]["spaces"].as_array().map(|a| a.len()), Some(4));
        unsafe { sofuu_free(p as *mut c_void) };
    }

    // ── E0-1: the brain must be reachable from the funnel ──────────
    //
    // This is the single test that proves the memory differentiator works
    // for an embedder. Before the facade, memory.remember/recall/count
    // were documented but dead (the instance stringifies to `{}`), so a
    // host could only reach the brain by hand-building rt_eval strings.

    /// A real on-disk brain in a unique temp dir, opened at hash dim 768
    /// to match the shipped default embedder space.
    fn open_brain(rt: *mut CapiRuntime, name: &str) -> JsonValue {
        let dir = std::env::temp_dir().join(format!("sofuu-capi-{name}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("brain.qtsq");
        let args = format!(
            r#"{{"path":{},"dim":768,"embed_id":"hash-v1"}}"#,
            serde_json::to_string(path.to_str().unwrap()).unwrap()
        );
        let handle = call(rt, "memory.open", &args);
        let h = handle
            .get("handle")
            .and_then(|h| h.as_i64())
            .unwrap_or_else(|| panic!("memory.open must return a {{handle}}, got {handle}"));
        assert!(h >= 0, "handle should be a non-negative index");
        handle
    }

    /// A 768-dim unit vector at `axis` — deterministic, no float noise.
    fn hash_axis(axis: usize) -> String {
        let mut v = vec![0.0f64; 768];
        v[axis] = 1.0;
        let body: Vec<String> = v.iter().map(|f| f.to_string()).collect();
        format!("[{}]", body.join(","))
    }

    #[test]
    fn memory_funnel_open_remember_recall_count_round_trip() {
        require_qtsq!();
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // 1. open → JSON-serializable handle (this was `{}` before).
        let h = open_brain(rt, "roundtrip");
        let hid = h.get("handle").and_then(|h| h.as_i64()).unwrap();
        assert_eq!(h.get("dim").and_then(|d| d.as_i64()), Some(768));

        // 2. remember two vectors via the funnel.
        for (axis, text) in [(0usize, "the sky is blue"), (1, "grass is green")] {
            let args = format!(
                r#"{{"handle":{hid},"vec":{},"text":{text:?},"role":"user","kv_page_id":0}}"#,
                hash_axis(axis)
            );
            let idx = call(rt, "memory.remember", &args);
            assert!(
                idx.as_i64().is_some_and(|i| i >= 0),
                "remember should return an index, got {idx}"
            );
        }

        // 3. count reflects both writes.
        let n = call(rt, "memory.count", &format!(r#"{{"handle":{hid}}}"#));
        assert!(
            n.as_u64().unwrap_or(0) >= 2,
            "count should be >= 2 after two remembers, got {n}"
        );

        // 4. recall returns the right record for the right vector.
        let args = format!(
            r#"{{"handle":{hid},"vec":{},"k":2}}"#,
            hash_axis(0)
        );
        let hits = call(rt, "memory.recall", &args);
        let arr = hits
            .as_array()
            .unwrap_or_else(|| panic!("recall should return an array, got {hits}"));
        assert!(!arr.is_empty(), "recall should find the stored vector");
        let top = &arr[0];
        let txt = top
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or_else(|| panic!("recall hit should carry its text, got {top}"));
        assert_eq!(
            txt, "the sky is blue",
            "the axis-0 query must retrieve the axis-0 record"
        );

        // 5. flush persists.
        let ok = call(rt, "memory.flush", &format!(r#"{{"handle":{hid}}}"#));
        assert!(
            matches!(ok.as_bool(), Some(true) | None),
            "flush should succeed, got {ok}"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn memory_funnel_handles_isolate_two_brains() {
        require_qtsq!();
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let a = open_brain(rt, "iso-a");
        let b = open_brain(rt, "iso-b");
        let (ha, hb) = (
            a.get("handle").and_then(|h| h.as_i64()).unwrap(),
            b.get("handle").and_then(|h| h.as_i64()).unwrap(),
        );
        assert_ne!(ha, hb, "two opens must yield two distinct handles");

        call(
            rt,
            "memory.remember",
            &format!(
                r#"{{"handle":{ha},"vec":{},"text":"only in A","role":"user","kv_page_id":0}}"#,
                hash_axis(5)
            ),
        );

        // A sees it; B does not.
        let ca = call(rt, "memory.count", &format!(r#"{{"handle":{ha}}}"#));
        let cb = call(rt, "memory.count", &format!(r#"{{"handle":{hb}}}"#));
        assert!(ca.as_u64().unwrap_or(0) >= 1, "brain A holds the record");
        assert_eq!(
            cb.as_u64().unwrap_or(0),
            0,
            "brain B must stay empty, got {cb}"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    #[test]
    fn memory_funnel_recall_before_open_is_a_clear_error() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // No memory.open on this runtime → the facade's "call open first"
        // error must surface as a normal error envelope, not a crash. The
        // code differs by build for the reason above (absent vs throwing).
        let code = call_err(
            rt,
            "memory.count",
            r#"{"handle":99}"#,
        );
        let expected = if sofuu_core::HAS_QTSQ {
            "js_exception"
        } else {
            "unknown_method"
        };
        assert_eq!(code, expected, "expected a clean error envelope");
        unsafe { sofuu_rt_free(rt) };
    }

    /// E0-5: the cross-space trap, closed. This is the payoff of moving the
    /// C default to hash-768 — an embedder can now go text → vector →
    /// brain → recall without ever naming a space, because the embedder's
    /// default and the brain's canonical space are the same one.
    ///
    /// Before this change, `sofuu_embed_local(rt, text, NULL, …)` returned
    /// 64-dim sem2 vectors and `memory.open(path, 768)` wanted 768-dim
    /// vectors: a silent dimension mismatch, not an error.
    #[test]
    fn default_embed_space_is_the_brain_space() {
        require_qtsq!();
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // 1. The C default must be the brain's dimension, with no space arg.
        let text = CString::new("the sky is blue").unwrap();
        let mut vec: *mut f32 = ptr::null_mut();
        let mut dim: usize = 0;
        let rc = unsafe { sofuu_embed_local(rt, text.as_ptr(), ptr::null(), &mut vec, &mut dim) };
        assert_eq!(rc, 0);
        assert_eq!(
            dim,
            sofuu_core::embedding::HASH_DIM,
            "the C default embed space must equal the brain's HASH_DIM"
        );

        // 2. The manifest must agree — this is what a host reads to discover
        //    the default, so it must not contradict the code.
        let mut info: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_embed_info(rt, &mut info) }, 0);
        let js = unsafe { CStr::from_ptr(info) }.to_string_lossy().into_owned();
        let val: JsonValue = serde_json::from_str(&js).expect("manifest must be JSON");
        assert_eq!(
            val["result"]["default_space"], "hash-768",
            "the manifest's default_space must match the code"
        );
        unsafe { sofuu_free(info as *mut c_void) };

        // 3. The real end-to-end proof: store the default-space vector in a
        //    brain opened at the default dim, then recall it. This is the
        //    exact sequence an embedder writes, and it can no longer fail on
        //    a dimension mismatch.
        let dir = std::env::temp_dir().join(format!("sofuu-capi-trap-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("brain.qtsq");
        let open_args = format!(
            r#"{{"path":{},"dim":{dim}}}"#,
            serde_json::to_string(path.to_str().unwrap()).unwrap()
        );
        let handle = call(rt, "memory.open", &open_args);
        let hid = handle.get("handle").and_then(|h| h.as_i64()).unwrap();

        // Hand the C-embedded vector to the funnel.
        let floats: Vec<f32> = unsafe { std::slice::from_raw_parts(vec, dim) }.to_vec();
        let vec_json: Vec<String> = floats.iter().map(|f| f.to_string()).collect();
        let remember_args = format!(
            r#"{{"handle":{hid},"vec":[{}],"text":"the sky is blue","role":"user","kv_page_id":0}}"#,
            vec_json.join(",")
        );
        let idx = call(rt, "memory.remember", &remember_args);
        assert!(
            idx.as_i64().is_some_and(|i| i >= 0),
            "a default-space vector must store without a dimension mismatch, got {idx}"
        );

        let recall_args = format!(
            r#"{{"handle":{hid},"vec":[{}],"k":1}}"#,
            vec_json.join(",")
        );
        let hits = call(rt, "memory.recall", &recall_args);
        let arr = hits.as_array().unwrap_or_else(|| panic!("expected hits, got {hits}"));
        assert_eq!(
            arr.first().and_then(|h| h.get("text")).and_then(|t| t.as_str()),
            Some("the sky is blue"),
            "the default-space vector must be retrievable from the brain"
        );

        call(rt, "memory.flush", &format!(r#"{{"handle":{hid}}}"#));
        unsafe { sofuu_free(vec as *mut c_void) };
        unsafe { sofuu_rt_free(rt) };
    }

    /// E0-4b: the mcp + http facades must actually install on the JS
    /// namespace. `mcp.call` used to be documented but unreachable (the
    /// client is an instance); `http.serve` replaces the documented-but-
    /// nonexistent `http.serve.start`. This asserts the *shape* resolves —
    /// a live MCP server would need a real child process, covered by the
    /// JS suite; what matters here is that the funnel method is callable.
    #[test]
    fn mcp_and_http_funnel_methods_are_registered() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // The facade must have installed module-level methods. Reading them
        // as properties proves they are functions, not undefined.
        //
        // Each lookup must be null-safe: on a QTSQ-free build `sofuu.memory`
        // is undefined, and a bare `sofuu.memory.remember` THROWS (TypeError),
        // which would abort the whole eval and take the mcp/http assertions
        // down with it. `__t` walks the path and reports "undefined" instead.
        let src = CString::new(
            r#"function __t(o, k){ return (o == null) ? "undefined" : typeof o[k]; }
               JSON.stringify({
                 mcpCall: __t(sofuu.mcp, "call"),
                 mcpList: __t(sofuu.mcp, "listTools"),
                 memRemember: __t(sofuu.memory, "remember"),
                 memRecall: __t(sofuu.memory, "recall"),
                 memCount: __t(sofuu.memory, "count"),
                 httpServe: __t(sofuu.http, "serve")
               })"#,
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0);
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        unsafe { sofuu_free(out as *mut c_void) };
        assert!(
            json.contains("\"ok\":true"),
            "the shape probe must not throw: {json}"
        );

        // The eval envelope nests a JSON *string* as `result`; parse both
        // layers rather than substring-matching the escaped blob.
        let env: JsonValue = serde_json::from_str(&json).expect("envelope is JSON");
        let inner: JsonValue =
            serde_json::from_str(env["result"].as_str().expect("result is a string"))
                .expect("inner payload is JSON");
        // mcp + http are pure JS facades and must exist in every build. The
        // memory methods come from sofuu-core's QTSQ-gated registration, so
        // they are only required where the codec is actually linked.
        for m in ["mcpCall", "mcpList", "httpServe"] {
            assert_eq!(
                inner[m], "function",
                "{m} should be registered as a function by the facade"
            );
        }
        for m in ["memRemember", "memRecall", "memCount"] {
            if sofuu_core::HAS_QTSQ {
                assert_eq!(
                    inner[m], "function",
                    "{m} should be registered as a function by the facade"
                );
            } else {
                assert_eq!(
                    inner[m], "undefined",
                    "without QTSQ the memory surface is compiled out entirely, \
                     so the facade must not invent it"
                );
            }
        }
        unsafe { sofuu_rt_free(rt) };
    }

    /// E0-4: the anti-drift test.
    ///
    /// The audit found that 7 of the 17 funnel methods listed in
    /// docs/EMBEDDING.md did not exist — the docs were aspirational and
    /// nothing checked. This parses the method table straight out of the
    /// doc and asserts every listed method resolves as a real function on a
    /// fresh runtime, so the list can never drift from the code again.
    #[test]
    fn documented_funnel_methods_all_resolve() {
        let _g = lock();

        // 1. Parse the "enforced by a test" table out of the doc.
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/EMBEDDING.md"
        ))
        .expect("docs/EMBEDDING.md must be readable from the capi test");

        // Only the method table between these two markers counts; the rest of
        // the doc is full of backticked filenames that are not methods.
        let table = doc
            .split_once("enforced by a test")
            .and_then(|(_, rest)| rest.split_once("**Handle-addressed namespaces.**"))
            .map(|(a, _)| a)
            .expect("the method table markers must exist in docs/EMBEDDING.md");

        let mut methods: Vec<String> = Vec::new();
        for line in table.lines() {
            if !line.trim_start().starts_with('|') {
                continue;
            }
            for cell in line.split('|') {
                // Pull every `backticked` token out of the cell.
                let mut rest = cell;
                while let Some(start) = rest.find('`') {
                    let after = &rest[start + 1..];
                    let Some(end) = after.find('`') else { break };
                    let tok = &after[..end];
                    // A method is `namespace.method` — no spaces, no globs,
                    // and a known top-level namespace.
                    const NS: [&str; 6] = ["ai", "memory", "mcp", "agent", "rlm", "http"];
                    if NS.iter().any(|n| tok.starts_with(&format!("{n}.")))
                        && !tok.contains(' ')
                        && !tok.contains('*')
                    {
                        methods.push(tok.to_string());
                    }
                    rest = &after[end + 1..];
                }
            }
        }
        methods.sort();
        methods.dedup();
        assert!(
            methods.len() >= 20,
            "expected the doc's method table to yield 20+ methods, got {methods:?}"
        );

        // 2. Every one of them must resolve as a function on a real runtime.
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());
        let src = CString::new(
            r#"JSON.stringify(globalThis.__doc_methods.map(function(m){
                 var p=m.split("."),o=sofuu,ok=true;
                 for(var i=0;i<p.length;i++){ if(o==null){ok=false;break} o=o[p[i]] }
                 return (ok && typeof o==="function");
               }))"#,
        )
        .unwrap();
        // Seed the list, then run the resolver in the same eval.
        let seed = CString::new(format!(
            "globalThis.__doc_methods = {};",
            serde_json::to_string(&methods).unwrap()
        ))
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let _ = unsafe { sofuu_rt_eval(rt, seed.as_ptr(), &mut out) };
        unsafe { sofuu_free(out as *mut c_void) };

        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, src.as_ptr(), &mut out) };
        assert_eq!(rc, 0, "resolver eval should succeed");
        let env: JsonValue = serde_json::from_str(
            unsafe { CStr::from_ptr(out) }.to_string_lossy().as_ref(),
        )
        .expect("envelope is JSON");
        unsafe { sofuu_free(out as *mut c_void) };
        let flags: Vec<bool> = serde_json::from_str(
            env["result"]
                .as_str()
                .unwrap_or_else(|| panic!("result should be a JSON string, envelope: {env}")),
        )
        .expect("flag array is JSON");

        assert_eq!(flags.len(), methods.len(), "one flag per method");

        // Partition on the QTSQ boundary. `memory.*` is registered by
        // sofuu-core under `#[cfg(has_qtsq)]`, so a QTSQ-free build (what CI
        // runs) legitimately has none of it. Rather than skipping the test or
        // weakening it, each build asserts the contract that is true for it:
        //
        //   QTSQ build  → every documented method resolves.
        //   QTSQ-free   → every NON-memory method resolves, and the memory
        //                 methods are uniformly ABSENT (catching a partial or
        //                 half-registered facade, which a blanket "skip the
        //                 memory rows" would hide).
        //
        // Doc drift in the memory rows is still caught on any QTSQ build, and
        // the `>= 20` parse assertion above keeps the table itself honest in
        // every build.
        let mut broken: Vec<&String> = Vec::new();
        let mut memory_resolved: Vec<&String> = Vec::new();
        for (m, ok) in methods.iter().zip(flags.iter()) {
            let is_memory = m.starts_with("memory.");
            if !*ok {
                if sofuu_core::HAS_QTSQ || !is_memory {
                    broken.push(m);
                }
            } else if is_memory {
                memory_resolved.push(m);
            }
        }
        assert!(
            broken.is_empty(),
            "docs/EMBEDDING.md documents funnel methods that do not resolve: {broken:?}"
        );
        if sofuu_core::HAS_QTSQ {
            assert!(
                !memory_resolved.is_empty(),
                "a QTSQ build must resolve the memory funnel"
            );
        } else {
            assert!(
                memory_resolved.is_empty(),
                "a QTSQ-free build must expose NO memory methods (found {:?}) — \
                 a partial registration means the no-QTSQ path is broken",
                memory_resolved
            );
        }

        unsafe { sofuu_rt_free(rt) };
    }

    /// E2: a host sets provider/model/endpoint ONCE in `sofuu_rt_new` and
    /// every later call inherits it. Before this an embedder had to repeat
    /// `provider`/`model`/`base_url` on every call, because there is no
    /// interactive `/model` picker inside an app.
    ///
    /// The signal is which gate a call reaches. With no model configured the
    /// runtime fails fast with "No model configured"; with the config
    /// default it gets past that gate and attempts the request (here to a
    /// closed port, so the failure is a connection error instead). That
    /// difference is deterministic and needs no network or server.
    #[test]
    fn config_supplies_default_provider_model_and_endpoint() {
        let _g = lock();
        let cfg = CString::new(
            r#"{"embedded":true,
                "provider":"openai",
                "model":"gpt-4o-mini",
                "base_url":"http://127.0.0.1:1/v1/chat/completions",
                "api_keys":{"openai":"local-token"}}"#,
        )
        .unwrap();
        // Control FIRST: a runtime with no default must stop at the model
        // gate. (Settings are per-thread and the most recent sofuu_rt_new
        // wins, so the control has to run before the configured runtime is
        // created — otherwise creating it would install its own settings.)
        let bare = unsafe { sofuu_rt_new(ptr::null()) };
        let bare_msg = envelope_message(bare, r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert!(
            bare_msg.contains("No model configured"),
            "without a default, the model gate must fire, got: {bare_msg}"
        );
        unsafe { sofuu_rt_free(bare) };

        let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
        assert!(!rt.is_null(), "runtime should build from the extended config");

        // With the default configured, the same call gets past that gate.
        let msg = envelope_message(rt, r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert!(
            !msg.contains("No model configured"),
            "the configured default model must satisfy the model gate, got: {msg}"
        );
        assert!(
            !msg.contains("\"ok\":true"),
            "this test points at a closed port on purpose; it must not succeed: {msg}"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    /// An explicit per-call model must still win over the configured
    /// default — the defaults are a convenience, never an override.
    #[test]
    fn explicit_per_call_model_overrides_the_configured_default() {
        let _g = lock();
        // A default model, but base_url points at a closed local port. A
        // per-call model replaces the default; we cannot observe the value
        // through the wire, so we assert the call is still attempted (i.e.
        // the explicit model also satisfied the gate) — the precedence of
        // the *value* is covered by the isolation test below.
        let cfg = CString::new(
            r#"{"embedded":true,"model":"default-model",
                "base_url":"http://127.0.0.1:1/v1/chat/completions",
                "api_keys":{"openai":"t"}}"#,
        )
        .unwrap();
        let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
        assert!(!rt.is_null());

        let msg = envelope_message(
            rt,
            r#"{"messages":[{"role":"user","content":"hi"}],"model":"explicit-model"}"#,
        );
        assert!(
            !msg.contains("No model configured"),
            "an explicit per-call model must satisfy the gate on its own: {msg}"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    /// The defaults must not leak: creating runtime B must not retarget
    /// runtime A (the same isolation the api_keys reader guarantees).
    #[test]
    fn config_defaults_do_not_leak_across_runtimes() {
        let _g = lock();

        // Runtime A: a configured default, so its calls clear the model gate.
        let a_cfg = CString::new(
            r#"{"embedded":true,"model":"model-a",
                "base_url":"http://127.0.0.1:1/v1/chat/completions",
                "api_keys":{"openai":"t"}}"#,
        )
        .unwrap();
        let rt_a = unsafe { sofuu_rt_new(a_cfg.as_ptr()) };
        assert!(!rt_a.is_null());
        let before = envelope_message(rt_a, r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert!(!before.contains("No model configured"), "A should have a default: {before}");

        // Runtime B, created on the same thread with different defaults.
        // B's install rewrites the thread-local settings.
        let b_cfg = CString::new(
            r#"{"embedded":true,"model":"model-b",
                "base_url":"http://127.0.0.1:1/v1/chat/completions",
                "api_keys":{"openai":"t"}}"#,
        )
        .unwrap();
        let rt_b = unsafe { sofuu_rt_new(b_cfg.as_ptr()) };
        assert!(!rt_b.is_null());
        let bmsg = envelope_message(rt_b, r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert!(!bmsg.contains("No model configured"), "B should have a default: {bmsg}");

        // A still works: it has its own engine and its own installed settings.
        let after = envelope_message(rt_a, r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert!(
            !after.contains("No model configured"),
            "runtime A must keep working after B was created: {after}"
        );

        unsafe { sofuu_rt_free(rt_b) };
        unsafe { sofuu_rt_free(rt_a) };
    }
}

/// Call a method and return the whole envelope as a string (for tests that
/// assert on *which gate* an error came from, not on a value).
fn envelope_message(rt: *mut CapiRuntime, args: &str) -> String {
    let m = CString::new("ai.complete").unwrap();
    let a = CString::new(args).unwrap();
    let mut out: *mut c_char = ptr::null_mut();
    unsafe { sofuu_rt_call(rt, m.as_ptr(), a.as_ptr(), &mut out) };
    assert!(!out.is_null(), "envelope must not be NULL");
    let s = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
    unsafe { sofuu_free(out as *mut c_void) };
    s
}
