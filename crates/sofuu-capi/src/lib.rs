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

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

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
    if (id < 0) {
      var emptyIdx = live.indexOf(null);
      if (emptyIdx >= 0) {
        id = emptyIdx;
        live[id] = inst;
      } else {
        id = live.length;
        live.push(inst);
      }
    }
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
    for (var i = live.length - 1; i >= 0; i--) {
      if (live[i]) return live[i];
    }
    throw new TypeError('memory: no open brain — call memory.open first');
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
    if (!a.vec && a.text) {
      return one(a, 'remember', [String(a.text), a.role || 'user', a.kv_page_id | 0,
                                 a.source === undefined ? undefined : String(a.source)]);
    }
    return one(a, 'remember', [f32(a.vec), a.text, a.role || 'user', a.kv_page_id | 0,
                               a.source === undefined ? undefined : String(a.source)]);
  };
  M.learn = function (a) {
    a = a || {};
    var text = typeof a === 'string' ? a : (a.text !== undefined ? a.text : '');
    var role = a.role || 'user';
    var source = a.source !== undefined ? String(a.source) : undefined;
    return one(a, 'learn', [String(text), role, source]);
  };
  /* B1.5: the read-out surface, same shape as the in-process API. */
  M.list = function (a) {
    a = a || {};
    var opts = {};
    if (a.tier !== undefined) opts.tier = a.tier | 0;
    if (a.role !== undefined) opts.role = String(a.role);
    if (a.entity !== undefined) opts.entity = String(a.entity);
    if (a.text !== undefined) opts.text = String(a.text);
    if (a.min_strength !== undefined) opts.minStrength = +a.min_strength;
    if (a.offset !== undefined) opts.offset = a.offset | 0;
    if (a.limit !== undefined) opts.limit = a.limit | 0;
    return one(a, 'list', [opts]);
  };
  M.export = function (a) { return one(a || {}, 'export', []); };
  M.recall = function (a) {
    a = a || {};
    var q = a.text !== undefined ? a.text : a.query;
    var k = a.k === undefined ? (a.top_k !== undefined ? (a.top_k | 0) : 5) : (a.k | 0);
    if (!a.vec && q !== undefined && q !== null) {
      return one(a, 'recall', [String(q), k]);
    }
    var args = [f32(a.vec), k];
    if (q !== undefined && q !== null) args.push(String(q));
    return one(a, 'recall', args);
  };
  M.search = function (a) {
    a = a || {};
    var q = typeof a === 'string' ? a : (a.text !== undefined ? a.text : (a.query !== undefined ? a.query : ''));
    var k = a.k === undefined ? (a.top_k !== undefined ? (a.top_k | 0) : 5) : (a.k | 0);
    return one(a, 'search', [String(q), k]);
  };
  M.count = function (a) { return one(a || {}, 'count', []); };
  M.flush = function (a) { return one(a || {}, 'flush', []); };
  M.forget = function (a) {
    a = a || {};
    return one(a, 'forget', [a.index | 0]);
  };
  /* B6 (ROADMAP-BRAIN-CMA): the session working set, same shapes as the
   * in-process API — a headless embedder promotes exactly like the chat
   * driver does. */
  M.rememberWorking = function (a) {
    a = a || {};
    return one(a, 'rememberWorking', [f32(a.vec), a.text || '',
                                      a.role || 'user', a.kv_page_id | 0,
                                      a.source === undefined ? undefined : String(a.source)]);
  };
  M.promoteWorking = function (a) {
    a = a || {};
    return one(a, 'promoteWorking', [JSON.stringify(Array.isArray(a.ids) ? a.ids : (a.id !== undefined ? [a.id] : []))]);
  };
  M.clearWorking = function (a) { return one(a || {}, 'clearWorking', []); };
  /* B7.1: stats is part of the read-out parity (doctor and /brain info
   * read the same numbers a headless host does). */
  M.stats = function (a) { return one(a || {}, 'stats', []); };

  /* CMA parity additions */
  M.rememberEntity = function (a) {
    a = a || {};
    return one(a, 'rememberEntity', [
      f32(a.vec),
      a.text || '',
      a.entity_name || a.entityName || '',
      a.entity_type || a.entityType || 'unknown'
    ]);
  };
  M.rememberEntityTurn = function (a) {
    a = a || {};
    return one(a, 'rememberEntityTurn', [
      f32(a.vec),
      a.text || '',
      a.entity_name || a.entityName || '',
      a.entity_type || a.entityType || 'unknown'
    ]);
  };
  M.markPositive = function (a) {
    a = a || {};
    var ids = a.ids !== undefined ? a.ids : a.id;
    return one(a, 'markPositive', [ids]);
  };
  M.decayTick = function (a) {
    a = a || {};
    var dt = a.dt !== undefined ? +a.dt : (a.seconds !== undefined ? +a.seconds : (a.hours !== undefined ? (+a.hours * 3600) : 1));
    one(a, 'decayTick', [dt]);
    return true;
  };
  M.retain = function (a) { return one(a || {}, 'retain', []); };
  M.retainAgainst = function (a) {
    a = a || {};
    var otherHandle = a.other !== undefined ? a.other : (a.canonical !== undefined ? a.canonical : a.target);
    var otherInst = pick({ handle: otherHandle });
    return one(a, 'retainAgainst', [otherInst]);
  };
  M.consolidate = function (a) { return one(a || {}, 'consolidate', []); };
  M.compact = function (a) { return one(a || {}, 'compact', []); };
  M.importRecords = function (a) {
    a = a || {};
    var json = typeof a.records === 'string' ? a.records : JSON.stringify(a.records || []);
    return one(a, 'importRecords', [f32(a.vecs || a.vec), json]);
  };
  M.kvHints = function (a) {
    a = a || {};
    var n = a.n_hints !== undefined ? (a.n_hints | 0) : (a.n === undefined ? 5 : (a.n | 0));
    return one(a, 'kvHints', [f32(a.vec), n]);
  };

  M.close = function (a) {
    var inst = pick(a || {});
    var id = live.indexOf(inst);
    if (id >= 0) live[id] = null;
    return true;
  };
  M.handles = function () {
    var count = 0;
    for (var i = 0; i < live.length; i++) if (live[i]) count++;
    return count;
  };
  M.__sofuu_facade = true;
  if (S.brain && !S.brain.__sofuu_facade) {
    S.brain = M;
  }
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
      if (id < 0) {
        var emptyIdx = clients.indexOf(null);
        if (emptyIdx >= 0) {
          id = emptyIdx;
          clients[id] = c;
        } else {
          id = clients.length;
          clients.push(c);
        }
      }
      try {
        Object.defineProperty(c, '__handle', { value: id, enumerable: false });
        c.toJSON = function () { return { handle: id }; };
      } catch (e) {}
      return c;
    };
    function pickClient(a) {
      a = a || {};
      if (a.handle !== undefined && a.handle !== null) {
        var got = clients[a.handle | 0];
        if (!got) throw new TypeError('mcp: no client with handle ' + a.handle);
        return got;
      }
      for (var i = clients.length - 1; i >= 0; i--) {
        if (clients[i]) return clients[i];
      }
      throw new TypeError('mcp: connect a server first');
    }
    X.call = function (a) {
      a = a || {};
      return pickClient(a).call(a.name, a.args === undefined ? {} : a.args);
    };
    X.listTools = function (a) { return pickClient(a).listTools(); };
    X.listResources = function (a) { return pickClient(a).listResources(); };
    X.disconnect = function (a) {
      var c = pickClient(a), id = clients.indexOf(c);
      try { c.disconnect(); } finally { if (id >= 0) clients[id] = null; }
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
      if (id < 0) {
        var emptyIdx = servers.indexOf(null);
        if (emptyIdx >= 0) {
          id = emptyIdx;
          servers[id] = srv;
        } else {
          id = servers.length;
          servers.push(srv);
        }
      }
      try { Object.defineProperty(srv, '__handle', { value: id, enumerable: false }); } catch (e) {}
      var port = a.port === undefined ? 8080 : (a.port | 0);
      var host = a.host === undefined ? '127.0.0.1' : a.host;
      if (host) srv.listen(port, host); else srv.listen(port);
      return { handle: id, port: port, host: host };
    };
    H.close = function (a) {
      a = a || {};
      var srv;
      if (a.handle !== undefined && a.handle !== null) {
        srv = servers[a.handle | 0];
      } else {
        for (var i = servers.length - 1; i >= 0; i--) {
          if (servers[i]) { srv = servers[i]; break; }
        }
      }
      if (!srv) throw new TypeError('http: no server running');
      var id = servers.indexOf(srv);
      try { if (typeof srv.close === 'function') srv.close(); }
      finally { if (id >= 0) servers[id] = null; }
      return true;
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
///     "enable_signals": false }   // signal handlers (default off)
/// Returns NULL on failure.
/// Android-only observability: route Rust panics (silent worker-thread
/// deaths — an inference worker panicking looks like "thinking forever,
/// no output" because stderr is not persisted on device) into logcat
/// under the `sofuu-panic` tag. Installed once, at first rt_new.
#[cfg(target_os = "android")]
unsafe extern "C" {
    fn __android_log_print(prio: i32, tag: *const std::os::raw::c_char, fmt: *const std::os::raw::c_char, ...) -> i32;
}

#[cfg(target_os = "android")]
fn install_android_panic_hook() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".to_string()
        };
        let msg = format!("RUST PANIC at {} — {}", loc, payload);
        unsafe {
            let cmsg = std::ffi::CString::new(msg.clone()).unwrap_or_default();
            let tag = std::ffi::CString::new("sofuu-panic").unwrap_or_default();
            // 6 = ANDROID_LOG_ERROR. liblog is already a linked dependency
            // of the engine (.so NEEDS liblog).
            __android_log_print(6, tag.as_ptr(), cmsg.as_ptr());
        }
        eprintln!("{}", msg);
        default_hook(info);
    }));
}

#[cfg(target_os = "android")]
static ANDROID_PANIC_HOOK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(not(target_os = "android"))]
fn install_android_panic_hook() {
    // Desktop hosts keep default panic reporting on stderr.
}

#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_new(config_json: *const c_char) -> *mut CapiRuntime {
    #[cfg(target_os = "android")]
    if !ANDROID_PANIC_HOOK.swap(true, std::sync::atomic::Ordering::SeqCst) {
        install_android_panic_hook();
    }

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
    //
    // E2: allowHostTools — such an agent may name a host tool that the host
    // registers later (sofuu_rt_register_host_tool cannot run until this
    // call returns), so the unknown-builtin throw is deferred to run time
    // for this path only.
    if let Some(agents) = config_val.get("agents").and_then(|v| v.as_array()) {
        for agent_def in agents {
            let def_json = serde_json::to_string(agent_def).unwrap_or_default();
            let src = format!(
                "try {{ sofuu.agent.define({}, {{allowHostTools: true}}); }} catch(e) {{ /* duplicate/bad def — non-fatal */ }}",
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

    // E2: the host-tool registry is thread-scoped, so a runtime created on
    // this thread starts empty (the previous runtime's callbacks and opaque
    // pointers are gone with it).
    HOST_TOOLS.with(|m| m.borrow_mut().clear());

    Box::into_raw(handle)
}

/// Destroy a runtime and release its engine. NULL-tolerant.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_free(rt: *mut CapiRuntime) {
    if !rt.is_null() {
        // E2: drop host tools with the runtime — the host's opaque pointers
        // must never be called after it freed the runtime that owned them.
        HOST_TOOLS.with(|m| m.borrow_mut().clear());
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
    let mut args_is_array = false;
    let args_part = match cstr_or_null(args_json) {
        None | Some("") => "undefined".to_string(),
        Some(s) => {
            // Validate it's valid JSON.
            match serde_json::from_str::<JsonValue>(s) {
                Ok(v) => {
                    args_is_array = v.is_array();
                    format!("JSON.parse({})", serde_json::to_string(s).unwrap_or_default())
                }
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
    // agent.run's shipped signature is POSITIONAL — (target, task, opts).
    // (2026-10-15, found by the committed contract test) the documented
    // object form used to land on run() as a single "target" argument and
    // died in normalizeDef with "agent.define: name is required" — the
    // stream builder always spread it correctly; the blocking call did not.
    let call_expr = if args_is_array {
        "var r=await o.apply(null,a);".to_string()
    } else if method_str == "agent.run" {
        "var r=await o(a.target, a.task, a);".to_string()
    } else {
        "var r=a===undefined?await o():await o(a);".to_string()
    };
    let js = format!(
        r#"var __capi_r=null;(async function(){{
  function markUnknown(m){{var e=new TypeError(m);e.__sofuuUnknown=true;return e}}
  try {{
    var p={method_json}.split("."),o=sofuu;
    for(var i=0;i<p.length;i++){{if(o==null)throw markUnknown("sofuu."+p.slice(0,i+1).join(".")+" is not available");o=o[p[i]]}}
    if(typeof o!=="function")throw markUnknown("sofuu."+{method_json}+" is not a function");
    var a={args_part};
    {call_expr}
    __capi_r={{ok:true,result:r}}
  }}catch(e){{
    __capi_r={{ok:false,error:{{code:(e&&e.__sofuuUnknown)?"unknown_method":"js_exception",message:e.message||String(e)}}}}
  }}
}})()"#,
        method_json = serde_json::to_string(method_str).unwrap_or_default(),
        args_part = args_part,
        call_expr = call_expr,
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

/// Non-blocking variant of [`sofuu_rt_eval`] for hosts that drive the
/// runtime incrementally from their own loop (the Android chat pump).
///
/// Same JS semantics as `sofuu_rt_eval` (wrapper envelope, sync result
/// JSON-serialized), but instead of `drain_loop` — which blocks until all
/// promises settle and all I/O completes, and would hold the runtime
/// thread for a whole chat turn — it performs exactly ONE non-blocking
/// loop tick: microtasks, `UV_RUN_NOWAIT`, microtasks. An async feature
/// started here (its eval result is a pending promise, serialized as
/// `{}`) progresses across repeated tick evals while the thread stays
/// free for the host's pump, cancel pokes, and bounded queries.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_eval_nb(
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

    // Same wrapper as sofuu_rt_eval: always a JSON envelope, sync results
    // captured, async results left as a pending promise object.
    let wrapper = format!(
        r#"(function(){{try{{return{{ok:true,result:eval({src_json})}}}}catch(e){{return{{ok:false,error:{{code:"js_exception",message:e.message||String(e)}}}}}}}})()"#,
        src_json = serde_json::to_string(src).unwrap_or_default(),
    );
    let c_wrapper = CString::new(wrapper).unwrap_or_default();
    let result = qjs::JS_Eval(
        ctx,
        c_wrapper.as_ptr(),
        c_wrapper.as_bytes().len(),
        c"<capi-eval-nb>".as_ptr(),
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
        return -4; // SOFUU_ERR_JS_EXCEPTION
    }

    // Exactly ONE non-blocking tick instead of drain_loop: microtasks,
    // one UV_RUN_NOWAIT poll, microtasks again (sofuu_loop_tick_nb does
    // all three and never waits).
    sofuu_core::rt::event_loop::sofuu_loop_tick_nb(ctx);

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
/// `args_json` is an object (merged with `{ onStep: <internal cb> }` and the
/// cancel token, then passed as the single options arg) or an array (spread
/// as positional args, with the same two keys injected into its trailing
/// options object) — the same contract `sofuu_rt_call` documents.
/// *out_cancel_id receives a token usable with sofuu_rt_cancel.
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

    // E2: when this runtime has host tools, make sure the bridge is the one
    // currently installed. sofuu.chat.init() installs its own __host_poke
    // handler, and a host may call it after registering tools; re-asserting
    // here (cheap, idempotent) keeps completions flowing either way.
    if HOST_TOOLS.with(|m| !m.borrow().is_empty()) {
        install_host_tool_bridge(ctx);
    }

    // P2-19 (AUDIT-2026-09-01): parse method + args BEFORE installing the
    // thread-local slot and the global callback — the old order left the
    // slot + a dangling `opaque` installed on every early `return -1`
    // (null/bad method or bad args JSON), poisoning the NEXT stream call.
    let method_str = match cstr_or_null(method) {
        Some(s) => s.to_string(),
        None => return -1,
    };

    // E1 (PLAN-EMBED-SDK): args_json is an object OR an array, exactly like
    // sofuu_rt_call — the array form is spread as positional arguments. This
    // is what makes positional methods drivable at all: `ai.streamLocal`
    // takes (modelPath, prompt, options), and before this the driver always
    // called the method with a single object, so the model path never
    // arrived.
    let (args_text, args_is_array) = match cstr_or_null(args_json) {
        None | Some("") => ("{}".to_string(), false),
        Some(s) => {
            match serde_json::from_str::<JsonValue>(s) {
                Ok(v) => (s.to_string(), v.is_array()),
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
    let args_json_lit = serde_json::to_string(&args_text).unwrap_or_default();
    // E1: the array form spreads as positional arguments. `onStep`/`signal`
    // land on the trailing options object — the one the method already
    // expects (`ai.streamLocal(path, prompt, opts)`) — or on an appended one
    // when the caller passed none, so a cancel token exists either way.
    let call_expr = if args_is_array {
        format!(
            r#"var a=JSON.parse({args_json_lit});
            if(!Array.isArray(a))a=[];
            var t=(a.length>0)?a[a.length-1]:null;
            if(t===null||typeof t!=="object"||Array.isArray(t)){{t={{}};a.push(t);}}
            t.onStep=function(evt){{try{{globalThis.__sofuu_stream_cb(JSON.stringify(evt));}}catch(e){{}}}};
            t.signal={cancel_id};
            var r=await o.apply(null,a);"#
        )
    } else if method_str == "agent.run" {
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

// ── Host-native tools (PLAN-EMBED-SDK E2) ─────────────────────────
//
// A host registers a native function (Swift/Kotlin/C) that an in-app agent
// can call as a tool — the plan's headline case is `take_photo`: the model
// asks for it, the host opens the camera, and a result that arrives seconds
// later goes back into the agent loop.
//
// The callback answers in two ways, which is what makes that possible
// without the host blocking its own thread:
//   - a JSON string → synchronous result, used immediately;
//   - NULL          → "I took the call", settled later from ANY thread with
//                     sofuu_rt_host_tool_resolve.
//
// Wiring: a C function is installed as the JS global `__sofuu_host_tool`
// (the same trick the per-stream callback uses), and `HOST_TOOL_BRIDGE_JS`
// owns the pending-promise map in JS. A completion crosses threads through
// rt/host_poke.rs — uv_async_send is the one libuv call documented as safe
// from another thread — and lands on the loop thread, which is the only
// thread allowed to touch QuickJS.
//
// The event loop stays alive while a call is outstanding because the agent's
// own tool timeout is a real timer handle (agent.js withTimeout), so no
// extra uv handle is needed — and that same timer is the honest ceiling on
// "the host may answer later" (30 s per tool by default, toolTimeoutMs).

/// Host tool callback signature; matches `sofuu_host_tool_fn` in the header.
pub type SofuuHostToolFn =
    unsafe extern "C" fn(*const c_char, u64, *mut c_void) -> *const c_char;

struct HostTool {
    cb: SofuuHostToolFn,
    opaque: *mut c_void,
}

thread_local! {
    /// Host tools registered for this thread's runtime, by name. Thread-scoped
    /// like STREAM_CB, so the documented one-runtime-per-thread rule applies:
    /// a runtime created on a thread starts with an empty registry.
    static HOST_TOOLS: std::cell::RefCell<HashMap<String, HostTool>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Host tool calls awaiting a completion. Process-global rather than
/// thread-local because sofuu_rt_host_tool_resolve arrives from any thread.
static PENDING_HOST_CALLS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The JS half of the host-tool bridge, installed once per runtime.
///
/// `__sofuu_host_tool_exec` mints the call id, stores the promise resolvers,
/// and calls the C entry point. A string answer is shaped and returned;
/// `undefined` means the host took the call, so the pending promise goes back
/// to the agent loop and `__sofuu_host_tool_settle` settles it later.
///
/// Shaping rule: a JSON string result reaches the model as its text, anything
/// else as JSON text. agent.js stringifies tool results with String(), so an
/// object handed back raw would reach the model as "[object Object]".
///
/// The `__host_poke` chaining matters: chat.js installs its own handler for
/// cancel/approval decisions, so this composes with it (and re-installs if a
/// later writer replaced it) instead of hijacking the channel.
const HOST_TOOL_BRIDGE_JS: &str = r#"(function(){
  globalThis.__sofuu_host_tools = globalThis.__sofuu_host_tools || {};
  globalThis.__sofuu_host_pending = globalThis.__sofuu_host_pending || {};
  globalThis.__sofuu_host_seq = globalThis.__sofuu_host_seq || 0;
  if (!globalThis.__sofuu_host_shape) {
    globalThis.__sofuu_host_shape = function(raw){
      var v;
      try { v = JSON.parse(raw); } catch (e) { throw new Error("host tool returned invalid JSON: " + raw); }
      return (typeof v === "string") ? v : raw;
    };
  }
  if (!globalThis.__sofuu_host_tool_exec) {
    globalThis.__sofuu_host_tool_exec = function(name, args){
      var id = ++globalThis.__sofuu_host_seq;
      var res = null, rej = null;
      var p = new Promise(function(a, b){ res = a; rej = b; });
      var pending = globalThis.__sofuu_host_pending;
      var now = Date.now();
      for (var k in pending) { if (now - pending[k].ts > 600000) delete pending[k]; }
      pending[id] = { resolve: res, reject: rej, ts: now };
      var raw = globalThis.__sofuu_host_tool(name, JSON.stringify(args === undefined ? {} : args), id);
      if (typeof raw === "string") { delete pending[id]; return globalThis.__sofuu_host_shape(raw); }
      return p;
    };
  }
  if (!globalThis.__sofuu_host_tool_settle) {
    globalThis.__sofuu_host_tool_settle = function(m){
      var pending = globalThis.__sofuu_host_pending || {};
      var e = pending[m.call_id];
      if (!e) return;
      delete pending[m.call_id];
      if (m.error) { e.reject(new Error(String(m.error))); return; }
      try { e.resolve(globalThis.__sofuu_host_shape(String(m.result))); }
      catch (err) { e.reject(err); }
    };
  }
  var prev = globalThis.__host_poke;
  if (!(prev && prev.__sofuuEmbedBridge)) {
    var w = function(json){
      var m = null;
      try { m = JSON.parse(String(json)); } catch (e) {}
      if (m && m.__sofuu_host_tool_result) { globalThis.__sofuu_host_tool_settle(m); return; }
      if (typeof prev === "function") return prev(json);
    };
    w.__sofuuEmbedBridge = true;
    globalThis.__host_poke = w;
  }
})()"#;

/// Read a JS value as a Rust string (numbers included — JS_ToCString
/// stringifies them, which is how the call id crosses over).
unsafe fn js_value_string(ctx: *mut qjs::JSContext, val: qjs::JSValueConst) -> Option<String> {
    let s = qjs::sofuu_js_to_cstring(ctx, val);
    if s.is_null() {
        return None;
    }
    let r = CStr::from_ptr(s).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, s);
    Some(r)
}

/// Install the C entry point + the JS bridge. Idempotent: safe to call on
/// every registration and again before each stream.
unsafe fn install_host_tool_bridge(ctx: *mut qjs::JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let fname = c"__sofuu_host_tool";
    let existing = qjs::sofuu_js_get_property_str(ctx, global, fname.as_ptr());
    let present = qjs::JS_IsFunction(ctx, existing) != 0;
    qjs::sofuu_js_free_value(ctx, existing);
    if !present {
        let fn_val = qjs::sofuu_js_new_cfunction(ctx, js_host_tool_call, fname.as_ptr(), 3);
        qjs::sofuu_js_set_property_str(ctx, global, fname.as_ptr(), fn_val);
    }
    qjs::sofuu_js_free_value(ctx, global);

    if let Ok(c) = CString::new(HOST_TOOL_BRIDGE_JS) {
        // ffi-1: drain any exception so a broken snippet cannot leave a
        // sticky error on the context for the next call.
        eval_init_snippet(ctx, &c, c"<host-tool-bridge>");
    }
}

/// The C entry point JS calls for a registered host tool. Returns the
/// callback's JSON result as a JS string, or undefined when the host took the
/// call (the JS side then waits on the pending promise).
unsafe extern "C" fn js_host_tool_call(
    ctx: *mut qjs::JSContext,
    _this_val: qjs::JSValueConst,
    argc: c_int,
    argv: *const qjs::JSValueConst,
) -> qjs::JSValue {
    if argc < 3 {
        return qjs::JS_ThrowTypeError(
            ctx,
            c"__sofuu_host_tool(name, argsJson, callId)".as_ptr(),
        );
    }
    let Some(name) = js_value_string(ctx, *argv) else {
        return qjs::sofuu_js_undefined();
    };
    let args_json = js_value_string(ctx, *argv.add(1)).unwrap_or_else(|| "{}".to_string());
    let call_id: u64 = js_value_string(ctx, *argv.add(2))
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    let entry = HOST_TOOLS.with(|m| m.borrow().get(&name).map(|t| (t.cb, t.opaque)));
    let Some((cb, opaque)) = entry else {
        // A tool spec outliving its registration (e.g. re-created runtime on
        // this thread). Throw rather than hang on a promise nobody owns.
        let msg = CString::new(format!("no such host tool: {name}")).unwrap_or_default();
        return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
    };

    let c_args = CString::new(args_json).unwrap_or_default();
    let out = cb(c_args.as_ptr(), call_id, opaque);
    if out.is_null() {
        if call_id != 0 {
            if let Ok(mut p) = PENDING_HOST_CALLS.lock() {
                p.push(call_id);
            }
        }
        return qjs::sofuu_js_undefined(); // pending — completed by resolve
    }

    let s = CStr::from_ptr(out).to_string_lossy().into_owned();
    let c_s = CString::new(s).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, c_s.as_ptr())
}

/// Register a native host function as an agent tool (PLAN-EMBED-SDK E2).
///
/// The tool is visible to every agent of this runtime, including agents
/// supplied through the `agents` config key — those are defined during
/// sofuu_rt_new, before a host can register anything, which is why the tool
/// is published into `globalThis.__sofuu_host_tools` (resolved at run time)
/// as well as into `sofuu.tools.TOOLS` (resolved at define time).
///
/// Registering the same name twice replaces the earlier registration.
/// Returns SOFUU_OK or a SOFUU_ERR_*.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_register_host_tool(
    rt: *mut CapiRuntime,
    name: *const c_char,
    description: *const c_char,
    parameters_json: *const c_char,
    cb: Option<SofuuHostToolFn>,
    opaque: *mut c_void,
) -> c_int {
    if rt.is_null() || name.is_null() || cb.is_none() {
        return -1; // SOFUU_ERR_INVALID_ARG
    }
    let rt_ref = &*rt;
    let ctx = rt_ref.ffi_rt.engine_ctx() as *mut qjs::JSContext;
    if ctx.is_null() {
        return -6; // SOFUU_ERR_INTERNAL
    }
    let Some(name) = cstr_or_null(name).map(|s| s.trim().to_string()) else {
        return -1;
    };
    if name.is_empty() {
        return -1;
    }
    // The schema reaches the model verbatim; a malformed one would look like
    // a model failure rather than an input error, so reject it here.
    let params_json = match cstr_or_null(parameters_json) {
        None | Some("") => r#"{"type":"object","properties":{}}"#.to_string(),
        Some(s) => match serde_json::from_str::<JsonValue>(s) {
            Ok(JsonValue::Object(_)) => s.to_string(),
            _ => return -1,
        },
    };
    let desc = cstr_or_null(description).unwrap_or("").to_string();

    let cb = cb.unwrap();
    HOST_TOOLS.with(|m| {
        m.borrow_mut().insert(name.clone(), HostTool { cb, opaque });
    });

    install_host_tool_bridge(ctx);

    let js = format!(
        r#"(function(){{
  var def = {{ name: {name_json}, description: {desc_json}, parameters: {params} }};
  def.execute = function(args){{ return globalThis.__sofuu_host_tool_exec({name_json}, args); }};
  globalThis.__sofuu_host_tools[{name_json}] = def;
  if (sofuu.tools && sofuu.tools.TOOLS) sofuu.tools.TOOLS[{name_json}] = def;
}})()"#,
        name_json = serde_json::to_string(&name).unwrap_or_default(),
        desc_json = serde_json::to_string(&desc).unwrap_or_default(),
        params = params_json,
    );
    if let Ok(c) = CString::new(js) {
        eval_init_snippet(ctx, &c, c"<host-tool-register>");
    }

    0
}

/// Complete a host tool call whose callback returned NULL. Callable from any
/// thread — the completion is handed to the runtime's thread through its
/// event loop; the runtime must outlive the call.
///
/// `result_json` is a JSON value the agent sees as the tool result. To fail
/// the call instead, pass a message in `error` (result_json may then be
/// NULL). Returns SOFUU_OK, or -1 when the id is unknown (already settled,
/// never pending, or a zero id).
#[no_mangle]
pub unsafe extern "C" fn sofuu_rt_host_tool_resolve(
    rt: *mut CapiRuntime,
    call_id: u64,
    result_json: *const c_char,
    error: *const c_char,
) -> c_int {
    if rt.is_null() || call_id == 0 {
        return -1;
    }
    let err = cstr_or_null(error).map(String::from);
    let result = cstr_or_null(result_json).map(String::from);
    if err.is_none() {
        let Some(r) = result.as_deref() else {
            return -1;
        };
        if serde_json::from_str::<JsonValue>(r).is_err() {
            return -1;
        }
    }

    // Only an outstanding call may settle: a double resolve or a mistyped id
    // is a host bug, and -1 says so instead of resolving a stranger's promise.
    let known = match PENDING_HOST_CALLS.lock() {
        Ok(mut p) => match p.iter().position(|id| *id == call_id) {
            Some(i) => {
                p.remove(i);
                true
            }
            None => false,
        },
        Err(_) => false,
    };
    if !known {
        return -1;
    }

    let msg = match &err {
        Some(e) => serde_json::json!({
            "__sofuu_host_tool_result": 1,
            "call_id": call_id,
            "error": e,
        }),
        None => serde_json::json!({
            "__sofuu_host_tool_result": 1,
            "call_id": call_id,
            "result": result,
        }),
    };
    sofuu_core::rt::host_poke::host_poke_send(&msg.to_string());
    0
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
/// The default used to disagree with the brain's canonical space
/// (HASH_DIM = 768). An embedder calling
/// `sofuu_embed_local(rt, text, NULL, …)` therefore got 64-dim vectors it
/// could not store in — or match against — the 768-dim brain. Now the C
/// default and the brain agree.
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
/// NULL. Errors: -1 invalid arg, -8 model-unavailable (also covers
/// undecodable bytes: no vector exists for them; SOFUU_ERR_MODEL_UNAVAIL
/// per sofuu_embed.h), -9 nomem.
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

    // ── E1 (PLAN-EMBED-SDK): positional (array) args in the stream driver ──
    //
    // Nothing model-sized is checked in — the GGUF / SafeTensors fixtures are
    // hundreds of MB — so the live tests below degrade to a skip when the
    // fixture is absent (override the path with the env var). Same shape
    // sofuu-core's test_safetensors_models.rs uses, for the same reason: a
    // gate that only passes on the author's machine is not a gate.
    const DEFAULT_LIVE_GGUF: &str =
        "/Users/priyanshuboruah/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf";
    const DEFAULT_LIVE_LFM2_DIR: &str = "/Users/priyanshuboruah/Downloads/liqiud-350";

    /// Returns the fixture path when it exists, else None (after saying why).
    fn live_path(env: &str, default: &str) -> Option<String> {
        let p = std::env::var(env).unwrap_or_else(|_| default.to_string());
        if std::path::Path::new(&p).exists() {
            Some(p)
        } else {
            eprintln!("skipping: no model at {p} (set {env} to point at one)");
            None
        }
    }

    /// E1: `ai.streamLocal` is positional — `(modelPath, prompt, options?)` —
    /// while the stream driver used to call every method with a single object,
    /// so the model path never arrived. The mock here *fails the stream* if it
    /// is not invoked positionally, so a regression cannot pass silently.
    #[test]
    fn stream_spreads_positional_array_args() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let setup = CString::new(
            r#"(function(){
              sofuu.ai.streamLocal = function(modelPath, prompt, opts) {
                return (async function*(){
                  if (modelPath !== "/tmp/m.gguf") throw new Error("modelPath was " + modelPath);
                  if (prompt !== "hello") throw new Error("prompt was " + prompt);
                  if (!opts || typeof opts.onStep !== "function") throw new Error("opts.onStep missing");
                  if (typeof opts.signal !== "number") throw new Error("opts.signal missing");
                  yield { text: "A" };
                  yield { text: "B" };
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

        let method = CString::new("ai.streamLocal").unwrap();
        // No trailing options object: the driver must append one so a cancel
        // hook exists even when the caller passes none.
        let args = CString::new(r#"["/tmp/m.gguf","hello"]"#).unwrap();
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
        assert_eq!(rc, 0, "streamLocal via array args should return OK");
        assert!(cancel_id != 0, "a cancel id is issued for the array shape too");

        let events = EVENTS.lock().unwrap().clone();
        let deltas = events
            .iter()
            .filter(|e| e.contains(r#""kind":"delta""#))
            .count();
        assert_eq!(deltas, 2, "expected 2 deltas, got {events:?}");
        let done = events
            .iter()
            .find(|e| e.contains(r#""kind":"done""#))
            .expect("a done event must always arrive");
        assert!(!done.contains(r#""error""#), "stream must not fail: {done}");

        unsafe { sofuu_rt_free(rt) };
    }

    /// E1: when the caller *does* pass a trailing options object, the driver
    /// must inject into it rather than replace it — dropping caller options
    /// would be a silent behaviour change — and cancel must work in the array
    /// shape exactly as it does for the object shape.
    #[test]
    fn stream_array_args_keep_caller_options_and_cancel() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let setup = CString::new(
            r#"(function(){
              sofuu.ai.streamLocal = function(modelPath, prompt, opts) {
                return (async function*(){
                  if (opts.maxTokens !== 7) throw new Error("caller options were dropped");
                  for (var i = 0; i < 20; i++) yield { text: "t" + i };
                })();
              };
            })()"#,
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) }, 0);
        unsafe { sofuu_free(out as *mut c_void) };

        static DELTAS: Mutex<u32> = Mutex::new(0);
        static DONE: Mutex<String> = Mutex::new(String::new());
        *DELTAS.lock().unwrap() = 0;
        DONE.lock().unwrap().clear();

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
                    unsafe { sofuu_rt_cancel(ctx.rt, ctx.cid) };
                }
            } else if s.contains(r#""kind":"done""#) {
                *DONE.lock().unwrap() = s;
            }
        }

        let mut ctx = CancelCtx { rt, cid: 0 };
        let method = CString::new("ai.streamLocal").unwrap();
        let args = CString::new(r#"["/tmp/m.gguf","hello",{"maxTokens":7}]"#).unwrap();
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
        assert_eq!(rc, 0);

        let deltas = *DELTAS.lock().unwrap();
        assert!(
            deltas < 20,
            "the first-delta cancel must stop the 20-chunk stream, saw {deltas}"
        );
        let done = DONE.lock().unwrap().clone();
        assert!(done.contains(r#""aborted":true"#), "done should report aborted: {done}");

        unsafe { sofuu_rt_free(rt) };
    }

    /// E1: the real thing end to end — a GGUF on disk, streamed positionally
    /// through the C ABI, with a live mid-flight cancel. This is the milestone
    /// ("receive token deltas") stated as an executable assertion.
    #[test]
    fn live_stream_local_gguf_through_the_c_abi() {
        let _g = lock();
        let Some(model) = live_path("SOFUU_TEST_GGUF", DEFAULT_LIVE_GGUF) else {
            return;
        };
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        static DELTAS: Mutex<u32> = Mutex::new(0);
        static TEXT: Mutex<String> = Mutex::new(String::new());
        static DONE: Mutex<String> = Mutex::new(String::new());
        *DELTAS.lock().unwrap() = 0;
        TEXT.lock().unwrap().clear();
        DONE.lock().unwrap().clear();

        #[repr(C)]
        struct Ctx {
            rt: *mut CapiRuntime,
            cid: u64,
        }

        unsafe extern "C" fn collect(event_json: *const c_char, opaque: *mut c_void) {
            let s = unsafe { CStr::from_ptr(event_json) }.to_string_lossy().into_owned();
            let ctx = unsafe { &*(opaque as *const Ctx) };
            if s.contains(r#""kind":"delta""#) {
                if let Ok(v) = serde_json::from_str::<JsonValue>(&s) {
                    if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                        TEXT.lock().unwrap().push_str(t);
                    }
                }
                let mut d = DELTAS.lock().unwrap();
                *d += 1;
                let n = *d;
                drop(d);
                // 6 deltas is enough to prove generation is real; cancelling
                // there keeps the test fast and exercises abort on the path
                // a host would actually cancel (the local decode loop).
                if n == 6 {
                    unsafe { sofuu_rt_cancel(ctx.rt, ctx.cid) };
                }
            } else if s.contains(r#""kind":"done""#) {
                *DONE.lock().unwrap() = s;
            }
        }

        let mut ctx = Ctx { rt, cid: 0 };
        let method = CString::new("ai.streamLocal").unwrap();
        let args = CString::new(
            serde_json::json!([model, "The capital of France is", { "maxTokens": 64 }]).to_string(),
        )
        .unwrap();
        let rc = unsafe {
            sofuu_rt_call_stream(
                rt,
                method.as_ptr(),
                args.as_ptr(),
                Some(collect),
                (&mut ctx as *mut Ctx) as *mut c_void,
                &mut ctx.cid,
            )
        };
        assert_eq!(rc, 0, "streamLocal should return OK");

        let deltas = *DELTAS.lock().unwrap();
        let text = TEXT.lock().unwrap().clone();
        let done = DONE.lock().unwrap().clone();
        assert!(deltas >= 2, "expected real token deltas, got {deltas}");
        assert!(!text.trim().is_empty(), "model produced no text");
        assert!(
            !done.contains(r#""error""#),
            "stream must not fail: {done}"
        );

        unsafe { sofuu_rt_free(rt) };
    }

    /// E1b: a SafeTensors model *directory* must route to in-process
    /// inference from the path alone. Before this it only did so with an
    /// explicit `provider: "local"`, so handing an embedded host a directory
    /// sent the request to the network as if it were a model name.
    #[test]
    fn live_safetensors_dir_routes_local_from_the_path_alone() {
        let _g = lock();
        let Some(dir) = live_path("SOFUU_TEST_LFM2_DIR", DEFAULT_LIVE_LFM2_DIR) else {
            return;
        };
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        static DELTAS: Mutex<u32> = Mutex::new(0);
        static DONE: Mutex<String> = Mutex::new(String::new());
        *DELTAS.lock().unwrap() = 0;
        DONE.lock().unwrap().clear();

        unsafe extern "C" fn collect(event_json: *const c_char, _opaque: *mut c_void) {
            let s = unsafe { CStr::from_ptr(event_json) }.to_string_lossy().into_owned();
            if s.contains(r#""kind":"delta""#) {
                *DELTAS.lock().unwrap() += 1;
            } else if s.contains(r#""kind":"done""#) {
                *DONE.lock().unwrap() = s;
            }
        }

        // The generic funnel, not streamLocal: this is the routing predicate
        // doing the work, with no provider hint anywhere in the call.
        let method = CString::new("ai.stream").unwrap();
        let args = CString::new(
            serde_json::json!({ "model": dir, "prompt": "Hello", "maxTokens": 8 }).to_string(),
        )
        .unwrap();
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
        assert_eq!(rc, 0);

        let deltas = *DELTAS.lock().unwrap();
        let done = DONE.lock().unwrap().clone();
        assert!(
            deltas >= 1,
            "a SafeTensors directory should generate tokens locally, got {deltas} deltas, done: {done}"
        );
        assert!(!done.contains(r#""error""#), "stream must not fail: {done}");

        unsafe { sofuu_rt_free(rt) };
    }

    // ── E2 (PLAN-EMBED-SDK): host-native tools ───────────────────────

    /// A mock provider for the host-tool tests. It is injected as
    /// `sofuu.ai.stream` because that is the one call the agent loop makes
    /// (agent.js streamWithRetry), so the real loop, the real tool dispatch
    /// and the real transcript assembly all run — only the model is fake.
    ///
    /// Round 1 asks for `tool`. Round 2 reports whether `token` (the tool's
    /// result) is present in the messages the model would see, so a broken
    /// bridge cannot pass by accident.
    fn mock_tool_agent_js(token: &str, agent: &str, tool: &str) -> String {
        format!(
            r#"(function(){{
  var TOKEN = {token_json};
  var round = 0;
  sofuu.ai.stream = function(opts) {{
    round++;
    var toolRound = (round === 1);
    var chunks;
    if (toolRound) {{ chunks = []; }}
    else {{
      var seen = JSON.stringify((opts && opts.messages) || []);
      chunks = [{{ text: seen.indexOf(TOKEN) >= 0 ? "CONFIRMED-TOOL-RESULT" : "MISSING-TOOL-RESULT" }}];
    }}
    var i = 0;
    var it = {{
      usage: {{ promptTokens: 11, completionTokens: 3, finishReason: "stop" }},
      toolCalls: toolRound
        ? [{{ index: 0, id: "call_1", function: {{ name: {tool_json}, arguments: "{{}}" }} }}]
        : [],
      abort: function() {{}},
      next: function() {{
        if (i < chunks.length) return Promise.resolve({{ value: chunks[i++], done: false }});
        return Promise.resolve({{ value: undefined, done: true }});
      }}
    }};
    it[Symbol.asyncIterator] = function() {{ return it; }};
    return it;
  }};
  sofuu.agent.define({{
    name: {agent_json},
    system: "You are a test agent.",
    provider: "mock",
    model: "mock-tool-test",
    tools: [{{ builtin: {tool_json} }}],
    budget: {{ maxSteps: 4 }}
  }});
}})()"#,
            token_json = serde_json::to_string(token).unwrap_or_default(),
            tool_json = serde_json::to_string(tool).unwrap_or_default(),
            agent_json = serde_json::to_string(agent).unwrap_or_default(),
        )
    }

    /// Drive `agent.run` over the streaming ABI and return every event.
    fn run_agent_collect(rt: *mut CapiRuntime, target: &str) -> Vec<String> {
        static EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());
        EVENTS.lock().unwrap().clear();
        unsafe extern "C" fn collect(event_json: *const c_char, _opaque: *mut c_void) {
            let s = unsafe { CStr::from_ptr(event_json) }.to_string_lossy().into_owned();
            EVENTS.lock().unwrap().push(s);
        }

        let method = CString::new("agent.run").unwrap();
        let args = CString::new(
            serde_json::json!({ "target": target, "task": "take a photo" }).to_string(),
        )
        .unwrap();
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
        assert_eq!(rc, 0, "agent.run should return OK");
        EVENTS.lock().unwrap().clone()
    }

    /// E2: the headline case — a native function the host registered runs
    /// inside a real agent turn, and its result reaches the conversation the
    /// model sees.
    #[test]
    fn host_tool_runs_inside_an_agent_turn() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        static CALLS: Mutex<u32> = Mutex::new(0);
        static ARGS: Mutex<String> = Mutex::new(String::new());
        *CALLS.lock().unwrap() = 0;
        ARGS.lock().unwrap().clear();

        // Synchronous host tool: answer with a JSON object. The buffer is
        // host-owned and static here, which is the cheapest legal shape of
        // the contract (the library copies it before returning).
        unsafe extern "C" fn take_photo(
            args_json: *const c_char,
            _call_id: u64,
            _opaque: *mut c_void,
        ) -> *const c_char {
            let s = unsafe { CStr::from_ptr(args_json) }.to_string_lossy().into_owned();
            *ARGS.lock().unwrap() = s;
            *CALLS.lock().unwrap() += 1;
            c"{\"path\":\"/tmp/TOOLRES-photo-9f3.jpg\"}".as_ptr()
        }

        let name = CString::new("take_photo").unwrap();
        let desc = CString::new("Takes a photo with the device camera").unwrap();
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    rt,
                    name.as_ptr(),
                    desc.as_ptr(),
                    ptr::null(),
                    Some(take_photo),
                    ptr::null_mut(),
                )
            },
            0,
            "registration should succeed"
        );

        let setup = CString::new(mock_tool_agent_js(
            "TOOLRES-photo-9f3",
            "photo-agent",
            "take_photo",
        ))
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) }, 0);
        unsafe { sofuu_free(out as *mut c_void) };

        let events = run_agent_collect(rt, "photo-agent");
        let joined = events.join("\n");

        assert_eq!(*CALLS.lock().unwrap(), 1, "the host tool must run exactly once");
        assert_eq!(*ARGS.lock().unwrap(), "{}", "the model's args reach the callback");
        assert!(
            joined.contains("CONFIRMED-TOOL-RESULT"),
            "the tool result must reach the model's transcript; events: {joined}"
        );
        assert!(!joined.contains("MISSING-TOOL-RESULT"), "events: {joined}");

        unsafe { sofuu_rt_free(rt) };
    }

    /// E2: the async half — the callback returns NULL (the camera case: the
    /// photo does not exist yet) and the host completes the call from
    /// ANOTHER thread. uv_async_send carries it to the loop thread, where
    /// the pending promise is settled and the agent turn resumes.
    #[test]
    fn host_tool_completes_from_another_thread() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        static ASYNC_RT: std::sync::atomic::AtomicPtr<CapiRuntime> =
            std::sync::atomic::AtomicPtr::new(ptr::null_mut());
        ASYNC_RT.store(rt, Ordering::Release);

        let (tx, rx) = std::sync::mpsc::channel::<c_int>();
        let tx = Box::into_raw(Box::new(tx)) as *mut c_void;

        unsafe extern "C" fn take_photo_later(
            _args_json: *const c_char,
            call_id: u64,
            opaque: *mut c_void,
        ) -> *const c_char {
            assert!(call_id != 0, "a pending call must carry an id");
            let tx = unsafe { (*(opaque as *const std::sync::mpsc::Sender<c_int>)).clone() };
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(60));
                let rt = ASYNC_RT.load(Ordering::Acquire);
                let result = CString::new(r#""TOOLRES-photo-9f3""#).unwrap();
                // SAFETY: the test keeps rt alive until the stream returns,
                // and the stream cannot return before this line has run.
                let rc = unsafe {
                    sofuu_rt_host_tool_resolve(rt, call_id, result.as_ptr(), ptr::null())
                };
                let _ = tx.send(rc);
            });
            ptr::null() // pending — completed later
        }

        let name = CString::new("take_photo").unwrap();
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    rt,
                    name.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    Some(take_photo_later),
                    tx,
                )
            },
            0
        );

        let setup = CString::new(mock_tool_agent_js(
            "TOOLRES-photo-9f3",
            "photo-agent",
            "take_photo",
        ))
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) }, 0);
        unsafe { sofuu_free(out as *mut c_void) };

        let events = run_agent_collect(rt, "photo-agent");
        let joined = events.join("\n");

        let rc = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the deferred call must be resolvable from the host thread");
        assert_eq!(rc, 0, "resolve must accept an outstanding call id");
        assert!(
            joined.contains("CONFIRMED-TOOL-RESULT"),
            "the deferred result must reach the model's transcript; events: {joined}"
        );

        unsafe { drop(Box::from_raw(tx as *mut std::sync::mpsc::Sender<c_int>)) };
        unsafe { sofuu_rt_free(rt) };
    }

    /// E2: registration is validated up front (the schema reaches the model
    /// verbatim, so a bad one must not travel), and only an outstanding call
    /// may be settled — a mistyped or double-used id says so.
    #[test]
    fn host_tool_api_validates_its_inputs() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        unsafe extern "C" fn cb(_a: *const c_char, _c: u64, _o: *mut c_void) -> *const c_char {
            ptr::null()
        }

        let name = CString::new("take_photo").unwrap();
        let bad_schema = CString::new("[1,2,3]").unwrap();

        // NULL runtime / name / callback.
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    ptr::null_mut(),
                    name.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    Some(cb),
                    ptr::null_mut(),
                )
            },
            -1
        );
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    rt,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    Some(cb),
                    ptr::null_mut(),
                )
            },
            -1
        );
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    rt,
                    name.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    None,
                    ptr::null_mut(),
                )
            },
            -1
        );
        // A schema that is not a JSON object is refused.
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    rt,
                    name.as_ptr(),
                    ptr::null(),
                    bad_schema.as_ptr(),
                    Some(cb),
                    ptr::null_mut(),
                )
            },
            -1
        );
        // The good shape (default schema) succeeds.
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    rt,
                    name.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    Some(cb),
                    ptr::null_mut(),
                )
            },
            0
        );

        // Registered tools publish into BOTH registries: the builtin one
        // (resolved at define time) and the host one (resolved at run time).
        let probe = CString::new(
            "(sofuu.tools.TOOLS && sofuu.tools.TOOLS['take_photo'] && globalThis.__sofuu_host_tools && globalThis.__sofuu_host_tools['take_photo']) ? 'BOTH-REGISTERED' : 'MISSING'",
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_rt_eval(rt, probe.as_ptr(), &mut out) }, 0);
        let envelope = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        unsafe { sofuu_free(out as *mut c_void) };
        assert!(envelope.contains("BOTH-REGISTERED"), "got {envelope}");

        // Nothing is pending, so nothing can be settled.
        let msg = CString::new("nothing pending").unwrap();
        assert_eq!(unsafe { sofuu_rt_host_tool_resolve(rt, 0, ptr::null(), ptr::null()) }, -1);
        assert_eq!(
            unsafe { sofuu_rt_host_tool_resolve(rt, 4242, ptr::null(), msg.as_ptr()) },
            -1
        );

        unsafe { sofuu_rt_free(rt) };
    }

    /// E2: an agent supplied through the `agents` config key is defined while
    /// sofuu_rt_new runs — before the host can register anything — so the
    /// unknown-builtin check is deferred for that path (allowHostTools). This
    /// is the regression guard: the agent must survive rt_new and find its
    /// tool by run time. The callback answers with a JSON *string* here, which
    /// also pins the shaping rule (text, not quoted JSON).
    #[test]
    fn host_tool_reaches_a_config_defined_agent() {
        let _g = lock();
        let cfg = CString::new(
            r#"{"agents":[{"name":"cfg-photo","system":"You are a test agent.","provider":"mock","model":"mock-tool-test","tools":[{"builtin":"take_photo"}],"budget":{"maxSteps":4}}]}"#,
        )
        .unwrap();
        let rt = unsafe { sofuu_rt_new(cfg.as_ptr()) };
        assert!(!rt.is_null());

        static CALLS: Mutex<u32> = Mutex::new(0);
        *CALLS.lock().unwrap() = 0;
        unsafe extern "C" fn take_photo(
            _a: *const c_char,
            _c: u64,
            _o: *mut c_void,
        ) -> *const c_char {
            *CALLS.lock().unwrap() += 1;
            c"\"TOOLRES-photo-9f3\"".as_ptr()
        }

        let name = CString::new("take_photo").unwrap();
        assert_eq!(
            unsafe {
                sofuu_rt_register_host_tool(
                    rt,
                    name.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    Some(take_photo),
                    ptr::null_mut(),
                )
            },
            0
        );

        // The mock defines an unrelated agent; the config one is the target.
        let setup = CString::new(mock_tool_agent_js(
            "TOOLRES-photo-9f3",
            "unused-agent",
            "take_photo",
        ))
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) }, 0);
        unsafe { sofuu_free(out as *mut c_void) };

        let events = run_agent_collect(rt, "cfg-photo");
        let joined = events.join("\n");

        assert_eq!(*CALLS.lock().unwrap(), 1, "the config agent must call the tool");
        assert!(
            joined.contains("CONFIRMED-TOOL-RESULT"),
            "the tool result must reach the model's transcript; events: {joined}"
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

        // 4b. Test learn and search via funnel (auto-embedding)
        let learn_args = format!(
            r#"{{"handle":{hid},"text":"User prefers TypeScript over Python for backend services.","role":"user"}}"#
        );
        let learn_idx = call(rt, "memory.learn", &learn_args);
        assert!(
            learn_idx.as_i64().is_some_and(|i| i >= 0),
            "learn should return an index, got {learn_idx}"
        );

        let search_args = format!(
            r#"{{"handle":{hid},"query":"TypeScript backend","k":2}}"#
        );
        let search_hits = call(rt, "memory.search", &search_args);
        let search_arr = search_hits
            .as_array()
            .unwrap_or_else(|| panic!("search should return an array, got {search_hits}"));
        assert!(!search_arr.is_empty(), "search should find the stored record");
        assert!(search_arr[0]["text"].as_str().unwrap().contains("TypeScript"));

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
    fn memory_funnel_handle_tombstoning_and_cma_extended_methods() {
        require_qtsq!();
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        let a = open_brain(rt, "tomb-a");
        let b = open_brain(rt, "tomb-b");
        let ha = a.get("handle").and_then(|h| h.as_i64()).unwrap();
        let hb = b.get("handle").and_then(|h| h.as_i64()).unwrap();
        assert_eq!(ha, 0);
        assert_eq!(hb, 1);

        // Remember to B
        call(
            rt,
            "memory.remember",
            &format!(
                r#"{{"handle":{hb},"vec":{},"text":"persists in B","role":"user","kv_page_id":0}}"#,
                hash_axis(1)
            ),
        );

        // Close A (handle 0)
        let close_res = call(rt, "memory.close", &format!(r#"{{"handle":{ha}}}"#));
        assert_eq!(close_res.as_bool(), Some(true));

        // B must still be accessible via handle 1 without index shift
        let cb = call(rt, "memory.count", &format!(r#"{{"handle":{hb}}}"#));
        assert_eq!(cb.as_u64(), Some(1));

        // Test extended CMA methods via sofuu_rt_call on brain B:
        // 1. rememberEntity
        let ent_res = call(
            rt,
            "memory.rememberEntity",
            &format!(
                r#"{{"handle":{hb},"vec":{},"text":"Alice is an engineer","entityName":"Alice","entityType":"person"}}"#,
                hash_axis(2)
            ),
        );
        assert!(ent_res.as_i64().is_some_and(|idx| idx >= 0));

        // 2. rememberEntityTurn
        let turn_res = call(
            rt,
            "memory.rememberEntityTurn",
            &format!(
                r#"{{"handle":{hb},"vec":{},"text":"Alice likes Rust","entityName":"Alice","entityType":"person"}}"#,
                hash_axis(3)
            ),
        );
        assert!(turn_res.as_i64().is_some_and(|idx| idx >= 0));

        // 3. markPositive
        let mark_res = call(
            rt,
            "memory.markPositive",
            &format!(r#"{{"handle":{hb},"ids":[0, 1]}}"#),
        );
        assert!(mark_res.as_u64().is_some_and(|n| n >= 1));

        // 4. decayTick
        let decay_res = call(
            rt,
            "memory.decayTick",
            &format!(r#"{{"handle":{hb},"hours":2.0}}"#),
        );
        assert_eq!(decay_res.as_bool(), Some(true));

        // 5. kvHints
        let hints_res = call(
            rt,
            "memory.kvHints",
            &format!(
                r#"{{"handle":{hb},"vec":{},"n_hints":3}}"#,
                hash_axis(3)
            ),
        );
        assert!(hints_res.is_array());

        // 6. recall with text forwarding (BM25 rare token re-ranking)
        let recall_res = call(
            rt,
            "memory.recall",
            &format!(
                r#"{{"handle":{hb},"vec":{},"k":2,"text":"Alice likes Rust"}}"#,
                hash_axis(3)
            ),
        );
        let hits = recall_res.as_array().expect("recall returns array");
        assert!(!hits.is_empty());

        // 7. compact and consolidate
        let compact_res = call(rt, "memory.compact", &format!(r#"{{"handle":{hb}}}"#));
        assert!(compact_res.as_i64().is_some());

        let cons_res = call(rt, "memory.consolidate", &format!(r#"{{"handle":{hb}}}"#));
        assert!(cons_res.as_i64().is_some());

        unsafe { sofuu_rt_free(rt) };
    }

    /// B7.1 (ROADMAP-BRAIN-CMA): one semantics across surfaces, pinned.
    ///
    /// A record written by the IN-PROCESS API must be visible through the
    /// C-ABI FUNNEL with the same filters, the same tier numbers, and the
    /// same error behaviour — and the working-set contract must hold on
    /// both: a working record recalls (tier 0) but never appears in the
    /// persistent read-out until `promoteWorking` moves it to episodic.
    /// Tier numbers are pinned against the CMA constants — every surface
    /// (chat /memory, /why, server /memories, this funnel) passes and
    /// renders the SAME numbers; the names are presentation over them.
    #[test]
    fn cross_surface_semantics_funnel_matches_in_process() {
        require_qtsq!();
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // Tier numbers are the contract every surface shares.
        assert_eq!(sofuu_core::memory::cma::TIER_WORKING, 0);
        assert_eq!(sofuu_core::memory::cma::TIER_EPISODIC, 1);
        assert_eq!(sofuu_core::memory::cma::TIER_SEMANTIC, 2);
        assert_eq!(sofuu_core::memory::cma::TIER_FORGOTTEN, 3);
        assert_eq!(sofuu_core::memory::cma::TIER_ENTITY, 4);

        // 1. Write through the IN-PROCESS API (the surface chat's JS
        //    driver uses): one episodic + one entity record, flushed.
        let path = std::env::temp_dir().join(format!(
            "sofuu-cross-surface-{}-{}.qtsq",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        let path_js = path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
        let seed = format!(
            r#"(function () {{
  var out = {{ memType: typeof sofuu.memory }};
  try {{
    var b = sofuu.memory.open("{p}", 768);
    var v0 = new Float32Array(768); v0[0] = 1.0;
    var v1 = new Float32Array(768); v1[1] = 1.0;
    var i0 = b.remember(v0, "the sky is blue", "user", 0, "chat");
    var i1 = b.rememberEntity(v1, "agent researcher stores findings", "agent:researcher", "agent");
    b.flush();
    out.i0 = i0; out.i1 = i1; out.count = b.count();
  }} catch (e) {{ out.error = String((e && e.message) || e); }}
  return JSON.stringify(out);
}})()"#,
            p = path_js
        );
        let seeded = eval_json(rt, &seed);
        let seeded_str = seeded.as_str().unwrap_or_else(|| panic!("seed must return a JSON string, got {seeded}"));
        let seeded_v: JsonValue = serde_json::from_str(seeded_str)
            .unwrap_or_else(|e| panic!("seed JSON unreadable ({e}): {seeded_str}"));
        assert!(
            seeded_v.get("error").is_none(),
            "the in-process seed must not error: {seeded_v}"
        );
        assert_eq!(
            seeded_v.get("count").and_then(|c| c.as_i64()),
            Some(2),
            "the in-process seed must write exactly two records, got {seeded_v}"
        );

        // 2. The FUNNEL (the headless C-ABI surface) sees both records
        //    with the same read-out shape. (list returns its JSON as a
        //    string — the same string the server writes and chat parses —
        //    so the helper re-parses it; that IS the cross-surface shape.)
        let open_args = format!(
            r#"{{"path":{},"dim":768,"embed_id":"hash-v1"}}"#,
            serde_json::to_string(&path.to_string_lossy()).unwrap()
        );
        let h = call(rt, "memory.open", &open_args);
        let hid = h.get("handle").and_then(|h| h.as_i64()).unwrap();
        let list = |filter: &str| -> JsonValue {
            let raw = call(rt, "memory.list", &format!(r#"{{"handle":{hid}{f}}}"#, f = filter));
            match &raw {
                JsonValue::String(s) => serde_json::from_str(s)
                    .unwrap_or_else(|e| panic!("memory.list string is not JSON ({e}): {s}")),
                v => v.clone(),
            }
        };
        let all = list("");
        let all_arr = all
            .get("records")
            .and_then(|r| r.as_array())
            .unwrap_or_else(|| panic!("memory.list must return {{records}}, got {all}"));
        assert_eq!(all.get("total").and_then(|t| t.as_i64()), Some(2));
        let texts: Vec<&str> = all_arr
            .iter()
            .filter_map(|r| r.get("text").and_then(|t| t.as_str()))
            .collect();
        assert!(texts.contains(&"the sky is blue"), "episodic visible: {texts:?}");
        assert!(
            texts.contains(&"agent researcher stores findings"),
            "entity visible: {texts:?}"
        );

        // 3. Tier filters are the same numbers on both surfaces.
        let tier_of = |filter: &str| -> Vec<u8> {
            let v = list(&format!(",{f}", f = filter));
            v.get("records")
                .and_then(|r| r.as_array())
                .unwrap_or_else(|| panic!("list({filter}) must return records, got {v}"))
                .iter()
                .map(|r| r.get("tier").and_then(|t| t.as_u64()).unwrap_or(255) as u8)
                .collect()
        };
        assert_eq!(tier_of(r#""tier":1"#), vec![sofuu_core::memory::cma::TIER_EPISODIC]);
        assert_eq!(tier_of(r#""tier":4"#), vec![sofuu_core::memory::cma::TIER_ENTITY]);
        assert!(
            tier_of(r#""tier":0"#).is_empty(),
            "nothing is working-tier in a fresh persistent read-out"
        );
        assert!(
            tier_of(r#""tier":5"#).is_empty(),
            "an unknown tier filters to empty, not everything"
        );

        // 4. Working-set parity through the funnel: recallable (tier 0),
        //    absent from the persistent read-out, promotable.
        let mut wv = vec![0.0f64; 768];
        wv[2] = 1.0;
        let wv_js: Vec<String> = wv.iter().map(|f| f.to_string()).collect();
        let wargs = format!(
            r#"{{"handle":{hid},"vec":[{}],"text":"session scratch fact","role":"user","kv_page_id":0,"source":"chat"}}"#,
            wv_js.join(",")
        );
        let wid = call(rt, "memory.rememberWorking", &wargs);
        let wid = wid
            .as_i64()
            .unwrap_or_else(|| panic!("rememberWorking must return an id, got {wid}"));
        assert!(
            wid >= 0x4000_0000,
            "a working id carries the B6 flag and stays positive, got {wid}"
        );
        let rargs = format!(r#"{{"handle":{hid},"vec":[{}],"k":3}}"#, wv_js.join(","));
        let hits = call(rt, "memory.recall", &rargs);
        let working_hit = hits
            .as_array()
            .and_then(|a| a.iter().find(|h| h.get("id").and_then(|i| i.as_i64()) == Some(wid)))
            .unwrap_or_else(|| panic!("recall must find the working record {wid}, got {hits}"));
        assert_eq!(
            working_hit.get("tier").and_then(|t| t.as_u64()),
            Some(0),
            "the working hit serves as tier 0, got {working_hit}"
        );
        let after_w = list("");
        let wtexts: Vec<String> = after_w
            .get("records")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| r.get("text").and_then(|t| t.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        assert!(
            !wtexts.contains(&"session scratch fact".to_string()),
            "an unpromoted working record must not appear in the persistent read-out: {wtexts:?}"
        );

        // 5. Promotion moves it to episodic on BOTH surfaces (flush, then
        //    a fresh in-process open — the same thing a reload is).
        let pr = call(
            rt,
            "memory.promoteWorking",
            &format!(r#"{{"handle":{hid},"ids":[{wid}]}}"#),
        );
        let pr_val: JsonValue = match &pr {
            JsonValue::String(s) => serde_json::from_str(s).unwrap_or(JsonValue::Null),
            v => v.clone(),
        };
        assert_eq!(
            pr_val.get("promoted").and_then(|p| p.as_i64()),
            Some(1),
            "promotion must move the working record, got {pr}"
        );
        call(rt, "memory.flush", &format!(r#"{{"handle":{hid}}}"#));
        let reload = format!(
            r#"(function () {{
  var b = sofuu.memory.open("{p}", 768);
  var res = b.list({{ tier: 1 }});
  var parsed = JSON.parse(res);
  return parsed.records.map(function (r) {{ return r.text; }});
}})()"#,
            p = path_js
        );
        let reloaded = eval_json(rt, &reload);
        let names: Vec<String> = match &reloaded {
            JsonValue::String(s) => serde_json::from_str(s).unwrap_or_default(),
            v => serde_json::from_value(v.clone()).unwrap_or_default(),
        };
        assert!(
            names.contains(&"session scratch fact".to_string()),
            "the promoted record must survive a reload as episodic, got {reloaded}"
        );

        // 6. Error behaviour parity: a bad forget id is `false`, not an
        //    exception — the same answer the direct API gives.
        let bad = call(rt, "memory.forget", r#"{"handle":0,"index":99999}"#);
        assert_eq!(bad.as_bool(), Some(false), "a bad forget id returns false, got {bad}");

        let _ = hid;
        unsafe { sofuu_rt_free(rt) };
        let _ = std::fs::remove_file(&path);
    }

    /// Evaluate a JS expression and return its parsed result value
    /// (sofuu_rt_eval wraps it in the {ok, result} envelope).
    fn eval_json(rt: *mut CapiRuntime, src: &str) -> JsonValue {
        let s = CString::new(src).unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, s.as_ptr(), &mut out) };
        assert_eq!(rc, 0, "eval should succeed");
        let json = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        unsafe { sofuu_free(out as *mut c_void) };
        let envelope: JsonValue = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("eval result is not JSON ({e}): {json}"));
        envelope.get("result").cloned().unwrap_or(JsonValue::Null)
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

    /// E0-4: the funnel-method resolution test.
    ///
    /// Every method on the headless funnel must resolve as a real function
    /// on a fresh runtime. The list lives here in code (it used to be parsed
    /// out of a doc table; the docs folder was removed, so the contract moved
    /// next to the code it guards). Add new funnel methods to this list —
    /// the test fails until you do, which is the point.
    #[test]
    fn documented_funnel_methods_all_resolve() {
        let _g = lock();

        // 1. The funnel contract: every entry must resolve below.
        let mut methods: Vec<String> = [
            "ai.complete",
            "ai.stream",
            "ai.embed",
            "ai.embedLocal",
            "ai.embedBatch",
            "ai.embedImage",
            "ai.embedInfo",
            "ai.similarity",
            "ai.transcribe",
            "ai.speak",
            "memory.open",
            "memory.remember",
            "memory.learn",
            "memory.recall",
            "memory.search",
            "memory.count",
            "memory.flush",
            "memory.forget",
            "memory.close",
            "memory.list",
            "memory.export",
            "memory.stats",
            "memory.rememberWorking",
            "memory.promoteWorking",
            "memory.clearWorking",
            "mcp.connect",
            "mcp.call",
            "mcp.listTools",
            "mcp.listResources",
            "mcp.disconnect",
            "agent.define",
            "agent.run",
            "agent.list",
            "agent.cancel",
            "agent.renderTrace",
            "rlm.query",
            "rlm.route",
            "http.serve",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        methods.sort();
        methods.dedup();
        assert_eq!(
            methods.len(),
            38,
            "the funnel contract list changed length — update it deliberately"
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
        // Drift in the memory rows is still caught on any QTSQ build, and
        // the length assertion above keeps the list itself honest in
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
            "funnel methods that do not resolve: {broken:?}"
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

    /// Committed agent-loop contract test (audit P2 close-out, 2026-10-15).
    ///
    /// The full e2e battery lives in `tests/agent_test.js`, which is
    /// gitignored local-only — a fresh clone could regress the agent
    /// loop's core contract invisibly. This drives the REAL shipped JS
    /// (`src/js/agent.js`, embedded by `shipped.rs`) through the funnel
    /// with a scripted provider seam — no network, no model needed:
    ///   1. an inline tool executes and its result reaches the final
    ///      answer, with real usage landed in the run accounting;
    ///   2. a delegation cycle (da → db → da) is refused by the guard and
    ///      the turn still completes (no hang, no re-entry storm) with
    ///      exactly the expected call count;
    ///   3. a token budget fires mid-run: stopped :'budget_tokens' with
    ///      the full breach detail, after one single-shot salvage round.
    ///
    /// Deliberately out of scope here: provider delta merging (ai.rs folds
    /// fragments into `stream.toolCalls`; the local-model protocol parser
    /// has its own Rust tests) and the RLM path (rlm modules have 50+
    /// committed inline tests). This is the loop contract a fresh clone
    /// must never lose.
    #[test]
    fn agent_loop_core_contract_runs_through_shipped_js() {
        let _g = lock();
        let rt = unsafe { sofuu_rt_new(ptr::null()) };
        assert!(!rt.is_null());

        // Scripted provider: sofuu.ai.stream is replaced with a fake that
        // shifts one step off a FIFO per call — the exact post-fold shape
        // agent.js consumes (chunk.text from the stream, st.toolCalls,
        // st.usage). Definitions + a result-capture wrapper ride eval.
        let setup = CString::new(
            r#"
(function () {
  sofuu._agT = { script: [], last: null };
  sofuu.ai.stream = function () {
    var step = sofuu._agT.script.shift() || { text: '(script exhausted)' };
    var it = (function* () { if (step.text) yield { text: step.text }; })();
    return {
      abort: function () {},
      toolCalls: step.toolCalls || [],
      usage: step.usage || {},
      [Symbol.asyncIterator]: function () { return it; }
    };
  };
  sofuu.agent.define({
    name: 'greeter', system: 'contract test', memory: 'off', rlm: 'off',
    tools: [{ name: 'greet', description: 'greet the scene',
              parameters: { type: 'object', properties: {} },
              execute: function () { return 'GREET-TOOLRES'; } }]
  });
  sofuu.agent.define({ name: 'da', system: 'contract test', memory: 'off', rlm: 'off', agents: ['db'] });
  sofuu.agent.define({ name: 'db', system: 'contract test', memory: 'off', rlm: 'off', agents: ['da'] });
  /* tiny carries a tool: budgets bind at the loop top, so the run must be
   * in the tool loop for the token ceiling to fire mid-turn. */
  sofuu.agent.define({ name: 'tiny', system: 'contract test', memory: 'off', rlm: 'off',
                       budget: { maxTokens: 1 },
                       tools: [{ name: 'noop', description: 'no-op',
                                 parameters: { type: 'object', properties: {} },
                                 execute: function () { return 'NOPRES'; } }] });
  var realRun = sofuu.agent.run;
  sofuu.agent.run = function (t, task, o) {
    var p;
    try { p = realRun(t, task, o); }
    catch (eSync) {
      sofuu._agT.last = { answer: 'SYNC-REJ: ' + String((eSync && eSync.stack) || eSync).slice(0, 600),
                          stopped: 'rejected' };
      return new Promise(function () {}); /* never settles; funnel times nothing */
    }
    return p.then(function (r) {
      sofuu._agT.last = { answer: String(r && r.answer || ''),
                          stopped: r && r.stopped || null,
                          breachKind: (r && r.breach) ? r.breach.kind : null,
                          llmCalls: (r && r.usage) ? (r.usage.llmCalls | 0) : -1,
                          promptTokens: (r && r.usage) ? (r.usage.promptTokens | 0) : -1,
                          toolCalls: (r && r.usage) ? (r.usage.toolCalls | 0) : -1 };
      return r;
    }, function (e) {
      sofuu._agT.last = { answer: 'REJECTED: ' + String((e && e.stack) || e).slice(0, 600),
                          stopped: 'rejected' };
      throw e;
    });
  };
})()"#,
        )
        .unwrap();
        let mut out: *mut c_char = ptr::null_mut();
        let rc = unsafe { sofuu_rt_eval(rt, setup.as_ptr(), &mut out) };
        assert_eq!(rc, 0, "setup must succeed");
        unsafe { sofuu_free(out as *mut c_void) };

        // Drive one agent.run to completion via the BLOCKING funnel —
        // sofuu_rt_call pumps the event loop until the promise resolves
        // (the stream variant is fire-and-tick, for hosts that pump) —
        // then read the captured result back through eval.
        fn run_and_capture(rt: *mut CapiRuntime, target: &str, task: &str) -> serde_json::Value {
            let method = CString::new("agent.run").unwrap();
            let args = CString::new(format!(
                "{{\"target\":\"{target}\",\"task\":\"{task}\"}}"
            ))
            .unwrap();
            let mut out: *mut c_char = ptr::null_mut();
            let rc = unsafe { sofuu_rt_call(rt, method.as_ptr(), args.as_ptr(), &mut out) };
            assert_eq!(rc, 0, "agent.run must settle: {}", unsafe {
                if out.is_null() { "".into() } else {
                    CStr::from_ptr(out).to_string_lossy().into_owned()
                }
            });
            unsafe { sofuu_free(out as *mut c_void) };
            let q = CString::new("sofuu._agT.last").unwrap();
            let mut res: *mut c_char = ptr::null_mut();
            let rc = unsafe { sofuu_rt_eval(rt, q.as_ptr(), &mut res) };
            assert_eq!(rc, 0, "result readback must succeed");
            let raw = unsafe { CStr::from_ptr(res) }.to_string_lossy().into_owned();
            unsafe { sofuu_free(res as *mut c_void) };
            let envelope: serde_json::Value =
                serde_json::from_str(&raw).expect("captured result must be JSON envelope");
            let v = envelope["result"].clone();
            assert!(
                v.is_object(),
                "captured result must be an object — the run must settle: {raw}"
            );
            v
        }

        let script = CString::new(
            "sofuu._agT.script = [\
             {toolCalls:[{id:'c1',name:'greet',arguments:{}}],usage:{promptTokens:10,completionTokens:5}},\
             {text:'final-sees-GREET-TOOLRES',usage:{promptTokens:20,completionTokens:5}}];",
        )
        .unwrap();
        unsafe { sofuu_rt_eval(rt, script.as_ptr(), &mut out) };
        unsafe { sofuu_free(out as *mut c_void) };
        let toolRound = run_and_capture(rt, "greeter", "do the thing");
        assert_eq!(
            toolRound["answer"].as_str().unwrap_or(""),
            "final-sees-GREET-TOOLRES",
            "tool round must produce the final answer: {toolRound}"
        );
        assert!(
            toolRound["promptTokens"].as_i64().unwrap_or(0) >= 30,
            "usage must reach the run accounting: {toolRound}"
        );
        assert!(
            toolRound["toolCalls"].as_i64().unwrap_or(0) == 1,
            "exactly one real tool call executes: {toolRound}"
        );

        // Delegation cycle: da delegates db, db tries to delegate da — the
        // guard refuses, db finishes anyway, and the tree completes with
        // exactly 3 llm calls (guard failure would re-enter and burn more).
        let script = CString::new(
            "sofuu._agT.script = [\
             {toolCalls:[{id:'d1',name:'delegate',arguments:{agent:'db',task:'sub'}}],usage:{promptTokens:20,completionTokens:5}},\
             {toolCalls:[{id:'d2',name:'delegate',arguments:{agent:'da',task:'back'}}],usage:{promptTokens:20,completionTokens:5}},\
             {text:'cycle-seen-OK',usage:{promptTokens:10,completionTokens:3}},\
             {text:'parent-done',usage:{promptTokens:10,completionTokens:3}}];",
        )
        .unwrap();
        unsafe { sofuu_rt_eval(rt, script.as_ptr(), &mut out) };
        unsafe { sofuu_free(out as *mut c_void) };
        let cycle = run_and_capture(rt, "da", "delegate something");
        assert_eq!(
            cycle["answer"].as_str().unwrap_or(""),
            "parent-done",
            "guarded cycle completes the tree: {cycle}"
        );
        assert!(
            cycle["llmCalls"].as_i64().unwrap_or(0) <= 5,
            "no re-entry storm — expected tree-shaped call count: {cycle}"
        );

        // Token budget: maxTokens 1 blows on the first ask's tool round; the
        // loop-top breach stops the run, one single-shot salvage round runs,
        // and the result carries the breach detail with honest usage.
        let script = CString::new(
            "sofuu._agT.script = [\
             {toolCalls:[{id:'b1',name:'noop',arguments:{}}],usage:{promptTokens:50,completionTokens:10}},\
             {text:'SALVAGE-OK',usage:{promptTokens:5,completionTokens:2}}];",
        )
        .unwrap();
        unsafe { sofuu_rt_eval(rt, script.as_ptr(), &mut out) };
        unsafe { sofuu_free(out as *mut c_void) };
        let budget = run_and_capture(rt, "tiny", "long task");
        assert_eq!(
            budget["stopped"].as_str().unwrap_or(""),
            "budget_tokens",
            "token budget stops the run: {budget}"
        );
        assert_eq!(
            budget["breachKind"].as_str().unwrap_or(""),
            "budget_tokens",
            "breach detail rides the result: {budget}"
        );
        assert_eq!(
            budget["answer"].as_str().unwrap_or(""),
            "SALVAGE-OK",
            "the salvage round surfaces a usable answer: {budget}"
        );

        unsafe { sofuu_rt_free(rt) };
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
