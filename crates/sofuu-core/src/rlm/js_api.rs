// rlm/js_api.rs — JS-facing API for the RLM loop (PLAN-RLM R2).
//
// The engine-facing seam `sofuu_rust_register_engine_js` (called from
// engine.c under SOFUU_RUST_CORE) registers seven `__rlm_*` host functions
// on the QuickJS global object and then evals the shipped JS driver
// (src/js/rlm.js), which wraps them into the async `sofuu.rlm` API. This is
// the headless entry point: no chat/TUI code on this path.
//
// Wire protocol (all stringly-typed, matching the __chat_* precedent in
// chat.rs — these functions never throw across the FFI boundary):
//
//   __rlm_new(reqJson)            → {"id":"7"} | {"error":"..."}
//   __rlm_start(id)               → action JSON | {"error":...}
//   __rlm_step(id, reply)         → action JSON | {"error":...}
//   __rlm_feed_llm(id, answers)   → action JSON | {"error":...}
//   __rlm_free(id)                → bool
//   __rlm_route(ctxTok, winTok, question) → "rlm" | "plain"
//   __rlm_log_route(jsonLine)     → bool (silent failure)
//
// Action JSON always carries a `kind` tag:
//   {"kind":"messages","messages":[...]}
//   {"kind":"resolve_llm","prompts":[...],"messages":[...]}
//   {"kind":"done","result":{answer,calls,rounds,ms,depth_reached,stopped,trace?}}
// (`trace` is included only when the episode's opts.trace is set.)

use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::Deserialize;
use sofuu_ffi::bridge::{
    js_new_bool, js_new_string, js_to_string, register_global_fn, JSCFunction, JSContext,
    JSValue, JSValueConst,
};
use sofuu_ffi::qjs;

use super::episode::{Episode, EpisodeAction};
use super::router::{self, Route};
use super::{RlmOpts, RlmRequest, ToolSpec};

/// The shipped async driver — single source of truth is src/js/rlm.js.
const DRIVER_JS: &str = include_str!("../../../../src/js/rlm.js");

/* ── Episode registry ────────────────────────────────────────────
 * Episode owns a nested QuickJS sandbox (raw pointers, !Send), so — like
 * the sandbox-state registry in sofuu-ffi/src/qjs_rt.rs — the map stores
 * Box pointers as usize. Every callback that dereferences them runs on the
 * single QuickJS engine thread that created them. */

fn registry() -> &'static Mutex<HashMap<u64, usize>> {
    static REG: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Look an episode up by its JS-side id string and run `f` on it.
fn with_episode<R>(id: &str, f: impl FnOnce(&mut Episode) -> R) -> Result<R, String> {
    let id_num: u64 = id
        .trim()
        .parse()
        .map_err(|_| format!("rlm: bad episode id '{id}'"))?;
    let ptr = registry()
        .lock()
        .ok()
        .and_then(|reg| reg.get(&id_num).copied())
        .ok_or_else(|| format!("rlm: unknown episode id {id_num}"))?;
    if ptr == 0 {
        return Err(format!("rlm: unknown episode id {id_num}"));
    }
    // SAFETY: the pointer was Box::into_raw'd by js_rlm_new and is only
    // removed (and dropped) by js_rlm_free; callbacks run on the engine
    // thread, so no concurrent &mut aliases exist.
    Ok(f(unsafe { &mut *(ptr as *mut Episode) }))
}

fn err_json(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

/* ── Request deserialization ───────────────────────────────────── */

#[derive(Debug, Deserialize)]
struct RequestJson {
    context: String,
    question: String,
    #[serde(default)]
    opts: OptsJson,
}

/// camelCase on the wire (`{"maxDepth":3, ...}`); None = RlmOpts default.
/// NOTE: provider credentials (`api_key`, `base_url`) are deliberately NOT
/// fields here — the JS driver consumes them for the provider call while
/// the Rust core never sees them, so they can't leak into traces/logs (G3).
/// `tools` is schema-only (PLAN-AGENTS A4): the EXECUTORS stay JS-side
/// (`opts.execTool` in the driver) and never cross this boundary.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct OptsJson {
    provider: String,
    model: String,
    max_depth: Option<u32>,
    max_llm_calls: Option<u32>,
    max_wall_ms: Option<u64>,
    eval_slice_ms: Option<u64>,
    max_rounds: Option<u32>,
    chunk_chars: Option<usize>,
    trace: bool,
    tools: Vec<ToolSpec>,
    max_tool_calls: Option<u32>,
}

fn apply<T: Copy>(slot: &mut T, v: Option<T>) {
    if let Some(v) = v {
        *slot = v;
    }
}

impl From<RequestJson> for RlmRequest {
    fn from(r: RequestJson) -> Self {
        let mut opts = RlmOpts {
            provider: r.opts.provider,
            model: r.opts.model,
            tools: r.opts.tools,
            ..RlmOpts::default()
        };
        apply(&mut opts.max_depth, r.opts.max_depth);
        apply(&mut opts.max_llm_calls, r.opts.max_llm_calls);
        apply(&mut opts.max_wall_ms, r.opts.max_wall_ms);
        apply(&mut opts.eval_slice_ms, r.opts.eval_slice_ms);
        apply(&mut opts.max_rounds, r.opts.max_rounds);
        apply(&mut opts.chunk_chars, r.opts.chunk_chars);
        apply(&mut opts.max_tool_calls, r.opts.max_tool_calls);
        opts.trace = r.opts.trace;
        RlmRequest {
            context: r.context,
            question: r.question,
            opts,
        }
    }
}

/* ── Action serialization ──────────────────────────────────────── */

fn action_json(action: &EpisodeAction, include_trace: bool) -> String {
    let v = match action {
        EpisodeAction::SendMessages(messages) => {
            serde_json::json!({ "kind": "messages", "messages": messages })
        }
        EpisodeAction::ResolveLlm {
            prompts,
            pending_messages,
        } => serde_json::json!({
            "kind": "resolve_llm",
            "prompts": prompts,
            "messages": pending_messages,
        }),
        EpisodeAction::ResolveTool {
            calls,
            pending_messages,
        } => serde_json::json!({
            "kind": "resolve_tools",
            "calls": calls,
            "messages": pending_messages,
        }),
        EpisodeAction::Done(result) => {
            let mut r = serde_json::to_value(result)
                .unwrap_or_else(|_| serde_json::json!({ "answer": "", "calls": 0 }));
            if !include_trace {
                if let Some(o) = r.as_object_mut() {
                    o.remove("trace");
                }
            }
            serde_json::json!({ "kind": "done", "result": r })
        }
    };
    v.to_string()
}

/* ── __rlm_* host functions ──────────────────────────────────────
 * Same shape as chat.rs's js_chat_*: no unwrap/expect on user input, no
 * panicking paths (a Rust panic across a QuickJS C frame is UB). */

/// `__rlm_new(reqJson)` → `{"id":"7"}` | `{"error":"..."}`. Depth is 0 —
/// recursive nested episodes are managed by the JS layer in a later chunk.
unsafe extern "C" fn js_rlm_new(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let arg = if argc >= 1 {
        js_to_string(ctx, *argv)
    } else {
        None
    };
    let out = rlm_new(arg);
    js_new_string(ctx, &out)
}

fn rlm_new(arg: Option<String>) -> String {
    let Some(raw) = arg else {
        return err_json("rlm: request JSON required");
    };
    let reqj: RequestJson = match serde_json::from_str(&raw) {
        Ok(r) => r,
        Err(e) => return err_json(&format!("rlm: bad request JSON: {e}")),
    };
    match Episode::new(reqj.into(), 0) {
        Err(e) => err_json(&e),
        Ok(ep) => {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let boxed = Box::into_raw(Box::new(ep)) as usize;
            match registry().lock() {
                Ok(mut reg) => {
                    reg.insert(id, boxed);
                    serde_json::json!({ "id": id.to_string() }).to_string()
                }
                Err(_) => {
                    // SAFETY: never inserted — reclaim and drop it here.
                    drop(unsafe { Box::from_raw(boxed as *mut Episode) });
                    err_json("rlm: registry lock poisoned")
                }
            }
        }
    }
}

/// `__rlm_start(id)` → first action (always `messages`) | {"error":...}.
unsafe extern "C" fn js_rlm_start(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let id = if argc >= 1 {
        js_to_string(ctx, *argv).unwrap_or_default()
    } else {
        String::new()
    };
    let out = with_episode(&id, |ep| {
        let a = ep.start();
        action_json(&a, ep.wants_trace())
    })
    .unwrap_or_else(|e| err_json(&e));
    js_new_string(ctx, &out)
}

/// `__rlm_step(id, reply)` — feed one model reply; → next action.
unsafe extern "C" fn js_rlm_step(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let id = if argc >= 1 {
        js_to_string(ctx, *argv).unwrap_or_default()
    } else {
        String::new()
    };
    let reply = if argc >= 2 {
        js_to_string(ctx, *argv.add(1)).unwrap_or_default()
    } else {
        String::new()
    };
    let out = with_episode(&id, |ep| {
        let a = ep.step(&reply);
        action_json(&a, ep.wants_trace())
    })
    .unwrap_or_else(|e| err_json(&e));
    js_new_string(ctx, &out)
}

/// `__rlm_feed_llm(id, answersJson)` — one answer per prompt of the last
/// `resolve_llm` action, in order; → next action.
unsafe extern "C" fn js_rlm_feed_llm(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let id = if argc >= 1 {
        js_to_string(ctx, *argv).unwrap_or_default()
    } else {
        String::new()
    };
    let raw = if argc >= 2 {
        js_to_string(ctx, *argv.add(1)).unwrap_or_default()
    } else {
        String::new()
    };
    let out = match serde_json::from_str::<Vec<String>>(&raw) {
        Err(e) => err_json(&format!("rlm: answers must be a JSON string array: {e}")),
        Ok(answers) => with_episode(&id, |ep| {
            let a = ep.feed_llm_answers(answers);
            action_json(&a, ep.wants_trace())
        })
        .unwrap_or_else(|e| err_json(&e)),
    };
    js_new_string(ctx, &out)
}

/// `__rlm_feed_tools(id, resultsJson)` — one result string per call of
/// the last `resolve_tools` action, in order; → next action. Tool results
/// are plain strings (the agent tool path's normalized output), so there
/// is no parse beyond the outer array.
unsafe extern "C" fn js_rlm_feed_tools(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let id = if argc >= 1 {
        js_to_string(ctx, *argv).unwrap_or_default()
    } else {
        String::new()
    };
    let raw = if argc >= 2 {
        js_to_string(ctx, *argv.add(1)).unwrap_or_default()
    } else {
        String::new()
    };
    let out = match serde_json::from_str::<Vec<String>>(&raw) {
        Err(e) => err_json(&format!("rlm: tool results must be a JSON string array: {e}")),
        Ok(results) => with_episode(&id, |ep| {
            let a = ep.feed_tool_results(results);
            action_json(&a, ep.wants_trace())
        })
        .unwrap_or_else(|e| err_json(&e)),
    };
    js_new_string(ctx, &out)
}

/// `__rlm_free(id)` → drops the episode (idempotent). The JS driver calls
/// this in a `finally` — episodes never leak.
unsafe extern "C" fn js_rlm_free(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let id = if argc >= 1 {
        js_to_string(ctx, *argv).unwrap_or_default()
    } else {
        String::new()
    };
    let id_num: Option<u64> = id.trim().parse().ok();
    let ptr = id_num.and_then(|n| registry().lock().ok().and_then(|mut reg| reg.remove(&n)));
    if let Some(p) = ptr {
        if p != 0 {
            // SAFETY: the pointer came from Box::into_raw in js_rlm_new and
            // is removed from the registry above, so this is the unique
            // owning reference; dropped on the engine thread that owns it.
            drop(unsafe { Box::from_raw(p as *mut Episode) });
        }
    }
    js_new_bool(ctx, ptr.is_some())
}

/// Like `bridge::js_to_string` but coerces (JS_ToCString semantics), so
/// numeric JS args arrive as their decimal string. Used by `__rlm_route`
/// and any numeric-accepting host fn.
unsafe fn js_arg_string(ctx: *mut JSContext, v: JSValueConst) -> String {
    // SAFETY: ctx live; v from this ctx; the returned pointer is freed
    // after copying (same contract as bridge::js_to_string).
    unsafe {
        let p = sofuu_ffi::bridge::sofuu_js_to_cstring(ctx, v);
        if p.is_null() {
            return String::new();
        }
        let s = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
        sofuu_ffi::bridge::sofuu_js_free_cstring(ctx, p);
        s
    }
}

/// `__rlm_route(ctxTokens, windowTokens, question)` → "rlm" | "plain".
/// Numeric args are coerced to their decimal string, then parsed.
unsafe extern "C" fn js_rlm_route(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let arg_str = |i: c_int| -> String {
        if i < argc {
            unsafe { js_arg_string(ctx, *argv.add(i as usize)) }
        } else {
            String::new()
        }
    };
    let ctx_tokens = arg_str(0).trim().parse::<i64>().unwrap_or(0).max(0) as usize;
    let win_tokens = arg_str(1).trim().parse::<i64>().unwrap_or(0).max(0) as usize;
    let question = arg_str(2);
    let route = router::route(ctx_tokens, win_tokens, &question);
    js_new_string(ctx, route.as_str())
}

/// `__rlm_log_route(jsonLine)` — append one decision to
/// ~/.sofuu/routing_log.jsonl. Never errors out loudly: logging must not
/// break a turn (missing HOME, unwritable dir → false).
unsafe extern "C" fn js_rlm_log_route(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let ok = if argc >= 1 {
        match js_to_string(ctx, *argv) {
            Some(raw) => rlm_log_route(&raw),
            None => false,
        }
    } else {
        false
    };
    js_new_bool(ctx, ok)
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct LogLine {
    ctx_tokens: usize,
    window_tokens: usize,
    question: String,
    route: String,
    latency_ms: Option<u64>,
    rlm_calls: Option<u32>,
}

fn rlm_log_route(raw: &str) -> bool {
    /* config_dir("") honors the embedded config_root (PLAN-HEADLESS H2.4). */
    let dir = PathBuf::from(crate::embed_config::config_dir(""));
    if std::fs::create_dir_all(&dir).is_err() {
        return false;
    }
    rlm_log_route_to(&dir.join("routing_log.jsonl"), raw)
}

fn rlm_log_route_to(path: &std::path::Path, raw: &str) -> bool {
    let Ok(line) = serde_json::from_str::<LogLine>(raw) else {
        return false;
    };
    let route = match line.route.as_str() {
        "rlm" => Route::Rlm,
        _ => Route::Plain,
    };
    router::log_decision_full(
        path,
        line.ctx_tokens,
        line.window_tokens,
        &line.question,
        route,
        line.latency_ms,
        line.rlm_calls,
    )
    .is_ok()
}

/* ── Engine seam (the only symbol engine.c knows) ──────────────── */

/// Called at the end of `engine_register_builtins` (engine.c, guarded by
/// SOFUU_RUST_CORE). Registers the `__rlm_*` globals, then evals the shipped
/// driver src/js/rlm.js, which installs `sofuu.rlm`.
///
/// # Safety
/// `ctx` must be the live QuickJS context of a SofuuEngine, called on the
/// engine thread during startup.
#[no_mangle]
pub unsafe extern "C" fn sofuu_rust_register_engine_js(ctx: *mut c_void) {
    if ctx.is_null() {
        return;
    }
    let ctx = ctx as *mut JSContext;
    // SAFETY: ctx is the engine's live context; all fns are 'static.
    unsafe {
        register_global_fn(ctx, "__rlm_new", js_rlm_new as JSCFunction);
        register_global_fn(ctx, "__rlm_start", js_rlm_start as JSCFunction);
        register_global_fn(ctx, "__rlm_step", js_rlm_step as JSCFunction);
        register_global_fn(ctx, "__rlm_feed_llm", js_rlm_feed_llm as JSCFunction);
        register_global_fn(ctx, "__rlm_feed_tools", js_rlm_feed_tools as JSCFunction);
        register_global_fn(ctx, "__rlm_free", js_rlm_free as JSCFunction);
        register_global_fn(ctx, "__rlm_route", js_rlm_route as JSCFunction);
        register_global_fn(ctx, "__rlm_log_route", js_rlm_log_route as JSCFunction);

        let src = std::ffi::CString::new(DRIVER_JS).unwrap_or_default();
        let r = qjs::JS_Eval(
            ctx,
            src.as_ptr(),
            DRIVER_JS.len(),
            c"<rlm-driver>".as_ptr(),
            qjs::JS_EVAL_TYPE_GLOBAL,
        );
        if qjs::is_exception(r) {
            // A driver failure must never take the engine down — the JS
            // guards make this unreachable in practice; log and move on.
            let exc = qjs::JS_GetException(ctx);
            let p = qjs::sofuu_js_to_cstring(ctx, exc);
            if !p.is_null() {
                let msg = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
                qjs::sofuu_js_free_cstring(ctx, p);
                eprintln!("[sofuu] rlm driver eval failed: {msg}");
            }
            qjs::sofuu_js_free_value(ctx, exc);
        }
        qjs::sofuu_js_free_value(ctx, r);

        // Shipped drivers beyond RLM (PLAN-AGENTS): the agent runtime
        // (sofuu.agent.define/run/…) and web search (sofuu.web.search/open).
        // Pure JS over the registered primitives — no host functions. The
        // seam runs after memory.rs registered sofuu.agent.create/prefetch,
        // and both drivers merge into the live namespace objects.
        crate::shipped::eval_shipped(
            ctx as *mut c_void,
            crate::shipped::AGENT_JS,
            "<agent-driver>",
        );
        crate::shipped::eval_shipped(ctx as *mut c_void, crate::shipped::WEB_JS, "<web-driver>");
        crate::shipped::eval_shipped(ctx as *mut c_void, crate::shipped::TOOLS_JS, "<tools-driver>");
    }
}

/* ── Tests ───────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_json_maps_camelcase_opts() {
        let r: RlmRequest = serde_json::from_str::<RequestJson>(
            r#"{"context":"c","question":"q","opts":{"maxDepth":5,"maxLlmCalls":7,"maxWallMs":9000,"maxRounds":4,"chunkChars":800,"trace":true,"model":"m"}}"#,
        )
        .expect("parse")
        .into();
        assert_eq!(r.opts.max_depth, 5);
        assert_eq!(r.opts.max_llm_calls, 7);
        assert_eq!(r.opts.max_wall_ms, 9000);
        assert_eq!(r.opts.max_rounds, 4);
        assert_eq!(r.opts.chunk_chars, 800);
        assert!(r.opts.trace);
        assert_eq!(r.opts.model, "m");
        // Omitted opts keep the R1 defaults.
        let r2: RlmRequest = serde_json::from_str::<RequestJson>(
            r#"{"context":"c","question":"q"}"#,
        )
        .expect("parse")
        .into();
        assert_eq!(r2.opts.max_depth, 3);
        assert_eq!(r2.opts.max_llm_calls, 24);
        assert!(!r2.opts.trace);
    }

    #[test]
    fn malformed_new_returns_error_json() {
        assert!(rlm_new(None).contains("\"error\""));
        let out = rlm_new(Some("not json".to_string()));
        assert!(out.contains("\"error\""), "{out}");
        let v: serde_json::Value = serde_json::from_str(&out).expect("error json parses");
        assert!(v["error"].as_str().unwrap().contains("bad request JSON"));
    }

    /// Registry smoke test through a REAL QuickJS context: register the
    /// __rlm_* globals on a bare JS_NewContext, then drive an episode from
    /// JS and assert the action JSON round-trips. (QjsSandbox's ctx field is
    /// private, so the test builds its own context via the qjs bindings —
    /// the __rlm_* functions only need a valid ctx, not a sandbox.)
    #[test]
    fn js_roundtrip_through_quickjs() {
        // SAFETY: single-threaded test; ctx/rt torn down at the end.
        unsafe {
            let rt = qjs::JS_NewRuntime();
            assert!(!rt.is_null());
            let ctx = qjs::JS_NewContext(rt);
            assert!(!ctx.is_null());
            register_global_fn(ctx, "__rlm_new", js_rlm_new as JSCFunction);
            register_global_fn(ctx, "__rlm_start", js_rlm_start as JSCFunction);
            register_global_fn(ctx, "__rlm_step", js_rlm_step as JSCFunction);
            register_global_fn(ctx, "__rlm_feed_llm", js_rlm_feed_llm as JSCFunction);
            register_global_fn(ctx, "__rlm_feed_tools", js_rlm_feed_tools as JSCFunction);
            register_global_fn(ctx, "__rlm_free", js_rlm_free as JSCFunction);
            register_global_fn(ctx, "__rlm_route", js_rlm_route as JSCFunction);

            let script = r#"
(function() {
  var bad = JSON.parse(__rlm_new('{'));
  if (!bad.error) throw new Error('malformed must error');
  var created = JSON.parse(__rlm_new(JSON.stringify({
    context: 'alpha beta gamma delta'.repeat(20), question: 'say hi', opts: { trace: true }
  })));
  if (!created.id) throw new Error('no id: ' + JSON.stringify(created));
  var a1 = JSON.parse(__rlm_start(created.id));
  if (a1.kind !== 'messages') throw new Error('start kind: ' + a1.kind);
  if (a1.messages.length !== 2) throw new Error('want 2 messages, got ' + a1.messages.length);
  if (a1.messages[0].role !== 'system') throw new Error('first message not system');
  var a2 = JSON.parse(__rlm_step(created.id, '```js\nfinal("all done")\n```'));
  if (a2.kind !== 'done') throw new Error('step kind: ' + a2.kind);
  if (a2.result.answer !== 'all done') throw new Error('answer: ' + a2.result.answer);
  if (a2.result.rounds !== 1) throw new Error('rounds: ' + a2.result.rounds);
  if (!Array.isArray(a2.result.trace)) throw new Error('trace missing with trace:true');
  var kinds = a2.result.trace.map(function(e) { return e.kind; });
  if (kinds.indexOf('final') < 0) throw new Error('no final event: ' + kinds);
  if (__rlm_free(created.id) !== true) throw new Error('free');
  if (__rlm_free(created.id) !== false) throw new Error('double free must be false');
  var gone = JSON.parse(__rlm_step(created.id, 'x'));
  if (!gone.error) throw new Error('freed episode must error');
  if (__rlm_route(900, 1000, 'where is X') !== 'rlm') throw new Error('route rlm');
  if (__rlm_route(100, 1000, 'where is X') !== 'plain') throw new Error('route plain');
  return 'ok';
})()
            "#;
            // QuickJS's reader peeks past input_len in places — pass a
            // NUL-terminated buffer (same discipline as qjs_rt.rs's eval;
            // a bare str::as_ptr here parses GARBAGE past the end).
            let c_script = std::ffi::CString::new(script).unwrap_or_default();
            let r = qjs::JS_Eval(
                ctx,
                c_script.as_ptr(),
                script.len(),
                c"<rlm-jsapi-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            if qjs::is_exception(r) {
                let exc = qjs::JS_GetException(ctx);
                let p = qjs::sofuu_js_to_cstring(ctx, exc);
                let msg = if p.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                };
                if !p.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, p);
                }
                qjs::sofuu_js_free_value(ctx, exc);
                qjs::sofuu_js_free_value(ctx, r);
                qjs::JS_FreeContext(ctx);
                qjs::JS_FreeRuntime(rt);
                panic!("JS driver smoke threw: {msg}");
            }
            let ctp = qjs::CtxPtr::new(ctx);
            let out = qjs::value_to_string(ctp, r);
            qjs::sofuu_js_free_value(ctx, r);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
            assert_eq!(out.as_deref(), Some("ok"));
        }
        // The JS freed its episode; the registry must be empty of high ids
        // only indirectly observable — assert via handle behavior already
        // covered above (double-free → false).
    }

    /// A4 full form through a REAL QuickJS context: tools in the request
    /// JSON → documented in the system prompt → snippet tool() call →
    /// resolve_tools action → feed → done with tool_calls accounted.
    #[test]
    fn js_tools_roundtrip_through_quickjs() {
        // SAFETY: single-threaded test; ctx/rt torn down at the end.
        unsafe {
            let rt = qjs::JS_NewRuntime();
            assert!(!rt.is_null());
            let ctx = qjs::JS_NewContext(rt);
            assert!(!ctx.is_null());
            register_global_fn(ctx, "__rlm_new", js_rlm_new as JSCFunction);
            register_global_fn(ctx, "__rlm_start", js_rlm_start as JSCFunction);
            register_global_fn(ctx, "__rlm_step", js_rlm_step as JSCFunction);
            register_global_fn(ctx, "__rlm_feed_tools", js_rlm_feed_tools as JSCFunction);
            register_global_fn(ctx, "__rlm_free", js_rlm_free as JSCFunction);

            let script = r#"
(function() {
  var created = JSON.parse(__rlm_new(JSON.stringify({
    context: 'filler text. '.repeat(50),
    question: 'use the tool',
    opts: { trace: true, tools: [
      { name: 'echo', description: 'echoes the message',
        parameters: { type: 'object', properties: { m: { type: 'string' } } } }
    ] }
  })));
  var a1 = JSON.parse(__rlm_start(created.id));
  if (a1.messages[0].content.indexOf('- echo — echoes the message') < 0)
    throw new Error('tool not documented in system prompt');
  var a2 = JSON.parse(__rlm_step(created.id,
    '```js\nvar r = tool("echo", { m: "hi" }); "got " + r\n```'));
  if (a2.kind !== 'resolve_tools') throw new Error('kind: ' + a2.kind);
  if (a2.calls.length !== 1 || a2.calls[0].name !== 'echo')
    throw new Error('calls: ' + JSON.stringify(a2.calls));
  if (a2.calls[0].args.m !== 'hi') throw new Error('args: ' + JSON.stringify(a2.calls[0]));
  var a3 = JSON.parse(__rlm_feed_tools(created.id, '["ECHOED"]'));
  if (a3.kind !== 'messages') throw new Error('after feed: ' + a3.kind);
  var last = a3.messages[a3.messages.length - 1].content;
  if (last.indexOf('got ECHOED') < 0) throw new Error('result not in transcript: ' + last);
  var a4 = JSON.parse(__rlm_step(created.id, '```js\nfinal("done")\n```'));
  if (a4.kind !== 'done') throw new Error('kind: ' + a4.kind);
  if (a4.result.tool_calls !== 1) throw new Error('tool_calls: ' + a4.result.tool_calls);
  if (__rlm_free(created.id) !== true) throw new Error('free');
  return 'ok';
})()
            "#;
            let c_script = std::ffi::CString::new(script).unwrap_or_default();
            let r = qjs::JS_Eval(
                ctx,
                c_script.as_ptr(),
                script.len(),
                c"<rlm-jsapi-tools-test>".as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL,
            );
            if qjs::is_exception(r) {
                let exc = qjs::JS_GetException(ctx);
                let p = qjs::sofuu_js_to_cstring(ctx, exc);
                let msg = if p.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                };
                if !p.is_null() {
                    qjs::sofuu_js_free_cstring(ctx, p);
                }
                qjs::sofuu_js_free_value(ctx, exc);
                qjs::sofuu_js_free_value(ctx, r);
                qjs::JS_FreeContext(ctx);
                qjs::JS_FreeRuntime(rt);
                panic!("JS tools roundtrip threw: {msg}");
            }
            let ctp = qjs::CtxPtr::new(ctx);
            let out = qjs::value_to_string(ctp, r);
            qjs::sofuu_js_free_value(ctx, r);
            qjs::JS_FreeContext(ctx);
            qjs::JS_FreeRuntime(rt);
            assert_eq!(out.as_deref(), Some("ok"));
        }
    }

    #[test]
    fn trace_stripped_unless_opted_in() {
        // Build a done-action with and without trace to check the strip.
        let res = super::super::RlmResult {
            answer: "a".into(),
            calls: 0,
            tool_calls: 0,
            depth_reached: 0,
            rounds: 1,
            ms: 3,
            trace: vec![super::super::RlmEvent {
                kind: "final".into(),
                t: 1,
                repr: "a".into(),
            }],
            dropped_events: 0,
            stopped: None,
        };
        let a = EpisodeAction::Done(res);
        let with = action_json(&a, true);
        let without = action_json(&a, false);
        assert!(with.contains("\"trace\""));
        assert!(!without.contains("\"trace\""), "{without}");
        assert!(without.contains("\"answer\":\"a\""));
    }

    #[test]
    fn log_route_appends_via_router() {
        let dir = std::env::temp_dir().join(format!(
            "sofuu-rlm-jsapi-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("routing_log.jsonl");
        let _ = std::fs::remove_file(&path);

        assert!(!rlm_log_route_to(&path, "not json"));
        assert!(rlm_log_route_to(
            &path,
            r#"{"ctxTokens":5000,"windowTokens":8192,"question":"summarize all","route":"rlm","latencyMs":4100,"rlmCalls":9}"#
        ));
        let text = std::fs::read_to_string(&path).expect("log written");
        let v: serde_json::Value = serde_json::from_str(text.trim()).expect("valid jsonl");
        assert_eq!(v["route"], "rlm");
        assert_eq!(v["ctx_tokens"], 5000);
        assert_eq!(v["latency_ms"], 4100);
        assert_eq!(v["rlm_calls"], 9);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// G3.10 — credentials passed to the JS provider layer must never
    /// cross into Rust-carried strings: OptsJson drops unknown fields at
    /// the door, and the result trace + router JSONL are structurally
    /// incapable of holding them. This test guards the regression where a
    /// future api_key field lands in RlmOpts and leaks.
    #[test]
    fn api_credentials_never_reach_results_or_logs() {
        let raw = r#"{
            "context": "deploy memo: region eu-west, image 3.1",
            "question": "what region is mentioned?",
            "opts": {
                "provider": "openai",
                "model": "m",
                "api_key": "sk-DECOY-123",
                "base_url": "https://decoy.example/v1",
                "trace": true
            }
        }"#;
        let req: RlmRequest = serde_json::from_str::<RequestJson>(raw).expect("parse").into();
        let opts_dbg = format!("{:?}", req.opts);
        assert!(!opts_dbg.contains("sk-DECOY-123"), "opts carry the key: {opts_dbg}");
        assert!(!opts_dbg.contains("decoy.example"), "opts carry base_url: {opts_dbg}");

        // Drive a full episode (scripted fake) and serialize the result.
        let mut ep = Episode::new(req, 0).expect("episode");
        let result = ep.run(&mut |_msgs| "```js\nfinal(\"eu-west\")\n```".to_string());
        let ser = serde_json::to_string(&result).expect("serialize");
        assert!(!ser.contains("sk-DECOY-123"), "result leaks key: {ser}");
        assert!(!ser.contains("decoy.example"), "result leaks base_url: {ser}");

        // The router JSONL line for the same turn holds only schema fields.
        let dir = std::env::temp_dir()
            .join(format!("sofuu-rlm-redact-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("routing_log.jsonl");
        router::log_decision_full(&path, 42, 8192, "what region is mentioned?", Route::Rlm, Some(7), Some(0))
            .expect("log");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(!text.contains("sk-DECOY-123"), "log leaks key: {text}");
        assert!(!text.contains("decoy.example"), "log leaks base_url: {text}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
