// rt/async_contract.rs — the R2 shared async contract (SPEC-RUNTIME-CONTRACT §3).
//
// One error taxonomy, one error shape, one deadline-inheritance rule, and the
// bounded-queue limits, callable from every surface (JS, funnel, native) so
// subsystems cannot drift into private error shapes. Enforcement stays in the
// callers (the agent loop, the tool drivers, the stream pump); this module is
// the contract they must speak.
//
// JS surface (JSON in, JSON out):
//   sofuu.async.codes()                         the taxonomy + retryability
//   sofuu.async.makeError({code, op, …})        the §1.4 error shape
//   sofuu.async.childDeadline({parent_ms, child_ms, margin_ms?})
//                                               nested deadline inheritance
//   sofuu.async.limits()                        bounded-queue + timeout defaults
//
// Phase P2 (R2.5/R2.2/R2.4). Pure functions; no engine behavior change.

use serde_json::{json, Value};

/// The §1.4 taxonomy with the retryability each code carries by default.
pub const CODES: [(&str, bool, &str); 10] = [
    ("E_CAPABILITY", false, "absent feature (R1)"),
    ("E_TIMEOUT", true, "deadline exceeded (R2)"),
    ("E_CANCELLED", false, "caller cancelled (R2)"),
    ("E_BACKPRESSURE", true, "consumer too slow (R2)"),
    ("E_RATE_LIMIT", true, "provider throttled"),
    ("E_AUTH", false, "credential rejected/expired"),
    ("E_NETWORK", true, "transport failure"),
    ("E_PROTOCOL", false, "wire format violation"),
    ("E_SCHEMA", true, "structured output invalid"),
    ("E_INTERNAL", false, "engine bug"),
];

fn default_retryable(code: &str) -> Option<bool> {
    CODES.iter().find(|(c, _, _)| *c == code).map(|(_, r, _)| *r)
}

/// Build the §1.4 error shape. An unknown code wraps as E_INTERNAL rather
/// than inventing a new taxonomy entry.
pub fn make_error(args: &Value) -> Value {
    let code_in = args.get("code").and_then(Value::as_str).unwrap_or("");
    let (code, note) = match default_retryable(code_in) {
        Some(_) => (code_in.to_string(), None),
        None => (
            "E_INTERNAL".to_string(),
            Some(format!("unknown code {code_in:?}")),
        ),
    };
    let retryable = args
        .get("retryable")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| default_retryable(&code).unwrap_or(false));
    let mut out = json!({
        "code": code,
        "op": args.get("op").and_then(Value::as_str).unwrap_or("?"),
        "surface": args.get("surface").and_then(Value::as_str).unwrap_or("js"),
        "retryable": retryable,
        "message": args.get("message").and_then(Value::as_str).unwrap_or(""),
    });
    if let Some(t) = args.get("deadline_ms").and_then(Value::as_u64) {
        out["deadline_ms"] = json!(t);
    }
    if let Some(t) = args.get("elapsed_ms").and_then(Value::as_u64) {
        out["elapsed_ms"] = json!(t);
    }
    if let Some(t) = args.get("turn_id").and_then(Value::as_str) {
        out["turn_id"] = json!(t);
    }
    if let Some(t) = args.get("seq").and_then(Value::as_u64) {
        out["seq"] = json!(t);
    }
    if let Some(n) = note {
        out["note"] = json!(n);
    }
    out
}

/// R2.2/R2.3: a child inherits the parent's remaining budget minus a fixed
/// margin, so nested work can never outlive its parent.
pub fn child_deadline(args: &Value) -> Value {
    let parent = args.get("parent_ms").and_then(Value::as_u64);
    let child = args.get("child_ms").and_then(Value::as_u64);
    let margin = args.get("margin_ms").and_then(Value::as_u64).unwrap_or(1000);
    let (Some(parent), Some(child)) = (parent, child) else {
        return make_error(&json!({
            "code": "E_SCHEMA", "op": "async.childDeadline",
            "message": "parent_ms and child_ms are required"
        }));
    };
    if parent <= margin {
        return make_error(&json!({
            "code": "E_TIMEOUT", "op": "async.childDeadline",
            "deadline_ms": parent, "elapsed_ms": parent,
            "message": format!("parent budget {parent} ms leaves no margin ({margin} ms)"),
        }));
    }
    let effective = child.min(parent - margin);
    json!({
        "ok": true,
        "deadline_ms": effective,
        "inherited": effective < child,
        "parent_ms": parent,
        "margin_ms": margin,
    })
}

/// R2.4: the bounded-queue limits and the per-operation default deadlines
/// (SPEC §3.2). Callers enforce them; this is the single place they live.
pub fn limits() -> Value {
    json!({
        "max_queue_events": 1024,
        "max_queue_bytes": 4194304,
        "deadline_margin_ms": 1000,
        "default_timeouts": {
            "ai_call_ms": 120000,
            "ttft_ms": 30000,
            "tool_ms": 60000,
            "memory_ms": 5000,
            "http_ms": 30000,
            "spawn_ms": 60000
        }
    })
}

// ── JS surface ───────────────────────────────────────────────────────────────

use std::ffi::CString;
use std::os::raw::c_int;
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

fn json_str(v: &str) -> CString {
    CString::new(v).unwrap_or_default()
}

unsafe fn read_json_arg(ctx: *mut JSContext, arg: JSValueConst) -> Option<Value> {
    let mut len = 0usize;
    let c = qjs::JS_ToCStringLen2(ctx, &mut len, arg, 0);
    if c.is_null() {
        return None;
    }
    let raw = std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, c);
    serde_json::from_str::<Value>(&raw).ok()
}

unsafe extern "C" fn js_async_make_error(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = if argc < 1 {
        make_error(&json!({"code": "E_INTERNAL", "op": "async.makeError",
            "message": "makeError requires an options object"}))
    } else {
        read_json_arg(ctx, *argv)
            .map(|a| make_error(&a))
            .unwrap_or_else(|| make_error(&json!({"code": "E_SCHEMA",
                "op": "async.makeError", "message": "argument must be a JSON object"})))
    };
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_async_child_deadline(
    ctx: *mut JSContext, _this: JSValueConst, argc: c_int, argv: *const JSValueConst,
) -> JSValue {
    let out = if argc < 1 {
        make_error(&json!({"code": "E_SCHEMA", "op": "async.childDeadline",
            "message": "childDeadline requires {parent_ms, child_ms}"}))
    } else {
        read_json_arg(ctx, *argv)
            .map(|a| child_deadline(&a))
            .unwrap_or_else(|| make_error(&json!({"code": "E_SCHEMA",
                "op": "async.childDeadline", "message": "argument must be a JSON object"})))
    };
    let cs = json_str(&serde_json::to_string(&out).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_async_limits(
    ctx: *mut JSContext, _this: JSValueConst, _argc: c_int, _argv: *const JSValueConst,
) -> JSValue {
    let cs = json_str(&serde_json::to_string(&limits()).unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

unsafe extern "C" fn js_async_codes(
    ctx: *mut JSContext, _this: JSValueConst, _argc: c_int, _argv: *const JSValueConst,
) -> JSValue {
    let codes: Vec<Value> = CODES
        .iter()
        .map(|(c, r, d)| json!({"code": c, "retryable": r, "description": d}))
        .collect();
    let cs = json_str(&serde_json::to_string(&json!({ "codes": codes }))
        .unwrap_or_else(|_| "{}".into()));
    qjs::sofuu_js_new_string(ctx, cs.as_ptr())
}

/// Register `sofuu.async.*`. Called from `engine_register_builtins`.
pub unsafe fn mod_async_contract_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"sofuu".as_ptr(),
            qjs::sofuu_js_dup_value(ctx, sofuu_obj),
        );
    }
    let obj = qjs::sofuu_js_new_object(ctx);
    let funcs: [(&std::ffi::CStr, unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue); 4] = [
        (c"makeError", js_async_make_error),
        (c"childDeadline", js_async_child_deadline),
        (c"limits", js_async_limits),
        (c"codes", js_async_codes),
    ];
    for (name, f) in funcs {
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            name.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, f, name.as_ptr(), 1),
        );
    }
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"async".as_ptr(), obj);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_shape_matches_the_spec() {
        let e = make_error(&json!({
            "code": "E_TIMEOUT", "op": "ai.stream", "surface": "http",
            "deadline_ms": 120000, "elapsed_ms": 120003,
            "message": "no first token", "turn_id": "t1", "seq": 41
        }));
        assert_eq!(e["code"], "E_TIMEOUT");
        assert_eq!(e["retryable"], true); // default per taxonomy
        assert_eq!(e["op"], "ai.stream");
        assert_eq!(e["deadline_ms"], 120000);
        assert_eq!(e["turn_id"], "t1");
        // Unknown codes wrap, they do not invent taxonomy entries.
        let w = make_error(&json!({"code": "E_WAT", "op": "x", "retryable": true}));
        assert_eq!(w["code"], "E_INTERNAL");
        assert!(w["note"].as_str().unwrap().contains("E_WAT"));
    }

    #[test]
    fn nested_deadline_inherits_and_refuses_exhausted_parents() {
        // Child within the parent budget: untouched.
        let r = child_deadline(&json!({"parent_ms": 10000, "child_ms": 5000}));
        assert_eq!(r["deadline_ms"], 5000);
        assert_eq!(r["inherited"], false);
        // Child beyond the budget: clamped to parent minus margin.
        let r = child_deadline(&json!({"parent_ms": 10000, "child_ms": 60000}));
        assert_eq!(r["deadline_ms"], 9000);
        assert_eq!(r["inherited"], true);
        // Exhausted parent: typed timeout, never a negative deadline.
        let r = child_deadline(&json!({"parent_ms": 500, "child_ms": 60000}));
        assert_eq!(r["code"], "E_TIMEOUT");
    }
}
