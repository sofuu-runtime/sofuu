// ml — the context-economy models (PLAN-ML-GATES).
//
// Four separate tiny on-device models (~8.5k params each, offline,
// deterministic, zero-API) that ADVISE the agent loop on what comes in,
// what actions get taken, what stays, and what gets trusted — never
// filtering, capping, or deleting (Principle 1).
//
// Phase 1 surface: the shared in-memory context working set (context.rs)
// + the rule-based supervisor subset (exact repeat calls, re-reads of
// unchanged files). Phase 2 added the trained freshness gate under
// freshness/ (baked weights, advise-only notices); Phase 3 added the
// trained compaction gate under compaction/ (scored, budgeted passes —
// the model selects, the caller acts); Phase 4 added the trained
// relevance gate under relevance/ (pre-retrieval use/skip advice);
// Phase 5 added the trained supervisor under supervisor/ (pre-call +
// loop-boundary checkpoints layered over the Phase-1 rules).
//
// JS surface (registered by mod_ml_register, wired in rt/engine.rs):
//   sofuu.ml.track(eventJson)          — feed the working set
//   sofuu.ml.workset()                 — aggregate counters (JSON string)
//   sofuu.ml.info()                    — models + working set state (JSON)
//   sofuu.ml.feedback(json)            — outcome/wrong labels (§13 online)
//   sofuu.ml.supervisor.check(json)    — pre-call checkpoint (rules + net)
//   sofuu.ml.supervisor.loop(json)     — loop-boundary checkpoint (net)
//   sofuu.ml.freshness.score(text, task, optsJson?) — staleness verdict
//   sofuu.ml.compaction.plan(segmentsJson, optsJson?) — compaction pass
//   sofuu.ml.relevance.plan(candsJson, optsJson?) — pre-retrieval advice
//
// Everything returns JSON STRINGS parsed on the JS side (same contract as
// sofuu.ai.modelCaps). SOFUU_NO_ML=1 skips registration entirely; every JS
// caller wraps calls in try/catch so a gate failure can never break a turn.

pub mod compaction;
pub mod context;
pub mod freshness;
pub mod net;
pub mod online;
pub mod relevance;
pub mod supervisor;

use std::ffi::{CStr, CString, c_int};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

use crate::rt::ai::json_escape;

/* ------------------------------------------------------------------ */
/* Small JSON helpers (serde_json is already a core dependency)         */
/* ------------------------------------------------------------------ */

fn s(v: &serde_json::Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

fn n(v: &serde_json::Value, key: &str) -> u32 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0) as u32
}

fn b(v: &serde_json::Value, key: &str) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(false)
}

/// Read a JS string argument into an owned String ("" when not a string).
unsafe fn arg_str(ctx: *mut JSContext, val: JSValueConst) -> String {
    if qjs::sofuu_js_is_string(val) == 0 {
        return String::new();
    }
    let ptr = qjs::sofuu_js_to_cstring(ctx, val);
    if ptr.is_null() {
        return String::new();
    }
    let out = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, ptr);
    out
}

fn parse_obj(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or(serde_json::Value::Null)
}

fn ret_json(ctx: *mut JSContext, json: String) -> JSValue {
    match CString::new(json) {
        Ok(c) => unsafe { qjs::sofuu_js_new_string(ctx, c.as_ptr()) },
        Err(_) => unsafe { qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()) },
    }
}

/* ------------------------------------------------------------------ */
/* sofuu.ml.track(eventJson) — feed the working set                     */
/*                                                                      */
/* kinds:                                                               */
/*   run_start   {run, task}                                            */
/*   run_end     {run}                                                  */
/*   tool_result {run, step, tool, target, chars, error}                */
/*   segment     {run, kind:0-3, chars, tokens, tool, target}           */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ml_track(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_undefined();
    }
    let v = parse_obj(&arg_str(ctx, *argv));
    match s(&v, "kind").as_str() {
        "run_start" => context::run_start(&s(&v, "run"), &s(&v, "task")),
        "run_end" => context::run_end(&s(&v, "run")),
        "tool_result" => context::postcall(
            &s(&v, "run"),
            n(&v, "step"),
            &s(&v, "tool"),
            &s(&v, "target"),
            n(&v, "chars"),
            b(&v, "error"),
        ),
        "segment" => context::track_segment(
            &s(&v, "run"),
            n(&v, "kind").min(3) as u8,
            n(&v, "chars"),
            n(&v, "tokens"),
            &s(&v, "tool"),
            &s(&v, "target"),
        ),
        _ => {}
    }
    qjs::sofuu_js_undefined()
}

/* ------------------------------------------------------------------ */
/* sofuu.ml.workset() → {"runs":..,"calls":..,"segments":..,"segTokens":..} */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ml_workset(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let (runs, calls, segs, seg_tokens) = context::summary();
    ret_json(
        ctx,
        format!(
            "{{\"runs\":{runs},\"calls\":{calls},\"segments\":{segs},\"segTokens\":{seg_tokens}}}"
        ),
    )
}

/* ------------------------------------------------------------------ */
/* sofuu.ml.info() → per-model state + working set counters             */
/* All four gates report "on" (trained, baked weights).                 */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ml_info(
    ctx: *mut JSContext,
    _this: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let (runs, calls, segs, seg_tokens) = context::summary();
    let (on_enabled, on_obs, on_examples, on_adapted, on_adaptations) = online::status();
    ret_json(
        ctx,
        format!(
            "{{\"version\":1,\"gates\":{{\"freshness\":\"on\",\"relevance\":\"on\",\"supervisor\":\"on\",\"compaction\":\"on\"}},\"online\":{{\"enabled\":{on_enabled},\"observations\":{on_obs},\"examples\":{on_examples},\"adapted\":{on_adapted},\"adaptations\":{on_adaptations}}},\"workset\":{{\"runs\":{runs},\"calls\":{calls},\"segments\":{segs},\"segTokens\":{seg_tokens}}}}}"
        ),
    )
}

/* ------------------------------------------------------------------ */
/* sofuu.ml.supervisor.check(json) — pre-call checkpoint                */
/* in:  {"run","step","tool","sig","target","argsText",                */
/*       "skipTargets":[...],"budget","task"}                          */
/* out: {"ok":bool,"reason":"dup_call"|"reread_unchanged"|            */
/*       "near_dup"|"too_broad"|"off_task"|"spinning"|                */
/*       "over_budget"|"skip_advised"|"waste_risk"|"",                */
/*       "nudge":".."|null,"score":0.123,"source":"rule"|"model"|""}  */
/* The rule layer (context.rs) speaks first; where it is silent the    */
/* trained net speaks at THRESHOLD. Advise-only: the call still runs.  */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ml_supervisor_check(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return ret_json(ctx, "{\"ok\":true,\"reason\":\"\",\"nudge\":null,\"score\":0,\"source\":\"\"}".to_string());
    }
    let v = parse_obj(&arg_str(ctx, *argv));
    let skip: Vec<String> = v
        .get("skipTargets")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let verdict = supervisor::model::check(
        &s(&v, "run"),
        n(&v, "step"),
        &s(&v, "tool"),
        &s(&v, "sig"),
        &s(&v, "target"),
        &s(&v, "argsText"),
        &skip,
        n(&v, "budget"),
        &s(&v, "task"),
    );
    let nudge = match &verdict.nudge {
        Some(t) => format!("\"{}\"", json_escape(Some(t))),
        None => "null".to_string(),
    };
    ret_json(
        ctx,
        format!(
            "{{\"ok\":{},\"reason\":\"{}\",\"nudge\":{},\"score\":{:.4},\"source\":\"{}\"}}",
            verdict.ok,
            json_escape(Some(&verdict.reason)),
            nudge,
            verdict.score,
            verdict.source
        ),
    )
}

/* ------------------------------------------------------------------ */
/* sofuu.ml.supervisor.loop(json) — loop-boundary checkpoint            */
/* in:  {"run","step","budget"}                                         */
/* out: same shape as check(); reasons carry the loop_ prefix           */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ml_supervisor_loop(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return ret_json(ctx, "{\"ok\":true,\"reason\":\"\",\"nudge\":null,\"score\":0,\"source\":\"\"}".to_string());
    }
    let v = parse_obj(&arg_str(ctx, *argv));
    let verdict = supervisor::model::loop_check(&s(&v, "run"), n(&v, "step"), n(&v, "budget"));
    let nudge = match &verdict.nudge {
        Some(t) => format!("\"{}\"", json_escape(Some(t))),
        None => "null".to_string(),
    };
    ret_json(
        ctx,
        format!(
            "{{\"ok\":{},\"reason\":\"{}\",\"nudge\":{},\"score\":{:.4},\"source\":\"{}\"}}",
            verdict.ok,
            json_escape(Some(&verdict.reason)),
            nudge,
            verdict.score,
            verdict.source
        ),
    )
}

/* ------------------------------------------------------------------ */
/* sofuu.ml.feedback(json) — outcome/feedback labels for online learning */
/* (§13). kinds:                                                        */
/*   outcome {model:"supervisor", run, step, wasted}                    */
/*   wrong   {model:"supervisor"}  — the most recent flag was a mistake */
/* ------------------------------------------------------------------ */

unsafe extern "C" fn js_ml_feedback(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return ret_json(ctx, "{\"ok\":false}".to_string());
    }
    let v = parse_obj(&arg_str(ctx, *argv));
    let ok = if s(&v, "model") == "supervisor" {
        match s(&v, "kind").as_str() {
            "outcome" => online::label_outcome(&s(&v, "run"), n(&v, "step"), b(&v, "wasted")),
            "wrong" => online::label_wrong(),
            _ => false,
        }
    } else {
        false
    };
    ret_json(ctx, format!("{{\"ok\":{ok}}}"))
}

/* ------------------------------------------------------------------ */
/* Module registration                                                  */
/* ------------------------------------------------------------------ */

thread_local! {
    static ML_FUNCS: [qjs::JSCFunctionListEntry; 4] = [
        qjs::JSCFunctionListEntry {
            name: c"track".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ml_track },
        },
        qjs::JSCFunctionListEntry {
            name: c"workset".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_ml_workset },
        },
        qjs::JSCFunctionListEntry {
            name: c"info".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 0, cproto: 0, _pad: [0; 6], cfunc: js_ml_info },
        },
        qjs::JSCFunctionListEntry {
            name: c"feedback".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ml_feedback },
        },
    ];
    static SUPERVISOR_FUNCS: [qjs::JSCFunctionListEntry; 2] = [
        qjs::JSCFunctionListEntry {
            name: c"check".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ml_supervisor_check },
        },
        qjs::JSCFunctionListEntry {
            name: c"loop".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_ml_supervisor_loop },
        },
    ];
}

/// # Safety
/// `ctx` must be the live engine context (called once at boot).
#[no_mangle]
pub unsafe extern "C" fn mod_ml_register(ctx: *mut JSContext) {
    /* SOFUU_NO_ML=1 disables the whole gate layer — registration is
     * skipped, agent.js sees no sofuu.ml and behaves exactly as before. */
    if std::env::var("SOFUU_NO_ML")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return;
    }

    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());

    if qjs::is_undefined(sofuu) {
        sofuu = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(ctx, global, c"sofuu".as_ptr(), qjs::sofuu_js_dup_value(ctx, sofuu));
    }

    let ml_obj = qjs::sofuu_js_new_object(ctx);
    let ml_funcs = ML_FUNCS.with(|f| f.as_ptr());
    qjs::JS_SetPropertyFunctionList(ctx, ml_obj, ml_funcs, ML_FUNCS.with(|f| f.len()) as c_int);

    let sup_obj = qjs::sofuu_js_new_object(ctx);
    let sup_funcs = SUPERVISOR_FUNCS.with(|f| f.as_ptr());
    qjs::JS_SetPropertyFunctionList(ctx, sup_obj, sup_funcs, SUPERVISOR_FUNCS.with(|f| f.len()) as c_int);
    qjs::sofuu_js_set_property_str(ctx, ml_obj, c"supervisor".as_ptr(), sup_obj);

    /* Freshness gate (§5): trained net, baked weights — score(text, task). */
    freshness::model::register(ctx, ml_obj);

    /* Compaction gate (§12): trained net, baked weights — plan(segments). */
    compaction::model::register(ctx, ml_obj);

    /* Relevance gate (§6): trained net, baked weights — plan(candidates). */
    relevance::model::register(ctx, ml_obj);

    qjs::sofuu_js_set_property_str(ctx, sofuu, c"ml".as_ptr(), ml_obj);

    qjs::sofuu_js_free_value(ctx, sofuu);
    qjs::sofuu_js_free_value(ctx, global);
}
