// ml/freshness/model.rs — the trained freshness gate (PLAN-ML-GATES §5).
//
// The runtime half of the freshness model: the SAME TinyMlp forward pass
// the trainer exercised (ml/net.rs — train/serve skew is structurally
// impossible), weights baked at compile time from the blob ml-train wrote
// after the acceptance gates passed (§10), and a JS shim so agent.js can
// ask "does this content look stale for this task?" The answer ADVISES
// the turn (one notice on the ephemeral context message, §16) — it never
// filters, caps, or deletes anything (Principle 1).

use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::features::{self, FreshnessInput, SourceKind, FRESHNESS_FEATURES};
use crate::ml::cap_str;
use crate::ml::net::TinyMlp;

pub const IN_DIM: u32 = FRESHNESS_FEATURES as u32; // 28
pub const H1: u32 = 112;
pub const H2: u32 = 48;
pub const PARAMS: u32 = 8721; // §5 arch: 28 → 112 → 48 → 1

/// Decision threshold, baked from the ml-train run that emitted
/// weights_v1.f32. Chosen on the VALIDATION fold only (§10 bar 3) at the
/// 0.95 recall-selection floor; the held-out test fold held recall 0.990
/// and precision 1.000 at this value.
pub const THRESHOLD: f32 = 0.95;

static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");

static NET: LazyLock<TinyMlp> = LazyLock::new(|| {
    // The blob is committed and CRC-checked by eval.rs on every test run;
    // a corrupt bake is a build-time mistake, not a runtime condition.
    TinyMlp::from_blob(WEIGHTS).expect("committed freshness weights must load")
});

pub struct FreshnessVerdict {
    /// P(possibly stale for this task) in (0, 1).
    pub score: f32,
    /// score ≥ THRESHOLD — the advise-the-turn bit.
    pub stale: bool,
    /// Apparent years behind the current year (0 when no year signal).
    pub years: f32,
    /// Short human-readable evidence for the notice ("dated 2019",
    /// "staleness wording", …) — empty when the net fired on the
    /// interaction alone.
    pub reason: String,
}

/// Score one piece of content against the current task. Deterministic for
/// a fixed `now_year` (scalar f32, fixed order — §10 bar 7).
pub fn score(
    text: &str,
    task: &str,
    kind: SourceKind,
    strength: f32,
    age_days: f32,
    now_year: u32,
) -> FreshnessVerdict {
    let inp = FreshnessInput { text, task, kind, strength, age_days, now_year };
    let mut feats = features::extract(&inp);
    // Phase 1.2: NaN must never reach the forward pass (sigmoid(NaN) is
    // NaN → "NaN" in the returned JSON → the caller's parse breaks).
    crate::ml::sanitize_features(&mut feats);
    let s = NET.forward(&feats);
    FreshnessVerdict {
        score: s,
        stale: s >= THRESHOLD,
        years: (feats[4] * 10.0).min(20.0), // years_behind_norm is ÷10, capped
        reason: evidence(&feats, text, now_year),
    }
}

/// Name the dominant evidence channel so the notice can say WHY. Priority
/// follows the feature design: explicit old date > staleness vocabulary >
/// hedging > legacy/version markers.
fn evidence(feats: &[f32; FRESHNESS_FEATURES], text: &str, now_year: u32) -> String {
    if feats[8] >= 0.5 {
        // Explicit date present — quote the newest year found, if it lags.
        let newest = features::scan_years(text)
            .into_iter()
            .filter(|&y| y >= 1900 && y <= now_year)
            .max();
        if let Some(y) = newest {
            if now_year.saturating_sub(y) >= 3 {
                return format!("dated {y}");
            }
        }
        return "explicit date".to_string();
    }
    if feats[0] >= 0.15 {
        return "staleness wording".to_string();
    }
    if feats[1] >= 0.15 {
        return "hedging wording".to_string();
    }
    if feats[3] >= 0.15 || (feats[11] >= 0.5 && feats[4] >= 0.2) {
        return "legacy version markers".to_string();
    }
    String::new()
}

/// The wall-clock year for live scoring (eval fixtures inject now_year
/// instead so they stay deterministic). Civil-from-days (Hinnant): the
/// era arithmetic uses a March-based year, so Jan/Feb need the +1.
pub fn current_year() -> u32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64 + 719_468; // days since 0000-03-01
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = era * 400 + yoe;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    (y + if mp >= 10 { 1 } else { 0 }) as u32
}

/* ── JS shim: sofuu.ml.freshness.score(text, task, optsJson?) ──────── */

use std::ffi::{CStr, CString, c_int};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

use crate::rt::ai::json_escape;

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

/// score(text, task, optsJson?) → JSON string
/// {"score":0.97,"stale":true,"years":4,"reason":"dated 2019"}
/// opts: {"kind":"web"|"memory"|"tool"|"file","strength":0..1,"ageDays":n}
unsafe extern "C" fn js_freshness_score(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let empty = String::new();
    // Bounded inputs (Phase 1.2): text/task feed whole-document feature
    // extraction; a megabyte blob must not turn a per-result gate into a
    // latency/memory hazard. The score caps stay as they were.
    let text = if argc >= 1 { cap_str(&arg_str(ctx, *argv), 20_000) } else { empty.clone() };
    let task = if argc >= 2 { cap_str(&arg_str(ctx, *argv.add(1)), 4_000) } else { empty.clone() };
    let opts = if argc >= 3 {
        serde_json::from_str::<serde_json::Value>(&arg_str(ctx, *argv.add(2)))
            .unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Null
    };
    let kind = match opts.get("kind").and_then(|v| v.as_str()).unwrap_or("web") {
        "memory" => SourceKind::Memory,
        "tool" => SourceKind::Tool,
        "file" => SourceKind::File,
        _ => SourceKind::Web,
    };
    let strength = opts
        .get("strength")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32;
    let age_days = opts
        .get("ageDays")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .max(0.0) as f32;
    // NaN/Inf from JSON exponent overflow must never reach the net
    // (Phase 1.2): a non-finite scalar feature renders NaN into the
    // returned JSON and breaks the caller's parse.
    let strength = if strength.is_finite() { strength } else { 0.0 };
    let age_days = if age_days.is_finite() { age_days } else { 0.0 };

    let v = score(&text, &task, kind, strength, age_days, current_year());
    // reason flows into a JSON string literal — escape it (a quote or
    // newline in the evidence text must not break the parse).
    let json = format!(
        "{{\"score\":{:.4},\"stale\":{},\"years\":{:.1},\"reason\":\"{}\"}}",
        v.score,
        v.stale,
        v.years,
        json_escape(Some(&v.reason))
    );
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
}

thread_local! {
    static FRESHNESS_FUNCS: [qjs::JSCFunctionListEntry; 1] = [
        qjs::JSCFunctionListEntry {
            name: c"score".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_freshness_score },
        },
    ];
}

/// Attach `sofuu.ml.freshness` onto an existing `sofuu.ml` object.
/// Called from mod_ml_register (ml/mod.rs) after the ml object exists.
///
/// # Safety
/// `ctx` must be the live engine context; `ml_obj` the sofuu.ml object.
pub unsafe fn register(ctx: *mut JSContext, ml_obj: JSValueConst) {
    let fresh_obj = qjs::sofuu_js_new_object(ctx);
    let funcs = FRESHNESS_FUNCS.with(|f| f.as_ptr());
    qjs::JS_SetPropertyFunctionList(ctx, fresh_obj, funcs, FRESHNESS_FUNCS.with(|f| f.len()) as c_int);
    qjs::sofuu_js_set_property_str(ctx, ml_obj, c"freshness".as_ptr(), fresh_obj);
}
