// ml/alloc/model.rs — the trained context allocator (the learned layer).
//
// The runtime half of the alloc model: the SAME TinyMlp forward pass the
// trainer exercised (ml/net.rs — train/serve skew is structurally
// impossible), weights baked at compile time from the blob ml-train
// wrote after the acceptance gates passed.
//
// The net predicts ONE scalar — context pressure p∈[0,1]: "without
// action, this session hits the safe budget within ~3 turns." A
// deterministic, clamped policy maps (p, resolved model caps) to the
// per-turn allocation: when to compact, how much window attachments /
// recall / tool results may take, and how much output to reserve. The
// mechanical layer (policy.rs) resolved the selected model's hard limits
// FIRST; nothing below ever escapes those bounds. Advise-only still
// (Principle 1): the caller acts, the safety nets stay.

use std::sync::LazyLock;

use super::features::{self, AllocInput, ALLOC_FEATURES, THINK_BUDGET, THINK_EFFORT, THINK_UNKNOWN};
use super::policy::{self, Resolved};
use crate::ml::cap_str;
use crate::ml::net::TinyMlp;
use crate::rt::model_caps;

pub const IN_DIM: u32 = ALLOC_FEATURES as u32; // 24
pub const H1: u32 = 96;
pub const H2: u32 = 40;
pub const PARAMS: u32 = 6321; // 24 → 96 → 40 → 1

/// Decision threshold baked from the ml-train run that emitted
/// weights_v1.f32 — selected on the VALIDATION fold only, at the
/// recall floor (a missed pressure event is exactly the provider-400
/// failure mode this model exists to prevent; a false alarm only
/// compacts a little early). Trainer: test acc 0.986 (majority 0.810,
/// logreg 0.969), recall 0.994, precision 0.988.
pub const THRESHOLD: f32 = 0.51;

/* ── Policy ranges — the allocation the net drives, all clamped ──────
 * Slack sessions (p→0) breathe: late compaction, generous tool/attach
 * budgets. Tight sessions (p→1) tighten BEFORE the provider error:
 * early compaction, small discretionary budgets, protected output. */

/// Compaction trigger as a fraction of the working budget: 0.70 slack →
/// 0.50 tight (replaces the fixed COMPACT_AT).
const COMPACT_AT_SLACK: f32 = 0.70;
const COMPACT_AT_TIGHT: f32 = 0.50;

/// Tool-result cap in chars: win/5 slack → win/12 tight, clamped.
const TOOL_CAP_FLOOR: i64 = 4_000;
const TOOL_CAP_CEIL: i64 = 32_768;

/// Recall budget in tokens: win×0.03 slack → win×0.01 tight, clamped.
const RECALL_FRACTION_SLACK: f32 = 0.03;
const RECALL_FRACTION_TIGHT: f32 = 0.01;
const RECALL_FLOOR: i64 = 1_024;
const RECALL_CEIL: i64 = 16_384;

/// Attachment budget in tokens: win×0.30 slack → win×0.15 tight.
const ATTACH_FRACTION_SLACK: f32 = 0.30;
const ATTACH_FRACTION_TIGHT: f32 = 0.15;
const ATTACH_FLOOR: i64 = 2_048;
const ATTACH_CEIL: i64 = 65_536;

/// Output reservation floor — below this a turn cannot answer at all.
const OUTPUT_FLOOR: i64 = 512;
/// Safety margin kept between prompt and window when reserving output.
const OUTPUT_MARGIN_FRACTION: f32 = 0.05;
/// Long-form tasks get up to 25% more output reservation.
const WRITING_OUTPUT_BOOST: f32 = 1.25;

/// Unknown models run at least this pressure — conservative allocation
/// until the model's limits are known (registry miss must never produce
/// an oversized request).
const UNKNOWN_PRESSURE_FLOOR: f32 = 0.50;

static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");

static NET: LazyLock<TinyMlp> = LazyLock::new(|| {
    TinyMlp::from_blob(WEIGHTS).expect("committed alloc weights must load")
});

/// One turn's allocation. Every number is clamped to the selected
/// model's resolved limits — a plan can be applied as-is.
#[derive(Clone, Debug)]
pub struct AllocPlan {
    /// Net output (after the unknown-model floor). Diagnostics + the
    /// caller's own heuristics.
    pub pressure: f32,
    /// Resolved working window (tokens).
    pub window: i64,
    /// Recommended max output tokens for this turn (≤ model max output,
    /// ≤ what the window can host after the prompt).
    pub max_output: i64,
    /// Compaction trigger, fraction of the working budget [0.50, 0.70].
    pub compact_at: f32,
    /// Tool-result cap in chars [4000, 32768].
    pub tool_cap_chars: i64,
    /// Recall budget in tokens.
    pub recall_budget_tok: i64,
    /// Attachment budget in tokens.
    pub attach_budget_tok: i64,
    /// Resolved limits known? (false → conservative defaults in use).
    pub known: bool,
    /// Where the limits came from (config/registry/learned/default).
    pub source: &'static str,
    /// ASCII notes where a clamp bit or a limit was learned — surfaced
    /// to the user, never silent.
    pub notes: Vec<String>,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Build one turn's allocation. `input` carries the session state; the
/// model's caps fields are OVERWRITTEN here — the allocator fetches the
/// selected model's details first (policy::resolve), whatever the caller
/// thought it knew.
pub fn plan(
    input: &AllocInput,
    model: Option<&str>,
    cfg_window: i64,
    cfg_max_output: i64,
    base_url: Option<&str>,
) -> AllocPlan {
    let resolved: Resolved = policy::resolve(model, cfg_window, cfg_max_output, base_url);
    plan_with(input, resolved)
}

/// Same policy on pre-resolved limits — the trainer and eval.rs use this
/// form directly so the policy is testable without a model name.
pub fn plan_with(input: &AllocInput, resolved: Resolved) -> AllocPlan {
    let mut notes: Vec<String> = Vec::new();
    if resolved.clamped_config {
        match resolved.config_exceeds_evidence {
            // Obeyed: the user asked for it, evidence merely advises.
            Some(bound) if resolved.window > bound => notes.push(format!(
                "requested window/output is above known evidence ({} tk) -- honored as requested; a provider limit error will correct it",
                bound
            )),
            _ => notes.push(format!(
                "config clamped to the model's limits: window {} tk, output {} tk",
                resolved.window, resolved.max_output
            )),
        }
    }
    if resolved.source == policy::Source::Learned {
        notes.push("limits learned from a provider error this session".to_string());
    }
    if !resolved.known {
        notes.push(format!(
            "unknown model -- conservative limits: window {} tk, output {} tk",
            resolved.window, resolved.max_output
        ));
    }

    // Features see the RESOLVED caps, not whatever the caller passed.
    let mut inp = input.clone();
    inp.ctx_window = resolved.window;
    inp.max_output = resolved.max_output;
    inp.thinking = if resolved.known { inp.thinking } else { THINK_UNKNOWN };
    let feats = features::extract(&inp);

    let mut pressure = NET.forward(&feats);
    if !resolved.known && pressure < UNKNOWN_PRESSURE_FLOOR {
        pressure = UNKNOWN_PRESSURE_FLOOR;
    }
    let p = pressure.clamp(0.0, 1.0);
    let slack = 1.0 - p; // p=0 → full slack

    let win = resolved.window;

    // Compaction trigger: late when slack, early when tight.
    let compact_at = lerp(COMPACT_AT_TIGHT, COMPACT_AT_SLACK, slack).clamp(COMPACT_AT_TIGHT, COMPACT_AT_SLACK);

    // Tool-result cap: generous on slack big windows (kills the
    // truncation→re-read cycle), tight when the window is under pressure.
    let tool_lo = (win / 12).clamp(TOOL_CAP_FLOOR, TOOL_CAP_CEIL);
    let tool_hi = (win / 5).clamp(tool_lo, TOOL_CAP_CEIL);
    let tool_cap_chars = lerp(tool_lo as f32, tool_hi as f32, slack) as i64;

    // Recall + attachment budgets.
    let recall = (win as f32 * lerp(RECALL_FRACTION_TIGHT, RECALL_FRACTION_SLACK, slack)) as i64;
    let recall_budget_tok = recall.clamp(RECALL_FLOOR.min(win / 4).max(256), RECALL_CEIL.min(win / 2));
    let attach = (win as f32 * lerp(ATTACH_FRACTION_TIGHT, ATTACH_FRACTION_SLACK, slack)) as i64;
    let attach_budget_tok = attach.clamp(ATTACH_FLOOR.min(win / 4), ATTACH_CEIL.min(win / 2));

    // Output reservation: what the window can host after overhead +
    // history + attachments + safety margin, bounded by the model's max
    // output; long-form tasks get a boost within that bound.
    let spoken_for = inp.overhead_tk.max(0) + inp.history_tk.max(0) + attach_budget_tok;
    let margin = (win as f32 * OUTPUT_MARGIN_FRACTION) as i64;
    let feasible = (win - spoken_for - margin).max(OUTPUT_FLOOR);
    let mut max_output = feasible.min(resolved.max_output);
    if inp.writing {
        max_output = ((max_output as f32 * WRITING_OUTPUT_BOOST) as i64).min(resolved.max_output);
    }
    let max_output = max_output.max(OUTPUT_FLOOR.min(resolved.max_output));
    if feasible < resolved.max_output.min(2_048) {
        notes.push(format!(
            "output reserve squeezed to {} tk -- prompt already uses {} of {} tk",
            max_output, spoken_for, win
        ));
    }

    AllocPlan {
        pressure: p,
        window: win,
        max_output,
        compact_at,
        tool_cap_chars,
        recall_budget_tok,
        attach_budget_tok,
        known: resolved.known,
        source: resolved.source.as_str(),
        notes,
    }
}

/// Thinking kind for a model name — helper for the JS shim so callers
/// never translate the enum themselves.
fn thinking_kind(model: Option<&str>) -> u8 {
    match model_caps::lookup(model).thinking {
        model_caps::Thinking::None => features::THINK_NONE,
        model_caps::Thinking::Effort(_) => THINK_EFFORT,
        model_caps::Thinking::Budget(_) => THINK_BUDGET,
        model_caps::Thinking::Unknown => THINK_UNKNOWN,
    }
}

/* ── JS shim: sofuu.ml.alloc.plan(stateJson) / noteLimit(model, err) ── */

use std::ffi::{CStr, CString, c_int};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

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

fn i64_field(v: &serde_json::Value, key: &str) -> i64 {
    // Bounded (Phase 1.2): a JSON number above i64 range parses as None
    // (0), and negatives are clamped to the sane floor per field below —
    // history/overhead/attach are magnitudes.
    v.get(key)
        .and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64)))
        .unwrap_or(0)
}

fn f32_field(v: &serde_json::Value, key: &str) -> f32 {
    // Non-finite (1e400 parses as Inf) → 0, never NaN into the net.
    let f = v.get(key).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    if f.is_finite() { f } else { 0.0 }
}

fn bool_field(v: &serde_json::Value, key: &str) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(false)
}

/// plan(stateJson) → JSON string
/// in:  {"model","baseUrl?","cfgWindow","cfgMaxOutput","overheadTk",
///       "calibrated","historyTk","turns","toolFrac","summary","growthTk",
///       "growthAccel","taskTk","attachTk","toolHeavy","writing",
///       "meanAnswerTk","maxAnswerTk","sawLengthStop"}
/// out: {"window","maxOutput","pressure","compactAt","toolCapChars",
///       "recallBudgetTok","attachBudgetTok","known","source","notes":[]}
/// The model's capabilities are resolved FIRST from the model name —
/// callers never pass caps, so a stale caller can't allocate against a
/// window the selected model doesn't have. `baseUrl` (optional) is the
/// endpoint the request will hit: when present, caps the endpoint itself
/// published for this exact model (the discovered store) take precedence
/// over the name-keyed registry.
unsafe extern "C" fn js_alloc_plan(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let v: serde_json::Value = if argc >= 1 {
        serde_json::from_str(&arg_str(ctx, *argv)).unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Null
    };

    let model = cap_str(v.get("model").and_then(|x| x.as_str()).unwrap_or(""), 200);
    let base_url = v.get("baseUrl").and_then(|x| x.as_str()).map(|s| cap_str(s, 2_048));
    let input = AllocInput {
        ctx_window: 0, // resolved inside plan() — never trusted from JS
        max_output: 0,
        thinking: thinking_kind(if model.is_empty() { None } else { Some(&model) }),
        overhead_tk: i64_field(&v, "overheadTk").clamp(0, 100_000_000),
        calibrated: bool_field(&v, "calibrated"),
        history_tk: i64_field(&v, "historyTk").clamp(0, 1 << 40),
        turns: i64_field(&v, "turns").clamp(0, u32::MAX as i64) as u32,
        tool_frac: f32_field(&v, "toolFrac").clamp(0.0, 1.0),
        summary_present: bool_field(&v, "summary"),
        growth_tk: f32_field(&v, "growthTk").max(0.0),
        growth_accel: f32_field(&v, "growthAccel").clamp(0.0, 10.0),
        task_tk: i64_field(&v, "taskTk").clamp(0, 1 << 30),
        attach_tk: i64_field(&v, "attachTk").clamp(0, 1 << 30),
        tool_heavy: bool_field(&v, "toolHeavy"),
        writing: bool_field(&v, "writing"),
        mean_answer_tk: f32_field(&v, "meanAnswerTk").max(0.0),
        max_answer_tk: f32_field(&v, "maxAnswerTk").max(0.0),
        saw_length_stop: bool_field(&v, "sawLengthStop"),
    };
    let p = plan(
        &input,
        if model.is_empty() { None } else { Some(&model) },
        i64_field(&v, "cfgWindow").clamp(0, 1 << 30),
        i64_field(&v, "cfgMaxOutput").clamp(0, 1 << 30),
        base_url.as_deref(),
    );

    let notes: Vec<String> = p
        .notes
        .iter()
        .map(|n| format!("\"{}\"", crate::rt::ai::json_escape(Some(n.as_str()))))
        .collect();
    let json = format!(
        "{{\"window\":{},\"maxOutput\":{},\"pressure\":{:.3},\"compactAt\":{:.2},\"toolCapChars\":{},\"recallBudgetTok\":{},\"attachBudgetTok\":{},\"known\":{},\"source\":\"{}\",\"notes\":[{}]}}",
        p.window,
        p.max_output,
        p.pressure,
        p.compact_at,
        p.tool_cap_chars,
        p.recall_budget_tok,
        p.attach_budget_tok,
        p.known,
        p.source,
        notes.join(","),
    );
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
}

/// noteLimit(model, errorText) → "context" | "output" | "" — record a
/// provider limit error; the returned kind tells the caller a retry is
/// worth attempting for that side (once per kind). The parsed limit feeds
/// future resolve() calls.
unsafe extern "C" fn js_alloc_note_limit(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let model = if argc >= 1 { cap_str(&arg_str(ctx, *argv), 200) } else { String::new() };
    let err = if argc >= 2 { cap_str(&arg_str(ctx, *argv.add(1)), 4_000) } else { String::new() };
    /* Returns the learned limit KIND ("context"/"output") so callers can
     * retry once per kind, or "" when nothing was learned. */
    let s = match policy::note_limit_error_kind(&model, &err) {
        Some(policy::LimitKind::Context) => "context",
        Some(policy::LimitKind::Output) => "output",
        None => "",
    };
    let c = std::ffi::CString::new(s).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, c.as_ptr())
}

/// ingestListing(baseUrl, listingJson) → count — feed a model listing
/// (whatever the endpoint returned for /models, verbatim) into the
/// discovered-caps store. Provider-agnostic: entries are keyed by the
/// endpoint's normalized API root + model id, and the union of field
/// spellings is accepted (context_length / context_window / ctx …,
/// max_completion_tokens / max_tokens …, nested top_provider /
/// per_request_limits objects). Returns how many entries carried caps.
/// Malformed JSON → -1 (the caller surfaces its own fetch error).
unsafe extern "C" fn js_alloc_ingest(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    // Bounded inputs (Phase 1.2): a listing is at most a few MB in
    // practice; the ingest parser itself is bounded by serde, but a
    // runaway argument must not be buffered here.
    let base_url = if argc >= 1 { cap_str(&arg_str(ctx, *argv), 2_048) } else { String::new() };
    let listing = if argc >= 2 { cap_str(&arg_str(ctx, *argv.add(1)), 8_000_000) } else { String::new() };
    let n = match crate::rt::model_caps_discovered::ingest_listing_json(&base_url, &listing) {
        Ok(n) => n as i64,
        Err(_) => -1,
    };
    let s = std::ffi::CString::new(n.to_string()).unwrap_or_default();
    qjs::sofuu_js_new_string(ctx, s.as_ptr())
}

thread_local! {
    static ALLOC_FUNCS: [qjs::JSCFunctionListEntry; 3] = [
        qjs::JSCFunctionListEntry {
            name: c"plan".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_alloc_plan },
        },
        qjs::JSCFunctionListEntry {
            name: c"noteLimit".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_alloc_note_limit },
        },
        qjs::JSCFunctionListEntry {
            name: c"ingestListing".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 2, cproto: 0, _pad: [0; 6], cfunc: js_alloc_ingest },
        },
    ];
}

/// Attach `sofuu.ml.alloc` onto an existing `sofuu.ml` object. Called
/// from mod_ml_register (ml/mod.rs) after the ml object exists.
///
/// # Safety
/// `ctx` must be the live engine context; `ml_obj` the sofuu.ml object.
pub unsafe fn register(ctx: *mut JSContext, ml_obj: JSValueConst) {
    let alloc_obj = qjs::sofuu_js_new_object(ctx);
    let funcs = ALLOC_FUNCS.with(|f| f.as_ptr());
    qjs::JS_SetPropertyFunctionList(ctx, alloc_obj, funcs, ALLOC_FUNCS.with(|f| f.len()) as c_int);
    qjs::sofuu_js_set_property_str(ctx, ml_obj, c"alloc".as_ptr(), alloc_obj);
}
