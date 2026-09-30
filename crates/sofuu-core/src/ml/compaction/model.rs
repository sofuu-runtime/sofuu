// ml/compaction/model.rs — the trained compaction gate (PLAN-ML-GATES §12).
//
// The runtime half of the compaction model: the SAME TinyMlp forward pass
// the trainer exercised (ml/net.rs — train/serve skew is structurally
// impossible), weights baked at compile time from the blob ml-train wrote
// after the acceptance gates passed (§10), and `plan()` — the policy layer
// that turns per-segment scores into a small, conservative compaction
// pass. The model ADVISES: it flags which segments look disposable and in
// which tier (free mechanical work vs LLM summarization). It never
// rewrites history itself — the caller decides what actually happens
// (Principle 1), and the drop-oldest guard stays the final safety net.

use std::sync::LazyLock;

use super::features::{self, SegKind, SegmentInput, COMPACTION_FEATURES};
use crate::ml::cap_str;
use crate::ml::net::TinyMlp;

pub const IN_DIM: u32 = COMPACTION_FEATURES as u32; // 33
pub const H1: u32 = 104;
pub const H2: u32 = 44;
pub const PARAMS: u32 = 8201; // §12 arch: 33 → 104 → 44 → 1

/// Decision threshold, baked from the ml-train run that emitted
/// weights_v1.f32. Selected on the VALIDATION fold only (§10 bar 3): the
/// fold (off-task musings, held-out decision phrasings, held-out
/// paraphrased references) separated with a wide margin — highest keep
/// score 0.000, lowest disposable score 0.824 — so the threshold is the
/// midpoint of that margin (0.41): the point farthest from both classes.
/// A segment must score above 0.41 before it is even a candidate; the
/// policy layer below adds hard keep-window protection on top. The
/// held-out test fold (held-out dups + held-out references) measured
/// precision 1.000 / recall 1.000 at it.
pub const THRESHOLD: f32 = 0.41;

/// Default per-pass budget when the caller doesn't name one: free ~12% of
/// the window (§12 "small budget per pass"). Context degrades gracefully
/// instead of cliff-dropping.
const DEFAULT_BUDGET_FRACTION: f32 = 0.12;

/// Segments in the protected keep-window: age ≤ 1 (this turn + the one
/// before it). Hard protection regardless of score — the caller also
/// passes ages, but the gate never trusts a flag it can enforce itself.
pub const KEEP_WINDOW_AGE: u32 = 1;

/// Mechanical dedupe floor (§12 free tier): a segment whose embedding
/// cosine with an EARLIER segment reaches this is a near-verbatim
/// repeat — dropping it is information-preserving whatever the net
/// says. The net's reference channel cannot see this case (a dup that
/// resembles the recent window looks "referenced" and scores keep), so
/// the deterministic rule runs below the model. 0.95 sits above the
/// measured distinct-text cosine ceiling (~0.81 on same-topic pairs).
pub const MECHANICAL_DUP_COSINE: f32 = 0.95;

static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");

static NET: LazyLock<TinyMlp> = LazyLock::new(|| {
    // The blob is committed and CRC-checked by eval.rs on every test run;
    // a corrupt bake is a build-time mistake, not a runtime condition.
    TinyMlp::from_blob(WEIGHTS).expect("committed compaction weights must load")
});

/// How a flagged segment can be compacted. The free tiers need no LLM
/// call (§12 "free tier below"): prefer them, spend an LLM summarization
/// only where one is actually needed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tier {
    /// Near-duplicate of an earlier segment — drop the repeat, keep the
    /// original. Free.
    Dup,
    /// Pure boilerplate (site chrome, legal text, widgets) — drop. Free.
    Boilerplate,
    /// Cheaply re-obtainable (file read, grep, listing) — drop; a
    /// compacted copy can be fetched again. Free.
    Retrievable,
    /// Needs a real summary — the only tier that spends an LLM call.
    Summarize,
}

impl Tier {
    fn as_str(self) -> &'static str {
        match self {
            Tier::Dup => "dup",
            Tier::Boilerplate => "boilerplate",
            Tier::Retrievable => "retrievable",
            Tier::Summarize => "summarize",
        }
    }

    fn free(self) -> bool {
        !matches!(self, Tier::Summarize)
    }
}

/// One flagged segment in the plan.
#[derive(Clone, Copy, Debug)]
pub struct Flagged {
    /// Index into the caller's segment array (stable id).
    pub id: usize,
    pub score: f32,
    pub tokens: u32,
    pub tier: Tier,
}

#[derive(Clone, Debug, Default)]
pub struct CompactionPlan {
    /// Disposable segments this pass, ordered free-tier-first then by
    /// score — the caller works down the list until its budget is spent.
    pub compact: Vec<Flagged>,
    /// Protected indices (keep-window, below-threshold, load-bearing).
    pub keep: Vec<usize>,
    /// Tokens freed if the whole `compact` list is processed.
    pub freeable: u32,
    /// Raw score per input segment (diagnostics / mlgate labels).
    pub scores: Vec<f32>,
}

/// Classify HOW a flagged segment gets compacted from its features.
/// Thresholds sit well inside the trained channels' firing ranges (dup
/// cosine ≥ 0.85 is paraphrase-or-better; boilerplate cosine ≥ 0.45).
fn tier_of(feats: &[f32; COMPACTION_FEATURES]) -> Tier {
    if feats[16] >= 0.85 {
        Tier::Dup
    } else if feats[14] >= 0.45 {
        Tier::Boilerplate
    } else if feats[13] >= 0.5 {
        Tier::Retrievable
    } else {
        Tier::Summarize
    }
}

/// Score every segment and build one conservative compaction pass.
///
/// Policy (§12): hard-protect the keep-window (age ≤ 1); candidate a
/// segment at score ≥ THRESHOLD, or unconditionally when it is a
/// near-verbatim duplicate of an earlier segment (mechanical dedupe —
/// information-preserving by construction); order free tiers first; cap
/// the pass at `budget_tokens` (0 → DEFAULT_BUDGET_FRACTION of the
/// window), but always take at least two candidates when available so a
/// caller that drops whole turns can make progress. Deterministic for
/// fixed inputs (scalar f32, fixed order — §10 bar 7).
pub fn plan(
    task: &str,
    summary: &str,
    recent: &str,
    segments: &[SegmentInput],
    budget_tokens: u32,
) -> CompactionPlan {
    let mut out = CompactionPlan::default();
    if segments.is_empty() {
        return out;
    }

    let mut feats = features::extract_all(&features::CompactionContext {
        task,
        summary,
        recent,
        segments,
    });
    // Phase 1.2: NaN must never reach a forward pass (sigmoid(NaN) is
    // NaN → "NaN" in the returned JSON → the caller's parse breaks).
    for f in feats.iter_mut() {
        crate::ml::sanitize_features(f);
    }
    out.scores = feats.iter().map(|f| NET.forward(f)).collect();

    // P1-19 (AUDIT-2026-09-01): plain `.sum()` wraps on hostile token
    // counts (release builds wrap silently) — the 2026-08-30 hardening
    // landed only in features.rs:220. 256×u32::MAX wrapped to a budget of
    // ~5 tokens, freeing the wrong amount. Saturating, matching features.rs.
    let total_tokens: u32 = segments
        .iter()
        .fold(0u32, |acc, s| acc.saturating_add(s.tokens));
    let budget = if budget_tokens > 0 {
        budget_tokens
    } else {
        (total_tokens as f32 * DEFAULT_BUDGET_FRACTION) as u32
    };

    let mut candidates: Vec<Flagged> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        let score = out.scores[i];
        let mechanical_dup = feats[i][16] >= MECHANICAL_DUP_COSINE;
        if seg.age_steps <= KEEP_WINDOW_AGE || (score < THRESHOLD && !mechanical_dup) {
            out.keep.push(i);
            continue;
        }
        candidates.push(Flagged { id: i, score, tokens: seg.tokens, tier: tier_of(&feats[i]) });
    }

    // Free-tier first (no LLM spend), then by confidence. Stable sort so
    // equal keys keep input order — determinism bar.
    candidates.sort_by(|a, b| {
        b.tier
            .free()
            .cmp(&a.tier.free())
            .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut spent = 0u32;
    for c in candidates {
        // The budget is a soft cap: the first two candidates always go
        // (a turn-granular caller needs both members of a pair to act).
        if spent.saturating_add(c.tokens) > budget && out.compact.len() >= 2 {
            break;
        }
        spent += c.tokens;
        out.freeable += c.tokens;
        out.compact.push(c);
    }
    out
}

/* ── JS shim: sofuu.ml.compaction.plan(segmentsJson, optsJson?) ────── */

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

/// plan(segmentsJson, optsJson?) → JSON string
/// in:  {"task","summary","recent","segments":[{"text","tokens","age",
///       "kind":0-3,"retrievable":bool,"compacted":bool}...]}
///      opts: {"budget":tokens}  (0/absent → ~12% of the window)
/// out: {"compact":[ids],"keep":[ids],"freeable":tok,"tiers":[...],
///       "scores":[...]}  — tiers[i] describes compact[i].
unsafe extern "C" fn js_compaction_plan(
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
    let opts: serde_json::Value = if argc >= 2 {
        serde_json::from_str(&arg_str(ctx, *argv.add(1))).unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Null
    };

    // Bounded inputs (Phase 1.2). Segment count: histories are hundreds
    // of messages at most; the extractor embeds every segment and scores
    // an O(n²) dup channel, so a giant array must never reach it. Texts
    // match the trainer's clip; token counts clamp into u32 (a huge JSON
    // number would wrap on the cast and feed the net garbage).
    const MAX_SEGMENTS: usize = 256;
    const MAX_SEG_CHARS: usize = 20_000;
    let task = cap_str(v.get("task").and_then(|x| x.as_str()).unwrap_or(""), 4_000);
    let summary = cap_str(v.get("summary").and_then(|x| x.as_str()).unwrap_or(""), 20_000);
    let recent = cap_str(v.get("recent").and_then(|x| x.as_str()).unwrap_or(""), 20_000);
    let budget = opts
        .get("budget")
        .and_then(|x| x.as_u64())
        .unwrap_or(0)
        .min(u32::MAX as u64) as u32;

    // Segment texts must outlive the SegmentInput borrows — collect first.
    let texts: Vec<String> = v
        .get("segments")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .take(MAX_SEGMENTS)
                .map(|s| cap_str(s.get("text").and_then(|x| x.as_str()).unwrap_or(""), MAX_SEG_CHARS))
                .collect()
        })
        .unwrap_or_default();
    let metas: Vec<(u32, u32, u8, bool, bool)> = v
        .get("segments")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .take(MAX_SEGMENTS)
                .map(|s| {
                    (
                        s.get("tokens").and_then(|x| x.as_u64()).unwrap_or(0).min(u32::MAX as u64) as u32,
                        s.get("age").and_then(|x| x.as_u64()).unwrap_or(0).min(u32::MAX as u64) as u32,
                        s.get("kind").and_then(|x| x.as_u64()).unwrap_or(1).min(3) as u8,
                        s.get("retrievable").and_then(|x| x.as_bool()).unwrap_or(false),
                        s.get("compacted").and_then(|x| x.as_bool()).unwrap_or(false),
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    let inputs: Vec<SegmentInput> = texts
        .iter()
        .zip(metas.iter())
        .map(|(text, &(tokens, age, kind, retr, compacted))| SegmentInput {
            text,
            tokens,
            age_steps: age,
            kind: match kind {
                0 => SegKind::User,
                1 => SegKind::Assistant,
                2 => SegKind::ToolCall,
                _ => SegKind::ToolResult,
            },
            retrievable: retr,
            already_compacted: compacted,
        })
        .collect();

    let p = plan(&task, &summary, &recent, &inputs, budget);

    let compact_ids: Vec<String> = p.compact.iter().map(|c| c.id.to_string()).collect();
    let keep_ids: Vec<String> = p.keep.iter().map(|k| k.to_string()).collect();
    let tiers: Vec<&str> = p.compact.iter().map(|c| c.tier.as_str()).collect();
    let scores: Vec<String> = p.scores.iter().map(|s| format!("{s:.4}")).collect();
    let json = format!(
        "{{\"compact\":[{}],\"keep\":[{}],\"freeable\":{},\"tiers\":[{}],\"scores\":[{}]}}",
        compact_ids.join(","),
        keep_ids.join(","),
        p.freeable,
        tiers.iter().map(|t| format!("\"{t}\"")).collect::<Vec<_>>().join(","),
        scores.join(",")
    );
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
}

thread_local! {
    static COMPACTION_FUNCS: [qjs::JSCFunctionListEntry; 1] = [
        qjs::JSCFunctionListEntry {
            name: c"plan".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_compaction_plan },
        },
    ];
}

/// Attach `sofuu.ml.compaction` onto an existing `sofuu.ml` object.
/// Called from mod_ml_register (ml/mod.rs) after the ml object exists.
///
/// # Safety
/// `ctx` must be the live engine context; `ml_obj` the sofuu.ml object.
pub unsafe fn register(ctx: *mut JSContext, ml_obj: JSValueConst) {
    let comp_obj = qjs::sofuu_js_new_object(ctx);
    let funcs = COMPACTION_FUNCS.with(|f| f.as_ptr());
    qjs::JS_SetPropertyFunctionList(ctx, comp_obj, funcs, COMPACTION_FUNCS.with(|f| f.len()) as c_int);
    qjs::sofuu_js_set_property_str(ctx, ml_obj, c"compaction".as_ptr(), comp_obj);
}
