// ml/relevance/model.rs — the trained relevance gate (PLAN-ML-GATES §6).
//
// The runtime half of the relevance model: the SAME TinyMlp forward pass
// the trainer exercised (ml/net.rs — train/serve skew is structurally
// impossible), weights baked at compile time from the blob ml-train wrote
// after the acceptance gates passed (§10), and `plan()` — the policy layer
// that turns per-candidate scores into use/skip advice. The model ADVISES
// the retriever before tokens are spent; it never touches the data path
// itself (Principle 1) — the guidance rides the ephemeral context message
// and the LLM decides what actually gets read in.

use std::sync::LazyLock;

use super::features::{self, CandKind, CandidateInput, RELEVANCE_FEATURES};
use crate::ml::net::TinyMlp;

pub const IN_DIM: u32 = RELEVANCE_FEATURES as u32; // 37
pub const H1: u32 = 104;
pub const H2: u32 = 44;
pub const PARAMS: u32 = 8617; // §6 arch: 37 → 104 → 44 → 1

/// Decision threshold, baked from the ml-train run that emitted
/// weights_v1.f32. Selected on the VALIDATION fold only (§10 bar 3) at the
/// 0.90-recall floor: val separated with a wide margin (lowest use score
/// 0.868, highest skip score 0.410), so 0.95 sits deep inside the margin.
/// The asymmetry is recall-first (§6): a false drop is invisible and costs
/// correctness, a false use only wastes tokens and stays visible — so a
/// candidate must clear a HIGH bar before the advisor recommends spending
/// tokens on it, and the mechanical rules below only ever add skips that
/// are information-preserving by construction.
pub const THRESHOLD: f32 = 0.95;

/// Mechanical skip floor: a candidate whose embedding cosine with an
/// already-KEPT candidate reaches this is a near-verbatim repeat — pulling
/// it in gains nothing whatever the net says (the content is already in
/// context). Mirrors the compaction gate's mechanical dedupe.
pub const MECHANICAL_DUP_COSINE: f32 = 0.90;

/// Mechanical skip floor for the never-use class (license / generated /
/// lockfile / minified): the marker channel at or above this is a
/// deterministic skip — such content never answers a coding task.
pub const MECHANICAL_NEVER_USE: f32 = 0.50;

static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");

static NET: LazyLock<TinyMlp> = LazyLock::new(|| {
    // The blob is committed and CRC-checked by eval.rs on every test run;
    // a corrupt bake is a build-time mistake, not a runtime condition.
    TinyMlp::from_blob(WEIGHTS).expect("committed relevance weights must load")
});

#[derive(Clone, Debug, Default)]
pub struct RelevancePlan {
    /// Candidates the advisor recommends spending tokens on, ordered by
    /// score (best first) — the caller works down the list.
    pub use_ids: Vec<usize>,
    /// Candidates the advisor recommends leaving out (below threshold or
    /// mechanical skip).
    pub skip_ids: Vec<usize>,
    /// Raw score per input candidate (diagnostics / mlgate labels).
    pub scores: Vec<f32>,
}

/// Score every candidate and build one advise-only relevance pass.
///
/// Policy (§6): mechanical skips run below the model (near-duplicate of an
/// already-kept candidate, never-use class) because they are
/// information-preserving by construction; everything else is `use` at
/// score ≥ THRESHOLD, `skip` below. Deterministic for fixed inputs (scalar
/// f32, fixed order — §10 bar 7).
pub fn plan(
    task: &str,
    recent: &str,
    candidates: &[CandidateInput],
    kept: &[usize],
) -> RelevancePlan {
    let mut out = RelevancePlan::default();
    if candidates.is_empty() {
        return out;
    }

    let feats = features::extract_all(&features::RelevanceContext {
        task,
        recent,
        candidates,
        kept,
    });
    out.scores = feats.iter().map(|f| NET.forward(f)).collect();

    for (i, f) in feats.iter().enumerate() {
        let mechanical_skip = f[16] >= MECHANICAL_DUP_COSINE || f[33] >= MECHANICAL_NEVER_USE;
        if out.scores[i] >= THRESHOLD && !mechanical_skip {
            out.use_ids.push(i);
        } else {
            out.skip_ids.push(i);
        }
    }
    // Best advice first. Stable sort keeps input order for equal scores —
    // determinism bar.
    out.use_ids
        .sort_by(|&a, &b| out.scores[b].partial_cmp(&out.scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/* ── JS shim: sofuu.ml.relevance.plan(candsJson, optsJson?) ────────── */

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

fn kind_of(s: &str) -> CandKind {
    match s {
        "file" => CandKind::File,
        "memory" => CandKind::Memory,
        "web" => CandKind::Web,
        _ => CandKind::Other,
    }
}

/// plan(candsJson, optsJson?) → JSON string
/// in:  {"task","recent","candidates":[{"text","kind":"file|memory|web|
///       other","strength","role","path"}...]}
///      opts: {"kept":[ids]}  (candidates already accepted into context)
/// out: {"use":[ids],"skip":[ids],"scores":[...]} — use ordered best-first.
unsafe extern "C" fn js_relevance_plan(
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

    let task = v.get("task").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let recent = v.get("recent").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let kept: Vec<usize> = opts
        .get("kept")
        .and_then(|x| x.as_array())
        .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|i| i as usize)).collect())
        .unwrap_or_default();

    // Candidate texts/paths must outlive the CandidateInput borrows.
    let texts: Vec<String> = v
        .get("candidates")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|c| c.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    let metas: Vec<(CandKind, f32, u8, String)> = v
        .get("candidates")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|c| {
                    (
                        kind_of(c.get("kind").and_then(|x| x.as_str()).unwrap_or("other")),
                        c.get("strength")
                            .and_then(|x| x.as_f64())
                            .unwrap_or(0.0)
                            .clamp(0.0, 1.0) as f32,
                        c.get("role").and_then(|x| x.as_u64()).unwrap_or(0).min(255) as u8,
                        c.get("path").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    let inputs: Vec<CandidateInput> = texts
        .iter()
        .zip(metas.iter())
        .map(|(text, (kind, strength, role, path))| CandidateInput {
            text,
            kind: *kind,
            strength: *strength,
            role: *role,
            path,
        })
        .collect();

    let p = plan(&task, &recent, &inputs, &kept);

    let use_ids: Vec<String> = p.use_ids.iter().map(|i| i.to_string()).collect();
    let skip_ids: Vec<String> = p.skip_ids.iter().map(|i| i.to_string()).collect();
    let scores: Vec<String> = p.scores.iter().map(|s| format!("{s:.4}")).collect();
    let json = format!(
        "{{\"use\":[{}],\"skip\":[{}],\"scores\":[{}]}}",
        use_ids.join(","),
        skip_ids.join(","),
        scores.join(",")
    );
    match CString::new(json) {
        Ok(c) => qjs::sofuu_js_new_string(ctx, c.as_ptr()),
        Err(_) => qjs::sofuu_js_new_string(ctx, c"{}".as_ptr()),
    }
}

thread_local! {
    static RELEVANCE_FUNCS: [qjs::JSCFunctionListEntry; 1] = [
        qjs::JSCFunctionListEntry {
            name: c"plan".as_ptr(),
            prop_flags: qjs::JS_PROP_WRITABLE | qjs::JS_PROP_CONFIGURABLE,
            def_type: qjs::JS_DEF_CFUNC,
            magic: 0,
            u: qjs::JSCFunctionListEntryFunc { length: 1, cproto: 0, _pad: [0; 6], cfunc: js_relevance_plan },
        },
    ];
}

/// Attach `sofuu.ml.relevance` onto an existing `sofuu.ml` object.
/// Called from mod_ml_register (ml/mod.rs) after the ml object exists.
///
/// # Safety
/// `ctx` must be the live engine context; `ml_obj` the sofuu.ml object.
pub unsafe fn register(ctx: *mut JSContext, ml_obj: JSValueConst) {
    let rel_obj = qjs::sofuu_js_new_object(ctx);
    let funcs = RELEVANCE_FUNCS.with(|f| f.as_ptr());
    qjs::JS_SetPropertyFunctionList(ctx, rel_obj, funcs, RELEVANCE_FUNCS.with(|f| f.len()) as c_int);
    qjs::sofuu_js_set_property_str(ctx, ml_obj, c"relevance".as_ptr(), rel_obj);
}
