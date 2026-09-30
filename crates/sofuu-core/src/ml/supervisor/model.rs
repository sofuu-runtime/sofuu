// ml/supervisor/model.rs -- the trained supervisor gate (PLAN-ML-GATES §11).
//
// The runtime half of the supervisor: the SAME TinyMlp forward pass the
// trainer exercised (ml/net.rs -- train/serve skew is structurally
// impossible), weights baked at compile time from the blob ml-train wrote
// after the acceptance gates passed (§10), layered OVER the Phase-1
// mechanical rules (context.rs dup_call / reread_unchanged): a rule hit is
// certain and wins; where the rules are silent the net speaks -- too-broad,
// off-task, spinning, skip-advised, over-budget. Advise-only (Principle 1):
// the nudge rides the tool result (pre-call) or the ephemeral context
// message (loop boundary) back in-band; the call still runs, the LLM
// decides. Two checkpoints:
//
//   check()      -- pre-call, the only place waste can be PREVENTED
//   loop_check() -- loop boundary, trajectory-level (the "__loop__"
//                  pseudo-action: spinning, drifting, over budget)

use std::sync::LazyLock;

use super::features::{self, SupervisorContext, TrajCall, LOOP_TOOL, SUPERVISOR_FEATURES};
use crate::ml::context::{self, CallRec};
use crate::ml::net::TinyMlp;

pub const IN_DIM: u32 = SUPERVISOR_FEATURES as u32; // 33
pub const H1: u32 = 104;
pub const H2: u32 = 44;
pub const PARAMS: u32 = 8201; // §11 arch: 33 → 104 → 44 → 1

/// Decision threshold, baked from the ml-train run that emitted
/// weights_v1.f32. Chosen on the VALIDATION fold only (§10 bar 3) at the
/// 0.90 recall-selection floor -- with the val fold perfectly separated
/// the margin-midpoint fallback placed it halfway between the highest
/// clean score (0.638) and the lowest waste score (0.998), so it carries
/// ~0.18 of margin on both sides against real-world messiness. The
/// held-out test fold held recall 1.000 and precision 1.000 at this
/// value. Still deliberately conservative: a nudge that cries wolf gets
/// ignored, which is its own failure mode (§11).
pub const THRESHOLD: f32 = 0.82;

static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");

static NET: LazyLock<TinyMlp> = LazyLock::new(|| {
    // The blob is committed and CRC-checked by eval.rs on every test run;
    // a corrupt bake is a build-time mistake, not a runtime condition.
    TinyMlp::from_blob(WEIGHTS).expect("committed supervisor weights must load")
});

/// The committed net (online.rs adapts only its output layer, §13).
pub(crate) fn net() -> &'static TinyMlp {
    &NET
}

static BLOB_HASH: LazyLock<u64> = LazyLock::new(|| fnv1a64(WEIGHTS));

/// FNV-1a 64 over the committed blob — the identity key that ties an
/// online-adapted layer to the exact pretrained weights it came from.
/// A re-bake changes the hash and invalidates every stored adaptation.
pub(crate) fn blob_hash() -> u64 {
    *BLOB_HASH
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One supervisor verdict. `ok` = nothing to flag; otherwise `reason` is
/// the machine label, `nudge` the in-band advisory text, `score` the net's
/// P(waste), and `source` which layer spoke ("rule" or "model").
#[derive(Clone, Debug, Default)]
pub struct SupervisorVerdict {
    pub ok: bool,
    pub score: f32,
    pub reason: String,
    pub nudge: Option<String>,
    pub source: &'static str,
}

fn trajectory(run_id: &str) -> (String, Vec<CallRec>) {
    context::snapshot(run_id).unwrap_or_default()
}

fn to_traj_calls(calls: &[CallRec]) -> Vec<TrajCall> {
    calls
        .iter()
        .map(|c| TrajCall {
            tool: c.tool.clone(),
            sig: c.sig.clone(),
            target: c.target.clone(),
            result_chars: c.result_chars,
            errored: c.errored,
        })
        .collect()
}

/// True when the target was read at least once and the most recent
/// read/write touch of it is a write (mirror of feature f7's semantics).
fn written_since_last_read(calls: &[CallRec], target: &str) -> bool {
    let mut seen_read = false;
    let mut written_since = false;
    for c in calls {
        if c.target != target {
            continue;
        }
        if c.tool == "read_file" {
            seen_read = true;
            written_since = false;
        } else if c.tool == "write_file" || c.tool == "edit_file" {
            if seen_read {
                written_since = true;
            }
        }
    }
    seen_read && written_since
}

/// Score one candidate action against the run so far (no side effects on
/// the working set — the caller decides when to record). Forward pass goes
/// through the online layer when one has been adopted (§13).
pub fn score_action(
    task: &str,
    calls: &[CallRec],
    skip_targets: &[String],
    budget: u32,
    step: u32,
    tool: &str,
    args_text: &str,
    target: &str,
    sig: &str,
) -> f32 {
    let feats = extract_features(task, calls, skip_targets, budget, step, tool, args_text, target, sig);
    forward_with_online(&feats)
}

fn extract_features(
    task: &str,
    calls: &[CallRec],
    skip_targets: &[String],
    budget: u32,
    step: u32,
    tool: &str,
    args_text: &str,
    target: &str,
    sig: &str,
) -> [f32; SUPERVISOR_FEATURES] {
    let ctx = SupervisorContext {
        task: task.to_string(),
        budget,
        calls: to_traj_calls(calls),
        skip_targets: skip_targets.to_vec(),
        tool: tool.to_string(),
        args_text: args_text.to_string(),
        target: target.to_string(),
        sig: sig.to_string(),
        step,
    };
    let mut f = features::extract(&ctx);
    // Phase 1.2: NaN must never reach the forward pass (the supervisor
    // runs on every tool call — a NaN score would break the check JSON).
    crate::ml::sanitize_features(&mut f);
    f
}

/// Forward through the adopted online output layer when present, else the
/// pretrained layer. Hidden activations are shared — only W3/b3 differ.
fn forward_with_online(feats: &[f32; SUPERVISOR_FEATURES]) -> f32 {
    let h = NET.forward_hidden(feats);
    match crate::ml::online::output_layer() {
        Some(w3) => NET.forward_output_with(&w3, &h),
        None => NET.forward_output(&h),
    }
}

/// Pre-call checkpoint. Layering: the mechanical rules speak first (they
/// are certain); where they are silent, the net speaks at THRESHOLD.
/// Either way the call is recorded and still runs — advisors, not filters.
pub fn check(
    run_id: &str,
    step: u32,
    tool: &str,
    sig: &str,
    target: &str,
    args_text: &str,
    skip_targets: &[String],
    budget: u32,
    task_fallback: &str,
) -> SupervisorVerdict {
    // Snapshot BEFORE precheck records the candidate call.
    let (task_ws, calls) = trajectory(run_id);
    let task = if task_ws.trim().is_empty() { task_fallback } else { &task_ws };
    let feats = extract_features(task, &calls, skip_targets, budget, step, tool, args_text, target, sig);
    let score = forward_with_online(&feats);

    // The rule layer records the call as it checks (advisors never block).
    let rule = context::precheck(run_id, step, tool, sig, target);
    if !rule.ok {
        // Online learning still sees the example: rule-flagged calls are
        // the certain positives of the runtime distribution.
        crate::ml::online::observe(run_id, step, &feats, true, score);
        return SupervisorVerdict {
            ok: false,
            score,
            reason: rule.reason,
            nudge: rule.nudge,
            source: "rule",
        };
    }

    // Mechanical excuse, mirrored from the rule layer: a write invalidates
    // everything the run knew about that file, so reading it again is fresh
    // information gathering even with identical args -- the net's dup
    // evidence (f3/f4) is stale by construction here. Write-read loops are
    // still policed at the loop boundary (trajectory level).
    if tool == "read_file" && !target.is_empty() && written_since_last_read(&calls, target) {
        crate::ml::online::observe(run_id, step, &feats, false, score);
        return SupervisorVerdict { ok: true, score, ..Default::default() };
    }

    if score >= THRESHOLD {
        let (reason, nudge) = precall_class(&feats);
        crate::ml::online::observe(run_id, step, &feats, true, score);
        return SupervisorVerdict {
            ok: false,
            score,
            reason,
            nudge: Some(nudge),
            source: "model",
        };
    }

    crate::ml::online::observe(run_id, step, &feats, false, score);
    SupervisorVerdict { ok: true, score, ..Default::default() }
}

/// Loop-boundary checkpoint: the "__loop__" pseudo-action asks the same
/// question of the run itself. Not recorded in the working set -- it is a
/// checkpoint, not a call.
pub fn loop_check(run_id: &str, step: u32, budget: u32) -> SupervisorVerdict {
    let (task, calls) = trajectory(run_id);
    // Mechanical guard, mirrored from the rule layer's philosophy: spin
    // and stall are PATTERN predicates, and fewer than three recorded
    // calls is not a pattern -- it is the start of a run. Without this
    // the net saturates on the tiny trajectory (score 1.0 "stalled") and
    // nags after the very first tool call of every run. The over-budget
    // case is still policed per-call by the pre-call checkpoint.
    if calls.len() < 3 {
        return SupervisorVerdict { ok: true, score: 0.0, ..Default::default() };
    }
    let score = score_action(&task, &calls, &[], budget, step, LOOP_TOOL, "", "", "");
    if score < THRESHOLD {
        return SupervisorVerdict { ok: true, score, ..Default::default() };
    }
    let ctx = SupervisorContext {
        task,
        budget,
        calls: to_traj_calls(&calls),
        skip_targets: Vec::new(),
        tool: LOOP_TOOL.to_string(),
        args_text: String::new(),
        target: String::new(),
        sig: String::new(),
        step,
    };
    let feats = features::extract(&ctx);
    let (reason, nudge) = loop_class(&feats);
    SupervisorVerdict { ok: false, score, reason, nudge: Some(nudge), source: "model" }
}

/// Name the pre-call waste class from the decisive channels so the nudge
/// says WHY. Priority follows certainty: duplication > skip advice >
/// breadth > budget > spin > off-task.
fn precall_class(f: &[f32; SUPERVISOR_FEATURES]) -> (String, String) {
    if f[3] >= 0.5 {
        return (
            "dup_call".into(),
            "this exact call already ran this run -- reuse its result unless you expect different output".into(),
        );
    }
    if f[4] >= 0.80 {
        return (
            "near_dup".into(),
            "a nearly identical call already ran -- reuse that result or change what you are asking for".into(),
        );
    }
    if f[7] >= 0.5 {
        return (
            "reread_unchanged".into(),
            "this target was read earlier and has not changed since -- reuse the earlier content".into(),
        );
    }
    if f[18] >= 0.5 {
        return (
            "skip_advised".into(),
            "this target was flagged 'probably not needed' for this turn -- confirm it is needed before spending tokens on it".into(),
        );
    }
    if f[16] >= 0.5 || f[17] >= 0.5 {
        return (
            "too_broad".into(),
            "this search is very broad -- narrow the pattern or scope before running it".into(),
        );
    }
    if f[13] >= 0.5 {
        return (
            "over_budget".into(),
            "the call budget is spent -- move toward the answer with what you have".into(),
        );
    }
    if f[29] >= 0.75 || f[24] >= 1.0 {
        return (
            "spinning".into(),
            "the run is repeating the same action -- change approach or conclude".into(),
        );
    }
    if f[0] < 0.15 && f[2] < 0.20 {
        return (
            "off_task".into(),
            "this looks unrelated to the task -- confirm it advances the goal before running it".into(),
        );
    }
    (
        "waste_risk".into(),
        "this call looks wasteful given the run so far -- reconsider before running it".into(),
    )
}

/// Name the loop-boundary class: budget > spin > stall.
fn loop_class(f: &[f32; SUPERVISOR_FEATURES]) -> (String, String) {
    if f[13] >= 0.5 {
        return (
            "loop_over_budget".into(),
            "the loop is past its call budget -- wrap up with the best answer you have".into(),
        );
    }
    if f[29] >= 0.75 || (f[25] >= 0.6 && f[27] < 0.01) {
        return (
            "loop_spinning".into(),
            "the loop is repeating itself without progress -- change approach or finish".into(),
        );
    }
    (
        "loop_stalled".into(),
        "the loop has stopped making progress -- wrap up or change approach".into(),
    )
}
