// ml/supervisor/features.rs — feature extraction for the supervisor gate
// (PLAN-ML-GATES §11).
//
// One feature vector per ACTION scored against the run's trajectory so
// far: "is this action the right one right now?" The checkpoint is the
// pre-call moment (execOneTool) — the only place waste can be PREVENTED,
// not observed — plus the loop boundary, where a "__loop__" pseudo-action
// (empty tool/target/args, trajectory channels only) asks the same
// question of the run itself: spinning, drifting, or over budget.
//
// Everything here is cheap and available at call time (§11): similarity
// of the call to the task, duplication against earlier calls, re-reads of
// unchanged targets, calls-so-far vs budget, whether this tool has paid
// off before, how broad the call is, and whether the target was in the
// pre-turn "probably not needed" advice. Deterministic: scalar f32, fixed
// order, injected context — the SAME code runs in the trainer and the
// runtime, so train/serve skew is structurally impossible (§4).

use std::collections::HashMap;

use crate::ml::freshness::features::cosine;
use crate::rt::ai::sofuu_tfidf_embed;

pub const SUPERVISOR_FEATURES: usize = 33;
const EMBED_DIM: usize = 768;

/// The loop-boundary pseudo-tool: the action slot is empty, only the
/// trajectory channels speak (spin / drift / budget / progress).
pub const LOOP_TOOL: &str = "__loop__";

/// Results below this carried no useful material (a grep with two short
/// hits, an empty glob) — "did this tool pay off" counts them as misses.
const USELESS_RESULT_CHARS: u32 = 80;

/// One earlier call in this run's trajectory (a mirror of context.rs
/// CallRec, decoupled so the trainer can synthesize trajectories without
/// touching the global working set).
#[derive(Clone, Debug, Default)]
pub struct TrajCall {
    pub tool: String,
    /// Canonical call signature (tool + sorted-key args JSON, computed
    /// JS-side) — identity key for duplication.
    pub sig: String,
    pub target: String,
    pub result_chars: u32,
    pub errored: bool,
}

/// Everything one action is scored against.
#[derive(Clone, Debug, Default)]
pub struct SupervisorContext {
    pub task: String,
    /// Pre-turn tool-call budget (the run's maxSteps); 0 → unknown.
    pub budget: u32,
    /// Earlier calls this run, oldest first. The candidate action is NOT
    /// in this list yet.
    pub calls: Vec<TrajCall>,
    /// Targets the pre-turn relevance advice flagged "probably not needed".
    pub skip_targets: Vec<String>,
    /// The candidate action.
    pub tool: String,
    pub args_text: String,
    pub target: String,
    /// Canonical call signature (tool + canonical args JSON) — the same
    /// identity key mlSig() computes JS-side; the trainer builds it as
    /// `tool:args_text`. Empty for the loop pseudo-action.
    pub sig: String,
    /// 1-based index of this call (calls-so-far + 1).
    pub step: u32,
}

/* ── Lexical helpers (same constructions as the relevance extractor) ── */

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

fn distinctive(text: &str) -> Vec<String> {
    tokens(text).into_iter().filter(|t| t.chars().count() >= 4).collect()
}

/// Identifier parts: snake_case / camelCase pieces of a token, len ≥ 4
/// ("gateway_upstream" → gateway, upstream; "verifySignature" → verify,
/// signature). An action echoes the task WORD by word, rarely as a whole
/// identifier — matching parts instead of whole tokens is what gives the
/// overlap channel its margin on paths and symbols.
fn ident_parts(tok: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for seg in tok.split('_') {
        let mut cur = String::new();
        for c in seg.chars() {
            if c.is_ascii_uppercase() && !cur.is_empty() {
                parts.push(cur.to_ascii_lowercase());
                cur = String::new();
            }
            cur.push(c);
        }
        if !cur.is_empty() {
            parts.push(cur.to_ascii_lowercase());
        }
    }
    parts.retain(|p| p.chars().count() >= 4);
    parts
}

/// Fraction of the action's distinctive identifier PARTS that appear in
/// `other` (the task).
fn part_overlap_ratio(text: &str, other: &str) -> f32 {
    let other_lower = other.to_lowercase();
    let parts: Vec<String> = distinctive(text)
        .iter()
        .flat_map(|t| ident_parts(t))
        .collect();
    if parts.is_empty() {
        return 0.0;
    }
    let hit = parts.iter().filter(|p| other_lower.contains(p.as_str())).count();
    (hit as f32 / parts.len() as f32).min(1.0)
}

/// Fraction of the task's distinctive tokens whose 5-char morphological
/// stem appears in `text` (migrates↔migration echoes).
fn stem_match(task: &str, text: &str) -> f32 {
    let mine = distinctive(task);
    if mine.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let hit = mine
        .iter()
        .filter(|t| {
            let stem: String = t.chars().take(5).collect();
            stem.len() >= 4 && lower.contains(&stem)
        })
        .count();
    (hit as f32 / mine.len() as f32).min(1.0)
}

/* ── Embedding (bounded inputs — the check runs per tool call) ───── */

fn embed_text(text: &str) -> Vec<f32> {
    let clipped: String = text.chars().take(512).collect();
    let mut v = vec![0.0f32; EMBED_DIM];
    sofuu_tfidf_embed(clipped.as_bytes(), &mut v, EMBED_DIM);
    v
}

fn embed_short(text: &str) -> Vec<f32> {
    let clipped: String = text.chars().take(256).collect();
    let mut v = vec![0.0f32; EMBED_DIM];
    sofuu_tfidf_embed(clipped.as_bytes(), &mut v, EMBED_DIM);
    v
}

/* ── Tool families ───────────────────────────────────────────────── */

/// 0 read · 1 write · 2 exec · 3 other (memory/delegate/web/__loop__).
fn tool_family(tool: &str) -> u8 {
    match tool {
        "read_file" | "grep" | "glob" | "list_dir" | "search" => 0,
        "write_file" | "edit_file" | "apply_patch" => 1,
        "bash" | "exec" | "run" | "shell" => 2,
        _ => 3,
    }
}

fn is_search_family(tool: &str) -> bool {
    matches!(tool, "grep" | "glob" | "list_dir" | "search")
}

fn is_read_family(tool: &str) -> bool {
    tool_family(tool) == 0
}

fn is_write_family(tool: &str) -> bool {
    tool_family(tool) == 1
}

/* ── Breadth ─────────────────────────────────────────────────────── */

/// How broad a search target is (§11 "grep ., list_dir on root"). Graded:
/// the fully-degenerate patterns are a 1.0, wildcard-heavy ones 0.5.
/// Non-search tools score 0 — reading one named file is never "broad".
fn breadth_score(tool: &str, target: &str) -> f32 {
    if !is_search_family(tool) {
        return 0.0;
    }
    let t = target.trim();
    if matches!(t, "." | "*" | "/" | "~" | "./" | "**" | "*.*") {
        return 1.0;
    }
    if t.contains("**") || t.ends_with("/*") || (t.len() <= 2 && t.contains('*')) {
        return 0.5;
    }
    0.0
}

/* ── Extraction ──────────────────────────────────────────────────── */

/// Extract the 33-feature vector for the candidate action.
///
/// Layout (see comments inline):
///   0-2   action↔task similarity (cosine, token overlap, stem echo)
///   3-5   duplication (exact sig, near-dup cosine, repeat count)
///   6-7   target re-read (prior reads, unchanged-since-last-read)
///   8-11  this tool's history (usage, errors, useless results, yield)
///   12-13 budget pressure (step/budget, over-budget flag)
///   14-15 trajectory health (error rate, useless-result rate)
///   16-17 breadth (degenerate pattern, search with no target)
///   18-19 target in the pre-turn "probably not needed" advice
///   20-23 tool family one-hots (read/write/exec/other)
///   24-29 trajectory shape (recent repeats, revisits, drift, progress,
///         error streak, same-target streak)
///   30    args size
///   31    interaction: off-task × late-in-run (the class whose label
///         flips with CONTEXT, not the action alone — what parameters
///         learn that no single-feature threshold can express)
///   32    trajectory adjacency — how closely the action's target relates
///         to what the run already touched (the run-4 lesson: the off-task
///         grep trap is indistinguishable from a legitimate follow-up grep
///         on echo alone; a real follow-up greps for something related to
///         a file just read, a wander does not)
pub fn extract(ctx: &SupervisorContext) -> [f32; SUPERVISOR_FEATURES] {
    let mut f = [0.0f32; SUPERVISOR_FEATURES];
    let is_loop = ctx.tool == LOOP_TOOL;

    // Action text: what the call says about itself. The loop pseudo-action
    // has nothing to say — its channels are the trajectory's.
    let action_text = if is_loop {
        String::new()
    } else {
        let mut t = String::with_capacity(ctx.target.len() + ctx.args_text.len() + 8);
        t.push_str(&ctx.tool);
        t.push(' ');
        t.push_str(&ctx.target);
        t.push(' ');
        t.push_str(&ctx.args_text);
        t
    };

    // 0-2: similarity of the action to the task.
    let task_empty = ctx.task.trim().is_empty();
    if !action_text.is_empty() && !task_empty {
        let a_emb = embed_text(&action_text);
        let t_emb = embed_text(&ctx.task);
        f[0] = cosine(&a_emb, &t_emb);
        f[1] = part_overlap_ratio(&action_text, &ctx.task);
        f[2] = stem_match(&ctx.task, &action_text);
    }

    // 3-5: duplication against earlier calls.
    let mut prior_same_sig = 0u32;
    let mut near_dup = 0.0f32;
    if !ctx.sig.is_empty() {
        let action_sig = ctx.sig.as_str();
        // Near-dup cosine over the most recent sigs (bounded: exact repeats
        // are caught by the string compare anyway; embedding 256 sigs per
        // check would be waste the supervisor itself should flag).
        let mut emb_cache: HashMap<String, Vec<f32>> = HashMap::new();
        let action_emb = embed_short(&action_sig);
        let start = ctx.calls.len().saturating_sub(16);
        for c in &ctx.calls[start..] {
            if c.sig.is_empty() {
                continue;
            }
            if c.sig == action_sig {
                prior_same_sig += 1;
                near_dup = 1.0;
                continue;
            }
            let e = emb_cache
                .entry(c.sig.clone())
                .or_insert_with(|| embed_short(&c.sig));
            let s = cosine(&action_emb, e);
            if s > near_dup {
                near_dup = s;
            }
        }
        // Exact-sig count over the FULL trajectory (cheap string compare).
        for c in &ctx.calls[..start] {
            if c.sig == action_sig {
                prior_same_sig += 1;
            }
        }
    }
    f[3] = if prior_same_sig > 0 { 1.0 } else { 0.0 };
    f[4] = near_dup;
    f[5] = (prior_same_sig as f32 / 4.0).min(1.0);

    // 6-7: re-reading this target — how often, and has anything written
    // to it since the last read (a re-read after a write is legitimate).
    if !is_loop && !ctx.target.is_empty() && is_read_family(&ctx.tool) {
        let mut reads = 0u32;
        let mut last_read: Option<usize> = None;
        let mut written_after_last_read = false;
        for (i, c) in ctx.calls.iter().enumerate() {
            if c.target == ctx.target {
                if is_read_family(&c.tool) {
                    reads += 1;
                    last_read = Some(i);
                    written_after_last_read = false;
                } else if is_write_family(&c.tool) {
                    if last_read.is_some() {
                        written_after_last_read = true;
                    }
                }
            }
        }
        f[6] = (reads as f32 / 4.0).min(1.0);
        f[7] = if reads > 0 && !written_after_last_read { 1.0 } else { 0.0 };
    }

    // 8-11: this tool's track record in this run.
    let mut same_tool = 0u32;
    let mut same_tool_err = 0u32;
    let mut same_tool_useless = 0u32;
    let mut same_tool_chars = 0u64;
    for c in &ctx.calls {
        if c.tool == ctx.tool {
            same_tool += 1;
            if c.errored {
                same_tool_err += 1;
            } else if c.result_chars < USELESS_RESULT_CHARS {
                same_tool_useless += 1;
            }
            same_tool_chars += c.result_chars as u64;
        }
    }
    if same_tool > 0 {
        f[8] = same_tool as f32 / ctx.calls.len().max(1) as f32;
        f[9] = same_tool_err as f32 / same_tool as f32;
        f[10] = same_tool_useless as f32 / same_tool as f32;
        f[11] = ((same_tool_chars as f32 / same_tool as f32) / 4000.0).min(1.0);
    }

    // 12-13: budget pressure.
    if ctx.budget > 0 {
        f[12] = (ctx.step as f32 / ctx.budget as f32).min(1.0);
        f[13] = if ctx.step > ctx.budget { 1.0 } else { 0.0 };
    }

    // 14-15: trajectory health.
    if !ctx.calls.is_empty() {
        let err = ctx.calls.iter().filter(|c| c.errored).count();
        let useless = ctx.calls.iter().filter(|c| !c.errored && c.result_chars < USELESS_RESULT_CHARS).count();
        f[14] = err as f32 / ctx.calls.len() as f32;
        f[15] = useless as f32 / ctx.calls.len() as f32;
    }

    // 16-17: breadth.
    f[16] = breadth_score(&ctx.tool, &ctx.target);
    f[17] = if is_search_family(&ctx.tool) && ctx.target.trim().is_empty() { 1.0 } else { 0.0 };

    // 18-19: target in the pre-turn "probably not needed" advice.
    if !ctx.target.is_empty() && !ctx.skip_targets.is_empty() {
        let target_lower = ctx.target.to_lowercase();
        let mut fuzzy = 0.0f32;
        let target_emb = embed_short(&ctx.target);
        for s in ctx.skip_targets.iter().take(8) {
            if s.is_empty() {
                continue;
            }
            if s.to_lowercase() == target_lower {
                f[18] = 1.0;
                fuzzy = 1.0;
                break;
            }
            let c = cosine(&target_emb, &embed_short(s));
            if c >= 0.90 {
                fuzzy = fuzzy.max(1.0);
            } else if stem_match(s, &ctx.target) >= 0.6 {
                fuzzy = fuzzy.max(0.5);
            }
        }
        f[19] = fuzzy;
    }

    // 20-23: tool family one-hots.
    let fam = tool_family(&ctx.tool);
    f[20 + fam as usize] = 1.0;

    // 24-29: trajectory shape (recent window).
    let n = ctx.calls.len();
    if n > 0 {
        // 24: same-sig calls among the last 4.
        let action_sig = ctx.sig.as_str();
        if !action_sig.is_empty() {
            let recent_same = ctx.calls[n.saturating_sub(4)..]
                .iter()
                .filter(|c| c.sig == action_sig)
                .count();
            f[24] = (recent_same as f32 / 2.0).min(1.0);
        }
        // 25: revisit rate among the last 6 — targets already seen earlier.
        let win_start = n.saturating_sub(6);
        let mut revisits = 0usize;
        let mut in_win = 0usize;
        for i in win_start..n {
            let t = &ctx.calls[i].target;
            if t.is_empty() {
                continue;
            }
            in_win += 1;
            if ctx.calls[..i].iter().any(|c| &c.target == t) {
                revisits += 1;
            }
        }
        if in_win > 0 {
            f[25] = revisits as f32 / in_win as f32;
        }
        // 26: drift — how far the recent targets sit from the task.
        if !task_empty {
            let recent_text: String = ctx.calls[win_start..]
                .iter()
                .map(|c| format!("{} {}", c.tool, c.target))
                .collect::<Vec<_>>()
                .join(" ");
            if !recent_text.trim().is_empty() {
                f[26] = cosine(&embed_text(&recent_text), &embed_text(&ctx.task));
            }
        }
        // 27: recent progress — edits/writes among the last 6.
        let writes = ctx.calls[win_start..].iter().filter(|c| is_write_family(&c.tool)).count();
        let window = (n - win_start).max(1);
        f[27] = writes as f32 / window as f32;
        // 28: trailing error streak.
        let mut streak = 0u32;
        for c in ctx.calls.iter().rev() {
            if c.errored {
                streak += 1;
            } else {
                break;
            }
        }
        f[28] = (streak as f32 / 3.0).min(1.0);
        // 29: trailing same-target streak (spinning on one file).
        let mut target_streak = 0u32;
        if !ctx.target.is_empty() {
            for c in ctx.calls.iter().rev() {
                if c.target == ctx.target {
                    target_streak += 1;
                } else {
                    break;
                }
            }
        } else if is_loop {
            // The loop pseudo-action inherits the streak of whatever the
            // run has been hammering last.
            if let Some(last) = ctx.calls.last() {
                if !last.target.is_empty() {
                    for c in ctx.calls.iter().rev() {
                        if c.target == last.target {
                            target_streak += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        f[29] = (target_streak as f32 / 4.0).min(1.0);
    }

    // 30: args size (a huge paste is its own smell).
    f[30] = (ctx.args_text.len() as f32 / 2000.0).min(1.0);

    // 31: off-task × late-in-run interaction.
    f[31] = (1.0 - f[0]) * f[12];

    // 32: trajectory adjacency — max cosine of the action's target against
    // the most recent distinct targets. A legitimate follow-up greps for
    // something related to a file the run just touched; an off-task wander
    // has no such anchor.
    if !is_loop && !ctx.target.is_empty() && !ctx.calls.is_empty() {
        let target_emb = embed_short(&ctx.target);
        let start = ctx.calls.len().saturating_sub(8);
        let mut seen: Vec<&str> = Vec::new();
        for c in ctx.calls[start..].iter().rev() {
            if c.target.is_empty() || seen.contains(&c.target.as_str()) {
                continue;
            }
            seen.push(&c.target);
            let s = cosine(&target_emb, &embed_short(&c.target));
            if s > f[32] {
                f[32] = s;
            }
        }
    }

    f
}

/* ── Tests ───────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    fn call(tool: &str, sig: &str, target: &str, chars: u32, errored: bool) -> TrajCall {
        TrajCall {
            tool: tool.to_string(),
            sig: sig.to_string(),
            target: target.to_string(),
            result_chars: chars,
            errored,
        }
    }

    fn sig_of(tool: &str, args: &str) -> String {
        format!("{tool}:{args}")
    }

    fn base_ctx() -> SupervisorContext {
        let args = "{\"path\":\"src/payments/webhook.rs\"}";
        SupervisorContext {
            task: "fix the verify_signature function in the payment webhook handler".to_string(),
            budget: 16,
            calls: Vec::new(),
            skip_targets: Vec::new(),
            tool: "read_file".to_string(),
            args_text: args.to_string(),
            target: "src/payments/webhook.rs".to_string(),
            sig: sig_of("read_file", args),
            step: 1,
        }
    }

    #[test]
    fn vector_shape_and_determinism() {
        let ctx = base_ctx();
        let a = extract(&ctx);
        let b = extract(&ctx);
        assert_eq!(a.len(), SUPERVISOR_FEATURES);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "extraction must be deterministic");
        }
    }

    /// Phase 1.3 (§5.3): NUL/Unicode tools+targets, a huge trajectory,
    /// u32::MAX steps against a zero budget, empty everything — the
    /// extractor must stay finite and in band (budget division is guarded
    /// by budget > 0; saturation handled by the clamps).
    #[test]
    fn hostile_contexts_stay_finite_and_bounded() {
        let mut ctx = base_ctx();
        ctx.task = "\u{0}配列😀task".to_string();
        ctx.tool = "\u{0}to\u{0}ol".to_string();
        ctx.target = String::new();
        ctx.sig = String::new();
        ctx.args_text = "x".repeat(100_000);
        ctx.step = u32::MAX;
        ctx.budget = 0; // over-budget division guard
        ctx.skip_targets = vec![String::new(); 20];
        // A long trajectory with repeated NUL-ish targets and huge chars.
        ctx.calls = (0..40)
            .map(|i| {
                call(
                    if i % 2 == 0 { "\u{0}grep" } else { "read_file" },
                    &format!("sig-\u{0}-{i}"),
                    if i % 3 == 0 { "" } else { "src/配列😀.rs" },
                    u32::MAX - i,
                    i % 5 == 0,
                )
            })
            .collect();
        let v = extract(&ctx);
        for (j, x) in v.iter().enumerate() {
            assert!(x.is_finite(), "feature {j} NaN/Inf: {x}");
            assert!((-1.0..=2.0).contains(x), "feature {j} out of band: {x}");
        }
        // And the loop pseudo-action on the same hostile trajectory.
        let mut loop_ctx = ctx.clone();
        loop_ctx.tool = LOOP_TOOL.to_string();
        loop_ctx.sig = String::new();
        loop_ctx.args_text = String::new();
        loop_ctx.target = String::new();
        let lv = extract(&loop_ctx);
        assert!(lv.iter().all(|x| x.is_finite()), "loop pseudo-action NaN/Inf");
    }

    #[test]
    fn exact_duplicate_fires() {
        let mut ctx = base_ctx();
        ctx.step = 4;
        ctx.calls = vec![
            call("grep", "grep:{\"pattern\":\"verify_signature\"}", "", 900, false),
            call("read_file", "read_file:{\"path\":\"src/payments/webhook.rs\"}", "src/payments/webhook.rs", 3000, false),
            call("grep", "grep:{\"pattern\":\"handle_delivery\"}", "", 700, false),
        ];
        let v = extract(&ctx);
        assert_eq!(v[3], 1.0, "exact sig repeat must fire f3");
        assert!(v[5] > 0.0, "repeat count channel must carry it");
        // A different call does not.
        let mut other = ctx.clone();
        other.args_text = "{\"path\":\"src/payments/sign.rs\"}".to_string();
        other.target = "src/payments/sign.rs".to_string();
        other.sig = sig_of("read_file", "{\"path\":\"src/payments/sign.rs\"}");
        let v2 = extract(&other);
        assert_eq!(v2[3], 0.0);
    }

    #[test]
    fn reread_unchanged_fires_and_write_invalidates() {
        let mut ctx = base_ctx();
        ctx.step = 3;
        ctx.args_text = "{\"path\":\"a.rs\",\"offset\":10}".to_string();
        ctx.target = "a.rs".to_string();
        ctx.sig = sig_of("read_file", "{\"path\":\"a.rs\",\"offset\":10}");
        ctx.calls = vec![
            call("read_file", "read_file:{\"path\":\"a.rs\"}", "a.rs", 2500, false),
            call("grep", "grep:{\"pattern\":\"foo\"}", "", 400, false),
        ];
        let v = extract(&ctx);
        assert!(v[6] > 0.0, "prior reads counted");
        assert_eq!(v[7], 1.0, "no write since → unchanged re-read");
        // A write to the target after the read makes it legitimate.
        let mut written = ctx.clone();
        written.calls.push(call("edit_file", "edit_file:{\"path\":\"a.rs\"}", "a.rs", 120, false));
        let v2 = extract(&written);
        assert_eq!(v2[7], 0.0, "write after last read invalidates");
    }

    #[test]
    fn broad_search_fires() {
        let mut ctx = base_ctx();
        ctx.tool = "grep".to_string();
        ctx.target = ".".to_string();
        ctx.args_text = "{\"pattern\":\".\"}".to_string();
        ctx.sig = sig_of("grep", "{\"pattern\":\".\"}");
        let v = extract(&ctx);
        assert_eq!(v[16], 1.0, "grep . is maximally broad");
        // A scoped pattern is not.
        let mut scoped = ctx.clone();
        scoped.target = "verify_signature".to_string();
        scoped.args_text = "{\"pattern\":\"verify_signature\"}".to_string();
        scoped.sig = sig_of("grep", "{\"pattern\":\"verify_signature\"}");
        assert_eq!(extract(&scoped)[16], 0.0);
        // Reading one named file is never broad.
        let mut read = ctx.clone();
        read.tool = "read_file".to_string();
        read.target = "src/main.rs".to_string();
        read.args_text = "{\"path\":\"src/main.rs\"}".to_string();
        read.sig = sig_of("read_file", "{\"path\":\"src/main.rs\"}");
        assert_eq!(extract(&read)[16], 0.0);
        // Search with no target at all.
        let mut empty = ctx.clone();
        empty.target = String::new();
        empty.args_text = "{}".to_string();
        empty.sig = sig_of("grep", "{}");
        assert_eq!(extract(&empty)[17], 1.0);
    }

    #[test]
    fn off_task_action_scores_low_similarity() {
        let on = base_ctx();
        let mut off = base_ctx();
        off.tool = "grep".to_string();
        off.target = "".to_string();
        off.args_text = "{\"pattern\":\"office plant delivery schedule\"}".to_string();
        off.sig = sig_of("grep", "{\"pattern\":\"office plant delivery schedule\"}");
        let v_on = extract(&on);
        let v_off = extract(&off);
        assert!(
            v_on[0] > v_off[0] + 0.05,
            "on-task cosine {} must beat off-task {}",
            v_on[0],
            v_off[0]
        );
        assert!(v_on[1] > v_off[1], "token overlap must agree");
    }

    #[test]
    fn skip_set_channels_fire() {
        let mut ctx = base_ctx();
        ctx.skip_targets = vec!["src/payments/webhook.rs".to_string()];
        let v = extract(&ctx);
        assert_eq!(v[18], 1.0, "exact skip-set match");
        assert_eq!(v[19], 1.0, "fuzzy channel carries it too");
    }

    #[test]
    fn tool_family_one_hots() {
        for (tool, slot) in [("read_file", 20), ("edit_file", 21), ("bash", 22), ("memory", 23)] {
            let mut ctx = base_ctx();
            ctx.tool = tool.to_string();
            ctx.sig = sig_of(tool, &ctx.args_text.clone());
            let v = extract(&ctx);
            assert_eq!(v[slot], 1.0, "{tool} must light slot {slot}");
            let others: f32 = [20, 21, 22, 23].iter().filter(|&&i| i != slot).map(|&i| v[i]).sum();
            assert_eq!(others, 0.0);
        }
    }

    #[test]
    fn spin_channels_fire_on_repeated_target() {
        let mut ctx = base_ctx();
        ctx.tool = "grep".to_string();
        ctx.target = "src/old.rs".to_string();
        ctx.args_text = "{\"pattern\":\"x1\"}".to_string();
        ctx.sig = sig_of("grep", "{\"pattern\":\"x1\"}");
        ctx.step = 6;
        ctx.calls = (0..4)
            .map(|i| call("grep", &format!("grep:{{\"pattern\":\"x{i}\"}}"), "src/old.rs", 30, false))
            .collect();
        let v = extract(&ctx);
        assert!(v[29] >= 0.99, "same-target streak must saturate: {}", v[29]);
        assert!(v[25] > 0.5, "revisit rate must fire: {}", v[25]);
        assert!(v[10] > 0.5, "useless-result history must fire: {}", v[10]);
    }

    #[test]
    fn loop_pseudo_action_is_trajectory_only() {
        let mut ctx = base_ctx();
        ctx.tool = LOOP_TOOL.to_string();
        ctx.target = String::new();
        ctx.args_text = String::new();
        ctx.sig = String::new();
        ctx.step = 9;
        ctx.calls = (0..4)
            .map(|i| call("grep", &format!("grep:{{\"pattern\":\"y{i}\"}}"), "src/old.rs", 20, false))
            .collect();
        let v = extract(&ctx);
        assert_eq!(v[23], 1.0, "loop rides the other family");
        assert_eq!(v[0], 0.0, "no action text → no similarity");
        assert_eq!(v[16], 0.0, "no breadth on a pseudo-action");
        assert_eq!(v[30], 0.0, "no args");
        assert!(v[29] >= 0.99, "the run's streak is inherited: {}", v[29]);
        assert!(v[12] > 0.5, "budget pressure still speaks");
    }

    #[test]
    fn budget_pressure_channels() {
        let mut ctx = base_ctx();
        ctx.step = 24; // budget 16 → over
        let v = extract(&ctx);
        assert_eq!(v[12], 1.0, "step/budget saturates");
        assert_eq!(v[13], 1.0, "over-budget flag fires");
        assert!(v[31] > 0.0 || v[0] > 0.99, "interaction carries late-run weight");
    }
}
