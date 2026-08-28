// ml/alloc/features.rs — feature extraction for the context allocator.
//
// One feature vector per ALLOCATION DECISION POINT (a turn about to be
// built). The load-bearing question is "how much pressure is this window
// under, given THIS model's real limits and THIS session's shape" — so
// the features start with the resolved model capabilities (the allocator
// fetches model details FIRST, then allocates), then the measured
// per-request overhead, the history state and its growth, the task shape,
// and the recent output behaviour. Deterministic: scalar f32, fixed
// order, no hidden state.

pub const ALLOC_FEATURES: usize = 24;

/// Thinking-mode kind as the allocator sees it (mirrors model_caps'
/// Thinking enum flattened to a u8 for the feature layer).
pub const THINK_NONE: u8 = 0;
pub const THINK_EFFORT: u8 = 1;
pub const THINK_BUDGET: u8 = 2;
pub const THINK_UNKNOWN: u8 = 3;

/// Everything the allocator knows at decision time. Caps fields carry the
/// RESOLVED values (registry lookup + config clamps already applied by
/// model::plan — 0 = unknown model, no live source reported a limit).
#[derive(Clone, Debug, Default)]
pub struct AllocInput {
    /* ── The selected model's capabilities (resolved first) ───────── */
    /// Context window in tokens. 0 = unknown.
    pub ctx_window: i64,
    /// Maximum output tokens. 0 = unknown.
    pub max_output: i64,
    /// Thinking mode kind (THINK_* consts).
    pub thinking: u8,

    /* ── Measured per-request overhead (the footer meter) ─────────── */
    /// System prompt + tool schemas + ephemeral context, calibrated from
    /// a real provider prompt_tokens report. 0 when uncalibrated.
    pub overhead_tk: i64,
    pub calibrated: bool,

    /* ── History state ─────────────────────────────────────────────── */
    pub history_tk: i64,
    pub turns: u32,
    /// Fraction of history tokens that are tool transcripts (results).
    pub tool_frac: f32,
    /// A compaction summary is present (history already folded once).
    pub summary_present: bool,
    /// Mean tokens added per turn over the last three turns.
    pub growth_tk: f32,
    /// Last turn's growth divided by the mean (1.0 = steady).
    pub growth_accel: f32,

    /* ── Task shape ────────────────────────────────────────────────── */
    pub task_tk: i64,
    /// Attachment tokens about to ride this turn (@file mentions…).
    pub attach_tk: i64,
    /// Task reads as tool-heavy (read/search/fix/debug/build…).
    pub tool_heavy: bool,
    /// Task reads as long-form output (write/summarize/explain/draft…).
    pub writing: bool,

    /* ── Output behaviour ──────────────────────────────────────────── */
    pub mean_answer_tk: f32,
    pub max_answer_tk: f32,
    /// A recent answer stopped on finish_reason length (output starved).
    pub saw_length_stop: bool,
}

fn log2_norm(v: f32, div: f32) -> f32 {
    if v <= 1.0 {
        return 0.0;
    }
    (v.log2() / div).min(1.0)
}

fn ratio(part: f32, win: f32) -> f32 {
    if win <= 0.0 {
        return 0.0;
    }
    (part / win).clamp(0.0, 1.5)
}

fn flag(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// Extract the fixed-order feature vector. SAME code trains and serves —
/// ml-train's data_alloc imports this function, so train/serve skew is
/// structurally impossible (§10 bar 5).
pub fn extract(inp: &AllocInput) -> [f32; ALLOC_FEATURES] {
    let win = inp.ctx_window as f32;
    let out = inp.max_output as f32;
    let hist = inp.history_tk as f32;
    let overhead = inp.overhead_tk as f32;
    let attach = inp.attach_tk as f32;
    let growth = inp.growth_tk.max(0.0);

    // Slack left for future turns if nothing changes: window minus what
    // is already spoken for (overhead, history, this turn's attachments,
    // and a couple of answers' worth of output).
    let spoken_for = overhead + hist + attach + inp.mean_answer_tk.max(0.0) * 2.0;
    let slack = if win > 0.0 { ((win - spoken_for) / win).clamp(0.0, 1.0) } else { 0.0 };

    // How many turns of current growth fit in the slack (log-scaled).
    let turns_to_fill = if win > 0.0 {
        let t = (slack * win) / growth.max(1.0);
        log2_norm(1.0 + t, 6.0)
    } else {
        0.0
    };

    // The output reservation the window can actually host right now.
    let out_reserve = if win > 0.0 {
        (out.min(win * 0.25).max(0.0)) / win
    } else {
        0.0
    };

    [
        log2_norm(win, 20.0),                          // 0: window size (2^20 = 1M)
        flag(inp.ctx_window > 0),                      // 1: caps known
        log2_norm(out, 17.0),                          // 2: max output size
        flag(inp.thinking == THINK_EFFORT),            // 3
        flag(inp.thinking == THINK_BUDGET),            // 4
        flag(inp.thinking == THINK_UNKNOWN),           // 5
        ratio(overhead, win),                          // 6: measured overhead / window
        flag(inp.calibrated),                          // 7: overhead calibrated
        ratio(hist, win),                              // 8: history fill
        log2_norm(inp.turns as f32 + 1.0, 8.0),        // 9: session length
        inp.tool_frac.clamp(0.0, 1.0),                 // 10: tool-transcript share
        flag(inp.summary_present),                     // 11: already compacted once
        ratio(growth, win),                            // 12: per-turn growth / window
        (inp.growth_accel.clamp(0.0, 3.0)) / 3.0,      // 13: growth acceleration
        ratio(inp.task_tk as f32, win),                // 14: task size
        ratio(attach, win),                            // 15: attachment load
        flag(inp.tool_heavy),                          // 16: tool-heavy task
        flag(inp.writing),                             // 17: long-form task
        ratio(inp.mean_answer_tk.max(0.0), win),       // 18: mean answer
        ratio(inp.max_answer_tk.max(0.0), win),        // 19: largest answer
        flag(inp.saw_length_stop),                     // 20: output starved recently
        slack,                                         // 21: slack fraction
        turns_to_fill,                                 // 22: turns until full at growth
        out_reserve,                                   // 23: feasible output reserve
    ]
}

/// Batch form — mirrors the other models' extract_all so the trainer can
/// map a whole dataset through the identical code path.
pub fn extract_all(inputs: &[AllocInput]) -> Vec<[f32; ALLOC_FEATURES]> {
    inputs.iter().map(extract).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_bounded_and_deterministic() {
        let inp = AllocInput {
            ctx_window: 131072,
            max_output: 16384,
            thinking: THINK_EFFORT,
            overhead_tk: 2500,
            calibrated: true,
            history_tk: 60000,
            turns: 14,
            tool_frac: 0.6,
            summary_present: false,
            growth_tk: 3000.0,
            growth_accel: 1.4,
            task_tk: 40,
            attach_tk: 4000,
            tool_heavy: true,
            writing: false,
            mean_answer_tk: 350.0,
            max_answer_tk: 900.0,
            saw_length_stop: false,
        };
        let a = extract(&inp);
        let b = extract(&inp);
        assert_eq!(a.len(), ALLOC_FEATURES);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()));
        for (i, v) in a.iter().enumerate() {
            assert!(v.is_finite(), "feature {i} not finite: {v}");
            assert!((0.0..=1.5).contains(v), "feature {i} out of band: {v}");
        }
    }

    #[test]
    fn unknown_model_is_expressible() {
        let inp = AllocInput::default(); // all zero = unknown model, fresh session
        let v = extract(&inp);
        assert_eq!(v[1], 0.0, "known flag must be 0 for unknown caps");
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
