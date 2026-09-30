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

/// i64 → f32 without the Inf blowup of a raw `as f32` on huge values
/// (Phase 1.3: a JSON-side magnitude like 1e18 must saturate, not turn
/// every ratio it touches into ±Inf/NaN).
fn sat_f32(v: i64) -> f32 {
    if v > (f32::MAX as i64) {
        f32::INFINITY
    } else if v < (f32::MIN as i64) {
        f32::NEG_INFINITY
    } else {
        v as f32
    }
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
    let win = sat_f32(inp.ctx_window);
    let out = sat_f32(inp.max_output);
    let hist = sat_f32(inp.history_tk);
    let overhead = sat_f32(inp.overhead_tk);
    let attach = sat_f32(inp.attach_tk);
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
        log2_norm(sat_f32(inp.turns as i64) + 1.0, 8.0),        // 9: session length
        inp.tool_frac.clamp(0.0, 1.0),                 // 10: tool-transcript share
        flag(inp.summary_present),                     // 11: already compacted once
        ratio(growth, win),                            // 12: per-turn growth / window
        (inp.growth_accel.clamp(0.0, 3.0)) / 3.0,      // 13: growth acceleration
        ratio(sat_f32(inp.task_tk), win),                // 14: task size
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

    /// Phase 1.3 (§5.3 "zero denominators and overflows"): every field
    /// pushed to a hostile extreme — saturating window, huge history,
    /// negative growth, oversized everything — must still yield a finite
    /// vector in the feature band. No input, however wrong, may produce
    /// NaN/Inf into the net.
    #[test]
    fn hostile_extremes_stay_finite_and_bounded() {
        let hostile = AllocInput {
            ctx_window: i64::MAX,          // saturates to f32::INFINITY
            max_output: i64::MIN,          // → f32::NEG_INFINITY
            thinking: THINK_UNKNOWN,
            overhead_tk: i64::MAX,
            calibrated: true,
            history_tk: i64::MAX - 1,      // slack computed on saturated floats
            turns: u32::MAX,
            tool_frac: 9.9,                // beyond [0,1] — clamped by extract
            summary_present: true,
            growth_tk: 1.0e30,             // f32-finite but huge
            growth_accel: -4.0,            // negative — clamped
            task_tk: i64::MIN + 1,
            attach_tk: i64::MAX - 2,
            tool_heavy: true,
            writing: true,
            mean_answer_tk: f32::MAX,
            max_answer_tk: f32::MAX,
            saw_length_stop: true,
        };
        let v = extract(&hostile);
        for (i, x) in v.iter().enumerate() {
            assert!(x.is_finite(), "feature {i} not finite under hostile input: {x}");
            assert!((-1.0..=2.0).contains(x), "feature {i} wildly out of band: {x}");
        }
        // A NaN-carrying input must survive the same way — sanitize is the
        // last line before the net (Phase 1.2).
        let mut nan_inp = hostile.clone();
        nan_inp.tool_frac = f32::NAN;
        nan_inp.growth_tk = f32::NAN;
        let mut v2 = extract(&nan_inp);
        crate::ml::sanitize_features(&mut v2);
        assert!(v2.iter().all(|x| x.is_finite()), "sanitize must clear NaN features");
    }

    /// Zero magnitudes: zero-window, zero-history, zero-growth must not
    /// divide by zero anywhere (§5.3).
    #[test]
    fn zero_magnitudes_never_diverge() {
        let zero = AllocInput {
            ctx_window: 0,
            max_output: 0,
            thinking: THINK_NONE,
            overhead_tk: 0,
            calibrated: false,
            history_tk: 0,
            turns: 0,
            tool_frac: 0.0,
            summary_present: false,
            growth_tk: 0.0,
            growth_accel: 0.0,
            task_tk: 0,
            attach_tk: 0,
            tool_heavy: false,
            writing: false,
            mean_answer_tk: 0.0,
            max_answer_tk: 0.0,
            saw_length_stop: false,
        };
        let v = extract(&zero);
        assert!(v.iter().all(|x| x.is_finite() && x >= &0.0), "zero state must be finite");
        assert_eq!(v[21], 0.0, "no slack computable on a zero window");
    }
}
