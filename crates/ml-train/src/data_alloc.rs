// ml-train/src/data_alloc.rs — synthetic dataset for the context
// allocator.
//
// Each example is one ALLOCATION DECISION POINT: a session mid-run, with
// a selected model's resolved limits, measured overhead, history state
// and growth, and the task about to be sent. Label 1 = PRESSURE: without
// action, the session hits the safe budget (85% of the window) within
// the next 3 turns. Labels are MECHANICAL — the future turns are
// simulated from the same growth regime, so the net can only pass by
// learning the signals the features carry: fill, growth, acceleration,
// overhead, attachments, output behaviour, and the model's window.
//
// Channel-coverage rule (the freshness lesson): every feature channel is
// ACTIVE in training. Holdout families test unseen CONSTRUCTIONS of
// trained channels (burst at a new window, heavy growth on a tiny
// window, the multi-window near-brink sweep), never an untrained
// channel. The unknown-caps channel trains (G14) — the conservative
// floor for it in model.rs is mechanical, but the net still sees it.

use sofuu_core::ml::alloc::features::{
    extract, AllocInput, THINK_BUDGET, THINK_EFFORT, THINK_NONE, THINK_UNKNOWN,
};

use crate::train::{Example, Rng};

/* ── Growth regimes (tokens added per turn) ──────────────────────── */

struct Growth {
    lo: f32,
    hi: f32,
    /// Fraction of history tokens that are tool transcripts.
    tool_lo: f32,
    tool_hi: f32,
}

const PROSE: Growth = Growth { lo: 150.0, hi: 500.0, tool_lo: 0.0, tool_hi: 0.2 };
const CODING: Growth = Growth { lo: 800.0, hi: 4000.0, tool_lo: 0.4, tool_hi: 0.8 };
const HEAVY: Growth = Growth { lo: 3000.0, hi: 9000.0, tool_lo: 0.6, tool_hi: 0.9 };
const BURST: Growth = Growth { lo: 300.0, hi: 900.0, tool_lo: 0.1, tool_hi: 0.4 };

/* ── Task kinds ──────────────────────────────────────────────────── */

#[derive(Clone, Copy)]
enum Task {
    Coding,
    Writing,
    Analysis,
    Chat,
}

impl Task {
    fn answer_range(self) -> (f32, f32) {
        match self {
            Task::Coding => (150.0, 500.0),
            Task::Writing => (400.0, 1500.0),
            Task::Analysis => (200.0, 700.0),
            Task::Chat => (80.0, 250.0),
        }
    }
    fn tool_heavy(self) -> bool {
        matches!(self, Task::Coding)
    }
    fn writing(self) -> bool {
        matches!(self, Task::Writing)
    }
    fn task_tk(self, rng: &mut Rng) -> f32 {
        let base = match self {
            Task::Coding => 20.0 + rng.uniform() * 60.0,
            Task::Writing => 30.0 + rng.uniform() * 90.0,
            Task::Analysis => 20.0 + rng.uniform() * 80.0,
            Task::Chat => 5.0 + rng.uniform() * 30.0,
        };
        base
    }
}

/* ── Family definitions ──────────────────────────────────────────── */

struct Family {
    group: u32,
    name: &'static str,
    window: i64,
    growth: &'static Growth,
    overhead: f32,
    task: Task,
    /// History fill range as a fraction of the window.
    fill_lo: f32,
    fill_hi: f32,
    /// Growth acceleration range (1.0 = steady).
    accel_lo: f32,
    accel_hi: f32,
    /// Chance the session carries an attachment this turn (fraction of
    /// the window, up to 10%).
    attach_chance: f32,
    /// Chance the overhead meter is calibrated.
    calibrated_chance: f32,
    /// Chance a compaction summary is already present.
    summary_chance: f32,
    /// Chance the recent output was starved (finish_reason length).
    length_stop_chance: f32,
    /// Thinking kind the resolved model reports.
    thinking: u8,
    /// Unknown caps (window/max_output reported as 0 → the allocator's
    /// conservative path). The simulated window stays the real one so
    /// the label is still mechanical.
    unknown_caps: bool,
    /// Examples in this family.
    n: usize,
}

fn families() -> Vec<Family> {
    vec![
        // Window tiers × coding growth — the core pressure channel.
        Family { group: 0, name: "A0 8k coding", window: 8_192, growth: &CODING, overhead: 300.0, task: Task::Coding, fill_lo: 0.10, fill_hi: 0.95, accel_lo: 0.7, accel_hi: 1.6, attach_chance: 0.1, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.05, thinking: THINK_EFFORT, unknown_caps: false, n: 200 },
        Family { group: 1, name: "A1 8k prose", window: 8_192, growth: &PROSE, overhead: 300.0, task: Task::Analysis, fill_lo: 0.05, fill_hi: 0.90, accel_lo: 0.7, accel_hi: 1.4, attach_chance: 0.05, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.03, thinking: THINK_NONE, unknown_caps: false, n: 190 },
        Family { group: 2, name: "A2 32k coding", window: 32_768, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.05, fill_hi: 0.95, accel_lo: 0.6, accel_hi: 2.0, attach_chance: 0.1, calibrated_chance: 0.9, summary_chance: 0.25, length_stop_chance: 0.05, thinking: THINK_EFFORT, unknown_caps: false, n: 220 },
        Family { group: 3, name: "A3 32k prose", window: 32_768, growth: &PROSE, overhead: 300.0, task: Task::Chat, fill_lo: 0.05, fill_hi: 0.85, accel_lo: 0.7, accel_hi: 1.3, attach_chance: 0.05, calibrated_chance: 0.9, summary_chance: 0.15, length_stop_chance: 0.02, thinking: THINK_NONE, unknown_caps: false, n: 190 },
        Family { group: 4, name: "A4 131k coding", window: 131_072, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.05, fill_hi: 0.95, accel_lo: 0.6, accel_hi: 2.2, attach_chance: 0.15, calibrated_chance: 0.9, summary_chance: 0.25, length_stop_chance: 0.05, thinking: THINK_EFFORT, unknown_caps: false, n: 220 },
        Family { group: 5, name: "A5 131k heavy", window: 131_072, growth: &HEAVY, overhead: 6000.0, task: Task::Coding, fill_lo: 0.10, fill_hi: 0.95, accel_lo: 0.7, accel_hi: 2.0, attach_chance: 0.2, calibrated_chance: 0.9, summary_chance: 0.3, length_stop_chance: 0.06, thinking: THINK_BUDGET, unknown_caps: false, n: 210 },
        Family { group: 6, name: "A6 200k coding", window: 200_000, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.05, fill_hi: 0.95, accel_lo: 0.6, accel_hi: 2.0, attach_chance: 0.15, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.04, thinking: THINK_BUDGET, unknown_caps: false, n: 200 },
        Family { group: 7, name: "A7 1M coding", window: 1_000_000, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.05, fill_hi: 0.95, accel_lo: 0.6, accel_hi: 2.0, attach_chance: 0.15, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.03, thinking: THINK_EFFORT, unknown_caps: false, n: 200 },
        Family { group: 8, name: "A8 1M heavy", window: 1_000_000, growth: &HEAVY, overhead: 6000.0, task: Task::Coding, fill_lo: 0.10, fill_hi: 0.95, accel_lo: 0.7, accel_hi: 2.5, attach_chance: 0.25, calibrated_chance: 0.9, summary_chance: 0.3, length_stop_chance: 0.05, thinking: THINK_EFFORT, unknown_caps: false, n: 200 },
        // Overhead channel: same window+growth, overhead varies widely.
        Family { group: 9, name: "A9 overhead sweep", window: 32_768, growth: &CODING, overhead: -1.0, task: Task::Coding, fill_lo: 0.30, fill_hi: 0.90, accel_lo: 0.8, accel_hi: 1.5, attach_chance: 0.1, calibrated_chance: 1.0, summary_chance: 0.2, length_stop_chance: 0.04, thinking: THINK_EFFORT, unknown_caps: false, n: 200 },
        // Burst channel: prose + attachment bursts (32k trains it).
        Family { group: 10, name: "A10 32k burst", window: 32_768, growth: &BURST, overhead: 800.0, task: Task::Analysis, fill_lo: 0.10, fill_hi: 0.90, accel_lo: 0.8, accel_hi: 1.4, attach_chance: 0.65, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.03, thinking: THINK_NONE, unknown_caps: false, n: 210 },
        // Held-out burst construction (131k) — val fold.
        Family { group: 11, name: "A11 131k burst", window: 131_072, growth: &BURST, overhead: 800.0, task: Task::Analysis, fill_lo: 0.10, fill_hi: 0.90, accel_lo: 0.8, accel_hi: 1.4, attach_chance: 0.65, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.03, thinking: THINK_EFFORT, unknown_caps: false, n: 200 },
        // Acceleration channel: decelerating → accelerating growth.
        Family { group: 12, name: "A12 accel sweep", window: 32_768, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.30, fill_hi: 0.85, accel_lo: 0.3, accel_hi: 3.0, attach_chance: 0.1, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.04, thinking: THINK_EFFORT, unknown_caps: false, n: 210 },
        // Output-pressure channel: starved outputs, big answers.
        Family { group: 13, name: "A13 output pressure", window: 131_072, growth: &CODING, overhead: 2500.0, task: Task::Writing, fill_lo: 0.20, fill_hi: 0.90, accel_lo: 0.8, accel_hi: 1.6, attach_chance: 0.1, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.45, thinking: THINK_BUDGET, unknown_caps: false, n: 200 },
        // Unknown-caps channel (trains; the floor is mechanical).
        Family { group: 14, name: "A14 unknown caps", window: 32_768, growth: &CODING, overhead: 1500.0, task: Task::Coding, fill_lo: 0.10, fill_hi: 0.95, accel_lo: 0.7, accel_hi: 1.8, attach_chance: 0.1, calibrated_chance: 0.5, summary_chance: 0.2, length_stop_chance: 0.05, thinking: THINK_UNKNOWN, unknown_caps: true, n: 200 },
        // Summary-recovery channel: big history, already compacted,
        // growth restarted — usually slack despite the fill.
        Family { group: 15, name: "A15 summary recovery", window: 131_072, growth: &PROSE, overhead: 2500.0, task: Task::Analysis, fill_lo: 0.40, fill_hi: 0.85, accel_lo: 0.6, accel_hi: 1.2, attach_chance: 0.05, calibrated_chance: 0.9, summary_chance: 1.0, length_stop_chance: 0.02, thinking: THINK_EFFORT, unknown_caps: false, n: 180 },
        // Boundary family: states engineered near the decision edge —
        // gives threshold selection real boundary samples (val fold).
        Family { group: 16, name: "A16 boundary", window: 32_768, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.62, fill_hi: 0.80, accel_lo: 0.9, accel_hi: 1.3, attach_chance: 0.15, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.05, thinking: THINK_EFFORT, unknown_caps: false, n: 240 },
        // Held-out heavy-on-tiny-window construction — test fold.
        Family { group: 17, name: "A17 8k heavy", window: 8_192, growth: &HEAVY, overhead: 600.0, task: Task::Coding, fill_lo: 0.10, fill_hi: 0.90, accel_lo: 0.7, accel_hi: 1.8, attach_chance: 0.2, calibrated_chance: 0.9, summary_chance: 0.25, length_stop_chance: 0.06, thinking: THINK_NONE, unknown_caps: false, n: 200 },
        Family { group: 18, name: "A18 200k prose", window: 200_000, growth: &PROSE, overhead: 300.0, task: Task::Chat, fill_lo: 0.02, fill_hi: 0.70, accel_lo: 0.7, accel_hi: 1.3, attach_chance: 0.05, calibrated_chance: 0.9, summary_chance: 0.1, length_stop_chance: 0.02, thinking: THINK_BUDGET, unknown_caps: false, n: 180 },
        Family { group: 19, name: "A19 writing", window: 131_072, growth: &PROSE, overhead: 800.0, task: Task::Writing, fill_lo: 0.05, fill_hi: 0.85, accel_lo: 0.7, accel_hi: 1.5, attach_chance: 0.15, calibrated_chance: 0.9, summary_chance: 0.15, length_stop_chance: 0.15, thinking: THINK_BUDGET, unknown_caps: false, n: 190 },
        // Held-out chat construction — val fold.
        Family { group: 20, name: "A20 32k chat", window: 32_768, growth: &PROSE, overhead: 300.0, task: Task::Chat, fill_lo: 0.02, fill_hi: 0.80, accel_lo: 0.7, accel_hi: 1.2, attach_chance: 0.03, calibrated_chance: 0.9, summary_chance: 0.1, length_stop_chance: 0.02, thinking: THINK_NONE, unknown_caps: false, n: 180 },
        // tool_frac channel: same growth volume, transcript share varies.
        Family { group: 21, name: "A21 toolfrac sweep", window: 32_768, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.20, fill_hi: 0.90, accel_lo: 0.8, accel_hi: 1.5, attach_chance: 0.1, calibrated_chance: 0.9, summary_chance: 0.2, length_stop_chance: 0.04, thinking: THINK_EFFORT, unknown_caps: false, n: 190 },
        // Uncalibrated meter channel.
        Family { group: 22, name: "A22 uncalibrated", window: 131_072, growth: &CODING, overhead: 0.0, task: Task::Coding, fill_lo: 0.10, fill_hi: 0.90, accel_lo: 0.7, accel_hi: 1.8, attach_chance: 0.1, calibrated_chance: 0.0, summary_chance: 0.2, length_stop_chance: 0.04, thinking: THINK_EFFORT, unknown_caps: false, n: 180 },
        // Held-out near-brink multi-window sweep — test fold.
        Family { group: 23, name: "A23 brink sweep", window: -1, growth: &CODING, overhead: 2500.0, task: Task::Coding, fill_lo: 0.60, fill_hi: 0.95, accel_lo: 0.8, accel_hi: 2.0, attach_chance: 0.15, calibrated_chance: 0.9, summary_chance: 0.3, length_stop_chance: 0.05, thinking: THINK_EFFORT, unknown_caps: false, n: 220 },
    ]
}

const BRINK_WINDOWS: &[i64] = &[8_192, 32_768, 131_072, 200_000, 1_000_000];

/// The safe budget the driver actually enforces (chat.rs ctxBudget).
const SAFE_BUDGET: f32 = 0.85;
/// Turns ahead the allocator must see coming.
const HORIZON: usize = 3;

fn simulate(f: &Family, rng: &mut Rng) -> (AllocInput, f32) {
    let window = if f.window < 0 {
        BRINK_WINDOWS[(rng.next_u32() as usize) % BRINK_WINDOWS.len()]
    } else {
        f.window
    };
    let win = window as f32;

    let overhead = if f.overhead < 0.0 {
        // Overhead sweep family: minimal → mesh-heavy, capped at win/8.
        (200.0 + rng.uniform() * 5800.0).min(win / 8.0)
    } else {
        (f.overhead * (0.8 + rng.uniform() * 0.4)).min(win / 8.0)
    };
    let calibrated = rng.uniform() < f.calibrated_chance;

    let fill = f.fill_lo + rng.uniform() * (f.fill_hi - f.fill_lo);
    let history = (fill * win).max(0.0);
    let growth_mean = f.growth.lo + rng.uniform() * (f.growth.hi - f.growth.lo);
    let accel = f.accel_lo + rng.uniform() * (f.accel_hi - f.accel_lo);
    let turns = ((history / growth_mean.max(1.0)).ceil() as u32).clamp(1, 60);

    let mut tool_frac = f.growth.tool_lo + rng.uniform() * (f.growth.tool_hi - f.growth.tool_lo);
    if f.group == 21 {
        tool_frac = rng.uniform() * 0.9; // toolfrac sweep family
    }

    let attach = if rng.uniform() < f.attach_chance {
        // Burst families carry big attachments; others small ones.
        let cap = if std::ptr::eq(f.growth, &BURST) { 0.10 } else { 0.03 };
        rng.uniform() * cap * win
    } else {
        0.0
    };

    let (ans_lo, ans_hi) = f.task.answer_range();
    let mean_answer = ans_lo + rng.uniform() * (ans_hi - ans_lo);
    let max_answer = mean_answer * (1.5 + rng.uniform() * 1.5);
    let saw_length = rng.uniform() < f.length_stop_chance;
    let summary = rng.uniform() < f.summary_chance;
    let task_tk = f.task.task_tk(rng);

    let input = AllocInput {
        ctx_window: if f.unknown_caps { 0 } else { window },
        max_output: if f.unknown_caps {
            0
        } else {
            (window / 4).clamp(512, 128_000)
        },
        thinking: f.thinking,
        overhead_tk: if calibrated { overhead as i64 } else { 0 },
        calibrated,
        history_tk: history as i64,
        turns,
        tool_frac,
        summary_present: summary,
        growth_tk: growth_mean,
        growth_accel: accel,
        task_tk: task_tk as i64,
        attach_tk: attach as i64,
        tool_heavy: f.task.tool_heavy(),
        writing: f.task.writing(),
        mean_answer_tk: mean_answer,
        max_answer_tk: max_answer,
        saw_length_stop: saw_length,
    };

    /* Mechanical label: simulate HORIZON future turns under the same
     * regime. Burst families may add one more attachment burst in the
     * window. Pressure = the safe budget is breached at any point. */
    let mut load = overhead + history + attach + task_tk;
    let mut g = growth_mean;
    let mut pressured = false;
    for t in 0..HORIZON {
        // Accelerating regimes compound; decelerating ones relax.
        g *= if t == 0 { 1.0 } else { accel.min(1.6) };
        let mut add = g * (0.8 + rng.uniform() * 0.4) + mean_answer;
        if std::ptr::eq(f.growth, &BURST) && t == 1 && rng.uniform() < 0.5 {
            add += rng.uniform() * 0.08 * win;
        }
        load += add;
        if load > SAFE_BUDGET * win {
            pressured = true;
            break;
        }
    }
    (input, if pressured { 1.0 } else { 0.0 })
}

pub fn build() -> Vec<Example> {
    let mut rng = Rng(0xA110_C7A7);
    let mut out = Vec::new();
    for f in families() {
        let mut pos = 0usize;
        for _ in 0..f.n {
            let (input, y) = simulate(&f, &mut rng);
            if y >= 0.5 {
                pos += 1;
            }
            let x = extract(&input).to_vec();
            out.push(Example {
                x,
                y,
                group: f.group,
                text: format!(
                    "win={} hist={} growth={:.0} accel={:.2} overhead={} attach={}",
                    input.ctx_window, input.history_tk, input.growth_tk,
                    input.growth_accel, input.overhead_tk, input.attach_tk
                ),
                task: f.name.to_string(),
            });
        }
        println!(
            "  {:<22} n={} positive={} ({:.0}%)",
            f.name,
            f.n,
            pos,
            100.0 * pos as f32 / f.n as f32
        );
    }
    out
}
