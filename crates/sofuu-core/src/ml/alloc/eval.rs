// ml/alloc/eval.rs — acceptance re-checks against the COMMITTED weights.
//
// The trainer proved the gates once; these tests prove the blob that
// shipped still meets them (a re-bake that regressed would fail here, in
// CI, not in a user session). Plus the property that makes the allocator
// safe to apply as-is: NO plan ever exceeds the resolved model's hard
// limits, whatever the session state.

use super::features::{AllocInput, THINK_BUDGET, THINK_EFFORT, THINK_NONE, THINK_UNKNOWN};
use super::model::{self, H1, H2, IN_DIM, PARAMS, THRESHOLD};
use super::policy::{self, Resolved, Source, UNKNOWN_MAX_OUTPUT, UNKNOWN_WINDOW};
use crate::ml::net::TinyMlp;

fn resolved(window: i64, max_output: i64, known: bool) -> Resolved {
    Resolved {
        window,
        max_output,
        known,
        source: if known { Source::Registry } else { Source::Default },
        clamped_config: false,
    }
}

fn fresh_session(window: i64, max_output: i64) -> AllocInput {
    AllocInput {
        ctx_window: window,
        max_output,
        thinking: THINK_EFFORT,
        overhead_tk: 400,
        calibrated: true,
        history_tk: window / 20,
        turns: 2,
        tool_frac: 0.3,
        summary_present: false,
        growth_tk: (window as f32) * 0.01,
        growth_accel: 1.0,
        task_tk: 30,
        attach_tk: 0,
        tool_heavy: true,
        writing: false,
        mean_answer_tk: 200.0,
        max_answer_tk: 400.0,
        saw_length_stop: false,
    }
}

/// Deterministic sweep of session states × model limits — the plan must
/// respect every hard bound on every point. This is the property that
/// lets the caller apply a plan without re-checking it.
#[test]
fn plans_respect_hard_limits_across_the_state_space() {
    let windows: &[i64] = &[4_096, 8_192, 32_768, 131_072, 200_000, 1_000_000];
    let fills: &[f32] = &[0.02, 0.2, 0.5, 0.7, 0.9, 0.98];
    let growths: &[f32] = &[0.0, 0.005, 0.02, 0.08];
    let thinks: &[u8] = &[THINK_NONE, THINK_EFFORT, THINK_BUDGET, THINK_UNKNOWN];
    let mut n = 0;
    for &win in windows {
        for &fill in fills {
            for &g in growths {
                for &th in thinks {
                    let inp = AllocInput {
                        ctx_window: win,
                        max_output: (win / 4).max(512),
                        thinking: th,
                        overhead_tk: win / 40,
                        calibrated: true,
                        history_tk: (win as f32 * fill) as i64,
                        turns: 12,
                        tool_frac: 0.5,
                        summary_present: fill > 0.6,
                        growth_tk: win as f32 * g,
                        growth_accel: 1.5,
                        task_tk: 50,
                        attach_tk: win / 20,
                        tool_heavy: true,
                        writing: false,
                        mean_answer_tk: 300.0,
                        max_answer_tk: 1200.0,
                        saw_length_stop: false,
                    };
                    let p = model::plan_with(&inp, resolved(win, win / 4, true));
                    assert!(p.max_output <= win / 4, "output exceeds model max: {}", p.max_output);
                    assert!(p.max_output >= 512 || p.max_output == win / 4, "output below floor: {}", p.max_output);
                    assert!(p.window == win);
                    assert!((0.50..=0.70).contains(&p.compact_at), "compactAt out of band: {}", p.compact_at);
                    assert!((4_000..=32_768).contains(&p.tool_cap_chars), "tool cap out of band: {}", p.tool_cap_chars);
                    assert!(p.recall_budget_tok <= win / 2, "recall too big: {}", p.recall_budget_tok);
                    assert!(p.attach_budget_tok <= win / 2, "attach too big: {}", p.attach_budget_tok);
                    assert!(p.pressure >= 0.0 && p.pressure <= 1.0);
                    n += 1;
                }
            }
        }
    }
    assert!(n >= 500, "sweep must cover the state space, got {n}");
}

/// Near-overflow with accelerating growth is a pressure event; a fresh
/// session on the same window is not. The behavioural anchors the
/// trainer's acceptance gates proved, re-checked on the committed blob.
#[test]
fn pressure_fires_near_overflow_and_stays_quiet_when_fresh() {
    let win = 32_768i64;
    let mut tight = fresh_session(win, 8_192);
    tight.history_tk = (win as f32 * 0.93) as i64;
    tight.growth_tk = win as f32 * 0.06;
    tight.growth_accel = 2.0;
    tight.attach_tk = win / 16;
    let p = model::plan_with(&tight, resolved(win, 8_192, true));
    assert!(
        p.pressure >= THRESHOLD,
        "near-overflow session must register pressure: {:.3} < {}",
        p.pressure,
        THRESHOLD
    );
    assert!(p.compact_at <= 0.58, "tight session compacts early, got {}", p.compact_at);

    let fresh = fresh_session(win, 8_192);
    let p = model::plan_with(&fresh, resolved(win, 8_192, true));
    assert!(
        p.pressure < THRESHOLD,
        "fresh session must stay quiet: {:.3} >= {}",
        p.pressure,
        THRESHOLD
    );
    assert!(p.compact_at >= 0.62, "slack session compacts late, got {}", p.compact_at);
}

/// Unknown model → conservative window, pressure floored, and the plan
/// still inside every bound. The allocator must never allocate against a
/// window the selected model might not have.
#[test]
fn unknown_model_allocates_conservatively() {
    let inp = fresh_session(0, 0);
    let p = model::plan_with(&inp, resolved(UNKNOWN_WINDOW, UNKNOWN_MAX_OUTPUT, false));
    assert_eq!(p.window, UNKNOWN_WINDOW);
    assert!(!p.known);
    assert!(p.pressure >= 0.5, "unknown model pressure floored, got {}", p.pressure);
    assert!(p.max_output <= UNKNOWN_MAX_OUTPUT);
    assert!(p.notes.iter().any(|n| n.contains("unknown model")));
}

/// Writing tasks reserve more output — within the model's max.
#[test]
fn writing_tasks_boost_output_within_caps() {
    let win = 131_072i64;
    let max_out = 16_384i64;
    let mut prose = fresh_session(win, max_out);
    prose.history_tk = win / 10;
    let mut writing = prose.clone();
    writing.writing = true;
    let p1 = model::plan_with(&prose, resolved(win, max_out, true));
    let p2 = model::plan_with(&writing, resolved(win, max_out, true));
    assert!(p2.max_output >= p1.max_output, "writing must not shrink output");
    assert!(p2.max_output <= max_out);
}

#[test]
fn plan_is_bit_exact() {
    let inp = fresh_session(131_072, 16_384);
    let a = model::plan_with(&inp, resolved(131_072, 16_384, true));
    let b = model::plan_with(&inp, resolved(131_072, 16_384, true));
    assert_eq!(a.pressure.to_bits(), b.pressure.to_bits(), "forward must be deterministic");
    assert_eq!(a.max_output, b.max_output);
    assert_eq!(a.tool_cap_chars, b.tool_cap_chars);
}

#[test]
fn blob_integrity_and_architecture() {
    static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");
    let net = TinyMlp::from_blob(WEIGHTS).expect("committed blob loads");
    assert_eq!((net.in_dim, net.h1, net.h2), (IN_DIM, H1, H2));
    assert_eq!(net.w.len() as u32, PARAMS);
    assert!(THRESHOLD > 0.0 && THRESHOLD < 1.0);

    // A single corrupted byte must be refused (CRC).
    let mut bad = WEIGHTS.to_vec();
    let mid = bad.len() / 2;
    bad[mid] ^= 0xFF;
    assert!(TinyMlp::from_blob(&bad).is_err());
}

/// The policy layer's resolve() is part of the shipped surface — pin the
/// ladder here too (registry above defaults, config clamped to caps).
#[test]
fn resolve_ladder_is_stable() {
    let r = policy::resolve(Some("gpt-4o-mini"), 0, 0);
    assert!(r.known && r.window > UNKNOWN_WINDOW, "registry model resolves above defaults");
    let r = policy::resolve(Some("gpt-4o-mini"), 4_096, 0);
    assert_eq!(r.window, 4_096, "smaller config honoured");
    let r = policy::resolve(Some("no-such-model-zzz"), 0, 0);
    assert_eq!(r.window, UNKNOWN_WINDOW);
}
