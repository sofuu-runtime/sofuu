// ml/supervisor/eval.rs — acceptance gates for the COMMITTED supervisor
// weights (PLAN-ML-GATES §10), mirrored from the ml-train gate run.
//
// The trainer proves the gates on its constructed dataset; these tests
// prove the COMMITTED blob still holds them on hand-built fixtures in a
// domain NO training family touches (weather-station calibration), plus
// the determinism and integrity bars (§10 bar 4/7). If a weight re-bake
// breaks these, the regression is caught at `cargo test`, not in the field.

#![cfg(test)]

use super::model;
use crate::ml::context;

const TASK: &str = "calibrate the weather station sensors before the storm season";
const P0: &str = "src/weather/calibrate.rs";
const P1: &str = "src/weather/sensors.rs";

/// Seed a run's trajectory through the SAME path the runtime uses
/// (precheck records, postcall accounts) so fixtures exercise the real
/// working-set state, not a synthetic one.
fn seed_run(run: &str, calls: &[(&str, String, &str, u32, bool)]) {
    context::run_start(run, TASK);
    for (i, (tool, sig, target, chars, err)) in calls.iter().enumerate() {
        let step = (i + 1) as u32;
        context::precheck(run, step, tool, sig, target);
        context::postcall(run, step, tool, target, *chars, *err);
    }
}

fn read_sig(path: &str) -> String {
    format!("read_file:{{\"path\":\"{path}\"}}")
}
fn grep_sig(pat: &str) -> String {
    format!("grep:{{\"pattern\":\"{pat}\"}}")
}

/// (tool, sig, target, args_text, skip_targets, budget, expect_flag)
fn precall_fixtures() -> Vec<(&'static str, String, &'static str, &'static str, Vec<String>, u32, bool)> {
    vec![
        // WASTE — exact duplicate of the call just made.
        ("read_file", read_sig(P0), P0, "", vec![], 20, true),
        // WASTE — re-read of an unchanged file (offset variant, so the
        // exact-sig rule does not shadow the re-read rule).
        ("read_file", format!("read_file:{{\"offset\":50,\"path\":\"{P1}\"}}"), P1, "", vec![], 20, true),
        // WASTE — off-task grep wander after an on-task read.
        ("grep", grep_sig("banana bread recipe contest"), "banana bread recipe contest", "", vec![], 20, true),
        // WASTE — degenerate broad search.
        ("grep", grep_sig("."), ".", "", vec![], 20, true),
        // WASTE — near-duplicate read (path respelling) of a file just read.
        ("read_file", read_sig("src//weather//calibrate.rs"), "src//weather//calibrate.rs", "", vec![], 20, true),
        // WASTE — target the pre-turn advice flagged "probably not needed".
        ("read_file", read_sig("docs/legacy_sensor_manual.md"), "docs/legacy_sensor_manual.md", "", vec!["docs/legacy_sensor_manual.md".to_string()], 20, true),
        // WASTE — first call of the run, zero echo of the task.
        ("read_file", read_sig("docs/office_snack_poll.md"), "docs/office_snack_poll.md", "", vec![], 20, true),
        // CLEAN — first call reads the task's core file.
        ("read_file", read_sig(P0), P0, "", vec![], 20, false),
        // CLEAN — follow-up grep for a symbol from the file just read.
        ("grep", grep_sig("sensor_offset"), "sensor_offset", "", vec![], 20, false),
        // CLEAN — similar-but-different: a second symbol after the first.
        ("grep", grep_sig("reading_drift"), "reading_drift", "", vec![], 20, false),
        // CLEAN — re-read AFTER a write invalidates the earlier content.
        ("read_file", read_sig(P0), P0, "", vec![], 20, false),
        // CLEAN — on-task directory listing after reading a file in it.
        ("list_dir", format!("list_dir:{{\"path\":\"src/weather\"}}"), "src/weather", "", vec![], 20, false),
    ]
}

/// The trajectory each precall fixture is scored against (index-aligned).
fn precall_trajectories() -> Vec<Vec<(&'static str, String, &'static str, u32, bool)>> {
    let read0 = ("read_file", read_sig(P0), P0, 1200u32, false);
    let read1 = ("read_file", read_sig(P1), P1, 900u32, false);
    let grep_sym = ("grep", grep_sig("sensor_offset"), "sensor_offset", 500u32, false);
    let edit0 = ("edit_file", format!("edit_file:{{\"path\":\"{P0}\"}}"), P0, 60u32, false);
    vec![
        vec![read0.clone()],                                        // dup
        vec![read1.clone(), grep_sym.clone()],                      // reread unchanged
        vec![read0.clone()],                                        // off-task grep
        vec![read0.clone()],                                        // broad
        vec![read0.clone()],                                        // near-dup
        vec![read0.clone()],                                        // skip-advised
        vec![],                                                     // zero-echo first call
        vec![],                                                     // clean first call
        vec![read0.clone()],                                        // follow-up grep
        vec![read0.clone(), grep_sym.clone()],                      // similar-but-different
        vec![read0.clone(), edit0.clone()],                         // reread after write
        vec![read0.clone()],                                        // list_dir
    ]
}

#[test]
fn committed_weights_meet_the_acceptance_gates() {
    let fx = precall_fixtures();
    let trajs = precall_trajectories();
    assert_eq!(fx.len(), trajs.len());
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    for (i, (tool, sig, target, args, skip, budget, expect_flag)) in fx.iter().enumerate() {
        let run = format!("sup-eval-gate-{i}");
        seed_run(&run, &trajs[i]);
        let step = trajs[i].len() as u32 + 1;
        let v = model::check(&run, step, tool, sig, target, args, skip, *budget, TASK);
        assert!(
            (0.0..=1.0).contains(&v.score),
            "score must be a probability, got {}",
            v.score
        );
        match (!v.ok, *expect_flag) {
            (true, true) => tp += 1,
            (true, false) => {
                fp += 1;
                eprintln!("FP fixture {i}: {tool} {target:?} reason={} score={:.3}", v.reason, v.score);
            }
            (false, true) => {
                fn_ += 1;
                eprintln!("FN fixture {i}: {tool} {target:?} score={:.3}", v.score);
            }
            (false, false) => {}
        }
    }
    let recall = tp as f32 / (tp + fn_).max(1) as f32;
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 1.0 };
    assert!(
        recall >= 0.85,
        "recall-on-waste {recall:.3} below the 0.85 floor (tp={tp} fn={fn_})"
    );
    assert!(
        precision >= 0.80,
        "precision {precision:.3} below the 0.80 floor (tp={tp} fp={fp})"
    );
}

#[test]
fn rule_layer_speaks_first() {
    let _g = crate::ml::online::TEST_LOCK.lock().unwrap();
    // The mechanical rules are certain — when they fire, the verdict must
    // carry their label and source, not the net's.
    let run = "sup-eval-rule-dup";
    seed_run(run, &[("read_file", read_sig(P0), P0, 1200, false)]);
    let v = model::check(run, 2, "read_file", &read_sig(P0), P0, "", &[], 20, TASK);
    assert!(!v.ok, "exact duplicate must be flagged");
    assert_eq!(v.reason, "dup_call");
    assert_eq!(v.source, "rule");

    let run = "sup-eval-rule-reread";
    seed_run(run, &[("read_file", read_sig(P1), P1, 900, false)]);
    let v = model::check(
        run, 2, "read_file",
        &format!("read_file:{{\"offset\":50,\"path\":\"{P1}\"}}"),
        P1, "", &[], 20, TASK,
    );
    assert!(!v.ok, "re-read of an unchanged file must be flagged");
    assert_eq!(v.reason, "reread_unchanged");
    assert_eq!(v.source, "rule");
}

#[test]
fn model_layer_catches_what_rules_miss() {
    let _g = crate::ml::online::TEST_LOCK.lock().unwrap();
    // Where the rules are silent, the net speaks: off-task wander, broad
    // search, skip-advised target, zero-echo first call.
    let cases: &[(&str, &str, &str, &str, &[&str])] = &[
        ("sup-eval-model-offtask", "grep", "banana bread recipe contest", "off_task", &[]),
        ("sup-eval-model-broad", "grep", ".", "too_broad", &[]),
        ("sup-eval-model-skip", "read_file", "docs/legacy_sensor_manual.md", "skip_advised", &["docs/legacy_sensor_manual.md"]),
        ("sup-eval-model-zeroecho", "read_file", "docs/office_snack_poll.md", "off_task", &[]),
    ];
    for (run, tool, target, want_reason, skip) in cases {
        if *run == "sup-eval-model-zeroecho" {
            context::run_start(run, TASK);
        } else {
            seed_run(run, &[("read_file", read_sig(P0), P0, 1200, false)]);
        }
        let skip_v: Vec<String> = skip.iter().map(|s| s.to_string()).collect();
        let sig = if *tool == "grep" { grep_sig(target) } else { read_sig(target) };
        let v = model::check(run, 2, tool, &sig, target, "", &skip_v, 20, TASK);
        assert!(!v.ok, "{run}: expected a flag, score={:.3}", v.score);
        assert_eq!(v.source, "model", "{run}: the net, not a rule, must speak");
        assert_eq!(v.reason, *want_reason, "{run}: wrong waste class");
    }
}

#[test]
fn loop_boundary_checkpoint() {
    let _g = crate::ml::online::TEST_LOCK.lock().unwrap();
    // Spinning: the run hammers one file with offset reads and no writes —
    // the boundary check must flag the RUN, not any single call.
    let run = "sup-eval-loop-spin";
    let spins: Vec<(&str, String, &str, u32, bool)> = (0..4)
        .map(|k| {
            (
                "read_file",
                format!("read_file:{{\"offset\":{},\"path\":\"{P1}\"}}", k * 50),
                P1,
                40,
                false,
            )
        })
        .collect();
    seed_run(run, &spins);
    let v = model::loop_check(run, 5, 20);
    assert!(!v.ok, "spinning run must be flagged at the boundary, score={:.3}", v.score);
    assert_eq!(v.source, "model");
    assert!(v.reason.starts_with("loop_"), "loop class expected, got {}", v.reason);

    // Healthy: read → grep → edit is progress; the boundary stays silent.
    let run = "sup-eval-loop-healthy";
    seed_run(
        run,
        &[
            ("read_file", read_sig(P0), P0, 1200, false),
            ("grep", grep_sig("sensor_offset"), "sensor_offset", 500, false),
            ("edit_file", format!("edit_file:{{\"path\":\"{P0}\"}}"), P0, 60, false),
        ],
    );
    let v = model::loop_check(run, 4, 20);
    assert!(v.ok, "healthy run must not be flagged, score={:.3} reason={}", v.score, v.reason);
}

#[test]
fn loop_boundary_needs_evidence() {
    let _g = crate::ml::online::TEST_LOCK.lock().unwrap();
    // Spin/stall are pattern predicates: with one or two recorded calls
    // the net saturates on the tiny trajectory (score 1.0 "stalled"), so
    // the boundary must stay silent until a pattern is even possible --
    // otherwise every run gets nagged after its first tool call.
    let run = "sup-eval-loop-one";
    seed_run(run, &[("read_file", read_sig(P0), P0, 1200, false)]);
    let v = model::loop_check(run, 2, 20);
    assert!(v.ok, "one-call run must not be flagged, score={:.3} reason={}", v.score, v.reason);

    let run = "sup-eval-loop-two";
    seed_run(
        run,
        &[
            ("read_file", read_sig(P0), P0, 1200, false),
            ("grep", grep_sig("sensor_offset"), "sensor_offset", 500, false),
        ],
    );
    let v = model::loop_check(run, 3, 20);
    assert!(v.ok, "two-call run must not be flagged, score={:.3} reason={}", v.score, v.reason);
}

#[test]
fn forward_pass_is_bit_exact() {
    let _g = crate::ml::online::TEST_LOCK.lock().unwrap();
    let run_a = "sup-eval-det-a";
    let run_b = "sup-eval-det-b";
    for run in [run_a, run_b] {
        seed_run(run, &[("read_file", read_sig(P0), P0, 1200, false)]);
    }
    let a = model::check(run_a, 2, "grep", &grep_sig("sensor_offset"), "sensor_offset", "", &[], 20, TASK);
    let b = model::check(run_b, 2, "grep", &grep_sig("sensor_offset"), "sensor_offset", "", &[], 20, TASK);
    assert_eq!(a.score.to_bits(), b.score.to_bits(), "forward must be deterministic");
}

#[test]
fn blob_integrity_and_architecture() {
    static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");
    let net = crate::ml::net::TinyMlp::from_blob(WEIGHTS).expect("committed blob loads");
    assert_eq!((net.in_dim, net.h1, net.h2), (model::IN_DIM, model::H1, model::H2));
    assert_eq!(net.w.len() as u32, model::PARAMS);
    assert!(model::THRESHOLD > 0.0 && model::THRESHOLD < 1.0);

    // A single corrupted byte must be refused (CRC).
    let mut bad = WEIGHTS.to_vec();
    let mid = bad.len() / 2;
    bad[mid] ^= 0xFF;
    assert!(crate::ml::net::TinyMlp::from_blob(&bad).is_err());
}

#[test]
fn flagged_verdicts_carry_ascii_nudges() {
    let _g = crate::ml::online::TEST_LOCK.lock().unwrap();
    // The nudge rides in-band (tool result / ephemeral notice); a flag
    // without a reason the LLM can act on is wasted tokens, and the TUI
    // renders ASCII only.
    let fx = precall_fixtures();
    let trajs = precall_trajectories();
    for (i, (tool, sig, target, args, skip, budget, expect_flag)) in fx.iter().enumerate() {
        if !*expect_flag {
            continue;
        }
        let run = format!("sup-eval-nudge-{i}");
        seed_run(&run, &trajs[i]);
        let step = trajs[i].len() as u32 + 1;
        let v = model::check(&run, step, tool, sig, target, args, skip, *budget, TASK);
        if !v.ok {
            let nudge = v.nudge.as_deref().unwrap_or("");
            assert!(!nudge.is_empty(), "flag without nudge: fixture {i}");
            assert!(nudge.is_ascii(), "nudge must be ASCII: fixture {i}");
            assert!(!v.reason.is_empty(), "flag without reason: fixture {i}");
        }
    }
}
