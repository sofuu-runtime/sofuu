// ml/freshness/eval.rs — acceptance gates for the COMMITTED weights
// (PLAN-ML-GATES §10), mirrored from the ml-train gate run.
//
// The trainer proves the gates on its constructed dataset; these tests
// prove the COMMITTED blob still holds them on hand-built fixtures that
// share NO text with the training families, plus the determinism and
// integrity bars (§10 bar 4/7). If a weight re-bake breaks these, the
// regression is caught at `cargo test`, not in the field.

#![cfg(test)]

use super::features::SourceKind;
use super::model;

const NOW_YEAR: u32 = 2026; // fixtures are deterministic: year is injected

/* ── Fixtures: strong cases, no overlap with trainer constructions ── */

/// (text, task, kind, expect_stale)
fn fixtures() -> Vec<(&'static str, &'static str, SourceKind, bool)> {
    vec![
        // Stale: explicit old date + deprecation vocabulary.
        (
            "Zephyr-db administration guide. Updated on March 4, 2019. This package \
             is deprecated and no longer maintained; the repository was archived \
             after support ended in 2020.",
            "what is the current recommended way to administer a Zephyr-db cluster?",
            SourceKind::Web,
            true,
        ),
        // Stale: hedging — the author already doubts currency.
        (
            "Quicksilver cache tuning notes. This information may be outdated; as of \
             last check the details were accurate. Verify before relying on it.",
            "how should I configure the Quicksilver cache eviction policy today?",
            SourceKind::Memory,
            true,
        ),
        // Stale: legacy version markers with an old year.
        (
            "Documents the legacy API kept for compatibility with the older release \
             line. Release v1.4 (2018-02-11): obsolete configuration format phased \
             out upstream.",
            "which API version is currently supported for new integrations?",
            SourceKind::File,
            true,
        ),
        // Stale: changelog whose newest entry is years old.
        (
            "Ironwood toolkit changelog. v2.1 (2019-06-01): breaking changes, removed \
             legacy endpoints. v2.0 (2019-01-15): initial stable API.",
            "what is the latest Ironwood toolkit release and its migration path?",
            SourceKind::Web,
            true,
        ),
        // Stale content, TIME-INSENSITIVE task — must NOT fire: the gate is
        // about the task×content interaction, not the content alone.
        (
            "Zephyr-db administration guide. Updated on March 4, 2019. This package \
             is deprecated and no longer maintained; the repository was archived \
             after support ended in 2020.",
            "summarize the writing style of this documentation page",
            SourceKind::Web,
            false,
        ),
        // Fresh: recent date + fresh vocabulary.
        (
            "Just shipped: the latest release rolled out this week and is now \
             available. Announced today; actively developed with weekly releases. \
             Updated March 4, 2026.",
            "what is the current recommended way to deploy a Rivermesh gateway?",
            SourceKind::Web,
            false,
        ),
        // Timeless: textbook content, never stale.
        (
            "A B-tree is a fundamental data structure used in systems programming. \
             This article explains the invariants, the core operations, and the \
             classic trade-offs of the B-tree. Complexity analysis included.",
            "what is the latest best practice for B-tree page splitting?",
            SourceKind::Web,
            false,
        ),
        // Future year is NOT staleness.
        (
            "Roadmap: the new storage engine will land in 2027 together with the \
             next generation interface and a redesigned query planner.",
            "when does the new storage engine ship and what does it change?",
            SourceKind::Web,
            false,
        ),
        // Neutral tool output, time-insensitive content.
        (
            "Directory listing: src, tests, crates, README.md, Makefile. 5 entries, \
             128 bytes total.",
            "list the project layout",
            SourceKind::Tool,
            false,
        ),
    ]
}

#[test]
fn committed_weights_meet_the_acceptance_gates() {
    let fx = fixtures();
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    for (text, task, kind, expect_stale) in &fx {
        let v = model::score(text, task, *kind, 0.0, 0.0, NOW_YEAR);
        assert!(
            v.score >= 0.0 && v.score <= 1.0,
            "score must be a probability, got {}",
            v.score
        );
        match (v.stale, *expect_stale) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => {}
        }
    }
    let recall = tp as f32 / (tp + fn_).max(1) as f32;
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 1.0 };
    assert!(
        recall >= 0.90,
        "recall-on-stale {recall:.3} below the 0.90 floor (tp={tp} fn={fn_})"
    );
    assert!(
        precision >= 0.70,
        "precision {precision:.3} below the 0.70 floor (tp={tp} fp={fp})"
    );
}

#[test]
fn forward_pass_is_bit_exact() {
    let (text, task, kind, _) = fixtures()[0];
    let a = model::score(text, task, kind, 0.0, 0.0, NOW_YEAR);
    let b = model::score(text, task, kind, 0.0, 0.0, NOW_YEAR);
    assert_eq!(a.score.to_bits(), b.score.to_bits(), "forward must be deterministic");
}

#[test]
fn current_year_is_sane() {
    // Wall-clock read for live scoring — the fixtures were built in 2026;
    // the year can only move forward from there.
    assert!(model::current_year() >= 2026, "civil-from-days year read");
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
fn stale_verdicts_carry_evidence() {
    // The notice rides the ephemeral context message WITH its reason (§16);
    // strong stale fixtures must not fire silently.
    let fx = fixtures();
    for (text, task, kind, expect_stale) in &fx {
        if !*expect_stale {
            continue;
        }
        let v = model::score(text, task, *kind, 0.0, 0.0, NOW_YEAR);
        if v.stale {
            assert!(
                !v.reason.is_empty(),
                "stale verdict without evidence: score={:.3} text={text:?}",
                v.score
            );
        }
    }
}
