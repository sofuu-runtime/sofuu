// tests/ml_baseline_freeze.rs — Phase 0 of PLAN-ML-IMPROVEMENT-NO-RETRAINING
// (§4.1 "Record immutable model facts").
//
// The frozen baseline lives in ../ml-baseline.json. This test proves the
// code still matches it on EVERY cargo test run: architecture constants,
// thresholds, mechanical-rule constants, feature widths, blob byte
// lengths, and FNV-1a blob hashes (the online-learning identity key).
// A failure here means a change altered the shipped models or their
// feature semantics — it must be reverted or wait for a versioned re-bake
// (the compatibility rule, §4.3).
//
// Pure Rust, no engine boot, no global-state mutation: parallel-safe by
// construction.

use sofuu_core::ml::alloc;
use sofuu_core::ml::compaction;
use sofuu_core::ml::freshness;
use sofuu_core::ml::net::TinyMlp;
use sofuu_core::ml::relevance;
use sofuu_core::ml::supervisor;

/// FNV-1a 64 over the whole blob — mirrors supervisor::model::blob_hash
/// (the online-adaptation identity key) without touching online state.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Load a committed blob, assert the exact frozen architecture, byte
/// length, and FNV-1a identity, and return the net.
fn pinned(name: &str, blob: &'static [u8], arch: (u32, u32, u32, u32), bytes: usize, fnv: &str) -> TinyMlp {
    let net = TinyMlp::from_blob(blob)
        .unwrap_or_else(|e| panic!("{name} committed blob must load: {e}"));
    assert_eq!(
        (net.in_dim, net.h1, net.h2, net.w.len() as u32),
        arch,
        "{name}: architecture changed — the weights are only valid with the schema they were trained against"
    );
    assert_eq!(blob.len(), bytes, "{name}: blob byte length drifted");
    assert_eq!(
        format!("{:016x}", fnv1a64(blob)),
        fnv,
        "{name}: blob identity changed — a silent re-bake or byte swap; online adaptations keyed to the old hash are invalid"
    );
    net
}

#[test]
fn baseline_freeze_matches_the_shipped_models() {
    // freshness: 28 → 112 → 48 → 1, 8721 params, threshold 0.95.
    pinned(
        "freshness",
        include_bytes!("../src/ml/freshness/weights_v1.f32"),
        (freshness::model::IN_DIM, freshness::model::H1, freshness::model::H2, freshness::model::PARAMS),
        34_912,
        "bf6e050ce597e0b8",
    );
    assert_eq!(
        (
            freshness::model::IN_DIM,
            freshness::model::H1,
            freshness::model::H2,
            freshness::model::PARAMS,
            freshness::model::THRESHOLD
        ),
        (28, 112, 48, 8721, 0.95)
    );
    assert_eq!(freshness::features::FRESHNESS_FEATURES, 28);

    // relevance: 37 → 104 → 44 → 1, 8617 params, threshold 0.95.
    pinned(
        "relevance",
        include_bytes!("../src/ml/relevance/weights_v1.f32"),
        (relevance::model::IN_DIM, relevance::model::H1, relevance::model::H2, relevance::model::PARAMS),
        34_496,
        "e99fe129f96a09fc",
    );
    assert_eq!(
        (
            relevance::model::IN_DIM,
            relevance::model::H1,
            relevance::model::H2,
            relevance::model::PARAMS,
            relevance::model::THRESHOLD
        ),
        (37, 104, 44, 8617, 0.95)
    );
    assert_eq!(relevance::features::RELEVANCE_FEATURES, 37);
    assert_eq!(relevance::model::MECHANICAL_DUP_COSINE, 0.90);
    assert_eq!(relevance::model::MECHANICAL_NEVER_USE, 0.50);

    // supervisor: 33 → 104 → 44 → 1, 8201 params, threshold 0.82.
    pinned(
        "supervisor",
        include_bytes!("../src/ml/supervisor/weights_v1.f32"),
        (supervisor::model::IN_DIM, supervisor::model::H1, supervisor::model::H2, supervisor::model::PARAMS),
        32_832,
        "64ab5b22694555ab",
    );
    assert_eq!(
        (
            supervisor::model::IN_DIM,
            supervisor::model::H1,
            supervisor::model::H2,
            supervisor::model::PARAMS,
            supervisor::model::THRESHOLD
        ),
        (33, 104, 44, 8201, 0.82)
    );
    assert_eq!(supervisor::features::SUPERVISOR_FEATURES, 33);
    // The online layer keys adaptations to this exact hash (§13); a
    // re-bake must invalidate them. (fnv1a64 here mirrors the private
    // supervisor::model::blob_hash — the online tests pin that copy.)
    assert_eq!(
        format!("{:016x}", fnv1a64(include_bytes!("../src/ml/supervisor/weights_v1.f32"))),
        "64ab5b22694555ab"
    );

    // compaction: 33 → 104 → 44 → 1, 8201 params, threshold 0.41.
    pinned(
        "compaction",
        include_bytes!("../src/ml/compaction/weights_v1.f32"),
        (compaction::model::IN_DIM, compaction::model::H1, compaction::model::H2, compaction::model::PARAMS),
        32_832,
        "c346aad8db948a11",
    );
    assert_eq!(
        (
            compaction::model::IN_DIM,
            compaction::model::H1,
            compaction::model::H2,
            compaction::model::PARAMS,
            compaction::model::THRESHOLD
        ),
        (33, 104, 44, 8201, 0.41)
    );
    assert_eq!(compaction::features::COMPACTION_FEATURES, 33);
    assert_eq!(compaction::model::KEEP_WINDOW_AGE, 1);
    assert_eq!(compaction::model::MECHANICAL_DUP_COSINE, 0.95);

    // alloc: 24 → 96 → 40 → 1, 6321 params, threshold 0.51.
    pinned(
        "alloc",
        include_bytes!("../src/ml/alloc/weights_v1.f32"),
        (alloc::model::IN_DIM, alloc::model::H1, alloc::model::H2, alloc::model::PARAMS),
        25_312,
        "71c6ddc29a182378",
    );
    assert_eq!(
        (
            alloc::model::IN_DIM,
            alloc::model::H1,
            alloc::model::H2,
            alloc::model::PARAMS,
            alloc::model::THRESHOLD
        ),
        (24, 96, 40, 6321, 0.51)
    );
    assert_eq!(alloc::features::ALLOC_FEATURES, 24);

    // The alloc policy layer's frozen mechanical constants (§4.1 "threshold
    // and mechanical-rule constants").
    use alloc::policy;
    assert_eq!(policy::UNKNOWN_WINDOW, 32_768);
    assert_eq!(policy::UNKNOWN_MAX_OUTPUT, 4_096);
    assert_eq!(policy::MIN_WINDOW, 2_048);
    assert_eq!(policy::MIN_MAX_OUTPUT, 512);

    // Every committed blob refuses trailing data (Phase 1.4): a corrupted
    // bake is a build-time mistake, never a silent wrong answer.
    let fresh = include_bytes!("../src/ml/freshness/weights_v1.f32");
    let mut trailing = fresh.to_vec();
    trailing.extend_from_slice(&[0u8; 4]);
    assert_eq!(TinyMlp::from_blob(&trailing).unwrap_err(), "truncated weights");
}

/// The frozen slot-name tables in ml-baseline.json are the human-readable
/// record; the load-bearing machine check is that each extractor still
/// returns the frozen WIDTH (drift = schema break). A JSON-side rename is
/// documentation-only.
#[test]
fn feature_widths_match_the_frozen_schema() {
    assert_eq!(freshness::features::FRESHNESS_FEATURES, 28);
    assert_eq!(relevance::features::RELEVANCE_FEATURES, 37);
    assert_eq!(supervisor::features::SUPERVISOR_FEATURES, 33);
    assert_eq!(compaction::features::COMPACTION_FEATURES, 33);
    assert_eq!(alloc::features::ALLOC_FEATURES, 24);

    // Round-trip every committed blob through from_blob: CRC + dims +
    // params + finiteness + trailing-data all verified at once.
    for blob in [
        include_bytes!("../src/ml/freshness/weights_v1.f32").as_slice(),
        include_bytes!("../src/ml/relevance/weights_v1.f32").as_slice(),
        include_bytes!("../src/ml/supervisor/weights_v1.f32").as_slice(),
        include_bytes!("../src/ml/compaction/weights_v1.f32").as_slice(),
        include_bytes!("../src/ml/alloc/weights_v1.f32").as_slice(),
    ] {
        assert!(TinyMlp::from_blob(blob).is_ok(), "committed blob failed integrity");
    }
}
