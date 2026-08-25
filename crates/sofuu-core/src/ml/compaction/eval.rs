// ml/compaction/eval.rs — acceptance gates for the COMMITTED weights
// (PLAN-ML-GATES §10), mirrored from the ml-train gate run.
//
// The trainer proves the gates on its constructed dataset; these tests
// prove the COMMITTED blob still holds them on a hand-built conversation
// that shares NO text with the training families, plus the determinism
// and integrity bars (§10 bar 4/7). Keep-safety is the asymmetric risk
// (§12): a missed junk segment costs a later pass, a compacted
// load-bearing segment destroys information — so the asserts are strict
// on the keep side.

#![cfg(test)]

use super::features::{SegKind, SegmentInput};
use super::model::{self, Tier};

const TASK: &str = "finish the payment webhook integration";

fn seg(text: &'static str, tokens: u32, age: u32, kind: SegKind, retr: bool) -> SegmentInput<'static> {
    SegmentInput { text, tokens, age_steps: age, kind, retrievable: retr, already_compacted: false }
}

/// A realistic mid-session history. Indices:
///  0 old verbose build log            → disposable (summarize tier)
///  1 old decision on the queue lib    → KEEP (decision language)
///  2 old file read, re-fetchable      → disposable (retrievable tier)
///  3 old user instruction             → KEEP
///  4 duplicate of segment 2           → disposable (dup tier)
///  5 cookie-banner boilerplate        → disposable (boilerplate tier)
///  6 old tool error, still relevant   → KEEP (error the agent must react to)
///  7 newest turn                      → KEEP (keep-window)
fn fixture() -> (Vec<SegmentInput<'static>>, &'static str) {
    let segs = vec![
        seg(
            "running the webhook build: compiling crate one, compiling crate two, \
             compiling crate three, linking, emitting artifacts, recording checksums \
             for every produced file, build finished with zero errors and nine warnings",
            320, 9, SegKind::ToolResult, false,
        ),
        seg(
            "we decided to route webhook retries through the durable queue library; \
             that is settled and every caller follows it from now on",
            90, 8, SegKind::Assistant, false,
        ),
        seg(
            "contents of src/payments/webhook.rs: the handler verifies the signature, \
             parses the payload, and enqueues the job for the worker pool",
            240, 7, SegKind::ToolResult, true,
        ),
        seg(
            "keep the webhook public endpoint unchanged while you refactor the handler",
            40, 8, SegKind::User, false,
        ),
        seg(
            "contents of src/payments/webhook.rs: the handler verifies the signature, \
             parses the payload, and enqueues the job for the worker pool",
            240, 3, SegKind::ToolResult, true,
        ),
        seg(
            "cookies help us deliver our services. by using this site you agree to our \
             use of cookies. privacy policy and terms of service apply. navigation: \
             home, about, contact, sitemap. all rights reserved.",
            180, 6, SegKind::ToolResult, false,
        ),
        seg(
            "error: the webhook delivery test failed with a timeout\nthe upstream \
             returned no response and the retry budget was exhausted\nstack trace \
             follows in the log",
            120, 5, SegKind::ToolResult, false,
        ),
        seg("what is left to finish on the webhook integration?", 20, 0, SegKind::User, false),
    ];
    let recent = "what is left to finish on the webhook integration?";
    (segs, recent)
}

const DISPOSABLE: &[usize] = &[0, 2, 4, 5];
const LOAD_BEARING: &[usize] = &[1, 3, 6, 7];

#[test]
fn committed_weights_meet_the_acceptance_gates() {
    let (segs, recent) = fixture();
    // Budget large enough to take every flagged segment — this test
    // measures SELECTION, not the budget cap (covered separately).
    let p = model::plan(TASK, "", recent, &segs, 100_000);

    let flagged: Vec<usize> = p.compact.iter().map(|c| c.id).collect();
    for &i in LOAD_BEARING {
        assert!(
            !flagged.contains(&i),
            "load-bearing segment {i} was flagged for compaction: score={:.3} ({:?})",
            p.scores[i],
            segs[i].text.chars().take(48).collect::<String>()
        );
        assert!(p.keep.contains(&i), "load-bearing segment {i} must be in keep");
    }
    for &i in DISPOSABLE {
        assert!(
            flagged.contains(&i),
            "disposable segment {i} was not flagged: score={:.3} ({:?})",
            p.scores[i],
            segs[i].text.chars().take(48).collect::<String>()
        );
    }
    // Keep-safety precision on the fixture: nothing flagged may be
    // load-bearing (already asserted) — and at least the junk was found.
    assert!(p.freeable > 0, "the pass must free tokens");
}

#[test]
fn free_tiers_are_classified() {
    let (segs, recent) = fixture();
    let p = model::plan(TASK, "", recent, &segs, 100_000);
    let tier_of = |id: usize| p.compact.iter().find(|c| c.id == id).map(|c| c.tier);
    assert_eq!(tier_of(4), Some(Tier::Dup), "verbatim repeat is the dup tier");
    assert_eq!(tier_of(5), Some(Tier::Boilerplate), "site chrome is the boilerplate tier");
    assert_eq!(tier_of(2), Some(Tier::Retrievable), "re-fetchable read is the retrievable tier");
    // Free tiers come first in the list (§12: prefer the free tier).
    let first_summarize = p.compact.iter().position(|c| c.tier == Tier::Summarize);
    let last_free = p.compact.iter().rposition(|c| c.tier != Tier::Summarize);
    match (first_summarize, last_free) {
        (Some(f), Some(l)) => assert!(f > l, "free tiers must precede summarize tiers"),
        _ => {}
    }
}

#[test]
fn keep_window_is_hard_protected() {
    // Junk text at age 0/1 must NEVER be flagged, whatever the score.
    let segs = vec![
        seg("ok", 4, 1, SegKind::User, false),
        seg(
            "cookies help us deliver our services. by using this site you agree to our \
             use of cookies. privacy policy and terms of service apply. all rights reserved.",
            160, 0, SegKind::ToolResult, false,
        ),
    ];
    let p = model::plan(TASK, "", "", &segs, 100_000);
    assert!(p.compact.is_empty(), "keep-window segments must never be compacted");
    assert_eq!(p.keep, vec![0, 1]);
}

#[test]
fn budget_caps_the_pass() {
    let (segs, recent) = fixture();
    // Budget of 300 tokens: the pass must stop once it is spent.
    let p = model::plan(TASK, "", recent, &segs, 300);
    assert!(p.freeable <= 300 + p.compact.first().map(|c| c.tokens).unwrap_or(0));
    assert!(p.freeable <= 600, "a 300-token budget must not free the whole window");
    let unlimited = model::plan(TASK, "", recent, &segs, 100_000);
    assert!(p.compact.len() <= unlimited.compact.len());
}

#[test]
fn plan_is_bit_exact() {
    let (segs, recent) = fixture();
    let a = model::plan(TASK, "", recent, &segs, 100_000);
    let b = model::plan(TASK, "", recent, &segs, 100_000);
    assert_eq!(a.scores.len(), b.scores.len());
    for (x, y) in a.scores.iter().zip(b.scores.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "forward must be deterministic");
    }
    assert_eq!(a.freeable, b.freeable);
    assert_eq!(a.compact.len(), b.compact.len());
}

#[test]
fn blob_integrity_and_architecture() {
    static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");
    let net = crate::ml::net::TinyMlp::from_blob(WEIGHTS).expect("committed blob loads");
    assert_eq!((net.in_dim, net.h1, net.h2), (model::IN_DIM, model::H1, model::H2));
    assert_eq!(net.w.len() as u32, model::PARAMS);
    // The shipped threshold must sit strictly inside the score range AND
    // well above the degenerate floor (a 0.05 gate flags everything).
    assert!(model::THRESHOLD >= 0.30 && model::THRESHOLD <= 0.90);
    assert!(model::KEEP_WINDOW_AGE >= 1);

    // A single corrupted byte must be refused (CRC).
    let mut bad = WEIGHTS.to_vec();
    let mid = bad.len() / 2;
    bad[mid] ^= 0xFF;
    assert!(crate::ml::net::TinyMlp::from_blob(&bad).is_err());
}

#[test]
fn empty_history_is_an_empty_plan() {
    let p = model::plan(TASK, "", "", &[], 100_000);
    assert!(p.compact.is_empty());
    assert!(p.keep.is_empty());
    assert_eq!(p.freeable, 0);
}

#[test]
fn mechanical_dup_survives_recent_similarity() {
    // The repeated content ALSO sits in the recent window, so the
    // reference channel fires on the old copy and the net calls it keep.
    // The deterministic dedupe rule (§12 free tier) must still flag it:
    // the content exists elsewhere, dropping the repeat loses nothing.
    let repeated = "deployment checklist for the gateway rollout: verify dns, \
                    verify tls certificates, verify upstream health";
    let segs = vec![
        seg(repeated, 60, 6, SegKind::Assistant, false),
        seg("an unrelated middle note about scheduling the next review", 30, 4, SegKind::Assistant, false),
        seg(repeated, 60, 2, SegKind::Assistant, false),
    ];
    let p = model::plan("roll out the gateway", "", repeated, &segs, 100_000);
    let dup = p.compact.iter().find(|c| c.id == 2);
    assert!(
        dup.is_some(),
        "near-verbatim repeat must be flagged despite recent-window similarity (score={:.3})",
        p.scores[2]
    );
    assert_eq!(dup.unwrap().tier, Tier::Dup);
    assert!(p.keep.contains(&0), "the original stays");
}

#[test]
fn short_identical_assistant_replies_are_dups() {
    // Mirrors the chat E2E: identical short assistant replies repeat,
    // interleaved with LONG padded user prompts (2000 x's). Each repeat
    // (age > keep-window) must register as a dup of the earlier copy.
    let pad = "x".repeat(2000);
    let q1 = format!("question number 1 with padding {pad}");
    let q2 = format!("question number 2 with padding {pad}");
    let q3 = format!("question number 3 with padding {pad}");
    let q4 = format!("question number 4 with padding {pad}");
    let q1l: &'static str = Box::leak(q1.into_boxed_str());
    let q2l: &'static str = Box::leak(q2.into_boxed_str());
    let q3l: &'static str = Box::leak(q3.into_boxed_str());
    let q4l: &'static str = Box::leak(q4.into_boxed_str());
    let reply: &'static str = "PLAIN-OK";
    let segs = vec![
        seg(q1l, 508, 7, SegKind::User, false),
        seg(reply, 2, 6, SegKind::Assistant, false),
        seg(q2l, 508, 5, SegKind::User, false),
        seg(reply, 2, 4, SegKind::Assistant, false),
        seg(q3l, 508, 3, SegKind::User, false),
        seg(reply, 2, 2, SegKind::Assistant, false),
        seg(q4l, 508, 1, SegKind::User, false),
    ];
    let feats = super::features::extract_all(&super::features::CompactionContext {
        task: q4l,
        summary: "",
        recent: q4l,
        segments: &segs,
    });
    eprintln!("f16 reply@3 = {}, f16 reply@5 = {}", feats[3][16], feats[5][16]);
    eprintln!("f16 q2@2 = {}, f16 q3@4 = {}", feats[2][16], feats[4][16]);
    assert!(feats[3][16] >= 0.95, "identical reply f16 must be ~1.0, got {}", feats[3][16]);
    assert!(feats[5][16] >= 0.95, "identical reply f16 must be ~1.0, got {}", feats[5][16]);
}
