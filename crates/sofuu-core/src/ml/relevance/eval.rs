// ml/relevance/eval.rs — acceptance gates for the COMMITTED weights
// (PLAN-ML-GATES §10), mirrored from the ml-train gate run.
//
// The trainer proves the gates on its constructed dataset; these tests
// prove the COMMITTED blob still holds them on hand-built candidate sets
// that share NO text with the training families (payments / invoicing /
// image resizing instead of the six training domains), plus the
// determinism and integrity bars (§10 bar 4/7). Recall is the asymmetric
// risk (§6): a false drop is invisible and costs correctness — so the
// asserts are strict on the use side.

#![cfg(test)]

use super::features::{CandKind, CandidateInput};
use super::model;

fn cand(text: &'static str, kind: CandKind) -> CandidateInput<'static> {
    CandidateInput { text, kind, strength: 0.0, role: 0, path: "" }
}

fn cand_file(text: &'static str, path: &'static str) -> CandidateInput<'static> {
    CandidateInput { text, kind: CandKind::File, strength: 0.0, role: 0, path }
}

const TASK: &str = "fix the verify_signature function in the payment webhook handler";

/// A realistic pre-retrieval menu. Indices:
///  0 definition site of the task-named symbol → USE
///  1 off-topic meeting note                   → skip
///  2 license boilerplate (never-use class)    → skip (mechanical)
///  3 unrelated call-site mention of the symbol→ skip
///  4 near-duplicate of candidate 0            → skip (dup of kept)
///  5 one-word wrong-topic trap                → skip
fn fixture() -> (Vec<CandidateInput<'static>>, Vec<usize>) {
    let cands = vec![
        cand_file(
            "fn verify_signature checks the payment webhook payload against the \
             shared secret and rejects the delivery when the digest does not match",
            "src/payments/webhook.rs",
        ),
        cand("meeting notes from the design review: the palette choices stay as \
              discussed, follow up next week", CandKind::Other),
        cand("permission is hereby granted, free of charge, to any person obtaining \
              a copy of this software. the software is provided as is, without \
              warranty of any kind. all rights reserved.", CandKind::File),
        cand_file(
            "the billing exporter calls verify_signature once at startup and \
             otherwise only formats the csv rows",
            "src/billing/export.rs",
        ),
        cand_file(
            "fn verify_signature checks the payment webhook payload against the \
             shared secret and rejects the delivery when the digest does not match",
            "src/payments/webhook.rs",
        ),
        cand("facilities notice: the payment for the new office plants is due; the \
              delivery of the ferns is scheduled for monday", CandKind::Other),
    ];
    (cands, vec![0]) // candidate 0 already accepted into context
}

#[test]
fn committed_weights_meet_the_acceptance_gates() {
    let (cands, kept) = fixture();
    let p = model::plan(TASK, "", &cands, &kept);

    assert!(
        p.use_ids.contains(&0),
        "definition site must be advised use: score={:.3}",
        p.scores[0]
    );
    for &i in &[1usize, 2, 3, 4, 5] {
        assert!(
            p.skip_ids.contains(&i),
            "candidate {i} must be advised skip: score={:.3} ({:?})",
            p.scores[i],
            cands[i].text.chars().take(48).collect::<String>()
        );
    }
    // Use advice is ordered best-first.
    assert_eq!(p.use_ids.first(), Some(&0));
}

#[test]
fn morphological_echo_is_caught() {
    // The flagship gap (§6): the answer echoes the task words in DIFFERENT
    // word forms (invoices↔invoice, reconciles↔reconciliation) — exact-word
    // overlap is partial, a flat overlap threshold misses it. Domain is
    // invoicing, absent from every training family.
    let cands = vec![
        cand("the nightly job invoices each account, reconciles the ledger \
              entries against the bank feed, and flags the mismatches for \
              review before the books close", CandKind::Web),
        cand("a recipe for sourdough bread with a long cold fermentation", CandKind::Web),
    ];
    let p = model::plan("finish the invoice reconciliation work", "", &cands, &[]);
    assert!(
        p.use_ids.contains(&0),
        "morphological echo must be advised use: score={:.3}",
        p.scores[0]
    );
    assert!(p.skip_ids.contains(&1), "unrelated candidate must be skip");
}

#[test]
fn near_duplicate_of_kept_is_mechanically_skipped() {
    // Even if the net scores a verbatim repeat of an already-kept candidate
    // highly, the deterministic dedupe rule must keep it out: the content
    // is already in context, pulling it again gains nothing.
    let repeated = "the deploy checklist for the gateway: verify dns, verify \
                    the tls certificates, and confirm the upstream health";
    let cands = vec![
        cand(repeated, CandKind::File),
        cand(repeated, CandKind::File),
        cand("an unrelated note about scheduling the next review", CandKind::Other),
    ];
    let p = model::plan("roll out the gateway", "", &cands, &[0]);
    assert!(p.use_ids.contains(&0), "the original stays advised use");
    assert!(
        p.skip_ids.contains(&1),
        "verbatim repeat of a kept candidate must be skipped (score={:.3})",
        p.scores[1]
    );
}

#[test]
fn plan_is_bit_exact() {
    let (cands, kept) = fixture();
    let a = model::plan(TASK, "", &cands, &kept);
    let b = model::plan(TASK, "", &cands, &kept);
    assert_eq!(a.scores.len(), b.scores.len());
    for (x, y) in a.scores.iter().zip(b.scores.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "forward must be deterministic");
    }
    assert_eq!(a.use_ids, b.use_ids);
    assert_eq!(a.skip_ids, b.skip_ids);
}

#[test]
fn blob_integrity_and_architecture() {
    static WEIGHTS: &[u8] = include_bytes!("weights_v1.f32");
    let net = crate::ml::net::TinyMlp::from_blob(WEIGHTS).expect("committed blob loads");
    assert_eq!((net.in_dim, net.h1, net.h2), (model::IN_DIM, model::H1, model::H2));
    assert_eq!(net.w.len() as u32, model::PARAMS);
    // The shipped threshold sits strictly inside the score range, on the
    // conservative side (recall-first: advice to spend tokens needs a high
    // bar) but not degenerate (a 1.0 gate advises nothing, ever).
    assert!(model::THRESHOLD >= 0.50 && model::THRESHOLD < 1.0);

    // A single corrupted byte must be refused (CRC).
    let mut bad = WEIGHTS.to_vec();
    let mid = bad.len() / 2;
    bad[mid] ^= 0xFF;
    assert!(crate::ml::net::TinyMlp::from_blob(&bad).is_err());
}

#[test]
fn empty_menu_is_an_empty_plan() {
    let p = model::plan(TASK, "", &[], &[]);
    assert!(p.use_ids.is_empty());
    assert!(p.skip_ids.is_empty());
    assert!(p.scores.is_empty());
}
