// crates/ml-train/src/gate_eval.rs — "do the gate models have a solid base?"
//
// Every other grader in this crate answers "did training converge on the
// data it was trained on". None of them answers the question that
// actually matters for shipping an 8 KB MLP that steers freshness,
// compaction, relevance, supervision and token budgets:
//
//     is it better than the trivial thing?
//
// A 1.000 accuracy on a synthetic set is not evidence of anything until
// you know what a constant predictor scores on the same fold. If the
// majority class is also 1.000, the network is decoration. This command
// grades each gate's COMMITTED weights against two references on the
// same held-out families:
//
//   * majority-class — predict the training prior, always. The floor any
//     model claiming to be useful must clear.
//   * logistic regression — trained on the same features and split
//     (PLAN-ML-GATES §10's "beat logistic regression" bar). A nonlinear
//     net that cannot beat a linear model on the same inputs is not
//     earning its bytes.
//
// Thresholds are chosen on the VALIDATION fold only (never the test
// fold), so the comparison cannot be tuned by peeking. The verdict line
// is deliberately blunt: PASS means the net beat both references, HOLD
// means it is statistically indistinguishable from at least one.
//
// The caveat this command cannot remove: all five datasets are
// SYNTHETIC with mechanical labels, so a PASS here means "learnable on
// the distribution we generate", not "correct on real user data". That
// limitation is printed at the end of every run, on purpose.

use crate::train::{
    Example, Metrics, Split, evaluate, predict, threshold_for_precision, threshold_for_recall,
    train_logreg,
};
use sofuu_core::ml::net::TinyMlp;

struct Gate {
    name: &'static str,
    /// Committed artifact, relative to sofuu-core/src/ml.
    weights: &'static str,
    data: fn() -> Vec<Example>,
    /// (train, val, test) family groups — whole families are held out,
    /// so a score reflects generalization to unseen constructions rather
    /// than to unseen rows of the same construction.
    groups: (&'static [u32], &'static [u32], &'static [u32]),
    /// The gate's own operating rule: a recall floor (miss a stale doc =
    /// bad) or a precision floor (false alarms = bad).
    select: Select,
}

#[derive(Clone, Copy, PartialEq)]
enum Select {
    Recall(f32),
    Precision(f32),
}

/// Where the committed artifacts live.
fn core_ml_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sofuu-core/src/ml")
}

fn gates() -> Vec<Gate> {
    vec![
        Gate {
            name: "freshness",
            weights: "freshness/weights_v1.f32",
            data: crate::data_freshness::build,
            groups: (&[0u32, 1u32, 2u32, 3u32, 4u32, 7u32, 10u32, 11u32, 12u32], &[5u32, 6u32], &[8u32, 9u32]),
            select: Select::Recall(0.95),
        },
        Gate {
            name: "compaction",
            weights: "compaction/weights_v1.f32",
            data: crate::data_compaction::build,
            // Same split the trainer uses: the XOR conflict (25/26) is in
            // TRAIN (a pattern never trained on is unlearnable by
            // definition), the held-out conflict families (17–24) are the
            // test. Generalization to unseen disagreements, measured
            // against a reference that provably cannot represent XOR.
            groups: (
                &[0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 8u32, 9u32, 10u32, 14u32, 15u32, 25u32, 26u32],
                &[7u32, 12u32, 16u32],
                &[17u32, 18u32, 19u32, 20u32, 21u32, 22u32, 23u32, 24u32],
            ),
            select: Select::Precision(0.95),
        },
        Gate {
            name: "relevance",
            weights: "relevance/weights_v1.f32",
            data: crate::data_relevance::build,
            groups: (
                &[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29,
                ],
                &[13u32, 14u32],
                &[15u32, 16u32],
            ),
            select: Select::Recall(0.90),
        },
        Gate {
            name: "supervisor",
            weights: "supervisor/weights_v1.f32",
            data: crate::data_supervisor::build,
            groups: (
                &[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22,
                    23,
                ],
                &[24u32, 26u32],
                &[16u32, 25u32],
            ),
            select: Select::Recall(0.90),
        },
        Gate {
            name: "alloc",
            weights: "alloc/weights_v1.f32",
            data: crate::data_alloc::build,
            groups: (
                &[0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 12u32, 13u32, 14u32, 15u32, 18u32, 19u32, 21u32, 22u32],
                &[11u32, 16u32, 20u32],
                &[17u32, 23u32],
            ),
            select: Select::Recall(0.92),
        },
    ]
}

fn pick_threshold(
    select: Select,
    preds: &[f32],
    labels: &[f32],
) -> f32 {
    match select {
        Select::Recall(r) => threshold_for_recall(preds, labels, r),
        Select::Precision(p) => threshold_for_precision(preds, labels, p),
    }
}

/// The trivial predictor: always answer the training prior. Its recall
/// and precision are both the positive rate, so any model at or below it
/// has learned nothing a constant would not.
fn majority_baseline(data: &[Example], split: &Split) -> (Vec<f32>, f32) {
    let pos = split.train.iter().filter(|&&i| data[i].y > 0.5).count() as f32;
    let n = split.train.len().max(1) as f32;
    let prior = pos / n;
    let preds: Vec<f32> = split
        .test
        .iter()
        .map(|_| if prior >= 0.5 { 1.0 } else { 0.0 })
        .collect();
    (preds, prior)
}

/// Z-score the features using TRAIN statistics only, so the linear
/// reference gets a fair shot. Without this the baseline is not a
/// baseline but a strawman: these features span very different scales
/// (lexical counts vs flags), and `train_logreg`'s lr=0.05 saturates the
/// sigmoid to exactly 0.0 on raw inputs, which made the linear model
/// score recall 0.000 and handed the MLP a free win. A comparison that
/// flatters the thing being audited is worse than no comparison.
fn standardize(data: &[Example], split: &Split) -> Vec<Example> {
    let n = split.train.len().max(1) as f32;
    let d = data.first().map(|e| e.x.len()).unwrap_or(0);
    let mut mean = vec![0.0f32; d];
    let mut var = vec![0.0f32; d];
    for &i in &split.train {
        for (j, &x) in data[i].x.iter().enumerate().take(d) {
            mean[j] += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }
    for &i in &split.train {
        for (j, &x) in data[i].x.iter().enumerate().take(d) {
            var[j] += (x - mean[j]) * (x - mean[j]);
        }
    }
    let std: Vec<f32> = var
        .iter()
        .map(|v| (v / n).sqrt().max(1e-3))
        .collect();
    data.iter()
        .map(|e| Example {
            x: e.x
                .iter()
                .enumerate()
                .take(d)
                .map(|(j, &x)| (x - mean[j]) / std[j])
                .collect(),
            ..e.clone()
        })
        .collect()
}

pub fn run() {
    println!("Gate model audit — committed weights vs trivial and linear baselines");
    println!("thresholds chosen on the validation fold; test families held out whole\n");

    let mut all_pass = true;
    let mut rows: Vec<(String, f64, f64, f64, f64, &'static str)> = Vec::new();

    for gate in gates() {
        let data = (gate.data)();
        let (tr, va, te) = gate.groups;
        let split = crate::train::split_by_groups_explicit(&data, tr, va, te);

        let blob = core_ml_dir().join(gate.weights);
        let Ok(bytes) = std::fs::read(&blob) else {
            println!("  {:<12} MISSING committed weights: {}", gate.name, blob.display());
            all_pass = false;
            continue;
        };
        let Ok(net) = TinyMlp::from_blob(&bytes) else {
            println!("  {:<12} committed weights failed to load (corrupt?)", gate.name);
            all_pass = false;
            continue;
        };

        // Test-fold predictions for the network.
        let tp: Vec<f32> = split.test.iter().map(|&i| predict(&net, &data[i].x)).collect();
        let ty: Vec<f32> = split.test.iter().map(|&i| data[i].y).collect();
        // Validation predictions pick the threshold — never the test fold.
        let vp: Vec<f32> = split.val.iter().map(|&i| predict(&net, &data[i].x)).collect();
        let vy: Vec<f32> = split.val.iter().map(|&i| data[i].y).collect();
        let threshold = pick_threshold(gate.select, &vp, &vy);
        let m_mlp = evaluate(&tp, &ty, threshold);

        // Reference 1: majority class.
        let (mp, prior) = majority_baseline(&data, &split);
        let m_maj = evaluate(&mp, &ty, 0.5);

        // Reference 2: logistic regression, same features, same split,
        // on standardized inputs (see `standardize` for why).
        let sdata = standardize(&data, &split);
        let lr = train_logreg(data[split.train[0]].x.len(), &sdata, &split, 0x5EED_C0DE);
        let lp: Vec<f32> = split.test.iter().map(|&i| lr.predict(&sdata[i].x)).collect();
        let lvp: Vec<f32> = split.val.iter().map(|&i| lr.predict(&sdata[i].x)).collect();
        let lr_threshold = pick_threshold(gate.select, &lvp, &vy);
        let m_lr = evaluate(&lp, &ty, lr_threshold);

        // The comparison is on the metric each gate is selected for,
        // because a gate that trades recall for precision on purpose
        // would look "worse" on the other one. F1 is reported alongside
        // because the selection metric alone can hide the real story: a
        // gate selected for recall that ties "always intervene" may still
        // dominate on F1 by refusing the false positives.
        let (mlp_score, maj_score, lr_score) = match gate.select {
            Select::Recall(_) => (m_mlp.recall, m_maj.recall, m_lr.recall),
            Select::Precision(_) => (m_mlp.precision, m_maj.precision, m_lr.precision),
        };
        let beats_maj = mlp_score > maj_score + 0.005;
        let beats_lr = mlp_score > lr_score + 0.005;
        let beats_maj_f1 = m_mlp.f1 > m_maj.f1 + 0.005;
        let beats_lr_f1 = m_mlp.f1 > m_lr.f1 + 0.005;
        let verdict = match (beats_maj || beats_maj_f1, beats_lr || beats_lr_f1) {
            (true, true) if beats_maj && beats_lr => "PASS",
            (true, true) => "PASS* (wins on F1, ties the selection metric)",
            (true, false) => "HOLD (does not beat the linear reference)",
            (false, true) => "HOLD (does not beat the constant)",
            (false, false) => "HOLD (ties both references)",
        };
        if !((beats_maj || beats_maj_f1) && (beats_lr || beats_lr_f1)) {
            all_pass = false;
        }

        let metric_name = match gate.select {
            Select::Recall(_) => "recall",
            Select::Precision(_) => "prec",
        };
        println!("  {} ({}, {} params, threshold {threshold:.2})", gate.name, net_dims(&net), net.w.len());
        println!(
            "    tiny MLP      acc={:.3} prec={:.3} rec={:.3} f1={:.3}   (n={}, prior {:.3})",
            m_mlp.accuracy, m_mlp.precision, m_mlp.recall, m_mlp.f1 as f64, m_mlp.n, prior
        );
        println!(
            "    logistic      acc={:.3} prec={:.3} rec={:.3} f1={:.3}",
            m_lr.accuracy, m_lr.precision, m_lr.recall, m_lr.f1
        );
        println!(
            "    majority      acc={:.3} prec={:.3} rec={:.3} f1={:.3}",
            m_maj.accuracy, m_maj.precision, m_maj.recall, m_maj.f1
        );
        println!(
            "    {metric_name}: mlp {mlp_score:.3} vs logreg {lr_score:.3} vs constant {maj_score:.3}  →  {verdict}\n"
        );

        /* Structural warnings last, so they sit next to the numbers they
         * qualify: a single-class fold or a silent baseline means a tie
         * is unmeasured, NOT won. */
        let test_pos_rate = if m_mlp.n > 0 {
            m_mlp.positives as f64 / m_mlp.n as f64
        } else {
            0.0
        };
        if test_pos_rate <= 0.001 || test_pos_rate >= 0.999 {
            println!(
                "    ⚠ the held-out fold is single-class ({:.1}% positive) — it CANNOT discriminate\n      any model from any other. This gate is UNVALIDATED, not proven.\n",
                test_pos_rate * 100.0
            );
        }
        if m_lr.recall <= 0.001 && m_lr.n > 0 {
            println!(
                "    ⚠ the logistic baseline never fired here (its validation recall target was\n      unreachable) — the \"beats linear\" bar is UNMEASURED for this gate.\n"
            );
        }

        rows.push((
            gate.name.to_string(),
            m_mlp.f1 as f64,
            m_lr.f1 as f64,
            m_maj.f1 as f64,
            m_mlp.n as f64,
            verdict,
        ));
    }

    println!("summary (F1 on held-out families)");
    println!("  {:<12} {:>8} {:>8} {:>10}", "gate", "mlp", "logreg", "constant");
    for (name, a, b, c, _, _) in &rows {
        println!("  {name:<12} {a:>8.3} {b:>8.3} {c:>10.3}");
    }
    println!();
    if all_pass {
        println!("VERDICT: every gate model beats both the constant and the linear reference.");
    } else {
        println!("VERDICT: at least one gate model does NOT beat a trivial baseline on held-out families.");
    }
    println!();
    println!("CAVEAT: all five datasets are synthetic with mechanical labels. A PASS means");
    println!("\"learnable on the distribution we generate\" — not \"correct on real user data\".");
    println!("What this cannot measure is the real-world base rate: on live traffic these");
    println!("gates see messier text than any generator produces, and the honest test is a");
    println!("shadow-mode A/B on real sessions, which does not exist yet.");
}

fn net_dims(net: &TinyMlp) -> String {
    format!("{}→{}→{}", net.in_dim, net.h1, net.h2)
}

/// Unused import guard: Metrics is referenced through evaluate()'s return
/// type in the summary above.
#[allow(dead_code)]
fn _metrics_type(_: &Metrics) {}
