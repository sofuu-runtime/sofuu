// ml-train/src/train.rs — generic training machinery for the four
// context-economy models (PLAN-ML-GATES §9/§10).
//
// Plain Adam over the shared TinyMlp (sofuu_core::ml::net — the SAME
// forward pass the runtime runs; train/serve skew is structurally
// impossible), grouped train/val/test splits, a logistic-regression
// baseline on the identical features (§10 bar 1b), threshold selection on
// the validation fold only (§10 bar 3), and per-class precision/recall
// reporting (§10 bar 2). Zero external dependencies: a seeded RNG keeps
// every run bit-reproducible.

use sofuu_core::ml::net::TinyMlp;

/* ── Seeded RNG (splitmix64) — reproducibility without a rand dep ── */

pub struct Rng(pub u64);

impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    pub fn uniform(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    /// Approximate standard normal (Irwin–Hall of 6 uniforms, centered).
    pub fn normal(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..6 {
            s += self.uniform();
        }
        s - 3.0
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u32() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

/* ── Dataset + grouped splits ────────────────────────────────────── */

#[derive(Clone, Debug)]
pub struct Example {
    pub x: Vec<f32>,
    pub y: f32,
    /// Group id for grouped splits — a fold never mixes constructions
    /// from the same group (generalization is proven, not assumed, §9).
    pub group: u32,
    /// Debug-only copies of the source text/task — the gray-zone dump on
    /// gate failure prints them so mis-separations are visible.
    pub text: String,
    pub task: String,
}

#[derive(Clone, Debug, Default)]
pub struct Split {
    pub train: Vec<usize>,
    pub val: Vec<usize>,
    pub test: Vec<usize>,
}

/// Split example indices by EXPLICIT group lists (train/val/test). The
/// caller decides which construction families are held out.
pub fn split_by_groups_explicit(
    examples: &[Example],
    train_g: &[u32],
    val_g: &[u32],
    test_g: &[u32],
) -> Split {
    let mut out = Split::default();
    for (i, e) in examples.iter().enumerate() {
        if train_g.contains(&e.group) {
            out.train.push(i);
        } else if val_g.contains(&e.group) {
            out.val.push(i);
        } else if test_g.contains(&e.group) {
            out.test.push(i);
        }
    }
    out
}

/// Split example indices by group: the first `n_val` groups → val, the
/// next `n_test` → test, the rest → train. Deterministic for sorted
/// group lists.
pub fn split_by_groups(examples: &[Example], n_val: usize, n_test: usize) -> Split {
    let mut groups: Vec<u32> = examples.iter().map(|e| e.group).collect();
    groups.sort_unstable();
    groups.dedup();
    let mut out = Split::default();
    for (gi, g) in groups.iter().enumerate() {
        let bucket = if gi < groups.len() - n_val - n_test {
            0
        } else if gi < groups.len() - n_test {
            1
        } else {
            2
        };
        for (i, e) in examples.iter().enumerate() {
            if e.group == *g {
                match bucket {
                    0 => out.train.push(i),
                    1 => out.val.push(i),
                    _ => out.test.push(i),
                }
            }
        }
    }
    out
}

/// Grouped k-fold: each fold holds out ~1/k of the GROUPS. Returns
/// (train, held-out) index lists per fold. `group_order` (when given)
/// fixes which groups land in which fold — position i → fold i % k — so
/// every fold can be given at least one positive-bearing group.
pub fn grouped_kfold(
    examples: &[Example],
    k: usize,
    group_order: Option<&[u32]>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut groups: Vec<u32> = examples.iter().map(|e| e.group).collect();
    groups.sort_unstable();
    groups.dedup();
    let ordered: Vec<u32> = match group_order {
        Some(o) => o.to_vec(),
        None => groups,
    };
    let mut folds = vec![Vec::new(); k];
    for (pos, g) in ordered.iter().enumerate() {
        let fold = pos % k;
        for (i, e) in examples.iter().enumerate() {
            if e.group == *g {
                folds[fold].push(i);
            }
        }
    }
    (0..k)
        .map(|f| {
            let mut train = Vec::new();
            for (g, idxs) in folds.iter().enumerate() {
                if g != f {
                    train.extend(idxs);
                }
            }
            (train, folds[f].clone())
        })
        .collect()
}

/* ── Forward/backward for training (BCE over the sigmoid output) ─── */

struct Fwd {
    a1: Vec<f32>,
    a2: Vec<f32>,
    logit: f32,
}

fn forward(net: &TinyMlp, x: &[f32]) -> Fwd {
    let (in_dim, h1, h2) = (net.in_dim as usize, net.h1 as usize, net.h2 as usize);
    let w = &net.w;
    let mut a1 = vec![0.0f32; h1];
    for j in 0..h1 {
        let mut acc = w[h1 * in_dim + j];
        let row = &w[j * in_dim..(j + 1) * in_dim];
        for (i, &xi) in x.iter().enumerate() {
            acc += row[i] * xi;
        }
        a1[j] = acc.tanh();
    }
    let w2 = h1 * in_dim + h1;
    let mut a2 = vec![0.0f32; h2];
    for j in 0..h2 {
        let mut acc = w[w2 + h2 * h1 + j];
        let row = &w[w2 + j * h1..w2 + (j + 1) * h1];
        for (i, &hi) in a1.iter().enumerate() {
            acc += row[i] * hi;
        }
        a2[j] = acc.tanh();
    }
    let w3 = w2 + h2 * h1 + h2;
    let mut logit = w[w3 + h2];
    for (i, &hi) in a2.iter().enumerate() {
        logit += w[w3 + i] * hi;
    }
    Fwd { a1, a2, logit }
}

pub fn predict(net: &TinyMlp, x: &[f32]) -> f32 {
    1.0 / (1.0 + (-forward(net, x).logit).exp())
}

fn bce(p: f32, y: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

/// One SGD step over a minibatch (manual backprop, Adam state external).
fn accumulate_grads(net: &TinyMlp, batch: &[&Example], grad: &mut [f32]) {
    let (in_dim, h1, h2) = (net.in_dim as usize, net.h1 as usize, net.h2 as usize);
    let w = &net.w;
    grad.fill(0.0);
    for ex in batch {
        let fwd = forward(net, &ex.x);
        let p = 1.0 / (1.0 + (-fwd.logit).exp());
        let mut dl_do = p - ex.y;
        if !dl_do.is_finite() {
            dl_do = 0.0;
        }
        let w2 = h1 * in_dim + h1;
        let w3 = w2 + h2 * h1 + h2;
        // Output layer
        grad[w3 + h2] += dl_do;
        let mut da2 = vec![0.0f32; h2];
        for i in 0..h2 {
            grad[w3 + i] += dl_do * fwd.a2[i];
            da2[i] = dl_do * w[w3 + i];
        }
        // Layer 2
        let mut da1 = vec![0.0f32; h1];
        for j in 0..h2 {
            let dz2 = da2[j] * (1.0 - fwd.a2[j] * fwd.a2[j]);
            grad[w2 + h2 * h1 + j] += dz2;
            let row = w2 + j * h1;
            for i in 0..h1 {
                grad[row + i] += dz2 * fwd.a1[i];
                da1[i] += dz2 * w[row + i];
            }
        }
        // Layer 1
        for j in 0..h1 {
            let dz1 = da1[j] * (1.0 - fwd.a1[j] * fwd.a1[j]);
            grad[h1 * in_dim + j] += dz1;
            let row = j * in_dim;
            for i in 0..in_dim {
                grad[row + i] += dz1 * ex.x[i];
            }
        }
    }
    let inv = 1.0 / batch.len() as f32;
    for g in grad.iter_mut() {
        *g *= inv;
    }
}

/* ── Adam ────────────────────────────────────────────────────────── */

pub struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    lr: f32,
}

impl Adam {
    pub fn new(n: usize, lr: f32) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n], t: 0, lr }
    }
    pub fn step(&mut self, w: &mut [f32], g: &[f32]) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let c1 = 1.0 - b1.powi(self.t as i32);
        let c2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..w.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            w[i] -= self.lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + eps);
        }
    }
}

/* ── Training loop with early stopping ───────────────────────────── */

pub struct TrainCfg {
    pub lr: f32,
    pub batch: usize,
    pub max_epochs: usize,
    pub patience: usize,
    /// Patience only starts counting after this many epochs — a noisy or
    /// shifted val fold must not be able to restore epoch-1 weights by
    /// diverging immediately (the collapse that motivated this guard).
    pub min_epochs: usize,
    pub seed: u64,
    /// L2 weight decay — with ~8.7k params and a few thousand curated
    /// examples, decay is what keeps the net learning the shared signal
    /// instead of memorizing construction-specific trigrams.
    pub weight_decay: f32,
}

pub fn init_weights(net: &mut TinyMlp, seed: u64) {
    let mut rng = Rng(seed);
    let (in_dim, h1, h2) = (net.in_dim as usize, net.h1 as usize, net.h2 as usize);
    // Xavier per LAYER — scaling every layer by the input fan-in (the old
    // bug) over-initializes layer 2/3, saturates the tanhs, and collapses
    // training to the constant-negative predictor.
    let mut init_block = |w: &mut [f32], fan_in: usize| {
        let scale = (1.0 / fan_in.max(1) as f32).sqrt();
        for v in w.iter_mut() {
            *v = rng.normal() * scale;
        }
    };
    let w = &mut net.w;
    let w1_end = h1 * in_dim; // W1
    init_block(&mut w[..w1_end], in_dim);
    // b1 stays zero
    let w2_start = w1_end + h1;
    let w2_end = w2_start + h2 * h1; // W2
    init_block(&mut w[w2_start..w2_end], h1);
    // b2 stays zero
    let w3_start = w2_end + h2;
    let w3_end = w3_start + h2; // W3
    init_block(&mut w[w3_start..w3_end], h2);
    // b3 stays zero
}

fn mean_loss(net: &TinyMlp, examples: &[Example], idxs: &[usize]) -> f32 {
    let mut s = 0.0f32;
    for &i in idxs {
        s += bce(predict(net, &examples[i].x), examples[i].y);
    }
    s / idxs.len().max(1) as f32
}

/// Train to early stopping on the val loss; restore best weights.
pub fn train_mlp(net: &mut TinyMlp, examples: &[Example], split: &Split, cfg: &TrainCfg) -> u32 {
    init_weights(net, cfg.seed);
    let mut adam = Adam::new(net.w.len(), cfg.lr);
    let mut grad = vec![0.0f32; net.w.len()];
    let mut rng = Rng(cfg.seed ^ 0x51_7C_C1_B7_27_22_0A_95);
    let mut order = split.train.clone();
    let mut best_w = net.w.clone();
    let mut best_loss = f32::INFINITY;
    let mut stale = 0usize;
    let mut epochs = 0u32;
    for epoch in 0..cfg.max_epochs {
        epochs = epoch as u32 + 1;
        rng.shuffle(&mut order);
        for start in (0..order.len()).step_by(cfg.batch) {
            let batch: Vec<&Example> = order[start..(start + cfg.batch).min(order.len())]
                .iter()
                .map(|&i| &examples[i])
                .collect();
            accumulate_grads(net, &batch, &mut grad);
            if cfg.weight_decay > 0.0 {
                for (g, w) in grad.iter_mut().zip(net.w.iter()) {
                    *g += cfg.weight_decay * *w;
                }
            }
            adam.step(&mut net.w, &grad);
        }
        let vl = mean_loss(net, examples, &split.val);
        if vl < best_loss - 1e-5 {
            best_loss = vl;
            best_w = net.w.clone();
            stale = 0;
        } else if epoch + 1 >= cfg.min_epochs {
            stale += 1;
            if stale >= cfg.patience {
                break;
            }
        }
    }
    net.w = best_w;
    epochs
}

/* ── Logistic regression baseline (§10 bar 1b) ───────────────────── */

pub struct LogReg {
    pub w: Vec<f32>,
    pub b: f32,
}

impl LogReg {
    pub fn predict(&self, x: &[f32]) -> f32 {
        let mut acc = self.b;
        for (i, &xi) in x.iter().enumerate() {
            acc += self.w[i] * xi;
        }
        1.0 / (1.0 + (-acc).exp())
    }
}

pub fn train_logreg(feats: usize, examples: &[Example], split: &Split, seed: u64) -> LogReg {
    let mut lr = LogReg { w: vec![0.0; feats], b: 0.0 };
    let mut adam = Adam::new(feats + 1, 0.05);
    let mut g = vec![0.0f32; feats + 1];
    let mut rng = Rng(seed ^ 0x2545_F491_4F6C_DD1D);
    let mut order = split.train.clone();
    let mut best = (f32::INFINITY, lr.w.clone(), lr.b);
    let mut stale = 0usize;
    for _ in 0..300 {
        rng.shuffle(&mut order);
        for start in (0..order.len()).step_by(64) {
            let end = (start + 64).min(order.len());
            g.fill(0.0);
            for &i in &order[start..end] {
                let ex = &examples[i];
                let p = lr.predict(&ex.x);
                let d = p - ex.y;
                g[feats] += d;
                for (j, &xj) in ex.x.iter().enumerate() {
                    g[j] += d * xj;
                }
            }
            let inv = 1.0 / (end - start) as f32;
            for v in g.iter_mut() {
                *v *= inv;
            }
            let mut all: Vec<f32> = lr.w.iter().copied().chain([lr.b]).collect();
            adam.step(&mut all, &g);
            lr.w.copy_from_slice(&all[..feats]);
            lr.b = all[feats];
        }
        let mut vl = 0.0f32;
        for &i in &split.val {
            vl += bce(lr.predict(&examples[i].x), examples[i].y);
        }
        vl /= split.val.len().max(1) as f32;
        if vl < best.0 - 1e-5 {
            best = (vl, lr.w.clone(), lr.b);
            stale = 0;
        } else {
            stale += 1;
            if stale >= 30 {
                break;
            }
        }
    }
    lr.w = best.1;
    lr.b = best.2;
    lr
}

/* ── Metrics + threshold selection ───────────────────────────────── */

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub n: usize,
    pub positives: usize,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

pub fn evaluate(preds: &[f32], labels: &[f32], threshold: f32) -> Metrics {
    let mut m = Metrics { n: preds.len(), ..Default::default() };
    let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
    let mut correct = 0usize;
    for (&p, &y) in preds.iter().zip(labels.iter()) {
        let pred = if p >= threshold { 1.0 } else { 0.0 };
        if y >= 0.5 {
            m.positives += 1;
        }
        if pred == y {
            correct += 1;
        }
        match (pred >= 0.5, y >= 0.5) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            _ => {}
        }
    }
    m.accuracy = correct as f32 / m.n.max(1) as f32;
    m.precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 1.0 };
    m.recall = if tp + fn_ > 0 { tp as f32 / (tp + fn_) as f32 } else { 1.0 };
    m.f1 = if m.precision + m.recall > 0.0 {
        2.0 * m.precision * m.recall / (m.precision + m.recall)
    } else {
        0.0
    };
    m
}

/// Threshold sweep on (preds, labels): highest threshold whose recall
/// stays ≥ `min_recall` (maximizes precision under a recall floor — the
/// freshness rule, §5). Models without an asymmetric floor can sweep for
/// max F1 instead. Used on the VALIDATION fold only.
///
/// When the fold is perfectly separable the sweep never sees recall bend,
/// so the floor is not a measured boundary — returning the top of the
/// sweep would ship a threshold pinned to the extreme with zero margin
/// against val→test score shift (the supervisor run-2/3 pathology: val
/// 1.000, threshold 0.95, test recall 0.50). In that case fall back to
/// the midpoint of the observed margin (highest negative vs lowest
/// positive score): a recall-first gate must sit INSIDE the gap.
pub fn threshold_for_recall(preds: &[f32], labels: &[f32], min_recall: f32) -> f32 {
    let mut best = 0.05f32;
    let mut t = 0.05f32;
    let mut broke = false;
    while t <= 0.951 {
        let m = evaluate(preds, labels, t);
        if m.recall >= min_recall {
            best = t;
        } else {
            broke = true;
            break;
        }
        t += 0.01;
    }
    if !broke {
        let pos_min = preds
            .iter()
            .zip(labels.iter())
            .filter(|(_, &y)| y >= 0.5)
            .map(|(&p, _)| p)
            .fold(1.0f32, f32::min);
        let neg_max = preds
            .iter()
            .zip(labels.iter())
            .filter(|(_, &y)| y < 0.5)
            .map(|(&p, _)| p)
            .fold(0.0f32, f32::max);
        best = ((pos_min + neg_max) / 2.0).clamp(0.05, 0.95);
    }
    best
}

/// Dual of `threshold_for_recall` for keep-safety gates (compaction, §12):
/// the LOWEST threshold whose precision stays ≥ `min_precision` — frees as
/// much disposable material as possible without crossing the floor that
/// protects load-bearing segments. Descends from the top; the first step
/// below the floor ends the sweep. Used on the VALIDATION fold only.
///
/// When the fold is perfectly separable the sweep never sees precision
/// bend, so the floor is not a measured boundary — returning the bottom
/// of the score range would flag nearly everything. In that case fall
/// back to the midpoint of the observed margin (highest keep score vs
/// lowest disposable score): a keep-safety gate must sit INSIDE the gap.
pub fn threshold_for_precision(preds: &[f32], labels: &[f32], min_precision: f32) -> f32 {
    let mut best = 0.95f32;
    let mut t = 0.95f32;
    let mut broke = false;
    while t >= 0.049 {
        let m = evaluate(preds, labels, t);
        if m.precision >= min_precision {
            best = t;
        } else {
            broke = true;
            break;
        }
        t -= 0.01;
    }
    if !broke {
        let keep_max = preds
            .iter()
            .zip(labels.iter())
            .filter(|(_, &y)| y < 0.5)
            .map(|(&p, _)| p)
            .fold(0.0f32, f32::max);
        let disp_min = preds
            .iter()
            .zip(labels.iter())
            .filter(|(_, &y)| y >= 0.5)
            .map(|(&p, _)| p)
            .fold(1.0f32, f32::min);
        best = ((keep_max + disp_min) / 2.0).clamp(0.05, 0.95);
    }
    best
}

/// Reliability report: mean predicted probability vs observed positive
/// rate per decile bucket (§10 bar 5 — reliability of output probabilities).
pub fn calibration_report(preds: &[f32], labels: &[f32]) -> String {
    let mut out = String::from("    bucket  n    mean_p  obs_rate\n");
    for b in 0..10 {
        let lo = b as f32 * 0.1;
        let hi = lo + 0.1;
        let mut n = 0usize;
        let mut sp = 0.0f32;
        let mut sy = 0.0f32;
        for (&p, &y) in preds.iter().zip(labels.iter()) {
            if p >= lo && (p < hi || (b == 9 && p <= hi)) {
                n += 1;
                sp += p;
                sy += y;
            }
        }
        if n == 0 {
            continue;
        }
        out.push_str(&format!(
            "    {:.1}-{:.1}  {:<4}  {:.3}   {:.3}\n",
            lo,
            hi,
            n,
            sp / n as f32,
            sy / n as f32
        ));
    }
    out
}
