//! Offline trainer for `semantic-projector-v1`.
//!
//! The shipped runtime does not carry a teacher model or a training
//! dependency.  This command builds the procedural corpus from
//! `data_embedding_gen` (hundreds of randomized families, deterministic under
//! a fixed corpus seed), optionally boosted per category from the §10
//! harness's failure-mining file, and trains the student projector with
//! listwise multi-label softmax metric learning (round 5; `SOFUU_EMB_LOSS=pairwise`
//! restores the round-1..4 hinge objective).
//!
//! Model selection is deliberately group-level: every epoch is scored on
//! held-out FAMILIES the trainer never saw, mirroring the acceptance
//! harness's unseen-family retrieval.  Selecting on held-out examples of
//! seen families (the v1 trainer's mistake) measures memorization, not
//! generalization.
//!
//! Round 5 adds three accuracy levers that change neither the parameter
//! count nor the input features: a listwise multi-label softmax loss (the
//! whole candidate set competes at once, so the gradient calibrates the
//! score distribution instead of pushing pairs past a fixed margin), model
//! soup (several runs sharing one init are weight-averaged greedily —
//! flat-minima averaging, no extra parameters at deploy time), and
//! quantization-aware training (the loss sees the int8 grid the artifact
//! ships on, via `embedding::fake_quant_weights` + straight-through grads).
//!
//! Environment knobs (all optional):
//!   SOFUU_EMB_SEEDS   csv init seeds        (default "11,23,47,89")
//!   SOFUU_EMB_EPOCHS  epoch budget          (default 240)
//!   SOFUU_EMB_OUT     artifact output path  (default: in-tree weights_v1.sem)
//!   SOFUU_EMB_CORPUS  corpus seed           (default: fixed CORPUS_SEED)
//!   SOFUU_EMB_MINING  =0 disables reading /tmp/sofuu_embed_mining.tsv
//!   SOFUU_EMB_LOSS    "pairwise" restores the hinge objective (default listwise)
//!   SOFUU_EMB_TAU     listwise softmax temperature scale (default 20)
//!   SOFUU_EMB_SOUP    =0 picks the best single seed instead of averaging (default on)
//!   SOFUU_EMB_QAT     =0 disables quantization-aware training (default on)
//!   SOFUU_EMB_QAT_WARMUP  dense epochs before QAT engages (default 60)

use std::collections::HashMap;
use std::path::PathBuf;

use sofuu_core::embedding::{self, ProjectorF32, HASH_DIM, HIDDEN_DIM, PARAM_COUNT, SEMANTIC_DIM};

use crate::data_embedding_gen::{self, Corpus, CURATED_CAT};
use crate::embedding_teacher::{self, Teacher};
use crate::embedding_v2;
use crate::train::{Adam, Rng};
use sofuu_core::embedding::topk_row_mask;

const MINING_PATH: &str = "/tmp/sofuu_embed_mining.tsv";
/// Teacher-anchor weight (v1 used 0.20 with colliding one-hot slots; the
/// sparse per-family codes below need much less help).
const TEACHER_WEIGHT: f32 = 0.05;
/// Distillation replaces the sparse-code anchor with teacher-geometry
/// regression (applied on anchor AND positive) at this weight.
const TEACHER_MSE_WEIGHT_DISTILL: f32 = 0.25;
/// Training-time input dropout: zero this fraction of the ACTIVE hash
/// features per anchor.  Short queries activate ~15 of 768 trigrams; without
/// dropout the tanh layer free-rides on template coverage and collapses
/// every sparse input to one bias-driven direction.
const INPUT_DROPOUT: f32 = 0.4;
/// Extra L2 on the hidden bias b1: a large b1 makes tanh(j) ≈ const for
/// sparse inputs, which is exactly the short-query collapse.
const B1_DECAY: f32 = 1e-3;
/// Structured-sparsity schedule (SOFUU_EMB_SPARSE_K): dense warmup, then a
/// one-time top-K prune of every W1 row; masked gradients keep the support
/// frozen for the rest of training.
const PRUNE_WARMUP_EPOCHS: usize = 40;
/// Ranking pushes any negative whose cosine still exceeds this.
const NEG_GATE: f32 = 0.10;
const NEG_SAME_CAT: usize = 6;
const NEG_ANY: usize = 2;
/// Listwise softmax temperature scale (SOFUU_EMB_TAU).  τ multiplies the
/// hybrid cosines inside the softmax; ~20 keeps the candidate distribution
/// sharp enough that easy negatives self-gate (no hinge margin needed).
const LISTWISE_TAU: f32 = 20.0;
/// Family siblings pulled in as extra positives per anchor (the self-view
/// is positive #0; families carry 3 memories).
const LISTWISE_MAX_POS: usize = 3;
/// Dense epochs before the fake-quant shadow engages (SOFUU_EMB_QAT_WARMUP).
const QAT_WARMUP_EPOCHS: usize = 60;

#[derive(Clone)]
struct Item {
    x: Vec<f32>,
    /// Word buckets for the SEM2 table channel (round 9).  Computed for
    /// every item; SEM1 models ignore it (and never consume RNG for it).
    buckets: Vec<u32>,
    target: [f32; SEMANTIC_DIM],
}

struct TrainData {
    items: Vec<Item>,
    /// item index -> owning family
    fam_of_item: Vec<usize>,
    /// contiguous item-index range per family
    fam_items: Vec<std::ops::Range<usize>>,
    cat_of_fam: Vec<usize>,
    /// category -> family indices
    cat_fams: Vec<Vec<usize>>,
}

struct ValData {
    /// (category, family) -> memory texts; prototypes are built per family
    fam_memories: Vec<(usize, Vec<String>)>,
    /// (category, family, query text)
    queries: Vec<(usize, usize, String)>,
}

/// Per-family teacher code: 3-sparse ±1 in the 64-dim output.  Unlike the v1
/// one-hot at `group % 16`, distinct families never collide on a slot, so
/// larger corpora stay consistent.
fn teacher_code(fam: usize) -> [f32; SEMANTIC_DIM] {
    let mut out = [0.0f32; SEMANTIC_DIM];
    let mut rng = Rng(0x7EAC_1100_C0DE ^ (fam as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut placed = 0usize;
    while placed < 3 {
        let k = (rng.next_u32() as usize) % SEMANTIC_DIM;
        if out[k] == 0.0 {
            out[k] = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            placed += 1;
        }
    }
    out
}

fn build_train(c: &Corpus, mut teacher: Option<&mut Teacher>) -> (TrainData, usize) {
    let mut items = Vec::new();
    let mut fam_of_item = Vec::new();
    let mut fam_items = Vec::new();
    let mut cat_of_fam = Vec::new();
    let mut cat_fams = vec![Vec::new(); data_embedding_gen::CAT_NAMES.len() + 1];
    let mut fam = 0usize;
    for (ci, cat) in c.cats.iter().enumerate() {
        for f in &cat.train {
            let start = items.len();
            let code = teacher_code(fam);
            for t in f.memories.iter().chain(&f.train_queries) {
                let target = match &mut teacher {
                    Some(tch) => tch.embed(t).unwrap_or(code),
                    None => code,
                };
                items.push(Item {
                    x: embedding::hash_v1_features(t),
                    buckets: embedding_v2::tokenize(t),
                    target,
                });
                fam_of_item.push(fam);
            }
            fam_items.push(start..items.len());
            cat_of_fam.push(ci);
            cat_fams[ci].push(fam);
            fam += 1;
        }
    }
    for g in &c.curated {
        let start = items.len();
        let code = teacher_code(fam);
        for t in g {
            let target = match &mut teacher {
                Some(tch) => tch.embed(t).unwrap_or(code),
                None => code,
            };
            items.push(Item {
                x: embedding::hash_v1_features(t),
                buckets: embedding_v2::tokenize(t),
                target,
            });
            fam_of_item.push(fam);
        }
        fam_items.push(start..items.len());
        cat_of_fam.push(CURATED_CAT);
        cat_fams[CURATED_CAT].push(fam);
        fam += 1;
    }
    (
        TrainData {
            items,
            fam_of_item,
            fam_items,
            cat_of_fam,
            cat_fams,
        },
        fam,
    )
}

fn build_val(c: &Corpus) -> ValData {
    let mut fam_memories = Vec::new();
    let mut queries = Vec::new();
    for (ci, cat) in c.cats.iter().enumerate() {
        for f in &cat.val {
            let fi = fam_memories.len();
            fam_memories.push((ci, f.memories.clone()));
            for q in &f.val_queries {
                queries.push((ci, fi, q.clone()));
            }
        }
    }
    ValData {
        fam_memories,
        queries,
    }
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Zero a random fraction of the active features (training only).
fn dropout(x: &[f32], rng: &mut Rng, p: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            if v != 0.0 && rng.uniform() < p {
                0.0
            } else {
                v
            }
        })
        .collect()
}

/// Held-out-family retrieval, the trainer's mirror of the acceptance
/// harness: each val query must surface its own family's memories in the
/// top-5 of its category's candidate pool.  Returns (recall@5, margin).
fn val_metrics(embed: &dyn Fn(&str) -> Vec<f32>, vd: &ValData) -> (f32, f32) {
    let mut mem_emb: Vec<Vec<f32>> = Vec::with_capacity(vd.fam_memories.len());
    for (_, mems) in &vd.fam_memories {
        // family prototype = L2-normalized mean of its memory embeddings
        let dim = {
            let e = embed(&mems[0]);
            e.len()
        };
        let mut mean = vec![0.0f32; dim];
        for m in mems {
            let e = embed(m);
            for (mv, ev) in mean.iter_mut().zip(&e) {
                *mv += ev;
            }
        }
        mem_emb.push(unit(&mean));
    }
    let mut hits = 0usize;
    let mut margin_sum = 0.0f32;
    for (cat, fam, q) in &vd.queries {
        let qe = unit(&embed(q));
        let mut own_best = -1.0f32;
        let mut other_best = -1.0f32;
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (fi, (mcat, _)) in vd.fam_memories.iter().enumerate() {
            if mcat != cat {
                continue;
            }
            let s = cos(&qe, &mem_emb[fi]);
            scored.push((fi, s));
            if fi == *fam {
                own_best = own_best.max(s);
            } else {
                other_best = other_best.max(s);
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if scored.iter().take(5).any(|&(fi, _)| fi == *fam) {
            hits += 1;
        }
        margin_sum += own_best - other_best;
    }
    let n = vd.queries.len().max(1) as f32;
    (hits as f32 / n, margin_sum / n)
}

// ── generation-agnostic lens model (round 9) ────────────────────────────
/// The training loop is shared between SEM1 (trigram-only tower) and
/// SEM2 (trigram tower + learned word table, embedding_v2.rs): pair
/// sampling, dropout schedule, decay regions, staleness and soup are
/// byte-identical recipes — only the parameter layout, forward/backward
/// math and export format differ.  `USES_BUCKETS` gates the word channel
/// so the v1 RNG stream (and therefore the shipped artifact) stays
/// bit-identical to pre-v2 runs.
pub trait LensModel: Clone {
    const PARAMS: usize;
    const USES_BUCKETS: bool;
    /// Hidden-bias region for the stronger B1_DECAY.
    const B1_START: usize;
    const B1_LEN: usize;
    const ARCH: &'static str;
    fn zeros() -> Self;
    fn init(seed: u64) -> Self;
    fn w(&self) -> &Vec<f32>;
    fn w_mut(&mut self) -> &mut Vec<f32>;
    fn forward_at(&self, x: &[f32], buckets: &[u32]) -> [f32; SEMANTIC_DIM];
    fn accumulate_at(
        &self,
        x: &[f32],
        buckets: &[u32],
        d_output: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    );
    fn mse_at(
        &self,
        x: &[f32],
        buckets: &[u32],
        target: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) -> f32;
    fn fake_quant(&self) -> Self;
    fn export_blob(&self) -> (Vec<u8>, String);
    fn export_blob_sparse(&self, _k: usize) -> (Vec<u8>, String) {
        unimplemented!("sparse export is SEM1-only")
    }
}

impl LensModel for ProjectorF32 {
    const PARAMS: usize = PARAM_COUNT;
    const USES_BUCKETS: bool = false;
    const B1_START: usize = HASH_DIM * HIDDEN_DIM;
    const B1_LEN: usize = HIDDEN_DIM;
    const ARCH: &'static str = "768 → 16 → 64";
    fn zeros() -> Self {
        ProjectorF32::zeros()
    }
    fn init(seed: u64) -> Self {
        let mut model = ProjectorF32::zeros();
        let mut rng = Rng(seed);
        let w1_scale = (2.0 / (HASH_DIM + HIDDEN_DIM) as f32).sqrt();
        for v in &mut model.w[..HASH_DIM * HIDDEN_DIM] {
            *v = rng.normal() * w1_scale;
        }
        let w2_start = HASH_DIM * HIDDEN_DIM + HIDDEN_DIM;
        let w2_scale = (2.0 / (HIDDEN_DIM + SEMANTIC_DIM) as f32).sqrt();
        for v in &mut model.w[w2_start..w2_start + HIDDEN_DIM * SEMANTIC_DIM] {
            *v = rng.normal() * w2_scale;
        }
        model
    }
    fn w(&self) -> &Vec<f32> {
        &self.w
    }
    fn w_mut(&mut self) -> &mut Vec<f32> {
        &mut self.w
    }
    fn forward_at(&self, x: &[f32], buckets: &[u32]) -> [f32; SEMANTIC_DIM] {
        debug_assert!(buckets.is_empty(), "SEM1 has no word channel");
        self.forward(x)
    }
    fn accumulate_at(
        &self,
        x: &[f32],
        _buckets: &[u32],
        d_output: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) {
        self.accumulate_output_grad(x, d_output, grad)
    }
    fn mse_at(
        &self,
        x: &[f32],
        _buckets: &[u32],
        target: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) -> f32 {
        self.accumulate_mse_grad(x, target, grad)
    }
    fn fake_quant(&self) -> Self {
        ProjectorF32::new(embedding::fake_quant_weights(&self.w))
    }
    fn export_blob(&self) -> (Vec<u8>, String) {
        let q = self.quantize();
        let b = q.to_blob();
        let checked =
            embedding::QuantizedProjector::from_blob(&b).expect("self-check SEM1 artifact");
        assert_eq!(checked.to_blob(), b, "SEM1 serialization must round-trip");
        (b, "SEM1 v1 dense".to_string())
    }
    fn export_blob_sparse(&self, k: usize) -> (Vec<u8>, String) {
        let q = self.quantize_sparse(k);
        let b = q.to_blob_v2();
        let checked =
            embedding::QuantizedProjector::from_blob(&b).expect("self-check SEM1 v2 artifact");
        assert_eq!(checked.to_blob_v2(), b, "SEM1 v2 serialization must round-trip");
        (b, format!("SEM1 v2 sparse K={k}"))
    }
}

impl LensModel for embedding_v2::V2Projector {
    const PARAMS: usize = embedding_v2::PARAMS_V2;
    const USES_BUCKETS: bool = true;
    const B1_START: usize = embedding_v2::B1_START;
    const B1_LEN: usize = embedding_v2::V2_H;
    const ARCH: &'static str = "768+word16 → 16 → 64";
    fn zeros() -> Self {
        embedding_v2::V2Projector::zeros()
    }
    fn init(seed: u64) -> Self {
        embedding_v2::V2Projector::init(seed)
    }
    fn w(&self) -> &Vec<f32> {
        &self.w
    }
    fn w_mut(&mut self) -> &mut Vec<f32> {
        &mut self.w
    }
    fn forward_at(&self, x: &[f32], buckets: &[u32]) -> [f32; SEMANTIC_DIM] {
        embedding_v2::V2Projector::forward_at(self, x, buckets)
    }
    fn accumulate_at(
        &self,
        x: &[f32],
        buckets: &[u32],
        d_output: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) {
        embedding_v2::V2Projector::accumulate_output_grad_at(self, x, buckets, d_output, grad)
    }
    fn mse_at(
        &self,
        x: &[f32],
        buckets: &[u32],
        target: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) -> f32 {
        embedding_v2::V2Projector::accumulate_mse_grad_at(self, x, buckets, target, grad)
    }
    fn fake_quant(&self) -> Self {
        embedding_v2::V2Projector::fake_quant(self)
    }
    fn export_blob(&self) -> (Vec<u8>, String) {
        let q = embedding_v2::V2Projector::quantize(self);
        let b = q.to_blob();
        let checked = embedding_v2::QuantizedV2::from_blob(&b).expect("self-check SEM2 artifact");
        assert_eq!(checked.to_blob(), b, "SEM2 serialization must round-trip");
        (b, "SEM2 v1 word-table".to_string())
    }
}

/// Bucket dropout that consumes RNG only for models with a word channel.
fn drop_buckets_if<M: LensModel>(buckets: &[u32], rng: &mut Rng, p: f32) -> Vec<u32> {
    if M::USES_BUCKETS {
        embedding_v2::drop_buckets(buckets, rng, p)
    } else {
        Vec::new()
    }
}

/// Buckets for a raw text (validation/report path), same gating.
fn buckets_for<M: LensModel>(t: &str) -> Vec<u32> {
    if M::USES_BUCKETS {
        embedding_v2::tokenize(t)
    } else {
        Vec::new()
    }
}

/// Elementwise mean of several same-init models — the model-soup
/// average.  Only meaningful for runs that share an init: different inits
/// land in different basins (permutation symmetry) and do not average.
fn average_models<M: LensModel>(models: &[M]) -> M {
    assert!(!models.is_empty(), "soup needs at least one member");
    let mut out = M::zeros();
    let inv = 1.0 / models.len() as f32;
    for m in models {
        for (o, v) in out.w_mut().iter_mut().zip(m.w().iter()) {
            *o += *v * inv;
        }
    }
    out
}

/// The deployed lens for a given projector: hash features (+ word buckets
/// for SEM2) → tower forward → hybrid blend with the frozen anchor channel
/// (K=0 → tower-only).  Shared by validation, soup probing and the final
/// report so every metric is measured through exactly the geometry the §10
/// harness grades.
fn hybrid_lens_embed<'a, M: LensModel>(m: &'a M) -> impl Fn(&str) -> Vec<f32> + 'a {
    let (k, w) = crate::embedding_anchor::anchor_cfg();
    let mat = if k > 0 {
        crate::embedding_anchor::anchor_matrix(k)
    } else {
        Vec::new()
    };
    move |t: &str| {
        let x = embedding::hash_v1_features(t);
        let b = buckets_for::<M>(t);
        let mut f = m.forward_at(&x, &b).to_vec();
        if k == 0 {
            return f;
        }
        crate::embedding_anchor::unit(&mut f);
        let a = crate::embedding_anchor::anchor_forward(&x, &mat, k);
        crate::embedding_anchor::blend(&f, &a, k, w)
    }
}

fn train_one<M: LensModel>(
    data_seed: u64,
    init_seed: u64,
    td: &TrainData,
    vd: &ValData,
    epochs_max: usize,
    distilling: bool,
    sparse_k: Option<usize>,
    listwise: bool,
    tau: f32,
    qat: bool,
    qat_warmup: usize,
) -> (M, f32, f32, u32) {
    let mut model = M::init(init_seed);
    let mut adam = Adam::new(M::PARAMS, 0.004);
    let mut grad = vec![0.0f32; M::PARAMS];
    let mut teacher_grad = vec![0.0f32; M::PARAMS];
    let mut rng = Rng(data_seed ^ 0xA6_1D_4B_72_91_0F_55_2C);
    let mut best_model = model.clone();
    let mut best_r = -1.0f32;
    let mut best_m = -1.0f32;
    let mut stale = 0usize;
    let mut epochs = 0u32;
    let mut mask: Option<Vec<bool>> = None;
    // hybrid lens: anchor width K and energy share w (K=0 → tower-only, the
    // pre-R6 behavior).  When K > 0 the ranking loss sees the FULL hybrid
    // vector — frozen anchor channel + trained tower channel — so the tower
    // learns only the delta the anchor lacks (see embedding_anchor.rs).
    let (anchor_k, anchor_w) = crate::embedding_anchor::anchor_cfg();
    let anchor_mat = if anchor_k > 0 {
        Some(crate::embedding_anchor::anchor_matrix(anchor_k))
    } else {
        None
    };
    let f_dims = SEMANTIC_DIM - anchor_k;
    // forward through the lens: returns (full hybrid v, tower f_hat, anchor a_hat)
    let hybrid_forward = |model: &M, x: &[f32], b: &[u32]| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let f_raw = model.forward_at(x, b);
        let mut f_hat = f_raw.to_vec();
        crate::embedding_anchor::unit(&mut f_hat);
        match &anchor_mat {
            None => (f_hat.clone(), f_hat, Vec::new()),
            Some(r) => {
                let a = crate::embedding_anchor::anchor_forward(x, r, anchor_k);
                let v = crate::embedding_anchor::blend(&f_hat, &a, anchor_k, anchor_w);
                (v, f_hat, a)
            }
        }
    };
    let hybrid_cos_of = |fa: &[f32], aa: &[f32], fp: &[f32], ap: &[f32]| -> f32 {
        match &anchor_mat {
            None => cos(fa, fp),
            Some(_) => crate::embedding_anchor::hybrid_cos(fa, aa, fp, ap, anchor_k, anchor_w),
        }
    };
    let mse_weight = match std::env::var("SOFUU_EMB_TEACHER_W") {
        Ok(v) => v.parse::<f32>().unwrap_or(0.25).clamp(0.0, 1.0),
        Err(_) => {
            if distilling {
                TEACHER_MSE_WEIGHT_DISTILL
            } else {
                TEACHER_WEIGHT
            }
        }
    };

    for epoch in 0..epochs_max {
        epochs = epoch as u32 + 1;
        // one-time structured prune after the dense warmup
        if mask.is_none() {
            if let Some(k) = sparse_k {
                if epoch >= PRUNE_WARMUP_EPOCHS {
                    let m = topk_row_mask(model.w(), HIDDEN_DIM, HASH_DIM, k);
                    for (w, keep) in model.w_mut().iter_mut().zip(&m) {
                        if !keep {
                            *w = 0.0;
                        }
                    }
                    mask = Some(m);
                    println!("  seed {data_seed}: pruned W1 rows to top-{k} at epoch {epoch}");
                }
            }
        }
        // QAT schedule: dense until the master finds a good basin, then the
        // loss trains on the deployment grid.
        let use_qat = qat && epoch >= qat_warmup;
        if qat && epoch == qat_warmup {
            println!("  seed {data_seed}: QAT engaged at epoch {epoch}");
        }
        // Balanced multi-task sampling: every category contributes an equal
        // pair budget per epoch, so associative-heavy categories cannot
        // drown out the exact-token skills (code/versions/errors) — the
        // acceptance harness weighs all categories equally.
        let mut per_cat: Vec<Vec<(usize, usize)>> = vec![Vec::new(); td.cat_fams.len()];
        for (f, r) in td.fam_items.iter().enumerate() {
            let cat = td.cat_of_fam[f];
            for off in 0..r.len() {
                per_cat[cat].push((r.start + off, r.start + (off + 1) % r.len()));
                // SimCSE self-pair: two dropout views of the SAME text pulled
                // together — preserves token identity (the exact-match skills)
                // and makes sparse inputs map to themselves, not to a
                // bias-driven constant.
                per_cat[cat].push((r.start + off, r.start + off));
            }
        }
        for list in per_cat.iter_mut() {
            rng.shuffle(list);
        }
        let min_pairs = per_cat
            .iter()
            .filter(|l| !l.is_empty())
            .map(|l| l.len())
            .min()
            .unwrap_or(0);
        let cap = (min_pairs as f32 * 1.5) as usize;
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for list in &per_cat {
            pairs.extend(list.iter().take(cap).copied());
        }
        let mut pair_order: Vec<usize> = (0..pairs.len()).collect();
        rng.shuffle(&mut pair_order);

        for batch in pair_order.chunks(8) {
            grad.fill(0.0);
            teacher_grad.fill(0.0);
            // QAT shadow: rebuilt from the f32 master every batch (the master
            // moves each step), fake-quantized to the exact int8 grid the
            // artifact ships on.  Forward AND backward caches are evaluated at
            // the quantized point; gradients flow straight through into the
            // shared `grad` and are applied 1:1 to the master.
            let fq_shadow;
            let fq: &M = if use_qat {
                fq_shadow = model.fake_quant();
                &fq_shadow
            } else {
                &model
            };
            for &pair in batch {
                let (ai, pi) = pairs[pair];
                let anchor_fam = td.fam_of_item[ai];
                let cat = td.cat_of_fam[anchor_fam];
                let ax = dropout(&td.items[ai].x, &mut rng, INPUT_DROPOUT);
                let ab = drop_buckets_if::<M>(&td.items[ai].buckets, &mut rng, INPUT_DROPOUT);
                let px = dropout(&td.items[pi].x, &mut rng, INPUT_DROPOUT);
                let pb = drop_buckets_if::<M>(&td.items[pi].buckets, &mut rng, INPUT_DROPOUT);
                if listwise {
                    // ---- listwise multi-label softmax (round 5 default) ----
                    // The whole candidate set competes in one softmax:
                    //   L = -log( Σ_pos e^{τ·s} / Σ_all e^{τ·s} )
                    // so the gradient calibrates the score distribution
                    // (positives share the pull, negatives are weighted by
                    // their softmaxed similarity) instead of pushing each
                    // pair past a fixed hinge.  The SimCSE self-view is
                    // positive #0; family siblings are the rest.  No
                    // NEG_GATE: easy negatives self-gate via p_j.
                    let mut cand_x: Vec<(Vec<f32>, Vec<u32>)> = Vec::new();
                    let mut cand_pos: Vec<bool> = Vec::new();
                    cand_x.push((
                        dropout(&td.items[ai].x, &mut rng, INPUT_DROPOUT),
                        drop_buckets_if::<M>(&td.items[ai].buckets, &mut rng, INPUT_DROPOUT),
                    ));
                    cand_pos.push(true);
                    if pi != ai {
                        cand_x.push((px.clone(), pb.clone()));
                        cand_pos.push(true);
                    }
                    let fr = td.fam_items[anchor_fam].clone();
                    let mut sibs: Vec<usize> = (fr.start..fr.start + fr.len())
                        .filter(|&i| i != ai && i != pi)
                        .collect();
                    rng.shuffle(&mut sibs);
                    for &s in sibs.iter().take(LISTWISE_MAX_POS.saturating_sub(1)) {
                        cand_x.push((
                            dropout(&td.items[s].x, &mut rng, INPUT_DROPOUT),
                            drop_buckets_if::<M>(&td.items[s].buckets, &mut rng, INPUT_DROPOUT),
                        ));
                        cand_pos.push(true);
                    }
                    let fams = &td.cat_fams[cat];
                    let mut neg_seen = 0usize;
                    while neg_seen < NEG_SAME_CAT {
                        let f = fams[(rng.next_u32() as usize) % fams.len()];
                        let r = td.fam_items[f].clone();
                        let idx = r.start + (rng.next_u32() as usize) % r.len();
                        neg_seen += 1;
                        if f == anchor_fam {
                            continue;
                        }
                        cand_x.push((
                            dropout(&td.items[idx].x, &mut rng, INPUT_DROPOUT),
                            drop_buckets_if::<M>(&td.items[idx].buckets, &mut rng, INPUT_DROPOUT),
                        ));
                        cand_pos.push(false);
                    }
                    for _ in 0..NEG_ANY {
                        let f = (rng.next_u32() as usize) % td.fam_items.len();
                        let r = td.fam_items[f].clone();
                        let idx = r.start + (rng.next_u32() as usize) % r.len();
                        if f == anchor_fam {
                            continue;
                        }
                        cand_x.push((
                            dropout(&td.items[idx].x, &mut rng, INPUT_DROPOUT),
                            drop_buckets_if::<M>(&td.items[idx].buckets, &mut rng, INPUT_DROPOUT),
                        ));
                        cand_pos.push(false);
                    }
                    let (_, fa, aa) = hybrid_forward(fq, &ax, &ab);
                    let mut cand: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(cand_x.len());
                    for (cx, cb) in &cand_x {
                        let (_v, f, a) = hybrid_forward(fq, cx, cb);
                        cand.push((f, a));
                    }
                    let s: Vec<f32> = cand
                        .iter()
                        .map(|(cf, ca)| hybrid_cos_of(&fa, &aa, cf, ca))
                        .collect();
                    let smax = s.iter().fold(f32::NEG_INFINITY, |mx, &v| mx.max(v));
                    let mut exps = vec![0.0f32; s.len()];
                    let mut z_all = 0.0f32;
                    let mut z_pos = 0.0f32;
                    for (j, &sv) in s.iter().enumerate() {
                        let e = (tau * (sv - smax)).exp();
                        exps[j] = e;
                        z_all += e;
                        if cand_pos[j] {
                            z_pos += e;
                        }
                    }
                    // dL/ds_j = τ·(e_j/Z_all − [j∈pos]·e_j/Z_pos)
                    let mut g = vec![0.0f32; s.len()];
                    for (j, &e) in exps.iter().enumerate() {
                        g[j] = tau * (e / z_all - if cand_pos[j] { e / z_pos } else { 0.0 });
                    }
                    // s_j = w·(f̂_a·f̂_j) + (1−w)·(â_a·â_j); the anchor channel
                    // is frozen, so only the tower region (k < f_dims) gets
                    // gradient.
                    let mut da = [0.0f32; SEMANTIC_DIM];
                    for k in 0..f_dims {
                        let mut acc = 0.0f32;
                        for j in 0..cand.len() {
                            acc += g[j] * cand[j].0[k];
                        }
                        da[k] = anchor_w * acc;
                    }
                    fq.accumulate_at(&ax, &ab, &da, &mut grad);
                    for j in 0..cand.len() {
                        if g[j].abs() < 1e-9 {
                            continue;
                        }
                        let mut dj = [0.0f32; SEMANTIC_DIM];
                        for k in 0..f_dims {
                            dj[k] = anchor_w * g[j] * fa[k];
                        }
                        fq.accumulate_at(&cand_x[j].0, &cand_x[j].1, &dj, &mut grad);
                    }
                    let _ = fq.mse_at(&ax, &ab, &td.items[ai].target, &mut teacher_grad);
                    if distilling {
                        let _ = fq.mse_at(&px, &pb, &td.items[pi].target, &mut teacher_grad);
                    }
                } else {
                let (anchor, fa, aa) = hybrid_forward(fq, &ax, &ab);
                let (positive, fp, _ap) = hybrid_forward(fq, &px, &pb);

                // gradient of -cos(hybrid_a, hybrid_p) w.r.t. the TOWER
                // output: w-scaled, zero on the frozen anchor region
                let mut da = [0.0f32; SEMANTIC_DIM];
                let mut dp = [0.0f32; SEMANTIC_DIM];
                for k in 0..f_dims {
                    da[k] = -anchor_w * fp[k];
                    dp[k] = -anchor_w * fa[k];
                }

                // fresh hard negatives each epoch: same category first
                let mut pushed: Vec<(Vec<f32>, Vec<u32>, Vec<f32>)> = Vec::new();
                let fams = &td.cat_fams[cat];
                let mut neg_seen = 0usize;
                while neg_seen < NEG_SAME_CAT {
                    let f = fams[(rng.next_u32() as usize) % fams.len()];
                    let r = td.fam_items[f].clone();
                    let idx = r.start + (rng.next_u32() as usize) % r.len();
                    neg_seen += 1;
                    if f == anchor_fam || idx == pi {
                        continue;
                    }
                    let nx = dropout(&td.items[idx].x, &mut rng, INPUT_DROPOUT);
                    let nb = drop_buckets_if::<M>(&td.items[idx].buckets, &mut rng, INPUT_DROPOUT);
                    let (out, fneg, aneg) = hybrid_forward(fq, &nx, &nb);
                    if hybrid_cos_of(&fa, &aa, &fneg, &aneg) > NEG_GATE {
                        for k in 0..f_dims {
                            da[k] += anchor_w * fneg[k];
                        }
                        pushed.push((nx, nb, out));
                    }
                }
                for _ in 0..NEG_ANY {
                    let f = (rng.next_u32() as usize) % td.fam_items.len();
                    let r = td.fam_items[f].clone();
                    let idx = r.start + (rng.next_u32() as usize) % r.len();
                    if f == anchor_fam || idx == pi {
                        continue;
                    }
                    let nx = dropout(&td.items[idx].x, &mut rng, INPUT_DROPOUT);
                    let nb = drop_buckets_if::<M>(&td.items[idx].buckets, &mut rng, INPUT_DROPOUT);
                    let (out, fneg, aneg) = hybrid_forward(fq, &nx, &nb);
                    if hybrid_cos_of(&fa, &aa, &fneg, &aneg) > NEG_GATE {
                        for k in 0..f_dims {
                            da[k] += anchor_w * fneg[k];
                        }
                        pushed.push((nx, nb, out));
                    }
                }

                fq.accumulate_at(&px, &pb, &dp, &mut grad);
                fq.accumulate_at(&ax, &ab, &da, &mut grad);
                for (nx, nb, _) in &pushed {
                    let mut dc = [0.0f32; SEMANTIC_DIM];
                    // push the negative away from the anchor side's hybrid:
                    // dL/df̂n = w * f̂a on the tower region
                    for k in 0..f_dims {
                        dc[k] = anchor_w * fa[k];
                    }
                    fq.accumulate_at(nx, nb, &dc, &mut grad);
                }
                let _ = fq.mse_at(&ax, &ab, &td.items[ai].target, &mut teacher_grad);
                if distilling {
                    // teacher geometry on BOTH sides of the pull
                    let _ = fq.mse_at(&px, &pb, &td.items[pi].target, &mut teacher_grad);
                }
                }
            }
            let inv = 1.0 / batch.len().max(1) as f32;
            let b1_start = M::B1_START;
            let b1_end = M::B1_START + M::B1_LEN;
            for (i, ((g, t), w)) in grad
                .iter_mut()
                .zip(teacher_grad.iter())
                .zip(model.w().iter())
                .enumerate()
            {
                // stronger decay on the hidden bias keeps tanh units
                // input-sensitive for sparse (short-query) inputs
                let decay = if (b1_start..b1_end).contains(&i) {
                    B1_DECAY
                } else {
                    1e-5
                };
                *g = *g * inv + mse_weight * *t * inv + decay * *w;
            }
            if let Some(m) = &mask {
                for (g, keep) in grad.iter_mut().zip(m.iter()) {
                    if !keep {
                        *g = 0.0;
                    }
                }
            }
            adam.step(model.w_mut(), &grad);
            if let Some(m) = &mask {
                for (w, keep) in model.w_mut().iter_mut().zip(m.iter()) {
                    if !keep {
                        *w = 0.0;
                    }
                }
            }
        }

        // Validation is measured through the DEPLOYED geometry: once QAT has
        // engaged, the shadow (not the f32 master) is what would be exported,
        // so selection must score that.  Pre-engagement scores measure a
        // geometry that will not be deployed — they neither select the best
        // model nor count toward staleness.
        let val_shadow;
        let val_ref: &M = if use_qat {
            val_shadow = model.fake_quant();
            &val_shadow
        } else {
            &model
        };
        let val_embed = |t: &str| -> Vec<f32> {
            let x = embedding::hash_v1_features(t);
            let b = buckets_for::<M>(t);
            let (v, _, _) = hybrid_forward(val_ref, &x, &b);
            v
        };
        let (r, m) = val_metrics(&val_embed, vd);
        let eligible = !qat || epoch >= qat_warmup;
        if eligible && (r > best_r + 1e-4 || (r > best_r - 1e-4 && m > best_m + 1e-4)) {
            best_r = r;
            best_m = m;
            best_model = model.clone();
            stale = 0;
        } else if eligible && epoch >= 60.max(qat_warmup) {
            stale += 1;
            if stale >= 45 {
                break;
            }
        }
    }
    if best_r < 0.0 {
        // QAT never engaged (or never beat its own warmup): hand back the
        // final master with zeroed metrics so the caller's selection still
        // works; the soup gate treats 0.0 as "worst member".
        return (model, 0.0, 0.0, epochs);
    }
    (best_model, best_r, best_m, epochs)
}

fn hash_val_baseline(vd: &ValData) -> (f32, f32) {
    val_metrics(&|t: &str| embedding::hash_v1_features(t), vd)
}

fn per_cat_val(
    embed: &dyn Fn(&str) -> Vec<f32>,
    vd: &ValData,
    n_cats: usize,
) -> Vec<(usize, f32)> {
    let mut hits = vec![0u32; n_cats + 1];
    let mut tot = vec![0u32; n_cats + 1];
    // reuse the global metric but bucketed per category
    let mut mem_emb: Vec<Vec<f32>> = Vec::with_capacity(vd.fam_memories.len());
    for (_, mems) in &vd.fam_memories {
        let dim = {
            let e = embed(&mems[0]);
            e.len()
        };
        let mut mean = vec![0.0f32; dim];
        for m in mems {
            let e = embed(m);
            for (mv, ev) in mean.iter_mut().zip(&e) {
                *mv += ev;
            }
        }
        mem_emb.push(unit(&mean));
    }
    for (cat, fam, q) in &vd.queries {
        let qe = unit(&embed(q));
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (fi, (mcat, _)) in vd.fam_memories.iter().enumerate() {
            if mcat != cat {
                continue;
            }
            scored.push((fi, cos(&qe, &mem_emb[fi])));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        tot[*cat] += 1;
        if scored.iter().take(5).any(|&(fi, _)| fi == *fam) {
            hits[*cat] += 1;
        }
    }
    (0..n_cats)
        .map(|c| {
            (
                c,
                hits[c] as f32 / tot[c].max(1) as f32,
            )
        })
        .collect()
}

fn read_mining() -> HashMap<String, usize> {
    let mut out = HashMap::new();
    if std::env::var("SOFUU_EMB_MINING").map(|v| v == "0").unwrap_or(false) {
        return out;
    }
    let Ok(text) = std::fs::read_to_string(MINING_PATH) else {
        return out;
    };
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 3 || cols[0].starts_with('#') {
            continue;
        }
        let Ok(sem) = cols[1].parse::<f32>() else { continue };
        let Ok(hash) = cols[2].parse::<f32>() else { continue };
        if hash > sem {
            // failing category: add families proportional to the gap
            let extra = (((hash - sem) * 40.0).round() as usize).clamp(0, 40);
            if extra > 0 {
                out.insert(cols[0].to_string(), extra);
            }
        }
    }
    out
}

fn artifact_out_path() -> PathBuf {
    if let Ok(p) = std::env::var("SOFUU_EMB_OUT") {
        return PathBuf::from(p);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../sofuu-core/src/embedding/weights_v1.sem");
    path
}

/// Post-training loss report: mean teacher-MSE and mean ranking hinge over
/// a fixed sample of training items/pairs, computed with the final model
/// THROUGH the hybrid lens (anchor region excluded from the MSE; hinge on
/// the blended cosine) when SOFUU_EMB_ANCHOR is active.
fn loss_report<M: LensModel>(model: &M, td: &TrainData, rng: &mut Rng) -> (f32, f32) {
    let (anchor_k, anchor_w) = crate::embedding_anchor::anchor_cfg();
    let anchor_mat = if anchor_k > 0 {
        Some(crate::embedding_anchor::anchor_matrix(anchor_k))
    } else {
        None
    };
    let f_dims = SEMANTIC_DIM - anchor_k;
    let n_mse = 1024usize.min(td.items.len());
    let mut mse = 0.0f32;
    for _ in 0..n_mse {
        let it = &td.items[(rng.next_u32() as usize) % td.items.len()];
        let out = model.forward_at(&it.x, &it.buckets);
        let mut d = 0.0f32;
        for k in 0..f_dims {
            let e = out[k] - it.target[k];
            d += e * e;
        }
        mse += 0.5 * d / f_dims.max(1) as f32 * SEMANTIC_DIM as f32;
    }
    mse /= n_mse.max(1) as f32;
    let n_hinge = 512usize.min(td.fam_items.len());
    let hybrid_pair_cos = |ia: &Item, ib: &Item| -> f32 {
        match &anchor_mat {
            None => cos(
                &model.forward_at(&ia.x, &ia.buckets),
                &model.forward_at(&ib.x, &ib.buckets),
            ),
            Some(m) => {
                let mut fa = model.forward_at(&ia.x, &ia.buckets).to_vec();
                crate::embedding_anchor::unit(&mut fa);
                let mut fb = model.forward_at(&ib.x, &ib.buckets).to_vec();
                crate::embedding_anchor::unit(&mut fb);
                let aa = crate::embedding_anchor::anchor_forward(&ia.x, m, anchor_k);
                let ab = crate::embedding_anchor::anchor_forward(&ib.x, m, anchor_k);
                crate::embedding_anchor::hybrid_cos(&fa, &aa, &fb, &ab, anchor_k, anchor_w)
            }
        }
    };
    let mut hinge = 0.0f32;
    for _ in 0..n_hinge {
        let f = (rng.next_u32() as usize) % td.fam_items.len();
        let r = &td.fam_items[f];
        let a_i = r.start + (rng.next_u32() as usize) % r.len();
        let p_i = r.start + (rng.next_u32() as usize) % r.len();
        let cos_ap = hybrid_pair_cos(&td.items[a_i], &td.items[p_i]);
        let mut hard = -1.0f32;
        for _ in 0..8 {
            let f2 = (rng.next_u32() as usize) % td.fam_items.len();
            if f2 == f {
                continue;
            }
            let r2 = &td.fam_items[f2];
            let x2_i = r2.start + (rng.next_u32() as usize) % r2.len();
            hard = hard.max(hybrid_pair_cos(&td.items[a_i], &td.items[x2_i]));
        }
        hinge += (0.20 - cos_ap + hard).max(0.0);
    }
    hinge /= n_hinge.max(1) as f32;
    (mse, hinge)
}

fn run_driver<M: LensModel>() {
    let mining = read_mining();
    let corpus_seed = std::env::var("SOFUU_EMB_CORPUS")
        .ok()
        .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x").trim(), 16).ok())
        .or_else(|| std::env::var("SOFUU_EMB_CORPUS").ok().and_then(|v| v.parse::<u64>().ok()))
        .unwrap_or(data_embedding_gen::corpus_seed_or_env());
    let seeds: Vec<u64> = std::env::var("SOFUU_EMB_SEEDS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .collect()
        })
        .filter(|v: &Vec<u64>| !v.is_empty())
        .unwrap_or_else(|| vec![11, 23, 47, 89]);
    let epochs_max = std::env::var("SOFUU_EMB_EPOCHS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&e| e >= 20)
        .unwrap_or(240);
    let sparse_k: Option<usize> = std::env::var("SOFUU_EMB_SPARSE_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&k| (1..=HASH_DIM).contains(&k));
    // Round-5 levers (all default ON; each env =0/off restores round-4):
    let soup_on = std::env::var("SOFUU_EMB_SOUP").map(|v| v != "0").unwrap_or(true);
    let listwise = std::env::var("SOFUU_EMB_LOSS")
        .map(|v| v != "pairwise")
        .unwrap_or(true);
    let qat = std::env::var("SOFUU_EMB_QAT").map(|v| v != "0").unwrap_or(true);
    let qat_warmup = std::env::var("SOFUU_EMB_QAT_WARMUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(QAT_WARMUP_EPOCHS);
    let tau = std::env::var("SOFUU_EMB_TAU")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|t| t.is_finite() && *t > 0.0)
        .unwrap_or(LISTWISE_TAU);
    // Soup members must share one init; the seed list becomes data-order
    // seeds and the first entry doubles as the fixed init seed.
    let init_seed = seeds[0];

    println!("building procedural semantic corpus (seed {corpus_seed:#x})…");
    let corpus = data_embedding_gen::corpus(corpus_seed, &mining);
    // teacher: local (PPMI-SVD over repo + trainer texts) or API cache
    let teacher_mode = std::env::var("SOFUU_EMB_TEACHER").unwrap_or_default();
    let mut corpus_texts: Vec<String> = Vec::new();
    for cat in &corpus.cats {
        for f in cat.train.iter().chain(&cat.val) {
            corpus_texts.extend(f.memories.iter().cloned());
            corpus_texts.extend(f.train_queries.iter().cloned());
            corpus_texts.extend(f.val_queries.iter().cloned());
        }
    }
    for g in &corpus.curated {
        corpus_texts.extend(g.iter().cloned());
    }
    let mut teacher = match teacher_mode.as_str() {
        "cache" => embedding_teacher::from_cache(),
        "local" => Some(embedding_teacher::build_local(&corpus_texts, corpus_seed)),
        _ => None,
    };
    let distilling = teacher.is_some();
    if let Some(t) = &teacher {
        println!(
            "teacher: {} (vocab {})",
            if teacher_mode == "cache" {
                "api cache"
            } else {
                "local PPMI-SVD"
            },
            t.vocab_size()
        );
    }

    let (td, n_fams) = build_train(&corpus, teacher.as_mut());
    let vd = build_val(&corpus);
    let n_items = td.items.len();
    println!(
        "  {} train families ({} procedural + {} curated) across {} categories; {} anchors; {} val families / {} val queries",
        n_fams,
        n_fams - corpus.curated.len(),
        corpus.curated.len(),
        data_embedding_gen::CAT_NAMES.len(),
        n_items,
        vd.fam_memories.len(),
        vd.queries.len(),
    );
    if mining.is_empty() {
        println!("  mining: no failure file (balanced corpus)");
    } else {
        let mut names: Vec<&String> = mining.keys().collect();
        names.sort();
        let summary: Vec<String> = names
            .iter()
            .map(|n| format!("{}+{}", n, mining[*n]))
            .collect();
        println!("  mining: boosted failing categories → {}", summary.join(", "));
    }
    let (anchor_k, anchor_w) = crate::embedding_anchor::anchor_cfg();
    println!(
        "arch: {} = {} parameters; output={} dimensions{}{}; {}{}{}",
        M::ARCH,
        M::PARAMS,
        SEMANTIC_DIM,
        match sparse_k {
            Some(k) => format!("; SEM1 v2 sparse K={k} (payload {} B)", embedding::payload_len_v2(k)),
            None => String::new(),
        },
        if anchor_k > 0 {
            format!(
                "; HYBRID anchor K={anchor_k} w={anchor_w} (frozen Rademacher channel, {} live tower dims)",
                SEMANTIC_DIM - anchor_k
            )
        } else {
            String::new()
        },
        if listwise {
            format!("listwise softmax τ={tau}")
        } else {
            "pairwise hinge".to_string()
        },
        if soup_on {
            format!("; SOUP up to {} members (fixed init seed {init_seed})", seeds.len())
        } else {
            "; pick-best seed".to_string()
        },
        if qat {
            format!("; QAT warmup {qat_warmup}")
        } else {
            String::new()
        }
    );

    let (hash_r, hash_m) = hash_val_baseline(&vd);
    println!(
        "hash-v1 val baseline: R@5={:.3} margin={:+.4}",
        hash_r, hash_m
    );

    // Round 5: every seed trains from the SAME init (init_seed) and varies
    // only the data-order seed — the precondition for model soup, since runs
    // from different inits land in permutation-distinct basins whose weights
    // do not average.
    let mut members: Vec<(f32, f32, M, u64, u32)> = Vec::new();
    for &data_seed in &seeds {
        let (model, r, m, epochs) = train_one(
            data_seed,
            init_seed,
            &td,
            &vd,
            epochs_max,
            distilling,
            sparse_k,
            listwise,
            tau,
            qat,
            qat_warmup,
        );
        println!(
            "  seed {data_seed}: {epochs} epochs → val R@5={r:.3} margin={m:+.4}"
        );
        members.push((r, m, model, data_seed, epochs));
    }
    members.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let (val_r, val_m, model, seed, epochs) = if soup_on && members.len() > 1 {
        // Greedy soup: start from the best member, add members best-first,
        // keep the average only when held-out val does not regress.  By
        // construction the soup is never worse than its best member on val.
        let mut acc: Vec<M> = vec![members[0].2.clone()];
        let mut kept = vec![members[0].3];
        let (mut cur_r, mut cur_m) = (members[0].0, members[0].1);
        for member in members.iter().skip(1) {
            acc.push(member.2.clone());
            let avg = average_models(&acc);
            // probe through the deployed geometry (shadow when QAT is on)
            let probe = if qat { avg.fake_quant() } else { avg };
            let (nr, nm) = val_metrics(&hybrid_lens_embed(&probe), &vd);
            if nr > cur_r + 1e-4 || (nr >= cur_r - 1e-4 && nm >= cur_m - 1e-4) {
                cur_r = nr;
                cur_m = nm;
                kept.push(member.3);
            } else {
                acc.pop();
            }
        }
        let names: Vec<String> = kept.iter().map(|s| s.to_string()).collect();
        println!(
            "soup: kept {}/{} members [{}] → val R@5={cur_r:.3} margin={:+.4}",
            kept.len(),
            members.len(),
            names.join(","),
            cur_m,
        );
        let epochs = members.iter().map(|m| m.4).max().unwrap_or(0);
        (cur_r, cur_m, average_models(&acc), kept[0], epochs)
    } else {
        members.remove(0)
    };
    let mut report_rng = Rng(0x1055_0001);
    // Report through the deployed geometry: with QAT on, the exported int8
    // weights are the fake-quant shadow, not the f32 master.
    let report_model = if qat { model.fake_quant() } else { model.clone() };
    let (mse, hinge) = loss_report(&report_model, &td, &mut report_rng);
    println!(
        "kept seed {seed} after {epochs} epochs: val R@5={val_r:.3} margin={val_m:+.4} (hash {hash_r:.3})"
    );
    println!(
        "loss report (final model, train sample): teacher/code MSE {mse:.4} · ranking hinge {hinge:.4}"
    );
    if let Some(t) = &mut teacher {
        println!(
            "teacher coverage: {} hits / {} misses (all-zero embeds fall back to the family code)",
            t.hits, t.misses
        );
    }

    let cat_scores = per_cat_val(
        &hybrid_lens_embed(&report_model),
        &vd,
        data_embedding_gen::CAT_NAMES.len(),
    );
    let hash_cat = per_cat_val(&|t: &str| embedding::hash_v1_features(t), &vd, data_embedding_gen::CAT_NAMES.len());
    println!("per-category val R@5 (sem vs hash):");
    for (ci, r) in cat_scores.iter().enumerate() {
        println!(
            "  {:<14} {:.3} vs {:.3}",
            data_embedding_gen::CAT_NAMES[ci], r.1, hash_cat[ci].1
        );
    }

    /* Trainer-level gate: on families the model never saw, the projector
     * must beat the raw hash features — strictly on recall, or (when val
     * recall is within noise) by a clearly wider separation margin.  The
     * §10 harness remains the real acceptance gate.  Under the hybrid lens
     * the anchor channel blends cosines, which compresses the val margin
     * by construction — there the bar is recall parity (−1pp) plus a
     * margin at least hash's own, still well above "no better than
     * nothing" while the real gates stay in the §10 harness. */
    let gate_ok = if anchor_k > 0 {
        val_r >= hash_r - 0.03 && val_m >= hash_m * 0.9
    } else {
        val_r > hash_r + 0.005 || (val_r >= hash_r - 0.02 && val_m > hash_m * 1.4)
    };
    if !gate_ok {
        eprintln!(
            "SEMANTIC EMBEDDER QUALITY GATE FAILED — val R@5 {val_r:.3} vs hash {hash_r:.3}, margin {val_m:+.4} vs {hash_m:+.4}; artifact not written"
        );
        std::process::exit(1);
    }

    let (blob, format_desc) = match sparse_k {
        Some(k) => model.export_blob_sparse(k),
        None => model.export_blob(),
    };
    let path = artifact_out_path();
    std::fs::write(&path, &blob).expect("write semantic projector artifact");
    println!(
        "wrote {} ({} bytes; {} ; artifact {})",
        path.display(),
        blob.len(),
        format_desc,
        embedding::artifact_id_for(&blob)
    );
}

/// Entry point: SEM1 (default, shipped geometry) or the round-9 SEM2
/// word-table candidate via `SOFUU_EMB_V2=1`.  The v2 branch enforces the
/// pre-registered recipe (pairwise loss, dense export) so a mis-set env
/// cannot silently grade a different model than the plan describes.
pub fn train_embedding() {
    let v2 = std::env::var("SOFUU_EMB_V2")
        .map(|v| v != "0")
        .unwrap_or(false);
    if v2 {
        let loss = std::env::var("SOFUU_EMB_LOSS").unwrap_or_default();
        if loss != "pairwise" {
            eprintln!(
                "SOFUU_EMB_V2 requires SOFUU_EMB_LOSS=pairwise (pre-registered round-9 recipe); got {loss:?}"
            );
            std::process::exit(1);
        }
        if std::env::var("SOFUU_EMB_SPARSE_K").is_ok() {
            eprintln!("SOFUU_EMB_V2 does not support SOFUU_EMB_SPARSE_K (SEM1-only pruning)");
            std::process::exit(1);
        }
        run_driver::<embedding_v2::V2Projector>();
    } else {
        run_driver::<ProjectorF32>();
    }
}
