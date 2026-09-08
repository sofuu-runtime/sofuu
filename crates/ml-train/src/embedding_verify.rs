//! Randomized accuracy verification for the semantic embedder.
//!
//!   cargo run -p ml-train --release -- embed-verify
//!
//! Builds R fresh random corpora (different generator seeds than any
//! training run) and measures retrieval on each, reporting mean ± std so
//! seed noise is visible instead of hidden behind a single number:
//!
//! - **seen-family**: probes are training-style queries of families whose
//!   memories are in the index (in-distribution accuracy);
//! - **unseen-family**: probes belong to held-out families also present in
//!   the index (generalization accuracy, the §10-shaped number);
//! - raw-space separation and per-category breakdown for the unseen probes;
//! - hash-v1 runs the identical protocol as the reference line.
//!
//! Informational by design: exit 0 unless a probe hard-fails.  Pair this
//! with the trainer's loss report (teacher/code MSE + ranking hinge) for
//! the loss picture.
//!
//! Round-6 generalization proof: with `SOFUU_EMBED_EVAL_MODE=twochannel`,
//! every seed also grades the fused two-channel pipeline (sem64 tower +
//! hash768, the shared RRF rule from embedding_eval's `fuse_top5`) so the
//! §10 candidate and the fresh-seed probe cannot drift apart.

use std::collections::HashMap;

use sofuu_core::embedding::{hash_v1_features, HASH_DIM, SEMANTIC_DIM};
use sofuu_core::memory::cma::Cma;

use crate::data_embedding_gen;
use crate::embedding_eval::{FusedStore, load_semantic_model, semantic_embed};

const TOP_K: usize = 5;
const SEEDS: [u64; 8] = [
    0xE1000_0001,
    0xE1000_0002,
    0xE1000_0003,
    0xE1000_0004,
    0xE1000_0005,
    0xE1000_0006,
    0xE1000_0007,
    0xE1000_0008,
];
const VAL_TAG_OFF: usize = 1_000;
const CURATED_CAT: usize = data_embedding_gen::CAT_NAMES.len();
const CHATTER_CAT: usize = CURATED_CAT + 1;

type Tag = (usize, usize);

struct ProbeResult {
    recall: f32,
    precision: f32,
    per_cat: Vec<f32>,
    sep_same: f64,
    sep_cross: f64,
}

fn run_probe(
    dim: usize,
    embed: &dyn Fn(&str) -> Vec<f32>,
    records: &[(String, Tag)],
    probes: &[(String, Tag)],
) -> ProbeResult {
    let mut cma = Cma::new(dim);
    let mut id_tag: HashMap<u32, Tag> = HashMap::new();
    for (text, tag) in records {
        let id = cma.remember(&embed(text), text, "note", 0);
        id_tag.insert(id as u32, *tag);
    }
    let mut hits = 0usize;
    let mut tgt_slots = 0u32;
    let mut tot_slots = 0u32;
    let mut per_cat_hits = vec![0u32; CURATED_CAT];
    let mut per_cat_tot = vec![0u32; CURATED_CAT];
    // separation accumulators
    let mut same = 0f64;
    let mut same_n = 0f64;
    let mut cross = 0f64;
    let mut cross_n = 0f64;
    for (q, tag) in probes {
        let qv = embed(q);
        let hits5 = cma.recall(&qv, TOP_K);
        let mut hit = false;
        let mut tgt = 0u32;
        for h in &hits5 {
            if id_tag.get(&h.id) == Some(tag) {
                hit = true;
                tgt += 1;
            }
        }
        if hit {
            hits += 1;
        }
        tgt_slots += tgt;
        tot_slots += TOP_K as u32;
        if tag.0 < CURATED_CAT {
            per_cat_tot[tag.0] += 1;
            if hit {
                per_cat_hits[tag.0] += 1;
            }
        }
        // separation over a deterministic subsample (first 60 records)
        for (text, rtag) in records.iter().take(60) {
            let tv = embed(text);
            let dot: f32 = qv.iter().zip(tv.iter()).map(|(a, b)| a * b).sum();
            if rtag == tag {
                same += dot as f64;
                same_n += 1.0;
            } else if rtag.0 == tag.0 {
                cross += dot as f64;
                cross_n += 1.0;
            }
        }
    }
    ProbeResult {
        recall: hits as f32 / probes.len().max(1) as f32,
        precision: tgt_slots as f32 / tot_slots.max(1) as f32,
        per_cat: per_cat_hits
            .iter()
            .zip(&per_cat_tot)
            .map(|(h, t)| *h as f32 / (*t).max(1) as f32)
            .collect(),
        sep_same: same / same_n.max(1.0),
        sep_cross: cross / cross_n.max(1.0),
    }
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    let n = xs.len().max(1) as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Round-6 fused-pipeline probe: identical protocol to `run_probe`, but the
/// top-5 comes from the shared `FusedStore` (sem64 tower + hash768, each
/// recalled to FUSE_TOP_N, fused by the pre-registered RRF rule) — the same
/// store the §10 harness grades, so the two cannot drift apart.
/// Separation is left 0.0: a single-space cosine gap is undefined for a
/// two-space pipeline.
fn run_probe_fused(
    sem_fn: &dyn Fn(&str) -> Vec<f32>,
    hash_fn: &dyn Fn(&str) -> Vec<f32>,
    records: &[(String, Tag)],
    probes: &[(String, Tag)],
) -> ProbeResult {
    let mut store = FusedStore::new(sem_fn, hash_fn, records);
    let mut hits = 0usize;
    let mut tgt_slots = 0u32;
    let mut tot_slots = 0u32;
    let mut per_cat_hits = vec![0u32; CURATED_CAT];
    let mut per_cat_tot = vec![0u32; CURATED_CAT];
    for (q, tag) in probes {
        let fused = store.top5(&sem_fn(q), &hash_fn(q));
        let mut hit = false;
        let mut tgt = 0u32;
        for id in &fused {
            if store.idx_tags().get(id) == Some(tag) {
                hit = true;
                tgt += 1;
            }
        }
        if hit {
            hits += 1;
        }
        tgt_slots += tgt;
        tot_slots += TOP_K as u32;
        if tag.0 < CURATED_CAT {
            per_cat_tot[tag.0] += 1;
            if hit {
                per_cat_hits[tag.0] += 1;
            }
        }
    }
    ProbeResult {
        recall: hits as f32 / probes.len().max(1) as f32,
        precision: tgt_slots as f32 / tot_slots.max(1) as f32,
        per_cat: per_cat_hits
            .iter()
            .zip(&per_cat_tot)
            .map(|(h, t)| *h as f32 / (*t).max(1) as f32)
            .collect(),
        sep_same: 0.0,
        sep_cross: 0.0,
    }
}

pub fn run_verify() -> i32 {
    println!("SEMANTIC EMBEDDER randomized accuracy verification");
    let sm = match load_semantic_model() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    println!(
        "artifact: {} ({} v{}, {})",
        sm.id, sm.kind, sm.blob_version, sm.path.display()
    );
    let sem_fn = semantic_embed(&sm);
    let twochannel = std::env::var("SOFUU_EMBED_EVAL_MODE").as_deref() == Ok("twochannel");
    if twochannel {
        println!("mode: twochannel — fused pipeline (S sem64 + H hash768, shared RRF rule) also graded");
    }

    let mut sem_seen = Vec::new();
    let mut sem_unseen = Vec::new();
    let mut hash_seen = Vec::new();
    let mut hash_unseen = Vec::new();
    let mut sem_prec_u = Vec::new();
    let mut per_cat_acc: Vec<Vec<f32>> = vec![Vec::new(); CURATED_CAT];
    let mut sep = (0.0f64, 0.0f64, 0usize);
    let mut fused_seen = Vec::new();
    let mut fused_unseen = Vec::new();
    let mut fused_prec_u = Vec::new();
    let mut per_cat_fused: Vec<Vec<f32>> = vec![Vec::new(); CURATED_CAT];

    for &seed in &SEEDS {
        let corpus = data_embedding_gen::corpus(seed, &HashMap::new());
        let mut records: Vec<(String, Tag)> = Vec::new();
        let mut probes_seen: Vec<(String, Tag)> = Vec::new();
        let mut probes_unseen: Vec<(String, Tag)> = Vec::new();
        for (ci, cat) in corpus.cats.iter().enumerate() {
            for (fi, f) in cat.train.iter().enumerate() {
                for m in &f.memories {
                    records.push((m.clone(), (ci, fi)));
                }
                for q in &f.train_queries {
                    probes_seen.push((q.clone(), (ci, fi)));
                }
            }
            for (fi, f) in cat.val.iter().enumerate() {
                for m in &f.memories {
                    records.push((m.clone(), (ci, VAL_TAG_OFF + fi)));
                }
                for q in &f.val_queries {
                    probes_unseen.push((q.clone(), (ci, VAL_TAG_OFF + fi)));
                }
            }
        }
        for (gi, g) in corpus.curated.iter().enumerate() {
            for t in g {
                records.push((t.clone(), (CURATED_CAT, gi)));
            }
        }
        for t in &corpus.chatter {
            records.push((t.clone(), (CHATTER_CAT, 0)));
        }

        let s_seen = run_probe(SEMANTIC_DIM, &sem_fn, &records, &probes_seen);
        let s_unseen = run_probe(SEMANTIC_DIM, &sem_fn, &records, &probes_unseen);
        let h_seen = run_probe(HASH_DIM, &hash_v1_features, &records, &probes_seen);
        let h_unseen = run_probe(HASH_DIM, &hash_v1_features, &records, &probes_unseen);
        println!(
            "seed {seed:#x}: seen sem {:.3}/hash {:.3} · unseen sem {:.3}/hash {:.3}",
            s_seen.recall, h_seen.recall, s_unseen.recall, h_unseen.recall
        );
        if twochannel {
            let f_seen = run_probe_fused(&sem_fn, &hash_v1_features, &records, &probes_seen);
            let f_unseen = run_probe_fused(&sem_fn, &hash_v1_features, &records, &probes_unseen);
            println!(
                "         fused: seen {:.3} · unseen {:.3} (P@5 {:.3})",
                f_seen.recall, f_unseen.recall, f_unseen.precision
            );
            fused_seen.push(f_seen.recall);
            fused_unseen.push(f_unseen.recall);
            fused_prec_u.push(f_unseen.precision);
            for c in 0..CURATED_CAT {
                per_cat_fused[c].push(f_unseen.per_cat[c]);
            }
        }
        sem_seen.push(s_seen.recall);
        sem_unseen.push(s_unseen.recall);
        hash_seen.push(h_seen.recall);
        hash_unseen.push(h_unseen.recall);
        sem_prec_u.push(s_unseen.precision);
        for c in 0..CURATED_CAT {
            per_cat_acc[c].push(s_unseen.per_cat[c]);
        }
        sep.0 += s_unseen.sep_same - s_unseen.sep_cross;
        sep.1 += h_unseen.sep_same - h_unseen.sep_cross;
        sep.2 += 1;
    }

    let (sm_s, sm_s_sd) = mean_std(&sem_seen);
    let (sm_u, sm_u_sd) = mean_std(&sem_unseen);
    let (h_s, h_s_sd) = mean_std(&hash_seen);
    let (h_u, h_u_sd) = mean_std(&hash_unseen);
    let (sp, sp_sd) = mean_std(&sem_prec_u);
    println!("\nrandomized accuracy over {} corpora:", SEEDS.len());
    println!(
        "  seen families   (in-distribution): sem R@5 {:.3} ±{:.3}   hash {:.3} ±{:.3}",
        sm_s, sm_s_sd, h_s, h_s_sd
    );
    println!(
        "  unseen families (generalization):  sem R@5 {:.3} ±{:.3}   hash {:.3} ±{:.3}",
        sm_u, sm_u_sd, h_u, h_u_sd
    );
    println!(
        "  unseen precision@5: sem {:.3} ±{:.3}   separation gap: sem {:+.3} vs hash {:+.3}",
        sp,
        sp_sd,
        sep.0 / sep.2.max(1) as f64,
        sep.1 / sep.2.max(1) as f64
    );
    println!("  unseen per-category (sem):");
    for (c, accs) in per_cat_acc.iter().enumerate() {
        let (m, sd) = mean_std(accs);
        println!("    {:<14} {:.3} ±{:.3}", data_embedding_gen::CAT_NAMES[c], m, sd);
    }
    if twochannel {
        let (f_s, f_s_sd) = mean_std(&fused_seen);
        let (f_u, f_u_sd) = mean_std(&fused_unseen);
        let (fp, fp_sd) = mean_std(&fused_prec_u);
        println!(
            "  fused two-channel (RRF): seen R@5 {:.3} ±{:.3}   unseen {:.3} ±{:.3}   unseen P@5 {:.3} ±{:.3}",
            f_s, f_s_sd, f_u, f_u_sd, fp, fp_sd
        );
        println!("  unseen per-category (fused):");
        for (c, accs) in per_cat_fused.iter().enumerate() {
            let (m, sd) = mean_std(accs);
            println!("    {:<14} {:.3} ±{:.3}", data_embedding_gen::CAT_NAMES[c], m, sd);
        }
    }
    println!("\nloss picture: see the trainer's loss report line (teacher/code MSE + ranking hinge) — accuracy here, optimization loss there.");
    0
}
