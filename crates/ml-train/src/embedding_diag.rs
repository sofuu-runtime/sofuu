// ml-train/src/embedding_diag.rs — WHY does the semantic projector lose to
// hash-v1?  Two answers, measured:
//
//   1. Margin evidence: for every query the hash backend retrieves but the
//      semantic candidate misses, compare the query's cosine margin to its
//      target family versus sibling families, once in the RAW 768-dim hash
//      space and once in the candidate's 64-dim space.  A positive input
//      margin on a semantic miss means the signal existed and the trained
//      projector destroyed it; a negative one means the features never saw it.
//
//   2. Anchor floors: a frozen Rademacher random projection of the hash
//      features (zero blob bytes, deterministic code) scored alone at
//      several widths — the retrieval quality ANY hybrid anchor+tower
//      candidate inherits for free.  Hybrid rows blend the frozen anchor
//      with the candidate's own 64-dim output.
//
//   cargo run -p ml-train --release -- embed-diag
//
// Honors SOFUU_EMBED_EVAL_ARTIFACT exactly like embed-eval.  Dev-only.

use std::collections::HashMap;

use sofuu_core::embedding::hash_v1_features;
use sofuu_core::memory::cma::Cma;

use crate::embedding_eval::{build_corpus, load_semantic_model, semantic_embed};

const TOP_K: usize = 5;
const CAT_NAMES: [&str; 8] = [
    "paraphrase", "facts", "documentation", "code", "paths", "errors", "versions", "dates",
];
const CHATTER_TAG: (usize, usize) = (CAT_NAMES.len(), 0);
const RP_SEED: u64 = 0xA11C_E500_0001;

fn unit(v: &mut Vec<f32>) {
    let mut n = 0.0f32;
    for x in v.iter() {
        n += x * x;
    }
    let n = n.sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Frozen Rademacher projection, row-major [768][dim]: entries are +1 or -1
/// scaled by 1/sqrt(dim), drawn from a fixed LCG.  Zero storage in the blob;
/// the same matrix is regenerable from RP_SEED on any host.
fn rp_matrix(dim: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; 768 * dim];
    let inv = 1.0 / (dim as f32).sqrt();
    let mut s = RP_SEED;
    for v in r.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = if (s >> 33) & 1 == 1 { inv } else { -inv };
    }
    r
}

fn rp_forward(x: &[f32], r: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    let mut i = 0;
    while i < x.len() {
        let xi = x[i];
        if xi != 0.0 {
            let row = i * dim;
            let mut j = 0;
            while j < dim {
                out[j] += xi * r[row + j];
                j += 1;
            }
        }
        i += 1;
    }
    unit(&mut out);
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n {
        d += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
        i += 1;
    }
    let den = na.sqrt() * nb.sqrt();
    if den > 1e-12 {
        d / den
    } else {
        0.0
    }
}

struct BackendResult {
    name: String,
    recall: [f32; 8],
    precision: [f32; 8],
    overall: f32,
    /// per-query hit flag, parallel to `queries`
    hits: Vec<bool>,
}

fn score_backend(
    name: &str,
    dim: usize,
    embed: &dyn Fn(&str) -> Vec<f32>,
    records: &[(String, (usize, usize))],
    queries: &[(String, (usize, usize))],
) -> BackendResult {
    let mut cma = Cma::new(dim);
    let mut id_tag: HashMap<u32, (usize, usize)> = HashMap::new();
    for (text, tag) in records {
        let id = cma.remember(&embed(text), text, "note", 0);
        id_tag.insert(id as u32, *tag);
    }
    let mut recall = [0f32; 8];
    let mut precision = [0f32; 8];
    let mut q_per_cat = [0u32; 8];
    let mut hits = Vec::with_capacity(queries.len());
    let mut tgt_all = 0usize;
    for (qtext, qtag) in queries {
        let hits5 = cma.recall(&embed(qtext), TOP_K);
        let mut tgt = 0usize;
        for h in &hits5 {
            let tag = id_tag.get(&h.id).copied().unwrap_or(CHATTER_TAG);
            if tag == *qtag {
                tgt += 1;
            }
        }
        let hit = tgt > 0;
        hits.push(hit);
        if hit {
            recall[qtag.0] += 1.0;
            tgt_all += 1;
        }
        precision[qtag.0] += tgt as f32 / TOP_K as f32;
        q_per_cat[qtag.0] += 1;
    }
    let mut i = 0;
    while i < 8 {
        recall[i] /= q_per_cat[i].max(1) as f32;
        precision[i] /= q_per_cat[i].max(1) as f32;
        i += 1;
    }
    BackendResult {
        name: name.to_string(),
        recall,
        precision,
        overall: tgt_all as f32 / queries.len().max(1) as f32,
        hits,
    }
}

pub fn run_diag() -> i32 {
    println!("SEMANTIC EMBEDDER failure diagnosis (why does the candidate lose to hash-v1?)");
    let sm = match load_semantic_model() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    println!(
        "candidate artifact: {} ({} v{}, {})",
        sm.id, sm.kind, sm.blob_version, sm.path.display()
    );
    let corpus = build_corpus();
    let sem_fn = std::rc::Rc::new(semantic_embed(&sm));

    // hash backend (also feeds the margin evidence)
    let hash_embed = |t: &str| {
        let mut v = hash_v1_features(t);
        unit(&mut v);
        v
    };
    let hash_res = score_backend("hash768", 768, &hash_embed, &corpus.records, &corpus.queries);
    let sem_res = score_backend("sem64", 64, &*sem_fn, &corpus.records, &corpus.queries);
    let hash_hits = hash_res.hits.clone();
    let sem_hits = sem_res.hits.clone();

    // anchor floors at several widths
    let mut results = vec![hash_res, sem_res];
    let mut mats: Vec<(usize, Vec<f32>)> = Vec::new();
    for &k in [16usize, 24, 32, 56].iter() {
        mats.push((k, rp_matrix(k)));
    }
    for (k, r) in mats.iter() {
        let kk = *k;
        let rr = r.clone();
        let embed = move |t: &str| rp_forward(&hash_v1_features(t), &rr, kk);
        results.push(score_backend(
            &format!("rp{k}"),
            *k,
            &embed,
            &corpus.records,
            &corpus.queries,
        ));
    }
    // frozen low-dim tf-idf anchors: the same deterministic embedder at a
    // smaller bucket count — sparse-to-sparse, no random noise
    for &k in [16usize, 24, 32, 48, 56].iter() {
        let embed = move |t: &str| {
            let mut v = sofuu_core::embedding::hash_features_at(t, k);
            unit(&mut v);
            v
        };
        results.push(score_backend(
            &format!("tf{k}"),
            k,
            &embed,
            &corpus.records,
            &corpus.queries,
        ));
    }
    // deterministic content-token anchor floors
    for &k in [16usize, 24, 32].iter() {
        let embed = move |t: &str| crate::embedding_anchor::token_anchor(t, k);
        results.push(score_backend(
            &format!("tok{k}"),
            k,
            &embed,
            &corpus.records,
            &corpus.queries,
        ));
    }
    // hybrids: concat(sqrt(w) * anchor_k, sqrt(1-w) * sem64)
    for &(k, w) in [(24usize, 0.7f32), (32usize, 0.5f32), (32usize, 0.7f32)].iter() {
        let r = rp_matrix(k);
        let rr = r.clone();
        let sf = std::rc::Rc::clone(&sem_fn);
        let embed = move |t: &str| {
            let a = rp_forward(&hash_v1_features(t), &rr, k);
            let mut s = sf(t);
            unit(&mut s);
            let sw = w.sqrt();
            let tw = (1.0 - w).sqrt();
            let mut v = Vec::with_capacity(k + s.len());
            let mut j = 0;
            while j < a.len() {
                v.push(sw * a[j]);
                j += 1;
            }
            for x in s.iter() {
                v.push(tw * x);
            }
            unit(&mut v);
            v
        };
        results.push(score_backend(
            &format!("hybrid k={k} w={w}"),
            k + 64,
            &embed,
            &corpus.records,
            &corpus.queries,
        ));
    }

    println!("\n=== backend floors (same corpus, same top-5 recall path) ===");
    println!(
        "{:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}  {:>7}",
        "backend",
        CAT_NAMES[0],
        CAT_NAMES[1],
        CAT_NAMES[2],
        CAT_NAMES[3],
        CAT_NAMES[4],
        CAT_NAMES[5],
        CAT_NAMES[6],
        CAT_NAMES[7],
        "P@5",
        "OVERALL"
    );
    for b in &results {
        let p_all: f32 = b.precision.iter().sum::<f32>() / 8.0;
        println!(
            "{:<22} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3}  {:>7.3}",
            b.name, b.recall[0], b.recall[1], b.recall[2], b.recall[3], b.recall[4], b.recall[5],
            b.recall[6], b.recall[7], p_all, b.overall,
        );
    }

    // ── margin evidence: hash right, candidate wrong ────────────────────
    // family mean vectors in hash space and candidate space
    let mut fam_hash: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
    let mut fam_sem: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
    let mut fam_n: HashMap<(usize, usize), usize> = HashMap::new();
    for (text, tag) in &corpus.records {
        if tag.0 >= CAT_NAMES.len() {
            continue;
        }
        let e = fam_hash
            .entry(*tag)
            .or_insert_with(|| vec![0.0f32; 768]);
        let h = hash_v1_features(text);
        let mut i = 0;
        while i < 768 {
            e[i] += h[i];
            i += 1;
        }
        let es = fam_sem.entry(*tag).or_insert_with(|| vec![0.0f32; 64]);
        let s = sem_fn(text);
        let mut i = 0;
        while i < 64 && i < s.len() {
            es[i] += s[i];
            i += 1;
        }
        *fam_n.entry(*tag).or_insert(0) += 1;
    }
    for (k, v) in fam_hash.iter_mut() {
        let n = *fam_n.get(k).unwrap_or(&1) as f32;
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    for (k, v) in fam_sem.iter_mut() {
        let n = *fam_n.get(k).unwrap_or(&1) as f32;
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    // sibling means per category for the max-wrong margin
    let margin_row =
        |qi: usize, space: &HashMap<(usize, usize), Vec<f32>>, emb: &dyn Fn(&str) -> Vec<f32>| -> f32 {
            let (qc, qf) = corpus.queries[qi].1;
            let qv = emb(&corpus.queries[qi].0);
            let tgt = cosine(&qv, space.get(&(qc, qf)).unwrap_or(&qv));
            let mut worst = f32::MIN;
            let mut f2 = 0;
            while f2 < corpus.n_fams {
                if f2 != qf {
                    if let Some(m) = space.get(&(qc, f2)) {
                        worst = worst.max(cosine(&qv, m));
                    }
                }
                f2 += 1;
            }
            tgt - worst
        };
    println!("\n=== margin evidence: hash retrieved it, candidate did not ===");
    println!(
        "{:<12} {:<58} {:>9} {:>9}",
        "category", "query", "hashmarg", "semmarg"
    );
    let mut destroyed = [0usize; 8];
    let mut absent = [0usize; 8];
    let mut qi = 0;
    while qi < corpus.queries.len() {
        if hash_hits[qi] && !sem_hits[qi] {
            let (qc, _) = corpus.queries[qi].1;
            let im = margin_row(qi, &fam_hash, &hash_embed);
            let smm = margin_row(qi, &fam_sem, &*sem_fn);
            if im > 0.0 {
                destroyed[qc] += 1;
            } else {
                absent[qc] += 1;
            }
            let short: String = corpus.queries[qi].0.chars().take(56).collect();
            println!(
                "{:<12} {:<58} {:>+9.3} {:>+9.3}",
                CAT_NAMES[qc], short, im, smm
            );
        }
        qi += 1;
    }
    println!("\nper-category miss breakdown (hash right, candidate wrong):");
    let mut c = 0;
    while c < 8 {
        if destroyed[c] + absent[c] > 0 {
            println!(
                "  {:<14} destroyed-by-projector {:2}   signal-absent-in-features {:2}",
                CAT_NAMES[c], destroyed[c], absent[c]
            );
        }
        c += 1;
    }
    0
}
