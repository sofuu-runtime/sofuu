// ml-train/src/embedding_probe.rs — WHY is G1 stuck at 0.750 across rounds 6–9?
//
// Table inspection on a SEM2 artifact: the §10 paraphrase families hinge on
// zero-lexical-overlap synonym pairs (query word ↔ memory word).  For each
// pair this probe answers, from the trained int8 table itself:
//
//   1. Did the training corpus ever SHOW the word?  (token census over the
//      exact train texts: procedural train families + curated groups)
//   2. How crowded is its bucket?  (distinct train tokens colliding there)
//   3. Did the two rows BRIDGE?  (row cosine vs the distribution over all
//      unrelated bucket pairs — percentile, not raw number)
//
// Verdict per pair:
//   ABSENT    — a word never appeared in training → no table size can bridge
//               it (data-bound).
//   BRIDGED   — pair cosine in the top 5% of unrelated pairs → the table
//               learned it; the G1 loss is downstream (tower/fusion).
//   BLURRED   — seen, but bucket load ≥ 30 distinct tokens and unbridged →
//               collision blur (capacity-bound; a bigger table could help).
//   UNLEARNED — seen, low load, unbridged → evidence and room existed but
//               the objective never connected them (training-bound).
//
//   SOFUU_EMBED_EVAL_ARTIFACT=<path to .sem> \
//       cargo run -p ml-train --release -- embed-probe
//
// The artifact path goes through embedding_eval::resolve_artifact_path —
// canonicalized, allow-listed to /tmp and the workspace, same contract as
// embed-eval.  Dev-only.  Reads the artifact; trains nothing; touches no §10
// number.

use std::collections::HashMap;

use crate::data_embedding_gen;
use crate::embedding_eval::resolve_artifact_path;
use crate::embedding_v2::{self, QuantizedV2, V2_D};

/// The synonym bridges §10's paraphrase queries demand (query word → memory
/// word), taken from the eval families verbatim.
const PAIRS: [(&str, &str); 6] = [
    ("codec", "compresses"),
    ("payment", "billing"),
    ("routine", "process"),
    ("sign", "login"),
    ("pulled", "recall"),
    ("toolchain", "msvc"),
];

/// Same rules as embedding_v2::tokenize, but keeps the strings.
fn token_strings(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for &b in text.as_bytes() {
        match b {
            b'A'..=b'Z' => cur.push((b + 32) as char),
            b'a'..=b'z' | b'0'..=b'9' => cur.push(b as char),
            _ => {
                if cur.len() >= 2 {
                    out.push(std::mem::take(&mut cur));
                }
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

fn bucket_of(tok: &str) -> u32 {
    embedding_v2::tokenize(tok)[0]
}

fn row_slice(q: &QuantizedV2, b: usize) -> &[i8] {
    let start = b * V2_D;
    &q.table[start..start + V2_D]
}

fn row_cos(q: &QuantizedV2, a: usize, b: usize) -> f32 {
    let ra = row_slice(q, a);
    let rb = row_slice(q, b);
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..V2_D {
        let (x, y) = (ra[i] as f64, rb[i] as f64);
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        (d / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    }
}

fn row_norm(q: &QuantizedV2, b: usize) -> f32 {
    row_slice(q, b)
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt() as f32
}

/// Percentile (0..=100) of `v` in a sorted distribution.
fn percentile(sorted: &[f32], v: f32) -> f32 {
    let lo = sorted.partition_point(|&x| x < v);
    let hi = sorted.partition_point(|&x| x <= v);
    ((lo + hi) / 2) as f32 * 100.0 / sorted.len() as f32
}

pub fn run_probe() -> i32 {
    if std::env::var("SOFUU_EMBED_EVAL_ARTIFACT").is_err() {
        eprintln!("usage: SOFUU_EMBED_EVAL_ARTIFACT=<path to .sem> ml-train embed-probe");
        eprintln!("(the probe needs a SEM2 candidate; the baked in-tree artifact is SEM1)");
        return 2;
    }
    let path = match resolve_artifact_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("artifact: {e}");
            return 2;
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read artifact: {e}");
            return 2;
        }
    };
    let q = match QuantizedV2::from_blob(&bytes) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("artifact is not a valid SEM2 blob: {e}");
            return 2;
        }
    };
    println!(
        "artifact: {} ({} B, SEM2 table {}×{V2_D})",
        path.display(),
        bytes.len(),
        q.table.len() / V2_D
    );

    // ── train census: the exact texts that put gradients on the table ──
    let corpus =
        data_embedding_gen::corpus(data_embedding_gen::corpus_seed_or_env(), &HashMap::new());
    let mut train: HashMap<String, usize> = HashMap::new();
    let mut bump = |m: &mut HashMap<String, usize>, text: &str| {
        for t in token_strings(text) {
            *m.entry(t).or_insert(0) += 1;
        }
    };
    for cat in &corpus.cats {
        for f in &cat.train {
            for t in f.memories.iter().chain(&f.train_queries) {
                bump(&mut train, t);
            }
        }
    }
    for g in &corpus.curated {
        for t in g {
            bump(&mut train, t);
        }
    }

    // distinct train tokens per bucket (collision load)
    let mut bucket_load: HashMap<u32, usize> = HashMap::new();
    for tok in train.keys() {
        *bucket_load.entry(bucket_of(tok)).or_insert(0) += 1;
    }
    let seen: Vec<u32> = bucket_load.keys().copied().collect();
    println!(
        "train census: {} distinct tokens, {} distinct buckets used (avg load {:.1}, max {})",
        train.len(),
        seen.len(),
        train.len() as f64 / seen.len().max(1) as f64,
        bucket_load.values().max().copied().unwrap_or(0),
    );

    // ── baseline: cosine over every unrelated pair of used buckets ──
    let mut dist: Vec<f32> = Vec::new();
    for i in 0..seen.len() {
        for j in i + 1..seen.len() {
            dist.push(row_cos(&q, seen[i] as usize, seen[j] as usize));
        }
    }
    dist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = dist.iter().sum::<f32>() / dist.len() as f32;
    println!(
        "baseline: {} unrelated row-pair cosines, mean {mean:+.3}, p50 {:+.3}, p95 {:+.3}, p99 {:+.3}",
        dist.len(),
        dist[dist.len() / 2],
        dist[dist.len() * 95 / 100],
        dist[dist.len() * 99 / 100],
    );

    // ── per-pair report ──
    let mut counts: HashMap<&str, usize> = HashMap::new();
    println!(
        "\n{:>10} {:>10}  {:>5} {:>5}  {:>5} {:>5}  {:>6} {:>6}  verdict",
        "query-word", "memory-word", "bktA", "bktB", "nA", "nB", "cos", "pctl"
    );
    for (a, b) in PAIRS {
        let (ba, bb) = (bucket_of(a) as usize, bucket_of(b) as usize);
        let (na, nb) = (
            train.get(a).copied().unwrap_or(0),
            train.get(b).copied().unwrap_or(0),
        );
        let (la, lb) = (
            bucket_load.get(&(ba as u32)).copied().unwrap_or(0),
            bucket_load.get(&(bb as u32)).copied().unwrap_or(0),
        );
        let cos = row_cos(&q, ba, bb);
        let pctl = percentile(&dist, cos);
        let verdict = if na == 0 || nb == 0 {
            "ABSENT"
        } else if pctl >= 95.0 {
            "BRIDGED"
        } else if la.max(lb) >= 30 {
            "BLURRED"
        } else {
            "UNLEARNED"
        };
        *counts.entry(verdict).or_insert(0) += 1;
        println!(
            "{a:>10} {b:>10}  {ba:>5} {bb:>5}  {na:>5} {nb:>5}  {cos:+.3} {pctl:>5.1}%  {verdict}  (load {la}/{lb}, |row| {:.2}/{:.2})",
            row_norm(&q, ba),
            row_norm(&q, bb),
        );
    }

    // ── aggregate verdict ──
    let absent = counts.get("ABSENT").copied().unwrap_or(0);
    let blurred = counts.get("BLURRED").copied().unwrap_or(0);
    let bridged = counts.get("BRIDGED").copied().unwrap_or(0);
    let unlearned = counts.get("UNLEARNED").copied().unwrap_or(0);
    println!("\nsummary: {absent} ABSENT, {blurred} BLURRED, {unlearned} UNLEARNED, {bridged} BRIDGED");
    if absent > 0 {
        println!("→ DATA-BOUND for the ABSENT pairs: the table never saw those words; a budget lift cannot create the evidence.");
    }
    if blurred > 0 && absent == 0 {
        println!("→ CAPACITY-BOUND signal: seen words sit in crowded buckets and stay unbridged; a bigger table is the right next experiment.");
    }
    if unlearned > 0 && blurred == 0 && absent == 0 {
        println!("→ TRAINING-BOUND signal: low-load rows with no bridge — neither size nor data is the blocker; the objective is.");
    }
    if bridged == PAIRS.len() {
        println!("→ Table is NOT the bottleneck: every bridge exists; G1 loss lives in the tower/fusion.");
    }
    0
}
