// crates/ml-train/src/embedding_bench.rs — the Q-phase public scoreboard.
//
// Everything else in this crate evaluates a MODEL ON OUR OWN SYNTHETIC
// CORPUS (8 categories x 6 families, procedural image scenes). That is
// the right instrument for training and gating, and the wrong thing to
// publish: it can only ever report "we beat our own baselines". This
// module is the external half — standard public sets, standard metrics —
// so the README numbers mean something to anyone who has not read our
// training code.
//
// Two independent measurements, no network access, no API keys:
//
//   * RETRIEVAL — BEIR SciFact (5,183 abstracts, 300 test queries with
//     339 relevance judgements). R@1/R@5/R@10, nDCG@10, MRR@10 against
//     the official qrels.
//   * SIMILARITY — STS benchmark test split (1,379 human-rated sentence
//     pairs). Spearman rho between cosine similarity and the gold score.
//
// Plus the numbers a footprint claim needs: artifact bytes, parameter
// count, median single-embed latency and batch-1k throughput, measured
// in this same process on the same machine that builds the binary.
//
// HONESTY RULE (same as the rest of the plan): these spaces are 64–768
// dim, offline-first and 7–30 KB. They are NOT general-purpose
// retrievers and will lose to a hosted 1536-dim model on SciFact. The
// scoreboard's job is to state the gap precisely, not to hide it — the
// value proposition is bytes and zero network, and the number that must
// hold up is "is it useful at all", not "does it beat OpenAI".
//
// Datasets: scripts/bench/fetch_datasets.sh → $SOFUU_BENCH_DIR
// (default /tmp/sofuu_bench). Missing data is reported, never faked.

use std::collections::HashMap;
use std::time::Instant;

use serde_json::Value;

// ── spaces under test ──────────────────────────────────────────────

/// One embedding space: how to embed, plus the identity it reports.
struct Space {
    id: &'static str,
    dim: usize,
    embed: fn(&str) -> Option<Vec<f32>>,
    info: fn() -> String,
    /// The baked artifact, measured on disk. A space with no artifact
    /// (hash-v1 is pure arithmetic) reports 0 — which is the point.
    artifact: Option<&'static str>,
}

/// Size of a baked artifact, in bytes (0 when the space has none).
fn artifact_size(rel: Option<&str>) -> u64 {
    let Some(rel) = rel else { return 0 };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sofuu-core/src/embedding")
        .join(rel);
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn spaces() -> Vec<Space> {
    vec![
        Space {
            id: "hash-v1",
            dim: sofuu_core::embedding::HASH_DIM,
            embed: |t| Some(sofuu_core::embedding::hash_v1_features(t)),
            info: || "{}".to_string(),
            artifact: None,
        },
        Space {
            id: "sem1-64",
            dim: sofuu_core::embedding::SEMANTIC_DIM,
            embed: sofuu_core::embedding::semantic_v1,
            info: sofuu_core::embedding::model_info_json,
            artifact: Some("weights_v1.sem"),
        },
        Space {
            id: "sem2-64",
            dim: sofuu_core::embedding::SEMANTIC_DIM,
            embed: sofuu_core::embedding::semantic_v2::semantic_v2,
            info: sofuu_core::embedding::semantic_v2::model_info_json_v2,
            artifact: Some("weights_v2.sem"),
        },
    ]
}

fn bench_dir() -> String {
    std::env::var("SOFUU_BENCH_DIR").unwrap_or_else(|_| "/tmp/sofuu_bench".to_string())
}

fn read_jsonl(path: &std::path::Path) -> Vec<Value> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect()
}

// ── vector helpers ─────────────────────────────────────────────────

/// L2-normalize defensively: the spaces already emit unit vectors, but a
/// scoreboard must not silently reward an implementation that does not.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── retrieval metrics ──────────────────────────────────────────────

struct RetrievalScores {
    queries_scored: usize,
    r1: f64,
    r5: f64,
    r10: f64,
    ndcg10: f64,
    mrr10: f64,
}

/// Score every query against the whole corpus and read the ranking.
/// `qrels: query id -> (doc id -> graded relevance)`.
fn score_retrieval(
    corpus: &[(String, Vec<f32>)],
    queries: &[(String, Vec<f32>)],
    qrels: &HashMap<String, HashMap<String, f64>>,
) -> RetrievalScores {
    let mut r1 = 0usize;
    let mut r5 = 0usize;
    let mut r10 = 0usize;
    let mut ndcg = 0.0f64;
    let mut mrr = 0.0f64;
    let mut scored = 0usize;

    for (qid, qvec) in queries {
        let Some(gold) = qrels.get(qid) else { continue };
        if gold.is_empty() {
            continue;
        }
        // Full ranking by cosine (vectors are unit → dot).
        let mut ranked: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i, dot(qvec, v)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored += 1;

        let relevance_at = |rank: usize| -> f64 {
            let (doc_id, _) = &corpus[ranked[rank].0];
            *gold.get(doc_id).unwrap_or(&0.0)
        };

        if relevance_at(0) > 0.0 {
            r1 += 1;
        }
        if (0..5).any(|r| relevance_at(r) > 0.0) {
            r5 += 1;
        }
        if (0..10).any(|r| relevance_at(r) > 0.0) {
            r10 += 1;
        }
        // MRR@10: reciprocal rank of the first relevant document.
        if let Some(rank) = (0..10).find(|&r| relevance_at(r) > 0.0) {
            mrr += 1.0 / (rank + 1) as f64;
        }
        // nDCG@10 with the standard 2^rel - 1 gain, binary judgements in
        // SciFact's qrels (rel = 1) so DCG == sum of 1/log2(rank + 2).
        let mut dcg = 0.0f64;
        for (rank, _) in ranked.iter().take(10).enumerate() {
            let rel = relevance_at(rank);
            if rel > 0.0 {
                dcg += (2.0f64.powf(rel) - 1.0) / ((rank + 2) as f64).log2();
            }
        }
        let ideal: f64 = {
            let mut rels: Vec<f64> = gold.values().copied().filter(|r| *r > 0.0).collect();
            rels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            rels.iter()
                .take(10)
                .enumerate()
                .map(|(i, r)| (2.0f64.powf(*r) - 1.0) / ((i + 2) as f64).log2())
                .sum()
        };
        if ideal > 0.0 {
            ndcg += dcg / ideal;
        }
    }

    let n = scored.max(1) as f64;
    RetrievalScores {
        queries_scored: scored,
        r1: r1 as f64 / n,
        r5: r5 as f64 / n,
        r10: r10 as f64 / n,
        ndcg10: ndcg / n,
        mrr10: mrr / n,
    }
}

// ── STS metrics ────────────────────────────────────────────────────

/// Average ranks (ties share a rank) — the standard tie handling.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![0.0; values.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len()
            && (values[idx[j]] - values[idx[i]]).abs() < f64::EPSILON
        {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0; // ranks are 1-based
        for &k in &idx[i..j] {
            out[k] = avg;
        }
        i = j;
    }
    out
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for i in 0..xs.len() {
        let a = xs[i] - mx;
        let b = ys[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    if dx <= 0.0 || dy <= 0.0 {
        return f64::NAN;
    }
    num / (dx.sqrt() * dy.sqrt())
}

fn spearman(pred: &[f64], gold: &[f64]) -> f64 {
    pearson(&ranks(pred), &ranks(gold))
}

// ── footprint + speed ──────────────────────────────────────────────

fn artifact_bytes(info: &str) -> Option<u64> {
    let v: Value = serde_json::from_str(info).ok()?;
    for key in ["bytes", "artifactBytes", "size"] {
        if let Some(n) = v.get(key).and_then(|x| x.as_u64()) {
            return Some(n);
        }
    }
    None
}

fn param_count(info: &str) -> Option<u64> {
    let v: Value = serde_json::from_str(info).ok()?;
    for key in ["params", "parameters", "paramCount"] {
        if let Some(n) = v.get(key).and_then(|x| x.as_u64()) {
            return Some(n);
        }
    }
    // A table-based space reports dims x rows instead of a scalar.
    let d = v.get("dimension").or_else(|| v.get("dim")).and_then(|x| x.as_u64())?;
    let r = v.get("rows").and_then(|x| x.as_u64())?;
    Some(d * r)
}

/// Median wall-clock for one embed (ms). The median, not the mean: a
/// single scheduler hiccup should not move a published number.
fn median_single_ms(embed: fn(&str) -> Option<Vec<f32>>, samples: &[String]) -> f64 {
    let mut times: Vec<f64> = Vec::with_capacity(samples.len());
    for text in samples {
        let t0 = Instant::now();
        let _ = embed(text);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if times.is_empty() {
        return f64::NAN;
    }
    times[times.len() / 2]
}

fn batch_ms_per_1k(embed: fn(&str) -> Option<Vec<f32>>, samples: &[String]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let t0 = Instant::now();
    let mut acc = 0usize;
    for text in samples {
        if let Some(v) = embed(text) {
            acc += v.len();
        }
    }
    let per = t0.elapsed().as_secs_f64() * 1000.0 / samples.len() as f64;
    let _ = acc;
    per * 1000.0
}

// ── BM25 reference ─────────────────────────────────────────────────

/// Classic Okapi BM25 over the same corpus, so the scoreboard can say
/// whether hash-v1's number is actually competitive with real lexical
/// search or merely respectable in isolation. Defaults k1=1.2, b=0.75.
struct Bm25 {
    /// term -> (doc frequency, total frequency)
    postings: HashMap<String, Vec<(usize, u32)>>,
    doc_len: Vec<u32>,
    avg_len: f64,
    n_docs: usize,
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn bm25_build(docs: &[String]) -> Bm25 {
    let mut postings: HashMap<String, Vec<(usize, u32)>> = HashMap::new();
    let mut doc_len = Vec::with_capacity(docs.len());
    for (i, text) in docs.iter().enumerate() {
        let toks = tokenize(text);
        doc_len.push(toks.len() as u32);
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for t in &toks {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        for (t, c) in counts {
            postings.entry(t.to_string()).or_default().push((i, c));
        }
    }
    let total: u32 = doc_len.iter().sum();
    Bm25 {
        postings,
        avg_len: if docs.is_empty() {
            0.0
        } else {
            total as f64 / docs.len() as f64
        },
        n_docs: docs.len(),
        doc_len,
    }
}

impl Bm25 {
    fn score_all(&self, query: &str) -> Vec<f64> {
        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        let mut scores = vec![0.0f64; self.n_docs];
        for term in tokenize(query) {
            let Some(posting) = self.postings.get(&term) else { continue };
            let df = posting.len() as f64;
            // Standard BM25 idf with the +1 that keeps it non-negative.
            let idf = ((self.n_docs as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();
            for (doc, tf) in posting {
                let tf = *tf as f64;
                let len = self.doc_len[*doc] as f64;
                let denom = tf + K1 * (1.0 - B + B * (len / self.avg_len.max(1.0)));
                scores[*doc] += idf * (tf * (K1 + 1.0)) / denom.max(f64::EPSILON);
            }
        }
        scores
    }
}

/// Rank `query_ids` for every judged query, then reuse the same metric
/// code as the dense spaces by turning BM25 scores into a fake ranking.
fn score_bm25(
    bm25: &Bm25,
    queries: &[(String, String)],
    corpus: &[(String, String)],
    qrels: &HashMap<String, HashMap<String, f64>>,
) -> RetrievalScores {
    // Represent BM25 as a pseudo-space: one "dimension" per document.
    let pseudo: Vec<(String, Vec<f32>)> = corpus
        .iter()
        .map(|(id, _)| (id.clone(), Vec::new()))
        .collect();
    let qvecs: Vec<(String, Vec<f32>)> = queries
        .iter()
        .map(|(id, _)| (id.clone(), Vec::new()))
        .collect();
    // Score by BM25 directly, then reuse metric helpers through a closure.
    let mut r1 = 0usize;
    let mut r5 = 0usize;
    let mut r10 = 0usize;
    let mut ndcg = 0.0f64;
    let mut mrr = 0.0f64;
    let mut scored = 0usize;
    for (qid, qtext) in queries {
        let Some(gold) = qrels.get(qid) else { continue };
        if gold.is_empty() {
            continue;
        }
        let scores = bm25.score_all(qtext);
        let mut ranked: Vec<usize> = (0..scores.len()).collect();
        ranked.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored += 1;
        let relevance_at = |rank: usize| -> f64 {
            let doc_id = &pseudo[ranked[rank]].0;
            *gold.get(doc_id).unwrap_or(&0.0)
        };
        if relevance_at(0) > 0.0 {
            r1 += 1;
        }
        if (0..5).any(|r| relevance_at(r) > 0.0) {
            r5 += 1;
        }
        if (0..10).any(|r| relevance_at(r) > 0.0) {
            r10 += 1;
        }
        if let Some(rank) = (0..10).find(|&r| relevance_at(r) > 0.0) {
            mrr += 1.0 / (rank + 1) as f64;
        }
        let mut dcg = 0.0f64;
        for (rank, _) in ranked.iter().take(10).enumerate() {
            let rel = relevance_at(rank);
            if rel > 0.0 {
                dcg += (2.0f64.powf(rel) - 1.0) / ((rank + 2) as f64).log2();
            }
        }
        let ideal: f64 = {
            let mut rels: Vec<f64> = gold.values().copied().filter(|r| *r > 0.0).collect();
            rels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            rels.iter()
                .take(10)
                .enumerate()
                .map(|(i, r)| (2.0f64.powf(*r) - 1.0) / ((i + 2) as f64).log2())
                .sum()
        };
        if ideal > 0.0 {
            ndcg += dcg / ideal;
        }
    }
    let n = scored.max(1) as f64;
    let _ = qvecs;
    RetrievalScores {
        queries_scored: scored,
        r1: r1 as f64 / n,
        r5: r5 as f64 / n,
        r10: r10 as f64 / n,
        ndcg10: ndcg / n,
        mrr10: mrr / n,
    }
}

// ── entry point ────────────────────────────────────────────────────

pub fn run() {
    let dir = std::path::PathBuf::from(bench_dir());
    println!("Sofuu embedding benchmark (Q phase)");
    println!("datasets: {}\n", dir.display());

    let corpus_rows = read_jsonl(&dir.join("scifact-corpus.jsonl"));
    let query_rows = read_jsonl(&dir.join("scifact-queries.jsonl"));
    let sts_rows = read_jsonl(&dir.join("sts-test.jsonl"));
    let qrels_raw = std::fs::read_to_string(dir.join("scifact-qrels.tsv")).unwrap_or_default();

    let have_retrieval = !corpus_rows.is_empty() && !query_rows.is_empty() && !qrels_raw.is_empty();
    let have_sts = !sts_rows.is_empty();
    if !have_retrieval && !have_sts {
        println!("No datasets found. Run scripts/bench/fetch_datasets.sh first.");
        return;
    }

    // qrels: TSV, header line, query-id \t corpus-id \t score
    let mut qrels: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (i, line) in qrels_raw.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let (Some(q), Some(d), Some(s)) = (cols.next(), cols.next(), cols.next()) else {
            continue;
        };
        let Ok(score) = s.trim().parse::<f64>() else { continue };
        qrels
            .entry(q.trim().to_string())
            .or_default()
            .insert(d.trim().to_string(), score);
    }

    // Document text = title + abstract (the BEIR convention).
    let corpus: Vec<(String, String)> = corpus_rows
        .iter()
        .filter_map(|r| {
            let id = r.get("_id")?.as_str()?.to_string();
            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or("");
            let text = r.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let joined = format!("{title} {text}").trim().to_string();
            Some((id, joined))
        })
        .collect();
    let queries: Vec<(String, String)> = query_rows
        .iter()
        .filter_map(|r| {
            let id = r.get("_id")?.as_str()?.to_string();
            let text = r.get("text").and_then(|v| v.as_str()).unwrap_or("");
            Some((id, text.to_string()))
        })
        .collect();
    // Keep only judged queries — an unjudged query would score as a miss.
    let judged: Vec<(String, String)> = queries
        .into_iter()
        .filter(|(id, _)| qrels.contains_key(id))
        .collect();

    if have_retrieval {
        println!(
            "SciFact: {} docs · {} judged queries · {} qrel pairs",
            corpus.len(),
            judged.len(),
            qrels.values().map(|m| m.len()).sum::<usize>()
        );
    }
    if have_sts {
        println!("STS test: {} pairs\n", sts_rows.len());
    }

    let mut results: Vec<Value> = Vec::new();

    // BM25 first: it is the reference every other row is read against.
    if have_retrieval {
        let texts: Vec<String> = corpus.iter().map(|(_, t)| t.clone()).collect();
        let bm25 = bm25_build(&texts);
        let r = score_bm25(&bm25, &judged, &corpus, &qrels);
        let mut obj = serde_json::Map::new();
        obj.insert("space".into(), Value::String("bm25".into()));
        obj.insert("dim".into(), Value::Null);
        obj.insert("params".into(), Value::Null);
        obj.insert("artifactBytes".into(), Value::from(0));
        obj.insert("queriesScored".into(), Value::from(r.queries_scored));
        obj.insert("R@1".into(), json_f(r.r1));
        obj.insert("R@5".into(), json_f(r.r5));
        obj.insert("R@10".into(), json_f(r.r10));
        obj.insert("nDCG@10".into(), json_f(r.ndcg10));
        obj.insert("MRR@10".into(), json_f(r.mrr10));
        let value = Value::Object(obj);
        results.push(value.clone());
        print_row(&value, true, false);
    }

    for space in spaces() {
        let info = (space.info)();
        let row = |k: &str, v: Value| {
            let mut m = serde_json::Map::new();
            m.insert(k.to_string(), v);
            Value::Object(m)
        };
        let _ = row;

        // Embed everything this space can.
        let mut corpus_vecs: Vec<(String, Vec<f32>)> = Vec::new();
        let mut query_vecs: Vec<(String, Vec<f32>)> = Vec::new();
        let mut sts_pred: Vec<f64> = Vec::new();
        let mut sts_gold: Vec<f64> = Vec::new();

        if have_retrieval {
            for (id, text) in &corpus {
                if let Some(mut v) = (space.embed)(text) {
                    normalize(&mut v);
                    corpus_vecs.push((id.clone(), v));
                }
            }
            for (id, text) in &judged {
                if let Some(mut v) = (space.embed)(text) {
                    normalize(&mut v);
                    query_vecs.push((id.clone(), v));
                }
            }
        }
        if have_sts {
            for r in &sts_rows {
                let (Some(s1), Some(s2)) = (
                    r.get("sentence1").and_then(|v| v.as_str()),
                    r.get("sentence2").and_then(|v| v.as_str()),
                ) else {
                    continue;
                };
                let Some(gold) = r.get("score").and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
                }) else {
                    continue;
                };
                let (Some(a), Some(b)) = ((space.embed)(s1), (space.embed)(s2)) else {
                    continue;
                };
                sts_pred.push(dot(&a, &b) as f64);
                sts_gold.push(gold);
            }
        }

        let retrieval = if have_retrieval {
            Some(score_retrieval(&corpus_vecs, &query_vecs, &qrels))
        } else {
            None
        };
        let rho = if have_sts && sts_pred.len() > 1 {
            spearman(&sts_pred, &sts_gold)
        } else {
            f64::NAN
        };

        // Speed on a fixed, modest sample (fast, reproducible, honest).
        let speed_samples: Vec<String> = corpus
            .iter()
            .take(200)
            .map(|(_, t)| t.chars().take(400).collect())
            .collect();
        let single_ms = median_single_ms(space.embed, &speed_samples);
        let batch_1k = batch_ms_per_1k(space.embed, &speed_samples);

        let mut obj = serde_json::Map::new();
        obj.insert("space".into(), Value::String(space.id.into()));
        obj.insert("dim".into(), Value::from(space.dim));
        obj.insert(
            "params".into(),
            param_count(&info).map(Value::from).unwrap_or(Value::Null),
        );
        obj.insert(
            "artifactBytes".into(),
            Value::from(artifact_size(space.artifact)),
        );
        obj.insert(
            "medianEmbedMs".into(),
            serde_json::Number::from_f64(single_ms)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        obj.insert(
            "per1kMs".into(),
            serde_json::Number::from_f64(batch_1k)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        obj.insert(
            "stsSpearman".into(),
            serde_json::Number::from_f64(rho)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
        if let Some(r) = &retrieval {
            obj.insert("queriesScored".into(), Value::from(r.queries_scored));
            obj.insert("R@1".into(), json_f(r.r1));
            obj.insert("R@5".into(), json_f(r.r5));
            obj.insert("R@10".into(), json_f(r.r10));
            obj.insert("nDCG@10".into(), json_f(r.ndcg10));
            obj.insert("MRR@10".into(), json_f(r.mrr10));
        }
        let value = Value::Object(obj);
        results.push(value.clone());
        print_row(&value, retrieval.is_some(), have_sts);
    }

    println!();
    if let Ok(path) = std::env::var("SOFUU_BENCH_JSON") {
        let doc = serde_json::json!({ "results": results });
        match std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap_or_default()) {
            Ok(()) => println!("wrote {}", path),
            Err(e) => eprintln!("could not write {}: {}", path, e),
        }
    }
}

fn json_f(v: f64) -> Value {
    serde_json::Number::from_f64(v).map(Value::Number).unwrap_or(Value::Null)
}

fn print_row(v: &Value, retrieval: bool, sts: bool) {
    let get = |k: &str| v.get(k).and_then(|x| x.as_f64());
    let mut line = format!("  {:<10}", v["space"].as_str().unwrap_or("?"));
    match v.get("dim").and_then(|x| x.as_u64()) {
        Some(d) => line.push_str(&format!("dim {:>4}  ", d)),
        None => line.push_str("dim    —  "),
    }
    if let Some(p) = get("params") {
        line.push_str(&format!("{:>7} params  ", p as i64));
    } else {
        line.push_str("      — params  ");
    }
    if let Some(b) = get("artifactBytes") {
        line.push_str(&format!("{:>6.1} KB", b / 1024.0));
    }
    if let Some(ms) = get("medianEmbedMs") {
        line.push_str(&format!("  {:>7.3} ms", ms));
    }
    if retrieval {
        line.push_str(&format!(
            "  R@1 {:.3}  R@5 {:.3}  nDCG@10 {:.3}",
            get("R@1").unwrap_or(f64::NAN),
            get("R@5").unwrap_or(f64::NAN),
            get("nDCG@10").unwrap_or(f64::NAN)
        ));
    }
    if sts {
        line.push_str(&format!("  rho {:.3}", get("stsSpearman").unwrap_or(f64::NAN)));
    }
    println!("{line}");
}
