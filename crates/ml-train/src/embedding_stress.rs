//! Extreme-condition stress suite for the semantic embedder candidate.
//!
//!   cargo run -p ml-train --release -- embed-stress
//!
//! Hard gates (exit 1 on failure): crash-safety on hostile inputs,
//! bit-determinism, and recall latency at scale.  Informational probes —
//! out-of-domain retrieval, case/prefix/typo robustness, multilingual,
//! tiny and oversized queries — feed /tmp/sofuu_embed_stress_mining.tsv so
//! the next training round can strengthen whatever broke.
//!
//! The candidate artifact comes from SOFUU_EMBED_EVAL_ARTIFACT (validated
//! allow-list) or the in-tree baked file, same as `embed-eval`.

use std::collections::HashMap;
use std::time::Instant;

use sofuu_core::embedding::{hash_v1_features, HASH_DIM, SEMANTIC_DIM};
use sofuu_core::memory::cma::Cma;

use crate::data_embedding_gen::{self, ood_domain_families, OOD_DOMAINS};
use crate::embedding_eval::{build_corpus, load_semantic_model, semantic_embed, FusedStore};

const TOP_K: usize = 5;
/// §10.3 latency bar is 10 ms for an 8-KiB forward; recall on a ~2k-record
/// index may reasonably cost more, but past this it is a pathology.
const BUDGET_RECALL_P95_MS: f64 = 25.0;
const BUDGET_FORWARD_P95_MS: f64 = 10.0;

type Tag = (usize, usize);

struct Backend {
    cma: Cma,
    id_tag: HashMap<u32, Tag>,
}

impl Backend {
    fn new(dim: usize, embed: &dyn Fn(&str) -> Vec<f32>, records: &[(String, Tag)]) -> Backend {
        let mut cma = Cma::new(dim);
        let mut id_tag = HashMap::new();
        for (text, tag) in records {
            let id = cma.remember(&embed(text), text, "note", 0);
            id_tag.insert(id as u32, *tag);
        }
        Backend { cma, id_tag }
    }

    /// does the query surface its own family anywhere in the top-5?
    fn hits_family(&mut self, embed: &dyn Fn(&str) -> Vec<f32>, q: &str, tag: &Tag) -> bool {
        for h in self.cma.recall(&embed(q), TOP_K) {
            if self.id_tag.get(&h.id) == Some(tag) {
                return true;
            }
        }
        false
    }
}

/// Recall@5 over a family-structured probe set: per query, own family
/// anywhere in the top-5 of the pool built from all families' memories.
fn recall_at_5(
    dim: usize,
    embed: &dyn Fn(&str) -> Vec<f32>,
    fams: &[(Vec<String>, Vec<String>)], // (memories, queries)
) -> f32 {
    let mut records: Vec<(String, Tag)> = Vec::new();
    let mut queries: Vec<(String, Tag)> = Vec::new();
    for (fi, (mems, qs)) in fams.iter().enumerate() {
        for m in mems {
            records.push((m.clone(), (0, fi)));
        }
        for q in qs {
            queries.push((q.clone(), (0, fi)));
        }
    }
    let mut be = Backend::new(dim, embed, &records);
    let mut hits = 0usize;
    for (q, tag) in &queries {
        if be.hits_family(embed, q, tag) {
            hits += 1;
        }
    }
    hits as f32 / queries.len().max(1) as f32
}

/// Round-6: recall@5 of the fused two-channel pipeline (sem64 tower +
/// hash768, shared `FusedStore` + pre-registered RRF rule) over the same
/// family-structured probe sets `recall_at_5` grades.
fn recall_at_5_fused(
    sem_fn: &dyn Fn(&str) -> Vec<f32>,
    hash_fn: &dyn Fn(&str) -> Vec<f32>,
    fams: &[(Vec<String>, Vec<String>)],
) -> f32 {
    let mut records: Vec<(String, Tag)> = Vec::new();
    let mut queries: Vec<(String, Tag)> = Vec::new();
    for (fi, (mems, qs)) in fams.iter().enumerate() {
        for m in mems {
            records.push((m.clone(), (0, fi)));
        }
        for q in qs {
            queries.push((q.clone(), (0, fi)));
        }
    }
    let mut store = FusedStore::new(sem_fn, hash_fn, &records);
    let mut hits = 0usize;
    for (q, tag) in &queries {
        let fused = store.top5(&sem_fn(q), &hash_fn(q));
        if fused.iter().any(|id| store.idx_tags().get(id) == Some(tag)) {
            hits += 1;
        }
    }
    hits as f32 / queries.len().max(1) as f32
}

// ── hard gate 1: hostile inputs ─────────────────────────────────────────

fn hostile_inputs(m: &crate::embedding_eval::Artifact) -> Result<(), String> {
    let mut eight_kib = String::from("deploy the staging cluster ");
    while eight_kib.len() < 8192 {
        eight_kib.push('x');
    }
    let cases = [
        "",
        "   ",
        "\n\t\r\n",
        "héllo wörld — ünïcode",
        "emoji 🦉 owl 日本語 नमस्ते",
        "combining a\u{0301}e\u{0301}",
        "/Users/dev/project/src/main.rs:142",
        "fn main() { let x: Option<Vec<u8>> = None; }",
        "error\u{0}with\u{1}control\u{7}chars",
        "\u{FFFD}\u{FFFE}",
        eight_kib.as_str(),
    ];
    for text in cases {
        let out = m.forward_text(text);
        if out.len() != SEMANTIC_DIM || out.iter().any(|v| !v.is_finite()) {
            return Err(format!("non-finite/short output for {text:?}"));
        }
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 1e-4 {
            return Err(format!("norm {norm} for {text:?}"));
        }
    }
    // raw bytes that are not valid UTF-8 reach the feature layer from the C ABI
    let raw_cases: [Vec<u8>; 4] = [
        Vec::new(),
        vec![0xFF, 0xFE, 0x80, 0xC3, 0x28, 0x00],
        vec![b'x'; 8192],
        (0..=255u8).cycle().take(4096).collect(),
    ];
    for raw in &raw_cases {
        let out = m.forward_text(&String::from_utf8_lossy(raw));
        if out.iter().any(|v| !v.is_finite()) {
            return Err("non-finite output on raw-byte probe".to_string());
        }
    }
    Ok(())
}

// ── hard gate 2: determinism ────────────────────────────────────────────

fn determinism(m: &crate::embedding_eval::Artifact) -> Result<(), String> {
    for text in [
        "paraphrase stability probe",
        "",
        "🦉 unicode probe 日本語",
        "/Users/dev/project/src/main.rs:142",
        "which tokio version does sofuu-cli use?",
    ] {
        let a = m.forward_text(text);
        let b = m.forward_text(text);
        if a.iter().zip(b.iter()).any(|(x, y)| x.to_bits() != y.to_bits()) {
            return Err(format!("forward not bit-stable for {text:?}"));
        }
    }
    Ok(())
}

// ── informational probes ────────────────────────────────────────────────

const MULTILINGUAL: [(&str, [&str; 3], [&str; 3]); 4] = [
    (
        "ja",
        [
            "展開スクリプトは毎朝7時に走る。失敗したら Slack に通知する。",
            "デプロイの前に必ず展開スクリプトを実行する決まりだ。",
            "朝の自動展開は7時開始で、通知は Slack だ。",
        ],
        [
            "展開スクリプトはいつ走る？",
            "朝の自動展開は何時？",
            "失敗通知はどこに行く？",
        ],
    ),
    (
        "de",
        [
            "Der Backup-Job läuft nachts um drei und schreibt nach Frankfurt.",
            "Nachts um drei startet der Backup-Job; das Ziel ist Frankfurt.",
            "Backup-Job: nächtlich, 03:00, Zielrechenzentrum Frankfurt.",
        ],
        [
            "Wann läuft der Backup-Job?",
            "Wohin schreibt der Backup-Job?",
            "Wie oft läuft der Backup-Job?",
        ],
    ),
    (
        "fr",
        [
            "L'API de facturation renvoie 402 quand le quota est dépassé.",
            "Erreur 402 sur l'API de facturation — quota dépassé, il faut purger.",
            "Quota dépassé ? L'API de facturation répond 402.",
        ],
        [
            "Pourquoi l'API de facturation renvoie 402 ?",
            "Que faire face à une erreur 402 ?",
            "Quand l'API renvoie-t-elle 402 ?",
        ],
    ),
    (
        "es",
        [
            "El despliegue requiere la firma del artefacto antes de subirlo.",
            "Sin firma del artefacto, el despliegue se rechaza siempre.",
            "Regla del despliegue: firmar el artefacto primero, subir después.",
        ],
        [
            "¿Qué requiere el despliegue?",
            "¿Por qué se rechaza el despliegue?",
            "¿Cuándo se firma el artefacto?",
        ],
    ),
];

/// Deterministic single-typo corruption: swap two adjacent chars inside the
/// longest ASCII word of the query.
fn typo(text: &str) -> String {
    let mut best: Option<(usize, usize)> = None; // (word start, word len)
    for (i, w) in text.split_whitespace().enumerate() {
        if w.len() >= 5 && w.chars().all(|c| c.is_ascii_alphanumeric()) {
            let byte_start = text.match_indices(w).nth(i).map(|(p, _)| p).unwrap_or(0);
            if best.map(|(_, bl)| w.len() > bl).unwrap_or(true) {
                best = Some((byte_start, w.len()));
            }
        }
    }
    let Some((start, len)) = best else {
        return text.to_string();
    };
    let mut out = text.to_string();
    let mid = start + len / 2;
    let swapped = format!("{}{}", &text[mid + 1..mid + 2], &text[mid..mid + 1]);
    out.replace_range(mid..mid + 2, &swapped);
    out
}

fn variant_hit_rate(
    be: &mut Backend,
    embed: &dyn Fn(&str) -> Vec<f32>,
    queries: &[(String, Tag)],
    make: &dyn Fn(&str) -> String,
) -> f32 {
    let mut hits = 0usize;
    for (q, tag) in queries {
        if be.hits_family(embed, &make(q), tag) {
            hits += 1;
        }
    }
    hits as f32 / queries.len().max(1) as f32
}

fn stress_mining_write(lines: &[(String, f32, f32)]) {
    let mut out = String::from("# probe\tsem_r5\thash_r5\n");
    for (name, s, h) in lines {
        out.push_str(&format!("{name}\t{s:.4}\t{h:.4}\n"));
    }
    let _ = std::fs::write("/tmp/sofuu_embed_stress_mining.tsv", out);
}

pub fn run_stress() -> i32 {
    println!("SEMANTIC EMBEDDER stress suite (extreme conditions)");
    let sm = match load_semantic_model() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    println!("artifact: {} ({})", sm.path.display(), sm.id);
    let sem_fn = semantic_embed(&sm);
    let twochannel = std::env::var("SOFUU_EMBED_EVAL_MODE").as_deref() == Ok("twochannel");
    if twochannel {
        println!("mode: twochannel — retrieval probes also grade the fused pipeline (S sem64 + H hash768, shared RRF rule)");
    }
    let mut failed: Vec<String> = Vec::new();
    let mut mining: Vec<(String, f32, f32)> = Vec::new();

    // ── hard: crash-safety + determinism ──
    if let Err(e) = hostile_inputs(&sm.model) {
        failed.push(format!("hostile-input crash-safety: {e}"));
    } else {
        println!("hostile inputs (empty/unicode/8KiB/control/raw bytes): OK (finite, unit-norm)");
    }
    if let Err(e) = determinism(&sm.model) {
        failed.push(format!("determinism: {e}"));
    } else {
        println!("bit-determinism (5 probes × 2 forwards): OK");
    }

    // ── hard: 8-KiB forward p95 (same bar as §10.3) ──
    {
        let unit_str = "fix the login timeout bug in the scheduler pool ";
        let mut big = unit_str.repeat(8192 / unit_str.len() + 1);
        big.truncate(8192);
        let mut samples = Vec::new();
        for i in 0..200 {
            let probe = format!("{big} variation {}", i % 7);
            let t = Instant::now();
            let _ = sm.model.forward_text(&probe);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95 = samples[190];
        let ok = p95 <= BUDGET_FORWARD_P95_MS && cfg!(debug_assertions) == false || p95 <= BUDGET_FORWARD_P95_MS;
        println!("8-KiB forward p95: {p95:.3} ms (bar {BUDGET_FORWARD_P95_MS:.0} ms) {}", if ok { "OK" } else { "FAIL" });
        if !ok {
            failed.push(format!("8-KiB forward p95 {p95:.3} ms"));
        }
    }

    // ── informational: OOD domains ──
    println!("\nout-of-domain retrieval (never trained, fresh instances):");
    if twochannel {
        println!("  {:<12} {:>8} {:>8} {:>9}", "domain", "sem R@5", "hash R@5", "fused R@5");
    } else {
        println!("  {:<12} {:>8} {:>8}", "domain", "sem R@5", "hash R@5");
    }
    let mut ood_sem_sum = 0.0f32;
    let mut ood_hash_sum = 0.0f32;
    let mut ood_fused_sum = 0.0f32;
    for (di, domain) in OOD_DOMAINS.iter().enumerate() {
        let fams: Vec<(Vec<String>, Vec<String>)> =
            ood_domain_families(di, 0x00D5_EED0_0001u64 + di as u64, 6)
                .into_iter()
                .map(|f| (f.memories, f.train_queries))
                .collect();
        let s = recall_at_5(SEMANTIC_DIM, &sem_fn, &fams);
        let h = recall_at_5(HASH_DIM, &hash_v1_features, &fams);
        ood_sem_sum += s;
        ood_hash_sum += h;
        if twochannel {
            let f = recall_at_5_fused(&sem_fn, &hash_v1_features, &fams);
            ood_fused_sum += f;
            println!("  {domain:<12} {s:>8.3} {h:>8.3} {f:>9.3}");
        } else {
            println!("  {domain:<12} {s:>8.3} {h:>8.3}");
        }
        mining.push((format!("ood:{domain}"), s, h));
    }
    let n_ood = OOD_DOMAINS.len() as f32;
    if twochannel {
        println!(
            "  {:<12} {:>8.3} {:>8.3} {:>9.3}",
            "OOD overall",
            ood_sem_sum / n_ood,
            ood_hash_sum / n_ood,
            ood_fused_sum / n_ood
        );
    } else {
        println!(
            "  {:<12} {:>8.3} {:>8.3}",
            "OOD overall",
            ood_sem_sum / n_ood,
            ood_hash_sum / n_ood
        );
    }

    // ── informational: scale (≈2k records) ──
    let scale_corpus = data_embedding_gen::corpus(0x5CA1E_0000_0001, &HashMap::new());
    {
        const VAL_OFF: usize = 1_000; // val-family tag offset (train tags are 0..)
        let mut records: Vec<(String, Tag)> = Vec::new();
        for (ci, cat) in scale_corpus.cats.iter().enumerate() {
            for (fi, f) in cat.train.iter().enumerate() {
                for t in f.memories.iter().chain(&f.train_queries) {
                    records.push((t.clone(), (ci, fi)));
                }
            }
            // val families' memories belong to the index too — probes are
            // their queries, tagged with the offset so the spaces match
            for (fi, f) in cat.val.iter().enumerate() {
                for t in &f.memories {
                    records.push((t.clone(), (ci, VAL_OFF + fi)));
                }
            }
        }
        for (gi, g) in scale_corpus.curated.iter().enumerate() {
            for t in g {
                records.push((t.clone(), (11, gi)));
            }
        }
        let mut probes: Vec<(String, Tag)> = Vec::new();
        for (ci, cat) in scale_corpus.cats.iter().enumerate() {
            for (fi, f) in cat.val.iter().enumerate() {
                for q in &f.val_queries {
                    probes.push((q.clone(), (ci, VAL_OFF + fi)));
                }
            }
        }
        println!(
            "\nscale: {} records, {} probes (unseen val instances)",
            records.len(),
            probes.len()
        );
        for (name, dim, embed) in [
            ("semantic", SEMANTIC_DIM, &sem_fn as &dyn Fn(&str) -> Vec<f32>),
            ("hash-v1", HASH_DIM, &hash_v1_features as &dyn Fn(&str) -> Vec<f32>),
        ] {
            let mut be = Backend::new(dim, embed, &records);
            let mut lat: Vec<f64> = Vec::new();
            let mut hits = 0usize;
            for (q, tag) in &probes {
                let t = Instant::now();
                let hit = be.hits_family(embed, q, tag);
                lat.push(t.elapsed().as_secs_f64() * 1000.0);
                if hit {
                    hits += 1;
                }
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p95 = lat[(lat.len() * 95) / 100];
            println!(
                "  {name}: R@5={:.3}  p95 recall={p95:.3} ms",
                hits as f32 / probes.len() as f32
            );
            if name == "semantic" && p95 > BUDGET_RECALL_P95_MS {
                failed.push(format!("scale recall p95 {p95:.3} ms > {BUDGET_RECALL_P95_MS:.0} ms"));
            }
        }
        if twochannel {
            let mut store = FusedStore::new(&sem_fn, &hash_v1_features, &records);
            let mut lat: Vec<f64> = Vec::new();
            let mut hits = 0usize;
            for (q, tag) in &probes {
                let t = Instant::now();
                let fused = store.top5(&sem_fn(q), &hash_v1_features(q));
                let hit = fused.iter().any(|id| store.idx_tags().get(id) == Some(tag));
                lat.push(t.elapsed().as_secs_f64() * 1000.0);
                if hit {
                    hits += 1;
                }
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p95 = lat[(lat.len() * 95) / 100];
            println!(
                "  fused (S+H RRF): R@5={:.3}  p95 recall={p95:.3} ms",
                hits as f32 / probes.len() as f32
            );
        }
    }

    // ── informational: variants, typos, multilingual, tiny, long ──
    let corpus = build_corpus();
    {
        // evaluation-corpus backend (records only; queries reused for probes)
        let mut sem_be = Backend::new(SEMANTIC_DIM, &sem_fn, &corpus.records);
        let mut hash_be = Backend::new(HASH_DIM, &hash_v1_features, &corpus.records);
        let qs: Vec<(String, Tag)> = corpus.queries.clone();

        let cases: [(&str, &dyn Fn(&str) -> String); 4] = [
            ("uppercase", &|q: &str| q.to_uppercase()),
            ("prefix", &|q: &str| format!("quick q: {q}")),
            ("spacing", &|q: &str| format!("{q}   ??")),
            ("typo", &|q: &str| typo(q)),
        ];
        println!("\nrobustness on the acceptance queries (own-family in top-{}):", TOP_K);
        for (name, make) in cases {
            let s = variant_hit_rate(&mut sem_be, &sem_fn, &qs, make);
            let h = variant_hit_rate(&mut hash_be, &hash_v1_features, &qs, make);
            println!("  {name:<10} sem {s:.3}  hash {h:.3}");
            mining.push((format!("variant:{name}"), s, h));
        }

        let mlfams: Vec<(Vec<String>, Vec<String>)> = MULTILINGUAL
            .iter()
            .map(|(_, m, q)| (m.iter().map(|s| s.to_string()).collect(), q.iter().map(|s| s.to_string()).collect()))
            .collect();
        let s = recall_at_5(SEMANTIC_DIM, &sem_fn, &mlfams);
        let h = recall_at_5(HASH_DIM, &hash_v1_features, &mlfams);
        if twochannel {
            let f = recall_at_5_fused(&sem_fn, &hash_v1_features, &mlfams);
            println!("multilingual (ja/de/fr/es): sem {s:.3}  hash {h:.3}  fused {f:.3}");
        } else {
            println!("multilingual (ja/de/fr/es): sem {s:.3}  hash {h:.3}");
        }
        mining.push(("multilingual".to_string(), s, h));

        // tiny queries: does the exact token route to the right family?
        let tiny: [(&str, Tag, &str); 5] = [
            ("qtc?", (0, 0), "paraphrase/session-compression"),
            ("E4102", (5, 0), "errors/E4102"),
            ("tokio", (6, 0), "versions/any-tokio"),
            ("config.rs", (4, 0), "paths/any"),
            ("retry", (2, 0), "documentation/any"),
        ];
        println!("tiny queries (top-1 tag):");
        for (q, tag, desc) in tiny {
            let s_top = sem_be
                .cma
                .recall(&sem_fn(q), 1)
                .first()
                .and_then(|h| sem_be.id_tag.get(&h.id).copied());
            let h_top = hash_be
                .cma
                .recall(&hash_v1_features(q), 1)
                .first()
                .and_then(|h| hash_be.id_tag.get(&h.id).copied());
            let mark = |got: Option<Tag>| -> String {
                match got {
                    Some(g) if g == tag => "hit".to_string(),
                    Some(g) => format!("miss ({},{})", g.0, g.1),
                    None => "empty".to_string(),
                }
            };
            println!("  {q:<12} [{desc:<32}] sem {}  hash {}", mark(s_top), mark(h_top));
        }

        // oversized query: a 4-KiB repetition of one memory's text
        {
            let base = corpus.records[0].0.clone();
            let long = base.repeat(4096 / base.len().max(1) + 1);
            let long = &long[..4096.min(long.len())];
            let tag = corpus.records[0].1;
            let s = sem_be.hits_family(&sem_fn, long, &tag);
            let h = hash_be.hits_family(&hash_v1_features, long, &tag);
            println!("4-KiB query from record 0: sem hit={s}  hash hit={h}");
        }
    }

    stress_mining_write(&mining);
    println!(
        "\nmining file: /tmp/sofuu_embed_stress_mining.tsv ({} probes)",
        mining.len()
    );

    if failed.is_empty() {
        println!("STRESS VERDICT: PASS (hard gates; informational probes above)");
        0
    } else {
        println!("\nSTRESS VERDICT: FAIL — {} hard gate(s):", failed.len());
        for f in &failed {
            println!("  - {f}");
        }
        1
    }
}
