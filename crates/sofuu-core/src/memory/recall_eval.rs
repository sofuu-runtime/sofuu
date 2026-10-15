// sofuu-core — B5.4 (ROADMAP-BRAIN-CMA): the recall evaluation harness.
//
// A fixed, deterministic fixture + query set for the CMA recall path. Every
// B5 ranking change is measured against these numbers instead of asserted:
//
//   cargo test -p sofuu-core --lib recall_eval -- --nocapture
//
// Reported per configuration: recall@k (fraction of the relevant records
// retrieved), MRR (1/rank of the first relevant hit), topic coverage@k (how
// many of the query's topics the top-k touches) and redundancy@k (1 -
// distinct topics / k). Near-duplicate crowding shows up as low coverage and
// high redundancy while recall@k stays flat — that is exactly the failure
// B5.1 fixes, so all four numbers move together or the change is not real.

use super::cma::Cma;
use super::hnsw::Hnsw;

const DIM: usize = 256;
/// One chatty topic (five paraphrases of the same fact) beside a spread of
/// sparse ones. Plenty of topics is what makes redundancy MEANINGFUL: with
/// only three topics, k=6 slots can carry at most three distinct topics, so
/// the redundancy@k metric could never read below 0.5 no matter how good
/// the selection is. Seven topics (15 relevant records) give the target
/// (redundancy ≤ 0.20 = at most one repeated topic among the six) room.
const TOPIC_SIZES: [usize; 7] = [6, 2, 2, 2, 1, 1, 1];
const NOISE: usize = 40;
const TOP_K: usize = 6;
/// Per-component paraphrase jitter, scaled so a member stays ~0.87-aligned
/// with its topic base (and ~0.69 with its siblings) at any DIM.
fn jitter() -> f32 {
    0.12 * (64.0 / DIM as f32).sqrt()
}

/// Deterministic xorshift — the fixture must be identical on every run and
/// on every machine.
struct Rng(u32);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn vec(&mut self) -> Vec<f32> {
        let mut v: Vec<f32> = (0..DIM).map(|_| self.unit()).collect();
        normalize(&mut v);
        v
    }
}

fn now_secs() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().min(u32::MAX as u64) as u32)
        .unwrap_or(0)
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// The fixture: a brain plus one query that touches all three topics.
struct Fixture {
    cma: Cma,
    query: Vec<f32>,
    /// Record ids per topic (the relevant set is their union).
    topics: Vec<Vec<u32>>,
}

fn build() -> Fixture {
    let mut rng = Rng(0x00C0_FFEE);
    let mut cma = Cma::new(DIM);
    let mut bases: Vec<Vec<f32>> = Vec::new();
    let mut topics: Vec<Vec<u32>> = Vec::new();
    for (t, &size) in TOPIC_SIZES.iter().enumerate() {
        let base = rng.vec();
        let mut ids = Vec::new();
        for _ in 0..size {
            /* Paraphrase: the topic's direction plus jitter — same meaning,
             * different surface. Well above DUP_DISTANCE, so remember()
             * keeps them as separate records (that is the point). */
            let mut v = base.clone();
            for x in v.iter_mut() {
                *x += rng.unit() * jitter();
            }
            normalize(&mut v);
            let id = cma.remember(&v, &format!("topic{t} paraphrase"), "user", 0);
            ids.push(id as u32);
        }
        bases.push(base);
        topics.push(ids);
    }
    for i in 0..NOISE {
        let v = rng.vec();
        cma.remember(&v, &format!("noise {i}"), "user", 0);
    }
    /* The query is dominated by topic 0 (the repeated fact is also the one
     * being asked about) with the other two topics as secondary intents. */
    let mut q = vec![0.0f32; DIM];
    for (i, b) in bases.iter().enumerate() {
        let w = if i == 0 { 3.0 } else { 2.0 };
        for d in 0..DIM {
            q[d] += b[d] * w;
        }
    }
    for x in q.iter_mut() {
        *x += rng.unit() * 0.05;
    }
    normalize(&mut q);
    Fixture { cma, query: q, topics }
}

impl Fixture {
    fn relevant(&self) -> Vec<u32> {
        self.topics.iter().flatten().copied().collect()
    }

    fn topic_of(&self, id: u32) -> Option<usize> {
        self.topics.iter().position(|t| t.contains(&id))
    }
}

/// recall@k, MRR, coverage@k, redundancy@k for one recall run.
struct Metrics {
    recall: f32,
    mrr: f32,
    coverage: f32,
    redundancy: f32,
}

fn measure(f: &mut Fixture, k: usize) -> Metrics {
    let hits = f.cma.recall(&f.query, k);
    let relevant = f.relevant();
    let got = hits.iter().filter(|h| relevant.contains(&h.id)).count();
    let recall = got as f32 / (k.min(relevant.len()) as f32);
    let mrr = match hits.iter().position(|h| relevant.contains(&h.id)) {
        Some(i) => 1.0 / (i as f32 + 1.0),
        None => 0.0,
    };
    let mut seen: Vec<usize> = Vec::new();
    for h in &hits {
        if let Some(t) = f.topic_of(h.id) {
            if !seen.contains(&t) {
                seen.push(t);
            }
        }
    }
    let coverage = seen.len() as f32 / f.topics.len() as f32;
    let redundancy = 1.0 - (seen.len() as f32 / hits.len().max(1) as f32);
    Metrics { recall, mrr, coverage, redundancy }
}

fn fmt(m: &Metrics) -> String {
    format!(
        "recall@{k} {r:.3}  MRR {mrr:.3}  coverage {c:.3}  redundancy {red:.3}",
        k = TOP_K,
        r = m.recall,
        mrr = m.mrr,
        c = m.coverage,
        red = m.redundancy
    )
}

/// B5.1+B5.2: the default configuration must touch more of the query's
/// topics and crowd out fewer slots with paraphrases, without losing the
/// best hit (MRR) or collapsing retrieval (recall).
#[test]
fn recall_eval_diversity_beats_baseline() {
    let mut base = build();
    base.cma.set_diversity_floor(None); /* pre-B5 order */
    let baseline = measure(&mut base, TOP_K);

    let mut tuned = build();
    let tuned_metrics = measure(&mut tuned, TOP_K);

    println!(
        "── B5 recall evaluation (topics {:?} + {} noise, dim {}, k {}) ──",
        TOPIC_SIZES, NOISE, DIM, TOP_K
    );
    println!("baseline  (floor off): {}", fmt(&baseline));
    println!(
        "default   (floor {:.2}): {}",
        tuned.cma.diversity_floor.unwrap_or(0.0),
        fmt(&tuned_metrics)
    );

    assert!(
        tuned_metrics.coverage > baseline.coverage,
        "diversity must touch more of the query's topics: {:.3} vs baseline {:.3}",
        tuned_metrics.coverage, baseline.coverage
    );
    assert!(
        tuned_metrics.redundancy < baseline.redundancy,
        "diversity must cut near-duplicate crowding: {:.3} vs baseline {:.3}",
        tuned_metrics.redundancy, baseline.redundancy
    );
    assert!(
        tuned_metrics.mrr >= baseline.mrr - 1e-6,
        "the best hit stays first: {:.3} vs baseline {:.3}",
        tuned_metrics.mrr, baseline.mrr
    );
    /* M1 (Brain Core II) targets — the precision bar for the ranking path:
     * with enough topics in the fixture, the top-k fills with DISTINCT
     * relevant records — every slot relevant (recall), the best first
     * (MRR), every topic reachable (coverage), and at most one repeat
     * across the six (redundancy). */
    assert!(
        tuned_metrics.recall >= 0.95,
        "M1: the top-{} must be (nearly) all relevant: {:.3}",
        TOP_K, tuned_metrics.recall
    );
    assert!(
        tuned_metrics.mrr >= 0.99,
        "M1: the first hit must be the right one: {:.3}",
        tuned_metrics.mrr
    );
    assert!(
        tuned_metrics.redundancy <= 0.20,
        "M1: at most one repeated topic across the top-{}: {:.3}",
        TOP_K, tuned_metrics.redundancy
    );
}

/// The floor sweep behind DIVERSITY_FLOOR_DEFAULT — printed, not asserted,
/// so the default can be re-chosen from numbers when the fixture changes.
#[test]
fn recall_eval_floor_sweep() {
    println!("── B5.1 diversity floor sweep (k {}) ──", TOP_K);
    for floor in [None, Some(0.4f32), Some(0.5), Some(0.6), Some(0.7), Some(0.8)] {
        let mut f = build();
        f.cma.set_diversity_floor(floor);
        let m = measure(&mut f, TOP_K);
        println!(
            "floor {}: {}",
            floor.map(|v| format!("{v:.2}")).unwrap_or_else(|| "off".into()),
            fmt(&m)
        );
    }
}

/// B5.2: two records at the same distance with the same strength — the
/// newer one wins. Inserted oldest-first so a stable sort alone would keep
/// the stale record on top.
#[test]
fn recall_eval_recency_breaks_ties() {
    /* Same angle off the query in two different directions: equal distance,
     * cosine 0.29 between them -- not near-duplicates, so the diversity
     * floor keeps both. */
    let mut cma = Cma::new(4);
    let theta = 1.0f32;
    let query = [1.0f32, 0.0, 0.0, 0.0];
    let stale = [theta.cos(), theta.sin(), 0.0, 0.0];
    let fresh = [theta.cos(), 0.0, theta.sin(), 0.0];
    let a = cma.remember(&stale, "stale fact", "user", 0) as u32;
    let b = cma.remember(&fresh, "fresh fact", "user", 0) as u32;
    let now = now_secs();
    cma.records[a as usize].created_at = now - 30 * 86400;
    cma.records[b as usize].created_at = now;

    let hits = cma.recall(&query, 2);
    assert_eq!(hits[0].id, b, "the fresh record must outrank the stale one");
    assert_eq!(hits[1].id, a);
}

/// B5.3: a wider search buys recall against brute force. Sizes are small
/// enough that the whole test runs in milliseconds.
#[test]
fn recall_eval_ef_knob_trades_recall() {
    let n = 600usize;
    let k = 10usize;
    let mut rng = Rng(0x5EED_1234);
    let mut index = Hnsw::new(DIM);
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let v = rng.vec();
        index.add_vector(&v);
        vecs.push(v);
    }
    let queries: Vec<Vec<f32>> = (0..20).map(|_| rng.vec()).collect();

    let brute = |q: &[f32]| -> Vec<u32> {
        let mut all: Vec<(u32, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
                (i as u32, d)
            })
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all.into_iter().take(k).map(|t| t.0).collect()
    };

    let mut recall_at = |ef: usize| -> f32 {
        let mut hit = 0usize;
        for q in &queries {
            let truth = brute(q);
            let got = index.search_with_ef(q, k, ef);
            hit += got.iter().filter(|(id, _)| truth.contains(id)).count();
        }
        hit as f32 / (queries.len() * k) as f32
    };

    let narrow = recall_at(8);
    let wide = recall_at(512);
    let default = recall_at((k * 4).max(8));
    println!(
        "── B5.3 ef sweep (n {}, k {}): ef 8 → {:.3}, default {} → {:.3}, ef 512 → {:.3} ──",
        n, k, narrow, (k * 4).max(8), default, wide
    );
    assert!(wide >= narrow, "a wider search cannot lose recall: {wide:.3} < {narrow:.3}");
    assert!(wide >= 0.95, "ef 512 over {n} vectors should be near-exact: {wide:.3}");
    assert!(default >= narrow, "the default must not be worse than the narrow floor");
}

/// B5.3 wiring: `set_ef_search` reaches the search itself. Recall's order
/// is pinned against a direct `search_with_ef` call on the same index
/// (diversity off, uniform strength, no resonance — so the two must agree
/// exactly). One recall per instance: recall itself mutates gravity, which
/// would pollute a second call on the same brain.
#[test]
fn recall_eval_ef_search_setting_is_used() {
    let n = 600usize;
    let k = 10usize;

    let build_index = |seed: u32| -> (Cma, Rng) {
        let mut rng = Rng(seed);
        let mut cma = Cma::new(DIM);
        for i in 0..n {
            let v = rng.vec();
            cma.remember(&v, &format!("m{i}"), "user", 0);
        }
        cma.set_diversity_floor(None);
        (cma, rng)
    };

    /* A query where the two widths actually disagree on this index. */
    let (mut a, mut rng) = build_index(0xABCD_0001);
    let q = (0..50)
        .map(|_| rng.vec())
        .find(|q| {
            let x: Vec<u32> = a.index.search_with_ef(q, 32, 8).into_iter().map(|t| t.0).collect();
            let y: Vec<u32> = a.index.search_with_ef(q, 32, 512).into_iter().map(|t| t.0).collect();
            x != y
        })
        .expect("a 600-vector graph must have a query where ef 8 and 512 differ");
    let direct8: Vec<u32> = a.index.search_with_ef(&q, 32, 8).into_iter().map(|t| t.0).collect();
    a.set_ef_search(Some(8));
    let got8: Vec<u32> = a.recall(&q, k).into_iter().map(|h| h.id).collect();
    assert_eq!(got8, direct8[..k].to_vec(), "ef 8 must reach the search");

    let (mut b, _) = build_index(0xABCD_0001);
    let direct512: Vec<u32> = b.index.search_with_ef(&q, 32, 512).into_iter().map(|t| t.0).collect();
    b.set_ef_search(Some(512));
    let got512: Vec<u32> = b.recall(&q, k).into_iter().map(|h| h.id).collect();
    assert_eq!(got512, direct512[..k].to_vec(), "ef 512 must reach the search");

    b.set_ef_search(None);
    assert!(b.ef_search.is_none());
}

