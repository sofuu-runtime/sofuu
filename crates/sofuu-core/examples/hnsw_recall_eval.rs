//! W4.5 (QTSQ_UPGRADE_PLAN §Upgrade 4): ground-truth recall harness for the
//! HNSW index.
//!
//! Builds an index, computes exact k-NN by brute force over the same corpus,
//! and reports recall@1 / recall@10 / recall@50 plus build time and query
//! latency — per neighbour-selection policy, so the W4 upgrade can be
//! measured BEFORE/AFTER on identical data:
//!
//!   * `heuristic`  — W4.1/W4.2, the shipped path;
//!   * `nearest`    — plain closest-M truncation (pre-W4 insert selection);
//!   * `legacy`     — the pre-W4 code verbatim (closest-M insert selection,
//!                    random level-0 replacement on backlink overflow,
//!                    silently dropped backlinks above level 0).
//!
//! Everything is deterministic: fixed splitmix64 seed, single thread, no
//! wall-clock or address input. Run it twice and the numbers match.
//!
//! ```text
//! cargo run --release -p sofuu-core --example hnsw_recall_eval
//! cargo run --release -p sofuu-core --example hnsw_recall_eval -- \
//!     --n 20000 --dim 128 --queries 200 --fixture uniform
//! cargo run --release -p sofuu-core --example hnsw_recall_eval -- \
//!     --n 20000 --dim 128 --queries 200 --fixture clustered --clusters 400 --sigma 0.3
//! cargo run --release -p sofuu-core --example hnsw_recall_eval -- \
//!     --n 20000 --queries 200 --policies legacy --tombstone 0
//! ```
//!
//! Flags: `--n --dim --queries --k --ef --fixture uniform|clustered
//! --clusters --sigma --policies a,b,c --tombstone <frac> --seed <hex|dec>`.
//! Defaults are the CI-sized run (n = 2000, dim = 128, 50 queries) which
//! takes seconds; `--n 20000` is the acceptance run.
//!
//! The corpus lives ABOVE `HNSW_SMALL_GRAPH` (1024) by default, so the
//! measured path is the budgeted approximate search — not the exhaustive
//! fallback that made the old in-crate recall test unable to fail.

use sofuu_core::memory::hnsw::{Hnsw, SelectPolicy, HNSW_EF_SEARCH, HNSW_SMALL_GRAPH};
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f32) / ((1u64 << 53) as f32)
    }
    fn vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.unit()).collect()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fixture {
    /// Uniform random in [0,1)^dim — no cluster structure to hide a bad
    /// graph behind, the hardest of the two fixtures.
    Uniform,
    /// `clusters` gaussian-ish blobs of `n/clusters` members each, with
    /// per-dimension noise of +-`sigma` in ALL `dim` dimensions. The C
    /// harness used sigma = 0.02 with 100 members per cluster; the default
    /// here is deliberately harder (sigma 0.3, ~50 members per cluster).
    /// NOTE: isotropic noise in all 128 dimensions is much HARDER than a
    /// real embedding — see `Manifold`.
    Clustered,
    /// Realistic embeddings: the same clusters, but the centres and the
    /// within-cluster noise live in a `subdim`-dimensional subspace of the
    /// `dim`-dimensional space, which is what a sentence embedding's
    /// intrinsic dimension actually looks like (~16-32, not 128).
    /// Distances concentrate far less, so the neighbourhoods an HNSW graph
    /// can navigate are the ones real corpora have.
    Manifold,
}

struct Args {
    n: usize,
    dim: usize,
    queries: usize,
    k: usize,
    ef: usize,
    fixture: Fixture,
    clusters: usize,
    sigma: f32,
    subdim: usize,
    policies: Vec<SelectPolicy>,
    tombstone: f64,
    seed: u64,
    shuffle: bool,
    query_mode: QueryMode,
    /// `--ef-sweep 16,32,64,128,256`: after each policy's graph is built,
    /// score recall@`k` at every one of these frontier widths. This is the
    /// §W4 measurement that separates "the graph is bad" from "the
    /// threshold was set on the wrong fixture": recall that keeps climbing
    /// with `ef` is an approximate-search trade, recall that plateaus below
    /// the target is a graph defect.
    ef_sweep: Vec<usize>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            n: 2000,
            dim: 128,
            queries: 50,
            k: 10,
            ef: HNSW_EF_SEARCH,
            fixture: Fixture::Uniform,
            clusters: 40,
            sigma: 0.3,
            subdim: 16,
            policies: vec![
                SelectPolicy::Heuristic,
                SelectPolicy::Nearest,
                SelectPolicy::LegacyRandom,
            ],
            tombstone: 0.5,
            seed: 0x5EED_0BEE_1234_5678,
            shuffle: true,
            query_mode: QueryMode::InDistribution,
            ef_sweep: Vec::new(),
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = argv.get(i + 1).cloned().unwrap_or_default();
        match key {
            "--n" => a.n = val.parse().expect("--n"),
            "--dim" => a.dim = val.parse().expect("--dim"),
            "--queries" => a.queries = val.parse().expect("--queries"),
            "--k" => a.k = val.parse().expect("--k"),
            "--ef" => a.ef = val.parse().expect("--ef"),
            "--ef-sweep" => {
                a.ef_sweep = val
                    .split(',')
                    .map(|e| e.trim().parse().expect("--ef-sweep item"))
                    .collect()
            }
            "--clusters" => a.clusters = val.parse().expect("--clusters"),
            "--sigma" => a.sigma = val.parse().expect("--sigma"),
            "--subdim" => a.subdim = val.parse().expect("--subdim"),
            "--tombstone" => a.tombstone = val.parse().expect("--tombstone"),
            "--query-mode" => match val.as_str() {
                "ind" | "distribution" => a.query_mode = QueryMode::InDistribution,
                "uniform" => a.query_mode = QueryMode::UniformProbe,
                other => panic!("unknown --query-mode {other}"),
            },
            "--order" => match val.as_str() {
                "shuffled" => a.shuffle = true,
                "generated" => a.shuffle = false,
                other => panic!("unknown --order {other}"),
            },
            "--seed" => {
                a.seed = u64::from_str_radix(val.trim_start_matches("0x"), 16)
                    .or_else(|_| val.parse::<u64>().map_err(|_| ()))
                    .expect("--seed")
            }
            "--fixture" => {
                a.fixture = match val.as_str() {
                    "uniform" => Fixture::Uniform,
                    "clustered" => Fixture::Clustered,
                    "manifold" => Fixture::Manifold,
                    other => panic!("unknown --fixture {other}"),
                }
            }
            "--policies" => {
                a.policies = val
                    .split(',')
                    .map(|p| match p.trim() {
                        "heuristic" => SelectPolicy::Heuristic,
                        "nearest" => SelectPolicy::Nearest,
                        "legacy" | "random" => SelectPolicy::LegacyRandom,
                        other => panic!("unknown policy {other}"),
                    })
                    .collect()
            }
            "--help" | "-h" => {
                println!(
                    "usage: hnsw_recall_eval [--n N] [--dim D] [--queries Q] [--k K] [--ef EF]\n\
                     \x20      [--ef-sweep 16,32,64,128,256] [--fixture uniform|clustered|manifold]\n\
                     \x20      [--clusters C] [--sigma S] [--policies heuristic,nearest,legacy]\n\
                     \x20      [--tombstone FRAC] [--seed S] [--query-mode ind|uniform]\n\
                     \x20      [--order shuffled|generated] [--subdim D]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown flag {other} (try --help)"),
        }
        i += 2;
    }
    a
}

fn policy_name(p: SelectPolicy) -> &'static str {
    match p {
        SelectPolicy::Heuristic => "heuristic",
        SelectPolicy::Nearest => "nearest",
        SelectPolicy::LegacyRandom => "legacy",
    }
}

/// How a probe is drawn. The default is the standard ANN-benchmark
/// protocol: a query is a fresh draw from the SAME distribution as the
/// corpus (a record that could have been stored, but was not), so it is a
/// near-miss of real content rather than either a duplicate of a stored
/// vector or an off-distribution point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum QueryMode {
    InDistribution,
    /// Off-distribution: uniform in the cube regardless of the corpus.
    /// Deliberate stress mode — with clusters it puts the probe at a
    /// near-equal distance from several of them, so its true top-10 spans
    /// clusters and measures the fixture's hardness, not the index.
    UniformProbe,
}

struct Corpus {
    /// corpus[i] is the vector stored under id i.
    vecs: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

fn build_corpus(a: &Args) -> Corpus {
    let mut rng = Rng::new(a.seed);
    let mut centers: Vec<Vec<f32>> = Vec::new();
    let mut basis: Vec<Vec<f32>> = Vec::new();
    match a.fixture {
        Fixture::Uniform => {}
        Fixture::Clustered => {
            centers = (0..a.clusters.max(1)).map(|_| rng.vec(a.dim)).collect();
        }
        Fixture::Manifold => {
            let sd = a.subdim.min(a.dim).max(1);
            for _ in 0..sd {
                let raw = rng.vec(a.dim);
                let n = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                basis.push(raw.iter().map(|x| x / n).collect());
            }
            centers = (0..a.clusters.max(1))
                .map(|_| {
                    let mut c = vec![0.0f32; a.dim];
                    for b in &basis {
                        let z = 2.0 * rng.unit() - 1.0;
                        for (d, bd) in c.iter_mut().zip(b) {
                            *d += z * bd;
                        }
                    }
                    c
                })
                .collect();
        }
    }
    let mut sample = |rng: &mut Rng| -> Vec<f32> {
        match a.fixture {
            Fixture::Uniform => rng.vec(a.dim),
            Fixture::Clustered => {
                let c = &centers[(rng.next_u64() as usize) % centers.len()];
                c.iter()
                    .map(|x| x + a.sigma * (2.0 * rng.unit() - 1.0))
                    .collect()
            }
            Fixture::Manifold => {
                let c = &centers[(rng.next_u64() as usize) % centers.len()];
                let mut v = c.clone();
                for b in &basis {
                    let z = a.sigma * (2.0 * rng.unit() - 1.0);
                    for (d, bd) in v.iter_mut().zip(b) {
                        *d += z * bd;
                    }
                }
                v
            }
        }
    };

    let mut vecs: Vec<Vec<f32>> = (0..a.n).map(|_| sample(&mut rng)).collect();
    /* Insertion order: real embeddings arrive in arbitrary order, while a
     * clustered fixture generated cluster-by-cluster would feed every
     * member of one blob before the next — the worst case for graph
     * construction, and not what a brain actually does. Deterministic
     * Fisher-Yates, so the run stays reproducible. ids are insertion ids:
     * corpus[i] is the vector stored under id i. */
    if a.shuffle {
        for i in (1..vecs.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            vecs.swap(i, j);
        }
    }
    let queries: Vec<Vec<f32>> = match a.query_mode {
        QueryMode::InDistribution => (0..a.queries).map(|_| sample(&mut rng)).collect(),
        QueryMode::UniformProbe => (0..a.queries).map(|_| rng.vec(a.dim)).collect(),
    };
    Corpus { vecs, queries }
}

/// Exact top-`kt` neighbours of each query over `allowed` ids (all ids when
/// `None`), by full scan. This is the ground truth the index is scored on.
fn ground_truth(
    vecs: &[Vec<f32>],
    queries: &[Vec<f32>],
    kt: usize,
    allowed: Option<&[u32]>,
) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| {
            let mut all: Vec<(f32, u32)> = match allowed {
                Some(ids) => ids
                    .iter()
                    .map(|&id| {
                        let v = &vecs[id as usize];
                        let d: f32 = q.iter().zip(v).map(|(x, y)| (x - y) * (x - y)).sum();
                        (d, id)
                    })
                    .collect(),
                None => vecs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let d: f32 = q.iter().zip(v).map(|(x, y)| (x - y) * (x - y)).sum();
                        (d, i as u32)
                    })
                    .collect(),
            };
            all.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
            all.into_iter().take(kt).map(|(_, id)| id).collect()
        })
        .collect()
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

/// Directed reachability from the enter point over level-0 out-links: a
/// node that cannot be reached that way can never be returned by a search,
/// whatever `ef` is. Also counts nodes with no inbound link at all.
fn reachability(h: &Hnsw) -> (f64, usize) {
    let n = h.len();
    if n == 0 {
        return (0.0, 0);
    }
    let mut seen = vec![false; n];
    let mut stack: Vec<u32> = vec![0];
    seen[0] = true;
    let mut reached = 1usize;
    while let Some(id) = stack.pop() {
        for &nb in h.neighbors_l0(id) {
            let i = nb as usize;
            if i < n && !seen[i] {
                seen[i] = true;
                reached += 1;
                stack.push(nb);
            }
        }
    }
    let mut zero_in = 0usize;
    let mut indeg = vec![0usize; n];
    for id in 0..n as u32 {
        for &nb in h.neighbors_l0(id) {
            indeg[nb as usize] += 1;
        }
    }
    for d in &indeg {
        if *d == 0 {
            zero_in += 1;
        }
    }
    (reached as f64 / n as f64, zero_in)
}

/// recall@`k` and latency at ONE frontier width, with the search asked for
/// exactly `k` results so that `search_with_ef`'s `ef = ef.max(k)` clamp does
/// not silently widen a small-`ef` probe. This is the ef-sweep measurement.
fn score_at_ef(
    h: &Hnsw,
    truth: &[Vec<u32>],
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    let mut hits = 0usize;
    let mut total = 0usize;
    let mut lat: Vec<f64> = Vec::with_capacity(queries.len());
    for (q, want) in queries.iter().zip(truth) {
        let t0 = Instant::now();
        let got: Vec<u32> = h
            .search_with_ef(q, k, ef)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
        let want: Vec<u32> = want.iter().take(k).copied().collect();
        for w in &want {
            total += 1;
            if got.contains(w) {
                hits += 1;
            }
        }
    }
    lat.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    (
        hits as f64 / total.max(1) as f64,
        pct(&lat, 0.50),
        pct(&lat, 0.95),
    )
}

struct Report {
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_50: f64,
    build_ms: f64,
    p50_us: f64,
    p95_us: f64,
    degree: f64,
    reverse_closed: f64,
    reachable: f64,
    zero_in: usize,
}

fn score(
    h: &Hnsw,
    truth: &[Vec<u32>],
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64, f64, f64) {
    let kt = truth.first().map(|t| t.len()).unwrap_or(0);
    let mut hit1 = 0usize;
    let mut hit10 = 0usize;
    let mut hit50 = 0usize;
    let mut lat: Vec<f64> = Vec::with_capacity(queries.len());
    for (q, want) in queries.iter().zip(truth) {
        let t0 = Instant::now();
        let got: Vec<u32> = h
            .search_with_ef(q, kt.max(k), ef)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
        let has = |id: u32, n: usize| want.iter().take(n).any(|w| *w == id);
        if got.first().is_some_and(|id| has(*id, 1)) {
            hit1 += 1;
        }
        hit10 += got.iter().take(10).filter(|&&id| has(id, 10)).count();
        hit50 += got.iter().take(50).filter(|&&id| has(id, 50)).count();
    }
    lat.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let nq = queries.len().max(1) as f64;
    (
        hit1 as f64 / nq,
        hit10 as f64 / (nq * 10.0),
        hit50 as f64 / (nq * 50.0),
        pct(&lat, 0.50),
        pct(&lat, 0.95),
    )
}

fn structure(h: &Hnsw) -> (f64, f64) {
    let nodes = h.len().max(1);
    let mut degree = 0usize;
    let mut edges = 0usize;
    let mut closed = 0usize;
    for id in 0..h.len() as u32 {
        degree += h.degree_l0(id);
        for &nb in h.neighbors_l0(id) {
            edges += 1;
            if h.neighbors_l0(nb).contains(&id) {
                closed += 1;
            }
        }
    }
    (
        degree as f64 / nodes as f64,
        closed as f64 / edges.max(1) as f64,
    )
}

fn measure(
    a: &Args,
    vecs: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth_all: &[Vec<u32>],
    policy: SelectPolicy,
) -> (Hnsw, Report) {
    let mut h = Hnsw::new_with_policy(a.dim, policy);
    let t0 = Instant::now();
    for v in vecs {
        h.add_vector(v);
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (recall_at_1, recall_at_10, recall_at_50, p50_us, p95_us) =
        score(&h, truth_all, queries, a.k, a.ef);
    let (degree, reverse_closed) = structure(&h);
    let (reachable, zero_in) = reachability(&h);
    (
        h,
        Report {
            recall_at_1,
            recall_at_10,
            recall_at_50,
            build_ms,
            p50_us,
            p95_us,
            degree,
            reverse_closed,
            reachable,
            zero_in,
        },
    )
}

fn main() {
    let a = parse_args();
    let kt = a.k.max(50);
    println!(
        "── HNSW recall eval ── n={} dim={} queries={} k={} ef={} fixture={} qmode={} seed=0x{:x}",
        a.n,
        a.dim,
        a.queries,
        a.k,
        a.ef,
        match a.fixture {
            Fixture::Uniform => "uniform",
            Fixture::Clustered => "clustered",
            Fixture::Manifold => "manifold",
        },
        match a.query_mode {
            QueryMode::InDistribution => "ind",
            QueryMode::UniformProbe => "uniform",
        },
        a.seed
    );
    if a.n <= HNSW_SMALL_GRAPH {
        println!(
            "!! n={} <= HNSW_SMALL_GRAPH={HNSW_SMALL_GRAPH}: the search is EXHAUSTIVE here, \
             so recall is ~1.0 by construction and this run says nothing about the graph",
            a.n
        );
    }
    let corpus = build_corpus(&a);
    let (vecs, queries) = (&corpus.vecs, &corpus.queries);
    let t = Instant::now();
    let truth_all = ground_truth(vecs, queries, kt, None);
    println!(
        "brute-force ground truth: {} queries x {} vectors in {:.0} ms",
        queries.len(),
        a.n,
        t.elapsed().as_secs_f64() * 1e3
    );

    let mut built: Vec<(SelectPolicy, Hnsw, Report)> = Vec::new();
    for &policy in &a.policies {
        let (h, r) = measure(&a, vecs, queries, &truth_all, policy);
        println!(
            "POLICY={:<9} n={} dim={} build={:.0}ms degree_l0={:.1} reverse_closed={:.3} \
             reachable={:.4} indeg0={} recall@1={:.4} recall@10={:.4} recall@50={:.4}              p50={:.1}us p95={:.1}us",
            policy_name(policy),
            a.n,
            a.dim,
            r.build_ms,
            r.degree,
            r.reverse_closed,
            r.reachable,
            r.zero_in,
            r.recall_at_1,
            r.recall_at_10,
            r.recall_at_50,
            r.p50_us,
            r.p95_us
        );
        built.push((policy, h, r));
    }

    /* §W4 measurement: recall@k as a function of the search frontier. The
     * question it answers is whether a miss is the GRAPH (recall plateaus
     * far below 1.0 no matter how wide the frontier) or the ef/recall TRADE
     * (recall climbs monotonically towards 1.0). */
    if !a.ef_sweep.is_empty() {
        println!();
        println!("EF-SWEEP fixture={} n={} dim={} k={}", match a.fixture {
            Fixture::Uniform => "uniform",
            Fixture::Clustered => "clustered",
            Fixture::Manifold => "manifold",
        }, a.n, a.dim, a.k);
        println!("{:<10} {:>6} {:>12} {:>10} {:>10}", "policy", "ef", format!("recall@{}", a.k), "p50_us", "p95_us");
        for (policy, h, _) in &built {
            for &ef in &a.ef_sweep {
                let (rec, p50, p95) = score_at_ef(h, &truth_all, queries, a.k, ef);
                println!(
                    "{:<10} {:>6} {:>12.4} {:>10.1} {:>10.1}",
                    policy_name(*policy),
                    ef,
                    rec,
                    p50,
                    p95
                );
            }
        }
        println!();
    }

    /* W4.4: tombstone stage — delete a fraction, then score against brute
     * force over the LIVE set. Uses the first graph (the heuristic one in
     * the default policy list) and a separate deterministic stream so the
     * deleted subset does not depend on how much RNG the build consumed. */
    if a.tombstone > 0.0 {
        let (policy, h, _) = built.first_mut().expect("at least one policy");
        let mut rng = Rng::new(a.seed ^ 0x7024_5017_EEE5_1234);
        let mut live: Vec<u32> = Vec::with_capacity(a.n);
        for id in 0..a.n as u32 {
            if (rng.unit() as f64) < a.tombstone {
                h.tombstone(id);
            } else {
                live.push(id);
            }
        }
        let truth_live = ground_truth(vecs, queries, kt, Some(&live));
        let (r1, r10, r50, p50, p95) = score(h, &truth_live, &queries, a.k, a.ef);
        let leaked = queries.iter().filter(|q| {
            h.search_with_ef(q, a.k.max(50), a.ef)
                .iter()
                .any(|(id, _)| h.is_tombstoned(*id))
        }).count();
        println!(
            "TOMBSTONE policy={} deleted={:.2} live={} recall@1={r1:.4} recall@10={r10:.4} \
             recall@50={r50:.4} p50={p50:.1}us p95={p95:.1}us tombstones_in_results={}",
            policy_name(*policy),
            a.tombstone,
            live.len(),
            leaked
        );
    }
}
