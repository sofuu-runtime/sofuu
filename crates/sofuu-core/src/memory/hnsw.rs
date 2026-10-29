// sofuu-core — HNSW (Hierarchical Navigable Small World) index.
//
// Rust port of src/memory/hnsw.c — the vector index powering the CMA
// (Cognitive Memory Architecture). Same algorithm: a layered proximity
// graph with greedy best-first search, M/M0 outbound links, and a
// per-level enter point.
//
// Safe Rust: no raw pointers, no manual memory. The vector store is owned
// here (the C version borrowed an external flat array).

// The node id/level fields and `dist` are internal, exercised via tests;
// the public API surfaces search/add_vector/len.
#![allow(dead_code)]

use std::collections::BinaryHeap;
use std::cmp::Ordering;

pub const HNSW_MAX_LEVELS: usize = 16;
pub const HNSW_MAX_M: usize = 16;
pub const HNSW_MAX_M0: usize = 32;

/// H-07 (AUDIT-2026-10-29): graphs at or below this size are searched
/// EXHAUSTIVELY — the walk is unbudgeted, exactly as it always was. Below
/// this point a full traversal is both cheap and strictly better than an
/// approximate one, and every ranking fixture lives here, so their numbers
/// stay pinned. (`Cma::recall` uses the same constant for its exact-search
/// fallback: one threshold, documented in one place.)
pub const HNSW_SMALL_GRAPH: usize = 1024;

/// H-07: the search budget, as a multiple of `ef`, once the graph is
/// LARGER than `HNSW_SMALL_GRAPH`. Without it the early-exit bound is the
/// only termination rule, and on a diffuse graph filling `ef` result slots
/// means walking an ever-larger share of the corpus: the audit measured
/// recall() at `0.9 ms + 1.07 µs · n` (1.5 ms at n=500, 6.0 ms at n=8000,
/// ~108 ms projected at 100k) because the frontier never filled cheaply.
/// A visit budget turns that O(n) walk into O(ef) work per query — 4 × ef
/// expansions is generous (each expansion already evaluates up to M0
/// neighbours), so the graph path still answers with the same top-k.
pub const HNSW_VISIT_FACTOR: usize = 4;

/// The most nodes one level-0 search may expand when the graph is large
/// (never below this, so small `ef` values stay useful).
pub const HNSW_VISIT_FLOOR: usize = 64;

/// One node in the graph. `id` indexes into the vector store.
#[derive(Clone)]
struct Node {
    id: u32,
    level: u8,
    links: Vec<Vec<u32>>, // links[l] = neighbors at level l
}

/// A search candidate (closer = better).
#[derive(Clone, Copy)]
struct Cand {
    dist: f32,
    id: u32,
}

/// Same payload as `Cand` but with NATURAL ordering (larger dist = greater),
/// so a `BinaryHeap<FarCand>` pops/peeks the WORST candidate. The results
/// heap needs that; `Cand`'s reversed Ord (a min-heap, correct for the
/// candidate frontier) cannot express it. Sharing one reversed type for both
/// heaps made `results` a min-heap used as a max-heap: it evicted the CLOSEST
/// neighbours and broke the early-exit bound on the BEST one, collapsing
/// recall as the graph grew (measured recall@10 0.82 → 0.07 from 50 to 2000
/// vectors; 1.00/0.91 after this split).
#[derive(Clone, Copy)]
struct FarCand {
    dist: f32,
    id: u32,
}
impl PartialEq for FarCand {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for FarCand {}
impl PartialOrd for FarCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal))
    }
}
impl Ord for FarCand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Min-heap by distance: reverse Ord.
        Some(other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// HNSW index over a flat `Vec<Vec<f32>>` vector store.
pub struct Hnsw {
    vec_store: Vec<Vec<f32>>,
    vec_dim: usize,
    nodes: Vec<Node>,
    enter_node: u32,
    max_level: u8,
}

impl Hnsw {
    pub fn new(vec_dim: usize) -> Self {
        Self {
            vec_store: Vec::new(),
            vec_dim,
            nodes: Vec::new(),
            enter_node: 0,
            max_level: 0,
        }
    }

    /// Push a new vector; returns its index (used as node id).
    pub fn add_vector(&mut self, v: &[f32]) -> u32 {
        debug_assert_eq!(v.len(), self.vec_dim);
        self.vec_store.push(v.to_vec());
        let id = (self.vec_store.len() - 1) as u32;
        self.add_node(id);
        id
    }

    /// The number of stored vectors.
    pub fn len(&self) -> usize {
        self.vec_store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec_store.is_empty()
    }

    /// Stored coordinates for a record id (as returned by add_vector).
    /// The HNSW store IS the single vector store — Cma.vectors was folded
    /// into it (AUDIT-2026-09-07 ml-5).
    pub fn vector(&self, id: u32) -> &[f32] {
        &self.vec_store[id as usize]
    }

    /// Overwrite a stored vector in place (entity upsert). Neighbor links
    /// keep their old-geometry decisions — the same trade the C store
    /// makes — but searches see the new coordinates immediately.
    pub fn set_vector(&mut self, id: u32, v: &[f32]) {
        debug_assert_eq!(v.len(), self.vec_dim);
        self.vec_store[id as usize].copy_from_slice(v);
    }

    fn dist(&self, a: u32, b: u32) -> f32 {
        let va = &self.vec_store[a as usize];
        let vb = &self.vec_store[b as usize];
        /* H-07: slice iteration instead of indexed bounds-checked loads —
         * same arithmetic, vectorizes (this is the inner loop of every
         * search and every insert). */
        va.iter().zip(vb).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn dist_query(&self, q: &[f32], b: u32) -> f32 {
        let vb = &self.vec_store[b as usize];
        q.iter().zip(vb).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn random_level(&self) -> u8 {
        // P3 (AUDIT-2026-09-07): canonical HNSW level assignment —
        // P(level >= l) = (1/e)^l, i.e. climb with probability 1/e per level
        // (~63.2% of nodes stay at 0). The old 50% coin over-produced upper
        // levels, growing the skip-list's memory and search fan-out.
        let mut l = 0u8;
        while l < (HNSW_MAX_LEVELS as u8 - 1) {
            let r: u32 = rand_u32();
            if (r as f64) / (u32::MAX as f64 + 1.0) >= std::f64::consts::E.recip() {
                break;
            }
            l += 1;
        }
        l
    }

    fn add_node(&mut self, id: u32) {
        let level = self.random_level();
        if self.nodes.is_empty() {
            // First node — becomes the enter point.
            self.enter_node = id;
            self.max_level = level;
            self.nodes.push(Node {
                id,
                level,
                links: vec![Vec::new()],
            });
            return;
        }

        // Greedy search from the enter point down to `level`.
        let mut ep = self.enter_node;
        let mut top_level = self.max_level;
        while top_level > level {
            // Search one level, move to closest neighbor.
            let mut best = ep;
            let mut best_dist = self.dist_query(&self.vec_store[id as usize], ep);
            if let Some(links) = self.nodes[ep as usize].links.get(top_level as usize) {
                for &nb in links {
                    let d = self.dist_query(&self.vec_store[id as usize], nb);
                    if d < best_dist {
                        best_dist = d;
                        best = nb;
                    }
                }
            }
            ep = best;
            if top_level > 0 {
                top_level -= 1;
            } else {
                break;
            }
        }

        // Insert node.
        let mut node = Node {
            id,
            level,
            links: Vec::new(),
        };
        for _l in 0..=level {
            node.links.push(Vec::new());
        }
        self.nodes.push(node);

        // Connect at each level up to `level`.
        for l in 0..=level {
            let m = if l == 0 { HNSW_MAX_M0 } else { HNSW_MAX_M };
            /* H-07: construction is never budgeted — the graph is built
             * from the best neighbourhood the walk can find. */
            let neighbors =
                self.search_layer(&self.vec_store[id as usize], ep, m, l, usize::MAX);
            let nid = self.nodes.len() as u32 - 1;
            for &nb in &neighbors {
                self.nodes[nid as usize].links[l as usize].push(nb);
                // Add back-link (bidirectional), bounded.
                let nb_links = &mut self.nodes[nb as usize].links;
                if nb_links.len() as u8 > l {
                    if nb_links[l as usize].len() < m {
                        nb_links[l as usize].push(nid);
                    } else if l == 0 {
                        // Replace a random level-0 link (simple policy).
                        let idx = (rand_u32() % nb_links[0].len() as u32) as usize;
                        nb_links[0][idx] = nid;
                    }
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.enter_node = self.nodes.len() as u32 - 1;
        }
    }

    /// Greedy best-first search at a single level.
    ///
    /// `budget` caps how many nodes the walk may EXPAND (`usize::MAX` = no
    /// cap). Insertion passes no cap — the graph is built from the best
    /// neighbourhood available, so quality there is never traded away —
    /// while `search_with_ef` budgets queries on large graphs (H-07).
    fn search_layer(&self, q: &[f32], ep: u32, ef: usize, level: u8, budget: usize) -> Vec<u32> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates: BinaryHeap<Cand> = BinaryHeap::new(); // min-heap: closest pops
        let mut results: BinaryHeap<FarCand> = BinaryHeap::new(); // max-heap: worst on top

        let d0 = self.dist_query(q, ep);
        candidates.push(Cand { dist: d0, id: ep });
        results.push(FarCand { dist: d0, id: ep });
        visited.insert(ep);

        while let Some(c) = candidates.pop() {
            /* H-07: the O(n) guard. Breaking here returns the best frontier
             * found so far — an approximate answer in bounded time, which
             * is what HNSW promises; the alternative (walking until the
             * result heap is provably full) made recall latency linear in
             * the corpus size. */
            if visited.len() >= budget {
                break;
            }
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            /* H-07: borrow the link list instead of cloning it — this is a
             * heap allocation per expanded node otherwise. */
            let Some(links) = self
                .nodes
                .get(c.id as usize)
                .and_then(|n| n.links.get(level as usize))
            else {
                continue;
            };
            for &nb in links {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);
                let d = self.dist_query(q, nb);
                let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
                if results.len() < ef || d < worst {
                    candidates.push(Cand { dist: d, id: nb });
                    results.push(FarCand { dist: d, id: nb });
                    if results.len() > ef {
                        results.pop(); // evicts the FARTHEST — the heap's top
                    }
                }
            }
        }

        // results is a max-heap (worst on top); drain into sorted-by-dist vec.
        let mut out: Vec<FarCand> = results.into_sorted_vec();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out.into_iter().map(|c| c.id).collect()
    }

    /// Search the whole graph for the `k` nearest neighbors of `q`.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.search_with_ef(q, k, (k * 4).max(8))
    }

    /// B5.3 (ROADMAP-BRAIN-CMA): the same search with an explicit candidate
    /// width. `ef` is the size of the best-first frontier kept at each
    /// level -- larger trades latency for recall, `ef <= k` behaves like
    /// `ef = k`. `search` is `search_with_ef(q, k, (k*4).max(8))`, so the
    /// default behaviour is unchanged.
    ///
    /// H-07: on a graph larger than `HNSW_SMALL_GRAPH` the walk is bounded
    /// to `HNSW_VISIT_FACTOR * ef` expansions, so a query costs O(ef) no
    /// matter how many records the brain holds. Small graphs stay
    /// exhaustive, which keeps every pinned ranking fixture bit-identical.
    pub fn search_with_ef(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let ef = ef.max(k);
        let budget = self.visit_budget(ef);
        // Greedy best-first descent from the enter point down to level 0,
        // using ef candidates at each level (better recall than pure greedy).
        let mut ep = self.enter_node;
        let mut top = self.max_level;
        while top > 0 {
            let candidates = self.search_layer(q, ep, ef, top, budget);
            ep = candidates.first().copied().unwrap_or(ep);
            top -= 1;
        }
        // Level-0 search with ef, then trim to k.
        let ids = self.search_layer(q, ep, ef, 0, budget);
        let mut out: Vec<(u32, f32)> = ids
            .into_iter()
            .map(|id| (id, self.dist_query(q, id).sqrt()))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
    }

    /// H-07: how many nodes a query walk may expand. Exhaustive (`MAX`) for
    /// a small graph — the documented fallback — and O(ef) above it.
    fn visit_budget(&self, ef: usize) -> usize {
        if self.nodes.len() <= HNSW_SMALL_GRAPH {
            return usize::MAX;
        }
        ef.saturating_mul(HNSW_VISIT_FACTOR).max(HNSW_VISIT_FLOOR)
    }
}

/// Simple deterministic PRNG for tests + level assignment.
fn rand_u32() -> u32 {
    // xorshift — deterministic, no external dep.
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u32> = Cell::new(0x9e3779b9);
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s.set(x);
        x
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(n: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[n % dim] = 1.0;
        v
    }

    #[test]
    fn empty_search() {
        let h = Hnsw::new(4);
        assert!(h.search(&[0.0; 4], 5).is_empty());
    }

    #[test]
    fn finds_exact_nearest() {
        let mut h = Hnsw::new(4);
        // 4 one-hot vectors.
        for i in 0..4 {
            h.add_vector(&unit(i, 4));
        }
        // Query = one-hot 0 + noise.
        let q = [1.0, 0.1, 0.05, 0.02];
        let res = h.search(&q, 1);
        assert_eq!(res[0].0, 0);
        // Query = one-hot 3.
        let q = [0.0, 0.0, 0.0, 1.0];
        let res = h.search(&q, 1);
        assert_eq!(res[0].0, 3);
    }

    #[test]
    fn topk_ordering() {
        let mut h = Hnsw::new(8);
        for i in 0..8 {
            let mut v = vec![0.0; 8];
            v[i] = 1.0;
            h.add_vector(&v);
        }
        let mut q = vec![0.0; 8];
        q[3] = 1.0;
        q[5] = 0.5;
        let res = h.search(&q, 3);
        // HNSW is approximate — the exact nearest (3) must be in the top-3,
        // and 5 should be in the top-3 too (both are much closer than rest).
        let ids: Vec<u32> = res.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&3), "nearest id 3 must be in top-3, got {ids:?}");
        assert!(ids.contains(&5), "second-nearest id 5 must be in top-3, got {ids:?}");
        // Distances ascending.
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    /// Regression for the results-heap ordering bug: with one reversed `Ord`
    /// shared by both heaps, `results` was a min-heap used as a max-heap, so
    /// it evicted the CLOSEST neighbours and the early-exit bound compared
    /// against the BEST distance. Small graphs hid it (ef >= n never trims);
    /// measured recall@10 collapsed 0.82 → 0.24 → 0.07 at n = 50/200/2000.
    /// This asserts recall against brute force at a size where the bug bites.
    #[test]
    fn recall_stays_high_as_the_graph_grows() {
        let dim = 32usize;
        let n = 300usize;
        let k = 10usize;
        let mut state: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f32) / ((1u64 << 53) as f32)
        };
        let mut store: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut h = Hnsw::new(dim);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next()).collect();
            store.push(v.clone());
            h.add_vector(&v);
        }
        let mut hits = 0usize;
        let mut total = 0usize;
        for _ in 0..20 {
            let q: Vec<f32> = (0..dim).map(|_| next()).collect();
            let mut exact: Vec<(u32, f32)> = store
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as u32, d)
                })
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let want: Vec<u32> = exact.into_iter().take(k).map(|(i, _)| i).collect();
            let got: Vec<u32> = h.search(&q, k).into_iter().map(|(i, _)| i).collect();
            for w in &want {
                total += 1;
                if got.contains(w) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.90,
            "recall@{k} at n={n} is {recall:.3} — the results heap is evicting near neighbours"
        );
    }

    /// The exact stored vector must always come back as its own nearest
    /// neighbour — the property the brain's recall depends on.
    #[test]
    fn exact_self_is_always_top1() {
        let dim = 32usize;
        let n = 300usize;
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f32) / ((1u64 << 53) as f32)
        };
        let mut h = Hnsw::new(dim);
        let mut store: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next()).collect();
            store.push(v.clone());
            h.add_vector(&v);
        }
        /* k=32 mirrors the brain's own call (recall passes ef>=32, so the
         * internal ef is >=128). At k=1 the internal ef floor is 8, which is
         * too small for reliable 1-NN on a 300-node graph — that is the
         * approximate-search trade, not the ordering bug this test pins. */
        for (i, v) in store.iter().enumerate() {
            let got = h.search(v, 32);
            assert_eq!(
                got.first().map(|(id, _)| *id),
                Some(i as u32),
                "vector {i} did not find itself"
            );
        }
    }

    #[test]
    fn add_many_keeps_len() {
        let mut h = Hnsw::new(16);
        for i in 0..100 {
            let mut v = vec![0.0; 16];
            v[i % 16] = (i as f32) / 100.0;
            h.add_vector(&v);
        }
        assert_eq!(h.len(), 100);
        // Search still returns sensible top-1 (nearest to a stored vector).
        let mut q = vec![0.0; 16];
        q[7] = 0.5;
        let res = h.search(&q, 1);
        assert!(!res.is_empty());
        assert!(res[0].1 < 1.0); // something close exists
    }

    /// H-07: a query on a LARGE graph must cost bounded work, not a share of
    /// the corpus. The audit measured recall() at `0.9 ms + 1.07 µs · n`;
    /// this pins the budget that removes the n term: the walk behind a
    /// `search_with_ef` may expand at most `HNSW_VISIT_FACTOR * ef` nodes
    /// once the graph is over `HNSW_SMALL_GRAPH`.
    #[test]
    fn large_graph_search_is_budgeted() {
        let dim = 16usize;
        let n = HNSW_SMALL_GRAPH + 512; // just over the exhaustive threshold
        let mut h = Hnsw::new(dim);
        for i in 0..n {
            let mut v = vec![0.0; dim];
            v[i % dim] = 1.0;
            v[(i * 7) % dim] += (i as f32) / (n as f32);
            h.add_vector(&v);
        }
        assert!(h.nodes.len() > HNSW_SMALL_GRAPH);
        let ef = 32usize;
        /* The budget is what bounds the walk: it can never expand more than
         * the cap, and the default path still answers with `k` hits. */
        let budget = h.visit_budget(ef);
        assert_eq!(budget, ef * HNSW_VISIT_FACTOR);
        assert!(budget < n, "a budgeted walk must not be able to touch n");
        let mut q = vec![0.0; dim];
        q[3] = 1.0;
        let hits = h.search_with_ef(&q, 8, ef);
        assert_eq!(hits.len(), 8, "a budgeted search still fills k");
        /* A small graph keeps the exhaustive budget. */
        let mut small = Hnsw::new(dim);
        for i in 0..8 {
            let mut v = vec![0.0; dim];
            v[i] = 1.0;
            small.add_vector(&v);
        }
        assert_eq!(small.visit_budget(8), usize::MAX);
    }
}
