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

    fn dist(&self, a: u32, b: u32) -> f32 {
        let va = &self.vec_store[a as usize];
        let vb = &self.vec_store[b as usize];
        let mut s = 0.0f32;
        for i in 0..self.vec_dim {
            let d = va[i] - vb[i];
            s += d * d;
        }
        s // squared L2 — monotonic with L2, matches C (L2 used in search)
    }

    fn dist_query(&self, q: &[f32], b: u32) -> f32 {
        let vb = &self.vec_store[b as usize];
        let mut s = 0.0f32;
        for i in 0..self.vec_dim {
            let d = q[i] - vb[i];
            s += d * d;
        }
        s
    }

    fn random_level(&self) -> u8 {
        // Geometric-ish level assignment: level 0 most likely, capped.
        let mut l = 0u8;
        while l < (HNSW_MAX_LEVELS as u8 - 1) {
            // ~50% chance to go up a level (matches C's simple rand gate).
            let r: u32 = rand_u32();
            if r % 2 == 0 {
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
            let neighbors = self.search_layer(&self.vec_store[id as usize], ep, m, l);
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
    fn search_layer(&self, q: &[f32], ep: u32, ef: usize, level: u8) -> Vec<u32> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates: BinaryHeap<Cand> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new(); // max-heap by dist (worst on top)

        let d0 = self.dist_query(q, ep);
        candidates.push(Cand { dist: d0, id: ep });
        results.push(Cand { dist: d0, id: ep });
        visited.insert(ep);

        while let Some(c) = candidates.pop() {
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            let links = self
                .nodes
                .get(c.id as usize)
                .and_then(|n| n.links.get(level as usize))
                .cloned()
                .unwrap_or_default();
            for nb in links {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);
                let d = self.dist_query(q, nb);
                let worst = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
                if results.len() < ef || d < worst {
                    candidates.push(Cand { dist: d, id: nb });
                    results.push(Cand { dist: d, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // results is a max-heap (worst on top); drain into sorted-by-dist vec.
        let mut out: Vec<Cand> = results.into_sorted_vec();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out.into_iter().map(|c| c.id).collect()
    }

    /// Search the whole graph for the `k` nearest neighbors of `q`.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        // Greedy best-first descent from the enter point down to level 0,
        // using ef candidates at each level (better recall than pure greedy).
        let mut ep = self.enter_node;
        let mut top = self.max_level;
        let ef = (k * 4).max(8); // ef > k for recall, trim at the end
        while top > 0 {
            let candidates = self.search_layer(q, ep, ef, top);
            ep = candidates.first().copied().unwrap_or(ep);
            top -= 1;
        }
        // Level-0 search with ef, then trim to k.
        let ids = self.search_layer(q, ep, ef, 0);
        let mut out: Vec<(u32, f32)> = ids
            .into_iter()
            .map(|id| (id, self.dist_query(q, id).sqrt()))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
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
}
