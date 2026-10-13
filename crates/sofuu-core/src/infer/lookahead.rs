// crates/sofuu-core/src/infer/lookahead.rs — Jacobi / Lookahead Parallel Tree Decoding.
//
// Amortizes memory bandwidth by evaluating candidate token trees in a single forward pass.
// Proposes candidate branches from n-grams learned from the prompt and generated stream,
// evaluates them concurrently using 2D block-causal tree attention masks, and accepts
// matching trajectories with zero perplexity degradation.

use std::collections::HashMap;

/// An N-Gram entry tracking continuation token frequencies.
#[derive(Debug, Clone, Default)]
pub struct NgramNode {
    pub counts: HashMap<u32, u32>,
}

impl NgramNode {
    pub fn record(&mut self, next_token: u32) {
        *self.counts.entry(next_token).or_insert(0) += 1;
    }

    pub fn top_continuations(&self, max_k: usize) -> Vec<u32> {
        let mut pairs: Vec<(u32, u32)> = self.counts.iter().map(|(&t, &c)| (t, c)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.into_iter().take(max_k).map(|(t, _)| t).collect()
    }
}

/// N-Gram cache maintaining 2-gram and 3-gram candidate projections.
#[derive(Debug, Clone, Default)]
pub struct NgramCache {
    /// Bigram mapping: token_a -> continuation
    pub bigrams: HashMap<u32, NgramNode>,
    /// Trigram mapping: (token_a, token_b) -> continuation
    pub trigrams: HashMap<(u32, u32), NgramNode>,
    /// Window buffer of recent tokens
    pub window: Vec<u32>,
    /// Max tokens stored in window
    pub max_window: usize,
}

impl NgramCache {
    pub fn new(max_window: usize) -> Self {
        Self {
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
            window: Vec::with_capacity(max_window),
            max_window,
        }
    }

    /// Ingest a bulk slice of tokens (e.g. from prompt prefill).
    pub fn ingest_sequence(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.record_token(t);
        }
    }

    /// Record a single new token into the n-gram index.
    pub fn record_token(&mut self, token: u32) {
        let len = self.window.len();
        if len >= 1 {
            let prev = self.window[len - 1];
            self.bigrams.entry(prev).or_default().record(token);
        }
        if len >= 2 {
            let prev2 = (self.window[len - 2], self.window[len - 1]);
            self.trigrams.entry(prev2).or_default().record(token);
        }

        if self.window.len() >= self.max_window {
            self.window.remove(0);
        }
        self.window.push(token);
    }

    /// Propose candidate next tokens given the recent context tail.
    pub fn propose(&self, context: &[u32], max_candidates: usize) -> Vec<u32> {
        let len = context.len();
        if len == 0 {
            return Vec::new();
        }

        // Try trigram match first (higher precision)
        if len >= 2 {
            let key = (context[len - 2], context[len - 1]);
            if let Some(node) = self.trigrams.get(&key) {
                let continuations = node.top_continuations(max_candidates);
                if !continuations.is_empty() {
                    return continuations;
                }
            }
        }

        // Fall back to bigram match
        let last = context[len - 1];
        if let Some(node) = self.bigrams.get(&last) {
            return node.top_continuations(max_candidates);
        }

        Vec::new()
    }
}

/// A node in a candidate token tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeNode {
    pub token: u32,
    pub parent_idx: Option<usize>,
    pub depth: usize,
}

/// A candidate token tree for single-pass parallel forward verification.
#[derive(Debug, Clone)]
pub struct CandidateTree {
    pub nodes: Vec<TreeNode>,
    pub max_depth: usize,
}

impl CandidateTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            max_depth: 0,
        }
    }

    /// Build a candidate tree given an n-gram cache and the current context.
    pub fn build(
        ngram: &NgramCache,
        context: &[u32],
        max_tree_size: usize,
        max_depth: usize,
    ) -> Self {
        let mut tree = Self::new();
        tree.max_depth = max_depth;

        if context.is_empty() || max_tree_size == 0 {
            return tree;
        }

        // Propose level 1 candidates from context tail
        let l1 = ngram.propose(context, 2);
        for &t1 in &l1 {
            if tree.nodes.len() >= max_tree_size {
                break;
            }
            let idx1 = tree.nodes.len();
            tree.nodes.push(TreeNode {
                token: t1,
                parent_idx: None,
                depth: 1,
            });

            // Propose level 2 candidates continuing from branch 1
            if max_depth >= 2 && tree.nodes.len() < max_tree_size {
                let mut branch_ctx = context.to_vec();
                branch_ctx.push(t1);
                let l2 = ngram.propose(&branch_ctx, 2);
                for &t2 in &l2 {
                    if tree.nodes.len() >= max_tree_size {
                        break;
                    }
                    let idx2 = tree.nodes.len();
                    tree.nodes.push(TreeNode {
                        token: t2,
                        parent_idx: Some(idx1),
                        depth: 2,
                    });

                    // Propose level 3 candidates continuing from branch 2
                    if max_depth >= 3 && tree.nodes.len() < max_tree_size {
                        branch_ctx.push(t2);
                        let l3 = ngram.propose(&branch_ctx, 1);
                        for &t3 in &l3 {
                            if tree.nodes.len() >= max_tree_size {
                                break;
                            }
                            tree.nodes.push(TreeNode {
                                token: t3,
                                parent_idx: Some(idx2),
                                depth: 3,
                            });
                        }
                        branch_ctx.pop();
                    }
                }
            }
        }

        tree
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return the list of candidate tokens in tree index order.
    pub fn tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.token).collect()
    }

    /// Check if node `ancestor` is an ancestor of node `descendant`.
    pub fn is_ancestor(&self, ancestor: usize, descendant: usize) -> bool {
        if ancestor == descendant {
            return true;
        }
        let mut curr = self.nodes[descendant].parent_idx;
        while let Some(p) = curr {
            if p == ancestor {
                return true;
            }
            curr = self.nodes[p].parent_idx;
        }
        false
    }

    /// Construct a 2D block-causal tree attention mask of shape `[K, K]`.
    ///
    /// For row `i` and col `j`:
    /// Value is `0.0` if `j` is an ancestor of `i` or `j == i`;
    /// Value is `-1e9` (neg infinity) otherwise.
    pub fn build_causal_mask(&self) -> Vec<f32> {
        let k = self.nodes.len();
        let mut mask = vec![-1e9f32; k * k];
        for i in 0..k {
            for j in 0..k {
                if self.is_ancestor(j, i) {
                    mask[i * k + j] = 0.0f32;
                }
            }
        }
        mask
    }

    /// Verify tree candidate outputs against model predictions.
    ///
    /// `predicted_tokens`: slice of length `K` where `predicted_tokens[i]` is the token
    /// sampled/predicted from the logits of tree node `i`.
    /// `root_predicted`: the token predicted from the base context position (preceding the tree).
    ///
    /// Returns:
    /// - `accepted_tokens`: all tokens verified in sequence (always starts with `root_predicted`).
    /// - `accepted_indices`: corresponding tree node indices for extra accepted tokens.
    pub fn verify(
        &self,
        root_predicted: u32,
        predicted_tokens: &[u32],
    ) -> (Vec<u32>, Vec<usize>) {
        let mut accepted_tokens = vec![root_predicted];
        let mut accepted_indices = Vec::new();

        if self.nodes.is_empty() || predicted_tokens.len() < self.nodes.len() {
            return (accepted_tokens, accepted_indices);
        }

        // Find root children matching `root_predicted`
        let mut current_accepted_idx: Option<usize> = None;
        for (i, node) in self.nodes.iter().enumerate() {
            if node.parent_idx.is_none() && node.token == root_predicted {
                current_accepted_idx = Some(i);
                accepted_indices.push(i);
                break;
            }
        }

        // Follow the branch greedily while predictions match candidate children
        while let Some(parent) = current_accepted_idx {
            let next_expected = predicted_tokens[parent];
            let mut matched_child = None;

            for (i, node) in self.nodes.iter().enumerate() {
                if node.parent_idx == Some(parent) && node.token == next_expected {
                    matched_child = Some(i);
                    accepted_tokens.push(next_expected);
                    accepted_indices.push(i);
                    break;
                }
            }

            current_accepted_idx = matched_child;
        }

        (accepted_tokens, accepted_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_cache_recording_and_proposals() {
        let mut cache = NgramCache::new(64);
        // "fn main() { return 42; }"
        let tokens = vec![101, 202, 303, 404, 505, 42, 606];
        cache.ingest_sequence(&tokens);

        // Bigram match for 202 ("main") -> 303 ("()")
        let p_bi = cache.propose(&[202], 2);
        assert_eq!(p_bi, vec![303]);

        // Trigram match for [101, 202] -> 303
        let p_tri = cache.propose(&[101, 202], 2);
        assert_eq!(p_tri, vec![303]);

        // Trigram match for [505, 42] -> 606
        let p_return = cache.propose(&[505, 42], 1);
        assert_eq!(p_return, vec![606]);
    }

    #[test]
    fn test_candidate_tree_mask_geometry() {
        let mut cache = NgramCache::new(64);
        // Branch 1: [10 -> 20 -> 30]
        cache.record_token(10);
        cache.record_token(20);
        cache.record_token(30);

        // Branch 2: [10 -> 25]
        cache.record_token(10);
        cache.record_token(25);

        let tree = CandidateTree::build(&cache, &[10], 5, 3);
        assert!(tree.size() >= 2);

        let mask = tree.build_causal_mask();
        let k = tree.size();

        // Diagonal must always be 0.0 (each token attends to itself)
        for i in 0..k {
            assert_eq!(mask[i * k + i], 0.0f32);
        }

        // Sibling branches must not attend to each other
        for i in 0..k {
            for j in 0..k {
                if !tree.is_ancestor(j, i) {
                    assert_eq!(mask[i * k + j], -1e9f32);
                }
            }
        }
    }

    #[test]
    fn test_tree_verification_greedy_equivalence() {
        let mut tree = CandidateTree::new();
        // Candidate 0: token 100 (root child)
        tree.nodes.push(TreeNode { token: 100, parent_idx: None, depth: 1 });
        // Candidate 1: token 200 (child of 0)
        tree.nodes.push(TreeNode { token: 200, parent_idx: Some(0), depth: 2 });
        // Candidate 2: token 300 (child of 1)
        tree.nodes.push(TreeNode { token: 300, parent_idx: Some(1), depth: 3 });
        // Candidate 3: token 150 (alternative root child)
        tree.nodes.push(TreeNode { token: 150, parent_idx: None, depth: 1 });

        // Case 1: Root prediction matches token 100, node 0 predicts 200, node 1 predicts 300, node 2 predicts 999
        let (accepted, indices) = tree.verify(100, &[200, 300, 999, 999]);
        assert_eq!(accepted, vec![100, 200, 300]);
        assert_eq!(indices, vec![0, 1, 2]);

        // Case 2: Root prediction matches token 150 (alternative branch), node 3 predicts 999
        let (accepted_b, indices_b) = tree.verify(150, &[999, 999, 999, 999]);
        assert_eq!(accepted_b, vec![150]);
        assert_eq!(indices_b, vec![3]);

        // Case 3: Root prediction does not match any candidate -> returns only root
        let (accepted_c, indices_c) = tree.verify(999, &[999, 999, 999, 999]);
        assert_eq!(accepted_c, vec![999]);
        assert!(indices_c.is_empty());
    }
}
