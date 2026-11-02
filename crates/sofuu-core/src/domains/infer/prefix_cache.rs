// crates/sofuu-core/src/infer/prefix_cache.rs — Shared-prefix KV reuse (serving).
//
// Why this exists
// ---------------
// Every request in an agent/RAG deployment carries the same system prompt, and
// that prompt is usually the dominant prefill cost (1000-2000 tokens of fixed
// text in front of a 20-token user turn). The multi-turn reuse in
// InferEngine::stream only helps when the NEW prompt EXTENDS the engine's
// previous history — which is what a single-user chat does and what a server
// never does: request N+1 shares the system prompt with request N but diverges
// at the first user turn, so common_prefix collapses to the system prompt and
// the old code reset the cache and re-prefilled those tokens from scratch.
//
// What it is
// ----------
// An immutable, refcounted block table indexed over TOKEN sequences:
//
//   * a block is KV_BLOCK_SIZE (16) consecutive positions of K and V for every
//     (layer, kv_head) of the model — the same granularity the paged KV pool
//     uses (paged_kv::KV_BLOCK_SIZE), so a prefix is "some whole blocks";
//   * blocks live behind Arc. A block is NEVER mutated after it is published:
//     restore copies its rows into the model's private contiguous KV cache, and
//     the running sequence writes only into its own cache. Two sequences that
//     share a prefix therefore hold refcounted handles to the same memory and a
//     sequence that diverges after it writes into its own rows — copy-on-write
//     by construction instead of by flag. There is no "borrowed block" state to
//     get wrong, which is the point: correctness first;
//   * the index is a trie over blocks. Depth d is position range
//     [16d, 16d+16) for every chain, so a node's KV payload is only ever valid
//     at its own depth. Sharing happens because two prompts that start with the
//     same tokens walk the same path; the block is stored once and both chains
//     point at it (Arc::strong_count is the refcount);
//   * lookup matches the longest chain of whole blocks and returns its blocks in
//     order; the caller copies them into the model cache at [0, 16*n);
//   * bounded: max shared bytes + max blocks, LRU eviction of LEAVES (a node
//     with children is on someone's path and cannot be dropped), and the whole
//     thing is off with SOFUU_PREFIX_CACHE=0 — the reproducible (cold) path then
//     allocates nothing and does not even hash the prompt.
//
// Restoring is a memcpy, not a recompute: the rows are bit-identical to the ones
// a fresh prefill would write, which is what makes the reuse safe to hand to a
// greedy decode.

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use super::paged_kv::KV_BLOCK_SIZE;

/// Default budget for shared KV bytes (256 MiB: ~1.1k cached tokens of
/// Qwen3-0.6B, ~4k of Llama-3.2-1B — see PrefixBlock::bytes).
pub const DEFAULT_MAX_BYTES: usize = 256 << 20;
/// Default cap on cached blocks (a block is 16 tokens, so 4096 ~= 65k tokens).
pub const DEFAULT_MAX_BLOCKS: usize = 4096;
/// Below this many reusable tokens the reuse is not worth a lookup (a hit has
/// to clear the prefill chunk alignment anyway, and a short prefix pays the
/// restore memcpy without removing real work).
pub const DEFAULT_MIN_TOKENS: usize = 64;

/// Immutable KV payload of one 16-token block.
///
/// Layout: k/v are flat [n_h][KV_BLOCK_SIZE * head_dim], where
/// n_h = n_layers * n_kv_heads and h = layer * n_kv_heads + kv_head — the same
/// head-major order the model's caches use
/// ([layer, kv_head, max_seq_len, head_dim]), so a restore walks h outer and
/// blocks inner and writes sequentially.
pub struct PrefixBlock {
    /// The token ids this block holds. Matching re-verifies these; the trie
    /// alone (a hash) is never trusted to identify a KV payload.
    pub tokens: Vec<u32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub n_h: usize,
    pub head_dim: usize,
}

impl PrefixBlock {
    pub fn new(tokens: Vec<u32>, k: Vec<f32>, v: Vec<f32>, n_h: usize, head_dim: usize) -> Self {
        Self { tokens, k, v, n_h, head_dim }
    }

    /// Resident bytes of this block (both caches, f32).
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }

    /// True when the payload matches the geometry the caller restores into.
    pub fn fits(&self, n_h: usize, head_dim: usize) -> bool {
        self.n_h == n_h
            && self.head_dim == head_dim
            && self.k.len() == n_h * KV_BLOCK_SIZE * head_dim
            && self.v.len() == self.k.len()
    }
}

/// Env-tunable knobs. Read once per engine (PrefixCache::from_env).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_bytes: usize,
    pub max_blocks: usize,
    pub min_tokens: usize,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

impl PrefixCacheConfig {
    /// SOFUU_PREFIX_CACHE=0 turns the feature off (no allocation, no hashing,
    /// the prefill path is exactly the pre-feature one).
    /// SOFUU_PREFIX_CACHE_MB / _ENTRIES / _MIN_TOKENS bound it.
    pub fn from_env() -> Self {
        let enabled = std::env::var("SOFUU_PREFIX_CACHE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let mb = env_usize("SOFUU_PREFIX_CACHE_MB", DEFAULT_MAX_BYTES >> 20);
        Self {
            enabled,
            max_bytes: mb.saturating_mul(1 << 20),
            max_blocks: env_usize("SOFUU_PREFIX_CACHE_ENTRIES", DEFAULT_MAX_BLOCKS).max(1),
            min_tokens: env_usize("SOFUU_PREFIX_MIN_TOKENS", DEFAULT_MIN_TOKENS),
        }
    }

    pub fn disabled() -> Self {
        Self { enabled: false, max_bytes: 0, max_blocks: 0, min_tokens: usize::MAX }
    }
}

/// Lifetime counters for the report / an operator dashboard.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// lookups that saw at least one whole matching block.
    pub hits: usize,
    pub misses: usize,
    /// tokens covered by the matched chains (before chunk alignment).
    pub hit_tokens: usize,
    /// tokens the callers actually skipped recomputing (after alignment).
    pub reused_tokens: usize,
    /// blocks a request snapshotted into the index.
    pub blocks_created: usize,
    pub blocks_evicted: usize,
    pub blocks: usize,
    pub bytes: usize,
}

struct Node {
    /// Hash under which this node hangs in its parent's child map.
    key: u64,
    /// The 16 tokens of this block (empty for the root).
    tokens: Vec<u32>,
    block: Option<Arc<PrefixBlock>>,
    children: HashMap<u64, usize>,
    parent: Option<usize>,
    last_used: u64,
}

/// Refcounted, immutable, LRU-bounded prefix index over token sequences.
pub struct PrefixCache {
    config: PrefixCacheConfig,
    nodes: Vec<Node>,
    free_slots: Vec<usize>,
    bytes: usize,
    blocks: usize,
    tick: u64,
    stats: PrefixCacheStats,
}

const ROOT: usize = 0;

fn block_key(tokens: &[u32]) -> u64 {
    // FNV-1a over the block's token ids. A collision can only make two
    // different blocks map to one child bucket; the token comparison in
    // lookup/insert rejects that (and simply declines to cache the branch), so
    // a KV payload is never served for the wrong tokens.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl PrefixCache {
    pub fn from_env() -> Self {
        Self::new(PrefixCacheConfig::from_env())
    }

    pub fn new(config: PrefixCacheConfig) -> Self {
        let mut nodes = Vec::new();
        nodes.push(Node {
            key: 0,
            tokens: Vec::new(),
            block: None,
            children: HashMap::new(),
            parent: None,
            last_used: 0,
        });
        Self {
            config,
            nodes,
            free_slots: Vec::new(),
            bytes: 0,
            blocks: 0,
            tick: 0,
            stats: PrefixCacheStats::default(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn min_tokens(&self) -> usize {
        self.config.min_tokens
    }

    pub fn config(&self) -> PrefixCacheConfig {
        self.config
    }

    pub fn stats(&self) -> PrefixCacheStats {
        let mut s = self.stats;
        s.blocks = self.blocks;
        s.bytes = self.bytes;
        s
    }

    /// Count tokens a caller actually skipped (post-alignment) for the report.
    pub fn note_reused(&mut self, tokens: usize) {
        self.stats.reused_tokens += tokens;
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Longest whole-block prefix of tokens present in the index, in order.
    /// Touches the LRU clock of every matched node.
    pub fn lookup(&mut self, tokens: &[u32]) -> Vec<Arc<PrefixBlock>> {
        let mut out = Vec::new();
        if !self.config.enabled || tokens.len() < KV_BLOCK_SIZE {
            return out;
        }
        let mut node = ROOT;
        let mut depth = 0usize;
        let mut path = Vec::new();
        while (depth + 1) * KV_BLOCK_SIZE <= tokens.len() {
            let start = depth * KV_BLOCK_SIZE;
            let key = block_key(&tokens[start..start + KV_BLOCK_SIZE]);
            let child = match self.nodes[node].children.get(&key) {
                Some(&c) => c,
                None => break,
            };
            if self.nodes[child].tokens != tokens[start..start + KV_BLOCK_SIZE] {
                break; // hash collision, not a match
            }
            path.push(child);
            node = child;
            depth += 1;
        }
        if path.is_empty() {
            self.stats.misses += 1;
            return out;
        }
        let tick = self.next_tick();
        for &n in &path {
            self.nodes[n].last_used = tick;
        }
        out.reserve(path.len());
        for n in path {
            if let Some(b) = self.nodes[n].block.clone() {
                out.push(b);
            } else {
                break; // cannot happen for a live node; keep the invariant anyway
            }
        }
        self.stats.hits += 1;
        self.stats.hit_tokens += out.len() * KV_BLOCK_SIZE;
        out
    }

    /// Ensure every whole block of tokens is in the index, asking snapshot for
    /// the payload of the blocks that are missing (blocks already on the path
    /// are shared, not copied — that is the memory win). tokens must be the
    /// KV-valid prefix of a prompt, i.e. the rows the caller just wrote at
    /// positions [0, tokens.len()).
    pub fn insert(&mut self, tokens: &[u32], snapshot: &mut dyn FnMut(usize) -> Option<PrefixBlock>) {
        if !self.config.enabled {
            return;
        }
        let n_blocks = tokens.len() / KV_BLOCK_SIZE;
        if n_blocks == 0 {
            return;
        }
        // Walk (and touch) what is already shared.
        let mut node = ROOT;
        let mut depth = 0usize;
        let mut tick: Option<u64> = None;
        while depth < n_blocks {
            let start = depth * KV_BLOCK_SIZE;
            let key = block_key(&tokens[start..start + KV_BLOCK_SIZE]);
            let child = match self.nodes[node].children.get(&key) {
                Some(&c) => c,
                None => break,
            };
            if self.nodes[child].tokens != tokens[start..start + KV_BLOCK_SIZE] {
                return; // colliding hash from a different token block: keep both trees honest
            }
            node = child;
            depth += 1;
            let t = match tick {
                Some(t) => t,
                None => {
                    let t = self.next_tick();
                    tick = Some(t);
                    t
                }
            };
            self.nodes[node].last_used = t;
        }
        // Append the missing tail.
        while depth < n_blocks {
            let start = depth * KV_BLOCK_SIZE;
            let block = match snapshot(depth) {
                Some(b) => b,
                None => break, // payload unavailable / geometry refused: stop cleanly
            };
            let key = block_key(&tokens[start..start + KV_BLOCK_SIZE]);
            let id = self.alloc_node();
            let t = self.next_tick();
            self.bytes += block.bytes();
            self.blocks += 1;
            self.stats.blocks_created += 1;
            self.nodes[id] = Node {
                key,
                tokens: tokens[start..start + KV_BLOCK_SIZE].to_vec(),
                block: Some(Arc::new(block)),
                children: HashMap::new(),
                parent: Some(node),
                last_used: t,
            };
            self.nodes[node].children.insert(key, id);
            node = id;
            depth += 1;
        }
        self.enforce_budget();
    }

    fn alloc_node(&mut self) -> usize {
        if let Some(id) = self.free_slots.pop() {
            return id;
        }
        self.nodes.push(Node {
            key: 0,
            tokens: Vec::new(),
            block: None,
            children: HashMap::new(),
            parent: None,
            last_used: 0,
        });
        self.nodes.len() - 1
    }

    /// Evict least-recently-used LEAVES until both budgets hold. A node with
    /// children is on some chain and cannot go; once its last child is evicted
    /// it becomes a leaf and joins the queue, so eviction always makes progress
    /// (the root is the only node that can never be removed).
    fn enforce_budget(&mut self) {
        if self.bytes <= self.config.max_bytes && self.blocks <= self.config.max_blocks {
            return;
        }
        let mut heap: BinaryHeap<std::cmp::Reverse<(u64, usize)>> = BinaryHeap::new();
        for (id, n) in self.nodes.iter().enumerate() {
            if id != ROOT && n.block.is_some() && n.children.is_empty() {
                heap.push(std::cmp::Reverse((n.last_used, id)));
            }
        }
        while (self.bytes > self.config.max_bytes || self.blocks > self.config.max_blocks)
            && !heap.is_empty()
        {
            let std::cmp::Reverse((_, id)) = heap.pop().unwrap();
            let (parent, key, live) = match self.nodes[id].parent {
                Some(p) => (p, self.nodes[id].key, self.nodes[id].block.is_some()),
                None => continue, // already evicted
            };
            if !live {
                continue;
            }
            if !self.nodes[id].children.is_empty() {
                continue; // gained children since the sweep: still needed
            }
            let freed = self.nodes[id].block.as_ref().map(|b| b.bytes()).unwrap_or(0);
            self.nodes[id].block = None;
            self.nodes[id].tokens = Vec::new();
            self.nodes[id].parent = None;
            self.nodes[parent].children.remove(&key);
            self.free_slots.push(id);
            self.bytes = self.bytes.saturating_sub(freed);
            self.blocks = self.blocks.saturating_sub(1);
            self.stats.blocks_evicted += 1;
            if parent != ROOT
                && self.nodes[parent].block.is_some()
                && self.nodes[parent].children.is_empty()
            {
                heap.push(std::cmp::Reverse((self.nodes[parent].last_used, parent)));
            }
        }
    }

    /// Drop every block (engine reset / model reload).
    pub fn clear(&mut self) {
        self.nodes.truncate(1);
        self.nodes[ROOT].children.clear();
        self.free_slots.clear();
        self.bytes = 0;
        self.blocks = 0;
    }
}
