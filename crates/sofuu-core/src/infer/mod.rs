// crates/sofuu-core/src/infer/mod.rs — Fast CPU-First GGUF Inference Engine.
//
// Combines zero-copy mmap, safe GGUF parsing, BPE tokenizer, and C++ SIMD microkernels
// into a lightweight, zero-dependency LLM engine for edge, laptop, and serverless environments.

pub mod block_graph;
pub mod continuous;
pub mod dynamic_ffn;
pub mod gguf;
pub mod gptx3;
pub mod qwen35;
pub mod lfm2;
pub mod kivi;
pub mod lookahead;
pub mod metal;
pub mod mmap;
pub mod model;
pub mod npy;
pub mod paged_kv;
pub mod pytorch;
pub mod quant;
#[cfg(test)]
mod quant_oracle;
pub mod safetensors;
pub mod sampler;
pub mod spark;
pub mod swarm;
pub mod tensor_table;
pub mod tokenizer;
pub mod zip_reader;

use std::path::Path;

use self::block_graph::BlockGraphModel;
use self::gguf::GgufFile;
use self::gptx3::Gptx3Model;
use self::lfm2::Lfm2Model;
use self::model::{ModelConfig, TransformerModel};
use self::sampler::{Sampler, SamplerConfig};
use self::spark::Spark25Model;
use self::qwen35::Qwen35Model;
use self::tensor_table::UnifiedTensorTable;
use self::tokenizer::BpeTokenizer;

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub threads: usize,
    pub stop_tokens: Vec<String>,
    pub sliding_window: usize,
    pub sink_tokens: usize,
    pub lookahead: bool,
    pub snapkv: bool,
    pub dynamic_ffn: bool,
    pub kivi: bool,
    pub kivi_bits: usize,
    pub streaming_weights: bool,
    pub trace_tokens: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            threads: 0, // 0 = utilize all available CPU cores
            stop_tokens: vec![
                "<|im_end|>".to_string(),
                "<|endoftext|>".to_string(),
                "<|eot_id|>".to_string(),
                "<｜end▁of▁sentence｜>".to_string(),
            ],
            sliding_window: 0, // 0 = full context / model default
            sink_tokens: 4,    // 4 sink tokens by default
            lookahead: false,  // Jacobi / Lookahead tree parallel decoding
            snapkv: false,     // SnapKV observation pooling compression
            dynamic_ffn: false, // Dynamic FFN Sparsity (PowerInfer / DejaVu style)
            kivi: false,       // KIVI / TurboKV asymmetric 2-bit & 4-bit KV cache
            kivi_bits: 4,      // 4-bit (Four) or 2-bit (Two)
            streaming_weights: false, // Non-temporal streaming weights (cache-bypass)
            trace_tokens: false, // record per-token timing + top logprobs (§8 P3)
        }
    }
}

/// One generated token's timing + logprob, recorded when
/// `GenerateOptions::trace_tokens` is set and readable afterwards from
/// [`InferEngine::last_token_trace`]. Long-context plan §8 P3: the surface
/// Gate 4 needs (per-position decode timing) and the logprobs Gate 2 wants.
#[derive(Debug, Clone)]
pub struct TokenTiming {
    /// Index within this generation (0 = first generated token).
    pub step: usize,
    /// KV position the token was fed back at.
    pub pos: usize,
    /// The sampled token id.
    pub token: u32,
    /// Wall-clock cost of the forward that produced this token's logits.
    pub forward_ns: u64,
    /// log P(this token was sampled) — full-vocab logsumexp over logits.
    pub logprob: f32,
}

pub enum ModelBackend {
    Gguf(TransformerModel),
    Lfm2(Lfm2Model),
    BlockGraph(BlockGraphModel),
    Gptx3(Gptx3Model),
    Spark(Spark25Model),
    Qwen35(Qwen35Model),
}

pub struct InferModel {
    pub config: ModelConfig,
    pub backend: ModelBackend,
    pub sliding_window: usize,
    pub sink_tokens: usize,
}

impl InferModel {
    pub fn from_gguf(gguf: GgufFile, max_seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        Self::from_gguf_with_gpu(gguf, max_seq_len, vocab_size, true, 0)
    }

    pub fn from_gguf_with_gpu(
        gguf: GgufFile,
        max_seq_len: usize,
        vocab_size: usize,
        gpu_allowed: bool,
        kv_ring_req: usize,
    ) -> Result<Self, String> {
        let arch = gguf.get_str("general.architecture").unwrap_or("llama").to_string();
        if arch == "spark2_5" {
            let m = Spark25Model::from_gguf(gguf, max_seq_len, vocab_size)?;
            let config = m.config.clone();
            return Ok(Self {
                config,
                backend: ModelBackend::Spark(m),
                sliding_window: 0,
                sink_tokens: 4,
            });
        }
        if arch == "qwen35" {
            let m = Qwen35Model::from_gguf(gguf, max_seq_len)?;
            let config = m.config.clone();
            return Ok(Self {
                config,
                backend: ModelBackend::Qwen35(m),
                sliding_window: 0,
                sink_tokens: 4,
            });
        }
        let m = TransformerModel::from_gguf_with_kv_ring(gguf, max_seq_len, vocab_size, kv_ring_req)?;
        let config = m.config.clone();
        Ok(Self {
            config,
            backend: ModelBackend::Gguf(m),
            sliding_window: 0,
            sink_tokens: 4,
        })
    }

    pub fn from_lfm2_dir<P: AsRef<Path>>(dir: P, max_seq_len: usize) -> Result<Self, String> {
        Self::from_lfm2_dir_with_gpu(dir, max_seq_len, true)
    }

    pub fn from_lfm2_dir_with_gpu<P: AsRef<Path>>(
        dir: P,
        max_seq_len: usize,
        gpu_allowed: bool,
    ) -> Result<Self, String> {
        let m = Lfm2Model::from_safetensors_dir_with_gpu(dir, max_seq_len, gpu_allowed)?;
        let config = m.config.clone();
        Ok(Self {
            config,
            backend: ModelBackend::Lfm2(m),
            sliding_window: 0,
            sink_tokens: 4,
        })
    }

    pub fn from_tensor_table(table: &UnifiedTensorTable, max_seq_len: usize) -> Result<Self, String> {
        if let Ok(m) = Gptx3Model::from_tensor_table(table, max_seq_len) {
            let config = m.config.clone();
            return Ok(Self {
                config,
                backend: ModelBackend::Gptx3(m),
                sliding_window: 0,
                sink_tokens: 4,
            });
        }
        let m = BlockGraphModel::from_tensor_table(table, max_seq_len)?;
        let config = m.config.clone();
        Ok(Self {
            config,
            backend: ModelBackend::BlockGraph(m),
            sliding_window: 0,
            sink_tokens: 4,
        })
    }

    pub fn set_gpu_layers(&mut self, n: usize) {
        match &mut self.backend {
            ModelBackend::BlockGraph(m) => m.set_gpu_layers(n),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ModelBackend::Lfm2(m) => m.set_gpu_layers(n),
            _ => {}
        }
    }

    pub fn reset_cache(&mut self) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.reset_cache(),
            ModelBackend::Lfm2(m) => m.reset_cache(),
            ModelBackend::BlockGraph(m) => m.reset_cache(),
            ModelBackend::Gptx3(m) => m.reset_cache(),
            ModelBackend::Spark(m) => m.reset_cache(),
            ModelBackend::Qwen35(m) => m.reset_cache(),
        }
    }

    pub fn compact_kv_cache(&mut self, cur_pos: usize, sink: usize, drop: usize) -> usize {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.compact_kv_cache(cur_pos, sink, drop),
            ModelBackend::Lfm2(m) => m.compact_kv_cache(cur_pos, sink, drop),
            ModelBackend::BlockGraph(m) => m.compact_kv_cache(cur_pos, sink, drop),
            ModelBackend::Gptx3(m) => m.compact_kv_cache(cur_pos, sink, drop),
            ModelBackend::Spark(m) => m.compact_kv_cache(cur_pos, sink, drop),
            ModelBackend::Qwen35(m) => m.compact_kv_cache(cur_pos, sink, drop),
        }
    }

    pub fn set_dynamic_ffn(&mut self, enabled: bool) {
        if let ModelBackend::Gguf(m) = &mut self.backend {
            m.dynamic_ffn.enabled = enabled;
        }
    }

    pub fn enable_kivi(&mut self, bits: usize, cur_pos: usize) {
        if let ModelBackend::Gguf(m) = &mut self.backend {
            m.enable_kivi(bits, cur_pos);
        }
    }

    /// P7: arm EMPTY KIVI streams before the streaming prefill — the chunks
    /// ingest themselves as they are computed (ring mode).
    pub fn enable_kivi_streaming(&mut self, bits: usize) {
        if let ModelBackend::Gguf(m) = &mut self.backend {
            m.enable_kivi_streaming(bits);
        }
    }

    /// P7: true when the f32 KV arrays are ring-strided and the quantized
    /// streams carry the history of record.
    pub fn kv_ring_mode(&self) -> bool {
        match &self.backend {
            ModelBackend::Gguf(m) => m.config.kv_ring > 0,
            _ => false,
        }
    }

    /// The LOGICAL window (may exceed the ring-sized f32 arrays).
    pub fn logical_seq_len(&self) -> usize {
        match &self.backend {
            ModelBackend::Gguf(m) => m.config.logical_max_seq_len,
            _ => self.config.max_seq_len,
        }
    }

    /// Rebuild the KIVI stream from the (post-compaction) f32 cache —
    /// F-LC3: after compact_kv_cache re-anchors positions, the quantized
    /// store must describe the same history or decode attends a ghost.
    /// No-op on backends without KIVI.
    pub fn rebuild_kivi_after_compact(&mut self, seq_len: usize) {
        if let ModelBackend::Gguf(m) = &mut self.backend {
            m.rebuild_kivi_after_compact(seq_len);
        }
    }

    /// Override the sliding-window size for backends that carry one.
    pub fn set_sliding_window(&mut self, w: usize) {
        self.sliding_window = w;
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.sliding_window = w,
            ModelBackend::Spark(m) => m.sliding_window = w,
            _ => {}
        }
    }

    /// Drop any KIVI state (used when a turn does not request KIVI but an
    /// earlier turn enabled it).
    pub fn disable_kivi(&mut self) {
        if let ModelBackend::Gguf(m) = &mut self.backend {
            m.kivi = None;
        }
    }

    /// Fill reusable scratch buffers with `v`. A debug instrument: if output
    /// changes when the scratch is poisoned, a kernel reads a row or column
    /// the current call never wrote, so results depend on whatever the
    /// previous call left there. `which` is a comma-separated list of buffer
    /// names, or "all".
    pub fn poison_scratch(&mut self, v: f32, which: &str) {
        let ModelBackend::Gguf(m) = &mut self.backend else {
            return;
        };
        let all = which == "all";
        for (name, buf) in [
            ("x", &mut m.x),
            ("x_norm", &mut m.x_norm),
            ("q", &mut m.q),
            ("k", &mut m.k),
            ("v", &mut m.v),
            ("attn_scores", &mut m.attn_scores),
            ("head_out", &mut m.head_out),
            ("attn_out", &mut m.attn_out),
            ("ffn_gate_buf", &mut m.ffn_gate_buf),
            ("ffn_up_buf", &mut m.ffn_up_buf),
            ("ffn_out", &mut m.ffn_out),
            ("logits", &mut m.logits),
            ("x_batch", &mut m.x_batch),
            ("x_norm_batch", &mut m.x_norm_batch),
            ("q_batch", &mut m.q_batch),
            ("k_batch", &mut m.k_batch),
            ("v_batch", &mut m.v_batch),
            ("head_out_batch", &mut m.head_out_batch),
            ("attn_out_batch", &mut m.attn_out_batch),
            ("ffn_gate_batch", &mut m.ffn_gate_batch),
            ("ffn_up_batch", &mut m.ffn_up_batch),
            ("ffn_out_batch", &mut m.ffn_out_batch),
        ] {
            if all || which.split(',').any(|w| w.trim() == name) {
                buf.fill(v);
            }
        }
    }

    /// Whether the speculative TREE may run on this model. It ropes the whole
    /// verify batch with the plain C kernel (tree_rope), so a model with a
    /// custom rope profile (phi3 partial rotary / longrope, llama3 scaling)
    /// would verify its drafts at wrong angles — greedy output would then
    /// depend on drafts instead of being the serial answer.
    pub fn spec_decode_supported(&self) -> bool {
        match &self.backend {
            ModelBackend::Gguf(m) => m.spec_decode_supported(),
            _ => true,
        }
    }

    /// Per-model speculative tree size (see TransformerModel::spec_rows_default).
    pub fn spec_rows_default(&self) -> usize {
        match &self.backend {
            ModelBackend::Gguf(m) => m.spec_rows_default(),
            _ => 5,
        }
    }

    /// Per-model spec-gate break-even (see TransformerModel::spec_min_accept_default).
    pub fn spec_min_accept_default(&self) -> f32 {
        match &self.backend {
            ModelBackend::Gguf(m) => m.spec_min_accept_default(),
            _ => 1.15,
        }
    }

    /// Per-model decode thread count (see TransformerModel::decode_threads_auto).
    pub fn decode_threads_auto(&self, max_threads: usize) -> usize {
        match &self.backend {
            ModelBackend::Gguf(m) => m.decode_threads_auto(max_threads),
            _ => max_threads,
        }
    }

    pub fn set_streaming_weights(&mut self, enabled: bool) {
        if let ModelBackend::Gguf(m) = &mut self.backend {
            m.set_streaming_weights(enabled);
        }
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward(token, pos),
            ModelBackend::Lfm2(m) => m.forward(token, pos),
            ModelBackend::BlockGraph(m) => m.forward(token, pos),
            ModelBackend::Gptx3(m) => m.forward(token, pos),
            ModelBackend::Spark(m) => m.forward(token, pos),
            ModelBackend::Qwen35(m) => m.forward(token, pos),
        }
    }

    pub fn forward_prefill(&mut self, token: u32, pos: usize) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_prefill(token, pos),
            ModelBackend::Lfm2(m) => m.forward_prefill(token, pos),
            ModelBackend::BlockGraph(m) => m.forward_prefill(token, pos),
            ModelBackend::Gptx3(m) => m.forward_prefill(token, pos),
            ModelBackend::Spark(m) => m.forward_prefill(token, pos),
            ModelBackend::Qwen35(m) => m.forward_prefill(token, pos),
        }
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], start_pos: usize) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_prefill_batch(tokens, start_pos),
            ModelBackend::Lfm2(m) => m.forward_prefill_batch(tokens, start_pos),
            ModelBackend::BlockGraph(m) => m.forward_prefill_batch(tokens, start_pos),
            ModelBackend::Gptx3(m) => m.forward_prefill_batch(tokens, start_pos),
            ModelBackend::Spark(m) => m.forward_prefill_batch(tokens, start_pos),
            ModelBackend::Qwen35(m) => m.forward_prefill_batch(tokens, start_pos),
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
            ModelBackend::Lfm2(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
            ModelBackend::BlockGraph(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
            ModelBackend::Gptx3(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
            ModelBackend::Spark(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
            ModelBackend::Qwen35(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
        }
    }

    pub fn forward_speculative_batch(&mut self, tokens: &[u32], start_pos: usize, out_logits: &mut [f32]) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_speculative_batch(tokens, start_pos, out_logits),
            _ => {
                let vocab_size = self.config.vocab_size;
                let mut pos = start_pos;
                for (b, &tok) in tokens.iter().enumerate() {
                    let logits = self.forward(tok, pos);
                    out_logits[b * vocab_size..(b + 1) * vocab_size].copy_from_slice(logits);
                    pos += 1;
                }
            }
        }
    }

    pub fn forward_speculative_head_item(&mut self, item_idx: usize, out_logits: &mut [f32]) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_speculative_head_item(item_idx, out_logits),
            _ => {}
        }
    }

    /// Whether the GPU chain pass supplies per-node head logits for the walk.
    pub fn gpu_chain_heads(&self) -> bool {
        match &self.backend {
            ModelBackend::Gguf(m) => m.gpu_chain_heads(),
            _ => false,
        }
    }

    /// How many speculative batch rows the CPU head pass can take at once.
    pub fn prefill_batch_rows(&self) -> usize {
        match &self.backend {
            ModelBackend::Gguf(m) => m.prefill_batch_rows(),
            _ => 0,
        }
    }

    /// LM-head logits for several speculative batch rows in one pass.
    pub fn spec_head_logits_batch(&mut self, first_row: usize, n_rows: usize, out: &mut [f32]) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.spec_head_logits_batch(first_row, n_rows, out),
            _ => {}
        }
    }

    /// Serve lazy per-row verify sampling from the last GPU chain pass's
    /// in-commit logits. Returns false when no chain pass covers this step
    /// (caller falls back to forward_speculative_head_item).
    pub fn spec_chain_head_logits(&mut self, item_idx: usize, out_logits: &mut [f32]) -> bool {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.spec_chain_head_logits(item_idx, out_logits),
            _ => false,
        }
    }

    pub fn forward_speculative_tree(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        ancestry_mask: &[u8],
        start_pos: usize,
        out_logits: &mut [f32],
    ) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => {
                m.forward_speculative_tree(tokens, positions, ancestry_mask, start_pos, out_logits);
            }
            _ => {
                let vocab_size = self.config.vocab_size;
                let l = self.forward(tokens[0], start_pos);
                out_logits[..vocab_size].copy_from_slice(l);
            }
        }
    }

    /// Whether the GPU spec-verify chain route will take the verify pass
    /// (drives the generate loop's chain-shaped draft tree).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn gpu_spec_chain_armed(&self) -> bool {
        match &self.backend {
            ModelBackend::Gguf(m) => m.gpu_spec_chain_armed(),
            _ => false,
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn gpu_spec_chain_armed(&self) -> bool {
        false
    }

    pub fn commit_speculative_candidates(&mut self, accepted_batch_indices: &[usize], start_pos: usize) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.commit_speculative_candidates(accepted_batch_indices, start_pos),
            _ => {}
        }
    }

    pub fn forward_embed(&mut self, token: u32, out_hidden: &mut [f32]) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_embed(token, out_hidden),
            ModelBackend::BlockGraph(m) => m.forward_embed(token, out_hidden),
            ModelBackend::Spark(m) => m.forward_embed(token, out_hidden),
            ModelBackend::Qwen35(_) => {
                let dim = self.config.dim;
                out_hidden[..dim].fill(0.0);
            }
            ModelBackend::Lfm2(_) | ModelBackend::Gptx3(_) => {
                let dim = self.config.dim;
                out_hidden[..dim].fill(0.0);
            }
        }
    }

    pub fn forward_layer_range(
        &mut self,
        pos: usize,
        start_layer: usize,
        end_layer: usize,
        in_hidden: &[f32],
        out_hidden: &mut [f32],
    ) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_layer_range(pos, start_layer, end_layer, in_hidden, out_hidden),
            ModelBackend::BlockGraph(m) => m.forward_layer_range(pos, start_layer, end_layer, in_hidden, out_hidden),
            ModelBackend::Spark(m) => m.forward_layer_range(pos, start_layer, end_layer, in_hidden, out_hidden),
            ModelBackend::Lfm2(_) | ModelBackend::Gptx3(_) | ModelBackend::Qwen35(_) => {
                let dim = self.config.dim;
                out_hidden[..dim].copy_from_slice(&in_hidden[..dim]);
            }
        }
    }

    pub fn forward_head(&mut self, in_hidden: &[f32], out_logits: &mut [f32]) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_head(in_hidden, out_logits),
            ModelBackend::BlockGraph(m) => m.forward_head(in_hidden, out_logits),
            ModelBackend::Spark(m) => m.forward_head(in_hidden, out_logits),
            ModelBackend::Lfm2(_) | ModelBackend::Gptx3(_) | ModelBackend::Qwen35(_) => {
                out_logits.fill(0.0);
            }
        }
    }
}

/// Number of history tokens that may be reused as cached context on the next
/// call. Zero after a KV compaction (positions no longer map onto token
/// order), otherwise exactly the tokens that reached the KV: the decode loop
/// pushes the last sampled token before the max-tokens break decides whether
/// to forward it, so `history` can be one longer than the KV.
fn reusable_history_len(compacted: bool, kv_len: usize, history_len: usize) -> usize {
    if compacted {
        0
    } else {
        kv_len.min(history_len)
    }
}

pub struct InferEngine {
    pub model: InferModel,
    pub tokenizer: BpeTokenizer,
    pub last_prefill_tokens: usize,
    pub last_prefill_duration: std::time::Duration,
    /// Tokenization, cache reset and SnapKV/KIVI setup for the last call.
    /// Reported apart from prefill so it cannot be mistaken for slow decoding.
    pub last_setup_duration: std::time::Duration,
    pub cached_tokens: Vec<u32>,
    /// Per-token per-token timing + top logprobs from the last stream call,
    /// filled only when [`GenerateOptions::trace_tokens`] is set (§8 P3).
    pub last_token_trace: Vec<TokenTiming>,
}

/// Normalize and sanitize a model path string:
/// - Strips quotes (e.g. `"/path/to/model"` or `'...'`)
/// - Strips leading `local/` or `local//` prefix (e.g. `local//Users/...`)
/// - Expands tilde `~` to user's HOME directory
/// - Trims surrounding whitespace and newlines
pub fn clean_model_path(path: &str) -> std::path::PathBuf {
    let mut s = path.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s = &s[1..s.len() - 1].trim();
    }
    if let Some(rest) = s.strip_prefix("local://") {
        s = rest.trim();
    } else if let Some(rest) = s.strip_prefix("local//") {
        s = rest.trim();
    } else if let Some(rest) = s.strip_prefix("local:") {
        s = rest.trim();
    } else if let Some(rest) = s.strip_prefix("local/") {
        s = rest.trim();
    }
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s = &s[1..s.len() - 1].trim();
    }
    if s.starts_with("~/") || s == "~" {
        if let Ok(home) = std::env::var("HOME") {
            let rest = if s == "~" { "" } else { &s[2..] };
            return std::path::PathBuf::from(home).join(rest);
        }
    }
    std::path::PathBuf::from(s)
}

/// Teacher-forced scoring result ([`InferEngine::score_text`]).
#[derive(Debug, Clone)]
pub struct ScoreStats {
    pub tokens_total: usize,
    /// Effective scoring window (max left context a scored token can see);
    /// the caller's 0 becomes `DEFAULT_SCORE_WINDOW`, clamped to the model's
    /// own `max_seq_len`. Reported so the number can be compared to another
    /// tool's `-c`.
    pub window: usize,
    pub scored: usize,
    pub windows: usize,
    pub nll_total: f64,
    pub ppl: f64,
}

/// Default scoring window (context a scored token may see). 512 is
/// llama-perplexity's default `n_ctx`, so `--score` and that tool measure the
/// same thing out of the box. The previous default (64) silently made every
/// sofuu ppl a short-context number and therefore incomparable.
pub const DEFAULT_SCORE_WINDOW: usize = 512;

/// Below this many scored tokens the mean NLL is a handful-of-samples estimate
/// — exp() of a few draws, not a perplexity. `--score` prints the count and
/// warns, so a 1-token number can never pass for a corpus perplexity again.
pub const MIN_SCORE_SAMPLE: usize = 32;

/// One window of the teacher-forced pass. `tokens[start..start+prefill]` only
/// fill the KV cache (the batched prefill returns the last row's logits only,
/// so those positions cannot be scored without paying a decode step each);
/// `tokens[first_scored..end]` are scored, one decode step apiece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreWindow {
    pub start: usize,
    pub prefill: usize,
    pub first_scored: usize,
    pub end: usize,
}

/// Plan the scoring pass: every token index in `1..len` is scored exactly
/// once, windows overlap by exactly one token (the last token of a window is
/// the first CONTEXT token of the next, so it is never scored twice), and the
/// context-only prefix of a window is capped at half its span — without that
/// cap a tail window shorter than the prefill batch degenerated to a single
/// scored token (a 24-token text reported "ppl 10.73" from ONE token, and a
/// 139-token text at window 512 from 11).
pub fn score_windows(len: usize, window: usize, batch: usize) -> Vec<ScoreWindow> {
    let mut out = Vec::new();
    if len < 3 {
        return out;
    }
    let window = window.max(2);
    let batch = batch.max(1);
    // Head: the one context-only prefix of the whole pass — the only tokens a
    // score run never sees scored. It is as large as the fast prefill path
    // allows, so the coverage loss is paid once per text instead of once per
    // window.
    let head = (len - 2).min(batch).min(window / 2).min((len / 4).max(1));
    let mut scored_from = head;
    while scored_from < len {
        let remaining = len - scored_from;
        // A window can hold `window` positions TOTAL (its context-only prefix
        // plus its scored run): exceeding that is what used to make the
        // runtime loop break early and skip tokens the plan had counted.
        let last = remaining <= window;
        let total = if last { remaining } else { window };
        let prefill = if out.is_empty() {
            head
        } else {
            // Each later window re-prefills the tail of what was just scored
            // as ITS context, so scored runs stay CONTIGUOUS: every token after
            // the head is scored exactly once. The old stride (one-token
            // overlap, prefix repeated per window) silently dropped
            // min(batch, span/2) - 1 tokens between windows — 25% of a corpus
            // at window 512 and 50% at the old default of 64 — and nothing in
            // the output said so.
            (total - 1).min(batch).min(total / 2).min(scored_from).max(1)
        };
        if total <= prefill {
            break;
        }
        let end = if last { len } else { scored_from + (total - prefill) };
        out.push(ScoreWindow { start: scored_from - prefill, prefill, first_scored: scored_from, end });
        scored_from = end;
    }
    out
}

/// NLL of `target` under the full-vocabulary log-softmax of `logits`, in
/// nats. One definition, used by `score_text` and pinned by the tests: base-e
/// log, max-subtracted for stability, whole vocabulary — the same quantity a
/// `SOFUU_DUMP_LOGITS` dump yields when recomputed in f64 off-line.
pub fn log_softmax_nll(logits: &[f32], target: usize) -> f64 {
    let mut mx = f32::NEG_INFINITY;
    for &l in logits {
        if l > mx {
            mx = l;
        }
    }
    let mut sum = 0.0f32;
    for &l in logits {
        sum += (l - mx).exp();
    }
    -((logits[target] - mx) - sum.ln()) as f64
}

/// ppl = exp(mean NLL) = exp(nll_total / scored); NaN when nothing was scored.
pub fn ppl_from_nll(nll_total: f64, scored: usize) -> f64 {
    if scored == 0 {
        f64::NAN
    } else {
        (nll_total / scored as f64).exp()
    }
}

impl InferEngine {
    /// Load a model and its tokenizer from disk (auto-detects context length).
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        Self::load_with_seq_len(model_path, 0)
    }

    /// Load a model with custom context length (0 = auto-detect from metadata).
    /// Auto-detects model format: GGUF, SafeTensors directory, or direct .safetensors file.
    pub fn load_with_seq_len<P: AsRef<Path>>(model_path: P, max_seq_len: usize) -> Result<Self, String> {
        Self::load_inner(model_path, max_seq_len, true, 0)
    }

    /// P7 (long-context §8): KIVI ring load for the CPU engine — the f32 KV
    /// arrays shrink to `kv_ring` rows; the logical window stays
    /// `max_seq_len`. History past the ring lives in the KIVI quantized
    /// streams (prefill ingests chunk-by-chunk; decode appends).
    pub fn load_with_seq_len_kv_ring_cpu<P: AsRef<Path>>(
        model_path: P,
        max_seq_len: usize,
        kv_ring: usize,
    ) -> Result<Self, String> {
        Self::load_inner(model_path, max_seq_len, false, kv_ring)
    }

    /// Load a model without creating or using GPU resources. Useful for explicit
    /// CPU-only requests, especially for large SafeTensors models.
    pub fn load_with_seq_len_cpu<P: AsRef<Path>>(model_path: P, max_seq_len: usize) -> Result<Self, String> {
        Self::load_inner(model_path, max_seq_len, false, 0)
    }

    fn load_inner<P: AsRef<Path>>(
        model_path: P,
        max_seq_len: usize,
        enable_gpu: bool,
        kv_ring: usize,
    ) -> Result<Self, String> {
        let raw_str = model_path.as_ref().to_str().unwrap_or("");
        let clean_path = clean_model_path(raw_str);

        // 1. Directory mode (HuggingFace / SafeTensors / PyTorch / NumPy model directory)
        if clean_path.is_dir() {
            let config_path = clean_path.join("config.json");
            let tokenizer_path = clean_path.join("tokenizer.json");

            if tokenizer_path.exists() {
                let tokenizer = BpeTokenizer::from_hf_file(&tokenizer_path)
                    .map_err(|e| format!("failed to load tokenizer.json from {}: {}", clean_path.display(), e))?;

                // GPT-X3's published tokenizer intentionally has no BOS token.
                // The upstream usage example explicitly passes add_special_tokens=false.
                let directory_architecture = if config_path.exists() {
                    std::fs::read_to_string(&config_path)
                        .ok()
                        .and_then(|content| {
                            serde_json::from_str::<serde_json::Value>(&content)
                                .ok()
                                .and_then(|cfg| {
                                    cfg.get("architectures")
                                        .and_then(|v| v.as_array())
                                        .and_then(|items| items.first())
                                        .and_then(|item| item.as_str())
                                        .map(|s| s.to_string())
                                })
                        })
                } else {
                    None
                };
                let mut tokenizer = tokenizer;
                if directory_architecture.as_deref() == Some("GPTX3ForCausalLM") {
                    tokenizer.bos_token_id = None;
                }

                // Check for LFM2 SafeTensors directory
                let is_lfm2 = if config_path.exists() {
                    let content = std::fs::read_to_string(&config_path).unwrap_or_default();
                    content.contains("Lfm2ForCausalLM") || content.contains("\"lfm2\"")
                } else {
                    false
                };

                if is_lfm2 && clean_path.join("model.safetensors").exists() {
                    let model = InferModel::from_lfm2_dir_with_gpu(&clean_path, max_seq_len, enable_gpu)?;
                    return Self::finalize_device(Self {
                        model,
                        tokenizer,
                        last_prefill_tokens: 0,
                        last_prefill_duration: std::time::Duration::ZERO,
                        last_setup_duration: std::time::Duration::ZERO,
                        last_token_trace: Vec::new(),
                        cached_tokens: Vec::new(),
                    }, enable_gpu);
                }

                // Check other weight files
                for weight_name in &["model.safetensors", "pytorch_model.bin", "model.pt", "model.npz"] {
                    let weight_path = clean_path.join(weight_name);
                    if weight_path.exists() {
                        if let Ok(table) = UnifiedTensorTable::open(&weight_path) {
                            if let Ok(model) = InferModel::from_tensor_table(&table, max_seq_len) {
                                return Self::finalize_device(Self {
                                    model,
                                    tokenizer,
                                    last_prefill_tokens: 0,
                                    last_prefill_duration: std::time::Duration::ZERO,
                                    last_setup_duration: std::time::Duration::ZERO,
                        last_token_trace: Vec::new(),
                                    cached_tokens: Vec::new(),
                                }, enable_gpu);
                            }
                        }
                    }
                }

                let architecture = if config_path.exists() {
                    std::fs::read_to_string(&config_path)
                        .ok()
                        .and_then(|content| {
                            serde_json::from_str::<serde_json::Value>(&content)
                                .ok()
                                .and_then(|cfg| {
                                    cfg.get("architectures")
                                        .and_then(|v| v.as_array())
                                        .and_then(|items| items.first())
                                        .and_then(|item| item.as_str())
                                        .map(|s| s.to_string())
                                })
                        })
                } else {
                    None
                };
                return Err(format!(
                    "unsupported model architecture in {}: {:?}; Sofuu currently supports LFM2 directories and standard GGUF decoder architectures",
                    clean_path.display(),
                    architecture.unwrap_or_else(|| "unknown".to_string())
                ));
            }
        }

        // 2. Direct file mode (.safetensors, .pt, .pth, .npz, .npy)
        let ext = clean_path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if ext == "safetensors" || ext == "pt" || ext == "pth" || ext == "npz" || ext == "npy" {
            if let Some(parent) = clean_path.parent() {
                let config_path = parent.join("config.json");
                let tokenizer_path = parent.join("tokenizer.json");
                if tokenizer_path.exists() {
                    let tokenizer = BpeTokenizer::from_hf_file(&tokenizer_path)
                        .map_err(|e| format!("failed to load tokenizer.json: {}", e))?;

                    let is_lfm2 = if config_path.exists() {
                        let content = std::fs::read_to_string(&config_path).unwrap_or_default();
                        content.contains("Lfm2ForCausalLM") || content.contains("\"lfm2\"")
                    } else {
                        false
                    };

                    if is_lfm2 && ext == "safetensors" {
                        let model = InferModel::from_lfm2_dir_with_gpu(parent, max_seq_len, enable_gpu)?;
                        return Self::finalize_device(Self {
                            model,
                            tokenizer,
                            last_prefill_tokens: 0,
                            last_prefill_duration: std::time::Duration::ZERO,
                            last_setup_duration: std::time::Duration::ZERO,
                        last_token_trace: Vec::new(),
                            cached_tokens: Vec::new(),
                        }, enable_gpu);
                    }

                    if let Ok(table) = UnifiedTensorTable::open(&clean_path) {
                        let model = InferModel::from_tensor_table(&table, max_seq_len)?;
                        return Self::finalize_device(Self {
                            model,
                            tokenizer,
                            last_prefill_tokens: 0,
                            last_prefill_duration: std::time::Duration::ZERO,
                            last_setup_duration: std::time::Duration::ZERO,
                        last_token_trace: Vec::new(),
                            cached_tokens: Vec::new(),
                        }, enable_gpu);
                    }
                }
            }
        }

        // 3. GGUF format
        let t0 = std::time::Instant::now();
        let mut gguf = GgufFile::open(&clean_path)
            .map_err(|e| format!("failed to open model {}: {}", clean_path.display(), e))?;
        let t_open = t0.elapsed();

        let t1 = std::time::Instant::now();
        let tokenizer = BpeTokenizer::from_gguf(&mut gguf)
            .ok_or_else(|| "failed to parse tokenizer from GGUF metadata".to_string())?;
        let t_tok = t1.elapsed();

        let t2 = std::time::Instant::now();
        let mut tokenizer = tokenizer;
        // Some GGUFs (e.g. spark2_5) explicitly disable BOS; honoring the flag
        // avoids injecting <unk> as a bogus sentence start.
        if matches!(
            gguf.metadata.get("tokenizer.ggml.add_bos_token"),
            Some(self::gguf::GgufValue::Bool(false))
        ) {
            tokenizer.bos_token_id = None;
        }
        let vocab_size = tokenizer.vocab.len();
        let model = InferModel::from_gguf_with_gpu(gguf, max_seq_len, vocab_size, enable_gpu, kv_ring)?;
        let t_model = t2.elapsed();

        if std::env::var("SOFUU_PROFILE_LOAD").is_ok() {
            eprintln!("  [profile] GgufFile::open: {:.2?}, BpeTokenizer: {:.2?}, InferModel: {:.2?}", t_open, t_tok, t_model);
        }

        Ok(Self {
            model,
            tokenizer,
            last_prefill_tokens: 0,
            last_prefill_duration: std::time::Duration::ZERO,
            last_setup_duration: std::time::Duration::ZERO,
                        last_token_trace: Vec::new(),
            cached_tokens: Vec::new(),
        })
    }

    #[allow(unused_variables)]
    fn finalize_device(mut engine: Self, enable_gpu: bool) -> Result<Self, String> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            engine.model.set_gpu_layers(if enable_gpu { usize::MAX } else { 0 });
        }
        Ok(engine)
    }

    /// Explicitly clear the KV cache and cached token history.
    pub fn reset_cache(&mut self) {
        self.model.reset_cache();
        self.cached_tokens.clear();
    }

    pub fn forward_embed(&mut self, token: u32, out_hidden: &mut [f32]) {
        self.model.forward_embed(token, out_hidden);
    }

    pub fn forward_layer_range(
        &mut self,
        pos: usize,
        start_layer: usize,
        end_layer: usize,
        in_hidden: &[f32],
        out_hidden: &mut [f32],
    ) {
        self.model.forward_layer_range(pos, start_layer, end_layer, in_hidden, out_hidden);
    }

    pub fn forward_head(&mut self, in_hidden: &[f32], out_logits: &mut [f32]) {
        self.model.forward_head(in_hidden, out_logits);
    }

    /// Teacher-forced NLL scoring (PLAN-LONG-CONTEXT-PERSISTENCE §8 P3): run
    /// the text through the model and accumulate the NLL of every next token
    /// given its left context, in sliding windows of `window` tokens (each
    /// window re-prefills the tail of the previous one as context, so the
    /// scored runs are contiguous; see `score_windows`).
    /// This is the surface Gate 2 needs (PPL ratio f16 vs KIVI) and the only
    /// honest perplexity measurement available without touching the prefill
    /// kernels: the batched prefill keeps only the LAST token's logits, so
    /// every other scored position costs one single-token `forward` (decode
    /// speed — a 4k-token slice is minutes, not seconds).
    ///
    /// Per window: one batched prefill of the first `prefill` tokens (the fast
    /// GEMM path; those rows are CONTEXT ONLY — their logits are not computed),
    /// then per-token forwards for the rest, scoring from the returned logits
    /// before each step. With `options.kivi`, KIVI is armed after every
    /// window's prefill (the same prefill-FP32 → decode-quantized flow as
    /// `stream`), so f16-vs-KIVI ratios measure exactly the shipped decode
    /// path. The multi-turn prefix cache is cleared and left cleared.
    ///
    /// CONVENTION (the numbers are only comparable to another tool when these
    /// match): ppl = exp(mean NLL) over the SCORED tokens, natural log,
    /// full-vocab log-softmax, token 0 always unscored, `window` = the left
    /// context a scored token can see (default `DEFAULT_SCORE_WINDOW` = 512 =
    /// llama-perplexity's default `n_ctx`), and each window's first `prefill`
    /// tokens are context only — so `scored` < `tokens_total` - 1 and is
    /// reported alongside the ppl. A ppl over a handful of tokens is a
    /// handful-of-samples estimate, not a perplexity: `MIN_SCORE_SAMPLE`
    /// exists because a 1-token "ppl" (45.19 on one token) was once reported
    /// next to a 1.42 over a thousand.
    pub fn score_text(
        &mut self,
        text: &str,
        options: GenerateOptions,
        window: usize,
    ) -> Result<ScoreStats, String> {
        let tokens = self.tokenizer.encode(text);
        if tokens.len() < 8 {
            return Err(format!("score_text needs >= 8 tokens, got {}", tokens.len()));
        }
        let vocab = self.model.config.vocab_size;
        let max_seq = self.model.config.max_seq_len;
        let window = if window == 0 { DEFAULT_SCORE_WINDOW } else { window };
        let window = window.max(8).min(max_seq);
        let batch = self::model::max_prefill_batch().min(window / 2).max(1);

        let mut stats = ScoreStats {
            tokens_total: tokens.len(),
            window,
            scored: 0,
            windows: 0,
            nll_total: 0.0,
            ppl: f64::NAN,
        };
        let mut logits = vec![0.0f32; vocab];
        for w in score_windows(tokens.len(), window, batch) {
            self.model.reset_cache();
            self.cached_tokens.clear();
            let lg = self
                .model
                .forward_prefill_all_with_logits(&tokens[w.start..w.start + w.prefill], 0);
            logits.copy_from_slice(lg);
            let mut kvpos = w.prefill;
            if options.kivi {
                self.model.enable_kivi(options.kivi_bits, kvpos);
            }
            let mut i = w.first_scored;
            loop {
                /* logits currently predict tokens[i] — score it. */
                let t = tokens[i] as usize;
                if t < vocab {
                    stats.nll_total += log_softmax_nll(&logits, t);
                    stats.scored += 1;
                }
                i += 1;
                if i >= w.end || kvpos + 1 >= window {
                    break;
                }
                let out = self.model.forward(tokens[i - 1], kvpos);
                logits.copy_from_slice(&out[..vocab]);
                kvpos += 1;
            }
            stats.windows += 1;
        }
        stats.ppl = ppl_from_nll(stats.nll_total, stats.scored);
        Ok(stats)
    }

    /// Generate text for a prompt, returning the complete generated string.
    pub fn generate(&mut self, prompt: &str, options: GenerateOptions) -> Result<String, String> {
        let mut output = String::new();
        self.stream(prompt, options, |token| {
            output.push_str(token);
            true
        })?;
        Ok(output)
    }

    /// Stream generated tokens one by one through a callback.
    /// The callback can return `false` to abort generation early.
    pub fn stream<F>(&mut self, prompt: &str, options: GenerateOptions, mut on_token: F) -> Result<String, String>
    where
        F: FnMut(&str) -> bool,
    {
        // A fresh trace per call — callers read it after stream returns.
        self.last_token_trace.clear();
        // Auto mode adjusts the pool per phase; restore the caller's count on
        // every exit path so a stream() call never leaks its decode
        // configuration to other users of the global pool.
        struct PoolRestore(usize);
        impl Drop for PoolRestore {
            fn drop(&mut self) {
                self::quant::set_thread_count(self.0);
            }
        }
        // Restore the caller's thread count on every exit, not only in auto
        // mode: an explicit --threads used to leak into later stream() calls
        // on the same engine.
        let prev_threads = self::quant::get_thread_count();
        let _pool_guard = PoolRestore(prev_threads);
        if let Some(which) = std::env::var_os("SOFUU_DEBUG_POISON") {
            let which = which.to_string_lossy().to_string();
            self.model.poison_scratch(1.0, &which);
        }
        let pool_base = if options.threads == 0 {
            Some(prev_threads)
        } else {
            None
        };
        if options.threads > 0 {
            self::quant::set_thread_count(options.threads);
        }
        if options.sliding_window > 0 {
            // Forward into the backend: attention reads the model's own
            // sliding_window, so writing only InferModel's field made --swa
            // and GenerateOptions.sliding_window inert.
            self.model.set_sliding_window(options.sliding_window);
        }
        if options.sink_tokens > 0 {
            self.model.sink_tokens = options.sink_tokens;
        }

        let max_seq_len = self.model.config.max_seq_len;
        // P7 ring-KV: the f32 arrays are ring-strided (config.max_seq_len is
        // the ring); the LOGICAL window governs refusals and compaction
        // triggers. Equal to max_seq_len when ring mode is off.
        let logical_max_seq_len = self.model.logical_seq_len();
        let ring_mode = options.kivi && self.model.kv_ring_mode();
        let vocab_size = self.model.config.vocab_size;
        let mut logits_buf = vec![0.0f32; vocab_size];
        // Setup work (tokenize, cache reset, SnapKV/KIVI) is reported apart
        // from prefill so a long-prompt turn cannot look like slow decode.
        // The bucket is filled twice: the part before prefill and the
        // SnapKV/KIVI part after it, never the prefill itself.
        let t_setup = std::time::Instant::now();
        self.last_setup_duration = std::time::Duration::ZERO;
        // Compaction rewrites KV positions, so the token order in
        // cached_tokens no longer maps onto KV rows afterwards.
        let mut compacted = false;
        let pool_failures_before = self::quant::pool_failures();

        // 1. Tokenize prompt
        let mut input_tokens = self.tokenizer.encode(prompt);
        // DIAGNOSTIC ONLY (never set in shipped runs): truncate the token
        // stream so a numerics/speed sweep can compare two arms at an
        // identical token prefix, including lengths that no text cut hits.
        if let Some(lim) = std::env::var("SOFUU_PREFILL_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            if lim > 0 && lim < input_tokens.len() {
                input_tokens.truncate(lim);
            }
        }
        if std::env::var_os("SOFUU_DEBUG_DETERMINISM").is_some() {
            eprintln!(
                "[determ] input_tokens.len {} = {:?}",
                input_tokens.len(),
                &input_tokens[..input_tokens.len().min(16)]
            );
        }
        if input_tokens.is_empty() {
            return Err("prompt produced zero tokens".to_string());
        }

        // 2. Multi-turn prefix cache reuse
        let mut common_prefix = 0;
        while common_prefix < self.cached_tokens.len()
            && common_prefix < input_tokens.len()
            && self.cached_tokens[common_prefix] == input_tokens[common_prefix]
        {
            common_prefix += 1;
        }

        // If not a pure extension of cached context, reset cache. A cache at
        // or beyond the context window is refused too: reusing it would start
        // the new sequence past max_seq_len and compaction would run on a
        // current_seq_len that no longer fits the buffer.
        if common_prefix == 0
            || common_prefix < self.cached_tokens.len()
            || self.cached_tokens.len() > max_seq_len
        {
            self.model.reset_cache();
            self.cached_tokens.clear();
            common_prefix = 0;
        }

        let sink = self.model.sink_tokens.min(max_seq_len / 4).max(1);
        let drop = ((max_seq_len - sink) / 4).max(1).min((max_seq_len - sink).saturating_sub(1));
        let chunk_size = self::model::max_prefill_batch().min((max_seq_len - sink).saturating_sub(drop).max(1));

        // P7 ring-KV: resumed prompts are unsupported — the quantized
        // streams carry history and expect a from-zero prefill.
        if ring_mode && common_prefix > 0 {
            return Err(
                "--kivi ring mode carries history in the quantized streams and does not support resumed prompts (common_prefix > 0) — start the session fresh".to_string(),
            );
        }

        // F-LC1 (PLAN-LONG-CONTEXT-PERSISTENCE §8 P2): a FRESH prompt that
        // fills the window used to trigger mid-PREFILL compaction — the loop
        // below silently dropped middle KV rows (a 50%-depth needle in an 8k
        // haystack vanished; the run still printed a normal stats line).
        // Refuse instead: the caller must raise --ctx or shorten the prompt.
        // Continuations (common_prefix > 0) keep the compaction path — chat
        // legitimately extends past the window — but it now warns loudly.
        if common_prefix == 0 && input_tokens.len() >= logical_max_seq_len {
            return Err(format!(
                "prompt is {} tokens but the context window is only {} — mid-prompt compaction would silently drop content; raise --ctx or shorten the prompt",
                input_tokens.len(),
                logical_max_seq_len
            ));
        }

        // 3. Prefill phase for new tokens
        let mut cur_pos = common_prefix;
        self.last_setup_duration = t_setup.elapsed();
        let t_prefill = std::time::Instant::now();

        if common_prefix == 0 && input_tokens.len() < logical_max_seq_len {
            // Ring mode arms the KIVI streams FIRST so the streaming prefill
            // ingests every chunk as it is computed (history past the f32
            // ring lives only in the quantized store; decode appends join
            // the same streams).
            if ring_mode {
                self.model.enable_kivi_streaming(options.kivi_bits);
            }
            // Fast-path: unified prefill with last-token logits projection
            let initial_logits = self.model.forward_prefill_all_with_logits(&input_tokens, 0);
            if std::env::var_os("SOFUU_DEBUG_DETERMINISM").is_some() {
                eprintln!("[determ] prefill_logits[..8] = {:?}", &initial_logits[..8]);
            }
            // DIAGNOSTIC ONLY: dump the WHOLE last-row logit vector so a
            // GPU-vs-CPU numeric comparison covers every vocabulary entry
            // instead of the 8 printed above. Raw little-endian f32.
            if let Some(p) = std::env::var_os("SOFUU_DUMP_LOGITS") {
                let mut bytes = Vec::with_capacity(initial_logits.len() * 4 + 64);
                bytes.extend_from_slice(b"SFLOG01\0");
                bytes.extend_from_slice(&(initial_logits.len() as u64).to_le_bytes());
                for v in initial_logits.iter() {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                if let Err(e) = std::fs::write(&p, &bytes) {
                    eprintln!("[logits-dump] write {} failed: {e}", p.to_string_lossy());
                } else {
                    let mut amax = 0.0f32;
                    let mut argmax = 0usize;
                    for (i, v) in initial_logits.iter().enumerate() {
                        let a = v.abs();
                        if a > amax { amax = a; }
                        if *v > initial_logits[argmax] { argmax = i; }
                    }
                    eprintln!(
                        "[logits-dump] {} n={} maxabs={:.6} argmax={} top={:.6}",
                        p.to_string_lossy(),
                        initial_logits.len(),
                        amax,
                        argmax,
                        initial_logits[argmax]
                    );
                }
            }
            logits_buf.copy_from_slice(initial_logits);
            cur_pos = input_tokens.len();
            self.last_prefill_tokens = input_tokens.len();
        } else if common_prefix < input_tokens.len() {
            let new_tokens = &input_tokens[common_prefix..];
            let n_new = new_tokens.len();
            self.last_prefill_tokens = n_new;

            // Prefill new tokens up to the last one
            if n_new > 1 {
                for chunk in new_tokens[..n_new - 1].chunks(chunk_size) {
                    while cur_pos + chunk.len() >= max_seq_len && cur_pos > sink + drop {
                        compacted = true;
                        cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
                    }
                    if cur_pos + chunk.len() >= max_seq_len {
                        for &tok in chunk {
                            while cur_pos >= max_seq_len && cur_pos > sink + drop {
                                compacted = true;
                                cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
                            }
                            self.model.forward_prefill(tok, cur_pos);
                            cur_pos += 1;
                        }
                    } else {
                        self.model.forward_prefill_batch(chunk, cur_pos);
                        cur_pos += chunk.len();
                    }
                }
            }

            // Final prompt token computes logits
            let last_tok = new_tokens[n_new - 1];
            while cur_pos >= max_seq_len && cur_pos > sink + drop {
                compacted = true;
                cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
            }
            let initial_logits = self.model.forward(last_tok, cur_pos);
            logits_buf.copy_from_slice(initial_logits);
            cur_pos += 1;
            if compacted {
                eprintln!(
                    "[sofuu infer] WARNING: the context window filled during prefill — KV compaction dropped middle tokens (prompt {} + prefix {} vs window {}); raise --ctx to keep the whole prompt",
                    input_tokens.len(),
                    common_prefix,
                    max_seq_len
                );
            }
        } else {
            // The prompt is already in the KV, so there are no new tokens to
            // compute logits from. Re-run its last token: forward() rewrites
            // that KV row and returns the logits the first turn produced.
            // (The rewrite goes through the single-token path, which can
            // differ from the batched row by round-off; it is the only way to
            // get logits without a full re-prefill.) Without this the buffer
            // stays zeroed and the sampler's argmax returns token 0.
            self.last_prefill_tokens = 0;
            let last_tok = input_tokens[input_tokens.len() - 1];
            if cur_pos > 0 {
                let logits = self.model.forward(last_tok, cur_pos - 1);
                logits_buf.copy_from_slice(logits);
            }
        }
        self.last_prefill_duration = t_prefill.elapsed();
        if self::quant::pool_failures() > pool_failures_before {
            return Err(format!(
                "a worker pool task failed during prefill ({} total); the attention/GEMM result would be incomplete",
                self::quant::pool_failures()
            ));
        }

        // SnapKV Observation Window Compaction (Prunes inactive middle blocks after prefill)
        let t_setup_post = std::time::Instant::now();
        if options.snapkv && !ring_mode && cur_pos > 64 {
            let snap_sink = self.model.sink_tokens.max(16).min(cur_pos / 4);
            let keep_recent = 32.min(cur_pos - snap_sink);
            let middle = cur_pos.saturating_sub(snap_sink + keep_recent);
            if middle >= 16 {
                let drop = (middle * 3) / 4;
                compacted = true;
                cur_pos = self.model.compact_kv_cache(cur_pos, snap_sink, drop);
            }
        }

        if options.dynamic_ffn {
            self.model.set_dynamic_ffn(true);
        }

        if options.kivi && !ring_mode {
            self.model.enable_kivi(options.kivi_bits, cur_pos);
        } else if ring_mode {
            // P7 streaming arm: KIVI was armed before the prefill and holds
            // every chunk already — the post-hoc f32 re-ingest would read a
            // wrapped ring and must not run.
        } else {
            // A previous turn may have enabled KIVI. Its cache does not hold
            // the tokens this call just prefilled, so decode would attend a
            // truncated history; drop it and use the plain KV path.
            self.model.disable_kivi();
        }

        if options.streaming_weights {
            self.model.set_streaming_weights(true);
        }
        self.last_setup_duration += t_setup_post.elapsed();

        // 4. Decode Phase
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: options.temperature,
            top_p: options.top_p,
            top_k: options.top_k,
            repetition_penalty: options.repetition_penalty,
        });

        let mut history = input_tokens.clone();
        let mut full_generated = String::new();
        let eos_id = self.tokenizer.eos_token_id;

        // Lookahead's speculative batches (2-3 rows) cost a single weight
        // stream only on the Gguf backend's batched tile path; every other
        // backend falls back to a serial per-token speculative loop that
        // multiplies weight streams, so the flag is ignored there.
        // See PLAN-TARGET-1200-1500.md 12c: speculation measured 2x SLOWER than serial decode on
        // this machine (13.1 vs 28.0 tk/s, 3 runs each, identical greedy output), and the
        // adaptive gate cannot see it because a spec step here costs ~2 serial forwards while
        // its threshold is 1.15 tokens/step. Timing-based switching is forbidden (it would make
        // greedy output depend on machine noise), so the default is OFF and constant.
        // SOFUU_SPEC=1 re-enables the tree.
        // A custom rope profile (phi3 longrope / llama3 scaling) is not
        // implemented in the tree verify pass, so the tree is off for those
        // models whatever SOFUU_SPEC says.
        let spec_wanted = std::env::var("SOFUU_SPEC").map(|v| v == "1").unwrap_or(false)
            && self.model.spec_decode_supported();
        let mut ngram_cache = if spec_wanted && options.lookahead && matches!(self.model.backend, ModelBackend::Gguf(_)) {
            let mut cache = lookahead::NgramCache::new(2048);
            cache.ingest_sequence(&input_tokens);
            Some(cache)
        } else {
            None
        };

        let mut spec_logits_buf = vec![0.0f32; vocab_size];
        // Per-node head logits for one speculative step (grown on first use).
        let mut spec_logits_batch: Vec<f32> = Vec::new();

        // Adaptive speculation: a spec step costs ~one serial step plus a
        // little extra row work, so it only pays when the draft chain lands
        // more than ~1.15 tokens per step. Models/prompts where drafts miss
        // (short generations, natural text on some models) measured SLOWER
        // with the tree on (MiniCPM 5.2 vs 5.7 tk/s, Llama-1B short EOS
        // window). After a warmup window the stream disables the tree for
        // the rest of the generation when acceptance is below break-even.
        // SOFUU_SPEC_ADAPT=0 restores always-on; SOFUU_SPEC_MIN_ACCEPT
        // overrides the 1.15 tokens/step threshold.
        let spec_adapt = std::env::var("SOFUU_SPEC_ADAPT").map(|v| v != "0").unwrap_or(true);
        let spec_warmup: usize = std::env::var("SOFUU_SPEC_WARMUP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4);
        let spec_min_accept = std::env::var("SOFUU_SPEC_MIN_ACCEPT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(self.model.spec_min_accept_default());
        // A2/W4: hoisted out of the generation loop (these were read once PER TOKEN).
        let dec_profile = std::env::var_os("SOFUU_DEC_PROFILE").is_some();
        let det_debug = std::env::var_os("SOFUU_DEBUG_DETERMINISM").is_some();
        let gpu_profile_spec = std::env::var_os("SOFUU_GPU_PROFILE").is_some();
        let mut spec_steps: u32 = 0;
        let mut spec_accepted: u32 = 0;
        let mut spec_disabled = false;
        // Self-calibrating gate state. The fixed tokens/step threshold cannot
        // be right for every model: a 5-row tree costs ~2.7 serial tokens of
        // time on the Pi (65-102 ms vs 42 ms), so acceptance has to clear
        // 2.7x, not 1.15x. Measure both rates instead of guessing.
        //
        // The rate comparison reads wall-clock time, so a gate that acts on it
        // makes greedy output depend on machine noise: two identical runs can
        // disable the tree at different steps and diverge on a near-tie.
        // Acceptance-based switching (spec_should_disable) is a function of
        // the token stream alone, so it is the default; set
        // SOFUU_SPEC_RATE_GATE=1 to also let measured rates switch the tree
        // off (faster in theory, not reproducible run to run).
        let spec_rate_adapt =
            std::env::var("SOFUU_SPEC_RATE_GATE").as_deref() == Ok("1");
        let mut serial_ns: u64 = 0;
        let mut serial_tokens: u32 = 0;
        let mut spec_ns: u64 = 0;
        let mut spec_tokens: u32 = 0;
        let mut spec_decided = false;

        let mut step = 0;
        let mut cur_token = sampler.sample(&mut logits_buf, &history);

        // Phase-tuned pool when the caller left threads on auto: prefill
        // keeps every core (GEMM panels scale), decode narrows for q8_0.
        if let Some(base) = pool_base {
            let decode_threads = self.model.decode_threads_auto(base);
            if quant::prof::enabled() {
                eprintln!("[pool] auto: base={base} decode={decode_threads}");
            }
            if decode_threads != base {
                self::quant::set_thread_count(decode_threads);
            }
        }

        while step < options.max_tokens {
            // Per-token profile scope (SOFUU_PROFILE): dropped at every
            // break path; report() prints every 32 steps with fresh buckets.
            let _tok_prof = quant::prof::TokenScope::new();
            if quant::prof::enabled() && step % 32 == 0 {
                quant::prof::report();
            }
            let next_token = cur_token;
            step += 1;

            if let Some(ref mut cache) = ngram_cache {
                cache.record_token(next_token);
            }

            // 1. Check EOS token ID
            if let Some(eos) = eos_id {
                if next_token == eos {
                    break;
                }
            }

            // 2. Check special stop tokens by raw string (e.g. <|im_end|>, <|endoftext|>, <|eot_id|>)
            if let Some(raw_tok) = self.tokenizer.decode_token(next_token) {
                let is_stop = raw_tok == "<|im_end|>"
                    || raw_tok == "<|endoftext|>"
                    || raw_tok == "<|eot_id|>"
                    || options.stop_tokens.iter().any(|s| s == raw_tok);
                if is_stop {
                    break;
                }
            }

            let piece = self.tokenizer.decode_piece(next_token);
            if !piece.is_empty() {
                // Check stop token strings in decoded piece
                let mut should_stop = false;
                for stop in &options.stop_tokens {
                    if piece.contains(stop) || full_generated.ends_with(stop) {
                        should_stop = true;
                        break;
                    }
                }
                if should_stop {
                    break;
                }

                full_generated.push_str(&piece);
                let continue_gen = on_token(&piece);
                if !continue_gen {
                    break;
                }
            }

            history.push(next_token);

            if step >= options.max_tokens {
                break;
            }

            // KV Cache auto-compaction before advancing. Ring mode skips it
            // entirely: the KIVI streams are the history of record, positions
            // never shift, and the ring's wrapped f32 rows are dead weight.
            while !ring_mode && cur_pos >= max_seq_len && cur_pos > sink + drop {
                compacted = true;
                cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
                // F-LC3: the compaction dropped middle tokens, shifted the
                // survivors and re-roped them — the KIVI stream built before
                // this point would keep attending the dropped history under
                // positions that no longer exist. Re-ingest from the
                // compacted cache.
                if options.kivi {
                    self.model.rebuild_kivi_after_compact(cur_pos);
                }
            }

            // Propose multi-path speculative candidate tree from n-gram cache
            let speculative_tree = if let Some(ref cache) = ngram_cache {
                if !spec_disabled && cur_pos + 6 < max_seq_len && step < options.max_tokens {
                    // Candidate tree knobs: SOFUU_SPEC_ROWS = max_tree_size (the
                    // spec tile batch pads to 8, so 1..=8 is the legal range),
                    // SOFUU_SPEC_DEPTH 1..=4 (level 4 only earns its keep with
                    // width[3] > 0), and SOFUU_SPEC_WIDTHS="l1,l2,l3,l4" for
                    // explicit sweeps. Device sweeps (Pi 5, greedy, Llama-1B
                    // Q8_0): level-1 width 2 dominates — every rank>=3
                    // candidate row costs attention/act-quant work that beats
                    // its acceptance rate, so the default stays the legacy
                    // (2,2,1) shape; rows=8 + "2,2,1,1" measured +1% on
                    // templated text but -20% on natural text.
                    let depth = std::env::var("SOFUU_SPEC_DEPTH")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(3)
                        .clamp(1, 4);
                    // GPU spec route: the verify pass runs the chain
                    // (root + depth candidates at consecutive positions) in
                    // ONE commit with the B=4 batched GEMVs — so shape the
                    // draft as the width-1 chain that pass accepts.
                    let (rows, widths) = if self.model.gpu_spec_chain_armed() {
                        let d = depth.min(3);
                        (d.max(1), [1, 1, 1, 0])
                    } else {
                        let rows = std::env::var("SOFUU_SPEC_ROWS")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or_else(|| self.model.spec_rows_default())
                            .clamp(1, 8);
                        let widths = std::env::var("SOFUU_SPEC_WIDTHS")
                            .ok()
                            .and_then(|s| {
                                let p: Vec<usize> = s
                                    .split(',')
                                    .filter_map(|x| x.trim().parse().ok())
                                    .collect();
                                (p.len() == 4).then_some([p[0], p[1], p[2], p[3]])
                            })
                            .unwrap_or(match rows {
                                8 => [3, 2, 2, if depth >= 4 { 1 } else { 0 }],
                                7 => [3, 2, 2, 0],
                                6 => [2, 2, 2, 0],
                                5 => [2, 2, 1, 0],
                                _ => [2, 1, 1, 0],
                            });
                        (rows, widths)
                    };
                    // The tree forward runs inside the prefill batch buffers, so
                    // a tree with more nodes than the configured batch overruns
                    // them (SOFUU_PREFILL_BATCH=4 with a 5-row tree used to
                    // panic in the batched embedding lookup). Clamp the draft
                    // to what the buffers hold; row 0 is the root.
                    let rows = rows.min(self.model.prefill_batch_rows().saturating_sub(1));
                    if rows == 0 {
                        None
                    } else {
                        let tree =
                            lookahead::CandidateTree::build_cfg(cache, &history, rows, depth, widths);
                        if !tree.is_empty() {
                            Some(tree)
                        } else {
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Warmup: run plain serial decode for a few steps so the gate has
            // a measured serial rate to compare speculation against.
            let speculative_tree = if spec_allowed_at(step, spec_warmup) {
                speculative_tree
            } else {
                None
            };

            if let Some(tree) = speculative_tree {
                // MULTI-PATH LOOKAHEAD TREE SPECULATION:
                // Evaluates [next_token, tree.tokens()...] in parallel in a single forward pass!
                // Model weights are read from DRAM ONLY ONCE.
                let mut batch_tokens = Vec::with_capacity(1 + tree.size());
                batch_tokens.push(next_token);
                batch_tokens.extend(tree.tokens());

                let positions = tree.build_positions(cur_pos);
                let ancestry_mask = tree.build_full_batch_ancestry_mask();
                let prof_spec = gpu_profile_spec;
                let t_verify = std::time::Instant::now();

                self.model.forward_speculative_tree(
                    &batch_tokens,
                    &positions,
                    &ancestry_mask,
                    cur_pos,
                    &mut spec_logits_buf,
                );

                // Sample prediction for root token (item 0 in batch)
                logits_buf.copy_from_slice(&spec_logits_buf[0..vocab_size]);
                let root_predicted = sampler.sample(&mut logits_buf, &history);
                decode_step_probe(&logits_buf, "gen_root", cur_pos, root_predicted);

                // Greedily traverse the tree along matching branches. The walk
                // asks for the head's prediction at every node it visits, so on
                // the CPU path compute all nodes' logits in one pass over the
                // lm head up front (lazily that is one full 127 MB read per
                // node, which made spec steps 65-102 ms and lost to serial
                // decode despite 2.06 tokens/step). Row 0 is the root, whose
                // logits the tree forward already produced.
                let tree_rows = tree.size();
                // Capacity check: the CPU head pass works on the prefill batch
                // buffers, so a tree that does not fit (SOFUU_PREFILL_BATCH
                // small) must fall back to the per-node path rather than
                // overrun them.
                let batched_heads = tree_rows > 0
                    && !self.model.gpu_chain_heads()
                    && 1 + tree_rows <= self.model.prefill_batch_rows();
                if batched_heads {
                    if spec_logits_batch.len() < tree_rows * vocab_size {
                        spec_logits_batch.resize(tree_rows * vocab_size, 0.0);
                    }
                    self.model.spec_head_logits_batch(
                        1,
                        tree_rows,
                        &mut spec_logits_batch[..tree_rows * vocab_size],
                    );
                }
                let (accepted_tokens, accepted_batch_indices, next_sampled) = tree.verify_path(
                    root_predicted,
                    |batch_idx| {
                        if batched_heads && batch_idx >= 1 {
                            let off = (batch_idx - 1) * vocab_size;
                            spec_logits_buf
                                .copy_from_slice(&spec_logits_batch[off..off + vocab_size]);
                        } else if !self
                            .model
                            .spec_chain_head_logits(batch_idx, &mut spec_logits_buf)
                        {
                            self.model
                                .forward_speculative_head_item(batch_idx, &mut spec_logits_buf);
                        }
                        let node_tok = sampler.sample(&mut spec_logits_buf[0..vocab_size], &history);
                        decode_step_probe(&spec_logits_buf, "gen_node", cur_pos, node_tok);
                        node_tok
                    },
                );

                // Commit accepted candidate KV vectors to persistent KV cache
                if !accepted_batch_indices.is_empty() {
                    self.model.commit_speculative_candidates(&accepted_batch_indices, cur_pos);
                }

                // Adaptive gate: measure tokens-per-spec-step and stop paying
                // for drafts that do not land. The step is charged for its
                // whole cost — forward, head logits, verify AND the KV commit —
                // so the rate compared against serial decode is like for like
                // (timing it before the commit silently flattered the tree).
                spec_steps += 1;
                spec_accepted += accepted_batch_indices.len() as u32;
                spec_ns += t_verify.elapsed().as_nanos() as u64;
                spec_tokens += 1 + accepted_batch_indices.len() as u32;
                if spec_adapt && !spec_disabled && !spec_decided {
                    let lose = spec_should_disable(spec_steps, spec_accepted, spec_min_accept)
                        || (spec_rate_adapt
                            && spec_is_losing(spec_ns, spec_tokens, serial_ns, serial_tokens));
                    // Decide ONCE, as soon as both samples exist: the serial
                    // rate only comes from the warmup window (no serial steps
                    // run afterwards), so re-deciding later would judge fresh
                    // tree costs against a stale baseline — that is how a
                    // transient warmup rate could switch a winning tree off
                    // mid-generation.
                    if lose || (spec_steps >= 8 && serial_tokens >= 4) {
                        spec_disabled = lose;
                        spec_decided = true;
                    }
                    if spec_disabled && quant::prof::enabled() {
                        let spec_rate = spec_tokens as f64 * 1e9 / spec_ns as f64;
                        let serial_rate = serial_tokens as f64 * 1e9 / serial_ns as f64;
                        eprintln!(
                            "[spec] gate: off — {spec_rate:.1} tok/s with tree ({spec_tokens}/{spec_steps} steps) vs {serial_rate:.1} serial"
                        );
                    }
                }
                if prof_spec {
                    eprintln!(
                        "[spec] fwd+verify+commit={}us accepted={} (of batch {})",
                        t_verify.elapsed().as_micros() as u64,
                        accepted_batch_indices.len(),
                        batch_tokens.len()
                    );
                }

                // Advance cur_pos by 1 (root) + accepted candidate count
                cur_pos += 1 + accepted_batch_indices.len();

                // Process bonus speculative accepted candidate tokens
                let mut early_stopped = false;
                for &acc_tok in accepted_tokens.iter().take(accepted_batch_indices.len()) {
                    step += 1;
                    if let Some(ref mut cache) = ngram_cache {
                        cache.record_token(acc_tok);
                    }

                    if let Some(eos) = eos_id {
                        if acc_tok == eos {
                            early_stopped = true;
                            break;
                        }
                    }

                    if let Some(raw_tok) = self.tokenizer.decode_token(acc_tok) {
                        let is_stop = raw_tok == "<|im_end|>"
                            || raw_tok == "<|endoftext|>"
                            || raw_tok == "<|eot_id|>"
                            || options.stop_tokens.iter().any(|s| s == raw_tok);
                        if is_stop {
                            early_stopped = true;
                            break;
                        }
                    }

                    let cand_piece = self.tokenizer.decode_piece(acc_tok);
                    if !cand_piece.is_empty() {
                        let mut should_stop = false;
                        for stop in &options.stop_tokens {
                            if cand_piece.contains(stop) || full_generated.ends_with(stop) {
                                should_stop = true;
                                break;
                            }
                        }
                        if should_stop {
                            early_stopped = true;
                            break;
                        }

                        full_generated.push_str(&cand_piece);
                        if !on_token(&cand_piece) {
                            early_stopped = true;
                            break;
                        }
                    }

                    history.push(acc_tok);
                    if step >= options.max_tokens {
                        early_stopped = true;
                        break;
                    }
                }

                if early_stopped {
                    break;
                }

                cur_token = next_sampled;
            } else {
                // Standard baseline forward pass (1 token)
                let t_serial = std::time::Instant::now();
                let __dt = if dec_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let next_logits = self.model.forward(next_token, cur_pos);
        if let Some(t) = __dt {
            eprintln!("[dp] pos={} forward={:.2}ms", cur_pos, t.elapsed().as_secs_f64() * 1000.0);
        }
                if det_debug && step <= 2 {
                    eprintln!(
                        "[determ] step {} tok {} pos {} logits[0..8] = {:?}",
                        step,
                        next_token,
                        cur_pos,
                        &next_logits[..8.min(next_logits.len())]
                    );
                }
                // DIAGNOSTIC ONLY: the decode path's own full-vocabulary logit
                // vector for this step (SOFUU_DUMP_LOGITS covers the prefill
                // projection, which does not run the b=1 GEMV at all). Written
                // raw little-endian f32 in the same SFLOG01 container so two
                // binaries/kernels can be compared bit for bit.
                if step == 1 {
                    if let Ok(p) = std::env::var("SOFUU_DUMP_LOGITS_DEC") {
                        let mut bytes = Vec::with_capacity(next_logits.len() * 4 + 64);
                        bytes.extend_from_slice(b"SFLOG01\0");
                        bytes.extend_from_slice(&(next_logits.len() as u64).to_le_bytes());
                        for v in next_logits.iter() {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                        match std::fs::write(&p, &bytes) {
                            Ok(()) => eprintln!(
                                "[logits-dump-dec] {} n={} argmax={} top={:.6}",
                                p,
                                next_logits.len(),
                                next_logits.iter().enumerate()
                                    .fold(0usize, |b, (i, v)| if *v > next_logits[b] { i } else { b }),
                                next_logits.iter().cloned().fold(f32::MIN, f32::max)
                            ),
                            Err(e) => eprintln!("[logits-dump-dec] write failed: {e}"),
                        }
                    }
                }
                // W3 probe: time everything AFTER forward returns (logits copy + sampler +
                // token probe). If this carries the bulk of the wall time, the [dp] forward
                // figure is encode-time only and the real wait is a late GPU sync.
                let __rd = if dec_profile {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                logits_buf.copy_from_slice(next_logits);
                let __cp = __rd.map(|t| t.elapsed().as_secs_f64() * 1000.0);
                cur_pos += 1;
                cur_token = sampler.sample(&mut logits_buf, &history);
                // §8 P3: per-token timing + top logprob when tracing is on.
                // logsumexp over the full vocab costs ~1 vocab pass — only
                // paid when a trace surface actually asked for it.
                if options.trace_tokens {
                    let forward_ns = t_serial.elapsed().as_nanos() as u64;
                    let mut max_v = f32::NEG_INFINITY;
                    for &v in logits_buf.iter() {
                        if v > max_v {
                            max_v = v;
                        }
                    }
                    let mut sum = 0.0f32;
                    for &v in logits_buf.iter() {
                        sum += (v - max_v).exp();
                    }
                    let lse = max_v + sum.ln();
                    let lp = logits_buf[cur_token as usize] - lse;
                    self.last_token_trace.push(TokenTiming {
                        step,
                        pos: cur_pos - 1,
                        token: cur_token,
                        forward_ns,
                        logprob: lp,
                    });
                }
                let __sm = __rd.map(|t| t.elapsed().as_secs_f64() * 1000.0);
                decode_step_probe(&logits_buf, "gen_serial", cur_pos, cur_token);
                if let (Some(_), Some(cp), Some(sm)) = (__rd, __cp, __sm) {
                    eprintln!(
                        "[dp] tail copy={:.2}ms sampler={:.2}ms probe={:.2}ms",
                        cp,
                        sm - cp,
                        __rd.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0) - sm
                    );
                }
                serial_ns += t_serial.elapsed().as_nanos() as u64;
                serial_tokens += 1;
            }
        }

        // Speculation receipt: steps vs accepted candidates is the number the
        // adaptive gate acts on, so a run that felt slower can be diagnosed
        // without a rebuild (tokens/step below the break-even means the tree
        // paid for drafts that missed).
        if quant::prof::enabled() && spec_steps > 0 {
            let tps = (spec_steps + spec_accepted) as f32 / spec_steps as f32;
            eprintln!(
                "[spec] steps={spec_steps} accepted={spec_accepted} tokens/step={tps:.3} gate_off={} min_accept={spec_min_accept:.2}",
                if spec_disabled { "yes" } else { "no" }
            );
        }

        // Cache the history for multi-turn continuations. Only tokens that
        // actually reached the KV may be reused: the loop pushes the last
        // sampled token before the max-tokens break decides whether to
        // forward it (so history can be one ahead of cur_pos), and a
        // compaction shifts positions so the token order no longer maps onto
        // KV rows at all.
        let reusable = reusable_history_len(compacted, cur_pos, history.len());
        self.cached_tokens = history[..reusable].to_vec();

        if self::quant::pool_failures() > pool_failures_before {
            return Err(format!(
                "a worker pool task failed during generation ({} total); the output would be incomplete",
                self::quant::pool_failures()
            ));
        }

        Ok(full_generated)
    }
}

/// Adaptive speculation gate: a speculative step delivers `1 + accepted/steps`
/// tokens for roughly one serial step plus a bit of extra row work, so a
/// draft chain that lands fewer than `min_tokens_per_step` tokens is a net
/// loss. Only evaluated after a 16-step warmup so early misses (cold n-gram
/// cache) cannot latch the tree off prematurely, while short generations
/// still stop paying for a losing draft within ~25% of their length.
/// Self-calibrating gate: turn speculation off when it is clearly slower than
/// plain serial decode. Both rates are measured in the same run (serial during
/// the warmup window), so the break-even follows the model, the prompt and the
/// kernel cost instead of a constant.
///
/// The margin is deliberately lenient (3%): the two losses are asymmetric. A
/// marginal tree costs a couple of percent, while switching off a winning one
/// costs ~12% (Qwen3-0.6B measured 27.0 t/s with the tree vs 23.7 serial), and
/// step-cost estimates carry a few percent of run-to-run noise because they are
/// sampled from a handful of steps.
/// Decode-step alignment probe (SOFUU_DECODE_PROBE=<path>): appends one line
/// per sampled step — tag, position, sampled token, top-8 (value,id) — from
/// the generator's sampling points, so a CPU run and a GPU-route run produce
/// step-aligned sequences whose first mismatching token pins the fork step.
/// Env-gated; zero cost unset.
fn decode_step_probe(logits: &[f32], tag: &str, pos: usize, sampled: u32) {
    if let Some(p) = std::env::var_os("SOFUU_DECODE_PROBE") {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let top8: Vec<String> = idx.iter().take(8)
            .map(|i| format!("({:.4e},{})", logits[*i], i))
            .collect();
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
            let _ = writeln!(f, "{tag} pos={pos} tok={sampled} {}", top8.join(" "));
        }
    }
}

fn spec_is_losing(spec_ns: u64, spec_tokens: u32, serial_ns: u64, serial_tokens: u32) -> bool {
    if spec_ns == 0 || serial_ns == 0 || spec_tokens < 8 || serial_tokens < 4 {
        return false;
    }
    let spec_rate = spec_tokens as f64 / spec_ns as f64;
    let serial_rate = serial_tokens as f64 / serial_ns as f64;
    spec_rate <= serial_rate * 0.97
}

/// Speculation is skipped for the first `warmup` steps so the self-calibrating
/// gate has a measured serial token cost to compare against.
fn spec_allowed_at(step: usize, warmup: usize) -> bool {
    step >= warmup
}

fn spec_should_disable(steps: u32, accepted: u32, min_tokens_per_step: f32) -> bool {
    // Two-tier: by 8 steps a chain near 1.0 tokens/step is a sure loss (cold
    // cache warmup is over by then); the full break-even threshold needs the
    // steadier 16-step estimate.
    if steps < 8 {
        return false;
    }
    let tokens_per_step = (steps + accepted) as f32 / steps as f32;
    tokens_per_step < 1.05 || (steps >= 16 && tokens_per_step < min_tokens_per_step)
}

#[cfg(test)]
mod adaptive_spec_tests {
    use super::{spec_allowed_at, spec_is_losing, spec_should_disable};

    #[test]
    fn warmup_boundary_is_exact() {
        // Steps before the warmup run serial (the gate needs that sample);
        // the step at the boundary and everything after may speculate.
        assert!(!spec_allowed_at(0, 4));
        assert!(!spec_allowed_at(3, 4));
        assert!(spec_allowed_at(4, 4));
        assert!(spec_allowed_at(5, 4));
        assert!(spec_allowed_at(0, 0));
    }

    #[test]
    fn calibrated_gate_keeps_a_fast_tree() {
        // 2.1 tokens per step at 87ms = 41.4ms/token vs 42ms serial: 1.4%
        // faster, inside the noise margin, so the tree stays.
        assert!(!spec_is_losing(10 * 87_000_000, 21, 10 * 42_000_000, 10));
        // 2.0 tokens per step at 87ms = 43.5ms/token: 3.6% slower, past the
        // margin, so it goes off.
        assert!(spec_is_losing(10 * 87_000_000, 20, 10 * 42_000_000, 10));
        // Same cost but 3.5 tokens/step: a clear win, keep it.
        assert!(!spec_is_losing(10 * 87_000_000, 35, 10 * 42_000_000, 10));
        // Cheap steps (e.g. q6_K-shaped trees at 60ms) pay at 2 tokens/step.
        assert!(!spec_is_losing(10 * 60_000_000, 20, 10 * 42_000_000, 10));
        // Clear loss: 1.14 tokens/step at 87ms is 76ms/token vs 42ms serial.
        assert!(spec_is_losing(10 * 87_000_000, 11, 10 * 42_000_000, 10));
    }

    #[test]
    fn calibrated_gate_needs_enough_samples() {
        assert!(!spec_is_losing(1_000_000_000, 7, 10_000_000, 10));
        assert!(!spec_is_losing(1_000_000_000, 20, 10_000_000, 3));
        assert!(!spec_is_losing(0, 20, 10, 10));
    }

    #[test]
    fn warmup_never_disables() {
        assert!(!spec_should_disable(7, 0, 1.15));
        assert!(!spec_should_disable(0, 0, 1.15));
    }

    #[test]
    fn below_break_even_disables() {
        // Sure-loss tier: 8 steps with no hits = 1.0 tokens/step.
        assert!(spec_should_disable(8, 0, 1.15));
        // 1.125 tokens/step at 8 steps is not sure-loss, and the full tier
        // needs 16 steps.
        assert!(!spec_should_disable(8, 1, 1.15));
        // 1.125 at 16 steps (< 1.15 threshold): full tier fires.
        assert!(spec_should_disable(16, 2, 1.15));
    }

    #[test]
    fn healthy_acceptance_keeps_speculating() {
        // Llama-1B counting-prompt-like: ~1.9 tokens/step.
        assert!(!spec_should_disable(16, 14, 1.15));
        // Qwen natural-text-like: ~1.2 tokens/step.
        assert!(!spec_should_disable(64, 13, 1.15));
    }
}


#[cfg(test)]
mod prefix_cache_tests {
    #[test]
    fn cache_never_outlives_the_kv_rows() {
        assert_eq!(super::reusable_history_len(false, 10, 11), 10);
        assert_eq!(super::reusable_history_len(false, 10, 10), 10);
        assert_eq!(super::reusable_history_len(true, 10, 20), 0);
        assert_eq!(super::reusable_history_len(false, 3, 1), 1);
    }
}


#[cfg(test)]
mod score_ppl_tests {
    use super::{log_softmax_nll, ppl_from_nll, score_windows, DEFAULT_SCORE_WINDOW, MIN_SCORE_SAMPLE};

    /// Convention pin: ppl is exp(mean NLL) in NATS. Uniform logits over a
    /// 4-way vocabulary put exactly 1/4 on every token, so one token's NLL is
    /// ln 4 and the perplexity of that single token is exactly 4 — if anyone
    /// switches the base (log2/log10), drops the normaliser or forgets the
    /// exp, this fails with an exact number instead of a tolerance.
    #[test]
    fn nll_is_natural_log_of_the_full_vocab_softmax() {
        let uniform = [0.0f32; 4];
        for t in 0..4 {
            assert!((log_softmax_nll(&uniform, t) - std::f64::consts::LN_2 * 2.0).abs() < 1e-6);
        }
        assert!((ppl_from_nll(log_softmax_nll(&uniform, 1), 1) - 4.0).abs() < 1e-5);
        // Two scored tokens, NLL ln 4 and ln 2: mean = (ln 4 + ln 2)/2 so the
        // ppl is exactly sqrt(8) — the averaging and the exp, both pinned.
        let two = [1.0f32, 0.0, 0.0, 0.0]; // log-softmax[1] = ln(e^-1/(e^-1+3))
        let nll_a = log_softmax_nll(&uniform, 0);
        let nll_b = log_softmax_nll(&two, 1);
        let expect = ((nll_a + nll_b) / 2.0).exp();
        assert!((ppl_from_nll(nll_a + nll_b, 2) - expect).abs() < 1e-9);
        assert!(expect > 4.0 && expect < 8.0);
        // Stability: a huge logit range must not overflow (f32 exp of 500 -> inf).
        let wide = [500.0f32, -500.0, 0.0, 0.0];
        let v = log_softmax_nll(&wide, 1);
        assert!(v.is_finite() && v > 0.0, "nll {v} must be finite and positive");
        // Zero scored tokens has no perplexity.
        assert!(ppl_from_nll(0.0, 0).is_nan());
    }

    /// The convention that produced the false "32x worse than llama.cpp":
    /// every scored token is scored exactly once, and the corpus is not
    /// silently reduced to a couple of samples.
    #[test]
    fn windows_score_every_token_exactly_once() {
        for &len in &[8usize, 24, 33, 64, 139, 913, 2579] {
            for &window in &[8usize, 64, 128, 512, 4096] {
                for &batch in &[1usize, 4, 32, 128] {
                    let plan = score_windows(len, window, batch);
                    let mut hits = vec![0u8; len];
                    for w in &plan {
                        assert!(w.prefill >= 1, "a window needs at least one context token");
                        assert!(w.first_scored > w.start && w.first_scored < w.end);
                        assert!(w.end <= len);
                        for i in w.first_scored..w.end {
                            hits[i] += 1;
                        }
                    }
                    // Token 0 never has a left context, and the head of the
                    // text is the first window's context-only prefix. From
                    // there on, coverage must be contiguous and single.
                    let head = plan.first().map(|w| w.first_scored).unwrap_or(0);
                    for (i, h) in hits.iter().enumerate() {
                        assert!(*h <= 1, "len={len} window={window} batch={batch}: token {i} scored {h} times");
                        if i >= head {
                            assert_eq!(*h, 1, "len={len} window={window} batch={batch}: token {i} never scored (head={head})");
                        } else {
                            assert_eq!(*h, 0, "len={len} window={window} batch={batch}: token {i} < head");
                        }
                    }
                    // No window may spend more than half of itself on context.
                    for w in &plan {
                        let win_span = w.end - w.start;
                        assert!(
                            w.prefill * 2 <= win_span.max(2),
                            "len={len} window={window}: prefix {} of window span {win_span}",
                            w.prefill
                        );
                    }
                }
            }
        }
    }

    /// The regression itself: with the shipped default window, a text of a few
    /// hundred tokens used to yield a ppl from ONE token (or 11), because the
    /// prefill prefix was allowed to eat the whole window. Pin the sample size.
    #[test]
    fn short_text_sample_is_not_degenerate() {
        let n = |len: usize, window: usize, batch: usize| -> usize {
            score_windows(len, window, batch).iter().map(|w| w.end - w.first_scored).sum()
        };
        // 24-token text, default window: was 1 scored token, ppl "10.73".
        assert!(n(24, DEFAULT_SCORE_WINDOW, 128) >= 11);
        // 139-token text at window 512: was 11.
        assert!(n(139, 512, 128) >= 64);
        // A window no longer than its own prefill batch still scores half.
        assert!(n(40, 32, 128) >= 20);
        // The head of every window is context only, so the scored FRACTION is
        // bounded below by (window-batch)/window: 75% at the default, 50% at
        // window 64. Pin it so no future change can quietly score a tenth.
        let scored = n(2579, DEFAULT_SCORE_WINDOW, 128);
        assert!(scored * 100 / (2579 - 1) >= 70, "scored {scored} of 2578");
        assert!(scored >= MIN_SCORE_SAMPLE);
        // Coverage is now the head plus everything after it, so a longer
        // window must not score LESS of the corpus than a shorter one.
        assert!(n(2579, 2048, 128) >= n(2579, DEFAULT_SCORE_WINDOW, 128));
    }
}

