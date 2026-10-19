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
        }
    }
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
        Self::from_gguf_with_gpu(gguf, max_seq_len, vocab_size, true)
    }

    pub fn from_gguf_with_gpu(
        gguf: GgufFile,
        max_seq_len: usize,
        vocab_size: usize,
        gpu_allowed: bool,
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
        let m = TransformerModel::from_gguf(gguf, max_seq_len, vocab_size)?;
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

pub struct InferEngine {
    pub model: InferModel,
    pub tokenizer: BpeTokenizer,
    pub last_prefill_tokens: usize,
    pub last_prefill_duration: std::time::Duration,
    pub cached_tokens: Vec<u32>,
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

impl InferEngine {
    /// Load a model and its tokenizer from disk (auto-detects context length).
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        Self::load_with_seq_len(model_path, 0)
    }

    /// Load a model with custom context length (0 = auto-detect from metadata).
    /// Auto-detects model format: GGUF, SafeTensors directory, or direct .safetensors file.
    pub fn load_with_seq_len<P: AsRef<Path>>(model_path: P, max_seq_len: usize) -> Result<Self, String> {
        Self::load_inner(model_path, max_seq_len, true)
    }

    /// Load a model without creating or using GPU resources. Useful for explicit
    /// CPU-only requests, especially for large SafeTensors models.
    pub fn load_with_seq_len_cpu<P: AsRef<Path>>(model_path: P, max_seq_len: usize) -> Result<Self, String> {
        Self::load_inner(model_path, max_seq_len, false)
    }

    fn load_inner<P: AsRef<Path>>(model_path: P, max_seq_len: usize, enable_gpu: bool) -> Result<Self, String> {
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
        let model = InferModel::from_gguf_with_gpu(gguf, max_seq_len, vocab_size, enable_gpu)?;
        let t_model = t2.elapsed();

        if std::env::var("SOFUU_PROFILE_LOAD").is_ok() {
            eprintln!("  [profile] GgufFile::open: {:.2?}, BpeTokenizer: {:.2?}, InferModel: {:.2?}", t_open, t_tok, t_model);
        }

        Ok(Self {
            model,
            tokenizer,
            last_prefill_tokens: 0,
            last_prefill_duration: std::time::Duration::ZERO,
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
        if options.threads > 0 {
            self::quant::set_thread_count(options.threads);
        }
        if options.sliding_window > 0 {
            self.model.sliding_window = options.sliding_window;
        }
        if options.sink_tokens > 0 {
            self.model.sink_tokens = options.sink_tokens;
        }

        let max_seq_len = self.model.config.max_seq_len;
        let vocab_size = self.model.config.vocab_size;
        let mut logits_buf = vec![0.0f32; vocab_size];

        // 1. Tokenize prompt
        let input_tokens = self.tokenizer.encode(prompt);
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

        // If not a pure extension of cached context, reset cache
        if common_prefix == 0 || common_prefix < self.cached_tokens.len() {
            self.model.reset_cache();
            self.cached_tokens.clear();
            common_prefix = 0;
        }

        let sink = self.model.sink_tokens.min(max_seq_len / 4).max(1);
        let drop = ((max_seq_len - sink) / 4).max(1).min((max_seq_len - sink).saturating_sub(1));
        let chunk_size = self::model::max_prefill_batch().min((max_seq_len - sink).saturating_sub(drop).max(1));

        // 3. Prefill phase for new tokens
        let mut cur_pos = common_prefix;
        let t_prefill = std::time::Instant::now();

        if common_prefix == 0 && input_tokens.len() < max_seq_len {
            // Fast-path: unified prefill with last-token logits projection
            let initial_logits = self.model.forward_prefill_all_with_logits(&input_tokens, 0);
            if std::env::var_os("SOFUU_DEBUG_DETERMINISM").is_some() {
                eprintln!("[determ] prefill_logits[..8] = {:?}", &initial_logits[..8]);
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
                        cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
                    }
                    if cur_pos + chunk.len() >= max_seq_len {
                        for &tok in chunk {
                            while cur_pos >= max_seq_len && cur_pos > sink + drop {
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
                cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
            }
            let initial_logits = self.model.forward(last_tok, cur_pos);
            logits_buf.copy_from_slice(initial_logits);
            cur_pos += 1;
        } else {
            // common_prefix == input_tokens.len() (prompt already prefilled)
            self.last_prefill_tokens = 0;
        }
        self.last_prefill_duration = t_prefill.elapsed();

        // SnapKV Observation Window Compaction (Prunes inactive middle blocks after prefill)
        if options.snapkv && cur_pos > 64 {
            let snap_sink = self.model.sink_tokens.max(16).min(cur_pos / 4);
            let keep_recent = 32.min(cur_pos - snap_sink);
            let middle = cur_pos.saturating_sub(snap_sink + keep_recent);
            if middle >= 16 {
                let drop = (middle * 3) / 4;
                cur_pos = self.model.compact_kv_cache(cur_pos, snap_sink, drop);
            }
        }

        if options.dynamic_ffn {
            self.model.set_dynamic_ffn(true);
        }

        if options.kivi {
            self.model.enable_kivi(options.kivi_bits, cur_pos);
        }

        if options.streaming_weights {
            self.model.set_streaming_weights(true);
        }

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
        let mut ngram_cache = if options.lookahead && matches!(self.model.backend, ModelBackend::Gguf(_)) {
            let mut cache = lookahead::NgramCache::new(2048);
            cache.ingest_sequence(&input_tokens);
            Some(cache)
        } else {
            None
        };

        let mut spec_logits_buf = vec![0.0f32; vocab_size];

        let mut step = 0;
        let mut cur_token = sampler.sample(&mut logits_buf, &history);

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

            // KV Cache auto-compaction before advancing
            while cur_pos >= max_seq_len && cur_pos > sink + drop {
                cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
            }

            // Propose multi-path speculative candidate tree from n-gram cache
            let speculative_tree = if let Some(ref cache) = ngram_cache {
                if cur_pos + 6 < max_seq_len && step < options.max_tokens {
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
                            .unwrap_or(5)
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
                    let tree =
                        lookahead::CandidateTree::build_cfg(cache, &history, rows, depth, widths);
                    if !tree.is_empty() {
                        Some(tree)
                    } else {
                        None
                    }
                } else {
                    None
                }
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
                let prof_spec = std::env::var_os("SOFUU_GPU_PROFILE").is_some();
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

                // Greedily traverse the tree along matching branches with lazy LM head projection
                let (accepted_tokens, accepted_batch_indices, next_sampled) = tree.verify_path(
                    root_predicted,
                    |batch_idx| {
                        // The GPU chain pass computed every row's logits
                        // in-commit; only the CPU-routed paths need the
                        // per-row matvec (~290ms each under load).
                        if !self
                            .model
                            .spec_chain_head_logits(batch_idx, &mut spec_logits_buf)
                        {
                            self.model
                                .forward_speculative_head_item(batch_idx, &mut spec_logits_buf);
                        }
                        sampler.sample(&mut spec_logits_buf[0..vocab_size], &history)
                    },
                );

                // Commit accepted candidate KV vectors to persistent KV cache
                if !accepted_batch_indices.is_empty() {
                    self.model.commit_speculative_candidates(&accepted_batch_indices, cur_pos);
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
                let next_logits = self.model.forward(next_token, cur_pos);
                if std::env::var_os("SOFUU_DEBUG_DETERMINISM").is_some() && step <= 2 {
                    eprintln!(
                        "[determ] step {} tok {} pos {} logits[0..8] = {:?}",
                        step,
                        next_token,
                        cur_pos,
                        &next_logits[..8.min(next_logits.len())]
                    );
                }
                logits_buf.copy_from_slice(next_logits);
                cur_pos += 1;
                cur_token = sampler.sample(&mut logits_buf, &history);
            }
        }

        // Cache the history for multi-turn continuations
        self.cached_tokens = history;

        Ok(full_generated)
    }
}
