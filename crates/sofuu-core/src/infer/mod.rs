// crates/sofuu-core/src/infer/mod.rs — Fast CPU-First GGUF Inference Engine.
//
// Combines zero-copy mmap, safe GGUF parsing, BPE tokenizer, and C++ SIMD microkernels
// into a lightweight, zero-dependency LLM engine for edge, laptop, and serverless environments.

pub mod gguf;
pub mod lfm2;
pub mod metal;
pub mod mmap;
pub mod model;
pub mod quant;
pub mod safetensors;
pub mod sampler;
pub mod tokenizer;

use std::path::Path;

use self::gguf::GgufFile;
use self::lfm2::Lfm2Model;
use self::model::{ModelConfig, TransformerModel};
use self::sampler::{Sampler, SamplerConfig};
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
            ],
            sliding_window: 0, // 0 = full context / model default
            sink_tokens: 4,    // 4 sink tokens by default
        }
    }
}

pub enum ModelBackend {
    Gguf(TransformerModel),
    Lfm2(Lfm2Model),
}

pub struct InferModel {
    pub config: ModelConfig,
    pub backend: ModelBackend,
    pub sliding_window: usize,
    pub sink_tokens: usize,
}

impl InferModel {
    pub fn from_gguf(gguf: GgufFile, max_seq_len: usize, vocab_size: usize) -> Result<Self, String> {
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
        let m = Lfm2Model::from_safetensors_dir(dir, max_seq_len)?;
        let config = m.config.clone();
        Ok(Self {
            config,
            backend: ModelBackend::Lfm2(m),
            sliding_window: 0,
            sink_tokens: 4,
        })
    }

    pub fn reset_cache(&mut self) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.reset_cache(),
            ModelBackend::Lfm2(m) => m.reset_cache(),
        }
    }

    pub fn compact_kv_cache(&mut self, cur_pos: usize, sink: usize, drop: usize) -> usize {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.compact_kv_cache(cur_pos, sink, drop),
            ModelBackend::Lfm2(m) => m.compact_kv_cache(cur_pos, sink, drop),
        }
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward(token, pos),
            ModelBackend::Lfm2(m) => m.forward(token, pos),
        }
    }

    pub fn forward_prefill(&mut self, token: u32, pos: usize) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_prefill(token, pos),
            ModelBackend::Lfm2(m) => m.forward_prefill(token, pos),
        }
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], start_pos: usize) {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_prefill_batch(tokens, start_pos),
            ModelBackend::Lfm2(m) => m.forward_prefill_batch(tokens, start_pos),
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        match &mut self.backend {
            ModelBackend::Gguf(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
            ModelBackend::Lfm2(m) => m.forward_prefill_all_with_logits(tokens, start_pos),
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
        let raw_str = model_path.as_ref().to_str().unwrap_or("");
        let clean_path = clean_model_path(raw_str);

        // 1. Directory mode (HuggingFace / SafeTensors model directory)
        if clean_path.is_dir() {
            let config_path = clean_path.join("config.json");
            let safetensors_path = clean_path.join("model.safetensors");
            let tokenizer_path = clean_path.join("tokenizer.json");

            if config_path.exists() && safetensors_path.exists() {
                let tokenizer = BpeTokenizer::from_hf_file(&tokenizer_path)
                    .map_err(|e| format!("failed to load tokenizer.json from {}: {}", clean_path.display(), e))?;
                let model = InferModel::from_lfm2_dir(&clean_path, max_seq_len)?;
                return Ok(Self {
                    model,
                    tokenizer,
                    last_prefill_tokens: 0,
                    last_prefill_duration: std::time::Duration::ZERO,
                    cached_tokens: Vec::new(),
                });
            }
        }

        // 2. Direct .safetensors file mode
        if clean_path.extension().map_or(false, |ext| ext == "safetensors") {
            if let Some(parent) = clean_path.parent() {
                let config_path = parent.join("config.json");
                let tokenizer_path = parent.join("tokenizer.json");
                if config_path.exists() {
                    let tokenizer = BpeTokenizer::from_hf_file(&tokenizer_path)
                        .map_err(|e| format!("failed to load tokenizer.json: {}", e))?;
                    let model = InferModel::from_lfm2_dir(parent, max_seq_len)?;
                    return Ok(Self {
                        model,
                        tokenizer,
                        last_prefill_tokens: 0,
                        last_prefill_duration: std::time::Duration::ZERO,
                        cached_tokens: Vec::new(),
                    });
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
        let vocab_size = tokenizer.vocab.len();
        let model = InferModel::from_gguf(gguf, max_seq_len, vocab_size)?;
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

    /// Explicitly clear the KV cache and cached token history.
    pub fn reset_cache(&mut self) {
        self.model.reset_cache();
        self.cached_tokens.clear();
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
        let chunk_size = self::model::MAX_PREFILL_BATCH.min((max_seq_len - sink).saturating_sub(drop).max(1));

        // 3. Prefill phase for new tokens
        let mut cur_pos = common_prefix;
        let t_prefill = std::time::Instant::now();

        if common_prefix == 0 && input_tokens.len() < max_seq_len {
            // Fast-path: unified prefill with last-token logits projection
            let initial_logits = self.model.forward_prefill_all_with_logits(&input_tokens, 0);
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

        for _ in 0..options.max_tokens {
            let next_token = sampler.sample(&mut logits_buf, &history);

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

            // KV Cache auto-compaction before advancing
            while cur_pos >= max_seq_len && cur_pos > sink + drop {
                cur_pos = self.model.compact_kv_cache(cur_pos, sink, drop);
            }

            // Forward next token to compute next logits
            let next_logits = self.model.forward(next_token, cur_pos);
            logits_buf.copy_from_slice(next_logits);
            cur_pos += 1;
        }

        // Cache the history for multi-turn continuations
        self.cached_tokens = history;

        Ok(full_generated)
    }
}
