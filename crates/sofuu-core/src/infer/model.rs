// crates/sofuu-core/src/infer/model.rs — Transformer graph and forward pass.
//
// Supports LLaMA 3, Qwen 2.5, and standard decoder-only architectures with:
// - Grouped-Query Attention (GQA)
// - Rotary Position Embeddings (RoPE)
// - SwiGLU activation
// - RMSNorm
// - Pre-allocated zero-garbage-collection working buffers
// - High-performance C++ SIMD microkernels for inner vector loops

use std::ops::{Deref, DerefMut};

use super::gguf::GgufFile;
use super::quant::{
    batched_attention_ext, dequantize_q4_0_row, dequantize_q4_k_row,
    dequantize_q6_k_row, dequantize_q8_0_row, f16_to_f32, kv_compact_style, matmat, matvec,
    rmsnorm, rmsnorm_in_place, rope_style, swiglu_in_place, type_row_size,
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
};

pub const MAX_PREFILL_BATCH: usize = 64;

/// An anonymous virtual memory buffer allocated directly via OS mmap.
///
/// On macOS and Linux, the OS kernel maps anonymous virtual memory to the
/// shared zero page (copy-on-write). Allocation takes ~10 microseconds
/// regardless of buffer size, and physical memory pages are only committed
/// when actually written to during inference.
pub struct AnonBuffer {
    ptr: *mut f32,
    len: usize,
}

unsafe impl Send for AnonBuffer {}
unsafe impl Sync for AnonBuffer {}

impl AnonBuffer {
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
        }
        let byte_len = len.checked_mul(std::mem::size_of::<f32>()).expect("buffer size overflow");
        let ptr = unsafe {
            #[cfg(unix)]
            {
                let p = libc::mmap(
                    std::ptr::null_mut(),
                    byte_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANON | libc::MAP_PRIVATE,
                    -1,
                    0,
                );
                if p == libc::MAP_FAILED {
                    let mut v = vec![0.0f32; len];
                    let p_fall = v.as_mut_ptr();
                    std::mem::forget(v);
                    p_fall
                } else {
                    p as *mut f32
                }
            }
            #[cfg(not(unix))]
            {
                let mut v = vec![0.0f32; len];
                let p = v.as_mut_ptr();
                std::mem::forget(v);
                p
            }
        };
        Self { ptr, len }
    }
}

impl Deref for AnonBuffer {
    type Target = [f32];
    #[inline(always)]
    fn deref(&self) -> &[f32] {
        if self.len == 0 || self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl DerefMut for AnonBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [f32] {
        if self.len == 0 || self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

impl Drop for AnonBuffer {
    fn drop(&mut self) {
        if self.len > 0 && !self.ptr.is_null() {
            let byte_len = self.len * std::mem::size_of::<f32>();
            unsafe {
                #[cfg(unix)]
                {
                    libc::munmap(self.ptr as *mut libc::c_void, byte_len);
                }
                #[cfg(not(unix))]
                {
                    let _ = Vec::from_raw_parts(self.ptr, self.len, self.len);
                }
            }
        }
    }
}

#[allow(dead_code)]
#[inline(always)]
fn inline_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        sum0 += ca[0] * cb[0];
        sum1 += ca[1] * cb[1];
        sum2 += ca[2] * cb[2];
        sum3 += ca[3] * cb[3];
    }
    let mut total = (sum0 + sum1) + (sum2 + sum3);
    for (va, vb) in rem_a.iter().zip(rem_b.iter()) {
        total += va * vb;
    }
    total
}

#[inline(always)]
fn inline_mad(dst: &mut [f32], src: &[f32], scalar: f32) {
    let len = dst.len().min(src.len());
    let chunks_dst = dst[..len].chunks_exact_mut(4);
    let chunks_src = src[..len].chunks_exact(4);
    let rem_len = len % 4;
    for (cd, cs) in chunks_dst.zip(chunks_src) {
        cd[0] += cs[0] * scalar;
        cd[1] += cs[1] * scalar;
        cd[2] += cs[2] * scalar;
        cd[3] += cs[3] * scalar;
    }
    if rem_len > 0 {
        let tail_start = len - rem_len;
        for (d, s) in dst[tail_start..len].iter_mut().zip(src[tail_start..len].iter()) {
            *d += *s * scalar;
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub rope_freq_base: f32,
    pub max_seq_len: usize,
    pub sliding_window: usize,
    /// GGUF has two common RoPE pairings. Llama-family uses consecutive pairs;
    /// Qwen/Gemma-family uses split-half pairs.
    pub rope_interleaved: bool,
}

#[derive(Clone)]
pub struct TensorView {
    pub offset: usize,
    pub size_bytes: usize,
    pub tensor_type: u32,
    pub shape: Vec<usize>,
}

#[derive(Clone)]
pub struct LayerWeights {
    pub attn_norm: TensorView,
    pub attn_q: TensorView,
    pub attn_k: TensorView,
    pub attn_v: TensorView,
    pub attn_output: TensorView,
    pub attn_q_bias: Option<TensorView>,
    pub attn_k_bias: Option<TensorView>,
    pub attn_v_bias: Option<TensorView>,
    pub attn_q_norm: Option<TensorView>,
    pub attn_k_norm: Option<TensorView>,
    pub ffn_norm: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
}

pub struct TransformerModel {
    pub config: ModelConfig,
    pub gguf: GgufFile,
    pub token_embd: TensorView,
    pub output_norm: TensorView,
    pub output: TensorView,
    pub layers: Vec<LayerWeights>,

    // Working state buffers (reused across all forward passes)
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub head_out: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate_buf: Vec<f32>,
    pub ffn_up_buf: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub logits: Vec<f32>,

    // Batched working state buffers for prompt prefill
    pub x_batch: Vec<f32>,
    pub x_norm_batch: Vec<f32>,
    pub q_batch: Vec<f32>,
    pub k_batch: Vec<f32>,
    pub v_batch: Vec<f32>,
    pub head_out_batch: Vec<f32>,
    pub attn_out_batch: Vec<f32>,
    pub ffn_gate_batch: Vec<f32>,
    pub ffn_up_batch: Vec<f32>,
    pub ffn_out_batch: Vec<f32>,

    // KV Cache: [layer, kv_head, max_seq_len, head_dim]
    pub k_cache: AnonBuffer,
    pub v_cache: AnonBuffer,
    pub kv_dim: usize,

    // Sliding Window Attention & Attention Sinks
    pub sliding_window: usize,
    pub sink_tokens: usize,
    pub is_gemma2: bool,
}

impl TransformerModel {
    pub fn from_gguf(gguf: GgufFile, max_seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        // 1. Detect architecture prefix
        let arch = gguf.get_str("general.architecture").unwrap_or("llama");
        let prefix = format!("{arch}.");
        let is_gemma2 = arch == "gemma2" || arch.contains("gemma2");

        // 2. Read model hyperparameters
        let dim = gguf
            .get_u32(&format!("{prefix}embedding_length"))
            .ok_or_else(|| "missing embedding_length".to_string())? as usize;
        let n_layers = gguf
            .get_u32(&format!("{prefix}block_count"))
            .ok_or_else(|| "missing block_count".to_string())? as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}attention.head_count"))
            .ok_or_else(|| "missing head_count".to_string())? as usize;
        let n_kv_heads = gguf
            .get_u32(&format!("{prefix}attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or(n_heads);
        let head_dim = gguf
            .get_u32(&format!("{prefix}attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let hidden_dim = gguf
            .get_u32(&format!("{prefix}feed_forward_length"))
            .ok_or_else(|| {
                if gguf.get_u32(&format!("{prefix}expert_count")).is_some() {
                    format!("unsupported GGUF architecture '{arch}': MoE models require a dedicated expert-routing graph")
                } else {
                    format!("missing feed_forward_length for GGUF architecture '{arch}'")
                }
            })? as usize;
        let norm_eps = gguf
            .get_f32(&format!("{prefix}attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_freq_base = gguf
            .get_f32(&format!("{prefix}rope.freq_base"))
            .unwrap_or(10000.0);
        let rope_interleaved = matches!(
            arch,
            "llama" | "llada" | "llama4" | "deci" | "baichuan" | "starcoder"
                | "internlm2" | "minicpm" | "xverse" | "command_r" | "cohere2"
                | "olmo" | "arctic" | "deepseek" | "deepseek2" | "granite"
                | "chameleon" | "smollm3" | "ernie4_5" | "mistral3"
        );

        // Vocab size from tokenizer or metadata fallback
        let vocab_size = if vocab_size > 0 {
            vocab_size
        } else {
            gguf.get_u32(&format!("{prefix}vocab_size"))
                .map(|v| v as usize)
                .ok_or_else(|| "missing vocab_size".to_string())?
        };

        let detected_ctx = gguf
            .get_u32(&format!("{prefix}context_length"))
            .or_else(|| gguf.get_u32("context_length"))
            .map(|v| v as usize)
            .unwrap_or(2048);

        let detected_sliding_window = gguf
            .get_u32(&format!("{prefix}attention.sliding_window"))
            .or_else(|| gguf.get_u32("attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(0);

        let final_seq_len = if max_seq_len > 0 {
            max_seq_len.min(detected_ctx)
        } else {
            detected_ctx.min(4096)
        };

        let config = ModelConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            vocab_size,
            norm_eps,
            rope_freq_base,
            max_seq_len: final_seq_len,
            sliding_window: detected_sliding_window,
            rope_interleaved,
        };

        // Helper to find tensor
        let find_tensor = |name: &str| -> Result<TensorView, String> {
            let info = gguf
                .tensors
                .get(name)
                .ok_or_else(|| format!("tensor not found: {name}"))?;
            Ok(TensorView {
                offset: info.offset,
                size_bytes: info.size_bytes,
                tensor_type: info.tensor_type,
                shape: info.shape.clone(),
            })
        };

        let find_opt_tensor = |name: &str| -> Option<TensorView> {
            let info = gguf.tensors.get(name)?;
            Some(TensorView {
                offset: info.offset,
                size_bytes: info.size_bytes,
                tensor_type: info.tensor_type,
                shape: info.shape.clone(),
            })
        };

        let token_embd = find_tensor("token_embd.weight")?;
        let output_norm = find_tensor("output_norm.weight")?;
        let output = find_tensor("output.weight").unwrap_or_else(|_| token_embd.clone());

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let attn_norm = find_tensor(&format!("blk.{i}.attn_norm.weight"))?;
            let attn_q = find_tensor(&format!("blk.{i}.attn_q.weight"))?;
            let attn_k = find_tensor(&format!("blk.{i}.attn_k.weight"))?;
            let attn_v = find_tensor(&format!("blk.{i}.attn_v.weight"))?;
            let attn_output = find_tensor(&format!("blk.{i}.attn_output.weight"))?;

            let attn_q_bias = find_opt_tensor(&format!("blk.{i}.attn_q.bias"));
            let attn_k_bias = find_opt_tensor(&format!("blk.{i}.attn_k.bias"));
            let attn_v_bias = find_opt_tensor(&format!("blk.{i}.attn_v.bias"));
            let attn_q_norm = find_opt_tensor(&format!("blk.{i}.attn_q_norm.weight"));
            let attn_k_norm = find_opt_tensor(&format!("blk.{i}.attn_k_norm.weight"));

            let ffn_norm = find_tensor(&format!("blk.{i}.ffn_norm.weight"))?;
            let ffn_gate = find_tensor(&format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_up = find_tensor(&format!("blk.{i}.ffn_up.weight"))?;
            let ffn_down = find_tensor(&format!("blk.{i}.ffn_down.weight"))?;

            layers.push(LayerWeights {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_q_norm,
                attn_k_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let kv_cache_size = n_layers * final_seq_len * kv_dim;

        Ok(Self {
            config,
            gguf,
            token_embd,
            output_norm,
            output,
            layers,

            x: vec![0.0f32; dim],
            x_norm: vec![0.0f32; dim],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_scores: vec![0.0f32; final_seq_len],
            head_out: vec![0.0f32; q_dim],
            attn_out: vec![0.0f32; dim],
            ffn_gate_buf: vec![0.0f32; hidden_dim],
            ffn_up_buf: vec![0.0f32; hidden_dim],
            ffn_out: vec![0.0f32; dim],
            logits: vec![0.0f32; vocab_size],

            x_batch: vec![0.0f32; MAX_PREFILL_BATCH * dim],
            x_norm_batch: vec![0.0f32; MAX_PREFILL_BATCH * dim],
            q_batch: vec![0.0f32; MAX_PREFILL_BATCH * q_dim],
            k_batch: vec![0.0f32; MAX_PREFILL_BATCH * kv_dim],
            v_batch: vec![0.0f32; MAX_PREFILL_BATCH * kv_dim],
            head_out_batch: vec![0.0f32; MAX_PREFILL_BATCH * q_dim],
            attn_out_batch: vec![0.0f32; MAX_PREFILL_BATCH * dim],
            ffn_gate_batch: vec![0.0f32; MAX_PREFILL_BATCH * hidden_dim],
            ffn_up_batch: vec![0.0f32; MAX_PREFILL_BATCH * hidden_dim],
            ffn_out_batch: vec![0.0f32; MAX_PREFILL_BATCH * dim],

            k_cache: AnonBuffer::new(kv_cache_size),
            v_cache: AnonBuffer::new(kv_cache_size),
            kv_dim,
            sliding_window: detected_sliding_window,
            sink_tokens: 4,
            is_gemma2,
        })
    }

    /// Returns the effective sliding window for a given layer.
    /// Gemma 2 alternates: even layers (0, 2, 4...) use local sliding window (4096),
    /// while odd layers (1, 3, 5...) use full global attention (sliding_window = 0).
    /// Mistral and other models apply uniform sliding window across all layers.
    #[inline]
    pub fn layer_sliding_window(&self, layer_idx: usize) -> usize {
        Self::calc_layer_sliding_window(self.is_gemma2, self.sliding_window, layer_idx)
    }

    #[inline]
    pub fn calc_layer_sliding_window(is_gemma2: bool, sliding_window: usize, layer_idx: usize) -> usize {
        if is_gemma2 {
            if layer_idx % 2 == 0 {
                if sliding_window > 0 { sliding_window } else { 4096 }
            } else {
                0
            }
        } else {
            sliding_window
        }
    }

    #[inline]
    fn tensor_slice<'a>(gguf: &'a GgufFile, tv: &TensorView) -> &'a [u8] {
        let start = gguf.data_offset + tv.offset;
        gguf.mmap
            .slice(start, tv.size_bytes)
            .expect("valid tensor slice")
    }

    /// Reset KV cache for a fresh conversation/prompt.
    pub fn reset_cache(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
    }

    /// In-place compact the KV cache when context window is full.
    /// Retains `sink_tokens` at [0 .. sink_tokens), drops `drop_tokens` from the middle,
    /// shifts surviving recent tokens left, and updates RoPE keys accordingly.
    /// Returns the new sequence length: `current_seq_len - drop_tokens`.
    pub fn compact_kv_cache(
        &mut self,
        current_seq_len: usize,
        sink_tokens: usize,
        drop_tokens: usize,
    ) -> usize {
        if drop_tokens == 0 || current_seq_len <= sink_tokens + drop_tokens {
            return current_seq_len;
        }
        kv_compact_style(
            &mut self.k_cache,
            &mut self.v_cache,
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
            self.config.max_seq_len,
            current_seq_len,
            sink_tokens,
            drop_tokens,
            self.config.rope_freq_base,
            if self.config.rope_interleaved { 1 } else { 0 },
        );
        current_seq_len - drop_tokens
    }

    /// Run single token step forward through the transformer.
    /// Returns reference to computed logits [vocab_size].
    pub fn forward(&mut self, token_id: u32, pos: usize) -> &[f32] {
        self.forward_impl(token_id, pos, true);
        &self.logits
    }

    /// Lookup embedding for a token and dequantize it into dst.
    fn lookup_embedding(
        gguf: &GgufFile,
        token_embd: &TensorView,
        token_id: u32,
        dim: usize,
        dst: &mut [f32],
    ) {
        let emb_bytes = Self::tensor_slice(gguf, token_embd);
        let row_size = type_row_size(token_embd.tensor_type, dim);
        let row_offset = (token_id as usize) * row_size;
        if row_offset + row_size > emb_bytes.len() {
            dst.fill(0.0);
            return;
        }
        let row_slice = &emb_bytes[row_offset..row_offset + row_size];

        match token_embd.tensor_type {
            GGML_TYPE_Q4_0 => {
                dequantize_q4_0_row(row_slice, dst);
            }
            GGML_TYPE_Q8_0 => {
                dequantize_q8_0_row(row_slice, dst);
            }
            GGML_TYPE_Q4_K => {
                dequantize_q4_k_row(row_slice, dst);
            }
            GGML_TYPE_Q6_K => {
                dequantize_q6_k_row(row_slice, dst);
            }
            GGML_TYPE_F32 => {
                let f32_ptr = row_slice.as_ptr() as *const f32;
                unsafe {
                    let slice = std::slice::from_raw_parts(f32_ptr, dim);
                    dst.copy_from_slice(slice);
                }
            }
            GGML_TYPE_F16 => {
                let u16_ptr = row_slice.as_ptr() as *const u16;
                unsafe {
                    let slice = std::slice::from_raw_parts(u16_ptr, dim);
                    for i in 0..dim {
                        dst[i] = f16_to_f32(slice[i]);
                    }
                }
            }
            _ => {
                dst.fill(0.0);
            }
        }
    }

    /// Prefill a batch of tokens: updates KV cache for the tokens without computing logits.
    /// Uses batched GEMM to reuse model weights across all tokens in the batch, eliminating
    /// redundant memory bandwidth stalls. Slices large token streams into MAX_PREFILL_BATCH tiles.
    pub fn forward_prefill_batch(&mut self, tokens: &[u32], mut start_pos: usize) {
        if tokens.is_empty() {
            return;
        }
        if tokens.len() == 1 {
            self.forward_prefill(tokens[0], start_pos);
            return;
        }
        for chunk in tokens.chunks(MAX_PREFILL_BATCH) {
            self.forward_prefill_batch_chunk(chunk, start_pos);
            start_pos += chunk.len();
        }
    }

    /// Prefill all prompt tokens and compute logits for the very last token in a single pass,
    /// avoiding an extra single-token evaluation over model weights.
    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        if tokens.is_empty() {
            return &self.logits;
        }
        if tokens.len() == 1 {
            return self.forward(tokens[0], start_pos);
        }

        let n_chunks = (tokens.len() + MAX_PREFILL_BATCH - 1) / MAX_PREFILL_BATCH;
        let mut cur_pos = start_pos;

        for (chunk_idx, chunk) in tokens.chunks(MAX_PREFILL_BATCH).enumerate() {
            let is_last_chunk = chunk_idx + 1 == n_chunks;
            if is_last_chunk {
                if chunk.len() == 1 {
                    return self.forward(chunk[0], cur_pos);
                }
                return self.forward_prefill_batch_chunk_internal(chunk, cur_pos, true);
            } else {
                self.forward_prefill_batch_chunk(chunk, cur_pos);
                cur_pos += chunk.len();
            }
        }
        &self.logits
    }

    fn forward_prefill_batch_chunk(&mut self, tokens: &[u32], start_pos: usize) {
        self.forward_prefill_batch_chunk_internal(tokens, start_pos, false);
    }

    fn forward_prefill_batch_chunk_internal(&mut self, tokens: &[u32], start_pos: usize, compute_last_logits: bool) -> &[f32] {
        let b_size = tokens.len();
        if b_size == 0 {
            return &self.logits;
        }
        if b_size == 1 {
            if compute_last_logits {
                return self.forward(tokens[0], start_pos);
            } else {
                self.forward_prefill(tokens[0], start_pos);
                return &self.logits;
            }
        }

        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let rope_freq_base = self.config.rope_freq_base;
        let max_seq_len = self.config.max_seq_len;
        let kv_dim = self.kv_dim;
        let q_dim = n_heads * head_dim;

        let gguf = &self.gguf;

        // 1. Batched Embedding lookup
        for (b, &tok) in tokens.iter().enumerate() {
            let dst = &mut self.x_batch[b * dim..(b + 1) * dim];
            Self::lookup_embedding(gguf, &self.token_embd, tok, dim, dst);
        }

        // 2. Transformer layers
        for layer_idx in 0..self.config.n_layers {
            let layer = &self.layers[layer_idx];

            // ── Attention Norm for batch ────────────────────────
            let attn_norm_bytes = Self::tensor_slice(gguf, &layer.attn_norm);
            let attn_norm_weights = unsafe {
                std::slice::from_raw_parts(attn_norm_bytes.as_ptr() as *const f32, dim)
            };
            for b in 0..b_size {
                let x_in = &self.x_batch[b * dim..(b + 1) * dim];
                let x_norm_out = &mut self.x_norm_batch[b * dim..(b + 1) * dim];
                rmsnorm(x_norm_out, x_in, attn_norm_weights, norm_eps);
            }

            // ── Batched Q, K, V Projections via matmat ──────────
            let q_bytes = Self::tensor_slice(gguf, &layer.attn_q);
            matmat(
                q_bytes,
                layer.attn_q.tensor_type,
                q_dim,
                dim,
                &self.x_norm_batch[0..b_size * dim],
                b_size,
                &mut self.q_batch[0..b_size * q_dim],
            );
            if let Some(ref bias) = layer.attn_q_bias {
                let b_bytes = Self::tensor_slice(gguf, bias);
                let b_f32 = unsafe {
                    std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, q_dim)
                };
                for b in 0..b_size {
                    let q_slice = &mut self.q_batch[b * q_dim..(b + 1) * q_dim];
                    for i in 0..q_dim {
                        q_slice[i] += b_f32[i];
                    }
                }
            }

            let k_bytes = Self::tensor_slice(gguf, &layer.attn_k);
            matmat(
                k_bytes,
                layer.attn_k.tensor_type,
                kv_dim,
                dim,
                &self.x_norm_batch[0..b_size * dim],
                b_size,
                &mut self.k_batch[0..b_size * kv_dim],
            );
            if let Some(ref bias) = layer.attn_k_bias {
                let b_bytes = Self::tensor_slice(gguf, bias);
                let b_f32 = unsafe {
                    std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, kv_dim)
                };
                for b in 0..b_size {
                    let k_slice = &mut self.k_batch[b * kv_dim..(b + 1) * kv_dim];
                    for i in 0..kv_dim {
                        k_slice[i] += b_f32[i];
                    }
                }
            }

            let v_bytes = Self::tensor_slice(gguf, &layer.attn_v);
            matmat(
                v_bytes,
                layer.attn_v.tensor_type,
                kv_dim,
                dim,
                &self.x_norm_batch[0..b_size * dim],
                b_size,
                &mut self.v_batch[0..b_size * kv_dim],
            );
            if let Some(ref bias) = layer.attn_v_bias {
                let b_bytes = Self::tensor_slice(gguf, bias);
                let b_f32 = unsafe {
                    std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, kv_dim)
                };
                for b in 0..b_size {
                    let v_slice = &mut self.v_batch[b * kv_dim..(b + 1) * kv_dim];
                    for i in 0..kv_dim {
                        v_slice[i] += b_f32[i];
                    }
                }
            }

            // ── Per-token QK-Norm ────────────────────────────────
            if let Some(ref q_norm) = layer.attn_q_norm {
                let q_norm_bytes = Self::tensor_slice(gguf, q_norm);
                let q_norm_w = unsafe {
                    std::slice::from_raw_parts(q_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                for b in 0..b_size {
                    let q_b = &mut self.q_batch[b * q_dim..(b + 1) * q_dim];
                    for h in 0..n_heads {
                        let head_slice = &mut q_b[h * head_dim..(h + 1) * head_dim];
                        rmsnorm_in_place(head_slice, q_norm_w, norm_eps);
                    }
                }
            }
            if let Some(ref k_norm) = layer.attn_k_norm {
                let k_norm_bytes = Self::tensor_slice(gguf, k_norm);
                let k_norm_w = unsafe {
                    std::slice::from_raw_parts(k_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                for b in 0..b_size {
                    let k_b = &mut self.k_batch[b * kv_dim..(b + 1) * kv_dim];
                    for kv_h in 0..n_kv_heads {
                        let head_slice = &mut k_b[kv_h * head_dim..(kv_h + 1) * head_dim];
                        rmsnorm_in_place(head_slice, k_norm_w, norm_eps);
                    }
                }
            }

            // ── RoPE & KV-Cache Storage for batch ────────────────
            for b in 0..b_size {
                let pos = start_pos + b;
                let q_slice = &mut self.q_batch[b * q_dim..(b + 1) * q_dim];
                let k_slice = &mut self.k_batch[b * kv_dim..(b + 1) * kv_dim];
                let rope_mode = if self.config.rope_interleaved { 1 } else { 0 };
                rope_style(q_slice, n_heads, head_dim, pos, rope_freq_base, rope_mode);
                rope_style(k_slice, n_kv_heads, head_dim, pos, rope_freq_base, rope_mode);

                let cache_pos = pos % max_seq_len;
                let v_slice = &self.v_batch[b * kv_dim..(b + 1) * kv_dim];
                for kv_h in 0..n_kv_heads {
                    let k_in = &k_slice[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let v_in = &v_slice[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim + cache_pos * head_dim;
                    self.k_cache[slot..slot + head_dim].copy_from_slice(k_in);
                    self.v_cache[slot..slot + head_dim].copy_from_slice(v_in);
                }
            }

            // ── Multi-Head / Grouped-Query Attention for batch ───
            let sliding_window = self.layer_sliding_window(layer_idx);
            batched_attention_ext(
                &self.q_batch[0..b_size * q_dim],
                &self.k_cache,
                &self.v_cache,
                &mut self.head_out_batch[0..b_size * q_dim],
                b_size,
                start_pos,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq_len,
                sliding_window,
                self.sink_tokens,
            );

            // ── Attention Output Projection via matmat ───────────
            let attn_out_bytes = Self::tensor_slice(gguf, &layer.attn_output);
            matmat(
                attn_out_bytes,
                layer.attn_output.tensor_type,
                dim,
                q_dim,
                &self.head_out_batch[0..b_size * q_dim],
                b_size,
                &mut self.attn_out_batch[0..b_size * dim],
            );

            // ── Residual Connection 1 ───────────────────────────
            for b in 0..b_size {
                let x_slice = &mut self.x_batch[b * dim..(b + 1) * dim];
                let out_slice = &self.attn_out_batch[b * dim..(b + 1) * dim];
                inline_mad(x_slice, out_slice, 1.0);
            }

            // ── FFN Norm for batch ──────────────────────────────
            let ffn_norm_bytes = Self::tensor_slice(gguf, &layer.ffn_norm);
            let ffn_norm_weights = unsafe {
                std::slice::from_raw_parts(ffn_norm_bytes.as_ptr() as *const f32, dim)
            };
            for b in 0..b_size {
                let x_in = &self.x_batch[b * dim..(b + 1) * dim];
                let x_norm_out = &mut self.x_norm_batch[b * dim..(b + 1) * dim];
                rmsnorm(x_norm_out, x_in, ffn_norm_weights, norm_eps);
            }

            // ── Batched SwiGLU FFN via matmat ───────────────────
            let gate_bytes = Self::tensor_slice(gguf, &layer.ffn_gate);
            matmat(
                gate_bytes,
                layer.ffn_gate.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm_batch[0..b_size * dim],
                b_size,
                &mut self.ffn_gate_batch[0..b_size * hidden_dim],
            );

            let up_bytes = Self::tensor_slice(gguf, &layer.ffn_up);
            matmat(
                up_bytes,
                layer.ffn_up.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm_batch[0..b_size * dim],
                b_size,
                &mut self.ffn_up_batch[0..b_size * hidden_dim],
            );

            for b in 0..b_size {
                let g = &mut self.ffn_gate_batch[b * hidden_dim..(b + 1) * hidden_dim];
                let u = &self.ffn_up_batch[b * hidden_dim..(b + 1) * hidden_dim];
                swiglu_in_place(g, u);
            }

            let down_bytes = Self::tensor_slice(gguf, &layer.ffn_down);
            matmat(
                down_bytes,
                layer.ffn_down.tensor_type,
                dim,
                hidden_dim,
                &self.ffn_gate_batch[0..b_size * hidden_dim],
                b_size,
                &mut self.ffn_out_batch[0..b_size * dim],
            );

            // ── Residual Connection 2 ───────────────────────────
            for b in 0..b_size {
                let x_slice = &mut self.x_batch[b * dim..(b + 1) * dim];
                let out_slice = &self.ffn_out_batch[b * dim..(b + 1) * dim];
                inline_mad(x_slice, out_slice, 1.0);
            }
        }

        if compute_last_logits {
            let last_b = b_size - 1;
            let x_last = &self.x_batch[last_b * dim..(last_b + 1) * dim];
            let norm_bytes = Self::tensor_slice(gguf, &self.output_norm);
            let norm_w = unsafe {
                std::slice::from_raw_parts(norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm(&mut self.x_norm, x_last, norm_w, norm_eps);

            let out_bytes = Self::tensor_slice(gguf, &self.output);
            matvec(
                out_bytes,
                self.output.tensor_type,
                self.config.vocab_size,
                dim,
                &self.x_norm,
                &mut self.logits,
            );
        }
        &self.logits
    }

    /// Prefill a token: updates KV cache for the token without computing logits.
    pub fn forward_prefill(&mut self, token_id: u32, pos: usize) {
        self.forward_impl(token_id, pos, false);
    }

    fn forward_impl(&mut self, token_id: u32, pos: usize, compute_logits: bool) {
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let rope_freq_base = self.config.rope_freq_base;
        let max_seq_len = self.config.max_seq_len;

        let gguf = &self.gguf;

        // 1. Embedding lookup
        Self::lookup_embedding(gguf, &self.token_embd, token_id, dim, &mut self.x);

        // 2. Transformer layers
        for layer_idx in 0..self.config.n_layers {
            let layer = &self.layers[layer_idx];

            // ── Attention Norm ──────────────────────────────
            let attn_norm_bytes = Self::tensor_slice(gguf, &layer.attn_norm);
            let attn_norm_weights = unsafe {
                std::slice::from_raw_parts(attn_norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm(&mut self.x_norm, &self.x, attn_norm_weights, norm_eps);

            // ── Q, K, V Projections ─────────────────────────
            let q_bytes = Self::tensor_slice(gguf, &layer.attn_q);
            matvec(
                q_bytes,
                layer.attn_q.tensor_type,
                n_heads * head_dim,
                dim,
                &self.x_norm,
                &mut self.q,
            );
            if let Some(ref bias) = layer.attn_q_bias {
                let b_bytes = Self::tensor_slice(gguf, bias);
                let b_f32 = unsafe {
                    std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, self.q.len())
                };
                for i in 0..self.q.len() {
                    self.q[i] += b_f32[i];
                }
            }

            let k_bytes = Self::tensor_slice(gguf, &layer.attn_k);
            matvec(
                k_bytes,
                layer.attn_k.tensor_type,
                n_kv_heads * head_dim,
                dim,
                &self.x_norm,
                &mut self.k,
            );
            if let Some(ref bias) = layer.attn_k_bias {
                let b_bytes = Self::tensor_slice(gguf, bias);
                let b_f32 = unsafe {
                    std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, self.k.len())
                };
                for i in 0..self.k.len() {
                    self.k[i] += b_f32[i];
                }
            }

            let v_bytes = Self::tensor_slice(gguf, &layer.attn_v);
            matvec(
                v_bytes,
                layer.attn_v.tensor_type,
                n_kv_heads * head_dim,
                dim,
                &self.x_norm,
                &mut self.v,
            );
            if let Some(ref bias) = layer.attn_v_bias {
                let b_bytes = Self::tensor_slice(gguf, bias);
                let b_f32 = unsafe {
                    std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, self.v.len())
                };
                for i in 0..self.v.len() {
                    self.v[i] += b_f32[i];
                }
            }

            // ── QK-Norm (per-head RMSNorm for Qwen 2.5 / 3) ──
            if let Some(ref q_norm) = layer.attn_q_norm {
                let q_norm_bytes = Self::tensor_slice(gguf, q_norm);
                let q_norm_w = unsafe {
                    std::slice::from_raw_parts(q_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                for h in 0..n_heads {
                    let head_slice = &mut self.q[h * head_dim..(h + 1) * head_dim];
                    rmsnorm_in_place(head_slice, q_norm_w, norm_eps);
                }
            }
            if let Some(ref k_norm) = layer.attn_k_norm {
                let k_norm_bytes = Self::tensor_slice(gguf, k_norm);
                let k_norm_w = unsafe {
                    std::slice::from_raw_parts(k_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                for kv_h in 0..n_kv_heads {
                    let head_slice = &mut self.k[kv_h * head_dim..(kv_h + 1) * head_dim];
                    rmsnorm_in_place(head_slice, k_norm_w, norm_eps);
                }
            }

            // ── RoPE (Rotary Position Embeddings) ───────────
            let rope_mode = if self.config.rope_interleaved { 1 } else { 0 };
            rope_style(&mut self.q, n_heads, head_dim, pos, rope_freq_base, rope_mode);
            rope_style(&mut self.k, n_kv_heads, head_dim, pos, rope_freq_base, rope_mode);

            // ── Store to KV-Cache at slot pos ───────────────
            let cache_pos = pos % max_seq_len;
            for kv_h in 0..n_kv_heads {
                let k_in = &self.k[kv_h * head_dim..(kv_h + 1) * head_dim];
                let v_in = &self.v[kv_h * head_dim..(kv_h + 1) * head_dim];
                let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim + cache_pos * head_dim;
                self.k_cache[slot..slot + head_dim].copy_from_slice(k_in);
                self.v_cache[slot..slot + head_dim].copy_from_slice(v_in);
            }

            // ── Multi-Head / Grouped-Query Attention ────────
            let sliding_window = self.layer_sliding_window(layer_idx);
            batched_attention_ext(
                &self.q,
                &self.k_cache,
                &self.v_cache,
                &mut self.head_out,
                1,
                pos,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq_len,
                sliding_window,
                self.sink_tokens,
            );

            // ── Attention Output Projection ─────────────────
            let attn_out_bytes = Self::tensor_slice(gguf, &layer.attn_output);
            matvec(
                attn_out_bytes,
                layer.attn_output.tensor_type,
                dim,
                n_heads * head_dim,
                &self.head_out,
                &mut self.attn_out,
            );

            // ── Residual Connection 1 ───────────────────────
            inline_mad(&mut self.x, &self.attn_out, 1.0);

            // ── FFN Norm ────────────────────────────────────
            let ffn_norm_bytes = Self::tensor_slice(gguf, &layer.ffn_norm);
            let ffn_norm_weights = unsafe {
                std::slice::from_raw_parts(ffn_norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm(&mut self.x_norm, &self.x, ffn_norm_weights, norm_eps);

            // ── SwiGLU FFN ──────────────────────────────────
            let gate_bytes = Self::tensor_slice(gguf, &layer.ffn_gate);
            matvec(
                gate_bytes,
                layer.ffn_gate.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm,
                &mut self.ffn_gate_buf,
            );

            let up_bytes = Self::tensor_slice(gguf, &layer.ffn_up);
            matvec(
                up_bytes,
                layer.ffn_up.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm,
                &mut self.ffn_up_buf,
            );

            swiglu_in_place(&mut self.ffn_gate_buf, &self.ffn_up_buf);

            let down_bytes = Self::tensor_slice(gguf, &layer.ffn_down);
            matvec(
                down_bytes,
                layer.ffn_down.tensor_type,
                dim,
                hidden_dim,
                &self.ffn_gate_buf,
                &mut self.ffn_out,
            );

            // ── Residual Connection 2 ───────────────────────
            inline_mad(&mut self.x, &self.ffn_out, 1.0);
        }

        if compute_logits {
            // 3. Final Output Norm
            let out_norm_bytes = Self::tensor_slice(gguf, &self.output_norm);
            let out_norm_weights = unsafe {
                std::slice::from_raw_parts(out_norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm_in_place(&mut self.x, out_norm_weights, norm_eps);

            // 4. Output Logits Projection
            let out_bytes = Self::tensor_slice(gguf, &self.output);
            matvec(
                out_bytes,
                self.output.tensor_type,
                self.config.vocab_size,
                dim,
                &self.x,
                &mut self.logits,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_sliding_window_gemma2_vs_uniform() {
        // Gemma 2 model: alternates between 4096 (even) and 0 (odd)
        assert_eq!(TransformerModel::calc_layer_sliding_window(true, 4096, 0), 4096);
        assert_eq!(TransformerModel::calc_layer_sliding_window(true, 4096, 1), 0);
        assert_eq!(TransformerModel::calc_layer_sliding_window(true, 4096, 2), 4096);
        assert_eq!(TransformerModel::calc_layer_sliding_window(true, 4096, 3), 0);

        // Mistral model: uniform across all layers
        assert_eq!(TransformerModel::calc_layer_sliding_window(false, 4096, 0), 4096);
        assert_eq!(TransformerModel::calc_layer_sliding_window(false, 4096, 1), 4096);
        assert_eq!(TransformerModel::calc_layer_sliding_window(false, 4096, 2), 4096);
        assert_eq!(TransformerModel::calc_layer_sliding_window(false, 4096, 3), 4096);
    }
}

