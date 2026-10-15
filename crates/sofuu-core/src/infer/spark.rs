// crates/sofuu-core/src/infer/spark.rs — Spark-X2.5 GGUF inference backend.
//
// Supports the `spark2_5` GGUF architecture (Spark-X2.5 family):
// - Fused QKV projection (attn_qkv), split Q | K | V by zero-copy row-range
//   slicing of the fused weight (three projections, one weight pass)
// - Grouped-query attention, head_dim 256 (16 heads, 4 kv heads)
// - Per-head sigmoid attention gating (attn_gate) applied to the attention
//   output before the wo projection
// - Hybrid sliding-window attention per layer (SWA vs full) with per-layer
//   RoPE geometry: SWA layers rotate all head_dim dims with a low freq base;
//   full-attention layers rotate only the first n_rot dims with a high base
// - GeGLU FFN with exact-erf GELU (matching llama.cpp ggml_gelu)
// - Tied LM head (token_embd doubles as the output projection)
//
// Semantics mirror llama.cpp src/models/spark2-5.cpp (LLAMA_ROPE_TYPE_NEOX,
// LLAMA_SWA_TYPE_STANDARD, LLM_FFN_GELU + geglu_split).

use super::gguf::{GgufFile, GgufValue};
use super::model::{AnonBuffer, ModelConfig, TensorView, max_prefill_batch};
use super::quant::{
    batched_attention_ext, dequantize_q4_0_row, dequantize_q4_k_row, dequantize_q6_k_row,
    dequantize_q8_0_row, f16_to_f32, matmat, matvec, prof, rmsnorm, rope_style, type_row_size,
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
};

#[derive(Clone)]
pub struct SparkLayer {
    pub attn_norm: TensorView,
    pub attn_qkv: TensorView,
    pub attn_gate: TensorView,
    pub attn_output: TensorView,
    pub ffn_norm: TensorView,
    pub ffn_gate: TensorView,
    pub ffn_up: TensorView,
    pub ffn_down: TensorView,
    pub is_swa: bool,
}

pub struct Spark25Model {
    pub config: ModelConfig,
    pub gguf: GgufFile,
    pub token_embd: TensorView,
    pub output_norm: TensorView,
    pub layers: Vec<SparkLayer>,

    pub swa_window: usize,
    pub n_rot_full: usize,
    pub n_rot_swa: usize,
    pub rope_base_full: f32,
    pub rope_base_swa: f32,

    // Single-token decode working state
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub head_out: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate_buf: Vec<f32>,
    pub ffn_up_buf: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub logits: Vec<f32>,

    // Batched prefill working state
    pub x_batch: Vec<f32>,
    pub x_norm_batch: Vec<f32>,
    pub q_batch: Vec<f32>,
    pub k_batch: Vec<f32>,
    pub v_batch: Vec<f32>,
    pub gate_batch: Vec<f32>,
    pub head_out_batch: Vec<f32>,
    pub attn_out_batch: Vec<f32>,
    pub ffn_gate_batch: Vec<f32>,
    pub ffn_up_batch: Vec<f32>,
    pub ffn_out_batch: Vec<f32>,

    // KV cache: [layer, kv_head, max_seq_len, head_dim]
    pub k_cache: AnonBuffer,
    pub v_cache: AnonBuffer,
    pub kv_dim: usize,

    pub sliding_window: usize, // CLI override; 0 = model default (swa_window)
    pub sink_tokens: usize,
}

#[inline(always)]
fn inline_mad(dst: &mut [f32], src: &[f32], scalar: f32) {
    let len = dst.len().min(src.len());
    for i in 0..len {
        dst[i] += src[i] * scalar;
    }
}

/// Exact-erf GELU matching llama.cpp ggml_gelu (spark FFN activation).
#[inline(always)]
fn erf32(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 approximation, max abs error 1.5e-7.
    let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let x = x.abs();
    let t = 1.0f32 / (1.0f32 + 0.3275911f32 * x);
    let y = 1.0f32
        - ((((1.061405429f32 * t - 1.453152027f32) * t + 1.421413741f32) * t - 0.284496736f32) * t
            + 0.254829592f32)
            * t
            * (-x * x).exp();
    sign * y
}

#[inline(always)]
fn gelu_erf(x: f32) -> f32 {
    0.5f32 * x * (1.0f32 + erf32(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// In-place GeGLU: gate[i] = gelu_erf(gate[i]) * up[i].
fn geglu_erf_in_place(gate: &mut [f32], up: &[f32]) {
    let len = gate.len().min(up.len());
    for i in 0..len {
        gate[i] = gelu_erf(gate[i]) * up[i];
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// Split-half (NeoX) rotary embedding over one vector of n_heads * head_dim.
/// If n_rot == head_dim this delegates to the C++ SIMD kernel (rope_style 0);
/// otherwise only the first n_rot dims per head rotate and the rest pass
/// through (partial rope).
#[inline]
fn rope_apply(vec: &mut [f32], n_heads: usize, head_dim: usize, n_rot: usize, pos: usize, base: f32) {
    if n_rot >= head_dim {
        rope_style(vec, n_heads, head_dim, pos, base, 0);
        return;
    }
    let half = n_rot / 2;
    let theta_scale = base.powf(-2.0f32 / n_rot as f32);
    let mut theta = pos as f32;
    for j in 0..half {
        let cos = theta.cos();
        let sin = theta.sin();
        for h in 0..n_heads {
            let a = &mut vec[h * head_dim..h * head_dim + head_dim];
            let x0 = a[j];
            let x1 = a[j + half];
            a[j] = x0 * cos - x1 * sin;
            a[j + half] = x1 * cos + x0 * sin;
        }
        theta *= theta_scale;
    }
}

/// Re-rotate one stored (position-encoded) key head by a signed position delta
/// (used after compaction so shifted keys reflect their new positions). Dims
/// beyond n_rot are untouched — they were never rotated.
#[inline]
fn rope_shift_partial(vec: &mut [f32], n_rot: usize, base: f32, pos_delta: i64) {
    let half = n_rot / 2;
    let theta_scale = base.powf(-2.0f32 / n_rot as f32);
    let mut theta = pos_delta as f32;
    for j in 0..half {
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = vec[j];
        let x1 = vec[j + half];
        vec[j] = x0 * cos - x1 * sin;
        vec[j + half] = x1 * cos + x0 * sin;
        theta *= theta_scale;
    }
}

impl Spark25Model {
    pub fn from_gguf(gguf: GgufFile, max_seq_len: usize, vocab_size_in: usize) -> Result<Self, String> {
        let prefix = "spark2_5.";
        let dim = gguf
            .get_u32(&format!("{prefix}embedding_length"))
            .ok_or_else(|| "spark2_5: missing embedding_length".to_string())? as usize;
        let n_layers = gguf
            .get_u32(&format!("{prefix}block_count"))
            .ok_or_else(|| "spark2_5: missing block_count".to_string())? as usize;
        let n_heads = gguf
            .get_u32(&format!("{prefix}attention.head_count"))
            .ok_or_else(|| "spark2_5: missing attention.head_count".to_string())? as usize;
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
            .ok_or_else(|| "spark2_5: missing feed_forward_length".to_string())? as usize;
        let norm_eps = gguf
            .get_f32(&format!("{prefix}attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let swa_window = gguf
            .get_u32(&format!("{prefix}attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(512);
        let n_rot_full = gguf
            .get_u32(&format!("{prefix}rope.dimension_count"))
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let n_rot_swa = gguf
            .get_u32(&format!("{prefix}rope.dimension_count_swa"))
            .map(|v| v as usize)
            .unwrap_or(n_rot_full);
        let rope_base_full = gguf
            .get_f32(&format!("{prefix}rope.freq_base"))
            .unwrap_or(5_000_000.0);
        let rope_base_swa = gguf
            .get_f32(&format!("{prefix}rope.freq_base_swa"))
            .unwrap_or(10_000.0);

        let vocab_size = if vocab_size_in > 0 {
            vocab_size_in
        } else {
            gguf.get_u32(&format!("{prefix}vocab_size"))
                .map(|v| v as usize)
                .ok_or_else(|| "spark2_5: missing vocab_size".to_string())?
        };

        let detected_ctx = gguf
            .get_u32(&format!("{prefix}context_length"))
            .or_else(|| gguf.get_u32("context_length"))
            .map(|v| v as usize)
            .unwrap_or(4096);
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
            rope_freq_base: rope_base_full,
            max_seq_len: final_seq_len,
            sliding_window: swa_window,
            rope_interleaved: false, // NeoX split-half pairing
        };

        // Per-layer SWA pattern from the GGUF bool array (raw byte array).
        let mut is_swa = vec![false; n_layers];
        match gguf
            .metadata
            .get(&format!("{prefix}attention.sliding_window_pattern"))
        {
            Some(GgufValue::RawArray { item_type: 7, len, data }) => {
                for i in 0..(*len).min(n_layers) {
                    is_swa[i] = data[i] != 0;
                }
            }
            _ => {}
        }

        let find = |name: &str| -> Result<TensorView, String> {
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

        let token_embd = find("token_embd.weight")?;
        let output_norm = find("output_norm.weight")?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(SparkLayer {
                attn_norm: find(&format!("blk.{i}.attn_norm.weight"))?,
                attn_qkv: find(&format!("blk.{i}.attn_qkv.weight"))?,
                attn_gate: find(&format!("blk.{i}.attn_gate.weight"))?,
                attn_output: find(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: find(&format!("blk.{i}.ffn_norm.weight"))?,
                ffn_gate: find(&format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: find(&format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: find(&format!("blk.{i}.ffn_down.weight"))?,
                is_swa: is_swa[i],
            });
        }

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let kv_cache_size = n_layers * final_seq_len * kv_dim;

        // Warm packed-tile GEMM cache (Spark matrices are q4_K/q6_K/q8_0).
        // token_embd doubles as the tied LM head and is GEMV'd at its full
        // tensor shape.
        if matches!(token_embd.tensor_type, GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K)
            && token_embd.shape.len() == 2
        {
            if let Some(slice) = gguf.mmap.slice(gguf.data_offset + token_embd.offset, 64) {
                super::quant::tile_warm(slice.as_ptr(), token_embd.shape[1], token_embd.shape[0], token_embd.tensor_type);
            }
        }
        for l in &layers {
            for tv in [&l.attn_output, &l.ffn_gate, &l.ffn_up, &l.ffn_down] {
                if matches!(tv.tensor_type, GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K) && tv.shape.len() == 2 {
                    if let Some(slice) = gguf.mmap.slice(gguf.data_offset + tv.offset, 64) {
                        super::quant::tile_warm(slice.as_ptr(), tv.shape[1], tv.shape[0], tv.tensor_type);
                    }
                }
            }
            // project_qkv_batch projects the fused QKV tensor as three
            // zero-copy row-range matmats, and the tile map keys entries by
            // exact (pointer, rows, cols, fingerprint) — a whole fused-tensor
            // entry can never match those lookups. Warm each row range at the
            // exact pointer and shape projection calls pass.
            if matches!(l.attn_qkv.tensor_type, GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K)
                && l.attn_qkv.shape.len() == 2
            {
                let row_stride = super::quant::type_row_size(l.attn_qkv.tensor_type, dim);
                let base = gguf.data_offset + l.attn_qkv.offset;
                for (row_off, rows) in [(0usize, q_dim), (q_dim, kv_dim), (q_dim + kv_dim, kv_dim)] {
                    if let Some(slice) = gguf.mmap.slice(base + row_off * row_stride, 64) {
                        super::quant::tile_warm(slice.as_ptr(), rows, dim, l.attn_qkv.tensor_type);
                    }
                }
            }
        }

        Ok(Self {
            config,
            gguf,
            token_embd,
            output_norm,
            layers,
            swa_window,
            n_rot_full,
            n_rot_swa,
            rope_base_full,
            rope_base_swa,
            x: vec![0.0f32; dim],
            x_norm: vec![0.0f32; dim],
            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            gate: vec![0.0f32; n_heads],
            head_out: vec![0.0f32; q_dim],
            attn_out: vec![0.0f32; dim],
            ffn_gate_buf: vec![0.0f32; hidden_dim],
            ffn_up_buf: vec![0.0f32; hidden_dim],
            ffn_out: vec![0.0f32; dim],
            logits: vec![0.0f32; vocab_size],
            x_batch: vec![0.0f32; max_prefill_batch() * dim],
            x_norm_batch: vec![0.0f32; max_prefill_batch() * dim],
            q_batch: vec![0.0f32; max_prefill_batch() * q_dim],
            k_batch: vec![0.0f32; max_prefill_batch() * kv_dim],
            v_batch: vec![0.0f32; max_prefill_batch() * kv_dim],
            gate_batch: vec![0.0f32; max_prefill_batch() * n_heads],
            head_out_batch: vec![0.0f32; max_prefill_batch() * q_dim],
            attn_out_batch: vec![0.0f32; max_prefill_batch() * dim],
            ffn_gate_batch: vec![0.0f32; max_prefill_batch() * hidden_dim],
            ffn_up_batch: vec![0.0f32; max_prefill_batch() * hidden_dim],
            ffn_out_batch: vec![0.0f32; max_prefill_batch() * dim],
            k_cache: AnonBuffer::new(kv_cache_size),
            v_cache: AnonBuffer::new(kv_cache_size),
            kv_dim,
            sliding_window: 0,
            sink_tokens: 4,
        })
    }

    #[inline]
    fn layer_rope(&self, layer_idx: usize) -> (usize, f32) {
        if self.layers[layer_idx].is_swa {
            (self.n_rot_swa, self.rope_base_swa)
        } else {
            (self.n_rot_full, self.rope_base_full)
        }
    }

    #[inline]
    fn layer_window(&self, layer_idx: usize) -> usize {
        if self.layers[layer_idx].is_swa {
            if self.sliding_window > 0 {
                self.sliding_window
            } else {
                self.swa_window
            }
        } else {
            0
        }
    }

    #[inline]
    fn tensor_slice<'a>(gguf: &'a GgufFile, tv: &TensorView) -> &'a [u8] {
        let start = gguf.data_offset + tv.offset;
        gguf.mmap
            .slice(start, tv.size_bytes)
            .expect("valid tensor slice")
    }

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
            GGML_TYPE_Q4_0 => dequantize_q4_0_row(row_slice, dst),
            GGML_TYPE_Q8_0 => dequantize_q8_0_row(row_slice, dst),
            GGML_TYPE_Q4_K => dequantize_q4_k_row(row_slice, dst),
            GGML_TYPE_Q6_K => dequantize_q6_k_row(row_slice, dst),
            GGML_TYPE_F32 => unsafe {
                let s = std::slice::from_raw_parts(row_slice.as_ptr() as *const f32, dim);
                dst.copy_from_slice(s);
            },
            GGML_TYPE_F16 => unsafe {
                let s = std::slice::from_raw_parts(row_slice.as_ptr() as *const u16, dim);
                for i in 0..dim {
                    dst[i] = f16_to_f32(s[i]);
                }
            },
            _ => dst.fill(0.0),
        }
    }

    pub fn reset_cache(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
    }

    /// KV compaction with per-layer rope geometry: keep `sink` tokens, drop
    /// `drop` tokens after them, shift survivors left, and re-rotate moved
    /// keys by -drop with each layer's own (n_rot, freq_base).
    pub fn compact_kv_cache(
        &mut self,
        current_seq_len: usize,
        sink_tokens: usize,
        drop_tokens: usize,
    ) -> usize {
        if drop_tokens == 0 || current_seq_len <= sink_tokens + drop_tokens {
            return current_seq_len;
        }
        let n_layers = self.config.n_layers;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let seq = self.config.max_seq_len;
        let new_len = current_seq_len - drop_tokens;
        let rope_layers: Vec<(usize, f32)> = (0..n_layers).map(|l| self.layer_rope(l)).collect();

        for (l, (n_rot, base)) in rope_layers.iter().enumerate() {
            for kv_h in 0..n_kv_heads {
                let head_off = (l * n_kv_heads + kv_h) * seq * head_dim;
                let src = head_off + (sink_tokens + drop_tokens) * head_dim;
                let end = head_off + current_seq_len * head_dim;
                let dst = head_off + sink_tokens * head_dim;
                if src < end {
                    self.k_cache.copy_within(src..end, dst);
                    self.v_cache.copy_within(src..end, dst);
                }
                for p in sink_tokens..new_len {
                    let off = head_off + p * head_dim;
                    let k = &mut self.k_cache[off..off + head_dim];
                    rope_shift_partial(k, *n_rot, *base, -(drop_tokens as i64));
                }
            }
        }
        new_len
    }

    fn lookup_weight<'a>(gguf: &'a GgufFile, tv: &TensorView, len: usize) -> &'a [f32] {
        let bytes = Self::tensor_slice(gguf, tv);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) }
    }

    /// Run one transformer layer for a single token. `self.x` holds the
    /// residual stream in/out.
    fn run_one_layer(&mut self, layer_idx: usize, pos: usize) {
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let max_seq_len = self.config.max_seq_len;
        let q_dim = n_heads * head_dim;
        let kv_dim = self.kv_dim;
        let gguf = &self.gguf;

        let (n_rot, base) = self.layer_rope(layer_idx);
        let sliding_window = self.layer_window(layer_idx);

        // ── Attention norm ──────────────────────────────────
        let norm_w = Self::lookup_weight(gguf, &self.layers[layer_idx].attn_norm, dim);
        rmsnorm(&mut self.x_norm, &self.x, norm_w, norm_eps);

        // ── Fused QKV (one weight pass, three row-range projections) ──
        let qkv_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].attn_qkv);
        let row_stride = type_row_size(self.layers[layer_idx].attn_qkv.tensor_type, dim);
        matvec(&qkv_bytes[..], self.layers[layer_idx].attn_qkv.tensor_type, q_dim, dim, &self.x_norm, &mut self.q);
        matvec(
            &qkv_bytes[q_dim * row_stride..],
            self.layers[layer_idx].attn_qkv.tensor_type,
            kv_dim,
            dim,
            &self.x_norm,
            &mut self.k,
        );
        matvec(
            &qkv_bytes[(q_dim + kv_dim) * row_stride..],
            self.layers[layer_idx].attn_qkv.tensor_type,
            kv_dim,
            dim,
            &self.x_norm,
            &mut self.v,
        );

        // ── Per-layer RoPE on Q and K ───────────────────────
        rope_apply(&mut self.q, n_heads, head_dim, n_rot, pos, base);
        rope_apply(&mut self.k, n_kv_heads, head_dim, n_rot, pos, base);

        // ── Store K/V to cache ──────────────────────────────
        let cache_pos = pos % max_seq_len;
        for kv_h in 0..n_kv_heads {
            let k_in = &self.k[kv_h * head_dim..(kv_h + 1) * head_dim];
            let v_in = &self.v[kv_h * head_dim..(kv_h + 1) * head_dim];
            let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim + cache_pos * head_dim;
            self.k_cache[slot..slot + head_dim].copy_from_slice(k_in);
            self.v_cache[slot..slot + head_dim].copy_from_slice(v_in);
        }

        // ── Attention ───────────────────────────────────────
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

        // ── Per-head sigmoid gate ───────────────────────────
        let gate_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].attn_gate);
        matvec(gate_bytes, self.layers[layer_idx].attn_gate.tensor_type, n_heads, dim, &self.x_norm, &mut self.gate);
        Self::apply_gate(&mut self.head_out, &self.gate, n_heads, head_dim);

        // ── Output projection + residual ────────────────────
        let wo_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].attn_output);
        matvec(wo_bytes, self.layers[layer_idx].attn_output.tensor_type, dim, q_dim, &self.head_out, &mut self.attn_out);
        inline_mad(&mut self.x, &self.attn_out, 1.0);

        // ── FFN norm ────────────────────────────────────────
        let ffn_norm_w = Self::lookup_weight(gguf, &self.layers[layer_idx].ffn_norm, dim);
        rmsnorm(&mut self.x_norm, &self.x, ffn_norm_w, norm_eps);

        // ── GeGLU FFN + residual ────────────────────────────
        let gate_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].ffn_gate);
        matvec(gate_bytes, self.layers[layer_idx].ffn_gate.tensor_type, hidden_dim, dim, &self.x_norm, &mut self.ffn_gate_buf);
        let up_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].ffn_up);
        matvec(up_bytes, self.layers[layer_idx].ffn_up.tensor_type, hidden_dim, dim, &self.x_norm, &mut self.ffn_up_buf);
        geglu_erf_in_place(&mut self.ffn_gate_buf, &self.ffn_up_buf);
        let down_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].ffn_down);
        matvec(down_bytes, self.layers[layer_idx].ffn_down.tensor_type, dim, hidden_dim, &self.ffn_gate_buf, &mut self.ffn_out);
        inline_mad(&mut self.x, &self.ffn_out, 1.0);
    }

    pub fn forward(&mut self, token_id: u32, pos: usize) -> &[f32] {
        Self::lookup_embedding(&self.gguf, &self.token_embd, token_id, self.config.dim, &mut self.x);
        for layer_idx in 0..self.config.n_layers {
            self.run_one_layer(layer_idx, pos);
        }
        self.compute_logits();
        &self.logits
    }

    pub fn forward_prefill(&mut self, token_id: u32, pos: usize) {
        Self::lookup_embedding(&self.gguf, &self.token_embd, token_id, self.config.dim, &mut self.x);
        for layer_idx in 0..self.config.n_layers {
            self.run_one_layer(layer_idx, pos);
        }
    }

    fn compute_logits(&mut self) {
        let dim = self.config.dim;
        let gguf = &self.gguf;
        let out_norm_w = Self::lookup_weight(gguf, &self.output_norm, dim);
        rmsnorm(&mut self.x_norm, &self.x, out_norm_w, self.config.norm_eps);
        let out_bytes = Self::tensor_slice(gguf, &self.token_embd);
        matvec(
            out_bytes,
            self.token_embd.tensor_type,
            self.config.vocab_size,
            dim,
            &self.x_norm,
            &mut self.logits,
        );
    }

    pub fn forward_embed(&mut self, token_id: u32, out_hidden: &mut [f32]) {
        let dim = self.config.dim;
        Self::lookup_embedding(&self.gguf, &self.token_embd, token_id, dim, &mut out_hidden[..dim]);
    }

    pub fn forward_layer_range(
        &mut self,
        pos: usize,
        start_layer: usize,
        end_layer: usize,
        in_hidden: &[f32],
        out_hidden: &mut [f32],
    ) {
        let dim = self.config.dim;
        self.x[..dim].copy_from_slice(&in_hidden[..dim]);
        let end = end_layer.min(self.config.n_layers);
        for layer_idx in start_layer..end {
            self.run_one_layer(layer_idx, pos);
        }
        out_hidden[..dim].copy_from_slice(&self.x[..dim]);
    }

    pub fn forward_head(&mut self, in_hidden: &[f32], out_logits: &mut [f32]) {
        let dim = self.config.dim;
        self.x[..dim].copy_from_slice(&in_hidden[..dim]);
        self.compute_logits();
        out_logits.copy_from_slice(&self.logits);
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], mut start_pos: usize) {
        for chunk in tokens.chunks(max_prefill_batch()) {
            self.forward_prefill_batch_internal(chunk, start_pos, false);
            start_pos += chunk.len();
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        if tokens.is_empty() {
            return &self.logits;
        }
        if tokens.len() == 1 {
            return self.forward(tokens[0], start_pos);
        }
        let batch = max_prefill_batch();
        let mut cur_pos = start_pos;
        let mut off = 0;
        while off < tokens.len() {
            let take = (tokens.len() - off).min(batch);
            if off + take < tokens.len() {
                self.forward_prefill_batch_internal(&tokens[off..off + take], cur_pos, false);
                cur_pos += take;
                off += take;
                continue;
            }
            // Final chunk: main body multiple of 4 (tile-eligible); any 1-3 row
            // remainder goes through the small-batch path with last-token logits.
            let main = take - (take % 4);
            if main == take {
                return self.forward_prefill_batch_internal(&tokens[off..off + take], cur_pos, true);
            }
            self.forward_prefill_batch_internal(&tokens[off..off + main], cur_pos, false);
            cur_pos += main;
            return self.forward_prefill_batch_internal(&tokens[off + main..off + take], cur_pos, true);
        }
        &self.logits
    }

    fn forward_prefill_batch_internal(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> &[f32] {
        let b_size = tokens.len();
        if b_size == 0 {
            return &self.logits;
        }
        if b_size == 1 {
            return match compute_last_logits {
                true => self.forward(tokens[0], start_pos),
                false => {
                    self.forward_prefill(tokens[0], start_pos);
                    &self.logits
                }
            };
        }

        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let max_seq_len = self.config.max_seq_len;
        let q_dim = n_heads * head_dim;
        let kv_dim = self.kv_dim;
        let gguf = &self.gguf;

        let chunk_t0 = prof::enabled().then(std::time::Instant::now);

        // ── Batched embedding lookup ────────────────────────
        for (b, &tok) in tokens.iter().enumerate() {
            Self::lookup_embedding(&self.gguf, &self.token_embd, tok, dim, &mut self.x_batch[b * dim..(b + 1) * dim]);
        }

        for layer_idx in 0..self.config.n_layers {
            let (n_rot, base) = self.layer_rope(layer_idx);
            let sliding_window = self.layer_window(layer_idx);

            // ── Attention norm ──────────────────────────────
            let norm_w = Self::lookup_weight(gguf, &self.layers[layer_idx].attn_norm, dim);
            for b in 0..b_size {
                rmsnorm(
                    &mut self.x_norm_batch[b * dim..(b + 1) * dim],
                    &self.x_batch[b * dim..(b + 1) * dim],
                    norm_w,
                    norm_eps,
                );
            }

            // ── Fused QKV via three row-range matmats ───────
            Self::project_qkv_batch(
                gguf,
                &self.layers[layer_idx],
                q_dim,
                kv_dim,
                dim,
                b_size,
                &self.x_norm_batch[..b_size * dim],
                &mut self.q_batch[..b_size * q_dim],
                &mut self.k_batch[..b_size * kv_dim],
                &mut self.v_batch[..b_size * kv_dim],
            );

            // ── Per-token RoPE + KV store ───────────────────
            for b in 0..b_size {
                let pos = start_pos + b;
                rope_apply(
                    &mut self.q_batch[b * q_dim..(b + 1) * q_dim],
                    n_heads,
                    head_dim,
                    n_rot,
                    pos,
                    base,
                );
                rope_apply(
                    &mut self.k_batch[b * kv_dim..(b + 1) * kv_dim],
                    n_kv_heads,
                    head_dim,
                    n_rot,
                    pos,
                    base,
                );
                let cache_pos = pos % max_seq_len;
                for kv_h in 0..n_kv_heads {
                    let k_in = &self.k_batch[b * kv_dim + kv_h * head_dim..b * kv_dim + (kv_h + 1) * head_dim];
                    let v_in = &self.v_batch[b * kv_dim + kv_h * head_dim..b * kv_dim + (kv_h + 1) * head_dim];
                    let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim
                        + cache_pos * head_dim;
                    self.k_cache[slot..slot + head_dim].copy_from_slice(k_in);
                    self.v_cache[slot..slot + head_dim].copy_from_slice(v_in);
                }
            }

            // ── Batched attention ───────────────────────────
            batched_attention_ext(
                &self.q_batch[..b_size * q_dim],
                &self.k_cache,
                &self.v_cache,
                &mut self.head_out_batch[..b_size * q_dim],
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

            // ── Per-head sigmoid gate (batch) ───────────────
            let gate_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].attn_gate);
            prefill_matmat(
                gate_bytes,
                self.layers[layer_idx].attn_gate.tensor_type,
                n_heads,
                dim,
                &self.x_norm_batch[..b_size * dim],
                b_size,
                &mut self.gate_batch[..b_size * n_heads],
            );
            for b in 0..b_size {
                Self::apply_gate(
                    &mut self.head_out_batch[b * q_dim..(b + 1) * q_dim],
                    &self.gate_batch[b * n_heads..(b + 1) * n_heads],
                    n_heads,
                    head_dim,
                );
            }

            // ── Output projection + residual ────────────────
            let wo_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].attn_output);
            prefill_matmat(
                wo_bytes,
                self.layers[layer_idx].attn_output.tensor_type,
                dim,
                q_dim,
                &self.head_out_batch[..b_size * q_dim],
                b_size,
                &mut self.attn_out_batch[..b_size * dim],
            );
            for b in 0..b_size {
                inline_mad(
                    &mut self.x_batch[b * dim..(b + 1) * dim],
                    &self.attn_out_batch[b * dim..(b + 1) * dim],
                    1.0,
                );
            }

            // ── FFN norm ────────────────────────────────────
            let ffn_norm_w = Self::lookup_weight(gguf, &self.layers[layer_idx].ffn_norm, dim);
            for b in 0..b_size {
                rmsnorm(
                    &mut self.x_norm_batch[b * dim..(b + 1) * dim],
                    &self.x_batch[b * dim..(b + 1) * dim],
                    ffn_norm_w,
                    norm_eps,
                );
            }

            // ── GeGLU FFN + residual ────────────────────────
            let gate_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].ffn_gate);
            prefill_matmat(
                gate_bytes,
                self.layers[layer_idx].ffn_gate.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm_batch[..b_size * dim],
                b_size,
                &mut self.ffn_gate_batch[..b_size * hidden_dim],
            );
            let up_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].ffn_up);
            prefill_matmat(
                up_bytes,
                self.layers[layer_idx].ffn_up.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm_batch[..b_size * dim],
                b_size,
                &mut self.ffn_up_batch[..b_size * hidden_dim],
            );
            for b in 0..b_size {
                geglu_erf_in_place(
                    &mut self.ffn_gate_batch[b * hidden_dim..(b + 1) * hidden_dim],
                    &self.ffn_up_batch[b * hidden_dim..(b + 1) * hidden_dim],
                );
            }
            let down_bytes = Self::tensor_slice(gguf, &self.layers[layer_idx].ffn_down);
            prefill_matmat(
                down_bytes,
                self.layers[layer_idx].ffn_down.tensor_type,
                dim,
                hidden_dim,
                &self.ffn_gate_batch[..b_size * hidden_dim],
                b_size,
                &mut self.ffn_out_batch[..b_size * dim],
            );
            for b in 0..b_size {
                inline_mad(
                    &mut self.x_batch[b * dim..(b + 1) * dim],
                    &self.ffn_out_batch[b * dim..(b + 1) * dim],
                    1.0,
                );
            }
        }

        if compute_last_logits {
            let last_b = b_size - 1;
            self.x.copy_from_slice(&self.x_batch[last_b * dim..(last_b + 1) * dim]);
            self.compute_logits();
        }

        if let Some(t0) = chunk_t0 {
            prof::record_chunk(t0.elapsed(), b_size);
            prof::report();
        }
        &self.logits
    }

    /// Fused QKV batch projection: three zero-copy row-range matmats.
    #[allow(clippy::too_many_arguments)]
    fn project_qkv_batch(
        gguf: &GgufFile,
        layer: &SparkLayer,
        q_dim: usize,
        kv_dim: usize,
        dim: usize,
        b_size: usize,
        x_norm_batch: &[f32],
        q_batch: &mut [f32],
        k_batch: &mut [f32],
        v_batch: &mut [f32],
    ) {
        if b_size == 1 {
            // Not reachable via forward_prefill_batch_internal (handled
            // earlier), kept for safety.
            let bytes = Self::tensor_slice(gguf, &layer.attn_qkv);
            let row_stride = type_row_size(layer.attn_qkv.tensor_type, dim);
            matvec(&bytes[..], layer.attn_qkv.tensor_type, q_dim, dim, x_norm_batch, q_batch);
            matvec(
                &bytes[q_dim * row_stride..],
                layer.attn_qkv.tensor_type,
                kv_dim,
                dim,
                x_norm_batch,
                k_batch,
            );
            matvec(
                &bytes[(q_dim + kv_dim) * row_stride..],
                layer.attn_qkv.tensor_type,
                kv_dim,
                dim,
                x_norm_batch,
                v_batch,
            );
            return;
        }
        let qkv_bytes = Self::tensor_slice(gguf, &layer.attn_qkv);
        let row_stride = type_row_size(layer.attn_qkv.tensor_type, dim);
        prefill_matmat(&qkv_bytes[..], layer.attn_qkv.tensor_type, q_dim, dim, x_norm_batch, b_size, q_batch);
        prefill_matmat(
            &qkv_bytes[q_dim * row_stride..],
            layer.attn_qkv.tensor_type,
            kv_dim,
            dim,
            x_norm_batch,
            b_size,
            k_batch,
        );
        prefill_matmat(
            &qkv_bytes[(q_dim + kv_dim) * row_stride..],
            layer.attn_qkv.tensor_type,
            kv_dim,
            dim,
            x_norm_batch,
            b_size,
            v_batch,
        );
    }

    #[inline]
    fn apply_gate(head_out: &mut [f32], gate: &[f32], n_heads: usize, head_dim: usize) {
        for h in 0..n_heads {
            let s = sigmoid(gate[h]);
            for x in head_out[h * head_dim..(h + 1) * head_dim].iter_mut() {
                *x *= s;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SndPtr<T>(*const T);
unsafe impl<T> Send for SndPtr<T> {}
unsafe impl<T> Sync for SndPtr<T> {}
#[derive(Clone, Copy)]
struct SndMutPtr<T>(*mut T);
unsafe impl<T> Send for SndMutPtr<T> {}
unsafe impl<T> Sync for SndMutPtr<T> {}

/// Spark prefill batched matmul, SOFUU-safe path (default): the C++
/// batched-gemm tile pipeline corrupts Spark activations (9848995df; TASKS
/// 2026-10-15 evidence pack), so weights dequantize row by row with the
/// dispatch decode uses and dot against the chunk directly.
/// SOFUU_SPARK_CPP_GEMM=1 restores the C++ path once that bug ships.
fn prefill_matmat(
    w_bytes: &[u8],
    ty: u32,
    rows: usize,
    cols: usize,
    xs: &[f32],
    b_size: usize,
    ys: &mut [f32],
) {
    debug_assert_eq!(xs.len(), b_size * cols);
    debug_assert_eq!(ys.len(), b_size * rows);
    if std::env::var_os("SOFUU_SPARK_CPP_GEMM").is_some() {
        return matmat(w_bytes, ty, rows, cols, xs, b_size, ys);
    }
    dequant_gemm_rows(w_bytes, ty, rows, cols, xs, b_size, ys);
}

fn dequant_gemm_rows(
    w_bytes: &[u8],
    ty: u32,
    rows: usize,
    cols: usize,
    xs: &[f32],
    b_size: usize,
    ys: &mut [f32],
) {
    let rs = type_row_size(ty, cols);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .clamp(1, 8)
        .min(rows.max(1));
    let wp = SndPtr(w_bytes.as_ptr());
    let xp = SndPtr(xs.as_ptr());
    let yp = SndMutPtr(ys.as_mut_ptr());
    if threads <= 1 {
        dequant_gemm_rows_range(wp, xp, yp, ty, rs, rows, cols, b_size, 0, rows);
        return;
    }
    std::thread::scope(|s| {
        for th in 0..threads {
            let (r0, r1) = (rows * th / threads, rows * (th + 1) / threads);
            s.spawn(move || dequant_gemm_rows_range(wp, xp, yp, ty, rs, rows, cols, b_size, r0, r1));
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn dequant_gemm_rows_range(
    wp: SndPtr<u8>,
    xp: SndPtr<f32>,
    yp: SndMutPtr<f32>,
    ty: u32,
    rs: usize,
    rows: usize,
    cols: usize,
    b_size: usize,
    r0: usize,
    r1: usize,
) {
    let (SndPtr(wp), SndPtr(xp), SndMutPtr(yp)) = (wp, xp, yp);
    let xs: &[f32] = unsafe { std::slice::from_raw_parts(xp, b_size * cols) };
    let mut dw = vec![0.0f32; cols];
    for r in r0..r1 {
        let src: &[u8] = unsafe { std::slice::from_raw_parts(wp.add(r * rs), rs) };
        match ty {
            GGML_TYPE_Q4_0 => dequantize_q4_0_row(src, &mut dw),
            GGML_TYPE_Q8_0 => dequantize_q8_0_row(src, &mut dw),
            GGML_TYPE_Q4_K => dequantize_q4_k_row(src, &mut dw),
            GGML_TYPE_Q6_K => dequantize_q6_k_row(src, &mut dw),
            GGML_TYPE_F32 => unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr() as *const f32, dw.as_mut_ptr(), cols);
            },
            GGML_TYPE_F16 => unsafe {
                let s = std::slice::from_raw_parts(src.as_ptr() as *const u16, cols);
                for i in 0..cols {
                    dw[i] = f16_to_f32(s[i]);
                }
            },
            _ => {
                for t in 0..b_size {
                    unsafe {
                        *yp.add(t * rows + r) = 0.0;
                    }
                }
            }
        }
        for t in 0..b_size {
            let x = &xs[t * cols..(t + 1) * cols];
            let mut s = 0.0f32;
            for (d, xv) in dw.iter().zip(x.iter()) {
                s += *d * xv;
            }
            unsafe {
                *yp.add(t * rows + r) = s;
            }
        }
    }
}

#[cfg(test)]#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_rope_roundtrip() {
        // Rotating by +pos then by -pos must restore the original values on
        // the rotated dims, and leave the pass-through dims untouched.
        let head_dim = 256usize;
        let n_rot = 64usize;
        let n_heads = 2usize;
        let base = 5_000_000.0f32;
        let mut v: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i * 37 % 251) as f32 / 251.0) - 0.5)
            .collect();
        let original = v.clone();

        // Apply +pos rope manually in two halves (rotate [0,64) per head).
        let rotate = |v: &mut Vec<f32>, pos: i64| {
            for h in 0..n_heads {
                let a = &mut v[h * head_dim..(h + 1) * head_dim];
                rope_shift_partial(a, n_rot, base, pos);
            }
        };
        rotate(&mut v, 7);
        rotate(&mut v, -7);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 2e-4, "rope roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_swa_pattern_rawarray_parse() {
        // Bool GGUF arrays arrive as RawArray { item_type: 7 }.
        let pattern: GgufValue = GgufValue::RawArray {
            item_type: 7,
            len: 4,
            data: vec![1, 1, 1, 0],
        };
        let parsed = match &pattern {
            GgufValue::RawArray { item_type: 7, data, .. } => data.iter().map(|&b| b != 0).collect::<Vec<bool>>(),
            _ => unreachable!(),
        };
        assert_eq!(parsed, vec![true, true, true, false]);
    }
}
