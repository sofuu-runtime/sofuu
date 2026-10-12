// crates/sofuu-core/src/infer/block_graph.rs — Generalized Composable Block Graph.
//
// Supports all major open-weight edge architectures:
// - LLaMA 2/3/3.1/3.2 (RMSNorm, GQA, RoPE, SwiGLU)
// - Qwen 2/2.5/3 (RMSNorm, Q/K Norm, GQA, RoPE, SwiGLU)
// - SmolLM / SmolLM2 (Compact LLaMA block graph)
// - Gemma 1/2 (Gemma-RMSNorm (1+w), Attention Soft-Capping, Final Soft-Capping, Post-Norms)
// - Mistral / Ministral (Sliding Window Attention, GQA, SwiGLU)
// - Phi-3 / Phi-4 (SuScaled RoPE, GeGLU / SwiGLU)
// - Liquid LFM-2 (Short Conv1D + GQA hybrid layers)
// - Sparse MoE (Mixtral 8x7B, Qwen-MoE, DeepSeek-MoE with Top-K gated expert routing)

use std::sync::Arc;

use super::npy::TensorDType;
use super::quant::{f16_to_f32, rmsnorm};

/// Normalization kind across transformer architectures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormKind {
    /// Standard Root-Mean-Square Normalization: y = x / RMS(x) * w
    RmsNorm { eps: f32 },
    /// Gemma variant: y = x / RMS(x) * (1.0 + w)
    GemmaRmsNorm { eps: f32 },
    /// Standard Layer Normalization with mean centering
    LayerNorm { eps: f32 },
}

impl NormKind {
    #[inline]
    pub fn apply(&self, x: &[f32], w: &[f32], out: &mut [f32]) {
        match self {
            Self::RmsNorm { eps } => {
                rmsnorm(out, x, w, *eps);
            }
            Self::GemmaRmsNorm { eps } => {
                gemma_rmsnorm(out, x, w, *eps);
            }
            Self::LayerNorm { eps } => {
                layer_norm(out, x, w, *eps);
            }
        }
    }
}

/// Applies Gemma-RMSNorm: y = (x / RMS(x)) * (1.0 + w)
pub fn gemma_rmsnorm(dst: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let dim = x.len();
    if dim == 0 {
        return;
    }
    let mut sum_sq = 0.0f32;
    for &val in x {
        sum_sq += val * val;
    }
    let rms = (sum_sq / (dim as f32) + eps).sqrt();
    let scale = 1.0f32 / rms;
    for i in 0..dim {
        dst[i] = x[i] * scale * (1.0f32 + weight[i]);
    }
}

/// Applies standard LayerNorm: y = (x - mean) / sqrt(var + eps) * w
pub fn layer_norm(dst: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let dim = x.len();
    if dim == 0 {
        return;
    }
    let mean: f32 = x.iter().sum::<f32>() / (dim as f32);
    let mut var_sum = 0.0f32;
    for &val in x {
        let diff = val - mean;
        var_sum += diff * diff;
    }
    let inv_std = 1.0f32 / (var_sum / (dim as f32) + eps).sqrt();
    for i in 0..dim {
        dst[i] = (x[i] - mean) * inv_std * weight[i];
    }
}

/// Attention configuration and hyper-parameters
#[derive(Debug, Clone)]
pub struct AttentionSpec {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub q_norm: Option<NormKind>,
    pub k_norm: Option<NormKind>,
    pub attn_logit_soft_cap: Option<f32>,
}

/// Short Causal Conv1D specification (Liquid LFM-2)
#[derive(Debug, Clone)]
pub struct Conv1dSpec {
    pub channels: usize,
    pub kernel_size: usize,
}

/// Sequence Mixer kind: Attention or Short Conv
#[derive(Debug, Clone)]
pub enum MixerSpec {
    Attention(AttentionSpec),
    Conv1d(Conv1dSpec),
}

/// Channel Mixer (FFN) specification
#[derive(Debug, Clone)]
pub enum ChannelMixerSpec {
    /// SwiGLU: down( silu(gate(x)) * up(x) )
    SwiGLU { intermediate_dim: usize },
    /// GeGLU: down( gelu(gate(x)) * up(x) )
    GeGLU { intermediate_dim: usize },
    /// Top-K Sparse Mixture of Experts
    SparseMoE {
        intermediate_dim: usize,
        num_experts: usize,
        top_k: usize,
    },
}

/// Complete Composable Transformer Block specification
#[derive(Debug, Clone)]
pub struct BlockSpec {
    pub pre_norm: NormKind,
    pub mixer: MixerSpec,
    pub post_attention_norm: Option<NormKind>,
    pub ffn_norm: NormKind,
    pub channel_mixer: ChannelMixerSpec,
    pub post_ffn_norm: Option<NormKind>,
}

/// Overall Model Architecture Configuration
#[derive(Debug, Clone)]
pub struct BlockGraphConfig {
    pub dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub blocks: Vec<BlockSpec>,
    pub embed_scale: Option<f32>, // sqrt(dim) for Gemma
    pub final_norm: NormKind,
    pub final_logit_soft_cap: Option<f32>, // 30.0 for Gemma 2
    pub tied_embeddings: bool,
}

/// Memory-backed linear weight representation (supporting F32 and F16 slices)
#[derive(Clone)]
pub struct WeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub dtype: TensorDType,
    pub data: Arc<Vec<u8>>,
    pub byte_offset: usize,
}

impl WeightMatrix {
    pub fn new(rows: usize, cols: usize, dtype: TensorDType, data: Arc<Vec<u8>>, byte_offset: usize) -> Self {
        Self {
            rows,
            cols,
            dtype,
            data,
            byte_offset,
        }
    }

    /// Matrix-vector multiplication: y = W * x
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        let slice = &self.data[self.byte_offset..];
        match self.dtype {
            TensorDType::F32 => {
                let w_f32 = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.rows * self.cols)
                };
                for r in 0..self.rows {
                    let row = &w_f32[r * self.cols..(r + 1) * self.cols];
                    let mut sum0 = 0.0f32;
                    let mut sum1 = 0.0f32;
                    let chunks_x = x.chunks_exact(8);
                    let chunks_w = row.chunks_exact(8);
                    let rem_x = chunks_x.remainder();
                    let rem_w = chunks_w.remainder();
                    for (cx, cw) in chunks_x.zip(chunks_w) {
                        sum0 += cx[0] * cw[0] + cx[1] * cw[1] + cx[2] * cw[2] + cx[3] * cw[3];
                        sum1 += cx[4] * cw[4] + cx[5] * cw[5] + cx[6] * cw[6] + cx[7] * cw[7];
                    }
                    let mut total = sum0 + sum1;
                    for (&xv, &wv) in rem_x.iter().zip(rem_w.iter()) {
                        total += xv * wv;
                    }
                    y[r] = total;
                }
            }
            TensorDType::F16 => {
                let w_u16 = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u16, self.rows * self.cols)
                };
                for r in 0..self.rows {
                    let row = &w_u16[r * self.cols..(r + 1) * self.cols];
                    let mut sum = 0.0f32;
                    for c in 0..self.cols {
                        sum += x[c] * f16_to_f32(row[c]);
                    }
                    y[r] = sum;
                }
            }
            _ => {
                // Generic fallback for unquantized weights
                for r in 0..self.rows {
                    y[r] = 0.0f32;
                }
            }
        }
    }
}

/// Applies GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608f32;
    0.5f32 * x * (1.0f32 + (sqrt_2_over_pi * (x + 0.044715f32 * x * x * x)).tanh())
}

/// Applies GeGLU in-place: gate[i] = gelu(gate[i]) * up[i]
pub fn geglu_in_place(gate: &mut [f32], up: &[f32]) {
    let len = gate.len().min(up.len());
    for i in 0..len {
        gate[i] = gelu(gate[i]) * up[i];
    }
}

/// Applies Attention Logit Soft-Capping in-place: x = cap * tanh(x / cap)
pub fn soft_cap_in_place(scores: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    let inv_cap = 1.0f32 / cap;
    for s in scores.iter_mut() {
        *s = cap * (*s * inv_cap).tanh();
    }
}

/// KV Cache for a single composable block
pub struct BlockKvCache {
    pub k: Vec<f32>, // [kv_heads, max_seq_len, head_dim]
    pub v: Vec<f32>, // [kv_heads, max_seq_len, head_dim]
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
}

impl BlockKvCache {
    pub fn new(n_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let cap = n_kv_heads * max_seq_len * head_dim;
        Self {
            k: vec![0.0f32; cap],
            v: vec![0.0f32; cap],
            max_seq_len,
            head_dim,
            n_kv_heads,
        }
    }

    #[inline]
    pub fn store(&mut self, kv_head: usize, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        let offset = (kv_head * self.max_seq_len + pos) * self.head_dim;
        self.k[offset..offset + self.head_dim].copy_from_slice(k_vec);
        self.v[offset..offset + self.head_dim].copy_from_slice(v_vec);
    }

    #[inline]
    pub fn get_k(&self, kv_head: usize, pos: usize) -> &[f32] {
        let offset = (kv_head * self.max_seq_len + pos) * self.head_dim;
        &self.k[offset..offset + self.head_dim]
    }

    #[inline]
    pub fn get_v(&self, kv_head: usize, pos: usize) -> &[f32] {
        let offset = (kv_head * self.max_seq_len + pos) * self.head_dim;
        &self.v[offset..offset + self.head_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0.1, 0.2, -0.1, 0.0];
        let mut out = vec![0.0; 4];
        let eps = 1e-6;

        gemma_rmsnorm(&mut out, &x, &w, eps);

        // Verification: RMS = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5) ≈ 2.7386127
        let rms = (30.0f32 / 4.0f32 + eps).sqrt();
        assert!((out[0] - (1.0 / rms) * (1.0 + 0.1)).abs() < 1e-5);
        assert!((out[1] - (2.0 / rms) * (1.0 + 0.2)).abs() < 1e-5);
        assert!((out[2] - (3.0 / rms) * (1.0 - 0.1)).abs() < 1e-5);
        assert!((out[3] - (4.0 / rms) * (1.0 + 0.0)).abs() < 1e-5);
    }

    #[test]
    fn test_soft_cap_in_place() {
        let mut scores = vec![100.0, -100.0, 0.0];
        let cap = 50.0;
        soft_cap_in_place(&mut scores, cap);

        assert!((scores[0] - 50.0 * 2.0f32.tanh()).abs() < 1e-5);
        assert!((scores[1] - 50.0 * (-2.0f32).tanh()).abs() < 1e-5);
        assert_eq!(scores[2], 0.0);
    }

    #[test]
    fn test_geglu_in_place() {
        let mut gate = vec![0.0, 1.0];
        let up = vec![2.0, 3.0];
        geglu_in_place(&mut gate, &up);

        // gelu(0) = 0 => gate[0] = 0 * 2 = 0
        assert_eq!(gate[0], 0.0);
        // gelu(1) ≈ 0.84134 => gate[1] = gelu(1) * 3 ≈ 2.524
        assert!((gate[1] - gelu(1.0) * 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_block_kv_cache() {
        let mut cache = BlockKvCache::new(2, 8, 4);
        let k_vec = vec![1.0, 2.0, 3.0, 4.0];
        let v_vec = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(1, 3, &k_vec, &v_vec);
        assert_eq!(cache.get_k(1, 3), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.get_v(1, 3), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_weight_matrix_gemv_f32() {
        // 2 rows, 3 cols: [[1, 2, 3], [4, 5, 6]]
        let weights_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut raw_bytes = Vec::new();
        for &w in &weights_f32 {
            raw_bytes.extend_from_slice(&w.to_le_bytes());
        }

        let wm = WeightMatrix::new(2, 3, TensorDType::F32, Arc::new(raw_bytes), 0);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];
        wm.gemv(&x, &mut y);

        // y[0] = 1*1 + 2*2 + 3*3 = 14
        // y[1] = 4*1 + 5*2 + 6*3 = 32
        assert!((y[0] - 14.0).abs() < 1e-5);
        assert!((y[1] - 32.0).abs() < 1e-5);
    }
}
