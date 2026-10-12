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

use super::model::ModelConfig;
use super::npy::TensorDType;
use super::quant::{bf16_to_f32, f16_to_f32, rmsnorm, rope, swiglu_in_place};
use super::tensor_table::{UnifiedTensorData, UnifiedTensorTable};

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

/// MLX 4-bit affine quantized weight components
#[derive(Clone)]
pub struct Mlx4Weights {
    pub scales: UnifiedTensorData,
    #[cfg(target_os = "macos")]
    pub scales_metal_buf: Option<Arc<crate::infer::metal::MetalBuffer>>,
    pub biases: UnifiedTensorData,
    #[cfg(target_os = "macos")]
    pub biases_metal_buf: Option<Arc<crate::infer::metal::MetalBuffer>>,
    pub group_size: usize,
    pub is_bf16: bool,
}

/// Memory-backed linear weight representation (supporting F32, F16, BF16, and MLX4)
#[derive(Clone)]
pub struct WeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub dtype: TensorDType,
    pub data: UnifiedTensorData,
    #[cfg(target_os = "macos")]
    pub metal_buf: Option<Arc<crate::infer::metal::MetalBuffer>>,
    pub mlx4: Option<Mlx4Weights>,
}

impl WeightMatrix {
    pub fn new(rows: usize, cols: usize, dtype: TensorDType, data: Arc<Vec<u8>>, byte_offset: usize) -> Self {
        let slice = if byte_offset > 0 { &data[byte_offset..] } else { &data[..] };
        Self {
            rows,
            cols,
            dtype,
            data: UnifiedTensorData::Heap(Arc::new(slice.to_vec())),
            #[cfg(target_os = "macos")]
            metal_buf: None,
            mlx4: None,
        }
    }

    pub fn new_from_data(rows: usize, cols: usize, dtype: TensorDType, data: UnifiedTensorData) -> Self {
        Self {
            rows,
            cols,
            dtype,
            data,
            #[cfg(target_os = "macos")]
            metal_buf: None,
            mlx4: None,
        }
    }

    pub fn new_mlx4(
        rows: usize,
        cols: usize,
        data: UnifiedTensorData,
        scales: UnifiedTensorData,
        biases: UnifiedTensorData,
        group_size: usize,
        is_bf16: bool,
    ) -> Self {
        Self {
            rows,
            cols,
            dtype: TensorDType::U32,
            data,
            #[cfg(target_os = "macos")]
            metal_buf: None,
            mlx4: Some(Mlx4Weights {
                scales,
                #[cfg(target_os = "macos")]
                scales_metal_buf: None,
                biases,
                #[cfg(target_os = "macos")]
                biases_metal_buf: None,
                group_size,
                is_bf16,
            }),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn with_metal(mut self, engine: &crate::infer::metal::MetalEngine) -> Self {
        let slice = self.data.as_slice();
        self.metal_buf = engine.create_buffer(slice).map(Arc::new);
        if let Some(ref mut mlx4) = self.mlx4 {
            let s_slice = mlx4.scales.as_slice();
            mlx4.scales_metal_buf = engine.create_buffer(s_slice).map(Arc::new);
            let b_slice = mlx4.biases.as_slice();
            mlx4.biases_metal_buf = engine.create_buffer(b_slice).map(Arc::new);
        }
        self
    }

    /// Matrix-vector multiplication: y = W * x
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        if let Some(ref mlx4) = self.mlx4 {
            #[cfg(target_os = "macos")]
            if let (Some(ref m_buf), Some(ref s_buf), Some(ref b_buf)) = (
                &self.metal_buf,
                &mlx4.scales_metal_buf,
                &mlx4.biases_metal_buf,
            ) {
                if let Some(engine) = crate::infer::metal::get_or_init_global_metal() {
                    engine.gemv_mlx4(
                        m_buf, 0,
                        s_buf, 0,
                        b_buf, 0,
                        x, y,
                        self.rows, self.cols, mlx4.group_size, mlx4.is_bf16,
                    );
                    return;
                }
            }

            let w_slice = self.data.as_slice();
            let s_slice = mlx4.scales.as_slice();
            let b_slice = mlx4.biases.as_slice();
            crate::infer::quant::gemv_mlx4(
                w_slice, s_slice, b_slice,
                x, y,
                self.rows, self.cols, mlx4.group_size, mlx4.is_bf16,
            );
            return;
        }

        #[cfg(target_os = "macos")]
        if let Some(ref m_buf) = self.metal_buf {
            if let Some(engine) = crate::infer::metal::get_or_init_global_metal() {
                match self.dtype {
                    TensorDType::BF16 => {
                        engine.gemv_bf16(m_buf, 0, x, y, self.rows, self.cols);
                        return;
                    }
                    TensorDType::F16 => {
                        engine.gemv_f16(m_buf, 0, x, y, self.rows, self.cols);
                        return;
                    }
                    TensorDType::F32 => {
                        engine.gemv_f32(m_buf, 0, x, y, self.rows, self.cols);
                        return;
                    }
                    _ => {}
                }
            }
        }

        let slice = self.data.as_slice();
        match self.dtype {
            TensorDType::F32 => {
                if slice.len() >= self.rows * self.cols * 4 && (slice.as_ptr() as usize) % 4 == 0 {
                    unsafe {
                        let f32_ptr = slice.as_ptr() as *const f32;
                        crate::infer::quant::sofuu_gemv_f32(f32_ptr, x.as_ptr(), y.as_mut_ptr(), self.rows, self.cols);
                    }
                    return;
                }
                // Unaligned or boundary fallback
                for r in 0..self.rows {
                    let base = r * self.cols * 4;
                    let mut sum = 0.0f32;
                    for c in 0..self.cols {
                        let raw = f32::from_ne_bytes(slice[base + c * 4..base + (c + 1) * 4].try_into().unwrap());
                        sum += x[c] * raw;
                    }
                    y[r] = sum;
                }
            }
            TensorDType::F16 => {
                if slice.len() >= self.rows * self.cols * 2 && (slice.as_ptr() as usize) % 2 == 0 {
                    unsafe {
                        let u16_ptr = slice.as_ptr() as *const u16;
                        crate::infer::quant::sofuu_gemv_f16(u16_ptr, x.as_ptr(), y.as_mut_ptr(), self.rows, self.cols);
                    }
                    return;
                }
                for r in 0..self.rows {
                    let mut sum = 0.0f32;
                    let base = r * self.cols * 2;
                    for c in 0..self.cols {
                        let raw = u16::from_ne_bytes(slice[base + c * 2..base + (c + 1) * 2].try_into().unwrap());
                        sum += x[c] * f16_to_f32(raw);
                    }
                    y[r] = sum;
                }
            }
            TensorDType::BF16 => {
                if slice.len() >= self.rows * self.cols * 2 {
                    crate::infer::quant::gemv_bf16(slice, x, &mut y[..self.rows], self.rows, self.cols);
                    return;
                }
                for r in 0..self.rows {
                    let mut sum = 0.0f32;
                    let base = r * self.cols * 2;
                    for c in 0..self.cols {
                        let raw = u16::from_ne_bytes(slice[base + c * 2..base + (c + 1) * 2].try_into().unwrap());
                        sum += x[c] * bf16_to_f32(raw);
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

    /// Read a single embedding row into destination f32 buffer.
    pub fn get_row(&self, row: usize, dst: &mut [f32]) {
        if row >= self.rows {
            dst.fill(0.0);
            return;
        }
        if let Some(ref mlx4) = self.mlx4 {
            let u32_per_row = self.cols / 8;
            let grp_per_row = self.cols / mlx4.group_size;
            let w_slice = &self.data.as_slice()[row * u32_per_row * 4..];
            let s_slice = &mlx4.scales.as_slice()[row * grp_per_row * 2..];
            let b_slice = &mlx4.biases.as_slice()[row * grp_per_row * 2..];
            let w_u32 = unsafe { std::slice::from_raw_parts(w_slice.as_ptr() as *const u32, u32_per_row) };
            let s_f16 = unsafe { std::slice::from_raw_parts(s_slice.as_ptr() as *const u16, grp_per_row) };
            let b_f16 = unsafe { std::slice::from_raw_parts(b_slice.as_ptr() as *const u16, grp_per_row) };
            for c in 0..self.cols {
                let u_idx = c / 8;
                let shift = (c % 8) * 4;
                let q = ((w_u32[u_idx] >> shift) & 0x0F) as f32;
                let grp = c / mlx4.group_size;
                let scale = if mlx4.is_bf16 { bf16_to_f32(s_f16[grp]) } else { f16_to_f32(s_f16[grp]) };
                let bias = if mlx4.is_bf16 { bf16_to_f32(b_f16[grp]) } else { f16_to_f32(b_f16[grp]) };
                dst[c] = q * scale + bias;
            }
            return;
        }

        let slice = self.data.as_slice();
        match self.dtype {
            TensorDType::F32 => {
                let w_f32 = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.rows * self.cols)
                };
                let r = &w_f32[row * self.cols..(row + 1) * self.cols];
                dst[..self.cols].copy_from_slice(r);
            }
            TensorDType::F16 => {
                let w_u16 = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u16, self.rows * self.cols)
                };
                let r = &w_u16[row * self.cols..(row + 1) * self.cols];
                for i in 0..self.cols {
                    dst[i] = f16_to_f32(r[i]);
                }
            }
            TensorDType::BF16 => {
                let w_u16 = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u16, self.rows * self.cols)
                };
                let r = &w_u16[row * self.cols..(row + 1) * self.cols];
                for i in 0..self.cols {
                    dst[i] = bf16_to_f32(r[i]);
                }
            }
            _ => {
                dst.fill(0.0);
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

/// Complete Composable Transformer Layer weights and spec
pub struct BlockGraphLayer {
    pub pre_norm: NormKind,
    pub pre_norm_weight: Vec<f32>,
    pub mixer_weights: BlockMixerWeights,
    pub post_attention_norm: Option<NormKind>,
    pub post_attention_norm_weight: Option<Vec<f32>>,
    pub ffn_norm: NormKind,
    pub ffn_norm_weight: Vec<f32>,
    pub channel_mixer_weights: BlockChannelMixerWeights,
    pub post_ffn_norm: Option<NormKind>,
    pub post_ffn_norm_weight: Option<Vec<f32>>,
}

pub enum BlockMixerWeights {
    Attention {
        spec: AttentionSpec,
        q_proj: WeightMatrix,
        k_proj: WeightMatrix,
        v_proj: WeightMatrix,
        o_proj: WeightMatrix,
        q_norm_weight: Option<Vec<f32>>,
        k_norm_weight: Option<Vec<f32>>,
    },
    Conv1d {
        spec: Conv1dSpec,
        conv_weight: Vec<f32>,
    },
}

pub enum BlockChannelMixerWeights {
    SwiGLU {
        gate_proj: WeightMatrix,
        up_proj: WeightMatrix,
        down_proj: WeightMatrix,
    },
    GeGLU {
        gate_proj: WeightMatrix,
        up_proj: WeightMatrix,
        down_proj: WeightMatrix,
    },
    SparseMoE {
        num_experts: usize,
        top_k: usize,
        router: WeightMatrix,
        experts_gate: Vec<WeightMatrix>,
        experts_up: Vec<WeightMatrix>,
        experts_down: Vec<WeightMatrix>,
    },
}

fn find_and_load_weight<S: AsRef<str>>(table: &UnifiedTensorTable, candidates: &[S]) -> Result<WeightMatrix, String> {
    for cand in candidates {
        if let Some((meta, data)) = table.get_tensor_data(cand.as_ref()) {
            if meta.dtype == TensorDType::U32 {
                // Check if this is an MLX 4-bit matrix with scales and biases
                let cand_str = cand.as_ref();
                let base = cand_str.trim_end_matches(".weight");
                let s_name = format!("{}.scales", base);
                let b_name = format!("{}.biases", base);
                if let (Some((s_meta, s_data)), Some((_b_meta, b_data))) =
                    (table.get_tensor_data(&s_name), table.get_tensor_data(&b_name))
                {
                    let rows = meta.shape[0];
                    let cols = if meta.shape.len() > 1 { meta.shape[1] * 8 } else { meta.shape[0] * 8 };
                    let group_size = if s_meta.shape.len() > 1 && s_meta.shape[1] > 0 {
                        cols / s_meta.shape[1]
                    } else {
                        64
                    };
                    let is_bf16 = s_meta.dtype == TensorDType::BF16;
                    return Ok(WeightMatrix::new_mlx4(
                        rows,
                        cols,
                        data,
                        s_data,
                        b_data,
                        group_size,
                        is_bf16,
                    ));
                }
            }

            let (rows, cols) = if meta.shape.len() == 2 {
                (meta.shape[0], meta.shape[1])
            } else if meta.shape.len() == 1 {
                (1, meta.shape[0])
            } else {
                (meta.shape[0], meta.shape[1..].iter().product())
            };
            return Ok(WeightMatrix::new_from_data(
                rows,
                cols,
                meta.dtype,
                data,
            ));
        }
    }
    Err(format!("failed to find tensor matching candidates"))
}

fn find_and_load_vec<S: AsRef<str>>(table: &UnifiedTensorTable, candidates: &[S]) -> Result<Vec<f32>, String> {
    for cand in candidates {
        if let Some((meta, slice)) = table.get_tensor(cand.as_ref()) {
            let len: usize = meta.shape.iter().product();
            let v = match meta.dtype {
                TensorDType::F32 => {
                    let s = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, len) };
                    s.to_vec()
                }
                TensorDType::F16 => {
                    let s = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u16, len) };
                    s.iter().map(|&h| f16_to_f32(h)).collect()
                }
                TensorDType::BF16 => {
                    let s = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u16, len) };
                    s.iter().map(|&h| bf16_to_f32(h)).collect()
                }
                _ => return Err(format!("unsupported dtype {:?} for norm vector", meta.dtype)),
            };
            return Ok(v);
        }
    }
    Err(format!("failed to find norm vector matching candidates"))
}

#[derive(serde::Deserialize, Default, Debug)]
struct HfConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    final_logit_soft_cap: Option<f32>,
    #[serde(default)]
    attn_logit_soft_cap: Option<f32>,
}

pub struct BlockGraphModel {
    pub config: ModelConfig,
    pub graph_config: BlockGraphConfig,
    pub embed_tokens: WeightMatrix,
    pub layers: Vec<BlockGraphLayer>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head: Option<WeightMatrix>,
    pub kv_caches: Vec<BlockKvCache>,
    pub logits: Vec<f32>,
    pub buf_x: Vec<f32>,
    pub buf_norm: Vec<f32>,
    pub buf_res: Vec<f32>,
    pub buf_attn_out: Vec<f32>,
    pub buf_q: Vec<f32>,
    pub buf_k: Vec<f32>,
    pub buf_v: Vec<f32>,
    pub buf_gate: Vec<f32>,
    pub buf_up: Vec<f32>,
    pub buf_ffn_out: Vec<f32>,
    pub buf_scores: Vec<f32>,
    pub buf_tmp: Vec<f32>,
    pub gpu_layers: usize,
    #[cfg(target_os = "macos")]
    pub metal_engine: Option<crate::infer::metal::MetalEngine>,
}

impl BlockGraphModel {
    pub fn from_tensor_table(table: &UnifiedTensorTable, max_seq_len: usize) -> Result<Self, String> {
        let names = table.tensor_names();

        // 1. Discover number of layers
        let mut max_layer_idx = 0;
        let mut found_layer = false;
        for name in &names {
            for part in name.split('.') {
                if let Ok(idx) = part.parse::<usize>() {
                    if name.contains("layer") || name.contains("blk") || name.contains("block") {
                        max_layer_idx = max_layer_idx.max(idx + 1);
                        found_layer = true;
                    }
                }
            }
        }
        let n_layers = if found_layer { max_layer_idx } else { 0 };
        if n_layers == 0 {
            return Err("no transformer layers detected in tensor table".to_string());
        }

        // 2. Discover embed tokens
        let embed_cands = [
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "tok_embeddings.weight",
            "transformer.wte.weight",
        ];
        let embed_tokens = find_and_load_weight(table, &embed_cands)?;
        let vocab_size = embed_tokens.rows;
        let dim = embed_tokens.cols;

        // Discover model metadata from config.json if available
        let hf_cfg: Option<HfConfig> = if let Some(ref path) = table.path {
            let dir = if path.is_dir() { path.clone() } else { path.parent().unwrap_or(path).to_path_buf() };
            let cfg_path = dir.join("config.json");
            if cfg_path.exists() {
                std::fs::read_to_string(&cfg_path)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
            } else {
                None
            }
        } else {
            None
        };

        // 3. Detect architecture variants (Gemma 1/2 vs standard LLaMA/Qwen/Mistral)
        let is_gemma_2 = names.iter().any(|n| n.contains("post_feedforward_layernorm"))
            || hf_cfg.as_ref().map(|c| c.model_type.contains("gemma2")).unwrap_or(false);
        let is_gemma_1 = hf_cfg.as_ref().map(|c| {
            c.model_type.to_lowercase() == "gemma" || c.architectures.iter().any(|a| a.to_lowercase().contains("gemma"))
        }).unwrap_or(false) || (vocab_size == 256000 && names.iter().any(|n| n.contains("post_attention_layernorm")));
        let is_gemma = is_gemma_1 || is_gemma_2;

        let norm_eps = hf_cfg.as_ref().and_then(|c| c.rms_norm_eps).unwrap_or(if is_gemma { 1e-6 } else { 1e-5 });
        let norm_kind = if is_gemma {
            NormKind::GemmaRmsNorm { eps: norm_eps }
        } else {
            NormKind::RmsNorm { eps: norm_eps }
        };
        let embed_scale = if is_gemma { Some((dim as f32).sqrt()) } else { None };
        let final_soft_cap = if is_gemma_2 {
            hf_cfg.as_ref().and_then(|c| c.final_logit_soft_cap).or(Some(30.0))
        } else {
            None
        };
        let attn_soft_cap = if is_gemma_2 {
            hf_cfg.as_ref().and_then(|c| c.attn_logit_soft_cap).or(Some(50.0))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(n_layers);
        let mut inferred_n_heads = 0;
        let mut inferred_n_kv_heads = 0;
        let mut inferred_head_dim = 0;
        let mut max_intermediate_dim = 0;

        for i in 0..n_layers {
            // Pre-norm
            let pre_norm_cands = [
                format!("model.layers.{i}.input_layernorm.weight"),
                format!("layers.{i}.input_layernorm.weight"),
                format!("layers.{i}.attention_norm.weight"),
            ];
            let pre_norm_weight = find_and_load_vec(table, &pre_norm_cands)?;

            // Attention projections
            let q_cands = [
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                format!("layers.{i}.self_attn.q_proj.weight"),
                format!("layers.{i}.attention.wq.weight"),
            ];
            let q_proj = find_and_load_weight(table, &q_cands)?;

            let k_cands = [
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                format!("layers.{i}.self_attn.k_proj.weight"),
                format!("layers.{i}.attention.wk.weight"),
            ];
            let k_proj = find_and_load_weight(table, &k_cands)?;

            let v_cands = [
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                format!("layers.{i}.self_attn.v_proj.weight"),
                format!("layers.{i}.attention.wv.weight"),
            ];
            let v_proj = find_and_load_weight(table, &v_cands)?;

            let o_cands = [
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                format!("layers.{i}.self_attn.o_proj.weight"),
                format!("layers.{i}.attention.wo.weight"),
            ];
            let o_proj = find_and_load_weight(table, &o_cands)?;

            if i == 0 {
                let q_rows = q_proj.rows;
                let k_rows = k_proj.rows;
                inferred_head_dim = if let Some(hd) = hf_cfg.as_ref().and_then(|c| c.head_dim) {
                    hd
                } else if is_gemma {
                    256
                } else if q_rows >= 128 && q_rows % 128 == 0 && q_rows / 128 <= 64 {
                    128
                } else if q_rows >= 64 && q_rows % 64 == 0 {
                    64
                } else {
                    q_rows.min(dim).max(1)
                };
                inferred_n_heads = if let Some(nh) = hf_cfg.as_ref().and_then(|c| c.num_attention_heads) {
                    nh
                } else {
                    (q_rows / inferred_head_dim).max(1)
                };
                inferred_n_kv_heads = if let Some(nkv) = hf_cfg.as_ref().and_then(|c| c.num_key_value_heads) {
                    nkv
                } else {
                    (k_rows / inferred_head_dim).max(1)
                };
            }

            // Q/K norms (Qwen 2.5 / 3)
            let q_norm_cands = [
                format!("model.layers.{i}.self_attn.q_norm.weight"),
                format!("layers.{i}.self_attn.q_norm.weight"),
            ];
            let q_norm_weight = find_and_load_vec(table, &q_norm_cands).ok();

            let k_norm_cands = [
                format!("model.layers.{i}.self_attn.k_norm.weight"),
                format!("layers.{i}.self_attn.k_norm.weight"),
            ];
            let k_norm_weight = find_and_load_vec(table, &k_norm_cands).ok();

            let spec = AttentionSpec {
                n_heads: inferred_n_heads,
                n_kv_heads: inferred_n_kv_heads,
                head_dim: inferred_head_dim,
                rope_theta: hf_cfg.as_ref().and_then(|c| c.rope_theta).unwrap_or(10000.0),
                sliding_window: None,
                q_norm: if q_norm_weight.is_some() { Some(NormKind::RmsNorm { eps: 1e-6 }) } else { None },
                k_norm: if k_norm_weight.is_some() { Some(NormKind::RmsNorm { eps: 1e-6 }) } else { None },
                attn_logit_soft_cap: attn_soft_cap,
            };

            let mixer_weights = BlockMixerWeights::Attention {
                spec,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight,
                k_norm_weight,
            };

            // Post-attention norm:
            // Gemma 2 applies post_attention_layernorm after attention residual.
            // Gemma 1 / LLaMA / Mistral use post_attention_layernorm as the pre-FFN norm!
            let post_attn_cands = [
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                format!("layers.{i}.post_attention_layernorm.weight"),
            ];
            let post_attention_norm_weight = if is_gemma_2 {
                find_and_load_vec(table, &post_attn_cands).ok()
            } else {
                None
            };
            let post_attention_norm = if post_attention_norm_weight.is_some() { Some(norm_kind) } else { None };

            // FFN norm
            let ffn_norm_cands = [
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                format!("layers.{i}.post_attention_layernorm.weight"),
                format!("layers.{i}.ffn_norm.weight"),
            ];
            let ffn_norm_weight = find_and_load_vec(table, &ffn_norm_cands)?;

            // FFN Channel Mixer
            let gate_cands = [
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                format!("layers.{i}.mlp.gate_proj.weight"),
                format!("layers.{i}.feed_forward.w1.weight"),
            ];
            let gate_proj = find_and_load_weight(table, &gate_cands)?;
            max_intermediate_dim = max_intermediate_dim.max(gate_proj.rows);

            let up_cands = [
                format!("model.layers.{i}.mlp.up_proj.weight"),
                format!("layers.{i}.mlp.up_proj.weight"),
                format!("layers.{i}.feed_forward.w3.weight"),
            ];
            let up_proj = find_and_load_weight(table, &up_cands)?;

            let down_cands = [
                format!("model.layers.{i}.mlp.down_proj.weight"),
                format!("layers.{i}.mlp.down_proj.weight"),
                format!("layers.{i}.feed_forward.w2.weight"),
            ];
            let down_proj = find_and_load_weight(table, &down_cands)?;

            let channel_mixer_weights = if is_gemma {
                BlockChannelMixerWeights::GeGLU {
                    gate_proj,
                    up_proj,
                    down_proj,
                }
            } else {
                BlockChannelMixerWeights::SwiGLU {
                    gate_proj,
                    up_proj,
                    down_proj,
                }
            };

            // Post-FFN norm (Gemma 2)
            let post_ffn_cands = [
                format!("model.layers.{i}.post_feedforward_layernorm.weight"),
                format!("layers.{i}.post_feedforward_layernorm.weight"),
            ];
            let post_ffn_norm_weight = if is_gemma_2 {
                find_and_load_vec(table, &post_ffn_cands).ok()
            } else {
                None
            };
            let post_ffn_norm = if post_ffn_norm_weight.is_some() { Some(norm_kind) } else { None };

            layers.push(BlockGraphLayer {
                pre_norm: norm_kind,
                pre_norm_weight,
                mixer_weights,
                post_attention_norm,
                post_attention_norm_weight,
                ffn_norm: norm_kind,
                ffn_norm_weight,
                channel_mixer_weights,
                post_ffn_norm,
                post_ffn_norm_weight,
            });
        }

        // Final norm
        let final_cands = [
            "model.norm.weight",
            "norm.weight",
        ];
        let final_norm_weight = find_and_load_vec(table, &final_cands)?;

        // LM head (optional; tied if absent)
        let lm_cands = ["lm_head.weight"];
        let lm_head = find_and_load_weight(table, &lm_cands).ok();

        let seq_len = if max_seq_len > 0 { max_seq_len } else { 2048 };

        let kv_caches = (0..n_layers)
            .map(|_| BlockKvCache::new(inferred_n_kv_heads, seq_len, inferred_head_dim))
            .collect();

        let config = ModelConfig {
            dim,
            n_layers,
            n_heads: inferred_n_heads,
            n_kv_heads: inferred_n_kv_heads,
            head_dim: inferred_head_dim,
            hidden_dim: max_intermediate_dim,
            vocab_size,
            norm_eps,
            rope_freq_base: hf_cfg.as_ref().and_then(|c| c.rope_theta).unwrap_or(10000.0),
            max_seq_len: seq_len,
            sliding_window: 0,
        };

        let graph_config = BlockGraphConfig {
            dim,
            vocab_size,
            max_seq_len: seq_len,
            blocks: Vec::new(),
            embed_scale,
            final_norm: norm_kind,
            final_logit_soft_cap: final_soft_cap,
            tied_embeddings: lm_head.is_none(),
        };

        #[cfg(target_os = "macos")]
        let metal_engine = crate::infer::metal::MetalEngine::new();

        let mut model = Self {
            config,
            graph_config,
            embed_tokens,
            layers,
            final_norm_weight,
            lm_head,
            kv_caches,
            logits: vec![0.0f32; vocab_size],
            buf_x: vec![0.0f32; dim],
            buf_norm: vec![0.0f32; dim],
            buf_res: vec![0.0f32; dim],
            buf_attn_out: vec![0.0f32; (inferred_n_heads * inferred_head_dim).max(dim)],
            buf_q: vec![0.0f32; inferred_n_heads * inferred_head_dim],
            buf_k: vec![0.0f32; inferred_n_kv_heads * inferred_head_dim],
            buf_v: vec![0.0f32; inferred_n_kv_heads * inferred_head_dim],
            buf_gate: vec![0.0f32; max_intermediate_dim],
            buf_up: vec![0.0f32; max_intermediate_dim],
            buf_ffn_out: vec![0.0f32; dim],
            buf_scores: vec![0.0f32; seq_len],
            buf_tmp: vec![0.0f32; dim.max(inferred_head_dim)],
            gpu_layers: n_layers,
            #[cfg(target_os = "macos")]
            metal_engine,
        };

        #[cfg(target_os = "macos")]
        if model.metal_engine.is_some() {
            model.set_gpu_layers(n_layers);
        }

        Ok(model)
    }

    /// Configure how many layers are offloaded to GPU (0 = all CPU, n_layers = all GPU).
    pub fn set_gpu_layers(&mut self, n: usize) {
        self.gpu_layers = n.min(self.layers.len());
        #[cfg(target_os = "macos")]
        if let Some(ref engine) = self.metal_engine {
            for (idx, layer) in self.layers.iter_mut().enumerate() {
                if idx < self.gpu_layers {
                    match &mut layer.mixer_weights {
                        BlockMixerWeights::Attention { q_proj, k_proj, v_proj, o_proj, .. } => {
                            if q_proj.metal_buf.is_none() {
                                *q_proj = q_proj.clone().with_metal(engine);
                                *k_proj = k_proj.clone().with_metal(engine);
                                *v_proj = v_proj.clone().with_metal(engine);
                                *o_proj = o_proj.clone().with_metal(engine);
                            }
                        }
                        _ => {}
                    }
                    match &mut layer.channel_mixer_weights {
                        BlockChannelMixerWeights::SwiGLU { gate_proj, up_proj, down_proj }
                        | BlockChannelMixerWeights::GeGLU { gate_proj, up_proj, down_proj } => {
                            if gate_proj.metal_buf.is_none() {
                                *gate_proj = gate_proj.clone().with_metal(engine);
                                *up_proj = up_proj.clone().with_metal(engine);
                                *down_proj = down_proj.clone().with_metal(engine);
                            }
                        }
                        _ => {}
                    }
                } else {
                    match &mut layer.mixer_weights {
                        BlockMixerWeights::Attention { q_proj, k_proj, v_proj, o_proj, .. } => {
                            q_proj.metal_buf = None;
                            k_proj.metal_buf = None;
                            v_proj.metal_buf = None;
                            o_proj.metal_buf = None;
                        }
                        _ => {}
                    }
                    match &mut layer.channel_mixer_weights {
                        BlockChannelMixerWeights::SwiGLU { gate_proj, up_proj, down_proj }
                        | BlockChannelMixerWeights::GeGLU { gate_proj, up_proj, down_proj } => {
                            gate_proj.metal_buf = None;
                            up_proj.metal_buf = None;
                            down_proj.metal_buf = None;
                        }
                        _ => {}
                    }
                }
            }

            if self.gpu_layers > 0 {
                if let Some(ref mut lm) = self.lm_head {
                    if lm.metal_buf.is_none() {
                        *lm = lm.clone().with_metal(engine);
                    }
                } else {
                    if self.embed_tokens.metal_buf.is_none() {
                        self.embed_tokens = self.embed_tokens.clone().with_metal(engine);
                    }
                }
            } else {
                if let Some(ref mut lm) = self.lm_head {
                    lm.metal_buf = None;
                }
                self.embed_tokens.metal_buf = None;
            }
        }
    }

    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.k.fill(0.0);
            cache.v.fill(0.0);
        }
    }

    pub fn compact_kv_cache(&mut self, cur_pos: usize, sink: usize, drop: usize) -> usize {
        if cur_pos <= sink + drop {
            return cur_pos;
        }
        let keep = cur_pos - drop;
        let new_len = keep;
        let head_dim = self.config.head_dim;
        let max_seq_len = self.config.max_seq_len;
        for cache in &mut self.kv_caches {
            for kv_h in 0..cache.n_kv_heads {
                let base = kv_h * max_seq_len * head_dim;
                for p in sink..new_len {
                    let old_p = p + drop;
                    let dst_off = base + p * head_dim;
                    let src_off = base + old_p * head_dim;
                    cache.k.copy_within(src_off..src_off + head_dim, dst_off);
                    cache.v.copy_within(src_off..src_off + head_dim, dst_off);
                }
            }
        }
        new_len
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let dim = self.config.dim;

        // 1. Embedding lookup
        self.embed_tokens.get_row(token as usize, &mut self.buf_x[..dim]);
        if let Some(scale) = self.graph_config.embed_scale {
            for v in &mut self.buf_x[..dim] {
                *v *= scale;
            }
        }

        // 2. Transformer layers
        self.execute_layers(pos, 0, self.layers.len());

        // 3. Final norm
        self.graph_config.final_norm.apply(&self.buf_x[..dim], &self.final_norm_weight, &mut self.buf_norm[..dim]);

        // 4. LM Head projection
        if let Some(lm_head) = &self.lm_head {
            lm_head.gemv(&self.buf_norm[..dim], &mut self.logits);
        } else {
            self.embed_tokens.gemv(&self.buf_norm[..dim], &mut self.logits);
        }

        // 5. Final logit soft-capping
        if let Some(cap) = self.graph_config.final_logit_soft_cap {
            soft_cap_in_place(&mut self.logits, cap);
        }

        &self.logits
    }

    /// Internal helper executing transformer blocks in range [start_layer, end_layer).
    pub fn execute_layers(&mut self, pos: usize, start_layer: usize, end_layer: usize) {
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let freq_base = self.config.rope_freq_base;

        let end = end_layer.min(self.layers.len());
        for l in start_layer..end {
            let layer = &self.layers[l];
            let is_gpu = l < self.gpu_layers;
            #[cfg(target_os = "macos")]
            let m_eng = if is_gpu { self.metal_engine.as_ref() } else { None };

            self.buf_res[..dim].copy_from_slice(&self.buf_x[..dim]);
            layer.pre_norm.apply(&self.buf_x[..dim], &layer.pre_norm_weight, &mut self.buf_norm[..dim]);

            match &layer.mixer_weights {
                BlockMixerWeights::Attention { spec, q_proj, k_proj, v_proj, o_proj, q_norm_weight, k_norm_weight } => {
                    // Batched Q, K, V Projections
                    #[cfg(target_os = "macos")]
                    let batched_qkv_done = if q_proj.mlx4.is_none() && k_proj.mlx4.is_none() && v_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref q_b), Some(ref k_b), Some(ref v_b)) =
                            (m_eng, &q_proj.metal_buf, &k_proj.metal_buf, &v_proj.metal_buf)
                        {
                            let dtype_code = match q_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_qkv_multi(
                                q_b, 0, q_proj.rows,
                                k_b, 0, k_proj.rows,
                                v_b, 0, v_proj.rows,
                                &self.buf_norm[..dim],
                                &mut self.buf_q[..n_heads * head_dim],
                                &mut self.buf_k[..n_kv_heads * head_dim],
                                &mut self.buf_v[..n_kv_heads * head_dim],
                                dim,
                                dtype_code,
                            );
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    #[cfg(not(target_os = "macos"))]
                    let batched_qkv_done = false;

                    if !batched_qkv_done {
                        q_proj.gemv(&self.buf_norm[..dim], &mut self.buf_q[..q_proj.rows]);
                        k_proj.gemv(&self.buf_norm[..dim], &mut self.buf_k[..k_proj.rows]);
                        v_proj.gemv(&self.buf_norm[..dim], &mut self.buf_v[..v_proj.rows]);
                    }

                    // Zero-allocation Q/K LayerNorms
                    if let (Some(q_nw), Some(qn)) = (q_norm_weight, spec.q_norm) {
                        for h in 0..n_heads {
                            let off = h * head_dim;
                            qn.apply(&self.buf_q[off..off + head_dim], q_nw, &mut self.buf_tmp[..head_dim]);
                            self.buf_q[off..off + head_dim].copy_from_slice(&self.buf_tmp[..head_dim]);
                        }
                    }
                    if let (Some(k_nw), Some(kn)) = (k_norm_weight, spec.k_norm) {
                        for h in 0..n_kv_heads {
                            let off = h * head_dim;
                            kn.apply(&self.buf_k[off..off + head_dim], k_nw, &mut self.buf_tmp[..head_dim]);
                            self.buf_k[off..off + head_dim].copy_from_slice(&self.buf_tmp[..head_dim]);
                        }
                    }

                    // Rotary Position Embedding on CPU
                    rope(&mut self.buf_q[..n_heads * head_dim], n_heads, head_dim, pos, freq_base);
                    rope(&mut self.buf_k[..n_kv_heads * head_dim], n_kv_heads, head_dim, pos, freq_base);

                    // Store K and V in Host RAM KV cache
                    let kv_cache = &mut self.kv_caches[l];
                    for kv_h in 0..n_kv_heads {
                        let off = kv_h * head_dim;
                        kv_cache.store(kv_h, pos, &self.buf_k[off..off + head_dim], &self.buf_v[off..off + head_dim]);
                    }

                    // Attention calculation using preallocated buf_scores (zero heap allocations)
                    let inv_sqrt_head_dim = 1.0f32 / (head_dim as f32).sqrt();
                    let group_size = (n_heads / n_kv_heads).max(1);

                    for h in 0..n_heads {
                        let kv_h = h / group_size;
                        let q_h = &self.buf_q[h * head_dim..(h + 1) * head_dim];

                        let mut max_score = -f32::INFINITY;
                        for t in 0..=pos {
                            let k_t = kv_cache.get_k(kv_h, t);
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_h[d] * k_t[d];
                            }
                            let mut score = dot * inv_sqrt_head_dim;
                            if let Some(cap) = spec.attn_logit_soft_cap {
                                score = cap * (score / cap).tanh();
                            }
                            if score > max_score {
                                max_score = score;
                            }
                            self.buf_scores[t] = score;
                        }

                        let mut exp_sum = 0.0f32;
                        for t in 0..=pos {
                            let s = (self.buf_scores[t] - max_score).exp();
                            self.buf_scores[t] = s;
                            exp_sum += s;
                        }
                        let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };

                        let h_out = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
                        h_out.fill(0.0);
                        for t in 0..=pos {
                            let w = self.buf_scores[t] * inv_sum;
                            let v_t = kv_cache.get_v(kv_h, t);
                            for d in 0..head_dim {
                                h_out[d] += w * v_t[d];
                            }
                        }
                    }

                    // Asynchronous GPU Attention Output Projection
                    #[cfg(target_os = "macos")]
                    let async_o_done = if o_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref o_b)) = (m_eng, &o_proj.metal_buf) {
                            let dtype_code = match o_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_async(
                                o_b, 0,
                                &self.buf_attn_out[..o_proj.cols],
                                &mut self.buf_x[..dim],
                                o_proj.rows, o_proj.cols,
                                dtype_code,
                            );
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    #[cfg(not(target_os = "macos"))]
                    let async_o_done = false;

                    if !async_o_done {
                        o_proj.gemv(&self.buf_attn_out[..o_proj.cols], &mut self.buf_x[..dim]);
                    } else {
                        #[cfg(target_os = "macos")]
                        if let Some(engine) = m_eng {
                            engine.sync();
                        }
                    }
                }
                BlockMixerWeights::Conv1d { .. } => {
                    self.buf_x[..dim].copy_from_slice(&self.buf_norm[..dim]);
                }
            }

            // Post-attention norm (Gemma) with preallocated buf_tmp
            if let (Some(pan), Some(pan_w)) = (layer.post_attention_norm, &layer.post_attention_norm_weight) {
                pan.apply(&self.buf_x[..dim], pan_w, &mut self.buf_tmp[..dim]);
                self.buf_x[..dim].copy_from_slice(&self.buf_tmp[..dim]);
            }

            // Residual add
            for i in 0..dim {
                self.buf_x[i] += self.buf_res[i];
            }

            // Save residual for FFN
            self.buf_res[..dim].copy_from_slice(&self.buf_x[..dim]);

            // FFN Pre-norm
            layer.ffn_norm.apply(&self.buf_x[..dim], &layer.ffn_norm_weight, &mut self.buf_norm[..dim]);

            // Channel mixer with Batched Gate/Up & Async Down projection
            match &layer.channel_mixer_weights {
                BlockChannelMixerWeights::SwiGLU { gate_proj, up_proj, down_proj } => {
                    #[cfg(target_os = "macos")]
                    let batched_gate_up = if gate_proj.mlx4.is_none() && up_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref g_b), Some(ref u_b)) =
                            (m_eng, &gate_proj.metal_buf, &up_proj.metal_buf)
                        {
                            let dtype_code = match gate_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_gate_up_multi(
                                g_b, 0,
                                u_b, 0,
                                &self.buf_norm[..dim],
                                &mut self.buf_gate[..gate_proj.rows],
                                &mut self.buf_up[..up_proj.rows],
                                gate_proj.rows,
                                dim,
                                dtype_code,
                            );
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    #[cfg(not(target_os = "macos"))]
                    let batched_gate_up = false;

                    if !batched_gate_up {
                        gate_proj.gemv(&self.buf_norm[..dim], &mut self.buf_gate[..gate_proj.rows]);
                        up_proj.gemv(&self.buf_norm[..dim], &mut self.buf_up[..up_proj.rows]);
                    }

                    swiglu_in_place(&mut self.buf_gate[..gate_proj.rows], &self.buf_up[..up_proj.rows]);

                    #[cfg(target_os = "macos")]
                    let async_down = if down_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref d_b)) = (m_eng, &down_proj.metal_buf) {
                            let dtype_code = match down_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_async(
                                d_b, 0,
                                &self.buf_gate[..down_proj.cols],
                                &mut self.buf_ffn_out[..dim],
                                down_proj.rows, down_proj.cols,
                                dtype_code,
                            );
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    #[cfg(not(target_os = "macos"))]
                    let async_down = false;

                    if !async_down {
                        down_proj.gemv(&self.buf_gate[..down_proj.cols], &mut self.buf_ffn_out[..dim]);
                    } else {
                        #[cfg(target_os = "macos")]
                        if let Some(engine) = m_eng {
                            engine.sync();
                        }
                    }
                }
                BlockChannelMixerWeights::GeGLU { gate_proj, up_proj, down_proj } => {
                    gate_proj.gemv(&self.buf_norm[..dim], &mut self.buf_gate[..gate_proj.rows]);
                    up_proj.gemv(&self.buf_norm[..dim], &mut self.buf_up[..up_proj.rows]);
                    geglu_in_place(&mut self.buf_gate[..gate_proj.rows], &self.buf_up[..up_proj.rows]);
                    down_proj.gemv(&self.buf_gate[..gate_proj.rows], &mut self.buf_ffn_out[..dim]);
                }
                BlockChannelMixerWeights::SparseMoE { router, experts_gate, experts_up, experts_down, .. } => {
                    let mut router_logits = vec![0.0f32; router.rows];
                    router.gemv(&self.buf_norm[..dim], &mut router_logits);
                    self.buf_ffn_out[..dim].fill(0.0);
                    let best_expert = router_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(idx, _)| idx).unwrap_or(0);
                    if best_expert < experts_gate.len() {
                        experts_gate[best_expert].gemv(&self.buf_norm[..dim], &mut self.buf_gate[..experts_gate[best_expert].rows]);
                        experts_up[best_expert].gemv(&self.buf_norm[..dim], &mut self.buf_up[..experts_up[best_expert].rows]);
                        swiglu_in_place(&mut self.buf_gate[..experts_gate[best_expert].rows], &self.buf_up[..experts_up[best_expert].rows]);
                        experts_down[best_expert].gemv(&self.buf_gate[..experts_gate[best_expert].rows], &mut self.buf_ffn_out[..dim]);
                    }
                }
            }

            // Post-FFN norm (Gemma) with preallocated buf_tmp
            if let (Some(pfn), Some(pfn_w)) = (layer.post_ffn_norm, &layer.post_ffn_norm_weight) {
                pfn.apply(&self.buf_ffn_out[..dim], pfn_w, &mut self.buf_tmp[..dim]);
                self.buf_ffn_out[..dim].copy_from_slice(&self.buf_tmp[..dim]);
            }

            // Residual add
            for i in 0..dim {
                self.buf_x[i] = self.buf_res[i] + self.buf_ffn_out[i];
            }
        }
    }

    /// Forward pass through a subset of layers [start_layer, end_layer).
    /// Used for Sequential Layer-Wise Activation Pipelining across distributed edge swarm nodes.
    pub fn forward_layer_range(
        &mut self,
        pos: usize,
        start_layer: usize,
        end_layer: usize,
        in_hidden: &[f32],
        out_hidden: &mut [f32],
    ) {
        let dim = self.config.dim;
        self.buf_x[..dim].copy_from_slice(&in_hidden[..dim]);
        self.execute_layers(pos, start_layer, end_layer);
        out_hidden[..dim].copy_from_slice(&self.buf_x[..dim]);
    }

    pub fn forward_prefill(&mut self, token: u32, pos: usize) {
        self.forward(token, pos);
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], start_pos: usize) {
        for (i, &tok) in tokens.iter().enumerate() {
            self.forward_prefill(tok, start_pos + i);
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        for (i, &tok) in tokens.iter().enumerate() {
            self.forward(tok, start_pos + i);
        }
        &self.logits
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
