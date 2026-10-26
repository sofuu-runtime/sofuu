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
use super::lfm2::{SendMutPtr, SendPtr};
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

/// DeepSeek Multi-Head Latent Attention (MLA) specification
#[derive(Debug, Clone)]
pub struct MlaSpec {
    pub n_heads: usize,
    pub kv_lora_rank: usize,      // Compressed latent dim d_c (e.g. 512)
    pub qk_nope_head_dim: usize,  // Unrotated query/key head dim d_nope (e.g. 128)
    pub qk_rope_head_dim: usize,  // Decoupled RoPE head dim d_R (e.g. 64)
    pub v_head_dim: usize,        // Value head dim d_v (e.g. 128)
    pub q_lora_rank: Option<usize>, // Low-rank query compression (e.g. 1536)
    pub rope_theta: f32,
    pub softmax_scale: f32,
}

/// Sequence Mixer kind: Attention, Short Conv, or DeepSeek Latent Attention
#[derive(Debug, Clone)]
pub enum MixerSpec {
    Attention(AttentionSpec),
    Conv1d(Conv1dSpec),
    LatentAttention(MlaSpec),
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
        shared_intermediate_dim: Option<usize>,
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
// Metal items in this file: macOS + iOS — the GPU engine compiles for
// both now (see the header of infer/metal.rs).
#[derive(Clone)]
pub struct Mlx4Weights {
    pub scales: UnifiedTensorData,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub scales_metal_buf: Option<Arc<crate::infer::metal::MetalBuffer>>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub scales_metal_offset: usize,
    pub biases: UnifiedTensorData,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub biases_metal_buf: Option<Arc<crate::infer::metal::MetalBuffer>>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub biases_metal_offset: usize,
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
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub metal_buf: Option<Arc<crate::infer::metal::MetalBuffer>>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub metal_offset: usize,
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
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_buf: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_offset: 0,
            mlx4: None,
        }
    }

    pub fn new_from_data(rows: usize, cols: usize, dtype: TensorDType, data: UnifiedTensorData) -> Self {
        Self {
            rows,
            cols,
            dtype,
            data,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_buf: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_offset: 0,
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
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_buf: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_offset: 0,
            mlx4: Some(Mlx4Weights {
                scales,
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                scales_metal_buf: None,
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                scales_metal_offset: 0,
                biases,
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                biases_metal_buf: None,
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                biases_metal_offset: 0,
                group_size,
                is_bf16,
            }),
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn with_metal(mut self, engine: &crate::infer::metal::MetalEngine) -> Self {
        match &self.data {
            UnifiedTensorData::Mmap { mmap, offset, .. } => {
                self.metal_buf = mmap.get_or_create_metal_buffer(engine);
                self.metal_offset = *offset;
            }
            UnifiedTensorData::Heap(vec) => {
                self.metal_buf = engine.create_buffer(vec.as_slice()).map(Arc::new);
                self.metal_offset = 0;
            }
        }
        if let Some(ref mut mlx4) = self.mlx4 {
            match &mlx4.scales {
                UnifiedTensorData::Mmap { mmap, offset, .. } => {
                    mlx4.scales_metal_buf = mmap.get_or_create_metal_buffer(engine);
                    mlx4.scales_metal_offset = *offset;
                }
                UnifiedTensorData::Heap(vec) => {
                    mlx4.scales_metal_buf = engine.create_buffer(vec.as_slice()).map(Arc::new);
                    mlx4.scales_metal_offset = 0;
                }
            }
            match &mlx4.biases {
                UnifiedTensorData::Mmap { mmap, offset, .. } => {
                    mlx4.biases_metal_buf = mmap.get_or_create_metal_buffer(engine);
                    mlx4.biases_metal_offset = *offset;
                }
                UnifiedTensorData::Heap(vec) => {
                    mlx4.biases_metal_buf = engine.create_buffer(vec.as_slice()).map(Arc::new);
                    mlx4.biases_metal_offset = 0;
                }
            }
        }
        self
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn clear_metal(&mut self) {
        self.metal_buf = None;
        self.metal_offset = 0;
        if let Some(ref mut mlx4) = self.mlx4 {
            mlx4.scales_metal_buf = None;
            mlx4.scales_metal_offset = 0;
            mlx4.biases_metal_buf = None;
            mlx4.biases_metal_offset = 0;
        }
    }

/// Batched prefill GEMM: ys[t * rows + r] = sum_c xs[t * cols + c] * W[r][c].
/// Routes to the shared C++ batch kernels (the same `sofuu_gemm_*` the GGUF
/// backend's fast prefill uses); dtypes without a batch kernel fall back to
/// the per-token `gemv` loop. Elementwise semantics match `gemv`; the batch
/// kernels may reorder the f32 accumulation.
pub fn gemm_batched(&self, xs: &[f32], n_tok: usize, ys: &mut [f32]) {
    let (rows, cols) = (self.rows, self.cols);
    debug_assert_eq!(xs.len(), n_tok * cols);
    debug_assert_eq!(ys.len(), n_tok * rows);
    if n_tok == 0 || rows == 0 || cols == 0 {
        return;
    }
    if self.mlx4.is_none() {
        let sl = self.data.as_slice();
        match self.dtype {
            TensorDType::F32 if sl.len() >= rows * cols * 4 => {
                if (sl.as_ptr() as usize) % 4 == 0 {
                    crate::infer::quant::gemm_batched_f32(sl, xs, ys, rows, cols, n_tok);
                    return;
                }
            }
            TensorDType::F16 if sl.len() >= rows * cols * 2 => {
                crate::infer::quant::gemm_batched_f16(sl, xs, ys, rows, cols, n_tok);
                return;
            }
            _ => {}
        }
    }
    if let Some(ref mlx4) = self.mlx4 {
        // Mlx4-packed weights: dequantize one 32-row panel at a time (stays
        // L2-resident) and run the panel as a cblas sgemm against every
        // token of the chunk. Each weight is unpacked once per call, and the
        // call covers the whole chunk, so a prefill touches every weight
        // O(prompt/512) times instead of O(prompt/token).
        Self::gemm_batched_mlx4(self, xs, n_tok, ys);
        return;
    }
    for t in 0..n_tok {
        self.gemv(&xs[t * cols..(t + 1) * cols], &mut ys[t * rows..(t + 1) * rows]);
    }
}

/// Panel-streamed Mlx4 batched GEMM: 32-row panels dequantized to f32 and
/// multiplied against the whole activation chunk on the cblas/AMX ladder.
fn gemm_batched_mlx4(wm: &WeightMatrix, xs: &[f32], n_tok: usize, ys: &mut [f32]) {
    let (rows, cols) = (wm.rows, wm.cols);
    let mlx4 = wm.mlx4.as_ref().expect("mlx4 variant");
    let w_sl = wm.data.as_slice();
    let s_sl = mlx4.scales.as_slice();
    let b_sl = mlx4.biases.as_slice();
    let gs = mlx4.group_size;
    let is_bf16 = mlx4.is_bf16;
    const PANEL: usize = 32;
    let u32_per_row = cols / 8;
    let groups_per_row = cols / gs;
    for r0 in (0..rows).step_by(PANEL) {
        let pr = PANEL.min(rows - r0);
        let mut fpbuf = vec![0.0f32; PANEL * cols];
        for i in 0..pr {
            let row = r0 + i;
            let wp = &w_sl[row * u32_per_row * 4..(row + 1) * u32_per_row * 4];
            let sp = &s_sl[row * groups_per_row * 2..(row + 1) * groups_per_row * 2];
            let bp = &b_sl[row * groups_per_row * 2..(row + 1) * groups_per_row * 2];
            let out = &mut fpbuf[i * cols..(i + 1) * cols];
            let mut scale = 0.0f32;
            let mut bias = 0.0f32;
            let mut cur_grp = usize::MAX;
            for u in 0..u32_per_row {
                let packed = u32::from_le_bytes(wp[u * 4..u * 4 + 4].try_into().unwrap());
                let c_base = u * 8;
                let grp = c_base / gs;
                if grp != cur_grp {
                    cur_grp = grp;
                    let s16 = u16::from_le_bytes(sp[grp * 2..grp * 2 + 2].try_into().unwrap());
                    let b16 = u16::from_le_bytes(bp[grp * 2..grp * 2 + 2].try_into().unwrap());
                    scale = if is_bf16 {
                        crate::infer::quant::bf16_to_f32(s16)
                    } else {
                        crate::infer::quant::f16_to_f32(s16)
                    };
                    bias = if is_bf16 {
                        crate::infer::quant::bf16_to_f32(b16)
                    } else {
                        crate::infer::quant::f16_to_f32(b16)
                    };
                }
                for j in 0..8 {
                    let nib = ((packed >> (j * 4)) & 0x0f) as f32;
                    out[c_base + j] = nib * scale + bias;
                }
            }
        }
        // C[t][r0..r0+pr) = X[t][c] * W_panel[r][c], strided via ldc = rows
        #[cfg(target_os = "macos")]
        if crate::infer::lfm2::blas_enabled() {
            unsafe {
                crate::infer::lfm2::blas_gemm_panel(
                    xs,
                    &fpbuf,
                    ys,
                    n_tok,
                    r0,
                    rows,
                    cols,
                    pr,
                );
            }
            continue;
        }
        // non-macos / blas-off fallback: f32 rows, per-token dots
        for t in 0..n_tok {
            let xr = &xs[t * cols..(t + 1) * cols];
            for i in 0..pr {
                let wrow = &fpbuf[i * cols..(i + 1) * cols];
                let mut sum = 0.0f32;
                for c in 0..cols {
                    sum += wrow[c] * xr[c];
                }
                ys[t * rows + r0 + i] = sum;
            }
        }
    }
}

    /// Matrix-vector multiplication: y = W * x
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        // Decode GEMV: on M2 Pro (2026-10-15) the threaded CPU GEMV beats
        // per-matvec Metal dispatch (same lfm2 decode finding), so macOS
        // decodes on CPU by default; SOFUU_METAL_DECODE=1 restores the Metal
        // hybrid. iOS keeps Metal.
        #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(unused_variables))]
        let metal_decode_wanted = cfg!(target_os = "ios")
            || std::env::var_os("SOFUU_METAL_DECODE").is_some();
        if let Some(ref mlx4) = self.mlx4 {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if metal_decode_wanted {
                if let (Some(ref m_buf), Some(ref s_buf), Some(ref b_buf)) = (
                    &self.metal_buf,
                    &mlx4.scales_metal_buf,
                    &mlx4.biases_metal_buf,
                ) {
                    if let Some(engine) = crate::infer::metal::get_or_init_global_metal() {
                        engine.gemv_mlx4(
                            m_buf, self.metal_offset,
                            s_buf, mlx4.scales_metal_offset,
                            b_buf, mlx4.biases_metal_offset,
                            x, y,
                            self.rows, self.cols, mlx4.group_size, mlx4.is_bf16,
                        );
                        return;
                    }
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

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if metal_decode_wanted {
            if let Some(ref m_buf) = self.metal_buf {
                if let Some(engine) = crate::infer::metal::get_or_init_global_metal() {
                    match self.dtype {
                        TensorDType::BF16 => {
                            engine.gemv_bf16(m_buf, self.metal_offset, x, y, self.rows, self.cols);
                            return;
                        }
                        TensorDType::F16 => {
                            engine.gemv_f16(m_buf, self.metal_offset, x, y, self.rows, self.cols);
                            return;
                        }
                        TensorDType::F32 => {
                            engine.gemv_f32(m_buf, self.metal_offset, x, y, self.rows, self.cols);
                            return;
                        }
                        _ => {}
                    }
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
                if slice.len() >= self.rows * self.cols * 2 {
                    unsafe {
                        crate::infer::quant::sofuu_gemv_f16(slice.as_ptr(), x.as_ptr(), y.as_mut_ptr(), self.rows, self.cols);
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

/// KV Cache for a single composable block (supports standard MHA/GQA and DeepSeek MLA latent cache)
pub struct BlockKvCache {
    pub k: Vec<f32>, // [kv_heads, max_seq_len, head_dim]
    pub v: Vec<f32>, // [kv_heads, max_seq_len, head_dim]
    pub latent_kv: Option<Vec<f32>>, // [max_seq_len, latent_dim] where latent_dim = kv_lora_rank + qk_rope_head_dim
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub latent_dim: usize,
}

impl BlockKvCache {
    pub fn new(n_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let cap = n_kv_heads * max_seq_len * head_dim;
        Self {
            k: vec![0.0f32; cap],
            v: vec![0.0f32; cap],
            latent_kv: None,
            max_seq_len,
            head_dim,
            n_kv_heads,
            latent_dim: 0,
        }
    }

    pub fn new_latent(max_seq_len: usize, latent_dim: usize) -> Self {
        let cap = max_seq_len * latent_dim;
        Self {
            k: Vec::new(),
            v: Vec::new(),
            latent_kv: Some(vec![0.0f32; cap]),
            max_seq_len,
            head_dim: 0,
            n_kv_heads: 0,
            latent_dim,
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

    #[inline]
    pub fn store_latent(&mut self, pos: usize, latent_vec: &[f32]) {
        if let Some(ref mut l) = self.latent_kv {
            let offset = pos * self.latent_dim;
            let len = latent_vec.len().min(self.latent_dim);
            l[offset..offset + len].copy_from_slice(&latent_vec[..len]);
        }
    }

    #[inline]
    pub fn get_latent(&self, pos: usize) -> &[f32] {
        if let Some(ref l) = self.latent_kv {
            let offset = pos * self.latent_dim;
            &l[offset..offset + self.latent_dim]
        } else {
            &[]
        }
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
    LatentAttention {
        spec: MlaSpec,
        q_a_proj: Option<WeightMatrix>,
        q_a_norm: Option<NormKind>,
        q_a_norm_weight: Option<Vec<f32>>,
        q_b_proj: WeightMatrix,
        kv_a_proj: WeightMatrix,
        kv_a_norm: Option<NormKind>,
        kv_a_norm_weight: Option<Vec<f32>>,
        w_uk: WeightMatrix,
        w_uv: WeightMatrix,
        o_proj: WeightMatrix,
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
        shared_gate: Option<WeightMatrix>,
        shared_up: Option<WeightMatrix>,
        shared_down: Option<WeightMatrix>,
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
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    num_local_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    n_routed_experts: Option<usize>,
    #[serde(default)]
    n_shared_experts: Option<usize>,
    #[serde(default)]
    q_lora_rank: Option<usize>,
    #[serde(default)]
    kv_lora_rank: Option<usize>,
    #[serde(default)]
    qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    v_head_dim: Option<usize>,
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
    pub buf_router: Vec<f32>,
    pub gpu_layers: usize,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
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
        let mut max_router_rows = 0;

        for i in 0..n_layers {
            // Pre-norm
            let pre_norm_cands = [
                format!("model.layers.{i}.input_layernorm.weight"),
                format!("layers.{i}.input_layernorm.weight"),
                format!("layers.{i}.attention_norm.weight"),
            ];
            let pre_norm_weight = find_and_load_vec(table, &pre_norm_cands)?;

            // Sequence mixer: check for DeepSeek Multi-Head Latent Attention (MLA) or standard Attention
            let mla_cands = [
                format!("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
                format!("layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
                format!("model.layers.{i}.self_attn.w_uk.weight"),
            ];
            let is_mla = mla_cands.iter().any(|c| table.get_tensor_data(c).is_some());

            let mixer_weights = if is_mla {
                let kv_lora_rank = hf_cfg.as_ref().and_then(|c| c.kv_lora_rank).unwrap_or(512);
                let qk_nope_head_dim = hf_cfg.as_ref().and_then(|c| c.qk_nope_head_dim).unwrap_or(128);
                let qk_rope_head_dim = hf_cfg.as_ref().and_then(|c| c.qk_rope_head_dim).unwrap_or(64);
                let v_head_dim = hf_cfg.as_ref().and_then(|c| c.v_head_dim).unwrap_or(128);
                let n_heads = hf_cfg.as_ref().and_then(|c| c.num_attention_heads).unwrap_or(inferred_n_heads.max(16));
                let q_lora_rank = hf_cfg.as_ref().and_then(|c| c.q_lora_rank);
                let rope_theta = hf_cfg.as_ref().and_then(|c| c.rope_theta).unwrap_or(10000.0);
                let softmax_scale = 1.0f32 / ((qk_nope_head_dim + qk_rope_head_dim) as f32).sqrt();

                let spec = MlaSpec {
                    n_heads,
                    kv_lora_rank,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    v_head_dim,
                    q_lora_rank,
                    rope_theta,
                    softmax_scale,
                };

                let kv_a_cands = [
                    format!("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
                    format!("layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
                    format!("model.layers.{i}.self_attn.kv_a_proj.weight"),
                ];
                let kv_a_proj = find_and_load_weight(table, &kv_a_cands)?;

                let kv_a_norm_cands = [
                    format!("model.layers.{i}.self_attn.kv_a_layernorm.weight"),
                    format!("layers.{i}.self_attn.kv_a_layernorm.weight"),
                ];
                let kv_a_norm_weight = find_and_load_vec(table, &kv_a_norm_cands).ok();
                let kv_a_norm = if kv_a_norm_weight.is_some() { Some(norm_kind) } else { None };

                let w_uk_cands = [
                    format!("model.layers.{i}.self_attn.w_uk.weight"),
                    format!("layers.{i}.self_attn.w_uk.weight"),
                    format!("model.layers.{i}.self_attn.kv_b_proj.weight"),
                ];
                let w_uk = find_and_load_weight(table, &w_uk_cands)?;

                let w_uv_cands = [
                    format!("model.layers.{i}.self_attn.w_uv.weight"),
                    format!("layers.{i}.self_attn.w_uv.weight"),
                ];
                let w_uv = find_and_load_weight(table, &w_uv_cands).unwrap_or_else(|_| w_uk.clone());

                let q_a_cands = [
                    format!("model.layers.{i}.self_attn.q_a_proj.weight"),
                    format!("layers.{i}.self_attn.q_a_proj.weight"),
                ];
                let q_a_proj = find_and_load_weight(table, &q_a_cands).ok();

                let q_a_norm_cands = [
                    format!("model.layers.{i}.self_attn.q_a_layernorm.weight"),
                    format!("layers.{i}.self_attn.q_a_layernorm.weight"),
                ];
                let q_a_norm_weight = find_and_load_vec(table, &q_a_norm_cands).ok();
                let q_a_norm = if q_a_norm_weight.is_some() { Some(norm_kind) } else { None };

                let q_b_cands = [
                    format!("model.layers.{i}.self_attn.q_b_proj.weight"),
                    format!("layers.{i}.self_attn.q_b_proj.weight"),
                    format!("model.layers.{i}.self_attn.q_proj.weight"),
                ];
                let q_b_proj = find_and_load_weight(table, &q_b_cands)?;

                let o_cands = [
                    format!("model.layers.{i}.self_attn.o_proj.weight"),
                    format!("layers.{i}.self_attn.o_proj.weight"),
                    format!("layers.{i}.attention.wo.weight"),
                ];
                let o_proj = find_and_load_weight(table, &o_cands)?;

                BlockMixerWeights::LatentAttention {
                    spec,
                    q_a_proj,
                    q_a_norm,
                    q_a_norm_weight,
                    q_b_proj,
                    kv_a_proj,
                    kv_a_norm,
                    kv_a_norm_weight,
                    w_uk,
                    w_uv,
                    o_proj,
                }
            } else {
                // Standard Attention projections
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

                // Gemma 2 alternates: even layers (0, 2, 4...) have sliding window (4096),
                // odd layers (1, 3, 5...) have full global attention.
                // Mistral applies uniform sliding window across all layers.
                let layer_sliding_window = if is_gemma_2 {
                    if i % 2 == 0 {
                        Some(hf_cfg.as_ref().and_then(|c| c.sliding_window).unwrap_or(4096))
                    } else {
                        None
                    }
                } else {
                    hf_cfg.as_ref().and_then(|c| c.sliding_window)
                };

                let spec = AttentionSpec {
                    n_heads: inferred_n_heads,
                    n_kv_heads: inferred_n_kv_heads,
                    head_dim: inferred_head_dim,
                    rope_theta: hf_cfg.as_ref().and_then(|c| c.rope_theta).unwrap_or(10000.0),
                    sliding_window: layer_sliding_window,
                    q_norm: if q_norm_weight.is_some() { Some(NormKind::RmsNorm { eps: 1e-6 }) } else { None },
                    k_norm: if k_norm_weight.is_some() { Some(NormKind::RmsNorm { eps: 1e-6 }) } else { None },
                    attn_logit_soft_cap: attn_soft_cap,
                };

                BlockMixerWeights::Attention {
                    spec,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm_weight,
                    k_norm_weight,
                }
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

            // FFN Channel Mixer: Sparse MoE or dense SwiGLU / GeGLU
            let router_cands = [
                format!("model.layers.{i}.block_sparse_moe.gate.weight"),
                format!("layers.{i}.block_sparse_moe.gate.weight"),
                format!("model.layers.{i}.mlp.router.weight"),
                format!("layers.{i}.feed_forward.router.weight"),
            ];
            let is_moe = router_cands.iter().any(|c| table.get_tensor_data(c).is_some());

            let channel_mixer_weights = if is_moe {
                let router = find_and_load_weight(table, &router_cands)?;
                max_router_rows = max_router_rows.max(router.rows);
                let num_experts = hf_cfg.as_ref().and_then(|c| c.num_local_experts.or(c.n_routed_experts)).unwrap_or(router.rows);
                let top_k = hf_cfg.as_ref().and_then(|c| c.num_experts_per_tok).unwrap_or(2);
                let mut experts_gate = Vec::with_capacity(num_experts);
                let mut experts_up = Vec::with_capacity(num_experts);
                let mut experts_down = Vec::with_capacity(num_experts);

                for e in 0..num_experts {
                    let eg_cands = [
                        format!("model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight"),
                        format!("model.layers.{i}.mlp.experts.{e}.gate_proj.weight"),
                        format!("layers.{i}.block_sparse_moe.experts.{e}.gate_proj.weight"),
                        format!("layers.{i}.feed_forward.experts.{e}.w1.weight"),
                    ];
                    let eg = find_and_load_weight(table, &eg_cands)?;
                    max_intermediate_dim = max_intermediate_dim.max(eg.rows);

                    let eu_cands = [
                        format!("model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight"),
                        format!("model.layers.{i}.mlp.experts.{e}.up_proj.weight"),
                        format!("layers.{i}.block_sparse_moe.experts.{e}.up_proj.weight"),
                        format!("layers.{i}.feed_forward.experts.{e}.w3.weight"),
                    ];
                    let eu = find_and_load_weight(table, &eu_cands)?;

                    let ed_cands = [
                        format!("model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight"),
                        format!("model.layers.{i}.mlp.experts.{e}.down_proj.weight"),
                        format!("layers.{i}.block_sparse_moe.experts.{e}.down_proj.weight"),
                        format!("layers.{i}.feed_forward.experts.{e}.w2.weight"),
                    ];
                    let ed = find_and_load_weight(table, &ed_cands)?;

                    experts_gate.push(eg);
                    experts_up.push(eu);
                    experts_down.push(ed);
                }

                // Shared experts (Qwen-MoE / DeepSeek MoE)
                let shared_gate_cands = [
                    format!("model.layers.{i}.mlp.shared_expert.gate_proj.weight"),
                    format!("layers.{i}.block_sparse_moe.shared_experts.gate_proj.weight"),
                    format!("layers.{i}.feed_forward.shared_expert.w1.weight"),
                ];
                let shared_gate = find_and_load_weight(table, &shared_gate_cands).ok();

                let shared_up_cands = [
                    format!("model.layers.{i}.mlp.shared_expert.up_proj.weight"),
                    format!("layers.{i}.block_sparse_moe.shared_experts.up_proj.weight"),
                    format!("layers.{i}.feed_forward.shared_expert.w3.weight"),
                ];
                let shared_up = find_and_load_weight(table, &shared_up_cands).ok();

                let shared_down_cands = [
                    format!("model.layers.{i}.mlp.shared_expert.down_proj.weight"),
                    format!("layers.{i}.block_sparse_moe.shared_experts.down_proj.weight"),
                    format!("layers.{i}.feed_forward.shared_expert.w2.weight"),
                ];
                let shared_down = find_and_load_weight(table, &shared_down_cands).ok();

                if let Some(ref sg) = shared_gate {
                    max_intermediate_dim = max_intermediate_dim.max(sg.rows);
                }

                BlockChannelMixerWeights::SparseMoE {
                    num_experts,
                    top_k,
                    router,
                    experts_gate,
                    experts_up,
                    experts_down,
                    shared_gate,
                    shared_up,
                    shared_down,
                }
            } else {
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

                if is_gemma {
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

        let kv_caches = layers
            .iter()
            .map(|l| match &l.mixer_weights {
                BlockMixerWeights::LatentAttention { spec, .. } => {
                    BlockKvCache::new_latent(seq_len, spec.kv_lora_rank + spec.qk_rope_head_dim)
                }
                _ => BlockKvCache::new(inferred_n_kv_heads, seq_len, inferred_head_dim),
            })
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
            logical_max_seq_len: seq_len,
            kv_ring: 0,
            sliding_window: 0,
            rope_interleaved: false,
            rope_n_rot: 0,
            rope_mscale: 1.0,
            rope_custom: false,
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

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let metal_engine = crate::infer::metal::MetalEngine::new();

        // Calculate maximum dimensions across all layer mixers (handling MLA and Attention)
        let mut max_q_dim = inferred_n_heads * inferred_head_dim;
        let mut max_k_dim = inferred_n_kv_heads * inferred_head_dim;
        let mut max_v_dim = inferred_n_kv_heads * inferred_head_dim;
        let mut max_attn_out_dim = (inferred_n_heads * inferred_head_dim).max(dim);

        for l in &layers {
            if let BlockMixerWeights::LatentAttention { spec, .. } = &l.mixer_weights {
                max_q_dim = max_q_dim.max(spec.n_heads * (spec.qk_nope_head_dim + spec.qk_rope_head_dim));
                max_k_dim = max_k_dim.max(spec.kv_lora_rank + spec.qk_rope_head_dim);
                max_v_dim = max_v_dim.max(spec.kv_lora_rank + spec.qk_rope_head_dim);
                max_attn_out_dim = max_attn_out_dim.max(spec.n_heads * spec.v_head_dim);
            }
        }

        let model = Self {
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
            buf_attn_out: vec![0.0f32; max_attn_out_dim],
            buf_q: vec![0.0f32; max_q_dim],
            buf_k: vec![0.0f32; max_k_dim],
            buf_v: vec![0.0f32; max_v_dim],
            buf_gate: vec![0.0f32; max_intermediate_dim.max(dim)],
            buf_up: vec![0.0f32; max_intermediate_dim.max(dim)],
            buf_ffn_out: vec![0.0f32; dim],
            buf_scores: vec![0.0f32; seq_len],
            buf_tmp: vec![0.0f32; dim.max(inferred_head_dim).max(max_intermediate_dim).max(2048)],
            buf_router: vec![0.0f32; max_router_rows.max(64)],
            gpu_layers: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_engine,
        };

        Ok(model)
    }

    /// Configure how many layers are offloaded to GPU (0 = all CPU, n_layers = all GPU).
    pub fn set_gpu_layers(&mut self, n: usize) {
        self.gpu_layers = n.min(self.layers.len());
        #[cfg(any(target_os = "macos", target_os = "ios"))]
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
                        BlockMixerWeights::LatentAttention { q_a_proj, q_b_proj, kv_a_proj, w_uk, w_uv, o_proj, .. } => {
                            if let Some(qa) = q_a_proj {
                                if qa.metal_buf.is_none() {
                                    *qa = qa.clone().with_metal(engine);
                                }
                            }
                            if q_b_proj.metal_buf.is_none() {
                                *q_b_proj = q_b_proj.clone().with_metal(engine);
                                *kv_a_proj = kv_a_proj.clone().with_metal(engine);
                                *w_uk = w_uk.clone().with_metal(engine);
                                *w_uv = w_uv.clone().with_metal(engine);
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
                        BlockChannelMixerWeights::SparseMoE { router, experts_gate, experts_up, experts_down, shared_gate, shared_up, shared_down, .. } => {
                            if router.metal_buf.is_none() {
                                *router = router.clone().with_metal(engine);
                                for g in experts_gate.iter_mut() { *g = g.clone().with_metal(engine); }
                                for u in experts_up.iter_mut() { *u = u.clone().with_metal(engine); }
                                for d in experts_down.iter_mut() { *d = d.clone().with_metal(engine); }
                                if let Some(sg) = shared_gate { *sg = sg.clone().with_metal(engine); }
                                if let Some(su) = shared_up { *su = su.clone().with_metal(engine); }
                                if let Some(sd) = shared_down { *sd = sd.clone().with_metal(engine); }
                            }
                        }
                    }
                } else {
                    match &mut layer.mixer_weights {
                        BlockMixerWeights::Attention { q_proj, k_proj, v_proj, o_proj, .. } => {
                            q_proj.clear_metal();
                            k_proj.clear_metal();
                            v_proj.clear_metal();
                            o_proj.clear_metal();
                        }
                        BlockMixerWeights::LatentAttention { q_a_proj, q_b_proj, kv_a_proj, w_uk, w_uv, o_proj, .. } => {
                            if let Some(qa) = q_a_proj { qa.clear_metal(); }
                            q_b_proj.clear_metal();
                            kv_a_proj.clear_metal();
                            w_uk.clear_metal();
                            w_uv.clear_metal();
                            o_proj.clear_metal();
                        }
                        _ => {}
                    }
                    match &mut layer.channel_mixer_weights {
                        BlockChannelMixerWeights::SwiGLU { gate_proj, up_proj, down_proj }
                        | BlockChannelMixerWeights::GeGLU { gate_proj, up_proj, down_proj } => {
                            gate_proj.clear_metal();
                            up_proj.clear_metal();
                            down_proj.clear_metal();
                        }
                        BlockChannelMixerWeights::SparseMoE { router, experts_gate, experts_up, experts_down, shared_gate, shared_up, shared_down, .. } => {
                            router.clear_metal();
                            for g in experts_gate.iter_mut() { g.clear_metal(); }
                            for u in experts_up.iter_mut() { u.clear_metal(); }
                            for d in experts_down.iter_mut() { d.clear_metal(); }
                            if let Some(sg) = shared_gate { sg.clear_metal(); }
                            if let Some(su) = shared_up { su.clear_metal(); }
                            if let Some(sd) = shared_down { sd.clear_metal(); }
                        }
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
                    lm.clear_metal();
                }
                self.embed_tokens.clear_metal();
            }

        }
    }

    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.k.fill(0.0);
            cache.v.fill(0.0);
            if let Some(ref mut l) = cache.latent_kv {
                l.fill(0.0);
            }
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
            if cache.n_kv_heads > 0 && head_dim > 0 {
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
            if let Some(ref mut l) = cache.latent_kv {
                let l_dim = cache.latent_dim;
                for p in sink..new_len {
                    let old_p = p + drop;
                    let dst_off = p * l_dim;
                    let src_off = old_p * l_dim;
                    l.copy_within(src_off..src_off + l_dim, dst_off);
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
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let m_eng = if is_gpu { self.metal_engine.as_ref() } else { None };

            self.buf_res[..dim].copy_from_slice(&self.buf_x[..dim]);
            layer.pre_norm.apply(&self.buf_x[..dim], &layer.pre_norm_weight, &mut self.buf_norm[..dim]);

            match &layer.mixer_weights {
                BlockMixerWeights::Attention { spec, q_proj, k_proj, v_proj, o_proj, q_norm_weight, k_norm_weight } => {
                    // Batched Q, K, V Projections
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let batched_qkv_done = if let (
                        Some(engine),
                        Some(ref q_b),
                        Some(ref k_b),
                        Some(ref v_b),
                        Some(ref q_m),
                        Some(ref k_m),
                        Some(ref v_m),
                    ) = (
                        m_eng,
                        &q_proj.metal_buf,
                        &k_proj.metal_buf,
                        &v_proj.metal_buf,
                        &q_proj.mlx4,
                        &k_proj.mlx4,
                        &v_proj.mlx4,
                    ) {
                        if let (
                            Some(ref qs_b),
                            Some(ref qb_b),
                            Some(ref ks_b),
                            Some(ref kb_b),
                            Some(ref vs_b),
                            Some(ref vb_b),
                        ) = (
                            &q_m.scales_metal_buf,
                            &q_m.biases_metal_buf,
                            &k_m.scales_metal_buf,
                            &k_m.biases_metal_buf,
                            &v_m.scales_metal_buf,
                            &v_m.biases_metal_buf,
                        ) {
                            engine.gemv_qkv_mlx4(
                                q_b, q_proj.metal_offset, q_proj.rows,
                                qs_b, q_m.scales_metal_offset,
                                qb_b, q_m.biases_metal_offset,
                                k_b, k_proj.metal_offset, k_proj.rows,
                                ks_b, k_m.scales_metal_offset,
                                kb_b, k_m.biases_metal_offset,
                                v_b, v_proj.metal_offset, v_proj.rows,
                                vs_b, v_m.scales_metal_offset,
                                vb_b, v_m.biases_metal_offset,
                                &self.buf_norm[..dim],
                                &mut self.buf_q[..n_heads * head_dim],
                                &mut self.buf_k[..n_kv_heads * head_dim],
                                &mut self.buf_v[..n_kv_heads * head_dim],
                                dim,
                                q_m.group_size,
                                q_m.is_bf16,
                            );
                            true
                        } else {
                            false
                        }
                    } else if q_proj.mlx4.is_none() && k_proj.mlx4.is_none() && v_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref q_b), Some(ref k_b), Some(ref v_b)) =
                            (m_eng, &q_proj.metal_buf, &k_proj.metal_buf, &v_proj.metal_buf)
                        {
                            let dtype_code = match q_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_qkv_multi(
                                q_b, q_proj.metal_offset, q_proj.rows,
                                k_b, k_proj.metal_offset, k_proj.rows,
                                v_b, v_proj.metal_offset, v_proj.rows,
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

                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
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

                    let min_t = if let Some(w) = spec.sliding_window {
                        if w > 0 { pos.saturating_sub(w - 1) } else { 0 }
                    } else {
                        0
                    };

                    for h in 0..n_heads {
                        let kv_h = h / group_size;
                        let q_h = &self.buf_q[h * head_dim..(h + 1) * head_dim];

                        let mut max_score = -f32::INFINITY;
                        for t in min_t..=pos {
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
                        for t in min_t..=pos {
                            let s = (self.buf_scores[t] - max_score).exp();
                            self.buf_scores[t] = s;
                            exp_sum += s;
                        }
                        let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };

                        let h_out = &mut self.buf_attn_out[h * head_dim..(h + 1) * head_dim];
                        h_out.fill(0.0);
                        for t in min_t..=pos {
                            let w = self.buf_scores[t] * inv_sum;
                            let v_t = kv_cache.get_v(kv_h, t);
                            for d in 0..head_dim {
                                h_out[d] += w * v_t[d];
                            }
                        }
                    }

                    // Asynchronous GPU Attention Output Projection
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let async_o_done = if o_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref o_b)) = (m_eng, &o_proj.metal_buf) {
                            let dtype_code = match o_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_async(
                                o_b, o_proj.metal_offset,
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
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let async_o_done = false;

                    if !async_o_done {
                        o_proj.gemv(&self.buf_attn_out[..o_proj.cols], &mut self.buf_x[..dim]);
                    } else {
                        #[cfg(any(target_os = "macos", target_os = "ios"))]
                        if let Some(engine) = m_eng {
                            engine.sync();
                        }
                    }
                }
                BlockMixerWeights::Conv1d { .. } => {
                    self.buf_x[..dim].copy_from_slice(&self.buf_norm[..dim]);
                }
                BlockMixerWeights::LatentAttention {
                    spec,
                    q_a_proj,
                    q_a_norm,
                    q_a_norm_weight,
                    q_b_proj,
                    kv_a_proj,
                    kv_a_norm,
                    kv_a_norm_weight,
                    w_uk,
                    w_uv,
                    o_proj,
                } => {
                    let d_c = spec.kv_lora_rank;
                    let d_nope = spec.qk_nope_head_dim;
                    let d_r = spec.qk_rope_head_dim;
                    let d_v = spec.v_head_dim;
                    let n_heads = spec.n_heads;
                    let latent_dim = d_c + d_r;

                    // 1. KV Down-Projection: c_t^{KV} (first d_c) and k_t^R (last d_r)
                    kv_a_proj.gemv(&self.buf_norm[..dim], &mut self.buf_k[..latent_dim]);
                    if let (Some(norm), Some(nw)) = (kv_a_norm, kv_a_norm_weight) {
                        norm.apply(&self.buf_k[..d_c], nw, &mut self.buf_tmp[..d_c]);
                        self.buf_k[..d_c].copy_from_slice(&self.buf_tmp[..d_c]);
                    }
                    rope(&mut self.buf_k[d_c..latent_dim], 1, d_r, pos, spec.rope_theta);

                    let kv_cache = &mut self.kv_caches[l];
                    kv_cache.store_latent(pos, &self.buf_k[..latent_dim]);

                    // 2. Query Computation
                    let q_total_dim = n_heads * (d_nope + d_r);
                    if let Some(qa) = q_a_proj {
                        qa.gemv(&self.buf_norm[..dim], &mut self.buf_tmp[..qa.rows]);
                        if let (Some(norm), Some(nw)) = (q_a_norm, q_a_norm_weight) {
                            norm.apply(&self.buf_tmp[..qa.rows], nw, &mut self.buf_gate[..qa.rows]);
                            q_b_proj.gemv(&self.buf_gate[..qa.rows], &mut self.buf_q[..q_total_dim]);
                        } else {
                            q_b_proj.gemv(&self.buf_tmp[..qa.rows], &mut self.buf_q[..q_total_dim]);
                        }
                    } else {
                        q_b_proj.gemv(&self.buf_norm[..dim], &mut self.buf_q[..q_total_dim]);
                    }

                    for h in 0..n_heads {
                        let base = h * (d_nope + d_r);
                        let q_r = &mut self.buf_q[base + d_nope..base + d_nope + d_r];
                        rope(q_r, 1, d_r, pos, spec.rope_theta);
                    }

                    // 3. Absorbed Query Attention & Softmax:
                    // For each head h: absorb W_h^{UK} into query q'_{t, h} = (q_nope)^T * W_h^{UK} in R^{d_c}
                    for h in 0..n_heads {
                        let q_base = h * (d_nope + d_r);
                        let q_nope = &self.buf_q[q_base..q_base + d_nope];
                        let q_r = &self.buf_q[q_base + d_nope..q_base + d_nope + d_r];

                        let q_absorbed = &mut self.buf_gate[..d_c];
                        q_absorbed.fill(0.0);
                        for i in 0..d_nope {
                            let qi = q_nope[i];
                            let row_idx = h * d_nope + i;
                            w_uk.get_row(row_idx, &mut self.buf_tmp[..d_c]);
                            for c in 0..d_c {
                                q_absorbed[c] += qi * self.buf_tmp[c];
                            }
                        }

                        let mut max_score = -f32::INFINITY;
                        for t in 0..=pos {
                            let cached = kv_cache.get_latent(t);
                            let c_kv = &cached[..d_c];
                            let k_r = &cached[d_c..latent_dim];

                            let mut dot_c = 0.0f32;
                            for c in 0..d_c {
                                dot_c += q_absorbed[c] * c_kv[c];
                            }
                            let mut dot_r = 0.0f32;
                            for r in 0..d_r {
                                dot_r += q_r[r] * k_r[r];
                            }
                            let score = (dot_c + dot_r) * spec.softmax_scale;
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

                        let v_latent = &mut self.buf_up[..d_c];
                        v_latent.fill(0.0);
                        for t in 0..=pos {
                            let w = self.buf_scores[t] * inv_sum;
                            let cached = kv_cache.get_latent(t);
                            let c_kv = &cached[..d_c];
                            for c in 0..d_c {
                                v_latent[c] += w * c_kv[c];
                            }
                        }

                        let h_out = &mut self.buf_attn_out[h * d_v..(h + 1) * d_v];
                        for j in 0..d_v {
                            let row_idx = h * d_v + j;
                            w_uv.get_row(row_idx, &mut self.buf_tmp[..d_c]);
                            let mut sum = 0.0f32;
                            for c in 0..d_c {
                                sum += self.buf_tmp[c] * v_latent[c];
                            }
                            h_out[j] = sum;
                        }
                    }

                    o_proj.gemv(&self.buf_attn_out[..n_heads * d_v], &mut self.buf_x[..dim]);
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
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let mlx4_gate_up_done = if let (Some(engine), Some(gb), Some(gsb), Some(gbb), Some(ub), Some(usb), Some(ubb)) = (
                        m_eng, &gate_proj.metal_buf, gate_proj.mlx4.as_ref().and_then(|g| g.scales_metal_buf.as_ref()), gate_proj.mlx4.as_ref().and_then(|g| g.biases_metal_buf.as_ref()),
                        &up_proj.metal_buf, up_proj.mlx4.as_ref().and_then(|u| u.scales_metal_buf.as_ref()), up_proj.mlx4.as_ref().and_then(|u| u.biases_metal_buf.as_ref()),
                    ) {
                        if let (Some(g), Some(_u)) = (&gate_proj.mlx4, &up_proj.mlx4) {
                            engine.gemv_gate_up_mlx4(
                                gb, gate_proj.metal_offset, gate_proj.rows,
                                gsb, g.scales_metal_offset,
                                gbb, g.biases_metal_offset,
                                ub, up_proj.metal_offset, up_proj.rows,
                                usb, _u.scales_metal_offset,
                                ubb, _u.biases_metal_offset,
                                &self.buf_norm[..dim],
                                &mut self.buf_gate[..gate_proj.rows],
                                &mut self.buf_up[..up_proj.rows],
                                dim, g.group_size, g.is_bf16,
                            );
                            true
                        } else { false }
                    } else { false };
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let mlx4_gate_up_done = false;
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let batched_gate_up = if mlx4_gate_up_done { false } else if gate_proj.mlx4.is_none() && up_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref g_b), Some(ref u_b)) =
                            (m_eng, &gate_proj.metal_buf, &up_proj.metal_buf)
                        {
                            let dtype_code = match gate_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_gate_up_multi(
                                g_b, gate_proj.metal_offset,
                                u_b, up_proj.metal_offset,
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
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let batched_gate_up = false;

                    if !batched_gate_up {
                        gate_proj.gemv(&self.buf_norm[..dim], &mut self.buf_gate[..gate_proj.rows]);
                        up_proj.gemv(&self.buf_norm[..dim], &mut self.buf_up[..up_proj.rows]);
                    }

                    swiglu_in_place(&mut self.buf_gate[..gate_proj.rows], &self.buf_up[..up_proj.rows]);

                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let async_down = if down_proj.mlx4.is_none() {
                        if let (Some(engine), Some(ref d_b)) = (m_eng, &down_proj.metal_buf) {
                            let dtype_code = match down_proj.dtype {
                                TensorDType::F16 => 1,
                                TensorDType::F32 => 2,
                                _ => 0,
                            };
                            engine.gemv_async(
                                d_b, down_proj.metal_offset,
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

                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let async_down = false;

                    if !async_down {
                        down_proj.gemv(&self.buf_gate[..down_proj.cols], &mut self.buf_ffn_out[..dim]);
                    } else {
                        #[cfg(any(target_os = "macos", target_os = "ios"))]
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
                BlockChannelMixerWeights::SparseMoE {
                    top_k,
                    router,
                    experts_gate,
                    experts_up,
                    experts_down,
                    shared_gate,
                    shared_up,
                    shared_down,
                    ..
                } => {
                    // 1. Router logits into preallocated buf_router (zero heap allocations)
                    router.gemv(&self.buf_norm[..dim], &mut self.buf_router[..router.rows]);

                    // 2. Select Top-K experts on the stack
                    let actual_k = (*top_k).min(8).min(experts_gate.len());
                    let mut top_indices = [0usize; 8];
                    let mut top_logits = [-f32::INFINITY; 8];

                    for k in 0..actual_k {
                        let mut best_idx = 0;
                        let mut best_val = -f32::INFINITY;
                        for idx in 0..router.rows.min(experts_gate.len()) {
                            let mut already = false;
                            for prev in 0..k {
                                if top_indices[prev] == idx {
                                    already = true;
                                    break;
                                }
                            }
                            if !already && self.buf_router[idx] > best_val {
                                best_val = self.buf_router[idx];
                                best_idx = idx;
                            }
                        }
                        top_indices[k] = best_idx;
                        top_logits[k] = best_val;
                    }

                    // 3. Numerically stable softmax over Top-K logits
                    let max_l = top_logits[0];
                    let mut sum_exp = 0.0f32;
                    let mut weights = [0.0f32; 8];
                    for k in 0..actual_k {
                        let e = (top_logits[k] - max_l).exp();
                        weights[k] = e;
                        sum_exp += e;
                    }
                    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                    for k in 0..actual_k {
                        weights[k] *= inv_sum;
                    }

                    // 4. Weighted accumulation across Top-K routed experts
                    self.buf_ffn_out[..dim].fill(0.0);
                    for k in 0..actual_k {
                        let exp_idx = top_indices[k];
                        let w = weights[k];
                        if exp_idx < experts_gate.len() && w > 0.0 {
                            let eg = &experts_gate[exp_idx];
                            let eu = &experts_up[exp_idx];
                            let ed = &experts_down[exp_idx];
                            eg.gemv(&self.buf_norm[..dim], &mut self.buf_gate[..eg.rows]);
                            eu.gemv(&self.buf_norm[..dim], &mut self.buf_up[..eu.rows]);
                            swiglu_in_place(&mut self.buf_gate[..eg.rows], &self.buf_up[..eu.rows]);
                            ed.gemv(&self.buf_gate[..ed.cols], &mut self.buf_tmp[..dim]);
                            for i in 0..dim {
                                self.buf_ffn_out[i] += w * self.buf_tmp[i];
                            }
                        }
                    }

                    // 5. Shared expert accumulation (unweighted, as in Qwen-MoE and DeepSeek)
                    if let (Some(sg), Some(su), Some(sd)) = (shared_gate, shared_up, shared_down) {
                        sg.gemv(&self.buf_norm[..dim], &mut self.buf_gate[..sg.rows]);
                        su.gemv(&self.buf_norm[..dim], &mut self.buf_up[..su.rows]);
                        swiglu_in_place(&mut self.buf_gate[..sg.rows], &self.buf_up[..su.rows]);
                        sd.gemv(&self.buf_gate[..sd.cols], &mut self.buf_tmp[..dim]);
                        for i in 0..dim {
                            self.buf_ffn_out[i] += self.buf_tmp[i];
                        }
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

    /// Embedding lookup for a token into out_hidden.
    pub fn forward_embed(&mut self, token: u32, out_hidden: &mut [f32]) {
        let dim = self.config.dim;
        self.embed_tokens.get_row(token as usize, &mut out_hidden[..dim]);
        if let Some(scale) = self.graph_config.embed_scale {
            for v in &mut out_hidden[..dim] {
                *v *= scale;
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

    /// Final norm and LM head projection into out_logits.
    pub fn forward_head(&mut self, in_hidden: &[f32], out_logits: &mut [f32]) {
        let dim = self.config.dim;
        self.buf_x[..dim].copy_from_slice(&in_hidden[..dim]);
        self.graph_config.final_norm.apply(&self.buf_x[..dim], &self.final_norm_weight, &mut self.buf_norm[..dim]);
        if let Some(lm_head) = &self.lm_head {
            lm_head.gemv(&self.buf_norm[..dim], out_logits);
        } else {
            self.embed_tokens.gemv(&self.buf_norm[..dim], out_logits);
        }
        if let Some(cap) = self.graph_config.final_logit_soft_cap {
            soft_cap_in_place(out_logits, cap);
        }
    }

    pub fn forward_prefill(&mut self, token: u32, pos: usize) {
        self.forward(token, pos);
    }

    /// Chunk size for batched prefill: each chunk's weight matrices become
    /// one gemm_batched call per projection, so AMX/NEON tiles see 32-row
    /// activations instead of one GEMV per token.
    const PRE_CHUNK: usize = 256;

    /// True when every layer is a standard Attention mixer over a SwiGLU or
    /// GeGLU channel mixer — exactly the shapes `forward_chunk` implements.
    /// MoE / MLA / conv mixers keep the per-token path.
    fn chunked_prefill_ok(&self) -> bool {
        // Uniform per-layer dims let the chunked path size buffers once.
        let mut q_ref = 0usize;
        let mut kv_ref = (0usize, 0usize);
        let mut inter_ref = 0usize;
        for l in &self.layers {
            let (q_rows, k_rows, v_rows, attn_cols, inter) = match (&l.mixer_weights, &l.channel_mixer_weights) {
                (
                    BlockMixerWeights::Attention { q_proj, k_proj, v_proj, o_proj, .. },
                    BlockChannelMixerWeights::SwiGLU { gate_proj, .. }
                    | BlockChannelMixerWeights::GeGLU { gate_proj, .. },
                ) => (
                    q_proj.rows,
                    k_proj.rows,
                    v_proj.rows,
                    o_proj.cols,
                    gate_proj.rows,
                ),
                _ => return false,
            };
            if q_ref == 0 {
                q_ref = q_rows;
                kv_ref = (k_rows, v_rows);
                inter_ref = inter;
            } else if q_ref != q_rows || inter_ref != inter || kv_ref != (k_rows, v_rows) {
                return false;
            }
            if attn_cols != q_rows {
                return false;
            }
        }
        true
    }

    /// Batched prefill over one chunk. Weight matmuls run once per layer for
    /// ALL tokens of the chunk (`WeightMatrix::gemm_batched`); rope, KV-cache
    /// stores and causal attention stay token-sequential exactly as the
    /// per-token path, so logits agree to f32 summation order only.
    pub fn forward_chunk(&mut self, tokens: &[u32], start_pos: usize) {
        let n = tokens.len();
        if n == 0 {
            return;
        }
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let freq_base = self.config.rope_freq_base;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let mut bx = vec![0.0f32; n * dim];
        let mut bres = vec![0.0f32; n * dim];
        let mut bnorm = vec![0.0f32; n * dim];
        let mut bq = vec![0.0f32; n * q_dim];
        let mut bk = vec![0.0f32; n * kv_dim];
        let mut bv = vec![0.0f32; n * kv_dim];
        let mut battn = vec![0.0f32; n * q_dim];
        let mut bo = vec![0.0f32; n * dim];
        let inter_max = self
            .layers
            .iter()
            .map(|l| match &l.channel_mixer_weights {
                BlockChannelMixerWeights::SwiGLU { gate_proj, .. } => gate_proj.rows,
                BlockChannelMixerWeights::GeGLU { gate_proj, .. } => gate_proj.rows,
                _ => 0,
            })
            .max()
            .unwrap_or(dim);
        let mut bgate = vec![0.0f32; n * inter_max];
        let mut bup = vec![0.0f32; n * inter_max];
        let mut bffn = vec![0.0f32; n * dim];
        let mut tmp = vec![0.0f32; dim.max(q_dim.max(kv_dim))];

        for (t, &tok) in tokens.iter().enumerate() {
            self.embed_tokens.get_row(tok as usize, &mut bx[t * dim..(t + 1) * dim]);
            if let Some(scale) = self.graph_config.embed_scale {
                for v in &mut bx[t * dim..(t + 1) * dim] {
                    *v *= scale;
                }
            }
        }

        for l in 0..self.layers.len() {
            let layer = &self.layers[l];
            let (
                spec_ref,
                q_proj_ref,
                k_proj_ref,
                v_proj_ref,
                o_proj_ref,
               qnw_ref,
               knw_ref,
            ) = match &layer.mixer_weights {
                BlockMixerWeights::Attention { spec, q_proj, k_proj, v_proj, o_proj, q_norm_weight, k_norm_weight } => {
                    (spec, q_proj, k_proj, v_proj, o_proj, q_norm_weight, k_norm_weight)
                }
                _ => unreachable!("chunked_prefill_ok guards the mixer kind"),
            };

            // Pre-norm for all tokens
            for t in 0..n {
                bres[t * dim..(t + 1) * dim].copy_from_slice(&bx[t * dim..(t + 1) * dim]);
                layer.pre_norm.apply(
                    &bx[t * dim..(t + 1) * dim],
                    &layer.pre_norm_weight,
                    &mut bnorm[t * dim..(t + 1) * dim],
                );
            }

            // Batched Q/K/V projections (one sgemm each per chunk)
            q_proj_ref.gemm_batched(&bnorm, n, &mut bq);
            k_proj_ref.gemm_batched(&bnorm, n, &mut bk);
            v_proj_ref.gemm_batched(&bnorm, n, &mut bv);

            // Token-sequential: qk-norms, rope, cache store, causal attention
            let inv_sqrt_head_dim = 1.0f32 / (head_dim as f32).sqrt();
            let group = (n_heads / n_kv_heads).max(1);
            for t in 0..n {
                let pos_t = start_pos + t;
                if let (Some(q_nw), Some(qn)) = (qnw_ref, spec_ref.q_norm) {
                    for h in 0..n_heads {
                        let off = t * q_dim + h * head_dim;
                        qn.apply(&bq[off..off + head_dim], q_nw, &mut tmp[..head_dim]);
                        bq[off..off + head_dim].copy_from_slice(&tmp[..head_dim]);
                    }
                }
                if let (Some(k_nw), Some(kn)) = (knw_ref, spec_ref.k_norm) {
                    for h in 0..n_kv_heads {
                        let off = t * kv_dim + h * head_dim;
                        kn.apply(&bk[off..off + head_dim], k_nw, &mut tmp[..head_dim]);
                        bk[off..off + head_dim].copy_from_slice(&tmp[..head_dim]);
                    }
                }
                rope(&mut bq[t * q_dim..(t + 1) * q_dim], n_heads, head_dim, pos_t, freq_base);
                rope(&mut bk[t * kv_dim..(t + 1) * kv_dim], n_kv_heads, head_dim, pos_t, freq_base);

                let kv_cache = &mut self.kv_caches[l];
                for kv_h in 0..n_kv_heads {
                    let off = t * kv_dim + kv_h * head_dim;
                    kv_cache.store(
                        kv_h,
                        pos_t,
                        &bk[off..off + head_dim],
                        &bv[off..off + head_dim],
                    );
                }

                let min_t = if let Some(w) = spec_ref.sliding_window {
                    if w > 0 { pos_t.saturating_sub(w - 1) } else { 0 }
                } else {
                    0
                };
                for h in 0..n_heads {
                    let kv_h = h / group;
                    let q_h = &bq[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
                    let mut max_score = -f32::INFINITY;
                    for tt in min_t..=pos_t {
                        let k_t = kv_cache.get_k(kv_h, tt);
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_h[d] * k_t[d];
                        }
                        let mut score = dot * inv_sqrt_head_dim;
                        if let Some(cap) = spec_ref.attn_logit_soft_cap {
                            score = cap * (score / cap).tanh();
                        }
                        if score > max_score {
                            max_score = score;
                        }
                        self.buf_scores[tt] = score;
                    }
                    let mut exp_sum = 0.0f32;
                    for tt in min_t..=pos_t {
                        let s = (self.buf_scores[tt] - max_score).exp();
                        self.buf_scores[tt] = s;
                        exp_sum += s;
                    }
                    let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                    let h_out = &mut battn[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
                    h_out.fill(0.0);
                    for tt in min_t..=pos_t {
                        let w = self.buf_scores[tt] * inv_sum;
                        let v_t = kv_cache.get_v(kv_h, tt);
                        for d in 0..head_dim {
                            h_out[d] += w * v_t[d];
                        }
                    }
                }
            }

            // Batched output projection + residual per token
            o_proj_ref.gemm_batched(&battn, n, &mut bo);
            for t in 0..n {
                bx[t * dim..(t + 1) * dim].copy_from_slice(&bo[t * dim..(t + 1) * dim]);
                if let (Some(pan), Some(pan_w)) = (layer.post_attention_norm, &layer.post_attention_norm_weight) {
                    pan.apply(&bx[t * dim..(t + 1) * dim], pan_w, &mut tmp[..dim]);
                    bx[t * dim..(t + 1) * dim].copy_from_slice(&tmp[..dim]);
                }
                for i in 0..dim {
                    bx[t * dim + i] += bres[t * dim + i];
                }
            }

            // FFN pre-norm per token
            for t in 0..n {
                bres[t * dim..(t + 1) * dim].copy_from_slice(&bx[t * dim..(t + 1) * dim]);
                layer.ffn_norm.apply(
                    &bx[t * dim..(t + 1) * dim],
                    &layer.ffn_norm_weight,
                    &mut bnorm[t * dim..(t + 1) * dim],
                );
            }
            match &layer.channel_mixer_weights {
                BlockChannelMixerWeights::SwiGLU { gate_proj, up_proj, down_proj } => {
                    gate_proj.gemm_batched(&bnorm, n, &mut bgate[..n * gate_proj.rows]);
                    up_proj.gemm_batched(&bnorm, n, &mut bup[..n * up_proj.rows]);
                    for t in 0..n {
                        swiglu_in_place(
                            &mut bgate[t * gate_proj.rows..(t + 1) * gate_proj.rows],
                            &bup[t * up_proj.rows..(t + 1) * up_proj.rows],
                        );
                    }
                    down_proj.gemm_batched(&bgate[..n * down_proj.cols], n, &mut bffn);
                }
                BlockChannelMixerWeights::GeGLU { gate_proj, up_proj, down_proj } => {
                    gate_proj.gemm_batched(&bnorm, n, &mut bgate[..n * gate_proj.rows]);
                    up_proj.gemm_batched(&bnorm, n, &mut bup[..n * up_proj.rows]);
                    for t in 0..n {
                        geglu_in_place(
                            &mut bgate[t * gate_proj.rows..(t + 1) * gate_proj.rows],
                            &bup[t * up_proj.rows..(t + 1) * up_proj.rows],
                        );
                    }
                    down_proj.gemm_batched(&bgate[..n * down_proj.cols], n, &mut bffn);
                }
                _ => unreachable!("chunked_prefill_ok guards the channel mixer"),
            }
            for t in 0..n {
                if let (Some(pfn), Some(pfn_w)) = (layer.post_ffn_norm, &layer.post_ffn_norm_weight) {
                    pfn.apply(&bffn[t * dim..(t + 1) * dim], pfn_w, &mut tmp[..dim]);
                    bffn[t * dim..(t + 1) * dim].copy_from_slice(&tmp[..dim]);
                }
                for i in 0..dim {
                    bx[t * dim + i] = bres[t * dim + i] + bffn[t * dim + i];
                }
            }
        }
        self.finish_chunk_logits(&bx, n);
    }

    /// Final norm + LM head for the LAST token of a chunk buffer.
    fn finish_chunk_logits(&mut self, bx: &[f32], n: usize) {
        let dim = self.config.dim;
        let last = &bx[(n - 1) * dim..n * dim];
        self.graph_config.final_norm.apply(last, &self.final_norm_weight, &mut self.buf_norm[..dim]);
        if let Some(lm_head) = &self.lm_head {
            lm_head.gemv(&self.buf_norm[..dim], &mut self.logits);
        } else {
            self.embed_tokens.gemv(&self.buf_norm[..dim], &mut self.logits);
        }
        if let Some(cap) = self.graph_config.final_logit_soft_cap {
            soft_cap_in_place(&mut self.logits, cap);
        }
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], start_pos: usize) {
        if tokens.is_empty() {
            return;
        }
        if self.chunked_prefill_ok() {
            for (ci, ch) in tokens.chunks(Self::PRE_CHUNK).enumerate() {
                self.forward_chunk(ch, start_pos + ci * Self::PRE_CHUNK);
            }
            return;
        }
        for (i, &tok) in tokens.iter().enumerate() {
            self.forward_prefill(tok, start_pos + i);
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        if tokens.is_empty() {
            return &self.logits;
        }
        if self.chunked_prefill_ok() {
            for (ci, ch) in tokens.chunks(Self::PRE_CHUNK).enumerate() {
                self.forward_chunk(ch, start_pos + ci * Self::PRE_CHUNK);
            }
            return &self.logits;
        }
        for (i, &tok) in tokens.iter().enumerate() {
            self.forward(tok, start_pos + i);
        }
        &self.logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gemm_batched vs the per-token gemv loop on F16 weights: identical
    /// element products and f32 accumulation, summation order may differ.
    #[test]
    fn weight_matrix_gemm_batched_matches_gemv() {
        let (rows, cols, n_tok) = (37usize, 128usize, 19usize);
        // small-magnitude normal f16s (0x3800..0x3900 band), deterministic
        let bits: Vec<u16> = (0..rows * cols).map(|i| 0x3800u16 + ((i * 7) % 0xFE) as u16).collect();
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for b in &bits {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        let wm = WeightMatrix::new_from_data(
            rows, cols, TensorDType::F16, UnifiedTensorData::Heap(std::sync::Arc::new(bytes)),
        );
        let xs: Vec<f32> = (0..n_tok * cols).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();
        let mut batched = vec![0.0f32; n_tok * rows];
        wm.gemm_batched(&xs, n_tok, &mut batched);
        for t in 0..n_tok {
            let mut want = vec![0.0f32; rows];
            wm.gemv(&xs[t * cols..(t + 1) * cols], &mut want);
            for r in 0..rows {
                let got = batched[t * rows + r];
                let err = (got - want[r]).abs() / want[r].abs().max(1e-3);
                assert!(err < 1e-4, "[{t},{r}] gemm {got} vs gemv {} rel {err}", want[r]);
            }
        }
    }

    /// Chunked prefill must land on the same logits (argmax + tight RMS) as
    /// the per-token path on gemma-2b — guards the whole batched rewrite.
    #[test]
    #[ignore = "needs gemma-2b on disk; run explicitly"]
    fn blockgraph_chunked_prefill_matches_per_token() {
        let table = UnifiedTensorTable::open("/Users/priyanshuboruah/Downloads/gemma-2b/model.safetensors")
            .expect("open gemma-2b");
        let mut m = BlockGraphModel::from_tensor_table(&table, 0).expect("load gemma-2b");
        assert!(m.chunked_prefill_ok(), "gemma-2b should be chunkable");
        let tokens: Vec<u32> = (0..40u32).cycle().take(45).collect();
        let per_token_ref = {
            let mut mm = BlockGraphModel::from_tensor_table(&table, 0).expect("reload");
            mm.forward_prefill_all_with_logits(&tokens, 0).to_vec()
        };
        let got = m.forward_prefill_all_with_logits(&tokens, 0);
        let n = got.len();
        let mut sum_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        let (mut ta, mut tb) = (0usize, 0usize);
        for i in 0..n {
            let d = (per_token_ref[i] - got[i]) as f64;
            sum_sq += d * d;
            ref_sq += per_token_ref[i] as f64 * per_token_ref[i] as f64;
            if per_token_ref[i] > per_token_ref[ta] { ta = i; }
            if got[i] > got[tb] { tb = i; }
        }
        // Summation-order-only difference on the AMX/F16 tile path.
        let rms = (sum_sq / n as f64).sqrt() / ((ref_sq / n as f64).sqrt().max(1e-9)) as f32 as f64;
        assert!(ta == tb, "argmax mismatch: {ta} vs {tb}");
        assert!(rms < 2e-3, "rel RMS {rms:.6} beyond summation-order tolerance");
let _ = rms;
    }

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

    #[test]
    fn test_block_graph_sliding_window_bounding() {
        let mut cache = BlockKvCache::new(1, 16, 4);
        for p in 0..10 {
            let k = vec![p as f32; 4];
            let v = vec![p as f32; 4];
            cache.store(0, p, &k, &v);
        }
        // If window is 4 at pos 8, min_t = 8 - (4 - 1) = 5
        let pos: usize = 8;
        let window: usize = 4;
        let min_t = pos.saturating_sub(window - 1);
        assert_eq!(min_t, 5);
        // Only tokens 5..=8 should be in the window
        assert_eq!(cache.get_k(0, min_t), &[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(cache.get_k(0, pos), &[8.0, 8.0, 8.0, 8.0]);
    }

    #[test]
    fn test_top_k_moe_routing_and_accumulation() {
        let router_data = vec![
            0.2f32, 0.0,
            0.9f32, 0.0,
            0.1f32, 0.0,
            0.8f32, 0.0,
        ];
        let mut raw_router = Vec::new();
        for &w in &router_data {
            raw_router.extend_from_slice(&w.to_le_bytes());
        }
        let router = WeightMatrix::new(4, 2, TensorDType::F32, Arc::new(raw_router), 0);

        let mut buf_router = vec![0.0f32; 4];
        let x = vec![1.0f32, 0.0];
        router.gemv(&x, &mut buf_router);

        let top_k = 2;
        let actual_k = top_k.min(8).min(4);
        let mut top_indices = [0usize; 8];
        let mut top_logits = [-f32::INFINITY; 8];

        for k in 0..actual_k {
            let mut best_idx = 0;
            let mut best_val = -f32::INFINITY;
            for idx in 0..4 {
                let mut already = false;
                for prev in 0..k {
                    if top_indices[prev] == idx {
                        already = true;
                        break;
                    }
                }
                if !already && buf_router[idx] > best_val {
                    best_val = buf_router[idx];
                    best_idx = idx;
                }
            }
            top_indices[k] = best_idx;
            top_logits[k] = best_val;
        }

        assert_eq!(top_indices[0], 1); // 0.9
        assert_eq!(top_indices[1], 3); // 0.8

        let max_l = top_logits[0];
        let mut sum_exp = 0.0f32;
        let mut weights = [0.0f32; 8];
        for k in 0..actual_k {
            let e = (top_logits[k] - max_l).exp();
            weights[k] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        for k in 0..actual_k {
            weights[k] *= inv_sum;
        }

        assert!((weights[0] + weights[1] - 1.0).abs() < 1e-5);
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_mla_latent_cache_and_absorption() {
        let max_seq_len = 16;
        let d_c = 8;
        let d_r = 4;
        let latent_dim = d_c + d_r;
        let mut cache = BlockKvCache::new_latent(max_seq_len, latent_dim);

        let mut latent_vec = vec![0.0f32; latent_dim];
        for i in 0..latent_dim {
            latent_vec[i] = (i + 1) as f32;
        }

        cache.store_latent(2, &latent_vec);
        let retrieved = cache.get_latent(2);
        assert_eq!(retrieved.len(), latent_dim);
        assert_eq!(retrieved, &latent_vec[..]);

        let c_kv = &retrieved[..d_c];
        let k_r = &retrieved[d_c..latent_dim];
        assert_eq!(c_kv.len(), 8);
        assert_eq!(k_r.len(), 4);
    }
}
