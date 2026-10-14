// crates/sofuu-core/src/infer/lfm2.rs — Liquid Foundation Models (LFM-2) Inference Engine.
//
// Supports:
// - Liquid AI LFM2 architecture (Lfm2ForCausalLM)
// - Dual operator layers: Gated Short Convolutions (1D causal) & GQA Attention
// - Unquantized BF16 SafeTensors (liqiud-350)
// - MLX 4-bit Affine Quantized SafeTensors (liquid)
// - Apple Silicon Metal Unified GPU Engine (zero-copy UMA) with CPU SIMD fallback

use std::path::Path;
use super::model::{AnonBuffer, ModelConfig};
use super::safetensors::{SafeDType, SafeTensorView, SafeTensorsFile};
// Metal items in this file: macOS + iOS — the GPU engine compiles for
// both now (see the header of infer/metal.rs).
#[cfg(any(target_os = "macos", target_os = "ios"))]
use super::metal::{MetalBuffer, MetalEngine};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2Config {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    #[serde(default = "default_conv_l_cache")]
    #[allow(non_snake_case)]
    pub conv_L_cache: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<Lfm2QuantConfig>,
    #[serde(default)]
    pub quantization_config: Option<Lfm2QuantConfig>,
}

fn default_conv_l_cache() -> usize { 3 }
fn default_norm_eps() -> f32 { 1e-5 }

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2QuantConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: String,
}

#[derive(Clone)]
pub enum LinearTensor {
    Bf16(SafeTensorView),
    Mlx4 {
        weight: SafeTensorView,
        scales: SafeTensorView,
        biases: SafeTensorView,
        group_size: usize,
    },
}

impl LinearTensor {
    #[inline]
    pub fn rows(&self) -> usize {
        match self {
            LinearTensor::Bf16(tv) => tv.shape[0],
            LinearTensor::Mlx4 { weight, .. } => weight.shape[0],
        }
    }
    #[inline]
    pub fn cols(&self) -> usize {
        match self {
            LinearTensor::Bf16(tv) => tv.shape[1],
            LinearTensor::Mlx4 { weight, .. } => weight.shape[1] * 8,
        }
    }
}

#[derive(Clone)]
pub enum Lfm2LayerWeights {
    Conv {
        operator_norm: SafeTensorView,
        in_proj: LinearTensor,
        conv: SafeTensorView,
        out_proj: LinearTensor,
        ffn_norm: SafeTensorView,
        w1: LinearTensor,
        w3: LinearTensor,
        w2: LinearTensor,
    },
    Attention {
        operator_norm: SafeTensorView,
        q_proj: LinearTensor,
        k_proj: LinearTensor,
        v_proj: LinearTensor,
        out_proj: LinearTensor,
        q_layernorm: SafeTensorView,
        k_layernorm: SafeTensorView,
        ffn_norm: SafeTensorView,
        w1: LinearTensor,
        w3: LinearTensor,
        w2: LinearTensor,
    },
}

pub struct Lfm2Model {
    pub config: ModelConfig,
    pub lfm_config: Lfm2Config,
    pub sf: SafeTensorsFile,
    pub embed_tokens: LinearTensor,
    pub embedding_norm: SafeTensorView,
    pub layers: Vec<Lfm2LayerWeights>,

    // Metal GPU Engine handle (if available on macOS)
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub metal_engine: Option<MetalEngine>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub metal_mmap_buf: Option<std::sync::Arc<MetalBuffer>>,

    // Working activation buffers
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub logits: Vec<f32>,

    // Conv layer working state buffers
    pub bcx_buf: Vec<f32>,
    pub conv_states: Vec<Vec<f32>>, // [layer_idx][channels * 2]

    // Attention layer working state buffers & KV cache
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub head_out: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub k_cache: AnonBuffer,
    pub v_cache: AnonBuffer,

    // MLP working state buffers
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_out: Vec<f32>,

    pub max_seq_len: usize,
    pub sliding_window: usize,
    pub sink_tokens: usize,
    pub gpu_layers: usize,
}

#[inline(always)]
pub fn bf16_to_f32(raw: u16) -> f32 {
    f32::from_bits((raw as u32) << 16)
}

#[inline(always)]
pub fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

#[inline(always)]
pub unsafe fn read_u16_unaligned(ptr: *const u8, idx: usize) -> u16 {
    let p = ptr.add(idx * 2) as *const u16;
    std::ptr::read_unaligned(p)
}

#[inline(always)]
pub unsafe fn read_u32_unaligned(ptr: *const u8, idx: usize) -> u32 {
    let p = ptr.add(idx * 4) as *const u32;
    std::ptr::read_unaligned(p)
}

impl Lfm2Model {
    pub fn from_safetensors_dir<P: AsRef<Path>>(dir: P, max_seq_len: usize) -> Result<Self, String> {
        let p = dir.as_ref();
        let config_path = p.join("config.json");
        let model_path = p.join("model.safetensors");

        if !config_path.exists() {
            return Err(format!("config.json not found in {}", p.display()));
        }
        if !model_path.exists() {
            return Err(format!("model.safetensors not found in {}", p.display()));
        }

        let cfg_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("failed to read config.json: {}", e))?;
        let lfm_config: Lfm2Config = serde_json::from_str(&cfg_str)
            .map_err(|e| format!("failed to parse LFM2 config.json: {}", e))?;

        let sf = SafeTensorsFile::open(&model_path)?;

        let dim = lfm_config.hidden_size;
        let n_layers = lfm_config.num_hidden_layers;
        let n_heads = lfm_config.num_attention_heads;
        let n_kv_heads = lfm_config.num_key_value_heads;
        let head_dim = lfm_config.head_dim.unwrap_or(dim / n_heads);
        let hidden_dim = lfm_config.intermediate_size;
        let vocab_size = lfm_config.vocab_size;
        let norm_eps = lfm_config.norm_eps;
        let rope_freq_base = lfm_config.rope_theta.unwrap_or(1000000.0);
        let effective_max_seq_len = if max_seq_len > 0 {
            max_seq_len
        } else {
            lfm_config.max_position_embeddings.unwrap_or(2048).min(4096)
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
            max_seq_len: effective_max_seq_len,
            sliding_window: 0,
            rope_interleaved: false,
        };

        // Helper to load linear tensor (supports BF16 or MLX 4-bit)
        let get_linear = |base: &str| -> Result<LinearTensor, String> {
            let w_key = format!("{}.weight", base);
            let tv = sf.tensors.get(&w_key)
                .ok_or_else(|| format!("missing {}", w_key))?;

            if tv.dtype == SafeDType::U32 {
                let s_key = format!("{}.scales", base);
                let b_key = format!("{}.biases", base);
                let scales = sf.tensors.get(&s_key)
                    .ok_or_else(|| format!("missing {}", s_key))?
                    .clone();
                let biases = sf.tensors.get(&b_key)
                    .ok_or_else(|| format!("missing {}", b_key))?
                    .clone();
                let group_size = lfm_config.quantization.as_ref()
                    .or(lfm_config.quantization_config.as_ref())
                    .map(|q| q.group_size)
                    .unwrap_or(64);
                Ok(LinearTensor::Mlx4 {
                    weight: tv.clone(),
                    scales,
                    biases,
                    group_size,
                })
            } else {
                Ok(LinearTensor::Bf16(tv.clone()))
            }
        };

        let embed_tokens = get_linear("model.embed_tokens")?;
        let embedding_norm = sf.tensors.get("model.embedding_norm.weight")
            .ok_or_else(|| "missing model.embedding_norm.weight".to_string())?
            .clone();

        // Load layer weights
        let mut layers = Vec::with_capacity(n_layers);
        let mut conv_states = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let prefix = format!("model.layers.{}", i);
            let layer_type = lfm_config.layer_types.get(i)
                .map(|s| s.as_str())
                .unwrap_or("conv");

            let operator_norm = sf.tensors.get(&format!("{}.operator_norm.weight", prefix))
                .ok_or_else(|| format!("missing {}.operator_norm.weight", prefix))?
                .clone();
            let ffn_norm = sf.tensors.get(&format!("{}.ffn_norm.weight", prefix))
                .ok_or_else(|| format!("missing {}.ffn_norm.weight", prefix))?
                .clone();

            let w1 = get_linear(&format!("{}.feed_forward.w1", prefix))?;
            let w3 = get_linear(&format!("{}.feed_forward.w3", prefix))?;
            let w2 = get_linear(&format!("{}.feed_forward.w2", prefix))?;

            if layer_type == "full_attention" {
                let q_proj = get_linear(&format!("{}.self_attn.q_proj", prefix))?;
                let k_proj = get_linear(&format!("{}.self_attn.k_proj", prefix))?;
                let v_proj = get_linear(&format!("{}.self_attn.v_proj", prefix))?;
                let out_proj = get_linear(&format!("{}.self_attn.out_proj", prefix))?;
                let q_layernorm = sf.tensors.get(&format!("{}.self_attn.q_layernorm.weight", prefix))
                    .ok_or_else(|| format!("missing {}.self_attn.q_layernorm.weight", prefix))?
                    .clone();
                let k_layernorm = sf.tensors.get(&format!("{}.self_attn.k_layernorm.weight", prefix))
                    .ok_or_else(|| format!("missing {}.self_attn.k_layernorm.weight", prefix))?
                    .clone();

                layers.push(Lfm2LayerWeights::Attention {
                    operator_norm,
                    q_proj,
                    k_proj,
                    v_proj,
                    out_proj,
                    q_layernorm,
                    k_layernorm,
                    ffn_norm,
                    w1,
                    w3,
                    w2,
                });
                conv_states.push(Vec::new());
            } else {
                let in_proj = get_linear(&format!("{}.conv.in_proj", prefix))?;
                let conv = sf.tensors.get(&format!("{}.conv.conv.weight", prefix))
                    .ok_or_else(|| format!("missing {}.conv.conv.weight", prefix))?
                    .clone();
                let out_proj = get_linear(&format!("{}.conv.out_proj", prefix))?;

                layers.push(Lfm2LayerWeights::Conv {
                    operator_norm,
                    in_proj,
                    conv,
                    out_proj,
                    ffn_norm,
                    w1,
                    w3,
                    w2,
                });
                let state_len = dim * (lfm_config.conv_L_cache.saturating_sub(1));
                conv_states.push(vec![0.0f32; state_len]);
            }
        }

        // Allocate working buffers
        let x = vec![0.0f32; dim];
        let x_norm = vec![0.0f32; dim];
        let logits = vec![0.0f32; vocab_size];

        let bcx_buf = vec![0.0f32; 3 * dim];

        let q = vec![0.0f32; n_heads * head_dim];
        let k = vec![0.0f32; n_kv_heads * head_dim];
        let v = vec![0.0f32; n_kv_heads * head_dim];
        let attn_scores = vec![0.0f32; effective_max_seq_len];
        let head_out = vec![0.0f32; n_heads * head_dim];
        let attn_out = vec![0.0f32; dim];

        let kv_cache_len = n_layers * n_kv_heads * effective_max_seq_len * head_dim;
        let k_cache = AnonBuffer::new(kv_cache_len);
        let v_cache = AnonBuffer::new(kv_cache_len);

        let actual_hidden_dim = layers.first().map(|l| match l {
            Lfm2LayerWeights::Conv { w1, .. } => w1.rows(),
            Lfm2LayerWeights::Attention { w1, .. } => w1.rows(),
        }).unwrap_or(hidden_dim);

        let ffn_gate = vec![0.0f32; actual_hidden_dim];
        let ffn_up = vec![0.0f32; actual_hidden_dim];
        let ffn_out = vec![0.0f32; dim];

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let (metal_engine, metal_mmap_buf) = {
            if let Some(engine) = MetalEngine::new() {
                let mmap_buf = sf.mmap.get_or_create_metal_buffer(&engine);
                (Some(engine), mmap_buf)
            } else {
                (None, None)
            }
        };


        Ok(Self {
            config,
            lfm_config,
            sf,
            embed_tokens,
            embedding_norm,
            layers,

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_engine,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_mmap_buf,

            x,
            x_norm,
            logits,

            bcx_buf,
            conv_states,

            q,
            k,
            v,
            attn_scores,
            head_out,
            attn_out,
            k_cache,
            v_cache,

            ffn_gate,
            ffn_up,
            ffn_out,

            max_seq_len: effective_max_seq_len,
            sliding_window: 0,
            sink_tokens: 4,
            gpu_layers: n_layers,
        })
    }

    pub fn reset_cache(&mut self) {
        for state in &mut self.conv_states {
            state.fill(0.0f32);
        }
        self.k_cache.fill(0.0f32);
        self.v_cache.fill(0.0f32);
    }

    /// Configure how many layers execute dense projections on GPU.
    /// CPU still owns attention, convolutions, normalization, and sampling.
    pub fn set_gpu_layers(&mut self, n: usize) {
        self.gpu_layers = n.min(self.layers.len());
    }

    pub fn compact_kv_cache(&mut self, _cur_pos: usize, _sink: usize, _drop: usize) -> usize {
        self.reset_cache();
        0
    }

    // Single vector-matrix multiply: y = W * x
    pub fn compute_matvec(
        mmap: &super::mmap::MmapFile,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        metal_engine: Option<&MetalEngine>,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        metal_mmap_buf: Option<&MetalBuffer>,
        lt: &LinearTensor,
        x: &[f32],
        y: &mut [f32],
    ) {
        match lt {
            LinearTensor::Bf16(tv) => {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                if let (Some(engine), Some(m_buf)) = (metal_engine, metal_mmap_buf) {
                    engine.gemv_bf16(m_buf, tv.data_offset, x, y, tv.shape[0], tv.shape[1]);
                    return;
                }

                // Multi-threaded CPU vectorized BF16 GEMV across all CPU P-cores
                let slice = mmap.slice(tv.data_offset, tv.size_bytes).unwrap();
                let rows = tv.shape[0];
                let cols = tv.shape[1];
                super::quant::gemv_bf16(slice, x, &mut y[..rows], rows, cols);
            }
            LinearTensor::Mlx4 { weight, scales, biases, group_size } => {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                if let (Some(engine), Some(m_buf)) = (metal_engine, metal_mmap_buf) {
                    let rows = weight.shape[0];
                    let cols = x.len();
                    engine.gemv_mlx4(
                        m_buf,
                        weight.data_offset,
                        m_buf,
                        scales.data_offset,
                        m_buf,
                        biases.data_offset,
                        x,
                        y,
                        rows,
                        cols,
                        *group_size,
                        true,
                    );
                    return;
                }

                // Multi-threaded CPU vectorized MLX 4-bit affine GEMV across all CPU P-cores
                let w_slice = mmap.slice(weight.data_offset, weight.size_bytes).unwrap();
                let s_slice = mmap.slice(scales.data_offset, scales.size_bytes).unwrap();
                let b_slice = mmap.slice(biases.data_offset, biases.size_bytes).unwrap();
                let rows = weight.shape[0];
                let cols = x.len();
                super::quant::gemv_mlx4(w_slice, s_slice, b_slice, x, &mut y[..rows], rows, cols, *group_size, true);
            }
        }
    }

    // RMSNorm: y = (x / rms) * w
    pub fn compute_rmsnorm(mmap: &super::mmap::MmapFile, norm_eps: f32, x: &[f32], tv: &SafeTensorView, y: &mut [f32]) {
        let dim = x.len();
        let slice = mmap.slice(tv.data_offset, tv.size_bytes).unwrap();
        let ptr = slice.as_ptr();

        let mut sum_sq = 0.0f32;
        for &v in x {
            sum_sq += v * v;
        }
        let rms = (sum_sq / dim as f32 + norm_eps).sqrt().recip();

        for i in 0..dim {
            let w = unsafe { read_u16_unaligned(ptr, i) };
            y[i] = x[i] * rms * bf16_to_f32(w);
        }
    }

    // Head RMSNorm (used for Q and K normalization in Attention layers)
    pub fn head_rmsnorm(x: &mut [f32], w_ptr: *const u8, head_dim: usize, eps: f32) {
        let mut sum_sq = 0.0f32;
        for &v in x.iter() {
            sum_sq += v * v;
        }
        let rms = (sum_sq / head_dim as f32 + eps).sqrt().recip();
        for i in 0..head_dim {
            let w = unsafe { read_u16_unaligned(w_ptr, i) };
            x[i] = x[i] * rms * bf16_to_f32(w);
        }
    }

    // Forward pass for single token at position `pos`
    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        let dim = self.config.dim;
        let tok = token as usize;

        // 1. Token embedding lookup
        match &self.embed_tokens {
            LinearTensor::Bf16(ref tv) => {
                let embed_slice = self.sf.mmap.slice(tv.data_offset, tv.size_bytes).unwrap();
                let row_bytes = dim * 2;
                let row_ptr = unsafe { embed_slice.as_ptr().add(tok * row_bytes) };
                for i in 0..dim {
                    let raw = unsafe { read_u16_unaligned(row_ptr, i) };
                    self.x[i] = bf16_to_f32(raw);
                }
            }
            LinearTensor::Mlx4 { ref weight, ref scales, ref biases, group_size } => {
                let w_slice = self.sf.mmap.slice(weight.data_offset, weight.size_bytes).unwrap();
                let s_slice = self.sf.mmap.slice(scales.data_offset, scales.size_bytes).unwrap();
                let b_slice = self.sf.mmap.slice(biases.data_offset, biases.size_bytes).unwrap();
                let u32_per_row = dim / 8;
                let groups_per_row = dim / *group_size;
                let row_w_ptr = unsafe { w_slice.as_ptr().add(tok * u32_per_row * 4) };
                let row_s_ptr = unsafe { s_slice.as_ptr().add(tok * groups_per_row * 2) };
                let row_b_ptr = unsafe { b_slice.as_ptr().add(tok * groups_per_row * 2) };

                for u in 0..u32_per_row {
                    let packed = unsafe { read_u32_unaligned(row_w_ptr, u) };
                    let col_base = u * 8;
                    let grp = col_base / *group_size;
                    let scale = bf16_to_f32(unsafe { read_u16_unaligned(row_s_ptr, grp) });
                    let bias = bf16_to_f32(unsafe { read_u16_unaligned(row_b_ptr, grp) });

                    for i in 0..8 {
                        let val_4b = ((packed >> (i * 4)) & 0x0f) as f32;
                        self.x[col_base + i] = val_4b * scale + bias;
                    }
                }
            }
        }

        let n_layers = self.config.n_layers;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let mmap = &self.sf.mmap;
        let norm_eps = self.config.norm_eps;
        // 2. Decoder layers
        for layer_idx in 0..n_layers {
            let layer = self.layers[layer_idx].clone();
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let (m_eng, m_buf) = if layer_idx < self.gpu_layers {
                (self.metal_engine.as_ref(), self.metal_mmap_buf.as_deref())
            } else {
                (None, None)
            };

            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            let layer_idx = layer_idx;
            match layer {
                Lfm2LayerWeights::Conv {
                    operator_norm,
                    in_proj,
                    conv,
                    out_proj,
                    ffn_norm,
                    w1,
                    w3,
                    w2,
                } => {
                    // Pre-norm
                    Self::compute_rmsnorm(mmap, norm_eps, &self.x, &operator_norm, &mut self.x_norm);

                    // BCx = in_proj(x_norm) [3 * dim]
                    Self::compute_matvec(
                        mmap,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                        &in_proj,
                        &self.x_norm,
                        &mut self.bcx_buf,
                    );

                    // B = [0..dim], C = [dim..2*dim], x = [2*dim..3*dim]
                    let (b_part, rest) = self.bcx_buf.split_at(dim);
                    let (c_part, x_part) = rest.split_at(dim);

                    let state_buf = &mut self.conv_states[layer_idx];
                    let conv_slice = mmap.slice(conv.data_offset, conv.size_bytes).unwrap();
                    let conv_ptr = conv_slice.as_ptr();

                    // Execute Causal 1D Convolution per channel
                    for c in 0..dim {
                        let u_val = b_part[c] * x_part[c];
                        let w0 = unsafe { read_u16_unaligned(conv_ptr, c * 3 + 0) };
                        let w1 = unsafe { read_u16_unaligned(conv_ptr, c * 3 + 1) };
                        let w2 = unsafe { read_u16_unaligned(conv_ptr, c * 3 + 2) };

                        let s0 = state_buf[c * 2 + 0];
                        let s1 = state_buf[c * 2 + 1];

                        // v = s0 * w0 + s1 * w1 + u * w2
                        let v_conv = s0 * bf16_to_f32(w0) + s1 * bf16_to_f32(w1) + u_val * bf16_to_f32(w2);

                        // Shift state: s0 <- s1, s1 <- u
                        state_buf[c * 2 + 0] = s1;
                        state_buf[c * 2 + 1] = u_val;

                        // Gate C: y = C * v
                        self.attn_out[c] = c_part[c] * v_conv;
                    }

                    // out_proj(y)
                    Self::compute_matvec(
                        mmap,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                        &out_proj,
                        &self.attn_out,
                        &mut self.x_norm,
                    );

                    // Residual: x = x + conv_out
                    for i in 0..dim {
                        self.x[i] += self.x_norm[i];
                    }

                    // FFN Pre-norm
                    Self::compute_rmsnorm(mmap, norm_eps, &self.x, &ffn_norm, &mut self.x_norm);

                    // SwiGLU: w1 (gate), w3 (up), w2 (down)
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let mlx4_gate_up_done = if let (Some(engine), Some(buf)) = (m_eng, m_buf) {
                        if let (
                            LinearTensor::Mlx4 { weight: g_w, scales: g_s, biases: g_b, group_size: g_g },
                            LinearTensor::Mlx4 { weight: u_w, scales: u_s, biases: u_b, group_size: _u_g },
                        ) = (&w1, &w3) {
                            engine.gemv_gate_up_mlx4(
                                buf, g_w.data_offset, g_w.shape[0], buf, g_s.data_offset, buf, g_b.data_offset,
                                buf, u_w.data_offset, u_w.shape[0], buf, u_s.data_offset, buf, u_b.data_offset,
                                &self.x_norm, &mut self.ffn_gate, &mut self.ffn_up,
                                dim, *g_g, true,
                            );
                            true
                        } else { false }
                    } else { false };
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let mlx4_gate_up_done = false;
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let used_batched_gate_up = if mlx4_gate_up_done { false } else if let (Some(engine), Some(buf), LinearTensor::Bf16(g_tv), LinearTensor::Bf16(u_tv)) = (m_eng, m_buf, &w1, &w3) {
                        engine.gemv_gate_up_bf16(
                            buf,
                            g_tv.data_offset,
                            u_tv.data_offset,
                            &self.x_norm,
                            &mut self.ffn_gate,
                            &mut self.ffn_up,
                            w1.rows(),
                            dim,
                        );
                        true
                    } else {
                        false
                    };
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let used_batched_gate_up = false;

                    if !used_batched_gate_up {
                        Self::compute_matvec(
                            mmap,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                            &w1,
                            &self.x_norm,
                            &mut self.ffn_gate,
                        );
                        Self::compute_matvec(
                            mmap,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                            &w3,
                            &self.x_norm,
                            &mut self.ffn_up,
                        );
                    }

                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    if let Some(engine) = m_eng {
                        engine.swiglu(&mut self.ffn_gate, &self.ffn_up);
                    } else {
                        super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);
                    }
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);

                    Self::compute_matvec(
                        mmap,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                        &w2,
                        &self.ffn_gate,
                        &mut self.ffn_out,
                    );

                    // Residual: x = x + ffn_out
                    for i in 0..dim {
                        self.x[i] += self.ffn_out[i];
                    }
                }
                Lfm2LayerWeights::Attention {
                    operator_norm,
                    q_proj,
                    k_proj,
                    v_proj,
                    out_proj,
                    q_layernorm,
                    k_layernorm,
                    ffn_norm,
                    w1,
                    w3,
                    w2,
                } => {
                    // Pre-norm
                    Self::compute_rmsnorm(mmap, norm_eps, &self.x, &operator_norm, &mut self.x_norm);

                    // Q, K, V projections
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let mlx4_qkv_done = if let (Some(engine), Some(buf)) = (m_eng, m_buf) {
                        if let (
                            LinearTensor::Mlx4 { weight: q_w, scales: q_s, biases: q_b, group_size: q_g },
                            LinearTensor::Mlx4 { weight: k_w, scales: k_s, biases: k_b, group_size: _k_g },
                            LinearTensor::Mlx4 { weight: v_w, scales: v_s, biases: v_b, group_size: _v_g },
                        ) = (&q_proj, &k_proj, &v_proj) {
                            engine.gemv_qkv_mlx4(
                                buf, q_w.data_offset, q_w.shape[0], buf, q_s.data_offset, buf, q_b.data_offset,
                                buf, k_w.data_offset, k_w.shape[0], buf, k_s.data_offset, buf, k_b.data_offset,
                                buf, v_w.data_offset, v_w.shape[0], buf, v_s.data_offset, buf, v_b.data_offset,
                                &self.x_norm, &mut self.q, &mut self.k, &mut self.v,
                                dim, *q_g, true,
                            );
                            true
                        } else { false }
                    } else { false };
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let mlx4_qkv_done = false;
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let used_batched_qkv = if mlx4_qkv_done { false } else if let (Some(engine), Some(buf), LinearTensor::Bf16(q_tv), LinearTensor::Bf16(k_tv), LinearTensor::Bf16(v_tv)) = (m_eng, m_buf, &q_proj, &k_proj, &v_proj) {
                        engine.gemv_qkv_bf16(
                            buf,
                            q_tv.data_offset,
                            k_tv.data_offset,
                            v_tv.data_offset,
                            &self.x_norm,
                            &mut self.q,
                            &mut self.k,
                            &mut self.v,
                            q_proj.rows(),
                            k_proj.rows(),
                            v_proj.rows(),
                            dim,
                        );
                        true
                    } else {
                        false
                    };
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let used_batched_qkv = false;

                    if !used_batched_qkv {
                        Self::compute_matvec(
                            mmap,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                            &q_proj,
                            &self.x_norm,
                            &mut self.q,
                        );
                        Self::compute_matvec(
                            mmap,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                            &k_proj,
                            &self.x_norm,
                            &mut self.k,
                        );
                        Self::compute_matvec(
                            mmap,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                            &v_proj,
                            &self.x_norm,
                            &mut self.v,
                        );
                    }

                    // Q & K Layernorm
                    let q_ln_slice = mmap.slice(q_layernorm.data_offset, q_layernorm.size_bytes).unwrap();
                    let q_ptr = q_ln_slice.as_ptr();
                    for h in 0..n_heads {
                        let q_head = &mut self.q[h * head_dim..(h + 1) * head_dim];
                        Self::head_rmsnorm(q_head, q_ptr, head_dim, norm_eps);
                    }

                    let k_ln_slice = mmap.slice(k_layernorm.data_offset, k_layernorm.size_bytes).unwrap();
                    let k_ptr = k_ln_slice.as_ptr();
                    for kv in 0..n_kv_heads {
                        let k_head = &mut self.k[kv * head_dim..(kv + 1) * head_dim];
                        Self::head_rmsnorm(k_head, k_ptr, head_dim, norm_eps);
                    }

                    // Rotary Position Embedding (RoPE) — split-half
                    let dim_half = head_dim / 2;
                    let freq_base = self.config.rope_freq_base;

                    for h in 0..n_heads {
                        let q_head = &mut self.q[h * head_dim..(h + 1) * head_dim];
                        for j in 0..dim_half {
                            let theta = pos as f32 / freq_base.powf((2 * j) as f32 / head_dim as f32);
                            let cos = theta.cos();
                            let sin = theta.sin();
                            let q0 = q_head[j];
                            let q1 = q_head[j + dim_half];
                            q_head[j] = q0 * cos - q1 * sin;
                            q_head[j + dim_half] = q1 * cos + q0 * sin;
                        }
                    }

                    for kv in 0..n_kv_heads {
                        let k_head = &mut self.k[kv * head_dim..(kv + 1) * head_dim];
                        for j in 0..dim_half {
                            let theta = pos as f32 / freq_base.powf((2 * j) as f32 / head_dim as f32);
                            let cos = theta.cos();
                            let sin = theta.sin();
                            let k0 = k_head[j];
                            let k1 = k_head[j + dim_half];
                            k_head[j] = k0 * cos - k1 * sin;
                            k_head[j + dim_half] = k1 * cos + k0 * sin;
                        }
                    }

                    // Store K and V into KV cache
                    let cur_seq = pos.min(self.max_seq_len - 1);
                    for kv in 0..n_kv_heads {
                        let cache_offset = (layer_idx * n_kv_heads + kv) * (self.max_seq_len * head_dim) + cur_seq * head_dim;
                        self.k_cache[cache_offset..cache_offset + head_dim]
                            .copy_from_slice(&self.k[kv * head_dim..(kv + 1) * head_dim]);
                        self.v_cache[cache_offset..cache_offset + head_dim]
                            .copy_from_slice(&self.v[kv * head_dim..(kv + 1) * head_dim]);
                    }

                    // Attention computation across all heads
                    let scale = (head_dim as f32).sqrt().recip();
                    let kv_group_size = n_heads / n_kv_heads;

                    for h in 0..n_heads {
                        let kv_idx = h / kv_group_size;
                        let q_head = &self.q[h * head_dim..(h + 1) * head_dim];
                        let kv_base = (layer_idx * n_kv_heads + kv_idx) * (self.max_seq_len * head_dim);

                        let seq_len = cur_seq + 1;
                        let mut max_score = f32::NEG_INFINITY;

                        for t in 0..seq_len {
                            let k_t = &self.k_cache[kv_base + t * head_dim..kv_base + (t + 1) * head_dim];
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_head[d] * k_t[d];
                            }
                            let score = dot * scale;
                            self.attn_scores[t] = score;
                            if score > max_score {
                                max_score = score;
                            }
                        }

                        // Softmax
                        let mut sum_exp = 0.0f32;
                        for t in 0..seq_len {
                            let exp = (self.attn_scores[t] - max_score).exp();
                            self.attn_scores[t] = exp;
                            sum_exp += exp;
                        }
                        let inv_sum = sum_exp.recip();
                        for t in 0..seq_len {
                            self.attn_scores[t] *= inv_sum;
                        }

                        // Weighted sum of V
                        let h_out = &mut self.head_out[h * head_dim..(h + 1) * head_dim];
                        h_out.fill(0.0f32);
                        for t in 0..seq_len {
                            let weight = self.attn_scores[t];
                            let v_t = &self.v_cache[kv_base + t * head_dim..kv_base + (t + 1) * head_dim];
                            for d in 0..head_dim {
                                h_out[d] += weight * v_t[d];
                            }
                        }
                    }

                    // Attention output projection
                    Self::compute_matvec(
                        mmap,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                        &out_proj,
                        &self.head_out,
                        &mut self.attn_out,
                    );

                    // Residual: x = x + attn_out
                    for i in 0..dim {
                        self.x[i] += self.attn_out[i];
                    }

                    // FFN Pre-norm
                    Self::compute_rmsnorm(mmap, norm_eps, &self.x, &ffn_norm, &mut self.x_norm);

                    // SwiGLU: w1 (gate), w3 (up), w2 (down)
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let mlx4_gate_up_done = if let (Some(engine), Some(buf)) = (m_eng, m_buf) {
                        if let (
                            LinearTensor::Mlx4 { weight: g_w, scales: g_s, biases: g_b, group_size: g_g },
                            LinearTensor::Mlx4 { weight: u_w, scales: u_s, biases: u_b, group_size: _ },
                        ) = (&w1, &w3) {
                            engine.gemv_gate_up_mlx4(
                                buf, g_w.data_offset, g_w.shape[0], buf, g_s.data_offset, buf, g_b.data_offset,
                                buf, u_w.data_offset, u_w.shape[0], buf, u_s.data_offset, buf, u_b.data_offset,
                                &self.x_norm, &mut self.ffn_gate, &mut self.ffn_up,
                                dim, *g_g, true,
                            );
                            true
                        } else { false }
                    } else { false };
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let mlx4_gate_up_done = false;
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let used_batched_gate_up = if mlx4_gate_up_done { false } else if let (Some(engine), Some(buf), LinearTensor::Bf16(g_tv), LinearTensor::Bf16(u_tv)) = (m_eng, m_buf, &w1, &w3) {
                        engine.gemv_gate_up_bf16(
                            buf,
                            g_tv.data_offset,
                            u_tv.data_offset,
                            &self.x_norm,
                            &mut self.ffn_gate,
                            &mut self.ffn_up,
                            w1.rows(),
                            dim,
                        );
                        true
                    } else {
                        false
                    };
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let used_batched_gate_up = false;

                    if !used_batched_gate_up {
                        Self::compute_matvec(
                            mmap,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                            &w1,
                            &self.x_norm,
                            &mut self.ffn_gate,
                        );
                        Self::compute_matvec(
                            mmap,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                            &w3,
                            &self.x_norm,
                            &mut self.ffn_up,
                        );
                    }

                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    if let Some(engine) = m_eng {
                        engine.swiglu(&mut self.ffn_gate, &self.ffn_up);
                    } else {
                        super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);
                    }
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);

                    Self::compute_matvec(
                        mmap,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
                        #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
                        &w2,
                        &self.ffn_gate,
                        &mut self.ffn_out,
                    );

                    // Residual: x = x + ffn_out
                    for i in 0..dim {
                        self.x[i] += self.ffn_out[i];
                    }
                }
            }
        }

        // 3. Final embedding norm
        let norm_tensor = self.embedding_norm.clone();
        Self::compute_rmsnorm(mmap, norm_eps, &self.x, &norm_tensor, &mut self.x_norm);

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let (m_eng, m_buf) = if self.gpu_layers > 0 {
            (self.metal_engine.as_ref(), self.metal_mmap_buf.as_deref())
        } else {
            (None, None)
        };


        // 4. Logits: tied embedding lm_head = embed_tokens @ x_norm
        Self::compute_matvec(
            mmap,
            #[cfg(any(target_os = "macos", target_os = "ios"))] m_eng,
            #[cfg(any(target_os = "macos", target_os = "ios"))] m_buf,
            &self.embed_tokens,
            &self.x_norm,
            &mut self.logits,
        );

        &self.logits
    }

    pub fn forward_prefill(&mut self, token: u32, pos: usize) {
        self.forward(token, pos);
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], start_pos: usize) {
        if tokens.len() >= LFM2_BATCH_MIN {
            self.forward_batched(tokens, start_pos);
            return;
        }
        for (i, &t) in tokens.iter().enumerate() {
            self.forward_prefill(t, start_pos + i);
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        let n = tokens.len();
        if n == 0 {
            return &self.logits;
        }
        if n >= LFM2_BATCH_MIN {
            self.forward_batched(tokens, start_pos);
            return &self.logits;
        }
        for (i, &t) in tokens[..n - 1].iter().enumerate() {
            self.forward_prefill(t, start_pos + i);
        }
        self.forward(tokens[n - 1], start_pos + n - 1)
    }
}

/// Minimum tokens before the batched prefill path pays for its scratch.
pub(crate) const LFM2_BATCH_MIN: usize = 8;

// ─── Batched CPU prefill ─────────────────────────────────────────────
//
// The per-token loop re-streams every weight matrix once per token, so a
// T-token prompt pays T full passes over the model (~0.9 GB on liquid-1.3B
// each). These helpers reorder the work per layer: one weight stream is
// reused for all T activations (row-panel GEMM), while conv, RoPE, KV
// write and attention stay token-sequential because that is where their
// state lives.

// Manual Clone/Copy without a `T: Copy` bound: the wrapper must stay
// copyable no matter what it points at.
struct SendPtr<T>(*const T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        SendPtr(self.0)
    }
}
impl<T> Copy for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

struct SendMutPtr<T>(*mut T);
impl<T> Clone for SendMutPtr<T> {
    fn clone(&self) -> Self {
        SendMutPtr(self.0)
    }
}
impl<T> Copy for SendMutPtr<T> {}
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

thread_local! {
    static ROW_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// f32 dot with four independent 4-lane FMA chains (16 floats/iter): keeps
/// the NEON pipes fed instead of serializing on one accumulator chain.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_f32_neon(a: *const f32, b: *const f32, n: usize) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a.add(i + 4)), vld1q_f32(b.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(a.add(i + 8)), vld1q_f32(b.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(a.add(i + 12)), vld1q_f32(b.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
            i += 4;
        }
        let mut sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut tmp = [0.0f32; 4];
        vst1q_f32(tmp.as_mut_ptr(), sum);
        let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        while i < n {
            s += *a.add(i) * *b.add(i);
            i += 1;
        }
        s
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        acc += va * vb;
    }
    acc
}

/// 4 output rows × 4 tokens register-tiled kernel: for each 4-float k chunk
/// it loads 4 row vectors + 4 token vectors once and issues 16 FMAs —
/// 0.125 loads/MAC vs 0.5 for the dot-based inner loop.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn gemm_tile4x4_neon(w: *const f32, x: *const f32, n: usize, out: &mut [f32; 16]) {
    use std::arch::aarch64::*;
    unsafe {
        let (mut c00, mut c01, mut c02, mut c03) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let (mut c10, mut c11, mut c12, mut c13) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let (mut c20, mut c21, mut c22, mut c23) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let (mut c30, mut c31, mut c32, mut c33) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let (x0, x1, x2, x3) = (x, x.add(n), x.add(2 * n), x.add(3 * n));
        let (w0, w1, w2, w3) = (w, w.add(n), w.add(2 * n), w.add(3 * n));
        let mut i = 0usize;
        while i + 4 <= n {
            let w0v = vld1q_f32(w0.add(i));
            let w1v = vld1q_f32(w1.add(i));
            let w2v = vld1q_f32(w2.add(i));
            let w3v = vld1q_f32(w3.add(i));
            let x0v = vld1q_f32(x0.add(i));
            let x1v = vld1q_f32(x1.add(i));
            let x2v = vld1q_f32(x2.add(i));
            let x3v = vld1q_f32(x3.add(i));
            c00 = vfmaq_f32(c00, w0v, x0v);
            c01 = vfmaq_f32(c01, w0v, x1v);
            c02 = vfmaq_f32(c02, w0v, x2v);
            c03 = vfmaq_f32(c03, w0v, x3v);
            c10 = vfmaq_f32(c10, w1v, x0v);
            c11 = vfmaq_f32(c11, w1v, x1v);
            c12 = vfmaq_f32(c12, w1v, x2v);
            c13 = vfmaq_f32(c13, w1v, x3v);
            c20 = vfmaq_f32(c20, w2v, x0v);
            c21 = vfmaq_f32(c21, w2v, x1v);
            c22 = vfmaq_f32(c22, w2v, x2v);
            c23 = vfmaq_f32(c23, w2v, x3v);
            c30 = vfmaq_f32(c30, w3v, x0v);
            c31 = vfmaq_f32(c31, w3v, x1v);
            c32 = vfmaq_f32(c32, w3v, x2v);
            c33 = vfmaq_f32(c33, w3v, x3v);
            i += 4;
        }
        let mut lanes = [0.0f32; 4];
        let reduces: [f32; 16] = [
            {
                vst1q_f32(lanes.as_mut_ptr(), c00);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c01);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c02);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c03);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c10);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c11);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c12);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c13);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c20);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c21);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c22);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c23);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c30);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c31);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c32);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
            {
                vst1q_f32(lanes.as_mut_ptr(), c33);
                lanes[0] + lanes[1] + lanes[2] + lanes[3]
            },
        ];
        // out[ri * 4 + t] = w_ri · x_t
        for ri in 0..4 {
            for t in 0..4 {
                out[ri * 4 + t] = reduces[ri * 4 + t];
            }
        }
        while i < n {
            for ri in 0..4 {
                let wv = *w.add(ri * n + i);
                out[ri * 4] += wv * *x0.add(i);
                out[ri * 4 + 1] += wv * *x1.add(i);
                out[ri * 4 + 2] += wv * *x2.add(i);
                out[ri * 4 + 3] += wv * *x3.add(i);
            }
            i += 1;
        }
    }
}

/// Prefill wants the topology's fast cores only: adding the efficiency
/// cores to a row-split barrier makes the P-cores idle waiting on the slow
/// tail (measured ~2x slower end to end when they join the split).
#[cfg(target_os = "macos")]
unsafe fn mac_p_core_count() -> Option<usize> {
    extern "C" {
        fn sysctlbyname(
            name: *const u8,
            oldp: *mut i32,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    let mut val: i32 = 0;
    let mut size: usize = std::mem::size_of::<i32>();
    let name = b"hw.perflevel0.physicalcpu\0";
    let rc = unsafe { sysctlbyname(name.as_ptr(), &mut val, &mut size, std::ptr::null_mut(), 0) };
    if rc == 0 && val > 0 {
        Some(val as usize)
    } else {
        None
    }
}

fn prefill_thread_count() -> usize {
    #[cfg(target_os = "macos")]
    if let Some(p) = unsafe { mac_p_core_count() } {
        return p;
    }
std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Dequantize row `row` of a linear tensor into `out` (exactly what the
/// per-token GEMV and embed lookup do, byte-for-byte semantics).
fn dequant_linear_row(
    lt: &LinearTensor,
    mmap: &super::mmap::MmapFile,
    row: usize,
    cols: usize,
    out: &mut [f32],
) {
    match lt {
        LinearTensor::Bf16(tv) => {
            let slice = mmap.slice(tv.data_offset, tv.size_bytes).unwrap();
            let base = slice.as_ptr();
            unsafe {
                let row_ptr = base.add(row * cols * 2);
                for i in 0..cols {
                    out[i] = bf16_to_f32(read_u16_unaligned(row_ptr, i));
                }
            }
        }
        LinearTensor::Mlx4 { weight, scales, biases, group_size } => {
            let w_slice = mmap.slice(weight.data_offset, weight.size_bytes).unwrap();
            let s_slice = mmap.slice(scales.data_offset, scales.size_bytes).unwrap();
            let b_slice = mmap.slice(biases.data_offset, biases.size_bytes).unwrap();
            let u32_per_row = cols / 8;
            let groups_per_row = cols / *group_size;
            unsafe {
                let wp = w_slice.as_ptr().add(row * u32_per_row * 4);
                let sp = s_slice.as_ptr().add(row * groups_per_row * 2);
                let bp = b_slice.as_ptr().add(row * groups_per_row * 2);
                let mut scale = 0.0f32;
                let mut bias = 0.0f32;
                let mut cur_grp = usize::MAX;
                for u in 0..u32_per_row {
                    let packed = read_u32_unaligned(wp, u);
                    let c_base = u * 8;
                    let grp = c_base / *group_size;
                    if grp != cur_grp {
                        cur_grp = grp;
                        scale = bf16_to_f32(read_u16_unaligned(sp, grp));
                        bias = bf16_to_f32(read_u16_unaligned(bp, grp));
                    }
                    for i in 0..8 {
                        let nib = ((packed >> (i * 4)) & 0x0f) as f32;
                        out[c_base + i] = nib * scale + bias;
                    }
                }
            }
        }
    }
}

    /// y[t, :] = W · x[t, :] for all tokens, weight row read once per call.
/// x must be token-major [t_count, cols]; y token-major [t_count, rows].
static GEMM_PROF_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GEMM_PROF_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn prof_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SOFUU_LFM2_PROF").is_ok())
}

fn gemm_batched(
    lt: &LinearTensor,
    mmap: &super::mmap::MmapFile,
    x: &[f32],
    t_count: usize,
    y: &mut [f32],
    threads: usize,
) {
    let rows = lt.rows();
    let cols = lt.cols();
    debug_assert_eq!(x.len(), t_count * cols);
    debug_assert_eq!(y.len(), t_count * rows);
    if t_count == 0 || rows == 0 {
        return;
    }
    let lt_ptr = SendPtr(lt as *const LinearTensor);
    let mmap_ptr = SendPtr(mmap as *const super::mmap::MmapFile);
    let x_ptr = SendPtr(x.as_ptr());
    let y_ptr = SendMutPtr(y.as_mut_ptr());
    let t_start = if prof_enabled() { Some(std::time::Instant::now()) } else { None };
    const ROW_PANEL: usize = 16;
    let panels = rows.div_ceil(ROW_PANEL);
    super::dynamic_ffn::parallel_for(0, panels, threads, move |p0, p1| unsafe {
        run_gemm_panels(lt_ptr, mmap_ptr, x_ptr, y_ptr, t_count, rows, cols, p0, p1);
    });
    if let Some(t0) = t_start {
        GEMM_PROF_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        GEMM_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if std::env::var("SOFUU_LFM2_PROF").as_deref() == Ok("verbose") {
            eprintln!("[gemm] rows={} cols={} T={} {:.1}ms", rows, cols, t_count, t0.elapsed().as_secs_f64() * 1e3);
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_gemm_panels(
    lt_ptr: SendPtr<LinearTensor>,
    mmap_ptr: SendPtr<super::mmap::MmapFile>,
    x_ptr: SendPtr<f32>,
    y_ptr: SendMutPtr<f32>,
    t_count: usize,
    rows: usize,
    cols: usize,
    p0: usize,
    p1: usize,
) {
    if p0 >= p1 {
        return;
    }
    const ROW_PANEL: usize = 16;
    let lt: &LinearTensor = unsafe { &*lt_ptr.0 };
    let mmap: &super::mmap::MmapFile = unsafe { &*mmap_ptr.0 };
    let x: &[f32] = unsafe { std::slice::from_raw_parts(x_ptr.0, t_count * cols) };
    let y: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(y_ptr.0, t_count * rows) };
    ROW_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(ROW_PANEL * cols, 0.0f32);
        for p in p0..p1 {
            let row0 = p * ROW_PANEL;
            let panel_rows = ROW_PANEL.min(rows - row0);
            for ri in 0..panel_rows {
                dequant_linear_row(lt, mmap, row0 + ri, cols, &mut buf[ri * cols..(ri + 1) * cols]);
            }
            // Register-blocked sweep: 4 rows × 4 tokens per tile; the k
            // stream is shared by all 16 FMAs of a tile.
            let mut t = 0usize;
            #[cfg(target_arch = "aarch64")]
            {
                let mut out = [0.0f32; 16];
                while t + 4 <= t_count {
                    let mut ri = 0usize;
                    while ri + 4 <= panel_rows {
                        unsafe {
                            gemm_tile4x4_neon(
                                buf.as_ptr().add(ri * cols),
                                x.as_ptr().add(t * cols),
                                cols,
                                &mut out,
                            );
                        }
                        for ti in 0..4 {
                            for tk in 0..4 {
                                y[(t + tk) * rows + row0 + ri + ti] = out[ti * 4 + tk];
                            }
                        }
                        ri += 4;
                    }
                    for r1 in ri..panel_rows {
                        let xr_base = t * cols;
                        for tk in 0..4 {
                            #[cfg(target_arch = "aarch64")]
                            let acc = unsafe {
                                dot_f32_neon(buf.as_ptr().add(r1 * cols), x.as_ptr().add(xr_base + tk * cols), cols)
                            };
                            #[cfg(not(target_arch = "aarch64"))]
                            let acc = 0.0f32;
                            y[(t + tk) * rows + row0 + r1] = acc;
                        }
                    }
                    t += 4;
                }
            }
            while t < t_count {
                let xr = &x[t * cols..(t + 1) * cols];
                for ri in 0..panel_rows {
                    #[cfg(target_arch = "aarch64")]
                    let acc = unsafe { dot_f32_neon(buf.as_ptr().add(ri * cols), xr.as_ptr(), cols) };
                    #[cfg(not(target_arch = "aarch64"))]
                    let acc = dot_f32_neon(&buf[ri * cols..(ri + 1) * cols], xr);
                    y[t * rows + row0 + ri] = acc;
                }
                t += 1;
            }
        }
    });
}

/// out[:, :] = rmsnorm(x[:, :]) · w for every token row.
fn batched_rmsnorm(
    x: &[f32],
    t_count: usize,
    dim: usize,
    mmap: &super::mmap::MmapFile,
    tv: &SafeTensorView,
    eps: f32,
    out: &mut [f32],
    threads: usize,
) {
    let slice = mmap.slice(tv.data_offset, tv.size_bytes).unwrap();
    let w_ptr = SendPtr(slice.as_ptr());
    let x_ptr = SendPtr(x.as_ptr());
    let out_ptr = SendMutPtr(out.as_mut_ptr());
    super::dynamic_ffn::parallel_for(0, t_count, threads, move |t0, t1| {
        let (SendPtr(w_ptr), SendPtr(x_ptr), SendMutPtr(out_ptr)) = (w_ptr, x_ptr, out_ptr);
        let x: &[f32] = unsafe { std::slice::from_raw_parts(x_ptr, t_count * dim) };
        let out: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out_ptr, t_count * dim) };
        for t in t0..t1 {
            let row = &x[t * dim..(t + 1) * dim];
            let orow = &mut out[t * dim..(t + 1) * dim];
            let mut sum_sq = 0.0f32;
            for &v in row.iter() {
                sum_sq += v * v;
            }
            let rms = (sum_sq / dim as f32 + eps).sqrt().recip();
            for i in 0..dim {
                let w = unsafe { read_u16_unaligned(w_ptr as *const u8, i) };
                orow[i] = row[i] * rms * bf16_to_f32(w);
            }
        }
    });
}

fn batched_swiglu(gate: &mut [f32], up: &[f32], t_count: usize, hidden: usize, threads: usize) {
    let g_ptr = SendMutPtr(gate.as_mut_ptr());
    let u_ptr = SendPtr(up.as_ptr());
    super::dynamic_ffn::parallel_for(0, t_count, threads, move |t0, t1| {
        let (SendMutPtr(g_ptr), SendPtr(u_ptr)) = (g_ptr, u_ptr);
        for t in t0..t1 {
            let g: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(g_ptr.add(t * hidden), hidden)
            };
            let u: &[f32] = unsafe { std::slice::from_raw_parts(u_ptr.add(t * hidden), hidden) };
            super::quant::swiglu_in_place(g, u);
        }
    });
}

fn batched_add_in_place(x: &mut [f32], add: &[f32]) {
    for (xv, av) in x.iter_mut().zip(add.iter()) {
        *xv += *av;
    }
}

/// Gather token embeddings into token-major `out` (rows dequantized in parallel).
fn embed_tokens_batched(
    lt: &LinearTensor,
    mmap: &super::mmap::MmapFile,
    tokens: &[u32],
    dim: usize,
    out: &mut [f32],
    threads: usize,
) {
    let n = tokens.len();
    if n == 0 {
        return;
    }
    let lt_ptr = SendPtr(lt as *const LinearTensor);
    let mmap_ptr = SendPtr(mmap as *const super::mmap::MmapFile);
    let toks_ptr = SendPtr(tokens.as_ptr());
    let out_ptr = SendMutPtr(out.as_mut_ptr());
    super::dynamic_ffn::parallel_for(0, n, threads, move |t0, t1| {
        let (SendPtr(lt_ptr), SendPtr(mmap_ptr), SendPtr(toks_ptr), SendMutPtr(out_ptr)) =
            (lt_ptr, mmap_ptr, toks_ptr, out_ptr);
        let lt: &LinearTensor = unsafe { &*lt_ptr };
        let mmap: &super::mmap::MmapFile = unsafe { &*mmap_ptr };
        let tokens: &[u32] = unsafe { std::slice::from_raw_parts(toks_ptr, n) };
        let out: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out_ptr, n * dim) };
        for t in t0..t1 {
            let row = tokens[t] as usize;
            dequant_linear_row(lt, mmap, row, dim, &mut out[t * dim..(t + 1) * dim]);
        }
    });
}

/// SwiGLU FFN on a token batch: gate = w1·x, up = w3·x, down = w2·swiglu(gate, up).
#[allow(clippy::too_many_arguments)]
fn batched_ffn(
    w1: &LinearTensor,
    w3: &LinearTensor,
    w2: &LinearTensor,
    mmap: &super::mmap::MmapFile,
    x: &[f32],
    t_count: usize,
    gate: &mut Vec<f32>,
    up: &mut Vec<f32>,
    down: &mut Vec<f32>,
    threads: usize,
) {
    let hidden = w1.rows();
    let needed_gate = t_count * hidden;
    let needed_down = t_count * w2.rows();
    if gate.len() < needed_gate {
        gate.resize(needed_gate, 0.0f32);
    }
    if up.len() < needed_gate {
        up.resize(needed_gate, 0.0f32);
    }
    if down.len() < needed_down {
        down.resize(needed_down, 0.0f32);
    }
    gemm_batched(w1, mmap, x, t_count, &mut gate[..needed_gate], threads);
    gemm_batched(w3, mmap, x, t_count, &mut up[..needed_gate], threads);
    batched_swiglu(gate, up, t_count, hidden, threads);
    gemm_batched(w2, mmap, &gate[..needed_gate], t_count, &mut down[..needed_down], threads);
}

impl Lfm2Model {
    /// Batched CPU prefill: every token goes through each layer with one
    /// weight stream per layer instead of one per token. Conv state, KV
    /// cache and per-token attention keep their sequential order; only the
    /// dense projections are reordered into row-panel GEMMs.
    pub fn forward_batched(&mut self, tokens: &[u32], start_pos: usize) {
        let dim = self.config.dim;
        let n_layers = self.config.n_layers;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let t_count = tokens.len();
        let threads = prefill_thread_count();
        let norm_eps = self.config.norm_eps;
        let max_seq_len = self.max_seq_len;
        let freq_base = self.config.rope_freq_base;
        if t_count == 0 {
            return;
        }

        let mut xb = vec![0.0f32; t_count * dim];
        {
            let lt = self.embed_tokens.clone();
            embed_tokens_batched(&lt, &self.sf.mmap, tokens, dim, &mut xb, threads);
        }

        let mut xn = vec![0.0f32; t_count * dim];
        let mut ffn_in = vec![0.0f32; t_count * dim];
        let mut res = vec![0.0f32; 0];
        let mut raw = vec![0.0f32; 0];
        let mut vout = vec![0.0f32; t_count * dim];
        let mut hb = vec![0.0f32; 0];
        let mut qbuf = vec![0.0f32; 0];
        let mut kbuf = vec![0.0f32; 0];
        let mut vbuf = vec![0.0f32; 0];
        let mut gate = vec![0.0f32; 0];
        let mut up = vec![0.0f32; 0];
        let mut down = vec![0.0f32; 0];
        let mut conv_ns = 0u64;
        let mut attn_ns = 0u64;
        let prof = prof_enabled();
        let t_wall = if prof { Some(std::time::Instant::now()) } else { None };

        for layer_idx in 0..n_layers {
            let layer = self.layers[layer_idx].clone();
            match layer {
                Lfm2LayerWeights::Conv { operator_norm, in_proj, conv, out_proj, ffn_norm, w1, w3, w2 } => {
                    batched_rmsnorm(&xb, t_count, dim, &self.sf.mmap, &operator_norm, norm_eps, &mut xn, threads);
                    let raw_rows = in_proj.rows();
                    if raw.len() < t_count * raw_rows {
                        raw.resize(t_count * raw_rows, 0.0f32);
                    }
                    gemm_batched(&in_proj, &self.sf.mmap, &xn, t_count, &mut raw[..t_count * raw_rows], threads);

                    // Causal short conv + gate: token-sequential, dim-scale work
                    let conv_slice = self.sf.mmap.slice(conv.data_offset, conv.size_bytes).unwrap();
                    let conv_ptr = conv_slice.as_ptr();
                    let state_buf: &mut [f32] = &mut self.conv_states[layer_idx];
                    let t_conv = if prof { Some(std::time::Instant::now()) } else { None };
                    for t in 0..t_count {
                        let base = t * raw_rows;
                        for c in 0..dim {
                            let u_val = raw[base + c] * raw[base + 2 * dim + c];
                            let ct0 = unsafe { read_u16_unaligned(conv_ptr, c * 3 + 0) };
                            let ct1 = unsafe { read_u16_unaligned(conv_ptr, c * 3 + 1) };
                            let ct2 = unsafe { read_u16_unaligned(conv_ptr, c * 3 + 2) };
                            let s0 = state_buf[c * 2 + 0];
                            let s1 = state_buf[c * 2 + 1];
                            let v_conv = s0 * bf16_to_f32(ct0) + s1 * bf16_to_f32(ct1) + u_val * bf16_to_f32(ct2);
                            state_buf[c * 2 + 0] = s1;
                            state_buf[c * 2 + 1] = u_val;
                            vout[t * dim + c] = raw[base + dim + c] * v_conv;
                        }
                    }
                    if let Some(t0) = t_conv {
                        conv_ns += t0.elapsed().as_nanos() as u64;
                    }

                    if res.len() < t_count * dim {
                        res.resize(t_count * dim, 0.0f32);
                    }
                    gemm_batched(&out_proj, &self.sf.mmap, &vout[..t_count * dim], t_count, &mut res[..t_count * dim], threads);
                    batched_add_in_place(&mut xb[..t_count * dim], &res[..t_count * dim]);

                    batched_rmsnorm(&xb, t_count, dim, &self.sf.mmap, &ffn_norm, norm_eps, &mut ffn_in, threads);
                    batched_ffn(&w1, &w3, &w2, &self.sf.mmap, &ffn_in, t_count, &mut gate, &mut up, &mut down, threads);
                    batched_add_in_place(&mut xb[..t_count * dim], &down[..t_count * dim]);
                }
                Lfm2LayerWeights::Attention { operator_norm, q_proj, k_proj, v_proj, out_proj, q_layernorm, k_layernorm, ffn_norm, w1, w3, w2 } => {
                    batched_rmsnorm(&xb, t_count, dim, &self.sf.mmap, &operator_norm, norm_eps, &mut xn, threads);
                    let q_rows = q_proj.rows();
                    let k_rows = k_proj.rows();
                    let v_rows = v_proj.rows();
                    if qbuf.len() < t_count * q_rows {
                        qbuf.resize(t_count * q_rows, 0.0f32);
                    }
                    if kbuf.len() < t_count * k_rows {
                        kbuf.resize(t_count * k_rows, 0.0f32);
                    }
                    if vbuf.len() < t_count * v_rows {
                        vbuf.resize(t_count * v_rows, 0.0f32);
                    }
                    gemm_batched(&q_proj, &self.sf.mmap, &xn, t_count, &mut qbuf[..t_count * q_rows], threads);
                    gemm_batched(&k_proj, &self.sf.mmap, &xn, t_count, &mut kbuf[..t_count * k_rows], threads);
                    gemm_batched(&v_proj, &self.sf.mmap, &xn, t_count, &mut vbuf[..t_count * v_rows], threads);
                    if hb.len() < t_count * q_rows {
                        hb.resize(t_count * q_rows, 0.0f32);
                    }

                    let ql_slice = self.sf.mmap.slice(q_layernorm.data_offset, q_layernorm.size_bytes).unwrap();
                    let kl_slice = self.sf.mmap.slice(k_layernorm.data_offset, k_layernorm.size_bytes).unwrap();
                    let q_ln_ptr = ql_slice.as_ptr();
                    let k_ln_ptr = kl_slice.as_ptr();
                    let dim_half = head_dim / 2;

                    let t_att = if prof { Some(std::time::Instant::now()) } else { None };
                    for t in 0..t_count {
                        let pos = start_pos + t;
                        for h in 0..n_heads {
                            let off = t * q_rows + h * head_dim;
                            let q_head = &mut qbuf[off..off + head_dim];
                            Self::head_rmsnorm(q_head, q_ln_ptr, head_dim, norm_eps);
                        }
                        for kv in 0..n_kv_heads {
                            let off = t * k_rows + kv * head_dim;
                            let k_head = &mut kbuf[off..off + head_dim];
                            Self::head_rmsnorm(k_head, k_ln_ptr, head_dim, norm_eps);
                        }
                        for h in 0..n_heads {
                            let off = t * q_rows + h * head_dim;
                            let q_head = &mut qbuf[off..off + head_dim];
                            for j in 0..dim_half {
                                let theta = pos as f32 / freq_base.powf((2 * j) as f32 / head_dim as f32);
                                let cos = theta.cos();
                                let sin = theta.sin();
                                let q0 = q_head[j];
                                let q1 = q_head[j + dim_half];
                                q_head[j] = q0 * cos - q1 * sin;
                                q_head[j + dim_half] = q1 * cos + q0 * sin;
                            }
                        }
                        for kv in 0..n_kv_heads {
                            let off = t * k_rows + kv * head_dim;
                            let k_head = &mut kbuf[off..off + head_dim];
                            for j in 0..dim_half {
                                let theta = pos as f32 / freq_base.powf((2 * j) as f32 / head_dim as f32);
                                let cos = theta.cos();
                                let sin = theta.sin();
                                let k0 = k_head[j];
                                let k1 = k_head[j + dim_half];
                                k_head[j] = k0 * cos - k1 * sin;
                                k_head[j + dim_half] = k1 * cos + k0 * sin;
                            }
                        }

                        let cur_seq = pos.min(max_seq_len - 1);
                        for kv in 0..n_kv_heads {
                            let cache_offset =
                                (layer_idx * n_kv_heads + kv) * (max_seq_len * head_dim) + cur_seq * head_dim;
                            self.k_cache[cache_offset..cache_offset + head_dim]
                                .copy_from_slice(&kbuf[t * k_rows + kv * head_dim..t * k_rows + (kv + 1) * head_dim]);
                            self.v_cache[cache_offset..cache_offset + head_dim]
                                .copy_from_slice(&vbuf[t * v_rows + kv * head_dim..t * v_rows + (kv + 1) * head_dim]);
                        }

                        let scale = (head_dim as f32).sqrt().recip();
                        let kv_group_size = n_heads / n_kv_heads;
                        for h in 0..n_heads {
                            let kv_idx = h / kv_group_size;
                            let q_off = t * q_rows + h * head_dim;
                            let q_head = &qbuf[q_off..q_off + head_dim];
                            let kv_base = (layer_idx * n_kv_heads + kv_idx) * (max_seq_len * head_dim);
                            let seq_len = cur_seq + 1;
                            let mut max_score = f32::NEG_INFINITY;
                            for tt in 0..seq_len {
                                let k_t = &self.k_cache[kv_base + tt * head_dim..kv_base + (tt + 1) * head_dim];
                                let mut dot = 0.0f32;
                                for d in 0..head_dim {
                                    dot += q_head[d] * k_t[d];
                                }
                                let score = dot * scale;
                                self.attn_scores[tt] = score;
                                if score > max_score {
                                    max_score = score;
                                }
                            }
                            let mut sum_exp = 0.0f32;
                            for tt in 0..seq_len {
                                let e = (self.attn_scores[tt] - max_score).exp();
                                self.attn_scores[tt] = e;
                                sum_exp += e;
                            }
                            let inv_sum = sum_exp.recip();
                            for tt in 0..seq_len {
                                self.attn_scores[tt] *= inv_sum;
                            }
                            let h_out = &mut hb[t * q_rows + h * head_dim..t * q_rows + (h + 1) * head_dim];
                            h_out.fill(0.0f32);
                            for tt in 0..seq_len {
                                let weight = self.attn_scores[tt];
                                let v_t = &self.v_cache[kv_base + tt * head_dim..kv_base + (tt + 1) * head_dim];
                                for d in 0..head_dim {
                                    h_out[d] += weight * v_t[d];
                                }
                            }
                        }
                    }
                    if let Some(t0) = t_att {
                        attn_ns += t0.elapsed().as_nanos() as u64;
                    }

                    if res.len() < t_count * dim {
                        res.resize(t_count * dim, 0.0f32);
                    }
                    gemm_batched(&out_proj, &self.sf.mmap, &hb[..t_count * q_rows], t_count, &mut res[..t_count * dim], threads);
                    batched_add_in_place(&mut xb[..t_count * dim], &res[..t_count * dim]);

                    batched_rmsnorm(&xb, t_count, dim, &self.sf.mmap, &ffn_norm, norm_eps, &mut ffn_in, threads);
                    batched_ffn(&w1, &w3, &w2, &self.sf.mmap, &ffn_in, t_count, &mut gate, &mut up, &mut down, threads);
                    batched_add_in_place(&mut xb[..t_count * dim], &down[..t_count * dim]);
                }
            }
        }

        // Final norm + tied-embedding logits for the last token.
        let norm_tensor = self.embedding_norm.clone();
        let last = &xb[(t_count - 1) * dim..t_count * dim];
        Self::compute_rmsnorm(&self.sf.mmap, norm_eps, last, &norm_tensor, &mut self.x_norm);
        Self::compute_matvec(
            &self.sf.mmap,
            #[cfg(any(target_os = "macos", target_os = "ios"))] None,
            #[cfg(any(target_os = "macos", target_os = "ios"))] None,
            &self.embed_tokens,
            &self.x_norm,
            &mut self.logits,
        );

        if let Some(t0) = t_wall {
            let gemm_ms = GEMM_PROF_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            let calls = GEMM_PROF_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[lfm2-prof] t={} wall={:.1}ms gemm={:.1}ms/calls={} conv={:.1}ms attn={:.1}ms other={:.1}ms",
                t_count,
                t0.elapsed().as_secs_f64() * 1e3,
                gemm_ms,
                calls,
                conv_ns as f64 / 1e6,
                attn_ns as f64 / 1e6,
                t0.elapsed().as_secs_f64() * 1e3 - gemm_ms - conv_ns as f64 / 1e6 - attn_ns as f64 / 1e6
            );
        }
    }
}



#[cfg(test)]
mod lfm2_batched_prefill_tests {
    use super::*;

    fn probe_tokens(n: usize) -> Vec<u32> {
        (0..n).map(|i| 64 + ((i * 37 + 11) % 64000) as u32).collect()
    }

    fn last_logits_m(model: &mut Lfm2Model, tokens: &[u32]) -> Vec<f32> {
        // The per-token reference path (pre-batched behaviour).
        for (i, &t) in tokens.iter().enumerate() {
            model.forward(t, i);
        }
        model.logits.clone()
    }

    fn compare_paths(dir: &str, greedy_steps: usize) {
        let mut ref_model = Lfm2Model::from_safetensors_dir(dir, 0).expect("load reference model");
        let mut batch_model = Lfm2Model::from_safetensors_dir(dir, 0).expect("load batched model");
        let tokens = probe_tokens(24);

        let ref_logits = last_logits_m(&mut ref_model, &tokens);
        let batch_logits_holder = batch_model.forward_prefill_all_with_logits(&tokens, 0);
        let batch_logits = batch_logits_holder.to_vec();

        assert_eq!(ref_logits.len(), batch_logits.len());
        let n = ref_logits.len() as f32;
        let mut sum_sq = 0.0f32;
        let mut ref_sq = 0.0f32;
        let mut top_ref: usize = 0;
        let mut top_batch: usize = 0;
        for i in 0..ref_logits.len() {
            let d = ref_logits[i] - batch_logits[i];
            sum_sq += d * d;
            ref_sq += ref_logits[i] * ref_logits[i];
            if ref_logits[i] > ref_logits[top_ref] {
                top_ref = i;
            }
            if batch_logits[i] > batch_logits[top_batch] {
                top_batch = i;
            }
        }
        let rms_rel = (sum_sq / n).sqrt() / (ref_sq / n).sqrt().max(1e-9);
        assert!(
            rms_rel < 2e-2,
            "batched prefill logits diverge from per-token path: rel RMS {:.6}",
            rms_rel
        );
        assert_eq!(top_ref, top_batch, "argmax token differs between paths");

        // Greedy continuation from the shared prefill state must agree.
        let mut tok = top_batch as u32;
        let mut pos = tokens.len();
        for _ in 0..greedy_steps {
            let logits = ref_model.forward(tok, pos).to_vec();
            let logits_b = batch_model.forward(tok, pos).to_vec();
            let mut ra = 0usize;
            let mut ba = 0usize;
            for i in 0..logits.len() {
                if logits[i] > logits[ra] {
                    ra = i;
                }
                if logits_b[i] > logits_b[ba] {
                    ba = i;
                }
            }
            assert_eq!(ra, ba, "greedy decode diverges at pos {}", pos);
            tok = ba as u32;
            pos += 1;
        }
    }

    #[test]
    fn batched_prefill_matches_per_token_bf16() {
        compare_paths("/Users/priyanshuboruah/Downloads/liqiud-350", 8);
    }

    #[test]
    #[ignore = "1.4 GB model load; run explicitly on a calm machine"]
    fn batched_prefill_matches_per_token_mlx4() {
        compare_paths("/Users/priyanshuboruah/Downloads/liquid", 8);
    }

    /// A/B timing for the batched prefill against the per-token loop.
    #[test]
    #[ignore = "release-only timing; run explicitly"]
    fn batched_prefill_speed() {
        let dir = "/Users/priyanshuboruah/Downloads/liquid";
        let tokens = probe_tokens(200);
        let mut m = Lfm2Model::from_safetensors_dir(dir, 0).expect("load model");

        let t0 = std::time::Instant::now();
        {
            let mut loop_model = Lfm2Model::from_safetensors_dir(dir, 0).unwrap();
            loop_model.set_gpu_layers(0); // force CPU: same charter as the batched path
            for (i, &t) in tokens.iter().enumerate() {
                loop_model.forward(t, i);
            }
        }
        let loop_secs = t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        m.forward_batched(&tokens, 0);
        let batch_secs = t1.elapsed().as_secs_f64();

        eprintln!(
            "lfm2 prefill 200 tokens: per-token {:.2} tk/s | batched {:.2} tk/s ({:.2}x)",
            200.0 / loop_secs,
            200.0 / batch_secs,
            loop_secs / batch_secs
        );
    }
}

#[cfg(test)]
mod lfm2_dot_micro {
    use super::*;

    #[test]
    #[ignore = "timing probe; run explicitly"]
    fn dot_f32_neon_throughput_main_thread() {
        let cols = 2048usize;
        let a: Vec<f32> = (0..cols + 1024).map(|i| (i % 7) as f32 * 0.001).collect();
        let b: Vec<f32> = (0..cols).map(|i| (i % 11) as f32 * 0.001).collect();
        let dits = 200_000usize; // 200k dots of 2048 MACs = 0.41 TMAC
        let t0 = std::time::Instant::now();
        let mut sink = 0.0f32;
        for d in 0..dits {
            let off = d % 1024;
            sink += unsafe { dot_f32_neon(a.as_ptr().add(off), b.as_ptr(), cols) };
        }
        let secs = t0.elapsed().as_secs_f64();
        let gmac = dits as f64 * cols as f64 / 1e9;
        eprintln!("dot: {:.2} GMAC/s ({:.1} GFLOP/s), sink {:.3}", gmac / secs, 2.0 * gmac / secs, sink);
    }
}

#[cfg(test)]
mod lfm2_tile_tests {
    use super::*;

    #[test]
    fn tile4x4_matches_reference_dots() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let cols = 512usize;
        let w: Vec<f32> = (0..4 * cols).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let x: Vec<f32> = (0..4 * cols).map(|i| (((i * (i / cols + 3)) % 17) as f32 - 8.0) * 0.02).collect();
        let mut got = [1.0f32; 16];
        unsafe {
            gemm_tile4x4_neon(w.as_ptr(), x.as_ptr(), cols, &mut got);
        }
        for ri in 0..4 {
            for t in 0..4 {
                let ref_dot: f32 = w[ri * cols..(ri + 1) * cols]
                    .iter()
                    .zip(x[t * cols..(t + 1) * cols].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let diff = (got[ri * 4 + t] - ref_dot).abs();
                assert!(
                    diff < 1e-4 + ref_dot.abs() * 1e-4,
                    "tile[{}][{}] {} vs ref {}",
                    ri, t, got[ri * 4 + t], ref_dot
                );
            }
        }
    }
}

#[cfg(test)]
mod lfm2_ceiling_probe {
    use super::*;

    #[test]
    #[ignore = "timing probe; run explicitly"]
    fn tile4x4_gflops_single_thread() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let cols = 2048usize;
        let w: Vec<f32> = (0..4 * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let x: Vec<f32> = (0..4 * cols).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let tiles = 100_000usize; // 100k tiles x 16 dots x 2048 MAC = 3.28 TMAC
        let mut out = [0f32; 16];
        let start = std::time::Instant::now();
        let mut sink = 0f32;
        for t in 0..tiles {
            let off = (t * 4) % (cols - 2048 + 2048); // vary inputs slightly
            let o = (t % 8) as f32;
            let _ = o;
            unsafe {
                gemm_tile4x4_neon(w.as_ptr(), x.as_ptr().add(off.min(cols)), cols, &mut out);
            }
            sink += out[5];
        }
        let secs = start.elapsed().as_secs_f64();
        let gmac = tiles as f64 * 16.0 * cols as f64 / 1e9;
        eprintln!(
            "tile4x4: {:.1} GMAC/s = {:.1} GFLOP/s single-thread (sink {:.4})",
            gmac / secs, 2.0 * gmac / secs, sink
        );
    }
}

#[cfg(test)]
mod lfm2_mt_ceiling_probe {
    use super::*;

    #[test]
    #[ignore = "timing probe; run explicitly"]
    fn tile4x4_gflops_six_threads() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let cols = 2048usize;
        let w: Vec<f32> = (0..4 * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let x: Vec<f32> = (0..4 * cols).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let tiles_per_thread = 100_000usize;
        let mut sink = 0f32;
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            for t in 0..6 {
                let (w, x) = (&w, &x);
                s.spawn(move || {
                    let mut local = 0f32;
                    let mut out = [0f32; 16];
                    for k in 0..tiles_per_thread {
                        let off = (t * 509 + k) % 8;
                        unsafe {
                            gemm_tile4x4_neon(
                                w.as_ptr().add(off & 0),
                                x.as_ptr().add(off * (cols / 8)),
                                cols,
                                &mut out,
                            );
                        }
                        local += out[5];
                    }
                    local
                });
            }
        });
        let secs = start.elapsed().as_secs_f64();
        let gmac = 6.0 * tiles_per_thread as f64 * 16.0 * cols as f64 / 1e9;
        let _ = sink;
        eprintln!(
            "tile4x4 6 threads: {:.1} GMAC/s aggregate = {:.1} GFLOP/s ({:.2}x vs single)",
            gmac / secs,
            2.0 * gmac / secs,
            (gmac / secs) / 42.6
        );
    }
}
