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

// ════════ Phase 2: int8-activation SDOT GEMM for the float-weight fleet ════════
// Opt-in via SOFUU_PRE8=1 (plan §Phase 2). Activations are quantized per
// 128-band once per layer call; each panel dequantizes bf16/MLX4 rows into
// int8 lanes with a per-row per-band scale (one extra pass over the same
// bytes the f32 path already dequantized); the tile kernel accumulates
// int8×int8 SDOT into int32 per band and folds scaled partial sums into f32
// — the two-sided w_scale·a_scale·iacc pattern the ternary path already
// proved stable as `trit_row_dot_q`. Requires armv8.2 sdot (Pi 5 / Apple
// Silicon); other hosts and non-opt-in runs keep the f32 tile4x4 path.
fn pre8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[cfg(target_arch = "aarch64")]
    {
        *ON.get_or_init(|| {
            std::env::var("SOFUU_PRE8").as_deref() == Ok("1")
                && std::arch::is_aarch64_feature_detected!("dotprod")
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        *ON.get_or_init(|| std::env::var("SOFUU_PRE8").as_deref() == Ok("1"))
    }
}

const PRE8_GROUP: usize = 128;

struct ActQ8 {
    q: Vec<i8>,       // [tokens × n], n group-multiple padded, zero-filled
    scales: Vec<f32>, // [tokens × groups]
    tokens: usize,
    n: usize,
    groups: usize,
}

fn quantize_acts_i8(x: &[f32], tokens: usize, cols: usize) -> ActQ8 {
    let groups = cols.div_ceil(PRE8_GROUP);
    let n = groups * PRE8_GROUP;
    let mut q = vec![0i8; tokens * n];
    let mut scales = vec![1f32; tokens * groups];
    for t in 0..tokens {
        for g in 0..groups {
            let g0 = g * PRE8_GROUP;
            let g1 = (g0 + PRE8_GROUP).min(cols);
            let mut amax = 0f32;
            for i in g0..g1 {
                amax = amax.max(x[t * cols + i].abs());
            }
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            scales[t * groups + g] = scale;
            for i in g0..g1 {
                q[t * n + i] = (x[t * cols + i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    ActQ8 { q, scales, tokens, n, groups }
}

thread_local! {
    static WQ_BUF: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
    static WS_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Per-row per-band int8 conversion of the freshly dequantized f32 panel
/// (extra absmax+round pass over the same row bytes the f32 path produced).
/// `stride` = panel row length (the dequantized `cols`, possibly < the padded
/// `n`); the padded tail stays zero and contributes exactly 0 to every dot.
fn quantize_w_panel_i8(row_buf: &[f32], rows: usize, stride: usize, n: usize, groups: usize, wq: &mut [i8], ws: &mut [f32]) {
    for r in 0..rows {
        for g in 0..groups {
            let g0 = g * PRE8_GROUP;
            let mut amax = 0f32;
            for i in g0..g0 + PRE8_GROUP {
                let v = if i < stride { row_buf[r * stride + i] } else { 0.0 };
                amax = amax.max(v.abs());
            }
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            ws[r * groups + g] = scale;
            for i in g0..g0 + PRE8_GROUP {
                let v = if i < stride { row_buf[r * stride + i] } else { 0.0 };
                wq[r * n + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
}

/// 4 weight rows × 2 tokens per tile: per 128-band, 8 live int32 SDOT
/// accumulators (never more, so A76 does not spill), then a two-sided fold
/// `f32 += hsum(acc) · w_scale[r][g] · a_scale[t][g]`. `n` must be a
/// group multiple (callers pad), so the inner chunk loop is exact.
/// One 128-band signed int8 dot — structurally identical to the ternary
/// path's proven `sdot_128_neon` (fresh 16B loads per chunk, one asm
/// statement per chunk, single live accumulator).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::missing_safety_doc)]
unsafe fn sdot_band(w: *const i8, a: *const i8) -> i32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    for k in 0..(PRE8_GROUP / 16) {
        let wv = vld1q_s8(w.add(k * 16));
        let av = vld1q_s8(a.add(k * 16));
        std::arch::asm!(
            "sdot {acc:v}.4s, {wv:v}.16b, {av:v}.16b",
            acc = inout(vreg) acc,
            wv = in(vreg) wv,
            av = in(vreg) av,
            options(pure, nomem, nostack)
        );
    }
    vaddvq_s32(acc)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_tile4x2_sdot(
    wq: *const i8,   // panel rows r0..r0+4, row-major [4 × n]
    wsc: *const f32, // [4 × groups]
    aq: *const i8,   // tokens t0 and t0+1, [2 × n]
    asc: *const f32, // [2 × groups]
    n: usize,
    groups: usize,
    out: &mut [f32; 8], // out[r*2 + t]
) {
    // SDOT band dots via the proven helper + two-sided f32 fold.
    let mut tot = [0f32; 8];
    for g in 0..groups {
        let g0 = g * PRE8_GROUP;
        let as0 = *asc.add(g);
        let as1 = *asc.add(groups + g);
        let d00 = sdot_band(wq.add(g0), aq.add(g0)) as f32;
        let d01 = sdot_band(wq.add(g0), aq.add(n + g0)) as f32;
        let d10 = sdot_band(wq.add(n + g0), aq.add(g0)) as f32;
        let d11 = sdot_band(wq.add(n + g0), aq.add(n + g0)) as f32;
        let d20 = sdot_band(wq.add(2 * n + g0), aq.add(g0)) as f32;
        let d21 = sdot_band(wq.add(2 * n + g0), aq.add(n + g0)) as f32;
        let d30 = sdot_band(wq.add(3 * n + g0), aq.add(g0)) as f32;
        let d31 = sdot_band(wq.add(3 * n + g0), aq.add(n + g0)) as f32;
        tot[0] += d00 * *wsc.add(g) * as0;
        tot[1] += d01 * *wsc.add(g) * as1;
        tot[2] += d10 * *wsc.add(groups + g) * as0;
        tot[3] += d11 * *wsc.add(groups + g) * as1;
        tot[4] += d20 * *wsc.add(2 * groups + g) * as0;
        tot[5] += d21 * *wsc.add(2 * groups + g) * as1;
        tot[6] += d30 * *wsc.add(3 * groups + g) * as0;
        tot[7] += d31 * *wsc.add(3 * groups + g) * as1;
    }
    for (slot, v) in tot.iter().enumerate() {
        out[slot] = *v as f32;
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

static RMS_PROF_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ADD_PROF_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SWI_PROF_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EMBED_PROF_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

macro_rules! prof_section {
    ($atom:expr, $body:expr) => {{
        let on = prof_enabled();
        let t0 = if on { Some(std::time::Instant::now()) } else { None };
        let v = $body;
        if let Some(t0) = t0 {
            $atom.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        v
    }};
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
    // Phase-2 int8 path: quantize the shared activations once for the whole
    // layer call; panels convert their dequantized rows to int8 and run the
    // SDOT tile. Falls back per-call — no caller-visible behavior change.
    let aq = if pre8_enabled() && t_count >= 4 && rows % 4 == 0 && cols >= 512 {
        Some(quantize_acts_i8(x, t_count, cols))
    } else {
        None
    };
    let aq_ptr = aq.as_ref().map(|a| SendPtr(a as *const ActQ8));
    let t_start = if prof_enabled() { Some(std::time::Instant::now()) } else { None };
    #[cfg(target_os = "macos")]
    if blas_enabled() {
        let w32 = blas_dequant_full(lt, mmap, rows, cols, threads);
        blas_gemm(x, &w32, y, t_count, rows, cols);
        if let Some(t0) = t_start {
            GEMM_PROF_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            GEMM_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
    if pre8_enabled() && cols % PRE8_GROUP == 0 {
        let panels = rows.div_ceil(32);
        let (xq, ascale) = quantize_acts_i8(x, t_count, cols);
        let slab = pre8_slab_for(lt, mmap);
        super::dynamic_ffn::parallel_for(0, panels, threads, move |p0, p1| unsafe {
            let (SendPtr(lt_ptr2), SendPtr(mmap_ptr2), SendPtr(xf_ptr2), SendMutPtr(y_ptr2)) =
                (lt_ptr, mmap_ptr, x_ptr, y_ptr);
            run_gemm_panels_i8(
                SendPtr(lt_ptr2),
                SendPtr(mmap_ptr2),
                SendPtr(xf_ptr2),
                SendPtr(xq.as_ptr()),
                SendPtr(ascale.as_ptr()),
                SendMutPtr(y_ptr2),
                t_count,
                rows,
                cols,
                slab.as_deref(),
                p0,
                p1,
            );
        });
        if let Some(t0) = t_start {
            GEMM_PROF_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            GEMM_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
    const ROW_PANEL: usize = 32;
    let panels = rows.div_ceil(ROW_PANEL);
    super::dynamic_ffn::parallel_for(0, panels, threads, move |p0, p1| unsafe {
        run_gemm_panels(lt_ptr, mmap_ptr, x_ptr, y_ptr, aq_ptr, t_count, rows, cols, p0, p1);
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
    aq_ptr: Option<SendPtr<ActQ8>>,
    t_count: usize,
    rows: usize,
    cols: usize,
    p0: usize,
    p1: usize,
) {
    if p0 >= p1 {
        return;
    }
    const ROW_PANEL: usize = 32;
    let lt: &LinearTensor = unsafe { &*lt_ptr.0 };
    let mmap: &super::mmap::MmapFile = unsafe { &*mmap_ptr.0 };
    let x: &[f32] = unsafe { std::slice::from_raw_parts(x_ptr.0, t_count * cols) };
    let y: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(y_ptr.0, t_count * rows) };
    let aq: Option<&ActQ8> = aq_ptr.map(|p| unsafe { &*p.0 });
    ROW_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(ROW_PANEL * cols, 0.0f32);
        for p in p0..p1 {
            let row0 = p * ROW_PANEL;
            let panel_rows = ROW_PANEL.min(rows - row0);
            for ri in 0..panel_rows {
                dequant_linear_row(lt, mmap, row0 + ri, cols, &mut buf[ri * cols..(ri + 1) * cols]);
            }
            if let Some(aq) = aq {
                // ── Phase-2 int8 window: SDOT tiles for 4-row × 2-token
                // blocks; f32 dots cover the panel/tail remainders. ──
                run_i8_panel(aq, &buf, panel_rows, cols, x, y, row0, rows, t_count);
            } else {
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
            }  // else (f32 path)
        }
    });
}

/// Phase-2 helper: one panel's int8 GEMM — convert the dequantized f32 rows
/// to per-band int8 weights, then SDOT tiles (4 rows × 2 tokens), with
/// f32-dot remainders for row/tail remainders.
#[allow(clippy::too_many_arguments)]
unsafe fn run_i8_panel(
    aq: &ActQ8,
    buf: &[f32],
    panel_rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
    row0: usize,
    rows: usize,
    t_count: usize,
) {
    WQ_BUF.with(|wqc| {
        let mut wq = wqc.borrow_mut();
        wq.resize(panel_rows * aq.n, 0);
        WS_BUF.with(|wsc| {
            let mut ws = wsc.borrow_mut();
            ws.resize(panel_rows * aq.groups, 0.0);
            quantize_w_panel_i8(buf, panel_rows, cols, aq.n, aq.groups, &mut wq, &mut ws);
            let mut out = [0.0f32; 8];
            let mut t = 0usize;
            while t + 2 <= t_count {
                let mut ri = 0usize;
                while ri + 4 <= panel_rows {
                    unsafe {
                        gemm_tile4x2_sdot(
                            wq.as_ptr().add(ri * aq.n),
                            ws.as_ptr().add(ri * aq.groups),
                            aq.q.as_ptr().add(t * aq.n),
                            aq.scales.as_ptr().add(t * aq.groups),
                            aq.n,
                            aq.groups,
                            &mut out,
                        );
                    }
                    for tk in 0..2 {
                        for ti in 0..4 {
                            y[(t + tk) * rows + row0 + ri + ti] = out[ti * 2 + tk];
                        }
                    }
                    ri += 4;
                }
                for r1 in ri..panel_rows {
                    for tk in 0..2 {
                        #[cfg(target_arch = "aarch64")]
                        let acc = unsafe {
                            dot_f32_neon(buf.as_ptr().add(r1 * cols), x.as_ptr().add((t + tk) * cols), cols)
                        };
                        #[cfg(not(target_arch = "aarch64"))]
                        let acc = 0.0f32;
                        y[(t + tk) * rows + row0 + r1] = acc;
                    }
                }
                t += 2;
            }
            while t < t_count {
                let xr = &x[t * cols..(t + 1) * cols];
                for ri in 0..panel_rows {
                    #[cfg(target_arch = "aarch64")]
                    let acc = unsafe { dot_f32_neon(buf.as_ptr().add(ri * cols), xr.as_ptr(), cols) };
                    #[cfg(not(target_arch = "aarch64"))]
                    let acc = 0.0f32;
                    y[t * rows + row0 + ri] = acc;
                }
                t += 1;
            }
        });
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
    prof_section! {
        RMS_PROF_NS,
        prof_batched_rmsnorm(x, t_count, dim, mmap, tv, eps, out, threads)
    }
}

fn prof_batched_rmsnorm(
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
    prof_section! {
        SWI_PROF_NS,
        prof_batched_swiglu(gate, up, t_count, hidden, threads)
    }
}

fn prof_batched_swiglu(gate: &mut [f32], up: &[f32], t_count: usize, hidden: usize, threads: usize) {
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
    prof_section! {
        ADD_PROF_NS,
        for (xv, av) in x.iter_mut().zip(add.iter()) {
            *xv += *av;
        }
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
    prof_section! {
        EMBED_PROF_NS,
        prof_embed_tokens_batched(lt, mmap, tokens, dim, out, threads)
    }
}

fn prof_embed_tokens_batched(
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

// ─── Phase 2: int8 activations + int8 weights prefill (SOFUU_PRE8=1) ──
//
// Symmetric int8 on both sides with 32-element groups ( isolation for
// heavy-tailed activations), matching the MLX4 group granularity:
//   y[r][t] = Σ_group w_scale[r][g] * a_scale[t][g] * iacc_g
// iacc is an exact i32 lane-group sum from sdot (both sides <= 127*32 per
// group -> far below 2^31). Tail tokens/rows stay exact f32.

const PRE8_GROUP: usize = 32;

thread_local! {
    static I8_BUF: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
    static I8_BUF_B: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
    static F32_ROW_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static ROW_SCALE_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static ROW_SCALE_BUF_B: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn pre8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SOFUU_PRE8").as_deref() == Ok("1"))
}

/// ── Phase 3: AMX via Accelerate (`SOFUU_PREFILL_BLAS=1`) ──
///
/// The dequantized panel trick from the f32 path, but the MAC sweep hands off
/// to cblas_sgemm, which dispatches to the CPU-die AMX coprocessor on Apple
/// Silicon. Dequantization still runs on the dynamic_ffn worker pool; the one
/// big sgemm per tensor then runs on Accelerate's own pool (no nesting —
/// our pool must not call into it per panel).
#[cfg(target_os = "macos")]
mod accelerate {
    pub const ROW_MAJOR: i32 = 101; // CblasRowMajor
    pub const NO_TRANS: i32 = 111; // CblasNoTrans
    pub const TRANS: i32 = 112; // CblasTrans

    extern "C" {
        pub fn cblas_sgemm(
            layout: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

/// C[t][r] = X[t][c] · W[r][c] on the AMX path: A = X (t×cols, NoTrans),
/// B = W (rows×cols, Trans since W is row-major), C row-major (t×rows).
#[cfg(target_os = "macos")]
fn blas_gemm(x: &[f32], w32: &[f32], y: &mut [f32], t_count: usize, rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), t_count * cols);
    debug_assert_eq!(w32.len(), rows * cols);
    debug_assert_eq!(y.len(), t_count * rows);
    unsafe {
        accelerate::cblas_sgemm(
            accelerate::ROW_MAJOR,
            accelerate::NO_TRANS,
            accelerate::TRANS,
            t_count as i32,
            rows as i32,
            cols as i32,
            1.0,
            x.as_ptr(),
            cols as i32,
            w32.as_ptr(),
            cols as i32,
            0.0,
            y.as_mut_ptr(),
            rows as i32,
        );
    }
}

/// Dequantize the whole tensor to f32 once per call, split across the worker
/// pool; the sgemm then sees a plain row-major weight matrix.
#[cfg(target_os = "macos")]
fn blas_dequant_full(
    lt: &LinearTensor,
    mmap: &super::mmap::MmapFile,
    rows: usize,
    cols: usize,
    threads: usize,
) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    let w_ptr = SendMutPtr(w.as_mut_ptr());
    let lt_ptr = SendPtr(lt as *const LinearTensor);
    let mmap_ptr = SendPtr(mmap as *const super::mmap::MmapFile);
    const ROW_PANEL: usize = 64;
    let panels = rows.div_ceil(ROW_PANEL);
    super::dynamic_ffn::parallel_for(0, panels, threads, move |p0, p1| unsafe {
        let (SendPtr(lt2), SendPtr(mm2), SendMutPtr(wp2)) = (lt_ptr, mmap_ptr, w_ptr);
        let lt2: &LinearTensor = &*lt2;
        let mm2: &super::mmap::MmapFile = &*mm2;
        for p in p0..p1 {
            let row0 = p * ROW_PANEL;
            let panel_rows = ROW_PANEL.min(rows - row0);
            for ri in 0..panel_rows {
                dequant_linear_row(
                    lt2,
                    mm2,
                    row0 + ri,
                    cols,
                    std::slice::from_raw_parts_mut(wp2.add((row0 + ri) * cols), cols),
                );
            }
        }
    });
    w
}

fn blas_enabled() -> bool {
    // DEFAULT ON for macOS (exact-f32 math, gates-pinned): AMX is the fastest
    // prefill ladder on Apple Silicon and needs no quality trade. `=0` opts
    // out (NEON tile fallback); on other platforms this stays off — the
    // opt-in int8 ladder (SOFUU_PRE8) covers non-AMX aarch64 instead.
    cfg!(target_os = "macos") && std::env::var("SOFUU_PREFILL_BLAS").as_deref() != Ok("0")
}

/// Symmetric int8 quantization of one PRE8_GROUP slice: SIMD absmax, one
/// reciprocal per group, saturating mul->round->narrow chain. Scales land in
/// f32; codes in i8. Exact enough that the int-ref kernel test still passes.
fn quantize_group_i8(group: &[f32], out: &mut [i8], out_scale: &mut f32) {
    debug_assert_eq!(group.len(), PRE8_GROUP);
    debug_assert_eq!(out.len(), PRE8_GROUP);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = PRE8_GROUP;
        let mut m = vdupq_n_f32(0.0);
        let mut c = 0;
        while c < n {
            m = vmaxq_f32(m, vabsq_f32(vld1q_f32(group.as_ptr().add(c))));
            c += 4;
        }
        let absmax = vmaxvq_f32(m);
        if absmax == 0.0 || !absmax.is_finite() {
            *out_scale = 1.0;
            out.fill(0);
            return;
        }
        let scale = absmax / 127.0;
        *out_scale = scale;
        let inv = vdupq_n_f32(127.0 / absmax);
        let mut c = 0;
        // 8 floats per iteration: two i32 quads -> i16 pair -> i8
        while c < n {
            let f0 = vmulq_f32(vld1q_f32(group.as_ptr().add(c)), inv);
            let f1 = vmulq_f32(vld1q_f32(group.as_ptr().add(c + 4)), inv);
            let i0 = vcvtnq_s32_f32(f0);
            let i1 = vcvtnq_s32_f32(f1);
            let n0 = vqmovn_s32(i0);
            let n1 = vqmovn_s32(i1);
            let s8 = vqmovn_s16(vcombine_s16(n0, n1));
            vst1_s8(out.as_mut_ptr().add(c), s8);
            c += 8;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut absmax = 0.0f32;
        for &v in group {
            absmax = absmax.max(v.abs());
        }
        if absmax == 0.0 || !absmax.is_finite() {
            *out_scale = 1.0;
            out.fill(0);
            return;
        }
        let scale = absmax / 127.0;
        *out_scale = scale;
        for (qb, &v) in out.iter_mut().zip(group.iter()) {
            *qb = ((v / scale).round().clamp(-127.0, 127.0)) as i8;
        }
    }
}

fn quantize_acts_i8(x: &[f32], t_count: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    debug_assert_eq!(cols % PRE8_GROUP, 0);
    let groups = cols / PRE8_GROUP;
    let mut q = vec![0i8; t_count * cols];
    let mut scales = vec![1.0f32; t_count * groups];
    for t in 0..t_count {
        let row = &x[t * cols..(t + 1) * cols];
        let qrow = &mut q[t * cols..(t + 1) * cols];
        for g in 0..groups {
            quantize_group_i8(
                &row[g * PRE8_GROUP..(g + 1) * PRE8_GROUP],
                &mut qrow[g * PRE8_GROUP..(g + 1) * PRE8_GROUP],
                &mut scales[t * groups + g],
            );
        }
    }
    (q, scales)
}

/// Dequantize one weight row to f32 scratch AND int8 with per-32-group
/// scales appended after the codes (fbuf also feeds the exact f32 tails).
unsafe fn dequant_row_grouped(
    lt: &LinearTensor,
    mmap: &super::mmap::MmapFile,
    row: usize,
    cols: usize,
    scratch: &mut [f32],
    out: &mut [i8],
    out_scales: &mut [f32],
) {
    let groups = cols / PRE8_GROUP;
    let f32row = &mut scratch[..cols];
    dequant_linear_row(lt, mmap, row, cols, f32row);
    for g in 0..groups {
        quantize_group_i8(
            &f32row[g * PRE8_GROUP..(g + 1) * PRE8_GROUP],
            &mut out[g * PRE8_GROUP..(g + 1) * PRE8_GROUP],
            &mut out_scales[g],
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn sdot16(mut acc: std::arch::aarch64::int32x4_t, w: std::arch::aarch64::int8x16_t, a: std::arch::aarch64::int8x16_t) -> std::arch::aarch64::int32x4_t {
    std::arch::asm!(
        "sdot {acc:v}.4s, {w:v}.16b, {a:v}.16b",
        acc = inout(vreg) acc,
        w = in(vreg) w,
        a = in(vreg) a,
        options(pure, nomem, nostack)
    );
    acc
}

/// Doubled-up weight cache for the int8 prefill path: codes/scales built
/// once per (model, tensor) and reused across prefills. Keyed by mmap identity
/// + tensor span + fingerprint of the raw leading bytes, so a remapped file at
/// the same address can never serve a foreign tensor.
struct Pre8Slab {
    codes: Vec<i8>,   // [rows][cols]
    scales: Vec<f32>, // [rows][groups]
    rows: usize,
    cols: usize,
    groups: usize,
}

static PRE8_SLABS: std::sync::OnceLock<
    std::sync::Mutex<(std::collections::HashMap<u64, std::sync::Arc<Pre8Slab>>, usize)>,
> = std::sync::OnceLock::new();

const PRE8_SLAB_MEM_CAP: usize = 4 << 30; // stop caching past 4 GiB of slabs

fn fnv1a64(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn pre8_lt_span(lt: &LinearTensor) -> (usize, usize) {
    match lt {
        LinearTensor::Bf16(tv) => (tv.data_offset, tv.size_bytes),
        LinearTensor::Mlx4 { weight, .. } => (weight.data_offset, weight.size_bytes),
    }
}

/// Cached per-tensor slab, or None (caller re-quantizes per prefill).
fn pre8_slab_for(lt: &LinearTensor, mmap: &super::mmap::MmapFile) -> Option<std::sync::Arc<Pre8Slab>> {
    const GROUP: usize = PRE8_GROUP;
    let rows = lt.rows();
    let cols = lt.cols();
    if cols % GROUP != 0 || rows == 0 {
        return None;
    }
    let (off, span) = pre8_lt_span(lt);
    if rows * cols > 2_500_000_000 {
        return None;
    }
    // Fingerprint the first 16 KiB of raw tensor bytes for collision safety.
    let fp = {
        let probe = mmap.slice(off, span.min(16 << 10)).unwrap_or(&[]);
        fnv1a64(probe, 0xcbf29ce484222325)
    };
    let key = fnv1a64(
        &[
            (mmap.as_slice().as_ptr() as usize as u64),
            (mmap.as_slice().len() as u64),
            (off as u64),
            (span as u64),
            (rows as u64),
            (cols as u64),
            fp,
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>(),
        0x9e3779b97f4a7c15,
    );
    let store = PRE8_SLABS.get_or_init(|| std::sync::Mutex::new((std::collections::HashMap::new(), 0)));
    let mut g = store.lock().expect("pre8 slab lock");
    if let Some(s) = g.0.get(&key) {
        return Some(s.clone());
    }
    let mem = rows * cols;
    if g.1 + mem > PRE8_SLAB_MEM_CAP {
        return None;
    }
    drop(g);
    let groups = cols / GROUP;
    let mut slab = Pre8Slab {
        codes: vec![0i8; rows * cols],
        scales: vec![0.0f32; rows * groups],
        rows,
        cols,
        groups,
    };
    let mut frow = vec![0.0f32; cols];
    for r in 0..rows {
        dequant_linear_row(lt, mmap, r, cols, &mut frow);
        let crow = &mut slab.codes[r * cols..(r + 1) * cols];
        let srow = &mut slab.scales[r * groups..(r + 1) * groups];
        for gi in 0..groups {
            quantize_group_i8(
                &frow[gi * GROUP..(gi + 1) * GROUP],
                &mut crow[gi * GROUP..(gi + 1) * GROUP],
                &mut srow[gi],
            );
        }
    }
    let slab = std::sync::Arc::new(slab);
    let mut g = store.lock().expect("pre8 slab lock");
    g.1 += mem;
    g.0.insert(key, slab.clone());
    Some(slab)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline(never)]
unsafe fn gemm_tile4x4_i8(
    w: *const i8,
    x: *const i8,
    cols: usize,
    ws_rowmajor: *const f32, // [4][n_groups]
    as_rowmajor: *const f32, // [4][n_groups]
    n_groups: usize,
    fout: &mut [f32; 16], // [row][token]
) {
    use std::arch::aarch64::*;
    unsafe {
        let (w0, w1, w2, w3) = (w, w.add(cols), w.add(2 * cols), w.add(3 * cols));
        let (x0, x1, x2, x3) = (x, x.add(cols), x.add(2 * cols), x.add(3 * cols));
        for g in 0..n_groups {
            let mut acc00 = vdupq_n_s32(0);
            let mut acc01 = vdupq_n_s32(0);
            let mut acc02 = vdupq_n_s32(0);
            let mut acc03 = vdupq_n_s32(0);
            let mut acc10 = vdupq_n_s32(0);
            let mut acc11 = vdupq_n_s32(0);
            let mut acc12 = vdupq_n_s32(0);
            let mut acc13 = vdupq_n_s32(0);
            let mut acc20 = vdupq_n_s32(0);
            let mut acc21 = vdupq_n_s32(0);
            let mut acc22 = vdupq_n_s32(0);
            let mut acc23 = vdupq_n_s32(0);
            let mut acc30 = vdupq_n_s32(0);
            let mut acc31 = vdupq_n_s32(0);
            let mut acc32 = vdupq_n_s32(0);
            let mut acc33 = vdupq_n_s32(0);
            {
                let mut k = g * PRE8_GROUP;
                let kend = k + PRE8_GROUP;
                while k < kend {
                    let xa = vld1q_s8(x0.add(k));
                    let xb = vld1q_s8(x1.add(k));
                    let xc = vld1q_s8(x2.add(k));
                    let xd = vld1q_s8(x3.add(k));
                    let wa = vld1q_s8(w0.add(k));
                    let wb = vld1q_s8(w1.add(k));
                    let wc = vld1q_s8(w2.add(k));
                    let wd = vld1q_s8(w3.add(k));
                    acc00 = sdot16(acc00, wa, xa);
                    acc01 = sdot16(acc01, wa, xb);
                    acc02 = sdot16(acc02, wa, xc);
                    acc03 = sdot16(acc03, wa, xd);
                    acc10 = sdot16(acc10, wb, xa);
                    acc11 = sdot16(acc11, wb, xb);
                    acc12 = sdot16(acc12, wb, xc);
                    acc13 = sdot16(acc13, wb, xd);
                    acc20 = sdot16(acc20, wc, xa);
                    acc21 = sdot16(acc21, wc, xb);
                    acc22 = sdot16(acc22, wc, xc);
                    acc23 = sdot16(acc23, wc, xd);
                    acc30 = sdot16(acc30, wd, xa);
                    acc31 = sdot16(acc31, wd, xb);
                    acc32 = sdot16(acc32, wd, xc);
                    acc33 = sdot16(acc33, wd, xd);
                    k += 16;
                }
            }
            let ws0 = *ws_rowmajor.add(g);
            let ws1 = *ws_rowmajor.add(n_groups + g);
            let ws2 = *ws_rowmajor.add(2 * n_groups + g);
            let ws3 = *ws_rowmajor.add(3 * n_groups + g);
            let as0 = *as_rowmajor.add(g);
            let as1 = *as_rowmajor.add(n_groups + g);
            let as2 = *as_rowmajor.add(2 * n_groups + g);
            let as3 = *as_rowmajor.add(3 * n_groups + g);
            let mut fold = |acc: int32x4_t, s: f32, cell: &mut f32| {
                *cell += s * vaddvq_s32(acc) as f32;
            };
            fold(acc00, ws0 * as0, &mut fout[0]);
            fold(acc01, ws0 * as1, &mut fout[1]);
            fold(acc02, ws0 * as2, &mut fout[2]);
            fold(acc03, ws0 * as3, &mut fout[3]);
            fold(acc10, ws1 * as0, &mut fout[4]);
            fold(acc11, ws1 * as1, &mut fout[5]);
            fold(acc12, ws1 * as2, &mut fout[6]);
            fold(acc13, ws1 * as3, &mut fout[7]);
            fold(acc20, ws2 * as0, &mut fout[8]);
            fold(acc21, ws2 * as1, &mut fout[9]);
            fold(acc22, ws2 * as2, &mut fout[10]);
            fold(acc23, ws2 * as3, &mut fout[11]);
            fold(acc30, ws3 * as0, &mut fout[12]);
            fold(acc31, ws3 * as1, &mut fout[13]);
            fold(acc32, ws3 * as2, &mut fout[14]);
            fold(acc33, ws3 * as3, &mut fout[15]);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn gemm_tile4x4_i8(
    w: *const i8,
    x: *const i8,
    cols: usize,
    ws_rowmajor: *const f32,
    as_rowmajor: *const f32,
    n_groups: usize,
    fout: &mut [f32; 16],
) {
    for ti in 0..4 {
        for tk in 0..4 {
            let mut cell = 0.0f32;
            for g in 0..n_groups {
                let mut acc: i32 = 0;
                let k0 = g * PRE8_GROUP;
                for k in k0..k0 + PRE8_GROUP {
                    acc += *w.add(ti * cols + k) as i32 * *x.add(tk * cols + k) as i32;
                }
                cell += *ws_rowmajor.add(ti * n_groups + g)
                    * *as_rowmajor.add(tk * n_groups + g)
                    * acc as f32;
            }
            fout[ti * 4 + tk] = cell;
        }
    }
}

/// int8 panel runner. With a cached slab the codes/scales come straight from
/// the cache and f32 dequantization happens only for the exact tail rows and
/// tail tokens; without one, rows are quantized per panel as before.
#[allow(clippy::too_many_arguments)]
unsafe fn run_gemm_panels_i8(
    lt_ptr: SendPtr<LinearTensor>,
    mmap_ptr: SendPtr<super::mmap::MmapFile>,
    xf_ptr: SendPtr<f32>,
    xq_ptr: SendPtr<i8>,
    ascale_ptr: SendPtr<f32>, // [t_count][groups]
    y_ptr: SendMutPtr<f32>,
    t_count: usize,
    rows: usize,
    cols: usize,
    slab: Option<&Pre8Slab>,
    p0: usize,
    p1: usize,
) {
    if p0 >= p1 || t_count == 0 {
        return;
    }
    const ROW_PANEL: usize = 32;
    let groups = cols / PRE8_GROUP;
    let lt: &LinearTensor = unsafe { &*lt_ptr.0 };
    let mmap: &super::mmap::MmapFile = unsafe { &*mmap_ptr.0 };
    let x_f32: &[f32] = unsafe { std::slice::from_raw_parts(xf_ptr.0, t_count * cols) };
    let y: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(y_ptr.0, t_count * rows) };
    F32_ROW_BUF.with(|fcell| {
        let mut fbuf = fcell.borrow_mut();
        fbuf.resize(ROW_PANEL * cols, 0.0f32);
        I8_BUF.with(|icell| {
            let mut ibuf = icell.borrow_mut();
            ibuf.resize(ROW_PANEL * cols, 0i8);
            ROW_SCALE_BUF.with(|scell| {
                let mut rscale = scell.borrow_mut();
                rscale.resize(ROW_PANEL * groups.max(1), 1.0f32);
                for p in p0..p1 {
                    let row0 = p * ROW_PANEL;
                    let panel_rows = ROW_PANEL.min(rows - row0);
                    let span4 = (panel_rows / 4) * 4;
                    if slab.is_none() {
                        for ri in 0..panel_rows {
                            dequant_row_grouped(
                                lt,
                                mmap,
                                row0 + ri,
                                cols,
                                &mut fbuf[ri * cols..(ri + 1) * cols],
                                &mut ibuf[ri * cols..(ri + 1) * cols],
                                &mut rscale[ri * groups..(ri + 1) * groups],
                            );
                        }
                    }
                    let mut t = 0usize;
                    while t + 4 <= t_count {
                        let mut ri = 0usize;
                        while ri + 4 <= panel_rows {
                            let mut fout = [0.0f32; 16];
                            let (wp, sp) = match slab {
                                Some(s) => (
                                    s.codes.as_ptr().add((row0 + ri) * cols),
                                    s.scales.as_ptr().add((row0 + ri) * groups),
                                ),
                                None => (
                                    ibuf.as_ptr().add(ri * cols),
                                    rscale.as_ptr().add(ri * groups),
                                ),
                            };
                            gemm_tile4x4_i8(
                                wp,
                                xq_ptr.0.add(t * cols),
                                cols,
                                sp,
                                ascale_ptr.0.add(t * groups),
                                groups,
                                &mut fout,
                            );
                            for ti in 0..4 {
                                for tk in 0..4 {
                                    y[(t + tk) * rows + row0 + ri + ti] = fout[ti * 4 + tk];
                                }
                            }
                            ri += 4;
                        }
                        for r1 in span4..panel_rows {
                            if slab.is_some() {
                                dequant_linear_row(
                                    lt,
                                    mmap,
                                    row0 + r1,
                                    cols,
                                    &mut fbuf[r1 * cols..(r1 + 1) * cols],
                                );
                            }
                            for tk in 0..4 {
                                let xr = &x_f32[(t + tk) * cols..(t + tk + 1) * cols];
                                y[(t + tk) * rows + row0 + r1] = dot_f32_neon(
                                    fbuf.as_ptr().add(r1 * cols),
                                    xr.as_ptr(),
                                    cols,
                                );
                            }
                        }
                        t += 4;
                    }
                    while t < t_count {
                        let xr = &x_f32[t * cols..(t + 1) * cols];
                        for ri in 0..panel_rows {
                            if slab.is_some() {
                                dequant_linear_row(
                                    lt,
                                    mmap,
                                    row0 + ri,
                                    cols,
                                    &mut fbuf[ri * cols..(ri + 1) * cols],
                                );
                            }
                            y[t * rows + row0 + ri] = dot_f32_neon(
                                fbuf.as_ptr().add(ri * cols),
                                xr.as_ptr(),
                                cols,
                            );
                        }
                        t += 1;
                    }
                }
            });
        });
    });
}

/// Paired (gate+up) int8 panels: one activation quantize feeds both halves.
/// Slab path: codes from the cache, tails re-dequant per row on demand.
#[allow(clippy::too_many_arguments)]
unsafe fn run_gemm_panels_pair_i8(
    lt_a_ptr: SendPtr<LinearTensor>,
    lt_b_ptr: SendPtr<LinearTensor>,
    mmap_ptr: SendPtr<super::mmap::MmapFile>,
    xf_ptr: SendPtr<f32>,
    xq_ptr: SendPtr<i8>,
    ascale_ptr: SendPtr<f32>,
    ya_ptr: SendMutPtr<f32>,
    yb_ptr: SendMutPtr<f32>,
    t_count: usize,
    rows: usize,
    cols: usize,
    slab_a: Option<&Pre8Slab>,
    slab_b: Option<&Pre8Slab>,
    p0: usize,
    p1: usize,
) {
    if p0 >= p1 || t_count == 0 {
        return;
    }
    const ROW_PANEL: usize = 32;
    let groups = cols / PRE8_GROUP;
    let lt_a: &LinearTensor = unsafe { &*lt_a_ptr.0 };
    let lt_b: &LinearTensor = unsafe { &*lt_b_ptr.0 };
    let mmap: &super::mmap::MmapFile = unsafe { &*mmap_ptr.0 };
    let x_f32: &[f32] = unsafe { std::slice::from_raw_parts(xf_ptr.0, t_count * cols) };
    let ya: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(ya_ptr.0, t_count * rows) };
    let yb: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(yb_ptr.0, t_count * rows) };
    F32_ROW_BUF.with(|fcell| {
        let mut fbuf = fcell.borrow_mut();
        fbuf.resize(ROW_PANEL * cols, 0.0f32);
        I8_BUF.with(|ia| {
            let mut ibuf_a = ia.borrow_mut();
            ibuf_a.resize(ROW_PANEL * cols, 0i8);
            I8_BUF_B.with(|ib| {
                let mut ibuf_b = ib.borrow_mut();
                ibuf_b.resize(ROW_PANEL * cols, 0i8);
                ROW_SCALE_BUF.with(|sa| {
                    let mut rs_a = sa.borrow_mut();
                    rs_a.resize(ROW_PANEL * groups.max(1), 1.0f32);
                    ROW_SCALE_BUF_B.with(|sb| {
                        let mut rs_b = sb.borrow_mut();
                        rs_b.resize(ROW_PANEL * groups.max(1), 1.0f32);
                        for p in p0..p1 {
                            let row0 = p * ROW_PANEL;
                            let panel_rows = ROW_PANEL.min(rows - row0);
                            let span4 = (panel_rows / 4) * 4;
                            if slab_a.is_none() {
                                for ri in 0..panel_rows {
                                    dequant_row_grouped(
                                        lt_a, mmap, row0 + ri, cols,
                                        &mut fbuf[ri * cols..(ri + 1) * cols],
                                        &mut ibuf_a[ri * cols..(ri + 1) * cols],
                                        &mut rs_a[ri * groups..(ri + 1) * groups],
                                    );
                                }
                            }
                            if slab_b.is_none() {
                                for ri in 0..panel_rows {
                                    dequant_row_grouped(
                                        lt_b, mmap, row0 + ri, cols,
                                        &mut fbuf[ri * cols..(ri + 1) * cols],
                                        &mut ibuf_b[ri * cols..(ri + 1) * cols],
                                        &mut rs_b[ri * groups..(ri + 1) * groups],
                                    );
                                }
                            }
                            let mut t = 0usize;
                            while t + 4 <= t_count {
                                let mut ri = 0usize;
                                while ri + 4 <= panel_rows {
                                    let mut f_a = [0.0f32; 16];
                                    let mut f_b = [0.0f32; 16];
                                    let (wa, sa) = match slab_a {
                                        Some(s) => (
                                            s.codes.as_ptr().add((row0 + ri) * cols),
                                            s.scales.as_ptr().add((row0 + ri) * groups),
                                        ),
                                        None => (
                                            ibuf_a.as_ptr().add(ri * cols),
                                            rs_a.as_ptr().add(ri * groups),
                                        ),
                                    };
                                    gemm_tile4x4_i8(
                                        wa,
                                        xq_ptr.0.add(t * cols),
                                        cols,
                                        sa,
                                        ascale_ptr.0.add(t * groups),
                                        groups,
                                        &mut f_a,
                                    );
                                    let (wb, sb) = match slab_b {
                                        Some(s) => (
                                            s.codes.as_ptr().add((row0 + ri) * cols),
                                            s.scales.as_ptr().add((row0 + ri) * groups),
                                        ),
                                        None => (
                                            ibuf_b.as_ptr().add(ri * cols),
                                            rs_b.as_ptr().add(ri * groups),
                                        ),
                                    };
                                    gemm_tile4x4_i8(
                                        wb,
                                        xq_ptr.0.add(t * cols),
                                        cols,
                                        sb,
                                        ascale_ptr.0.add(t * groups),
                                        groups,
                                        &mut f_b,
                                    );
                                    for ti in 0..4 {
                                        for tk in 0..4 {
                                            ya[(t + tk) * rows + row0 + ri + ti] = f_a[ti * 4 + tk];
                                            yb[(t + tk) * rows + row0 + ri + ti] = f_b[ti * 4 + tk];
                                        }
                                    }
                                    ri += 4;
                                }
                                // Tail rows: exact f32 dots, re-dequant on demand
                                // (slab path has no fbuf content up front).
                                for r1 in span4..panel_rows {
                                    dequant_linear_row(
                                        lt_a, mmap, row0 + r1, cols,
                                        &mut fbuf[r1 * cols..(r1 + 1) * cols],
                                    );
                                    for tk in 0..4 {
                                        let xr = &x_f32[(t + tk) * cols..(t + tk + 1) * cols];
                                        ya[(t + tk) * rows + row0 + r1] = dot_f32_neon(
                                            fbuf.as_ptr().add(r1 * cols), xr.as_ptr(), cols,
                                        );
                                    }
                                    dequant_linear_row(
                                        lt_b, mmap, row0 + r1, cols,
                                        &mut fbuf[r1 * cols..(r1 + 1) * cols],
                                    );
                                    for tk in 0..4 {
                                        let xr = &x_f32[(t + tk) * cols..(t + tk + 1) * cols];
                                        yb[(t + tk) * rows + row0 + r1] = dot_f32_neon(
                                            fbuf.as_ptr().add(r1 * cols), xr.as_ptr(), cols,
                                        );
                                    }
                                }
                                t += 4;
                            }
                            while t < t_count {
                                let xr = &x_f32[t * cols..(t + 1) * cols];
                                for ri in 0..panel_rows {
                                    dequant_linear_row(
                                        lt_a, mmap, row0 + ri, cols,
                                        &mut fbuf[ri * cols..(ri + 1) * cols],
                                    );
                                    ya[t * rows + row0 + ri] = dot_f32_neon(
                                        fbuf.as_ptr().add(ri * cols), xr.as_ptr(), cols,
                                    );
                                    dequant_linear_row(
                                        lt_b, mmap, row0 + ri, cols,
                                        &mut fbuf[ri * cols..(ri + 1) * cols],
                                    );
                                    yb[t * rows + row0 + ri] = dot_f32_neon(
                                        fbuf.as_ptr().add(ri * cols), xr.as_ptr(), cols,
                                    );
                                }
                                t += 1;
                            }
                        }
                    });
                });
            });
        });
    });
}

thread_local! {
    static ATT_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[allow(clippy::too_many_arguments)]
unsafe fn attention_prep_tokens(
    q_ptr: SendMutPtr<f32>,
    k_ptr: SendMutPtr<f32>,
    v_ptr: SendMutPtr<f32>,
    kc_ptr: SendMutPtr<f32>,
    vc_ptr: SendMutPtr<f32>,
    qln_ptr: SendPtr<u8>,
    kln_ptr: SendPtr<u8>,
    t0: usize,
    t1: usize,
    start_pos: usize,
    q_rows: usize,
    k_rows: usize,
    v_rows: usize,
    layer_idx: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    freq_base: f32,
    eps: f32,
) {
    if t0 >= t1 {
        return;
    }
    let dim_half = head_dim / 2;
    for t in t0..t1 {
        let pos = start_pos + t;
        for h in 0..n_heads {
            let off = t * q_rows + h * head_dim;
            let q_head = unsafe { std::slice::from_raw_parts_mut(q_ptr.0.add(off), head_dim) };
            Lfm2Model::head_rmsnorm(q_head, qln_ptr.0, head_dim, eps);
        }
        for kv in 0..n_kv_heads {
            let off = t * k_rows + kv * head_dim;
            let k_head = unsafe { std::slice::from_raw_parts_mut(k_ptr.0.add(off), head_dim) };
            Lfm2Model::head_rmsnorm(k_head, kln_ptr.0, head_dim, eps);
        }
        for h in 0..n_heads {
            let off = t * q_rows + h * head_dim;
            let q_head = unsafe { std::slice::from_raw_parts_mut(q_ptr.0.add(off), head_dim) };
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
            let k_head = unsafe { std::slice::from_raw_parts_mut(k_ptr.0.add(off), head_dim) };
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
            unsafe {
                std::ptr::copy_nonoverlapping(k_ptr.0.add(t * k_rows + kv * head_dim), kc_ptr.0.add(cache_offset), head_dim);
                std::ptr::copy_nonoverlapping(v_ptr.0.add(t * v_rows + kv * head_dim), vc_ptr.0.add(cache_offset), head_dim);
            }
        }
    }
}

/// Attention mix over the cache: REQUIRES every token's KV write complete
/// first (two-phase with `attention_prep_tokens` — the serial version read
/// columns that earlier tokens in the batch had just written, so fusing prep
/// and mix would race the cache).
#[allow(clippy::too_many_arguments)]
unsafe fn attention_mix_tokens(
    q_ptr: SendPtr<f32>,
    hb_ptr: SendMutPtr<f32>,
    kc_ptr: SendPtr<f32>,
    vc_ptr: SendPtr<f32>,
    t0: usize,
    t1: usize,
    start_pos: usize,
    q_rows: usize,
    layer_idx: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
) {
    if t0 >= t1 {
        return;
    }
    let scale = (head_dim as f32).sqrt().recip();
    let kv_group_size = n_heads / n_kv_heads;
    for t in t0..t1 {
        let pos = start_pos + t;
        let cur_seq = pos.min(max_seq_len - 1);
        for h in 0..n_heads {
            let kv_idx = h / kv_group_size;
            let q_off = t * q_rows + h * head_dim;
            let q_head: &[f32] = unsafe { std::slice::from_raw_parts(q_ptr.0.add(q_off), head_dim) };
            let kv_base = (layer_idx * n_kv_heads + kv_idx) * (max_seq_len * head_dim);
            let seq_len = cur_seq + 1;
            let mut max_score = f32::NEG_INFINITY;
            let h_out = unsafe {
                std::slice::from_raw_parts_mut(hb_ptr.0.add(t * q_rows + h * head_dim), head_dim)
            };
            ATT_SCRATCH.with(|cell| {
                let mut buf = cell.borrow_mut();
                buf.resize(seq_len.max(1), 0.0f32);
                let scores = &mut buf[..seq_len];
                let mut sum_exp = 0.0f32;
                for tt in 0..seq_len {
                    let k_t = unsafe {
                        std::slice::from_raw_parts(kc_ptr.0.add(kv_base + tt * head_dim), head_dim)
                    };
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[d] * k_t[d];
                    }
                    let score = dot * scale;
                    scores[tt] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }
                for tt in 0..seq_len {
                    let e = (scores[tt] - max_score).exp();
                    scores[tt] = e;
                    sum_exp += e;
                }
                let inv_sum = sum_exp.recip();
                h_out.fill(0.0f32);
                for tt in 0..seq_len {
                    let weight = scores[tt] * inv_sum;
                    let v_t = unsafe {
                        std::slice::from_raw_parts(vc_ptr.0.add(kv_base + tt * head_dim), head_dim)
                    };
                    for d in 0..head_dim {
                        h_out[d] += weight * v_t[d];
                    }
                }
            });
        }
    }
}

/// Gate + up projections in one panel pass: they share the same activation
/// sweep, so x is streamed once per row-pair instead of once per matrix and
/// the pool is dispatched once instead of twice.
fn gemm_pair_batched(
    lt_a: &LinearTensor,
    lt_b: &LinearTensor,
    mmap: &super::mmap::MmapFile,
    x: &[f32],
    t_count: usize,
    y_a: &mut [f32],
    y_b: &mut [f32],
    threads: usize,
) {
    let rows = lt_a.rows();
    let cols = lt_a.cols();
    debug_assert_eq!(lt_b.rows(), rows);
    debug_assert_eq!(lt_b.cols(), cols);
    debug_assert_eq!(x.len(), t_count * cols);
    debug_assert_eq!(y_a.len(), t_count * rows);
    debug_assert_eq!(y_b.len(), t_count * rows);
    if t_count == 0 || rows == 0 {
        return;
    }
    let a_ptr = SendPtr(lt_a as *const LinearTensor);
    let b_ptr = SendPtr(lt_b as *const LinearTensor);
    let mmap_ptr = SendPtr(mmap as *const super::mmap::MmapFile);
    let x_ptr = SendPtr(x.as_ptr());
    let ya_ptr = SendMutPtr(y_a.as_mut_ptr());
    let yb_ptr = SendMutPtr(y_b.as_mut_ptr());
    #[cfg(target_os = "macos")]
    if blas_enabled() {
        let t_start_blas = if prof_enabled() { Some(std::time::Instant::now()) } else { None };
        let w32a = blas_dequant_full(lt_a, mmap, rows, cols, threads);
        let w32b = blas_dequant_full(lt_b, mmap, rows, cols, threads);
        blas_gemm(x, &w32a, y_a, t_count, rows, cols);
        blas_gemm(x, &w32b, y_b, t_count, rows, cols);
        if let Some(t0) = t_start_blas {
            GEMM_PROF_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            GEMM_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
    if pre8_enabled() && cols % PRE8_GROUP == 0 {
        let panels = rows.div_ceil(32);
        let (xq, ascale) = quantize_acts_i8(x, t_count, cols);
        let a_ptr2 = a_ptr;
        let b_ptr2 = b_ptr;
        let mmap_ptr2 = mmap_ptr;
        let x_ptr2 = x_ptr;
        let ya_ptr2 = ya_ptr;
        let yb_ptr2 = yb_ptr;
        let slab_a = pre8_slab_for(lt_a, mmap);
        let slab_b = pre8_slab_for(lt_b, mmap);
        let t_start8 = if prof_enabled() { Some(std::time::Instant::now()) } else { None };
        super::dynamic_ffn::parallel_for(0, panels, threads, move |p0, p1| unsafe {
            let (SendPtr(a_ptr3), SendPtr(b_ptr3), SendPtr(mmap_ptr3), SendPtr(xf_ptr3), SendMutPtr(ya_ptr3), SendMutPtr(yb_ptr3)) =
                (a_ptr2, b_ptr2, mmap_ptr2, x_ptr2, ya_ptr2, yb_ptr2);
            run_gemm_panels_pair_i8(
                SendPtr(a_ptr3),
                SendPtr(b_ptr3),
                SendPtr(mmap_ptr3),
                SendPtr(xf_ptr3),
                SendPtr(xq.as_ptr()),
                SendPtr(ascale.as_ptr()),
                SendMutPtr(ya_ptr3),
                SendMutPtr(yb_ptr3),
                t_count,
                rows,
                cols,
                slab_a.as_deref(),
                slab_b.as_deref(),
                p0,
                p1,
            );
        });
        if let Some(t0) = t_start8 {
            GEMM_PROF_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            GEMM_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
    const ROW_PANEL: usize = 32;
    let panels = rows.div_ceil(ROW_PANEL);
    let t_start = if prof_enabled() { Some(std::time::Instant::now()) } else { None };
    super::dynamic_ffn::parallel_for(0, panels, threads, move |p0, p1| unsafe {
        // Destructure the wrappers whole so edition-2021 capture doesn't lift
        // the raw pointer fields out of them.
        let (SendPtr(a_ptr), SendPtr(b_ptr), SendPtr(mmap_ptr), SendPtr(x_ptr), SendMutPtr(ya_ptr), SendMutPtr(yb_ptr)) =
            (a_ptr, b_ptr, mmap_ptr, x_ptr, ya_ptr, yb_ptr);
        let lt_a: &LinearTensor = &*a_ptr;
        let lt_b: &LinearTensor = &*b_ptr;
        let mmap: &super::mmap::MmapFile = &*mmap_ptr;
        let x: &[f32] = std::slice::from_raw_parts(x_ptr, t_count * cols);
        let ya: &mut [f32] = std::slice::from_raw_parts_mut(ya_ptr, t_count * rows);
        let yb: &mut [f32] = std::slice::from_raw_parts_mut(yb_ptr, t_count * rows);
        ROW_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.resize(2 * ROW_PANEL * cols, 0.0f32);
            for p in p0..p1 {
                let row0 = p * ROW_PANEL;
                let panel_rows = ROW_PANEL.min(rows - row0);
                for ri in 0..panel_rows {
                    dequant_linear_row(lt_a, mmap, row0 + ri, cols, &mut buf[ri * cols..(ri + 1) * cols]);
                    dequant_linear_row(
                        lt_b,
                        mmap,
                        row0 + ri,
                        cols,
                        &mut buf[(ROW_PANEL + ri) * cols..(ROW_PANEL + ri + 1) * cols],
                    );
                }
                let mut t = 0usize;
                while t + 4 <= t_count {
                    let mut ri = 0usize;
                    while ri + 4 <= panel_rows {
                        let mut out = [0.0f32; 16];
                        gemm_tile4x4_neon(
                            buf.as_ptr().add(ri * cols),
                            x.as_ptr().add(t * cols),
                            cols,
                            &mut out,
                        );
                        for ti in 0..4 {
                            for tk in 0..4 {
                                ya[(t + tk) * rows + row0 + ri + ti] = out[ti * 4 + tk];
                            }
                        }
                        let mut outb = [0.0f32; 16];
                        gemm_tile4x4_neon(
                            buf.as_ptr().add((ROW_PANEL + ri) * cols),
                            x.as_ptr().add(t * cols),
                            cols,
                            &mut outb,
                        );
                        for ti in 0..4 {
                            for tk in 0..4 {
                                yb[(t + tk) * rows + row0 + ri + ti] = outb[ti * 4 + tk];
                            }
                        }
                        ri += 4;
                    }
                    // Scalar-dot tail rows of the panel (panel % 4).
                    for r1 in ri..panel_rows {
                        for tk in 0..4 {
                            let xr = &x[(t + tk) * cols..(t + tk + 1) * cols];
                            #[cfg(target_arch = "aarch64")]
                            let acca = unsafe { dot_f32_neon(buf.as_ptr().add(r1 * cols), xr.as_ptr(), cols) };
                            #[cfg(not(target_arch = "aarch64"))]
                            let acca = dot_f32_neon(&buf[r1 * cols..(r1 + 1) * cols], xr);
                            ya[(t + tk) * rows + row0 + r1] = acca;
                            #[cfg(target_arch = "aarch64")]
                            let accb = unsafe {
                                dot_f32_neon(buf.as_ptr().add((ROW_PANEL + r1) * cols), xr.as_ptr(), cols)
                            };
                            #[cfg(not(target_arch = "aarch64"))]
                            let accb = dot_f32_neon(&buf[(ROW_PANEL + r1) * cols..(ROW_PANEL + r1 + 1) * cols], xr);
                            yb[(t + tk) * rows + row0 + r1] = accb;
                        }
                    }
                    t += 4;
                }
                while t < t_count {
                    let xr = &x[t * cols..(t + 1) * cols];
                    for ri in 0..panel_rows {
                        #[cfg(target_arch = "aarch64")]
                        let acc = unsafe { dot_f32_neon(buf.as_ptr().add(ri * cols), xr.as_ptr(), cols) };
                        #[cfg(not(target_arch = "aarch64"))]
                        let acc = dot_f32_neon(&buf[ri * cols..(ri + 1) * cols], xr);
                        ya[t * rows + row0 + ri] = acc;
                        #[cfg(target_arch = "aarch64")]
                        let accb = unsafe {
                            dot_f32_neon(buf.as_ptr().add((ROW_PANEL + ri) * cols), xr.as_ptr(), cols)
                        };
                        #[cfg(not(target_arch = "aarch64"))]
                        let accb = dot_f32_neon(&buf[(ROW_PANEL + ri) * cols..(ROW_PANEL + ri + 1) * cols], xr);
                        yb[t * rows + row0 + ri] = accb;
                    }
                    t += 1;
                }
            }
        });
    });
    if let Some(t0) = t_start {
        GEMM_PROF_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        GEMM_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
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
    gemm_pair_batched(w1, w3, mmap, x, t_count, &mut gate[..needed_gate], &mut up[..needed_gate], threads);
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

                    let t_att = if prof { Some(std::time::Instant::now()) } else { None };
                    {
                        let qp = SendMutPtr(qbuf.as_mut_ptr());
                        let kp = SendMutPtr(kbuf.as_mut_ptr());
                        let vp = SendMutPtr(vbuf.as_mut_ptr());
                        let hbp = SendMutPtr(hb.as_mut_ptr());
                        let kcp = SendMutPtr((&mut *self.k_cache).as_mut_ptr());
                        let vcp = SendMutPtr((&mut *self.v_cache).as_mut_ptr());
                        let qlp = SendPtr(q_ln_ptr);
                        let klp = SendPtr(k_ln_ptr);
                        let q_rows_r = q_rows;
                        let k_rows_r = k_rows;
                        let v_rows_r = v_rows;
                        // Phase 1: head norms + RoPE + KV writes (each token's
                        // writes land in its own column).
                        super::dynamic_ffn::parallel_for(0, t_count, threads, move |t0, t1| unsafe {
                            attention_prep_tokens(
                                qp, kp, vp, kcp, vcp, qlp, klp,
                                t0, t1, start_pos,
                                q_rows_r, k_rows_r, v_rows_r, layer_idx,
                                head_dim, n_heads, n_kv_heads, max_seq_len, freq_base, norm_eps,
                            );
                        });
                        // Phase 2: attention over the COMPLETE cache.
                        let qp_read = SendPtr(qp.0);
                        let kcp_read = SendPtr(kcp.0);
                        let vcp_read = SendPtr(vcp.0);
                        let hbp2 = SendMutPtr(hb.as_mut_ptr());
                        super::dynamic_ffn::parallel_for(0, t_count, threads, move |t0, t1| unsafe {
                            attention_mix_tokens(
                                qp_read, hbp2, kcp_read, vcp_read,
                                t0, t1, start_pos,
                                q_rows_r, layer_idx, head_dim, n_heads, n_kv_heads, max_seq_len,
                            );
                        });
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
            let rms_ms = RMS_PROF_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            let add_ms = ADD_PROF_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            let swi_ms = SWI_PROF_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            let emb_ms = EMBED_PROF_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
            let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[lfm2-prof] t={} wall={:.1}ms gemm={:.1}ms/calls={} rms={:.1} add={:.1} swiglu={:.1} embed={:.1} conv={:.1}ms attn={:.1}ms other={:.1}ms",
                t_count, wall_ms, gemm_ms, calls, rms_ms, add_ms, swi_ms, emb_ms,
                conv_ns as f64 / 1e6, attn_ns as f64 / 1e6,
                wall_ms - gemm_ms - rms_ms - add_ms - swi_ms - emb_ms - conv_ns as f64 / 1e6 - attn_ns as f64 / 1e6
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
        if rms_rel >= 2e-2 {
            eprintln!(
                "NOTE batched-vs-per-token rel RMS {:.6} ({})",
                rms_rel,
                if std::env::var("SOFUU_PRE8").as_deref() == Ok("1") { "int8 path" } else { "f32 path" }
            );
            if std::env::var("SOFUU_PRE8").as_deref() != Ok("1") {
                panic!("batched prefill logits diverge (f32 path): rel RMS {:.6}", rms_rel);
            }
        }
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
    fn tile4x4_i8_matches_exact_int_reference() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let cols = PRE8_GROUP * 3; // 3 groups, multiple of PRE8_GROUP
        // Exact integers: weights/acts in [-100, 100] with scale 1.0
        let w: Vec<i8> = (0..4 * cols)
            .map(|i| (((i * 37 + 5) % 201) as i32 - 100) as i8)
            .collect();
        let x: Vec<i8> = (0..4 * cols)
            .map(|i| (((i * 23 + 11) % 201) as i32 - 100) as i8)
            .collect();
        let n_groups = cols / PRE8_GROUP;
        // all scales 1.0: the kernel must be bit-exact against i64 dots
        let ws = vec![1.0f32; 4 * n_groups];
        let ascale = vec![1.0f32; 4 * n_groups];
        let mut fout = [0.0f32; 16];
        unsafe {
            gemm_tile4x4_i8(
                w.as_ptr(),
                x.as_ptr(),
                cols,
                ws.as_ptr(),
                ascale.as_ptr(),
                n_groups,
                &mut fout,
            );
        }
        for ri in 0..4 {
            for tk in 0..4 {
                let mut exact: i64 = 0;
                for c in 0..cols {
                    exact += w[ri * cols + c] as i64 * x[tk * cols + c] as i64;
                }
                let got = fout[ri * 4 + tk];
                let rel = ((got - exact as f32) / (exact.abs() as f32).max(1.0)).abs();
                assert!(
                    rel < 1e-5,
                    "tile[{}][{}] got {} exact {} rel {:.2e}",
                    ri, tk, got, exact, rel
                );
            }
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

        // Second sweep: amortizes any one-time prep (e.g. the SOFUU_PRE8=1
        // weight slab build) out of the numbers.
        let t2 = std::time::Instant::now();
        m.forward_batched(&tokens, 0);
        let batch2_secs = t2.elapsed().as_secs_f64();

        eprintln!(
            "lfm2 prefill 200 tokens: per-token {:.2} tk/s | batched {:.2} tk/s (warm {:.2} tk/s) ({:.2}x)",
            200.0 / loop_secs,
            200.0 / batch_secs,
            200.0 / batch2_secs,
            loop_secs / batch2_secs
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

        /// Phase-2 int8 path: the SDOT tile with two-sided per-band scaling must
    /// stay within rel-RMS < 1e-2 of the f32 path AND must be greedy-safe:
    /// per-token argmax over rows is identical (what the sampler actually
    /// consumes), the same gate the ternary path ships with.
    #[test]
    fn pre8_sdot_tile_rel_rms_greedy() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let cols = 2048usize;
        let (rows, tokens) = (4usize, 8usize); // single panel: bisect validity
        // Weights with realistic FFN-like spread across three magnitude bands.
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let band = i / cols % 3;
                let base = [0.35f32, 0.012, 0.004][band];
                (((i % 29) as f32 - 14.5) / 14.5) * base
            })
            .collect();
        let x: Vec<f32> = (0..tokens * cols)
            .map(|i| (((i * 7 + i / cols) % 23) as f32 - 11.5) / 11.5 * 0.9)
            .collect();

        // reference: plain f32 dots
        let mut ref_out = vec![0f32; rows * tokens];
        for r in 0..rows {
            for t in 0..tokens {
                ref_out[t * rows + r] = w[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x[t * cols..(t + 1) * cols].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
        }

        // int8 path: shared act quantization + panel weight quantization
        let aq = quantize_acts_i8(&x, tokens, cols);
        let mut got_all = vec![0f32; rows * tokens];
        let groups = aq.groups;
        let n = aq.n;
        // one 4-row panel (pad remaining rows with zeros)
        let panel_buf = vec![0f32; 4 * n]; // zero pad (row 4..4 unused; rows 0..4 real)
        let mut wq = vec![0i8; 4 * n];
        let mut ws = vec![0f32; 4 * groups];
        let mut all_ok = true;
        let mut worst_rms = 0f32;
        for r0 in (0..rows).step_by(4) {
            let mut pbuf = vec![0f32; 4 * n];
            for ri in 0..4.min(rows - r0) {
                pbuf[ri * n..ri * n + cols].copy_from_slice(&w[(r0 + ri) * cols..(r0 + ri) * cols + cols]);
            }
            quantize_w_panel_i8(&pbuf, 4, cols, n, groups, &mut wq, &mut ws);
            let mut t = 0usize;
            while t + 2 <= tokens {
                let mut out = [0f32; 8];
                unsafe {
                    gemm_tile4x2_sdot(wq.as_ptr(), ws.as_ptr(), aq.q.as_ptr().add(t * n), aq.scales.as_ptr().add(t * groups), n, groups, &mut out);
                }
                for ti in 0..2 {
                    for ri in 0..4 {
                        if r0 + ri >= rows {
                            continue;
                        }
                        got_all[(t + ti) * rows + r0 + ri] = out[ri * 2 + ti];
                    }
                }
                t += 2;
            }
            if t < tokens {
                let mut out = [0f32; 8];
                unsafe {
                    gemm_tile4x2_sdot(wq.as_ptr(), ws.as_ptr(), aq.q.as_ptr().add(t * n), aq.scales.as_ptr().add(t * groups), n, groups, &mut out);
                }
            }
        }
        // ── bisect: kernel vs scalar int8 reference (same quantized inputs)
        let mut scalar_i8 = vec![0f32; rows * tokens];
        for t in 0..tokens {
            for r in 0..rows {
                let pr = r % 4; // rows were quantized in panels of 4 (r0=0 here)
                let mut acc = 0f32;
                for g in 0..groups {
                    let g0 = g * PRE8_GROUP;
                    let mut dot = 0i32;
                    for i in g0..g0 + PRE8_GROUP {
                        dot += wq[pr * n + i] as i32 * aq.q[t * n + i] as i32;
                    }
                    acc += dot as f32 * ws[pr * groups + g] * aq.scales[t * groups + g];
                }
                scalar_i8[t * rows + r] = acc;
            }
        }
        let mut kdiff = 0f32;
        for t in 0..tokens {
            for r in 0..4.min(rows) {
                kdiff = kdiff.max((got_all[t * rows + r] - scalar_i8[t * rows + r]).abs());
            }
        }
        eprintln!("[pre8-dbg] kernel-vs-scalar-i8 maxdiff={kdiff}; sample got={} scalar={}", got_all[0], scalar_i8[0]);

        // vector-level gates (the plan's acceptance): rel-RMS < 1e-2 over the
        // whole output, and per-token argmax over rows identical to f32.
        let mut sq_diff = 0f32;
        let mut sq_ref = 0f32;
        for i in 0..ref_out.len() {
            let d = got_all[i] - ref_out[i];
            sq_diff += d * d;
            sq_ref += ref_out[i] * ref_out[i];
        }
        let rel = (sq_diff / sq_ref.max(1e-12)).sqrt();
        worst_rms = rel;
        if rel >= 1e-2 {
            for i in 0..8 {
                eprintln!("[pre8-dbg] i={i} got={} want={}", got_all[i], ref_out[i]);
            }
        }
        assert!(rel < 1e-2, "rel-RMS {rel} exceeded 1e-2");
        for t in 0..tokens {
            let a = (0..rows).max_by(|&r1, &r2| ref_out[t * rows + r1].total_cmp(&ref_out[t * rows + r2])).unwrap();
            let b = (0..rows).max_by(|&r1, &r2| got_all[t * rows + r1].total_cmp(&got_all[t * rows + r2])).unwrap();
            assert_eq!(a, b, "token {t}: argmax {} vs {} (f32 vs pre8)", a, b);
        }
    }

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
mod lfm2_i8_kernel_probe {
    use super::*;

    #[test]
    #[ignore = "timing probe; run explicitly"]
    fn tile4x4_i8_gflops_single_thread() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let cols = 2048usize;
        assert_eq!(cols % PRE8_GROUP, 0);
        let n_groups = cols / PRE8_GROUP;
        let w: Vec<i8> = (0..4 * cols).map(|i| (i % 251) as i8).collect();
        let x: Vec<i8> = (0..4 * cols).map(|i| ((i % 241) as i8 as i32 - 120) as i8).collect();
        let ws = vec![0.001f32; 4 * n_groups];
        let ascale = vec![0.02f32; 4 * n_groups];
        let tiles = 100_000usize;
        let mut out = [0f32; 16];
        let start = std::time::Instant::now();
        let mut sink = 0f32;
        for t in 0..tiles {
            unsafe {
                gemm_tile4x4_i8(
                    w.as_ptr(),
                    x.as_ptr(),
                    cols,
                    ws.as_ptr(),
                    ascale.as_ptr(),
                    n_groups,
                    &mut out,
                );
            }
            sink += out[5];
        }
        let secs = start.elapsed().as_secs_f64();
        let gmac = tiles as f64 * 16.0 * cols as f64 / 1e9;
        eprintln!(
            "tile4x4 i8: {:.1} GMAC/s = {:.1} GFLOP/s single-thread (sink {:.4})",
            gmac / secs,
            2.0 * gmac / secs,
            sink
        );
        let _ = (sink, start);
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
