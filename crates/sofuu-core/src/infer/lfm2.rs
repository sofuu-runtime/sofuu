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
#[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
    pub metal_engine: Option<MetalEngine>,
    #[cfg(target_os = "macos")]
    pub metal_mmap_buf: Option<MetalBuffer>,

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

        #[cfg(target_os = "macos")]
        let (metal_engine, metal_mmap_buf) = {
            if let Some(engine) = MetalEngine::new() {
                let mmap_slice = sf.mmap.as_slice();
                let mmap_buf = engine.register_zerocopy_buffer(mmap_slice);
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

            #[cfg(target_os = "macos")]
            metal_engine,
            #[cfg(target_os = "macos")]
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
        })
    }

    pub fn reset_cache(&mut self) {
        for state in &mut self.conv_states {
            state.fill(0.0f32);
        }
        self.k_cache.fill(0.0f32);
        self.v_cache.fill(0.0f32);
    }

    pub fn compact_kv_cache(&mut self, _cur_pos: usize, _sink: usize, _drop: usize) -> usize {
        self.reset_cache();
        0
    }

    // Single vector-matrix multiply: y = W * x
    pub fn compute_matvec(
        mmap: &super::mmap::MmapFile,
        #[cfg(target_os = "macos")]
        metal_engine: Option<&MetalEngine>,
        #[cfg(target_os = "macos")]
        metal_mmap_buf: Option<&MetalBuffer>,
        lt: &LinearTensor,
        x: &[f32],
        y: &mut [f32],
    ) {
        match lt {
            LinearTensor::Bf16(tv) => {
                #[cfg(target_os = "macos")]
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
                #[cfg(target_os = "macos")]
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
                    );
                    return;
                }

                // Multi-threaded CPU vectorized MLX 4-bit affine GEMV across all CPU P-cores
                let w_slice = mmap.slice(weight.data_offset, weight.size_bytes).unwrap();
                let s_slice = mmap.slice(scales.data_offset, scales.size_bytes).unwrap();
                let b_slice = mmap.slice(biases.data_offset, biases.size_bytes).unwrap();
                let rows = weight.shape[0];
                let cols = x.len();
                super::quant::gemv_mlx4(w_slice, s_slice, b_slice, x, &mut y[..rows], rows, cols, *group_size);
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
        #[cfg(target_os = "macos")]
        let (m_eng, m_buf) = (self.metal_engine.as_ref(), self.metal_mmap_buf.as_ref());

        // 2. Decoder layers
        for layer_idx in 0..n_layers {
            let layer = self.layers[layer_idx].clone();
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
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
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
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
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
                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
                        &w1,
                        &self.x_norm,
                        &mut self.ffn_gate,
                    );
                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
                        &w3,
                        &self.x_norm,
                        &mut self.ffn_up,
                    );

                    #[cfg(target_os = "macos")]
                    if let Some(engine) = m_eng {
                        engine.swiglu(&mut self.ffn_gate, &self.ffn_up);
                    } else {
                        super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);
                    }
                    #[cfg(not(target_os = "macos"))]
                    super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);

                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
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
                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
                        &q_proj,
                        &self.x_norm,
                        &mut self.q,
                    );
                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
                        &k_proj,
                        &self.x_norm,
                        &mut self.k,
                    );
                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
                        &v_proj,
                        &self.x_norm,
                        &mut self.v,
                    );

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
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
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
                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
                        &w1,
                        &self.x_norm,
                        &mut self.ffn_gate,
                    );
                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
                        &w3,
                        &self.x_norm,
                        &mut self.ffn_up,
                    );

                    #[cfg(target_os = "macos")]
                    if let Some(engine) = m_eng {
                        engine.swiglu(&mut self.ffn_gate, &self.ffn_up);
                    } else {
                        super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);
                    }
                    #[cfg(not(target_os = "macos"))]
                    super::quant::swiglu_in_place(&mut self.ffn_gate, &self.ffn_up);

                    Self::compute_matvec(
                        mmap,
                        #[cfg(target_os = "macos")] m_eng,
                        #[cfg(target_os = "macos")] m_buf,
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

        // 4. Logits: tied embedding lm_head = embed_tokens @ x_norm
        Self::compute_matvec(
            mmap,
            #[cfg(target_os = "macos")] m_eng,
            #[cfg(target_os = "macos")] m_buf,
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
        for (i, &t) in tokens.iter().enumerate() {
            self.forward_prefill(t, start_pos + i);
        }
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        let n = tokens.len();
        if n == 0 {
            return &self.logits;
        }
        for (i, &t) in tokens[..n - 1].iter().enumerate() {
            self.forward_prefill(t, start_pos + i);
        }
        self.forward(tokens[n - 1], start_pos + n - 1)
    }
}
