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

use super::gguf::{GgufFile, GgufValue, TensorInfo};
use super::quant::{
    batched_add_bias, batched_mad, batched_rmsnorm, batched_rmsnorm_in_place, batched_rope_kv,
    batched_rope_kv_f16,
    batched_swiglu, matmat_residual, matmat_pair,
    batched_attention_ext, batched_attention_ext_k16, dequantize_q4_0_row, dequantize_q4_k_row,
    dequantize_q6_k_row, dequantize_q8_0_row, f16_to_f32, kv_compact_style, matmat, matvec,
    prefetch_streaming_slice, prefetch_temporal_slice, prof,
    rmsnorm, rmsnorm_in_place, rope_style, swiglu_in_place, tree_attention, tree_rope, type_row_size,
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
};

/// Mirror a freshly written f32 KV row into the f16 shadow. Both stores stay
/// on every write path, so the shadow can never describe a stale cache.
#[inline]
fn kv16_store(shadow: Option<&mut AnonBuffer>, slot: usize, src: &[f32]) {
    if let Some(buf) = shadow {
        let dst = buf.as_u16_mut();
        super::quant::kv16_convert(src, &mut dst[slot..slot + src.len()]);
    }
}

/// 128 leaves a half-long tail chunk, which measures better on the Pi 5 than
/// two full 64 chunks (+4% Qwen prefill: one chunk less, better tile amortization;
/// Spark/Llama-1B neutral). Mostly GEMM-bucket time — attention is untouched.
pub const DEFAULT_PREFILL_BATCH: usize = 128;

/// Cap on tokens per prefill batch chunk. Overridable at process start with
/// `SOFUU_PREFILL_BATCH` (snapped to a multiple of 4 so the tiled GEMM path
/// stays eligible); read once per process.
pub fn max_prefill_batch() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("SOFUU_PREFILL_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= 4)
            .map(|v| (v / 4) * 4)
            .unwrap_or(DEFAULT_PREFILL_BATCH)
    })
}

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

impl AnonBuffer {
    /// f16 view of the same allocation: one f32 slot holds two u16 halves.
    /// Used by the KV shadow, which is allocated at half the f32 element
    /// count so its byte footprint equals the f16 payload exactly.
    #[inline]
    pub fn as_u16(&self) -> &[u16] {
        if self.len == 0 || self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr as *const u16, self.len * 2) }
        }
    }

    #[inline]
    pub fn as_u16_mut(&mut self) -> &mut [u16] {
        if self.len == 0 || self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u16, self.len * 2) }
        }
    }
}

#[inline(always)]
fn inline_mad(dst: &mut [f32], src: &[f32], scalar: f32) {
    let _p = super::quant::prof::Scope::new(&super::quant::prof::GLUE_NS, &super::quant::prof::GLUE_CALLS);
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

    /// GPU2 (PLAN-GPU-METAL): GGUF decode client. When armed, execute_layers
    /// runs the seven per-layer GEMVs through Metal batches (three commits
    /// per layer + the lm_head); norms, RoPE, the KV cache and attention stay
    /// on the CPU, shared verbatim with the CPU path. SOFUU_GPU_DECODE=1.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub gpu: Option<GgufGpu>,

    // Batched working state buffers for prompt prefill
    pub x_batch: Vec<f32>,
    pub x_norm_batch: Vec<f32>,
    /// True while the gpu slab's xn region holds output-normed chain rows
    /// (last GPU spec pass) — enables the lazy per-row head commit.
    pub spec_rows_valid: bool,
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
    /// f16 shadows of the KV caches, read by the CPU attention path to halve
    /// KV DRAM traffic (llama.cpp's default KV precision). `None` when the
    /// f16 path is inactive; the f32 caches remain authoritative either way.
    pub k16_cache: Option<AnonBuffer>,
    pub v16_cache: Option<AnonBuffer>,
    pub kv_f16: bool,
    /// Element-weighted dominant quant type of the projection matrices.
    pub dominant_proj_type: u32,
    pub kv_dim: usize,

    // Sliding Window Attention & Attention Sinks
    pub sliding_window: usize,
    pub sink_tokens: usize,
    pub is_gemma2: bool,
    pub dynamic_ffn: super::dynamic_ffn::DynamicFfnConfig,
    pub kivi: Option<super::kivi::KiviCache>,
    pub streaming_weights: bool,
    pub tree_k_layers: Vec<f32>,
    pub tree_v_layers: Vec<f32>,
    pub tree_max_nodes: usize,
}

impl TransformerModel {
    pub fn from_gguf(gguf: GgufFile, max_seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        Self::from_gguf_with_gpu(gguf, max_seq_len, vocab_size, true)
    }

    /// `gpu_allowed = false` (engine --cpu / CPU-load contract) keeps the
    /// SOFUU_GPU_DECODE client off even when the env flag is set.
    pub fn from_gguf_with_gpu(
        gguf: GgufFile,
        max_seq_len: usize,
        vocab_size: usize,
        gpu_allowed: bool,
    ) -> Result<Self, String> {
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
            // Gate unsupported quant types at load with a loud, actionable
            // error instead of a mid-forward panic (type_row_size's fallback
            // sizes Q3_K/IQ* as f32 and corrupts the slice math). The llama
            // matvec path handles exactly these types.
            const SUPPORTED: [u32; 6] = [
                crate::infer::quant::GGML_TYPE_F32,
                crate::infer::quant::GGML_TYPE_F16,
                crate::infer::quant::GGML_TYPE_Q4_0,
                crate::infer::quant::GGML_TYPE_Q8_0,
                crate::infer::quant::GGML_TYPE_Q4_K,
                crate::infer::quant::GGML_TYPE_Q6_K,
            ];
            if !SUPPORTED.contains(&info.tensor_type) {
                return Err(format!(
                    "tensor `{name}` uses GGML type {} — not supported by the CPU GGUF backend (supported: f32, f16, q4_0, q8_0, q4_k, q6_k). Re-quantize the model to one of these, e.g. `llama-quantize` to Q4_K_M.",
                    info.tensor_type
                ));
            }
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
        // Tied embeddings: fall back to token_embd ONLY when output.weight is
        // genuinely absent. A present-but-invalid output.weight (e.g. an
        // unsupported quant type) must fail the load — the type gate's error
        // is not a tie signal.
        let output = match find_tensor("output.weight") {
            Ok(tv) => tv,
            Err(e) => {
                if gguf.tensors.contains_key("output.weight") {
                    return Err(e);
                }
                token_embd.clone()
            }
        };

        // Dominant projection quant type: drives the speculative tree default.
        // q6_K's batched GEMM has no 8-act panel kernel, so a batch-8 draft
        // step is compute-bound; measured on the Pi, a rows=3 tree (batch 4)
        // beats rows=5 there, while q4_K/q8_0 models prefer rows=5.
        let mut type_elems: Vec<(u32, u64)> = Vec::new();
        let mut note_type = |ty: u32, shape: &Vec<usize>| {
            if shape.len() != 2 {
                return;
            }
            let elems = (shape[0] as u64) * (shape[1] as u64);
            match type_elems.iter_mut().find(|(t, _)| *t == ty) {
                Some((_, e)) => *e += elems,
                None => type_elems.push((ty, elems)),
            }
        };
        note_type(token_embd.tensor_type, &token_embd.shape);
        note_type(output.tensor_type, &output.shape);

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

        for l in &layers {
            for tv in [&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output, &l.ffn_gate, &l.ffn_up, &l.ffn_down] {
                note_type(tv.tensor_type, &tv.shape);
            }
        }
        let dominant_proj_type = type_elems
            .iter()
            .max_by_key(|(_, e)| *e)
            .map(|(t, _)| *t)
            .unwrap_or(GGML_TYPE_F32);

        // Warm packed-tile GEMM/GEMV cache for all quantized 2-D matrices
        // (q8_0 / q4_K / q6_K). Non-quantized or odd-geometry tensors are
        // ignored by the tile layer and fall back to the generic kernels.
        for tv in std::iter::once(&token_embd)
            .chain(std::iter::once(&output))
            .chain(layers.iter().flat_map(|l| {
                [&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output, &l.ffn_gate, &l.ffn_up, &l.ffn_down]
            }))
        {
            if matches!(tv.tensor_type, GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K) && tv.shape.len() == 2 {
                if let Some(slice) = gguf.mmap.slice(gguf.data_offset + tv.offset, 64) {
                    super::quant::tile_warm(slice.as_ptr(), tv.shape[1], tv.shape[0], tv.tensor_type);
                }
            }
        }

        // KV caches hoisted so the GPU client can register them zero-copy:
        // one physical cache, written by whichever path runs (GPU kernels or
        // the CPU fallback), read by both.
        let k_cache = AnonBuffer::new(kv_cache_size);
        let v_cache = AnonBuffer::new(kv_cache_size);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let gpu = {
            let wanted = gpu_allowed
                && cfg!(target_os = "macos")
                && std::env::var_os("SOFUU_GPU_DECODE").is_some()
                && gguf.resident.is_none();
            if wanted {
                GgufGpu::arm(&gguf, &config, &layers, vocab_size, &k_cache, &v_cache, is_gemma2, &output, &output_norm)
            } else {
                None
            }
        };
        // f16 KV shadows: CPU-only Linux (the Pi lane). The GPU client binds
        // the f32 caches zero-copy, KIVI owns its own packed store, and the
        // Android app keeps its tuned f32 path — all of those stay f32.
        // SOFUU_KV_F16=0 is the kill switch.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let gpu_armed = gpu.is_some();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let gpu_armed = false;
        let kv_f16 = cfg!(all(target_os = "linux", not(target_os = "android")))
            && !gpu_armed
            && kv_cache_size % 2 == 0
            && std::env::var("SOFUU_KV_F16").map(|v| v != "0").unwrap_or(true);
        let k16_cache = kv_f16.then(|| AnonBuffer::new(kv_cache_size / 2));
        let v16_cache = kv_f16.then(|| AnonBuffer::new(kv_cache_size / 2));
        if kv_f16 && std::env::var_os("SOFUU_PROFILE").is_some() {
            let feat = super::quant::simd_features();
            eprintln!(
                "[infer] f16 KV shadows on ({} MB x2, CPU attention path, vector f16 kernels: {})",
                kv_cache_size / (2 << 20),
                if super::quant::kv16_fast_path() { "yes" } else { "NO (scalar fallback)" }
            );
            eprintln!(
                "[infer] simd features: neon={} dotprod={} fp16v={} i8mm={}",
                feat & 1 != 0,
                feat & 2 != 0,
                feat & 4 != 0,
                feat & 8 != 0
            );
        }

        Ok(Self {
            config,
            gguf,
            token_embd,
            output_norm,
            output,
            layers,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            gpu,

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

            x_batch: vec![0.0f32; max_prefill_batch() * dim],
            x_norm_batch: vec![0.0f32; max_prefill_batch() * dim],
            spec_rows_valid: false,
            q_batch: vec![0.0f32; max_prefill_batch() * q_dim],
            k_batch: vec![0.0f32; max_prefill_batch() * kv_dim],
            v_batch: vec![0.0f32; max_prefill_batch() * kv_dim],
            head_out_batch: vec![0.0f32; max_prefill_batch() * q_dim],
            attn_out_batch: vec![0.0f32; max_prefill_batch() * dim],
            ffn_gate_batch: vec![0.0f32; max_prefill_batch() * hidden_dim],
            ffn_up_batch: vec![0.0f32; max_prefill_batch() * hidden_dim],
            ffn_out_batch: vec![0.0f32; max_prefill_batch() * dim],

            k_cache,
            v_cache,
            k16_cache,
            v16_cache,
            kv_f16,
            dominant_proj_type,
            kv_dim,
            sliding_window: detected_sliding_window,
            sink_tokens: 4,
            is_gemma2,
            dynamic_ffn: super::dynamic_ffn::DynamicFfnConfig::default(),
            kivi: None,
            streaming_weights: false,
            tree_k_layers: vec![0.0f32; n_layers * 16 * kv_dim],
            tree_v_layers: vec![0.0f32; n_layers * 16 * kv_dim],
            tree_max_nodes: 16,
        })
    }

    /// Returns the effective sliding window for a given layer.
    /// Gemma 2 alternates: even layers (0, 2, 4...) use local sliding window (4096),
    /// while odd layers (1, 3, 5...) use full global attention (sliding_window = 0).
    /// Mistral and other models apply uniform sliding window across all layers.
    #[inline]
    /// Default speculative tree size for this model. q6_K projections have no
    /// 8-act panel GEMM, so a rows=5 draft (batch 8) is compute-bound; device
    /// sweeps on the Pi put rows=3 (batch 4) ~7% ahead there, while q4_K and
    /// q8_0 models prefer the rows=5 tree.
    #[inline]
    pub fn spec_rows_default(&self) -> usize {
        if self.dominant_proj_type == GGML_TYPE_Q6_K { 3 } else { 5 }
    }

    /// Decode-phase thread count (auto mode, Pi lane). q8_0's GEMV is a pure
    /// DRAM stream: two streams already saturate the Pi 5 memory controller
    /// and four streams thrash it (SmolLM2 26.5→31.5 t/s, 1B 16.2→18.3).
    /// k-quant GEMVs carry per-block scale math and keep every core.
    #[inline]
    pub fn decode_threads_auto(&self, max_threads: usize) -> usize {
        #[cfg(all(target_os = "linux", not(target_os = "android")))]
        if self.dominant_proj_type == GGML_TYPE_Q8_0 {
            return max_threads.min(2);
        }
        max_threads
    }

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
        if let Some(k16) = self.k16_cache.as_mut() {
            k16.as_u16_mut().fill(0);
        }
        if let Some(v16) = self.v16_cache.as_mut() {
            v16.as_u16_mut().fill(0);
        }
        self.kivi = None;
    }

    /// Configure Non-Temporal Streaming Weights (Cache-Bypass).
    /// When enabled, advises the OS kernel sequentially via libc::madvise(MADV_SEQUENTIAL)
    /// and activates non-temporal streaming prefetch in the layer execution pipeline to bypass
    /// CPU cache thrashing during single-token autoregressive decoding.
    pub fn set_streaming_weights(&mut self, enabled: bool) {
        self.streaming_weights = enabled;
        if enabled {
            self.gguf.mmap.advise_sequential();
        } else {
            self.gguf.mmap.advise_normal();
        }
    }

    /// Enable KIVI 2-bit or 4-bit Asymmetric Quantized KV Cache.
    /// Ingests any prefilled tokens from `k_cache` and `v_cache` into historical quantized blocks.
    pub fn enable_kivi(&mut self, bits: usize, cur_pos: usize) {
        let kivi_bits = match bits {
            2 => super::kivi::KiviBits::Two,
            _ => super::kivi::KiviBits::Four,
        };
        let config = super::kivi::KiviConfig {
            enabled: true,
            bits: kivi_bits,
            residual_window: 32,
            group_size: 32,
        };
        let mut kivi = super::kivi::KiviCache::new(
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
            config,
        );

        let n_layers = self.config.n_layers;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let max_seq_len = self.config.max_seq_len;

        // Ingest prefilled tokens from k_cache & v_cache
        if cur_pos > 0 {
            for layer_idx in 0..n_layers {
                for p in 0..cur_pos {
                    let cache_pos = p % max_seq_len;
                    for kv_h in 0..n_kv_heads {
                        let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim + cache_pos * head_dim;
                        let k_slice = &self.k_cache[slot..slot + head_dim];
                        let v_slice = &self.v_cache[slot..slot + head_dim];
                        let head_idx = layer_idx * n_kv_heads + kv_h;
                        kivi.heads[head_idx].append(k_slice, v_slice);
                    }
                }
            }
        }

        self.kivi = Some(kivi);
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
        // Re-mirror the compacted window into the f16 shadow (rare path; a
        // stale shadow here would feed attention wrong history).
        if self.kv_f16 {
            let new_len = current_seq_len - drop_tokens;
            let hd = self.config.head_dim;
            let seq = self.config.max_seq_len;
            let heads = self.config.n_layers * self.config.n_kv_heads;
            let k16 = self.k16_cache.as_mut().unwrap().as_u16_mut();
            let v16 = self.v16_cache.as_mut().unwrap().as_u16_mut();
            for h in 0..heads {
                let base = h * seq * hd;
                for p in 0..new_len {
                    let row = base + p * hd;
                    super::quant::kv16_convert(
                        &self.k_cache[row..row + hd],
                        &mut k16[row..row + hd],
                    );
                    super::quant::kv16_convert(
                        &self.v_cache[row..row + hd],
                        &mut v16[row..row + hd],
                    );
                }
            }
        }
        current_seq_len - drop_tokens
    }

    /// Run single token step forward through the transformer.
    /// Returns reference to computed logits [vocab_size].
    pub fn forward(&mut self, token_id: u32, pos: usize) -> &[f32] {
        // A serial step between chain passes invalidates the saved rows.
        self.spec_rows_valid = false;
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
        let _p = super::quant::prof::Scope::new(&super::quant::prof::EMBED_NS, &super::quant::prof::EMBED_CALLS);
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
    /// redundant memory bandwidth stalls. Slices large token streams into max_prefill_batch() tiles,
    /// snapped to a multiple of 4 so the packed-tile GEMM path stays eligible for every chunk.
    pub fn forward_prefill_batch(&mut self, tokens: &[u32], mut start_pos: usize) {
        if tokens.is_empty() {
            return;
        }
        if tokens.len() == 1 {
            self.forward_prefill(tokens[0], start_pos);
            return;
        }
        for chunk in tokens.chunks(max_prefill_batch()) {
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

        let batch = max_prefill_batch();
        let mut cur_pos = start_pos;
        let mut off = 0;

        while off < tokens.len() {
            let take = (tokens.len() - off).min(batch);
            if off + take < tokens.len() {
                self.forward_prefill_batch_chunk(&tokens[off..off + take], cur_pos);
                cur_pos += take;
                off += take;
                continue;
            }
            // Final chunk: keep the main body a multiple of 4 (tile-eligible) and
            // push any 1-3 row remainder through the per-row/small-batch path.
            let main = take - (take % 4);
            if main == take {
                return self.forward_prefill_batch_chunk_internal(&tokens[off..off + take], cur_pos, true);
            }
            self.forward_prefill_batch_chunk_internal(&tokens[off..off + main], cur_pos, false);
            cur_pos += main;
            return self.forward_prefill_batch_chunk_internal(&tokens[off + main..off + take], cur_pos, true);
        }
        &self.logits
    }

    fn forward_prefill_batch_chunk(&mut self, tokens: &[u32], start_pos: usize) {
        self.forward_prefill_batch_chunk_internal(tokens, start_pos, false);
    }

    #[inline(never)]
    fn forward_batch_layers_chunk(&mut self, tokens: &[u32], start_pos: usize) {
        let b_size = tokens.len();
        if b_size == 0 {
            return;
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
            batched_rmsnorm(
                &mut self.x_norm_batch[0..b_size * dim],
                &self.x_batch[0..b_size * dim],
                attn_norm_weights,
                b_size,
                dim,
                norm_eps,
            );

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
                batched_add_bias(&mut self.q_batch[0..b_size * q_dim], b_f32, b_size, q_dim);
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
                batched_add_bias(&mut self.k_batch[0..b_size * kv_dim], b_f32, b_size, kv_dim);
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
                batched_add_bias(&mut self.v_batch[0..b_size * kv_dim], b_f32, b_size, kv_dim);
            }

            // ── Per-token QK-Norm ────────────────────────────────
            if let Some(ref q_norm) = layer.attn_q_norm {
                let q_norm_bytes = Self::tensor_slice(gguf, q_norm);
                let q_norm_w = unsafe {
                    std::slice::from_raw_parts(q_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                batched_rmsnorm_in_place(
                    &mut self.q_batch[0..b_size * q_dim],
                    q_norm_w,
                    b_size * n_heads,
                    head_dim,
                    norm_eps,
                );
            }
            if let Some(ref k_norm) = layer.attn_k_norm {
                let k_norm_bytes = Self::tensor_slice(gguf, k_norm);
                let k_norm_w = unsafe {
                    std::slice::from_raw_parts(k_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                batched_rmsnorm_in_place(
                    &mut self.k_batch[0..b_size * kv_dim],
                    k_norm_w,
                    b_size * n_kv_heads,
                    head_dim,
                    norm_eps,
                );
            }

            // ── RoPE & KV-Cache Storage for batch ────────────────
            let rope_mode = if self.config.rope_interleaved { 1 } else { 0 };
            if self.kv_f16 {
                batched_rope_kv_f16(
                    &mut self.q_batch[0..b_size * q_dim],
                    &mut self.k_batch[0..b_size * kv_dim],
                    &self.v_batch[0..b_size * kv_dim],
                    &mut self.k_cache,
                    &mut self.v_cache,
                    self.k16_cache.as_mut().unwrap().as_u16_mut(),
                    self.v16_cache.as_mut().unwrap().as_u16_mut(),
                    b_size,
                    start_pos,
                    layer_idx,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq_len,
                    rope_freq_base,
                    rope_mode,
                );
            } else {
                batched_rope_kv(
                    &mut self.q_batch[0..b_size * q_dim],
                    &mut self.k_batch[0..b_size * kv_dim],
                    &self.v_batch[0..b_size * kv_dim],
                    &mut self.k_cache,
                    &mut self.v_cache,
                    b_size,
                    start_pos,
                    layer_idx,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq_len,
                    rope_freq_base,
                    rope_mode,
                );
            }

            // ── Multi-Head / Grouped-Query Attention for batch ───
            // Batched prefill keeps the f32 score kernel: at 8+ rows the
            // shares-K-strip pass is compute-bound, and the f16 converts cost
            // more than the halved bytes save there. The f16 shadows are still
            // written above, so decode reads them.
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

            // ── Attention Output Projection + Residual 1 (fused when the
            // tile path takes it; bit-identical either way) ──────────────
            let attn_out_bytes = Self::tensor_slice(gguf, &layer.attn_output);
            if !matmat_residual(
                attn_out_bytes,
                layer.attn_output.tensor_type,
                dim,
                q_dim,
                &self.head_out_batch[0..b_size * q_dim],
                b_size,
                &mut self.x_batch[0..b_size * dim],
            ) {
                matmat(
                    attn_out_bytes,
                    layer.attn_output.tensor_type,
                    dim,
                    q_dim,
                    &self.head_out_batch[0..b_size * q_dim],
                    b_size,
                    &mut self.attn_out_batch[0..b_size * dim],
                );
                batched_mad(
                    &mut self.x_batch[0..b_size * dim],
                    &self.attn_out_batch[0..b_size * dim],
                    1.0,
                    b_size,
                    dim,
                );
            }

            // ── FFN Norm for batch ──────────────────────────────
            let ffn_norm_bytes = Self::tensor_slice(gguf, &layer.ffn_norm);
            let ffn_norm_weights = unsafe {
                std::slice::from_raw_parts(ffn_norm_bytes.as_ptr() as *const f32, dim)
            };
            batched_rmsnorm(
                &mut self.x_norm_batch[0..b_size * dim],
                &self.x_batch[0..b_size * dim],
                ffn_norm_weights,
                b_size,
                dim,
                norm_eps,
            );

            // ── Batched SwiGLU FFN: gate+up share one act-quant dispatch ──
            let gate_bytes = Self::tensor_slice(gguf, &layer.ffn_gate);
            let up_bytes = Self::tensor_slice(gguf, &layer.ffn_up);
            if !matmat_pair(
                gate_bytes,
                up_bytes,
                layer.ffn_gate.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm_batch[0..b_size * dim],
                b_size,
                &mut self.ffn_gate_batch[0..b_size * hidden_dim],
                &mut self.ffn_up_batch[0..b_size * hidden_dim],
            ) {
                matmat(
                    gate_bytes,
                    layer.ffn_gate.tensor_type,
                    hidden_dim,
                    dim,
                    &self.x_norm_batch[0..b_size * dim],
                    b_size,
                    &mut self.ffn_gate_batch[0..b_size * hidden_dim],
                );

                matmat(
                    up_bytes,
                    layer.ffn_up.tensor_type,
                    hidden_dim,
                    dim,
                    &self.x_norm_batch[0..b_size * dim],
                    b_size,
                    &mut self.ffn_up_batch[0..b_size * hidden_dim],
                );
            }

            batched_swiglu(
                &mut self.ffn_gate_batch[0..b_size * hidden_dim],
                &self.ffn_up_batch[0..b_size * hidden_dim],
                b_size,
                hidden_dim,
            );

            let down_bytes = Self::tensor_slice(gguf, &layer.ffn_down);
            if self.dynamic_ffn.enabled {
                for item in 0..b_size {
                    self.dynamic_ffn.sparse_down_proj(
                        down_bytes,
                        layer.ffn_down.tensor_type,
                        dim,
                        hidden_dim,
                        &self.ffn_gate_batch[item * hidden_dim..(item + 1) * hidden_dim],
                        &mut self.ffn_out_batch[item * dim..(item + 1) * dim],
                    );
                }
            } else if !matmat_residual(
                down_bytes,
                layer.ffn_down.tensor_type,
                dim,
                hidden_dim,
                &self.ffn_gate_batch[0..b_size * hidden_dim],
                b_size,
                &mut self.x_batch[0..b_size * dim],
            ) {
                matmat(
                    down_bytes,
                    layer.ffn_down.tensor_type,
                    dim,
                    hidden_dim,
                    &self.ffn_gate_batch[0..b_size * hidden_dim],
                    b_size,
                    &mut self.ffn_out_batch[0..b_size * dim],
                );
                batched_mad(
                    &mut self.x_batch[0..b_size * dim],
                    &self.ffn_out_batch[0..b_size * dim],
                    1.0,
                    b_size,
                    dim,
                );
            }
        }
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

        let chunk_t0 = prof::enabled().then(std::time::Instant::now);

        // P1 (llama.cpp parity): B<=32 chunks through the prefill GEMM class
        // (SOFUU_GPU_PREFILL_GEMM=1). Any unsupported chunk falls back to the
        // b4 route / CPU tiles below.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if std::env::var("SOFUU_GPU_PREFILL_GEMM")
            .map(|v| v == "1")
            .unwrap_or(false)
            && b_size >= 2
            && self.gpu_forward_prefill_gemm_chunks(tokens, start_pos, compute_last_logits)
        {
            return &self.logits;
        }

        // GPU route: B≤4 chunk through the chain pass in ONE commit (the
        // b4 GEMVs; head GEMV only when the caller needs last-row logits).
        // Anything unsupported (b>4, boundaries) falls to the CPU tiles.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.gpu_forward_prefill_chain(tokens, start_pos, compute_last_logits) {
            return &self.logits;
        }

        self.forward_batch_layers_chunk(tokens, start_pos);

        if compute_last_logits {
            let dim = self.config.dim;
            let norm_eps = self.config.norm_eps;
            let last_b = b_size - 1;
            let x_last = &self.x_batch[last_b * dim..(last_b + 1) * dim];
            let norm_bytes = Self::tensor_slice(&self.gguf, &self.output_norm);
            let norm_w = unsafe {
                std::slice::from_raw_parts(norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm(&mut self.x_norm, x_last, norm_w, norm_eps);

            let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
            matvec(
                out_bytes,
                self.output.tensor_type,
                self.config.vocab_size,
                dim,
                &self.x_norm,
                &mut self.logits,
            );
        }

        if let Some(t0) = chunk_t0 {
            prof::record_chunk(t0.elapsed(), b_size);
            prof::report();
        }
        &self.logits
    }

    /// Evaluates speculative candidate tokens in parallel in a single forward pass,
    /// streaming model weights from DRAM exactly ONCE.
    ///
    /// Computes and returns logits for token 0 (the base token to verify candidate).
    /// If candidate matches, `forward_speculative_head_item` computes logits for the candidate
    /// without re-evaluating any transformer layers!
    #[inline(never)]
    pub fn forward_speculative_batch(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        out_logits: &mut [f32],
    ) {
        let b_size = tokens.len();
        if b_size == 0 {
            return;
        }
        let vocab_size = self.config.vocab_size;
        let dim = self.config.dim;
        let norm_eps = self.config.norm_eps;

        if b_size == 1 {
            let l = self.forward(tokens[0], start_pos);
            out_logits[0..vocab_size].copy_from_slice(l);
            return;
        }

        self.forward_batch_layers_chunk(tokens, start_pos);

        let norm_bytes = Self::tensor_slice(&self.gguf, &self.output_norm);
        let norm_w = unsafe {
            std::slice::from_raw_parts(norm_bytes.as_ptr() as *const f32, dim)
        };
        for b in 0..b_size {
            let x_b = &self.x_batch[b * dim..(b + 1) * dim];
            let x_norm_b = &mut self.x_norm_batch[b * dim..(b + 1) * dim];
            rmsnorm(x_norm_b, x_b, norm_w, norm_eps);
        }

        // Compute LM head ONLY for token 0 (saving 150k-300k dot products if candidate misses!)
        let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
        matvec(
            out_bytes,
            self.output.tensor_type,
            vocab_size,
            dim,
            &self.x_norm_batch[0..dim],
            &mut out_logits[0..vocab_size],
        );
        self.logits.copy_from_slice(&out_logits[0..vocab_size]);
    }

    /// Lazy per-row LM head for verify_path's accepted rows: ONE GPU head
    /// GEMV commit over the output-normed xn row the chain pass left in the
    /// slab. The lm_head is ~514MB of q6_K for the 8B vocab, so it must run
    /// only for rows verify actually samples (the CPU matvec fallback
    /// measured ~290ms per row under load; the GPU commit ~13ms).
    /// Returns false — caller falls back to forward_speculative_head_item —
    /// when no chain pass covers this step.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn spec_chain_head_logits(&mut self, item_idx: usize, out_logits: &mut [f32]) -> bool {
        if !self.spec_rows_valid || item_idx >= 4 {
            return false;
        }
        let g = match self.gpu.as_mut() {
            Some(g) if g.token_commit => g,
            _ => return false,
        };
        let dim = self.config.dim;
        let vocab_size = self.config.vocab_size;
        let gguf = &self.gguf;
        g.engine.batch_open();
        g.engine.batch_gguf(
            self.output.tensor_type,
            &g.w_buf, g.w_offset(gguf, &self.output),
            &g.act_buf, g.slots.xn + item_idx * dim * 4,
            &g.act_buf, g.slots.logits,
            vocab_size, dim,
        );
        g.engine.batch_end();
        out_logits[0..vocab_size].copy_from_slice(g.slot_f32(g.slots.logits, vocab_size));
        self.logits[0..vocab_size].copy_from_slice(&out_logits[0..vocab_size]);
        true
    }

    /// No GPU slab on other targets: the caller's CPU fallback is the path.
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn spec_chain_head_logits(&mut self, _item_idx: usize, _out_logits: &mut [f32]) -> bool {
        false
    }

    /// Compute LM head logits for an already normalized speculative batch item (e.g. candidate upon hit).
    #[inline(never)]
    pub fn forward_speculative_head_item(&mut self, item_idx: usize, out_logits: &mut [f32]) {
        let dim = self.config.dim;
        let vocab_size = self.config.vocab_size;
        let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
        matvec(
            out_bytes,
            self.output.tensor_type,
            vocab_size,
            dim,
            &self.x_norm_batch[item_idx * dim..(item_idx + 1) * dim],
            &mut out_logits[0..vocab_size],
        );
        self.logits.copy_from_slice(&out_logits[0..vocab_size]);
    }

    /// Multi-Path Lookahead Tree Speculation:
    /// Evaluates root token and all candidate tree branches in parallel in a single forward pass,
    /// streaming model weights from DRAM exactly ONCE.
    ///
    /// Tokens: [root_token, cand_1, cand_2, ...] (length M)
    /// Positions: [pos_0, pos_1, pos_2, ...] (length M)
    /// Ancestry mask: [M * M] u8 row-major matrix: 1 if col j is ancestor of row i or j == i, 0 otherwise.
    ///
    /// Computes and returns logits for token 0 (root). If any candidate branch matches,
    /// `forward_speculative_head_item` computes logits for the accepted branch nodes lazily,
    /// and `commit_speculative_candidates` commits only the accepted branch KV vectors to persistent cache.
    #[inline(never)]
    pub fn forward_speculative_tree(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        ancestry_mask: &[u8],
        start_pos: usize,
        out_logits: &mut [f32],
    ) {
        let b_size = tokens.len();
        if b_size == 0 {
            return;
        }
        let vocab_size = self.config.vocab_size;
        let dim = self.config.dim;
        let norm_eps = self.config.norm_eps;

        if b_size == 1 {
            let l = self.forward(tokens[0], start_pos);
            out_logits[0..vocab_size].copy_from_slice(l);
            return;
        }

        // Any non-GPU-chain route invalidates the saved per-row logits.
        self.spec_rows_valid = false;

        // GPU route: chain-shaped verify pass (root + candidates at strictly
        // consecutive positions) through the one-commit client. Anything else
        // (true multi-branch trees, unsupported shapes) falls through to the
        // CPU tree path below, untouched.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            if self.gpu_forward_spec_chain(tokens, positions, start_pos, out_logits) {
                return;
            }
        }

        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
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

            // ── Attention Norm for tree batch ───────────────────────
            let attn_norm_bytes = Self::tensor_slice(gguf, &layer.attn_norm);
            let attn_norm_weights = unsafe {
                std::slice::from_raw_parts(attn_norm_bytes.as_ptr() as *const f32, dim)
            };
            batched_rmsnorm(
                &mut self.x_norm_batch[0..b_size * dim],
                &self.x_batch[0..b_size * dim],
                attn_norm_weights,
                b_size,
                dim,
                norm_eps,
            );

            // ── Batched Q, K, V Projections via matmat ──────────────
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
                batched_add_bias(&mut self.q_batch[0..b_size * q_dim], b_f32, b_size, q_dim);
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
                batched_add_bias(&mut self.k_batch[0..b_size * kv_dim], b_f32, b_size, kv_dim);
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
                batched_add_bias(&mut self.v_batch[0..b_size * kv_dim], b_f32, b_size, kv_dim);
            }

            // ── Per-token QK-Norm ───────────────────────────────────
            if let Some(ref q_norm) = layer.attn_q_norm {
                let q_norm_bytes = Self::tensor_slice(gguf, q_norm);
                let q_norm_w = unsafe {
                    std::slice::from_raw_parts(q_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                batched_rmsnorm_in_place(
                    &mut self.q_batch[0..b_size * q_dim],
                    q_norm_w,
                    b_size * n_heads,
                    head_dim,
                    norm_eps,
                );
            }
            if let Some(ref k_norm) = layer.attn_k_norm {
                let k_norm_bytes = Self::tensor_slice(gguf, k_norm);
                let k_norm_w = unsafe {
                    std::slice::from_raw_parts(k_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                batched_rmsnorm_in_place(
                    &mut self.k_batch[0..b_size * kv_dim],
                    k_norm_w,
                    b_size * n_kv_heads,
                    head_dim,
                    norm_eps,
                );
            }

            // ── Tree RoPE for each candidate position ───────────────
            let rope_mode = if self.config.rope_interleaved { 1 } else { 0 };
            tree_rope(
                &mut self.q_batch[0..b_size * q_dim],
                &mut self.k_batch[0..b_size * kv_dim],
                positions,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_freq_base,
                rope_mode,
            );

            // ── Save candidate keys and values to tree_k_layers & tree_v_layers ───
            for b in 0..b_size {
                let base = (layer_idx * self.tree_max_nodes + b) * kv_dim;
                self.tree_k_layers[base..base + kv_dim]
                    .copy_from_slice(&self.k_batch[b * kv_dim..(b + 1) * kv_dim]);
                self.tree_v_layers[base..base + kv_dim]
                    .copy_from_slice(&self.v_batch[b * kv_dim..(b + 1) * kv_dim]);
            }

            // ── Node 0 is the root token: commit its K and V to persistent KV cache directly ───
            let cache_pos = start_pos % max_seq_len;
            for kv_h in 0..n_kv_heads {
                let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim + cache_pos * head_dim;
                let k_src = &self.k_batch[kv_h * head_dim..(kv_h + 1) * head_dim];
                let v_src = &self.v_batch[kv_h * head_dim..(kv_h + 1) * head_dim];
                self.k_cache[slot..slot + head_dim].copy_from_slice(k_src);
                self.v_cache[slot..slot + head_dim].copy_from_slice(v_src);
                kv16_store(self.k16_cache.as_mut(), slot, k_src);
                kv16_store(self.v16_cache.as_mut(), slot, v_src);
            }

            // ── 2D Block-Causal Tree Self-Attention ─────────────────
            let sliding_window = self.layer_sliding_window(layer_idx);
            tree_attention(
                &self.q_batch[0..b_size * q_dim],
                &self.k_batch[0..b_size * kv_dim],
                &self.v_batch[0..b_size * kv_dim],
                &self.k_cache,
                &self.v_cache,
                &mut self.head_out_batch[0..b_size * q_dim],
                ancestry_mask,
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

            // ── Attention Output Projection via matmat ───────────────
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

            // ── Residual Connection 1 ───────────────────────────────
            batched_mad(
                &mut self.x_batch[0..b_size * dim],
                &self.attn_out_batch[0..b_size * dim],
                1.0,
                b_size,
                dim,
            );

            // ── FFN Norm for tree batch ─────────────────────────────
            let ffn_norm_bytes = Self::tensor_slice(gguf, &layer.ffn_norm);
            let ffn_norm_weights = unsafe {
                std::slice::from_raw_parts(ffn_norm_bytes.as_ptr() as *const f32, dim)
            };
            batched_rmsnorm(
                &mut self.x_norm_batch[0..b_size * dim],
                &self.x_batch[0..b_size * dim],
                ffn_norm_weights,
                b_size,
                dim,
                norm_eps,
            );

            // ── Batched SwiGLU FFN via matmat ───────────────────────
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

            batched_swiglu(
                &mut self.ffn_gate_batch[0..b_size * hidden_dim],
                &self.ffn_up_batch[0..b_size * hidden_dim],
                b_size,
                hidden_dim,
            );

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

            // ── Residual Connection 2 ───────────────────────────────
            batched_mad(
                &mut self.x_batch[0..b_size * dim],
                &self.ffn_out_batch[0..b_size * dim],
                1.0,
                b_size,
                dim,
            );
        }

        // ── Output Norm for all tree nodes ──────────────────────────
        let norm_bytes = Self::tensor_slice(&self.gguf, &self.output_norm);
        let norm_w = unsafe {
            std::slice::from_raw_parts(norm_bytes.as_ptr() as *const f32, dim)
        };
        for b in 0..b_size {
            let x_b = &self.x_batch[b * dim..(b + 1) * dim];
            let x_norm_b = &mut self.x_norm_batch[b * dim..(b + 1) * dim];
            rmsnorm(x_norm_b, x_b, norm_w, norm_eps);
        }

        // ── Compute LM head ONLY for token 0 (the root token) ───────
        let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
        matvec(
            out_bytes,
            self.output.tensor_type,
            vocab_size,
            dim,
            &self.x_norm_batch[0..dim],
            &mut out_logits[0..vocab_size],
        );
        self.logits.copy_from_slice(&out_logits[0..vocab_size]);
    }

    /// Commit accepted candidate nodes from `tree_k_layers` and `tree_v_layers`
    /// into the persistent KV cache starting at `start_pos + 1`.
    pub fn commit_speculative_candidates(&mut self, accepted_batch_indices: &[usize], start_pos: usize) {
        let n_layers = self.config.n_layers;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let max_seq_len = self.config.max_seq_len;
        let kv_dim = self.kv_dim;

        for (offset, &batch_idx) in accepted_batch_indices.iter().enumerate() {
            let dest_pos = (start_pos + 1 + offset) % max_seq_len;
            for l in 0..n_layers {
                let base = (l * self.tree_max_nodes + batch_idx) * kv_dim;
                let k_cand = &self.tree_k_layers[base..base + kv_dim];
                let v_cand = &self.tree_v_layers[base..base + kv_dim];

                for kv_h in 0..n_kv_heads {
                    let slot = (l * n_kv_heads + kv_h) * max_seq_len * head_dim + dest_pos * head_dim;
                    let k_row = &k_cand[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let v_row = &v_cand[kv_h * head_dim..(kv_h + 1) * head_dim];
                    self.k_cache[slot..slot + head_dim].copy_from_slice(k_row);
                    self.v_cache[slot..slot + head_dim].copy_from_slice(v_row);
                    kv16_store(self.k16_cache.as_mut(), slot, k_row);
                    kv16_store(self.v16_cache.as_mut(), slot, v_row);
                }
            }
        }
    }

    /// Prefill a token: updates KV cache for the token without computing logits.
    pub fn forward_prefill(&mut self, token_id: u32, pos: usize) {
        self.forward_impl(token_id, pos, false);
    }

    pub fn forward_embed(&mut self, token_id: u32, out_hidden: &mut [f32]) {
        let dim = self.config.dim;
        Self::lookup_embedding(&self.gguf, &self.token_embd, token_id, dim, &mut out_hidden[..dim]);
    }

    pub fn execute_layers(&mut self, pos: usize, start_layer: usize, end_layer: usize) {
        // GPU2: whole-layer Metal batches when armed and the CPU-side
        // special paths are all off (streaming/dynamic-FFN/KIVI fall back).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.gpu.is_some()
            && !self.streaming_weights
            && !self.dynamic_ffn.enabled
            && self.kivi.is_none()
        {
            self.execute_layers_gpu(pos, start_layer, end_layer);
            return;
        }
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let rope_freq_base = self.config.rope_freq_base;
        let max_seq_len = self.config.max_seq_len;
        let gguf = &self.gguf;

        let end = end_layer.min(self.config.n_layers);
        let streaming = self.streaming_weights;
        for layer_idx in start_layer..end {
            let layer = &self.layers[layer_idx];

            let attn_norm_bytes = Self::tensor_slice(gguf, &layer.attn_norm);
            let q_bytes = Self::tensor_slice(gguf, &layer.attn_q);
            let k_bytes = Self::tensor_slice(gguf, &layer.attn_k);
            let v_bytes = Self::tensor_slice(gguf, &layer.attn_v);
            let attn_out_bytes = Self::tensor_slice(gguf, &layer.attn_output);
            let ffn_norm_bytes = Self::tensor_slice(gguf, &layer.ffn_norm);
            let gate_bytes = Self::tensor_slice(gguf, &layer.ffn_gate);
            let up_bytes = Self::tensor_slice(gguf, &layer.ffn_up);
            let down_bytes = Self::tensor_slice(gguf, &layer.ffn_down);

            if streaming {
                // Non-temporal prefetch Q & K projection weights
                prefetch_streaming_slice(q_bytes);
                prefetch_streaming_slice(k_bytes);
                // Temporal prefetch working hidden states
                prefetch_temporal_slice(&self.x);
            }

            // ── Attention Norm ──────────────────────────────
            let attn_norm_weights = unsafe {
                std::slice::from_raw_parts(attn_norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm(&mut self.x_norm, &self.x, attn_norm_weights, norm_eps);

            // ── Q, K, V Projections ─────────────────────────
            if streaming {
                prefetch_streaming_slice(v_bytes);
            }
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

            if streaming {
                prefetch_streaming_slice(attn_out_bytes);
            }
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
            // One batched call per tensor instead of per-head FFI rounds:
            // at 128-wide heads and 24 heads per layer these were ~670
            // near-empty C calls per decode token.
            if let Some(ref q_norm) = layer.attn_q_norm {
                let q_norm_bytes = Self::tensor_slice(gguf, q_norm);
                let q_norm_w = unsafe {
                    std::slice::from_raw_parts(q_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                batched_rmsnorm_in_place(&mut self.q, q_norm_w, n_heads, head_dim, norm_eps);
            }
            if let Some(ref k_norm) = layer.attn_k_norm {
                let k_norm_bytes = Self::tensor_slice(gguf, k_norm);
                let k_norm_w = unsafe {
                    std::slice::from_raw_parts(k_norm_bytes.as_ptr() as *const f32, head_dim)
                };
                batched_rmsnorm_in_place(&mut self.k, k_norm_w, n_kv_heads, head_dim, norm_eps);
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
                kv16_store(self.k16_cache.as_mut(), slot, k_in);
                kv16_store(self.v16_cache.as_mut(), slot, v_in);
            }

            // ── Multi-Head / Grouped-Query Attention ────────
            if streaming {
                let cache_len = (cache_pos + 1).min(max_seq_len);
                let slot_start = layer_idx * n_kv_heads * max_seq_len * head_dim;
                let slot_len = n_kv_heads * cache_len * head_dim;
                prefetch_temporal_slice(&self.k_cache[slot_start..slot_start + slot_len]);
                prefetch_temporal_slice(&self.v_cache[slot_start..slot_start + slot_len]);
                prefetch_temporal_slice(&self.q);
                prefetch_streaming_slice(gate_bytes);
                prefetch_streaming_slice(up_bytes);
            }
            if let Some(ref mut kivi) = self.kivi {
                kivi.append_token(layer_idx, &self.k, &self.v);
                kivi.compute_layer_attention(layer_idx, n_heads, &self.q, &mut self.head_out);
            } else if self.kv_f16 {
                let sliding_window = self.layer_sliding_window(layer_idx);
                batched_attention_ext_k16(
                    &self.q,
                    self.k16_cache.as_ref().unwrap().as_u16(),
                    self.v16_cache.as_ref().unwrap().as_u16(),
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
            } else {
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
            }

            // ── Attention Output Projection ─────────────────
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
            let ffn_norm_weights = unsafe {
                std::slice::from_raw_parts(ffn_norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm(&mut self.x_norm, &self.x, ffn_norm_weights, norm_eps);

            // ── SwiGLU FFN ──────────────────────────────────
            if streaming {
                prefetch_streaming_slice(down_bytes);
            }
            matvec(
                gate_bytes,
                layer.ffn_gate.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm,
                &mut self.ffn_gate_buf,
            );

            matvec(
                up_bytes,
                layer.ffn_up.tensor_type,
                hidden_dim,
                dim,
                &self.x_norm,
                &mut self.ffn_up_buf,
            );

            swiglu_in_place(&mut self.ffn_gate_buf, &self.ffn_up_buf);

            if streaming {
                if layer_idx + 1 < end {
                    let next_layer = &self.layers[layer_idx + 1];
                    prefetch_streaming_slice(Self::tensor_slice(gguf, &next_layer.attn_q));
                } else {
                    prefetch_streaming_slice(Self::tensor_slice(gguf, &self.output));
                }
            }

            if self.dynamic_ffn.enabled {
                self.dynamic_ffn.sparse_down_proj(
                    down_bytes,
                    layer.ffn_down.tensor_type,
                    dim,
                    hidden_dim,
                    &self.ffn_gate_buf,
                    &mut self.ffn_out,
                );
            } else {
                matvec(
                    down_bytes,
                    layer.ffn_down.tensor_type,
                    dim,
                    hidden_dim,
                    &self.ffn_gate_buf,
                    &mut self.ffn_out,
                );
            }

            // ── Residual Connection 2 ───────────────────────
            inline_mad(&mut self.x, &self.ffn_out, 1.0);
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
        let dim = self.config.dim;
        self.x[..dim].copy_from_slice(&in_hidden[..dim]);
        self.execute_layers(pos, start_layer, end_layer);
        out_hidden[..dim].copy_from_slice(&self.x[..dim]);
    }

    pub fn forward_head(&mut self, in_hidden: &[f32], out_logits: &mut [f32]) {
        let dim = self.config.dim;
        let norm_eps = self.config.norm_eps;
        self.x[..dim].copy_from_slice(&in_hidden[..dim]);
        let out_norm_bytes = Self::tensor_slice(&self.gguf, &self.output_norm);
        let out_norm_weights = unsafe {
            std::slice::from_raw_parts(out_norm_bytes.as_ptr() as *const f32, dim)
        };
        rmsnorm_in_place(&mut self.x, out_norm_weights, norm_eps);

        let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
        if self.streaming_weights {
            prefetch_streaming_slice(out_bytes);
            prefetch_temporal_slice(&self.x);
        }
        matvec(
            out_bytes,
            self.output.tensor_type,
            self.config.vocab_size,
            dim,
            &self.x,
            out_logits,
        );
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn ggpu_head_active(&self) -> bool {
        self.gpu.is_some()
            && matches!(
                self.output.tensor_type,
                GGML_TYPE_Q4_0 | GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K
            )
    }

    /// lm_head as one Metal batch (CPU norm; staged slot in, logits out).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn ggpu_head(&mut self) {
        if let Some(g) = self.gpu.as_mut() {
            let dim = self.config.dim;
            let vocab = self.config.vocab_size;
            let self_x: &mut [f32] = &mut self.x;
            let dst = g.slot_f32_mut(g.slots.xn, dim);
            dst[..dim].copy_from_slice(&self_x[..dim]);
            let off = g.w_offset(&self.gguf, &self.output);
            assert!(g.engine.batch_open());
            g.engine.batch_gguf(
                self.output.tensor_type,
                &g.w_buf, off,
                &g.act_buf, g.slots.xn,
                &g.act_buf, g.slots.logits,
                vocab, dim,
            );
            g.engine.batch_end();
            let src = g.slot_f32(g.slots.logits, vocab);
            self.logits.copy_from_slice(src);
        }
    }

    /// GPU2 decode path: same CPU glue (norms, QK-norms, RoPE, KV, attention,
    /// residuals) with the seven per-layer GEMVs batched into three commits.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn execute_layers_gpu(&mut self, pos: usize, start_layer: usize, end_layer: usize) {
        let end = end_layer.min(self.config.n_layers);
        // Wrap-around (context past max_seq_len) stays on the 3-commit path,
        // whose KV writes match the CPU modulo arithmetic.
        if self.gpu.as_ref().map(|g| g.one_commit).unwrap_or(false) && pos < self.config.max_seq_len {
            self.layers_one_commit(pos, start_layer, end);
            return;
        }
        let windows: Vec<usize> = (0..self.config.n_layers)
            .map(|i| self.layer_sliding_window(i))
            .collect();
        for layer_idx in start_layer..end {
            self.gpu_layer_step(layer_idx, pos, windows[layer_idx]);
        }
    }

    /// GPU2.1 decode step: ALL layers across ONE commit each. Norms run as
    /// in-batch kernels reading f32 weights straight from the weight mmap;
    /// q/k RoPE + QK-norm fuse into the KV-write/attention kernels; x, the
    /// activation slots and the KV caches (registered zero-copy, shared with
    /// the CPU) are all command-buffer-resident between layers; x syncs back
    /// to Vec once per token.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn layers_one_commit(&mut self, pos: usize, start_layer: usize, end_layer: usize) {
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let rope_freq_base = self.config.rope_freq_base;
        let max_seq_len = self.config.max_seq_len;
        let rope_mode: u32 = if self.config.rope_interleaved { 1 } else { 0 };
        let gguf = &self.gguf;

        let g = self.gpu.as_mut().expect("one_commit without gpu");
        let token_commit = g.token_commit && start_layer == 0 && end_layer == self.config.n_layers;
        g.slot_f32_mut(g.slots.x, dim).copy_from_slice(&self.x[..dim]);
        let prof = std::env::var_os("SOFUU_GPU_PROFILE").is_some();
        let t_start = std::time::Instant::now();
        if token_commit {
            assert!(g.engine.batch_open());
        }

        for layer_idx in start_layer..end_layer {
            let (attn_norm_off, ffn_norm_off, q_norm_off, k_norm_off) = {
                let l = &self.layers[layer_idx];
                (
                    gguf.data_offset + l.attn_norm.offset,
                    gguf.data_offset + l.ffn_norm.offset,
                    l.attn_q_norm.as_ref().map(|tv| gguf.data_offset + tv.offset),
                    l.attn_k_norm.as_ref().map(|tv| gguf.data_offset + tv.offset),
                )
            };
            let has_q = u32::from(q_norm_off.is_some());
            let has_k = u32::from(k_norm_off.is_some());
            let kv_head_bytes = n_kv_heads * max_seq_len * head_dim * 4;
            let kc_layer = layer_idx * kv_head_bytes;
            let meta_chunks = (max_seq_len + 127) / 128;
            let meta_per_layer = n_heads * meta_chunks * 2 + n_heads * meta_chunks * head_dim;
            let (kln_buf, kln_off) = (k_norm_off.map(|off| &g.w_buf), k_norm_off.unwrap_or(0));
            let l = &self.layers[layer_idx];
            if !token_commit {
                assert!(g.engine.batch_open());
            }
            g.engine.batch_rmsnorm_f32w(
                &g.act_buf, g.slots.x,
                &g.w_buf, attn_norm_off,
                &g.act_buf, g.slots.xn,
                dim, norm_eps,
            );
            g.engine.batch_gguf(l.attn_q.tensor_type, &g.w_buf,
                gguf.data_offset + l.attn_q.offset,
                &g.act_buf, g.slots.xn, &g.act_buf, g.slots.qa,
                n_heads * head_dim, dim);
            g.engine.batch_gguf(l.attn_k.tensor_type, &g.w_buf,
                gguf.data_offset + l.attn_k.offset,
                &g.act_buf, g.slots.xn, &g.act_buf, g.slots.ka,
                n_kv_heads * head_dim, dim);
            g.engine.batch_gguf(l.attn_v.tensor_type, &g.w_buf,
                gguf.data_offset + l.attn_v.offset,
                &g.act_buf, g.slots.xn, &g.act_buf, g.slots.va,
                n_kv_heads * head_dim, dim);
            let run_attn = !g.skip_attn;
            if run_attn {
            g.engine.batch_llama_kv_write(
                &g.act_buf, g.slots.ka,
                &g.act_buf, g.slots.va,
                &g.k_buf, kc_layer,
                &g.v_buf, kc_layer,
                kln_buf, kln_off,
                has_k, pos as u32,
                head_dim as u32, n_kv_heads as u32, max_seq_len as u32,
                norm_eps, rope_freq_base, rope_mode,
            );
            // GPU3 chunked attention (default; the meta-overlap bug is fixed —
            // head-0 partials were overwriting the {max,sum} block).
            // SOFUU_GPU_ATTN=0 restores the pre-GPU3 serial kernel.
            let attn_new = std::env::var("SOFUU_GPU_ATTN")
                .map(|v| v != "0")
                .unwrap_or(true);
            if attn_new {
                // GPU3: q-norm + q-RoPE as their own dispatch (no intra-dispatch
                // race), then chunked attention partials + finish rescale.
                g.engine.batch_head_rmsnorm(
                    &g.act_buf, g.slots.qa,
                    q_norm_off.map(|_| &g.w_buf), q_norm_off.unwrap_or(0),
                    head_dim as u32, n_heads as u32,
                    has_q, pos as u32, norm_eps, rope_freq_base, rope_mode,
                );
                g.engine.batch_llama_attn_part(
                    &g.act_buf, g.slots.qa,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    None, 0,
                    &g.meta_buf, layer_idx * meta_per_layer * 4,
                    0, pos as u32,
                    head_dim as u32, n_heads as u32, n_kv_heads as u32, max_seq_len as u32,
                    norm_eps, rope_freq_base, rope_mode,
                );
                g.engine.batch_llama_attn_fin(
                    &g.meta_buf, layer_idx * meta_per_layer * 4,
                    &g.act_buf, g.slots.ao,
                    pos as u32, n_heads as u32, head_dim as u32,
                );
            } else {
                let (qln_buf, qln_off) =
                    (q_norm_off.map(|off| &g.w_buf), q_norm_off.unwrap_or(0));
                g.engine.batch_llama_attn(
                    &g.act_buf, g.slots.qa,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    qln_buf, qln_off,
                    &g.act_buf, g.slots.ao,
                    has_q, pos as u32,
                    head_dim as u32, n_heads as u32, n_kv_heads as u32, max_seq_len as u32,
                    norm_eps, rope_freq_base, rope_mode,
                );
            }
            let trace = std::env::var_os("SOFUU_GPU_TRACE").is_some();
            if trace && (layer_idx == 0 || layer_idx == 17 || layer_idx == 35) {
                let ao = g.slot_f32(g.slots.ao, n_heads * head_dim);
                let r: f32 = (ao.iter().map(|v| v * v).sum::<f32>() / (n_heads * head_dim) as f32)
                    .sqrt();
                eprintln!("[trace] L{layer_idx} pos{pos} attn_out_rms={r:.6}");
            }
            }
            g.engine.batch_gguf(l.attn_output.tensor_type, &g.w_buf,
                gguf.data_offset + l.attn_output.offset,
                &g.act_buf, g.slots.ao, &g.act_buf, g.slots.xo,
                dim, n_heads * head_dim);
            g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, dim);
            g.engine.batch_rmsnorm_f32w(
                &g.act_buf, g.slots.x,
                &g.w_buf, ffn_norm_off,
                &g.act_buf, g.slots.xn,
                dim, norm_eps,
            );
            let run_ffn = !g.skip_ffn;
            if run_ffn {
            g.engine.batch_gguf(l.ffn_gate.tensor_type, &g.w_buf,
                gguf.data_offset + l.ffn_gate.offset,
                &g.act_buf, g.slots.xn, &g.act_buf, g.slots.gate,
                hidden_dim, dim);
            g.engine.batch_gguf(l.ffn_up.tensor_type, &g.w_buf,
                gguf.data_offset + l.ffn_up.offset,
                &g.act_buf, g.slots.xn, &g.act_buf, g.slots.up,
                hidden_dim, dim);
            g.engine.batch_swiglu(&g.act_buf, g.slots.gate, &g.act_buf, g.slots.up, hidden_dim);
            g.engine.batch_gguf(l.ffn_down.tensor_type, &g.w_buf,
                gguf.data_offset + l.ffn_down.offset,
                &g.act_buf, g.slots.gate, &g.act_buf, g.slots.xo,
                dim, hidden_dim);
            g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, dim);
            }
            if !token_commit {
                g.engine.batch_end();
            }
        }
        let host_us = t_start.elapsed().as_micros() as u64;
        let gguf2 = &self.gguf;
        let g = self.gpu.as_mut().expect("one_commit without gpu");
        let head_on = token_commit && !g.skip_head;
        if head_on {
            // Final norm + lm_head in the SAME command buffer, then one drain.
            g.engine.batch_rmsnorm_f32w(
                &g.act_buf, g.slots.x,
                &g.w_buf, gguf2.data_offset + self.output_norm.offset,
                &g.act_buf, g.slots.xn,
                dim, self.config.norm_eps,
            );
            g.engine.batch_gguf(
                self.output.tensor_type,
                &g.w_buf, g.w_offset(gguf2, &self.output),
                &g.act_buf, g.slots.xn,
                &g.act_buf, g.slots.logits,
                self.config.vocab_size, dim,
            );
            g.engine.batch_end();
            self.logits[..self.config.vocab_size]
                .copy_from_slice(g.slot_f32(g.slots.logits, self.config.vocab_size));
        }
        if prof {
            let total_us = t_start.elapsed().as_micros() as u64;
            PROFILE_HOST_US.fetch_add(host_us, std::sync::atomic::Ordering::Relaxed);
            PROFILE_TOTAL_US.fetch_add(total_us, std::sync::atomic::Ordering::Relaxed);
            let h = PROFILE_HOST_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            let tt = PROFILE_TOTAL_US.swap(0, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[gpu-profile] token: host_encode={}us total_after_wait={}us gpu_busy~={}us",
                h,
                tt,
                tt.saturating_sub(h)
            );
        }
        self.x[..dim].copy_from_slice(g.slot_f32(g.slots.x, dim));
    }

    /// GPU spec pass eligibility, for the generate loop's draft shaping.
    /// True only when the whole-token one-commit client is armed and the
    /// spec chain is not disabled (SOFUU_GPU_SPEC=0 or a SOFUU_GPU_SKIP
    /// ablation).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn gpu_spec_chain_armed(&self) -> bool {
        match self.gpu.as_ref() {
            Some(g) => g.spec_chain_ready(),
            None => false,
        }
    }

    /// GPU3/G4: speculative-verify CHAIN pass (root + up to 3 candidates at
    /// strictly consecutive positions start_pos..start_pos+B-1) through the
    /// whole-token one-commit client. GEMVs run as B=4 batched q4_K GEMVs
    /// (batch_gguf_b4: the DRAM weight stream — the dominant cost — is
    /// amortized over all B rows); norms/adds/swiglu and attention run per
    /// row inside the SAME command buffer. In-order execution + per-op
    /// buffer barriers give row t's attention visibility of row t-1's KV
    /// written earlier in the same commit — exactly the chain-causal
    /// semantics (for chains, is_ancestor == prefix, so the CPU tree path's
    /// ancestry mask is satisfied by construction).
    ///
    /// Candidate K/V land in the persistent cache at their positions; a
    /// rejected candidate's stale rows are always rewritten before any
    /// later pass can attend them (each row's KV write precedes its own
    /// attention within the commit). Returns false (caller falls back to
    /// the CPU tree path) for anything that is not a supported chain.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn gpu_forward_spec_chain(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        start_pos: usize,
        out_logits: &mut [f32],
    ) -> bool {
        if std::env::var("SOFUU_GPU_SPEC")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            return false;
        }
        let head_row = if self.gpu.as_ref().map(|g| g.skip_head).unwrap_or(true) {
            None
        } else {
            Some(0usize)
        };
        if !self.gpu_forward_chain_core(
            tokens,
            positions,
            start_pos,
            head_row,
            true,
            out_logits,
        ) {
            return false;
        }
        if head_row.is_none() {
            // ablation: no head anywhere — fill zeros so the caller contract
            // (root logits present) is still met shape-wise.
            out_logits[..self.config.vocab_size].fill(0.0);
        }
        true
    }

    /// Prefill route: B≤4 chunk through the GPU chain pass (no head unless
    /// the caller needs the last row's logits). Returns true when taken.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn gpu_forward_prefill_chain(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        last_logits: bool,
    ) -> bool {
        // OPT-IN (SOFUU_GPU_PREFILL=1): the b4 chunk pass is ~4x the SERIAL
        // GPU prefill, but at current q4_K kernel rates (~50GB/s class) it
        // still loses to the CPU tile path on most models — it becomes the
        // default together with the higher-bandwidth kernel generation.
        if std::env::var("SOFUU_GPU_PREFILL")
            .map(|v| v != "1")
            .unwrap_or(true)
        {
            return false;
        }
        let b = tokens.len();
        if b < 2 || b > 4 || start_pos + b > self.config.max_seq_len {
            return false;
        }
        let t0 = std::time::Instant::now();
        let positions: Vec<usize> = (0..b).map(|t| start_pos + t).collect();
        if !self.gpu_forward_chain_core(
            tokens,
            &positions,
            start_pos,
            if last_logits { Some(b - 1) } else { None },
            false,
            &mut [],
        ) {
            return false;
        }
        if std::env::var_os("SOFUU_GPU_PROFILE").is_some() {
            eprintln!(
                "[gpu-profile] prefill-chunk: n={b} start_pos={start_pos} wall={}{}",
                t0.elapsed().as_micros() as u64,
                if last_logits { " (last-logits)" } else { "" }
            );
        }
        true
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn gpu_forward_chain_core(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        start_pos: usize,
        head_row: Option<usize>,
        tree_refill: bool,
        out_logits: &mut [f32],
    ) -> bool {
        let g = match self.gpu.as_mut() {
            Some(g) if g.token_commit => g,
            _ => return false,
        };
        let b = tokens.len();
        if b < 2 || b > 4 || positions.len() != b || start_pos + b > self.config.max_seq_len {
            return false;
        }
        for (t, &p) in positions.iter().enumerate() {
            if p != start_pos + t {
                return false;
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
        let rope_mode: u32 = if self.config.rope_interleaved { 1 } else { 0 };
        let q_rows = n_heads * head_dim;
        let kv_rows = n_kv_heads * head_dim;
        let vocab_size = self.config.vocab_size;
        let pass_t = std::time::Instant::now();
        let mut ffn_ops = 0usize;
        let pos_of_last = start_pos + b - 1;
        let fuse_on = std::env::var("SOFUU_GPU_FUSE")
            .map(|v| v != "0")
            .unwrap_or(true);

        // Embedding lookup stays on CPU by design (GPU-first directive:
        // embedding/sampler are CPU work; q-row staging is a host copy).
        for (t, &tok) in tokens.iter().enumerate() {
            let dst = g.slot_f32_mut(g.slots.x + t * dim * 4, dim);
            Self::lookup_embedding(&self.gguf, &self.token_embd, tok, dim, dst);
        }
        // Per-row positions for the b4 attention family (u32s bit-written
        // into the posb slot).
        {
            let pb = g.slot_f32_mut(g.slots.posb, 16);
            let pbb =
                unsafe { std::slice::from_raw_parts_mut(pb.as_ptr() as *mut u32, 16) };
            for (t, item) in pbb.iter_mut().enumerate().take(b) {
                *item = (start_pos + t) as u32;
            }
        }
        // Bound above the macro so macro hygiene can see it (the macro body
        // expands it at def-site scope); a field borrow is disjoint from
        // self.gpu, so holding both across the pass is fine.
        let gguf = &self.gguf;

        // GEMV of one tensor over the whole chain: batch_gguf_b4 when the
        // tensor is q4_K (the b4 kernel's token-major [4][cols]/[4][rows]
        // layout matches the x/xn/qa/gate... slot regions exactly), one
        // batch_gguf per row otherwise. Both paths use identical offsets.
        macro_rules! gemv_b {
            ($tv:expr, $x_off:expr, $y_off:expr, $rows:expr, $cols:expr) => {
                if $tv.tensor_type == GGML_TYPE_Q4_K {
                    g.engine.batch_gguf_b4(
                        &g.w_buf, gguf.data_offset + $tv.offset,
                        &g.act_buf, $x_off,
                        &g.act_buf, $y_off,
                        $rows, $cols,
                    );
                } else {
                    for t in 0..b {
                        g.engine.batch_gguf(
                            $tv.tensor_type, &g.w_buf, gguf.data_offset + $tv.offset,
                            // offsets are BYTES; slot sizes are element counts
                            &g.act_buf, $x_off + t * $cols * 4,
                            &g.act_buf, $y_off + t * $rows * 4,
                            $rows, $cols,
                        );
                    }
                }
            };
        }

        // P2 decode diet: the SAME fused multi-section GEMM kernel as the
        // prefill route — B-generic (mask subtiles), sections land in the
        // pre-existing qa/ka/va (gate/up) slots, per-tensor b4 GEMV fallback.
        // SHIPPED ON (SOFUU_FUSE_MULTI=0 disables): the in-model divergence
        // was the section blob's base — base was taken from the FIRST
        // section's aligned offset, but sections are not monotonic in file
        // order (1B attn_k < attn_q) so the k delta underflowed u32 → the
        // kernel read base+u32::MAX+rem = garbage ~1e-1-off activations.
        // base = MIN aligned offset across the run fixes it: 1B + MiniCPM +
        // 0.6B FULL-TEXT == CPU with the type-grouped mixed dispatch.
        let fuse_multi = std::env::var("SOFUU_FUSE_MULTI")
            .map(|v| v != "0")
            .unwrap_or(true);
        #[inline]
        fn f16_bits_to_f32(h: u16) -> f32 {
            let sign = ((h >> 15) as u32) << 31;
            let exp = ((h >> 10) & 0x1F) as u32;
            let man = (h & 0x3FF) as u32;
            if exp == 0 {
                f32::from_bits(sign | (man << 13))
            } else if exp == 31 {
                f32::from_bits(sign | 0x7F80_0000 | (man << 13))
            } else {
                f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
            }
        }
        let multi_dbg_dec = std::env::var_os("SOFUU_MULTI_DEBUG").is_some();
        let multi_dbg_dec_done = std::cell::Cell::new(false);
        let multi_dbg_dec_dumped = std::cell::Cell::new(false);
        macro_rules! gemm_fused_dec {
            ($secs:expr, $rows_total:expr, $cols:expr, $x_off:expr) => {{
                // P1/P2: type-grouped section runs — consecutive sections of
                // one type ride ONE dispatch. base = MIN aligned weight offset
                // across the run (sections may not be monotonic in file order —
                // the 1B's attn_k sits BELOW attn_q; a first-section base made
                // the k delta underflow u32 = the b4blob OOB receipt).
                let mut all_ok = fuse_multi && !$secs.is_empty();
                let mut fast = false;
                let mut groups: Vec<(u32, Vec<(usize, u32, u32, u32)>)> = Vec::new(); // (type, [(secs idx, rows, y_off, y_stride)])
                for (i, (tv, rows, y_off, y_stride)) in $secs.iter().enumerate().take(4) {
                    let ty = tv.tensor_type;
                    if ty != GGML_TYPE_Q4_K && ty != GGML_TYPE_Q6_K && ty != GGML_TYPE_Q8_0 {
                        all_ok = false;
                        break;
                    }
                    match groups.last_mut() {
                        Some(g) if g.0 == ty => g.1.push((i, *rows, *y_off as u32, *y_stride)),
                        _ => groups.push((ty, vec![(i, *rows, *y_off as u32, *y_stride)])),
                    }
                }
                let mut flushed_fast = true;
                let mut first_dumped = !multi_dbg_dec;
                for (run_type, run) in groups.iter() {
                    let mut blob: [[u32; 5]; 4] = [[0u32; 5]; 4];
                    let mut base = usize::MAX;
                    for (idx, _rr, _yo, _ys) in run {
                        let raw = gguf.data_offset + $secs[*idx].0.offset;
                        base = base.min(raw & !255usize);
                    }
                    let mut end = 0u32;
                    let mut rt = 0u32;
                    for (j, (idx, rr, yo, ys)) in run.iter().enumerate() {
                        let raw = gguf.data_offset + $secs[*idx].0.offset;
                        blob[j][0] = ((raw & !255usize) - base) as u32;
                        blob[j][1] = (raw & 255usize) as u32;
                        end += *rr;
                        blob[j][2] = end;
                        blob[j][3] = *yo;
                        blob[j][4] = *ys;
                        rt = end;
                    }
                    if multi_dbg_dec && !first_dumped {
                        first_dumped = true;
                        let fb8 = gguf.as_bytes();
                        for (j, (idx, rr, yo, ys)) in run.iter().enumerate() {
                            let raw = gguf.data_offset + $secs[*idx].0.offset;
                            let slot = base + blob[j][0] as usize + blob[j][1] as usize;
                            if slot + 2 > fb8.len() || raw + 2 > fb8.len() {
                                eprintln!("[b4blob] sec{j} OOB base={base} off={} rem={} raw={raw} file_len={}",
                                    blob[j][0], blob[j][1], fb8.len());
                                continue;
                            }
                            let bits = u16::from_le_bytes([fb8[raw], fb8[raw + 1]]);
                            let d_file = f16_bits_to_f32(bits);
                            let bits2 = u16::from_le_bytes([fb8[slot], fb8[slot + 1]]);
                            let d_slot = f16_bits_to_f32(bits2);
                            eprintln!(
                                "[b4blob] sec{j} ty={} raw={raw} base={base} off={} rem={} rows_end={} y_off={} stride={} d_file={d_file:.6e} d_slot={d_slot:.6e} match={}",
                                *run_type, blob[j][0], blob[j][1], blob[j][2], blob[j][3], blob[j][4],
                                (bits == bits2) as u8
                            );
                        }
                    }
                    let ok = match *run_type {
                        GGML_TYPE_Q6_K => g.engine.batch_gguf6_b4multi(
                            14, &g.w_buf, base, &g.act_buf, $x_off, &g.act_buf,
                            rt as usize, $cols, &blob,
                        ),
                        GGML_TYPE_Q8_0 => g.engine.batch_gguf8_b4r4(
                            8, &g.w_buf, base, &g.act_buf, $x_off, &g.act_buf,
                            rt as usize, $cols, &blob,
                        ),
                        _ => g.engine.batch_gguf_b4multi(
                            12, &g.w_buf, base, &g.act_buf, $x_off, &g.act_buf,
                            rt as usize, $cols, &blob,
                        ),
                    };
                    flushed_fast &= ok;
                }
                fast = all_ok && flushed_fast;
                fast
            }};
        }

        assert!(g.engine.batch_open());
        for layer_idx in 0..self.config.n_layers {
            let l = &self.layers[layer_idx];
            let attn_norm_off = gguf.data_offset + l.attn_norm.offset;
            let ffn_norm_off = gguf.data_offset + l.ffn_norm.offset;
            let q_norm_off = l.attn_q_norm.as_ref().map(|tv| gguf.data_offset + tv.offset);
            let k_norm_off = l.attn_k_norm.as_ref().map(|tv| gguf.data_offset + tv.offset);
            let has_q = u32::from(q_norm_off.is_some());
            let has_k = u32::from(k_norm_off.is_some());
            let kv_head_bytes = n_kv_heads * max_seq_len * head_dim * 4;
            let kc_layer = layer_idx * kv_head_bytes;
            let meta_chunks = (max_seq_len + 127) / 128;
            let meta_per_layer = n_heads * meta_chunks * 2 + n_heads * meta_chunks * head_dim;
            let meta_layer_off = layer_idx * meta_per_layer * 4;
            let (kln_buf, kln_off) = (k_norm_off.map(|off| &g.w_buf), k_norm_off.unwrap_or(0));

            g.engine.batch_rmsnorm_f32w_b4(
                &g.act_buf, g.slots.x,
                &g.w_buf, attn_norm_off,
                &g.act_buf, g.slots.xn,
                dim, norm_eps, b,
            );
            let qkv_ok = gemm_fused_dec!(
                [
                    (&l.attn_q, q_rows as u32, g.slots.qa, q_rows as u32),
                    (&l.attn_k, kv_rows as u32, g.slots.ka, kv_rows as u32),
                    (&l.attn_v, kv_rows as u32, g.slots.va, kv_rows as u32),
                ],
                q_rows + 2 * kv_rows, dim, g.slots.xn
            );
            if !qkv_ok {
                gemv_b!(l.attn_q, g.slots.xn, g.slots.qa, q_rows, dim);
                gemv_b!(l.attn_k, g.slots.xn, g.slots.ka, kv_rows, dim);
                gemv_b!(l.attn_v, g.slots.xn, g.slots.va, kv_rows, dim);
            }

            // Attention in four b4 dispatches (rows × heads × chunks grid):
            // kv_write_b4 lands every row's KV first; attn_part_b4 bounds
            // row r's softmax to positions[r], so later rows' fresh cache
            // content is excluded — chain causality without per-row
            // dispatches. (SOFUU_GPU_SKIP=attn timing ablation skips it.)
            if !g.skip_attn {
                g.engine.batch_llama_kv_write_b4(
                    &g.act_buf, g.slots.ka,
                    &g.act_buf, g.slots.va,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    kln_buf, kln_off,
                    has_k,
                    &g.act_buf, g.slots.posb,
                    b,
                    head_dim as u32, n_kv_heads as u32, max_seq_len as u32,
                    norm_eps, rope_freq_base, rope_mode,
                );
                g.engine.batch_head_rmsnorm_b4(
                    &g.act_buf, g.slots.qa,
                    q_norm_off.map(|_| &g.w_buf), q_norm_off.unwrap_or(0),
                    &g.act_buf, g.slots.posb,
                    b,
                    head_dim as u32, n_heads as u32,
                    has_q, norm_eps, rope_freq_base, rope_mode,
                );
                // pos < 128 (one softmax chunk): part+fin fuse into one
                // dispatch — one barrier drain instead of two
                let fused = (pos_of_last + 1) <= 128
                    && fuse_on
                    && g.engine.batch_llama_attn_1c_b4(
                        &g.act_buf, g.slots.qa,
                        &g.k_buf, kc_layer,
                        &g.v_buf, kc_layer,
                        &g.act_buf, g.slots.ao,
                        &g.act_buf, g.slots.posb,
                        b,
                        head_dim as u32, n_heads as u32, n_kv_heads as u32,
                        max_seq_len as u32,
                    );
                if !fused {
                    g.engine.batch_llama_attn_part_b4(
                        &g.act_buf, g.slots.qa,
                        &g.k_buf, kc_layer,
                        &g.v_buf, kc_layer,
                        &g.meta_buf, meta_layer_off,
                        &g.act_buf, g.slots.posb,
                        b,
                        head_dim as u32, n_heads as u32, n_kv_heads as u32,
                        max_seq_len as u32,
                        meta_chunks as u32, meta_per_layer as u32,
                    );
                    g.engine.batch_llama_attn_fin_b4(
                        &g.meta_buf, meta_layer_off,
                        &g.act_buf, g.slots.ao,
                        &g.act_buf, g.slots.posb,
                        b,
                        n_heads as u32, head_dim as u32,
                        meta_chunks as u32, meta_per_layer as u32,
                    );
                }
            }

            // o projection + resid-add fused (rows == dim == acc width)
            if !(fuse_on
                && l.attn_output.tensor_type == GGML_TYPE_Q4_K
                && g.engine.batch_gguf_b4r(
                    &g.w_buf, gguf.data_offset + l.attn_output.offset,
                    &g.act_buf, g.slots.ao,
                    &g.act_buf, g.slots.xo,
                    &g.act_buf, g.slots.x,
                    dim, q_rows, dim,
                ))
            {
                gemv_b!(l.attn_output, g.slots.ao, g.slots.xo, dim, q_rows);
                g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, b * dim);
            }
            for t in 0..b {
                g.engine.batch_rmsnorm_f32w(
                    &g.act_buf, g.slots.x + t * dim * 4,
                    &g.w_buf, ffn_norm_off,
                    &g.act_buf, g.slots.xn + t * dim * 4,
                    dim, norm_eps,
                );
            }
            if !g.skip_ffn {
            ffn_ops += 1;
            let gu_ok = gemm_fused_dec!(
                [
                    (&l.ffn_gate, hidden_dim as u32, g.slots.gate, hidden_dim as u32),
                    (&l.ffn_up, hidden_dim as u32, g.slots.up, hidden_dim as u32),
                ],
                2 * hidden_dim, dim, g.slots.xn
            );
            if !gu_ok {
                gemv_b!(l.ffn_gate, g.slots.xn, g.slots.gate, hidden_dim, dim);
                gemv_b!(l.ffn_up, g.slots.xn, g.slots.up, hidden_dim, dim);
            }
            if !g.skip_ffn {
                ffn_ops += 3;
            }
            g.engine.batch_swiglu(&g.act_buf, g.slots.gate, &g.act_buf, g.slots.up, b * hidden_dim);
            // down projection + resid-add fused
            if !(fuse_on
                && l.ffn_down.tensor_type == GGML_TYPE_Q4_K
                && g.engine.batch_gguf_b4r(
                    &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                    &g.act_buf, g.slots.gate,
                    &g.act_buf, g.slots.xo,
                    &g.act_buf, g.slots.x,
                    dim, hidden_dim, dim,
                ))
            {
                gemv_b!(l.ffn_down, g.slots.gate, g.slots.xo, dim, hidden_dim);
                g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, b * dim);
            }
            }
        }

        // Head: output-norm every row (cheap; feeds lazy per-row logits),
        // then ONE lm_head GEMV — row 0 (the root logits the tree contract
        // needs eagerly). The lm_head is ~514MB of q6_K for the 8B vocab:
        // committing it for all B rows streams 2GB per pass for rows verify
        // usually never samples; accepted rows get a lazy GPU head commit
        // (gpu_spec_head_row, ~13ms each) and CPU matvec as the fallback.
        if let Some(hr) = head_row {
            for t in 0..b {
                g.engine.batch_rmsnorm_f32w(
                    &g.act_buf, g.slots.x + t * dim * 4,
                    &g.w_buf, gguf.data_offset + self.output_norm.offset,
                    &g.act_buf, g.slots.xn + t * dim * 4,
                    dim, norm_eps,
                );
            }
            g.engine.batch_gguf(
                self.output.tensor_type,
                &g.w_buf, g.w_offset(gguf, &self.output),
                &g.act_buf, g.slots.xn + hr * dim * 4,
                &g.act_buf, g.slots.logits,
                vocab_size, dim,
            );
        }
        g.engine.batch_end();
        let commit_us = pass_t.elapsed().as_micros() as u64;
        if std::env::var("SOFUU_GPU_TIMING").map(|v| v.trim() == "2").unwrap_or(false) {
            g.engine.print_op_timing("spec-pass");
        }

        if head_row.is_some() {
            out_logits[0..vocab_size].copy_from_slice(g.slot_f32(g.slots.logits, vocab_size));
            self.logits[0..vocab_size].copy_from_slice(&out_logits[0..vocab_size]);
            let xn_host = g.slot_f32(g.slots.xn, b * dim);
            self.x_norm_batch[0..b * dim].copy_from_slice(xn_host);
        }
        let post_us = pass_t.elapsed().as_micros() as u64;
        // The lazy per-row head commits (spec verify) read the xn region;
        // valid only when the output norms above ran.
        self.spec_rows_valid = head_row.is_some();
        if std::env::var_os("SOFUU_GPU_PROFILE").is_some() {
            eprintln!(
                "[gpu-profile] spec-pass: batch={b} wall={}us commit={}us post-head={}us refill={}us",
                pass_t.elapsed().as_micros() as u64,
                commit_us,
                post_us.saturating_sub(commit_us),
                pass_t.elapsed().as_micros() as u64 - post_us
            );
        }

        // Refill the CPU tree-K/V staging from the shared cache so the
        // caller's commit_speculative_candidates stays byte-identical in
        // behavior: the GPU wrote these rows into the cache (idempotent
        // copy); rejected rows get rewritten before their positions are
        // ever attended again. (Not needed on the prefill route.)
        if tree_refill {
            for layer_idx in 0..self.config.n_layers {
                for t in 0..b {
                    let cache_pos = (start_pos + t) % max_seq_len;
                    let base = (layer_idx * self.tree_max_nodes + t) * kv_rows;
                    for kv_h in 0..n_kv_heads {
                        let src = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim
                            + cache_pos * head_dim;
                        self.tree_k_layers[base + kv_h * head_dim..base + (kv_h + 1) * head_dim]
                            .copy_from_slice(&self.k_cache[src..src + head_dim]);
                        self.tree_v_layers[base + kv_h * head_dim..base + (kv_h + 1) * head_dim]
                            .copy_from_slice(&self.v_cache[src..src + head_dim]);
                    }
                }
            }
        }
        true
    }

    fn bytemuck_or_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// P1 (llama.cpp parity): prefill a B<=32-token chunk through the GEMM
    /// class in ONE commit — gemm_q4_k reuses every weight byte across the
    /// chunk's tokens; the glue is the b4 family at rows=B (all its kernels
    /// are rows-generic). Q4_K weights ride batch_gguf_gemm; any other type
    /// (or a missing GEMM PSO) falls back to per-row GEMVs in the same
    /// commit — correct, just not the parity rate. Causality: the b4
    /// attention bounds each row's softmax to positions[r], so a chunk's
    /// later rows never see earlier rows' fresh KV beyond their position.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn gpu_forward_prefill_gemm_core(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> bool {
        let b = tokens.len();
        if !(1..=32).contains(&b)
            || start_pos + b > self.config.max_seq_len
        {
            return false;
        }
        let g = match self.gpu.as_mut() {
            Some(gg) if gg.token_commit && gg.pref_buf.is_some() => gg,
            _ => return false,
        };
        let pslots = match g.pref_slots {
            Some(p) => p,
            None => return false,
        };

        let pref_bytes = g.pref_bytes;
        let pbuf = match g.pref_buf.as_ref() {
            Some(pb) => pb,
            None => return false,
        };
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let rope_freq_base = self.config.rope_freq_base;
        let max_seq_len = self.config.max_seq_len;
        let rope_mode: u32 = if self.config.rope_interleaved { 1 } else { 0 };
        let q_rows = n_heads * head_dim;
        let kv_rows = n_kv_heads * head_dim;
        let vocab_size = self.config.vocab_size;
        let gguf = &self.gguf;
        let pos_of_last = start_pos + b - 1;

        // Embedding lookup + per-row positions on the host slab.
        {
            let ph = g.pref_act.as_mut().unwrap();
            let ph_buf: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(ph.as_ptr() as *mut f32, pref_bytes / 4)
            };
            for (t, &tok) in tokens.iter().enumerate() {
                let dst = &mut ph_buf[pslots.x / 4 + t * dim..pslots.x / 4 + (t + 1) * dim];
                Self::lookup_embedding(&self.gguf, &self.token_embd, tok, dim, dst);
            }
            let pbb = unsafe {
                std::slice::from_raw_parts_mut(
                    ph_buf[pslots.posb / 4..].as_ptr() as *mut u32,
                    b,
                )
            };
            for (t, item) in pbb.iter_mut().enumerate() {
                *item = (start_pos + t) as u32;
            }
        }

        let pass_t = std::time::Instant::now();
        macro_rules! gemm {
            ($tv:expr, $x_off:expr, $y_off:expr, $rows:expr, $cols:expr) => {{
                let fast = $tv.tensor_type == GGML_TYPE_Q4_K
                    && g.engine.batch_gguf_gemm(
                        12, &g.w_buf, gguf.data_offset + $tv.offset,
                        pbuf, $x_off, pbuf, $y_off,
                        $rows, $cols, b,
                    );
                if !fast {
                    for t in 0..b {
                        g.engine.batch_gguf(
                            $tv.tensor_type, &g.w_buf, gguf.data_offset + $tv.offset,
                            pbuf, $x_off + t * $cols * 4,
                            pbuf, $y_off + t * $rows * 4,
                            $rows, $cols,
                        );
                    }
                }
            }};
        }

        let acc_fuse = std::env::var("SOFUU_GEMM_ACC")
            .map(|v| v != "0")
            .unwrap_or(true);
        let multi_dbg = std::env::var_os("SOFUU_MULTI_DEBUG").is_some();
        let multi_dbg_done = std::cell::Cell::new(false);
        // P1 op-diet: fused multi-section GEMM (qkv / gate_up). Section
        // weight offsets are u32 DELTAS from the first section's 256-aligned
        // base (the C side binds w at that base; absolute offsets overflow
        // u32 on the 8B's 5.0GB file). Falls back to per-tensor gemm! on any
        // non-q4_K tensor or a missing fused PSO.
        macro_rules! gemm_fused {
            ($secs:expr, $rows_total:expr, $cols:expr, $x_off:expr) => {{
                let all_q4k = $secs.iter().all(|s| s.0.tensor_type == GGML_TYPE_Q4_K);
                let mut fast = false;
                if multi_dbg && !multi_dbg_done.replace(true) {
                    eprintln!(
                        "[multi-debug] all_q4k={} types={:?} rows_total={} cols={} b={} x_off={} y_offs={:?}",
                        all_q4k,
                        $secs.iter().map(|s| s.0.tensor_type).collect::<Vec<_>>(),
                        $rows_total, $cols, b,
                        $x_off,
                        $secs.iter().map(|s| s.3).collect::<Vec<_>>()
                    );
                }
                if all_q4k {
                    let mut blob: [[u32; 5]; 4] = [[0u32; 5]; 4];
                    let mut end = 0u32;
                    // base = MIN aligned offset across the sections — the
                    // decode fusion's b4blob receipt proved sections are not
                    // monotonic in file order (1B attn_k < attn_q); a first-
                    // section base underflowed the u32 delta to 2^32.
                    let base = (0..$secs.len().min(4))
                        .map(|i| (gguf.data_offset + $secs[i].0.offset) & !255usize)
                        .min()
                        .unwrap_or(0);
                    for (i, (tv, rows, y_off, y_stride)) in $secs.iter().enumerate().take(4) {
                        let raw = gguf.data_offset + tv.offset;
                        blob[i][0] = ((raw & !255usize) - base) as u32;
                        blob[i][1] = (raw & 255usize) as u32;
                        end += *rows as u32;
                        blob[i][2] = end;
                        blob[i][3] = *y_off as u32;
                        blob[i][4] = *y_stride as u32;
                    }
                    if multi_dbg && !multi_dbg_done.replace(true) {
                        eprintln!(
                            "[multi-debug] base={base} secs={:?} rows_total={} cols={} b={}",
                            &blob[..], $rows_total, $cols, b
                        );
                    }
                    fast = g.engine.batch_gguf_gemm_multi(
                        12, &g.w_buf, base, pbuf, $x_off, pbuf, 0,
                        $rows_total, $cols, b, &blob,
                    );
                }
                fast
            }};
        }

        assert!(g.engine.batch_open());
        for layer_idx in 0..self.config.n_layers {
            let l = &self.layers[layer_idx];
            let attn_norm_off = gguf.data_offset + l.attn_norm.offset;
            let ffn_norm_off = gguf.data_offset + l.ffn_norm.offset;
            let q_norm_off = l.attn_q_norm.as_ref().map(|tv| gguf.data_offset + tv.offset);
            let k_norm_off = l.attn_k_norm.as_ref().map(|tv| gguf.data_offset + tv.offset);
            let has_q = u32::from(q_norm_off.is_some());
            let has_k = u32::from(k_norm_off.is_some());
            let kv_head_bytes = n_kv_heads * max_seq_len * head_dim * 4;
            let kc_layer = layer_idx * kv_head_bytes;
            let meta_chunks = (max_seq_len + 127) / 128;
            let meta_per_layer = n_heads * meta_chunks * 2 + n_heads * meta_chunks * head_dim;
            let meta_layer_off = layer_idx * meta_per_layer * 4;
            let (kln_buf, kln_off) = (k_norm_off.map(|off| &g.w_buf), k_norm_off.unwrap_or(0));

            g.engine.batch_rmsnorm_f32w_b4(
                pbuf, pslots.x,
                &g.w_buf, attn_norm_off,
                pbuf, pslots.xn,
                dim, norm_eps, b,
            );
            let qkv_fused = gemm_fused!(
                [
                    (&l.attn_q, q_rows as u32, pslots.qa, q_rows as u32),
                    (&l.attn_k, kv_rows as u32, pslots.ka, kv_rows as u32),
                    (&l.attn_v, kv_rows as u32, pslots.va, kv_rows as u32),
                ],
                q_rows + 2 * kv_rows, dim, pslots.xn
            ) && std::env::var("SOFUU_FUSE_QKV").map(|v| v == "1").unwrap_or(false);
            if !qkv_fused {
                gemm!(l.attn_q, pslots.xn, pslots.qa, q_rows, dim);
                gemm!(l.attn_k, pslots.xn, pslots.ka, kv_rows, dim);
                gemm!(l.attn_v, pslots.xn, pslots.va, kv_rows, dim);
            }

            if !g.skip_attn {
                g.engine.batch_llama_kv_write_b4(
                    pbuf, pslots.ka,
                    pbuf, pslots.va,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    kln_buf, kln_off,
                    has_k,
                    pbuf, pslots.posb,
                    b,
                    head_dim as u32, n_kv_heads as u32, max_seq_len as u32,
                    norm_eps, rope_freq_base, rope_mode,
                );
                g.engine.batch_head_rmsnorm_b4(
                    pbuf, pslots.qa,
                    q_norm_off.map(|_| &g.w_buf), q_norm_off.unwrap_or(0),
                    pbuf, pslots.posb,
                    b,
                    head_dim as u32, n_heads as u32,
                    has_q, norm_eps, rope_freq_base, rope_mode,
                );
                let fused = (pos_of_last + 1) <= 128
                    && g.engine.batch_llama_attn_1c_b4(
                        pbuf, pslots.qa,
                        &g.k_buf, kc_layer,
                        &g.v_buf, kc_layer,
                        pbuf, pslots.ao,
                        pbuf, pslots.posb,
                        b,
                        head_dim as u32, n_heads as u32, n_kv_heads as u32,
                        max_seq_len as u32,
                    );
                if !fused {
                    g.engine.batch_llama_attn_part_b4(
                        pbuf, pslots.qa,
                        &g.k_buf, kc_layer,
                        &g.v_buf, kc_layer,
                        &g.meta_buf, meta_layer_off,
                        pbuf, pslots.posb,
                        b,
                        head_dim as u32, n_heads as u32, n_kv_heads as u32,
                        max_seq_len as u32,
                        meta_chunks as u32, meta_per_layer as u32,
                    );
                    g.engine.batch_llama_attn_fin_b4(
                        &g.meta_buf, meta_layer_off,
                        pbuf, pslots.ao,
                        pbuf, pslots.posb,
                        b,
                        n_heads as u32, head_dim as u32,
                        meta_chunks as u32, meta_per_layer as u32,
                    );
                }
            }
            let o_ok = !g.skip_attn
                && acc_fuse
                && l.attn_output.tensor_type == GGML_TYPE_Q4_K
                && g.engine.batch_gguf_gemm_acc(
                    12, &g.w_buf, gguf.data_offset + l.attn_output.offset,
                    pbuf, pslots.ao,
                    pbuf, pslots.x, dim,
                    pbuf, pslots.x,
                    dim, q_rows, b,
                );
            if !o_ok {
                gemm!(l.attn_output, pslots.ao, pslots.xo, dim, q_rows);
                g.engine.batch_add(pbuf, pslots.x, pbuf, pslots.xo, b * dim);
            }

            g.engine.batch_rmsnorm_f32w_b4(
                pbuf, pslots.x,
                &g.w_buf, ffn_norm_off,
                pbuf, pslots.xn,
                dim, norm_eps, b,
            );
            if !g.skip_ffn {
                let ffn_fused = gemm_fused!(
                    [
                        (&l.ffn_gate, hidden_dim as u32, pslots.gate, hidden_dim as u32),
                        (&l.ffn_up, hidden_dim as u32, pslots.up, hidden_dim as u32),
                    ],
                    2 * hidden_dim, dim, pslots.xn
                ) && std::env::var("SOFUU_FUSE_GUP").map(|v| v != "0").unwrap_or(true);
                if !ffn_fused {
                    gemm!(l.ffn_gate, pslots.xn, pslots.gate, hidden_dim, dim);
                    gemm!(l.ffn_up, pslots.xn, pslots.up, hidden_dim, dim);
                }
                g.engine.batch_swiglu(pbuf, pslots.gate, pbuf, pslots.up, b * hidden_dim);
                let d_ok = acc_fuse
                    && l.ffn_down.tensor_type == GGML_TYPE_Q4_K
                    && g.engine.batch_gguf_gemm_acc(
                        12, &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                        pbuf, pslots.gate,
                        pbuf, pslots.x, dim,
                        pbuf, pslots.x,
                        dim, hidden_dim, b,
                    );
                if !d_ok {
                    gemm!(l.ffn_down, pslots.gate, pslots.xo, dim, hidden_dim);
                    g.engine.batch_add(pbuf, pslots.x, pbuf, pslots.xo, b * dim);
                }
            }
        }

        // Last-row logits inside the same commit (output-norm + head GEMV).
        if want_logits {
            // the last loop op released xo — it is the head's norm scratch
            g.engine.batch_rmsnorm_f32w(
                pbuf, pslots.x + (b - 1) * dim * 4,
                &g.w_buf, gguf.data_offset + self.output_norm.offset,
                pbuf, pslots.xo,
                dim, norm_eps,
            );
            g.engine.batch_gguf(
                self.output.tensor_type,
                &g.w_buf, g.w_offset(gguf, &self.output),
                pbuf, pslots.xo,
                pbuf, pslots.logits,
                vocab_size, dim,
            );
        }
        g.engine.batch_end();
        if std::env::var("SOFUU_GPU_TIMING").map(|v| v.trim() == "2").unwrap_or(false) {
            g.engine.print_op_timing("prefill-gemm");
        }
        if std::env::var_os("SOFUU_PREFILL_DEBUG_DUMP").is_some() {
            let ph = g.pref_act.as_ref().unwrap();
            let idx: Vec<f32> = unsafe {
                std::slice::from_raw_parts(ph.as_ptr() as *const f32, pref_bytes / 4)
            }
            .to_vec();
            let _ = std::fs::write(
                format!(
                    "/tmp/prefdump_{}_{}x{}.bin",
                    std::process::id(),
                    start_pos,
                    b
                ),
                Self::bytemuck_or_bytes(&idx),
            );
        }

        if want_logits {
            let lg: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    (g.pref_act.as_ref().unwrap().as_ptr() as *const u8)
                        .add(pslots.logits) as *const f32,
                    vocab_size,
                )
            };
            self.logits[0..vocab_size].copy_from_slice(lg);
        }
        if std::env::var_os("SOFUU_GPU_PROFILE").is_some() {
            eprintln!(
                "[gpu-profile] prefill-gemm: n={b} start_pos={start_pos} wall={}us logits={}",
                pass_t.elapsed().as_micros() as u64,
                want_logits
            );
        }
        true
    }

    /// P1: full prefill through the GEMM route in ~32-token chunks. The KV
    /// lands in the shared caches positions-contiguous; the CPU's next decode
    /// step continues exactly where the chunks stopped. Returns false when
    /// anything unsupported appeared (caller falls back to b4/CPU tiles).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn gpu_forward_prefill_gemm_chunks(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> bool {
        if self.gpu.as_ref().map(|g| g.pref_slots.is_none()).unwrap_or(true) {
            return false;
        }
        let vocab = self.config.vocab_size;
        // SPILL SAFETY established by receipts: b=32 rows span meta slots
        // [L, L+31) inside the layers*4-slot buffer at ANY prompt length
        // (max 58 of 112 slots on the 0.6B) — the original all-pos 5-chunk
        // receipts were bit-exact through this spill. b=128 would overflow
        // (L+127 > 112) → that needs the meta resize; chunks stay <=32 until
        // then (SOFUU_PREFILL_CHUNK caps at 32 for now).
        let pre_chunk = std::env::var("SOFUU_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| (1..=32).contains(v))
            .unwrap_or(32);
        let mut pos = start_pos;
        let mut off = 0usize;
        while off < tokens.len() {
            let take = (tokens.len() - off).min(pre_chunk);
            let last = compute_last_logits && off + take == tokens.len();
            if !self.gpu_forward_prefill_gemm_core(&tokens[off..off + take], pos, last) {
                return false;
            }
            pos += take;
            off += take;
        }
        let _ = vocab;
        true
    }

    /// One decoder layer: CPU glue + three Metal commits. All GPU ops read
    /// and write slab slots; the CPU copies f32 values in and out around the
    /// commits (the same between-submissions coherence the G0 rig proves).
    /// Kept as the fallback for shapes the one-commit kernels don't cover
    /// (sliding windows, biases, past-4096 wraparound).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn gpu_layer_step(&mut self, layer_idx: usize, pos: usize, sliding_window: usize) {
        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        let norm_eps = self.config.norm_eps;
        let rope_freq_base = self.config.rope_freq_base;
        let max_seq_len = self.config.max_seq_len;
        let gguf = &self.gguf;
        let g = match self.gpu.as_mut() {
            Some(g) => g,
            None => return,
        };

        let layer = &self.layers[layer_idx];
        let q_rows = n_heads * head_dim;
        let kv_rows = n_kv_heads * head_dim;

        // ── Attention norm (CPU, f32 weights) → stage x_norm ──
        let attn_norm_bytes = Self::tensor_slice(gguf, &layer.attn_norm);
        let attn_norm_w = unsafe {
            std::slice::from_raw_parts(attn_norm_bytes.as_ptr() as *const f32, dim)
        };
        rmsnorm(&mut self.x_norm, &self.x, attn_norm_w, norm_eps);
        g.slot_f32_mut(g.slots.xn, dim).copy_from_slice(&self.x_norm);

        // ── Commit A: q, k, v projections ──────────────────
        assert!(g.engine.batch_open());
        let tensors = [
            (&layer.attn_q, g.slots.qa, q_rows),
            (&layer.attn_k, g.slots.ka, kv_rows),
            (&layer.attn_v, g.slots.va, kv_rows),
        ];
        for (tv, y_slot, rows) in tensors {
            g.engine.batch_gguf(
                tv.tensor_type,
                &g.w_buf, g.w_offset(gguf, tv),
                &g.act_buf, g.slots.xn,
                &g.act_buf, y_slot,
                rows, dim,
            );
        }
        g.engine.batch_end();

        self.q.copy_from_slice(g.slot_f32(g.slots.qa, q_rows));
        self.k.copy_from_slice(g.slot_f32(g.slots.ka, kv_rows));
        self.v.copy_from_slice(g.slot_f32(g.slots.va, kv_rows));

        // ── CPU glue: biases, QK-norms, RoPE, KV write, GQA attention ──
        if let Some(ref bias) = layer.attn_q_bias {
            let b = unsafe {
                std::slice::from_raw_parts(Self::tensor_slice(gguf, bias).as_ptr() as *const f32, q_rows)
            };
            for i in 0..q_rows { self.q[i] += b[i]; }
        }
        if let Some(ref bias) = layer.attn_k_bias {
            let b = unsafe {
                std::slice::from_raw_parts(Self::tensor_slice(gguf, bias).as_ptr() as *const f32, kv_rows)
            };
            for i in 0..kv_rows { self.k[i] += b[i]; }
        }
        if let Some(ref bias) = layer.attn_v_bias {
            let b = unsafe {
                std::slice::from_raw_parts(Self::tensor_slice(gguf, bias).as_ptr() as *const f32, kv_rows)
            };
            for i in 0..kv_rows { self.v[i] += b[i]; }
        }
        if let Some(ref q_norm) = layer.attn_q_norm {
            let w = unsafe {
                std::slice::from_raw_parts(Self::tensor_slice(gguf, q_norm).as_ptr() as *const f32, head_dim)
            };
            super::quant::batched_rmsnorm_in_place(&mut self.q, w, n_heads, head_dim, norm_eps);
        }
        if let Some(ref k_norm) = layer.attn_k_norm {
            let w = unsafe {
                std::slice::from_raw_parts(Self::tensor_slice(gguf, k_norm).as_ptr() as *const f32, head_dim)
            };
            super::quant::batched_rmsnorm_in_place(&mut self.k, w, n_kv_heads, head_dim, norm_eps);
        }
        let rope_mode = if self.config.rope_interleaved { 1 } else { 0 };
        super::quant::rope_style(&mut self.q, n_heads, head_dim, pos, rope_freq_base, rope_mode);
        super::quant::rope_style(&mut self.k, n_kv_heads, head_dim, pos, rope_freq_base, rope_mode);

        let cache_pos = pos % max_seq_len;
        for kv_h in 0..n_kv_heads {
            let k_in = &self.k[kv_h * head_dim..(kv_h + 1) * head_dim];
            let v_in = &self.v[kv_h * head_dim..(kv_h + 1) * head_dim];
            let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim + cache_pos * head_dim;
            self.k_cache[slot..slot + head_dim].copy_from_slice(k_in);
            self.v_cache[slot..slot + head_dim].copy_from_slice(v_in);
            kv16_store(self.k16_cache.as_mut(), slot, k_in);
            kv16_store(self.v16_cache.as_mut(), slot, v_in);
        }
        if self.kv_f16 {
            super::quant::batched_attention_ext_k16(
                &self.q,
                self.k16_cache.as_ref().unwrap().as_u16(),
                self.v16_cache.as_ref().unwrap().as_u16(),
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
        } else {
            super::quant::batched_attention_ext(
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
        }

        // ── Commit B: attention output projection ──────────
        g.slot_f32_mut(g.slots.ao, q_rows).copy_from_slice(&self.head_out);
        assert!(g.engine.batch_open());
        g.engine.batch_gguf(
            layer.attn_output.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.attn_output),
            &g.act_buf, g.slots.ao,
            &g.act_buf, g.slots.xo,
            dim, q_rows,
        );
        g.engine.batch_end();
        self.attn_out.copy_from_slice(g.slot_f32(g.slots.xo, dim));
        inline_mad(&mut self.x, &self.attn_out, 1.0);

        // ── FFN norm (CPU) → stage ─────────────────────────
        let ffn_norm_bytes = Self::tensor_slice(gguf, &layer.ffn_norm);
        let ffn_norm_w = unsafe {
            std::slice::from_raw_parts(ffn_norm_bytes.as_ptr() as *const f32, dim)
        };
        rmsnorm(&mut self.x_norm, &self.x, ffn_norm_w, norm_eps);
        g.slot_f32_mut(g.slots.xn, dim).copy_from_slice(&self.x_norm);

        // ── Commit C: gate, up, swiglu (GPU), down ─────────
        assert!(g.engine.batch_open());
        g.engine.batch_gguf(
            layer.ffn_gate.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.ffn_gate),
            &g.act_buf, g.slots.xn,
            &g.act_buf, g.slots.gate,
            hidden_dim, dim,
        );
        g.engine.batch_gguf(
            layer.ffn_up.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.ffn_up),
            &g.act_buf, g.slots.xn,
            &g.act_buf, g.slots.up,
            hidden_dim, dim,
        );
        g.engine
            .batch_swiglu(&g.act_buf, g.slots.gate, &g.act_buf, g.slots.up, hidden_dim);
        g.engine.batch_gguf(
            layer.ffn_down.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.ffn_down),
            &g.act_buf, g.slots.gate,
            &g.act_buf, g.slots.xo,
            dim, hidden_dim,
        );
        g.engine.batch_end();
        self.ffn_out.copy_from_slice(g.slot_f32(g.slots.xo, dim));
        inline_mad(&mut self.x, &self.ffn_out, 1.0);
    }
    fn forward_impl(&mut self, token_id: u32, pos: usize, compute_logits: bool) {
        let dim = self.config.dim;

        // 1. Embedding lookup
        Self::lookup_embedding(&self.gguf, &self.token_embd, token_id, dim, &mut self.x);

        // 2. Transformer layers
        self.execute_layers(pos, 0, self.config.n_layers);

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let token_head = self
            .gpu
            .as_ref()
            .map(|g| g.token_commit)
            .unwrap_or(false);
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let token_head = false;

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if token_head && compute_logits {
            // Head ran in-batch inside layers_one_commit; logits already home.
            let trace = std::env::var_os("SOFUU_GPU_TRACE").is_some();
            if trace {
                let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
                let mut ss = 0f32;
                for (i, v) in self.logits.iter().enumerate() {
                    ss += v * v;
                    if *v > best {
                        best = *v;
                        bi = i;
                    }
                }
                eprintln!(
                    "[trace] tok pos{pos} logits_rms={:.4} argmax={bi}({best:.3})",
                    (ss / self.logits.len() as f32).sqrt()
                );
            }
            return;
        }

        if compute_logits {
            let norm_eps = self.config.norm_eps;
            // 3. Final Output Norm
            let out_norm_bytes = Self::tensor_slice(&self.gguf, &self.output_norm);
            let out_norm_weights = unsafe {
                std::slice::from_raw_parts(out_norm_bytes.as_ptr() as *const f32, dim)
            };
            rmsnorm_in_place(&mut self.x, out_norm_weights, norm_eps);

            // GPU2: lm_head GEMV as one Metal batch when eligible.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if self.ggpu_head_active() {
                self.ggpu_head();
                return;
            }

            // 4. Output Logits Projection
            let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
            if self.streaming_weights {
                prefetch_streaming_slice(out_bytes);
                prefetch_temporal_slice(&self.x);
            }
            matvec(
                out_bytes,
                self.output.tensor_type,
                self.config.vocab_size,
                dim,
                &self.x,
                &mut self.logits,
            );
        }

        // The `gpu` field only exists on Apple targets (cfg above); on other
        // targets the CPU path always owns the head, so the trace helper
        // prints unconditionally there and consults the field on Apple.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let gpu_took_head = self.gpu.is_some();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let gpu_took_head = false;
        if compute_logits && std::env::var_os("SOFUU_GPU_TRACE").is_some() && !gpu_took_head {
            let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
            for (j, v) in self.logits.iter().enumerate() {
                if *v > best {
                    best = *v;
                    bi = j;
                }
            }
            eprintln!("[trace] tok pos{pos} argmax={bi}({best:.3})");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic 1-layer llama GGUF over a temp mmap so the loader's
    /// type gate and tied-output handling can be tested without a real model.
    fn synth_gguf(output_type: Option<u32>) -> GgufFile {
        use std::collections::HashMap;
        let path = std::env::temp_dir().join(format!(
            "sofuu_synth_gguf_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::write(&path, vec![0u8; 1 << 20]).unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("general.architecture".into(), GgufValue::String("llama".into()));
        metadata.insert("llama.embedding_length".into(), GgufValue::Uint32(64));
        metadata.insert("llama.block_count".into(), GgufValue::Uint32(1));
        metadata.insert("llama.attention.head_count".into(), GgufValue::Uint32(2));
        metadata.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(2));
        metadata.insert("llama.feed_forward_length".into(), GgufValue::Uint32(96));
        metadata.insert("llama.vocab_size".into(), GgufValue::Uint32(256));
        let mut tensors = HashMap::new();
        let mut add = |name: &str, shape: [usize; 2], ty: u32, off: &mut usize| {
            let vals: usize = shape[0] * shape[1];
            let bytes = crate::infer::quant::type_row_size(ty, vals);
            tensors.insert(
                name.to_string(),
                TensorInfo {
                    name: name.to_string(),
                    shape: shape.to_vec(),
                    tensor_type: ty,
                    offset: *off,
                    size_bytes: bytes,
                },
            );
            *off += bytes;
        };
        let mut off = 0usize;
        add("token_embd.weight", [64, 256], GGML_TYPE_Q8_0, &mut off);
        add("blk.0.attn_norm.weight", [64, 1], GGML_TYPE_F32, &mut off);
        add("blk.0.attn_q.weight", [64, 64], GGML_TYPE_Q4_K, &mut off);
        add("blk.0.attn_k.weight", [64, 64], GGML_TYPE_Q4_K, &mut off);
        add("blk.0.attn_v.weight", [64, 64], GGML_TYPE_Q4_K, &mut off);
        add("blk.0.attn_output.weight", [64, 64], GGML_TYPE_Q4_K, &mut off);
        add("blk.0.ffn_norm.weight", [64, 1], GGML_TYPE_F32, &mut off);
        add("blk.0.ffn_gate.weight", [96, 64], GGML_TYPE_Q4_K, &mut off);
        add("blk.0.ffn_up.weight", [96, 64], GGML_TYPE_Q4_K, &mut off);
        add("blk.0.ffn_down.weight", [64, 96], GGML_TYPE_Q4_K, &mut off);
        if let Some(ty) = output_type {
            add("output.weight", [64, 256], ty, &mut off);
        }
        add("output_norm.weight", [64, 1], GGML_TYPE_F32, &mut off);
        GgufFile {
            mmap: std::sync::Arc::new(crate::infer::mmap::MmapFile::open(&path).unwrap()),
            resident: None,
            version: 3,
            metadata,
            tensors,
            data_offset: 0,
        }
    }

    #[test]
    fn unsupported_tensor_type_fails_load_loudly() {
        // token_embd in an unsupported quant (Q3_K family, type 11): the load
        // must reject the file with an actionable error, never reach the
        // mid-forward tensor_slice panic.
        let mut gguf = synth_gguf(Some(GGML_TYPE_Q8_0));
        let bad = gguf.tensors.get("token_embd.weight").unwrap().clone();
        gguf.tensors.insert(
            "token_embd.weight".into(),
            TensorInfo { tensor_type: 11, ..bad },
        );
        let err = match TransformerModel::from_gguf(gguf, 512, 256) {
            Err(e) => e,
            Ok(_) => panic!("unsupported type must fail the load"),
        };
        assert!(err.contains("not supported"), "unexpected error: {err}");
        assert!(err.contains("token_embd.weight"), "error must name the tensor: {err}");
    }

    #[test]
    fn present_but_unsupported_output_weight_is_not_treated_as_tied() {
        // output.weight EXISTS in an unsupported type: the load must fail —
        // silently tying embeddings would produce wrong logits.
        let gguf = synth_gguf(Some(11));
        let err = match TransformerModel::from_gguf(gguf, 512, 256) {
            Err(e) => e,
            Ok(_) => panic!("unsupported output.weight must fail the load"),
        };
        assert!(err.contains("output.weight"), "unexpected error: {err}");
    }

    #[test]
    fn absent_output_weight_ties_embeddings() {
        // No output.weight at all: tied embeddings, load must succeed.
        let gguf = synth_gguf(None);
        let m = TransformerModel::from_gguf(gguf, 512, 256).expect("tied load must succeed");
        assert_eq!(m.output.tensor_type, GGML_TYPE_Q8_0);
        assert_eq!(m.output.size_bytes, m.token_embd.size_bytes);
    }

    // Live-model checks: 1.3 GB mmaps thrash the parallel test suite (the
    // qwen35 live tests share the constraint); run explicitly:
    //   cargo test -p sofuu-core --lib live_llama1b -- --ignored
    #[test]
    #[ignore]
    fn live_llama1b_prefill_logits_repeat_in_process() {
        // Env-gated live-model check: repeated prefills INSIDE one process must
        // produce bit-identical last-position logits. Divergence here means
        // per-call state (not process layout/ASLR) breaks determinism.
        let path = std::env::var_os("SOFUU_DET_GGUF")
            .unwrap_or_else(|| "/Users/priyanshuboruah/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf".into());
        if !std::path::Path::new(&path).exists() {
            println!("model not present; skipping");
            return;
        }
        let gguf = GgufFile::open(&path).expect("open gguf");
        let vocab = gguf
            .metadata
            .get("llama.vocab_size")
            .and_then(|v| match v {
                GgufValue::Uint32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(128256);
        let mut m = TransformerModel::from_gguf(gguf, 4096, vocab).expect("model load");
        // Fixed prompt token ids ("Count: one two three four five six seven." + BOS).
        let toks: Vec<u32> = vec![128000, 2568, 25, 832, 1403, 2380, 3116, 4330, 4848, 8254, 13];
        let mut runs: Vec<Vec<f32>> = Vec::new();
        for r in 0..3 {
            m.reset_cache();
            let logits = m.forward_prefill_all_with_logits(&toks, 0);
            runs.push(logits[..8].to_vec());
        }
        println!("[det] run0[0..8] = {:?}", runs[0]);
        for r in 1..runs.len() {
            assert_eq!(
                runs[0], runs[r],
                "in-process repeat {} diverged: {:?} vs {:?}",
                r, runs[0], runs[r]
            );
        }
    }
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


// ── GPU2: GGUF decode client (PLAN-GPU-METAL all-arch coverage) ────────────
// Activations live in a page-aligned zero-copy slab; the CPU runs norms,
// QK-norms, RoPE, the KV cache and attention (f32, shared with the CPU
// path) and stages inputs into slab slots; the GPU batch runs the seven
// per-layer GEMVs as three commits: [q,k,v], [attn_output], and
// [gate, up, swiglu GPU, down]. Eligible dtypes: q4_0/q8_0/q4_K/q6_K.
// Dispatch-overhead repair, not a decode flip: the M2 Pro streaming wall
// (~48-50 GB/s for BOTH engines) still governs — see PLAN-GPU-METAL.md.

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn gguf_gpu_slot(cursor: &mut usize, n_f32: usize) -> usize {
    let off = (*cursor + 255) & !255;
    *cursor = off + ((n_f32 * 4 + 255) & !255);
    off
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn gguf_gpu_page(bytes: usize) -> usize {
    ((bytes + 16383) / 16384) * 16384
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
static PROFILE_HOST_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(any(target_os = "macos", target_os = "ios"))]
static PROFILE_TOTAL_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
use std::sync::atomic::Ordering;

#[cfg(any(target_os = "macos", target_os = "ios"))]
struct GgufGpuSlots {
    x: usize,
    posb: usize,
    xn: usize,
    qa: usize,
    ka: usize,
    va: usize,
    ao: usize,
    xo: usize,
    gate: usize,
    up: usize,
    logits: usize,
}

/// P1 (llama.cpp parity): 32-row GEMM prefill slab layout (byte offsets
/// into pref_buf, same gguf_gpu_slot cursor rule as GgufGpuSlots).
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Clone, Copy)]
struct PrefillSlots {
    x: usize,
    posb: usize,
    xn: usize,
    qa: usize,
    ka: usize,
    va: usize,
    ao: usize,
    xo: usize,
    gate: usize,
    up: usize,
    logits: usize,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub struct GgufGpu {
    engine: crate::infer::metal::MetalEngine,
    /// Zero-copy wrap of the whole GGUF file mmap: every weight bind is
    /// gguf.data_offset + tv.offset inside it.
    w_buf: crate::infer::metal::MetalBuffer,
    /// Buffers BEFORE pages (fields drop in declaration order): the
    /// MTLBuffers are released before the mmaps they wrap.
    act_buf: crate::infer::metal::MetalBuffer,
    act: crate::infer::model::AnonBuffer,
    /// The MODEL's own KV caches registered zero-copy: the GPU kernels and
    /// the CPU fallback read and write the same memory, so no per-layer sync
    /// copy exists and a fallback forward is coherent by construction.
    /// Layout per layer slice [kv][seq][head_dim], matching the CPU caches.
    k_buf: crate::infer::metal::MetalBuffer,
    v_buf: crate::infer::metal::MetalBuffer,
    /// GPU2.1: whole layer in ONE command buffer (norms/QK-norm/RoPE/KV/
    /// attention all as batch ops). False = the 3-commit G2 client.
    one_commit: bool,
    /// GPU3: the WHOLE TOKEN (36 layers + final norm + lm_head) in one
    /// command buffer; the CPU waits once per token. SOFUU_GPU_LAYER_COMMIT=1
    /// restores per-layer waits for A/B receipts.
    token_commit: bool,
    /// Timing-only ablations (results are garbage text; never shipped on).
    /// GPU3 attention meta buffer (shared across layers, barrier-ordered):
    /// per head-chunk {max,sum} pair + per head-chunk partial V sums.
    meta_buf: crate::infer::metal::MetalBuffer,
    meta: crate::infer::model::AnonBuffer,
    skip_attn: bool,
    skip_ffn: bool,
    skip_head: bool,
    slots: GgufGpuSlots,
    /// P1 (llama.cpp parity): 32-row GEMM prefill slab. All None unless
    /// SOFUU_GPU_PREFILL_GEMM=1 at engine construction (memory is not paid
    /// by the default fleet path).
    pref_buf: Option<crate::infer::metal::MetalBuffer>,
    pref_act: Option<crate::infer::model::AnonBuffer>,
    pref_slots: Option<PrefillSlots>,
    pref_bytes: usize,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl GgufGpu {
    /// The spec-verify chain pass needs the whole-token one-commit client.
    /// SOFUU_GPU_SKIP ablations still run the route (sections are skipped
    /// for timing work); SOFUU_GPU_SPEC=0 routes the verify pass to CPU.
    fn spec_chain_ready(&self) -> bool {
        self.token_commit
            && std::env::var("SOFUU_GPU_SPEC")
                .map(|v| v != "0")
                .unwrap_or(true)
    }

    fn arm(
        gguf: &GgufFile,
        config: &ModelConfig,
        layers: &[LayerWeights],
        _vocab: usize,
        k_cache: &AnonBuffer,
        v_cache: &AnonBuffer,
        is_gemma2: bool,
        output: &TensorView,
        output_norm: &TensorView,
    ) -> Option<Self> {
        if config.max_seq_len > 4096 {
            eprintln!("[gguf-gpu] disabled: seq > 4096 above the kernel range");
            return None;
        }
        if gguf.resident.is_some() {
            eprintln!("[gguf-gpu] disabled: resident-weights path is the 27B CPU optimum");
            return None;
        }
        // Every GEMV tensor of every layer must be an eligible dequant dtype.
        let eligible = |ty: u32| matches!(ty, GGML_TYPE_Q4_0 | GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K);
        for (i, l) in layers.iter().enumerate() {
            let ok = [&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output, &l.ffn_gate, &l.ffn_up, &l.ffn_down]
                .iter()
                .all(|tv| eligible(tv.tensor_type));
            if !ok {
                eprintln!("[gguf-gpu] disabled: layer {i} has an ineligible GEMV dtype");
                return None;
            }
        }
        let engine = crate::infer::metal::MetalEngine::new()?;
        let w_buf = engine.register_zerocopy_only(gguf.mmap.as_slice())?;

        // ── One-commit gates (GPU2.1): any miss keeps the 3-commit client ──
        let one_commit_reason = Self::one_commit_blocker(gguf, config, layers, is_gemma2);
        let one_commit = one_commit_reason.is_none();
        if let Some(ref why) = one_commit_reason {
            eprintln!("[gguf-gpu] one-commit disabled ({why}); using the 3-commit client");
        }

        // Whole-token gate: on top of one_commit, the final norm must be f32
        // 4-byte aligned and lm_head an eligible GEMV dtype (it runs in-batch).
        let head_ok = output_norm.tensor_type == GGML_TYPE_F32
            && (gguf.data_offset + output_norm.offset) % 4 == 0
            && matches!(
                output.tensor_type,
                GGML_TYPE_Q4_0 | GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K
            );
        let token_commit = one_commit
            && head_ok
            && std::env::var_os("SOFUU_GPU_LAYER_COMMIT").is_none();

        let dim = config.dim;
        let q_rows = config.n_heads * config.head_dim;
        let kv_rows = config.n_kv_heads * config.head_dim;
        let inter = config.hidden_dim;
        let vocab = config.vocab_size;
        let kv_region_bytes =
            layers.len() * config.n_kv_heads * config.max_seq_len * config.head_dim * 4;
        let mut cursor = 0usize;
        // B=4 regions: the spec-verify chain pass (forward_speculative_tree
        // GPU route) feeds q/k/v/o/ffn GEMVs through batch_gguf_b4 with
        // token-major [4][cols] x and [4][rows] y. The single-token client
        // uses row 0 of each region. ~3MB extra slab on 8B-sized models.
        const SPEC_B: usize = 4;
        let slots = GgufGpuSlots {
            x: gguf_gpu_slot(&mut cursor, SPEC_B * dim),
            posb: gguf_gpu_slot(&mut cursor, SPEC_B * 16),
            xn: gguf_gpu_slot(&mut cursor, SPEC_B * dim),
            qa: gguf_gpu_slot(&mut cursor, SPEC_B * q_rows),
            ka: gguf_gpu_slot(&mut cursor, SPEC_B * kv_rows),
            va: gguf_gpu_slot(&mut cursor, SPEC_B * kv_rows),
            ao: gguf_gpu_slot(&mut cursor, SPEC_B * q_rows),
            xo: gguf_gpu_slot(&mut cursor, SPEC_B * dim),
            gate: gguf_gpu_slot(&mut cursor, SPEC_B * inter.max(dim)),
            up: gguf_gpu_slot(&mut cursor, SPEC_B * inter.max(dim)),
            logits: gguf_gpu_slot(&mut cursor, SPEC_B * vocab),
        };
        let act_bytes = gguf_gpu_page(cursor);
        let act = crate::infer::model::AnonBuffer::new(act_bytes / 4);
        let act_slice: &[u8] =
            unsafe { std::slice::from_raw_parts(act.as_ptr() as *const u8, act_bytes) };
        let act_buf = engine.register_zerocopy_only(act_slice)?;
        // P1 (llama.cpp parity): the GEMM prefill slab, built only when the
        // route is requested at load (SOFUU_GPU_PREFILL_GEMM=1). PRE_B=32
        // rows; every stage's [B x rows] region sized accordingly.
        let pre_enabled = std::env::var("SOFUU_GPU_PREFILL_GEMM")
            .map(|v| v == "1")
            .unwrap_or(false);
        let (pref_buf, pref_act, pref_slots, pref_bytes) = if pre_enabled {
            const PRE_B: usize = 32;
            let mut pc = 0usize;
            let ps = PrefillSlots {
                x: gguf_gpu_slot(&mut pc, PRE_B * dim),
                posb: gguf_gpu_slot(&mut pc, PRE_B * 16),
                xn: gguf_gpu_slot(&mut pc, PRE_B * dim),
                qa: gguf_gpu_slot(&mut pc, PRE_B * q_rows),
                ka: gguf_gpu_slot(&mut pc, PRE_B * kv_rows),
                va: gguf_gpu_slot(&mut pc, PRE_B * kv_rows),
                ao: gguf_gpu_slot(&mut pc, PRE_B * q_rows),
                xo: gguf_gpu_slot(&mut pc, PRE_B * dim),
                gate: gguf_gpu_slot(&mut pc, PRE_B * inter.max(dim)),
                up: gguf_gpu_slot(&mut pc, PRE_B * inter.max(dim)),
                logits: gguf_gpu_slot(&mut pc, vocab),
            };
            let pb = gguf_gpu_page(pc);
            let pa = crate::infer::model::AnonBuffer::new(pb / 4);
            let pslice: &[u8] =
                unsafe { std::slice::from_raw_parts(pa.as_ptr() as *const u8, pb) };
            let pbuf = engine.register_zerocopy_only(pslice)?;
            (Some(pbuf), Some(pa), Some(ps), pb)
        } else {
            (None, None, None, 0usize)
        };
        let k_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(k_cache.as_ptr() as *const u8, kv_region_bytes)
        };
        let v_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(v_cache.as_ptr() as *const u8, kv_region_bytes)
        };
        let k_buf = engine.register_zerocopy_only(k_bytes)?;
        let v_buf = engine.register_zerocopy_only(v_bytes)?;
        let meta_chunks = (config.max_seq_len + 127) / 128;
        let meta_head_f32 =
            config.n_heads * meta_chunks * 2 + config.n_heads * meta_chunks * config.head_dim;
        // per-layer region (layers overwrite each other's chunks otherwise);
        // the spec chain's b4 attention needs SPEC_B per-row meta regions
        let meta_per_layer = meta_head_f32;
        let meta_f32 =
            ((meta_per_layer * config.n_layers * SPEC_B * 4 + 16383) / 16384) * 16384 / 4;
        let meta = crate::infer::model::AnonBuffer::new(meta_f32);
        let meta_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(meta.as_ptr() as *const u8, meta_f32 * 4)
        };
        let meta_buf = engine.register_zerocopy_only(meta_bytes)?;
        eprintln!(
            "[gguf-gpu] decode client armed ({}): dim={dim} heads={}/{} hd={} inter={inter} vocab={vocab} layers={} act={}MB kv-shared={}MB",
            if token_commit { "ONE commit/TOKEN" } else if one_commit { "ONE commit/layer" } else { "3-commit" },
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            layers.len(),
            act_bytes / (1024 * 1024),
            kv_region_bytes / (1024 * 1024),
        );
        let skips = std::env::var("SOFUU_GPU_SKIP").unwrap_or_default();
        Some(Self {
            engine,
            w_buf,
            act_buf,
            act,
            k_buf,
            v_buf,
            meta_buf,
            meta,
            one_commit,
            token_commit,
            skip_attn: skips.contains("attn"),
            skip_ffn: skips.contains("ffn"),
            skip_head: skips.contains("head"),
            pref_buf,
            pref_act,
            pref_slots,
            pref_bytes,
            slots,
        })
    }

    /// One-commit eligibility: every layer full-attention, no biases, f32
    /// norms 4-byte aligned, head geometry the kernels support. Any miss
    /// names itself so the arm log is self-explanatory.
    fn one_commit_blocker(
        gguf: &GgufFile,
        config: &ModelConfig,
        layers: &[LayerWeights],
        is_gemma2: bool,
    ) -> Option<&'static str> {
        if config.head_dim > 128 {
            return Some("head_dim > 128 above kernel accumulators");
        }
        if config.n_kv_heads == 0 || config.n_heads % config.n_kv_heads != 0 {
            return Some("n_heads not divisible by n_kv_heads");
        }
        for i in 0..layers.len() {
            if TransformerModel::calc_layer_sliding_window(is_gemma2, config.sliding_window, i) != 0 {
                return Some("sliding-window layers present");
            }
            let l = &layers[i];
            if l.attn_q_bias.is_some() || l.attn_k_bias.is_some() || l.attn_v_bias.is_some() {
                return Some("attention biases present");
            }
            let f32_4b = |tv: &Option<TensorView>, what: &'static str| -> Option<&'static str> {
                if let Some(tv) = tv {
                    if tv.tensor_type != GGML_TYPE_F32 {
                        return Some(what);
                    }
                    if (gguf.data_offset + tv.offset) % 4 != 0 {
                        return Some("norm weights not 4-byte aligned");
                    }
                }
                None
            };
            if l.attn_norm.tensor_type != GGML_TYPE_F32 || l.ffn_norm.tensor_type != GGML_TYPE_F32 {
                return Some("block norms not f32");
            }
            if (gguf.data_offset + l.attn_norm.offset) % 4 != 0
                || (gguf.data_offset + l.ffn_norm.offset) % 4 != 0
            {
                return Some("norm weights not 4-byte aligned");
            }
            if let Some(why) = f32_4b(&l.attn_q_norm, "q_norm not f32") {
                return Some(why);
            }
            if let Some(why) = f32_4b(&l.attn_k_norm, "k_norm not f32") {
                return Some(why);
            }
        }
        if layers.len() * config.n_kv_heads * config.max_seq_len * config.head_dim * 4 % 16384 != 0 {
            return Some("k-v region not page-multiple");
        }
        None
    }

    #[inline]
    fn slot_f32_mut(&mut self, off: usize, n: usize) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(
                (self.act.as_ptr() as *mut u8).add(off) as *mut f32,
                n,
            )
        }
    }

    #[inline]
    fn slot_f32(&self, off: usize, n: usize) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                (self.act.as_ptr() as *const u8).add(off) as *const f32,
                n,
            )
        }
    }

    fn w_offset(&self, gguf: &GgufFile, tv: &TensorView) -> usize {
        gguf.data_offset + tv.offset
    }
}
