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
    dequantize_q3_k_row, dequantize_q6_k_row, dequantize_q8_0_row, f16_to_f32, kv_compact_style,
    matmat, matvec,
    prefetch_streaming_slice, prefetch_temporal_slice, prof,
    rmsnorm, rmsnorm_in_place, rope_style, swiglu_in_place, tree_attention, tree_rope, type_row_size,
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
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

/// One softmax chunk of the b4 prefill attention family — `ATTN_CHUNK` in
/// src/metal/kernels.metal. `llama_attn_1c_b4` keeps its softmax weight table
/// in a threadgroup `float tw[ATTN_CHUNK]` addressed by the row's ABSOLUTE
/// position, so it computes the right answer only while every row of the pass
/// sits inside the first chunk (max position <= 127). Past that the writes walk
/// off the table and the reads come back as something else: a silently wrong
/// softmax, not a slow one.
pub const ATTN_SOFTMAX_CHUNK: usize = 128;

/// True when a prefill pass of `rows` rows starting at `start_pos` may use the
/// fused one-chunk attention (`llama_attn_1c_b4`). `false` means the split
/// part+fin pair (which walks `ceil((pos+1)/ATTN_CHUNK)` chunks through meta)
/// owns the pass. EVERY `batch_llama_attn_1c_b4` call site must go through this
/// gate — the kernel itself cannot see the positions, and the .mm dispatch has
/// no way to refuse.
#[inline]
pub fn attn_1c_valid(start_pos: usize, rows: usize) -> bool {
    rows > 0 && start_pos + rows <= ATTN_SOFTMAX_CHUNK
}

/// Cap on tokens per prefill batch chunk. Overridable at process start with
/// `SOFUU_PREFILL_BATCH` (snapped to a multiple of 4 so the tiled GEMM path
/// stays eligible); read once per process.
pub fn max_prefill_batch() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let raw = std::env::var("SOFUU_PREFILL_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= 4);
        let limit = raw.map(|v| (v / 4) * 4).unwrap_or(DEFAULT_PREFILL_BATCH);
        // The batch size multiplies into nine per-layer buffers; an unbounded
        // value from a typo aborts the process on allocation failure.
        if limit > 512 {
            eprintln!("[infer] SOFUU_PREFILL_BATCH={limit} clamped to 512 (batch buffers scale linearly)");
            512
        } else {
            limit
        }
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
    /// True when the bytes came from the `vec!` fallback (heap) rather than
    /// mmap. Drop must use the matching deallocator: `munmap` on a heap
    /// pointer tears down allocator pages under concurrent threads.
    heap: bool,
}

unsafe impl Send for AnonBuffer {}
unsafe impl Sync for AnonBuffer {}

impl AnonBuffer {
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                heap: false,
            };
        }
        let byte_len = len.checked_mul(std::mem::size_of::<f32>()).expect("buffer size overflow");
        let mut heap = false;
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
                    heap = true;
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
                heap = true;
                let mut v = vec![0.0f32; len];
                let p = v.as_mut_ptr();
                std::mem::forget(v);
                p
            }
        };
        Self { ptr, len, heap }
    }

    /// Fallible, mmap-only variant for the buffers that dominate a model's
    /// resident footprint (the KV caches). An exhausted address space comes
    /// back as an error the loader can report and the CLI can print, instead
    /// of the Rust allocator aborting the process (release is panic = "abort").
    pub fn try_new(len: usize) -> Result<Self, String> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                heap: false,
            });
        }
        let byte_len = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "buffer size overflow".to_string())?;
        #[cfg(unix)]
        {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    byte_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANON | libc::MAP_PRIVATE,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(format!(
                    "cannot reserve {} MB: {}",
                    byte_len >> 20,
                    std::io::Error::last_os_error()
                ));
            }
            Ok(Self {
                ptr: p as *mut f32,
                len,
                heap: false,
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self::new(len))
        }
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
                    if self.heap {
                        let _ = Vec::from_raw_parts(self.ptr, self.len, self.len);
                    } else {
                        libc::munmap(self.ptr as *mut libc::c_void, byte_len);
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = byte_len;
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
    /// F-LC/P7 ring-KV mode: the LOGICAL conversation window the caller
    /// asked for (F-LC1 refusals and compaction triggers use this), while
    /// `max_seq_len` — which sizes and strides the f32 KV arrays — may be
    /// the smaller ring when the KIVI streaming prefill carries history in
    /// the quantized slab. Equals max_seq_len when ring mode is off.
    pub logical_max_seq_len: usize,
    /// 0 = off. Otherwise rows f32 KV cache holds (the ring); positions past
    /// it live only in the KIVI quantized store.
    pub kv_ring: usize,
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
    /// P1b: llama3 per-pair rope frequency multipliers (head_dim/2 entries).
    /// Empty = the C kernel path unchanged. Filled from GGUF
    /// `{arch}rope.scaling.*` or the explicit SOFUU_ROPE_LLAMA3 knob for
    /// checkpoints whose quant dropped the metadata.
    pub rope_freq_adjust: Vec<f32>,
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
    /// f16 KV shadows are present and the BATCH (prefill) attention should read them.
    /// Separate from kv_f16 so GPU decode can stay armed on the zero-copy f32 caches.
    pub k16_prefill: bool,
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

/// Which kernel carries one chain projection. Pure so the rule can be tested
/// without a GPU - and because the failure mode of getting it wrong is silent:
/// batch_gguf_b4 is a FOUR-TOKEN kernel with no row-count argument, so handing
/// it a 128-row chunk leaves 124 rows holding stale activations and nothing
/// in the log says a word.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ChainGemvArm {
    /// q4_K simdgroup-matrix GEMM over the whole chunk (SOFUU_GEMM_MMA8).
    Mma8,
    /// q6_K typed tiled GEMM over the whole chunk, B-generic.
    Q6Tiled,
    /// Any of q4_K/q6_K/q8_0 over a WIDE chunk through the typed tiled GEMM.
    Tiled,
    /// The four-token fused GEMV - only valid for a chunk of <= 4 rows.
    B4,
    /// One batch_gguf dispatch per row: the correct catch-all.
    PerRow,
}

/// The arm rule. Every dispatched arm reports whether it really enqueued, so
/// "false" here means "that kernel could not take this shape"; the fallbacks
/// are then chosen by shape alone. A four-row kernel is never returned for a
/// chunk wider than four rows, whatever the other arms did.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) fn chain_gemv_arm(
    ty: u32,
    b: usize,
    mma8_ok: bool,
    q6_tiled_ok: bool,
    tiled_ok: bool,
) -> ChainGemvArm {
    if mma8_ok {
        return ChainGemvArm::Mma8;
    }
    if q6_tiled_ok {
        return ChainGemvArm::Q6Tiled;
    }
    if tiled_ok {
        return ChainGemvArm::Tiled;
    }
    if ty == GGML_TYPE_Q4_K && b <= 4 {
        return ChainGemvArm::B4;
    }
    ChainGemvArm::PerRow
}
impl TransformerModel {
    pub fn from_gguf(gguf: GgufFile, max_seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        Self::from_gguf_with_kv_ring(gguf, max_seq_len, vocab_size, 0)
    }

    /// P7 (long-context §8): load with a RINGED f32 KV cache — the arrays
    /// hold only `kv_ring_req` rows while the logical window stays
    /// `max_seq_len`; prompt/decode positions past the ring are carried by
    /// the KIVI quantized streams, which the streaming prefill ingests
    /// chunk-by-chunk and attends causally.
    pub fn from_gguf_with_kv_ring(
        gguf: GgufFile,
        max_seq_len: usize,
        vocab_size: usize,
        kv_ring_req: usize,
    ) -> Result<Self, String> {
        Self::from_gguf_with_gpu(gguf, max_seq_len, vocab_size, true, kv_ring_req)
    }

    /// `gpu_allowed = false` (engine --cpu / CPU-load contract) keeps the
    /// SOFUU_GPU_DECODE client off even when the env flag is set.
    pub fn from_gguf_with_gpu(
        gguf: GgufFile,
        max_seq_len: usize,
        vocab_size: usize,
        gpu_allowed: bool,
        kv_ring_req: usize,
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
        // Zero-valued hyperparameters divide by zero deep inside the kernels
        // (n_rep = heads / kv_heads, pos % max_seq_len) and produce a SIGFPE
        // instead of a load error. Reject them by name here.
        if dim == 0 {
            return Err(format!("invalid GGUF '{arch}': {prefix}embedding_length is 0"));
        }
        if n_layers == 0 {
            return Err(format!("invalid GGUF '{arch}': {prefix}block_count is 0"));
        }
        if n_heads == 0 {
            return Err(format!("invalid GGUF '{arch}': {prefix}attention.head_count is 0"));
        }
        if n_kv_heads == 0 {
            return Err(format!("invalid GGUF '{arch}': {prefix}attention.head_count_kv is 0"));
        }
        if head_dim == 0 {
            return Err(format!("invalid GGUF '{arch}': head_dim resolves to 0"));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(format!(
                "invalid GGUF '{arch}': head_count {n_heads} is not a multiple of head_count_kv {n_kv_heads}"
            ));
        }
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

        if detected_ctx == 0 {
            return Err(format!("invalid GGUF '{arch}': context_length is 0"));
        }
        // NTK-aware rope scaling (PLAN-LONG-CONTEXT-PERSISTENCE §4.2 / §8 P1).
        // A request past the model's TRAINED context drifts rotary angles
        // out of distribution; scaling the base by α^(d/(d-2)) with
        // α = requested/trained restores sharp token-distance perception
        // without retraining. Static α chosen once at load from the REQUESTED
        // window (the standard llama.cpp-style formulation): every position
        // is roped with the SAME base, so the cache stays phase-consistent —
        // the per-position dynamic variant would need a full cache re-rope
        // on every base change. Engages only when requested > trained, so
        // every in-window run is bit-identical to before;
        // SOFUU_ROPE_NTK=0 restores the hard clamp, SOFUU_NTK_ALPHA forces a
        // factor for A/B receipts.
        let ntk_requested = if max_seq_len > 0 { max_seq_len } else { detected_ctx };
        let ntk_forced = std::env::var("SOFUU_NTK_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|a| *a >= 1.0);
        let ntk_disabled = std::env::var("SOFUU_ROPE_NTK").map(|v| v == "0").unwrap_or(false);
        let ntk_alpha = if let Some(a) = ntk_forced {
            a
        } else if ntk_disabled || ntk_requested <= detected_ctx {
            1.0
        } else {
            ntk_requested as f32 / detected_ctx as f32
        };
        let rope_freq_base = if ntk_alpha > 1.0 {
            let d = head_dim as f32;
            let scaled = rope_freq_base * ntk_alpha.powf(d / (d - 2.0));
            eprintln!(
                "[sofuu] NTK rope scaling: requested ctx {ntk_requested} > trained {detected_ctx} → α={ntk_alpha:.3}, base {rope_freq_base:.0} → {scaled:.0}"
            );
            scaled
        } else {
            rope_freq_base
        };

        let final_seq_len = if max_seq_len > 0 {
            if ntk_alpha > 1.0 {
                // Scaling IS the mechanism that makes the larger window
                // valid — clamping here would silently ignore --ctx again.
                max_seq_len
            } else {
                max_seq_len.min(detected_ctx)
            }
        } else {
            detected_ctx.min(4096)
        };
        // Below 4 tokens sink+drop collapse to 0 and compact_kv_cache cannot
        // make progress, so the decode loop would spin forever.
        if final_seq_len < 4 {
            return Err(format!(
                "context window of {final_seq_len} tokens is too small: the decoder needs at least 4 (raise --ctx)"
            ));
        }

        // P7 ring-KV: the f32 arrays shrink to the ring; everything cache-
        // indexed reads config.max_seq_len (= the ring) so every kernel call
        // site stays stride-correct, while compaction triggers / F-LC1
        // refusals consult logical_max_seq_len.
        let logical_max_seq_len = final_seq_len;
        let kv_ring = if kv_ring_req > 0 && kv_ring_req < final_seq_len {
            kv_ring_req
        } else {
            0
        };
        let cache_seq_len = if kv_ring > 0 { kv_ring } else { final_seq_len };
        // P1b llama3 rope-frequency table: HF-trained models (Llama-3.x)
        // were trained WITH this per-pair scaling — it is part of the
        // weights' semantics, so it applies at every position once present,
        // not only past the trained window. Source 1: the GGUF's own
        // `{arch}rope.scaling.*` keys. Source 2 (for checkpoints whose quant
        // dropped the keys, e.g. our Llama-3.2-1B Q8_0): the explicit knob
        // SOFUU_ROPE_LLAMA3="factor/orig_ctx[/low/high]" e.g. "32/8192".
        let mut rope_freq_adjust: Vec<f32> = Vec::new();
        let gguf_scaling = gguf.get_str(&format!("{prefix}rope.scaling.type"))
            .map(|t| t.to_string());
        let knob = std::env::var("SOFUU_ROPE_LLAMA3").ok();
        let llama3_spec: Option<(f32, usize, f32, f32, &'static str)> = if let Some(k) = knob {
            let parts: Vec<&str> = k.split('/').map(|x| x.trim()).collect();
            let parse = |x: &str| x.parse::<f64>().ok();
            match (parts.first().and_then(|v| parse(v)), parts.get(1).and_then(|v| parse(v))) {
                (Some(factor), Some(orig)) if factor > 1.0 && orig > 0.0 => {
                    let low = parts.get(2).and_then(|v| parse(v)).unwrap_or(1.0);
                    let high = parts.get(3).and_then(|v| parse(v)).unwrap_or(4.0);
                    Some((factor as f32, orig as usize, low as f32, high as f32, "SOFUU_ROPE_LLAMA3"))
                }
                _ => {
                    eprintln!("[sofuu] SOFUU_ROPE_LLAMA3: expected factor/orig_ctx[/low/high], got '{k}' — ignored");
                    None
                }
            }
        } else if gguf_scaling.as_deref() == Some("llama3") {
            let factor = gguf.get_f32(&format!("{prefix}rope.scaling.factor"))
                .or_else(|| gguf.metadata.get(&format!("{prefix}rope.scaling.factor")).and_then(|v| match v {
                    GgufValue::Float32(f) => Some(*f),
                    GgufValue::Uint32(u) => Some(*u as f32),
                    GgufValue::Float64(f) => Some(*f as f32),
                    _ => None,
                }))
                .unwrap_or(0.0);
            let orig = gguf.get_u32(&format!("{prefix}rope.scaling.original_context_length")).unwrap_or(0) as usize;
            let low = gguf.get_f32(&format!("{prefix}rope.scaling.low_freq_factor")).unwrap_or(1.0);
            let high = gguf.get_f32(&format!("{prefix}rope.scaling.high_freq_factor")).unwrap_or(4.0);
            if factor > 1.0 && orig > 0 {
                Some((factor, orig, low, high, "gguf metadata"))
            } else {
                None
            }
        } else {
            None
        };
        if let Some((factor, orig, low, high, source)) = llama3_spec {
            if head_dim >= 2 {
                rope_freq_adjust = super::quant::llama3_rope_adjust(head_dim, rope_freq_base, factor, orig, low, high);
                eprintln!(
                    "[rope] llama3 scaling armed ({source}): factor {factor}, orig ctx {orig}, low/high freq factors {low}/{high} — applies at every position (trained semantics)"
                );
            }
        }

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
            max_seq_len: cache_seq_len,
            logical_max_seq_len,
            kv_ring,
            sliding_window: detected_sliding_window,
            rope_interleaved,
        };
        if kv_ring > 0 {
            eprintln!(
                "[sofuu] KIVI ring KV cache: logical window {logical_max_seq_len}, f32 ring {kv_ring} rows — history past the ring lives in the quantized streams (P7)"
            );
        }

        // Helper to find tensor
        let find_tensor = |name: &str| -> Result<TensorView, String> {
            let info = gguf
                .tensors
                .get(name)
                .ok_or_else(|| format!("tensor not found: {name}"))?;
            // Gate unsupported quant types at load with a loud, actionable
            // error instead of a mid-forward panic (type_row_size's fallback
            // sizes IQ* as f32 and corrupts the slice math). The llama
            // matvec path handles exactly these types.
            const SUPPORTED: [u32; 7] = [
                crate::infer::quant::GGML_TYPE_F32,
                crate::infer::quant::GGML_TYPE_F16,
                crate::infer::quant::GGML_TYPE_Q4_0,
                crate::infer::quant::GGML_TYPE_Q8_0,
                crate::infer::quant::GGML_TYPE_Q4_K,
                crate::infer::quant::GGML_TYPE_Q6_K,
                crate::infer::quant::GGML_TYPE_Q3_K,
            ];
            if !SUPPORTED.contains(&info.tensor_type) {
                return Err(format!(
                    "tensor `{name}` uses GGML type {} — not supported by the CPU GGUF backend (supported: f32, f16, q4_0, q8_0, q4_k, q6_k, q3_k). Re-quantize the model to one of these, e.g. `llama-quantize` to Q4_K_M.",
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
        let kv_cache_size = n_layers * cache_seq_len * kv_dim;

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

        // KV caches hoisted so the GPU client can register them zero-copy:
        // one physical cache, written by whichever path runs (GPU kernels or
        // the CPU fallback), read by both. They are allocated *before* the
        // optional packed tiles: under memory pressure the tile layer must
        // yield (it falls back to the per-row kernels) rather than starve the
        // KV cache and abort the load.
        let kv_mb = kv_cache_size / (1 << 18); // f32 bytes / 2^20
        let k_cache = AnonBuffer::try_new(kv_cache_size).map_err(|e| {
            format!("cannot allocate the KV cache ({kv_mb} MB x2 for the configured context): {e}")
        })?;
        let v_cache = AnonBuffer::try_new(kv_cache_size)
            .map_err(|e| format!("cannot allocate the second KV cache ({kv_mb} MB): {e}"))?;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let gpu = {
            let wanted = gpu_allowed
                && cfg!(target_os = "macos")
                // DEFAULT ON. The round-69 "regression" was retracted in round 70: prefill routing
                // is env-gated (SOFUU_GPU_PREFILL*) and client arming does not touch it - both
                // paths showed [prof] gemm 168 ops, gemv 0. Same-run measurement (load 4.7):
                // GPU decode forward 18.6ms (kernels 15.2 + host 3.3) vs CPU 25.6ms.
                // SOFUU_GPU_DECODE=0 opts out.
                && std::env::var("SOFUU_GPU_DECODE").map(|v| v != "0").unwrap_or(true)
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
        // The f16 shadows are an optimization, not a requirement: when the
        // extra reservation fails under pressure, run with the f32 caches
        // (the pre-shadow behaviour) instead of aborting the load.
        // Prefill-time f16 KV (opt-out SOFUU_PREFILL_K16=0): the batch attention then reads
        // halved K/V bytes. Independent of kv_f16 because GPU decode binds the f32 caches
        // zero-copy, but prefill never touches the GPU - so both can coexist.
        let prefill_k16_want = !kv_f16
            && cfg!(target_arch = "aarch64")
            && kv_cache_size % 2 == 0
            && std::env::var("SOFUU_PREFILL_K16").map(|v| v != "0").unwrap_or(false);
        let want_shadow = kv_f16 || prefill_k16_want;
        let (k16_cache, v16_cache, k16_ok) = if want_shadow {
            match (
                AnonBuffer::try_new(kv_cache_size / 2),
                AnonBuffer::try_new(kv_cache_size / 2),
            ) {
                (Ok(k), Ok(v)) => (Some(k), Some(v), true),
                _ => {
                    eprintln!(
                        "[infer] f16 KV shadows disabled: cannot reserve {} MB more \
                         (running with the f32 cache)",
                        kv_cache_size >> 18
                    );
                    (None, None, false)
                }
            }
        } else {
            (None, None, false)
        };
        let kv_f16 = kv_f16 && k16_ok;
        let k16_prefill = prefill_k16_want && k16_ok;

        if kv_f16 && super::quant::env_flag("SOFUU_PROFILE") {
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

        let model = Self {
            config,
            rope_freq_adjust,
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
            attn_scores: vec![0.0f32; cache_seq_len.max(4)],
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
            k16_prefill,
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
        };

        // Warm the packed-tile GEMM/GEMV cache LAST: it is the only large
        // allocation the engine can do without, so the KV caches and every
        // fixed working buffer above are claimed first and the tiles take
        // whatever remains (the warm is OOM-safe; kernels fall back to the
        // per-row path, same math, slower).
        for tv in std::iter::once(&model.token_embd)
            .chain(std::iter::once(&model.output))
            .chain(model.layers.iter().flat_map(|l| {
                [&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output, &l.ffn_gate, &l.ffn_up, &l.ffn_down]
            }))
        {
            if matches!(tv.tensor_type, GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K | GGML_TYPE_Q6_K | GGML_TYPE_Q3_K) && tv.shape.len() == 2 {
                if let Some(slice) = model.gguf.mmap.slice(model.gguf.data_offset + tv.offset, 64) {
                    super::quant::tile_warm(slice.as_ptr(), tv.shape[1], tv.shape[0], tv.tensor_type);
                }
            }
        }
        Ok(model)
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

    /// Default break-even for the adaptive spec gate, in tokens per
    /// speculative step. The verify rows ride the batched GEMM path, whose
    /// cost per serial token depends on the weight format: tiled q4_K/q8_0
    /// verify near prefill-class rates (q4_K break-even ~2.2 — the default
    /// tree earns its keep at 2.4 accepted/step), but q3_K verify batches hit
    /// the ~3x-fatter native unpack and break even at ~2.8. Battery #12
    /// (Pi 5, 2.8 GHz+ovd): Qwen3-0.6B Q3_K_S at acceptance 2.400 ran 26.0
    /// tk/s with the tree vs 30.0 serial (llama.cpp 28.4), so q3_K defaults
    /// above 2.8 and the gate disables the tree after warmup.
    #[inline]
    pub fn spec_min_accept_default(&self) -> f32 {
        if self.dominant_proj_type == GGML_TYPE_Q3_K { 2.9 } else { 1.15 }
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
    ///
    /// The zero-fill is skipped by default. Every reader bounds its scan to
    /// the live window (`pos + 1`), every live position is written before it
    /// is read (`kv16_store` mirrors both stores on every write path), and the
    /// caches are anonymous mmap — lazily committed to the shared zero page —
    /// so a fill would only force the full 1.3 GB of a 4096-token context into
    /// physical memory (≈0.5 s on the Pi 5, once per fresh prompt) and keep it
    /// resident for a conversation that may never use it.
    /// `SOFUU_KV_ZERO=1` restores the fill for deterministic cache dumps.
    pub fn reset_cache(&mut self) {
        static ZERO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let zero = *ZERO.get_or_init(|| super::quant::env_flag("SOFUU_KV_ZERO"));
        if zero {
            self.k_cache.fill(0.0);
            self.v_cache.fill(0.0);
            if let Some(k16) = self.k16_cache.as_mut() {
                k16.as_u16_mut().fill(0);
            }
            if let Some(v16) = self.v16_cache.as_mut() {
                v16.as_u16_mut().fill(0);
            }
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
        // P6 contract (long-context §8): the quantized-KV decode path runs
        // the CPU layer loop only — execute_layers_checked bypasses the GPU
        // whole-layer batch while KIVI is armed (`kivi.is_none()` gate), so a
        // --kivi run on a GPU-visible load still computes on the CPU. Say so
        // once per process instead of silently dropping the GPU.
        static KIVI_CONTRACT_NOTICE: std::sync::Once = std::sync::Once::new();
        KIVI_CONTRACT_NOTICE.call_once(|| {
            // gpu is macOS/iOS-only (see the field); no GPU layer batch
            // exists on Linux/CI, so there is nothing to announce there.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if self.gpu.is_some() {
                eprintln!(
                    "[kivi] quantized-KV path: CPU layer loop only — the GPU whole-layer batch is bypassed while KIVI is armed (long-context contract, P6)"
                );
            }
        });
        let kivi_bits = match bits {
            2 => super::kivi::KiviBits::Two,
            _ => super::kivi::KiviBits::Four,
        };
        if cur_pos > 0 {
            let mut kivi = self.build_kivi_cache(kivi_bits);
            self.ingest_rows_into_kivi(&mut kivi, 0, cur_pos);
            self.kivi = Some(kivi);
        } else {
            self.kivi = Some(self.build_kivi_cache(kivi_bits));
        }
    }

    /// P7: arm EMPTY KIVI streams before the streaming prefill — each
    /// prefill chunk ingests itself right after its rope (ring mode);
    /// no f32 re-ingest ever runs.
    pub fn enable_kivi_streaming(&mut self, bits: usize) {
        let kivi_bits = match bits {
            2 => super::kivi::KiviBits::Two,
            _ => super::kivi::KiviBits::Four,
        };
        self.kivi = Some(self.build_kivi_cache(kivi_bits));
        if self.config.kv_ring == 0 {
            eprintln!(
                "[kivi] streaming arm requested without a ring cache (SOFUU_KV_RING / --kv-ring): history past the f32 window cannot be carried — falling back to post-prefill ingest semantics"
            );
        }
    }

    /// Shared KIVI construction (sink-pinned config, P4).
    fn build_kivi_cache(&self, kivi_bits: super::kivi::KiviBits) -> super::kivi::KiviCache {
        let config = super::kivi::KiviConfig {
            enabled: true,
            bits: kivi_bits,
            residual_window: 32,
            group_size: 32,
            sink_tokens: 4,
        };
        super::kivi::KiviCache::new(
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
            config,
        )
    }

    /// Append cache rows [pos .. seq_len) of every layer/head into `kivi`
    /// (shared by enable_kivi and the post-compaction rebuild).
    fn ingest_rows_into_kivi(&self, kivi: &mut super::kivi::KiviCache, pos: usize, seq_len: usize) {
        let n_layers = self.config.n_layers;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let max_seq_len = self.config.max_seq_len;
        for layer_idx in 0..n_layers {
            for p in pos..seq_len {
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

    /// F-LC3 (long-context §8 P5): compact_kv_cache() drops middle tokens,
    /// shifts the survivors left and re-ropes them to their new positions —
    /// a KIVI stream built before the compaction would keep attending the
    /// dropped history under positions that no longer exist. Rebuild the
    /// stream from the compacted cache so the quantized store and the f32
    /// store describe the same (sinks + survivors) history. No-op when KIVI
    /// is not armed.
    pub fn rebuild_kivi_after_compact(&mut self, seq_len: usize) {
        let old = match self.kivi.take() {
            Some(k) => k.config,
            None => return,
        };
        let mut kivi = super::kivi::KiviCache::new(
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
            old,
        );
        if seq_len > 0 {
            self.ingest_rows_into_kivi(&mut kivi, 0, seq_len);
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
            GGML_TYPE_Q3_K => {
                dequantize_q3_k_row(row_slice, dst);
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

    /// SOFUU_KV_DUMP=<prefix>: after prompt prefill, write the model's KV
    /// caches and the residual x — the CPU-route vs GPU-route byte diff at
    /// the same pipeline point separates kv_write bugs from attention-read
    /// bugs (the caches are the same zero-copy memory both routes use).
    pub fn maybe_dump_kv(&self) {
        if let Some(prefix) = std::env::var_os("SOFUU_KV_DUMP") {
            let prefix = prefix.to_string_lossy().to_string();
            let kvb = self.layers.len()
                * self.config.n_kv_heads
                * self.config.max_seq_len
                * self.config.head_dim
                * 4;
            let write = |suffix: &str, bytes: &[u8]| {
                let _ = std::fs::write(format!("{prefix}.{suffix}"), bytes);
            };
            let k: &[u8] = unsafe {
                std::slice::from_raw_parts(self.k_cache.as_ptr() as *const u8, kvb)
            };
            let v: &[u8] = unsafe {
                std::slice::from_raw_parts(self.v_cache.as_ptr() as *const u8, kvb)
            };
            let x: &[u8] = unsafe {
                std::slice::from_raw_parts(self.x.as_ptr() as *const u8, self.x.len() * 4)
            };
            let lg: &[u8] = unsafe {
                std::slice::from_raw_parts(self.logits.as_ptr() as *const u8, self.logits.len() * 4)
            };
            write("k.bin", k);
            write("v.bin", v);
            write("x.bin", x);
            write("logits.bin", lg);
            if !self.x_norm_batch.is_empty() {
                let n = self.x_norm_batch.len().min(4096);
                let xn: &[u8] = unsafe {
                    std::slice::from_raw_parts(self.x_norm_batch.as_ptr() as *const u8, n * 4)
                };
                write("xn.bin", xn);
            }
            eprintln!("[kv-dump] wrote {prefix}.k/v/x/logits ({} vars)", self.x.len());
        }
    }

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
        self.maybe_dump_kv();
    }

    /// Prefill all prompt tokens and compute logits for the very last token in a single pass,
    /// avoiding an extra single-token evaluation over model weights.
    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {

        // LEDGER #141: macOS defaults to 4 CPU threads (platform_topology.h:17) and measures 427 tk/s,
        // while 6 threads measures 605 tk/s on the same prompt (+41%). The C++ default is in the peer
        // lane, so the correction is applied here: leave an explicit SOFUU_NUM_THREADS alone, and on
        // a machine with 8+ logical cores use 6 - the measured knee on M2 Pro (4/6/8/10 -> 405/481/373/339).
        #[cfg(target_os = "macos")]
        {
            if std::env::var_os("SOFUU_NUM_THREADS").is_none() {
                let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
                let want = if cores >= 8 { 6 } else { cores.saturating_sub(1).clamp(1, 4) };
                if crate::infer::quant::get_thread_count() != want {
                    crate::infer::quant::set_thread_count(want);
                }
            }
        }        if tokens.is_empty() {
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
                self.forward_prefill_batch_chunk_internal(&tokens[off..off + take], cur_pos, true);
                self.maybe_dump_kv();
                return &self.logits;
            }
            self.forward_prefill_batch_chunk_internal(&tokens[off..off + main], cur_pos, false);
            cur_pos += main;
            self.forward_prefill_batch_chunk_internal(&tokens[off + main..off + take], cur_pos, true);
            self.maybe_dump_kv();
            return &self.logits;
        }
        self.maybe_dump_kv();
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

        let gguf = &self.gguf;

        // 1. Batched Embedding lookup
        for (b, &tok) in tokens.iter().enumerate() {
            let dst = &mut self.x_batch[b * dim..(b + 1) * dim];
            Self::lookup_embedding(gguf, &self.token_embd, tok, dim, dst);
        }
        if let Some(prefix) = std::env::var_os("SOFUU_LAYER_SNAP") {
            let prefix = prefix.to_string_lossy().to_string();
            let esl: &[u8] = unsafe {
                std::slice::from_raw_parts(self.x_batch.as_ptr() as *const u8, tokens.len() * dim * 4)
            };
            let _ = std::fs::write(format!("{prefix}.emb"), esl);
        }

        // 2. Transformer layers
        let snap_layer: usize = std::env::var("SOFUU_QKV_SNAP_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        for layer_idx in 0..self.config.n_layers {
            self.forward_batch_layer_rows(layer_idx, b_size, start_pos, snap_layer);
        }
    }

    /// W1 row split: ONE layer of the batched prefill tile path over rows
    /// [0, b_size) of the *_batch buffers, at positions
    /// [start_pos, start_pos + b_size). This is the loop body of
    /// forward_batch_layers_chunk moved verbatim into its own function so the
    /// CPU+GPU split driver can interleave the CPU's rows with the GPU's rows
    /// one layer at a time (the CPU's rows are the FIRST rows of the chunk, so
    /// they never need the GPU's same-chunk K/V: their attention covers
    /// positions <= start_pos + i). Behaviour is unchanged: the batching loop
    /// calls it once per layer exactly as before.
    fn forward_batch_layer_rows(
        &mut self,
        layer_idx: usize,
        b_size: usize,
        start_pos: usize,
        snap_layer: usize,
    ) {
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
            // SOFUU_LAYER_SNAP=<prefix>: per-layer residual + normed-x dump
            // (the CPU tiles' mirror of the route's SOFUU_PREFILL_DEBUG_DUMP
            // slab dumps — the L-granularity divergence pin).
            if let Some(prefix) = std::env::var_os("SOFUU_LAYER_SNAP") {
                let prefix = prefix.to_string_lossy().to_string();
                let xsl: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        self.x_batch.as_ptr() as *const u8,
                        b_size * dim * 4,
                    )
                };
                let nsl: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        self.x_norm_batch.as_ptr() as *const u8,
                        b_size * dim * 4,
                    )
                };
                let _ = std::fs::write(format!("{prefix}.L{layer_idx}.xb"), xsl);
                let _ = std::fs::write(format!("{prefix}.L{layer_idx}.xn"), nsl);
            }
            // SOFUU_QKV_SNAP stage s1 (attn-norm out) — GPU mirror in the
            // prefill gemm core.
            if layer_idx == snap_layer {
                if let Some(prefix) = std::env::var_os("SOFUU_QKV_SNAP") {
                    let prefix = prefix.to_string_lossy().to_string();
                    let nsl: Vec<u8> = self.x_norm_batch[0..b_size * dim]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let _ = std::fs::write(format!("{prefix}.s1.xn"), &nsl);
                }
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
            if layer_idx == snap_layer {
                if let Some(prefix) = std::env::var_os("SOFUU_QKV_SNAP") {
                    let prefix = prefix.to_string_lossy().to_string();
                    let seg = |src: &[f32], n: usize| -> Vec<u8> {
                        src[..n].iter().flat_map(|v| v.to_le_bytes()).collect()
                    };
                    let _ = std::fs::write(
                        format!("{prefix}.s2.qa"),
                        seg(&self.q_batch, b_size * q_dim),
                    );
                    let _ = std::fs::write(
                        format!("{prefix}.s2.ka"),
                        seg(&self.k_batch, b_size * kv_dim),
                    );
                    let _ = std::fs::write(
                        format!("{prefix}.s2.va"),
                        seg(&self.v_batch, b_size * kv_dim),
                    );
                }
            }
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
            if !self.rope_freq_adjust.is_empty() {
                // P1b: llama3-scaled rope — the batched C kernel takes only a
                // scalar base, so rope each row here (rope is ~1% of a prefill
                // step) and store the rows to the cache ourselves.
                for b in 0..b_size {
                    let p = start_pos + b;
                    super::quant::rope_style_adjust(
                        &mut self.q_batch[b * q_dim..(b + 1) * q_dim],
                        n_heads, head_dim, p, rope_freq_base, rope_mode, &self.rope_freq_adjust,
                    );
                    super::quant::rope_style_adjust(
                        &mut self.k_batch[b * kv_dim..(b + 1) * kv_dim],
                        n_kv_heads, head_dim, p, rope_freq_base, rope_mode, &self.rope_freq_adjust,
                    );
                    let cache_pos = p % max_seq_len;
                    for kv_h in 0..n_kv_heads {
                        let src = b * kv_dim + kv_h * head_dim;
                        let slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim
                            + cache_pos * head_dim;
                        let krow: Vec<f32> = self.k_batch[src..src + head_dim].to_vec();
                        let vrow: Vec<f32> = self.v_batch[src..src + head_dim].to_vec();
                        self.k_cache[slot..slot + head_dim].copy_from_slice(&krow);
                        self.v_cache[slot..slot + head_dim].copy_from_slice(&vrow);
                        kv16_store(self.k16_cache.as_mut(), slot, &krow);
                        kv16_store(self.v16_cache.as_mut(), slot, &vrow);
                    }
                }
            } else if self.kv_f16 || self.k16_prefill {
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


            // SOFUU_QKV_SNAP s3 (K/V cache rows post k-norm+rope) + s4
            // (q post q-norm+rope) — GPU mirror after kv_write/head_rmsnorm.
            if layer_idx == snap_layer {
                if let Some(prefix) = std::env::var_os("SOFUU_QKV_SNAP") {
                    let prefix = prefix.to_string_lossy().to_string();
                    let msl = self.config.max_seq_len;
                    let hd = head_dim;
                    let mut kseg = Vec::with_capacity(b_size * n_kv_heads * hd * 4);
                    let mut vseg = kseg.clone();
                    for kh in 0..n_kv_heads {
                        let base = (kh * msl + start_pos) * hd;
                        for i in base..base + b_size * hd {
                            kseg.extend_from_slice(&self.k_cache[i].to_le_bytes());
                        }
                        for i in base..base + b_size * hd {
                            vseg.extend_from_slice(&self.v_cache[i].to_le_bytes());
                        }
                    }
                    let _ = std::fs::write(format!("{prefix}.s3.kc"), &kseg);
                    let _ = std::fs::write(format!("{prefix}.s3.vc"), &vseg);
                    let qsl: Vec<u8> = self.q_batch[0..b_size * q_dim]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let _ = std::fs::write(format!("{prefix}.s4.qa"), &qsl);
                }
            }

            // ── Multi-Head / Grouped-Query Attention for batch ───
            // Batched prefill keeps the f32 score kernel: at 8+ rows the
            // shares-K-strip pass is compute-bound, and the f16 converts cost
            // more than the halved bytes save there. The f16 shadows are still
            // written above, so decode reads them.
            let sliding_window = self.layer_sliding_window(layer_idx);
            if let Some(mut kivi) = self.kivi.take() {
                // P7 streaming quantized prefill: this chunk's rows were just
                // roped (k_batch/v_batch) — ingest them per head, then causal
                // softmax-attend over sinks + quantized history + this chunk
                // (stream index = position, so causality is a row cap).
                for kv_h in 0..n_kv_heads {
                    for r in 0..b_size {
                        let base = r * kv_dim + kv_h * head_dim;
                        let idx = layer_idx * n_kv_heads + kv_h;
                        kivi.heads[idx].append(
                            &self.k_batch[base..base + head_dim],
                            &self.v_batch[base..base + head_dim],
                        );
                    }
                }
                for i in 0..b_size {
                    let qrow = &self.q_batch[i * q_dim..(i + 1) * q_dim];
                    let outrow = &mut self.head_out_batch[i * q_dim..(i + 1) * q_dim];
                    kivi.compute_layer_attention_causal(layer_idx, n_heads, qrow, outrow, start_pos + i + 1);
                }
                self.kivi = Some(kivi);
            } else if self.k16_prefill {
                super::quant::batched_attention_ext_k16(
                    &self.q_batch[0..b_size * q_dim],
                    self.k16_cache.as_ref().unwrap().as_u16(),
                    self.v16_cache.as_ref().unwrap().as_u16(),
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
            } else {
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
            }

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

        // W1: CPU+GPU ROW SPLIT (SOFUU_SPLIT=1, default OFF - shipped config
        // unchanged). Rows [0,kc) of the chunk run through the CPU tile path
        // while rows [kc,b) run on the GPU in the same wall-clock window; see
        // gpu_forward_prefill_split for the ordering argument. A false return
        // means "not taken" and the chunk falls through to the whole-chunk
        // routes below, so the split can never be why a chunk fails.
        // W2: CPU+GPU FEATURE split (SOFUU_SPLIT_AXIS=feat). Checked before
        // the row split so the axis knob actually selects the axis; a false
        // return falls through to the row split and then the whole-chunk
        // routes, so the arm can never be why a chunk fails.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(sh) = self.prefill_feat_shares(b_size) {
            if self.gpu_forward_prefill_feat(tokens, start_pos, sh, compute_last_logits) {
                self.logits_gap_probe_tagged("feat_head", start_pos + b_size - 1);
                return &self.logits;
            }
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(kg) = self.prefill_split_gpu_rows(b_size) {
            if self.gpu_forward_prefill_split(tokens, start_pos, kg, compute_last_logits) {
                self.logits_gap_probe_tagged("split_head", start_pos + b_size - 1);
                return &self.logits;
            }
        }

        // P1 (llama.cpp parity): B<=pre_chunk chunks through the prefill
        // GEMM class (SOFUU_GPU_PREFILL_GEMM=1; cap = SOFUU_PREFILL_CHUNK,
        // clamped 1..=128, default 32). Any unsupported chunk falls back to
        // the b4 route / CPU tiles below.
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
        // Chain chunk width: 128 rows is the measured knee with the simdgroup-matrix GEMM
        // (the b4c2 kernels compute only 4 rows); everything else walks 4 rows so the
        // fused 4-row kernels stay in play. The walk must NOT depend on the MMA flag,
        // or plain SOFUU_GPU_PREFILL=1 silently falls through to the CPU path and looks
        // like a slow GPU route (LEDGER #47).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let chain_chunk = if crate::infer::metal::MetalEngine::mma8_enabled()
            && std::env::var_os("SOFUU_MMA8_CHUNK4").is_none()
        {
            std::env::var("SOFUU_CHAIN_CHUNK")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(128)
        } else {
            4
        };
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if b_size > chain_chunk {
            let mut off = 0usize;
            let mut all_ok = true;
            while off < b_size {
                let end = (off + chain_chunk).min(b_size);
                let is_last = end == b_size;
                if !self.gpu_forward_prefill_chain(
                    &tokens[off..end], start_pos + off, compute_last_logits && is_last,
                ) {
                    all_ok = false;
                    break;
                }
                off = end;
            }
            if all_ok {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                self.logits_gap_probe_tagged("chain_head", start_pos + b_size - 1);
                return &self.logits;
            }
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.gpu_forward_prefill_chain(tokens, start_pos, compute_last_logits) {
            self.logits_gap_probe_tagged("chain_head", start_pos + b_size - 1);
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
            self.logits_gap_probe_tagged("tiles_head", start_pos + tokens.len() - 1);
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
        self.logits_gap_probe_tagged("cpu_tiles_head", start_pos + tokens.len() - 1);
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
        if !g.engine.batch_end() {
            return false;
        }
        out_logits[0..vocab_size].copy_from_slice(g.slot_f32(g.slots.logits, vocab_size));
        self.logits[0..vocab_size].copy_from_slice(&out_logits[0..vocab_size]);
        true
    }

    /// No GPU slab on other targets: the caller's CPU fallback is the path.
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn spec_chain_head_logits(&mut self, _item_idx: usize, _out_logits: &mut [f32]) -> bool {
        false
    }

    /// Whether the GPU chain pass supplies per-node head logits (then the CPU
    /// batched pass below is unnecessary).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn gpu_chain_heads(&self) -> bool {
        self.spec_rows_valid && self.gpu.as_ref().map(|g| g.token_commit).unwrap_or(false)
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn gpu_chain_heads(&self) -> bool {
        false
    }

    /// LM-head logits for `n_rows` speculative batch items starting at
    /// `first_row`, in ONE pass over the output tensor.
    ///
    /// The tree walk needs the head's prediction at every node it visits. Doing
    /// that lazily costs a full read of the lm head per node — 127 MB on
    /// Qwen3-0.6B, i.e. ~10.5 ms on the Pi, and the walk visits 3-5 nodes per
    /// step — which is why speculative steps measured 65-102 ms and the default
    /// (speculation on) lost to serial decode.
    /// Rows available in the shared batch buffers (`x_batch`, `x_norm_batch`)
    /// that BOTH the speculative tree forward and the batched head pass write
    /// into, so one number bounds both.
    pub fn prefill_batch_rows(&self) -> usize {
        max_prefill_batch()
    }

    pub fn spec_head_logits_batch(&mut self, first_row: usize, n_rows: usize, out: &mut [f32]) {
        let dim = self.config.dim;
        let vocab_size = self.config.vocab_size;
        let n_rows = n_rows.min(max_prefill_batch().saturating_sub(first_row));
        assert!(n_rows > 0, "spec_head_logits_batch needs at least one row");
        assert!(
            first_row + n_rows <= self.x_norm_batch.len() / dim,
            "spec batch row {first_row}+{n_rows} exceeds x_norm_batch"
        );
        assert_eq!(out.len(), n_rows * vocab_size, "out must hold n_rows x vocab");
        let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
        matmat(
            out_bytes,
            self.output.tensor_type,
            vocab_size,
            dim,
            &self.x_norm_batch[first_row * dim..(first_row + n_rows) * dim],
            n_rows,
            &mut out[0..n_rows * vocab_size],
        );
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
        self.execute_layers_checked(pos, start_layer, end_layer);
    }

    /// Runs the transformer layers for one token. Returns `true` only when the
    /// GPU whole-layer path ran to completion — its in-batch head has then
    /// already written `self.logits`, and `forward_impl` may skip the CPU
    /// head. Returns `false` when the CPU layer loop ran (KIVI,
    /// streaming-weights, dynamic-FFN and rejected Metal command buffers all
    /// fall back this way), in which case the CPU head MUST run.
    /// F-LC4: the early-return in `forward_impl` used to fire on
    /// `token_commit` alone, so with KIVI armed the CPU layer loop ran but the
    /// head was skipped and `self.logits` stayed frozen at the prefill
    /// values — the sampler then re-picked the prefill argmax every step
    /// ("Berlin Berlin Berlin…").
    pub fn execute_layers_checked(&mut self, pos: usize, start_layer: usize, end_layer: usize) -> bool {
        // GPU2: whole-layer Metal batches when armed and the CPU-side
        // special paths are all off (streaming/dynamic-FFN/KIVI fall back).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            if self.gpu.is_some()
                && !self.streaming_weights
                && !self.dynamic_ffn.enabled
                && self.kivi.is_none()
                && self.rope_freq_adjust.is_empty()
            {
                if self.execute_layers_gpu(pos, start_layer, end_layer) {
                    return true;
                }
                // A rejected command buffer tears (partial writes, later ops
                // killed). Falling through to the full CPU layer loop from
                // start_layer is still coherent: every layer's KV write for the
                // token's positions is overwritten (GPU and CPU KV are
                // bit-identical per the parity receipts, so the rerun is
                // idempotent), and self.x here is the token's layer INPUT — the
                // GPU paths only sync x back to the host at their successful
                // tails, never on a failure return.
                // layers 0..(start_layer) were never touched by this call.
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
            if !self.rope_freq_adjust.is_empty() {
                super::quant::rope_style_adjust(&mut self.q, n_heads, head_dim, pos, rope_freq_base, rope_mode, &self.rope_freq_adjust);
                super::quant::rope_style_adjust(&mut self.k, n_kv_heads, head_dim, pos, rope_freq_base, rope_mode, &self.rope_freq_adjust);
            } else {
                rope_style(&mut self.q, n_heads, head_dim, pos, rope_freq_base, rope_mode);
                rope_style(&mut self.k, n_kv_heads, head_dim, pos, rope_freq_base, rope_mode);
            }

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
        false
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

    /// SOFUU_LOGITS_GAP_PROBE=<path>: append the top-8 (value, id) of the
    /// just-computed logits — one line per token, CPU and GPU routes
    /// append to their own files; the top-2 gap at the first divergent
    /// line classifies a text fork as near-tie (reassociation) vs real.
    /// Shared by every target: the CPU tile/head call sites are not Apple-only.
    fn logits_gap_probe_tagged(&self, tag: &str, pos: usize) {
        if let Some(p) = std::env::var_os("SOFUU_LOGITS_GAP_PROBE") {
            let mut idx: Vec<usize> = (0..self.logits.len()).collect();
            idx.sort_by(|a, b| self.logits[*b].partial_cmp(&self.logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let line: Vec<String> = idx.iter().take(8)
                .map(|i| format!("({:.6e},{})", self.logits[*i], i))
                .collect();
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                let _ = writeln!(f, "{} pos={} {}", tag, pos, line.join(" "));
            }
        }
    }

/// Arm A (plan): the act-quantized decode GEMVs (a76 dotprod serial
    /// mirrors of the CPU matvec) are now the DEFAULT, because they were
    /// gate-clean fleet-wide and are the decode-parity counterpart of the
    /// prefill tile MMQ. Opt out with SOFUU_DECODE_MMQ=0.
    fn fuse_norm_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("SOFUU_FUSE_NORM").map(|v| v == "1").unwrap_or(false))
    }

    fn fuse_qkv_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // measured 10% SLOWER than three separate dispatches (86 vs 96 tg/s), so off.
        *ON.get_or_init(|| std::env::var("SOFUU_FUSE_QKV").map(|v| v == "1").unwrap_or(false))
    }

    fn decode_mmq_enabled() -> bool {
        // Opt-in since the coalesced-2 GEMV landed: MMQ bypasses that kernel,
        // and with both armed the 0.6B decodes at 50.6 tg/s against 71.6 with
        // MMQ off (max of 3, same session). Arm A's receipt attributed the
        // whole 38.3 -> 52.0 gain to MMQ while SOFUU_Q4K_COALESCED=2 shipped in
        // the same commit; the honest split is that coalesced-2 carries it and
        // MMQ costs 21 tg/s on top of it.
        matches!(
            std::env::var("SOFUU_DECODE_MMQ").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
    }

    /// lm_head as one Metal batch (CPU norm; staged slot in, logits out).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn ggpu_head(&mut self) -> bool {
        let decode_mmq_head = Self::decode_mmq_enabled();
        if let Some(g) = self.gpu.as_mut() {
            let dim = self.config.dim;
            let vocab = self.config.vocab_size;
            let self_x: &mut [f32] = &mut self.x;
            let dst = g.slot_f32_mut(g.slots.xn, dim);
            dst[..dim].copy_from_slice(&self_x[..dim]);
            let off = g.w_offset(&self.gguf, &self.output);
            assert!(g.engine.batch_open());
            {
                // q6-only decode head mirror (same probe receipt as above).
                let mmqh = decode_mmq_head
                    && self.output.tensor_type == GGML_TYPE_Q6_K
                    && g.engine.batch_quantize_q8_k(
                        &g.act_buf, g.slots.xn, &g.act_buf, g.slots.xq8e, dim, 1,
                    )
                    && g.engine.batch_gemm_q6_k_mmq(
                        &g.w_buf, off,
                        &g.act_buf, g.slots.xq8e, &g.act_buf, g.slots.logits,
                        vocab, dim, 1,
                    );
                if !mmqh {
                    g.engine.batch_gguf(
                        self.output.tensor_type,
                        &g.w_buf, off,
                        &g.act_buf, g.slots.xn,
                        &g.act_buf, g.slots.logits,
                        vocab, dim,
                    );
                }
            }
            if !g.engine.batch_end() {
                return false;
            }
            let src = g.slot_f32(g.slots.logits, vocab);
            self.logits.copy_from_slice(src);
        }
        self.logits_gap_probe_tagged("ggpu_head", usize::MAX);
        true
    }

    fn logits_gap_probe(&self) {
        self.logits_gap_probe_tagged("cpu_head", usize::MAX);
    }

    /// GPU2 decode path: same CPU glue (norms, QK-norms, RoPE, KV, attention,
    /// residuals) with the seven per-layer GEMVs batched into three commits.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn execute_layers_gpu(&mut self, pos: usize, start_layer: usize, end_layer: usize) -> bool {
        let end = end_layer.min(self.config.n_layers);
        // Wrap-around (context past max_seq_len) stays on the 3-commit path,
        // whose KV writes match the CPU modulo arithmetic.
        if self.gpu.as_ref().map(|g| g.one_commit).unwrap_or(false) && pos < self.config.max_seq_len {
            if !self.layers_one_commit(pos, start_layer, end) {
                return false;
            }
            return true;
        }
        let windows: Vec<usize> = (0..self.config.n_layers)
            .map(|i| self.layer_sliding_window(i))
            .collect();
        for layer_idx in start_layer..end {
            if !self.gpu_layer_step(layer_idx, pos, windows[layer_idx]) {
                return false;
            }
        }
        true
    }

    /// GPU2.1 decode step: ALL layers across ONE commit each. Norms run as
    /// in-batch kernels reading f32 weights straight from the weight mmap;
    /// q/k RoPE + QK-norm fuse into the KV-write/attention kernels; x, the
    /// activation slots and the KV caches (registered zero-copy, shared with
    /// the CPU) are all command-buffer-resident between layers; x syncs back
    /// to Vec once per token.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn layers_one_commit(&mut self, pos: usize, start_layer: usize, end_layer: usize) -> bool {
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

        // SOFUU_DECODE_MMQ=1: per-token GEMVs through the act-quantized
        // kernels that bit-mirror the CPU's a76 dotprod serial matvec (the
        // decode-parity counterpart of SOFUU_GPU_TILE_MMQ).
        let decode_mmq = Self::decode_mmq_enabled();
        // Bisect filter: SOFUU_DECODE_MMQ_OPS=q4|q6 (both by default)
        let dmmq_ops = std::env::var("SOFUU_DECODE_MMQ_OPS").unwrap_or_default();
        let dmmq_q4 = dmmq_ops.is_empty() || dmmq_ops.contains("q4");
        let dmmq_q6 = dmmq_ops.is_empty() || dmmq_ops.contains("q6");
        macro_rules! dec_gguf {
            ($tv:expr, $w_off:expr, $src:expr, $dst:expr, $rows:expr, $cols:expr) => {{
                // timing ablation: SOFUU_GPU_SKIP=mm drops every projection dispatch
                if g.skip_mm { } else {
                // q6_K only: the q6 tile GEMV is numerically the per-256
                // batch-mmq pipeline at B=1 (probe-exact); the q4 tile GEMV
                // uses its own blocklen-8 interleave — booked separately.
                // The decode mirror uses the POS row-quantizer for BOTH
                // types: the signed-max kernel triggered a combo-only live
                // regression (fork step 19→2) while pos is sign-equivalent
                // and bisects clean (SOFUU_DECODE_MMQ_OPS=q6pos receipt).
                let (mmqd, is_q4) = if decode_mmq && dmmq_q6 && $tv.tensor_type == GGML_TYPE_Q6_K {
                    (
                        g.engine.batch_quantize_q8_k_pos(
                            &g.act_buf, $src, &g.act_buf, g.slots.xq8e, $cols, 1,
                        ) && g.engine.batch_gemm_q6_k_mmq(
                            &g.w_buf, $w_off, &g.act_buf, g.slots.xq8e,
                            &g.act_buf, $dst, $rows, $cols, 1,
                        ),
                        false,
                    )
                } else if decode_mmq && dmmq_q4 && $tv.tensor_type == GGML_TYPE_Q4_K {
                    // q4 tile GEMV consumes the ROW quantizer convention
                    // (positive d); the gemm mirror kernel is convention-free.
                    (
                        g.engine.batch_quantize_q8_k_pos(
                            &g.act_buf, $src, &g.act_buf, g.slots.xq8e, $cols, 1,
                        ) && g.engine.batch_gemm_q4_k_mmq(
                            &g.w_buf, $w_off, &g.act_buf, g.slots.xq8e,
                            &g.act_buf, $dst, $rows, $cols, 1,
                        ),
                        true,
                    )
                } else {
                    (false, false)
                };
                let _ = is_q4;
                if !mmqd {
                    if !g.engine.batch_gguf($tv.tensor_type, &g.w_buf, $w_off,
                        &g.act_buf, $src, &g.act_buf, $dst, $rows, $cols) {
                        return false;
                    }
                }
                }
            }};
        }
        g.slot_f32_mut(g.slots.x, dim).copy_from_slice(&self.x[..dim]);
        let prof = std::env::var_os("SOFUU_GPU_PROFILE").is_some();
        let t_start = std::time::Instant::now();
        if token_commit {
            assert!(g.engine.batch_open());
        }

        // LEDGER #192: arm the flat GEMV sweep for this decode token only.
        // LEDGER #193: OPT-IN only. Forcing the flat sweep here made the GPU one-commit
        // batch get REJECTED (sofuu gpu_batch=false -> silent CPU fallback at 30-220ms/token),
        // so #192s measured "win" was the fallback, not a faster GPU path.
        // W0/A2: hoist every env/OnceLock read out of the per-layer loop. These were read
        // 28x per token (each std::env::var walks environ under a lock).
        let dec_attn_env = std::env::var("SOFUU_GPU_ATTN").map(|v| v != "0").unwrap_or(true);
        let dec_attn_min = crate::infer::metal::MetalEngine::attn_chunk_min() as usize;
        let dec_trace = std::env::var_os("SOFUU_GPU_TRACE").is_some();
        let dec_fnorm_dbg = std::env::var_os("SOFUU_FNORM_DBG").is_some();

        let _flat_guard = if std::env::var_os("SOFUU_GEMV_FLAT_FORCE").is_some() {
            crate::infer::metal::MetalEngine::set_gemv_flat_forced(true);
            Some(crate::infer::metal::FlatGuard)
        } else {
            None
        };
        for layer_idx in start_layer..end_layer {
            // LEDGER #190: sibling GEMVs within a layer read the same input and write
            // disjoint outputs, so only the layer boundary needs a real barrier.
            g.engine.batch_barrier_mark();
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
            if !g.engine.batch_rmsnorm_f32w(
                &g.act_buf, g.slots.x,
                &g.w_buf, attn_norm_off,
                &g.act_buf, g.slots.xn,
                dim, norm_eps,
            ) {
                return false;
            }
            // q, k and v share x and cols and write disjoint slots, so they ride in
            // ONE dispatch: each dispatch in a batch pays a fixed ramp that is larger
            // than the streaming time of a decode-sized projection.
            let qkv = [&l.attn_q, &l.attn_k, &l.attn_v];
            let qkv_rows = [n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim];
            let qkv_dst = [g.slots.qa, g.slots.ka, g.slots.va];
            let qkv_fused = Self::fuse_qkv_enabled()
                && qkv.iter().all(|tv| tv.tensor_type == GGML_TYPE_Q4_K)
                && {
                    let offs: Vec<usize> = qkv.iter().map(|tv| gguf.data_offset + tv.offset).collect();
                    let base = *offs.iter().min().unwrap();
                    let fits = offs.iter().all(|o| (*o - base) <= u32::MAX as usize);
                    if !fits {
                        false
                    } else {
                        let secs: Vec<[u32; 3]> = offs.iter().zip(qkv_dst.iter().zip(qkv_rows.iter()))
                            .map(|(o, (dst, rows))| [(*o - base) as u32, *dst as u32, *rows as u32])
                            .collect();
                        g.engine.batch_gguf_gemv_q4k_secs(
                            &g.w_buf, base, &g.act_buf, g.slots.xn, &g.act_buf, dim, &secs,
                        )
                    }
                };
            if !qkv_fused {
                dec_gguf!(l.attn_q, gguf.data_offset + l.attn_q.offset,
                    g.slots.xn, g.slots.qa, n_heads * head_dim, dim);
                dec_gguf!(l.attn_k, gguf.data_offset + l.attn_k.offset,
                    g.slots.xn, g.slots.ka, n_kv_heads * head_dim, dim);
                dec_gguf!(l.attn_v, gguf.data_offset + l.attn_v.offset,
                    g.slots.xn, g.slots.va, n_kv_heads * head_dim, dim);
            }
            let run_attn = !g.skip_attn;
            if run_attn {
            if !g.engine.batch_llama_kv_write(
                &g.act_buf, g.slots.ka,
                &g.act_buf, g.slots.va,
                &g.k_buf, kc_layer,
                &g.v_buf, kc_layer,
                kln_buf, kln_off,
                has_k, pos as u32,
                head_dim as u32, n_kv_heads as u32, max_seq_len as u32,
                norm_eps, rope_freq_base, rope_mode,
            ) {
                return false;
            }
            // GPU3 chunked attention (default; the meta-overlap bug is fixed —
            // head-0 partials were overwriting the {max,sum} block).
            // SOFUU_GPU_ATTN=0 restores the pre-GPU3 serial kernel.
            // the serial kernel is 14% cheaper per token (9.1 vs 10.6 ms/token, GPU timer).
            let attn_new = dec_attn_env && pos >= dec_attn_min;
            if attn_new {
                // GPU3: q-norm + q-RoPE as their own dispatch (no intra-dispatch
                // race), then chunked attention partials + finish rescale.
                if !g.engine.batch_head_rmsnorm(
                    &g.act_buf, g.slots.qa,
                    q_norm_off.map(|_| &g.w_buf), q_norm_off.unwrap_or(0),
                    head_dim as u32, n_heads as u32,
                    has_q, pos as u32, norm_eps, rope_freq_base, rope_mode,
                ) {
                    return false;
                }
                if !g.engine.batch_llama_attn_part(
                    &g.act_buf, g.slots.qa,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    None, 0,
                    &g.meta_buf, layer_idx * meta_per_layer * 4,
                    0, pos as u32,
                    head_dim as u32, n_heads as u32, n_kv_heads as u32, max_seq_len as u32,
                    norm_eps, rope_freq_base, rope_mode,
                ) {
                    return false;
                }
                if !g.engine.batch_llama_attn_fin(
                    &g.meta_buf, layer_idx * meta_per_layer * 4,
                    &g.act_buf, g.slots.ao,
                    pos as u32, n_heads as u32, head_dim as u32,
                ) {
                    return false;
                }
            } else if crate::infer::metal::MetalEngine::attn_w4_enabled() {
                let (qln_buf, qln_off) =
                    (q_norm_off.map(|off| &g.w_buf), q_norm_off.unwrap_or(0));
                let warps = crate::infer::metal::MetalEngine::attn_warps();
                let attn_ok = if warps >= 16 {
                    g.engine.batch_llama_attn_w16(
                        &g.act_buf, g.slots.qa,
                        &g.k_buf, kc_layer,
                        &g.v_buf, kc_layer,
                        qln_buf, qln_off,
                        &g.act_buf, g.slots.ao,
                        has_q != 0, pos as u32,
                        head_dim as u32, n_heads as u32, n_kv_heads as u32, max_seq_len as u32,
                        norm_eps, rope_freq_base, rope_mode,
                    )
                } else if warps >= 8 {
                    g.engine.batch_llama_attn_w8(
                        &g.act_buf, g.slots.qa,
                        &g.k_buf, kc_layer,
                        &g.v_buf, kc_layer,
                        qln_buf, qln_off,
                        &g.act_buf, g.slots.ao,
                        has_q != 0, pos as u32,
                        head_dim as u32, n_heads as u32, n_kv_heads as u32, max_seq_len as u32,
                        norm_eps, rope_freq_base, rope_mode,
                    )
                } else {
                    g.engine.batch_llama_attn_w4(
                        &g.act_buf, g.slots.qa,
                        &g.k_buf, kc_layer,
                        &g.v_buf, kc_layer,
                        qln_buf, qln_off,
                        &g.act_buf, g.slots.ao,
                        has_q != 0, pos as u32,
                        head_dim as u32, n_heads as u32, n_kv_heads as u32, max_seq_len as u32,
                        norm_eps, rope_freq_base, rope_mode,
                    )
                };
                if !attn_ok {
                    return false;
                }
            } else if crate::infer::metal::MetalEngine::attn_fast_enabled() {
                let (qln_buf, qln_off) =
                    (q_norm_off.map(|off| &g.w_buf), q_norm_off.unwrap_or(0));
                if !g.engine.batch_llama_attn_fast(
                    &g.act_buf, g.slots.qa,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    qln_buf, qln_off,
                    &g.act_buf, g.slots.ao,
                    has_q != 0, pos as u32,
                    head_dim as u32, n_heads as u32, n_kv_heads as u32, max_seq_len as u32,
                    norm_eps, rope_freq_base, rope_mode,
                ) {
                    return false;
                }
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
            let trace = dec_trace;
            if trace && (layer_idx == 0 || layer_idx == 17 || layer_idx == 35) {
                let ao = g.slot_f32(g.slots.ao, n_heads * head_dim);
                let r: f32 = (ao.iter().map(|v| v * v).sum::<f32>() / (n_heads * head_dim) as f32)
                    .sqrt();
                eprintln!("[trace] L{layer_idx} pos{pos} attn_out_rms={r:.6}");
            }
            }
            dec_gguf!(l.attn_output, gguf.data_offset + l.attn_output.offset,
                g.slots.ao, g.slots.xo, dim, n_heads * head_dim);
            // residual add + norm in ONE dispatch (SOFUU_FUSE_NORM=0 reverts)
            if !(Self::fuse_norm_enabled()
                && g.engine.batch_add_rmsnorm_f32w(
                    &g.act_buf, g.slots.x,
                    &g.act_buf, g.slots.xo,
                    &g.act_buf, g.slots.x,
                    &g.act_buf, g.slots.xn,
                    &g.w_buf, ffn_norm_off,
                    dim, norm_eps,
                ))
            {
                if !g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, dim) {
                    return false;
                }
                if !g.engine.batch_rmsnorm_f32w(
                    &g.act_buf, g.slots.x,
                    &g.w_buf, ffn_norm_off,
                    &g.act_buf, g.slots.xn,
                    dim, norm_eps,
                ) {
                    return false;
                }
            }
            if layer_idx == 0 && dec_fnorm_dbg {
                let xs = g.slot_f32(g.slots.x, dim);
                let xn = g.slot_f32(g.slots.xn, dim);
                eprintln!("[fnorm] x[0..4]={:?} xn[0..4]={:?}", &xs[..4], &xn[..4]);
            }
            let run_ffn = !g.skip_ffn;
            if run_ffn {
            dec_gguf!(l.ffn_gate, gguf.data_offset + l.ffn_gate.offset,
                g.slots.xn, g.slots.gate, hidden_dim, dim);
            dec_gguf!(l.ffn_up, gguf.data_offset + l.ffn_up.offset,
                g.slots.xn, g.slots.up, hidden_dim, dim);
            if !g.engine.batch_swiglu(&g.act_buf, g.slots.gate, &g.act_buf, g.slots.up, hidden_dim) {
                return false;
            }
            dec_gguf!(l.ffn_down, gguf.data_offset + l.ffn_down.offset,
                g.slots.gate, g.slots.xo, dim, hidden_dim);
            if !g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, dim) {
                return false;
            }
            }
            if !token_commit && !g.engine.batch_end() {
                return false;
            }
        }
        let host_us = t_start.elapsed().as_micros() as u64;
        let gguf2 = &self.gguf;
        let g = self.gpu.as_mut().expect("one_commit without gpu");
        let head_on = token_commit && !g.skip_head;
        if head_on {
            // Final norm + lm_head in the SAME command buffer, then one drain.
            if !g.engine.batch_rmsnorm_f32w(
                &g.act_buf, g.slots.x,
                &g.w_buf, gguf2.data_offset + self.output_norm.offset,
                &g.act_buf, g.slots.xn,
                dim, self.config.norm_eps,
            ) {
                return false;
            }
            if !g.engine.batch_gguf(
                self.output.tensor_type,
                &g.w_buf, g.w_offset(gguf2, &self.output),
                &g.act_buf, g.slots.xn,
                &g.act_buf, g.slots.logits,
                self.config.vocab_size, dim,
            ) {
                return false;
            }
            if !g.engine.batch_end() {
                return false;
            }
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
        true
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
        // LEDGER #174 test: the last op of a chunk has no successor to sync it (#112), so the
        // caller may read logits with work still in flight. Force a wait at the end of the chunk.
        if std::env::var_os("SOFUU_CHAIN_WAIT").is_some() {
            if let Some(g) = self.gpu.as_mut() {
                g.engine.sync();
            }
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

        if std::env::var_os("SOFUU_CHAIN_TRACE").is_some() {
            eprintln!("[chain] n={} start={}", tokens.len(), start_pos);
        }        // OPT-IN (SOFUU_GPU_PREFILL=1): the b4 chunk pass is ~4x the SERIAL
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
        if b < 2 || b > 128 || start_pos + b > self.config.max_seq_len {
            return false;
        }
        // A rope-frequency table (llama3 scaling: factor/orig-ctx/low/high) is
        // applied by the CPU rope path and by nothing on the GPU side, so the
        // whole GPU chain pass would silently use plain rope for a scaled
        // model. The decode client already refuses this case; the prefill chain
        // did not, which made SOFUU_GPU_PREFILL=1 on a llama3-scaled model a
        // quiet wrong-answer switch rather than a slow one.
        if !self.rope_freq_adjust.is_empty() {
            if std::env::var_os("SOFUU_CHAIN_TRACE").is_some() {
                eprintln!("[chain] refused: rope frequency table (llama3 scaling) is CPU-only");
            }
            return false;
        }
        let t0 = std::time::Instant::now();
        let positions: Vec<usize> = (0..b).map(|t| start_pos + t).collect();
        // The core writes the last row's head logits into this slice (and then
        // mirrors them into self.logits). Passing &mut [] made every chunk that
        // needs logits panic with "range end index 151936 out of range for slice
        // of length 0", so SOFUU_GPU_PREFILL never completed a prompt.
        let mut head_logits = vec![0.0f32; if last_logits { self.config.vocab_size } else { 0 }];
        if !self.gpu_forward_chain_core(
            tokens,
            &positions,
            start_pos,
            if last_logits { Some(b - 1) } else { None },
            false,
            &mut head_logits,
        ) {
            return false;
        }
        // Probe the logits the chain actually produced (LEDGER #52: self.logits is
        // still zero at this point, so the earlier probe measured the wrong buffer).
        if last_logits {
            if let Some(p) = std::env::var_os("SOFUU_LOGITS_GAP_PROBE") {
                let mut idx: Vec<usize> = (0..head_logits.len()).collect();
                idx.sort_by(|a, b| {
                    head_logits[*b]
                        .partial_cmp(&head_logits[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let line: Vec<String> = idx
                    .iter()
                    .take(8)
                    .map(|i| format!("({:.6e},{})", head_logits[*i], i))
                    .collect();
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                    let _ = writeln!(f, "chain_core pos={} {}", start_pos + b - 1, line.join(" "));
                }
            }
        }
        // Bit-identity receipt for the fp16 weight arm (SOFUU_LOGITS_HASH=1):
        // FNV-1a over the whole f32 logit row the GPU chain produced. Two runs
        // that differ only in how the q4_K weights were staged must print the
        // same 64-bit hash.
        if last_logits {
            if std::env::var_os("SOFUU_LOGITS_HASH").is_some() {
                let mut h: u64 = 0xcbf29ce484222325;
                for v in head_logits.iter() {
                    for &byte in v.to_le_bytes().iter() {
                        h = (h ^ byte as u64).wrapping_mul(0x100000001b3);
                    }
                }
                eprintln!(
                    "[logits-hash] pos={} n={} fnv={h:016x} l0={:.9} l1={:.9}",
                    start_pos + b - 1,
                    head_logits.len(),
                    head_logits.first().copied().unwrap_or(0.0),
                    head_logits.get(1).copied().unwrap_or(0.0),
                );
            }
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

    /// W1 row split: the slab prologue shared by the whole-chunk chain pass
    /// and the CPU+GPU split driver - embedding lookup for the pass tokens plus
    /// the per-row position table. Slab row t is token t (position
    /// start_pos + t).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn chain_embed_posb(
        g: &mut GgufGpu,
        gguf: &GgufFile,
        embd: &TensorView,
        tokens: &[u32],
        start_pos: usize,
        dim: usize,
        b: usize,
    ) {
        // Embedding lookup stays on CPU by design (GPU-first directive:
        // embedding/sampler are CPU work; q-row staging is a host copy).
        for (t, &tok) in tokens.iter().enumerate() {
            let dst = g.slot_f32_mut(g.slots.x + t * dim * 4, dim);
            Self::lookup_embedding(gguf, embd, tok, dim, dst);
        }
        // Per-row positions for the b4 attention family (u32s bit-written
        // into the posb slot).
        {
            // BUG FIX (audit round 76): this wrote only 16 positions while a chain chunk runs
            // up to b = 128 rows, so rows 16..127 kept stale/zero positions - RoPE ran at
            // position 0 and the causal bound collapsed, making GPU chain prefill numerically
            // wrong (measured max |delta| 8.32 vs CPU on prefill_logits[..8]). The slot is
            // SPEC_B * 16 floats (model.rs:5220, SPEC_B = 128), so b contiguous u32s are well
            // inside it; the sibling writer at model.rs:3806 and the CPU path both use
            // contiguous layout.
            let pb = g.slot_f32_mut(g.slots.posb, b);
            let pbb =
                unsafe { std::slice::from_raw_parts_mut(pb.as_ptr() as *mut u32, b) };
            for (t, item) in pbb.iter_mut().enumerate().take(b) {
                *item = (start_pos + t) as u32;
            }
        }
    }

    /// W1 row split: output-norm the pass rows and run ONE lm_head GEMV for
    /// head_row. Shared by the whole-chunk chain pass and the split driver.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn chain_head_into(
        g: &mut GgufGpu,
        gguf: &GgufFile,
        output: &TensorView,
        output_norm: &TensorView,
        cfg: &ModelConfig,
        b: usize,
        head_row: Option<usize>,
    ) -> bool {
        let dim = cfg.dim;
        let norm_eps = cfg.norm_eps;
        let vocab_size = cfg.vocab_size;
        if head_row.is_none() {
            return true;
        }
        // Head: output-norm every row (cheap; feeds lazy per-row logits),
        // then ONE lm_head GEMV — row 0 (the root logits the tree contract
        // needs eagerly). The lm_head is ~514MB of q6_K for the 8B vocab:
        // committing it for all B rows streams 2GB per pass for rows verify
        // usually never samples; accepted rows get a lazy GPU head commit
        // (gpu_spec_head_row, ~13ms each) and CPU matvec as the fallback.
        if let Some(hr) = head_row {
            // Same batching as the ffn_norm above: one dispatch for the chunk.
            if !g.engine.batch_rmsnorm_f32w_b4(
                &g.act_buf, g.slots.x,
                &g.w_buf, gguf.data_offset + output_norm.offset,
                &g.act_buf, g.slots.xn,
                dim, norm_eps, b,
            ) {
                for t in 0..b {
                    if !g.engine.batch_rmsnorm_f32w(
                        &g.act_buf, g.slots.x + t * dim * 4,
                        &g.w_buf, gguf.data_offset + output_norm.offset,
                        &g.act_buf, g.slots.xn + t * dim * 4,
                        dim, norm_eps,
                    ) {
                        return false;
                    }
                }
            }
            if !g.engine.batch_gguf(
                output.tensor_type,
                &g.w_buf, g.w_offset(gguf, output),
                &g.act_buf, g.slots.xn + hr * dim * 4,
                &g.act_buf, g.slots.logits,
                vocab_size, dim,
            ) {
                return false;
            }
        }
        true
    }

    // ─── W1: CPU+GPU prefill ROW SPLIT (SOFUU_SPLIT=1, default OFF) ───────────
    //
    // Split each prefill chunk's ROWS between the two engines and run them in
    // the SAME wall-clock window: rows [0, kc) through the CPU tile path
    // (forward_batch_layer_rows) and rows [kc, b) through the GPU chain
    // (chain_layer_into), with the host encoding one GPU layer group per
    // command buffer and draining only at the end of the chunk.
    //
    // Why this shape and not "two independent passes per chunk":
    //   * The CPU arm owns the FIRST rows, so row i's attention covers
    //     positions <= start_pos + i - all of them written by the CPU arm
    //     itself (or by an earlier, already-drained chunk). It never reads the
    //     GPU arm's same-chunk K/V, so the CPU arm needs NO wait at all.
    //   * The GPU arm owns the LATER rows, whose attention does need the CPU
    //     arm's rows' K/V of the SAME layer. The per-layer order is therefore
    //     CPU layer L -> encode GPU layer L -> commit the group, i.e. a group is
    //     committed only after the CPU finished every layer in it. The CPU's
    //     next layer then runs on the host while the GPU executes the group
    //     just committed - that overlap is the whole split.
    //   * KV: both arms write the same f32 cache (g.k_buf/g.v_buf are the
    //     zero-copy wrappers of self.k_cache/self.v_cache). Each arm writes
    //     only its own rows, only in its own layer's region, so the concurrent
    //     host and GPU writes never share a cache line, and the end-of-chunk
    //     drain is the barrier that makes chunk t+1 see all of chunk t.
    //   * The chunk's LAST row is a GPU row, so the head GEMV (and therefore
    //     the numerics class of the returned logits) is the GPU route's.
    //
    // Every unsupported shape returns false and the caller falls through to a
    // whole-chunk route: a split is never the reason a chunk fails.

    /// Per-chunk row target for the GPU arm - FITTED, not guessed.
    ///
    /// Each arm costs a per-chunk FIXED part (both re-stream every weight once
    /// per chunk) plus a per-row slope, and the arms run concurrently, so a
    /// chunk costs max(CPU_FIX + kc*c_cpu, GPU_FIX + kg*c_gpu). Measured on
    /// 128-row chunks at start_pos=128 over interleaved rounds (M2 Pro):
    ///     CPU arm: 18.9ms at kc=4 .. 183.5ms at kc=124  -> c_cpu ~= 1.4ms/row
    ///     GPU arm: 65.0ms at kg=124, 68.3ms at kg=112   -> c_gpu ~= 0.3ms/row
    ///     whole-chunk GPU route, same b and start       = 64.5-67.7ms
    /// c_cpu is ~5x c_gpu, so the equalising row count
    ///     kc* = (GPU_FIX - CPU_FIX + c_gpu*b) / (c_cpu + c_gpu)
    /// has NO interior solution: every row handed to the CPU lengthens the
    /// critical path by c_cpu - c_gpu ~= 1.1ms. The fitted target is therefore
    /// the smallest CPU share the tile path can take - 4 rows, its b4
    /// granularity - and kg = b - 4. A full SOFUU_SPLIT_GPU_ROWS sweep agrees:
    /// on 128-row chunks kg=124 measured 1.036s of steady chunks against 1.60s
    /// at the old kc=40 (kg=88) formula, and the curve is monotonic in kg.
    /// Even at kg=124 the split only TIES the whole-chunk GPU route (1.036s vs
    /// 1.012s; GPU-only faster in 5 of 5 interleaved rounds), because a
    /// layer-group commit costs more than the 4 GPU rows it saves. That is why
    /// SOFUU_SPLIT stays opt-in and the GPU route is the fast arm.
    ///
    /// SOFUU_SPLIT_GPU_ROWS=<n> pins the absolute GPU row count;
    /// SOFUU_SPLIT_GPU_FRAC=<pct> pins the GPU share instead. Both are clamped so
    /// each arm keeps rows, and the CPU count is rounded down to a multiple of 4.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn split_gpu_row_target(b: usize) -> Option<usize> {
        static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        static FRAC: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let pinned = *ROWS.get_or_init(|| {
            std::env::var("SOFUU_SPLIT_GPU_ROWS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
        });
        let frac = *FRAC.get_or_init(|| {
            std::env::var("SOFUU_SPLIT_GPU_FRAC")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0 && *v < 100)
                .unwrap_or(0)
        });
        let want = if pinned > 0 {
            pinned
        } else if frac > 0 {
            (b * frac) / 100
        } else {
            // Fitted balance (see above): the CPU arm's marginal row is ~5x the
            // GPU arm's, so it keeps only the 4-row tile-granularity floor.
            b.saturating_sub(4).max(4)
        };
        let kg = want.clamp(4, b.saturating_sub(4).max(4));
        // The CPU arm's row count must stay a multiple of 4: the tile path
        // walks its rows in b4 kernels and its non-multiple-of-4 tail is
        // measured FOUR times slower (a 128-row chunk at kc=57 costs 460ms of
        // CPU arm against 120ms at kc=60), which would swamp the split. The
        // leftover rows go to the GPU arm, whose kernels are row-generic.
        let kc = (b - kg) & !3usize;
        let kg = b.saturating_sub(kc);
        if kc < 4 || kg < 4 {
            return None;
        }
        Some(kg)
    }

    /// Layers per split command buffer (SOFUU_SPLIT_GROUP, default 2). 1 is the
    /// finest interleave; larger groups trade GPU idle time for fewer commits
    /// (measured on 128-row chunks: G=2 < G=4 < G=7).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn split_layer_group() -> usize {
        static G: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *G.get_or_init(|| {
            std::env::var("SOFUU_SPLIT_GROUP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v >= 1)
                .unwrap_or(2)
        })
    }

    /// SOFUU_SPLIT=1 arms the row split (default OFF - the shipped
    /// configuration is exactly the whole-chunk routes).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn split_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("SOFUU_SPLIT")
                .map(|v| v == "1")
                .unwrap_or(false)
        })
    }

    /// Some(gpu_rows) when this chunk can be split; None means "use a whole-chunk
    /// route". The GPU side is the b4/chain client, whose only wide-chunk GEMV
    /// class is mma8 (the plain q4_K path is a FOUR-row kernel), so mma8 is part
    /// of the contract; the CPU arm's attention reads the f32 cache, so the f16
    /// KV shadows (decode-only on this target) must be off.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn prefill_split_gpu_rows(&self, b: usize) -> Option<usize> {
        if !Self::split_enabled() || b < 8 {
            return None;
        }
        let g = self.gpu.as_ref()?;
        if !g.token_commit || !crate::infer::metal::MetalEngine::mma8_enabled() {
            return None;
        }
        if self.k16_prefill || self.kv_f16 {
            return None;
        }
        if self.kivi.is_some() || self.streaming_weights || self.dynamic_ffn.enabled {
            return None;
        }
        if self.config.n_layers == 0 || b > 128 {
            return None;
        }
        Self::split_gpu_row_target(b)
    }

    /// W1: split one prefill chunk's rows across the CPU and the GPU. See the
    /// block comment above for the ordering argument. Returns false when the
    /// pass could not be run to completion (the caller then falls through).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[inline(never)]
    fn gpu_forward_prefill_split(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        gpu_rows: usize,
        want_logits: bool,
    ) -> bool {
        if start_pos + tokens.len() > self.config.max_seq_len {
            return false;
        }
        // The CPU arm holds &mut self (the tile path's batch buffers), so the
        // GPU client has to be an independent local for the duration.
        let mut gpu = match self.gpu.take() {
            Some(g) => g,
            None => return false,
        };
        let ok = self.prefill_split_inner(&mut gpu, tokens, start_pos, gpu_rows, want_logits);
        if !ok {
            // A rejected/aborted pass must not leave a command buffer in flight
            // for the next chunk to trip over.
            gpu.engine.batch_wait_all();
        }
        self.gpu = Some(gpu);
        ok
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[inline(never)]
    fn prefill_split_inner(
        &mut self,
        g: &mut GgufGpu,
        tokens: &[u32],
        start_pos: usize,
        gpu_rows: usize,
        want_logits: bool,
    ) -> bool {
        let b = tokens.len();
        if gpu_rows < 4 || gpu_rows > b.saturating_sub(4) {
            return false;
        }
        let kc = b - gpu_rows; // CPU rows [0, kc)
        let kg = gpu_rows; // GPU rows [kc, b), slab rows 0..kg
        let dim = self.config.dim;
        let n_layers = self.config.n_layers;
        let vocab_size = self.config.vocab_size;
        let gpu_start = start_pos + kc;
        let t_pass = std::time::Instant::now();

        // CPU arm: embeddings for its own rows (the GPU arm stages its rows in
        // the slab via the shared prologue below).
        for (i, &tok) in tokens[..kc].iter().enumerate() {
            let dst = &mut self.x_batch[i * dim..(i + 1) * dim];
            Self::lookup_embedding(&self.gguf, &self.token_embd, tok, dim, dst);
        }
        Self::chain_embed_posb(
            g,
            &self.gguf,
            &self.token_embd,
            &tokens[kc..],
            gpu_start,
            dim,
            kg,
        );

        let fuse_on = std::env::var("SOFUU_GPU_FUSE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let group = Self::split_layer_group().min(n_layers.max(1));
        let snap_layer: usize = std::env::var("SOFUU_QKV_SNAP_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut ffn_ops = 0usize;
        let mut cpu_us = 0u64;

        if !g.engine.batch_open() {
            return false;
        }
        for layer_idx in 0..n_layers {
            // ---- CPU arm: its rows, this layer. No wait: its attention never
            //      looks past its own last position.
            let t_cpu = std::time::Instant::now();
            self.forward_batch_layer_rows(layer_idx, kc, start_pos, snap_layer);
            cpu_us += t_cpu.elapsed().as_micros() as u64;
            // ---- GPU arm: its rows, same layer, into the OPEN command buffer.
            if !Self::chain_layer_into(
                g,
                &self.layers[layer_idx],
                &self.gguf,
                &self.config,
                layer_idx,
                kg,
                gpu_start,
                fuse_on,
                &mut ffn_ops,
            ) {
                return false;
            }
            let done = layer_idx + 1;
            if done % group == 0 || done == n_layers {
                // Commit only now: the CPU arm's K/V for EVERY layer in this
                // group is in the shared cache, so the group's attention may
                // read those rows. Then keep encoding while the GPU runs.
                if !g.engine.batch_commit_async() {
                    return false;
                }
                if done < n_layers && !g.engine.batch_open_async() {
                    return false;
                }
            }
        }

        if want_logits {
            if !g.engine.batch_open_async() {
                return false;
            }
            if !Self::chain_head_into(
                g,
                &self.gguf,
                &self.output,
                &self.output_norm,
                &self.config,
                kg,
                Some(kg - 1),
            ) {
                return false;
            }
            if !g.engine.batch_commit_async() {
                return false;
            }
        }
        // One drain per chunk: the queue is serial, so waiting here waits for
        // every group - the end-of-chunk barrier the KV handoff needs.
        if !g.engine.batch_wait_all() {
            return false;
        }
        let wall_us = t_pass.elapsed().as_micros() as u64;

        if want_logits {
            self.logits[0..vocab_size].copy_from_slice(g.slot_f32(g.slots.logits, vocab_size));
            // The slab xn rows are the GPU arm's chunk rows, not rows 0..kg of
            // the chunk, so the GPU-only route's xn copy does not apply here.
            self.spec_rows_valid = false;
        }
        if std::env::var_os("SOFUU_GPU_PROFILE").is_some() {
            eprintln!(
                "[gpu-profile] split-chunk: start={start_pos} b={b} cpu_rows={kc} gpu_rows={kg} wall={}us cpu_arm={}us groups={}",
                wall_us,
                cpu_us,
                (n_layers + group - 1) / group.max(1)
            );
        }
        true
    }

    /// Section blob for a group of q4_K projections that read the SAME activation
    /// rows (q/k/v off the normed input; gate/up off the ffn-normed one) so they
    /// can ride ONE sectioned mma8 dispatch instead of one dispatch each.
    /// Returns (w_base, secs) where secs[i] = {w_off - w_base, w_rem, rows,
    /// y_off bytes}, or None when the group cannot be expressed: fewer than two
    /// members (nothing to merge), more than the kernel's four sections, any
    /// member that is not q4_K, or an aligned delta that does not fit the u32
    /// the kernel reads (a >4 GiB weight buffer would truncate).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn mma8_secs_blob(
        gguf: &GgufFile,
        members: &[(&TensorView, u32, u32)],
    ) -> Option<(usize, Vec<[u32; 4]>)> {
        if members.len() < 2 || members.len() > 4 {
            return None;
        }
        if members.iter().any(|(tv, _, _)| tv.tensor_type != GGML_TYPE_Q4_K) {
            return None;
        }
        let mut base = usize::MAX;
        for (tv, _, _) in members {
            base = base.min((gguf.data_offset + tv.offset) & !255usize);
        }
        let mut secs = Vec::with_capacity(members.len());
        for (tv, rows, y_off) in members {
            let raw = gguf.data_offset + tv.offset;
            let aligned = raw & !255usize;
            if aligned < base || aligned - base > u32::MAX as usize {
                return None;
            }
            secs.push([(aligned - base) as u32, (raw & 255) as u32, *rows, *y_off]);
        }
        Some((base, secs))
    }

    // ─── W2: CPU+GPU FEATURE SPLIT (SOFUU_SPLIT=1 SOFUU_SPLIT_AXIS=feat) ─────
    //
    // The shipped split (W1) cuts the chunk's ROWS: each engine owns whole
    // activation rows, so both stream EVERY weight of every layer and the model
    // is read twice per chunk. This arm cuts the OUTPUT FEATURES instead: for
    // every projection each engine computes a contiguous range of W's rows for
    // ALL chunk rows, so the weight stream is divided - every weight byte is
    // read by exactly one engine. A projection's output is the next
    // projection's input, so the layer has to be walked in ROUNDS:
    //
    //   R1 attention  q/k/v (head split) + KV write + RoPE + attention
    //                   -> the full [b][q_dim] head_out        (exchange)
    //   R2 o proj     output-feature split -> a [b][dim] partial pair
    //                   -> folded into x                       (exchange)
    //   R3 ffn        ffn_norm + gate/up (feature split) + SwiGLU
    //                   -> the full [b][hidden] swiGLU rows     (exchange)
    //   R4 down proj  output-feature split -> a [b][dim] partial pair
    //                   -> folded into x                       (exchange)
    //
    // ORDERING: the GPU's round k is encoded and committed BEFORE the CPU
    // starts round k, so the CPU's share of a round runs underneath the GPU's
    // command buffer; the wait that ends the round is its exchange point (the
    // CPU cannot start round k+1 without the GPU's round-k output, the same
    // dependency the row split expresses with one commit per layer group). The
    // next buffer is opened only after that wait, so no round reads a
    // half-written neighbour.
    //
    // x has ONE writer per round: the GPU folds the two residual partials into
    // the slab with resid_split_add_f32, while the CPU folds the same two
    // operand blocks into its own [b][dim] view with the same f32 add - the two
    // copies agree bit for bit, so the CPU's KV rows and partials carry the
    // arithmetic GPU-only would have produced.
    //
    // Attention head split under GQA: the KV head is the awkward part (a q head
    // must share its group's KV head), so each engine owns WHOLE KV heads plus
    // the q heads of their groups. Its K/V slice binds at
    // kc_layer + kc*max_seq*head_dim*4 with the COMPACT head counts, and its q
    // block is compact with the attention output written at its own head offset
    // of the canonical head_out (the kernel already takes an output stride), so
    // every shipped attention kernel runs unchanged.
    //
    // o and ffn_down CANNOT be K-split: a column slice of a quantized row is
    // not a contiguous byte range, so those two are output-feature split and
    // their partials are folded - which is what forces R2 and R4.

    /// SOFUU_SPLIT_AXIS=feat selects the output-feature split. Unset (or
    /// "rows", or anything else) keeps the shipped row split, so an existing
    /// SOFUU_SPLIT=1 run is the same configuration it was before this arm.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_split_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("SOFUU_SPLIT_AXIS")
                .map(|v| v.trim().eq_ignore_ascii_case("feat"))
                .unwrap_or(false)
        })
    }

    /// Packed bytes of ONE weight row (mirror of the loader's private
    /// packed_row_bytes): the split addresses a row range by byte offset.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_row_bytes(type_id: u32, cols: usize) -> Option<usize> {
        match type_id {
            GGML_TYPE_Q4_0 if cols % 32 == 0 => Some((cols / 32) * 18),
            GGML_TYPE_Q8_0 if cols % 32 == 0 => Some((cols / 32) * 34),
            GGML_TYPE_Q4_K if cols % 256 == 0 => Some((cols / 256) * 144),
            GGML_TYPE_Q6_K if cols % 256 == 0 => Some((cols / 256) * 210),
            GGML_TYPE_F32 => Some(cols * 4),
            GGML_TYPE_F16 => Some(cols * 2),
            _ => None,
        }
    }

    /// One projection's CPU share: a 32-output-feature multiple, never all and
    /// never none (the point of the arm is that BOTH engines stream weights).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_part(total: usize, pct: usize) -> Option<usize> {
        if total < 64 {
            return None;
        }
        let mut r = total * pct.clamp(1, 99) / 100;
        r -= r % 32;
        r = r.clamp(32, total - 32);
        if r == 0 || r >= total || r % 32 != 0 {
            return None;
        }
        Some(r)
    }

    /// f32 view of a norm tensor (the CPU tile path's pattern).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_f32w<'a>(gguf: &'a GgufFile, tv: &TensorView, n: usize) -> &'a [f32] {
        let bytes = Self::tensor_slice(gguf, tv);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) }
    }

    /// Some(shares) when this chunk can run the feature split. Everything that
    /// could fail later is checked HERE, before a single dispatch, so an
    /// unsupported model falls through to the whole-chunk routes untouched.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn prefill_feat_shares(&self, b: usize) -> Option<FeatSplit> {
        if !Self::split_enabled() || !Self::feat_split_enabled() || b < 8 || b > 128 {
            return None;
        }
        let cfg = &self.config;
        if cfg.n_layers == 0 || cfg.n_layers != self.layers.len() {
            return None;
        }
        if self.k16_prefill
            || self.kv_f16
            || self.kivi.is_some()
            || self.streaming_weights
            || self.dynamic_ffn.enabled
        {
            return None;
        }
        {
            // The GPU arm is the chain client: its wide-chunk GEMV class is
            // mma8 (the plain q4_K route is a FOUR-row kernel), and the CPU arm
            // reads the f32 cache - the row split's contract verbatim.
            let g = self.gpu.as_ref()?;
            if !g.token_commit || !crate::infer::metal::MetalEngine::mma8_enabled() {
                return None;
            }
        }
        let (n_heads, n_kv_heads, hd, dim, hidden) = (
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.dim,
            cfg.hidden_dim,
        );
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 || hd < 4 || hd % 4 != 0 {
            return None;
        }
        let q_rows = n_heads * hd;
        // The GPU's GEMM class reads 256-column superblocks; the o projection's
        // K is q_rows and ffn_down's K is hidden, so all three must divide.
        if dim % 256 != 0 || hidden % 256 != 0 || q_rows % 256 != 0 {
            return None;
        }
        let group = n_heads / n_kv_heads;
        if group == 0 || n_heads < 2 * group {
            return None;
        }
        // Head split: whole KV groups only, at least one group on each side.
        let mut hc = std::env::var("SOFUU_FEAT_HEADS_CPU")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or((n_heads / 2 / group) * group);
        hc -= hc % group;
        if hc < group || hc > n_heads - group {
            hc = (n_heads / 2 / group) * group;
        }
        if hc == 0 || hc >= n_heads || hc % group != 0 {
            return None;
        }
        let pct = |k: &str, d: usize| -> usize {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(d)
        };
        let o_cpu = Self::feat_part(dim, pct("SOFUU_FEAT_O_CPU_PCT", 50))?;
        let f_cpu = Self::feat_part(hidden, pct("SOFUU_FEAT_FFN_CPU_PCT", 50))?;
        let d_cpu = Self::feat_part(dim, pct("SOFUU_FEAT_DOWN_CPU_PCT", 50))?;
        for l in self.layers.iter() {
            if l.attn_q_bias.is_some() || l.attn_k_bias.is_some() || l.attn_v_bias.is_some() {
                return None;
            }
            for (tv, cols) in [
                (&l.attn_q, dim),
                (&l.attn_k, dim),
                (&l.attn_v, dim),
                (&l.attn_output, q_rows),
                (&l.ffn_gate, dim),
                (&l.ffn_up, dim),
                (&l.ffn_down, hidden),
            ] {
                if Self::feat_row_bytes(tv.tensor_type, cols).is_none() {
                    return None;
                }
            }
        }
        Some(FeatSplit {
            hc,
            kc: hc / group,
            o_cpu,
            f_cpu,
            d_cpu,
        })
    }

    /// W2: y[0..rows) = W[row0..row0+rows) · x for all b rows, compact
    /// [b][rows] out. mma8 for q4_K (the chain's wide-chunk class), the
    /// sectioned tiled GEMM otherwise (q6_K). Both address the weights by byte
    /// offset, so a contiguous OUTPUT-FEATURE range is a base plus a row count.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(clippy::too_many_arguments)]
    fn feat_gemm_rows(
        g: &mut GgufGpu,
        gguf: &GgufFile,
        tv: &TensorView,
        row0: usize,
        rows: usize,
        cols: usize,
        b: usize,
        x_off: usize,
        y_off: usize,
    ) -> bool {
        if rows == 0 {
            return true;
        }
        let rb = match Self::feat_row_bytes(tv.tensor_type, cols) {
            Some(v) => v,
            None => return false,
        };
        let woff = gguf.data_offset + tv.offset + row0 * rb;
        if tv.tensor_type == GGML_TYPE_Q4_K
            && b > 4
            && crate::infer::metal::MetalEngine::mma8_enabled()
            && g.engine.batch_gguf_mma8(
                &g.w_buf,
                woff,
                &g.act_buf,
                x_off,
                &g.act_buf,
                y_off,
                rows,
                cols,
                b,
            )
        {
            return true;
        }
        let base = woff & !255usize;
        let rem = (woff - base) as u32;
        let secs = [[0u32, rem, rows as u32, 0u32, rows as u32]];
        g.engine.batch_gguf_gemm_tiled_multi_chain(
            tv.tensor_type,
            &g.w_buf,
            base,
            &g.act_buf,
            x_off,
            &g.act_buf,
            y_off,
            rows,
            cols,
            b,
            &secs,
        )
    }

    /// W2 R1 (GPU): its share of the attention block for all b slab rows -
    /// attn-norm, q/k/v over its head range, KV write + RoPE on its KV heads,
    /// q/k-norm + RoPE, attention into its head slice of the canonical
    /// head_out. fold = the previous round's residual partial pair to apply to
    /// x first (R1 of every layer but the first).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(clippy::too_many_arguments)]
    /// Warm the CPU arm's row range of a projection into the tiled-GEMM
    /// cache. That cache is keyed by (pointer, rows, cols), so a feature-split
    /// range is a different entry from the whole tensor the other arms use: the
    /// first use pays the repack (a per-row-group shuffle, bit-identical math),
    /// every later chunk hits it. Without it the CPU half runs the untiled
    /// kernels, measured ~3x slower per MAC on this tree.
    fn feat_warm(gguf: &GgufFile, tv: &TensorView, rows: usize, cols: usize) {
        if rows == 0 || cols == 0 {
            return;
        }
        let Some(rb) = Self::feat_row_bytes(tv.tensor_type, cols) else {
            return;
        };
        let bytes = Self::tensor_slice(gguf, tv);
        if rows * rb > bytes.len() {
            return;
        }
        super::quant::tile_warm(bytes.as_ptr(), rows, cols, tv.tensor_type);
    }

    /// One GQA-paired prefill-attention stage through the GPU chain.
    /// macOS/iOS only: the types in the signature (GgufGpu, FeatSplit) and
    /// the callees exist behind the same gate.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_gpu_attn(
        g: &mut GgufGpu,
        gguf: &GgufFile,
        l: &LayerWeights,
        cfg: &ModelConfig,
        layer_idx: usize,
        b: usize,
        sh: FeatSplit,
        fold: Option<(usize, usize)>,
    ) -> bool {
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let (n_heads, n_kv_heads) = (cfg.n_heads, cfg.n_kv_heads);
        let max_seq = cfg.max_seq_len;
        let q_rows = n_heads * hd;
        let hg = n_heads - sh.hc;
        let kgv = n_kv_heads - sh.kc;
        if hg == 0 || kgv == 0 {
            return false;
        }
        let rope_mode: u32 = if cfg.rope_interleaved { 1 } else { 0 };
        if let Some((n_a, n_b)) = fold {
            // FFI offsets are f32 ELEMENT counts; slot offsets are bytes.
            if !g.engine.batch_resid_split_add_f32(
                &g.act_buf,
                g.slots.x / 4,
                &g.act_buf,
                g.slots.pc / 4,
                n_a,
                &g.act_buf,
                g.slots.xo / 4,
                n_b,
                dim,
                b,
            ) {
                return false;
            }
        }
        if !g.engine.batch_rmsnorm_f32w_b4(
            &g.act_buf,
            g.slots.x,
            &g.w_buf,
            gguf.data_offset + l.attn_norm.offset,
            &g.act_buf,
            g.slots.xn,
            dim,
            cfg.norm_eps,
            b,
        ) {
            return false;
        }
        if !Self::feat_gemm_rows(g, gguf, &l.attn_q, sh.hc * hd, hg * hd, dim, b, g.slots.xn, g.slots.qa)
            || !Self::feat_gemm_rows(g, gguf, &l.attn_k, sh.kc * hd, kgv * hd, dim, b, g.slots.xn, g.slots.ka)
            || !Self::feat_gemm_rows(g, gguf, &l.attn_v, sh.kc * hd, kgv * hd, dim, b, g.slots.xn, g.slots.va)
        {
            return false;
        }
        let kv_head_bytes = n_kv_heads * max_seq * hd * 4;
        let kc_layer = layer_idx * kv_head_bytes + sh.kc * max_seq * hd * 4;
        let k_norm_off = l.attn_k_norm.as_ref().map(|tv| gguf.data_offset + tv.offset);
        let q_norm_off = l.attn_q_norm.as_ref().map(|tv| gguf.data_offset + tv.offset);
        let has_q = u32::from(q_norm_off.is_some());
        let has_k = u32::from(k_norm_off.is_some());
        let (kln_buf, kln_off) = (k_norm_off.map(|_| &g.w_buf), k_norm_off.unwrap_or(0));
        g.engine.batch_llama_kv_write_b4(
            &g.act_buf,
            g.slots.ka,
            &g.act_buf,
            g.slots.va,
            &g.k_buf,
            kc_layer,
            &g.v_buf,
            kc_layer,
            kln_buf,
            kln_off,
            has_k,
            &g.act_buf,
            g.slots.posb,
            b,
            hd as u32,
            kgv as u32,
            max_seq as u32,
            cfg.norm_eps,
            cfg.rope_freq_base,
            rope_mode,
            None,
        );
        g.engine.batch_head_rmsnorm_b4(
            &g.act_buf,
            g.slots.qa,
            q_norm_off.map(|_| &g.w_buf),
            q_norm_off.unwrap_or(0),
            &g.act_buf,
            g.slots.posb,
            b,
            hd as u32,
            hg as u32,
            has_q,
            cfg.norm_eps,
            cfg.rope_freq_base,
            rope_mode,
            None,
        );
        // Whole-row attention on the compact head count, landing its heads at
        // its own offset of the canonical [b][q_rows] block: a false return
        // means NOTHING was dispatched (pipeline missing, unsupported
        // geometry, SOFUU_PRE_ATTN=0 or a position ceiling), so the round is
        // aborted and the chunk falls through to a whole-chunk route.
        if !g.engine.batch_llama_attn_row_os(
            &g.act_buf,
            g.slots.qa,
            &g.k_buf,
            kc_layer,
            &g.v_buf,
            kc_layer,
            &g.act_buf,
            g.slots.ao + sh.hc * hd * 4,
            &g.act_buf,
            g.slots.posb,
            b,
            hd as u32,
            hg as u32,
            kgv as u32,
            max_seq as u32,
            q_rows as u32,
        ) {
            return false;
        }
        true
    }

    /// W2 R2 (GPU): its output-feature range of the o projection, off the
    /// canonical head_out rows, into its [b][dim] partial block.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_gpu_oproj(
        g: &mut GgufGpu,
        gguf: &GgufFile,
        l: &LayerWeights,
        cfg: &ModelConfig,
        b: usize,
        sh: FeatSplit,
    ) -> bool {
        let q_rows = cfg.n_heads * cfg.head_dim;
        Self::feat_gemm_rows(
            g,
            gguf,
            &l.attn_output,
            sh.o_cpu,
            cfg.dim - sh.o_cpu,
            q_rows,
            b,
            g.slots.ao,
            g.slots.xo,
        )
    }

    /// W2 R3 (GPU): fold the o halves into x, ffn-norm, gate/up over its
    /// output-feature range, SwiGLU, then place its compact range into the
    /// canonical [b][hidden] swiGLU rows.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_gpu_ffn(
        g: &mut GgufGpu,
        gguf: &GgufFile,
        l: &LayerWeights,
        cfg: &ModelConfig,
        b: usize,
        sh: FeatSplit,
    ) -> bool {
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let f_gpu = hidden - sh.f_cpu;
        if f_gpu == 0 {
            return false;
        }
        if !g.engine.batch_resid_split_add_f32(
            &g.act_buf,
            g.slots.x / 4,
            &g.act_buf,
            g.slots.pc / 4,
            sh.o_cpu,
            &g.act_buf,
            g.slots.xo / 4,
            dim - sh.o_cpu,
            dim,
            b,
        ) {
            return false;
        }
        if !g.engine.batch_rmsnorm_f32w_b4(
            &g.act_buf,
            g.slots.x,
            &g.w_buf,
            gguf.data_offset + l.ffn_norm.offset,
            &g.act_buf,
            g.slots.xn,
            dim,
            cfg.norm_eps,
            b,
        ) {
            return false;
        }
        if !Self::feat_gemm_rows(g, gguf, &l.ffn_gate, sh.f_cpu, f_gpu, dim, b, g.slots.xn, g.slots.gate)
            || !Self::feat_gemm_rows(g, gguf, &l.ffn_up, sh.f_cpu, f_gpu, dim, b, g.slots.xn, g.slots.up)
        {
            return false;
        }
        if !g.engine.batch_swiglu(&g.act_buf, g.slots.gate, &g.act_buf, g.slots.up, b * f_gpu) {
            return false;
        }
        // The CPU arm's half is already in place (it owns the prefix), so only
        // the GPU's suffix has to be moved; the destination stride is the
        // canonical row width, the source stride is its compact range.
        g.engine.batch_strided_copy_f32(
            &g.act_buf,
            g.slots.swg / 4 + sh.f_cpu,
            hidden,
            &g.act_buf,
            g.slots.gate / 4,
            f_gpu,
            f_gpu,
            b,
        )
    }

    /// W2 R4 (GPU): its output-feature range of ffn_down off the canonical
    /// swiGLU rows, into its [b][dim] partial block.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_gpu_down(
        g: &mut GgufGpu,
        gguf: &GgufFile,
        l: &LayerWeights,
        cfg: &ModelConfig,
        b: usize,
        sh: FeatSplit,
    ) -> bool {
        Self::feat_gemm_rows(
            g,
            gguf,
            &l.ffn_down,
            sh.d_cpu,
            cfg.dim - sh.d_cpu,
            cfg.hidden_dim,
            b,
            g.slots.swg,
            g.slots.xo,
        )
    }

    /// W2 R1 (CPU): its share of the attention block, off the CPU's folded x
    /// view (the slab x has a single writer - the GPU), writing its q heads
    /// into the canonical head_out at its own offset.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(clippy::too_many_arguments)]
    fn feat_cpu_attn(
        cfg: &ModelConfig,
        gguf: &GgufFile,
        l: &LayerWeights,
        k_cache: &mut [f32],
        v_cache: &mut [f32],
        x_src: &[f32],
        xn: &mut [f32],
        qb: &mut [f32],
        kb: &mut [f32],
        vb: &mut [f32],
        head_c: &mut [f32],
        ao: &mut [f32],
        b: usize,
        start_pos: usize,
        layer_idx: usize,
        sliding_window: usize,
        sink_tokens: usize,
        sh: FeatSplit,
    ) -> bool {
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let (n_heads, n_kv_heads) = (cfg.n_heads, cfg.n_kv_heads);
        let max_seq = cfg.max_seq_len;
        let q_d = sh.hc * hd;
        let kv_d = sh.kc * hd;
        let q_rows = n_heads * hd;
        let rope_mode: i32 = if cfg.rope_interleaved { 1 } else { 0 };
        if q_d == 0 || kv_d == 0 || dim == 0 || b == 0 {
            return false;
        }
        let wnorm = Self::feat_f32w(gguf, &l.attn_norm, dim);
        batched_rmsnorm(&mut xn[..b * dim], &x_src[..b * dim], wnorm, b, dim, cfg.norm_eps);
        let (q_rb, k_rb, v_rb) = match (
            Self::feat_row_bytes(l.attn_q.tensor_type, dim),
            Self::feat_row_bytes(l.attn_k.tensor_type, dim),
            Self::feat_row_bytes(l.attn_v.tensor_type, dim),
        ) {
            (Some(a), Some(c), Some(d)) => (a, c, d),
            _ => return false,
        };
        Self::feat_warm(gguf, &l.attn_q, q_d, dim);
        Self::feat_warm(gguf, &l.attn_k, kv_d, dim);
        Self::feat_warm(gguf, &l.attn_v, kv_d, dim);
        let q_bytes = Self::tensor_slice(gguf, &l.attn_q);
        matmat(
            &q_bytes[..q_d * q_rb],
            l.attn_q.tensor_type,
            q_d,
            dim,
            &xn[..b * dim],
            b,
            &mut qb[..b * q_d],
        );
        let k_bytes = Self::tensor_slice(gguf, &l.attn_k);
        matmat(
            &k_bytes[..kv_d * k_rb],
            l.attn_k.tensor_type,
            kv_d,
            dim,
            &xn[..b * dim],
            b,
            &mut kb[..b * kv_d],
        );
        let v_bytes = Self::tensor_slice(gguf, &l.attn_v);
        matmat(
            &v_bytes[..kv_d * v_rb],
            l.attn_v.tensor_type,
            kv_d,
            dim,
            &xn[..b * dim],
            b,
            &mut vb[..b * kv_d],
        );
        if let Some(ref qn) = l.attn_q_norm {
            let w = Self::feat_f32w(gguf, qn, hd);
            batched_rmsnorm_in_place(&mut qb[..b * q_d], w, b * sh.hc, hd, cfg.norm_eps);
        }
        if let Some(ref kn) = l.attn_k_norm {
            let w = Self::feat_f32w(gguf, kn, hd);
            batched_rmsnorm_in_place(&mut kb[..b * kv_d], w, b * sh.kc, hd, cfg.norm_eps);
        }
        // The cache slice starts at this layer with layer_idx = 0 and the CPU's
        // COMPACT head counts, so the shipped slot arithmetic
        // (kh*max_seq + pos)*head_dim lands on this layer's own first kc heads -
        // i.e. on exactly the heads this engine owns.
        let base = layer_idx
            .saturating_mul(n_kv_heads)
            .saturating_mul(max_seq)
            .saturating_mul(hd);
        let span = sh
            .kc
            .saturating_mul(max_seq)
            .saturating_mul(hd);
        if base + span > k_cache.len() || base + span > v_cache.len() {
            return false;
        }
        batched_rope_kv(
            &mut qb[..b * q_d],
            &mut kb[..b * kv_d],
            &vb[..b * kv_d],
            &mut k_cache[base..],
            &mut v_cache[base..],
            b,
            start_pos,
            0,
            sh.hc,
            sh.kc,
            hd,
            max_seq,
            cfg.rope_freq_base,
            rope_mode,
        );
        batched_attention_ext(
            &qb[..b * q_d],
            &k_cache[base..],
            &v_cache[base..],
            &mut head_c[..b * q_d],
            b,
            start_pos,
            0,
            sh.hc,
            sh.kc,
            hd,
            max_seq,
            sliding_window,
            sink_tokens,
        );
        // Its heads are a prefix of the q row, so the canonical block only
        // changes the row stride.
        for t in 0..b {
            ao[t * q_rows..t * q_rows + q_d].copy_from_slice(&head_c[t * q_d..(t + 1) * q_d]);
        }
        true
    }

    /// W2 R2 (CPU): its output-feature range of the o projection off the
    /// canonical head_out rows, into its [b][dim] partial block.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_cpu_oproj(
        gguf: &GgufFile,
        l: &LayerWeights,
        cfg: &ModelConfig,
        head_canon: &[f32],
        pc: &mut [f32],
        b: usize,
        sh: FeatSplit,
    ) -> bool {
        let cols = cfg.n_heads * cfg.head_dim;
        let rb = match Self::feat_row_bytes(l.attn_output.tensor_type, cols) {
            Some(v) => v,
            None => return false,
        };
        Self::feat_warm(gguf, &l.attn_output, sh.o_cpu, cols);
        let bytes = Self::tensor_slice(gguf, &l.attn_output);
        matmat(
            &bytes[..sh.o_cpu * rb],
            l.attn_output.tensor_type,
            sh.o_cpu,
            cols,
            head_canon,
            b,
            &mut pc[..b * sh.o_cpu],
        );
        true
    }

    /// W2 R3 (CPU): ffn_norm of its folded x view, gate/up over its
    /// output-feature range, SwiGLU, then its range written into the canonical
    /// row block (it owns the prefix, so only the row stride changes).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(clippy::too_many_arguments)]
    fn feat_cpu_ffn(
        gguf: &GgufFile,
        l: &LayerWeights,
        cfg: &ModelConfig,
        xn: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        swg: &mut [f32],
        b: usize,
        sh: FeatSplit,
    ) -> bool {
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let f = sh.f_cpu;
        if f == 0 {
            return false;
        }
        let (grb, urb) = match (
            Self::feat_row_bytes(l.ffn_gate.tensor_type, dim),
            Self::feat_row_bytes(l.ffn_up.tensor_type, dim),
        ) {
            (Some(a), Some(c)) => (a, c),
            _ => return false,
        };
        Self::feat_warm(gguf, &l.ffn_gate, f, dim);
        Self::feat_warm(gguf, &l.ffn_up, f, dim);
        let gb = Self::tensor_slice(gguf, &l.ffn_gate);
        let ub = Self::tensor_slice(gguf, &l.ffn_up);
        let mut done = false;
        if l.ffn_gate.tensor_type == l.ffn_up.tensor_type && grb == urb {
            // The tiled pair kernel only runs for geometries it has a tiled
            // copy of; a partial row range can miss that cache, in which case
            // it returns false with the outputs untouched. Fall through to the
            // plain GEMMs instead of publishing zeros.
            done = matmat_pair(
                &gb[..f * grb],
                &ub[..f * urb],
                l.ffn_gate.tensor_type,
                f,
                dim,
                &xn[..b * dim],
                b,
                &mut gate[..b * f],
                &mut up[..b * f],
            );
        }
        if !done {
            matmat(
                &gb[..f * grb],
                l.ffn_gate.tensor_type,
                f,
                dim,
                &xn[..b * dim],
                b,
                &mut gate[..b * f],
            );
            matmat(
                &ub[..f * urb],
                l.ffn_up.tensor_type,
                f,
                dim,
                &xn[..b * dim],
                b,
                &mut up[..b * f],
            );
        }
        batched_swiglu(&mut gate[..b * f], &up[..b * f], b, f);
        for t in 0..b {
            swg[t * hidden..t * hidden + f].copy_from_slice(&gate[t * f..(t + 1) * f]);
        }
        true
    }

    /// W2 R4 (CPU): its output-feature range of ffn_down off the canonical
    /// swiGLU rows, into its [b][dim] partial block.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn feat_cpu_down(
        gguf: &GgufFile,
        l: &LayerWeights,
        cfg: &ModelConfig,
        swg: &[f32],
        pc: &mut [f32],
        b: usize,
        sh: FeatSplit,
    ) -> bool {
        let hidden = cfg.hidden_dim;
        let rb = match Self::feat_row_bytes(l.ffn_down.tensor_type, hidden) {
            Some(v) => v,
            None => return false,
        };
        Self::feat_warm(gguf, &l.ffn_down, sh.d_cpu, hidden);
        let bytes = Self::tensor_slice(gguf, &l.ffn_down);
        matmat(
            &bytes[..sh.d_cpu * rb],
            l.ffn_down.tensor_type,
            sh.d_cpu,
            hidden,
            &swg[..b * hidden],
            b,
            &mut pc[..b * sh.d_cpu],
        );
        true
    }

    /// W2: the CPU's view of the residual state - x_slab + (j < n_a ? pa : pb).
    /// Same f32 add, same operand order as the GPU's resid_split_add_f32, so
    /// the two copies of x agree bit for bit without a copy-back.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(clippy::too_many_arguments)]
    fn feat_fold(
        dst: &mut [f32],
        slab: &[f32],
        x_off: usize,
        pa_off: usize,
        pb_off: usize,
        b: usize,
        cols: usize,
        n_a: usize,
        n_b: usize,
    ) {
        for t in 0..b {
            let xr = &slab[x_off + t * cols..x_off + (t + 1) * cols];
            let d = &mut dst[t * cols..(t + 1) * cols];
            if n_b == 0 {
                d.copy_from_slice(xr);
                continue;
            }
            let ar = &slab[pa_off + t * n_a..pa_off + (t + 1) * n_a];
            let br = &slab[pb_off + t * n_b..pb_off + (t + 1) * n_b];
            for j in 0..n_a {
                d[j] = xr[j] + ar[j];
            }
            for j in 0..n_b {
                d[n_a + j] = xr[n_a + j] + br[j];
            }
        }
    }

    /// W2: entry point from the chunk dispatcher (mirrors
    /// gpu_forward_prefill_split: the GPU client leaves self for the duration,
    /// so the CPU arm can hold its own batch buffers).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[inline(never)]
    fn gpu_forward_prefill_feat(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        sh: FeatSplit,
        want_logits: bool,
    ) -> bool {
        if start_pos + tokens.len() > self.config.max_seq_len {
            return false;
        }
        let mut gpu = match self.gpu.take() {
            Some(g) => g,
            None => return false,
        };
        let ok = self.prefill_feat_inner(&mut gpu, tokens, start_pos, sh, want_logits);
        if !ok {
            // A rejected round must not leave a command buffer in flight for
            // the next chunk (or the fall-through route) to trip over.
            gpu.engine.batch_wait_all();
        }
        self.gpu = Some(gpu);
        ok
    }

    /// W2: the round loop (see the block comment above for the ordering
    /// argument). Returns false when a round could not be encoded - the caller
    /// then falls through to a whole-chunk route, which rewrites the same K/V
    /// positions idempotently, so a partial pass is never observable.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[inline(never)]
    fn prefill_feat_inner(
        &mut self,
        g: &mut GgufGpu,
        tokens: &[u32],
        start_pos: usize,
        sh: FeatSplit,
        want_logits: bool,
    ) -> bool {
        let b = tokens.len();
        let dim = self.config.dim;
        let hidden = self.config.hidden_dim;
        let n_heads = self.config.n_heads;
        let hd = self.config.head_dim;
        let q_rows = n_heads * hd;
        let n_layers = self.config.n_layers;
        let vocab_size = self.config.vocab_size;
        let (hc, kc) = (sh.hc, sh.kc);
        let (o_cpu, f_cpu, d_cpu) = (sh.o_cpu, sh.f_cpu, sh.d_cpu);
        let (o_gpu, d_gpu) = (dim - o_cpu, dim - d_cpu);
        if b == 0 || b > 128 || q_rows == 0 || n_layers == 0 {
            return false;
        }
        // slab exchange blocks, in f32 ELEMENT units
        let ex = g.slots.x / 4;
        let exo = g.slots.xo / 4;
        let epc = g.slots.pc / 4;
        let eao = g.slots.ao / 4;
        let eswg = g.slots.swg / 4;
        let t_pass = std::time::Instant::now();
        let mut cpu_ns = [0u64; 5];
        let mut wait_ns = [0u64; 4];

        let dbg = std::env::var_os("SOFUU_GPU_PROFILE").is_some();
        if dbg {
        }
        Self::chain_embed_posb(g, &self.gguf, &self.token_embd, tokens, start_pos, dim, b);
        if !g.engine.batch_open() {
            return false;
        }
        // Which residual fold the CPU's x view still owes: 0 none, 1 the o
        // partials, 2 the ffn_down partials.
        let mut pend = 0u8;
        for layer_idx in 0..n_layers {
            let layer = &self.layers[layer_idx];
            let sw = self.layer_sliding_window(layer_idx);
            let sink = self.sink_tokens;
            let last = layer_idx + 1 == n_layers;

            // ── R1: attention (head split) ───────────────────────────────
            let fold1 = if pend == 2 { Some((d_cpu, d_gpu)) } else { None };
            if !Self::feat_gpu_attn(g, &self.gguf, layer, &self.config, layer_idx, b, sh, fold1) {
                return false;
            }
            if !g.engine.batch_commit_async() {
                return false;
            }
            let t0 = std::time::Instant::now();
            {
                let na = if pend == 2 { d_cpu } else { dim };
                let nb = dim - na;
                let slab: &mut [f32] = &mut g.act[..];
                Self::feat_fold(&mut self.x_batch[..b * dim], slab, ex, epc, exo, b, dim, na, nb);
                let ok = Self::feat_cpu_attn(
                    &self.config,
                    &self.gguf,
                    layer,
                    &mut self.k_cache,
                    &mut self.v_cache,
                    &self.x_batch[..b * dim],
                    &mut self.x_norm_batch[..b * dim],
                    &mut self.q_batch[..b * hc * hd],
                    &mut self.k_batch[..b * kc * hd],
                    &mut self.v_batch[..b * kc * hd],
                    &mut self.head_out_batch[..b * hc * hd],
                    // its heads land in the head_out slot, at its own column
                    // slice (the GPU writes the suffix of the same block)
                    &mut slab[eao..],
                    b,
                    start_pos,
                    layer_idx,
                    sw,
                    sink,
                    sh,
                );
                if !ok {
                    return false;
                }
            }
            cpu_ns[0] += t0.elapsed().as_nanos() as u64;
            let tw = std::time::Instant::now();
            let wok = g.engine.batch_wait_all();
            wait_ns[0] += tw.elapsed().as_nanos() as u64;
            if !wok {
                return false;
            }
            pend = 0;
            if !g.engine.batch_open_async() {
                return false;
            }

            // ── R2: o projection (output-feature split) ──────────────────
            if !Self::feat_gpu_oproj(g, &self.gguf, layer, &self.config, b, sh) {
                return false;
            }
            if !g.engine.batch_commit_async() {
                return false;
            }
            let t0 = std::time::Instant::now();
            {
                let slab: &mut [f32] = &mut g.act[..];
                let (left, right) = slab.split_at_mut(epc);
                let ok = Self::feat_cpu_oproj(
                    &self.gguf,
                    layer,
                    &self.config,
                    &left[eao..eao + b * q_rows],
                    &mut right[..b * o_cpu],
                    b,
                    sh,
                );
                if !ok {
                    return false;
                }
            }
            cpu_ns[1] += t0.elapsed().as_nanos() as u64;
            let tw = std::time::Instant::now();
            let wok = g.engine.batch_wait_all();
            wait_ns[1] += tw.elapsed().as_nanos() as u64;
            if !wok {
                return false;
            }
            pend = 1;
            if !g.engine.batch_open_async() {
                return false;
            }

            // ── R3: ffn gate/up + SwiGLU (output-feature split) ──────────
            if !Self::feat_gpu_ffn(g, &self.gguf, layer, &self.config, b, sh) {
                return false;
            }
            if !g.engine.batch_commit_async() {
                return false;
            }
            let t0 = std::time::Instant::now();
            {
                let slab: &mut [f32] = &mut g.act[..];
                Self::feat_fold(
                    &mut self.x_batch[..b * dim],
                    slab,
                    ex,
                    epc,
                    exo,
                    b,
                    dim,
                    o_cpu,
                    o_gpu,
                );
                let wnorm = Self::feat_f32w(&self.gguf, &layer.ffn_norm, dim);
                batched_rmsnorm(
                    &mut self.x_norm_batch[..b * dim],
                    &self.x_batch[..b * dim],
                    wnorm,
                    b,
                    dim,
                    self.config.norm_eps,
                );
                let (_, right) = slab.split_at_mut(eswg);
                let ok = Self::feat_cpu_ffn(
                    &self.gguf,
                    layer,
                    &self.config,
                    &self.x_norm_batch[..b * dim],
                    &mut self.ffn_gate_batch[..b * f_cpu],
                    &mut self.ffn_up_batch[..b * f_cpu],
                    &mut right[..b * hidden],
                    b,
                    sh,
                );
                if !ok {
                    return false;
                }
            }
            cpu_ns[2] += t0.elapsed().as_nanos() as u64;
            let tw = std::time::Instant::now();
            let wok = g.engine.batch_wait_all();
            wait_ns[2] += tw.elapsed().as_nanos() as u64;
            if !wok {
                return false;
            }
            pend = 0;
            if !g.engine.batch_open_async() {
                return false;
            }

            // ── R4: ffn_down (output-feature split) ─────────────────────
            if !Self::feat_gpu_down(g, &self.gguf, layer, &self.config, b, sh) {
                return false;
            }
            if !g.engine.batch_commit_async() {
                return false;
            }
            let t0 = std::time::Instant::now();
            {
                let slab: &mut [f32] = &mut g.act[..];
                let (left, right) = slab.split_at_mut(eswg);
                let ok = Self::feat_cpu_down(
                    &self.gguf,
                    layer,
                    &self.config,
                    &right[..b * hidden],
                    &mut left[epc..epc + b * d_cpu],
                    b,
                    sh,
                );
                if !ok {
                    return false;
                }
            }
            cpu_ns[3] += t0.elapsed().as_nanos() as u64;
            let tw = std::time::Instant::now();
            let wok = g.engine.batch_wait_all();
            wait_ns[3] += tw.elapsed().as_nanos() as u64;
            if !wok {
                return false;
            }
            pend = 2;
            if !last || want_logits {
                if !g.engine.batch_open_async() {
                    return false;
                }
            }
        }

        if want_logits {
            // The last layer's ffn_down partials are still unfolded; the head
            // is the final reader of x, so the fold rides in its buffer.
            if !g.engine.batch_resid_split_add_f32(
                &g.act_buf,
                g.slots.x / 4,
                &g.act_buf,
                g.slots.pc / 4,
                d_cpu,
                &g.act_buf,
                g.slots.xo / 4,
                d_gpu,
                dim,
                b,
            ) {
                return false;
            }
            if !Self::chain_head_into(
                g,
                &self.gguf,
                &self.output,
                &self.output_norm,
                &self.config,
                b,
                Some(b - 1),
            ) {
                return false;
            }
            if !g.engine.batch_commit_async() {
                return false;
            }
        }
        if !g.engine.batch_wait_all() {
            return false;
        }
        let wall_us = t_pass.elapsed().as_micros() as u64;
        if want_logits {
            self.logits[0..vocab_size].copy_from_slice(g.slot_f32(g.slots.logits, vocab_size));
            self.spec_rows_valid = false;
        }
        if std::env::var_os("SOFUU_GPU_PROFILE").is_some() {
            eprintln!(
                "[gpu-profile] feat-chunk: start={} b={} heads_cpu={}/{} o_cpu={}/{} f_cpu={}/{} d_cpu={}/{} wall={}us cpu_arm={}us r1={}us r2={}us r3={}us r4={}us wait={}us w1={}us w2={}us w3={}us w4={}us",
                start_pos,
                b,
                hc,
                n_heads,
                o_cpu,
                dim,
                f_cpu,
                hidden,
                d_cpu,
                dim,
                wall_us,
                cpu_ns.iter().sum::<u64>() / 1000,
                cpu_ns[0] / 1000,
                cpu_ns[1] / 1000,
                cpu_ns[2] / 1000,
                cpu_ns[3] / 1000,
                wait_ns.iter().sum::<u64>() / 1000,
                wait_ns[0] / 1000,
                wait_ns[1] / 1000,
                wait_ns[2] / 1000,
                wait_ns[3] / 1000,
            );
        }
        true
    }

    /// W1 CPU+GPU row split: ONE chain layer over b contiguous slab rows
    /// (slab row t = token t of this pass, its position start_pos + t). This is
    /// the body of gpu_forward_chain_core's layer loop moved verbatim into its
    /// own function, so the split driver can interleave it with the CPU arm and
    /// commit one layer group per command buffer AFTER the CPU arm has written
    /// its rows' K/V for the same layers. Returns false when a dispatch was
    /// rejected, exactly like the in-loop returns did.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(clippy::too_many_arguments)]
    fn chain_layer_into(
        g: &mut GgufGpu,
        l: &LayerWeights,
        gguf: &GgufFile,
        cfg: &ModelConfig,
        layer_idx: usize,
        b: usize,
        start_pos: usize,
        fuse_on: bool,
        ffn_ops: &mut usize,
    ) -> bool {
        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let hidden_dim = cfg.hidden_dim;
        let norm_eps = cfg.norm_eps;
        let rope_freq_base = cfg.rope_freq_base;
        let max_seq_len = cfg.max_seq_len;
        let rope_mode: u32 = if cfg.rope_interleaved { 1 } else { 0 };
        let q_rows = n_heads * head_dim;
        let kv_rows = n_kv_heads * head_dim;
    // GEMV of one tensor over the whole chain: batch_gguf_b4 when the
    // tensor is q4_K (the b4 kernel's token-major [4][cols]/[4][rows]
    // layout matches the x/xn/qa/gate... slot regions exactly), one
    // batch_gguf per row otherwise. Both paths use identical offsets.
    macro_rules! gemv_b {
        ($tv:expr, $x_off:expr, $y_off:expr, $rows:expr, $cols:expr) => {{
            // One dispatch attempt per arm, then chain_gemv_arm decides. The
            // decision is a pure function (and is tested as one) so "which
            // kernel carries a chunk" cannot drift back into a shape that
            // silently computes only part of it.
            let __ty = $tv.tensor_type;
            let __woff = gguf.data_offset + $tv.offset;
            let __mma8_ok = __ty == GGML_TYPE_Q4_K
                && crate::infer::metal::MetalEngine::mma8_enabled()
                && g.engine.batch_gguf_mma8(
                    &g.w_buf, __woff,
                    &g.act_buf, $x_off,
                    &g.act_buf, $y_off,
                    $rows, $cols, b,
                );
            let __q6_ok = !__mma8_ok
                && __ty == GGML_TYPE_Q6_K
                && crate::infer::metal::MetalEngine::mma8_enabled()
                && {
                    // The q6_K half of the chain's projections (attn_v on
                    // the i1 Q4_K_M quants; ffn_down below uses the same
                    // class with a resid epilogue) used to walk the per-row
                    // batch_gguf loop: one dispatch per row, each
                    // re-streaming the whole weight matrix - 128x traffic
                    // and 128 encoder round-trips per projection per
                    // 128-row chunk. The tiled fused GEMM is the same f32
                    // dequant (block constants first, no f16 staging) with
                    // the chunk's tokens as its second tile dimension, so
                    // one typed section over the whole grid is ONE
                    // dispatch. Sections are [base-relative off, 256-byte
                    // remainder, rows_end, y_off, y_stride]; padded slots
                    // carry rows_end 0 and are skipped by the host.
                    let mut blob = [[0u32; 5]; 4];
                    blob[0] = [0, (__woff & 255) as u32, $rows as u32, 0, $rows as u32];
                    g.engine.batch_gguf_gemm_tiled_multi_chain(
                        GGML_TYPE_Q6_K, &g.w_buf, __woff & !255usize,
                        &g.act_buf, $x_off,
                        &g.act_buf, $y_off,
                        $rows, $cols, b, &blob,
                    )
                };
            let __tiled_ok = !__mma8_ok && !__q6_ok && b > 4
                && matches!(__ty, 8u32 | 12u32 | 14u32)
                // SOFUU_CHAIN_NO_TILED=1 restores the per-row walk for the
                // whole chunk, which is the A/B arm for this route's
                // numeric receipt (same chunk width, different GEMM).
                && std::env::var_os("SOFUU_CHAIN_NO_TILED").is_none()
                && {
                    // WIDE chunk (b > 4) on a tensor class with no mma8
                    // section of its own: q8_0 always (the Llama-3.2-1B /
                    // SmolLM2 / K2-Horizon class is pure q8_0), and q4_K /
                    // q6_K when their own forms above refused. Before this
                    // arm the remaining routes were both wrong for a wide
                    // chunk: the b4 GEMV computes FOUR rows (124 of 128
                    // would stay stale) and the per-row loop costs b
                    // dispatches per projection, each re-streaming the whole
                    // matrix - measured 4x SLOWER than the CPU tile path on
                    // a Q8_0 1B (117 vs 532 tk/s), which made
                    // SOFUU_GPU_PREFILL=1 a regression rather than a route.
                    // This is the same typed tiled GEMM the q6_K arm above
                    // already ships (f32 dequant, token-major y, B-generic
                    // second tile dimension), so a whole projection is ONE
                    // dispatch. cols % 256 != 0 or a missing pipeline makes
                    // it refuse, and the per-row loop still owns the tensor.
                    let mut blob = [[0u32; 5]; 4];
                    blob[0] = [0, (__woff & 255) as u32, $rows as u32, 0, $rows as u32];
                    g.engine.batch_gguf_gemm_tiled_multi_chain(
                        __ty, &g.w_buf, __woff & !255usize,
                        &g.act_buf, $x_off,
                        &g.act_buf, $y_off,
                        $rows, $cols, b, &blob,
                    )
                };
            let __trace = std::env::var_os("SOFUU_MMA8_TRACE").is_some();
            match chain_gemv_arm(__ty, b, __mma8_ok, __q6_ok, __tiled_ok) {
                ChainGemvArm::Mma8 => {
                    // simdgroup-matrix GEMM (SOFUU_GEMM_MMA8=1)
                    if __trace {
                        eprintln!("[gemv] MMA8 rows={} cols={} b={}", $rows, $cols, b);
                    }
                }
                ChainGemvArm::Q6Tiled => {
                    if __trace {
                        eprintln!("[gemv] Q6TILED rows={} cols={} b={}", $rows, $cols, b);
                    }
                }
                ChainGemvArm::Tiled => {
                    if __trace {
                        eprintln!("[gemv] TILED rows={} cols={} b={}", $rows, $cols, b);
                    }
                }
                ChainGemvArm::B4 => {
                // batch_gguf_b4 is a FOUR-TOKEN kernel (no B argument): it
                // writes rows 0..3 of x/y and nothing else. It is only
                // reachable for a chunk the chain actually walks four rows
                // at a time (b <= 4); a wider chunk whose mma8 form refused
                // (cols % 256 != 0, missing pipeline) must take the
                // B-generic arms below or the per-row loop, never this one -
                // that mismatch is what kept two wide-chunk corruption bugs
                // alive.
                if __trace {
                    eprintln!("[gemv] B4   rows={} cols={} b={}", $rows, $cols, b);
                }
                g.engine.batch_gguf_b4(
                    &g.w_buf, __woff,
                    &g.act_buf, $x_off,
                    &g.act_buf, $y_off,
                    $rows, $cols,
                );
                }
                ChainGemvArm::PerRow => {
                    for t in 0..b {
                        g.engine.batch_gguf(
                            __ty, &g.w_buf, __woff,
                            // offsets are BYTES; slot sizes are element counts
                            &g.act_buf, $x_off + t * $cols * 4,
                            &g.act_buf, $y_off + t * $rows * 4,
                            $rows, $cols,
                        );
                    }
                }
            }
        }};
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
        // WIDE chunks (b > 4): q, k and v read the SAME normed rows and write
        // disjoint slots, so one sectioned mma8 dispatch replaces three - two
        // dispatch ramps and two barriers per layer less, with the same tile,
        // the same accumulation order and therefore the same bits in qa/ka/va.
        // Narrow chunks keep the b<=4 fused path (which also handles mixed
        // tensor types); a false return anywhere falls through to the
        // per-tensor walk, never to a skipped projection.
        // v is q6_K on the Q4_K_M layer class, so the three-way merge fires only
        // on the layers whose v is q4_K; the two-way q+k merge covers the rest
        // (q and k are q4_K in both classes), leaving v to its own dispatch as
        // before. Both are the SAME kernel with a shorter section list, so no
        // output bit changes either way.
        let qkv_secs = b > 4
            && crate::infer::metal::MetalEngine::mma8_enabled()
            && {
                let q3 = [
                    (&l.attn_q, q_rows as u32, g.slots.qa as u32),
                    (&l.attn_k, kv_rows as u32, g.slots.ka as u32),
                    (&l.attn_v, kv_rows as u32, g.slots.va as u32),
                ];
                let secs_run = |base: usize, secs: &[[u32; 4]]| {
                    g.engine.batch_gguf_mma8_secs(
                        &g.w_buf, base, &g.act_buf, g.slots.xn, &g.act_buf, dim, b, secs,
                    )
                };
                match Self::mma8_secs_blob(gguf, &q3) {
                    Some((base, secs)) => secs_run(base, &secs),
                    None => match Self::mma8_secs_blob(gguf, &q3[..2]) {
                        Some((base, secs)) => {
                            if secs_run(base, &secs) {
                                gemv_b!(l.attn_v, g.slots.xn, g.slots.va, kv_rows, dim);
                                true
                            } else {
                                false
                            }
                        }
                        None => false,
                    },
                }
            };
        let qkv_ok = qkv_secs
            || (b <= 4
                && gemm_fused_dec!(
                    [
                        (&l.attn_q, q_rows as u32, g.slots.qa, q_rows as u32),
                        (&l.attn_k, kv_rows as u32, g.slots.ka, kv_rows as u32),
                        (&l.attn_v, kv_rows as u32, g.slots.va, kv_rows as u32),
                    ],
                    q_rows + 2 * kv_rows, dim, g.slots.xn
                ));
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
                None,
            );
            g.engine.batch_head_rmsnorm_b4(
                &g.act_buf, g.slots.qa,
                q_norm_off.map(|_| &g.w_buf), q_norm_off.unwrap_or(0),
                &g.act_buf, g.slots.posb,
                b,
                head_dim as u32, n_heads as u32,
                has_q, norm_eps, rope_freq_base, rope_mode,
                None,
            );
            // Whole-row attention first (ONE dispatch for the whole chunk,
            // every row's full softmax): llama_attn_pre_row on the 128-row
            // chain — 32 rows share one K/V stream per threadgroup — and
            // llama_attn_pre_w32 as the A/B arm. It took 55% of the route
            // (3.18 s of 5.76 s at 1454 tokens) because the b4 family below
            // walks 4 rows per threadgroup over a grid of
            // rows*n_heads*n_chunks one-warp threadgroups. A false return
            // means nothing was dispatched (pipeline missing, unsupported
            // geometry, SOFUU_PRE_ATTN=0, or SOFUU_PREW_MAXPOS below this
            // chunk), so the b4 family still owns attention in that case.
            let row_attn = g.engine.batch_llama_attn_row(
                &g.act_buf, g.slots.qa,
                &g.k_buf, kc_layer,
                &g.v_buf, kc_layer,
                &g.act_buf, g.slots.ao,
                &g.act_buf, g.slots.posb,
                b,
                head_dim as u32, n_heads as u32, n_kv_heads as u32,
                max_seq_len as u32,
            );
            // pos < 128 (one softmax chunk): part+fin fuse into one
            // dispatch — one barrier drain instead of two
            let flashed = !row_attn && g.engine.batch_llama_attn_flash_b4(
                &g.act_buf, g.slots.qa,
                &g.k_buf, kc_layer,
                &g.v_buf, kc_layer,
                &g.act_buf, g.slots.ao,
                &g.act_buf, g.slots.posb,
                b,
                head_dim as u32, n_heads as u32, n_kv_heads as u32,
                max_seq_len as u32,
            );
            // `llama_attn_1c_b4` is WRONG for pos >= 128 (LEDGER #64/#65: it agrees with
            // the split pair only below one 128-position chunk). Gate it to the range it
            // is actually valid for rather than letting it corrupt long prompts.
            let one_chunk_pos = attn_1c_valid(start_pos, b);
            // row_attn already wrote slots.ao: nothing below may run for
            // this layer, or it would overwrite a finished chunk.
            let fused = row_attn || flashed || (fuse_on
                && one_chunk_pos
                && g.engine.batch_llama_attn_1c_b4(
                    &g.act_buf, g.slots.qa,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    &g.act_buf, g.slots.ao,
                    &g.act_buf, g.slots.posb,
                    b,
                    head_dim as u32, n_heads as u32, n_kv_heads as u32,
                    max_seq_len as u32,
                    None,
                ));
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

        // o projection + resid-add fused (rows == dim == acc width).
        // gemv_q4_k_b4c2r is a FOUR-TOKEN kernel (no B argument): on the
        // wide-chain route (mma8, chunk 128) it would compute rows 0..3 and
        // leave the other 124 rows of xo stale. Only the narrow b4 walk can
        // use it; wider chunks fall through to gemv_b!, which is batched.
        // Wide chunks: the mma8 residual epilogue writes x directly, so the
        // separate batch_add dispatch and the xo round trip are gone. Same f32
        // add of the same two operands, so x is bit-identical.
        let o_acc = b > 4
            && crate::infer::metal::MetalEngine::mma8_enabled()
            && l.attn_output.tensor_type == GGML_TYPE_Q4_K
            && g.engine.batch_gguf_mma8_acc(
                &g.w_buf, gguf.data_offset + l.attn_output.offset,
                &g.act_buf, g.slots.ao,
                &g.act_buf, g.slots.x,
                dim, q_rows, b,
            );
        if !o_acc
            && !(fuse_on
                && b <= 4
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
            if !g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, b * dim) {
                return false;
            }
        }
        // ONE dispatch for the whole chunk's ffn_norm, not b of them: the
        // b-row kernel is one 32-thread threadgroup per row off a single
        // grid, so 128 per-row dispatches (+128 memoryBarrier-ordered
        // encoder steps) collapse into one with identical math. That loop
        // was ~1.27 s of the 5.76 s route. Falls back to the per-row
        // dispatch only when the pipeline/rows ceiling makes the batched
        // form unavailable — never silently skips the norm.
        if !g.engine.batch_rmsnorm_f32w_b4(
            &g.act_buf, g.slots.x,
            &g.w_buf, ffn_norm_off,
            &g.act_buf, g.slots.xn,
            dim, norm_eps, b,
        ) {
            for t in 0..b {
                if !g.engine.batch_rmsnorm_f32w(
                    &g.act_buf, g.slots.x + t * dim * 4,
                    &g.w_buf, ffn_norm_off,
                    &g.act_buf, g.slots.xn + t * dim * 4,
                    dim, norm_eps,
                ) {
                    return false;
                }
            }
        }
        if !g.skip_ffn {
        *ffn_ops += 1;
        // gate/up share the ffn-normed rows: ONE sectioned mma8 dispatch on the
        // wide chunks (same argument as q/k/v above), b<=4 fused path otherwise.
        // (A two-GEMM-per-threadgroup variant that also applied the SwiGLU in
        // the epilogue was measured: it halves the in-flight threadgroup count
        // and cost +16 ms per 128-row chunk, so the three dispatches stay.)
        let gu_secs = b > 4
            && crate::infer::metal::MetalEngine::mma8_enabled()
            && Self::mma8_secs_blob(
                gguf,
                &[
                    (&l.ffn_gate, hidden_dim as u32, g.slots.gate as u32),
                    (&l.ffn_up, hidden_dim as u32, g.slots.up as u32),
                ],
            )
            .is_some_and(|(base, secs)| {
                g.engine.batch_gguf_mma8_secs(
                    &g.w_buf, base, &g.act_buf, g.slots.xn, &g.act_buf, dim, b, &secs,
                )
            });
        let gu_ok = gu_secs
            || (b <= 4
                && gemm_fused_dec!(
                    [
                        (&l.ffn_gate, hidden_dim as u32, g.slots.gate, hidden_dim as u32),
                        (&l.ffn_up, hidden_dim as u32, g.slots.up, hidden_dim as u32),
                    ],
                    2 * hidden_dim, dim, g.slots.xn
                ));
        if !gu_ok {
            gemv_b!(l.ffn_gate, g.slots.xn, g.slots.gate, hidden_dim, dim);
            gemv_b!(l.ffn_up, g.slots.xn, g.slots.up, hidden_dim, dim);
        }
        if !g.skip_ffn {
            *ffn_ops += 3;
        }
        if !g.engine.batch_swiglu(&g.act_buf, g.slots.gate, &g.act_buf, g.slots.up, b * hidden_dim) {
            return false;
        }
        // down projection + resid-add fused
        let down_fused = (fuse_on
            && b <= 4 // gemv_q4_k_b4c2r is 4 tokens wide (see o above)
            && l.ffn_down.tensor_type == GGML_TYPE_Q4_K
            && g.engine.batch_gguf_b4r(
                &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                &g.act_buf, g.slots.gate,
                &g.act_buf, g.slots.xo,
                &g.act_buf, g.slots.x,
                dim, hidden_dim, dim,
            ))
            // q6_K down (the i1 Q4_K_M class — half the layers): the tiled
            // fused kernel's resid epilogue is the same f32 dequant as
            // above with the residual add folded in, so the 128 per-row
            // dispatches AND the separate batch_add collapse into ONE
            // dispatch. Every output element is read and written by the
            // same thread, so binding y and res to the same slot region is
            // not a race.
            || (l.ffn_down.tensor_type == GGML_TYPE_Q6_K
                && crate::infer::metal::MetalEngine::mma8_enabled()
                && g.engine.batch_gguf_gemm_tiled_acc_chain(
                    GGML_TYPE_Q6_K,
                    &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                    &g.act_buf, g.slots.gate,
                    &g.act_buf, g.slots.x, dim,
                    &g.act_buf, g.slots.x,
                    dim, hidden_dim, b,
                ));
        // Wide chunks whose down is q4_K: same residual epilogue as o above.
        let down_acc = !down_fused
            && b > 4
            && crate::infer::metal::MetalEngine::mma8_enabled()
            && l.ffn_down.tensor_type == GGML_TYPE_Q4_K
            && g.engine.batch_gguf_mma8_acc(
                &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                &g.act_buf, g.slots.gate,
                &g.act_buf, g.slots.x,
                dim, hidden_dim, b,
            );
        if !down_fused && !down_acc {
            gemv_b!(l.ffn_down, g.slots.gate, g.slots.xo, dim, hidden_dim);
            if !g.engine.batch_add(&g.act_buf, g.slots.x, &g.act_buf, g.slots.xo, b * dim) {
                return false;
            }
        }
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
        if std::env::var_os("SOFUU_CHAIN_TRACE").is_some() {
            eprintln!("[chain] core token_commit={} b={} start={}", g.token_commit, b, start_pos);
        }
        if b < 2 || b > 128 || positions.len() != b || start_pos + b > self.config.max_seq_len {
            return false;
        }
        // LEDGER #118 (fixed in #153): stale-read test. Zero every activation slot region, bounded by
        // the NEXT slot's offset so the fill can never run past the allocation. Previously placed inside
        // the shape guard above, where it could only run on chunks that were already invalid.
        if std::env::var_os("SOFUU_ZERO_ACT").is_some() {
            let mut offs = vec![
                g.slots.x, g.slots.xn, g.slots.xo, g.slots.qa,
                g.slots.ka, g.slots.va, g.slots.ao, g.slots.gate, g.slots.up, g.slots.posb,
            ];
            offs.sort_unstable();
            offs.dedup();
            for w in offs.windows(2) {
                let (a, c) = (w[0], w[1]);
                if c > a {
                    let n = (c - a) / 4;
                    g.slot_f32_mut(a, n).fill(0.0);
                }
            }
        }
        for (t, &p) in positions.iter().enumerate() {
            if p != start_pos + t {
                return false;
            }
        }

        let dim = self.config.dim;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let max_seq_len = self.config.max_seq_len;
        let kv_rows = n_kv_heads * head_dim;
        let vocab_size = self.config.vocab_size;
        let pass_t = std::time::Instant::now();
        let mut ffn_ops = 0usize;
        let fuse_on = std::env::var("SOFUU_GPU_FUSE")
            .map(|v| v != "0")
            .unwrap_or(true);

        Self::chain_embed_posb(g, &self.gguf, &self.token_embd, tokens, start_pos, dim, b);
        // Bound above the macro so macro hygiene can see it (the macro body
        // expands it at def-site scope); a field borrow is disjoint from
        // self.gpu, so holding both across the pass is fine.
        let gguf = &self.gguf;

        // SOFUU_LAYER_SNAP: split the chunk into one command buffer PER
        // LAYER and dump xn/x after each — the L-granularity divergence pin
        // against the CPU tiles' SOFUU_LAYER_SNAP (the slab is unreadable
        // mid-buffer otherwise).
        assert!(g.engine.batch_open());
        for layer_idx in 0..self.config.n_layers {
            if !Self::chain_layer_into(
                g,
                &self.layers[layer_idx],
                gguf,
                &self.config,
                layer_idx,
                b,
                start_pos,
                fuse_on,
                &mut ffn_ops,
            ) {
                return false;
            }
        }

        if !Self::chain_head_into(
            g,
            gguf,
            &self.output,
            &self.output_norm,
            &self.config,
            b,
            head_row,
        ) {
            return false;
        }
        if !g.engine.batch_end() {
            return false;
        }
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
        // The chunk cap the slab was sized for (parsed once at arm()).
        let cap = self.gpu.as_ref().map(|g| g.pre_chunk).unwrap_or(0);
        if !(1..=cap).contains(&b) || start_pos + b > self.config.max_seq_len {
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
        if std::env::var_os("SOFUU_SLAB_DUMP").is_some() {
            let n = 8.min(self.x.len());
            eprintln!("[slab] x_cpu[0..{n}] = {:?}", &self.x[..n]);
            eprintln!("[slab] slab[0..{n}]  = {:?} (off={})", &g.pref_f32(pslots.x, n), pslots.x);
        }
        // LEDGER #176: the act-slot zeroing lived inside gpu_forward_chain_core, which these
        // wide-tile routes never call - so every earlier SOFUU_ZERO_ACT negative was meaningless
        // here. Zero the act_buf slots HERE, next to the slab, and cover all 12 fields
        // (including logits and xq8e, which the previous list missed).
        if std::env::var_os("SOFUU_ZERO_ACT").is_some() {
            let mut offs = vec![
                g.slots.x, g.slots.posb, g.slots.xn, g.slots.qa, g.slots.ka, g.slots.va,
                g.slots.ao, g.slots.xo, g.slots.gate, g.slots.up, g.slots.logits, g.slots.xq8e,
            ];
            offs.sort_unstable();
            offs.dedup();
            for w in offs.windows(2) {
                if w[1] > w[0] {
                    let n = (w[1] - w[0]) / 4;
                    g.slot_f32_mut(w[0], n).fill(0.0);
                }
            }
        }
        // LEDGER #173 test: the MMA/GEMM slab is a DIFFERENT buffer from act_buf, so zeroing
        // g.slots.* (SOFUU_ZERO_ACT) can never affect it. Zero the slab regions themselves,
        // bounded by the next slot offset exactly like the act version.
        if std::env::var_os("SOFUU_ZERO_SLAB").is_some() {
            let mut offs = vec![
                pslots.x, pslots.posb, pslots.xn, pslots.qa, pslots.ka,
                pslots.va, pslots.ao, pslots.xo, pslots.gate, pslots.up, pslots.rope,
            ];
            offs.sort_unstable();
            offs.dedup();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for w in offs.windows(2) {
                if w[1] > w[0] {
                    spans.push((w[0], (w[1] - w[0]) / 4));
                }
            }
            for (a, n) in spans {
                if n > 0 {
                    g.pref_f32_mut(a, n).fill(0.0);
                }
            }
        }
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
            if let Some(prefix) = std::env::var_os("SOFUU_LAYER_SNAP") {
                let prefix = prefix.to_string_lossy().to_string();
                let esl: Vec<u8> = ph_buf[pslots.x / 4..pslots.x / 4 + b * dim]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let _ = std::fs::write(format!("{prefix}.emb"), &esl);
            }
            // rope table: EXACT mirror of sofuu_rope_fill_rows —
            // freqs[j] = 1/powf(base, 2j/hd) (f32 powf), theta = pos × freq,
            // layout [row] = cos(hd/2) | sin(hd/2). The kernels take this
            // instead of on-device pow so the routes share one value source.
            let half = head_dim / 2;
            let mut freqs = vec![0f32; half];
            for (j, f) in freqs.iter_mut().enumerate() {
                *f = 1.0f32 / rope_freq_base.powf((2 * j) as f32 / head_dim as f32);
            }
            let dst = &mut ph_buf[pslots.rope / 4..pslots.rope / 4 + b * head_dim];
            for t in 0..b {
                let pos_f = (start_pos + t) as f32;
                for j in 0..half {
                    let theta = pos_f * freqs[j];
                    dst[t * head_dim + j] = theta.cos();
                    dst[t * head_dim + half + j] = theta.sin();
                }
            }
        }

        let pass_t = std::time::Instant::now();
        // Phase 3 q8_0/q6_K GEMM kernels exist and are BIT-EXACT standalone
        // (gemm_q8_0/gemm_q6_k oracles, worst_rel=0 at B=5/32/128), but the
        // all-q8_0 1B produced silently-wrong values in-model at 128-row
        // passes (garbage logits → the CPU repack's nearest_int assert on
        // downstream activation magnitudes; deterministic per prompt,
        // B<=32 passes clean, skip=attn is clean) — the defect is booked,
        // UNTIL it is pinned the extra dtypes are env opt-in only:
        // SOFUU_GEMM_TYPES=all. q4_K (the receipted state since Phase 1)
        // stays the default gate for every model. Declared before the gemm!
        // macro below (macro_rules hygiene: the body resolves in the
        // definition context).
        // q6_K and q8_0 now share the batched route by default: leaving them
        // out put every o and down projection (q6_K in Q4_K_M, q8_0's everywhere)
        // through the single-token GEMV once per token per layer, which the
        // SOFUU_GGUF_DBG histogram shows as one call per layer per token. On
        // the 8B that is 9.6 s -> 3.8 s for a 123-token prefill (2.52x).
        // SOFUU_GEMM_TYPES=q4k restores the old q4_K-only gate.
        let gemm_types_all = std::env::var("SOFUU_GEMM_TYPES")
            .map(|v| v.trim() != "q4k")
            .unwrap_or(true);
        // SOFUU_QKV_SNAP_LAYER: which layer the QKV stage bisect snapshots
        // (default 0; set 1+ to localize mid-network divergence).
        let snap_layer: usize = std::env::var("SOFUU_QKV_SNAP_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        // 8x4-tile MMQ mirror: route the q4_K batch gemms through the
        // act-quantized kernels that bit-mirror the CPU tiles' integer/f32
        // order (default pair route parity; see PLAN-LLAMA-CPP-PARITY).
        let tile_mmq = std::env::var("SOFUU_GPU_TILE_MMQ")
            .map(|v| v == "1")
            .unwrap_or(false);

        macro_rules! gemm {
            ($tv:expr, $x_off:expr, $y_off:expr, $rows:expr, $cols:expr) => {{
                // Phase 3: the real-GEMM class dispatches per TYPE (8/12/14);
                // q6_K/q8_0 ride it only under SOFUU_GEMM_TYPES=all (the 1B
                // in-model defect above), other dtypes take the per-row loop.
                let mmq_quant = tile_mmq
                    && matches!($tv.tensor_type, GGML_TYPE_Q4_K | GGML_TYPE_Q6_K)
                    && g.engine.batch_quantize_q8_k(
                        pbuf, $x_off, pbuf, pslots.xq8, $cols, b,
                    );
                let mmq_fast = mmq_quant
                    && if $tv.tensor_type == GGML_TYPE_Q4_K {
                        g.engine.batch_gemm_q4_k_mmq(
                            &g.w_buf, gguf.data_offset + $tv.offset,
                            pbuf, pslots.xq8,
                            pbuf, $y_off,
                            $rows, $cols, b,
                        )
                    } else {
                        g.engine.batch_gemm_q6_k_mmq(
                            &g.w_buf, gguf.data_offset + $tv.offset,
                            pbuf, pslots.xq8,
                            pbuf, $y_off,
                            $rows, $cols, b,
                        )
                    };
                let fast = mmq_fast
                    || (($tv.tensor_type == GGML_TYPE_Q4_K
                        || (gemm_types_all
                            && matches!($tv.tensor_type, GGML_TYPE_Q4_K | GGML_TYPE_Q6_K | GGML_TYPE_Q8_0)))
                        && g.engine.batch_gguf_gemm(
                            $tv.tensor_type, &g.w_buf, gguf.data_offset + $tv.offset,
                            pbuf, $x_off, pbuf, $y_off,
                            $rows, $cols, b,
                        ));
                if !fast {
                    for t in 0..b {
                        if !g.engine.batch_gguf(
                            $tv.tensor_type, &g.w_buf, gguf.data_offset + $tv.offset,
                            pbuf, $x_off + t * $cols * 4,
                            pbuf, $y_off + t * $rows * 4,
                            $rows, $cols,
                        ) {
                            return false;
                        }
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
                // Phase 3 (mixed-type qkv): type-GROUPED fused GEMM —
                // consecutive same-type sections ride ONE multi-gemm
                // dispatch (uniform qkv/gate_up = 1 dispatch; mixed
                // [q4,q4,q6] = 2). base = MIN aligned offset across the
                // GROUP (the b4blob receipt: sections are not monotonic in
                // file order — 1B attn_k < attn_q).
                let mut fast = !$secs.is_empty();
                let mut groups: Vec<(u32, Vec<usize>)> = Vec::new();
                for (i, s) in $secs.iter().enumerate().take(4) {
                    let ty = s.0.tensor_type;
                    let eligible = ty == GGML_TYPE_Q4_K
                        || (gemm_types_all
                            && matches!(ty, GGML_TYPE_Q4_K | GGML_TYPE_Q6_K | GGML_TYPE_Q8_0));
                    if !eligible {
                        fast = false;
                        break;
                    }
                    match groups.last_mut() {
                        Some(g) if g.0 == ty => g.1.push(i),
                        _ => groups.push((ty, vec![i])),
                    }
                }
                if fast && multi_dbg && !multi_dbg_done.replace(true) {
                    eprintln!(
                        "[multi-debug] type-groups={} types={:?} rows_total={} cols={} b={}",
                        groups.len(),
                        groups.iter().map(|(ty, _)| ty).collect::<Vec<_>>(),
                        $rows_total, $cols, b
                    );
                }
                if fast {
                    fast = groups.iter().all(|(ty, idxs)| {
                        let mut blob: [[u32; 5]; 4] = [[0u32; 5]; 4];
                        // zero rows_end on the padded slots can never match
                        // a row span (end < prev_end) — same rule as the
                        // host-padded decoder tables
                        let base = idxs.iter()
                            .map(|&i| (gguf.data_offset + $secs[i].0.offset) & !255usize)
                            .min()
                            .unwrap_or(0);
                        let mut end = 0u32;
                        for (j, &i) in idxs.iter().enumerate() {
                            let (tv, rows, y_off, y_stride) = &$secs[i];
                            let raw = gguf.data_offset + tv.offset;
                            blob[j][0] = ((raw & !255usize) - base) as u32;
                            blob[j][1] = (raw & 255usize) as u32;
                            end += *rows;
                            blob[j][2] = end;
                            blob[j][3] = *y_off as u32;
                            blob[j][4] = *y_stride as u32;
                        }
                        // matrix pipe (LEDGER #7, opt-in), then the 64-row
                        // tile for the tall (gate/up) shapes, then the 32-row
                        // tile, then the shipped 8-token multi
                        g.engine.batch_gguf_gemm_mma64_multi(
                            *ty, &g.w_buf, base, pbuf, $x_off, pbuf, 0,
                            end as usize, $cols, b, &blob,
                        ) || g.engine.batch_gguf_gemm_tiled_wide_multi(
                            *ty, &g.w_buf, base, pbuf, $x_off, pbuf, 0,
                            end as usize, $cols, b, &blob,
                        ) || g.engine.batch_gguf_gemm_tiled_multi(
                            *ty, &g.w_buf, base, pbuf, $x_off, pbuf, 0,
                            end as usize, $cols, b, &blob,
                        ) || g.engine.batch_gguf_gemm_multi(
                            *ty, &g.w_buf, base, pbuf, $x_off, pbuf, 0,
                            end as usize, $cols, b, &blob,
                        )
                    });
                }
                fast
            }};
        }

        let layer_snap = std::env::var_os("SOFUU_LAYER_SNAP").is_some();
        if !layer_snap {
            assert!(g.engine.batch_open());
        }
        for layer_idx in 0..self.config.n_layers {
            if layer_snap {
                assert!(g.engine.batch_open());
            }
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
            // SOFUU_QKV_SNAP=<prefix> (layer 0): stage bisect of the L0
            // attention entry — s1 attn-norm out, s2 qkv-gemm out (pre-norm),
            // s3 K/V cache rows after kv_write (post k-norm+rope), s4 qa after
            // head_rmsnorm (post q-norm+rope). Same checkpoints exist in the
            // CPU tiles path.
            if layer_idx == snap_layer {
                if let Some(prefix) = std::env::var_os("SOFUU_QKV_SNAP") {
                    let prefix = prefix.to_string_lossy().to_string();
                    if !g.engine.batch_end() {
                        return false;
                    }
                    let lt = g.pref_act.as_ref().unwrap();
                    let lp: &[f32] = unsafe {
                        std::slice::from_raw_parts(lt.as_ptr() as *const f32, pref_bytes / 4)
                    };
                    let seg = |off: usize, n: usize| -> Vec<u8> {
                        lp[off / 4..off / 4 + n]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect()
                    };
                    let _ = std::fs::write(format!("{prefix}.s1.xn"), seg(pslots.xn, b * dim));
                    assert!(g.engine.batch_open());
                }
            }
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
            if layer_idx == snap_layer {
                if let Some(prefix) = std::env::var_os("SOFUU_QKV_SNAP") {
                    let prefix = prefix.to_string_lossy().to_string();
                    if !g.engine.batch_end() {
                        return false;
                    }
                    let lt = g.pref_act.as_ref().unwrap();
                    let lp: &[f32] = unsafe {
                        std::slice::from_raw_parts(lt.as_ptr() as *const f32, pref_bytes / 4)
                    };
                    let seg = |off: usize, n: usize| -> Vec<u8> {
                        lp[off / 4..off / 4 + n]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect()
                    };
                    let _ = std::fs::write(format!("{prefix}.s2.qa"), seg(pslots.qa, b * q_rows));
                    let _ = std::fs::write(format!("{prefix}.s2.ka"), seg(pslots.ka, b * kv_rows));
                    let _ = std::fs::write(format!("{prefix}.s2.va"), seg(pslots.va, b * kv_rows));
                    assert!(g.engine.batch_open());
                }
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
                    Some((pbuf, pslots.rope)),
                );
                g.engine.batch_head_rmsnorm_b4(
                    pbuf, pslots.qa,
                    q_norm_off.map(|_| &g.w_buf), q_norm_off.unwrap_or(0),
                    pbuf, pslots.posb,
                    b,
                    head_dim as u32, n_heads as u32,
                    has_q, norm_eps, rope_freq_base, rope_mode,
                    Some((pbuf, pslots.rope)),
                );
                if layer_idx == snap_layer {
                    if let Some(prefix) = std::env::var_os("SOFUU_QKV_SNAP") {
                        let prefix = prefix.to_string_lossy().to_string();
                        if !g.engine.batch_end() {
                            return false;
                        }
                        let lt = g.pref_act.as_ref().unwrap();
                        let lp: &[f32] = unsafe {
                            std::slice::from_raw_parts(lt.as_ptr() as *const f32, pref_bytes / 4)
                        };
                        let mut kseg =
                            Vec::with_capacity(b * n_kv_heads * head_dim * 4);
                        let mut vseg = kseg.clone();
                        for kh in 0..n_kv_heads {
                            let base = (kh * max_seq_len + start_pos) * head_dim;
                            for i in base..base + b * head_dim {
                                kseg.extend_from_slice(&self.k_cache[i].to_le_bytes());
                            }
                            for i in base..base + b * head_dim {
                                vseg.extend_from_slice(&self.v_cache[i].to_le_bytes());
                            }
                        }
                        let _ = std::fs::write(format!("{prefix}.s3.kc"), &kseg);
                        let _ = std::fs::write(format!("{prefix}.s3.vc"), &vseg);
                        let qsl: Vec<u8> = lp[pslots.qa / 4..pslots.qa / 4 + b * q_rows]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect();
                        let _ = std::fs::write(format!("{prefix}.s4.qa"), &qsl);
                        assert!(g.engine.batch_open());
                    }
                }
                // Per-pass TRUE chunk count for part/fin: the config-level
                // meta_chunks (32 at msl 4096) spawns 131k mostly-early-return
                // threadgroups on 32-head models when b=128 lands at pos>=128
                // — the deterministic 1B queue fault pass. n_chunks <=
                // ncm_pass by construction, the meta layout stays sized for
                // the config chunks (row stride unchanged, content fits the
                // smaller prefix), and fin loops c < n_chunks — this is
                // index-space only, bit-identical values.
                let ncm_pass = ((pos_of_last / 128) + 1) as u32;
                let flashed = g.engine.batch_llama_attn_flash_b4(
                    pbuf, pslots.qa,
                    &g.k_buf, kc_layer,
                    &g.v_buf, kc_layer,
                    pbuf, pslots.ao,
                    pbuf, pslots.posb,
                    b,
                    head_dim as u32, n_heads as u32, n_kv_heads as u32,
                    max_seq_len as u32,
                );
                // SGate: `llama_attn_1c_b4` is valid for ONE 128-position softmax
                // chunk only (its threadgroup weight table is tw[ATTN_CHUNK] indexed
                // by the ABSOLUTE position; kernels.metal llama_attn_1c_b4). This
                // route used to dispatch it unconditionally, so EVERY pass past
                // position 127 — the second 128-row chunk of a long prompt, or a
                // tail chunk that starts at 128 — read its softmax weights from
                // off the end of that table and returned 10-25-logit-wrong logits
                // with a changed argmax and no warning. Same gate the chain route
                // has carried since LEDGER #64/#65.
                let one_chunk_pos = attn_1c_valid(start_pos, b);
                let fused = flashed
                    || (one_chunk_pos
                        && g.engine.batch_llama_attn_1c_b4(
                            pbuf, pslots.qa,
                            &g.k_buf, kc_layer,
                            &g.v_buf, kc_layer,
                            pbuf, pslots.ao,
                            pbuf, pslots.posb,
                            b,
                            head_dim as u32, n_heads as u32, n_kv_heads as u32,
                            max_seq_len as u32,
                            Some((pbuf, pslots.rope)),
                        ));
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
                        ncm_pass, meta_per_layer as u32,
                    );
                    g.engine.batch_llama_attn_fin_b4(
                        &g.meta_buf, meta_layer_off,
                        pbuf, pslots.ao,
                        pbuf, pslots.posb,
                        b,
                        n_heads as u32, head_dim as u32,
                        ncm_pass, meta_per_layer as u32,
                    );
                }
            }
            let o_ok = !g.skip_attn
                && acc_fuse
                // tile_mmq takes the o-proj through gemm!+add below (the mmq
                // mirror has no fused-acc form; the add = the CPU's one vaddq).
                && !(tile_mmq && matches!(l.attn_output.tensor_type, GGML_TYPE_Q4_K | GGML_TYPE_Q6_K))
                && (l.attn_output.tensor_type == GGML_TYPE_Q4_K
                    || (gemm_types_all
                        && matches!(
                            l.attn_output.tensor_type,
                            GGML_TYPE_Q6_K | GGML_TYPE_Q8_0
                        )))
                && (g.engine.batch_gguf_gemm_mma64_acc(
                    l.attn_output.tensor_type, &g.w_buf, gguf.data_offset + l.attn_output.offset,
                    pbuf, pslots.ao,
                    pbuf, pslots.x, dim,
                    pbuf, pslots.x,
                    dim, q_rows, b,
                ) || g.engine.batch_gguf_gemm_tiled_acc(
                    l.attn_output.tensor_type, &g.w_buf, gguf.data_offset + l.attn_output.offset,
                    pbuf, pslots.ao,
                    pbuf, pslots.x, dim,
                    pbuf, pslots.x,
                    dim, q_rows, b,
                ) || g.engine.batch_gguf_gemm_acc(
                    l.attn_output.tensor_type, &g.w_buf, gguf.data_offset + l.attn_output.offset,
                    pbuf, pslots.ao,
                    pbuf, pslots.x, dim,
                    pbuf, pslots.x,
                    dim, q_rows, b,
                ));
            if !o_ok {
                gemm!(l.attn_output, pslots.ao, pslots.xo, dim, q_rows);
                if !g.engine.batch_add(pbuf, pslots.x, pbuf, pslots.xo, b * dim) {
                    return false;
                }
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
                ) && std::env::var("SOFUU_FUSE_GUP").map(|v| v != "0").unwrap_or(true)
                    && !tile_mmq;
                if !ffn_fused {
                    gemm!(l.ffn_gate, pslots.xn, pslots.gate, hidden_dim, dim);
                    gemm!(l.ffn_up, pslots.xn, pslots.up, hidden_dim, dim);
                }
                if !g.engine.batch_swiglu(pbuf, pslots.gate, pbuf, pslots.up, b * hidden_dim) {
                    return false;
                }
                let d_ok = acc_fuse
                    && !(tile_mmq && matches!(l.ffn_down.tensor_type, GGML_TYPE_Q4_K | GGML_TYPE_Q6_K))
                    && (l.ffn_down.tensor_type == GGML_TYPE_Q4_K
                        || (gemm_types_all
                            && matches!(
                                l.ffn_down.tensor_type,
                                GGML_TYPE_Q6_K | GGML_TYPE_Q8_0
                            )))
                    && (g.engine.batch_gguf_gemm_mma64_acc(
                        l.ffn_down.tensor_type, &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                        pbuf, pslots.gate,
                        pbuf, pslots.x, dim,
                        pbuf, pslots.x,
                        dim, hidden_dim, b,
                    ) || g.engine.batch_gguf_gemm_tiled_acc(
                        l.ffn_down.tensor_type, &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                        pbuf, pslots.gate,
                        pbuf, pslots.x, dim,
                        pbuf, pslots.x,
                        dim, hidden_dim, b,
                    ) || g.engine.batch_gguf_gemm_acc(
                        l.ffn_down.tensor_type, &g.w_buf, gguf.data_offset + l.ffn_down.offset,
                        pbuf, pslots.gate,
                        pbuf, pslots.x, dim,
                        pbuf, pslots.x,
                        dim, hidden_dim, b,
                    ));
                if !d_ok {
                    gemm!(l.ffn_down, pslots.gate, pslots.xo, dim, hidden_dim);
                    if !g.engine.batch_add(pbuf, pslots.x, pbuf, pslots.xo, b * dim) {
                        return false;
                    }
                }
            }
            if layer_snap {
                if !g.engine.batch_end() {
                    return false;
                }
                let lt = g.pref_act.as_ref().unwrap();
                let lp: &[f32] = unsafe {
                    std::slice::from_raw_parts(lt.as_ptr() as *const f32, pref_bytes / 4)
                };
                let xsl: Vec<u8> = lp[pslots.x / 4..pslots.x / 4 + b * dim]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let nsl: Vec<u8> = lp[pslots.xn / 4..pslots.xn / 4 + b * dim]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                if let Some(prefix) = std::env::var_os("SOFUU_LAYER_SNAP") {
                    let prefix = prefix.to_string_lossy().to_string();
                    let _ = std::fs::write(format!("{prefix}.L{layer_idx}.xb"), &xsl);
                    let _ = std::fs::write(format!("{prefix}.L{layer_idx}.xn"), &nsl);
                }
                assert!(g.engine.batch_open());
            }
        }
        if layer_snap {
            // the logits tail (below) needs the encoder it assumes open
            assert!(g.engine.batch_open());
        }

        // Last-row logits inside the same commit (output-norm + head GEMV).
        if want_logits {
            // the last loop op released xo — it is the head's norm scratch
            if !g.engine.batch_rmsnorm_f32w(
                pbuf, pslots.x + (b - 1) * dim * 4,
                &g.w_buf, gguf.data_offset + self.output_norm.offset,
                pbuf, pslots.xo,
                dim, norm_eps,
            ) {
                return false;
            }
            if !g.engine.batch_gguf(
                self.output.tensor_type,
                &g.w_buf, g.w_offset(gguf, &self.output),
                pbuf, pslots.xo,
                pbuf, pslots.logits,
                vocab_size, dim,
            ) {
                return false;
            }
        }
        // A rejected command buffer leaves outputs untouched: treat the
        // pass as not-happened so the caller falls back to the CPU route
        // instead of reading stale slab values as results.
        if !g.engine.batch_end() {
            return false;
        }
        if std::env::var("SOFUU_GPU_TIMING").map(|v| v.trim() == "2").unwrap_or(false) {
            g.engine.print_op_timing("prefill-gemm");
            g.engine.print_gguf_histogram("prefill-gemm");
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
            self.logits_gap_probe_tagged("prefill_head", pos_of_last);
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

    /// P1: full prefill through the GEMM route in pre_chunk-token chunks
    /// (SOFUU_PREFILL_CHUNK, default 32, up to 128). The KV
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
        // SPILL SAFETY: attention part/fin rows write meta slots [L, L+b)
        // (bind base = layer_idx) inside the n_layers*16-slot buffer, and
        // every layer rewrites its own range before its own fin reads it,
        // so the only bound is (n_layers-1)+pre_chunk < n_layers*16 —
        // 155 vs 448 on the 0.6B even at the 128-row cap (arm() clamps
        // pre_chunk to that bound). Receipts: 5-chunk all-pos and the
        // 757-token prompt were bit-exact through the same spill at b=32.
        // (SOFUU_PREFILL_CHUNK is parsed once in arm(); pre_chunk here is
        // the slab-sized value.)
        let pre_chunk = self.gpu.as_ref().map(|g| g.pre_chunk).unwrap_or(32);
        if std::env::var_os("SOFUU_PREFILL_DEBUG").is_some() {
            eprintln!("[prefill-dbg] chunks: n_tokens={} pre_chunk={} start_pos={}", tokens.len(), pre_chunk, start_pos);
        }
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
    fn gpu_layer_step(&mut self, layer_idx: usize, pos: usize, sliding_window: usize) -> bool {
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
            None => return true,
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
            if !g.engine.batch_gguf(
                tv.tensor_type,
                &g.w_buf, g.w_offset(gguf, tv),
                &g.act_buf, g.slots.xn,
                &g.act_buf, y_slot,
                rows, dim,
            ) {
                return false;
            }
        }
        if !g.engine.batch_end() {
            return false;
        }

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
        if !g.engine.batch_gguf(
            layer.attn_output.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.attn_output),
            &g.act_buf, g.slots.ao,
            &g.act_buf, g.slots.xo,
            dim, q_rows,
        ) {
            return false;
        }
        if !g.engine.batch_end() {
            return false;
        }
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
        if !g.engine.batch_gguf(
            layer.ffn_gate.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.ffn_gate),
            &g.act_buf, g.slots.xn,
            &g.act_buf, g.slots.gate,
            hidden_dim, dim,
        ) {
            return false;
        }
        if !g.engine.batch_gguf(
            layer.ffn_up.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.ffn_up),
            &g.act_buf, g.slots.xn,
            &g.act_buf, g.slots.up,
            hidden_dim, dim,
        ) {
            return false;
        }
        g.engine
            .batch_swiglu(&g.act_buf, g.slots.gate, &g.act_buf, g.slots.up, hidden_dim);
        if !g.engine.batch_gguf(
            layer.ffn_down.tensor_type,
            &g.w_buf, g.w_offset(gguf, &layer.ffn_down),
            &g.act_buf, g.slots.gate,
            &g.act_buf, g.slots.xo,
            dim, hidden_dim,
        ) {
            return false;
        }
        if !g.engine.batch_end() {
            return false;
        }
        self.ffn_out.copy_from_slice(g.slot_f32(g.slots.xo, dim));
        inline_mad(&mut self.x, &self.ffn_out, 1.0);
        true
    }
    fn forward_impl(&mut self, token_id: u32, pos: usize, compute_logits: bool) {
        let dim = self.config.dim;

        // 1. Embedding lookup
        Self::lookup_embedding(&self.gguf, &self.token_embd, token_id, dim, &mut self.x);

        // 2. Transformer layers
        let __lt = if std::env::var_os("SOFUU_DEC_PROFILE").is_some() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let gpu_did_layers = self.execute_layers_checked(pos, 0, self.config.n_layers);
        if let Some(t) = __lt {
            // `gpu` is macOS/iOS-only (see the field); Linux/CI builds have
            // no field to read, so the profiler reports false there.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let gpu = self.gpu.as_ref().map(|g| g.token_commit).unwrap_or(false);
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            let gpu = false;
            eprintln!(
                "[dp] pos={} layers={:.2}ms gpu_batch={}",
                pos,
                t.elapsed().as_secs_f64() * 1000.0,
                gpu
            );
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let token_head = self
            .gpu
            .as_ref()
            .map(|g| g.token_commit)
            .unwrap_or(false);
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let token_head = false;
        // The head is only pre-paid when the GPU whole-layer path ACTUALLY
        // ran this token (execute_layers_checked): with KIVI /
        // streaming-weights / dynamic-FFN / a rejected command buffer the CPU
        // layer loop ran instead, and skipping the head would leave
        // `self.logits` frozen at the previous forward's values (F-LC4).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if token_head && gpu_did_layers && compute_logits {
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
                if self.ggpu_head() {
                    return;
                }
                // head batch rejected: self.x is already output-normed —
                // finish with the CPU matvec only (re-norming would be wrong)
                let out_bytes = Self::tensor_slice(&self.gguf, &self.output);
                matvec(
                    out_bytes,
                    self.output.tensor_type,
                    self.config.vocab_size,
                    dim,
                    &self.x,
                    &mut self.logits,
                );
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
            self.logits_gap_probe();
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

    /// The chain's projection kernels are chosen by shape, and one of them
    /// (batch_gguf_b4) computes FOUR rows with no row-count argument. Sending
    /// it a wider chunk loses every row past the fourth while every dispatch
    /// still returns success, so the rule that keeps it on <=4-row chunks is a
    /// test rather than a comment.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn chain_gemv_arm_never_gives_the_four_row_kernel_a_wide_chunk() {
        // q4_K on a 128-row chunk whose mma8 form could not take the shape
        // (cols % 256 != 0, or the pipeline is not on this GPU): not b4.
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q4_K, 128, false, false, false),
            ChainGemvArm::PerRow
        );
        // Every chunk width the four-token kernel really sees.
        for b in 1..=4usize {
            assert_eq!(
                chain_gemv_arm(GGML_TYPE_Q4_K, b, false, false, false),
                ChainGemvArm::B4,
                "b={b} must keep the fused four-token GEMV"
            );
        }
        // q8_0 has no mma8 section at all: wide chunks belong to the typed
        // tile, and a refused tile falls back to one dispatch per row.
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q8_0, 128, false, false, true),
            ChainGemvArm::Tiled
        );
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q8_0, 128, false, false, false),
            ChainGemvArm::PerRow
        );
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q8_0, 4, false, false, false),
            ChainGemvArm::PerRow
        );
        // The routes the shipped model was measured on stay first in line.
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q4_K, 128, true, false, false),
            ChainGemvArm::Mma8
        );
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q4_K, 128, true, false, true),
            ChainGemvArm::Mma8,
            "an enrolled mma8 dispatch must not be second-guessed"
        );
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q6_K, 128, false, true, false),
            ChainGemvArm::Q6Tiled
        );
        assert_eq!(
            chain_gemv_arm(GGML_TYPE_Q6_K, 128, false, true, true),
            ChainGemvArm::Q6Tiled
        );
        // A type no tile knows about never leaves the catch-all.
        for ty in [0u32, 1, 2, GGML_TYPE_Q3_K] {
            assert_eq!(
                chain_gemv_arm(ty, 128, false, false, false),
                ChainGemvArm::PerRow,
                "type {ty}"
            );
        }
    }

    /// Build a synthetic 1-layer llama GGUF over a temp mmap so the loader's
    /// type gate and tied-output handling can be tested without a real model.
    pub(super) fn synth_gguf(output_type: Option<u32>) -> GgufFile {
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
        // token_embd in an unsupported quant (IQ3_XXS, type 18): the load
        // must reject the file with an actionable error, never reach the
        // mid-forward tensor_slice panic.
        let mut gguf = synth_gguf(Some(GGML_TYPE_Q8_0));
        let bad = gguf.tensors.get("token_embd.weight").unwrap().clone();
        gguf.tensors.insert(
            "token_embd.weight".into(),
            TensorInfo { tensor_type: 18, ..bad },
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
        let gguf = synth_gguf(Some(18));
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
    /// The CPU speculative path precomputes every tree node's head logits in
    /// one GEMM and the walk samples each node from that buffer (row i-1 =
    /// batch row i). It must agree with the per-row matvec it replaced, or the
    /// walk verifies against another node's prediction and silently accepts a
    /// different branch.
    ///
    /// Needs a real checkpoint (the synthetic fixtures are all-zero weights, so
    /// every logit would be zero and the comparison vacuous). Skips when the
    /// file is absent; point SOFUU_SPEC_HEAD_GGUF at any small GGUF to run it.
    #[test]
    fn spec_head_logits_batch_matches_per_row_matvec() {
        let path = std::env::var_os("SOFUU_SPEC_HEAD_GGUF").unwrap_or_else(|| {
            "/Users/priyanshuboruah/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf".into()
        });
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
        let mut m = TransformerModel::from_gguf(gguf, 64, vocab).expect("model load");
        m.reset_cache();
        // Root plus two depth-1 children (the ancestry mask is row-major, 1
        // where the column is an ancestor of the row).
        let tokens = vec![128000u32, 2568, 25];
        let positions = vec![0usize, 1, 1];
        let mask = vec![
            1u8, 0, 0,
            1, 1, 0,
            1, 0, 1,
        ];
        let mut root_logits = vec![0.0f32; vocab];
        m.forward_speculative_tree(&tokens, &positions, &mask, 0, &mut root_logits);
        assert!(
            root_logits.iter().any(|v| v.abs() > 1e-6),
            "root logits are all zero — the comparison below would be vacuous"
        );

        let rows = 2usize;
        let mut batched = vec![0.0f32; rows * vocab];
        m.spec_head_logits_batch(1, rows, &mut batched);

        for r in 0..rows {
            let mut per_row = vec![0.0f32; vocab];
            m.forward_speculative_head_item(1 + r, &mut per_row);
            let batched_row = &batched[r * vocab..(r + 1) * vocab];
            for i in 0..vocab {
                let tol = 1e-4 * (1.0 + per_row[i].abs());
                assert!(
                    (batched_row[i] - per_row[i]).abs() <= tol,
                    "row {r} logit {i}: batched {} vs per-row {}",
                    batched_row[i], per_row[i]
                );
            }
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

    /// SGate (TASK 1): the fused one-chunk attention `llama_attn_1c_b4` is valid
    /// for exactly ONE 128-position softmax chunk. Its softmax table is a
    /// threadgroup `float tw[ATTN_CHUNK]` addressed by the row's ABSOLUTE
    /// position, so a pass whose rows reach past position 127 writes off the end
    /// of that table and reads the weights back from somewhere else. Nothing
    /// prints: the answer is simply 10-25 logits (and an argmax) wrong.
    ///
    /// The prefill-GEMM route used to dispatch it unconditionally — one
    /// `flashed || 1c(...)` with no range gate — so EVERY pass past position
    /// 127 (the second 128-row chunk of a long prompt, or a tail chunk that
    /// starts at 128) was silently wrong. Boundaries below are the lengths that
    /// were measured: 128 tokens (one chunk) fine; 137/148/160/255/256/257/383/
    /// 384/385/637/1454 wrong until the gate went in.
    #[test]
    fn attn_1c_gate_is_exactly_one_softmax_chunk() {
        // Inside the first chunk: dispatched.
        assert!(attn_1c_valid(0, 1));
        assert!(attn_1c_valid(0, DEFAULT_PREFILL_BATCH));
        assert!(attn_1c_valid(96, 32));
        assert!(attn_1c_valid(127, 1));
        // Reaching position 128 or beyond: the split part+fin pair owns the pass.
        assert!(!attn_1c_valid(0, 129));
        assert!(!attn_1c_valid(96, 33));
        assert!(!attn_1c_valid(128, 1));
        assert!(!attn_1c_valid(128, DEFAULT_PREFILL_BATCH));
        assert!(!attn_1c_valid(256, 1));
        assert!(!attn_1c_valid(384, 1));
        // Every pass the 128-row top-level prefill walk produces at the lengths
        // that were measured, including the tails (few rows, start_pos >= 128).
        for &(start, rows) in &[
            (0usize, 128usize),
            (128, 128),
            (256, 128),
            (384, 128),
            (1280, 128),
            (128, 9),   // 137 tokens
            (128, 32),  // 160 tokens
            (256, 127), // 383 tokens
            (0, 33),
            (0, 96),
        ] {
            assert_eq!(
                attn_1c_valid(start, rows),
                start + rows <= ATTN_SOFTMAX_CHUNK,
                "start={start} rows={rows}"
            );
        }
        // rows == 0 never happens and must never dispatch attention.
        assert!(!attn_1c_valid(0, 0));
    }

    /// SGate (TASK 1): the gate constant is the KERNEL's chunk, not a number
    /// that happens to agree today — and no prefill route may dispatch
    /// `batch_llama_attn_1c_b4` without it above the call. The kernel cannot see
    /// the positions and the .mm dispatch cannot refuse, so this Rust-side gate
    /// is the only thing between a long prompt and a silently wrong softmax.
    /// Reads the source (repo precedent: the kernels.metal guards in
    /// infer/metal.rs) so a NEW call site cannot be added ungated.
    #[test]
    fn every_prefill_attn_1c_dispatch_sits_behind_the_position_gate() {
        const KERNELS: &str = include_str!("../../../../src/metal/kernels.metal");
        assert_eq!(ATTN_SOFTMAX_CHUNK, 128);
        let start = KERNELS
            .find("kernel void llama_attn_1c_b4(")
            .expect("llama_attn_1c_b4 missing from kernels.metal");
        let rest = &KERNELS[start..];
        let end = rest[1..]
            .find("\nkernel void ")
            .map(|i| i + 1)
            .unwrap_or(rest.len());
        let body = &rest[..end];
        assert!(
            body.contains("threadgroup float tw[128]") && body.contains("tw[t] = exp("),
            "llama_attn_1c_b4 no longer carries the absolute-position tw[128] table this gate exists for"
        );
        let src = include_str!("model.rs");
        // Only the prefill code, never this test's own needle (the string below
        // occurs in here too): everything from the first test module on is out.
        let src = &src[..src.find("\n#[cfg(test)]").unwrap_or(src.len())];
        let calls: Vec<usize> = src
            .match_indices("g.engine.batch_llama_attn_1c_b4(")
            .map(|(i, _)| i)
            .collect();
        assert!(
            calls.len() >= 2,
            "expected the chain route and the prefill-GEMM route, found {}",
            calls.len()
        );
        for at in calls {
            let fn_start = src[..at]
                .rfind("\n    fn ")
                .expect("a 1c dispatch outside any fn");
            let body = &src[fn_start..at];
            assert!(
                body.contains("attn_1c_valid(start_pos"),
                "ungated batch_llama_attn_1c_b4 dispatch at byte {at}: the kernel is valid for one 128-position chunk only"
            );
            assert!(
                body.contains("one_chunk_pos"),
                "the gate above the 1c dispatch at byte {at} is computed but not used"
            );
        }
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
    /// W2 feature split: the CPU arm's residual partial block ([b][dim] f32,
    /// compact: the CPU's output-feature range of attn_output / ffn_down) -
    /// the other half of the pair whose GPU half lands in xo.
    pc: usize,
    /// W2 feature split: the canonical [b][hidden] SwiGLU rows. The CPU arm
    /// owns a prefix, the GPU arm a suffix, both written in place, so ffn_down
    /// (an output-feature split: its rows read the whole swiGLU row) sees one
    /// contiguous block on either engine.
    swg: usize,
    gate: usize,
    up: usize,
    logits: usize,
    xq8e: usize,
}

/// P1 (llama.cpp parity): B-row GEMM prefill slab layout (byte offsets
/// into pref_buf, same gguf_gpu_slot cursor rule as GgufGpuSlots). B =
/// pre_chunk (SOFUU_PREFILL_CHUNK, clamped 1..=128, default 32).
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
    rope: usize,
    xq8: usize,
    logits: usize,
}

/// W2 CPU+GPU feature split: which slice of every projection each engine
/// owns. CPU owns q heads [0, hc) (and so KV heads [0, hc/group)), and output
/// features [0, n) of attn_output / ffn_gate+ffn_up / ffn_down.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Clone, Copy, Debug)]
struct FeatSplit {
    hc: usize,
    kc: usize,
    o_cpu: usize,
    f_cpu: usize,
    d_cpu: usize,
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
    skip_mm: bool,
    slots: GgufGpuSlots,
    /// P1 (llama.cpp parity): B-row GEMM prefill slab. All None unless
    /// SOFUU_GPU_PREFILL_GEMM=1 at engine construction (memory is not paid
    /// by the default fleet path).
    pref_buf: Option<crate::infer::metal::MetalBuffer>,
    pref_act: Option<crate::infer::model::AnonBuffer>,
    pref_slots: Option<PrefillSlots>,
    pref_bytes: usize,
    /// The chunk cap the slab was sized for (parsed ONCE at arm(); the
    /// forward paths read this field, never the env again, so slab and
    /// chunk cannot diverge). 0 when the prefill slab was not built.
    pre_chunk: usize,
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

        // ── fp16 weight materialisation (SOFUU_GPU_F16W=1) ────────────────
        // The chain GEMM tile unpacks q4_K -> fp16 in every 32-value k-slice of
        // every tile, and measured staging is ~38% of tile time. The values it
        // stages are exactly q4k_f16_val(ds, nib, mm), so one dequantisation at
        // load into a plain fp16 matrix is BIT-IDENTICAL and deletes the unpack
        // from the inner loop. Only the q4_K chain projections qualify: the q6_K
        // ones (attn_v / ffn_down on the i1 class) run the f32-FMA tiled kernel,
        // whose arithmetic is a different class and would NOT be bit-identical.
        if crate::infer::metal::MetalEngine::f16_weights_enabled() {
            let mut want: Vec<(usize, usize, usize)> = Vec::new();
            for l in layers.iter() {
                for tv in [&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
                           &l.ffn_gate, &l.ffn_up, &l.ffn_down] {
                    if tv.tensor_type != GGML_TYPE_Q4_K { continue; }
                    if tv.shape.len() != 2 { continue; }
                    let (cols, rows) = (tv.shape[0], tv.shape[1]);
                    if cols % 256 != 0 || rows == 0 { continue; }
                    want.push((gguf.data_offset + tv.offset, rows, cols));
                }
            }
            want.sort_unstable();
            want.dedup();
            engine.materialize_f16(&w_buf, &want);
        }

        // ── temporary instrumentation: SOFUU_BW_PROBE=1 prints achievable read
        // bandwidth over the weight buffer at several thread counts, so the next
        // GEMV is designed against a measured ceiling instead of a guess.
        if std::env::var_os("SOFUU_BW_PROBE").is_some() {
            let n = gguf.mmap.as_slice().len();
            for threads in [1024usize, 4096, 16384, 65536, 262144] {
                let mut best = 0.0f32;
                for _ in 0..3 {
                    let g = engine.bw_probe(w_buf.handle, n, threads, 4, 32, 0, 0, 0);
                    if g > best { best = g; }
                }
                eprintln!("[bw-probe] threads={threads:6} lanes=32 -> {best:.1} GB/s");
            }
            // the shipped GEMV gives a 1024-col row 4 working lanes of 32, and a
            // 3072-col row 12; this measures what that costs
            // thread lifetime: how many blocks each thread streams before it retires.
            // A decode GEMV lane does ONE block (144 B) and exits.
            // threadgroup size: a decode GEMV launches 32-thread groups, which caps
            // the number of resident threads no matter how many are dispatched.
            // granularity: total bytes held constant, slice size swept. A decode
            // projection is 1.7-5.3 MB; the whole buffer is 396 MB.
            // flat vs row-major addressing, same bytes, same block count. Mode 0
            // sweeps flat; mode 4 gives each 32-lane group one row of 4 blocks, the
            // shipped c2 mapping, with the same 144-byte block reads.
            for (tag, mode) in [("flat", 0usize), ("row-major", 4), ("row-major + c2 body", 5)] {
                let slice_blocks = 12_000_000usize / 144;
                let iters = (2_750_000usize / slice_blocks).max(1);
                let mut best = 0.0f32;
                for _ in 0..3 {
                    let g = engine.bw_probe(w_buf.handle, n, 98304, iters, 32, 32, mode, slice_blocks);
                    if g > best { best = g; }
                }
                eprintln!("[bw-probe] {tag:16} -> {best:.1} GB/s");
            }
        }

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
        // 16 rows per chain chunk: every activation slot below is sized from this.
const SPEC_B: usize = 128;
        let slots = GgufGpuSlots {
            x: gguf_gpu_slot(&mut cursor, SPEC_B * dim),
            posb: gguf_gpu_slot(&mut cursor, SPEC_B * 16),
            xn: gguf_gpu_slot(&mut cursor, SPEC_B * dim),
            qa: gguf_gpu_slot(&mut cursor, SPEC_B * q_rows),
            ka: gguf_gpu_slot(&mut cursor, SPEC_B * kv_rows),
            va: gguf_gpu_slot(&mut cursor, SPEC_B * kv_rows),
            ao: gguf_gpu_slot(&mut cursor, SPEC_B * q_rows),
            xo: gguf_gpu_slot(&mut cursor, SPEC_B * dim),
            // W2 feature split exchange blocks (~2MB on a 0.6B, paid once).
            pc: gguf_gpu_slot(&mut cursor, SPEC_B * dim),
            swg: gguf_gpu_slot(&mut cursor, SPEC_B * inter.max(dim)),
            gate: gguf_gpu_slot(&mut cursor, SPEC_B * inter.max(dim)),
            up: gguf_gpu_slot(&mut cursor, SPEC_B * inter.max(dim)),
            logits: gguf_gpu_slot(&mut cursor, SPEC_B * vocab),
            xq8e: gguf_gpu_slot(
                &mut cursor,
                (dim.max(q_rows).max(inter.max(dim)) / 32) * 10,
            ),
        };
        let act_bytes = gguf_gpu_page(cursor);
        let act = crate::infer::model::AnonBuffer::new(act_bytes / 4);
        let act_slice: &[u8] =
            unsafe { std::slice::from_raw_parts(act.as_ptr() as *const u8, act_bytes) };
        let act_buf = engine.register_zerocopy_only(act_slice)?;

        // ── temporary instrumentation: SOFUU_GEMV_BENCH=1 times the shipped decode
        // GEMV at real geometry inside one commit. Until now "the kernel is slow"
        // and "the kernel is slow inside the token batch" were the same sentence.
        if std::env::var_os("SOFUU_GEMV_BENCH").is_some() {
            let head_only = std::env::var("SOFUU_GEMV_BENCH").map(|v| v == "head").unwrap_or(false);
            // SOFUU_GEMV_BENCH_N truncates the plan, so a run at K dispatches and a
            // run at 2K separate the per-dispatch fixed cost from the stream rate.
            let bench_n: usize = std::env::var("SOFUU_GEMV_BENCH_N")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
            let mut plan: Vec<(u32, usize, usize, usize)> = Vec::new();
            if head_only {
                for _ in 0..8 {
                    plan.push((output.tensor_type, vocab, dim, gguf.data_offset + output.offset));
                }
            } else {
                for l in layers.iter() {
                    plan.push((l.ffn_gate.tensor_type, inter, dim, gguf.data_offset + l.ffn_gate.offset));
                    plan.push((l.attn_q.tensor_type, q_rows, dim, gguf.data_offset + l.attn_q.offset));
                }
            }
            let per = |ty: u32| -> f64 {
                match ty { 12 => 144.0 / 256.0, 14 => 210.0 / 256.0, 8 => 34.0 / 32.0, _ => 4.0 }
            };
            // NOTE: computed from the FULL plan on purpose-visible order: truncation
            // happens below, so this must be recomputed after it or every truncated
            // run reports the whole plan's bytes (which is exactly what made an
            // 8-dispatch run look like 81.6 GB/s while reading 16.8 MB).
            // zero the activation slab: uninitialised floats include denormals and
            // NaNs, and a denormal-bound kernel looks exactly like a slow kernel.
            unsafe { std::ptr::write_bytes(act.as_ptr() as *mut u8, 0, act_bytes); }
            // one dispatch, to separate per-dispatch cost from streaming cost
            for _ in 0..3 {
                if engine.batch_open() {
                    if plan.len() > bench_n { plan.truncate(bench_n); }
            let t0 = std::time::Instant::now();
                    let (ty, rows, cols, w_off) = plan[0];
                    let _ = engine.batch_gguf(ty, &w_buf, w_off, &act_buf, slots.x, &act_buf, slots.xo, rows, cols);
                    let _ = engine.batch_end();
                    let mb = rows as f64 * cols as f64 * per(ty) / 1e6;
                    eprintln!("[gemv-bench] 1 dispatch {rows}x{cols} = {mb:.2} MB -> {:.1} us",
                        t0.elapsed().as_secs_f64() * 1e6);
                }
            }
            if plan.len() > bench_n { plan.truncate(bench_n); }
            let total_bytes: usize = plan.iter()
                .map(|(ty, rows, cols, _)| (*rows as f64 * *cols as f64 * per(*ty)) as usize)
                .sum();
            let mut best = 0.0f64;
            for rep in 0..4 {
                if !engine.batch_open() { eprintln!("[gemv-bench] open failed"); break; }
                let t0 = std::time::Instant::now();
                // alternate the output slot: writing every projection into the same
                // buffer makes each dispatch wait on the previous one's store.
                let outs = [slots.xo, slots.gate, slots.up, slots.qa];
                for (i, (ty, rows, cols, w_off)) in plan.iter().enumerate() {
                    if !engine.batch_gguf(*ty, &w_buf, *w_off, &act_buf, slots.x,
                                          &act_buf, outs[i % outs.len()], *rows, *cols) {
                        eprintln!("[gemv-bench] gguf failed ty={ty} rows={rows} cols={cols}");
                        break;
                    }
                }
                let ok = engine.batch_end();
                let dt = t0.elapsed().as_secs_f64();
                if !ok { eprintln!("[gemv-bench] end failed"); break; }
                let gbs = total_bytes as f64 / 1e9 / dt;
                if gbs > best { best = gbs; }
                if rep == 0 {
                    eprintln!("[gemv-bench] {} dispatches, {:.1} MB, types q4k={} -> {:.1} GB/s",
                        plan.len(), total_bytes as f64 / 1e6,
                        plan.iter().filter(|(ty, _, _, _)| *ty == 12).count(), gbs);
                }
            }
            eprintln!("[gemv-bench] best {best:.1} GB/s");
        }
        // P1 (llama.cpp parity): the GEMM prefill slab, built only when the
        // route is requested at load (SOFUU_GPU_PREFILL_GEMM=1). PRE_B =
        // pre_chunk rows (SOFUU_PREFILL_CHUNK, clamped 1..=128, default 32);
        // every stage's [B x rows] region sized accordingly.
        let pre_enabled = std::env::var("SOFUU_GPU_PREFILL_GEMM")
            .map(|v| v == "1")
            .unwrap_or(false);
        let pre_chunk = if pre_enabled {
            let requested = std::env::var("SOFUU_PREFILL_CHUNK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| (1..=128).contains(v))
                .unwrap_or(32);
            // Attention part/fin rows land at meta slots [L, L+b) where the
            // bind base is layer_idx slots inside an n_layers*SPEC_B*4-slot
            // buffer: worst index (n_layers-1)+b must stay below
            // n_layers*16. 128-row chunks on a 28-layer model peak at slot
            // 155 of 448 — the 32-row receipts' spill was the same hazard
            // at b=32 (peak 58 of 112 by an off-by-4 in the old note; the
            // capacity here is 16 slots per layer, not 4).
            requested.min(15 * layers.len() + 1)
        } else {
            0
        };
        let (pref_buf, pref_act, pref_slots, pref_bytes) = if pre_enabled {
            let PRE_B = pre_chunk;
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
                rope: gguf_gpu_slot(&mut pc, PRE_B * config.head_dim),
                xq8: gguf_gpu_slot(
                    &mut pc,
                    PRE_B * (dim.max(q_rows).max(config.hidden_dim) / 256) * 65,
                ),
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
            skip_mm: skips.contains("mm"),
            pref_buf,
            pref_act,
            pref_slots,
            pref_bytes,
            pre_chunk,
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

    /// LEDGER #173: mutable slab view, mirror of slot_f32_mut for pref_buf.
    #[inline]
    fn pref_f32_mut(&mut self, off: usize, n: usize) -> &mut [f32] {
        unsafe {
            let b = self.pref_buf.as_ref().expect("pref_f32_mut: no slab");
            let base = crate::infer::metal::buffer_contents_raw(b);
            std::slice::from_raw_parts_mut(base.add(off) as *mut f32, n)
        }
    }

    /// LEDGER #164: read the PREFILL SLAB (pref_buf) from the CPU. Mirrors slot_f32_mut, which is
    /// hard-wired to act_buf; the slab lives in its own buffer and needs the contents FFI.
    #[inline]
    fn pref_f32(&self, off: usize, n: usize) -> &[f32] {
        unsafe {
            let b = self.pref_buf.as_ref().expect("pref_f32: no slab");
            let base = crate::infer::metal::buffer_contents_raw(b);
            std::slice::from_raw_parts(base.add(off) as *const f32, n)
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


#[cfg(test)]
mod kivi_compact_tests {
    use super::tests::synth_gguf;
    use super::*;

    /// F-LC3 (§8 P5): compact_kv_cache() drops middle tokens, shifts survivors
    /// left and re-ropes them; the KIVI stream must describe the SAME
    /// (sinks + survivors) history afterwards, row for row.
    #[test]
    fn kivi_rebuild_after_compact_matches_compacted_cache() {
        let gguf = synth_gguf(None);
        let mut m = TransformerModel::from_gguf(gguf, 64, 256).expect("synth load");
        let (layers, kvh, hd, seq) = (
            m.config.n_layers,
            m.config.n_kv_heads,
            m.config.head_dim,
            m.config.max_seq_len,
        );
        let n_heads = m.config.n_heads;
        // Rows 0..11 of every layer/head hold distinct values.
        for layer in 0..layers {
            for p in 0..11 {
                for h in 0..kvh {
                    let base = (layer * kvh + h) * seq * hd + p * hd;
                    for c in 0..hd {
                        let t = ((base + c) % 251) as f32;
                        m.k_cache[base + c] = 0.01 + t * 0.003;
                        m.v_cache[base + c] = 0.02 + t * 0.002;
                    }
                }
            }
        }
        m.enable_kivi(4, 11);
        assert!(m.kivi.is_some());
        let pre_total = m.kivi.as_ref().unwrap().heads[0].total_tokens();
        assert_eq!(pre_total, 11);

        // Compact like the decode loop: keep 4 sinks, drop 4 middle tokens.
        let new_len = m.compact_kv_cache(11, 4, 4);
        assert_eq!(new_len, 7);

        // Snapshot the post-compaction f32 rows (sinks 0..4, survivors 4..7)
        // — these are the ground truth the rebuilt stream must match.
        let mut k_expected = vec![vec![0.0f32; hd]; layers * kvh * new_len];
        let mut v_expected = vec![vec![0.0f32; hd]; layers * kvh * new_len];
        for head in 0..layers * kvh {
            for p in 0..new_len {
                let slot = head * seq * hd + p * hd;
                k_expected[head * new_len + p].copy_from_slice(&m.k_cache[slot..slot + hd]);
                v_expected[head * new_len + p].copy_from_slice(&m.v_cache[slot..slot + hd]);
            }
        }

        m.rebuild_kivi_after_compact(new_len);
        let kivi = m.kivi.as_mut().expect("kivi must still be armed");
        for head in 0..layers * kvh {
            let st = &kivi.heads[head];
            assert_eq!(st.sink_len, 4, "sinks preserved");
            assert_eq!(st.recent_tokens, new_len - 4, "survivors re-ingested");
            assert_eq!(st.total_tokens(), new_len);
            assert!(st.num_quantized_tokens >= 0);
            for t in 0..4 {
                assert_eq!(&st.sink_k[t * hd..(t + 1) * hd], &k_expected[head * new_len + t][..]);
                assert_eq!(&st.sink_v[t * hd..(t + 1) * hd], &v_expected[head * new_len + t][..]);
            }
            for t in 4..new_len {
                assert_eq!(
                    &st.recent_k[(t - 4) * hd..(t - 3) * hd],
                    &k_expected[head * new_len + t][..]
                );
                assert_eq!(
                    &st.recent_v[(t - 4) * hd..(t - 3) * hd],
                    &v_expected[head * new_len + t][..]
                );
            }
        }

        // Attention over the rebuilt stream computes.
        let q = vec![0.3f32; n_heads * hd];
        let mut out = vec![0.0f32; n_heads * hd];
        kivi.compute_layer_attention(0, n_heads, &q, &mut out);
        assert!(out.iter().any(|&v| v != 0.0));
    }
}

#[cfg(test)]
mod kivi_ring_tests {
    use super::*;

    /// P7: the streaming ring prefill (quantized causal attention over the
    /// KIVI streams) must produce the same next-token distribution class as
    /// the plain f32 prefill on a short prompt — same argmax, close logits.
    /// Env-gated on a real checkpoint: SOFUU_LC_RING_GGUF=<small llama gguf>.
    #[test]
    fn kivi_ring_prefill_matches_f32_argmax() {
        let path = match std::env::var_os("SOFUU_LC_RING_GGUF") {
            Some(p) => p,
            None => {
                println!("SOFUU_LC_RING_GGUF not set; skipping");
                return;
            }
        };
        if !std::path::Path::new(&path).exists() {
            println!("model file missing; skipping");
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

        // Plain f32 reference.
        let mut plain = TransformerModel::from_gguf_with_gpu(GgufFile::open(&path).unwrap(), 4096, vocab, false, 0)
            .expect("plain load");
        let toks: Vec<u32> = vec![128000, 791, 6864, 315, 9822, 374, 12366, 13, 578, 6864, 315, 10057, 374];
        let logits_plain: Vec<f32> = plain.forward_prefill_all_with_logits(&toks, 0).to_vec();

        // Ring-KIVI streaming prefill (ring 32 forces multiple causal groups).
        let mut ring = TransformerModel::from_gguf_with_gpu(GgufFile::open(&path).unwrap(), 4096, vocab, false, 32)
            .expect("ring load");
        ring.enable_kivi_streaming(4);
        let logits_ring: Vec<f32> = ring.forward_prefill_all_with_logits(&toks, 0).to_vec();

        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let (ap, ar) = (argmax(&logits_plain), argmax(&logits_ring));
        let dot: f32 = logits_plain.iter().zip(&logits_ring).map(|(a, b)| a * b).sum();
        let np = (logits_plain.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let nr = (logits_ring.iter().map(|v| v * v).sum::<f32>()).sqrt();
        let cos = dot / (np * nr);
        println!("[ring-parity] argmax plain={ap} ring={ar} cosine={cos:.5}");
        assert_eq!(ap, ar, "ring-KIVI streaming prefill changed the argmax");
        assert!(cos > 0.95, "ring-KIVI logits diverge from f32 (cosine {cos:.5})");
    }
}
