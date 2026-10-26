// crates/sofuu-core/src/infer/qwen35.rs — Qwen3.5 hybrid inference backend (CPU).
//
// Single-sequence, deterministic backend for the Qwen3.5 architecture:
// gated-delta-net (GDN) layers interleaved with periodic gated full attention.
// Big matmul weights are Prism-Hadamard-folded PTQ1_0 ternary tensors that stay
// raw in the mmapped GGUF and are dequantized per row into a f32 scratch buffer
// right before each dot. The Hadamard side of the fold is applied to
// activations as an exact in-place normalized Sylvester Walsh–Hadamard
// transform per 1024-wide block; the token-embedding table is inverse-folded
// (transform applied after row lookup, in the opposite order).

use std::collections::HashSet;

use super::gguf::{GgufFile, GgufValue};
use super::model::ModelConfig;
use super::quant::{bf16_to_f32, f16_to_f32, prof, ptq1_0_dequant_row};

/// GGML tensor type id for PTQ1_0 (prism ternary, 28 bytes per 128 values).
const GGML_TYPE_PTQ1_0: u32 = 143;
/// GGML tensor type id for PQ2_0 (prism 2-bit, 34 bytes per 128 values —
/// 2 B f16 scale FIRST, then 32 B of little-endian 2-bit codes; code c
/// decodes to (c − 1) · d, i.e. the {−1, 0, +1, +2}·d lattice). Same
/// hadamard-fold family as PTQ1_0; the Ternary-Bonsai-2-9B stores its
/// mainline weights here.
const GGML_TYPE_PQ2_0: u32 = 142;

/// One packed PQ2_0 row on the wire: 34 B per 128 values.
const PQ2_0_BLOCK_BYTES: usize = 34;
const PQ2_0_BLOCK_VALUES: usize = 128;
/// GGML tensor type id for bfloat16.
const GGML_TYPE_BF16: u32 = 30;

/// Hadamard fold block size (prism.hadamard.block_size for this model).
const HAD_BLOCK: usize = 1024;

/// Byte length of one PTQ1_0 block: f16 scale + 24B qs + 2B qh = 128 values.
const PTQ1_0_BLOCK_BYTES: usize = 28;
const PTQ1_0_BLOCK_VALUES: usize = 128;

/// Minimum row·input product (MACs) before a PTQ1_0 GEMV/GEMM fans out to
/// threads; below it the single-thread path is cheaper than spawning.
const PAR_MAC_THRESHOLD: usize = 2_000_000;
/// One 128-value PTQ1_0 block on the wire: 24B qs + 2B qh + 2B f16 scale.
const PTQ1_0_ROW_BLOCK_BYTES: usize = 28;
/// f32 byte budget of one thread's dequantized row slab in the batched GEMM.
/// Measured on 6P+4E Apple Silicon: 8 MB beats 1 MB (act streams beat slab
/// streaming); the slab is dequantize-once per row, so this only sets the
/// per-thread re-read window.
const GEMM_SLAB_BYTES: usize = 8 * 1024 * 1024;
/// Tokens per batched-prefill sub-chunk; bounds transient activation scratch
/// independently of prompt length.
const PRE_FILL_CHUNK: usize = 128;

/// Thread fan-out uses available cores but is not part of the result: chunk
/// boundaries are pure index math, so every output row runs the identical
/// dequant + dot sequence no matter how many threads join in.
///
/// `available_parallelism()` reports 1 in the static-musl Pi 5 build even
/// with four cores online and an unrestricted affinity mask (its
/// sched_getaffinity path fails and Rust falls back to 1), which silently
/// pinned every fan-out here to one core. The C++ worker pool detects the
/// real count, so borrow it.
fn thread_budget() -> usize {
    super::quant::get_thread_count().clamp(1, 8)
}

/// GDN state geometry (default; actual values come from the GGUF header).
const DEF_HEAD_K_DIM: usize = 128;
const DEF_HEAD_V_DIM: usize = 128;

/// How the runtime must prepare the activation for a weight's matmul.
#[derive(Clone, Copy)]
enum Xform {
    /// x ← per-1024-block H · (S ⊙ x) with sign vector `signs[si]`.
    Folded(usize),
    /// Inverse fold (embedding tables): row ← S ⊙ (H · row) after lookup.
    Inverse(usize),
    /// Plain weight; activation passes through.
    None,
}

/// A 2D weight matrix kept raw in the mmapped GGUF. GGUF shape convention:
/// ne0 (row length, matmul input width) is the fastest-varying axis, so
/// output row `r` occupies bytes [off + r·row_bytes, off + (r+1)·row_bytes).
#[derive(Clone)]
struct RawMat {
    off: usize,
    n_in: usize,  // ne0
    n_out: usize, // ne1
    row_bytes: usize,
    ty: u32, // GGML type of the packed rows (143 PTQ1_0, 142 PQ2_0, ...)
    xform: Xform,
}

impl RawMat {
    #[inline]
    fn row_slice<'a>(&self, data: &'a [u8], r: usize) -> &'a [u8] {
        let start = self.off + r * self.row_bytes;
        &data[start..start + self.row_bytes]
    }

    /// Index into the model's sign table, or None when unfolded.
    #[inline]
    fn sign_idx(&self) -> Option<usize> {
        match self.xform {
            Xform::Folded(i) | Xform::Inverse(i) => Some(i),
            Xform::None => None,
        }
    }
}

#[derive(Clone)]
enum Mixer {
    Gdn {
        qkv: RawMat,      // [dim -> 10240]: q(2048) | k(2048) | v(6144)
        gate: RawMat,     // [dim -> 6144] (z)
        conv1d: Vec<f32>, // [4 x 10240]: channel c's taps at [c*4 .. c*4+4]
        dt_bias: Vec<f32>,// [48]
        a: Vec<f32>,      // [48] A = -exp(A_log), per-head decay multiplier
        alpha: Vec<f32>,  // [48 x dim] f32, row r = output r (unfolded)
        beta: Vec<f32>,   // [48 x dim] f32 (unfolded)
        ssm_norm: Vec<f32>,   // [head_v_dim]
        out: RawMat,         // [6144 -> dim], folded, GDN perm on activation
    },
    Attention {
        q: RawMat,        // [dim -> 12288]: per head q(256) | gate(256)
        k: RawMat,        // [dim -> 1024]
        v: RawMat,        // [dim -> 1024]
        q_norm: Vec<f32>, // [256]
        k_norm: Vec<f32>, // [256]
        o: RawMat,        // [6144 -> dim]
    },
}

struct Layer {
    attn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_gate: RawMat, // [dim -> hidden]
    ffn_up: RawMat,   // [dim -> hidden]
    ffn_down: RawMat, // [hidden -> dim]
    mixer: Mixer,
}

pub struct Qwen35Model {
    pub config: ModelConfig,
    gguf: GgufFile,
    embd: RawMat,   // token_embd [dim -> vocab], Xform::Inverse on this model
    output: RawMat, // [dim -> vocab]
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
    /// f32 sign vectors (±1) keyed by input width, from prism.hadamard.
    signs: Vec<(usize, Vec<f32>)>,
    /// prism.hadamard.block_size for this model (power of two; 27B = 1024).
    had_block: usize,
    rope_sections: [usize; 4],
    /// GDN slot (0..n_gdn) per layer; usize::MAX for attention layers.
    gdn_index: Vec<usize>,
    /// Attention cache slot (0..n_attn) per layer; usize::MAX for GDN layers.
    attn_index: Vec<usize>,
    /// EOS id from the GGUF header (surfaced for the sampler/wiring).
    pub eos_token: u32,

    // Recurrent state
    ssm_state: Vec<f32>,  // [gdn layers][v heads][k_dim*v_dim], S[i][j] at i*k_dim+j
    conv_state: Vec<f32>, // [gdn layers][conv channels][3 ring slots]
    k_cache: Vec<f32>,    // [attn layer][kv head][pos][head_dim]
    v_cache: Vec<f32>,

    // Reused scratch (allocation-light steady state)
    rowbuf: Vec<f32>,    // one dequantized weight row (widest row: hidden)
    x: Vec<f32>,
    hbuf: Vec<f32>,      // post attn_norm hidden (raw basis)
    prep512: Vec<f32>,   // staged hidden, width = dim
    prep_mid: Vec<f32>,  // staged GDN output / perm target, width = d_inner
    prep_ff: Vec<f32>,   // staged FFN intermediate, width = hidden
    qkv: Vec<f32>,       // conv output (silu applied in place), conv_dim
    zg: Vec<f32>,        // 6144 gate (z)
    g48a: Vec<f32>,      // raw alpha per v-head
    g48b: Vec<f32>,      // raw beta per v-head
    gdn_out: Vec<f32>,   // GDN core output, v_dim * v_heads
    mix_out: Vec<f32>,   // mixer/FFN projection output, dim
    qg: Vec<f32>,        // fused q+gate, 2 * heads * head_dim
    kh: Vec<f32>,        // kv heads * head_dim
    vh: Vec<f32>,        // kv heads * head_dim
    attn3: Vec<f32>,     // attention output pre-projection, heads * head_dim
    scores: Vec<f32>,    // attention scores, max_seq_len
    ffn_g: Vec<f32>,     // hidden
    ffn_u: Vec<f32>,     // hidden
    logits: Vec<f32>,
}

// ── small numeric helpers ────────────────────────────────────────────

#[inline(always)]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

#[inline(always)]
fn softplus20(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline(always)]
fn silu(v: f32) -> f32 {
    v * sigmoid(v)
}

/// Deterministic 8-way unrolled dot product (fixed summation order).
/// Accumulator j sums a[j], a[8+j], a[16+j], … in that exact sequence and the
/// tail folds into s0 — the operation order is identical to writing eight
/// scalar chains, only expressed as an [f32; 8] array so the lane indexing
/// (and its bounds checks) optimizes out.
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let b = &b[..n];
    let mut s = [0.0f32; 8];
    for (wa, wb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for j in 0..8 {
            s[j] += wa[j] * wb[j];
        }
    }
    let n_full = (n / 8) * 8;
    let mut s0 = s[0];
    for k in n_full..n {
        s0 += a[k] * b[k];
    }
    let s01 = s0 + s[1];
    let s23 = s[2] + s[3];
    let s45 = s[4] + s[5];
    let s67 = s[6] + s[7];
    (s01 + s23) + (s45 + s67)
}

/// RMS norm: x ← x · (w / sqrt(mean(x²) + eps)); f64 accumulation matches the
/// reference's ggml_float sums.
fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean = sum_sq / x.len() as f64;
    let scale = 1.0 / ((mean + eps as f64).sqrt() as f32);
    for (v, w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

/// L2-normalize in place: scale = 1 / max(l2(x), eps); eps floors the norm.
fn l2norm_slice(x: &mut [f32], eps: f32) {
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let norm = (sum_sq.sqrt() as f32).max(eps);
    let scale = 1.0 / norm;
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// One Sylvester Walsh–Hadamard block, normalized: computes exactly
/// out[i] = Σ_j (±1)^popcount(i&j) · x[j] / sqrt(len). The fast butterfly is
/// the standard unsorted WHT; the final 1/sqrt(n) scaling is a power of two
/// (n = 1024 → 1/32), so no rounding is introduced by the normalization.
fn wht_block(block: &mut [f32]) {
    let n = block.len();
    debug_assert!(n.is_power_of_two());
    let mut len = 1;
    while len < n {
        let mut start = 0;
        while start < n {
            for i in start..start + len {
                let a = block[i];
                let b = block[i + len];
                block[i] = a + b;
                block[i + len] = a - b;
            }
            start += len * 2;
        }
        len *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in block.iter_mut() {
        *v *= scale;
    }
}

fn wht_blocks(block_len: usize, x: &mut [f32]) {
    debug_assert!(block_len.is_power_of_two());
    for block in x.chunks_exact_mut(block_len) {
        wht_block(block);
    }
}

/// Prepared folded-matmul activation: dst[i] = S[i]·src[i], then per-block H.
fn raw_gdn_v() -> bool {
    std::env::var_os("SOFUU_GDN_V_RAW").is_some_and(|v| v != "0")
}

/// prism.hadamard.gdn_v_grouped is set on EVERY prism checkpoint (27B and 9B
/// alike) — it marks that the [hd, nk, rep] → [hd, rep, nk] regroup rides the
/// ssm_out transform, not that it may be skipped. SOFUU_GDN_V_RAW=1 keeps the
/// raw order for A/B only.
fn hadamard_prep(block: usize, dst: &mut [f32], src: &[f32], signs: &[f32]) {
    for (d, (&s, &v)) in dst.iter_mut().zip(signs.iter().zip(src.iter())) {
        *d = s * v;
    }
    wht_blocks(block, dst);
}

/// Signs + Hadamard applied in place (activation already staged in x).
fn hadamard_prep_in_place(block: usize, x: &mut [f32], signs: &[f32]) {
    let _p = prof::Scope::new(&prof::ROPE_NS, &prof::ROPE_CALLS);
    for (v, &s) in x.iter_mut().zip(signs) {
        *v *= s;
    }
    wht_blocks(block, x);
}

/// GDN regrouping for the folded ssm_out activation. The folded weight expects
/// the activation tiled [hd=128, nk=16, rep=3] (feature f = d + 128·(nk + 16·rep));
/// regroup to [hd, rep, nk]: f' = d + 128·rep + 384·nk (fork: permute(0,2,1,3)).
fn gdn_perm(dst: &mut [f32], src: &[f32]) {
    const HD: usize = 128;
    const NK: usize = 16;
    let rep = src.len() / (HD * NK); // 27B rep 3, 9B rep 2 (= ssm_dt_rank/nk)
    debug_assert_eq!(src.len(), HD * NK * rep);
    for rep_i in 0..rep {
        for nk in 0..NK {
            for d in 0..HD {
                let fi = d + HD * (nk + NK * rep_i);
                let fo = d + HD * rep_i + HD * rep * nk;
                dst[fo] = src[fi];
            }
        }
    }
}

/// Dequantize a PTQ1_0 row into per-value trits in {-1, 0, +1} plus the
/// per-128-value f16 block scale. The trit walk is the frozen decoder from
/// `ptq1_0_dequant_row` (quant.rs) with the final `· d` folded into `scales`,
/// so the two agree value-for-value: `trits[i] as f32 · scales[i / 128]` equals
/// the f32-dequant output bit for bit. Storing trits instead of scaled f32
/// quarters the slab write/read traffic the GEMV/GEMM kernels pay per sweep.
fn dequant_trits_i8(src: &[u8], trits: &mut [i8], scales: &mut [f32]) {
    const QS_LEN: usize = (PTQ1_0_BLOCK_VALUES - 4 * PTQ1_0_BLOCK_VALUES / 64) / 5; // 24
    const QH_LEN: usize = PTQ1_0_BLOCK_VALUES / 64; // 2
    const BLOCK: usize = QS_LEN + QH_LEN + 2; // qs | qh | f16 scale = 28
    const STAGES: [usize; 3] = [32, 16, 8];
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

    let n_blocks = trits.len() / PTQ1_0_BLOCK_VALUES;
    debug_assert!(src.len() >= n_blocks * BLOCK);
    debug_assert_eq!(scales.len(), n_blocks);

    for blk in 0..n_blocks {
        let b_offset = blk * BLOCK;
        scales[blk] = f16_to_f32(u16::from_le_bytes([
            src[b_offset + QS_LEN + QH_LEN],
            src[b_offset + QS_LEN + QH_LEN + 1],
        ]));
        let mut o = blk * PTQ1_0_BLOCK_VALUES;

        let mut j = 0;
        for s in 0..STAGES.len() {
            let c = STAGES[s];
            while j + c <= QS_LEN {
                for n in 0..5usize {
                    for m in 0..c {
                        let q = src[b_offset + j + m].wrapping_mul(POW3[n]);
                        let xi = ((q as u16) * 3) >> 8;
                        trits[o] = (xi as i16 - 1) as i8;
                        o += 1;
                    }
                }
                j += c;
            }
        }
        for n in 0..4usize {
            for h in 0..QH_LEN {
                let q = src[b_offset + QS_LEN + h].wrapping_mul(POW3[n]);
                let xi = ((q as u16) * 3) >> 8;
                trits[o] = (xi as i16 - 1) as i8;
                o += 1;
            }
        }
        debug_assert_eq!(o, (blk + 1) * PTQ1_0_BLOCK_VALUES);
    }
}

/// Expand one PQ2_0 block's 32 code bytes into 128 int8 levels.
///
/// Value 4k+m lives in byte k at shift 2m, so the four shift/and passes are
/// exactly the four interleaved streams `vst4` writes back in value order.
/// The scalar loop this replaces ran 128 shift/and/sub triples per block; the
/// integer band dot is exact regardless of how the levels were produced, so
/// results stay bit-identical.
#[cfg(target_arch = "aarch64")]
#[inline]
fn decode_pq2_codes_neon(codes: &[u8], out: &mut [i8]) {
    debug_assert_eq!(codes.len(), PQ2_0_BLOCK_VALUES / 4);
    debug_assert_eq!(out.len(), PQ2_0_BLOCK_VALUES);
    unsafe {
        use std::arch::aarch64::*;
        let m3 = vdupq_n_u8(0x03);
        let one = vdupq_n_s8(1);
        for half in 0..(PQ2_0_BLOCK_VALUES / 64) {
            let v = vld1q_u8(codes.as_ptr().add(half * 16));
            let c0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m3)), one);
            let c1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), m3)), one);
            let c2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), m3)), one);
            let c3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 6)), one);
            vst4q_s8(out.as_mut_ptr().add(half * 64), int8x16x4_t(c0, c1, c2, c3));
        }
    }
}

/// Fused PQ2_0 block dot: expand the 32 code bytes and sdot them against the
/// activation band without leaving registers. The buffer shape decoded into a
/// 128-byte stack array and re-loaded it (`vst4q` then eight `vld1q` from the
/// same lines), which stalls on store forwarding on A76. Value j is 4k+m in
/// byte k at shift 2m, so `vld4q_s8` of the band hands back exactly the four
/// interleaved activation streams the four decoded groups need.
///
/// Integer dot products are exact and order-independent, so this returns the
/// same i32 as `sdot_group(decoded_levels, band)`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn p2_block_dot_neon(codes: *const u8, a: *const i8) -> i32 {
    use std::arch::aarch64::*;
    let m3 = vdupq_n_u8(0x03);
    let one = vdupq_n_s8(1);
    let mut acc = vdupq_n_s32(0);
    for q in 0..(PQ2_0_BLOCK_VALUES / 64) {
        let v = vld1q_u8(codes.add(q * 16));
        let c0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m3)), one);
        let c1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), m3)), one);
        let c2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), m3)), one);
        let c3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 6)), one);
        let a4 = vld4q_s8(a.add(q * 64));
        for (wv, av) in [(c0, a4.0), (c1, a4.1), (c2, a4.2), (c3, a4.3)] {
            std::arch::asm!(
                "sdot {acc:v}.4s, {wv:v}.16b, {av:v}.16b",
                acc = inout(vreg) acc,
                wv = in(vreg) wv,
                av = in(vreg) av,
                options(pure, nomem, nostack)
            );
        }
    }
    vaddvq_s32(acc)
}

/// Dequantize one PQ2_0 (GGML type 142, prism 2-bit at group 128) row into
/// per-value int8 levels in {-1, 0, +1, +2} plus the per-128-value f16 block
/// scale — the same slab contract as `dequant_trits_i8`, so every sdot/slab
/// consumer runs either codec unchanged: the band dot folds as
/// `levels[i] · scales[i / 128]` because the code lattice is (code − 1) · d.
/// Block layout: f16 d FIRST (bytes 0..2), then 32 B of codes; value j lives
/// at byte j/4 low bits first: code = (qs[j / 4] >> ((j % 4) · 2)) & 3.
fn dequant_pq2_i8(src: &[u8], levels: &mut [i8], scales: &mut [f32]) {
    const QK: usize = PQ2_0_BLOCK_VALUES;
    const BLOCK: usize = PQ2_0_BLOCK_BYTES;
    let n_blocks = levels.len() / QK;
    debug_assert!(src.len() >= n_blocks * BLOCK);
    debug_assert_eq!(scales.len(), n_blocks);
    for blk in 0..n_blocks {
        let b_offset = blk * BLOCK;
        scales[blk] = f16_to_f32(u16::from_le_bytes([src[b_offset], src[b_offset + 1]]));
        let codes = &src[b_offset + 2..b_offset + BLOCK];
        #[cfg(target_arch = "aarch64")]
        decode_pq2_codes_neon(codes, &mut levels[blk * QK..(blk + 1) * QK]);
        #[cfg(not(target_arch = "aarch64"))]
        for j in 0..QK {
            let code = codes[j / 4] >> ((j % 4) * 2);
            levels[blk * QK + j] = (code & 3) as i8 - 1;
        }
    }
}

/// Codec-dispatched row decode for any packed RawMat (both prism codecs).
#[inline]
fn dequant_row_i8(m: &RawMat, src: &[u8], levels: &mut [i8], scales: &mut [f32]) {
    match m.ty {
        GGML_TYPE_PQ2_0 => dequant_pq2_i8(src, levels, scales),
        _ => dequant_trits_i8(src, levels, scales),
    }
}

/// Codec-dispatched row decode straight to f32 values (embedding-table path).
#[inline]
fn dequant_row_to_f32(ty: u32, n_in: usize, src: &[u8], out: &mut [f32]) {
    match ty {
        GGML_TYPE_PQ2_0 => {
            let qk = PQ2_0_BLOCK_VALUES;
            for c in 0..n_in / qk {
                let blk = &src[c * PQ2_0_BLOCK_BYTES..(c + 1) * PQ2_0_BLOCK_BYTES];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for (j, o) in out[c * qk..(c + 1) * qk].iter_mut().enumerate() {
                    let code = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
                    *o = (code as i8 - 1) as f32 * d;
                }
            }
        }
        _ => ptq1_0_dequant_row(src, out),
    }
}

/// Dot product of one trit row against one activation vector: per 128-value
/// band the raw trit·act partial is accumulated and then scaled by that
/// band's f16 scale — the same value set as `dot8(dequant_row, act)`, only a
/// different (band-structured) f32 summation order.
#[inline]
fn trit_row_dot(trits: &[i8], scales: &[f32], act: &[f32]) -> f32 {
    let blocks = scales.len();
    debug_assert_eq!(trits.len(), blocks * PTQ1_0_BLOCK_VALUES);
    let mut acc = 0.0f32;
    for c in 0..blocks {
        let col0 = c * PTQ1_0_BLOCK_VALUES;
        let mut bacc = 0.0f32;
        for i in col0..col0 + PTQ1_0_BLOCK_VALUES {
            bacc += trits[i] as f32 * act[i];
        }
        // Same two-op fold as the tile kernel's band step, bit for bit, so
        // the single-token GEMV and the tiled GEMM agree exactly.
        bacc *= scales[c];
        acc += bacc;
    }
    acc
}

// ── packed-domain int8 path (arm64 dotprod) ──────────────────────────
//
// Trits ∈ {-1,0,1} multiply int8-quantized activations exactly: per 128-value
// band, `iacc = Σ trit·act_q` is an exact i32 (|acc| ≤ 127·128 ≪ 2^31) and the
// band folds as w_scale · a_scale · iacc — the same two-sided per-group
// scaling GGUF Q8_0 uses, bounded by absmax quantization (~1/254 relative
// error worst case). Every token's activation is quantized once per call and
// reused across all weight rows, and the MAC stream goes through the dotprod
// `sdot` instruction (16 int8 MACs per instruction) instead of 4-lane f32.

fn dotprod_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static DOTPROD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *DOTPROD.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Signed int8 dot of two 128-byte groups: 8 × `vsdotq_s32` (16 MACs each)
/// on dotprod cores, plain i32 widening loop otherwise. Callers pass windows
/// of exactly PTQ1_0_BLOCK_VALUES bytes on both sides.
#[inline]
fn sdot_group(w: &[i8], a: &[i8]) -> i32 {
    debug_assert_eq!(w.len(), PTQ1_0_BLOCK_VALUES);
    debug_assert_eq!(a.len(), PTQ1_0_BLOCK_VALUES);
    #[cfg(target_arch = "aarch64")]
    {
        if dotprod_available() {
            // SAFETY: both slices are exactly 128 bytes (checked above) and
            // i8 is always properly aligned.
            return unsafe { sdot_128_neon(w.as_ptr(), a.as_ptr()) };
        }
    }
    let mut acc = 0i32;
    for k in 0..PTQ1_0_BLOCK_VALUES {
        acc += w[k] as i32 * a[k] as i32;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_128_neon(w: *const i8, a: *const i8) -> i32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    for k in 0..(PTQ1_0_BLOCK_VALUES / 16) {
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

/// 4-token variant of `sdot_128_neon`: the 128-byte weight band is loaded
/// once into registers and each of the four activation bands gets its own
/// sdot chain. Band i32 sums are order-independent, so this returns exactly
/// what four separate `sdot_group` calls return (each accumulator walks the
/// same lane pattern as the single-token kernel).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline(never)]
unsafe fn sdot_group_4_neon(
    w: *const i8,
    a0: *const i8,
    a1: *const i8,
    a2: *const i8,
    a3: *const i8,
) -> [i32; 4] {
    use std::arch::aarch64::*;
    let (mut acc0, mut acc1, mut acc2, mut acc3) =
        (vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0));
    let mut k = 0usize;
    while k < PTQ1_0_BLOCK_VALUES {
        let wv = vld1q_s8(w.add(k));
        let av0 = vld1q_s8(a0.add(k));
        let av1 = vld1q_s8(a1.add(k));
        let av2 = vld1q_s8(a2.add(k));
        let av3 = vld1q_s8(a3.add(k));
        std::arch::asm!(
            "sdot {acc0:v}.4s, {wv:v}.16b, {av0:v}.16b",
            "sdot {acc1:v}.4s, {wv:v}.16b, {av1:v}.16b",
            "sdot {acc2:v}.4s, {wv:v}.16b, {av2:v}.16b",
            "sdot {acc3:v}.4s, {wv:v}.16b, {av3:v}.16b",
            acc0 = inout(vreg) acc0,
            acc1 = inout(vreg) acc1,
            acc2 = inout(vreg) acc2,
            acc3 = inout(vreg) acc3,
            wv = in(vreg) wv,
            av0 = in(vreg) av0,
            av1 = in(vreg) av1,
            av2 = in(vreg) av2,
            av3 = in(vreg) av3,
            options(pure, nomem, nostack)
        );
        k += 16;
    }
    [
        vaddvq_s32(acc0),
        vaddvq_s32(acc1),
        vaddvq_s32(acc2),
        vaddvq_s32(acc3),
    ]
}

fn sdot_group_4(w: &[i8], a0: &[i8], a1: &[i8], a2: &[i8], a3: &[i8]) -> [i32; 4] {
    debug_assert_eq!(w.len(), PTQ1_0_BLOCK_VALUES);
    debug_assert_eq!(a0.len(), PTQ1_0_BLOCK_VALUES);
    debug_assert_eq!(a1.len(), PTQ1_0_BLOCK_VALUES);
    debug_assert_eq!(a2.len(), PTQ1_0_BLOCK_VALUES);
    debug_assert_eq!(a3.len(), PTQ1_0_BLOCK_VALUES);
    #[cfg(target_arch = "aarch64")]
    {
        if dotprod_available() {
            // SAFETY: all five slices are exactly 128 bytes (checked above).
            return unsafe {
                sdot_group_4_neon(w.as_ptr(), a0.as_ptr(), a1.as_ptr(), a2.as_ptr(), a3.as_ptr())
            };
        }
    }
    let mut out = [0i32; 4];
    for (k, wv) in w.iter().enumerate() {
        let wv = *wv as i32;
        out[0] += wv * a0[k] as i32;
        out[1] += wv * a1[k] as i32;
        out[2] += wv * a2[k] as i32;
        out[3] += wv * a3[k] as i32;
    }
    out
}

/// File-direct band dot for one packed PTQ1_0 block (24B qs + 2B qh + 2B f16
/// scale): the mixed-radix trits are expanded in NEON digit-planes and fed
/// straight into `sdot` against the activation band — no i8 slab round-trip.
/// The band's integer sum is order-independent (|iacc| <= 127*128 << 2^31),
/// so this yields the same i32 as `sdot_group` on the dequantized row.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn packed_block_dot_t143(block: *const u8, mut aq_band: *const i8) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let qs0 = vld1q_u8(block); // qs bytes [0..16) => o = n*16+m, m in 0..16
        let mut acc = vdupq_n_s32(0);
        let mut r = qs0; // r holds byte * 3^n (mod 256)
        for _n in 0..5u32 {
            // Expand both 8-lane halves: t = ((r16 * 3) >> 8) - 1 in u16,
            // saturate-narrowed to u8 (255 == -1 as i8).
            let lo = vmulq_u8(r, vdupq_n_u8(1)); // no-op split for clarity below
            let r_lo = vget_low_u8(lo);
            let r_hi = vget_high_u8(lo);
            let w16_lo = vshrq_n_u16(vmulq_u16(vmovl_u8(r_lo), vdupq_n_u16(3)), 8);
            let w16_hi = vshrq_n_u16(vmulq_u16(vmovl_u8(r_hi), vdupq_n_u16(3)), 8);
            let t_lo = vqmovn_u16(vsubq_u16(w16_lo, vdupq_n_u16(1)));
            let t_hi = vqmovn_u16(vsubq_u16(w16_hi, vdupq_n_u16(1)));
            let wv = vcombine_s8(vreinterpret_s8_u8(t_lo), vreinterpret_s8_u8(t_hi));
            let av = vld1q_s8(aq_band);
            std::arch::asm!(
                "sdot {acc:v}.4s, {wv:v}.16b, {av:v}.16b",
                acc = inout(vreg) acc,
                wv = in(vreg) wv,
                av = in(vreg) av,
                options(pure, nomem, nostack)
            );
            r = vmulq_u8(r, vdupq_n_u8(3)); // advance mixed-radix digit
            aq_band = aq_band.add(16);
        }
        // qs bytes [16..24): 8 lanes => o = 80 + n*8 + m, m in 0..8.
        // Zero-extend the 8-lane planes to 16 so `sdot` still applies.
        let qs1 = vld1_u8(block.add(16));
        let mut r2 = qs1;
        let zeros_u8 = vdup_n_u8(0);
        for _n in 0..5u32 {
            let r_lo2 = vmul_u8(r2, vdup_n_u8(1));
            let w16 = vshrq_n_u16(vmulq_u16(vmovl_u8(r_lo2), vdupq_n_u16(3)), 8);
            let t8 = vqmovn_u16(vsubq_u16(w16, vdupq_n_u16(1)));
            let wv8 = vreinterpret_s8_u8(t8);
            let av8 = vld1_s8(aq_band);
            let wv = vcombine_s8(wv8, vreinterpret_s8_u8(zeros_u8));
            let av = vcombine_s8(av8, vreinterpret_s8_u8(zeros_u8));
            std::arch::asm!(
                "sdot {acc:v}.4s, {wv:v}.16b, {av:v}.16b",
                acc = inout(vreg) acc,
                wv = in(vreg) wv,
                av = in(vreg) av,
                options(pure, nomem, nostack)
            );
            r2 = vmul_u8(r2, vdup_n_u8(3));
            aq_band = aq_band.add(8);
        }
        // qh: 2 bytes x 4 digits => o = 120 + n*2 + h.
        let mut acc_i = vaddvq_s32(acc);
        for n in 0..4usize {
            for h in 0..2usize {
                let b = block.add(24 + h).read();
                let q = b.wrapping_mul(POW3_PACKET[n]);
                let xi = ((q as u16) * 3) >> 8;
                let t = (xi as i16 - 1) as i8;
                acc_i += t as i32 * aq_band.read() as i32;
                aq_band = aq_band.add(1);
            }
        }
        acc_i
    }
}

const POW3_PACKET: [u8; 4] = [1, 3, 9, 27];

/// Whole-row file-direct dot over packed blocks. Fold mirrors
/// `trit_row_dot_q` exactly (wscale * ascale * iacc as f32), and the block
/// scale is read from the block's trailing f16 — the same value
/// `dequant_trits_i8` would produce — so results are bit-identical.
#[cfg(target_arch = "aarch64")]
unsafe fn packed_row_dot_t143(row: &[u8], aq: &ActQ, arow: usize, n_in: usize) -> f32 {
    let blocks = n_in / PTQ1_0_BLOCK_VALUES;
    debug_assert_eq!(row.len(), blocks * PTQ1_0_ROW_BLOCK_BYTES);
    let mut acc = 0.0f32;
    for c in 0..blocks {
        let blk = &row[c * 28..c * 28 + 28];
        let scale = f16_to_f32(u16::from_le_bytes([blk[26], blk[27]]));
        let iacc = unsafe {
            packed_block_dot_t143(
                blk.as_ptr(),
                aq.q[arow * n_in + c * PTQ1_0_BLOCK_VALUES..].as_ptr(),
            )
        };
        acc += scale * aq.scales[arow * blocks + c] * iacc as f32;
    }
    acc
}

/// Whole-row file-direct dot over PQ2_0 (type 142) blocks: each 128-value
/// band's codes decode into a stack level buffer and run the identical
/// `sdot_group` fold the slab paths use, so file-direct and slab results are
/// bit-identical. The scale rides block-first (`d` at bytes 0..2).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn packed_row_dot_t142(row: &[u8], aq: &ActQ, arow: usize, n_in: usize) -> f32 {
    let blocks = n_in / PQ2_0_BLOCK_VALUES;
    debug_assert_eq!(row.len(), blocks * PQ2_0_BLOCK_BYTES);
    let mut acc = 0.0f32;
    let band_base = aq.q.as_ptr().add(arow * n_in);
    if dotprod_available() {
        for c in 0..blocks {
            let blk = row.as_ptr().add(c * PQ2_0_BLOCK_BYTES);
            let scale = f16_to_f32(u16::from_le_bytes([*blk, *blk.add(1)]));
            let iacc = unsafe {
                p2_block_dot_neon(blk.add(2), band_base.add(c * PQ2_0_BLOCK_VALUES))
            };
            acc += scale * *aq.scales.get_unchecked(arow * blocks + c) * iacc as f32;
        }
        return acc;
    }
    let mut levels = [0i8; PQ2_0_BLOCK_VALUES];
    for c in 0..blocks {
        let blk = &row[c * PQ2_0_BLOCK_BYTES..(c + 1) * PQ2_0_BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let codes = &blk[2..];
        #[cfg(target_arch = "aarch64")]
        decode_pq2_codes_neon(codes, &mut levels);
        #[cfg(not(target_arch = "aarch64"))]
        for (j, lv) in levels.iter_mut().enumerate() {
            let code = codes[j / 4] >> ((j % 4) * 2);
            *lv = (code & 3) as i8 - 1;
        }
        let band = &aq.q[arow * n_in + c * PQ2_0_BLOCK_VALUES..][..PQ2_0_BLOCK_VALUES];
        let iacc = sdot_group(&levels, band);
        acc += scale * aq.scales[arow * blocks + c] * iacc as f32;
    }
    acc
}

/// Codec-dispatched whole-row file-direct dot.
#[cfg(target_arch = "aarch64")]
unsafe fn packed_row_dot(m: &RawMat, row: &[u8], aq: &ActQ, arow: usize, n_in: usize) -> f32 {
    match m.ty {
        GGML_TYPE_PQ2_0 => unsafe { packed_row_dot_t142(row, aq, arow, n_in) },
        _ => unsafe { packed_row_dot_t143(row, aq, arow, n_in) },
    }
}

/// A token activation set quantized per PTQ1_0 band: `q[row * n_in + i]` is
/// the signed int8 code, `scales[row * blocks + c] = band_absmax / 127`.
struct ActQ {
    q: Vec<i8>,
    scales: Vec<f32>,
}

fn quantize_acts(acts: &[f32], n_tok: usize, n_in: usize, blocks: usize) -> ActQ {
    debug_assert_eq!(acts.len(), n_tok * n_in);
    let mut q = vec![0i8; n_tok * n_in];
    let mut scales = vec![1.0f32; n_tok * blocks];
    for t in 0..n_tok {
        let row = &acts[t * n_in..(t + 1) * n_in];
        let qrow = &mut q[t * n_in..(t + 1) * n_in];
        for c in 0..blocks {
            let band = &row[c * PTQ1_0_BLOCK_VALUES..(c + 1) * PTQ1_0_BLOCK_VALUES];
            let mut absmax = 0.0f32;
            for &v in band {
                absmax = absmax.max(v.abs());
            }
            let srow = t * blocks + c;
            if absmax == 0.0 || !absmax.is_finite() {
                scales[srow] = 1.0;
                continue;
            }
            let scale = absmax / 127.0;
            scales[srow] = scale;
            for (qb, &v) in qrow[c * PTQ1_0_BLOCK_VALUES..(c + 1) * PTQ1_0_BLOCK_VALUES]
                .iter_mut()
                .zip(band.iter())
            {
                let x = (v / scale).round();
                *qb = x.clamp(-127.0, 127.0) as i8;
            }
        }
    }
    ActQ { q, scales }
}

/// Single-row dot with a per-band-quantized activation: the sdot counterpart
/// of `trit_row_dot` with the identical band-fold order.
#[inline]
fn trit_row_dot_q(trits: &[i8], wscales: &[f32], actq: &ActQ, arow: usize, n_in: usize) -> f32 {
    let blocks = wscales.len();
    let mut acc = 0.0f32;
    for c in 0..blocks {
        let col0 = c * PTQ1_0_BLOCK_VALUES;
        let w = &trits[col0..col0 + PTQ1_0_BLOCK_VALUES];
        let a = &actq.q[arow * n_in + col0..arow * n_in + col0 + PTQ1_0_BLOCK_VALUES];
        let iacc = sdot_group(w, a);
        acc += wscales[c] * actq.scales[arow * blocks + c] * iacc as f32;
    }
    acc
}

/// Register-tiled PTQ1_0 GEMM over the int8 path: activations quantized per
/// band once per call, rows dequantized once to the trit slab, the MAC stream
/// through `sdot_group`. Same band/fold order as `trit_row_dot_q`, so tiled
/// prefill and single-token decode agree bit for bit on this path too.
fn gemm_rows_t143_sdot(
    data: &[u8],
    m: &RawMat,
    n_tok: usize,
    acts: &[f32],
    r_off: usize,
    rows: usize,
) -> Vec<f32> {
    const TR: usize = 4;
    const TT: usize = 8;
    let n_in = m.n_in;
    let blocks = n_in / PTQ1_0_BLOCK_VALUES;
    debug_assert!(blocks > 0 && blocks * PTQ1_0_BLOCK_VALUES == n_in);
    let aq = quantize_acts(acts, n_tok, n_in, blocks);
    let mut res = vec![0.0f32; rows * n_tok];
    let rows_per_pass = (GEMM_SLAB_BYTES / n_in).clamp(1, rows.max(1));
    let mut slab_t = vec![0i8; rows_per_pass * n_in];
    let mut slab_d = vec![0.0f32; rows_per_pass * blocks];
    let mut base = 0usize;
    while base < rows {
        let take = rows_per_pass.min(rows - base);
        for i in 0..take {
            dequant_row_i8(m, 
                m.row_slice(data, r_off + base + i),
                &mut slab_t[i * n_in..(i + 1) * n_in],
                &mut slab_d[i * blocks..(i + 1) * blocks],
            );
        }
        for r in 0..take {
            let row = &slab_t[r * n_in..(r + 1) * n_in];
            let rowd = &slab_d[r * blocks..(r + 1) * blocks];
            for t in 0..n_tok {
                let v = trit_row_dot_q(row, rowd, &aq, t, n_in);
                res[(base + r) * n_tok + t] = v;
            }
        }
        base += take;
    }
    res
}

/// GEMV against a PTQ1_0 matrix still in the mmap: dequantize each row into
/// trit bytes plus band scales (quarter the traffic of an f32 slab) and dot
/// with the prepared activation via `trit_row_dot`. Jobs at or above
/// PAR_MAC_THRESHOLD split the output rows into contiguous deterministic
/// chunks via `std::thread::scope`; each worker allocates its own row
/// scratch (two small vecs per call, noise next to a multi-second decode
/// step). Every row runs the same dequantize-then-band-dot sequence, so
/// results are thread-count independent.
/// Lazy dequantized weight slab for PTQ1_0 matrices. Decoding re-runs the
/// NEON trit digit-expansion over every weight byte on every token; with the
/// slab built once (SOFUU_T143_SLAB=1) the per-token sweep is a plain sdot
/// stream over i8 trits through `trit_row_dot_q` — the same pinned reference
/// fold the prefill path uses, so decode results are bit-identical to the
/// file-direct path. A 27B slab is ~2.9 GB anonymous vs 5.9 GB for the packed
/// residency copy, and the packed pages stay file-backed (evictable).
struct SlabT143 {
    trits: Vec<i8>, // n_out * n_in
    rowd: Vec<f32>, // n_out * blocks band scales
}

static T143_SLABS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), std::sync::Arc<SlabT143>>>,
> = std::sync::OnceLock::new();

fn t143_slab_enabled() -> bool {
    std::env::var_os("SOFUU_T143_SLAB").is_some_and(|v| v != "0")
}

fn ensure_t143_slab(data: &[u8], m: &RawMat) -> Option<std::sync::Arc<SlabT143>> {
    if m.n_in % PTQ1_0_BLOCK_VALUES != 0 || m.n_out == 0 {
        return None;
    }
    let key = (m.off, m.n_in, m.n_out);
    let map = T143_SLABS
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(guard) = map.lock() {
        if let Some(s) = guard.get(&key) {
            return Some(s.clone());
        }
    }
    let blocks = m.n_in / PTQ1_0_BLOCK_VALUES;
    let n_in = m.n_in;
    let rows = m.n_out;
    let mut trits = vec![0i8; rows * m.n_in];
    let mut rowd = vec![0.0f32; rows * blocks];
    let threads = thread_budget().min(rows.max(1));
    // Row ranges are disjoint per worker — same split as gemv_t143's dot loop.
    std::thread::scope(|scope| {
        let mut rest_t: &mut [i8] = trits.as_mut_slice();
        let mut rest_d: &mut [f32] = rowd.as_mut_slice();
        for t in 0..threads.max(1) {
            let lo = rows * t / threads.max(1);
            let hi = rows * (t + 1) / threads.max(1);
            let take = hi - lo;
            if take == 0 {
                continue;
            }
            let tmp_t = rest_t;
            let (chunk_t, tail_t) = tmp_t.split_at_mut(take * n_in);
            rest_t = tail_t;
            let tmp_d = rest_d;
            let (chunk_d, tail_d) = tmp_d.split_at_mut(take * blocks);
            rest_d = tail_d;
            scope.spawn(move || {
                for i in 0..take {
                    let r = lo + i;
                    dequant_row_i8(m, 
                        m.row_slice(data, r),
                        &mut chunk_t[i * n_in..(i + 1) * n_in],
                        &mut chunk_d[i * blocks..(i + 1) * blocks],
                    );
                }
            });
        }
    });
    let arc = std::sync::Arc::new(SlabT143 { trits, rowd });
    if let Ok(mut guard) = map.lock() {
        guard.insert(key, arc.clone());
    }
    Some(arc)
}

fn gemv_t143(data: &[u8], m: &RawMat, act: &[f32], out: &mut [f32]) {
    let _p = prof::Scope::new(&prof::GLUE_NS, &prof::GLUE_CALLS);
    let n_in = m.n_in;
    let blocks = n_in / PTQ1_0_BLOCK_VALUES;
    debug_assert!(blocks > 0 && blocks * PTQ1_0_BLOCK_VALUES == n_in);
    let n_out = out.len();
    let use_sdot = dotprod_available();
    let empty = ActQ { q: Vec::new(), scales: Vec::new() };
    // Quantize the activation once (one row) and share it across workers.
    let aq = if use_sdot {
        quantize_acts(&act[..n_in], 1, n_in, blocks)
    } else {
        empty
    };
    let threads = if n_out * n_in >= PAR_MAC_THRESHOLD {
        thread_budget().min(n_out)
    } else {
        1
    };
    // SOFUU_T143_SLAB: dequantize each matrix once, then decode as a plain
    // i8 trit sdot sweep (bit-identical fold; saves the per-token digit-gen).
    if t143_slab_enabled() && n_out * n_in >= PAR_MAC_THRESHOLD {
        if let Some(slab) = ensure_t143_slab(data, m) {
            let act = &act[..n_in];
            std::thread::scope(|scope| {
                let mut rest: &mut [f32] = out;
                let mut r0 = 0usize;
                for t in 0..threads.max(1) {
                    let take = n_out * (t + 1) / threads.max(1) - r0;
                    let tmp = rest;
                    let (chunk, tail) = tmp.split_at_mut(take);
                    rest = tail;
                    let rows = r0;
                    let slab_ref = slab.clone();
                    let aq_ref = &aq;
                    scope.spawn(move || {
                        for (i, o) in chunk.iter_mut().enumerate() {
                            let r = rows + i;
                            let row = &slab_ref.trits[r * n_in..(r + 1) * n_in];
                            let rowd = &slab_ref.rowd[r * blocks..(r + 1) * blocks];
                            *o = if use_sdot {
                                trit_row_dot_q(row, rowd, aq_ref, 0, n_in)
                            } else {
                                trit_row_dot(row, rowd, act)
                            };
                        }
                    });
                    r0 += take;
                }
            });
            return;
        }
    }
    if threads <= 1 {
        if use_sdot {
            for (r, o) in out.iter_mut().enumerate() {
                *o = unsafe { packed_row_dot(m, m.row_slice(data, r), &aq, 0, n_in) };
            }
            return;
        }
        let mut trits = vec![0i8; n_in];
        let mut rowd = vec![0.0f32; blocks];
        let act = &act[..n_in];
        for (r, o) in out.iter_mut().enumerate() {
            dequant_row_i8(m, m.row_slice(data, r), &mut trits, &mut rowd);
            *o = trit_row_dot(&trits, &rowd, act);
        }
        return;
    }
    let act = &act[..n_in];
    std::thread::scope(|scope| {
        let mut rest: &mut [f32] = out;
        let mut r0 = 0usize;
        for t in 0..threads {
            let take = n_out * (t + 1) / threads - r0;
            let tmp = rest;
            let (chunk, tail) = tmp.split_at_mut(take);
            rest = tail;
            let rows = r0;
            let aq_ref = &aq;
            scope.spawn(move || {
                if use_sdot {
                    for (i, o) in chunk.iter_mut().enumerate() {
                        *o = unsafe { packed_row_dot(m, m.row_slice(data, rows + i), aq_ref, 0, n_in) };
                    }
                    return;
                }
                let mut trits = vec![0i8; n_in];
                let mut rowd = vec![0.0f32; blocks];
                for (i, o) in chunk.iter_mut().enumerate() {
                    dequant_row_i8(m, m.row_slice(data, rows + i), &mut trits, &mut rowd);
                    *o = trit_row_dot(&trits, &rowd, act);
                }
            });
            r0 += take;
        }
    });
}

/// Register-tiled PTQ1_0 GEMM core: weight rows [r_off, r_off + rows) are
/// dequantized exactly once each (trit bytes + band scales into per-pass
/// slabs bounded by GEMM_SLAB_BYTES) and every row is dotted, one
/// 128-value band at a time, against every token's activation from a
/// 4-row × 8-token register tile. The old per-(row, token) `dot8` sweep
/// re-read the whole f32 slab and the whole activation set for every token;
/// the tile loads the slab once per 8 tokens and the activations once per
/// 4 rows, and keeps 32 independent FMA chains in registers. Value set is
/// unchanged (trit·d == dequant value); only the f32 summation order
/// differs from the scalar path, so results match to rounding, not bitwise.
fn gemm_rows_t143(
    data: &[u8],
    m: &RawMat,
    n_tok: usize,
    acts: &[f32],
    r_off: usize,
    rows: usize,
) -> Vec<f32> {
    const TR: usize = 4;
    const TT: usize = 8;
    let n_in = m.n_in;
    let blocks = n_in / PTQ1_0_BLOCK_VALUES;
    debug_assert!(blocks > 0 && blocks * PTQ1_0_BLOCK_VALUES == n_in);
    let mut res = vec![0.0f32; rows * n_tok];
    let rows_per_pass = (GEMM_SLAB_BYTES / n_in).clamp(1, rows.max(1));
    let mut slab_t = vec![0i8; rows_per_pass * n_in];
    let mut slab_d = vec![0.0f32; rows_per_pass * blocks];
    let mut base = 0usize;
    while base < rows {
        let take = rows_per_pass.min(rows - base);
        for i in 0..take {
            dequant_row_i8(m, 
                m.row_slice(data, r_off + base + i),
                &mut slab_t[i * n_in..(i + 1) * n_in],
                &mut slab_d[i * blocks..(i + 1) * blocks],
            );
        }
        let rows_full = take - take % TR;
        let toks_full = n_tok - n_tok % TT;
        for rb in (0..rows_full).step_by(TR) {
            for tb in (0..toks_full).step_by(TT) {
                let mut acc = [0.0f32; TR * TT]; // acc[t * TR + r]
                for c in 0..blocks {
                    let col0 = c * PTQ1_0_BLOCK_VALUES;
                    let col1 = col0 + PTQ1_0_BLOCK_VALUES;
                    // Iterator windows, not indexed access: the zips let the
                    // optimizer drop every per-element bounds check, which is
                    // what kept the inner loop scalar in earlier drafts.
                    let w0 = &slab_t[rb * n_in + col0..rb * n_in + col1];
                    let w1 = &slab_t[(rb + 1) * n_in + col0..(rb + 1) * n_in + col1];
                    let w2 = &slab_t[(rb + 2) * n_in + col0..(rb + 2) * n_in + col1];
                    let w3 = &slab_t[(rb + 3) * n_in + col0..(rb + 3) * n_in + col1];
                    let d = [
                        slab_d[rb * blocks + c],
                        slab_d[(rb + 1) * blocks + c],
                        slab_d[(rb + 2) * blocks + c],
                        slab_d[(rb + 3) * blocks + c],
                    ];
                    for t in 0..TT {
                        let arow = &acts[(tb + t) * n_in + col0..(tb + t) * n_in + col1];
                        let (mut b0, mut b1, mut b2, mut b3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        for (((av, t0), t1), (t2, t3)) in arow
                            .iter()
                            .zip(w0.iter())
                            .zip(w1.iter())
                            .zip(w2.iter().zip(w3.iter()))
                        {
                            b0 += *t0 as f32 * *av;
                            b1 += *t1 as f32 * *av;
                            b2 += *t2 as f32 * *av;
                            b3 += *t3 as f32 * *av;
                        }
                        // Same two-op band fold as trit_row_dot, bit for bit.
                        acc[t * TR] += d[0] * b0;
                        acc[t * TR + 1] += d[1] * b1;
                        acc[t * TR + 2] += d[2] * b2;
                        acc[t * TR + 3] += d[3] * b3;
                    }
                }
                for t in 0..TT {
                    for r in 0..TR {
                        res[(base + rb + r) * n_tok + tb + t] = acc[t * TR + r];
                    }
                }
            }
            // tail tokens for this row tile
            for r in 0..TR {
                let row = &slab_t[(rb + r) * n_in..(rb + r + 1) * n_in];
                let rowd = &slab_d[(rb + r) * blocks..(rb + r + 1) * blocks];
                for t in toks_full..n_tok {
                    let v = trit_row_dot(row, rowd, &acts[t * n_in..(t + 1) * n_in]);
                    res[(base + rb + r) * n_tok + t] = v;
                }
            }
        }
        // tail rows for every token
        for r in rows_full..take {
            let row = &slab_t[r * n_in..(r + 1) * n_in];
            let rowd = &slab_d[r * blocks..(r + 1) * blocks];
            for t in 0..n_tok {
                let v = trit_row_dot(row, rowd, &acts[t * n_in..(t + 1) * n_in]);
                res[(base + r) * n_tok + t] = v;
            }
        }
        base += take;
    }
    res
}

fn direct_tile_enabled() -> bool {
    // DEFAULT ON: bit-identical to the sdot slab path (pinned by
    // direct_tile_bitmatches_sdot_path) and 2x+ faster. `=0` restores the
    // old per-(row, token) runner.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SOFUU_T143_DIRECT").as_deref() != Ok("0"))
}

/// Phase 4a: token-tiled sdot GEMM over the trit slab. Two changes vs
/// `gemm_rows_t143_sdot`, both invisible in output:
/// (1) every weight band is loaded once and sdot'ed against a 4-token tile of
///     activation bands (`sdot_group_4`) instead of one full
///     `trit_row_dot_q` walk per (row, token) — the per-band call overhead
///     (slice bounds, dotprod re-check, re-loads) was paid ~rows×tokens×bands
///     times before;
/// (2) the dequant slab is sized for the SHARED P-cluster L2 (~2 MiB per
///     worker) instead of 8 MiB — at 6 workers the old slab re-read every
///     token only from L3/DRAM, which dominated the whole prefill.
/// Fold order per band (`wscale * ascale * iacc`, bands ascending) is the
/// same as `trit_row_dot_q`, so results are bit-identical on the dotprod
/// path (pinned by `direct_tile_bitmatches_sdot_path`).
fn gemm_rows_t143_direct(
    data: &[u8],
    m: &RawMat,
    n_tok: usize,
    acts: &[f32],
    r_off: usize,
    rows: usize,
) -> Vec<f32> {
    const TT: usize = 4; // activation bands of the tile stay in L1
    const L2_TILE_BYTES: usize = 2 << 20;
    let n_in = m.n_in;
    let blocks = n_in / PTQ1_0_BLOCK_VALUES;
    debug_assert!(blocks > 0 && blocks * PTQ1_0_BLOCK_VALUES == n_in);
    let aq = quantize_acts(acts, n_tok, n_in, blocks);
    let mut res = vec![0.0f32; rows * n_tok];
    let rows_per_pass = (L2_TILE_BYTES / n_in).clamp(1, rows.max(1));
    let mut slab_t = vec![0i8; rows_per_pass * n_in];
    let mut slab_d = vec![0.0f32; rows_per_pass * blocks];
    let toks_full = n_tok - n_tok % TT;
    let mut base = 0usize;
    while base < rows {
        let take = rows_per_pass.min(rows - base);
        for i in 0..take {
            dequant_row_i8(m, 
                m.row_slice(data, r_off + base + i),
                &mut slab_t[i * n_in..(i + 1) * n_in],
                &mut slab_d[i * blocks..(i + 1) * blocks],
            );
        }
        for r in 0..take {
            let row = &slab_t[r * n_in..(r + 1) * n_in];
            let rowd = &slab_d[r * blocks..(r + 1) * blocks];
            let mut t = 0usize;
            while t < toks_full {
                let mut acc = [0.0f32; TT];
                for c in 0..blocks {
                    let col0 = c * PTQ1_0_BLOCK_VALUES;
                    let w = &row[col0..col0 + PTQ1_0_BLOCK_VALUES];
                    let band_base = t * n_in + col0;
                    let iacc = sdot_group_4(
                        w,
                        &aq.q[band_base..band_base + PTQ1_0_BLOCK_VALUES],
                        &aq.q[band_base + n_in..band_base + n_in + PTQ1_0_BLOCK_VALUES],
                        &aq.q[band_base + 2 * n_in..band_base + 2 * n_in + PTQ1_0_BLOCK_VALUES],
                        &aq.q[band_base + 3 * n_in..band_base + 3 * n_in + PTQ1_0_BLOCK_VALUES],
                    );
                    acc[0] += rowd[c] * aq.scales[t * blocks + c] * iacc[0] as f32;
                    acc[1] += rowd[c] * aq.scales[(t + 1) * blocks + c] * iacc[1] as f32;
                    acc[2] += rowd[c] * aq.scales[(t + 2) * blocks + c] * iacc[2] as f32;
                    acc[3] += rowd[c] * aq.scales[(t + 3) * blocks + c] * iacc[3] as f32;
                }
                for j in 0..TT {
                    res[(base + r) * n_tok + t + j] = acc[j];
                }
                t += TT;
            }
            for t in toks_full..n_tok {
                let v = trit_row_dot_q(row, rowd, &aq, t, n_in);
                res[(base + r) * n_tok + t] = v;
            }
        }
        base += take;
    }
    res
}

/// Batched GEMM against a PTQ1_0 matrix: `acts` holds `n_tok` contiguous
/// activation rows of m.n_in f32 and out receives, token-major,
/// out[t * m.n_out + r] = dot8(dequant(m.row r), acts[t..t + n_in]). Each
/// weight row is dequantized once per call (amortized over the token chunk
/// instead of per token); rows split into contiguous deterministic thread
/// chunks exactly like gemv_t143, so prefill results are bit-identical to
/// the single-token path. Nothing here writes shared mutable state — the
/// only outputs are the per-thread tiles scattered into `out` after join.
fn gemm_t143(data: &[u8], m: &RawMat, n_tok: usize, acts: &[f32], out: &mut [f32]) {
    if n_tok == 0 {
        debug_assert!(out.is_empty());
        return;
    }
    debug_assert_eq!(acts.len(), n_tok * m.n_in);
    debug_assert_eq!(out.len(), n_tok * m.n_out);
    let rows = m.n_out;
    let use_sdot = dotprod_available();
    let threads = if rows * m.n_in * n_tok >= PAR_MAC_THRESHOLD {
        thread_budget().min(rows)
    } else {
        1
    };
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for t in 0..threads {
            let r0 = rows * t / threads;
            let r1 = rows * (t + 1) / threads;
            handles.push((r0, scope.spawn(move || {
                if use_sdot {
                    if direct_tile_enabled() && n_tok >= 8 {
                        gemm_rows_t143_direct(data, m, n_tok, acts, r0, r1 - r0)
                    } else {
                        gemm_rows_t143_sdot(data, m, n_tok, acts, r0, r1 - r0)
                    }
                } else {
                    gemm_rows_t143(data, m, n_tok, acts, r0, r1 - r0)
                }
            })));
        }
        for (r0, h) in handles {
            let tile = h.join().expect("qwen35 gemm worker panicked");
            for (i, srow) in tile.chunks_exact(n_tok).enumerate() {
                let r = r0 + i;
                for (tv, &v) in srow.iter().enumerate() {
                    out[tv * rows + r] = v;
                }
            }
        }
    });
}

/// GEMV against a small dense f32 weight (row-major [n_out][n_in]).
fn gemv_f32(w: &[f32], n_in: usize, act: &[f32], out: &mut [f32]) {
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot8(&w[r * n_in..(r + 1) * n_in], &act[..n_in]);
    }
}

/// One gated-delta-net recurrence step for one v-head.
/// State S is contiguous [q/k dim i][v dim j] at i*k_dim+j.
#[inline]
fn gdn_head_step(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, o: &mut [f32]) {
    const D: usize = 128;
    debug_assert_eq!(s.len(), D * D);
    // S ← S·exp(g)
    let decay = g.exp();
    for e in s.iter_mut() {
        *e *= decay;
    }
    // u[j] = Σ_i S[i][j]·k[i]  (accumulate row-major for contiguous access)
    let mut u = [0.0f32; D];
    for (i, &ki) in k.iter().enumerate() {
        if ki != 0.0 {
            let row = &s[i * D..(i + 1) * D];
            for (uj, &si) in u.iter_mut().zip(row.iter()) {
                *uj += si * ki;
            }
        }
    }
    // delta[j] = beta·(v[j] − u[j])
    let mut delta = [0.0f32; D];
    for j in 0..D {
        delta[j] = beta * (v[j] - u[j]);
    }
    // S[i][j] += k[i]·delta[j]; out[j] = (Σ_i S[i][j]·q[i]) / sqrt(128) — the
    // output reads the post-update state.
    for e in o.iter_mut() {
        *e = 0.0;
    }
    let scale = 1.0 / (D as f32).sqrt();
    for (i, &ki) in k.iter().enumerate() {
        let row = &mut s[i * D..(i + 1) * D];
        if ki != 0.0 {
            for (sj, &dj) in row.iter_mut().zip(delta.iter()) {
                *sj += ki * dj;
            }
        }
        let qi = q[i];
        for (oj, &sj) in o.iter_mut().zip(row.iter()) {
            *oj += sj * qi;
        }
    }
    for e in o.iter_mut() {
        *e *= scale;
    }
}

/// Partial multi-modal RoPE with text-only positions (all four section cursors
/// equal to `pos`): IMRoPE sector selection as in the reference engine, then a
/// split-half rotation of pairs (d, d + n_rot/2) inside the first n_rot dims;
/// dims ≥ n_rot pass through untouched.
fn rope_imrope(head: &mut [f32], n_rot: usize, pos: usize, base: f32, sections: &[usize; 4]) {
    let pos_f = pos as f32;
    let (pos_t, pos_h, pos_w, pos_e) = (pos_f, pos_f, pos_f, pos_f);
    let pairs = n_rot / 2;
    let sect_dims: usize = sections.iter().sum();
    for j in 0..pairs {
        // IMRoPE sector rule (ggml mrope cache init, is_imrope branch).
        let sector = j % sect_dims;
        let p = if sector % 3 == 1 && sector < 3 * sections[1] {
            pos_h
        } else if sector % 3 == 2 && sector < 3 * sections[2] {
            pos_w
        } else if sector % 3 == 0 && sector < 3 * sections[0] {
            pos_t
        } else {
            pos_e
        };
        let theta = p * base.powf(-2.0 * j as f32 / n_rot as f32);
        let (sin, cos) = theta.sin_cos();
        let d = j;
        let e = j + pairs;
        let a = head[d];
        let b = head[e];
        head[d] = a * cos - b * sin;
        head[e] = a * sin + b * cos;
    }
}

// ── GGUF loading helpers ─────────────────────────────────────────────

fn read_f32(data: &[u8], name: &str, off: usize, n: usize) -> Result<Vec<f32>, String> {
    let end = off.checked_add(n * 4).ok_or_else(|| format!("tensor {name}: size overflow"))?;
    if end > data.len() {
        return Err(format!("tensor {name}: F32 data out of range"));
    }
    Ok(data[off..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_bf16_mat(data: &[u8], name: &str, off: usize, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let n = rows.checked_mul(cols).ok_or_else(|| format!("tensor {name}: size overflow"))?;
    let end = off.checked_add(n * 2).ok_or_else(|| format!("tensor {name}: size overflow"))?;
    if end > data.len() {
        return Err(format!("tensor {name}: BF16 data out of range"));
    }
    Ok(data[off..end]
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn row_bytes_ptq1_0(n_in: usize) -> usize {
    n_in.div_ceil(PTQ1_0_BLOCK_VALUES) * PTQ1_0_BLOCK_BYTES
}

fn raw_mat_of(data: &[u8], gguf: &GgufFile, name: &str, xform: Xform, expect_type: u32) -> Result<RawMat, String> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing Qwen35 tensor: {name}"))?;
    let ty = info.tensor_type;
    // PTQ1_0-expecting calls accept either prism codec (same geometry class);
    // everything else validates exactly.
    let ty_ok = match expect_type {
        GGML_TYPE_PTQ1_0 => ty == GGML_TYPE_PTQ1_0 || ty == GGML_TYPE_PQ2_0,
        t => ty == t,
    };
    if !ty_ok {
        return Err(format!(
            "tensor {name}: unexpected GGUF type {} (expected {expect_type})",
            info.tensor_type
        ));
    }
    if info.shape.len() != 2 {
        return Err(format!("tensor {name}: expected 2D, got shape {:?}", info.shape));
    }
    let n_in = info.shape[0];
    let n_out = info.shape[1];
    let row_bytes = match ty {
        GGML_TYPE_PTQ1_0 => row_bytes_ptq1_0(n_in),
        GGML_TYPE_PQ2_0 => n_in.div_ceil(PQ2_0_BLOCK_VALUES) * PQ2_0_BLOCK_BYTES,
        GGML_TYPE_BF16 => n_in * 2,
        _ => n_in * 4,
    };
    let off = gguf.data_offset + info.offset;
    let total = row_bytes
        .checked_mul(n_out)
        .ok_or_else(|| format!("tensor {name}: byte size overflow"))?;
    if off + total > data.len() {
        return Err(format!("tensor {name}: data out of mmap range"));
    }
    Ok(RawMat { off, n_in, n_out, row_bytes, ty, xform })
}

fn f32_meta_vec(gguf: &GgufFile, data: &[u8], name: &str) -> Result<Vec<f32>, String> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing Qwen35 tensor: {name}"))?;
    if info.tensor_type != 0 {
        return Err(format!("tensor {name}: expected F32, got type {}", info.tensor_type));
    }
    let n: usize = info.shape.iter().product();
    read_f32(data, name, gguf.data_offset + info.offset, n)
}

fn bf16_meta_mat(gguf: &GgufFile, data: &[u8], name: &str, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("missing Qwen35 tensor: {name}"))?;
    if info.tensor_type != GGML_TYPE_BF16 && info.tensor_type != 0 {
        return Err(format!(
            "tensor {name}: expected BF16 or F32, got type {}",
            info.tensor_type
        ));
    }
    let n: usize = info.shape.iter().product();
    if n != rows * cols {
        return Err(format!("tensor {name}: shape {:?} ≠ [{cols},{rows}]", info.shape));
    }
    if info.tensor_type == 0 {
        // the 9B ships the same small dense tensors as F32 (27B = BF16).
        let off = gguf.data_offset + info.offset;
        let end = off
            .checked_add(n * 4)
            .ok_or_else(|| format!("tensor {name}: size overflow"))?;
        if end > data.len() {
            return Err(format!("tensor {name}: F32 data out of range"));
        }
        return Ok(data[off..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect());
    }
    read_bf16_mat(data, name, gguf.data_offset + info.offset, rows, cols)
}

fn meta_string_list(gguf: &GgufFile, key: &str) -> Vec<String> {
    match gguf.metadata.get(key) {
        Some(GgufValue::StringArray(v)) => v.clone(),
        Some(GgufValue::Array(v)) => v
            .iter()
            .filter_map(|x| match x {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn meta_i32_list(gguf: &GgufFile, key: &str) -> Vec<i32> {
    match gguf.metadata.get(key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .filter_map(|x| match x {
                GgufValue::Int32(v) => Some(*v),
                GgufValue::Uint32(v) => Some(*v as i32),
                GgufValue::Int64(v) => Some(*v as i32),
                GgufValue::Uint64(v) => Some(*v as i32),
                GgufValue::Int8(v) => Some(*v as i32),
                GgufValue::Uint8(v) => Some(*v as i32),
                GgufValue::Float32(v) => Some(*v as i32),
                _ => None,
            })
            .collect(),
        Some(GgufValue::RawArray { item_type: 5, len, data }) => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .take(*len)
            .collect(),
        _ => Vec::new(),
    }
}

impl Qwen35Model {
    pub fn from_gguf(mut gguf: GgufFile, max_seq_len: usize) -> Result<Self, String> {
        // ── hyperparameters (read from KVs; spec defaults as fallback) ──
        let dim = gguf.get_u32("qwen35.embedding_length").unwrap_or(5120) as usize;
        let n_layers = gguf.get_u32("qwen35.block_count").unwrap_or(64) as usize;
        // MTP/nextn checkpoints pack the draft layer(s) as the last blk.N —
        // they are Q8_0 planned for speculative decode, not part of the
        // single-token main model. Skip them everywhere.
        let nextn_layers = gguf.get_u32("qwen35.nextn_predict_layers").unwrap_or(0) as usize;
        let n_main_layers = n_layers.saturating_sub(nextn_layers);
        let n_heads = gguf.get_u32("qwen35.attention.head_count").unwrap_or(24) as usize;
        let n_kv_heads = gguf.get_u32("qwen35.attention.head_count_kv").unwrap_or(4) as usize;
        let head_dim = gguf.get_u32("qwen35.attention.key_length").unwrap_or(256) as usize;
        let hidden_dim = gguf.get_u32("qwen35.feed_forward_length").unwrap_or(17408) as usize;
        let norm_eps = gguf
            .get_f32("qwen35.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6);
        let rope_base = gguf.get_f32("qwen35.rope.freq_base").unwrap_or(1e7);
        let attn_interval = gguf.get_u32("qwen35.full_attention_interval").unwrap_or(4) as usize;
        let sections: [usize; 4] = {
            let v = meta_i32_list(&gguf, "qwen35.rope.dimension_sections");
            if v.len() == 4 {
                [v[0] as usize, v[1] as usize, v[2] as usize, v[3] as usize]
            } else {
                [11, 11, 10, 0]
            }
        };
        let eos_token = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(248046);
        let seq_cap = if max_seq_len > 0 { max_seq_len.min(4096) } else { 4096 };

        // ── GDN geometry from KVs ──
        let head_k_dim = gguf.get_u32("qwen35.ssm.state_size").unwrap_or(DEF_HEAD_K_DIM as u32) as usize;
        let ssm_dt_rank = gguf.get_u32("qwen35.ssm.time_step_rank").unwrap_or(48) as usize;
        let n_k_heads = gguf.get_u32("qwen35.ssm.group_count").unwrap_or(16) as usize;
        let ssm_d_inner = gguf.get_u32("qwen35.ssm.inner_size").unwrap_or(6144) as usize;
        let d_conv = gguf.get_u32("qwen35.ssm.conv_kernel").unwrap_or(4) as usize;
        let head_v_dim = ssm_d_inner / ssm_dt_rank.max(1);
        let head_v_dim = if head_v_dim == 0 { DEF_HEAD_V_DIM } else { head_v_dim };
        let conv_dim = head_k_dim * n_k_heads * 2 + head_v_dim * ssm_dt_rank;
        let conv_depth = d_conv - 1; // rolling ring of prior tokens
        if d_conv != 4 {
            return Err(format!("qwen35: unsupported ssm.conv_kernel {d_conv} (expected 4)"));
        }

        // ── Prism Hadamard fold metadata ──
        // The Sylvester butterfly in wht_block works for any power of two;
        // block_size is a per-model parameter now (27B = 1024, 9B = 4096).
        let had_version = gguf.get_u32("prism.hadamard.version").unwrap_or(0);
        let had_block = if had_version != 0 {
            let hb = gguf.get_u32("prism.hadamard.block_size").unwrap_or(0) as usize;
            if hb == 0 || !hb.is_power_of_two() {
                return Err(format!(
                    "qwen35: unsupported prism.hadamard.block_size {hb} (need a power of two)"
                ));
            }
            hb
        } else {
            HAD_BLOCK
        };
        let signs = Self::load_signs(&mut gguf)?;
        let folded_names = meta_string_list(&gguf, "prism.hadamard.weight_names");
        let inverse_names = meta_string_list(&gguf, "prism.hadamard.inverse_weight_names");
        let folded: HashSet<String> = folded_names.into_iter().collect();
        let inverses: HashSet<String> = inverse_names.into_iter().collect();
        let sign_idx = |name: &str, width: usize| -> Result<Option<usize>, String> {
            let listed = folded.contains(name) || inverses.contains(name);
            if !listed {
                return Ok(None);
            }
            match signs.iter().position(|(w, _)| *w == width) {
                Some(i) => Ok(Some(i)),
                None => Err(format!("prism.hadamard: no sign vector for width {width} ({name})")),
            }
        };

        let data = gguf.mmap.as_slice();

        // ── global weights ──
        let embd_xform = match sign_idx("token_embd.weight", dim)? {
            Some(i) => Xform::Inverse(i),
            None => Xform::None,
        };
        let embd = raw_mat_of(data, &gguf, "token_embd.weight", embd_xform, GGML_TYPE_PTQ1_0)?;
        let out_xform = match sign_idx("output.weight", dim)? {
            Some(i) => Xform::Folded(i),
            None => Xform::None,
        };
        let output = raw_mat_of(data, &gguf, "output.weight", out_xform, GGML_TYPE_PTQ1_0)?;
        let output_norm = f32_meta_vec(&gguf, data, "output_norm.weight")?;
        let vocab_size = embd.n_out;

        let xf = |name: &str, width: usize| -> Result<Xform, String> {
            Ok(match sign_idx(name, width)? {
                Some(i) => Xform::Folded(i),
                None => Xform::None,
            })
        };

        // ── per-layer weights ──
        let mut layers = Vec::with_capacity(n_layers);
        let mut gdn_index = Vec::with_capacity(n_layers);
        let mut attn_index = Vec::with_capacity(n_layers);
        let mut n_gdn = 0usize;
        let mut n_attn = 0usize;

        for li in 0..n_main_layers {
            let base = format!("blk.{li}");
            let is_attn = (li + 1) % attn_interval == 0;
            let attn_norm = f32_meta_vec(&gguf, data, &format!("{base}.attn_norm.weight"))?;
            let post_attention_norm =
                f32_meta_vec(&gguf, data, &format!("{base}.post_attention_norm.weight"))?;
            let ffn_gate_name = format!("{base}.ffn_gate.weight");
            let ffn_up_name = format!("{base}.ffn_up.weight");
            let ffn_down_name = format!("{base}.ffn_down.weight");
            let ffn_gate = raw_mat_of(data, &gguf, &ffn_gate_name, xf(&ffn_gate_name, dim)?, GGML_TYPE_PTQ1_0)?;
            let ffn_up = raw_mat_of(data, &gguf, &ffn_up_name, xf(&ffn_up_name, dim)?, GGML_TYPE_PTQ1_0)?;
            let ffn_down =
                raw_mat_of(data, &gguf, &ffn_down_name, xf(&ffn_down_name, hidden_dim)?, GGML_TYPE_PTQ1_0)?;
            let mixer = if is_attn {
                let qname = format!("{base}.attn_q.weight");
                let kname = format!("{base}.attn_k.weight");
                let vname = format!("{base}.attn_v.weight");
                let oname = format!("{base}.attn_output.weight");
                let mixer = Mixer::Attention {
                    q: raw_mat_of(data, &gguf, &qname, xf(&qname, dim)?, GGML_TYPE_PTQ1_0)?,
                    k: raw_mat_of(data, &gguf, &kname, xf(&kname, dim)?, GGML_TYPE_PTQ1_0)?,
                    v: raw_mat_of(data, &gguf, &vname, xf(&vname, dim)?, GGML_TYPE_PTQ1_0)?,
                    q_norm: f32_meta_vec(&gguf, data, &format!("{base}.attn_q_norm.weight"))?,
                    k_norm: f32_meta_vec(&gguf, data, &format!("{base}.attn_k_norm.weight"))?,
                    o: raw_mat_of(
                        data,
                        &gguf,
                        &oname,
                        xf(&oname, n_heads * head_dim)?,
                        GGML_TYPE_PTQ1_0,
                    )?,
                };
                attn_index.push(n_attn);
                gdn_index.push(usize::MAX);
                n_attn += 1;
                mixer
            } else {
                let qkv_name = format!("{base}.attn_qkv.weight");
                let gate_name = format!("{base}.attn_gate.weight");
                let out_name = format!("{base}.ssm_out.weight");
                // ssm_alpha/ssm_beta consume the raw normalized hidden: they are
                // not part of prism.hadamard.weight_names, so no activation
                // transform applies to their inputs.
                let alpha = bf16_meta_mat(&gguf, data, &format!("{base}.ssm_alpha.weight"), ssm_dt_rank, dim)?;
                let beta = bf16_meta_mat(&gguf, data, &format!("{base}.ssm_beta.weight"), ssm_dt_rank, dim)?;
                let conv1d = f32_meta_vec(&gguf, data, &format!("{base}.ssm_conv1d.weight"))?;
                if conv1d.len() != conv_dim * d_conv {
                    return Err(format!(
                        "{base}.ssm_conv1d: {} taps, expected {}",
                        conv1d.len(),
                        conv_dim * d_conv
                    ));
                }
                let dt_bias = f32_meta_vec(&gguf, data, &format!("{base}.ssm_dt.bias"))?;
                let a = f32_meta_vec(&gguf, data, &format!("{base}.ssm_a"))?;
                let ssm_norm = f32_meta_vec(&gguf, data, &format!("{base}.ssm_norm.weight"))?;
                if dt_bias.len() != ssm_dt_rank || a.len() != ssm_dt_rank || ssm_norm.len() != head_v_dim {
                    return Err(format!("{base}: ssm vector length mismatch"));
                }
                let mixer = Mixer::Gdn {
                    qkv: raw_mat_of(data, &gguf, &qkv_name, xf(&qkv_name, dim)?, GGML_TYPE_PTQ1_0)?,
                    gate: raw_mat_of(data, &gguf, &gate_name, xf(&gate_name, dim)?, GGML_TYPE_PTQ1_0)?,
                    out: raw_mat_of(
                        data,
                        &gguf,
                        &out_name,
                        xf(&out_name, ssm_d_inner)?,
                        GGML_TYPE_PTQ1_0,
                    )?,
                    conv1d,
                    dt_bias,
                    a,
                    alpha,
                    beta,
                    ssm_norm,
                };
                gdn_index.push(n_gdn);
                attn_index.push(usize::MAX);
                n_gdn += 1;
                mixer
            };
            layers.push(Layer { attn_norm, post_attention_norm, ffn_gate, ffn_up, ffn_down, mixer });
        }

        let config = ModelConfig {
            dim,
            n_layers: n_main_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            vocab_size,
            norm_eps,
            rope_freq_base: rope_base,
            max_seq_len: seq_cap,
            logical_max_seq_len: seq_cap,
            kv_ring: 0,
            sliding_window: 0,
            rope_interleaved: true,
        };

        let kv_elems = n_attn * n_kv_heads * seq_cap * head_dim;
        let d_inner = head_v_dim * ssm_dt_rank;

        Ok(Self {
            config,
            gguf,
            embd,
            output,
            output_norm,
            layers,
            signs,
            had_block,
            rope_sections: sections,
            gdn_index,
            attn_index,
            eos_token,
            ssm_state: vec![0.0; n_gdn * ssm_dt_rank * head_k_dim * head_v_dim],
            conv_state: vec![0.0; n_gdn * conv_dim * conv_depth],
            k_cache: vec![0.0; kv_elems],
            v_cache: vec![0.0; kv_elems],
            rowbuf: vec![0.0; hidden_dim.max(dim)],
            x: vec![0.0; dim],
            hbuf: vec![0.0; dim],
            prep512: vec![0.0; dim],
            prep_mid: vec![0.0; d_inner],
            prep_ff: vec![0.0; hidden_dim],
            qkv: vec![0.0; conv_dim],
            zg: vec![0.0; d_inner],
            g48a: vec![0.0; ssm_dt_rank],
            g48b: vec![0.0; ssm_dt_rank],
            gdn_out: vec![0.0; d_inner],
            mix_out: vec![0.0; dim],
            qg: vec![0.0; 2 * n_heads * head_dim],
            kh: vec![0.0; n_kv_heads * head_dim],
            vh: vec![0.0; n_kv_heads * head_dim],
            attn3: vec![0.0; n_heads * head_dim],
            scores: vec![0.0; seq_cap],
            ffn_g: vec![0.0; hidden_dim],
            ffn_u: vec![0.0; hidden_dim],
            logits: vec![0.0; vocab_size],
        })
    }

    /// prism.hadamard.sign_values is one i32 array that concatenates the sign
    /// vector for each width in prism.hadamard.sign_widths.
    fn load_signs(gguf: &mut GgufFile) -> Result<Vec<(usize, Vec<f32>)>, String> {
        let widths = meta_i32_list(gguf, "prism.hadamard.sign_widths");
        let values = meta_i32_list(gguf, "prism.hadamard.sign_values");
        if widths.is_empty() {
            return Ok(Vec::new());
        }
        let mut signs = Vec::with_capacity(widths.len());
        let mut off = 0usize;
        for w in &widths {
            let w = *w as usize;
            if w == 0 || off + w > values.len() {
                return Err(format!("invalid prism.hadamard sign width {w}"));
            }
            let vec: Vec<f32> = values[off..off + w]
                .iter()
                .map(|&v| if v < 0 { -1.0f32 } else { 1.0f32 })
                .collect();
            off += w;
            signs.push((w, vec));
        }
        if off != values.len() {
            return Err(format!(
                "prism.hadamard.sign_values length {} does not match widths sum {off}",
                values.len()
            ));
        }
        Ok(signs)
    }

    pub fn reset_cache(&mut self) {
        self.ssm_state.fill(0.0);
        self.conv_state.fill(0.0);
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
    }

    pub fn compact_kv_cache(&mut self, _current_seq_len: usize, _sink: usize, _drop: usize) -> usize {
        // GDN conv/ssm state cannot shift losslessly; long-context continuation
        // is intentionally disallowed, like the GPT-X3 hybrid backend.
        self.reset_cache();
        0
    }

    /// Embedding lookup. For inverse-folded tables the transform runs in the
    /// opposite order of matmul folding: dequantized row ← H per block, then
    /// signs ("h = s * (H z)").
    fn embed_token(&mut self, token: u32) {
        let dim = self.config.dim;
        let t = (token as usize).min(self.embd.n_out.saturating_sub(1));
        let data = self.gguf.as_bytes();
        let row = self.embd.row_slice(data, t);
        dequant_row_to_f32(self.embd.ty, dim, row, &mut self.rowbuf[..dim]);
        match self.embd.xform {
            Xform::Inverse(i) => {
                wht_blocks(self.had_block, &mut self.rowbuf[..dim]);
                let signs = &self.signs[i].1;
                for (d, (&r, &s)) in self.x.iter_mut().zip(self.rowbuf[..dim].iter().zip(signs)) {
                    *d = r * s;
                }
            }
            _ => {
                self.x.copy_from_slice(&self.rowbuf[..dim]);
            }
        }
    }

    fn gdn_forward(&mut self, li: usize, gi: usize) {
        let cfg = &self.config;
        let dim = cfg.dim;
        let norm_eps = cfg.norm_eps;
        let (qkv_mat, gate_mat, out_mat, conv1d, dt_bias, a, alpha, beta, ssm_norm) =
            match &self.layers[li].mixer {
                Mixer::Gdn { qkv, gate, out, conv1d, dt_bias, a, alpha, beta, ssm_norm } => {
                    (qkv, gate, out, conv1d, dt_bias, a, alpha, beta, ssm_norm)
                }
                _ => unreachable!(),
            };
        let n_k_heads = 16;
        let n_v_heads = self.g48a.len();
        let head_v_dim = self.zg.len() / n_v_heads;
        let conv_dim = self.qkv.len();
        let conv_depth = 3;
        let s_dim = qkv_mat.sign_idx();

        // Folded projections read the sign+hadamard transformed hidden;
        // alpha/beta (unfolded) read the raw normalized hidden.
        let mut prep = std::mem::take(&mut self.prep512);
        match (s_dim, &self.signs) {
            (Some(i), table) => hadamard_prep(self.had_block, &mut prep, &self.hbuf, &table[i].1),
            _ => prep.copy_from_slice(&self.hbuf),
        }
        let data = self.gguf.as_bytes();
        gemv_t143(data, qkv_mat, &prep, &mut self.qkv);
        gemv_t143(data, gate_mat, &prep, &mut self.zg);
        gemv_f32(alpha, dim, &self.hbuf, &mut self.g48a);
        gemv_f32(beta, dim, &self.hbuf, &mut self.g48b);
        self.prep512 = prep;

        // Causal conv over qkv_mixed (pre-activation) + silu, 3-deep ring.
        let cs_base = gi * conv_dim * conv_depth;
        for c in 0..conv_dim {
            let s0 = self.conv_state[cs_base + c * 3];
            let s1 = self.conv_state[cs_base + c * 3 + 1];
            let s2 = self.conv_state[cs_base + c * 3 + 2];
            let cur = self.qkv[c];
            let y = s0 * conv1d[c * 4]
                + s1 * conv1d[c * 4 + 1]
                + s2 * conv1d[c * 4 + 2]
                + cur * conv1d[c * 4 + 3];
            self.conv_state[cs_base + c * 3] = s1;
            self.conv_state[cs_base + c * 3 + 1] = s2;
            self.conv_state[cs_base + c * 3 + 2] = cur;
            self.qkv[c] = silu(y);
        }

        // q/k L2 norm over the joint [head × 2·k_heads] view, per 128 slice.
        let eps = norm_eps;
        for sl in 0..2 * n_k_heads {
            let start = sl * head_v_dim;
            l2norm_slice(&mut self.qkv[start..start + head_v_dim], eps);
        }

        // Per-v-head recurrence; q/k repeat across v-heads (head = iv % n_k).
        let s_head = head_v_dim * head_v_dim;
        let s_base = gi * n_v_heads * s_head;
        for iv in 0..n_v_heads {
            let hq = iv % n_k_heads;
            let k_off = head_v_dim * n_k_heads;
            let v_off = 2 * head_v_dim * n_k_heads;
            let q = &self.qkv[hq * head_v_dim..(hq + 1) * head_v_dim];
            let k = &self.qkv[k_off + hq * head_v_dim..k_off + (hq + 1) * head_v_dim];
            let v = &self.qkv[v_off + iv * head_v_dim..v_off + (iv + 1) * head_v_dim];
            let g = a[iv] * softplus20(self.g48a[iv] + dt_bias[iv]);
            let b = sigmoid(self.g48b[iv]);
            let s = &mut self.ssm_state[s_base + iv * s_head..s_base + (iv + 1) * s_head];
            let o = &mut self.gdn_out[iv * head_v_dim..(iv + 1) * head_v_dim];
            gdn_head_step(s, q, k, v, g, b, o);
        }

        // Gated RMS norm: n = rms(out per head, ssm_norm); final = silu(z)·n.
        for iv in 0..n_v_heads {
            let o = &mut self.gdn_out[iv * head_v_dim..(iv + 1) * head_v_dim];
            rmsnorm(o, ssm_norm, eps);
            let z = &self.zg[iv * head_v_dim..(iv + 1) * head_v_dim];
            for (ov, &zv) in o.iter_mut().zip(z.iter()) {
                *ov *= silu(zv);
            }
        }

        // ssm_out fold: GDN perm first, then signs+hadamard, then gemv.
        if raw_gdn_v() {
            self.prep_mid.copy_from_slice(&self.gdn_out);
        } else {
            gdn_perm(&mut self.prep_mid, &self.gdn_out);
        }
        let data = self.gguf.as_bytes();
        match (out_mat.sign_idx(), &self.signs) {
            (Some(i), table) => hadamard_prep_in_place(self.had_block, &mut self.prep_mid, &table[i].1),
            _ => {}
        }
        gemv_t143(data, out_mat, &self.prep_mid, &mut self.mix_out);
    }

    fn attn_forward(&mut self, li: usize, ai: usize, pos: usize) {
        let cfg = &self.config;
        let norm_eps = cfg.norm_eps;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let rep = n_heads / n_kv;
        let n_rot = 64usize;
        let (q_mat, k_mat, v_mat, o_mat, q_norm, k_norm) = match &self.layers[li].mixer {
            Mixer::Attention { q, k, v, o, q_norm, k_norm } => (q, k, v, o, q_norm, k_norm),
            _ => unreachable!(),
        };

        // Prepared hidden shared by q/k/v projections (fork memoization).
        let mut prep = std::mem::take(&mut self.prep512);
        match (q_mat.sign_idx(), &self.signs) {
            (Some(i), table) => hadamard_prep(self.had_block, &mut prep, &self.hbuf, &table[i].1),
            _ => prep.copy_from_slice(&self.hbuf),
        }
        let data = self.gguf.as_bytes();
        gemv_t143(data, q_mat, &prep, &mut self.qg);
        gemv_t143(data, k_mat, &prep, &mut self.kh);
        gemv_t143(data, v_mat, &prep, &mut self.vh);
        self.prep512 = prep;

        // Split fused q+gate; per-head q RMS norm, rope, and gate staging.
        for h in 0..n_heads {
            let qg = self.qg[h * 2 * head_dim..(h + 1) * 2 * head_dim].split_at_mut(head_dim);
            let (q_part, gate_part) = qg;
            rmsnorm(q_part, q_norm, norm_eps);
            rope_imrope(q_part, n_rot, pos, cfg.rope_freq_base, &self.rope_sections);
            self.prep_mid[h * head_dim..(h + 1) * head_dim].copy_from_slice(gate_part);
        }
        // Split-half pairing: dims (d, d + n_rot/2) for d < n_rot/2.
        for kk in 0..n_kv {
            let kpart = &mut self.kh[kk * head_dim..(kk + 1) * head_dim];
            rmsnorm(kpart, k_norm, norm_eps);
            rope_imrope(kpart, n_rot, pos, cfg.rope_freq_base, &self.rope_sections);
        }
        // v: no norm, no rope.

        // KV cache store at this position (f32 cache, no cache rotations).
        let head_span = self.config.max_seq_len * head_dim;
        let base = (ai * n_kv) * head_span;
        for kvh in 0..n_kv {
            let off = base + kvh * head_span + pos * head_dim;
            self.k_cache[off..off + head_dim].copy_from_slice(&self.kh[kvh * head_dim..(kvh + 1) * head_dim]);
            self.v_cache[off..off + head_dim].copy_from_slice(&self.vh[kvh * head_dim..(kvh + 1) * head_dim]);
        }

        // Causal attention, GQA repeat h → kvh = h / rep.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq = pos + 1;
        for h in 0..n_heads {
            let kvh = h / rep;
            let kbase = base + kvh * head_span;
            let q = &self.qg[h * 2 * head_dim..h * 2 * head_dim + head_dim];
            let mut mx = f32::NEG_INFINITY;
            for p in 0..seq {
                let kseg = &self.k_cache[kbase + p * head_dim..kbase + (p + 1) * head_dim];
                let sc = dot8(q, kseg) * scale;
                self.scores[p] = sc;
                mx = mx.max(sc);
            }
            let mut sum = 0.0f32;
            for sc in self.scores[..seq].iter_mut() {
                *sc = (*sc - mx).exp();
                sum += *sc;
            }
            let inv = 1.0 / sum;
            let out = &mut self.attn3[h * head_dim..(h + 1) * head_dim];
            out.fill(0.0);
            for p in 0..seq {
                let w = self.scores[p] * inv;
                let vseg = &self.v_cache[kbase + p * head_dim..kbase + (p + 1) * head_dim];
                for (ov, &vv) in out.iter_mut().zip(vseg.iter()) {
                    *ov += w * vv;
                }
            }
            // Gated output: out ⊙ sigmoid(gate).
            let gate = &self.prep_mid[h * head_dim..(h + 1) * head_dim];
            for (ov, &gv) in out.iter_mut().zip(gate.iter()) {
                *ov *= sigmoid(gv);
            }
        }

        // Output projection: signs+hadamard on the 6144 activation, then gemv.
        match (o_mat.sign_idx(), &self.signs) {
            (Some(i), table) => hadamard_prep_in_place(self.had_block, &mut self.attn3, &table[i].1),
            _ => {}
        }
        let data = self.gguf.as_bytes();
        gemv_t143(data, o_mat, &self.attn3, &mut self.mix_out);
    }

    fn forward_impl(&mut self, token: u32, pos: usize, compute_logits: bool) {
        // A zero-position forward is the start of a brand-new context; the
        // recurrent state (GDN + conv) cannot be positionally overwritten the
        // way a KV cache can — reset whenever the first position is requested
        // (same semantics as the GPT-X3 hybrid backend).
        if pos == 0 {
            self.reset_cache();
        }
        if pos >= self.config.max_seq_len {
            eprintln!(
                "[qwen35] position {pos} beyond max_seq_len {}; step ignored",
                self.config.max_seq_len
            );
            return;
        }
        let eps = self.config.norm_eps;

        self.embed_token(token);

        for li in 0..self.config.n_layers {
            // h = rms(attn_norm · x)
            self.hbuf.copy_from_slice(&self.x);
            rmsnorm(&mut self.hbuf, &self.layers[li].attn_norm, eps);

            let gi = self.gdn_index[li];
            if gi != usize::MAX {
                self.gdn_forward(li, gi);
            } else {
                let ai = self.attn_index[li];
                self.attn_forward(li, ai, pos);
            }

            for (x, o) in self.x.iter_mut().zip(&self.mix_out) {
                *x += o;
            }

            // h2 = rms(post_attention_norm · x); FFN: silu(gate·h2) ⊙ (up·h2)
            self.hbuf.copy_from_slice(&self.x);
            rmsnorm(&mut self.hbuf, &self.layers[li].post_attention_norm, eps);

            {
                let mut prep = std::mem::take(&mut self.prep512);
                match (self.layers[li].ffn_gate.sign_idx(), &self.signs) {
                    (Some(i), table) => hadamard_prep(self.had_block, &mut prep, &self.hbuf, &table[i].1),
                    _ => prep.copy_from_slice(&self.hbuf),
                }
                let data = self.gguf.as_bytes();
                gemv_t143(data, &self.layers[li].ffn_gate, &prep, &mut self.ffn_g);
                gemv_t143(data, &self.layers[li].ffn_up, &prep, &mut self.ffn_u);
                self.prep512 = prep;
            }
            for (g, &u) in self.ffn_g.iter_mut().zip(self.ffn_u.iter()) {
                *g = silu(*g) * u;
            }

            {
                let mut prep = std::mem::take(&mut self.prep_ff);
                match (self.layers[li].ffn_down.sign_idx(), &self.signs) {
                    (Some(i), table) => hadamard_prep(self.had_block, &mut prep, &self.ffn_g, &table[i].1),
                    _ => prep.copy_from_slice(&self.ffn_g),
                }
                let data = self.gguf.as_bytes();
                gemv_t143(data, &self.layers[li].ffn_down, &prep, &mut self.mix_out);
                self.prep_ff = prep;
            }
            for (x, f) in self.x.iter_mut().zip(&self.mix_out) {
                *x += f;
            }
        }

        if compute_logits {
            rmsnorm(&mut self.x, &self.output_norm, eps);
            match (self.output.sign_idx(), &self.signs) {
                (Some(i), table) => hadamard_prep_in_place(self.had_block, &mut self.x, &table[i].1),
                _ => {}
            }
            let data = self.gguf.as_bytes();
            gemv_t143(data, &self.output, &self.x, &mut self.logits);
        }

        // Debug hook: one env-gated top-5 print per logits computation.
        if compute_logits && std::env::var_os("SOFUU_DEBUG_QWEN35").is_some() {
            let mut idx: Vec<u32> = (0..self.logits.len() as u32).collect();
            idx.sort_unstable_by(|&a, &b| {
                self.logits[b as usize]
                    .partial_cmp(&self.logits[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let tops: Vec<String> = idx[..5]
                .iter()
                .map(|&i| format!("{}={:.4}", i, self.logits[i as usize]))
                .collect();
            eprintln!("[qwen35] pos={pos} top5: {}", tops.join(" "));
        }
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        self.forward_impl(token, pos, true);
        &self.logits
    }

    pub fn forward_prefill(&mut self, token: u32, pos: usize) {
        self.forward_impl(token, pos, false);
    }

    pub fn forward_prefill_batch(&mut self, tokens: &[u32], start_pos: usize) {
        self.forward_batch_impl(tokens, start_pos, false);
    }

    pub fn forward_prefill_all_with_logits(&mut self, tokens: &[u32], start_pos: usize) -> &[f32] {
        if tokens.is_empty() {
            return &self.logits;
        }
        self.forward_batch_impl(tokens, start_pos, true);
        &self.logits
    }

    /// Sub-chunked entry for the batched prefill. Each chunk processes its
    /// tokens as one GEMM batch; logits (when requested) come from the very
    /// last token, matching forward_impl's single-row logits contract.
    fn forward_batch_impl(&mut self, tokens: &[u32], start_pos: usize, last_logits: bool) {
        let max_seq = self.config.max_seq_len;
        let n_chunks = tokens.len().div_ceil(PRE_FILL_CHUNK);
        for ci in 0..n_chunks {
            let lo = ci * PRE_FILL_CHUNK;
            let want_hi = ((ci + 1) * PRE_FILL_CHUNK).min(tokens.len());
            let cstart = start_pos + lo;
            if cstart >= max_seq {
                eprintln!(
                    "[qwen35] position {cstart} beyond max_seq_len {max_seq}; prefill chunk ignored"
                );
                continue;
            }
            let hi = want_hi.min(cstart + (max_seq - cstart));
            if hi < want_hi {
                eprintln!(
                    "[qwen35] prefill stopped at position {hi} (max_seq_len {max_seq}); remaining tokens ignored"
                );
            }
            self.forward_chunk(&tokens[lo..hi], cstart, last_logits && ci + 1 == n_chunks);
        }
    }

    /// Batched prefill over one token chunk. Every weight matmul runs as a
    /// dequant-once GEMM (gemm_t143) across the chunk; conv/GDN state, KV
    /// cache writes and attention stay sequential and single-writer in token
    /// order, with token i sitting at start_pos + i. With `last_logits`, the
    /// final token's head (rmsnorm → fold → gemv_t143 → self.logits) is the
    /// unchanged single-token path, so prefill and decode agree bit-for-bit.
    /// All per-chunk activations live in locals; the persistent struct
    /// scratch stays reserved for the single-token forward path.
    fn forward_chunk(&mut self, tokens: &[u32], start_pos: usize, last_logits: bool) {
        let n = tokens.len();
        if n == 0 {
            return;
        }
        if start_pos == 0 {
            self.reset_cache();
        }
        let dim = self.config.dim;
        let eps = self.config.norm_eps;
        let n_heads = self.config.n_heads;
        let n_kv = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let max_seq = self.config.max_seq_len;
        let rep = n_heads / n_kv;
        let rope_base = self.config.rope_freq_base;
        let sections = self.rope_sections;
        let hidden = self.ffn_g.len();
        let d_inner = self.zg.len();
        let conv_dim = self.qkv.len();
        let qg_w = self.qg.len();
        let kv_w = self.kh.len();
        let attn_w = self.attn3.len();
        let n_v_heads = self.g48a.len();
        let head_v_dim = d_inner / n_v_heads;
        let n_k_heads = 16usize;

        let mut bx = vec![0.0f32; n * dim];
        let mut bh = vec![0.0f32; n * dim];
        let mut bp = vec![0.0f32; n * dim];
        let mut bqkv = vec![0.0f32; n * conv_dim];
        let mut bzg = vec![0.0f32; n * d_inner];
        let mut bg48a = vec![0.0f32; n * n_v_heads];
        let mut bg48b = vec![0.0f32; n * n_v_heads];
        let mut bgdn = vec![0.0f32; n * d_inner];
        let mut bpm = vec![0.0f32; n * d_inner];
        let mut bqg = vec![0.0f32; n * qg_w];
        let mut bkh = vec![0.0f32; n * kv_w];
        let mut bvh = vec![0.0f32; n * kv_w];
        let mut bgate = vec![0.0f32; n * attn_w];
        let mut battn = vec![0.0f32; n * attn_w];
        let mut bfg = vec![0.0f32; n * hidden];
        let mut bfup = vec![0.0f32; n * hidden];
        let mut bo = vec![0.0f32; n * dim];
        let mut scores = vec![0.0f32; max_seq];
        let mut escratch = vec![0.0f32; dim];

        let data = self.gguf.as_bytes();
        for (t, &tok) in tokens.iter().enumerate() {
            let tid = (tok as usize).min(self.embd.n_out.saturating_sub(1));
            let row = self.embd.row_slice(data, tid);
            dequant_row_to_f32(self.embd.ty, dim, row, &mut escratch[..dim]);
            match self.embd.xform {
                Xform::Inverse(i) => {
                    wht_blocks(self.had_block, &mut escratch[..dim]);
                    let signs = &self.signs[i].1;
                    for (d, (&r, &s)) in bx[t * dim..(t + 1) * dim]
                        .iter_mut()
                        .zip(escratch[..dim].iter().zip(signs))
                    {
                        *d = r * s;
                    }
                }
                _ => {
                    bx[t * dim..(t + 1) * dim].copy_from_slice(&escratch[..dim]);
                }
            }
        }

        for li in 0..self.config.n_layers {
            for t in 0..n {
                let hrow = &mut bh[t * dim..(t + 1) * dim];
                hrow.copy_from_slice(&bx[t * dim..(t + 1) * dim]);
                rmsnorm(hrow, &self.layers[li].attn_norm, eps);
            }
            let gi = self.gdn_index[li];
            if gi != usize::MAX {
                let (qkv_mat, gate_mat, out_mat, conv1d, dt_bias, a, alpha, beta, ssm_norm) =
                    match &self.layers[li].mixer {
                        Mixer::Gdn { qkv, gate, out, conv1d, dt_bias, a, alpha, beta, ssm_norm } => {
                            (qkv, gate, out, conv1d, dt_bias, a, alpha, beta, ssm_norm)
                        }
                        _ => unreachable!(),
                    };
                for t in 0..n {
                    let trow = t * dim;
                    match (qkv_mat.sign_idx(), &self.signs) {
                        (Some(i), table) => {
                            hadamard_prep(self.had_block, &mut bp[trow..trow + dim], &bh[trow..trow + dim], &table[i].1)
                        }
                        _ => bp[trow..trow + dim].copy_from_slice(&bh[trow..trow + dim]),
                    }
                    let grow = t * n_v_heads;
                    gemv_f32(alpha, dim, &bh[trow..trow + dim], &mut bg48a[grow..grow + n_v_heads]);
                    gemv_f32(beta, dim, &bh[trow..trow + dim], &mut bg48b[grow..grow + n_v_heads]);
                }
                gemm_t143(data, qkv_mat, n, &bp, &mut bqkv);
                gemm_t143(data, gate_mat, n, &bp, &mut bzg);

                let cs_base = gi * conv_dim * 3;
                let s_head = head_v_dim * head_v_dim;
                let s_base = gi * n_v_heads * s_head;
                for t in 0..n {
                    let trow = t * conv_dim;
                    let row = &mut bqkv[trow..trow + conv_dim];
                    for c in 0..conv_dim {
                        let s0 = self.conv_state[cs_base + c * 3];
                        let s1 = self.conv_state[cs_base + c * 3 + 1];
                        let s2 = self.conv_state[cs_base + c * 3 + 2];
                        let cur = row[c];
                        let y = s0 * conv1d[c * 4]
                            + s1 * conv1d[c * 4 + 1]
                            + s2 * conv1d[c * 4 + 2]
                            + cur * conv1d[c * 4 + 3];
                        self.conv_state[cs_base + c * 3] = s1;
                        self.conv_state[cs_base + c * 3 + 1] = s2;
                        self.conv_state[cs_base + c * 3 + 2] = cur;
                        row[c] = silu(y);
                    }
                    for sl in 0..2 * n_k_heads {
                        l2norm_slice(&mut row[sl * head_v_dim..(sl + 1) * head_v_dim], eps);
                    }
                    let grow = t * d_inner;
                    for iv in 0..n_v_heads {
                        let hq = iv % n_k_heads;
                        let k_off = head_v_dim * n_k_heads;
                        let v_off = 2 * head_v_dim * n_k_heads;
                        let q = &row[hq * head_v_dim..(hq + 1) * head_v_dim];
                        let k = &row[k_off + hq * head_v_dim..k_off + (hq + 1) * head_v_dim];
                        let v = &row[v_off + iv * head_v_dim..v_off + (iv + 1) * head_v_dim];
                        let g = a[iv] * softplus20(bg48a[t * n_v_heads + iv] + dt_bias[iv]);
                        let b = sigmoid(bg48b[t * n_v_heads + iv]);
                        let s = &mut self.ssm_state[s_base + iv * s_head..s_base + (iv + 1) * s_head];
                        let o = &mut bgdn[grow + iv * head_v_dim..grow + (iv + 1) * head_v_dim];
                        gdn_head_step(s, q, k, v, g, b, o);
                    }
                    for iv in 0..n_v_heads {
                        let o = &mut bgdn[grow + iv * head_v_dim..grow + (iv + 1) * head_v_dim];
                        rmsnorm(o, ssm_norm, eps);
                        let z = &bzg[grow + iv * head_v_dim..grow + (iv + 1) * head_v_dim];
                        for (ov, &zv) in o.iter_mut().zip(z.iter()) {
                            *ov *= silu(zv);
                        }
                    }
                    let bm = &mut bpm[grow..grow + d_inner];
                    if raw_gdn_v() {
                        bm.copy_from_slice(&bgdn[grow..grow + d_inner]);
                    } else {
                        gdn_perm(bm, &bgdn[grow..grow + d_inner]);
                    }
                    match (out_mat.sign_idx(), &self.signs) {
                        (Some(i), table) => {
                            hadamard_prep_in_place(self.had_block, &mut bpm[grow..grow + d_inner], &table[i].1)
                        }
                        _ => {}
                    }
                }
                gemm_t143(data, out_mat, n, &bpm, &mut bo);
            } else {
                let ai = self.attn_index[li];
                let (q_mat, k_mat, v_mat, o_mat, q_norm, k_norm) = match &self.layers[li].mixer {
                    Mixer::Attention { q, k, v, o, q_norm, k_norm } => (q, k, v, o, q_norm, k_norm),
                    _ => unreachable!(),
                };
                for t in 0..n {
                    let trow = t * dim;
                    match (q_mat.sign_idx(), &self.signs) {
                        (Some(i), table) => {
                            hadamard_prep(self.had_block, &mut bp[trow..trow + dim], &bh[trow..trow + dim], &table[i].1)
                        }
                        _ => bp[trow..trow + dim].copy_from_slice(&bh[trow..trow + dim]),
                    }
                }
                gemm_t143(data, q_mat, n, &bp, &mut bqg);
                gemm_t143(data, k_mat, n, &bp, &mut bkh);
                gemm_t143(data, v_mat, n, &bp, &mut bvh);

                let head_span = max_seq * head_dim;
                let base = (ai * n_kv) * head_span;
                let n_rot = 64usize;
                for t in 0..n {
                    let pos = start_pos + t;
                    let qrow = &mut bqg[t * qg_w..(t + 1) * qg_w];
                    for h in 0..n_heads {
                        let (q_part, gate_part) =
                            qrow[h * 2 * head_dim..(h + 1) * 2 * head_dim].split_at_mut(head_dim);
                        rmsnorm(q_part, q_norm, eps);
                        rope_imrope(q_part, n_rot, pos, rope_base, &sections);
                        bgate[t * attn_w + h * head_dim..t * attn_w + (h + 1) * head_dim]
                            .copy_from_slice(gate_part);
                    }
                    for kk in 0..n_kv {
                        let kpart =
                            &mut bkh[t * kv_w + kk * head_dim..t * kv_w + (kk + 1) * head_dim];
                        rmsnorm(kpart, k_norm, eps);
                        rope_imrope(kpart, n_rot, pos, rope_base, &sections);
                    }
                    for kvh in 0..n_kv {
                        let off = base + kvh * head_span + pos * head_dim;
                        self.k_cache[off..off + head_dim].copy_from_slice(
                            &bkh[t * kv_w + kvh * head_dim..t * kv_w + (kvh + 1) * head_dim],
                        );
                        self.v_cache[off..off + head_dim].copy_from_slice(
                            &bvh[t * kv_w + kvh * head_dim..t * kv_w + (kvh + 1) * head_dim],
                        );
                    }
                    let scale = 1.0 / (head_dim as f32).sqrt();
                    let seq = pos + 1;
                    for h in 0..n_heads {
                        let kvh = h / rep;
                        let kbase = base + kvh * head_span;
                        let q = &bqg[t * qg_w + h * 2 * head_dim..t * qg_w + h * 2 * head_dim + head_dim];
                        let mut mx = f32::NEG_INFINITY;
                        for p in 0..seq {
                            let kseg = &self.k_cache[kbase + p * head_dim..kbase + (p + 1) * head_dim];
                            let sc = dot8(q, kseg) * scale;
                            scores[p] = sc;
                            mx = mx.max(sc);
                        }
                        let mut sum = 0.0f32;
                        for sc in scores[..seq].iter_mut() {
                            *sc = (*sc - mx).exp();
                            sum += *sc;
                        }
                        let inv = 1.0 / sum;
                        let out = &mut battn[t * attn_w + h * head_dim..t * attn_w + (h + 1) * head_dim];
                        out.fill(0.0);
                        for p in 0..seq {
                            let w = scores[p] * inv;
                            let vseg = &self.v_cache[kbase + p * head_dim..kbase + (p + 1) * head_dim];
                            for (ov, &vv) in out.iter_mut().zip(vseg.iter()) {
                                *ov += w * vv;
                            }
                        }
                        let gate = &bgate[t * attn_w + h * head_dim..t * attn_w + (h + 1) * head_dim];
                        for (ov, &gv) in out.iter_mut().zip(gate.iter()) {
                            *ov *= sigmoid(gv);
                        }
                    }
                    match (o_mat.sign_idx(), &self.signs) {
                        (Some(i), table) => {
                            hadamard_prep_in_place(self.had_block, &mut battn[t * attn_w..(t + 1) * attn_w], &table[i].1)
                        }
                        _ => {}
                    }
                }
                gemm_t143(data, o_mat, n, &battn, &mut bo);
            }
            for t in 0..n {
                for (x, o) in bx[t * dim..(t + 1) * dim].iter_mut().zip(&bo[t * dim..(t + 1) * dim]) {
                    *x += o;
                }
            }

            for t in 0..n {
                let hrow = &mut bh[t * dim..(t + 1) * dim];
                hrow.copy_from_slice(&bx[t * dim..(t + 1) * dim]);
                rmsnorm(hrow, &self.layers[li].post_attention_norm, eps);
                match (self.layers[li].ffn_gate.sign_idx(), &self.signs) {
                    (Some(i), table) => {
                        hadamard_prep(self.had_block, &mut bp[t * dim..(t + 1) * dim], &bh[t * dim..(t + 1) * dim], &table[i].1)
                    }
                    _ => bp[t * dim..(t + 1) * dim].copy_from_slice(&bh[t * dim..(t + 1) * dim]),
                }
            }
            gemm_t143(data, &self.layers[li].ffn_gate, n, &bp, &mut bfg);
            gemm_t143(data, &self.layers[li].ffn_up, n, &bp, &mut bfup);
            for t in 0..n {
                let frow = t * hidden;
                for (g, &u) in bfg[frow..frow + hidden]
                    .iter_mut()
                    .zip(&bfup[frow..frow + hidden])
                {
                    *g = silu(*g) * u;
                }
                match (self.layers[li].ffn_down.sign_idx(), &self.signs) {
                    (Some(i), table) => {
                        hadamard_prep_in_place(self.had_block, &mut bfg[frow..frow + hidden], &table[i].1)
                    }
                    _ => {}
                }
            }
            gemm_t143(data, &self.layers[li].ffn_down, n, &bfg, &mut bo);
            for t in 0..n {
                for (x, o) in bx[t * dim..(t + 1) * dim].iter_mut().zip(&bo[t * dim..(t + 1) * dim]) {
                    *x += o;
                }
            }
        }

        if last_logits {
            let pos = start_pos + n - 1;
            self.x.copy_from_slice(&bx[(n - 1) * dim..n * dim]);
            rmsnorm(&mut self.x, &self.output_norm, eps);
            match (self.output.sign_idx(), &self.signs) {
                (Some(i), table) => hadamard_prep_in_place(self.had_block, &mut self.x, &table[i].1),
                _ => {}
            }
            gemv_t143(data, &self.output, &self.x, &mut self.logits);

            if std::env::var_os("SOFUU_DEBUG_QWEN35").is_some() {
                let mut idx: Vec<u32> = (0..self.logits.len() as u32).collect();
                idx.sort_unstable_by(|&a, &b| {
                    self.logits[b as usize]
                        .partial_cmp(&self.logits[a as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let tops: Vec<String> = idx[..5]
                    .iter()
                    .map(|&i| format!("{}={:.4}", i, self.logits[i as usize]))
                    .collect();
                eprintln!("[qwen35] pos={pos} top5: {}", tops.join(" "));
            }
        }
    }
}

// ── validation test (skipped unless the target GGUF is present) ──────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Isolated throughput of the file-direct PQ2_0 row dot (the ternary
    /// decode hot loop). Not an assertion — run with
    /// `cargo test -p sofuu-core --lib bench_packed_row_dot_t142 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_packed_row_dot_t142() {
        for n_in in [4096usize, 8192] {
            let blocks = n_in / PQ2_0_BLOCK_VALUES;
            let mut row = vec![0u8; blocks * PQ2_0_BLOCK_BYTES];
            for (i, chunk) in row.chunks_mut(PQ2_0_BLOCK_BYTES).enumerate() {
                // d = 2^-9, codes spread over all four values.
                chunk[0..2].copy_from_slice(&0x1C00u16.to_le_bytes());
                for k in 2..PQ2_0_BLOCK_BYTES {
                    chunk[k] = (k as u8).wrapping_mul(37).wrapping_add(i as u8);
                }
            }
            let acts: Vec<f32> = (0..n_in).map(|i| ((i * 37 % 251) as f32 / 251.0) - 0.5).collect();
            let aq = quantize_acts(&acts, 1, n_in, blocks);
            let iters = 400_000usize;
            let mut acc = 0.0f32;
            let t0 = Instant::now();
            for _ in 0..iters {
                acc += unsafe { packed_row_dot_t142(&row, &aq, 0, n_in) };
            }
            let dt = t0.elapsed().as_secs_f64();
            let gbs = (iters * row.len()) as f64 / dt / 1e9;
            println!(
                "packed_row_dot_t142 n_in={n_in}: {:.2} GB/s, {:.1} ns/block, acc={acc:.3}",
                gbs,
                dt * 1e9 / (iters * blocks) as f64
            );
        }
    }

    /// PQ2_0 decode: d rides block-first, codes are little-endian 2-bit and
    /// map (c − 1) — exactly the PrismML-Eng/llama.cpp prism-branch contract.
    #[test]
    fn pq2_0_dequant_hand_block() {
        let mut src = vec![0u8; PQ2_0_BLOCK_BYTES];
        // d = 1.0 (0x3C00) block-first; one byte with codes 0,1,2,3 packed
        // low-bits-first, everything else zero.
        src[0] = 0x00;
        src[1] = 0x3C;
        src[2] = 0b11_10_01_00;
        let mut levels = vec![0i8; PQ2_0_BLOCK_VALUES];
        let mut scales = vec![0.0f32; 1];
        dequant_pq2_i8(&src, &mut levels, &mut scales);
        assert_eq!(scales[0], 1.0);
        assert_eq!(&levels[..4], &[-1i8, 0, 1, 2]);
        assert!(levels[4..].iter().all(|&v| v == -1));
    }

    /// File-backed decode reference: dequantize the first 128 values of the
    /// 9B's token_embd through the engine codec dispatch and compare the f32
    /// bit-pattern sha256 against an independent Python reader of the same
    /// GGUF bytes. Skips when the checkpoint is absent.
    #[test]
    fn pq2_0_ternary9b_token_embd_decode_reference() {
        if !std::path::Path::new(MODEL_PATH_9B).exists() {
            return;
        }
        let gguf = GgufFile::open(MODEL_PATH_9B).unwrap();
        let data = gguf.mmap.as_slice();
        let m = raw_mat_of(data, &gguf, "token_embd.weight", Xform::None, GGML_TYPE_PTQ1_0).unwrap();
        assert_eq!(m.ty, GGML_TYPE_PQ2_0, "9B mainline tensors must be type 142");
        // MTP regression: the draft layer (blk.32, nextn_predict_layers = 1)
        // must not become layer 32 of the main model.
        let model = Qwen35Model::from_gguf(gguf, 0).unwrap();
        assert_eq!(model.config.n_layers, 32, "9B main layers must be 32 (MTP draft skipped)");
        let gguf = GgufFile::open(MODEL_PATH_9B).unwrap();
        let data = gguf.mmap.as_slice();
        let m = raw_mat_of(data, &gguf, "token_embd.weight", Xform::None, GGML_TYPE_PTQ1_0)
            .unwrap();
        let mut levels = vec![0i8; m.n_in];
        let mut scales = vec![0.0f32; m.n_in / PQ2_0_BLOCK_VALUES];
        let row0 = m.row_slice(data, 0);
        dequant_pq2_i8(row0, &mut levels, &mut scales);
        // The Python reference pins block 0 of row 0 (34 B: f16 d then 32 B
        // of little-endian 2-bit codes) — compare the SAME 128 values, and
        // pin the first 8 bytes so a header-walk disagreement between our
        // gguf parser and the independent reader fails as bytes, not as a
        // checksum mystery.
        let head_hex: String = row0[..8].iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            head_hex,
            "bb2205960aaa6054",
            "token_embd block-0 bytes disagree with the Python header walk (data_offset mismatch?)"
        );
        assert_eq!(
            scales[0],
            0.01314544677734375,
            "block-0 f16 scale disagrees with the Python reader"
        );
        let vals: Vec<f32> = levels[..128].iter().map(|&lv| lv as f32 * scales[0]).collect();
        let bits: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sha = crate::rt::wal::sha256_hex(&bits);
        assert_eq!(
            sha,
            "79ef9d8b5a06cd43e648b72e64f5fd5335b46b733cd331a668a0ede1ceb437b8",
            "PQ2_0 decode diverges from the independent Python reference on token_embd row 0"
        );
    }


    const MODEL_PATH: &str = "/Users/priyanshuboruah/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf";
    /// Ternary-Bonsai-2-9B (qwen35 / PQ2_0 + MTP draft) — the live-file decode
    /// reference below is pinned to this checkpoint's token_embd.
    const MODEL_PATH_9B: &str =
        "/Users/priyanshuboruah/Downloads/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf";

    /// The Hadamard side of the fold must compute exactly the specified
    /// normalized Sylvester WH: H[i][j] = (±1)^popcount(i&j) / sqrt(1024).
    #[test]
    fn packed_row_dot_bitmatches_dequant_sdot_path() {
        // Random packed PTQ1_0 rows (f16 scales set to values that survive
        // f32 folding) must produce bit-identical dots to the
        // dequantize-then-sdot reference, because per-band integer sums are
        // exact regardless of accumulation order.
        let blocks = 7usize;
        let n_in = blocks * PTQ1_0_BLOCK_VALUES;
        let mut row = vec![0u8; blocks * PTQ1_0_ROW_BLOCK_BYTES];
        let mut state = 987654321u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        for c in 0..blocks {
            let base = c * PTQ1_0_ROW_BLOCK_BYTES;
            for b in 0..26 {
                row[base + b] = rand();
            }
            // Scale f16 = 0x3C00 (1.0): int sums stay exactly representable.
            row[base + 26] = 0x00;
            row[base + 27] = 0x3C;
        }
        let acts: Vec<f32> = (0..n_in)
            .map(|i| (((i * 37 % 201) as f32) - 100.0) / 50.0)
            .collect();
        let aq = quantize_acts(&acts, 1, n_in, blocks);

        let mut trits = vec![0i8; n_in];
        let mut rowd = vec![0.0f32; blocks];
        dequant_trits_i8(&row, &mut trits, &mut rowd);
        let ref_dot = trit_row_dot_q(&trits, &rowd, &aq, 0, n_in);
        let packed_dot = unsafe { packed_row_dot_t143(&row, &aq, 0, n_in) };
        assert_eq!(
            ref_dot.to_bits(),
            packed_dot.to_bits(),
            "packed dot {packed_dot} != dequant+sdot reference {ref_dot}"
        );
    }

    #[test]
    fn wht_matches_sylvester_hadamard_matrix() {
        let n = HAD_BLOCK;
        let mut x = Vec::with_capacity(n);
        let mut state = 12345u64;
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x.push(((state >> 33) % 2001) as f32 / 1000.0 - 1.0);
        }
        let mut fast = x.clone();
        wht_block(&mut fast);
        let scale = 1.0f32 / (n as f32).sqrt();
        for i in 0..n {
            let mut acc = 0.0f32;
            for j in 0..n {
                let sign = if (i & j).count_ones() & 1 == 1 { -scale } else { scale };
                acc += sign * x[j];
            }
            assert!(
                (fast[i] - acc).abs() < 1e-2,
                "WHT mismatch at {i}: {} vs {}",
                fast[i],
                acc
            );
        }
    }

    fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.into_iter().take(5).map(|i| (i, logits[i])).collect()
    }

    #[test]
    fn qwen35_load_prefill_and_greedy_generate() {
        if !std::path::Path::new(MODEL_PATH).exists() {
            println!("model not found at {MODEL_PATH}; skipping");
            return;
        }
        let t0 = Instant::now();
        let mut gguf = GgufFile::open(MODEL_PATH).expect("open gguf");
        let eos = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(248046);
        let vocab = gguf.take_string_array("tokenizer.ggml.tokens").expect("vocab");
        let mut prompt_ids = Vec::new();
        for piece in ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"] {
            let id = vocab.iter().position(|t| t == piece).expect(piece) as u32;
            println!("token {piece:?} -> id {id}");
            prompt_ids.push(id);
        }
        println!(
            "config: dim={} layers={} heads={} kv={} hd={} ffn={} vocab={}",
            5120, 64, 24, 4, 256, 17408, vocab.len()
        );
        let mut model = Qwen35Model::from_gguf(gguf, 4096).expect("load qwen35");
        println!("load time: {:?}", t0.elapsed());
        let t1 = Instant::now();
        let (first_next, first_top) = {
            let logits = model.forward_prefill_all_with_logits(&prompt_ids, 0);
            println!("first-run top-5: {:?}", top5(logits));
            let t = top5(logits);
            (t[0].0 as u32, t)
        };
        println!("prefill+logits time: {:?}", t1.elapsed());
        let _ = first_top;

        let mut pos = prompt_ids.len();
        let mut next = first_next;
        let mut generated = String::new();
        for step in 0..8 {
            let piece = vocab.get(next as usize).cloned().unwrap_or_default();
            println!("gen[{step}] id={next} piece={piece:?}");
            generated.push_str(&piece.replace('Ġ', " ").replace('Ċ', "\n"));
            if next == eos {
                println!("[eos reached]");
                break;
            }
            let nxt = {
                let logits = model.forward(next, pos);
                top5(logits)[0].0 as u32
            };
            pos += 1;
            next = nxt;
        }
        println!("PROMPT: The capital of France is");
        println!("GENERATED: {generated}");
        assert!(!generated.is_empty());
    }

    /// LCG noise for synthetic packed rows (deterministic, no rand dep).
    fn noise_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect()
    }

    /// The trit-i8 dequant must agree value-for-value with the frozen f32
    /// reference decoder: trits[i] · scales[i/128] == out[i] exactly.
    #[test]
    fn trit_i8_dequant_matches_f32_reference() {
        let rows = 6usize;
        let n_in = 3 * PTQ1_0_BLOCK_VALUES; // 3 bands
        let row_bytes = row_bytes_ptq1_0(n_in);
        let data = noise_bytes(0x9E3779B97F4A7C15, rows * row_bytes);
        for r in 0..rows {
            let src = &data[r * row_bytes..(r + 1) * row_bytes];
            let mut want = vec![0.0f32; n_in];
            ptq1_0_dequant_row(src, &mut want);
            let mut trits = vec![0i8; n_in];
            let mut scales = vec![0.0f32; n_in / PTQ1_0_BLOCK_VALUES];
            dequant_trits_i8(src, &mut trits, &mut scales);
            for i in 0..n_in {
                let got = trits[i] as f32 * scales[i / PTQ1_0_BLOCK_VALUES];
                assert_eq!(got.to_bits(), want[i].to_bits(), "row {r} value {i}");
            }
        }
    }

    /// The resident trit slab (SOFUU_T143_SLAB decode path) must dot to
    /// bit-identical results as the file-direct packed path, because the
    /// slab row is exactly `dequant_trits_i8`'s output and both fold through
    /// `trit_row_dot_q`'s band order. Threshold-sized so the slab path is
    /// what a real 27B decode takes (n_out*n_in >= PAR_MAC_THRESHOLD).
    #[test]
    fn t143_slab_dot_bitmatches_file_direct_gemv() {
        let n_in = 8 * PTQ1_0_BLOCK_VALUES;
        let rows = 2112usize; // 2112 * 1024 = 2,162,688 MACs >= threshold
        let row_bytes = row_bytes_ptq1_0(n_in);
        let mut data = noise_bytes(0xBA5EBA11F00DFACE, rows * row_bytes);
        let blocks = n_in / PTQ1_0_BLOCK_VALUES;
        for blk in 0..(rows * blocks) {
            let off = blk * PTQ1_0_BLOCK_BYTES;
            data[off + PTQ1_0_BLOCK_BYTES - 2] = 0x00;
            data[off + PTQ1_0_BLOCK_BYTES - 1] = 0x3C;
        }
        let acts: Vec<f32> = noise_bytes(0x0123456789ABCDEF, n_in)
            .iter()
            .map(|&b| (b as f32 / 255.0 - 0.5) * 4.0)
            .collect();
        let m = RawMat { off: 0, n_in, n_out: rows, row_bytes, ty: GGML_TYPE_PTQ1_0, xform: Xform::None };

        let mut file_direct = vec![0.0f32; rows];
        gemv_t143(&data, &m, &acts, &mut file_direct);

        let slab = ensure_t143_slab(&data, &m).expect("slab build");
        let aq = quantize_acts(&acts, 1, n_in, blocks);
        for r in 0..rows {
            let row = &slab.trits[r * n_in..(r + 1) * n_in];
            let rowd = &slab.rowd[r * blocks..(r + 1) * blocks];
            let got = trit_row_dot_q(row, rowd, &aq, 0, n_in);
            assert_eq!(
                got.to_bits(),
                file_direct[r].to_bits(),
                "slab row {r} != file-direct"
            );
        }
    }

    /// The tiled GEMM must reproduce the scalar per-(row, token) band dot
    /// path for every element, tail shapes included (37 rows × 19 tokens is
    /// ragged on both axes).
    #[test]
    fn tiled_gemm_matches_scalar_band_dot() {
        let n_in = 2 * PTQ1_0_BLOCK_VALUES;
        let rows = 37usize;
        let n_tok = 19usize;
        let row_bytes = row_bytes_ptq1_0(n_in);
        let mut data = noise_bytes(0xDEADBEEFCAFEBABE, rows * row_bytes);
        // Pin f16 scales to 1.0: random scale words are sometimes NaN/inf,
        // and a NaN row makes bit comparisons meaningless noise.
        for blk in 0..(rows * (n_in / PTQ1_0_BLOCK_VALUES)) {
            let off = blk * PTQ1_0_BLOCK_BYTES;
            data[off + PTQ1_0_BLOCK_BYTES - 2] = 0x00;
            data[off + PTQ1_0_BLOCK_BYTES - 1] = 0x3C;
        }
        let acts: Vec<f32> = noise_bytes(0x1234567890ABCDEF, n_tok * n_in)
            .iter()
            .map(|&b| (b as f32 / 255.0 - 0.5) * 4.0)
            .collect();
        let m = RawMat { off: 0, n_in, n_out: rows, row_bytes, ty: GGML_TYPE_PTQ1_0, xform: Xform::None };

        // Expected: per row, per token — the same band-ordered scalar dot.
        let mut want = vec![0.0f32; rows * n_tok];
        let mut trits = vec![0i8; n_in];
        let mut rowd = vec![0.0f32; n_in / PTQ1_0_BLOCK_VALUES];
        for r in 0..rows {
            dequant_trits_i8(&data[r * row_bytes..(r + 1) * row_bytes], &mut trits, &mut rowd);
            for t in 0..n_tok {
                want[r * n_tok + t] = trit_row_dot(&trits, &rowd, &acts[t * n_in..(t + 1) * n_in]);
            }
        }
        // Pass the full matrix through the tiled kernel in uneven row slabs
        // to exercise the slab-boundary tail as well (rows_per_pass clamps to
        // GEMM_SLAB_BYTES / n_in — huge here — so split manually).
        let mut got = vec![0.0f32; rows * n_tok];
        for (off, len) in [(0usize, 7usize), (7, 13), (20, 17)] {
            let tile = gemm_rows_t143(&data, &m, n_tok, &acts, off, len);
            got[off * n_tok..(off + len) * n_tok].copy_from_slice(&tile);
        }
        assert_eq!(got.len(), want.len());
        for i in 0..want.len() {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "elem {i}");
        }
    }

    /// Phase 4a: the token-tiled direct runner must be BIT-identical to the
    /// slab sdot path it replaces (same band/fold order, same lane pattern).
    #[test]
    fn direct_tile_bitmatches_sdot_path() {
        let n_in = 2 * PTQ1_0_BLOCK_VALUES;
        let rows = 37usize;
        let n_tok = 19usize; // uneven: exercises the 4-token tail
        let row_bytes = row_bytes_ptq1_0(n_in);
        let mut data = noise_bytes(0xDEADBEEFCAFEBABE, rows * row_bytes);
        for blk in 0..(rows * (n_in / PTQ1_0_BLOCK_VALUES)) {
            let off = blk * PTQ1_0_BLOCK_BYTES;
            data[off + PTQ1_0_BLOCK_BYTES - 2] = 0x00;
            data[off + PTQ1_0_BLOCK_BYTES - 1] = 0x3C;
        }
        let acts: Vec<f32> = noise_bytes(0x1234567890ABCDEF, n_tok * n_in)
            .iter()
            .map(|&b| (b as f32 / 255.0 - 0.5) * 4.0)
            .collect();
        let m = RawMat { off: 0, n_in, n_out: rows, row_bytes, ty: GGML_TYPE_PTQ1_0, xform: Xform::None };
        let want = gemm_rows_t143_sdot(&data, &m, n_tok, &acts, 0, rows);
        let got = gemm_rows_t143_direct(&data, &m, n_tok, &acts, 0, rows);
        for i in 0..want.len() {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "elem {i}: direct {} vs sdot {}",
                got[i],
                want[i]
            );
        }
    }

    /// Speed harness (run with --release -- --ignored --nocapture): the
    /// register-tiled sweep should be several times faster than the old
    /// per-(row, token) dot8 sweep on PTQ1_0-shaped work.
    #[test]
    #[ignore]
    fn tiled_gemm_speed_vs_scalar_sweep() {
        let n_in = 5120usize;
        let rows = 17408usize;
        let n_tok = 128usize;
        let row_bytes = row_bytes_ptq1_0(n_in);
        let mut data = noise_bytes(0xFEED, rows * row_bytes);
        // Random bytes make a few percent of f16 scales NaN/inf, which
        // poisons whole rows and makes any comparison meaningless — pin the
        // scale word of every block to 1.0 and exercise only the trit path.
        for blk in 0..(rows * (n_in / PTQ1_0_BLOCK_VALUES)) {
            let off = blk * PTQ1_0_BLOCK_BYTES;
            data[off + PTQ1_0_BLOCK_BYTES - 2] = 0x00;
            data[off + PTQ1_0_BLOCK_BYTES - 1] = 0x3C;
        }
        let acts: Vec<f32> = noise_bytes(0xBEEF, n_tok * n_in)
            .iter()
            .map(|&b| (b as f32 / 255.0 - 0.5) * 2.0)
            .collect();
        let m = RawMat { off: 0, n_in, n_out: rows, row_bytes, ty: GGML_TYPE_PTQ1_0, xform: Xform::None };

        let t0 = std::time::Instant::now();
        let mut scalar = vec![0.0f32; rows * n_tok];
        let mut slab = vec![0.0f32; n_in];
        let mut rowd = vec![0.0f32; n_in / PTQ1_0_BLOCK_VALUES];
        let mut trits = vec![0i8; n_in];
        for r in 0..rows {
            dequant_trits_i8(&data[r * row_bytes..(r + 1) * row_bytes], &mut trits, &mut rowd);
            for t in 0..n_tok {
                scalar[r * n_tok + t] = trit_row_dot(&trits, &rowd, &acts[t * n_in..(t + 1) * n_in]);
            }
        }
        let t_scalar = t0.elapsed();

        let t1 = std::time::Instant::now();
        let tiled = gemm_rows_t143(&data, &m, n_tok, &acts, 0, rows);
        let t_tiled = t1.elapsed();

        let t2 = std::time::Instant::now();
        let direct = gemm_rows_t143_direct(&data, &m, n_tok, &acts, 0, rows);
        let t_direct = t2.elapsed();

        // The path actually used on dotprod cores today — the fair A/B for
        // the direct runner (both int8): these must agree BIT for BIT.
        let t3 = std::time::Instant::now();
        let sdot = gemm_rows_t143_sdot(&data, &m, n_tok, &acts, 0, rows);
        let t_sdot = t3.elapsed();
        let bit = direct
            .iter()
            .zip(sdot.iter())
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();

        let mut ok = 0usize;
        for (a, b) in scalar.iter().zip(tiled.iter()) {
            if (a - b).abs() <= a.abs().max(1.0) * 1e-4 {
                ok += 1;
            }
        }
        let agree_direct = direct
            .iter()
            .zip(scalar.iter())
            .filter(|(a, b)| (**a - **b).abs() <= a.abs().max(1.0) * 1e-4)
            .count();
        println!(
            "scalar band-dot sweep: {t_scalar:?} | tiled(f32): {t_tiled:?} | sdot-slab: {t_sdot:?} | direct: {t_direct:?} | direct {:.2}x vs sdot-slab | agree(f32 paths) {ok}/{} | direct-vs-scalar (int8, quant err expected) {agree_direct}/{} | direct-vs-sdot BIT {bit}/{}",
            t_sdot.as_secs_f64() / t_direct.as_secs_f64().max(1e-9),
            scalar.len(),
            scalar.len(),
            direct.len()
        );
        assert!(ok == scalar.len(), "tile differs from scalar beyond 1e-4 on {}/{}", scalar.len() - ok, scalar.len());
    }

    /// Probe the real model's ternary sparsity: zero-trit fraction, sign
    /// balance and f16-scale sanity over sampled rows from several weight
    /// classes. Run with --ignored --nocapture. The zero fraction decides
    /// whether a zero-skipping packed GEMM can chase 50+ tk/s prefill on CPU.
    #[test]
    #[ignore]
    fn probe_ternary_zero_fraction() {
        let gguf = GgufFile::open(MODEL_PATH).expect("open gguf");
        let data = gguf.mmap.as_slice();
        let probes: &[(&str, usize)] = &[
            ("token_embd.weight", 2000),
            ("blk.0.ffn_gate.weight", 4000),
            ("blk.47.ffn_up.weight", 2000),
            ("blk.10.attn_qkv.weight", 2000),
            ("blk.63.attn_out.weight", 600),
        ];
        let mut trits: Vec<i8> = vec![0; 5120];
        let mut rowd: Vec<f32> = vec![0.0; 5120 / PTQ1_0_BLOCK_VALUES];
        for &(want, max_rows) in probes {
            let Some(info) = gguf.tensors.get(want) else {
                println!("  {want}: not found");
                continue;
            };
            if info.tensor_type != GGML_TYPE_PTQ1_0 {
                println!("  {want}: type {} (not PTQ1_0)", info.tensor_type);
                continue;
            }
            let n_in = info.shape[0];
            if trits.len() < n_in {
                trits.resize(n_in, 0);
                rowd.resize(n_in / PTQ1_0_BLOCK_VALUES, 0.0);
            }
            let row_bytes = row_bytes_ptq1_0(n_in);
            let rows = info.shape[1].min(max_rows);
            let off0 = gguf.data_offset + info.offset;
            let (mut z, mut pos, mut neg) = (0u64, 0u64, 0u64);
            let mut nonfinite_scales = 0u64;
            for r in 0..rows {
                let off = off0 + r * row_bytes;
                dequant_trits_i8(&data[off..off + row_bytes], &mut trits[..n_in], &mut rowd[..n_in / PTQ1_0_BLOCK_VALUES]);
                for (b, &d) in rowd[..n_in / PTQ1_0_BLOCK_VALUES].iter().enumerate() {
                    if !d.is_finite() {
                        nonfinite_scales += 1;
                    }
                }
                for &v in &trits[..n_in] {
                    match v {
                        0 => z += 1,
                        1 => pos += 1,
                        -1 => neg += 1,
                        _ => {}
                    }
                }
            }
            let total = rows as u64 * n_in as u64;
            let pct = |x: u64| (x as f64 * 100.0 / total.max(1) as f64);
            println!(
                "{want}: rows {rows} | zeros {:.1}% | +1 {:.1}% | -1 {:.1}% | non-finite scales {nonfinite_scales}",
                pct(z),
                pct(pos),
                pct(neg)
            );
        }
    }

    /// The int8 sdot path must stay within per-band absmax activation
    /// quantization error of the exact f32 path, and the sdot GEMV must agree
    /// bit-for-bit with the sdot GEMM (same band/fold order).
    #[test]
    fn sdot_path_matches_f32_within_quantization_error() {
        let n_in = 2 * PTQ1_0_BLOCK_VALUES;
        let rows = 23usize;
        let n_tok = 5usize;
        let row_bytes = row_bytes_ptq1_0(n_in);
        let mut data = noise_bytes(0xA5A5_5A5A_A5A5_5A5A, rows * row_bytes);
        for blk in 0..(rows * (n_in / PTQ1_0_BLOCK_VALUES)) {
            let off = blk * PTQ1_0_BLOCK_BYTES;
            data[off + PTQ1_0_BLOCK_BYTES - 2] = 0x00;
            data[off + PTQ1_0_BLOCK_BYTES - 1] = 0x3C; // f16 scale 1.0
        }
        let acts: Vec<f32> = noise_bytes(0x0F1E_2D3C_4B5A_6978, n_tok * n_in)
            .iter()
            .map(|&b| (b as f32 / 255.0 - 0.5) * 4.0)
            .collect();
        let m = RawMat { off: 0, n_in, n_out: rows, row_bytes, ty: GGML_TYPE_PTQ1_0, xform: Xform::None };

        let exact = gemm_rows_t143(&data, &m, n_tok, &acts, 0, rows);
        let approx = gemm_rows_t143_sdot(&data, &m, n_tok, &acts, 0, rows);
        // Per-element relative error is meaningless where the exact dot
        // cancels near zero; absmax band quantization bounds the error
        // relative to the signal RMS (~0.4% by construction).
        let (mut se, mut ss) = (0.0f64, 0.0f64);
        for i in 0..approx.len() {
            let d = (approx[i] - exact[i]) as f64;
            se += d * d;
            ss += exact[i] as f64 * exact[i] as f64;
        }
        let rel = (se / ss.max(1e-12)).sqrt();
        assert!(
            rel < 0.02,
            "sdot path deviates {rel} rms-relative from the exact path (> 0.02)"
        );

        // GEMV (single-token path) must reproduce the GEMM's sdot column
        // bit-for-bit: same quantization, same band fold order.
        let aq = quantize_acts(&acts, n_tok, n_in, n_in / PTQ1_0_BLOCK_VALUES);
        let mut trits = vec![0i8; n_in];
        let mut rowd = vec![0.0f32; n_in / PTQ1_0_BLOCK_VALUES];
        for r in 0..rows {
            dequant_trits_i8(&data[r * row_bytes..(r + 1) * row_bytes], &mut trits, &mut rowd);
            for t in 0..n_tok {
                let gemv_result = trit_row_dot_q(&trits, &rowd, &aq, t, n_in);
                let gemm_result = approx[r * n_tok + t];
                assert_eq!(
                    gemv_result.to_bits(),
                    gemm_result.to_bits(),
                    "row {r} token {t} disagrees between GEMV and GEMM sdot paths"
                );
            }
        }
    }
}
