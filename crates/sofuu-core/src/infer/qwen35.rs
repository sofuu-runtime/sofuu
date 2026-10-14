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
use super::quant::{bf16_to_f32, ptq1_0_dequant_row};

/// GGML tensor type id for PTQ1_0 (prism ternary, 28 bytes per 128 values).
const GGML_TYPE_PTQ1_0: u32 = 143;
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
fn thread_budget() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8)
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

fn wht_blocks(x: &mut [f32]) {
    for block in x.chunks_exact_mut(HAD_BLOCK) {
        wht_block(block);
    }
}

/// Prepared folded-matmul activation: dst[i] = S[i]·src[i], then per-block H.
fn hadamard_prep(dst: &mut [f32], src: &[f32], signs: &[f32]) {
    for (d, (&s, &v)) in dst.iter_mut().zip(signs.iter().zip(src.iter())) {
        *d = s * v;
    }
    wht_blocks(dst);
}

/// Signs + Hadamard applied in place (activation already staged in x).
fn hadamard_prep_in_place(x: &mut [f32], signs: &[f32]) {
    for (v, &s) in x.iter_mut().zip(signs) {
        *v *= s;
    }
    wht_blocks(x);
}

/// GDN regrouping for the folded ssm_out activation. The folded weight expects
/// the activation tiled [hd=128, nk=16, rep=3] (feature f = d + 128·(nk + 16·rep));
/// regroup to [hd, rep, nk]: f' = d + 128·rep + 384·nk (fork: permute(0,2,1,3)).
fn gdn_perm(dst: &mut [f32], src: &[f32]) {
    const HD: usize = 128;
    const NK: usize = 16;
    const REP: usize = 3;
    debug_assert_eq!(src.len(), HD * NK * REP);
    for rep in 0..REP {
        for nk in 0..NK {
            for d in 0..HD {
                let fi = d + HD * (nk + NK * rep);
                let fo = d + HD * rep + HD * REP * nk;
                dst[fo] = src[fi];
            }
        }
    }
}

/// GEMV against a PTQ1_0 matrix still in the mmap: dequantize each row into
/// `scratch` (must hold n_in f32) and dot with the prepared activation. Jobs
/// at or above PAR_MAC_THRESHOLD split the output rows into contiguous
/// deterministic chunks via `std::thread::scope` (one spawn per chunk); each
/// thread allocates its own n_in-wide row scratch — the shared `scratch` is
/// only used by the single-thread path. Every row runs the same
/// dequantize-then-dot8 sequence as the sequential version, so results are
/// bit-identical regardless of thread count.
fn gemv_t143(data: &[u8], m: &RawMat, act: &[f32], out: &mut [f32], scratch: &mut [f32]) {
    let n_out = out.len();
    let threads = if n_out * m.n_in >= PAR_MAC_THRESHOLD {
        thread_budget().min(n_out)
    } else {
        1
    };
    if threads <= 1 {
        let act = &act[..m.n_in];
        for (r, o) in out.iter_mut().enumerate() {
            let row = m.row_slice(data, r);
            ptq1_0_dequant_row(row, &mut scratch[..m.n_in]);
            *o = dot8(&scratch[..m.n_in], act);
        }
        return;
    }
    let act = &act[..m.n_in];
    std::thread::scope(|scope| {
        let mut rest: &mut [f32] = out;
        let mut r0 = 0usize;
        for t in 0..threads {
            let take = n_out * (t + 1) / threads - r0;
            let tmp = rest;
            let (chunk, tail) = tmp.split_at_mut(take);
            rest = tail;
            let rows = r0;
            scope.spawn(move || {
                let mut row = vec![0.0f32; m.n_in];
                for (i, o) in chunk.iter_mut().enumerate() {
                    ptq1_0_dequant_row(m.row_slice(data, rows + i), &mut row);
                    *o = dot8(&row, act);
                }
            });
            r0 += take;
        }
    });
}

/// One deterministic row range of the batched GEMM: weight rows
/// [r_off, r_off + rows) are dequantized exactly once each into per-pass
/// slabs (bounded by GEMM_SLAB_BYTES) and each slab row is dotted, with the
/// same dot8 call gemv_t143 makes, against every token's activation. Returns
/// the r-major tile [rows][n_tok].
fn gemm_rows_t143(
    data: &[u8],
    m: &RawMat,
    n_tok: usize,
    acts: &[f32],
    r_off: usize,
    rows: usize,
) -> Vec<f32> {
    let n_in = m.n_in;
    let mut res = vec![0.0f32; rows * n_tok];
    let rows_per_pass = (GEMM_SLAB_BYTES / (n_in * 4)).clamp(1, rows.max(1));
    let mut slab = vec![0.0f32; rows_per_pass * n_in];
    let mut base = 0usize;
    while base < rows {
        let take = rows_per_pass.min(rows - base);
        for i in 0..take {
            ptq1_0_dequant_row(
                m.row_slice(data, r_off + base + i),
                &mut slab[i * n_in..(i + 1) * n_in],
            );
        }
        for i in 0..take {
            let w = &slab[i * n_in..(i + 1) * n_in];
            let tile = &mut res[(base + i) * n_tok..(base + i + 1) * n_tok];
            for (t, o) in tile.iter_mut().enumerate() {
                *o = dot8(w, &acts[t * n_in..(t + 1) * n_in]);
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
                gemm_rows_t143(data, m, n_tok, acts, r0, r1 - r0)
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
    if info.tensor_type != expect_type {
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
    let row_bytes = match expect_type {
        GGML_TYPE_PTQ1_0 => row_bytes_ptq1_0(n_in),
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
    Ok(RawMat { off, n_in, n_out, row_bytes, xform })
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
    if info.tensor_type != GGML_TYPE_BF16 {
        return Err(format!("tensor {name}: expected BF16, got type {}", info.tensor_type));
    }
    let n: usize = info.shape.iter().product();
    if n != rows * cols {
        return Err(format!("tensor {name}: shape {:?} ≠ [{cols},{rows}]", info.shape));
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
        let had_version = gguf.get_u32("prism.hadamard.version").unwrap_or(0);
        let had_block = gguf.get_u32("prism.hadamard.block_size").unwrap_or(0) as usize;
        if had_version != 0 && had_block != HAD_BLOCK {
            return Err(format!("qwen35: unsupported prism.hadamard.block_size {had_block}"));
        }
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

        for li in 0..n_layers {
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
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            vocab_size,
            norm_eps,
            rope_freq_base: rope_base,
            max_seq_len: seq_cap,
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
        let data = self.gguf.mmap.as_slice();
        let row = self.embd.row_slice(data, t);
        ptq1_0_dequant_row(row, &mut self.rowbuf[..dim]);
        match self.embd.xform {
            Xform::Inverse(i) => {
                wht_blocks(&mut self.rowbuf[..dim]);
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
        let mut rowbuf = std::mem::take(&mut self.rowbuf);
        match (s_dim, &self.signs) {
            (Some(i), table) => hadamard_prep(&mut prep, &self.hbuf, &table[i].1),
            _ => prep.copy_from_slice(&self.hbuf),
        }
        let data = self.gguf.mmap.as_slice();
        gemv_t143(data, qkv_mat, &prep, &mut self.qkv, &mut rowbuf);
        gemv_t143(data, gate_mat, &prep, &mut self.zg, &mut rowbuf);
        gemv_f32(alpha, dim, &self.hbuf, &mut self.g48a);
        gemv_f32(beta, dim, &self.hbuf, &mut self.g48b);
        self.prep512 = prep;
        self.rowbuf = rowbuf;

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
        gdn_perm(&mut self.prep_mid, &self.gdn_out);
        let data = self.gguf.mmap.as_slice();
        let mut rowbuf = std::mem::take(&mut self.rowbuf);
        match (out_mat.sign_idx(), &self.signs) {
            (Some(i), table) => hadamard_prep_in_place(&mut self.prep_mid, &table[i].1),
            _ => {}
        }
        gemv_t143(data, out_mat, &self.prep_mid, &mut self.mix_out, &mut rowbuf);
        self.rowbuf = rowbuf;
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
        let mut rowbuf = std::mem::take(&mut self.rowbuf);
        match (q_mat.sign_idx(), &self.signs) {
            (Some(i), table) => hadamard_prep(&mut prep, &self.hbuf, &table[i].1),
            _ => prep.copy_from_slice(&self.hbuf),
        }
        let data = self.gguf.mmap.as_slice();
        gemv_t143(data, q_mat, &prep, &mut self.qg, &mut rowbuf);
        gemv_t143(data, k_mat, &prep, &mut self.kh, &mut rowbuf);
        gemv_t143(data, v_mat, &prep, &mut self.vh, &mut rowbuf);
        self.prep512 = prep;
        self.rowbuf = rowbuf;

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
            (Some(i), table) => hadamard_prep_in_place(&mut self.attn3, &table[i].1),
            _ => {}
        }
        let data = self.gguf.mmap.as_slice();
        let mut rowbuf = std::mem::take(&mut self.rowbuf);
        gemv_t143(data, o_mat, &self.attn3, &mut self.mix_out, &mut rowbuf);
        self.rowbuf = rowbuf;
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
                let mut rowbuf = std::mem::take(&mut self.rowbuf);
                match (self.layers[li].ffn_gate.sign_idx(), &self.signs) {
                    (Some(i), table) => hadamard_prep(&mut prep, &self.hbuf, &table[i].1),
                    _ => prep.copy_from_slice(&self.hbuf),
                }
                let data = self.gguf.mmap.as_slice();
                gemv_t143(data, &self.layers[li].ffn_gate, &prep, &mut self.ffn_g, &mut rowbuf);
                gemv_t143(data, &self.layers[li].ffn_up, &prep, &mut self.ffn_u, &mut rowbuf);
                self.prep512 = prep;
                self.rowbuf = rowbuf;
            }
            for (g, &u) in self.ffn_g.iter_mut().zip(self.ffn_u.iter()) {
                *g = silu(*g) * u;
            }

            {
                let mut prep = std::mem::take(&mut self.prep_ff);
                let mut rowbuf = std::mem::take(&mut self.rowbuf);
                match (self.layers[li].ffn_down.sign_idx(), &self.signs) {
                    (Some(i), table) => hadamard_prep(&mut prep, &self.ffn_g, &table[i].1),
                    _ => prep.copy_from_slice(&self.ffn_g),
                }
                let data = self.gguf.mmap.as_slice();
                gemv_t143(data, &self.layers[li].ffn_down, &prep, &mut self.mix_out, &mut rowbuf);
                self.prep_ff = prep;
                self.rowbuf = rowbuf;
            }
            for (x, f) in self.x.iter_mut().zip(&self.mix_out) {
                *x += f;
            }
        }

        if compute_logits {
            rmsnorm(&mut self.x, &self.output_norm, eps);
            match (self.output.sign_idx(), &self.signs) {
                (Some(i), table) => hadamard_prep_in_place(&mut self.x, &table[i].1),
                _ => {}
            }
            let data = self.gguf.mmap.as_slice();
            let mut rowbuf = std::mem::take(&mut self.rowbuf);
            gemv_t143(data, &self.output, &self.x, &mut self.logits, &mut rowbuf);
            self.rowbuf = rowbuf;
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

        let data = self.gguf.mmap.as_slice();
        for (t, &tok) in tokens.iter().enumerate() {
            let tid = (tok as usize).min(self.embd.n_out.saturating_sub(1));
            let row = self.embd.row_slice(data, tid);
            ptq1_0_dequant_row(row, &mut escratch[..dim]);
            match self.embd.xform {
                Xform::Inverse(i) => {
                    wht_blocks(&mut escratch[..dim]);
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
                            hadamard_prep(&mut bp[trow..trow + dim], &bh[trow..trow + dim], &table[i].1)
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
                    gdn_perm(&mut bpm[grow..grow + d_inner], &bgdn[grow..grow + d_inner]);
                    match (out_mat.sign_idx(), &self.signs) {
                        (Some(i), table) => {
                            hadamard_prep_in_place(&mut bpm[grow..grow + d_inner], &table[i].1)
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
                            hadamard_prep(&mut bp[trow..trow + dim], &bh[trow..trow + dim], &table[i].1)
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
                            hadamard_prep_in_place(&mut battn[t * attn_w..(t + 1) * attn_w], &table[i].1)
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
                        hadamard_prep(&mut bp[t * dim..(t + 1) * dim], &bh[t * dim..(t + 1) * dim], &table[i].1)
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
                        hadamard_prep_in_place(&mut bfg[frow..frow + hidden], &table[i].1)
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
                (Some(i), table) => hadamard_prep_in_place(&mut self.x, &table[i].1),
                _ => {}
            }
            let mut rowbuf = std::mem::take(&mut self.rowbuf);
            gemv_t143(data, &self.output, &self.x, &mut self.logits, &mut rowbuf);
            self.rowbuf = rowbuf;

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

    const MODEL_PATH: &str = "/Users/priyanshuboruah/Downloads/Ternary-Bonsai-2-27B-PTQ1_0.gguf";

    /// The Hadamard side of the fold must compute exactly the specified
    /// normalized Sylvester WH: H[i][j] = (±1)^popcount(i&j) / sqrt(1024).
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
}
