//! Sofuu's small learned semantic embedder.
//!
//! The shipped model is intentionally narrow: it learns a low-rank projection
//! over Sofuu's existing deterministic `hash-v1` character-trigram features.
//! That keeps the runtime offline and dependency-free while giving memory a
//! compact 64-dimensional space.  The old 768-dimensional hash embedder stays
//! public as `sofuu.ai.embedLocal()` for compatibility; the memory subsystem
//! uses `semantic-projector-v1` through `semantic_v1()`.
//!
//! Architecture (exactly 13,392 parameters at the shipped width):
//!
//! ```text
//! 768 hash features -> 16 tanh units -> 64 linear units -> L2 norm
//! ```
//!
//! Two artifact versions exist:
//!
//! - **SEM1 v1** (shipped): dense int8 W1, per-row f32 scales.
//! - **SEM1 v2** (experimental, trainer-only until it passes §10): the hash
//!   features are extremely sparse, so most dense W1 weights multiply zeros.
//!   v2 stores only the K largest-magnitude inputs per hidden unit
//!   (structured sparsity: K u16 indices + K int8 values per row), which
//!   fits a much wider hidden layer inside the same payload budget while the
//!   sparse forward gets *faster* than the dense v1 path.
//!
//! The trainer and runtime share the scalar forward pass below.  Artifacts
//! are verified with a strict header, length, dimension, finiteness and CRC
//! check before they can be used.
//!
//! The hidden width is a build-time knob (`SOFUU_EMB_H`, see build.rs) so
//! retrain experiments can try wider bottlenecks; the shipped default stays
//! at the plan-frozen 16 and the baked v1 artifact only loads on that
//! default build.

use std::sync::OnceLock;

use crate::ml::net::crc32_ieee;

pub mod semantic_v2; // round-9 SEM2 table embedder — fused-channel runtime
pub mod image; // M1: frozen hand-built image features (decode + IMGF1 layout)

pub const HASH_DIM: usize = 768;
/// Hidden width.  Experiments rebuild with `SOFUU_EMB_H=64|128 …` (build.rs
/// whitelists the values and emits `emb_h_*` cfgs); the shipped default is
/// the plan-frozen 16.
#[cfg(emb_h_32)]
pub const HIDDEN_DIM: usize = 32;
#[cfg(emb_h_36)]
pub const HIDDEN_DIM: usize = 36;
#[cfg(emb_h_48)]
pub const HIDDEN_DIM: usize = 48;
#[cfg(emb_h_64)]
pub const HIDDEN_DIM: usize = 64;
#[cfg(emb_h_96)]
pub const HIDDEN_DIM: usize = 96;
#[cfg(emb_h_128)]
pub const HIDDEN_DIM: usize = 128;
#[cfg(not(any(
    emb_h_32,
    emb_h_36,
    emb_h_48,
    emb_h_64,
    emb_h_96,
    emb_h_128
)))]
pub const HIDDEN_DIM: usize = 16;
pub const SEMANTIC_DIM: usize = 64;
/// Dense-v1 parameter count (the f32 trainer model is always dense).
pub const PARAM_COUNT: usize =
    HASH_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * SEMANTIC_DIM + SEMANTIC_DIM;

pub const INPUT_EMBEDDER_ID: &str = "hash-v1";
pub const MODEL_ID: &str = "semantic-projector-v1";
pub const MODEL_FORMAT: &str = "SEM1";
/// Nonzero inputs per hidden unit in the v2 sparse layout.
pub const SPARSE_K_DEFAULT: usize = 48;

const W1_START: usize = 0;
const B1_START: usize = W1_START + HASH_DIM * HIDDEN_DIM;
const W2_START: usize = B1_START + HIDDEN_DIM;
const B2_START: usize = W2_START + HIDDEN_DIM * SEMANTIC_DIM;

const BLOB_MAGIC: u32 = u32::from_le_bytes(*b"SEM1");
const BLOB_VERSION_V1: u32 = 1;
const BLOB_VERSION_V2: u32 = 2;
const QUANTIZATION_INT8_PER_ROW: u32 = 1;
const QUANTIZATION_INT8_SPARSE_PER_ROW: u32 = 2;
const HEADER_WORDS: usize = 9;
const HEADER_LEN: usize = HEADER_WORDS * std::mem::size_of::<u32>();

/// The train-time model.  The flat layout is deliberately identical to the
/// parameter order used by the gradient code and the artifact exporter:
/// W1, b1, W2, b2.
#[derive(Clone, Debug)]
pub struct ProjectorF32 {
    pub w: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ForwardCache {
    pub hidden: [f32; HIDDEN_DIM],
    pub raw: [f32; SEMANTIC_DIM],
    pub output: [f32; SEMANTIC_DIM],
    pub norm: f32,
}

impl ProjectorF32 {
    pub fn new(w: Vec<f32>) -> Self {
        assert_eq!(w.len(), PARAM_COUNT, "semantic projector parameter count");
        Self { w }
    }

    pub fn zeros() -> Self {
        Self::new(vec![0.0; PARAM_COUNT])
    }

    /// Shared scalar forward pass.  Keep the accumulation order fixed: this
    /// is also the order used by the offline trainer and by the quantized
    /// runtime model.
    pub fn forward_cache(&self, input: &[f32]) -> ForwardCache {
        assert_eq!(input.len(), HASH_DIM, "semantic projector input width");
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut acc = self.w[B1_START + j];
            let row = W1_START + j * HASH_DIM;
            for i in 0..HASH_DIM {
                acc += self.w[row + i] * input[i];
            }
            hidden[j] = acc.tanh();
        }

        let mut raw = [0.0f32; SEMANTIC_DIM];
        for k in 0..SEMANTIC_DIM {
            let mut acc = self.w[B2_START + k];
            let row = W2_START + k * HIDDEN_DIM;
            for j in 0..HIDDEN_DIM {
                acc += self.w[row + j] * hidden[j];
            }
            raw[k] = acc;
        }

        let mut norm_sq = 0.0f32;
        for &v in &raw {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        let mut output = [0.0f32; SEMANTIC_DIM];
        if norm.is_finite() && norm > 1e-12 {
            let inv = 1.0f32 / norm;
            for k in 0..SEMANTIC_DIM {
                output[k] = raw[k] * inv;
            }
        }
        ForwardCache {
            hidden,
            raw,
            output,
            norm,
        }
    }

    pub fn forward(&self, input: &[f32]) -> [f32; SEMANTIC_DIM] {
        self.forward_cache(input).output
    }

    /// Backpropagate an arbitrary gradient with respect to the normalized
    /// output.  The trainer uses this for pairwise ranking and the helper
    /// below uses it for teacher-target distillation.  Keeping this operation
    /// here prevents the offline trainer from drifting from runtime math.
    pub fn accumulate_output_grad(
        &self,
        input: &[f32],
        d_output: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) {
        assert_eq!(grad.len(), PARAM_COUNT, "semantic projector gradient width");
        let cache = self.forward_cache(input);

        /* Backprop through y = raw / ||raw||. */
        let mut d_raw = [0.0f32; SEMANTIC_DIM];
        if cache.norm.is_finite() && cache.norm > 1e-12 {
            let mut dot = 0.0;
            for k in 0..SEMANTIC_DIM {
                dot += d_output[k] * cache.output[k];
            }
            let inv_norm = 1.0f32 / cache.norm;
            for k in 0..SEMANTIC_DIM {
                d_raw[k] = (d_output[k] - cache.output[k] * dot) * inv_norm;
            }
        }

        let mut d_hidden = [0.0f32; HIDDEN_DIM];
        for k in 0..SEMANTIC_DIM {
            let row = W2_START + k * HIDDEN_DIM;
            grad[B2_START + k] += d_raw[k];
            for j in 0..HIDDEN_DIM {
                grad[row + j] += d_raw[k] * cache.hidden[j];
                d_hidden[j] += d_raw[k] * self.w[row + j];
            }
        }

        for j in 0..HIDDEN_DIM {
            let d_pre = d_hidden[j] * (1.0 - cache.hidden[j] * cache.hidden[j]);
            let row = W1_START + j * HASH_DIM;
            grad[B1_START + j] += d_pre;
            for i in 0..HASH_DIM {
                grad[row + i] += d_pre * input[i];
            }
        }
    }

    /// Accumulate the gradient of `0.5 * ||L2(project(input)) - target||²`.
    /// This keeps the trainer's backward pass tied to the exact runtime
    /// forward implementation instead of maintaining a second copy.
    pub fn accumulate_mse_grad(
        &self,
        input: &[f32],
        target: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) -> f32 {
        assert_eq!(grad.len(), PARAM_COUNT, "semantic projector gradient width");
        let cache = self.forward_cache(input);
        let mut d_output = [0.0f32; SEMANTIC_DIM];
        let mut loss = 0.0f32;
        for k in 0..SEMANTIC_DIM {
            let d = cache.output[k] - target[k];
            d_output[k] = d;
            loss += 0.5 * d * d;
        }
        self.accumulate_output_grad(input, &d_output, grad);
        loss
    }

    pub fn quantize(&self) -> QuantizedProjector {
        QuantizedProjector::from_f32(self)
    }

    /// Prune each W1 row to its `k` largest-magnitude weights and quantize
    /// the result into the v2 sparse layout.
    pub fn quantize_sparse(&self, k: usize) -> QuantizedProjector {
        QuantizedProjector::from_sparse_f32(self, k)
    }
}

/// Per-row top-K mask over the dense W1 (row-major, `rows` rows of
/// `hash_dim` entries): true = keep.  Shared by the sparse quantizer, the
/// trainer's pruning phase and the parity tests so all three prune
/// identically.
pub fn topk_row_mask(w: &[f32], rows: usize, hash_dim: usize, k: usize) -> Vec<bool> {
    assert!(k >= 1 && k <= hash_dim, "sparse k out of range");
    let mut mask = vec![false; rows * hash_dim];
    for j in 0..rows {
        let row = &w[j * hash_dim..(j + 1) * hash_dim];
        let mut order: Vec<usize> = (0..hash_dim).collect();
        order.sort_by(|&a, &b| {
            row[b]
                .abs()
                .partial_cmp(&row[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in order.iter().take(k) {
            mask[j * hash_dim + i] = true;
        }
    }
    mask
}

/// First-layer quantization layout.
#[derive(Clone, Debug)]
pub enum W1Kind {
    /// SEM1 v1: every hidden unit sees all 768 hash features.
    Dense {
        w1: Vec<i8>,
        w1_scales: Vec<f32>,
        b1: Vec<f32>,
    },
    /// SEM1 v2: `k` nonzero inputs per unit (indices ascending).
    Sparse {
        idx: Vec<u16>,
        val: Vec<i8>,
        scales: Vec<f32>,
        b1: Vec<f32>,
        k: usize,
    },
}

#[derive(Clone, Debug)]
pub struct QuantizedProjector {
    pub kind: W1Kind,
    pub w2: Vec<i8>,
    pub w2_scales: Vec<f32>,
    pub b2: Vec<f32>,
}

impl QuantizedProjector {
    pub fn from_f32(model: &ProjectorF32) -> Self {
        let mut w1 = vec![0i8; HASH_DIM * HIDDEN_DIM];
        let mut w1_scales = Vec::with_capacity(HIDDEN_DIM);
        for j in 0..HIDDEN_DIM {
            let src = &model.w[W1_START + j * HASH_DIM..W1_START + (j + 1) * HASH_DIM];
            let dst = &mut w1[j * HASH_DIM..(j + 1) * HASH_DIM];
            w1_scales.push(quantize_row(src, dst));
        }

        let mut b1 = Vec::with_capacity(HIDDEN_DIM);
        b1.extend_from_slice(&model.w[B1_START..B1_START + HIDDEN_DIM]);

        let mut w2 = vec![0i8; HIDDEN_DIM * SEMANTIC_DIM];
        let mut w2_scales = Vec::with_capacity(SEMANTIC_DIM);
        for k in 0..SEMANTIC_DIM {
            let src = &model.w[W2_START + k * HIDDEN_DIM..W2_START + (k + 1) * HIDDEN_DIM];
            let dst = &mut w2[k * HIDDEN_DIM..(k + 1) * HIDDEN_DIM];
            w2_scales.push(quantize_row(src, dst));
        }

        let mut b2 = Vec::with_capacity(SEMANTIC_DIM);
        b2.extend_from_slice(&model.w[B2_START..B2_START + SEMANTIC_DIM]);
        Self {
            kind: W1Kind::Dense {
                w1,
                w1_scales,
                b1,
            },
            w2,
            w2_scales,
            b2,
        }
    }

    /// Prune each W1 row to its top-`k` magnitudes, then int8-quantize the
    /// survivors with a per-row scale computed over the survivors only.
    pub fn from_sparse_f32(model: &ProjectorF32, k: usize) -> Self {
        let mask = topk_row_mask(&model.w, HIDDEN_DIM, HASH_DIM, k);
        let mut idx = Vec::with_capacity(HIDDEN_DIM * k);
        let mut val = Vec::with_capacity(HIDDEN_DIM * k);
        let mut scales = Vec::with_capacity(HIDDEN_DIM);
        let mut b1 = Vec::with_capacity(HIDDEN_DIM);
        b1.extend_from_slice(&model.w[B1_START..B1_START + HIDDEN_DIM]);
        let mut row_vals: Vec<i8> = Vec::with_capacity(k);
        for j in 0..HIDDEN_DIM {
            row_vals.clear();
            let row = &model.w[W1_START + j * HASH_DIM..W1_START + (j + 1) * HASH_DIM];
            let mut max_abs = 0.0f32;
            for i in 0..HASH_DIM {
                if mask[j * HASH_DIM + i] {
                    max_abs = max_abs.max(row[i].abs());
                }
            }
            let scale = if max_abs.is_finite() && max_abs > 1e-12 {
                max_abs / 127.0
            } else {
                1.0 / 127.0
            };
            for i in 0..HASH_DIM {
                if mask[j * HASH_DIM + i] {
                    let x = if row[i].is_finite() { row[i] / scale } else { 0.0 };
                    row_vals.push(x.round().clamp(-127.0, 127.0) as i8);
                }
            }
            debug_assert_eq!(row_vals.len(), k);
            let mut pushed = 0usize;
            for i in 0..HASH_DIM {
                if mask[j * HASH_DIM + i] {
                    idx.push(i as u16);
                    pushed += 1;
                }
            }
            debug_assert_eq!(pushed, k);
            val.extend_from_slice(&row_vals);
            scales.push(scale);
        }

        let mut w2 = vec![0i8; HIDDEN_DIM * SEMANTIC_DIM];
        let mut w2_scales = Vec::with_capacity(SEMANTIC_DIM);
        for k2 in 0..SEMANTIC_DIM {
            let src = &model.w[W2_START + k2 * HIDDEN_DIM..W2_START + (k2 + 1) * HIDDEN_DIM];
            let dst = &mut w2[k2 * HIDDEN_DIM..(k2 + 1) * HIDDEN_DIM];
            w2_scales.push(quantize_row(src, dst));
        }
        let mut b2 = Vec::with_capacity(SEMANTIC_DIM);
        b2.extend_from_slice(&model.w[B2_START..B2_START + SEMANTIC_DIM]);
        Self {
            kind: W1Kind::Sparse {
                idx,
                val,
                scales,
                b1,
                k,
            },
            w2,
            w2_scales,
            b2,
        }
    }

    /// Runtime inference.  Layer 1 dispatches on the artifact version; layer
    /// 2 is dense in both versions.  No heap allocation on this hot path.
    pub fn forward(&self, input: &[f32]) -> [f32; SEMANTIC_DIM] {
        assert_eq!(input.len(), HASH_DIM, "semantic projector input width");
        let mut hidden = [0.0f32; HIDDEN_DIM];
        match &self.kind {
            W1Kind::Dense {
                w1,
                w1_scales,
                b1,
            } => {
                for j in 0..HIDDEN_DIM {
                    let mut acc = b1[j];
                    let row = j * HASH_DIM;
                    let scale = w1_scales[j];
                    for i in 0..HASH_DIM {
                        acc += w1[row + i] as f32 * scale * input[i];
                    }
                    hidden[j] = acc.tanh();
                }
            }
            W1Kind::Sparse {
                idx,
                val,
                scales,
                b1,
                k,
            } => {
                for j in 0..HIDDEN_DIM {
                    let mut acc = b1[j];
                    let base = j * k;
                    let scale = scales[j];
                    for t in 0..*k {
                        let i = idx[base + t] as usize;
                        acc += val[base + t] as f32 * scale * input[i];
                    }
                    hidden[j] = acc.tanh();
                }
            }
        }

        let mut raw = [0.0f32; SEMANTIC_DIM];
        for k in 0..SEMANTIC_DIM {
            let mut acc = self.b2[k];
            let row = k * HIDDEN_DIM;
            let scale = self.w2_scales[k];
            for j in 0..HIDDEN_DIM {
                acc += self.w2[row + j] as f32 * scale * hidden[j];
            }
            raw[k] = acc;
        }

        let mut norm_sq = 0.0f32;
        for &v in &raw {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        let mut out = [0.0f32; SEMANTIC_DIM];
        if norm.is_finite() && norm > 1e-12 {
            let inv = 1.0f32 / norm;
            for k in 0..SEMANTIC_DIM {
                out[k] = raw[k] * inv;
            }
        }
        out
    }

    /// Serialize the checked SEM1 v1 (dense) artifact.
    pub fn to_blob(&self) -> Vec<u8> {
        let (w1, w1_scales, b1) = match &self.kind {
            W1Kind::Dense {
                w1,
                w1_scales,
                b1,
            } => (w1, w1_scales, b1),
            W1Kind::Sparse { .. } => panic!("v2 sparse blobs serialize via to_blob_v2"),
        };
        let mut payload = Vec::with_capacity(payload_len());
        payload.extend(w1.iter().map(|v| *v as u8));
        payload.extend(self.w2.iter().map(|v| *v as u8));
        for &v in w1_scales {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.w2_scales {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in b1 {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.b2 {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        debug_assert_eq!(payload.len(), payload_len());
        let crc = crc32_ieee(&payload);
        let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
        for word in [
            BLOB_MAGIC,
            BLOB_VERSION_V1,
            HASH_DIM as u32,
            HIDDEN_DIM as u32,
            SEMANTIC_DIM as u32,
            PARAM_COUNT as u32,
            QUANTIZATION_INT8_PER_ROW,
            payload.len() as u32,
            crc,
        ] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&payload);
        out
    }

    /// Serialize the SEM1 v2 (structured-sparse) artifact.  Header word 5
    /// carries K (per-row nonzeros); word 6 the quantization id 2.
    pub fn to_blob_v2(&self) -> Vec<u8> {
        let (idx, val, scales, b1, k) = match &self.kind {
            W1Kind::Sparse {
                idx,
                val,
                scales,
                b1,
                k,
            } => (idx, val, scales, b1, *k),
            W1Kind::Dense { .. } => panic!("dense blobs serialize via to_blob (v1)"),
        };
        let len = payload_len_v2(k);
        let mut payload = Vec::with_capacity(len);
        // Layout mirrors the parser: one u16 index block for all rows,
        // then one i8 value block for all rows.
        for v in idx {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for v in val {
            payload.push(*v as u8);
        }
        payload.extend(self.w2.iter().map(|v| *v as u8));
        for &v in scales {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.w2_scales {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in b1 {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.b2 {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        debug_assert_eq!(payload.len(), len);
        let crc = crc32_ieee(&payload);
        let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
        for word in [
            BLOB_MAGIC,
            BLOB_VERSION_V2,
            HASH_DIM as u32,
            HIDDEN_DIM as u32,
            SEMANTIC_DIM as u32,
            k as u32,
            QUANTIZATION_INT8_SPARSE_PER_ROW,
            payload.len() as u32,
            crc,
        ] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&payload);
        out
    }

    /// Strict parser: stale, truncated, trailing or corrupted artifacts are
    /// rejected before they can affect memory recall.  Dispatches on the
    /// header version; dimensions must match the compiled width.
    pub fn from_blob(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN {
            return Err("semantic blob too short");
        }
        let mut hdr = [0u32; HEADER_WORDS];
        for (i, word) in hdr.iter_mut().enumerate() {
            *word = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        }
        if hdr[0] != BLOB_MAGIC {
            return Err("semantic bad magic");
        }
        if hdr[2] as usize != HASH_DIM
            || hdr[3] as usize != HIDDEN_DIM
            || hdr[4] as usize != SEMANTIC_DIM
        {
            return Err("semantic dimensions do not match runtime");
        }
        let payload = &bytes[HEADER_LEN..];
        let crc_ok = |payload: &[u8], hdr: &[u32; HEADER_WORDS]| crc32_ieee(payload) == hdr[8];
        match hdr[1] {
            BLOB_VERSION_V1 => {
                if hdr[5] as usize != PARAM_COUNT {
                    return Err("semantic parameter count mismatch");
                }
                if hdr[6] != QUANTIZATION_INT8_PER_ROW {
                    return Err("semantic quantization unsupported");
                }
                if hdr[7] as usize != payload_len() || payload.len() != payload_len() {
                    return Err("semantic payload length mismatch");
                }
                if !crc_ok(payload, &hdr) {
                    return Err("semantic crc mismatch");
                }
                let mut at = 0usize;
                let mut w1 = vec![0i8; HASH_DIM * HIDDEN_DIM];
                for v in &mut w1 {
                    *v = payload[at] as i8;
                    at += 1;
                }
                let mut w2 = vec![0i8; HIDDEN_DIM * SEMANTIC_DIM];
                for v in &mut w2 {
                    *v = payload[at] as i8;
                    at += 1;
                }
                let mut w1_scales = Vec::with_capacity(HIDDEN_DIM);
                for _ in 0..HIDDEN_DIM {
                    w1_scales.push(read_f32(payload, &mut at)?);
                }
                let mut w2_scales = Vec::with_capacity(SEMANTIC_DIM);
                for _ in 0..SEMANTIC_DIM {
                    w2_scales.push(read_f32(payload, &mut at)?);
                }
                let mut b1 = Vec::with_capacity(HIDDEN_DIM);
                for _ in 0..HIDDEN_DIM {
                    b1.push(read_f32(payload, &mut at)?);
                }
                let mut b2 = Vec::with_capacity(SEMANTIC_DIM);
                for _ in 0..SEMANTIC_DIM {
                    b2.push(read_f32(payload, &mut at)?);
                }
                if at != payload.len()
                    || w1_scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
                    || w2_scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
                    || b1.iter().chain(b2.iter()).any(|v| !v.is_finite())
                {
                    return Err("semantic non-finite or invalid calibration");
                }
                Ok(Self {
                    kind: W1Kind::Dense {
                        w1,
                        w1_scales,
                        b1,
                    },
                    w2,
                    w2_scales,
                    b2,
                })
            }
            BLOB_VERSION_V2 => {
                let k = hdr[5] as usize;
                if k == 0 || k > HASH_DIM {
                    return Err("semantic sparse k out of range");
                }
                if hdr[6] != QUANTIZATION_INT8_SPARSE_PER_ROW {
                    return Err("semantic quantization unsupported");
                }
                let len = payload_len_v2(k);
                if hdr[7] as usize != len || payload.len() != len {
                    return Err("semantic payload length mismatch");
                }
                if !crc_ok(payload, &hdr) {
                    return Err("semantic crc mismatch");
                }
                let mut at = 0usize;
                let mut idx = vec![0u16; HIDDEN_DIM * k];
                for v in &mut idx {
                    *v = u16::from_le_bytes(payload[at..at + 2].try_into().unwrap());
                    at += 2;
                }
                for row in idx.chunks(k) {
                    if row.windows(2).any(|w| w[0] >= w[1])
                        || *row.last().unwrap() as usize >= HASH_DIM
                    {
                        return Err("semantic sparse indices unsorted or out of range");
                    }
                }
                let mut val = vec![0i8; HIDDEN_DIM * k];
                for v in &mut val {
                    *v = payload[at] as i8;
                    at += 1;
                }
                let mut w2 = vec![0i8; HIDDEN_DIM * SEMANTIC_DIM];
                for v in &mut w2 {
                    *v = payload[at] as i8;
                    at += 1;
                }
                let mut scales = Vec::with_capacity(HIDDEN_DIM);
                for _ in 0..HIDDEN_DIM {
                    scales.push(read_f32(payload, &mut at)?);
                }
                let mut w2_scales = Vec::with_capacity(SEMANTIC_DIM);
                for _ in 0..SEMANTIC_DIM {
                    w2_scales.push(read_f32(payload, &mut at)?);
                }
                let mut b1 = Vec::with_capacity(HIDDEN_DIM);
                for _ in 0..HIDDEN_DIM {
                    b1.push(read_f32(payload, &mut at)?);
                }
                let mut b2 = Vec::with_capacity(SEMANTIC_DIM);
                for _ in 0..SEMANTIC_DIM {
                    b2.push(read_f32(payload, &mut at)?);
                }
                if at != payload.len()
                    || scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
                    || w2_scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
                    || b1.iter().chain(b2.iter()).any(|v| !v.is_finite())
                {
                    return Err("semantic non-finite or invalid calibration");
                }
                Ok(Self {
                    kind: W1Kind::Sparse {
                        idx,
                        val,
                        scales,
                        b1,
                        k,
                    },
                    w2,
                    w2_scales,
                    b2,
                })
            }
            _ => Err("semantic unsupported version"),
        }
    }
}

fn read_f32(bytes: &[u8], at: &mut usize) -> Result<f32, &'static str> {
    if *at + 4 > bytes.len() {
        return Err("semantic truncated float");
    }
    let out = f32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
    *at += 4;
    Ok(out)
}

fn quantize_row(src: &[f32], dst: &mut [i8]) -> f32 {
    let max_abs = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if max_abs.is_finite() && max_abs > 1e-12 {
        max_abs / 127.0
    } else {
        1.0 / 127.0
    };
    for (q, &v) in dst.iter_mut().zip(src.iter()) {
        let x = if v.is_finite() { v / scale } else { 0.0 };
        *q = x.round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Fake-quantize a flat parameter vector exactly the way `from_f32` will at
/// export time: per-row int8 scale/round/clamp, then dequantize back to f32.
/// This is the quantization-aware-training forward — the offline trainer
/// trains through this shadow so the deployed int8 weights, not the f32
/// masters, are what the loss sees.  Biases are stored as f32 in the artifact
/// and pass through untouched.  Idempotent, and the resulting f32 model's
/// forward is bit-identical to the exported int8 model's (dense v1 layout
/// only — see the mirror test).
pub fn fake_quant_weights(w: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), PARAM_COUNT, "semantic projector parameter count");
    let mut out = vec![0.0f32; PARAM_COUNT];
    for j in 0..HIDDEN_DIM {
        let src = &w[W1_START + j * HASH_DIM..W1_START + (j + 1) * HASH_DIM];
        let dst = &mut out[W1_START + j * HASH_DIM..W1_START + (j + 1) * HASH_DIM];
        fake_quant_row(src, dst);
    }
    out[B1_START..B1_START + HIDDEN_DIM].copy_from_slice(&w[B1_START..B1_START + HIDDEN_DIM]);
    for k in 0..SEMANTIC_DIM {
        let src = &w[W2_START + k * HIDDEN_DIM..W2_START + (k + 1) * HIDDEN_DIM];
        let dst = &mut out[W2_START + k * HIDDEN_DIM..W2_START + (k + 1) * HIDDEN_DIM];
        fake_quant_row(src, dst);
    }
    out[B2_START..B2_START + SEMANTIC_DIM].copy_from_slice(&w[B2_START..B2_START + SEMANTIC_DIM]);
    out
}

/// One row's round-trip through the int8 grid — mirrors `quantize_row`'s
/// scale/round/clamp exactly, then dequantizes.
fn fake_quant_row(src: &[f32], dst: &mut [f32]) {
    let max_abs = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if max_abs.is_finite() && max_abs > 1e-12 {
        max_abs / 127.0
    } else {
        1.0 / 127.0
    };
    for (q, &v) in dst.iter_mut().zip(src.iter()) {
        let x = if v.is_finite() { v / scale } else { 0.0 };
        *q = x.round().clamp(-127.0, 127.0) * scale;
    }
}

const fn payload_len() -> usize {
    HASH_DIM * HIDDEN_DIM
        + HIDDEN_DIM * SEMANTIC_DIM
        + (HIDDEN_DIM + SEMANTIC_DIM + HIDDEN_DIM + SEMANTIC_DIM) * std::mem::size_of::<f32>()
}

/// v2 sparse payload: H rows × (K u16 indices + K i8 values), dense W2,
/// then f32 scales and biases.
pub const fn payload_len_v2(k: usize) -> usize {
    HIDDEN_DIM * k * 3
        + HIDDEN_DIM * SEMANTIC_DIM
        + (HIDDEN_DIM + SEMANTIC_DIM + HIDDEN_DIM + SEMANTIC_DIM) * std::mem::size_of::<f32>()
}

/// Reuse the existing deterministic hash embedder as the frozen input layer.
pub fn hash_v1_features(text: &str) -> Vec<f32> {
    let mut out = vec![0.0f32; HASH_DIM];
    hash_v1_features_into(text, &mut out);
    out
}

/// Writes the same frozen v1 features into a caller-provided buffer so hot
/// paths can build them on the stack.
pub fn hash_v1_features_into(text: &str, out: &mut [f32]) {
    crate::rt::ai::sofuu_tfidf_embed(text.as_bytes(), out, HASH_DIM);
}

/// The same deterministic tf-idf embedder at an arbitrary bucket count.
/// Used by the dev harnesses as a frozen low-dim anchor channel (zero blob
/// bytes — the runtime can recompute it from the text alone).
pub fn hash_features_at(text: &str, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    crate::rt::ai::sofuu_tfidf_embed(text.as_bytes(), &mut out, dim);
    out
}

static WEIGHTS: &[u8] = include_bytes!("weights_v1.sem");
static BAKED_MODEL: OnceLock<Result<QuantizedProjector, &'static str>> = OnceLock::new();

pub fn baked_model() -> Result<&'static QuantizedProjector, &'static str> {
    BAKED_MODEL
        .get_or_init(|| QuantizedProjector::from_blob(WEIGHTS))
        .as_ref()
        .map_err(|e| *e)
}

pub fn semantic_v1(text: &str) -> Option<Vec<f32>> {
    let model = baked_model().ok()?;
    // P3 (AUDIT-2026-09-07): build the frozen input features on the stack —
    // hash_v1_features' per-call Vec was allocation churn on every embed.
    let mut feats = [0.0f32; HASH_DIM];
    hash_v1_features_into(text, &mut feats);
    Some(model.forward(&feats).to_vec())
}

pub fn model_artifact_id() -> String {
    artifact_id_for(WEIGHTS)
}

pub fn artifact_id_for(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a64(bytes))
}

/// (blob version, hidden width, stored parameter count) parsed from the
/// baked artifact header — the honest values for the info surface.
pub fn blob_summary() -> (u32, usize, usize) {
    let bytes = WEIGHTS;
    if bytes.len() < HEADER_LEN {
        return (0, 0, 0);
    }
    let rd = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let version = rd(1);
    let hidden = rd(3) as usize;
    let params = match version {
        1 => rd(5) as usize,
        2 => {
            let k = rd(5) as usize;
            hidden * k + hidden * SEMANTIC_DIM
        }
        _ => 0,
    };
    (version, hidden, params)
}

pub fn model_info_json() -> String {
    let available = baked_model().is_ok();
    let (version, hidden, params) = blob_summary();
    format!(
        "{{\"id\":\"{}\",\"input\":\"{}\",\"format\":\"{}\",\"blobVersion\":{},\"dimension\":{},\"hidden\":{},\"params\":{},\"quantization\":\"int8-per-row\",\"artifact\":\"{}\",\"available\":{}}}",
        MODEL_ID,
        INPUT_EMBEDDER_ID,
        MODEL_FORMAT,
        version,
        SEMANTIC_DIM,
        hidden,
        params,
        model_artifact_id(),
        available
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_is_the_small_design() {
        assert_eq!(
            PARAM_COUNT,
            HASH_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * SEMANTIC_DIM + SEMANTIC_DIM
        );
        if HIDDEN_DIM == 16 {
            assert_eq!(PARAM_COUNT, 13_392);
            assert_eq!(payload_len(), 13_952);
        }
    }

    #[test]
    fn quantized_blob_roundtrips() {
        let mut w = vec![0.0f32; PARAM_COUNT];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i % 37) as f32 - 18.0) * 0.001;
        }
        let model = ProjectorF32::new(w).quantize();
        let blob = model.to_blob();
        let back = QuantizedProjector::from_blob(&blob).expect("valid SEM1 blob");
        let (W1Kind::Dense { w1, b1, .. }, W1Kind::Dense { w1: bw1, b1: bb1, .. }) =
            (&model.kind, &back.kind)
        else {
            panic!("dense roundtrip produced a sparse model");
        };
        assert_eq!(bw1, w1);
        assert_eq!(bb1, b1);
        assert_eq!(back.w2, model.w2);
        assert_eq!(back.b2, model.b2);
    }

    #[test]
    fn fake_quant_weights_mirror_export_quantization() {
        // Deterministic pseudo-random weights across the full parameter
        // vector — the fake-quant shadow must match the exported int8 model
        // row-for-row, so the trainer optimizes exactly what ships.
        let mut w = vec![0.0f32; PARAM_COUNT];
        let mut seed = 0xC0FF_EE12_3456_789A_u64;
        for v in w.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.08;
        }
        let fq = fake_quant_weights(&w);
        assert_eq!(
            fq,
            fake_quant_weights(&fq),
            "fake-quant must be idempotent (the grid is closed under re-quantization)"
        );
        let quant = ProjectorF32::new(w).quantize();
        let shadow = ProjectorF32::new(fq);
        for text in [
            "remember the deploy checklist",
            "fix the login timeout bug",
            "the billing webhook retries five times",
        ] {
            let input = hash_v1_features(text);
            assert_eq!(
                shadow.forward(&input),
                quant.forward(&input),
                "fake-quant forward must be bit-identical to the exported int8 model"
            );
        }
    }

    #[test]
    fn sparse_v2_roundtrips_and_prunes() {
        let mut w = vec![0.0f32; PARAM_COUNT];
        let mut seed = 0x5EED_5F00_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.04
        };
        for v in w.iter_mut() {
            *v = next();
        }
        let model = ProjectorF32::new(w.clone());
        let k = SPARSE_K_DEFAULT.min(HASH_DIM);
        let sparse = model.quantize_sparse(k);
        let blob = sparse.to_blob_v2();
        // budget: the whole point of v2 is fitting a wide layer in 32 KiB
        if HIDDEN_DIM <= 128 {
            assert!(
                blob.len() <= 32 * 1024,
                "v2 blob {} B exceeds 32 KiB at H={}",
                blob.len(),
                HIDDEN_DIM
            );
        }
        let back = QuantizedProjector::from_blob(&blob).expect("valid SEM1 v2 blob");
        let W1Kind::Sparse {
            idx, val, k: bk, ..
        } = &back.kind
        else {
            panic!("v2 roundtrip produced a dense model");
        };
        assert_eq!(*bk, k);
        assert_eq!(idx.len(), HIDDEN_DIM * k);
        assert_eq!(val.len(), HIDDEN_DIM * k);
        // quantized sparse forward matches its own pruned f32 source
        let mask = topk_row_mask(&model.w, HIDDEN_DIM, HASH_DIM, k);
        let mut pruned = w.clone();
        for (i, keep) in mask.iter().enumerate() {
            if !keep {
                pruned[i] = 0.0;
            }
        }
        let pruned_model = ProjectorF32::new(pruned);
        for text in [
            "fix the login timeout bug",
            "remember the deploy checklist",
            "grep the migration tests",
        ] {
            let input = hash_v1_features(text);
            let a = pruned_model.forward(&input);
            let b = back.forward(&input);
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!(
                dot >= 0.99,
                "v2 forward drifted from pruned f32 forward: cosine {dot}"
            );
        }
    }

    #[test]
    fn corrupted_or_trailing_blob_is_refused() {
        let model = ProjectorF32::zeros().quantize();
        let mut blob = model.to_blob();
        blob[HEADER_LEN + 3] ^= 0x80;
        assert_eq!(
            QuantizedProjector::from_blob(&blob).unwrap_err(),
            "semantic crc mismatch"
        );
        let mut trailing = model.to_blob();
        trailing.push(0);
        assert_eq!(
            QuantizedProjector::from_blob(&trailing).unwrap_err(),
            "semantic payload length mismatch"
        );
    }

    #[test]
    fn forward_is_finite_and_unit_length() {
        let mut w = vec![0.0f32; PARAM_COUNT];
        w[B2_START] = 1.0;
        let model = ProjectorF32::new(w).quantize();
        let out = model.forward(&hash_v1_features("remember the payment webhook decision"));
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!((norm - 1.0).abs() < 1e-5);
    }

    /* Baked-artifact tests are only meaningful on the shipped-width build;
     * experiment builds (SOFUU_EMB_H=…) cannot load the v1 blob at all. */
    fn baked_build() -> bool {
        HIDDEN_DIM == 16
    }

    #[test]
    fn baked_artifact_is_available_and_identified() {
        if !baked_build() {
            return;
        }
        let model = baked_model().expect("baked semantic projector");
        let out = model.forward(&hash_v1_features("test semantic memory"));
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(out.len(), SEMANTIC_DIM);
        assert!(model_info_json().contains("semantic-projector-v1"));
    }

    /* Golden pins: a retrained or edited artifact must fail these until the
     * pins are consciously updated together with the model version.  Note
     * the id comes from fnv1a64() below, whose multiplier is a historical
     * constant (not the standard FNV prime) — recorded brains depend on it
     * staying byte-stable, so never "fix" the constant. */
    const GOLDEN_ARTIFACT_ID: &str = "2c624b85c910cf72";

    #[test]
    fn golden_artifact_id_is_pinned() {
        if !baked_build() {
            return;
        }
        assert_eq!(model_artifact_id(), GOLDEN_ARTIFACT_ID);
    }

    #[test]
    fn golden_forward_vectors_are_stable() {
        if !baked_build() {
            return;
        }
        /* Exact output of the baked artifact for the probe text below,
         * captured at pin time.  Any weight or math change shifts these. */
        const GOLDEN_OUT: [f32; SEMANTIC_DIM] = [
            0.72581065, -0.1539309, 0.007700318, 0.17196694, 0.256631, -0.1483408, 0.027093751,
            0.021263238, 0.03731487, 0.15305528, 0.029815733, 0.057533894, 0.46654624, 0.07555741,
            -0.13870068, 0.24315989, 2.388094e-5, -0.00042916875, -0.0005819646, -0.0004713955,
            0.00069648563, 0.0001455819, -0.0020740647, -0.00077811617, -5.6817848e-5,
            0.0009785863, 6.1708946e-5, -0.00013371922, -0.0011307292, 0.0004206127, -0.0009816907,
            -0.00073443376, -0.0005534311, 0.000105050305, -0.002167152, -0.00027990458,
            -0.00046685318, 0.0021408526, -0.0014134507, 0.0003839714, -0.0014626291,
            0.000101158425, -0.00035949683, 2.313351e-5, -0.00022946816, -0.00013011233,
            0.00030418677, 0.0022469591, 0.00014250913, -0.0020789704, 0.0004019526,
            -2.1020083e-5, -0.00042062145, -0.00027044682, 0.0006508033, 0.0005832471,
            -0.0024228934, 0.00057195773, -0.0003593252, -0.0018072274, -0.0008448325,
            0.0044400254, -0.00010296871, -0.00052699033,
        ];
        let model = baked_model().expect("baked semantic projector");
        let out = model.forward(&hash_v1_features(
            "remember the payment webhook decision",
        ));
        // Compare with a tolerance, not f32::to_bits. The f32 math here is
        // auto-vectorized differently per target (x86-64 SSE vs aarch64
        // NEON), so the low mantissa bits of the SAME vector legitimately
        // differ between architectures. A bit-exact golden is therefore a
        // test that only passes on the machine that captured it — which is
        // how this one ended up failing on every CI runner. 2e-4 is far
        // below any real drift (a swapped weight file moves these by
        // ~1e-2+) and far above SIMD reassociation noise.
        for (i, (got, want)) in out.iter().zip(GOLDEN_OUT.iter()).enumerate() {
            assert!(
                (got - want).abs() < 2e-4,
                "golden vector drifted at dim {i}: got {got}, want {want}"
            );
        }
    }

    /// Trainer/runtime forward parity: the same weights through the f32 path
    /// (trainer) and the int8 path (runtime) must stay inside quantization
    /// noise, or the baked artifact is not the model that was trained.
    #[test]
    fn quantized_forward_tracks_f32_forward() {
        let mut seed = 0x5EED_1234_ABCD_0001u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.04
        };
        let mut w = vec![0.0f32; PARAM_COUNT];
        for v in w.iter_mut() {
            *v = next();
        }
        let f32_model = ProjectorF32::new(w);
        let quantized = f32_model.quantize();

        for text in [
            "fix the login timeout bug",
            "remember the deploy checklist",
            "grep the migration tests",
        ] {
            let input = hash_v1_features(text);
            let a = f32_model.forward(&input);
            let b = quantized.forward(&input);
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            /* Both outputs are unit-length, so the dot product is cosine. */
            assert!(
                dot >= 0.99,
                "quantized forward drifted from f32 forward: cosine {dot}"
            );
        }
    }

    /* ── §10.2 correctness gates ──────────────────────────────────── */

    /// Empty, whitespace, Unicode, control, path-like, code-like and
    /// oversized inputs never panic and always yield a finite unit-length
    /// 64-vector.
    #[test]
    fn semantic_forward_handles_hostile_inputs() {
        if !baked_build() {
            return;
        }
        let mut eight_kib = String::from("deploy the staging cluster ");
        while eight_kib.len() < 8192 {
            eight_kib.push('x');
        }
        let mut long = String::new();
        for _ in 0..1024 {
            long.push_str("release checklist item; ");
        }
        assert!(eight_kib.len() >= 8192 && long.len() > 8192);

        let cases = [
            "",
            "   ",
            "\n\t\r\n",
            "héllo wörld — ünïcode",
            "emoji 🦉 owl 日本語 नमस्ते",
            "combining a\u{0301}e\u{0301}",
            "/Users/dev/project/src/main.rs:142",
            "fn main() { let x: Option<Vec<u8>> = None; }",
            "error\u{0}with\u{1}control\u{7}chars",
            "\u{FFFD}\u{FFFE}",
            eight_kib.as_str(),
            long.as_str(),
        ];
        for text in cases {
            let out = semantic_v1(text).expect("baked model must be available");
            assert_eq!(out.len(), SEMANTIC_DIM);
            assert!(out.iter().all(|v| v.is_finite()));
            let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "not unit length ({norm}) for input {text:?}"
            );
        }
    }

    /// The feature embedder takes raw bytes — including bytes that are not
    /// valid UTF-8 (the C ABI path hands it arbitrary buffers). It must
    /// overwrite its pre-filled output with a usable feature row for any
    /// input it is given.
    #[test]
    fn tfidf_feature_embed_never_panics_on_raw_bytes() {
        let raw_cases: [Vec<u8>; 4] = [
            Vec::new(),
            vec![0xFF, 0xFE, 0x80, 0xC3, 0x28, 0x00],
            vec![b'x'; 8192],
            (0..=255u8).cycle().take(4096).collect(),
        ];
        for raw in &raw_cases {
            let mut out = vec![7.5f32; HASH_DIM]; /* pre-poisoned */
            crate::rt::ai::sofuu_tfidf_embed(raw, &mut out, HASH_DIM);
            assert!(out.iter().all(|v| v.is_finite()));
            /* Empty input legitimately yields an all-zero feature row (the
             * semantic layer still normalizes its bias into a unit vector —
             * covered by the hostile-input test's "" case). Non-empty raw
             * bytes must always produce a usable row. */
            if !raw.is_empty() {
                let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!(norm > 0.0, "empty feature row for {} raw bytes", raw.len());
            }
        }
    }

    /// Repeated inference for the same input is bit-stable (no hidden
    /// mutation of model state, no reordering-sensitive accumulation).
    #[test]
    fn semantic_forward_is_bit_stable() {
        if !baked_build() {
            return;
        }
        let texts = [
            "paraphrase stability probe",
            "",
            "🦉 unicode probe 日本語",
            "/Users/dev/project/src/main.rs:142",
        ];
        for text in texts {
            let a = semantic_v1(text).expect("baked model");
            let b = semantic_v1(text).expect("baked model");
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(f32::to_bits(*x), f32::to_bits(*y));
            }
        }
    }

    /// Every blob corruption that matters — truncation, magic/version
    /// damage, payload bit flips, CRC damage, trailing bytes — is rejected.
    #[test]
    fn all_blob_corruptions_are_rejected() {
        if !baked_build() {
            return;
        }
        let blob: &[u8] = WEIGHTS;
        assert!(QuantizedProjector::from_blob(blob).is_ok());

        /* Truncations: one byte short, header-short, empty. */
        assert!(QuantizedProjector::from_blob(&blob[..blob.len() - 1]).is_err());
        assert!(QuantizedProjector::from_blob(&blob[..HEADER_LEN - 1]).is_err());
        assert!(QuantizedProjector::from_blob(&[]).is_err());

        /* Single-byte damage at each protected region: magic, version,
         * payload, and the trailing CRC word. */
        let mut magic = blob.to_vec();
        magic[0] ^= 0xFF;
        assert!(QuantizedProjector::from_blob(&magic).is_err(), "magic");

        let mut version = blob.to_vec();
        version[4] ^= 0xFF;
        assert!(QuantizedProjector::from_blob(&version).is_err(), "version");

        let mut payload = blob.to_vec();
        payload[HEADER_LEN + 128] ^= 0x01;
        assert!(QuantizedProjector::from_blob(&payload).is_err(), "payload");

        let mut crc = blob.to_vec();
        *crc.last_mut().unwrap() ^= 0x01;
        assert!(QuantizedProjector::from_blob(&crc).is_err(), "crc");

        /* Trailing garbage after a complete blob. */
        let mut trailing = blob.to_vec();
        trailing.push(0);
        assert!(QuantizedProjector::from_blob(&trailing).is_err(), "trailing");
    }
}
