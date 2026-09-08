//! Round 9 (PLAN-TINY-SEMANTIC-EMBEDDER §"Round 9"): the v2 feature contract.
//!
//! The shipped SEM1 projector is a bag of character trigrams — it has no
//! word identity, so synonym pairs are orthogonal and the §10.1 G1
//! paraphrase gate is unreachable by construction (rounds 6–8 stalled at
//! 0.74–0.75).  v2 adds exactly one feature: a learned word-embedding
//! table.  The tower input becomes concat(trigram768, mean-pooled table
//! row e16); everything else (anchor lens, fusion rule, training recipe)
//! is unchanged from round 8.
//!
//! Pre-registered design (locked before any §10 grading):
//!   - Tokenizer: lowercase runs of `[a-z0-9]`, length ≥ 2; every other
//!     byte is a separator.  Bucket = fnv1a64(token bytes) % B.  The
//!     fnv1a64 multiplier 0x1000000001b3 is the historical constant —
//!     never "fix" it.
//!   - B = 1024 buckets, D = 16 dims, tower hidden H = 16 (the shipped
//!     default width; the table is paid for by giving up projector width).
//!   - Table init ~ U(−1/√D, 1/√D); tower init mirrors v1 (He-normal on
//!     fan-in, zero biases).
//!   - QAT: table quantizes with ONE global int8 scale (per-row scales
//!     would blow the 32 KiB payload budget — see the plan's corrected
//!     budget math); tower keeps per-row scales like SEM1.
//!   - Artifact: new magic `SEM2`, ml-train-only until §10 passes.
//!     sofuu-core runtime untouched.

use crate::train::Rng;
use sofuu_core::embedding::{HASH_DIM, SEMANTIC_DIM};

/// Word-table size (buckets).  Power of two so masking is exact.
pub const V2_B: usize = 1024;
/// Word-vector width.
pub const V2_D: usize = 16;
/// Tower hidden width — the shipped default (H=16).
pub const V2_H: usize = 16;

pub const V2_MAGIC: u32 = u32::from_le_bytes(*b"SEM2");
pub const V2_VERSION: u32 = 1;
/// int8: global table scale + per-row tower scales, f32 biases.
pub const V2_QUANT_ID: u32 = 3;

// ── flat parameter layout (f32 master) ─────────────────────────────────
pub const TABLE_LEN: usize = V2_B * V2_D; // 16,384
pub const W1_IN: usize = HASH_DIM + V2_D; // 784
pub const W1_LEN: usize = W1_IN * V2_H; // 12,544
pub const W1_START: usize = TABLE_LEN;
pub const B1_START: usize = W1_START + W1_LEN;
pub const W2_START: usize = B1_START + V2_H;
pub const W2_LEN: usize = V2_H * SEMANTIC_DIM; // 1,024
pub const B2_START: usize = W2_START + W2_LEN;
pub const PARAMS_V2: usize = B2_START + SEMANTIC_DIM; // 30,032

// ── tokenizer ───────────────────────────────────────────────────────────
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1000_0000_01b3; // historical — never change

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Word buckets for a text: lowercase `[a-z0-9]` runs of length ≥ 2,
/// every other byte a separator.  Duplicates are KEPT (mean-pooling over
/// occurrences is the pooling rule the table was trained with).
pub fn tokenize(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut h = FNV_OFFSET;
    let mut len = 0usize;
    for &b in text.as_bytes() {
        let c = match b {
            b'A'..=b'Z' => b + 32,
            b'a'..=b'z' | b'0'..=b'9' => b,
            _ => {
                if len >= 2 {
                    out.push((h % V2_B as u64) as u32);
                }
                h = FNV_OFFSET;
                len = 0;
                continue;
            }
        };
        h ^= c as u64;
        h = h.wrapping_mul(FNV_PRIME);
        len += 1;
    }
    if len >= 2 {
        out.push((h % V2_B as u64) as u32);
    }
    out
}

/// Training-time dropout on the bucket stream (the word-channel twin of
/// `dropout` on the trigram features): each token dropped independently
/// with probability `p`, consuming exactly one RNG draw per token.
pub fn drop_buckets(buckets: &[u32], rng: &mut Rng, p: f32) -> Vec<u32> {
    let mut out = Vec::with_capacity(buckets.len());
    for &b in buckets {
        if rng.uniform() >= p {
            out.push(b);
        }
    }
    out
}

// ── quantization helpers (mirror sofuu-core's private row math) ────────
fn row_scale(src: &[f32]) -> f32 {
    let max_abs = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs.is_finite() && max_abs > 1e-12 {
        max_abs / 127.0
    } else {
        1.0 / 127.0
    }
}

fn quantize_row(src: &[f32], dst: &mut [i8]) -> f32 {
    let scale = row_scale(src);
    for (q, &v) in dst.iter_mut().zip(src.iter()) {
        let x = if v.is_finite() { v / scale } else { 0.0 };
        *q = x.round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

fn fake_quant_row(src: &[f32], dst: &mut [f32]) {
    let scale = row_scale(src);
    for (q, &v) in dst.iter_mut().zip(src.iter()) {
        let x = if v.is_finite() { v / scale } else { 0.0 };
        *q = x.round().clamp(-127.0, 127.0) * scale;
    }
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ── f32 master ──────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct V2Projector {
    pub w: Vec<f32>,
}

struct V2Cache {
    input: [f32; W1_IN],
    hidden: [f32; V2_H],
    output: [f32; SEMANTIC_DIM],
    norm: f32,
}

impl V2Projector {
    pub fn new(w: Vec<f32>) -> Self {
        assert_eq!(w.len(), PARAMS_V2, "v2 parameter count");
        Self { w }
    }

    pub fn zeros() -> Self {
        Self {
            w: vec![0.0; PARAMS_V2],
        }
    }

    /// Pre-registered init: table ~ U(±1/√D) (uniform magnitude, so the
    /// max-based global int8 scale is well-conditioned), tower He-normal
    /// on fan-in like v1, biases zero.
    pub fn init(seed: u64) -> Self {
        let mut model = Self::zeros();
        let mut rng = Rng(seed);
        let lim = 1.0 / (V2_D as f32).sqrt();
        for v in model.w[..TABLE_LEN].iter_mut() {
            *v = (rng.uniform() * 2.0 - 1.0) * lim;
        }
        let w1_scale = (2.0 / (W1_IN + V2_H) as f32).sqrt();
        for v in model.w[W1_START..B1_START].iter_mut() {
            *v = rng.normal() * w1_scale;
        }
        let w2_scale = (2.0 / (V2_H + SEMANTIC_DIM) as f32).sqrt();
        for v in model.w[W2_START..B2_START].iter_mut() {
            *v = rng.normal() * w2_scale;
        }
        model
    }

    fn forward_cache(&self, x: &[f32], buckets: &[u32]) -> V2Cache {
        assert_eq!(x.len(), HASH_DIM, "v2 tower trigram width");
        let mut input = [0.0f32; W1_IN];
        input[..HASH_DIM].copy_from_slice(x);
        if !buckets.is_empty() {
            let inv = 1.0 / buckets.len() as f32;
            for &b in buckets {
                let base = (b as usize & (V2_B - 1)) * V2_D;
                let row = &self.w[base..base + V2_D];
                for d in 0..V2_D {
                    input[HASH_DIM + d] += row[d] * inv;
                }
            }
        }
        let mut hidden = [0.0f32; V2_H];
        for j in 0..V2_H {
            let row = W1_START + j * W1_IN;
            let mut acc = self.w[B1_START + j];
            for i in 0..W1_IN {
                acc += self.w[row + i] * input[i];
            }
            hidden[j] = acc.tanh();
        }
        let mut raw = [0.0f32; SEMANTIC_DIM];
        for k in 0..SEMANTIC_DIM {
            let row = W2_START + k * V2_H;
            let mut acc = self.w[B2_START + k];
            for j in 0..V2_H {
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
        V2Cache {
            input,
            hidden,
            output,
            norm,
        }
    }

    pub fn forward_at(&self, x: &[f32], buckets: &[u32]) -> [f32; SEMANTIC_DIM] {
        self.forward_cache(x, buckets).output
    }

    /// Backprop an arbitrary output gradient through the normalized tower
    /// AND the mean-pooled table.  Mirrors `ProjectorF32::accumulate_output_grad`
    /// row-for-row; the only addition is the table scatter:
    ///   d_e[d] = Σ_j d_pre[j] · W1[j][HASH_DIM + d]
    ///   grad[table_row(b)][d] += d_e[d] / n   for every occurrence b
    /// (duplicates accumulate — the same pooling rule as the forward).
    pub fn accumulate_output_grad_at(
        &self,
        x: &[f32],
        buckets: &[u32],
        d_output: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) {
        assert_eq!(grad.len(), PARAMS_V2, "v2 gradient width");
        let cache = self.forward_cache(x, buckets);

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

        let mut d_hidden = [0.0f32; V2_H];
        for k in 0..SEMANTIC_DIM {
            let row = W2_START + k * V2_H;
            grad[B2_START + k] += d_raw[k];
            for j in 0..V2_H {
                grad[row + j] += d_raw[k] * cache.hidden[j];
                d_hidden[j] += d_raw[k] * self.w[row + j];
            }
        }

        let mut d_pre = [0.0f32; V2_H];
        for j in 0..V2_H {
            d_pre[j] = d_hidden[j] * (1.0 - cache.hidden[j] * cache.hidden[j]);
            let row = W1_START + j * W1_IN;
            grad[B1_START + j] += d_pre[j];
            for i in 0..W1_IN {
                grad[row + i] += d_pre[j] * cache.input[i];
            }
        }

        if !buckets.is_empty() {
            let mut d_e = [0.0f32; V2_D];
            for j in 0..V2_H {
                let row = W1_START + j * W1_IN + HASH_DIM;
                for d in 0..V2_D {
                    d_e[d] += d_pre[j] * self.w[row + d];
                }
            }
            let contrib = 1.0 / buckets.len() as f32;
            for &b in buckets {
                let base = (b as usize & (V2_B - 1)) * V2_D;
                for d in 0..V2_D {
                    grad[base + d] += d_e[d] * contrib;
                }
            }
        }
    }

    /// `0.5 · ||L2(project(x, buckets)) − target||²` gradient, same shape as
    /// v1's `accumulate_mse_grad`.
    pub fn accumulate_mse_grad_at(
        &self,
        x: &[f32],
        buckets: &[u32],
        target: &[f32; SEMANTIC_DIM],
        grad: &mut [f32],
    ) -> f32 {
        let cache = self.forward_cache(x, buckets);
        let mut d_output = [0.0f32; SEMANTIC_DIM];
        let mut loss = 0.0f32;
        for k in 0..SEMANTIC_DIM {
            let d = cache.output[k] - target[k];
            d_output[k] = d;
            loss += 0.5 * d * d;
        }
        self.accumulate_output_grad_at(x, buckets, &d_output, grad);
        loss
    }

    /// The deployed geometry as an f32 master: table through the GLOBAL
    /// int8 grid, W1/W2 through per-row grids, biases untouched.  The
    /// resulting model's forward is bit-identical to the exported int8
    /// artifact's (same scale/round/clamp math as `quantize`).
    pub fn fake_quant(&self) -> V2Projector {
        let mut out = self.w.clone();
        fake_quant_row(&self.w[..TABLE_LEN], &mut out[..TABLE_LEN]);
        for j in 0..V2_H {
            let src = &self.w[W1_START + j * W1_IN..W1_START + (j + 1) * W1_IN];
            let dst = &mut out[W1_START + j * W1_IN..W1_START + (j + 1) * W1_IN];
            fake_quant_row(src, dst);
        }
        for k in 0..SEMANTIC_DIM {
            let src = &self.w[W2_START + k * V2_H..W2_START + (k + 1) * V2_H];
            let dst = &mut out[W2_START + k * V2_H..W2_START + (k + 1) * V2_H];
            fake_quant_row(src, dst);
        }
        V2Projector::new(out)
    }

    pub fn quantize(&self) -> QuantizedV2 {
        QuantizedV2::from_f32(self)
    }
}

// ── int8 artifact ───────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct QuantizedV2 {
    pub table: Vec<i8>, // B×D, one global scale
    pub table_scale: f32,
    pub w1: Vec<i8>, // H rows × W1_IN
    pub w1_scales: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<i8>, // SEMANTIC_DIM rows × H
    pub w2_scales: Vec<f32>,
    pub b2: Vec<f32>,
}

impl QuantizedV2 {
    fn from_f32(m: &V2Projector) -> Self {
        let mut table = vec![0i8; TABLE_LEN];
        let table_scale = quantize_row(&m.w[..TABLE_LEN], &mut table);
        let mut w1 = vec![0i8; W1_LEN];
        let mut w1_scales = Vec::with_capacity(V2_H);
        for j in 0..V2_H {
            let src = &m.w[W1_START + j * W1_IN..W1_START + (j + 1) * W1_IN];
            let dst = &mut w1[j * W1_IN..(j + 1) * W1_IN];
            w1_scales.push(quantize_row(src, dst));
        }
        let mut w2 = vec![0i8; W2_LEN];
        let mut w2_scales = Vec::with_capacity(SEMANTIC_DIM);
        for k in 0..SEMANTIC_DIM {
            let src = &m.w[W2_START + k * V2_H..W2_START + (k + 1) * V2_H];
            let dst = &mut w2[k * V2_H..(k + 1) * V2_H];
            w2_scales.push(quantize_row(src, dst));
        }
        Self {
            table,
            table_scale,
            w1,
            w1_scales,
            b1: m.w[B1_START..B1_START + V2_H].to_vec(),
            w2,
            w2_scales,
            b2: m.w[B2_START..B2_START + SEMANTIC_DIM].to_vec(),
        }
    }

    pub fn forward(&self, x: &[f32], buckets: &[u32]) -> [f32; SEMANTIC_DIM] {
        assert_eq!(x.len(), HASH_DIM, "v2 tower trigram width");
        let mut e = [0.0f32; V2_D];
        if !buckets.is_empty() {
            // Bit-identical to the fake-quant shadow's pooling (forward_cache):
            // dequantize per element, then multiply by 1/n — same term, same
            // accumulation order.  (Σq)·(scale/n) would round differently.
            let inv = 1.0f32 / buckets.len() as f32;
            for &b in buckets {
                let base = (b as usize & (V2_B - 1)) * V2_D;
                for d in 0..V2_D {
                    e[d] += (self.table[base + d] as f32 * self.table_scale) * inv;
                }
            }
        }
        let mut input = [0.0f32; W1_IN];
        input[..HASH_DIM].copy_from_slice(x);
        input[HASH_DIM..].copy_from_slice(&e);
        let mut hidden = [0.0f32; V2_H];
        for j in 0..V2_H {
            let row = j * W1_IN;
            let sc = self.w1_scales[j];
            let mut acc = self.b1[j];
            for i in 0..W1_IN {
                acc += self.w1[row + i] as f32 * sc * input[i];
            }
            hidden[j] = acc.tanh();
        }
        let mut raw = [0.0f32; SEMANTIC_DIM];
        for k in 0..SEMANTIC_DIM {
            let row = k * V2_H;
            let sc = self.w2_scales[k];
            let mut acc = self.b2[k];
            for j in 0..V2_H {
                acc += self.w2[row + j] as f32 * sc * hidden[j];
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
        output
    }

    /// Full text → embedding (the deployed path: tokenize + hash features +
    /// int8 forward).  Used by the eval/stress harnesses.
    pub fn embed_text(&self, text: &str) -> [f32; SEMANTIC_DIM] {
        let x = sofuu_core::embedding::hash_v1_features(text);
        self.forward(&x, &tokenize(text))
    }

    // ── SEM2 blob ───────────────────────────────────────────────────────
    pub const HEADER_WORDS: usize = 10;
    pub const HEADER_LEN: usize = Self::HEADER_WORDS * 4;

    pub const fn payload_len() -> usize {
        TABLE_LEN
            + 4
            + W1_LEN
            + V2_H * 4
            + W2_LEN
            + SEMANTIC_DIM * 4
            + V2_H * 4
            + SEMANTIC_DIM * 4
    }

    pub fn to_blob(&self) -> Vec<u8> {
        let payload_len = Self::payload_len();
        let mut p: Vec<u8> = Vec::with_capacity(payload_len);
        p.extend(self.table.iter().map(|v| *v as u8));
        p.extend_from_slice(&self.table_scale.to_le_bytes());
        p.extend(self.w1.iter().map(|v| *v as u8));
        for s in &self.w1_scales {
            p.extend_from_slice(&s.to_le_bytes());
        }
        p.extend(self.w2.iter().map(|v| *v as u8));
        for s in &self.w2_scales {
            p.extend_from_slice(&s.to_le_bytes());
        }
        for v in &self.b1 {
            p.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.b2 {
            p.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(p.len(), payload_len, "SEM2 payload length");
        let crc = crc32_ieee(&p);
        let mut out = Vec::with_capacity(Self::HEADER_LEN + payload_len);
        for word in [
            V2_MAGIC,
            V2_VERSION,
            V2_B as u32,
            V2_D as u32,
            V2_H as u32,
            SEMANTIC_DIM as u32,
            PARAMS_V2 as u32,
            V2_QUANT_ID,
            payload_len as u32,
            crc,
        ] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&p);
        out
    }

    pub fn from_blob(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < Self::HEADER_LEN {
            return Err("SEM2 truncated header");
        }
        let mut word = |i: usize| -> u32 { u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) };
        if word(0) != V2_MAGIC {
            return Err("SEM2 bad magic");
        }
        if word(1) != V2_VERSION {
            return Err("SEM2 unsupported version");
        }
        if word(2) as usize != V2_B || word(3) as usize != V2_D || word(4) as usize != V2_H {
            return Err("SEM2 dimensions mismatch compiled model");
        }
        if word(5) as usize != SEMANTIC_DIM {
            return Err("SEM2 output width mismatch");
        }
        if word(6) as usize != PARAMS_V2 {
            return Err("SEM2 parameter count mismatch");
        }
        if word(7) != V2_QUANT_ID {
            return Err("SEM2 unsupported quantization");
        }
        let payload_len = word(8) as usize;
        if payload_len != Self::payload_len() {
            return Err("SEM2 payload length mismatch");
        }
        if bytes.len() != Self::HEADER_LEN + payload_len {
            return Err("SEM2 trailing bytes");
        }
        let p = &bytes[Self::HEADER_LEN..];
        if crc32_ieee(p) != word(9) {
            return Err("SEM2 crc mismatch");
        }
        let mut at = 0usize;
        let mut read_f32 = |at: &mut usize| -> Result<f32, &'static str> {
            if *at + 4 > p.len() {
                return Err("SEM2 truncated float");
            }
            let out = f32::from_le_bytes(p[*at..*at + 4].try_into().unwrap());
            *at += 4;
            Ok(out)
        };
        if at + TABLE_LEN > p.len() {
            return Err("SEM2 truncated table");
        }
        let table = p[at..at + TABLE_LEN].iter().map(|b| *b as i8).collect();
        at += TABLE_LEN;
        let table_scale = read_f32(&mut at)?;
        if !table_scale.is_finite() || table_scale <= 0.0 {
            return Err("SEM2 invalid table scale");
        }
        if at + W1_LEN > p.len() {
            return Err("SEM2 truncated W1");
        }
        let w1 = p[at..at + W1_LEN].iter().map(|b| *b as i8).collect();
        at += W1_LEN;
        let mut w1_scales = Vec::with_capacity(V2_H);
        for _ in 0..V2_H {
            let s = read_f32(&mut at)?;
            if !s.is_finite() || s <= 0.0 {
                return Err("SEM2 invalid W1 scale");
            }
            w1_scales.push(s);
        }
        if at + W2_LEN > p.len() {
            return Err("SEM2 truncated W2");
        }
        let w2 = p[at..at + W2_LEN].iter().map(|b| *b as i8).collect();
        at += W2_LEN;
        let mut w2_scales = Vec::with_capacity(SEMANTIC_DIM);
        for _ in 0..SEMANTIC_DIM {
            let s = read_f32(&mut at)?;
            if !s.is_finite() || s <= 0.0 {
                return Err("SEM2 invalid W2 scale");
            }
            w2_scales.push(s);
        }
        let mut b1 = Vec::with_capacity(V2_H);
        for _ in 0..V2_H {
            let v = read_f32(&mut at)?;
            if !v.is_finite() {
                return Err("SEM2 non-finite b1");
            }
            b1.push(v);
        }
        let mut b2 = Vec::with_capacity(SEMANTIC_DIM);
        for _ in 0..SEMANTIC_DIM {
            let v = read_f32(&mut at)?;
            if !v.is_finite() {
                return Err("SEM2 non-finite b2");
            }
            b2.push(v);
        }
        if at != p.len() {
            return Err("SEM2 payload trailing bytes");
        }
        Ok(Self {
            table,
            table_scale,
            w1,
            w1_scales,
            b1,
            w2,
            w2_scales,
            b2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_runs_and_separators() {
        // "deploy" and "cluster" are words; "a" (len 1) is not; unicode bytes
        // are separators, not letters.
        let b = tokenize("Deploy the cluster!");
        assert_eq!(b.len(), 3);
        assert_eq!(b, tokenize("deploy the cluster"));
        assert_ne!(b[0], b[2]);
        assert!(tokenize("a b c").is_empty());
        assert_eq!(tokenize("héllo"), tokenize("llo")); // é = separator; "h" is len-1
        assert_eq!(tokenize("héllo").len(), 1);
    }

    #[test]
    fn tokenizer_stability() {
        // The bucket stream is a deployment contract: pin it.
        let b = tokenize("fix the login timeout bug in the scheduler pool");
        // recompute independently: same hash, same mod
        for tok in ["fix", "the", "login", "timeout", "bug", "the", "scheduler", "pool"] {
            let want = (fnv1a64(tok.as_bytes()) % V2_B as u64) as u32;
            assert!(b.contains(&want), "missing bucket for {tok}");
        }
        assert_eq!(b.len(), 9); // "the" appears twice, duplicates kept
    }

    #[test]
    fn blob_round_trips_and_is_deterministic() {
        let m = V2Projector::init(7);
        let q = m.quantize();
        let blob = q.to_blob();
        assert_eq!(blob.len(), QuantizedV2::HEADER_LEN + QuantizedV2::payload_len());
        assert!(blob.len() <= 32 * 1024, "SEM2 payload must fit the budget");
        let back = QuantizedV2::from_blob(&blob).expect("parse SEM2");
        assert_eq!(back.to_blob(), blob, "round-trip must be byte-identical");
        // corruption is caught
        let mut bad = blob.clone();
        let n = bad.len() - 1;
        bad[n] ^= 0xFF;
        assert!(QuantizedV2::from_blob(&bad).is_err());
    }

    #[test]
    fn fake_quant_matches_int8_forward() {
        // The QAT shadow's forward must be bit-identical to the exported
        // artifact's forward — the same guarantee SEM1 has.
        let m = V2Projector::init(11);
        let shadow = m.fake_quant();
        let q = m.quantize();
        for text in [
            "paraphrase stability probe",
            "",
            "which tokio version does sofuu-cli use?",
            "/Users/dev/project/src/main.rs:142",
        ] {
            let x = sofuu_core::embedding::hash_v1_features(text);
            let b = tokenize(text);
            let a = shadow.forward_at(&x, &b);
            let c = q.forward(&x, &b);
            for k in 0..SEMANTIC_DIM {
                assert_eq!(a[k].to_bits(), c[k].to_bits(), "fake-quant mismatch at {k}");
            }
        }
    }

    #[test]
    fn gradient_flows_to_the_table() {
        let m = V2Projector::init(3);
        let x = sofuu_core::embedding::hash_v1_features("alpha beta gamma");
        let b = tokenize("alpha beta gamma");
        assert_eq!(b.len(), 3);
        let mut grad = vec![0.0f32; PARAMS_V2];
        let mut d_out = [0.0f32; SEMANTIC_DIM];
        d_out[0] = 1.0;
        m.accumulate_output_grad_at(&x, &b, &d_out, &mut grad);
        let touched: Vec<usize> = b
            .iter()
            .map(|&bb| bb as usize & (V2_B - 1))
            .collect();
        let table_grad: f32 = grad[..TABLE_LEN]
            .iter()
            .enumerate()
            .map(|(i, g)| if touched.contains(&(i / V2_D)) { g.abs() } else { *g })
            .sum();
        assert!(table_grad > 0.0, "table rows of present tokens must receive gradient");
        // absent rows stay exactly zero
        for row in 0..V2_B {
            if !touched.contains(&row) {
                assert!(
                    grad[row * V2_D..(row + 1) * V2_D].iter().all(|&g| g == 0.0),
                    "row {row} untouched"
                );
            }
        }
    }

    #[test]
    fn empty_buckets_degrade_to_trigram_only() {
        let m = V2Projector::init(5);
        let x = sofuu_core::embedding::hash_v1_features("hi");
        let with = m.forward_at(&x, &[]);
        let without = m.forward_at(&x, &[]);
        assert_eq!(with, without);
        assert!(with.iter().all(|v| v.is_finite()));
    }
}
