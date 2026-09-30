//! Runtime home of the round-9 SEM2 embedder (PLAN-TINY-SEMANTIC-EMBEDDER
//! §"Round 9").  Owner decision 2026-09-08: ship the trained embedder into
//! the brain as a second retrieval channel, with hash-v1 as the fallback
//! store.  This module is the deployed path the eval measured:
//!
//! ```text
//! sem64(text) = anchor_lens( SEM2_tower( hash768(text), tokens(text) ) )
//! ```
//!
//! with the lens at K=32, w=0.6 — the constants the fused OVERALL 0.917 was
//! graded under (§10, single pass).  The record on the table is honest: the
//! fused pipeline passes G2/G4/G5 and the budgets, but G1 paraphrase
//! (0.750 < 0.85), G3-paths (0.667) and G6 (3/8) FAILED.  Shipping is the
//! owner's call, not a §10 pass; retrieval is FUSED (this channel + the
//! hash-v1 channel, RRF), never this channel alone (sem-alone scored
//! 0.807 < hash's 0.870).
//!
//! Bit-identity contract: this forward is THE model the int8 artifact was
//! trained to export (QAT shadow == exported int8 == here).  It mirrors
//! `ml-train/src/embedding_v2.rs::QuantizedV2` and
//! `ml-train/src/embedding_anchor.rs` term-for-term — the pooling is
//! per-element `(table[q] * table_scale) * inv` (a Σq·(scale/n) factor-out
//! would round differently), the fnv1a64 multiplier 0x1000000001b3 is the
//! historical constant (never "fix" it), and ANCHOR_SEED is frozen.  A
//! cross-crate test in ml-train pins runtime == trainer on the graded
//! artifact.
//!
//! Space hygiene (§12): this is a 64-dimensional space distinct from BOTH
//! the 768-dim hash-v1 store and the 64-dim `semantic-projector-v1` store —
//! the brain manifest pins id + dimension + artifact hash on every open, so
//! no file can ever hydrate vectors from a different space.

use std::sync::OnceLock;

use crate::embedding::{hash_v1_features, HASH_DIM, SEMANTIC_DIM};
use crate::ml::net::crc32_ieee;

/* ── identity ─────────────────────────────────────────────────────────── */

/// Manifest id of the SEM2 vector space.  Distinct from MODEL_ID
/// ("semantic-projector-v1", the SEM1 tower): two 64-dim spaces are NOT
/// interchangeable.
pub const MODEL_ID_V2: &str = "semantic-table-v2";
/// v2 keeps hash-v1 as its frozen input layer — same lineage as SEM1.
pub const INPUT_EMBEDDER_ID_V2: &str = "hash-v1";
pub const MODEL_FORMAT_V2: &str = "SEM2";

/* ── v2 contract constants (mirror embedding_v2.rs) ───────────────────── */

/// Word-table buckets (power of two so masking is exact).
pub const V2_B: usize = 1024;
/// Word-vector width.
pub const V2_D: usize = 16;
/// Tower hidden width — the shipped default (H=16).
pub const V2_H: usize = 16;

pub const V2_MAGIC: u32 = u32::from_le_bytes(*b"SEM2");
pub const V2_VERSION: u32 = 1;
/// int8: global table scale + per-row tower scales, f32 biases.
pub const V2_QUANT_ID: u32 = 3;

pub const TABLE_LEN: usize = V2_B * V2_D; // 16,384
pub const W1_IN: usize = HASH_DIM + V2_D; // 784
pub const W1_LEN: usize = W1_IN * V2_H; // 12,544
pub const W2_LEN: usize = V2_H * SEMANTIC_DIM; // 1,024
pub const PARAMS_V2: usize = TABLE_LEN + W1_LEN + V2_H + W2_LEN + SEMANTIC_DIM; // 30,032

/* ── anchor lens (mirror embedding_anchor.rs; K/w frozen at the graded
 *    round-9 config) ─────────────────────────────────────────────────── */

/// Fixed seed for the Rademacher anchor matrix — changing it invalidates
/// every stored v2 vector.
pub const ANCHOR_SEED: u64 = 0xA11C_E500_0001;
/// Anchor width K (tower keeps the first 64−K dims; the channels occupy
/// disjoint indices, so cos(v1,v2) = w·cos_tower + (1−w)·cos_anchor exactly).
pub const ANCHOR_K: usize = 32;
/// Anchor energy share w.
pub const ANCHOR_W: f32 = 0.6;

/* ── tokenizer (verbatim from the trainer) ────────────────────────────── */

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1000_0000_01b3; // historical — never change

/// Local copy: same construction as the runtime's historical hash
/// (multiplier 0x1000000001b3 is deliberate, do NOT "fix").
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Word buckets for a text: lowercase `[a-z0-9]` runs of length ≥ 2, every
/// other byte a separator.  Duplicates are KEPT — mean-pooling over
/// occurrences is the pooling rule the table was trained with.  The bucket
/// stream is a deployment contract.
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

/* ── anchor math (verbatim from embedding_anchor.rs) ──────────────────── */

/// Row-major [768][k] Rademacher matrix (+1/−1 scaled by 1/sqrt(k)),
/// generated from ANCHOR_SEED by the frozen LCG.  Zero blob bytes.
fn anchor_matrix(k: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; HASH_DIM * k];
    let inv = 1.0 / (k as f32).sqrt();
    let mut s = ANCHOR_SEED;
    for v in r.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = if (s >> 33) & 1 == 1 { inv } else { -inv };
    }
    r
}

/// The unit anchor vector for a 768-feature input (sparse-skip dense
/// projection).  Iteration order and accumulation order are part of the
/// bit-identity contract.
fn anchor_forward(x: &[f32], r: &[f32], k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k];
    let mut i = 0;
    while i < x.len() {
        let xi = x[i];
        if xi != 0.0 {
            let row = i * k;
            let mut j = 0;
            while j < k {
                out[j] += xi * r[row + j];
                j += 1;
            }
        }
        i += 1;
    }
    unit(&mut out);
    out
}

fn unit(v: &mut [f32]) {
    let mut n = 0.0f32;
    for x in v.iter() {
        n += x * x;
    }
    let n = n.sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// The hybrid vector from an L2-normalized tower output and a precomputed
/// unit anchor.  Result length is exactly 64 (SEMANTIC_DIM contract).
fn blend(f_hat: &[f32], a_hat: &[f32], k: usize, w: f32) -> Vec<f32> {
    let f_dims = SEMANTIC_DIM - k;
    let sw = w.sqrt();
    let tw = (1.0 - w).sqrt();
    let mut v = Vec::with_capacity(SEMANTIC_DIM);
    let mut j = 0;
    while j < f_dims {
        v.push(sw * f_hat[j]);
        j += 1;
    }
    for x in a_hat.iter() {
        v.push(tw * x);
    }
    unit(&mut v);
    v
}

fn anchor_rng() -> &'static Vec<f32> {
    static MATRIX: OnceLock<Vec<f32>> = OnceLock::new();
    MATRIX.get_or_init(|| anchor_matrix(ANCHOR_K))
}

/* ── int8 model (field-for-field mirror of QuantizedV2) ───────────────── */

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
    /// The deployed forward: trigram input `x` (768) + word-bucket stream →
    /// unit 64-dim tower output.  Term order is the QAT contract: pool
    /// per-element `(q * scale) * inv`, then raw int8 matmul dequant per
    /// row scale, tanh, row-scale output, L2.
    pub fn forward(&self, x: &[f32], buckets: &[u32]) -> [f32; SEMANTIC_DIM] {
        assert_eq!(x.len(), HASH_DIM, "v2 tower trigram width");
        let mut e = [0.0f32; V2_D];
        if !buckets.is_empty() {
            // Bit-identical to the fake-quant shadow's pooling: dequantize
            // per element, then multiply by 1/n — same term, same
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

    /// Tower embedding of a text (no lens) — the graded channel-S body.
    pub fn embed_text(&self, text: &str) -> [f32; SEMANTIC_DIM] {
        self.forward(&hash_v1_features(text), &tokenize(text))
    }

    // ── SEM2 blob ──────────────────────────────────────────────────────
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

    /// Strict parse: every header word, exact trailing length, CRC, and
    /// per-value finiteness/positivity are checked.  A corrupt artifact must
    /// make the model UNAVAILABLE (the brain then runs on the hash-v1
    /// fallback), never half-load.
    pub fn from_blob(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < Self::HEADER_LEN {
            return Err("SEM2 truncated header");
        }
        let word = |i: usize| -> u32 {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        };
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
        let read_f32 = |at: &mut usize| -> Result<f32, &'static str> {
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

/* ── baked artifact + deployed API ────────────────────────────────────── */

/// The graded round-9 soup artifact (seed 47 kept): fnv1a64 id
/// dffb00185d090662, 30,636 B.  Never retrain silently — a new artifact
/// gets a new file AND fails the golden test below on purpose.
static WEIGHTS_V2: &[u8] = include_bytes!("weights_v2.sem");
static BAKED_MODEL_V2: OnceLock<Result<QuantizedV2, &'static str>> = OnceLock::new();

/// Raw bytes of the baked SEM2 artifact (for cross-checks).
pub fn weights_v2_bytes() -> &'static [u8] {
    WEIGHTS_V2
}

pub fn baked_model_v2() -> Result<&'static QuantizedV2, &'static str> {
    BAKED_MODEL_V2
        .get_or_init(|| QuantizedV2::from_blob(WEIGHTS_V2))
        .as_ref()
        .map_err(|e| *e)
}

/// The deployed sem-channel vector: tower embedding through the frozen
/// anchor lens (K=32, w=0.6).  None when the artifact fails validation —
/// the brain then runs hash-only (the fallback path).
pub fn semantic_v2(text: &str) -> Option<Vec<f32>> {
    let model = baked_model_v2().ok()?;
    let x = hash_v1_features(text);
    let f_hat = model.forward(&x, &tokenize(text));
    let a_hat = anchor_forward(&x, anchor_rng(), ANCHOR_K);
    Some(blend(&f_hat, &a_hat, ANCHOR_K, ANCHOR_W))
}

pub fn model_artifact_id_v2() -> String {
    format!("{:016x}", fnv1a64(WEIGHTS_V2))
}

/// Identity + availability JSON for the JS side (mirrors model_info_json;
/// the availability flag is what turns the fused channel on or off).
pub fn model_info_json_v2() -> String {
    let available = baked_model_v2().is_ok();
    format!(
        "{{\"id\":\"{}\",\"input\":\"{}\",\"format\":\"{}\",\"blobVersion\":{},\"dimension\":{},\"hidden\":{},\"table\":{{\"buckets\":{},\"width\":{}}},\"params\":{},\"quantization\":\"int8 (global table scale, per-row tower scales)\",\"anchor\":{{\"kind\":\"rp\",\"k\":{},\"w\":{},\"seed\":\"0x{:016x}\"}},\"artifact\":\"{}\",\"available\":{}}}",
        MODEL_ID_V2,
        INPUT_EMBEDDER_ID_V2,
        MODEL_FORMAT_V2,
        V2_VERSION,
        SEMANTIC_DIM,
        V2_H,
        V2_B,
        V2_D,
        PARAMS_V2,
        ANCHOR_K,
        ANCHOR_W,
        ANCHOR_SEED,
        model_artifact_id_v2(),
        available
    )
}

/* ── tests ────────────────────────────────────────────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_runs_and_separators() {
        // Ported from the trainer: "Deploy the cluster!" is three words
        // ("a" len-1 is not); unicode bytes are separators, not letters.
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
        for tok in [
            "fix",
            "the",
            "login",
            "timeout",
            "bug",
            "the",
            "scheduler",
            "pool",
        ] {
            let want = (fnv1a64(tok.as_bytes()) % V2_B as u64) as u32;
            assert!(b.contains(&want), "missing bucket for {tok}");
        }
        assert_eq!(b.len(), 9); // "the" appears twice, duplicates kept
    }

    #[test]
    fn baked_artifact_is_the_graded_round9_soup() {
        // dffb00185d090662 is the §10-graded artifact (fused OVERALL 0.917).
        // If this fires, the baked bytes changed — that is a re-grade event,
        // not a fix-forward.
        assert_eq!(WEIGHTS_V2.len(), 30_636);
        assert!(WEIGHTS_V2.len() <= 32 * 1024, "SEM2 must fit the budget");
        assert_eq!(model_artifact_id_v2(), "dffb00185d090662");
        assert!(baked_model_v2().is_ok(), "graded artifact must parse");
    }

    #[test]
    fn blob_round_trips_byte_identically() {
        let q = baked_model_v2().unwrap();
        assert_eq!(q.to_blob(), WEIGHTS_V2, "round-trip must be byte-identical");
    }

    #[test]
    fn corruption_matrix_refuses_every_hostile_shape() {
        // Mirror of the SEM1 discipline: the model must go UNAVAILABLE,
        // never load bad weights (an unavailable v2 channel is a clean
        // fallback to hash-only).
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", vec![]),
            ("short header", WEIGHTS_V2[..20].to_vec()),
            (
                "bad magic",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[0] ^= 0xFF;
                            b
                        },
            ),
            (
                "bad version",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[4] = 99;
                            b
                        },
            ),
            (
                "bad B",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[8] = 1; // 1025 ≠ 1024 (the low byte of 1024 is already 0)
                            b
                        },
            ),
            (
                "bad D",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[12] = 0;
                            b
                        },
            ),
            (
                "bad H",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[16] = 9;
                            b
                        },
            ),
            (
                "bad out width",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[20] = 32;
                            b
                        },
            ),
            (
                "bad params",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[24] = 1;
                            b
                        },
            ),
            (
                "bad quant",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[28] = 7;
                            b
                        },
            ),
            (
                "bad payload len",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b[32] = 0;
                            b
                        },
            ),
            (
                "trailing byte",
                {
                            let mut b = WEIGHTS_V2.to_vec();
                            b.push(0);
                            b
                        },
            ),
            (
                "crc break",
                {
                            let n = WEIGHTS_V2.len() - 1;
                            let mut b = WEIGHTS_V2.to_vec();
                            b[n] ^= 0x01;
                            b
                        },
            ),
            (
                "truncated payload",
                WEIGHTS_V2[..WEIGHTS_V2.len() - 16].to_vec(),
            ),
        ];
        for (name, bytes) in cases {
            assert!(
                QuantizedV2::from_blob(&bytes).is_err(),
                "corruption escaped: {name}"
            );
        }
    }

    #[test]
    fn forward_is_bit_stable() {
        // Regression lock: the fused pipeline's graded numbers only mean
        // something if this forward stops drifting.  The cross-crate proof
        // vs the trainer lives in ml-train (runtime_equals_trainer); the
        // fnv-of-bits lock here catches drift on this side alone.
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "fix the login timeout bug in the scheduler pool",
            "",
        ];
        let mut acc = FNV_OFFSET;
        for t in texts {
            let v = semantic_v2(t).expect("baked model must be available");
            assert_eq!(v.len(), SEMANTIC_DIM);
            assert!(v.iter().all(|x| x.is_finite()));
            let norm = v.iter().fold(0.0f32, |m, x| m + x * x).sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "unit lens vector expected");
            for x in &v {
                acc ^= (x.to_bits() as u64) & 0xFFFFFFFF;
                acc = acc.wrapping_mul(FNV_PRIME);
            }
        }
        assert_eq!(format!("{acc:016x}"), GOLDEN_SEM2_BITS);
    }

    /// Captured 2026-09-08 from the graded dffb00185d090662 artifact; the
    /// ml-train cross-check (`trainer_and_runtime_are_bit_identical`) proves
    /// this matches the trainer's own forward + anchor lens, not just the port.
    const GOLDEN_SEM2_BITS: &str = "3a00025c59cb864a";

    #[test]
    fn hostile_inputs_never_panic() {
        let emoji = "😀".repeat(4000);
        let big = "a".repeat(1_000_000);
        let hostile = ["\u{0}", emoji.as_str(), big.as_str(), "\u{0}x\u{0}y", "%%%\n\t\r"];
        for t in hostile {
            let v = semantic_v2(t).expect("model available");
            assert_eq!(v.len(), SEMANTIC_DIM);
            assert!(v.iter().all(|x| x.is_finite()));
        }
    }
}
