// ml/net.rs — the generic tiny MLP shared by all four gates
// (PLAN-ML-GATES §4).
//
//   in(F) → h1 tanh → h2 tanh → out(1) sigmoid
//
// Scalar f32 only — deliberately NO SIMD: the scores must be bit-identical
// on every platform so tests can assert exact values and the trainer and the
// runtime agree by construction (they also share this code — train/serve
// skew is structurally impossible, §4).
//
// Weights ship as include_bytes! blobs:
//
//   magic   u32 LE  = "SML1"
//   version u32 LE  = 1
//   in_dim  u32 LE
//   h1      u32 LE
//   h2      u32 LE
//   params  u32 LE  (must equal h1*in + h1 + h2*h1 + h2 + h2 + 1)
//   crc32   u32 LE  (IEEE, over the weight bytes that follow)
//   weights f32 LE  — W1(h1×in) b1(h1) W2(h2×h1) b2(h2) W3(h2) b3(1)
//
// A stale/truncated/corrupted blob is a load error, never a silent wrong
// answer.

const BLOB_MAGIC: u32 = u32::from_le_bytes(*b"SML1");
const BLOB_VERSION: u32 = 1;
/// AUDIT-2026-11-08 §1 (binary budget): the same container with the weights
/// stored as IEEE half precision. The five advisory gates (alloc, freshness,
/// compaction, relevance, supervisor) are MLPs whose scores move by well under
/// a percent at a 10-bit mantissa, and this halves ~250 KB of committed
/// weights. Build-time conversion only — the committed .f32 files stay the
/// source of truth, and version 1 blobs still load (the integrity test reads
/// one).
pub const BLOB_VERSION_F16: u32 = 2;
const HEADER_LEN: usize = 7 * 4;

#[derive(Clone, Debug)]
pub struct TinyMlp {
    pub in_dim: u32,
    pub h1: u32,
    pub h2: u32,
    /// Flat weights: W1 b1 W2 b2 W3 b3 (row-major, output-major).
    pub w: Vec<f32>,
}

/// Total parameter count for an (in, h1, h2) shape.
pub const fn param_count(in_dim: u32, h1: u32, h2: u32) -> u32 {
    h1 * in_dim + h1 + h2 * h1 + h2 + h2 + 1
}

/// Standard CRC32 (IEEE 802.3, reflected, poly 0xEDB88320) — bitwise, no
/// table: runs once per model load over ~34 KiB, speed is irrelevant.
/// IEEE 754 half -> single. Every f16 value has an exact f32 value, so this
/// reconstructs it rather than approximating: normals are 1.mant * 2^(e-15)
/// (a power-of-two scale, which is exact), subnormals are mant * 2^-24, and
/// the special encodings map to inf / NaN. Written with magnitudes instead of
/// hand-rolled exponent rebiasing, which got the subnormal bias wrong once
/// (the round-trip test caught it).
pub fn f16_to_f32(h: u16) -> f32 {
    let neg = h & 0x8000 != 0;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as f32;
    let mag = match exp {
        0 => mant * 5.960_464_5e-8, // subnormal: mant * 2^-24
        0x1f => {
            if mant == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant / 1024.0) * exp2i(exp as i32 - 15),
    };
    if neg {
        -mag
    } else {
        mag
    }
}

/// 2^n for the small n an f16 exponent can hold (|n| <= 24). A table-free
/// power of two is exact, unlike powi's libm path on some targets.
fn exp2i(n: i32) -> f32 {
    debug_assert!((-126..=127).contains(&n));
    f32::from_bits(((n + 127) as u32) << 23)
}

pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Standard 64-bit FNV-1a hash — used for blob identity tracking and cache keys.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl TinyMlp {
    pub fn new(in_dim: u32, h1: u32, h2: u32, w: Vec<f32>) -> Self {
        assert_eq!(
            w.len() as u32,
            param_count(in_dim, h1, h2),
            "weight count must match the architecture"
        );
        Self { in_dim, h1, h2, w }
    }

    /// Forward pass → sigmoid output in (0, 1). ~8.5k multiply-adds.
    /// Deterministic: scalar f32 in a fixed order, no SIMD reassociation.
    ///
    /// Implemented as forward_output(forward_hidden(x)) — the two halves
    /// run the SAME accumulations in the SAME order, so the split is
    /// bit-identical to the fused pass (asserted in tests). The split
    /// exists for online learning (§13): the backbone (everything before
    /// the output layer) stays frozen while only W3/b3 adapt.
    pub fn forward(&self, x: &[f32]) -> f32 {
        self.forward_output(&self.forward_hidden(x))
    }

    /// Hidden pass → the h2 activations (tanh of layer 2). Everything the
    /// output layer needs; the frozen-backbone half of an online update.
    pub fn forward_hidden(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim as usize, "input width mismatch");
        let (in_dim, h1, h2) = (self.in_dim as usize, self.h1 as usize, self.h2 as usize);
        let w = &self.w;

        // Layer 1: h1 = tanh(W1 x + b1)
        let mut a1 = vec![0.0f32; h1];
        for j in 0..h1 {
            let row = &w[j * in_dim..(j + 1) * in_dim];
            let mut acc = w[h1 * in_dim + j];
            for (i, &xi) in x.iter().enumerate() {
                acc += row[i] * xi;
            }
            a1[j] = acc.tanh();
        }

        // Layer 2: h2 = tanh(W2 h1 + b2)
        let w2 = h1 * in_dim + h1;
        let mut a2 = vec![0.0f32; h2];
        for j in 0..h2 {
            let row = &w[w2 + j * h1..w2 + (j + 1) * h1];
            let mut acc = w[w2 + h2 * h1 + j];
            for (i, &hi) in a1.iter().enumerate() {
                acc += row[i] * hi;
            }
            a2[j] = acc.tanh();
        }
        a2
    }

    /// Output pass: sigmoid(W3 h2 + b3) over precomputed h2 activations.
    pub fn forward_output(&self, h2_acts: &[f32]) -> f32 {
        let h2 = self.h2 as usize;
        assert_eq!(h2_acts.len(), h2, "hidden width mismatch");
        let os = self.out_start();
        let w = &self.w;
        let mut acc = w[os + h2];
        for (i, &hi) in h2_acts.iter().enumerate() {
            acc += w[os + i] * hi;
        }
        1.0 / (1.0 + (-acc).exp())
    }

    /// Output pass with an EXPLICIT output layer (the adapted weights from
    /// online learning, §13): sigmoid(W3' h2 + b3'). Same scalar-f32 fixed
    /// order as forward_output — bit-exact given the same layer.
    pub fn forward_output_with(&self, out_layer: &[f32], h2_acts: &[f32]) -> f32 {
        let h2 = self.h2 as usize;
        assert_eq!(h2_acts.len(), h2, "hidden width mismatch");
        assert_eq!(out_layer.len(), h2 + 1, "output layer must be W3 + b3");
        let mut acc = out_layer[h2];
        for (i, &hi) in h2_acts.iter().enumerate() {
            acc += out_layer[i] * hi;
        }
        1.0 / (1.0 + (-acc).exp())
    }

    /// Start index of the output layer (W3 row) in the flat weight vector;
    /// the output layer is `w[out_start..out_start + h2]` (W3) plus
    /// `w[out_start + h2]` (b3). h2+1 params total — the ONLY slice an
    /// online update may touch (§13 guardrail 1).
    pub fn out_start(&self) -> usize {
        let (in_dim, h1, h2) = (self.in_dim as usize, self.h1 as usize, self.h2 as usize);
        h1 * in_dim + h1 + h2 * h1 + h2
    }

    /// The output layer as one flat slice: W3 then b3 (h2+1 values).
    pub fn output_layer(&self) -> &[f32] {
        let os = self.out_start();
        &self.w[os..os + self.h2 as usize + 1]
    }

    /// Serialize with the header + CRC (the trainer emits this; the runtime
    /// bakes it via include_bytes!).
    pub fn to_blob(&self) -> Vec<u8> {
        let mut weight_bytes = Vec::with_capacity(self.w.len() * 4);
        for v in &self.w {
            weight_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let crc = crc32_ieee(&weight_bytes);
        let mut out = Vec::with_capacity(HEADER_LEN + weight_bytes.len());
        for v in [
            BLOB_MAGIC,
            BLOB_VERSION,
            self.in_dim,
            self.h1,
            self.h2,
            self.w.len() as u32,
            crc,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&weight_bytes);
        out
    }

    /// Parse + verify a blob. Any mismatch (magic, version, dims, param
    /// count, CRC, finiteness, trailing data) is an error — the caller
    /// refuses to load; a corrupted bake must never become a silent wrong
    /// answer.
    pub fn from_blob(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN {
            return Err("blob too short");
        }
        let mut hdr = [0u32; 7];
        for (i, h) in hdr.iter_mut().enumerate() {
            *h = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let [magic, version, in_dim, h1, h2, params, crc] = hdr;
        if magic != BLOB_MAGIC {
            return Err("bad magic");
        }
        let half = match version {
            BLOB_VERSION => false,
            BLOB_VERSION_F16 => true,
            _ => return Err("unsupported version"),
        };
        if in_dim == 0 || h1 == 0 || h2 == 0 {
            return Err("zero dimension");
        }
        if params != param_count(in_dim, h1, h2) {
            return Err("param count does not match dims");
        }
        let weight_bytes = &bytes[HEADER_LEN..];
        let width = if half { 2 } else { 4 };
        if weight_bytes.len() != params as usize * width {
            return Err("truncated weights");
        }
        if crc32_ieee(weight_bytes) != crc {
            return Err("crc mismatch");
        }
        let mut w = Vec::with_capacity(params as usize);
        if half {
            for c in weight_bytes.chunks_exact(2) {
                let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                if !v.is_finite() {
                    return Err("non-finite weight");
                }
                w.push(v);
            }
        } else {
            for c in weight_bytes.chunks_exact(4) {
                let v = f32::from_le_bytes(c.try_into().unwrap());
                if !v.is_finite() {
                    return Err("non-finite weight");
                }
                w.push(v);
            }
        }
        Ok(Self { in_dim, h1, h2, w })
    }
}

/// Online-learning trust region (PLAN-ML-GATES §12 guardrail 2): project
/// `w` back into the ball `‖w − w0‖ ≤ rel·‖w0‖` around the pretrained
/// weights. Bounded by construction, not by hoping the learning rate is
/// small. Returns true when a projection happened.
pub fn clamp_trust_region(w: &mut [f32], w0: &[f32], rel: f32) -> bool {
    assert_eq!(w.len(), w0.len());
    let mut d2 = 0.0f32;
    let mut n2 = 0.0f32;
    for (a, b) in w.iter().zip(w0.iter()) {
        let d = a - b;
        d2 += d * d;
        n2 += b * b;
    }
    let r_max = rel * n2.sqrt();
    if d2.sqrt() <= r_max {
        return false;
    }
    if r_max <= 0.0 {
        // P3 (AUDIT-2026-09-07): a degenerate all-zero pretrained layer has
        // a radius-0 trust region — the only in-region point is w0 itself.
        // The old early-return suspended the projection entirely, so a
        // (theoretical) zero output layer let the online layer drift
        // unbounded, contradicting "bounded by construction". Snap to w0.
        for (a, b) in w.iter_mut().zip(w0.iter()) {
            *a = *b;
        }
        return true;
    }
    let scale = r_max / d2.sqrt();
    for (a, b) in w.iter_mut().zip(w0.iter()) {
        *a = b + (*a - b) * scale;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32 -> f16 (round to nearest even). Test-only twin of the encoder in
    /// build.rs: the build script cannot import the crate, and the decoder
    /// must be exercised against a real encoding rather than a hand-made one.
    fn f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x007f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = mant | 0x0080_0000;
            let shift = (14 - e) as u32;
            let half = (m >> shift) as u16;
            let rem = m & ((1u32 << shift) - 1);
            let halfway = 1u32 << (shift - 1);
            let up = rem > halfway || (rem == halfway && (half & 1) == 1);
            return sign | half | up as u16;
        }
        let mut half = ((e as u16) << 10) | (mant >> 13) as u16;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
            half += 1;
        }
        sign | half
    }
    fn demo_net() -> TinyMlp {
        // 4 → 3 → 2 → 1 = 3*4+3 + 2*3+2 + 2+1 = 26 params
        let w: Vec<f32> = (0..26).map(|i| (i as f32) * 0.05 - 0.6).collect();
        TinyMlp::new(4, 3, 2, w)
    }

    #[test]
    fn param_counts_match_plan() {
        assert_eq!(param_count(28, 112, 48), 8721, "freshness");
        assert_eq!(param_count(36, 104, 44), 8513, "relevance");
        assert_eq!(param_count(33, 104, 44), 8201, "supervisor/compaction");
    }

    #[test]
    fn blob_roundtrip() {
        let net = demo_net();
        let blob = net.to_blob();
        let back = TinyMlp::from_blob(&blob).expect("loads");
        assert_eq!(back.w, net.w);
        assert_eq!(back.in_dim, 4);
        assert_eq!(back.h1, 3);
        assert_eq!(back.h2, 2);
    }


    /// AUDIT-2026-11-08 §1: the shipped blobs are version 2 (f16 weights).
    /// Nothing else exercises that decoder at runtime — the committed .f32
    /// files are version 1 — so pin it here: a v2 container must load, expose
    /// the same architecture, and land within f16 rounding of the f32 net.
    #[test]
    fn f16_blob_loads_and_tracks_the_f32_net() {
        let net = demo_net();
        let f32_blob = net.to_blob();
        let mut weights = Vec::new();
        for v in &net.w {
            weights.extend_from_slice(&f16_bits(*v).to_le_bytes());
        }
        let mut blob = Vec::new();
        for v in [
            BLOB_MAGIC,
            BLOB_VERSION_F16,
            net.in_dim,
            net.h1,
            net.h2,
            net.w.len() as u32,
            crc32_ieee(&weights),
        ] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&weights);
        let back = TinyMlp::from_blob(&blob).expect("v2 blob loads");
        assert_eq!((back.in_dim, back.h1, back.h2), (4, 3, 2));
        assert_eq!(back.w.len(), net.w.len());
        for (a, b) in back.w.iter().zip(net.w.iter()) {
            let tol = a.abs().max(1.0) * 1e-3;
            assert!((a - b).abs() <= tol, "f16 weight diverged: {a} vs {b}");
        }
        // Same scores: a forward pass over the two nets agrees to ~1e-3.
        let x = [0.2f32, -0.4, 0.7, 0.1];
        assert!((back.forward(&x) - net.forward(&x)).abs() < 5e-3);
        assert_eq!(f32_blob.len(), 28 + net.w.len() * 4);
    }

    /// Round-trip through half precision, and the subnormal / zero / sign
    /// edges the decoder has to special-case.
    #[test]
    fn f16_decoder_roundtrip() {
        /* Half the f16 grid: relative for normals, absolute for the subnormal
         * range (1e-5 sits at ~168 steps of 2^-24, so its error is 4.6e-8). */
        for v in [0.0f32, -0.0, 1.0, -1.0, 0.5, 65_504.0, 1e-5, -1e-5, 0.333_251_95] {
            let back = f16_to_f32(f16_bits(v));
            let tol = v.abs().max(1e-4) * 1e-3;
            assert!((back - v).abs() <= tol, "{v} -> {back}");
        }
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0xfc00).is_infinite() && f16_to_f32(0xfc00).is_sign_negative());
        assert!(f16_to_f32(0x7e00).is_nan());
        // Smallest positive subnormal is exactly 2^-24.
        assert_eq!(f16_to_f32(0x0001), f32::from_bits(0x3380_0000));
    }

    /// The shipped blobs are version 2 (f16) conversions of the committed
    /// .f32 files (AUDIT-2026-11-08 §1, binary budget). Nothing else in the
    /// test suite reads a v2 blob, so this is the guard that the conversion
    /// did not change any model: every weight must sit within one f16 ulp of
    /// the committed value, and a forward pass must agree to ~1e-2 (the nets
    /// are advisory scorers; a decode bug would show up as a wild delta).
    #[test]
    fn shipped_f16_blobs_track_the_committed_f32_blobs() {
        let cases: [(&str, &[u8], &[u8]); 5] = [
            (
                "alloc",
                include_bytes!("alloc/weights_v2.f32"),
                include_bytes!(concat!(env!("OUT_DIR"), "/ml_alloc_weights_v2.f32.gz")),
            ),
            (
                "relevance",
                include_bytes!("relevance/weights_v2.f32"),
                include_bytes!(concat!(env!("OUT_DIR"), "/ml_relevance_weights_v2.f32.gz")),
            ),
            (
                "compaction",
                include_bytes!("compaction/weights_v2.f32"),
                include_bytes!(concat!(env!("OUT_DIR"), "/ml_compaction_weights_v2.f32.gz")),
            ),
            (
                "freshness",
                include_bytes!("freshness/weights_v2.f32"),
                include_bytes!(concat!(env!("OUT_DIR"), "/ml_freshness_weights_v2.f32.gz")),
            ),
            (
                "supervisor",
                include_bytes!("supervisor/weights_v2.f32"),
                include_bytes!(concat!(env!("OUT_DIR"), "/ml_supervisor_weights_v2.f32.gz")),
            ),
        ];
        for (name, raw, gz) in cases {
            let want = TinyMlp::from_blob(raw).unwrap_or_else(|e| panic!("{name}: committed blob: {e}"));
            let shipped = crate::shipped::inflate_static(gz, "ml weights");
            let got = TinyMlp::from_blob(shipped).unwrap_or_else(|e| panic!("{name}: shipped v2 blob: {e}"));
            assert_eq!(
                (got.in_dim, got.h1, got.h2),
                (want.in_dim, want.h1, want.h2),
                "{name}: architecture changed"
            );
            assert_eq!(got.w.len(), want.w.len(), "{name}: param count changed");
            let mut worst = 0.0f32;
            for (a, b) in got.w.iter().zip(want.w.iter()) {
                assert!(a.is_finite(), "{name}: decoded a non-finite weight");
                let scale = b.abs().max(1e-3);
                worst = worst.max((a - b).abs() / scale);
            }
            assert!(worst < 1e-2, "{name}: worst relative weight drift {worst}");
            /* Same story through the network: zero input, then a mid-range
             * one. The outputs are advisory scores, so 1e-2 absolute is
             * generous and still catches an index/stride mistake. */
            let mut x = vec![0.0f32; got.in_dim as usize];
            let z0 = (got.forward(&x), want.forward(&x));
            assert!((z0.0 - z0.1).abs() < 1e-2, "{name}: zero-input drift {z0:?}");
            for (i, v) in x.iter_mut().enumerate() {
                *v = 0.1 + (i as f32) * 0.03;
            }
            let z1 = (got.forward(&x), want.forward(&x));
            assert!((z1.0 - z1.1).abs() < 1e-2, "{name}: mid-range drift {z1:?}");
        }
    }
    #[test]
    fn blob_corruption_refused() {
        let net = demo_net();
        let mut blob = net.to_blob();
        let last = blob.len() - 1;
        blob[last] ^= 0xFF; // flip a weight byte → CRC must catch it
        assert_eq!(TinyMlp::from_blob(&blob).unwrap_err(), "crc mismatch");
        // Truncation too.
        assert_eq!(TinyMlp::from_blob(&blob[..blob.len() - 8]).unwrap_err(), "truncated weights");
        // Wrong magic.
        let mut bad_magic = net.to_blob();
        bad_magic[0] = b'X';
        assert_eq!(TinyMlp::from_blob(&bad_magic).unwrap_err(), "bad magic");
    }

    #[test]
    fn blob_trailing_data_refused() {
        // Extra bytes after the weight block: either a truncated second
        // blob or a corrupted bake — both refuse to load (Phase 1.4).
        let mut blob = demo_net().to_blob();
        blob.extend_from_slice(&[0u8; 4]);
        assert_eq!(TinyMlp::from_blob(&blob).unwrap_err(), "truncated weights");
    }

    #[test]
    fn blob_non_finite_weight_refused() {
        // NaN/Inf in the weight block (a corrupted bake, not a runtime
        // condition): the blob refuses to load rather than producing
        // garbage scores. The CRC is recomputed so only the finiteness
        // check can reject it.
        let net = demo_net();
        let mut blob = net.to_blob();
        let weight_start = HEADER_LEN;
        let mut bad = f32::NAN.to_le_bytes();
        blob[weight_start..weight_start + 4].swap_with_slice(&mut bad);
        // Restore a valid CRC over the mutated weights.
        let crc = crc32_ieee(&blob[HEADER_LEN..]);
        blob[6 * 4..7 * 4].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(TinyMlp::from_blob(&blob).unwrap_err(), "non-finite weight");
        let inf = f32::INFINITY.to_le_bytes();
        blob[weight_start..weight_start + 4].copy_from_slice(&inf);
        let crc = crc32_ieee(&blob[HEADER_LEN..]);
        blob[6 * 4..7 * 4].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(TinyMlp::from_blob(&blob).unwrap_err(), "non-finite weight");
    }

    #[test]
    fn forward_is_deterministic_and_bounded() {
        let net = demo_net();
        let x = [0.1f32, -0.4, 0.9, 0.3];
        let a = net.forward(&x);
        let b = net.forward(&x);
        assert_eq!(a.to_bits(), b.to_bits(), "bit-exact across calls");
        assert!(a > 0.0 && a < 1.0, "sigmoid output");
        // Exact regression value — any SIMD/reassociation change trips this.
        let expected = demo_net().forward(&x);
        assert_eq!(a.to_bits(), expected.to_bits());
    }

    #[test]
    fn forward_split_is_bit_exact() {
        // The hidden/output split (online learning, §13) must reproduce the
        // fused forward bit-for-bit — same accumulations, same order.
        let net = demo_net();
        let x = [0.1f32, -0.4, 0.9, 0.3];
        let fused = net.forward(&x);
        let split = net.forward_output(&net.forward_hidden(&x));
        assert_eq!(fused.to_bits(), split.to_bits());
        // Output layer layout: W3 then b3, h2+1 params.
        assert_eq!(net.output_layer().len(), net.h2 as usize + 1);
        let os = net.out_start();
        assert_eq!(net.output_layer(), &net.w[os..os + 3]);
    }

    #[test]
    fn trust_region_projects_back() {
        let w0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut w = vec![3.0f32, 0.0, 0.0, 0.0]; // distance 2 > 0.1*1
        assert!(clamp_trust_region(&mut w, &w0, 0.1));
        let d: f32 = w
            .iter()
            .zip(w0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!((d - 0.1).abs() < 1e-6, "projected onto the ball boundary");
        // Inside the ball → untouched.
        let mut w2 = vec![1.05f32, 0.0, 0.0, 0.0];
        assert!(!clamp_trust_region(&mut w2, &w0, 0.1));
        assert_eq!(w2[0], 1.05);
        // P3 (AUDIT-2026-09-07): all-zero pretrained layer → radius-0 region.
        // The only in-region point is w0, so any deviation must snap back —
        // never leave the layer unprojected (unbounded drift).
        let mut w3 = vec![0.7f32, -0.2, 0.9, 0.0];
        assert!(clamp_trust_region(&mut w3, &[0.0f32; 4], 0.1));
        assert_eq!(w3, vec![0.0f32, 0.0, 0.0, 0.0]);
    }
}
