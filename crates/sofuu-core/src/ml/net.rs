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
    /// count, CRC) is an error — the caller refuses to load.
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
        if version != BLOB_VERSION {
            return Err("unsupported version");
        }
        if in_dim == 0 || h1 == 0 || h2 == 0 {
            return Err("zero dimension");
        }
        if params != param_count(in_dim, h1, h2) {
            return Err("param count does not match dims");
        }
        let weight_bytes = &bytes[HEADER_LEN..];
        if weight_bytes.len() != params as usize * 4 {
            return Err("truncated weights");
        }
        if crc32_ieee(weight_bytes) != crc {
            return Err("crc mismatch");
        }
        let w = weight_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
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
    if d2.sqrt() <= r_max || r_max <= 0.0 {
        return false;
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
    }
}
