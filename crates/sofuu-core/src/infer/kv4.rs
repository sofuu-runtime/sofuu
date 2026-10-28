//! KV4: the KIVI-layout QUANTIZED KV cache on the GPU decode/prefill path.
//!
//! The point of this module is memory: a f32 KV cache costs
//! `2 * n_kv_heads * head_dim * 4` bytes per token per layer, which on a
//! 28-layer 8-kv-head 128-dim model is 224 KB/token - 940 MB at a 4096-token
//! context. KV4 stores 32-token blocks as 4-bit codes (or 2-bit) with an f32
//! scale+min, exactly like `crate::infer::kivi` does for the CPU path, and
//! keeps only a small f32 ring of the newest rows, so the same context costs
//! ~43 KB/token and NO attention kernel ever materializes an f32 K/V row.
//!
//! # Layout (byte-compatible with kivi.rs)
//!
//! The blocks are the SAME bytes a CPU reader of
//! `crate::infer::kivi::QuantizedKBlock` / `QuantizedVBlock` sees, which is
//! what `kv4_pack_k_block` and `kv4_pack_v_token` prove in the unit tests:
//!
//! * K is quantized **per channel** over `BLOCK` consecutive tokens
//!   (`data[c * k_bpc + t/2]`, even token in the low nibble),
//! * V is quantized **per token** over channel groups of `GROUP`
//!   (`data[t * v_tp + c/2]`, even channel in the low nibble),
//! * one f32 scale and one f32 min per (K block, channel) and per
//!   (V token, channel group),
//! * the newest rows stay f32 in a ring of `Kv4Layout::ring` rows.
//!
//! # Which rows are where
//!
//! `nq` is a pure function of the position, so host and kernel agree without
//! any extra state: the quantized prefix is `nq(pos)` rows and every row in
//! `nq(pos)..=pos` is read from the f32 ring. `blocks_needed` says how many
//! blocks must have been quantized before the attention of `pos` can run.
//!
//! A block is quantized only once it is `BLOCK` tokens older than the newest
//! row, i.e. while 32 rows behind it are still f32 - the "recent window" of the
//! KIVI paper and the `residual_window` of kivi.rs. The ring must therefore
//! hold at least `BLOCK - 1` residual rows PLUS the rows of one prefill chunk,
//! which is why `ring = MAX_CHUNK + RING_SLACK`.

/// Tokens per quantization block (kivi.rs block size / V group size).
pub const BLOCK: usize = 32;
/// Channels per V quantization group.
pub const GROUP: usize = 32;
/// The largest prefill chunk KV4 accepts (the b4 chain's row cap).
pub const MAX_CHUNK: usize = 128;
/// f32 residual rows the ring must keep on top of one chunk: the 32 newest
/// rows are never quantized and a block boundary can leave up to 31 more.
pub const RING_SLACK: usize = 64;

/// Ring rows for a given chunk cap.
pub const fn ring_rows(max_chunk: usize) -> usize {
    max_chunk + RING_SLACK
}

/// How many blocks must be quantized before position `pos` can attend.
/// Equals `nq(pos) / BLOCK` by construction.
#[inline]
pub fn blocks_needed(pos: usize) -> usize {
    let seq = pos + 1;
    if seq >= 2 * BLOCK { seq / BLOCK - 1 } else { 0 }
}

/// Rows `[0, nq)` are read from the packed blocks; `[nq, pos]` from the ring.
#[inline]
pub fn nq(pos: usize) -> usize {
    blocks_needed(pos) * BLOCK
}

/// The ring-size argument the kv4 kernels take: the f16-K layout rides in
/// bit 31 (the kernels mask it off and branch on it), since the dispatch
/// has no spare scalar binding.
pub fn ring_arg(layout: &Kv4Layout, ring: usize) -> u32 {
    (ring as u32) | if layout.k_f16 { 1u32 << 31 } else { 0 }
}
/// Which pair of kv4 attention kernels to dispatch.
///
/// `true` (default) selects the v2 pair in src/metal/kernels.metal: one warp
/// per key computing all R q heads of that key (one dequant feeds R dots), the
/// per-block K scale/min vectors formed once per 32 keys, V dequantized once per
/// key, and - decode only - the key range split across threadgroups so that
/// reuse does not cost occupancy. `SOFUU_KV4_KERNEL=old` pins the shipped
/// per-head kernels, which stay in the source as the A/B arm and as the fallback
/// for a (bits, GQA ratio) pair v2 has no compiled form for.
pub fn kernel_v2() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            std::env::var("SOFUU_KV4_KERNEL").ok().as_deref(),
            Some("old") | Some("0") | Some("v1") | Some("shipped")
        )
    })
}

/// The KV precision the GPU decode path uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvDtype {
    /// Byte-identical to the shipped f32 cache (the reproducibility contract).
    F32,
    /// 4-bit K and V (the default on the GPU decode path).
    Q4,
    /// 2-bit K and V (receipt only; the accuracy band does not hold).
    Q2,
    /// f16 K + 4-bit V: K is stored as raw halves (no scale/min), V as
    /// 4-bit codes. The band-holding candidate where a uniform 4-bit cache
    /// costs too much accuracy.
    F16K,
    /// 8-bit K and V: the 'does more precision hold the band' probe.
    Q8,
}

impl KvDtype {
    pub fn bits(self) -> u32 {
        match self {
            KvDtype::F32 => 32,
            KvDtype::Q4 => 4,
            KvDtype::Q2 => 2,
            KvDtype::F16K => 4,
            KvDtype::Q8 => 8,
        }
    }
    /// True when K is stored as f16 halves instead of packed codes.
    pub fn k_f16(self) -> bool {
        matches!(self, KvDtype::F16K)
    }
    pub fn is_quantized(self) -> bool {
        !matches!(self, KvDtype::F32)
    }
    /// CLI / env parsing. Accepts f32/fp32/0, 4/q4/4bit, 2/q2/2bit, plus the
    /// f16 spelling the score JSON used before (an f32 cache was what "f16"
    /// meant on this path).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "0" | "f16" | "fp16" | "none" => Some(KvDtype::F32),
            "4" | "q4" | "kv4" | "4bit" | "int4" => Some(KvDtype::Q4),
            "2" | "q2" | "kv2" | "2bit" | "int2" => Some(KvDtype::Q2),
            "f16k" | "k16" | "f16-k" | "kv16k" => Some(KvDtype::F16K),
            "8" | "q8" | "8bit" | "kv8" | "int8" => Some(KvDtype::Q8),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            KvDtype::F32 => "f32",
            KvDtype::Q4 => "kv4",
            KvDtype::Q2 => "kv2",
            KvDtype::F16K => "f16k",
            KvDtype::Q8 => "kv8",
        }
    }
    /// Env default: SOFUU_KV_DTYPE when set. The CLI flag overrides it, and
    /// --kv-dtype f32 is the documented reproducibility escape hatch.
    pub fn from_env() -> Option<Self> {
        std::env::var("SOFUU_KV_DTYPE").ok().and_then(|v| Self::parse(&v))
    }
}

/// Byte geometry of one layer's quantized cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kv4Layout {
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub nblk: usize,
    pub ring: usize,
    pub bits: u32,
    /// K stored as f16 halves (V still packed at `bits`).
    pub k_f16: bool,
}

impl Kv4Layout {
    pub fn new(
        head_dim: usize, n_kv_heads: usize, max_seq: usize, ring: usize, bits: u32,
        k_f16: bool,
    ) -> Kv4Layout {
        Kv4Layout { head_dim, n_kv_heads, nblk: max_seq / BLOCK, ring, bits, k_f16 }
    }
    #[inline]
    pub fn per_byte(&self) -> usize {
        (8 / self.bits) as usize
    }
    /// Packed bytes per K channel (one block of BLOCK tokens).
    #[inline]
    pub fn k_bpc(&self) -> usize {
        BLOCK * self.bits as usize / 8
    }
    /// Packed bytes per V token (all channels).
    #[inline]
    pub fn v_tp(&self) -> usize {
        self.head_dim * self.bits as usize / 8
    }
    /// Channel groups per V token.
    #[inline]
    pub fn ng(&self) -> usize {
        self.head_dim / GROUP
    }
    /// Bytes of the f32 residual ring the GPU binds as k_buf/v_buf.
    pub fn ring_bytes(&self) -> usize {
        self.n_kv_heads * self.ring * self.head_dim * 4
    }
    pub fn kq_bytes(&self) -> usize {
        if self.k_f16 {
            self.n_kv_heads * self.nblk * BLOCK * self.head_dim * 2
        } else {
            self.n_kv_heads * self.nblk * self.head_dim * self.k_bpc()
        }
    }
    pub fn k_scale_bytes(&self) -> usize {
        // f16 K carries no scale; keep one byte so layer offsets stay valid.
        if self.k_f16 { 1 } else { self.n_kv_heads * self.nblk * self.head_dim * 4 }
    }
    pub fn qv_bytes(&self) -> usize {
        self.n_kv_heads * self.nblk * BLOCK * self.v_tp()
    }
    pub fn v_scale_bytes(&self) -> usize {
        self.n_kv_heads * self.nblk * BLOCK * self.ng() * 4
    }
    /// Bytes per token per layer: K codes + K scale/min + V codes + V
    /// scale/min, summed over the kv heads.
    pub fn bytes_per_token(&self) -> usize {
        // K codes are BLOCK-token packed; K scale+min are per (block,
        // channel); V codes and V scale+min are per token.
        if self.k_f16 {
            return self.n_kv_heads * (self.head_dim * 2 + self.v_tp() + self.ng() * 8);
        }
        let per_head = self.head_dim * self.k_bpc() / BLOCK
            + self.head_dim * 8 / BLOCK
            + self.v_tp()
            + self.ng() * 8;
        self.n_kv_heads * per_head
    }
}

// ---------------------------------------------------------------------------
// Host reference quantizer / dequantizer (bit-identical to kivi.rs).
// ---------------------------------------------------------------------------

#[inline]
fn quant_code(x: f32, scale: f32, min: f32, maxv: f32) -> u8 {
    let inv = if scale > 1e-12 { 1.0 / scale } else { 0.0 };
    ((x - min) * inv).round().clamp(0.0, maxv) as u8
}

/// Per-channel K block quantizer. `tokens` is `[BLOCK][head_dim]` row-major.
pub fn kv4_pack_k_block(tokens: &[f32], head_dim: usize, bits: u32)
    -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    assert_eq!(tokens.len(), BLOCK * head_dim);
    let maxv = ((1u32 << bits) - 1) as f32;
    let per_byte = (8 / bits) as usize;
    let bpc = BLOCK * bits as usize / 8;
    let mut data = vec![0u8; head_dim * bpc];
    let mut scales = vec![0.0f32; head_dim];
    let mut mins = vec![0.0f32; head_dim];
    for c in 0..head_dim {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for t in 0..BLOCK {
            let v = tokens[t * head_dim + c];
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
        let (scale, min) = if hi <= lo { (0.0, lo) } else { ((hi - lo) / maxv, lo) };
        scales[c] = scale;
        mins[c] = min;
        let base = c * bpc;
        for t in 0..BLOCK {
            let q = quant_code(tokens[t * head_dim + c], scale, min, maxv);
            let sh = (t % per_byte) * bits as usize;
            data[base + t / per_byte] |= (q & (maxv as u8)) << sh;
        }
    }
    (data, scales, mins)
}

/// Per-token V quantizer. `tokens` is `[BLOCK][head_dim]` row-major.
pub fn kv4_pack_v_token(tokens: &[f32], head_dim: usize, bits: u32)
    -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    assert_eq!(tokens.len(), BLOCK * head_dim);
    assert_eq!(head_dim % GROUP, 0);
    let maxv = ((1u32 << bits) - 1) as f32;
    let per_byte = (8 / bits) as usize;
    let tp = head_dim * bits as usize / 8;
    let ng = head_dim / GROUP;
    let mut data = vec![0u8; BLOCK * tp];
    let mut scales = vec![0.0f32; BLOCK * ng];
    let mut mins = vec![0.0f32; BLOCK * ng];
    for t in 0..BLOCK {
        for g in 0..ng {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for j in 0..GROUP {
                let v = tokens[t * head_dim + g * GROUP + j];
                if v < lo { lo = v; }
                if v > hi { hi = v; }
            }
            let (scale, min) = if hi <= lo { (0.0, lo) } else { ((hi - lo) / maxv, lo) };
            scales[t * ng + g] = scale;
            mins[t * ng + g] = min;
            let base = t * tp + g * (GROUP / per_byte);
            for j in 0..GROUP {
                let q = quant_code(tokens[t * head_dim + g * GROUP + j], scale, min, maxv);
                let sh = (j % per_byte) * bits as usize;
                data[base + j / per_byte] |= (q & (maxv as u8)) << sh;
            }
        }
    }
    (data, scales, mins)
}

/// Dequantize one channel of one K block (the kernel's fold, channel-major).
#[inline]
pub fn kv4_get_k(data: &[u8], scales: &[f32], mins: &[f32], head_dim: usize, bits: u32,
                 b: usize, c: usize, t_in_block: usize) -> f32 {
    let per_byte = (8 / bits) as usize;
    let bpc = BLOCK * bits as usize / 8;
    let idx = (b * head_dim + c) * bpc + t_in_block / per_byte;
    let code = (data[idx] >> ((t_in_block % per_byte) * bits as usize)) & ((1u8 << bits) - 1);
    code as f32 * scales[b * head_dim + c] + mins[b * head_dim + c]
}

/// Dequantize one channel of one V row (`t_abs` is the absolute token index).
#[inline]
pub fn kv4_get_v(data: &[u8], scales: &[f32], mins: &[f32], head_dim: usize, bits: u32,
                 t_abs: usize, c: usize) -> f32 {
    let per_byte = (8 / bits) as usize;
    let tp = head_dim * bits as usize / 8;
    let ng = head_dim / GROUP;
    let g = c / GROUP;
    let idx = t_abs * tp + c / per_byte;
    let code = (data[idx] >> ((c % per_byte) * bits as usize)) & ((1u8 << bits) - 1);
    code as f32 * scales[t_abs * ng + g] + mins[t_abs * ng + g]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::kivi::{KiviBits, QuantizedKBlock, QuantizedVBlock};

    fn lcg(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 0.5) * 6.0
        }).collect()
    }

    /// The host packer must produce the byte-for-byte layout kivi.rs writes:
    /// same codes, same scale/min, same nibble order. This is what lets the
    /// GPU kernel read the SAME block bytes a CPU KIVI reader would.
    #[test]
    fn pack_k_matches_kivi_block() {
        for &(bits, kb) in &[(4u32, KiviBits::Four), (2u32, KiviBits::Two)] {
            for &hd in &[32usize, 64, 128] {
                let toks = lcg(7 + hd as u64 + bits as u64, BLOCK * hd);
                let (data, scales, mins) = kv4_pack_k_block(&toks, hd, bits);
                let k = QuantizedKBlock::new(&toks, BLOCK, hd, kb);
                assert_eq!(data, k.data, "K data hd={hd} bits={bits}");
                assert_eq!(scales, k.scales, "K scales hd={hd} bits={bits}");
                assert_eq!(mins, k.mins, "K mins hd={hd} bits={bits}");
                // and a dequantized row equals the kivi reader's own value
                for c in 0..hd {
                    for t in [0usize, 1, 5, 31] {
                        let mine = kv4_get_k(&data, &scales, &mins, hd, bits, 0, c, t);
                        let theirs = (k.data[(c * k.data.len() / hd) + t / (8 / bits) as usize]
                            >> ((t % (8 / bits) as usize) * bits as usize))
                            & ((1u8 << bits) - 1);
                        let theirs = theirs as f32 * k.scales[c] + k.mins[c];
                        assert_eq!(mine, theirs);
                    }
                }
            }
        }
    }

    #[test]
    fn pack_v_matches_kivi_block() {
        for &(bits, kb) in &[(4u32, KiviBits::Four), (2u32, KiviBits::Two)] {
            for &hd in &[32usize, 64, 128] {
                let toks = lcg(11 + hd as u64 + bits as u64, BLOCK * hd);
                let (data, scales, mins) = kv4_pack_v_token(&toks, hd, bits);
                let v = QuantizedVBlock::new(&toks, BLOCK, hd, kb, GROUP);
                assert_eq!(data, v.data, "V data hd={hd} bits={bits}");
                assert_eq!(scales, v.scales, "V scales hd={hd} bits={bits}");
                assert_eq!(mins, v.mins, "V mins hd={hd} bits={bits}");
                for t in 0..BLOCK {
                    for c in [0usize, 1, 31, hd - 1] {
                        let mine = kv4_get_v(&data, &scales, &mins, hd, bits, t, c);
                        let per_byte = (8 / bits) as usize;
                        let tp = hd * bits as usize / 8;
                        let g = c / GROUP;
                        let byte = data[t * tp + c / per_byte];
                        let code = (byte >> ((c % per_byte) * bits as usize)) & ((1u8 << bits) - 1);
                        assert_eq!(mine, code as f32 * scales[t * (hd / GROUP) + g]
                                              + mins[t * (hd / GROUP) + g]);
                    }
                }
            }
        }
    }

    /// The kernel's fold order (dequantize in registers, online softmax) must
    /// equal a plain f32 attention over the dequantized rows.
    #[test]
    fn fold_matches_f32_reference_attention() {
        let hd = 64usize;
        let bits = 4u32;
        let pos = 150usize;
        let seq = pos + 1;
        let nqi = nq(pos);
        let toks = lcg(23, seq * hd);
        // pack every block below nq
        let mut kq = Vec::new();
        let mut kqs = Vec::new();
        let mut kqm = Vec::new();
        let mut qv = Vec::new();
        let mut qvs = Vec::new();
        let mut qvm = Vec::new();
        let nb = nqi / BLOCK;
        for b in 0..nb {
            let blk = &toks[b * BLOCK * hd..(b + 1) * BLOCK * hd];
            let (d, s, m) = kv4_pack_k_block(blk, hd, bits);
            kq.extend_from_slice(&d);
            kqs.extend_from_slice(&s);
            kqm.extend_from_slice(&m);
            let (d, s, m) = kv4_pack_v_token(blk, hd, bits);
            qv.extend_from_slice(&d);
            qvs.extend_from_slice(&s);
            qvm.extend_from_slice(&m);
        }
        let q = lcg(99, hd);
        // reference: dequantize everything into f32 and run plain softmax
        let mut kref = vec![0.0f32; seq * hd];
        let mut vref = vec![0.0f32; seq * hd];
        for t in 0..seq {
            for c in 0..hd {
                if t < nqi {
                    kref[t * hd + c] = kv4_get_k(&kq, &kqs, &kqm, hd, bits, t / BLOCK, c, t % BLOCK);
                    vref[t * hd + c] = kv4_get_v(&qv, &qvs, &qvm, hd, bits, t, c);
                } else {
                    kref[t * hd + c] = toks[t * hd + c];
                    vref[t * hd + c] = toks[t * hd + c];
                }
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let mut sc = vec![0.0f32; seq];
        for t in 0..seq {
            let mut acc = 0.0f32;
            for c in 0..hd { acc += q[c] * kref[t * hd + c]; }
            sc[t] = acc * scale;
        }
        let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
        let mut outf = vec![0.0f32; hd];
        for t in 0..seq {
            for c in 0..hd { outf[c] += sc[t] * vref[t * hd + c]; }
        }
        for c in 0..hd { outf[c] /= sum; }
        // kernel-order fold
        let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
        let mut acc = vec![0.0f32; hd];
        for t in 0..seq {
            let mut s = 0.0f32;
            for c in 0..hd {
                let kv = if t < nqi {
                    kv4_get_k(&kq, &kqs, &kqm, hd, bits, t / BLOCK, c, t % BLOCK)
                } else { toks[t * hd + c] };
                s += q[c] * kv;
            }
            let s = s * scale;
            let vv: Vec<f32> = (0..hd).map(|c| {
                if t < nqi { kv4_get_v(&qv, &qvs, &qvm, hd, bits, t, c) } else { toks[t * hd + c] }
            }).collect();
            let mnew = m.max(s);
            let corr = (m - mnew).exp();
            let w = (s - mnew).exp();
            l = l * corr + w;
            for c in 0..hd { acc[c] = acc[c] * corr + w * vv[c]; }
            m = mnew;
        }
        for c in 0..hd {
            let got = acc[c] / l;
            assert!((got - outf[c]).abs() < 1e-5, "c={c} got={got} want={}", outf[c]);
        }
    }

    /// The residual window and the block prefix must cover the whole history
    /// for every position a decode or a <=128-row chunk can ask for.
    #[test]
    fn nq_and_ring_cover_every_position() {
        let ring = ring_rows(MAX_CHUNK);
        for pos in 0..4096usize {
            let nqi = nq(pos);
            assert_eq!(nqi % BLOCK, 0);
            assert!(nqi <= pos + 1);
            // rows [nq, pos] are the f32 residual rows
            let resid = pos + 1 - nqi;
            assert!(resid <= 2 * BLOCK - 1, "pos={pos} resid={resid}");
            // a chunk ending at pos writes rows [pos + 1 - chunk, pos]
            for chunk in [1usize, 7, 32, 127, MAX_CHUNK] {
                if chunk > pos + 1 { continue; }
                let oldest_needed = nq(pos + 1 - chunk);
                let newest = pos;
                assert!(newest - oldest_needed + 1 <= ring,
                        "chunk={chunk} pos={pos} span={}", newest - oldest_needed + 1);
                // every block the chunk's last row needs must already exist
                assert!(blocks_needed(pos) <= nq(pos) / BLOCK + 1);
            }
        }
    }
}
