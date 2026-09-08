// ml-train/src/embedding_anchor.rs — the frozen anchor channel of a hybrid
// semantic embedding.
//
// Diagnosis (embed-diag, 2026-09-06): 34 of 35 queries that hash-v1
// retrieved and the trained projector missed had a POSITIVE cosine margin
// in the raw 768-dim hash space — the trained tower destroyed signal the
// input features already carried.  The fix is architectural: the final
// 64-dim vector blends a FROZEN, code-only anchor channel (a Rademacher
// random projection of the hash features, zero blob bytes, bit-identical
// on every host) with the trained tower's output, and the tower is trained
// THROUGH that blend so it only learns the delta the anchor lacks.
//
//   v(x) = unit([ sqrt(w) * f_hat(x)[0..64-K] , sqrt(1-w) * a_hat(x) ])
//
// with f_hat the L2-normalized tower output, a_hat the L2-normalized
// anchor, K the anchor width and w the anchor energy share.  Because the
// channels occupy disjoint indices, cos(v1, v2) = w*cos(f1, f2) +
// (1-w)*cos(a1, a2) exactly.  The blob stays a standard SEM1 v1 artifact
// (tower only); K and w travel via SOFUU_EMB_ANCHOR / SOFUU_EMB_ANCHOR_W
// and are printed by every harness that applies them.  If a candidate ever
// passes §10, the same ~30 lines land in the runtime forward as part of
// the cutover — until then this module is dev-only.

use sofuu_core::embedding::hash_v1_features;

/// Fixed seed for the Rademacher anchor matrix — changing it invalidates
/// every anchor-carrying candidate.
pub const ANCHOR_SEED: u64 = 0xA11C_E500_0001;

/// (anchor width K, anchor energy share w) from the environment.
/// SOFUU_EMB_ANCHOR = 0 (default) disables the blend entirely.
pub fn anchor_cfg() -> (usize, f32) {
    let k = std::env::var("SOFUU_EMB_ANCHOR")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        .min(64);
    let w = std::env::var("SOFUU_EMB_ANCHOR_W")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    (k, w)
}

/// Row-major [768][k] Rademacher matrix (+1/-1 scaled by 1/sqrt(k)).
pub fn anchor_matrix(k: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; 768 * k];
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

/// The unit anchor vector for a text (sparse input, dense ±1 projection).
pub fn anchor_forward(x: &[f32], r: &[f32], k: usize) -> Vec<f32> {
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

pub fn unit(v: &mut Vec<f32>) {
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

/// The full hybrid vector from a tower output and a precomputed anchor.
/// `f_hat` must already be L2-normalized; `a_hat` comes from anchor_forward.
/// Result length is exactly 64 (SEMANTIC_DIM contract).
pub fn blend(f_hat: &[f32], a_hat: &[f32], k: usize, w: f32) -> Vec<f32> {
    let f_dims = 64 - k;
    let sw = w.sqrt();
    let tw = (1.0 - w).sqrt();
    let mut v = Vec::with_capacity(64);
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

/// Exact hybrid cosine from the two unit channels (disjoint indices).
pub fn hybrid_cos(f1: &[f32], a1: &[f32], f2: &[f32], a2: &[f32], k: usize, w: f32) -> f32 {
    let f_dims = 64 - k;
    let mut cf = 0.0f32;
    let mut i = 0;
    while i < f_dims {
        cf += f1[i] * f2[i];
        i += 1;
    }
    let mut ca = 0.0f32;
    let mut i = 0;
    while i < a1.len() {
        ca += a1[i] * a2[i];
        i += 1;
    }
    let _ = k;
    w * cf + (1.0 - w) * ca
}

/// Gradient of the hybrid cosine w.r.t. the TOWER's raw output, given the
/// other side's hybrid vector.  Chain rule: d/df1_j = sqrt(w) * v2_j, and
/// v2_j = sqrt(w) * f2_hat_j on the tower region, 0 beyond it — so the
/// model-output gradient is w * f2_hat on the first 64-K dims and zero on
/// the anchor region (the anchor has no model path).
pub fn hybrid_out_grad(other_v: &[f32], k: usize, w: f32, out: &mut [f32]) {
    let f_dims = 64 - k;
    let mut j = 0;
    while j < f_dims {
        out[j] = w * other_v[j];
        j += 1;
    }
    while j < out.len() {
        out[j] = 0.0;
        j += 1;
    }
}

/// Convenience: anchor vector straight from text.
pub fn anchor_of_text(t: &str, r: &[f32], k: usize) -> Vec<f32> {
    anchor_forward(&hash_v1_features(t), r, k)
}

// ── token anchor (SOFUU_EMB_ANCHOR_KIND=tok) ────────────────────────────
//
// The Rademacher anchor is a lossy lottery for the exact-token categories:
// it spreads every trigram ±1 across all K dims, so shared identifiers
// collide with unrelated noise and the code/errors floors swing wildly
// with K (diag: code 0.333 at K=32 but 0.917 at K=56).  The token anchor
// instead buckets CONTENT TOKENS deterministically — maximal alphanumeric
// runs (identifiers, codes, version strings, path segments) — so two texts
// that share a token share its bucket EXACTLY and unshared tokens
// contribute nothing to each other.  Still zero blob bytes, pure code.

fn fnv1a64(bytes: &[u8]) -> u64 {
    // local copy: same construction as the runtime's historical hash
    // (multiplier 0x1000000001b3 is deliberate, do NOT "fix")
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000000001b3);
    }
    h
}

fn is_tok_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.'
}

/// Unit token-anchor vector for a text at `k` buckets.
pub fn token_anchor(t: &str, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k];
    let bytes = t.as_bytes();
    let mut start: Option<usize> = None;
    let mut i = 0usize;
    while i <= bytes.len() {
        let c = if i < bytes.len() { bytes[i] } else { b' ' };
        if is_tok_byte(c) && i < bytes.len() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s0) = start {
            let len = i - s0;
            let mut has_special = false;
            let mut j = s0;
            while j < i {
                let b = bytes[j];
                if b.is_ascii_digit() || b == b'_' || b == b'-' || b == b'.' {
                    has_special = true;
                }
                j += 1;
            }
            // content tokens: identifiers/codes/versions/paths and ordinary
            // words of 4+ chars; skip short pure-alpha fragments
            if len >= 4 || (len >= 2 && has_special) {
                let h = fnv1a64(&bytes[s0..i]);
                let bidx = ((h >> 17) as usize) % k;
                out[bidx] += if has_special { 1.5 } else { 1.0 };
            }
            start = None;
        }
        i += 1;
    }
    unit(&mut out);
    out
}

/// Anchor kind: "rp" (Rademacher projection of the 768 features, default)
/// or "tok" (deterministic content-token bucketing).
pub fn anchor_kind() -> &'static str {
    match std::env::var("SOFUU_EMB_ANCHOR_KIND").as_deref() {
        Ok("tok") => "tok",
        _ => "rp",
    }
}

/// Anchor vector for text under the configured kind.  `r` is the
/// precomputed Rademacher matrix (used only for kind "rp").
pub fn anchor_text(t: &str, r: &[f32], k: usize) -> Vec<f32> {
    if anchor_kind() == "tok" {
        token_anchor(t, k)
    } else {
        anchor_forward(&hash_v1_features(t), r, k)
    }
}
