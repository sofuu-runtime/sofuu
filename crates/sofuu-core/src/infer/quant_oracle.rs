// crates/sofuu-core/src/infer/quant_oracle.rs — kernel-level quant oracles.
//
// The loader's accept-list (TransformerModel::SUPPORTED_QUANT_TYPES) is a
// PROMISE: every type on it loads and produces correct logits. Before this
// module the promise was only words — Q3_K had a dequant, three kernels and
// no model anywhere on the machine, so "supported" meant "a constant lists
// it". A wrong nibble order, a transposed sub-block scale or a wrong hmask
// half loads happily and produces fluent garbage, which is the worst class
// of bug this engine can ship.
//
// Every test here builds SYNTHETIC blocks from a value spec, decodes them
// with the kernel under test and compares against an INDEPENDENT f64
// reference implemented in this file from the format description (ggml's
// block layouts). No reference ever calls engine code: f16 scales are
// decoded by ref_f16 (IEEE 754 binary16 from scratch), block layouts are
// re-derived from the byte offsets, and the dot reference is a plain f64
// sum. The kernel is the only thing on trial.
//
// Coverage table (the answer to "type -> dequant -> dot -> testable?"):
//
//   type      CPU dequant                     CPU dot / matvec                    testable
//   F32 (0)   none (row is f32)               matvec -> sofuu_gemv_f32             yes, exact f32
//   F16 (1)   f16_to_f32 (Rust, per element)   matvec -> sofuu_gemv_f16             yes, exact f32
//   Q4_0 (2)  dequantize_q4_0_row (Rust)      vec_dot_q4_0 (Rust) / matvec         yes, exact f32
//   Q8_0 (8)  dequantize_q8_0_row (Rust)      vec_dot_q8_0 (Rust) / matvec         yes, exact f32
//   Q4_K (12) dequantize_q4_k_row (C)         vec_dot_raw_q4_k (C, f32) / matvec   yes, exact f32
//   Q6_K (14) dequantize_q6_k_row (C)         vec_dot_raw_q6_k (C, f32) / matvec   yes, exact f32
//   Q3_K (11) dequantize_q3_k_row (C)         no exported f32 vec_dot; matvec only  yes, via matvec
//
// F32/F16/Q4_0/Q4_K/Q6_K have an f32 dot reachable from Rust, so their dot
// oracle runs the real vec_dot on arbitrary activations. Every dot is gated
// twice (see check_dot): <= 1e-5 relative on every well-conditioned row
// (|result| >= L1/16) AND <= 1e-6 * L1 on every row, so a fixture that
// cancels to near zero cannot hide an error behind a big relative number.
// Q3_K
// does NOT: sofuu_vec_dot_q3_k is file-static and the only entry point is
// matvec/matmat, which quantizes the activation to q8_K and runs an int8
// SDOT kernel on aarch64 (the f32 dequant+dot only runs on archs without
// SDOT). For those paths the fixtures use INTEGER activations whose maximum
// magnitude is exactly 127: amax/127 == 1.0 exactly, so q8_K quantisation
// is lossless (q == x) and the int8 kernel computes the exact integer dot.
// That makes the int8 path testable at the same 1e-5 gate — it proves the
// weight/scale/hmask decode and the accumulation, i.e. everything a
// "silently garbage" bug would break. What it does not prove is the
// activation quantiser's accuracy, which is an accuracy (not correctness)
// property shared with llama.cpp's own q3_K/q4_K/q6_K GEMV and is gated
// separately by the L1-scale checks in quant::tests.
//
//
// Every oracle test prints its receipt — "[quant-oracle] <TYPE> dequant N ulp
// of f32 (limit 1); dot X rel (limit 1e-5); every row <= 1e-6 of L1 — PASS" —
// visible with --nocapture, so the numbers are auditable in CI logs.
//
// These gates have teeth (each verified by injecting the bug into the kernel
// and watching this suite go red, then reverting): the q3_K hmask polarity
// flip, a transposed q6_K sub-block scale inside vec_dot, q4_K's high nibble
// given the low nibble's scale, q4_0's packed nibble order swapped, and the
// f16 subnormal shift off by one.
// every_accepted_quant_type_has_a_kernel_oracle() is the structural gate:
// it walks TransformerModel::SUPPORTED_QUANT_TYPES and RUNS each type's
// oracle, so adding a type to the accept-list without an oracle here turns
// the suite red instead of quietly widening the promise.

#![allow(clippy::needless_range_loop)]

use super::model::TransformerModel;
use super::quant::{
    dequantize_q3_k_row, dequantize_q4_0_row, dequantize_q4_k_row, dequantize_q6_k_row,
    dequantize_q8_0_row, f16_to_f32, matmat, matvec, tile_warm, type_row_size, vec_dot_q4_0,
    vec_dot_q8_0, vec_dot_raw_q4_k, vec_dot_raw_q6_k, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, QK_K,
};

// ─────────────────────────────────────────────────────────────────────
// Independent reference primitives. Nothing below calls engine code.
// ─────────────────────────────────────────────────────────────────────

/// Exact IEEE 754 binary16 -> f64, written from the format definition.
/// (Every binary16 value is exactly representable in f64, so this is the
/// exact value of the encoded number — no rounding anywhere.)
fn ref_f16(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    if e == 0 {
        s * m * 2f64.powi(-24) // subnormal (and +/-0)
    } else if e == 31 {
        if m == 0.0 {
            s * f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        s * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

/// binary16 bits for a value that is EXACTLY representable in binary16.
/// Fixtures pick dyadic scales on purpose so a scale can never be the thing
/// under test (the kernel's f16 decode is proven separately, exhaustively,
/// by oracle_f16_conversion_is_exact_for_all_65536_patterns).
fn f16_bits_exact(v: f64) -> u16 {
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    let bits = if a == 0.0 {
        sign
    } else if a < 2f64.powi(-14) {
        let m = (a / 2f64.powi(-24)).round();
        assert_eq!(m * 2f64.powi(-24), a, "{v} is not exact in binary16");
        assert!((1.0..=1023.0).contains(&m), "{v} out of subnormal range");
        sign | (m as u16)
    } else {
        let mut e = 0i32;
        while 2f64.powi(e + 1) <= a {
            e += 1;
        }
        while 2f64.powi(e) > a {
            e -= 1;
        }
        assert!((-14..=15).contains(&e), "{v} out of normal range");
        let m = (a / 2f64.powi(e) - 1.0) * 1024.0;
        let mr = m.round();
        assert_eq!(mr, m, "{v} is not exact in binary16");
        sign | (((e + 15) as u16) << 10) | (mr as u16)
    };
    assert_eq!(ref_f16(bits), v, "f16_bits_exact round-trip broke for {v}");
    bits
}

/// Largest allowed distance, in f32 ulps, between a kernel dequant and the
/// f64 reference rounded to f32. One multiply can round once, so 1 ulp is
/// the honest bound when the reference is the exact value.
fn dequant_ulps(got: f32, want: f64) -> u32 {
    let want32 = want as f32;
    if got == want32 && got == 0.0 {
        return 0; // +0.0 and -0.0 agree as values; sign of zero is not a bug
    }
    let a = got.to_bits() as i32 as i64;
    let b = want32.to_bits() as i32 as i64;
    (a - b).unsigned_abs() as u32
}

/// Compare a kernel dequant against the f64 reference. Returns the worst
/// ulp distance seen so the caller can report it.
fn check_dequant(label: &str, got: &[f32], want: &[f64], tol_ulps: u32) -> u32 {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut worst = 0u32;
    let mut worst_at = 0usize;
    for i in 0..got.len() {
        let u = dequant_ulps(got[i], want[i]);
        if u > worst {
            worst = u;
            worst_at = i;
        }
    }
    assert!(
        worst <= tol_ulps,
        "{label}: dequant is {worst} ulp away at index {worst_at}: kernel {} vs reference {} (got bits {:08x}, want bits {:08x})",
        got[worst_at],
        want[worst_at],
        got[worst_at].to_bits(),
        (want[worst_at] as f32).to_bits()
    );
    worst
}

/// Compare a kernel dot against the f64 reference. Two gates, both of which
/// must hold for EVERY fixture:
///
///   * `err <= 1e-6 * L1`, the most an f32 accumulation of these terms can
///     explain (L1 = sum |w_i x_i|), so a fixture that cancels to near zero
///     cannot hide a real error behind a large relative number;
///   * `<= 1e-5 relative` wherever the fixture is NOT cancellation-dominated
///     (|result| >= L1/16 — 99% of real projections), which is the gate the
///     task asks for and the number this function returns.
fn check_dot(label: &str, got: f32, want: f64, l1: f64) -> f64 {
    let abs_err = (got as f64 - want).abs();
    let rel = if want.abs() > 1e-30 {
        abs_err / want.abs()
    } else {
        abs_err
    };
    let rel_l1 = abs_err / l1.max(1e-30);
    assert!(
        rel_l1 <= 1e-6,
        "{label}: dot {got} vs reference {want} — {abs_err:.3e} absolute is {rel_l1:.3e} of L1 {l1:.3e} (f32 accumulation bound 1e-6)",
    );
    let well_conditioned = want.abs() >= l1 / 16.0;
    if well_conditioned {
        assert!(
            rel <= 1e-5,
            "{label}: dot {got} vs reference {want} — rel {rel:.3e} > 1e-5 ({:.1}% of L1, so not a cancellation case)",
            100.0 * want.abs() / l1.max(1e-30),
        );
    }
    if well_conditioned {
        rel
    } else {
        0.0
    }
}

/// f64 dot of a row of reference weights with an activation row.
fn ref_dot(w: &[f64], x: &[f32]) -> (f64, f64) {
    let mut sum = 0f64;
    let mut l1 = 0f64;
    for i in 0..w.len() {
        let t = w[i] * x[i] as f64;
        sum += t;
        l1 += t.abs();
    }
    (sum, l1)
}

/// Integer activations whose max magnitude is exactly 127, so activation
/// quantisation (d = amax/127 = 1.0) is LOSSLESS and the int8 kernels
/// compute the exact integer dot. Period 8 => every 32/256-value block of
/// every row contains the 127, and every rotation of the pattern does too.
const XPAT: [f32; 8] = [127.0, -64.0, 32.0, -16.0, 8.0, -4.0, 2.0, 0.0];

fn act_row(cols: usize, rot: usize) -> Vec<f32> {
    (0..cols).map(|i| XPAT[(i + rot) % XPAT.len()]).collect()
}

/// One synthetic weight matrix plus the independent expectation for it.
struct Fx {
    ty: u32,
    rows: usize,
    cols: usize,
    bytes: Vec<u8>,
    row_bytes: usize,
    ref_w: Vec<f64>,
}

impl Fx {
    fn new(ty: u32, rows: usize, cols: usize) -> Fx {
        let row_bytes = type_row_size(ty, cols);
        Fx {
            ty,
            rows,
            cols,
            bytes: vec![0u8; rows * row_bytes],
            row_bytes,
            ref_w: vec![0f64; rows * cols],
        }
    }

    /// Dequantise every row with the kernel's row dequantizer and gate it
    /// against the reference; returns the worst ulp distance.
    fn check_row_dequant(&self, label: &str, tol_ulps: u32) -> u32 {
        let mut got = vec![0f32; self.cols];
        let mut worst = 0;
        for r in 0..self.rows {
            let start = r * self.row_bytes;
            let row = &self.bytes[start..start + self.row_bytes];
            got.fill(f32::NAN);
            match self.ty {
                GGML_TYPE_Q4_0 => dequantize_q4_0_row(row, &mut got),
                GGML_TYPE_Q8_0 => dequantize_q8_0_row(row, &mut got),
                GGML_TYPE_Q4_K => dequantize_q4_k_row(row, &mut got),
                GGML_TYPE_Q6_K => dequantize_q6_k_row(row, &mut got),
                GGML_TYPE_Q3_K => dequantize_q3_k_row(row, &mut got),
                GGML_TYPE_F16 => {
                    for i in 0..self.cols {
                        let h = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
                        got[i] = f16_to_f32(h);
                    }
                }
                GGML_TYPE_F32 => {
                    for i in 0..self.cols {
                        got[i] = f32::from_le_bytes([
                            row[i * 4],
                            row[i * 4 + 1],
                            row[i * 4 + 2],
                            row[i * 4 + 3],
                        ]);
                    }
                }
                other => panic!("no dequant oracle for type {other}"),
            }
            let want = &self.ref_w[r * self.cols..(r + 1) * self.cols];
            worst = worst.max(check_dequant(
                &format!("{label} row {r} dequant"),
                &got,
                want,
                tol_ulps,
            ));
        }
        worst
    }

    /// The dispatch path every transformer projection actually takes:
    /// matvec (decode) and matmat (prefill) for every row, gated against
    /// the f64 reference. Returns the worst relative dot error.
    fn check_dispatch(&self, label: &str, batch: usize) -> f64 {
        let x = act_row(self.cols, 0);
        let mut y = vec![0f32; self.rows];
        matvec(&self.bytes, self.ty, self.rows, self.cols, &x, &mut y);

        let mut xb = Vec::with_capacity(batch * self.cols);
        for b in 0..batch {
            xb.extend_from_slice(&act_row(self.cols, b * 3));
        }
        let mut yb = vec![0f32; batch * self.rows];
        matmat(
            &self.bytes,
            self.ty,
            self.rows,
            self.cols,
            &xb,
            batch,
            &mut yb,
        );

        let mut worst = 0f64;
        for r in 0..self.rows {
            let w = &self.ref_w[r * self.cols..(r + 1) * self.cols];
            let (want, l1) = ref_dot(w, &x);
            worst = worst.max(check_dot(
                &format!("{label} matvec row {r}"),
                y[r],
                want,
                l1,
            ));
            for b in 0..batch {
                let xrow = &xb[b * self.cols..(b + 1) * self.cols];
                let (want, l1) = ref_dot(w, xrow);
                worst = worst.max(check_dot(
                    &format!("{label} matmat row {r} batch {b}"),
                    yb[b * self.rows + r],
                    want,
                    l1,
                ));
            }
        }
        worst
    }
}

// ─────────────────────────────────────────────────────────────────────
// Fixtures. Each builder WRITES the bytes from a value spec and computes
// the expectation from the same spec in f64 (never by reading the bytes
// back through engine code).
// ─────────────────────────────────────────────────────────────────────

// ── F32 / F16 ────────────────────────────────────────────────────────

/// variant 0: ordinary dyadic values / 1: zeros / 2: f16 extremes /
/// 3: identical / 4: alternating sign
fn fx_f16(rows: usize, cols: usize, variant: u8) -> Fx {
    let mut fx = Fx::new(GGML_TYPE_F16, rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let v: f64 = match variant {
                0 => ((c % 7) as f64 - 3.0) * 0.25,
                1 => 0.0,
                2 => {
                    if (c + r) % 4 == 0 {
                        65504.0
                    } else if (c + r) % 4 == 1 {
                        -65504.0
                    } else if (c + r) % 4 == 2 {
                        2f64.powi(-24)
                    } else {
                        -2f64.powi(-24)
                    }
                }
                3 => 1.5,
                _ => {
                    if (c + r) % 2 == 0 {
                        3.0
                    } else {
                        -3.0
                    }
                }
            };
            let bits = f16_bits_exact(v);
            let off = r * fx.row_bytes + c * 2;
            fx.bytes[off..off + 2].copy_from_slice(&bits.to_le_bytes());
            fx.ref_w[r * cols + c] = v;
        }
    }
    fx
}

fn fx_f32(rows: usize, cols: usize, variant: u8) -> Fx {
    let mut fx = Fx::new(GGML_TYPE_F32, rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let v: f32 = match variant {
                0 => (c as f32 * 0.125) - 2.0,
                1 => 0.0,
                2 => {
                    if c % 2 == 0 {
                        1.0e30
                    } else {
                        -1.0e-30
                    }
                }
                3 => 7.5,
                _ => {
                    if (c + r) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
            };
            let off = r * fx.row_bytes + c * 4;
            fx.bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
            fx.ref_w[r * cols + c] = v as f64;
        }
    }
    fx
}

// ── Q4_0 (18 B / 32 values: f16 d, 16 packed nibbles) ────────────────

/// variant 0: distinct mixed values / 1: all-zero scale / 2: f16 extreme
/// scales / 3: one identical value per block / 4: lattice extremes (sign
/// pattern, -8/7) / 5: alternating +/-7
fn fx_q4_0(rows: usize, cols: usize, variant: u8) -> Fx {
    let mut fx = Fx::new(GGML_TYPE_Q4_0, rows, cols);
    let nb = cols / 32;
    for r in 0..rows {
        for blk in 0..nb {
            let d: f64 = match variant {
                1 => 0.0,
                2 => {
                    if blk % 2 == 0 {
                        65504.0
                    } else {
                        2f64.powi(-24)
                    }
                }
                3 => 0.5,
                _ => 0.5 * (1 + (blk % 3)) as f64,
            };
            let off = r * fx.row_bytes + blk * 18;
            fx.bytes[off..off + 2].copy_from_slice(&f16_bits_exact(d).to_le_bytes());
            for j in 0..16 {
                let q = |i: usize| -> i32 {
                    match variant {
                        3 => 3,
                        4 => {
                            if (i + r) % 2 == 0 {
                                -8
                            } else {
                                7
                            }
                        }
                        5 => {
                            if (i + blk) % 2 == 0 {
                                7
                            } else {
                                -7
                            }
                        }
                        _ => (((i * 5 + r * 3 + blk * 11) % 16) as i32) - 8,
                    }
                };
                let q0 = q(j);
                let q1 = q(j + 16);
                fx.bytes[off + 2 + j] = ((q0 + 8) as u8) | (((q1 + 8) as u8) << 4);
                fx.ref_w[r * cols + blk * 32 + j] = d * q0 as f64;
                fx.ref_w[r * cols + blk * 32 + j + 16] = d * q1 as f64;
            }
        }
    }
    fx
}

// ── Q8_0 (34 B / 32 values: f16 d, 32 int8) ──────────────────────────

fn fx_q8_0(rows: usize, cols: usize, variant: u8) -> Fx {
    let mut fx = Fx::new(GGML_TYPE_Q8_0, rows, cols);
    let nb = cols / 32;
    for r in 0..rows {
        for blk in 0..nb {
            let d: f64 = match variant {
                1 => 0.0,
                2 => {
                    if blk % 2 == 0 {
                        65504.0
                    } else {
                        2f64.powi(-24)
                    }
                }
                3 => 0.25,
                _ => 0.5 * (1 + (blk % 2)) as f64,
            };
            let off = r * fx.row_bytes + blk * 34;
            fx.bytes[off..off + 2].copy_from_slice(&f16_bits_exact(d).to_le_bytes());
            for j in 0..32 {
                let q: i32 = match variant {
                    3 => -5,
                    4 => {
                        if (j + r) % 2 == 0 {
                            -128
                        } else {
                            127
                        }
                    }
                    5 => {
                        if (j + blk) % 2 == 0 {
                            127
                        } else {
                            -127
                        }
                    }
                    _ => (((j * 37 + r * 11 + blk * 7) % 256) as i32) - 128,
                };
                fx.bytes[off + 2 + j] = q as i8 as u8;
                fx.ref_w[r * cols + blk * 32 + j] = d * q as f64;
            }
        }
    }
    fx
}

// ── Q4_K (144 B / 256 values, 8 sub-blocks of 32) ────────────────────
//
// Layout (ggml type 12): f16 d | f16 dmin | 12 B scales | 128 B nibbles.
//   get_scale_min_k4(j): j<4 -> sc=q[j]&63, mn=q[j+4]&63
//                        j>=4 -> sc=(q[j+4]&0xF)|((q[j-4]>>6)<<4),
//                                mn=(q[j+4]>>4)|((q[j]>>6)<<4)
//   nibbles: four 32-byte groups; group g holds sub-block 2g in the LOW
//   nibble and 2g+1 in the HIGH nibble, element j and j+32.
//   value = d*sc[j/32]*q - dmin*mn[j/32]
//
// The sub-block scales are DISTINCT on purpose (0, 1, 2^j .. 63): a
// per-256 reference or a transposed sub-block order cannot pass this.

fn q4_k_sc(variant: u8, j: usize) -> (u8, u8) {
    match variant {
        1 => (0, 0),   // all-zero scale AND min
        2 => (63, 63), // both at the 6-bit max
        3 => (17, 5),  // identical everywhere
        4 => (
            if j % 2 == 0 { 0 } else { 63 },
            if j % 2 == 0 { 63 } else { 0 },
        ),
        _ => (((1u32 << (j % 6)) - 1) as u8, (j * 9) as u8), // 0,1,3,7,15,31 + 0..63
    }
}

fn fx_q4_k(rows: usize, cols: usize, variant: u8) -> Fx {
    assert_eq!(cols % QK_K, 0);
    let mut fx = Fx::new(GGML_TYPE_Q4_K, rows, cols);
    let nb = cols / QK_K;
    for r in 0..rows {
        for blk in 0..nb {
            let d: f64 = match variant {
                1 => 0.0,
                2 => 65504.0,
                3 => 1.0,
                _ => 0.5 * (1 + (blk % 2)) as f64,
            };
            let dmin: f64 = match variant {
                1 => 0.0,
                2 => 65504.0,
                3 => 0.25,
                _ => 0.125 * (1 + (blk % 3)) as f64,
            };
            let off = r * fx.row_bytes + blk * 144;
            fx.bytes[off..off + 2].copy_from_slice(&f16_bits_exact(d).to_le_bytes());
            fx.bytes[off + 2..off + 4].copy_from_slice(&f16_bits_exact(dmin).to_le_bytes());

            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                let (s, m) = q4_k_sc(variant, j);
                sc[j] = s;
                mn[j] = m;
            }
            // Inverse of get_scale_min_k4.
            for j in 0..4 {
                fx.bytes[off + 4 + j] = (sc[j] & 0x3f) | ((sc[j + 4] & 0x30) << 2);
                fx.bytes[off + 8 + j] = (mn[j] & 0x3f) | ((mn[j + 4] & 0x30) << 2);
                fx.bytes[off + 12 + j] = (sc[j + 4] & 0x0f) | ((mn[j + 4] & 0x0f) << 4);
            }
            for g in 0..4 {
                for l in 0..32 {
                    let q_of = |i: usize| -> u8 {
                        match variant {
                            3 => 9,
                            4 => {
                                if (i + r) % 2 == 0 {
                                    0
                                } else {
                                    15
                                }
                            }
                            _ => ((i * 13 + r * 7 + blk * 5 + g) % 16) as u8,
                        }
                    };
                    let lo = q_of(g * 64 + l); // sub-block 2g
                    let hi = q_of(g * 64 + 32 + l); // sub-block 2g+1
                    fx.bytes[off + 16 + g * 32 + l] = lo | (hi << 4);
                    let sb_lo = 2 * g;
                    let sb_hi = 2 * g + 1;
                    fx.ref_w[r * cols + blk * 256 + sb_lo * 32 + l] =
                        d * sc[sb_lo] as f64 * lo as f64 - dmin * mn[sb_lo] as f64;
                    fx.ref_w[r * cols + blk * 256 + sb_hi * 32 + l] =
                        d * sc[sb_hi] as f64 * hi as f64 - dmin * mn[sb_hi] as f64;
                }
            }
        }
    }
    fx
}

// ── Q6_K (210 B / 256 values, 16 sub-blocks of 16) ───────────────────
//
// Layout (ggml type 14): 128 B ql | 64 B qh | 16 int8 scales | f16 d.
//   element order per 128-value half n: for l in 0..32 (is = l/16)
//     y[l]     = d*sc[is+0] * ((ql[l]&0xF)    | ((qh[l]>>0 &3)<<4) - 32)
//     y[l+32]  = d*sc[is+2] * ((ql[l+32]&0xF) | ((qh[l]>>2 &3)<<4) - 32)
//     y[l+64]  = d*sc[is+4] * ((ql[l]>>4)     | ((qh[l]>>4 &3)<<4) - 32)
//     y[l+96]  = d*sc[is+6] * ((ql[l+32]>>4)  | ((qh[l]>>6 &3)<<4) - 32)
//   then ql += 64, qh += 32, sc += 8.
//   So global element e: n=e/128, t=e%128, which=t/32, l=t%32,
//   part=l/16, scale slot = n*8 + part + 2*which, qh shift = 2*which.

fn q6_k_scale(variant: u8, slot: usize) -> i8 {
    match variant {
        1 => 0,
        2 => {
            if slot % 2 == 0 {
                127
            } else {
                -128
            }
        }
        3 => -7,
        4 => {
            if slot % 2 == 0 {
                -128
            } else {
                127
            }
        }
        _ => [
            1i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ][slot],
    }
}

fn fx_q6_k(rows: usize, cols: usize, variant: u8) -> Fx {
    assert_eq!(cols % QK_K, 0);
    let mut fx = Fx::new(GGML_TYPE_Q6_K, rows, cols);
    let nb = cols / QK_K;
    for r in 0..rows {
        for blk in 0..nb {
            let d: f64 = match variant {
                1 => 0.0,
                2 => 65504.0,
                3 => 2.0,
                _ => 0.5 * (1 + (blk % 2)) as f64,
            };
            let off = r * fx.row_bytes + blk * 210;
            fx.bytes[off + 208..off + 210].copy_from_slice(&f16_bits_exact(d).to_le_bytes());
            for slot in 0..16 {
                let s = q6_k_scale(variant, slot);
                fx.bytes[off + 192 + slot] = s as u8;
            }
            for e in 0..256 {
                let q: i32 = match variant {
                    3 => -17,
                    4 => {
                        if (e + r) % 2 == 0 {
                            -32
                        } else {
                            31
                        }
                    }
                    _ => (((e * 29 + r * 13 + blk * 3) % 64) as i32) - 32,
                };
                let uq = (q + 32) as u8; // 6-bit unsigned
                let n = e / 128;
                let t = e % 128;
                let which = t / 32;
                let l = t % 32;
                let part = l / 16;
                // which 0/2 read ql[l] (low/high nibble), which 1/3 ql[l+32].
                let ql_idx = off + n * 64 + if which % 2 == 1 { 32 } else { 0 } + l;
                if which >= 2 {
                    fx.bytes[ql_idx] |= (uq & 0x0f) << 4;
                } else {
                    fx.bytes[ql_idx] |= uq & 0x0f;
                }
                let qh_idx = off + 128 + n * 32 + l;
                fx.bytes[qh_idx] |= ((uq >> 4) & 0x03) << (2 * which);
                let slot = n * 8 + part + 2 * which;
                fx.ref_w[r * cols + blk * 256 + e] =
                    d * q6_k_scale(variant, slot) as f64 * q as f64;
            }
        }
    }
    fx
}

// ── Q3_K (110 B / 256 values, 16 sub-blocks of 16) ───────────────────
//
// Layout (ggml type 11): 32 B hmask | 64 B qs | 12 B scales | f16 d.
//   12 scale bytes -> 16 signed 6-bit values (biased -32) by the aux[4]
//   permutation; per 128-value half, four 32-value groups j (2-bit shift
//   2j), each split into part 0 (qs[0..16]) and part 1 (qs[16..32]).
//   hmask bit (1 << (half*4 + j)) SET selects 0..3, CLEAR selects -4..-1.
//   value = d * (sc[half*8 + j*2 + part] - 32) * v

/// Inverse of the aux[4] permutation: 16 desired scale bytes -> 12 bytes.
/// (forward: b0=(a0&0x0f0f)|((T>>0&3)<<4), b1=(a1&0x0f0f)|((T>>2&3)<<4),
///  b2=((a0>>4)&0x0f0f)|((T>>4&3)<<4), b3=((a1>>4)&0x0f0f)|((T>>6&3)<<4),
///  with T = a2; byte k of b_i is scale[4*i+k])
fn q3_k_scale_bytes(f: &[u8; 16]) -> [u8; 12] {
    let mut a = [0u8; 12];
    for k in 0..4 {
        a[k] = (f[k] & 0x0f) | ((f[8 + k] & 0x0f) << 4);
        a[4 + k] = (f[4 + k] & 0x0f) | ((f[12 + k] & 0x0f) << 4);
        // The forward permutation shifts the 2-bit fragment << 4, so the
        // fragment of output byte f[i] is its bits 4-5 (output bytes hold
        // a 6-bit value in bits 0-5; bits 6-7 are never set).
        a[8 + k] = ((f[k] >> 4) & 0x03)
            | (((f[4 + k] >> 4) & 0x03) << 2)
            | (((f[8 + k] >> 4) & 0x03) << 4)
            | (((f[12 + k] >> 4) & 0x03) << 6);
    }
    a
}

fn q3_k_scale_raw(variant: u8, slot: usize) -> u8 {
    match variant {
        1 => 32, // biased to exactly 0
        2 => {
            if slot % 2 == 0 {
                63
            } else {
                0
            }
        } // +31 / -32, the lattice extremes
        3 => 40, // identical everywhere
        4 => {
            if slot % 2 == 0 {
                0
            } else {
                63
            }
        }
        _ => (32 + (slot as i32 - 16)) as u8, // 16..47 -> biased -16..+15
    }
}

fn fx_q3_k(rows: usize, cols: usize, variant: u8) -> Fx {
    assert_eq!(cols % QK_K, 0);
    let mut fx = Fx::new(GGML_TYPE_Q3_K, rows, cols);
    let nb = cols / QK_K;
    for r in 0..rows {
        for blk in 0..nb {
            let d: f64 = match variant {
                1 => 0.0,
                2 => 65504.0,
                3 => 4.0,
                _ => 0.5 * (1 + (blk % 2)) as f64,
            };
            let off = r * fx.row_bytes + blk * 110;
            fx.bytes[off + 108..off + 110].copy_from_slice(&f16_bits_exact(d).to_le_bytes());
            let mut f = [0u8; 16];
            for slot in 0..16 {
                f[slot] = q3_k_scale_raw(variant, slot);
            }
            fx.bytes[off + 96..off + 108].copy_from_slice(&q3_k_scale_bytes(&f));

            for e in 0..256 {
                let half = e / 128;
                let rem = e % 128;
                let j = rem / 32;
                let part = (rem % 32) / 16;
                let l = rem % 16;
                let (q, negative) = match variant {
                    3 => (2u8, false),
                    4 => {
                        if (e + r) % 2 == 0 {
                            (3, false)
                        } else {
                            (0, true)
                        }
                    }
                    _ => (((e * 7 + r * 5 + blk * 3) % 4) as u8, (e / 4) % 2 == 0),
                };
                // qs lives AFTER the 32-byte hmask region of the block.
                let qs_idx = off + 32 + half * 32 + part * 16 + l;
                fx.bytes[qs_idx] |= q << (2 * j);
                if !negative {
                    // hmask bit SET: value = q (0..3)
                    fx.bytes[off + part * 16 + l] |= 1u8 << (half * 4 + j);
                }
                let slot = half * 8 + j * 2 + part;
                let v = q as i32 - if negative { 4 } else { 0 };
                fx.ref_w[r * cols + blk * 256 + e] = d * (f[slot] as i32 - 32) as f64 * v as f64;
            }
        }
    }
    fx
}

// ─────────────────────────────────────────────────────────────────────
// Per-type oracles. Each one gates the dequant AND the dot/matvec for
// every edge-case variant, and returns (worst dequant ulp, worst dot rel).
// ─────────────────────────────────────────────────────────────────────

/// One receipt line per type: the numbers this gate actually observed.
fn report(name: &str, worst_ulp: u32, worst_rel: f64) {
    eprintln!(
        "[quant-oracle] {name:5} dequant {worst_ulp} ulp of f32 (limit 1); dot {worst_rel:.2e} rel (limit 1e-5); every row <= 1e-6 of L1 — PASS"
    );
}

fn oracle_f32() -> (u32, f64) {
    let (mut ulp, mut rel) = (0u32, 0f64);
    for v in 0..5u8 {
        let fx = fx_f32(9, 512, v);
        let u = fx.check_row_dequant("f32", 0);
        assert_eq!(u, 0, "f32 dequant must be bit-exact");
        ulp = ulp.max(u);
        rel = rel.max(fx.check_dispatch("f32", 4));
    }
    (ulp, rel)
}

fn oracle_f16() -> (u32, f64) {
    let (mut ulp, mut rel) = (0u32, 0f64);
    for v in 0..5u8 {
        let fx = fx_f16(9, 512, v);
        // f16 -> f32 is exact for every binary16 value, so this gate is
        // bit-exact (0 ulp), not 1.
        let u = fx.check_row_dequant("f16", 0);
        assert_eq!(u, 0, "f16 dequant must be bit-exact");
        ulp = ulp.max(u);
        rel = rel.max(fx.check_dispatch("f16", 4));
    }
    (ulp, rel)
}
fn oracle_q4_0() -> (u32, f64) {
    let (mut ulp, mut rel) = (0u32, 0f64);
    for v in 0..6u8 {
        let fx = fx_q4_0(9, 512, v);
        ulp = ulp.max(fx.check_row_dequant("q4_0", 1));
        // The Rust f32 dot (an exact-algorithm kernel) on arbitrary
        // activations, row by row.
        let x = act_row(fx.cols, 0);
        for r in 0..fx.rows {
            let row = &fx.bytes[r * fx.row_bytes..(r + 1) * fx.row_bytes];
            let got = vec_dot_q4_0(row, &x);
            let (want, l1) = ref_dot(&fx.ref_w[r * fx.cols..(r + 1) * fx.cols], &x);
            rel = rel.max(check_dot(
                &format!("q4_0 vec_dot row {r} (variant {v})"),
                got,
                want,
                l1,
            ));
        }
        rel = rel.max(fx.check_dispatch("q4_0", 4));
    }
    (ulp, rel)
}

fn oracle_q8_0() -> (u32, f64) {
    let (mut ulp, mut rel) = (0u32, 0f64);
    for v in 0..6u8 {
        let fx = fx_q8_0(9, 512, v);
        ulp = ulp.max(fx.check_row_dequant("q8_0", 1));
        let x = act_row(fx.cols, 0);
        for r in 0..fx.rows {
            let row = &fx.bytes[r * fx.row_bytes..(r + 1) * fx.row_bytes];
            let got = vec_dot_q8_0(row, &x);
            let (want, l1) = ref_dot(&fx.ref_w[r * fx.cols..(r + 1) * fx.cols], &x);
            rel = rel.max(check_dot(
                &format!("q8_0 vec_dot row {r} (variant {v})"),
                got,
                want,
                l1,
            ));
        }
        rel = rel.max(fx.check_dispatch("q8_0", 4));
    }
    (ulp, rel)
}

fn oracle_q4_k() -> (u32, f64) {
    let (mut ulp, mut rel) = (0u32, 0f64);
    for v in 0..5u8 {
        let fx = fx_q4_k(9, 512, v);
        ulp = ulp.max(fx.check_row_dequant("q4_K", 1));
        let x = act_row(fx.cols, 0);
        let nb = fx.cols / QK_K;
        for r in 0..fx.rows {
            let ptr = fx.bytes[r * fx.row_bytes..].as_ptr();
            let got = vec_dot_raw_q4_k(ptr, x.as_ptr(), nb);
            let (want, l1) = ref_dot(&fx.ref_w[r * fx.cols..(r + 1) * fx.cols], &x);
            rel = rel.max(check_dot(
                &format!("q4_K vec_dot row {r} (variant {v})"),
                got,
                want,
                l1,
            ));
        }
        // Cold dispatch (per-row kernels) ...
        rel = rel.max(fx.check_dispatch("q4_K", 4));
        // ... and the packed-tile path prefill actually uses.
        let _guard = super::quant::tests::tile_test_guard();
        tile_warm(fx.bytes.as_ptr(), fx.rows, fx.cols, GGML_TYPE_Q4_K);
        rel = rel.max(fx.check_dispatch("q4_K tiled", 8));
    }
    (ulp, rel)
}

fn oracle_q6_k() -> (u32, f64) {
    let (mut ulp, mut rel) = (0u32, 0f64);
    for v in 0..5u8 {
        let fx = fx_q6_k(9, 512, v);
        ulp = ulp.max(fx.check_row_dequant("q6_K", 1));
        let x = act_row(fx.cols, 0);
        let nb = fx.cols / QK_K;
        for r in 0..fx.rows {
            let ptr = fx.bytes[r * fx.row_bytes..].as_ptr();
            let got = vec_dot_raw_q6_k(ptr, x.as_ptr(), nb);
            let (want, l1) = ref_dot(&fx.ref_w[r * fx.cols..(r + 1) * fx.cols], &x);
            rel = rel.max(check_dot(
                &format!("q6_K vec_dot row {r} (variant {v})"),
                got,
                want,
                l1,
            ));
        }
        rel = rel.max(fx.check_dispatch("q6_K", 4));
        let _guard = super::quant::tests::tile_test_guard();
        tile_warm(fx.bytes.as_ptr(), fx.rows, fx.cols, GGML_TYPE_Q6_K);
        rel = rel.max(fx.check_dispatch("q6_K tiled", 8));
    }
    (ulp, rel)
}

fn oracle_q3_k() -> (u32, f64) {
    let (mut ulp, mut rel) = (0u32, 0f64);
    for v in 0..5u8 {
        let fx = fx_q3_k(9, 512, v);
        ulp = ulp.max(fx.check_row_dequant("q3_K", 1));
        // There is no exported f32 vec_dot for q3_K: the only entry point is
        // the dispatch, whose aarch64 route is the q8_K + SDOT kernel. Integer
        // activations (max |x| == 127) make that quantisation lossless, so
        // this compares exact integer arithmetic against the f64 reference.
        rel = rel.max(fx.check_dispatch("q3_K", 4));
        // The warmed path repacks q3_K blocks into the q6_K tile layout and
        // runs the q6_K GEMM — an exact rewrite of the same values.
        let _guard = super::quant::tests::tile_test_guard();
        tile_warm(fx.bytes.as_ptr(), fx.rows, fx.cols, GGML_TYPE_Q3_K);
        rel = rel.max(fx.check_dispatch("q3_K tiled", 8));
    }
    (ulp, rel)
}

/// The type -> (dequant, dot) coverage registry. every_accepted... walks the
/// loader's accept-list and RUNS these, so a type can only be accepted once
/// an oracle here proves it.
const ORACLES: [(u32, &str, fn() -> (u32, f64)); 7] = [
    (GGML_TYPE_F32, "F32", oracle_f32),
    (GGML_TYPE_F16, "F16", oracle_f16),
    (GGML_TYPE_Q4_0, "Q4_0", oracle_q4_0),
    (GGML_TYPE_Q8_0, "Q8_0", oracle_q8_0),
    (GGML_TYPE_Q4_K, "Q4_K", oracle_q4_k),
    (GGML_TYPE_Q6_K, "Q6_K", oracle_q6_k),
    (GGML_TYPE_Q3_K, "Q3_K", oracle_q3_k),
];
// ─────────────────────────────────────────────────────────────────────
// The oracles, as tests.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn oracle_f32_dequant_and_dot() {
    let (u, r) = oracle_f32();
    report("F32", u, r);
}

#[test]
fn oracle_f16_dequant_and_dot() {
    let (u, r) = oracle_f16();
    report("F16", u, r);
}

#[test]
fn oracle_q4_0_dequant_and_dot() {
    let (u, r) = oracle_q4_0();
    report("Q4_0", u, r);
}

#[test]
fn oracle_q8_0_dequant_and_dot() {
    let (u, r) = oracle_q8_0();
    report("Q8_0", u, r);
}

#[test]
fn oracle_q4_k_dequant_and_dot() {
    let (u, r) = oracle_q4_k();
    report("Q4_K", u, r);
}

#[test]
fn oracle_q6_k_dequant_and_dot() {
    let (u, r) = oracle_q6_k();
    report("Q6_K", u, r);
}

#[test]
fn oracle_q3_k_dequant_and_dot() {
    let (u, r) = oracle_q3_k();
    report("Q3_K", u, r);
}

/// The f16 decode is both the F16 dequant and the scale decoder every
/// k-quant depends on, so it is checked against the exact value of all
/// 65536 binary16 patterns — not a sample.
#[test]
fn oracle_f16_conversion_is_exact_for_all_65536_patterns() {
    for bits in 0..=u16::MAX {
        let want = ref_f16(bits);
        let got = f16_to_f32(bits) as f64;
        if want.is_nan() {
            assert!(got.is_nan(), "0x{bits:04x}: kernel {got} is not NaN");
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "0x{bits:04x}: kernel {got} vs exact {want}"
            );
        }
    }
}

/// A scale encoder that disagrees with the kernel decoder would let every
/// other q3_K test pass while the real layout is wrong; this pins the
/// 12-byte -> 16-scale permutation against the forward formula written out
/// here separately (ggml dequantize_row_q3_K expand, as the kernel has it).
#[test]
fn oracle_q3_k_scale_permutation_round_trips() {
    let mut f = [0u8; 16];
    for i in 0..16 {
        f[i] = ((i as u32 * 37 + 11) % 64) as u8;
    }
    let a = q3_k_scale_bytes(&f);
    let word = |i: usize| u32::from_le_bytes([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]);
    let (a0, a1, t) = (word(0), word(1), word(2));
    let b = [
        (a0 & 0x0f0f_0f0f) | (((t >> 0) & 0x0303_0303) << 4),
        (a1 & 0x0f0f_0f0f) | (((t >> 2) & 0x0303_0303) << 4),
        ((a0 >> 4) & 0x0f0f_0f0f) | (((t >> 4) & 0x0303_0303) << 4),
        ((a1 >> 4) & 0x0f0f_0f0f) | (((t >> 6) & 0x0303_0303) << 4),
    ];
    for i in 0..4 {
        for k in 0..4 {
            assert_eq!((b[i] >> (8 * k)) as u8, f[i * 4 + k], "slot {}", i * 4 + k);
        }
    }
}

/// THE STRUCTURAL GATE. Every type the loader accepts must have an oracle
/// here that RUNS and passes; a type added to SUPPORTED_QUANT_TYPES without
/// one fails this test instead of shipping as "supported" on a constant.
// Gate this on the same platforms as the list it reads: model.rs:3420 carries
// a stray cfg(macos|ios) between a removed GPU fn doc block and
// SUPPORTED_QUANT_TYPES, so the const only exists on Apple targets.
// (Pre-existing; reported rather than "fixed" here, because the const and
// gate_quant_type disagreeing about it is a loader question, not a test one.)
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn every_accepted_quant_type_has_a_kernel_oracle() {
    for ty in TransformerModel::SUPPORTED_QUANT_TYPES {
        let entry = ORACLES.iter().find(|(t, _, _)| *t == ty);
        let (_, name, run) = entry.unwrap_or_else(|| {
            panic!(
                "type {ty} ({}) is in TransformerModel::SUPPORTED_QUANT_TYPES but has no kernel oracle in quant_oracle.rs — a type may only be accepted once its dequant and dot are proven against an independent reference",
                super::quant::ggml_type_name(ty)
            )
        });
        let (ulp, rel) = run();
        report(name, ulp, rel);
    }
    // ... and no oracle may exist for a type the loader would refuse, so the
    // two lists stay in step from both directions.
    for (ty, name, _) in ORACLES {
        assert!(
            TransformerModel::SUPPORTED_QUANT_TYPES.contains(&ty),
            "oracle for {name} ({ty}) exists but the loader does not accept that type",
        );
    }
}
