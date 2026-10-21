// crates/sofuu-core/src/infer/quant.rs — Quantization formats and SIMD GEMV kernels.
//
// Supports Q4_0, Q8_0, F16, and F32 quantized tensor types.

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q6_K: u32 = 14;
/// Prism-private ternary at group size 128 (ggml type 143).
pub const GGML_TYPE_PTQ1_0: u32 = 143;

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;
pub const QK_PTQ1_0: usize = 128;

/// Convert IEEE 754 half-precision (16-bit) float to 32-bit single precision.
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 113;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        return f32::from_bits((sign << 31) | (e << 23) | ((m & 0x3ff) << 13));
    } else if exp == 31 {
        return f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13));
    }
    let exp_f32 = exp + 127 - 15;
    let mant_f32 = mant << 13;
    f32::from_bits((sign << 31) | (exp_f32 << 23) | mant_f32)
}

/// Convert bfloat16 to 32-bit single precision IEEE 754 float.
#[inline(always)]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Convert f32 to f16 (for kv-cache / quantization).
#[inline(always)]
pub fn f32_to_f16(val: f32) -> u16 {
    let u = val.to_bits();
    let sign = ((u >> 31) & 1) as u16;
    let exp = ((u >> 23) & 0xff) as i32;
    let mant = (u & 0x7fffff) as u32;

    if exp == 255 {
        return (sign << 15) | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        let m = (mant | 0x800000) >> (1 - new_exp);
        return (sign << 15) | ((m >> 13) as u16);
    }
    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Calculate byte length for a 1D tensor row of length `n` given GGML tensor type.
pub fn type_row_size(type_id: u32, n: usize) -> usize {
    match type_id {
        GGML_TYPE_F32 => n * 4,
        GGML_TYPE_F16 => n * 2,
        GGML_TYPE_Q4_0 => (n / QK4_0) * 18,
        GGML_TYPE_Q8_0 => (n / QK8_0) * 34,
        GGML_TYPE_Q4_K => (n / QK_K) * 144,
        GGML_TYPE_Q6_K => (n / QK_K) * 210,
        _ => n * 4,
    }
}

/// Dequantize Q4_0 block row into destination f32 buffer.
pub fn dequantize_q4_0_row(src: &[u8], dst: &mut [f32]) {
    let n_blocks = dst.len() / QK4_0;
    for b in 0..n_blocks {
        let b_offset = b * 18;
        let d_raw = u16::from_le_bytes([src[b_offset], src[b_offset + 1]]);
        let d = f16_to_f32(d_raw);
        let qs = &src[b_offset + 2..b_offset + 18];
        let out_offset = b * QK4_0;

        for j in 0..16 {
            let byte = qs[j];
            let v0 = (byte & 0x0f) as i8 - 8;
            let v1 = (byte >> 4) as i8 - 8;
            dst[out_offset + j] = (v0 as f32) * d;
            dst[out_offset + j + 16] = (v1 as f32) * d;
        }
    }
}

/// Dequantize Q8_0 block row into destination f32 buffer.
pub fn dequantize_q8_0_row(src: &[u8], dst: &mut [f32]) {
    let n_blocks = dst.len() / QK8_0;
    for b in 0..n_blocks {
        let b_offset = b * 34;
        let d_raw = u16::from_le_bytes([src[b_offset], src[b_offset + 1]]);
        let d = f16_to_f32(d_raw);
        let qs = &src[b_offset + 2..b_offset + 34];
        let out_offset = b * QK8_0;

        for j in 0..QK8_0 {
            dst[out_offset + j] = (qs[j] as i8 as f32) * d;
        }
    }
}

/// Dequantize a PTQ1_0 (GGML type 143, Prism ternary at group 128) row into
/// destination f32 buffer. 28-byte block per 128 values: qs[24] (120 values,
/// 5 base-3 trits per byte) + qh[2] (8 values, 4 trits per byte) + f16 scale
/// `d` last. Same base-3 trit packing as upstream TQ1_0 with the 32-then-16
/// byte staging generalised to 32/16/8, so a 24-byte qs region walks chunks
/// of 16 then 8; each byte stores its trits pre-scaled (ceil q*256/243), and
/// the decoder re-extracts them via the modular trick `(byte * 3^n)` with u8
/// wrapping, then `(q * 3) >> 8`. `out.len()` must be a multiple of 128.
pub fn ptq1_0_dequant_row(src: &[u8], out: &mut [f32]) {
    const QS_LEN: usize = (QK_PTQ1_0 - 4 * QK_PTQ1_0 / 64) / 5; // 24
    const QH_LEN: usize = QK_PTQ1_0 / 64; // 2
    const BLOCK: usize = QS_LEN + QH_LEN + 2; // qs | qh | f16 scale = 28
    const STAGES: [usize; 3] = [32, 16, 8];
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

    let n_blocks = out.len() / QK_PTQ1_0;
    debug_assert!(src.len() >= n_blocks * BLOCK);

    for blk in 0..n_blocks {
        let b_offset = blk * BLOCK;
        let d = f16_to_f32(u16::from_le_bytes([
            src[b_offset + QS_LEN + QH_LEN],
            src[b_offset + QS_LEN + QH_LEN + 1],
        ]));
        let mut o = blk * QK_PTQ1_0;

        let mut j = 0;
        for s in 0..STAGES.len() {
            let c = STAGES[s];
            while j + c <= QS_LEN {
                for n in 0..5usize {
                    for m in 0..c {
                        let q = src[b_offset + j + m].wrapping_mul(POW3[n]);
                        let xi = ((q as u16) * 3) >> 8;
                        out[o] = (xi as i16 - 1) as f32 * d;
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
                out[o] = (xi as i16 - 1) as f32 * d;
                o += 1;
            }
        }
        debug_assert_eq!(o, (blk + 1) * QK_PTQ1_0);
    }
}

/// Dot-product of Q4_0 quantized weight row with float32 activation vector.
#[inline]
pub fn vec_dot_q4_0(weight_bytes: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / QK4_0;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let b_offset = b * 18;
        let d_raw = u16::from_le_bytes([weight_bytes[b_offset], weight_bytes[b_offset + 1]]);
        let d = f16_to_f32(d_raw);
        let qs = &weight_bytes[b_offset + 2..b_offset + 18];
        let x_block = &x[b * QK4_0..(b + 1) * QK4_0];

        let mut block_acc = 0.0f32;
        for j in 0..16 {
            let byte = qs[j];
            let v0 = (byte & 0x0f) as i8 - 8;
            let v1 = (byte >> 4) as i8 - 8;
            block_acc += (v0 as f32) * x_block[j] + (v1 as f32) * x_block[j + 16];
        }
        sum += block_acc * d;
    }
    sum
}

/// Dot-product of Q8_0 quantized weight row with float32 activation vector.
#[inline]
pub fn vec_dot_q8_0(weight_bytes: &[u8], x: &[f32]) -> f32 {
    let n_blocks = x.len() / QK8_0;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let b_offset = b * 34;
        let d_raw = u16::from_le_bytes([weight_bytes[b_offset], weight_bytes[b_offset + 1]]);
        let d = f16_to_f32(d_raw);
        let qs = &weight_bytes[b_offset + 2..b_offset + 34];
        let x_block = &x[b * QK8_0..(b + 1) * QK8_0];

        let mut block_acc = 0.0f32;
        for j in 0..QK8_0 {
            block_acc += (qs[j] as i8 as f32) * x_block[j];
        }
        sum += block_acc * d;
    }
    sum
}

/// Dequantize Q4_K block row into destination f32 buffer.
pub fn dequantize_q4_k_row(src: &[u8], dst: &mut [f32]) {
    assert_eq!(dst.len() % QK_K, 0);
    unsafe {
        sofuu_dequantize_row_q4_k(src.as_ptr(), dst.as_mut_ptr(), dst.len());
    }
}

/// Dequantize Q6_K block row into destination f32 buffer.
pub fn dequantize_q6_k_row(src: &[u8], dst: &mut [f32]) {
    assert_eq!(dst.len() % QK_K, 0);
    unsafe {
        sofuu_dequantize_row_q6_k(src.as_ptr(), dst.as_mut_ptr(), dst.len());
    }
}

/// Configure the number of CPU threads used for matrix-vector compute.
/// If `n == 0`, defaults to all available CPU cores.
/// Warm the packed-tile GEMM/GEMV cache for one weight matrix (q4_K/q6_K/q8_0).
/// Subsequent matvec/matmat calls on this weight pointer route to the tile
/// kernels. `ty` is the GGML type id (8 = q8_0, 12 = q4_K, 14 = q6_K).
pub fn tile_warm(ptr: *const u8, rows: usize, cols: usize, ty: u32) {
    unsafe {
        sofuu_tile_warm(ptr, rows, cols, ty);
    }
}

/// Force the q4_K decoded-scale sidecar off (`on = true`) or back on
/// (`on = false`) for the next tile warm. Tests use this to drive the
/// original kernels and the sidecar kernels on identical tiles. The
/// sidecar is opt-in at runtime via `SOFUU_SCALE_SIDECAR=1`: on Pi 5 the
/// prefill GEMM is unchanged by it (scale decode overlaps the NEON wall)
/// while its extra bytes slow DRAM-bound GEMV decode, so it ships disabled.
pub fn tile_scale_sidecar_disable(on: bool) {
    unsafe {
        sofuu_tile_scale_sidecar_mode(on as i32);
    }
}

/// Device GEMM micro-bench (SOFUU_BENCH_GEMM diagnostics): splits the q4_K /
/// q8_0 tile prefill cost between the activation-quantize prologue and the
/// kernel, printing GMAC/s to stderr.
pub fn gemm_bench(ty: u32, rows: usize, cols: usize, batch: usize, iters: usize) {
    unsafe {
        sofuu_bench_gemm(ty, rows as i32, cols as i32, batch as i32, iters as i32);
    }
}

/// Device micro-bench for the serial decode GEMV path (kernel + prologue):
/// prints GB/s of weight bytes streamed per call, the decode bandwidth metric.
pub fn gemv_bench(ty: u32, rows: usize, cols: usize, iters: usize) {
    unsafe {
        sofuu_bench_gemv(ty, rows as i32, cols as i32, iters as i32);
    }
}

/// Device micro-bench for the q4_K nibble-plane prefill kernel vs the packed
/// one (kernel-only, identical synthetic tensors); prints both GMAC/s rates.
pub fn gemm_bench_plane(rows: usize, cols: usize, batch: usize, iters: usize) {
    unsafe {
        sofuu_bench_gemm_plane(rows as i32, cols as i32, batch as i32, iters as i32);
    }
}

/// Bit-parity gate for the q4_K nibble-plane GEMM vs the packed kernel.
/// Returns 0 when outputs are bit-identical.
pub fn gemm_plane_parity(rows: usize, cols: usize, batch: usize) -> i32 {
    unsafe { sofuu_test_gemm_q4_k_plane_parity(rows as i32, cols as i32, batch as i32) }
}

/// Bit-parity gate for the T1 pre-decoded-scale tiles (gemm_s) vs the
/// stock packed-tile kernel. Returns 0 when outputs are bit-identical.
pub fn gemm_q4_s_parity(rows: usize, cols: usize, batch: usize) -> i32 {
    unsafe { sofuu_test_q4_k_s_bit_exact(rows as i32, cols as i32, batch as i32) }
}

pub fn set_thread_count(n: usize) {
    unsafe {
        sofuu_set_threads(n as i32);
    }
}

/// Get the current number of CPU threads used for matrix-vector compute.
pub fn get_thread_count() -> usize {
    unsafe {
        sofuu_get_threads() as usize
    }
}

/// Non-temporal streaming prefetch for single-use weight buffers.
///
/// On ARM64 (aarch64), executes `prfm pldl1strm` (Prefetch Load L1 Streaming),
/// which instructs the memory controller and hardware prefetcher to stream data
/// without allocating or evicting long-lived cache lines in L2/L3 unified caches.
/// On x86_64, executes `_mm_prefetch` with `_MM_HINT_NTA` (Non-Temporal Access hint).
///
/// Limits prefetch length to `bytes.min(64 * 1024)` to avoid saturating hardware
/// Miss Status Holding Registers (MSHRs).
#[inline]
pub fn prefetch_streaming_bytes(ptr: *const u8, bytes: usize) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let prefetch_len = bytes.min(64 * 1024);
    const STEP: usize = 64;
    let mut offset = 0;
    while offset < prefetch_len {
        let p = unsafe { ptr.add(offset) };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "prfm pldl1strm, [{x}]",
                x = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_NTA);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = p;
        }
        offset += STEP;
    }
}

/// Temporal prefetch for recurrently accessed data (KV Cache and working activation buffers).
///
/// On ARM64 (aarch64), executes `prfm pldl1keep` (Prefetch Load L1 Retain / Keep),
/// pinning the cache lines in L1/L2 caches with high temporal locality.
/// On x86_64, executes `_mm_prefetch` with `_MM_HINT_T0` (Temporal locality in all cache levels).
#[inline]
pub fn prefetch_temporal_bytes(ptr: *const u8, bytes: usize) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let prefetch_len = bytes.min(64 * 1024);
    const STEP: usize = 64;
    let mut offset = 0;
    while offset < prefetch_len {
        let p = unsafe { ptr.add(offset) };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{x}]",
                x = in(reg) p,
                options(nostack, preserves_flags, readonly)
            );
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = p;
        }
        offset += STEP;
    }
}

/// Non-temporal streaming prefetch for a byte slice.
#[inline(always)]
pub fn prefetch_streaming_slice(slice: &[u8]) {
    prefetch_streaming_bytes(slice.as_ptr(), slice.len());
}

/// Temporal prefetch for a typed slice (e.g. KV Cache or f32 activations).
#[inline(always)]
pub fn prefetch_temporal_slice<T>(slice: &[T]) {
    prefetch_temporal_bytes(slice.as_ptr() as *const u8, slice.len() * std::mem::size_of::<T>());
}


#[inline(always)]
pub fn vec_dot_raw_q4_0(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32 {
    unsafe { sofuu_vec_dot_q4_0(weight_bytes, x, n_blocks) }
}

#[inline(always)]
pub fn vec_dot_raw_q8_0(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32 {
    unsafe { sofuu_vec_dot_q8_0(weight_bytes, x, n_blocks) }
}

#[inline(always)]
pub fn vec_dot_raw_q4_k(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32 {
    unsafe { sofuu_vec_dot_q4_k(weight_bytes, x, n_blocks) }
}

#[inline(always)]
pub fn vec_dot_raw_q6_k(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32 {
    unsafe { sofuu_vec_dot_q6_k(weight_bytes, x, n_blocks) }
}

#[inline(always)]
pub fn vec_dot_raw_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
    unsafe { sofuu_vec_dot_f32(a, b, n) }
}

extern "C" {
    fn sofuu_set_threads(n: i32);
    fn sofuu_get_threads() -> i32;
    fn sofuu_tile_warm(w: *const u8, rows: usize, cols: usize, ty: u32) -> i32;
    fn sofuu_tile_scale_sidecar_mode(disable: i32);
    fn sofuu_bench_gemm(ty: u32, rows: i32, cols: i32, batch: i32, iters: i32);
    fn sofuu_bench_gemv(ty: u32, rows: i32, cols: i32, iters: i32);
    fn sofuu_bench_gemm_plane(rows: i32, cols: i32, batch: i32, iters: i32);
    fn sofuu_test_gemm_q4_k_plane_parity(rows: i32, cols: i32, batch: i32) -> i32;
    fn sofuu_test_q4_k_s_bit_exact(rows: i32, cols: i32, batch: i32) -> i32;
    // (sofuu_dequantize_row_q4_0 / _q8_0 are called from C++ only —
    // infer_kernels.cpp:2159/2329 — no Rust extern decl needed.)
    fn sofuu_dequantize_row_q4_k(src: *const u8, dst: *mut f32, k: usize);
    fn sofuu_dequantize_row_q6_k(src: *const u8, dst: *mut f32, k: usize);
    fn sofuu_vec_dot_q4_0(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32;
    fn sofuu_vec_dot_q8_0(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32;
    fn sofuu_vec_dot_q4_k(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32;
    fn sofuu_vec_dot_q6_k(weight_bytes: *const u8, x: *const f32, n_blocks: usize) -> f32;
    fn sofuu_gemv_q4_0(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize);
    fn sofuu_gemv_q8_0(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize);
    fn sofuu_gemv_q4_k(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize);
    fn sofuu_gemv_q6_k(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize);
    pub(crate) fn sofuu_gemv_f32(weight_bytes: *const f32, x: *const f32, out: *mut f32, rows: usize, cols: usize);
    pub(crate) fn sofuu_gemv_f16(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize);
    pub(crate) fn sofuu_gemv_bf16(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize);
    fn sofuu_gemv_mlx4_multi(
        w_bytes: *const *const u8,
        scales_bytes: *const *const u8,
        biases_bytes: *const *const u8,
        x: *const f32,
        outs: *const *mut f32,
        rows: *const usize,
        num_mats: usize,
        cols: usize,
        group_size: usize,
        is_bf16: i32,
    );
    fn sofuu_gemv_mlx4(
        w_bytes: *const u8,
        scales_bytes: *const u8,
        biases_bytes: *const u8,
        x: *const f32,
        out: *mut f32,
        rows: usize,
        cols: usize,
        group_size: usize,
        is_bf16: i32,
    );
    fn sofuu_gemm_q4_0(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_gemm_q8_0(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_gemm_q4_k(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_gemm_q6_k(weight_bytes: *const u8, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_gemm_q8_0_acc(weight_bytes: *const u8, x: *const f32, out_acc: *mut f32, rows: usize, cols: usize, batch_size: usize) -> i32;
    fn sofuu_gemm_q4_k_acc(weight_bytes: *const u8, x: *const f32, out_acc: *mut f32, rows: usize, cols: usize, batch_size: usize) -> i32;
    fn sofuu_gemm_q6_k_acc(weight_bytes: *const u8, x: *const f32, out_acc: *mut f32, rows: usize, cols: usize, batch_size: usize) -> i32;
    fn sofuu_gemm_q4_k_pair(w1: *const u8, w2: *const u8, x: *const f32, out1: *mut f32, out2: *mut f32, rows: usize, cols: usize, batch_size: usize) -> i32;
    fn sofuu_gemm_q8_0_pair(w1: *const u8, w2: *const u8, x: *const f32, out1: *mut f32, out2: *mut f32, rows: usize, cols: usize, batch_size: usize) -> i32;
    fn sofuu_gemm_f32(weight_bytes: *const f32, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_gemm_f16(weight_bytes: *const u16, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_rmsnorm(out: *mut f32, x: *const f32, weight: *const f32, n: usize, eps: f32);
    fn sofuu_batched_rmsnorm(out: *mut f32, x: *const f32, weight: *const f32, rows: usize, cols: usize, eps: f32);
    fn sofuu_batched_mad(dst: *mut f32, src: *const f32, alpha: f32, rows: usize, cols: usize);
    fn sofuu_batched_add_bias(x: *mut f32, bias: *const f32, rows: usize, cols: usize);
    fn sofuu_batched_swiglu(gate: *mut f32, up: *const f32, rows: usize, cols: usize);
    fn sofuu_batched_rope_kv(
        q_batch: *mut f32, k_batch: *mut f32, v_batch: *const f32,
        k_cache: *mut f32, v_cache: *mut f32,
        b_size: usize, start_pos: usize, layer_idx: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq_len: usize, freq_base: f32, rope_mode: i32,
    );
    fn sofuu_kv16_convert(src: *const f32, dst: *mut u16, n: usize);
    fn sofuu_batched_rope_kv_f16(
        q_batch: *mut f32, k_batch: *mut f32, v_batch: *const f32,
        k_cache: *mut f32, v_cache: *mut f32,
        k16_cache: *mut u16, v16_cache: *mut u16,
        b_size: usize, start_pos: usize, layer_idx: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq_len: usize, freq_base: f32, rope_mode: i32,
    );
    fn sofuu_swiglu(out: *mut f32, gate: *const f32, up: *const f32, n: usize);
    fn sofuu_softmax(scores: *mut f32, n: usize);
    fn sofuu_rope(vec: *mut f32, n_heads: usize, head_dim: usize, pos: usize, freq_base: f32, rope_style: i32);
    fn sofuu_rope_shift(vec: *mut f32, n_heads: usize, head_dim: usize, delta_pos: i64, freq_base: f32, rope_style: i32);
    fn sofuu_vec_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    fn sofuu_vec_mad_f32(dst: *mut f32, src: *const f32, scalar: f32, n: usize);
    fn sofuu_kv_compact(
        k_cache: *mut f32,
        v_cache: *mut f32,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        current_seq_len: usize,
        sink_tokens: usize,
        drop_tokens: usize,
        freq_base: f32,
        rope_style: i32,
    );
    fn sofuu_batched_attention(
        q_batch: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        head_out_batch: *mut f32,
        b_size: usize,
        start_pos: usize,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    );
    fn sofuu_batched_attention_ext(
        q_batch: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        head_out_batch: *mut f32,
        b_size: usize,
        start_pos: usize,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        sliding_window: usize,
        sink_tokens: usize,
    );
    fn sofuu_batched_attention_ext_k16(
        q_batch: *const f32,
        k_cache: *const u16,
        v_cache: *const u16,
        head_out_batch: *mut f32,
        b_size: usize,
        start_pos: usize,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        sliding_window: usize,
        sink_tokens: usize,
    );
    fn sofuu_test_attention_x4_parity(n_heads: i32, n_kv_heads: i32, head_dim: i32, max_seq: i32) -> i32;
    fn sofuu_test_softmax_vec_parity() -> i32;
    fn sofuu_test_attention_fa2_parity(n_heads: i32, n_kv_heads: i32, head_dim: i32, max_seq: i32) -> i32;
    fn sofuu_test_rope_table_parity(n_heads: i32, head_dim: i32, max_pos: i32) -> i32;
    fn sofuu_tree_rope(
        q_batch: *mut f32,
        k_batch: *mut f32,
        node_positions: *const usize,
        tree_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        freq_base: f32,
        rope_style: i32,
    );
    fn sofuu_tree_attention(
        q_batch: *const f32,
        k_batch: *const f32,
        v_batch: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        head_out_batch: *mut f32,
        ancestry_mask: *const u8,
        tree_size: usize,
        start_pos: usize,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        sliding_window: usize,
        sink_tokens: usize,
    );

    // Apple Silicon Metal Unified GPU Engine
    pub fn sofuu_metal_available() -> bool;
    pub fn sofuu_metal_init() -> *mut std::ffi::c_void;
    pub fn sofuu_metal_free(ctx: *mut std::ffi::c_void);
    pub fn sofuu_metal_buffer_create_zerocopy(
        ctx: *mut std::ffi::c_void,
        ptr: *const std::ffi::c_void,
        size_bytes: usize,
    ) -> *mut std::ffi::c_void;
    pub fn sofuu_metal_buffer_create_copy(
        ctx: *mut std::ffi::c_void,
        ptr: *const std::ffi::c_void,
        size_bytes: usize,
    ) -> *mut std::ffi::c_void;
    pub fn sofuu_metal_buffer_free(buf: *mut std::ffi::c_void);
    pub fn sofuu_metal_gemv_bf16(
        ctx: *mut std::ffi::c_void,
        w_buf: *mut std::ffi::c_void,
        w_offset: usize,
        x: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
    );
    pub fn sofuu_metal_gemv_qkv_bf16(
        ctx: *mut std::ffi::c_void,
        w_buf: *mut std::ffi::c_void,
        q_offset: usize,
        k_offset: usize,
        v_offset: usize,
        x: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        v_out: *mut f32,
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
        cols: usize,
    );
    pub fn sofuu_metal_gemv_gate_up_bf16(
        ctx: *mut std::ffi::c_void,
        w_buf: *mut std::ffi::c_void,
        gate_offset: usize,
        up_offset: usize,
        x: *const f32,
        gate_out: *mut f32,
        up_out: *mut f32,
        inter_rows: usize,
        cols: usize,
    );
    pub fn sofuu_metal_gemv_f16(
        ctx: *mut std::ffi::c_void,
        w_buf: *mut std::ffi::c_void,
        w_offset: usize,
        x: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
    );
    pub fn sofuu_metal_gemv_f32(
        ctx: *mut std::ffi::c_void,
        w_buf: *mut std::ffi::c_void,
        w_offset: usize,
        x: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
    );
    pub fn sofuu_metal_rmsnorm(
        ctx: *mut std::ffi::c_void,
        x: *const f32,
        w_buf: *mut std::ffi::c_void,
        w_offset: usize,
        y: *mut f32,
        dim: usize,
        eps: f32,
    );
    pub fn sofuu_metal_swiglu(
        ctx: *mut std::ffi::c_void,
        g: *mut f32,
        u: *const f32,
        size: usize,
    );
    pub fn sofuu_metal_conv1d_3(
        ctx: *mut std::ffi::c_void,
        conv_w_buf: *mut std::ffi::c_void,
        conv_w_offset: usize,
        state: *const f32,
        out: *mut f32,
        channels: usize,
    );
    pub fn sofuu_metal_gemv_mlx4(
        ctx: *mut std::ffi::c_void,
        w_buf: *mut std::ffi::c_void,
        w_offset: usize,
        scales_buf: *mut std::ffi::c_void,
        scales_offset: usize,
        biases_buf: *mut std::ffi::c_void,
        biases_offset: usize,
        x: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
        group_size: usize,
        is_bf16: i32,
    );
    pub fn sofuu_metal_buffer_create(
        ctx: *mut std::ffi::c_void,
        ptr: *const std::ffi::c_void,
        size_bytes: usize,
    ) -> *mut std::ffi::c_void;
    pub fn sofuu_metal_gemv_qkv_mlx4(
        ctx: *mut std::ffi::c_void,
        q_buf: *mut std::ffi::c_void,
        q_offset: usize,
        q_rows: usize,
        q_scales_buf: *mut std::ffi::c_void,
        q_scales_offset: usize,
        q_biases_buf: *mut std::ffi::c_void,
        q_biases_offset: usize,
        k_buf: *mut std::ffi::c_void,
        k_offset: usize,
        k_rows: usize,
        k_scales_buf: *mut std::ffi::c_void,
        k_scales_offset: usize,
        k_biases_buf: *mut std::ffi::c_void,
        k_biases_offset: usize,
        v_buf: *mut std::ffi::c_void,
        v_offset: usize,
        v_rows: usize,
        v_scales_buf: *mut std::ffi::c_void,
        v_scales_offset: usize,
        v_biases_buf: *mut std::ffi::c_void,
        v_biases_offset: usize,
        x: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        v_out: *mut f32,
        cols: usize,
        group_size: usize,
        is_bf16: i32,
    );
    pub fn sofuu_metal_gemv_gate_up_mlx4(
        ctx: *mut std::ffi::c_void,
        gate_buf: *mut std::ffi::c_void,
        gate_offset: usize,
        gate_rows: usize,
        gate_scales_buf: *mut std::ffi::c_void,
        gate_scales_offset: usize,
        gate_biases_buf: *mut std::ffi::c_void,
        gate_biases_offset: usize,
        up_buf: *mut std::ffi::c_void,
        up_offset: usize,
        up_rows: usize,
        up_scales_buf: *mut std::ffi::c_void,
        up_scales_offset: usize,
        up_biases_buf: *mut std::ffi::c_void,
        up_biases_offset: usize,
        x: *const f32,
        gate_out: *mut f32,
        up_out: *mut f32,
        cols: usize,
        group_size: usize,
        is_bf16: i32,
    );
    pub fn sofuu_metal_gemv_qkv_multi(
        ctx: *mut std::ffi::c_void,
        q_buf: *mut std::ffi::c_void,
        q_offset: usize,
        q_rows: usize,
        k_buf: *mut std::ffi::c_void,
        k_offset: usize,
        k_rows: usize,
        v_buf: *mut std::ffi::c_void,
        v_offset: usize,
        v_rows: usize,
        x: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        v_out: *mut f32,
        cols: usize,
        dtype: std::os::raw::c_int,
    );
    pub fn sofuu_metal_gemv_gate_up_multi(
        ctx: *mut std::ffi::c_void,
        gate_buf: *mut std::ffi::c_void,
        gate_offset: usize,
        up_buf: *mut std::ffi::c_void,
        up_offset: usize,
        x: *const f32,
        gate_out: *mut f32,
        up_out: *mut f32,
        inter_rows: usize,
        cols: usize,
        dtype: std::os::raw::c_int,
    );
    pub fn sofuu_metal_gemv_async(
        ctx: *mut std::ffi::c_void,
        w_buf: *mut std::ffi::c_void,
        w_offset: usize,
        x: *const f32,
        y: *mut f32,
        rows: usize,
        cols: usize,
        dtype: std::os::raw::c_int,
    );
    pub fn sofuu_metal_sync(ctx: *mut std::ffi::c_void);
}

// Test-gate harnesses live under cfg(test) so release builds don't carry
// unused-FFI declarations (the C++ symbols are always compiled; the gate
// tests link them from here alone).
#[cfg(test)]
extern "C" {
    fn sofuu_test_quant_q8_K_parity(cols: i32, iters: i32) -> i32;
    fn sofuu_test_gemm_resid_pair_parity(rows: i32, cols: i32, batch: i32) -> i32;
}
/// Shift RoPE position of key vectors by `delta_pos` (negative shifts left).
pub fn rope_shift(vec: &mut [f32], n_heads: usize, head_dim: usize, delta_pos: i64, freq_base: f32) {
    rope_shift_style(vec, n_heads, head_dim, delta_pos, freq_base, 0);
}

pub fn rope_shift_style(
    vec: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    delta_pos: i64,
    freq_base: f32,
    rope_style: i32,
) {
    unsafe {
        sofuu_rope_shift(vec.as_mut_ptr(), n_heads, head_dim, delta_pos, freq_base, rope_style);
    }
}

/// In-place compact the KV cache across all layers.
pub fn kv_compact(
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    current_seq_len: usize,
    sink_tokens: usize,
    drop_tokens: usize,
    freq_base: f32,
) {
    kv_compact_style(
        k_cache,
        v_cache,
        n_layers,
        n_kv_heads,
        head_dim,
        max_seq_len,
        current_seq_len,
        sink_tokens,
        drop_tokens,
        freq_base,
        0,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn kv_compact_style(
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    current_seq_len: usize,
    sink_tokens: usize,
    drop_tokens: usize,
    freq_base: f32,
    rope_style: i32,
) {
    unsafe {
        sofuu_kv_compact(
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            current_seq_len,
            sink_tokens,
            drop_tokens,
            freq_base,
            rope_style,
        );
    }
}

/// Multi-core parallel causal self-attention across all heads and tokens.
pub fn batched_attention(
    q_batch: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_out_batch: &mut [f32],
    b_size: usize,
    start_pos: usize,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) {
    unsafe {
        sofuu_batched_attention(
            q_batch.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            head_out_batch.as_mut_ptr(),
            b_size,
            start_pos,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq_len,
        );
    }
}

/// Multi-core parallel causal self-attention with Sliding Window Attention (SWA) and Attention Sinks.
/// Profiling hooks (enabled with `SOFUU_PROFILE=1`): split prefill wall time
/// into gemm / gemv / attention buckets; `report` prints and clears them.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub static GEMM_NS: AtomicU64 = AtomicU64::new(0);
    pub static GEMM_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GEMV_NS: AtomicU64 = AtomicU64::new(0);
    pub static GEMV_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static NORM_NS: AtomicU64 = AtomicU64::new(0);
    pub static NORM_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static ROPE_NS: AtomicU64 = AtomicU64::new(0);
    pub static ROPE_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static EMBED_NS: AtomicU64 = AtomicU64::new(0);
    pub static EMBED_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GLUE_NS: AtomicU64 = AtomicU64::new(0);
    pub static GLUE_CALLS: AtomicU64 = AtomicU64::new(0);
    static CHUNK_NS: AtomicU64 = AtomicU64::new(0);
    static CHUNK_TOK: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("SOFUU_PROFILE").is_some())
    }

    /// RAII bucket: accumulates wall time (and one call) while on the stack.
    pub struct Scope(&'static AtomicU64, &'static AtomicU64, Option<Instant>);
    impl Scope {
        pub fn new(ns: &'static AtomicU64, calls: &'static AtomicU64) -> Self {
            Scope(ns, calls, enabled().then(Instant::now))
        }
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            if let Some(t0) = self.2.take() {
                self.0.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                self.1.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record a prefill chunk's total time and token count.
    pub fn record_chunk(elapsed: std::time::Duration, tokens: usize) {
        if !enabled() {
            return;
        }
        CHUNK_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        CHUNK_TOK.fetch_add(tokens as u64, Ordering::Relaxed);
    }

    /// One generated-token step, recorded on drop so every early-return and
    /// break path in the decode loop is covered. Decode steps share the
    /// prefill buckets; the label in `report()` says "prefilled" either way.
    pub struct TokenScope(Option<Instant>);
    impl TokenScope {
        pub fn new() -> Self {
            TokenScope(enabled().then(Instant::now))
        }
    }
    impl Default for TokenScope {
        fn default() -> Self {
            Self::new()
        }
    }
    impl Drop for TokenScope {
        fn drop(&mut self) {
            if let Some(t0) = self.0.take() {
                record_chunk(t0.elapsed(), 1);
            }
        }
    }

    /// Print accumulated profile buckets and reset them.
    pub fn report() {
        if !enabled() {
            return;
        }
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
        let chunk = ms(&CHUNK_NS);
        let gemm = ms(&GEMM_NS);
        let gemv = ms(&GEMV_NS);
        let attn = ms(&ATTN_NS);
        let tok = CHUNK_TOK.load(Ordering::Relaxed);
        let norm = ms(&NORM_NS);
        let rope = ms(&ROPE_NS);
        let embed = ms(&EMBED_NS);
        let glue = ms(&GLUE_NS);
        let other = chunk - gemm - gemv - attn - norm - rope - embed - glue;
        eprintln!(
            "[prof] prefilled {tok} tok in {chunk:.0}ms | gemm {gemm:.0}ms ({}) | gemv {gemv:.0}ms ({}) | attn {attn:.0}ms ({}) | norm {norm:.0}ms({}) rope {rope:.0}ms({}) embed {embed:.0}ms({}) glue {glue:.0}ms({}) | other {other:.0}ms",
            GEMM_CALLS.load(Ordering::Relaxed),
            GEMV_CALLS.load(Ordering::Relaxed),
            ATTN_CALLS.load(Ordering::Relaxed),
            NORM_CALLS.load(Ordering::Relaxed),
            ROPE_CALLS.load(Ordering::Relaxed),
            EMBED_CALLS.load(Ordering::Relaxed),
            GLUE_CALLS.load(Ordering::Relaxed),
        );
        for a in [
            &GEMM_NS, &GEMM_CALLS, &GEMV_NS, &GEMV_CALLS, &ATTN_NS, &ATTN_CALLS,
            &NORM_NS, &NORM_CALLS, &ROPE_NS, &ROPE_CALLS, &EMBED_NS, &EMBED_CALLS,
            &GLUE_NS, &GLUE_CALLS, &CHUNK_NS, &CHUNK_TOK,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }
}

pub fn batched_attention_ext(
    q_batch: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_out_batch: &mut [f32],
    b_size: usize,
    start_pos: usize,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
        sliding_window: usize,
        sink_tokens: usize,
    ) {
        let _prof = prof::Scope::new(&prof::ATTN_NS, &prof::ATTN_CALLS);
        unsafe {
            sofuu_batched_attention_ext(
                q_batch.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                head_out_batch.as_mut_ptr(),
                b_size,
                start_pos,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq_len,
                sliding_window,
                sink_tokens,
            );
        }
    }

    /// f16-KV twin of `batched_attention_ext`: identical work split, halves
    /// the KV bytes the score/AV passes stream. Numerically equals the f32
    /// path up to f16 storage rounding (llama.cpp's default KV precision).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_attention_ext_k16(
        q_batch: &[f32],
        k16_cache: &[u16],
        v16_cache: &[u16],
        head_out_batch: &mut [f32],
        b_size: usize,
        start_pos: usize,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        sliding_window: usize,
        sink_tokens: usize,
    ) {
        let _prof = prof::Scope::new(&prof::ATTN_NS, &prof::ATTN_CALLS);
        unsafe {
            sofuu_batched_attention_ext_k16(
                q_batch.as_ptr(),
                k16_cache.as_ptr(),
                v16_cache.as_ptr(),
                head_out_batch.as_mut_ptr(),
                b_size,
                start_pos,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq_len,
                sliding_window,
                sink_tokens,
            );
        }
    }

    /// Bit-parity gate for the x4-position attention inner loops vs the
    /// original per-position dot/mad path. Returns 0 when outputs match.
    pub fn attention_x4_parity(n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize) -> i32 {
        unsafe { sofuu_test_attention_x4_parity(n_heads as i32, n_kv_heads as i32, head_dim as i32, max_seq as i32) }
    }

    /// Numerical gate for the NEON softmax (magic-rounded poly exp) vs the
    /// double-precision scalar reference: max rel ≤ 5e-7, sum ≈ 1.
    pub fn softmax_vec_parity() -> i32 {
        unsafe { sofuu_test_softmax_vec_parity() }
    }

    /// Numerical gate for the FA2 tiled online-softmax attention vs the
    /// two-pass reference (not bit-exact: block max/exp reassociates float
    /// rounding; the C harness enforces rel element error ≤ 1e-4). Returns 0
    /// when within tolerance.
    pub fn attention_fa2_parity(n_heads: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize) -> i32 {
        unsafe { sofuu_test_attention_fa2_parity(n_heads as i32, n_kv_heads as i32, head_dim as i32, max_seq as i32) }
    }

    /// Bit-parity gate for the RoPE sincos table vs the inline
    /// powf/cosf/sinf computation. Returns 0 when outputs match.
    pub fn rope_table_parity(n_heads: usize, head_dim: usize, max_pos: usize) -> i32 {
        unsafe { sofuu_test_rope_table_parity(n_heads as i32, head_dim as i32, max_pos as i32) }
    }

/// Multi-path speculative tree attention across historical KV cache plus candidate tree nodes.
#[allow(clippy::too_many_arguments)]
pub fn tree_attention(
    q_batch: &[f32],
    k_batch: &[f32],
    v_batch: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_out_batch: &mut [f32],
    ancestry_mask: &[u8],
    tree_size: usize,
    start_pos: usize,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    sliding_window: usize,
    sink_tokens: usize,
) {
    if tree_size == 0 {
        return;
    }
    let _prof = prof::Scope::new(&prof::ATTN_NS, &prof::ATTN_CALLS);
    unsafe {
        sofuu_tree_attention(
            q_batch.as_ptr(),
            k_batch.as_ptr(),
            v_batch.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            head_out_batch.as_mut_ptr(),
            ancestry_mask.as_ptr(),
            tree_size,
            start_pos,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq_len,
            sliding_window,
            sink_tokens,
        );
    }
}

/// Compute SIMD dot product of two f32 slices.
#[inline(always)]
pub fn vec_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice lengths must match for dot product");
    unsafe {
        sofuu_vec_dot_f32(a.as_ptr(), b.as_ptr(), a.len())
    }
}

/// In-place multiply-accumulate: dst[i] += src[i] * scalar via hardware SIMD.
#[inline(always)]
pub fn vec_mad_f32(dst: &mut [f32], src: &[f32], scalar: f32) {
    assert_eq!(dst.len(), src.len(), "slice lengths must match for vec_mad");
    unsafe {
        sofuu_vec_mad_f32(dst.as_mut_ptr(), src.as_ptr(), scalar, dst.len())
    }
}

/// Multi-threaded hardware SIMD matrix-vector multiplication for BF16 weights: out = W * x
#[inline]
pub fn gemv_bf16(weight_bytes: &[u8], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(x.len(), cols, "input x dimension mismatch");
    assert_eq!(out.len(), rows, "output dimension mismatch");
    assert!(weight_bytes.len() >= rows * cols * 2, "weight slice too small for BF16 GEMV");
    unsafe {
        sofuu_gemv_bf16(weight_bytes.as_ptr(), x.as_ptr(), out.as_mut_ptr(), rows, cols);
    }
}

/// Multi-threaded hardware SIMD matrix-vector multiplication for MLX 4-bit affine quantized weights: out = W * x
#[inline]
pub fn gemv_mlx4(
    w_bytes: &[u8],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    is_bf16: bool,
) {
    assert_eq!(x.len(), cols, "input x dimension mismatch");
    assert_eq!(out.len(), rows, "output dimension mismatch");
    let u32_per_row = cols / 8;
    let grp_per_row = cols / group_size;
    assert!(w_bytes.len() >= rows * u32_per_row * 4, "weight slice too small for MLX4 GEMV");
    assert!(scales_bytes.len() >= rows * grp_per_row * 2, "scales slice too small for MLX4 GEMV");
    assert!(biases_bytes.len() >= rows * grp_per_row * 2, "biases slice too small for MLX4 GEMV");
    unsafe {
        sofuu_gemv_mlx4(
            w_bytes.as_ptr(),
            scales_bytes.as_ptr(),
            biases_bytes.as_ptr(),
            x.as_ptr(),
            out.as_mut_ptr(),
            rows,
            cols,
            group_size,
            is_bf16 as i32,
        );
    }
}

/// Batched MLX4 matrix-vector multiplication against one activation vector.
/// This reduces per-matrix dispatch and activation-quantization overhead.
pub fn gemv_mlx4_multi(
    weights: &[&[u8]],
    scales: &[&[u8]],
    biases: &[&[u8]],
    x: &[f32],
    outs: &mut [&mut [f32]],
    rows: &[usize],
    cols: usize,
    group_size: usize,
    is_bf16: bool,
) {
    assert_eq!(weights.len(), scales.len());
    assert_eq!(weights.len(), biases.len());
    assert_eq!(weights.len(), outs.len());
    assert_eq!(weights.len(), rows.len());
    let w_ptrs: Vec<*const u8> = weights.iter().map(|w| w.as_ptr()).collect();
    let s_ptrs: Vec<*const u8> = scales.iter().map(|s| s.as_ptr()).collect();
    let b_ptrs: Vec<*const u8> = biases.iter().map(|b| b.as_ptr()).collect();
    let out_ptrs: Vec<*mut f32> = outs.iter_mut().map(|o| o.as_mut_ptr()).collect();
    unsafe {
        sofuu_gemv_mlx4_multi(
            w_ptrs.as_ptr(),
            s_ptrs.as_ptr(),
            b_ptrs.as_ptr(),
            x.as_ptr(),
            out_ptrs.as_ptr(),
            rows.as_ptr(),
            rows.len(),
            cols,
            group_size,
            is_bf16 as i32,
        );
    }
}

/// Matrix-vector multiplication: y = W * x
/// where W has shape [rows, cols] in row-major layout with GGML type `type_id`.
/// Executes via hardware-tuned C++ SIMD microkernels (NEON / AVX2).
pub fn matvec(w_bytes: &[u8], type_id: u32, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), cols, "x.len() must match cols");
    assert_eq!(y.len(), rows, "y.len() must match rows");

    let _prof = prof::Scope::new(&prof::GEMV_NS, &prof::GEMV_CALLS);
    match type_id {
        GGML_TYPE_Q4_0 => {
            unsafe {
                sofuu_gemv_q4_0(w_bytes.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols);
            }
        }
        GGML_TYPE_Q8_0 => {
            unsafe {
                sofuu_gemv_q8_0(w_bytes.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols);
            }
        }
        GGML_TYPE_Q4_K => {
            unsafe {
                sofuu_gemv_q4_k(w_bytes.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols);
            }
        }
        GGML_TYPE_Q6_K => {
            unsafe {
                sofuu_gemv_q6_k(w_bytes.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols);
            }
        }
        GGML_TYPE_F32 => {
            unsafe {
                let f32_ptr = w_bytes.as_ptr() as *const f32;
                sofuu_gemv_f32(f32_ptr, x.as_ptr(), y.as_mut_ptr(), rows, cols);
            }
        }
        GGML_TYPE_F16 => {
            unsafe {
                sofuu_gemv_f16(w_bytes.as_ptr(), x.as_ptr(), y.as_mut_ptr(), rows, cols);
            }
        }
        _ => {
            y.fill(0.0);
        }
    }
}

/// Batched Matrix-matrix multiplication: Y = X * W^T
/// where W has shape [rows, cols] in row-major layout with GGML type `type_id`,
/// X has shape [batch_size, cols] in row-major layout,
/// and Y has shape [batch_size, rows] in row-major layout.
/// Executes via hardware-tuned C++ SIMD microkernels (NEON / AVX2).
pub fn matmat(
    w_bytes: &[u8],
    type_id: u32,
    rows: usize,
    cols: usize,
    x_batch: &[f32],
    batch_size: usize,
    y_batch: &mut [f32],
) {
    assert_eq!(x_batch.len(), batch_size * cols, "x_batch.len() must match batch_size * cols");
    assert_eq!(y_batch.len(), batch_size * rows, "y_batch.len() must match batch_size * rows");

    if batch_size == 0 {
        return;
    }
    if batch_size == 1 {
        matvec(w_bytes, type_id, rows, cols, x_batch, y_batch);
        return;
    }

    let _prof = prof::Scope::new(&prof::GEMM_NS, &prof::GEMM_CALLS);
    match type_id {
        GGML_TYPE_Q4_0 => unsafe {
            sofuu_gemm_q4_0(w_bytes.as_ptr(), x_batch.as_ptr(), y_batch.as_mut_ptr(), rows, cols, batch_size);
        },
        GGML_TYPE_Q8_0 => unsafe {
            sofuu_gemm_q8_0(w_bytes.as_ptr(), x_batch.as_ptr(), y_batch.as_mut_ptr(), rows, cols, batch_size);
        },
        GGML_TYPE_Q4_K => unsafe {
            sofuu_gemm_q4_k(w_bytes.as_ptr(), x_batch.as_ptr(), y_batch.as_mut_ptr(), rows, cols, batch_size);
        },
        GGML_TYPE_Q6_K => unsafe {
            sofuu_gemm_q6_k(w_bytes.as_ptr(), x_batch.as_ptr(), y_batch.as_mut_ptr(), rows, cols, batch_size);
        },
        GGML_TYPE_F32 => unsafe {
            let f32_ptr = w_bytes.as_ptr() as *const f32;
            sofuu_gemm_f32(f32_ptr, x_batch.as_ptr(), y_batch.as_mut_ptr(), rows, cols, batch_size);
        },
        GGML_TYPE_F16 => unsafe {
            let u16_ptr = w_bytes.as_ptr() as *const u16;
            sofuu_gemm_f16(u16_ptr, x_batch.as_ptr(), y_batch.as_mut_ptr(), rows, cols, batch_size);
        },
        _ => {
            y_batch.fill(0.0);
        }
    }
}

/// Residual-fused matmat: computes `w·x` and adds it in place onto `acc`,
/// which must be preloaded with the residual (`acc` is both source and
/// destination, so `x += w·x_norm` is one kernel pass instead of GEMM +
/// separate row-add). Returns false when the fused tile path was not taken —
/// `acc` is untouched and the caller falls back to `matmat` + `batched_mad`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn matmat_residual(
    w_bytes: &[u8],
    type_id: u32,
    rows: usize,
    cols: usize,
    x_batch: &[f32],
    batch_size: usize,
    acc: &mut [f32],
) -> bool {
    if batch_size < 2 || batch_size > 512 {
        return false;
    }
    assert_eq!(x_batch.len(), batch_size * cols, "x_batch.len() must match batch_size * cols");
    assert_eq!(acc.len(), batch_size * rows, "acc.len() must match batch_size * rows");
    if !cfg!(target_arch = "aarch64") {
        return false;
    }
    let dbg = std::env::var_os("SOFUU_FUSE_DEBUG").is_some();
    let _prof = prof::Scope::new(&prof::GEMM_NS, &prof::GEMM_CALLS);
    let fused = match type_id {
        GGML_TYPE_Q4_K => unsafe {
            sofuu_gemm_q4_k_acc(w_bytes.as_ptr(), x_batch.as_ptr(), acc.as_mut_ptr(), rows, cols, batch_size) == 0
        },
        GGML_TYPE_Q8_0 => unsafe {
            sofuu_gemm_q8_0_acc(w_bytes.as_ptr(), x_batch.as_ptr(), acc.as_mut_ptr(), rows, cols, batch_size) == 0
        },
        GGML_TYPE_Q6_K => unsafe {
            sofuu_gemm_q6_k_acc(w_bytes.as_ptr(), x_batch.as_ptr(), acc.as_mut_ptr(), rows, cols, batch_size) == 0
        },
        _ => false,
    };
    if dbg {
        static CALLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = CALLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 8 || n % 512 == 0 {
            eprintln!("[fuse-debug] residual type={} {}x{} batch={} -> fused={}", type_id, rows, cols, batch_size, fused);
        }
    }
    fused
}

/// Dual matmat for the gate/up pair (same input, two weights): computes both
/// `w1·x` and `w2·x` in one dispatch with a single activation-quantize pass.
/// Returns false (buffers untouched) when the fused tile path is unavailable;
/// the caller then falls back to two `matmat` calls.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn matmat_pair(
    w1_bytes: &[u8],
    w2_bytes: &[u8],
    type_id: u32,
    rows: usize,
    cols: usize,
    x_batch: &[f32],
    batch_size: usize,
    y1_batch: &mut [f32],
    y2_batch: &mut [f32],
) -> bool {
    if batch_size < 4 || batch_size % 4 != 0 || batch_size > 512 {
        return false;
    }
    assert_eq!(x_batch.len(), batch_size * cols, "x_batch.len() must match batch_size * cols");
    assert_eq!(y1_batch.len(), batch_size * rows);
    assert_eq!(y2_batch.len(), batch_size * rows);
    if !cfg!(target_arch = "aarch64") {
        return false;
    }
    match type_id {
        GGML_TYPE_Q4_K => unsafe {
            let _prof = prof::Scope::new(&prof::GEMM_NS, &prof::GEMM_CALLS);
            sofuu_gemm_q4_k_pair(w1_bytes.as_ptr(), w2_bytes.as_ptr(), x_batch.as_ptr(), y1_batch.as_mut_ptr(), y2_batch.as_mut_ptr(), rows, cols, batch_size) == 0
        },
        GGML_TYPE_Q8_0 => unsafe {
            let _prof = prof::Scope::new(&prof::GEMM_NS, &prof::GEMM_CALLS);
            sofuu_gemm_q8_0_pair(w1_bytes.as_ptr(), w2_bytes.as_ptr(), x_batch.as_ptr(), y1_batch.as_mut_ptr(), y2_batch.as_mut_ptr(), rows, cols, batch_size) == 0
        },
        _ => false,
    }
}

/// Out-place Root Mean Square Normalization (RMSNorm) via C++ SIMD.
#[inline]
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let _p = prof::Scope::new(&prof::NORM_NS, &prof::NORM_CALLS);
    assert_eq!(out.len(), x.len());
    assert_eq!(out.len(), weight.len());
    unsafe {
        sofuu_rmsnorm(out.as_mut_ptr(), x.as_ptr(), weight.as_ptr(), x.len(), eps);
    }
}

/// In-place Root Mean Square Normalization (RMSNorm) via C++ SIMD.
#[inline]
pub fn rmsnorm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    let _p = prof::Scope::new(&prof::NORM_NS, &prof::NORM_CALLS);
    assert_eq!(x.len(), weight.len());
    unsafe {
        sofuu_rmsnorm(x.as_mut_ptr(), x.as_ptr(), weight.as_ptr(), x.len(), eps);
    }
}

/// Batched (out-place) RMSNorm over `rows` contiguous rows of `cols` floats;
/// one FFI call, internally parallel. `out` may alias `x`.
#[inline]
pub fn batched_rmsnorm(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let _p = prof::Scope::new(&prof::NORM_NS, &prof::NORM_CALLS);
    debug_assert_eq!(out.len(), rows * cols);
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(weight.len(), cols);
    unsafe {
        sofuu_batched_rmsnorm(out.as_mut_ptr(), x.as_ptr(), weight.as_ptr(), rows, cols, eps);
    }
}

/// Batched in-place RMSNorm (single buffer, aliased through the kernel).
#[inline]
pub fn batched_rmsnorm_in_place(x: &mut [f32], weight: &[f32], rows: usize, cols: usize, eps: f32) {
    let _p = prof::Scope::new(&prof::NORM_NS, &prof::NORM_CALLS);
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(weight.len(), cols);
    unsafe {
        let p = x.as_mut_ptr();
        sofuu_batched_rmsnorm(p, p, weight.as_ptr(), rows, cols, eps);
    }
}

/// Batched in-place multiply-accumulate: `dst[r] += alpha * src[r]` per row.
#[inline]
pub fn batched_mad(dst: &mut [f32], src: &[f32], alpha: f32, rows: usize, cols: usize) {
    let _p = prof::Scope::new(&prof::GLUE_NS, &prof::GLUE_CALLS);
    debug_assert_eq!(dst.len(), rows * cols);
    debug_assert_eq!(src.len(), rows * cols);
    unsafe {
        sofuu_batched_mad(dst.as_mut_ptr(), src.as_ptr(), alpha, rows, cols);
    }
}

/// Batched per-row bias add: `x[r][c] += bias[c]`.
#[inline]
pub fn batched_add_bias(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    let _p = prof::Scope::new(&prof::GLUE_NS, &prof::GLUE_CALLS);
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(bias.len(), cols);
    unsafe {
        sofuu_batched_add_bias(x.as_mut_ptr(), bias.as_ptr(), rows, cols);
    }
}

/// Batched in-place SwiGLU over `rows` rows of `cols` floats.
#[inline]
pub fn batched_swiglu(gate: &mut [f32], up: &[f32], rows: usize, cols: usize) {
    let _p = prof::Scope::new(&prof::GLUE_NS, &prof::GLUE_CALLS);
    debug_assert_eq!(gate.len(), rows * cols);
    debug_assert_eq!(up.len(), rows * cols);
    unsafe {
        sofuu_batched_swiglu(gate.as_mut_ptr(), up.as_ptr(), rows, cols);
    }
}

/// Batched RoPE on q/k rows + KV-cache scatter, parallel over tokens.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn batched_rope_kv(
    q_batch: &mut [f32],
    k_batch: &mut [f32],
    v_batch: &[f32],
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    b_size: usize,
    start_pos: usize,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    freq_base: f32,
    rope_mode: i32,
) {
    let _p = prof::Scope::new(&prof::ROPE_NS, &prof::ROPE_CALLS);
    unsafe {
        sofuu_batched_rope_kv(
            q_batch.as_mut_ptr(),
            k_batch.as_mut_ptr(),
            v_batch.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
            b_size,
            start_pos,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq_len,
            freq_base,
            rope_mode,
        );
    }
}

/// Vectorized f32 -> f16 row convert; the single rounding path every f16 KV
/// shadow writer uses (matches the prefill writer bit for bit).
#[inline]
pub fn kv16_convert(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    unsafe {
        sofuu_kv16_convert(src.as_ptr(), dst.as_mut_ptr(), src.len());
    }
}

/// Batched RoPE + KV scatter that also writes the f16 shadow the CPU
/// attention reads (one pass, so the shadow cannot drift).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn batched_rope_kv_f16(
    q_batch: &mut [f32],
    k_batch: &mut [f32],
    v_batch: &[f32],
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    k16_cache: &mut [u16],
    v16_cache: &mut [u16],
    b_size: usize,
    start_pos: usize,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    freq_base: f32,
    rope_mode: i32,
) {
    let _p = prof::Scope::new(&prof::ROPE_NS, &prof::ROPE_CALLS);
    unsafe {
        sofuu_batched_rope_kv_f16(
            q_batch.as_mut_ptr(),
            k_batch.as_mut_ptr(),
            v_batch.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
            k16_cache.as_mut_ptr(),
            v16_cache.as_mut_ptr(),
            b_size,
            start_pos,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq_len,
            freq_base,
            rope_mode,
        );
    }
}

/// Rotates Q and K for speculative tree candidate nodes at their specific positions without committing to KV cache.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn tree_rope(
    q_batch: &mut [f32],
    k_batch: &mut [f32],
    node_positions: &[usize],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    freq_base: f32,
    rope_mode: i32,
) {
    let tree_size = node_positions.len();
    if tree_size == 0 {
        return;
    }
    unsafe {
        sofuu_tree_rope(
            q_batch.as_mut_ptr(),
            k_batch.as_mut_ptr(),
            node_positions.as_ptr(),
            tree_size,
            n_heads,
            n_kv_heads,
            head_dim,
            freq_base,
            rope_mode,
        );
    }
}

/// Fused SwiGLU activation: out = SiLU(gate) * up via C++ SIMD.
#[inline]
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    assert_eq!(out.len(), gate.len());
    assert_eq!(out.len(), up.len());
    unsafe {
        sofuu_swiglu(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), out.len());
    }
}

/// In-place SwiGLU activation: gate = SiLU(gate) * up via C++ SIMD.
#[inline]
pub fn swiglu_in_place(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    unsafe {
        sofuu_swiglu(gate.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), gate.len());
    }
}

/// Numerically stable in-place Softmax via C++ SIMD.
#[inline]
pub fn softmax(scores: &mut [f32]) {
    unsafe {
        sofuu_softmax(scores.as_mut_ptr(), scores.len());
    }
}

/// Rotary Position Embedding (RoPE) via C++ SIMD.
#[inline]
pub fn rope(vec: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, freq_base: f32) {
    rope_style(vec, n_heads, head_dim, pos, freq_base, 0);
}

pub fn rope_style(
    vec: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    freq_base: f32,
    rope_style: i32,
) {
    assert_eq!(vec.len(), n_heads * head_dim);
    unsafe {
        sofuu_rope(vec.as_mut_ptr(), n_heads, head_dim, pos, freq_base, rope_style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tile-cache entries are keyed by (weight pointer, rows, cols, type);
    /// tests running in parallel can hand the allocator the same address for
    /// a different fixture, so every tile-warming test takes this lock.
    static TILE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn tile_test_guard() -> std::sync::MutexGuard<'static, ()> {
        TILE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn test_tile_gemv_q8_0_matches_reference() {
        let _guard = tile_test_guard();
        // 16 rows x 32 cols (single q8_0 block per row, one 4-row tile pair).
        let rows = 16usize;
        let cols = 32usize;
        let mut w_bytes = vec![0u8; rows * (cols + 2)]; // 32 qs + 2 d per row
        let mut weights = vec![0f32; rows * cols];
        for r in 0..rows {
            // f16-exact delta (exp 8, mantissa r*64), matching the gemm test below
            let d_bits: u16 = (8u16 << 10) | ((r as u16) * 64);
            let d = f16_to_f32(d_bits);
            let bytes = &mut w_bytes[r * (cols + 2)..(r + 1) * (cols + 2)];
            bytes[0] = (d_bits & 0xff) as u8;
            bytes[1] = (d_bits >> 8) as u8;
            for c in 0..cols {
                let v = ((r * 13 + c * 7) % 30) as i8 - 15;
                bytes[2 + c] = v as u8;
                weights[r * cols + c] = v as f32 * d;
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.11 - 1.5).sin()).collect();

        let mut ref_out = vec![0f32; rows];
        for r in 0..rows {
            let mut sum = 0f32;
            for c in 0..cols {
                sum += weights[r * cols + c] * x[c];
            }
            ref_out[r] = sum;
        }

        tile_warm(w_bytes.as_ptr(), rows, cols, GGML_TYPE_Q8_0);

        let mut out = vec![0f32; rows];
        matvec(&w_bytes, GGML_TYPE_Q8_0, rows, cols, &x, &mut out);

        for r in 0..rows {
            let a = out[r];
            let b = ref_out[r];
            // Tolerance is relative to the row's L1 magnitude (Σ|w·x|): int8
            // activation quantization is ≤0.4% per element, which only shows
            // up against the net sum when signs cancel.
            let l1: f32 = (0..cols).map(|c| (weights[r * cols + c] * x[c]).abs()).sum();
            let ok = (a - b).abs() <= 0.02 * l1 + 1e-4;
            assert!(ok, "row {r}: tile {a} vs ref {b}");
        }
    }

    #[test]
    fn test_tile_gemm_q8_0_matches_reference() {
        let _guard = tile_test_guard();
        let rows = 16usize;
        let cols = 32usize;
        let batch = 8usize;
        let mut w_bytes = vec![0u8; rows * (cols + 2)];
        let mut weights = vec![0f32; rows * cols];
        for r in 0..rows {
            // f16-exact delta (exp 8, mantissa r*64)
            let d_bits: u16 = (8u16 << 10) | ((r as u16) * 64);
            let d = f16_to_f32(d_bits);
            let bytes = &mut w_bytes[r * (cols + 2)..(r + 1) * (cols + 2)];
            bytes[0] = (d_bits & 0xff) as u8;
            bytes[1] = (d_bits >> 8) as u8;
            for c in 0..cols {
                let v = ((r * 13 + c * 7) % 30) as i8 - 15;
                bytes[2 + c] = v as u8;
                weights[r * cols + c] = v as f32 * d;
            }
        }
        let x: Vec<f32> = (0..batch * cols)
            .map(|i| (((i % 17) as f32) * 0.13 - 1.0).cos())
            .collect();

        tile_warm(w_bytes.as_ptr(), rows, cols, GGML_TYPE_Q8_0);

        let mut out = vec![0f32; batch * rows];
        matmat(&w_bytes, GGML_TYPE_Q8_0, rows, cols, &x, batch, &mut out);

        for b in 0..batch {
            for r in 0..rows {
                let a = out[b * rows + r];
                let mut b_ref = 0f32;
                for c in 0..cols {
                    b_ref += weights[r * cols + c] * x[b * cols + c];
                }
                let ok = (a - b_ref).abs() <= 0.05 * b_ref.abs().max(1e-3) + 1e-2;
                assert!(ok, "({b},{r}): tile {a} vs ref {b_ref}");
            }
        }
    }

    /// GGUF-layout q4_K fixture: per 256-elem super-block an f16 d/dmin, the
    /// 12-byte 6-bit scales/mins pack (inverse of get_scale_min_k4), and
    /// nibbles stored as low = element g*64+e, high = element g*64+32+e —
    /// the layout the tile kernels and the SDOT fallback both decode.
    fn build_q4_k_fixture(rows: usize, cols: usize) -> Vec<u8> {
        const QK: usize = 256;
        assert_eq!(cols % QK, 0);
        let blks_per_row = cols / QK;
        let row_len = blks_per_row * 144;
        let mut bytes = vec![0u8; rows * row_len];
        for r in 0..rows {
            for sb in 0..blks_per_row {
                let base = r * row_len + sb * 144;
                let d_bits: u16 = (8u16 << 10) | (((r * 2 + sb) % 16) as u16 * 64);
                let dmin_bits: u16 = (6u16 << 10) | (((r + 2 * sb + 3) % 8) as u16 * 128);
                bytes[base] = (d_bits & 0xff) as u8;
                bytes[base + 1] = (d_bits >> 8) as u8;
                bytes[base + 2] = (dmin_bits & 0xff) as u8;
                bytes[base + 3] = (dmin_bits >> 8) as u8;
                let sc: Vec<u8> = (0..8).map(|j| 1 + ((r * 7 + sb * 5 + j * 11) % 63) as u8).collect();
                let mn: Vec<u8> = (0..8).map(|j| ((r * 3 + sb * 13 + j * 29) % 64) as u8).collect();
                for j in 0..4 {
                    bytes[base + 4 + j] = sc[j] | ((sc[j + 4] & 0x30) << 2);
                    bytes[base + 8 + j] = mn[j] | ((mn[j + 4] & 0x30) << 2);
                    bytes[base + 12 + j] = (sc[j + 4] & 0x0F) | ((mn[j + 4] & 0x0F) << 4);
                }
                for g in 0..4 {
                    for e in 0..32 {
                        let lo = ((r * 71 + sb * 37 + g * 19 + e * 5) % 16) as u8;
                        let hi = ((r * 23 + sb * 41 + g * 7 + e * 3) % 16) as u8;
                        bytes[base + 16 + g * 32 + e] = lo | (hi << 4);
                    }
                }
            }
        }
        bytes
    }

    /// Dot-convention q4_K reference: w = d·sc·q4 − dmin·mn per 32-element
    /// sub-block (matches the runtime kernels and SDOT vecdot).
    fn q4_k_reference_weights(w_bytes: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let mut weights = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let sb = c / 256;
                let t = c % 256;
                let grp = t / 32;
                let base = r * (cols / 256) * 144 + sb * 144;
                let d = f16_to_f32(((w_bytes[base + 1] as u16) << 8) | w_bytes[base] as u16);
                let dmin = f16_to_f32(((w_bytes[base + 3] as u16) << 8) | w_bytes[base + 2] as u16);
                let j = grp % 4;
                let sc = if grp < 4 {
                    w_bytes[base + 4 + j] & 0x3F
                } else {
                    (w_bytes[base + 12 + j] & 0x0F) | ((w_bytes[base + 4 + j] >> 6) << 4)
                };
                let mn = if grp < 4 {
                    w_bytes[base + 8 + j] & 0x3F
                } else {
                    (w_bytes[base + 12 + j] >> 4) | ((w_bytes[base + 8 + j] >> 6) << 4)
                };
                let g = t / 64;
                let e = t % 64;
                let byte = w_bytes[base + 16 + g * 32 + (e % 32)];
                let q = if e < 32 { byte & 0x0F } else { byte >> 4 };
                weights[r * cols + c] = d * (sc as f32) * (q as f32) - dmin * (mn as f32);
            }
        }
        weights
    }

    #[test]
    fn test_tile_q4_k_scale_sidecar_bit_exact() {
        let _guard = tile_test_guard();
        // (16, 256) takes the serial warm path (2 row groups); (128, 256)
        // crosses the parallel thresholds (16 row groups, 16 tile blocks)
        // so the ranged repack + sidecar fill are covered too.
        for (rows, cols) in [(16usize, 256usize), (128usize, 256usize)] {
            let batch = 8usize;
            let w_bytes = build_q4_k_fixture(rows, cols);
            let weights = q4_k_reference_weights(&w_bytes, rows, cols);

            let x: Vec<f32> =
                (0..batch * cols).map(|i| ((i % 37) as f32) * 0.098 - 1.4).collect();

            let run = |out: &mut Vec<f32>, outv: &mut Vec<f32>| {
                tile_warm(w_bytes.as_ptr(), rows, cols, GGML_TYPE_Q4_K);
                *out = vec![0f32; batch * rows];
                matmat(&w_bytes, GGML_TYPE_Q4_K, rows, cols, &x, batch, out);
                *outv = vec![0f32; rows];
                matvec(&w_bytes, GGML_TYPE_Q4_K, rows, cols, &x[..cols], outv);
            };

            tile_scale_sidecar_disable(false);
            let mut out_sc = Vec::new();
            let mut outv_sc = Vec::new();
            run(&mut out_sc, &mut outv_sc);

            // Rebuild the entry without a sidecar so the same tiles are
            // computed by the original kernels; outputs must be bit-identical.
            tile_scale_sidecar_disable(true);
            let mut out_nosc = Vec::new();
            let mut outv_nosc = Vec::new();
            run(&mut out_nosc, &mut outv_nosc);
            tile_scale_sidecar_disable(false);

            for i in 0..out_sc.len() {
                assert_eq!(
                    out_sc[i].to_bits(),
                    out_nosc[i].to_bits(),
                    "gemm ({rows}x{cols}, {}, row {}): sidecar {:?} vs {}",
                    i / rows,
                    i % rows,
                    out_sc[i],
                    out_nosc[i]
                );
            }
            for r in 0..rows {
                assert_eq!(outv_sc[r].to_bits(), outv_nosc[r].to_bits(), "gemv row {r}");
            }

            // Sanity: both paths agree with the dequant reference within the
            // usual int8-activation tolerance, guarding against fixture drift.
            for b in 0..batch {
                for r in 0..rows {
                    let a = out_sc[b * rows + r];
                    let mut l1 = 0f32;
                    let mut want = 0f32;
                    for c in 0..cols {
                        let p = weights[r * cols + c] * x[b * cols + c];
                        l1 += p.abs();
                        want += p;
                    }
                    assert!((a - want).abs() <= 0.05 * l1 + 1e-3, "({b},{r}): {a} vs {want}");
                }
            }
        }
    }

    use super::sofuu_test_quant_q8_K_parity;

    /// The NEON q8_K 4x4 activation quantizer must stay bit-identical to the
    /// scalar generic (qs/d/bsums) — guards the prefill numerics.
    #[test]
    fn test_quant_q8_k_4x4_neon_bit_exact() {
        for cols in [256usize, 1024, 3072] {
            let r = unsafe { sofuu_test_quant_q8_K_parity(cols as i32, 25) };
            assert_eq!(r, 0, "NEON q8_K quantizer diverged at cols={cols}");
        }
    }

    /// The nibble-plane q4_K GEMM (R3) must be bit-identical to the packed
    /// kernel: the plane expansion relocates the vand/vshr nibble unpack to
    /// warm time and must not change the int math.
    #[test]
    fn test_gemm_q4_k_nibble_plane_bit_exact() {
        let _guard = tile_test_guard();
        for (rows, cols) in [(64usize, 256usize), (64, 1024), (128, 3072)] {
            let r = gemm_plane_parity(rows, cols, 8);
            assert_eq!(r, 0, "nibble-plane q4_K GEMM diverged at {rows}x{cols}");
        }
    }

    /// The T1 pre-decoded-scale GEMM (gemm_s over the stock tile + ss
    /// section) must be bit-identical to the stock packed-tile kernel: the
    /// reorganization replaces the 6-bit group unpack with repack-time
    /// loads and moves the sub-block scale multiply to the f32 pipe, where
    /// every product is exact.
    #[test]
    fn test_gemm_q4_k_s_tiles_bit_exact() {
        let _guard = tile_test_guard();
        for (rows, cols) in [(64usize, 256usize), (64, 1024), (128, 3072)] {
            let r = gemm_q4_s_parity(rows, cols, 8);
            assert_eq!(r, 0, "pre-decoded-scale q4_K tiles diverged at {rows}x{cols}");
        }
    }

    /// Small speculative batches (2, 3, 5 rows) are padded to a 4-multiple
    /// inside the gemm dispatch; the pad must not perturb the real rows, so
    /// the padded result is bit-equal to an explicit zero-padded batch-4 (or
    /// batch-8) tile computation on the shared rows. Warms explicitly and
    /// uses a shape unlike the sidecar test's fixtures: tests run in
    /// parallel and the tile registry keys on (pointer, shape, fingerprint),
    /// so an identical sibling fixture could alias mid-test.
    #[test]
    fn test_tile_gemm_small_batch_pad_bit_exact() {
        let _guard = tile_test_guard();
        let rows = 24usize;
        let cols = 256usize;
        let w_bytes = build_q4_k_fixture(rows, cols);
        tile_warm(w_bytes.as_ptr(), rows, cols, GGML_TYPE_Q4_K);

        for (small, big) in [(2usize, 4usize), (3usize, 4usize), (5usize, 8usize)] {
            let big_x: Vec<f32> = (0..big * cols)
                .map(|i| if i >= small * cols { 0.0 } else { ((i % 37) as f32) * 0.098 - 1.4 })
                .collect();
            let small_x = big_x[..small * cols].to_vec();

            let mut out_big = vec![0f32; big * rows];
            matmat(&w_bytes, GGML_TYPE_Q4_K, rows, cols, &big_x, big, &mut out_big);
            let mut out_small = vec![0f32; small * rows];
            matmat(&w_bytes, GGML_TYPE_Q4_K, rows, cols, &small_x, small, &mut out_small);

            for b in 0..small {
                for r in 0..rows {
                    assert_eq!(
                        out_small[b * rows + r].to_bits(),
                        out_big[b * rows + r].to_bits(),
                        "pad({small}->{big}) ({b},{r}): {:?} vs {:?}",
                        out_small[b * rows + r],
                        out_big[b * rows + r]
                    );
                }
            }
        }
    }

    #[test]
    fn test_cpp_rmsnorm() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &weight, 1e-5);

        // mean(x^2) = 30 / 4 = 7.5. sqrt(7.5) = 2.7386127
        let expected_0 = 1.0 / 7.5f32.sqrt();
        assert!((out[0] - expected_0).abs() < 1e-4);
        assert!((out[1] - 2.0 * expected_0).abs() < 1e-4);
        assert!((out[2] - 3.0 * expected_0).abs() < 1e-4);
        assert!((out[3] - 4.0 * expected_0).abs() < 1e-4);
    }

    #[test]
    fn test_cpp_softmax() {
        let mut scores = [1.0f32, 2.0, 3.0];
        softmax(&mut scores);

        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(scores[0] < scores[1]);
        assert!(scores[1] < scores[2]);
    }

    #[test]
    fn test_cpp_swiglu() {
        let gate = [0.0f32, 2.0];
        let up = [5.0f32, 3.0];
        let mut out = [0.0f32; 2];
        swiglu(&mut out, &gate, &up);

        assert_eq!(out[0], 0.0);
        let silu_2 = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((out[1] - silu_2 * 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_cpp_gemv_f32() {
        let w = [1.0f32, 2.0, 3.0, 4.0]; // 2 rows, 2 cols
        let x = [2.0f32, 1.0];
        let mut y = [0.0f32; 2];

        let w_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4)
        };
        matvec(w_bytes, GGML_TYPE_F32, 2, 2, &x, &mut y);

        assert_eq!(y[0], 4.0); // 1*2 + 2*1 = 4
        assert_eq!(y[1], 10.0); // 3*2 + 4*1 = 10
    }

    #[test]
    fn test_cpp_thread_controls() {
        let default_threads = get_thread_count();
        assert!(default_threads > 0);

        set_thread_count(4);
        assert_eq!(get_thread_count(), 4);

        set_thread_count(2);
        assert_eq!(get_thread_count(), 2);

        // Reset to default
        set_thread_count(default_threads);
    }

    #[test]
    fn test_cpp_dequant_and_gemv_q4_k() {
        // Construct 1 super-block of Q4_K (256 elements = 144 bytes)
        let mut block = vec![0u8; 144];
        // d = 1.0 (f16 0x3c00), dmin = 0.0 (f16 0x0000)
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = 0x00;
        block[3] = 0x00;
        // scales[0] = 1, rest 0
        block[4] = 1; // scale 0 has sc = 1, m = 0
        // qs: set first byte: low nibble = 5, high nibble = 3
        block[16] = 0x35;

        let mut dst = vec![0.0f32; 256];
        dequantize_q4_k_row(&block, &mut dst);

        // First weight should be d * sc * (qs[0] & 0xF) - min * m = 1.0 * 1 * 5 - 0 = 5.0
        assert_eq!(dst[0], 5.0);
        // Weight at index 32 should be d * sc * (qs[0] >> 4) = 1.0 * 0 * 3 = 0.0 (since sc[1]=0)
        assert_eq!(dst[32], 0.0);

        // Test GEMV
        let mut x = vec![0.0f32; 256];
        x[0] = 2.0;
        let mut y = vec![0.0f32; 1];
        matvec(&block, GGML_TYPE_Q4_K, 1, 256, &x, &mut y);
        // y[0] should be dst[0] * x[0] = 5.0 * 2.0 = 10.0 (within quantization error)
        assert!((y[0] - 10.0).abs() < 1e-3, "expected 10.0, got {}", y[0]);
    }

    #[test]
    fn test_cpp_dequant_and_gemv_q6_k() {
        // Construct 1 super-block of Q6_K (256 elements = 210 bytes)
        // Layout: ql[128], qh[64], scales[16], d[2]
        let mut block = vec![0u8; 210];
        // scales[0] = 2
        block[128 + 64] = 2; // scale 0 is at offset 192
        // d = 1.0 (f16 0x3c00) at offset 208
        block[208] = 0x00;
        block[209] = 0x3c;
        // ql[0] = 32 (so q1 = 32 - 32 = 0)
        // Let's set ql[0] = 34: (34 & 0xF) = 2, qh[0] = 0 -> (2 | (0 << 4)) - 32 = 2 - 32 = -30
        // Or to get positive 5: q = 37 -> 37 - 32 = 5.
        // 37 in binary: 0b100101 -> lower 4 bits = 0b0101 (5), upper 2 bits = 0b10 (2)
        block[0] = 5; // ql[0] lower 4 bits = 5
        block[128] = 2; // qh[0] bits 0..1 = 2 -> (5 | (2 << 4)) = 37. 37 - 32 = 5.

        let mut dst = vec![0.0f32; 256];
        dequantize_q6_k_row(&block, &mut dst);

        // dst[0] = d * sc[0] * (q - 32) = 1.0 * 2 * (37 - 32) = 10.0
        assert_eq!(dst[0], 10.0);

        // Test GEMV
        let mut x = vec![0.0f32; 256];
        x[0] = 3.0;
        let mut y = vec![0.0f32; 1];
        matvec(&block, GGML_TYPE_Q6_K, 1, 256, &x, &mut y);
        // y[0] should be dst[0] * x[0] = 10.0 * 3.0 = 30.0
        assert_eq!(y[0], 30.0);
    }

    #[test]
    fn test_simd_vec_dot_and_mad_f32() {
        let n = 128;
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for i in 0..n {
            a[i] = (i + 1) as f32;
            b[i] = 0.5f32;
        }

        let dot = vec_dot_f32(&a, &b);
        // sum(1..=128) * 0.5 = (128 * 129 / 2) * 0.5 = 8256 * 0.5 = 4128.0
        assert!((dot - 4128.0).abs() < 1e-3, "expected 4128.0, got {dot}");

        let mut dst = vec![1.0f32; n];
        vec_mad_f32(&mut dst, &a, 2.0);
        for i in 0..n {
            let expected = 1.0 + a[i] * 2.0;
            assert_eq!(dst[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_cpp_gemm_f32() {
        // 2 rows, 3 cols
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 2 batches, 3 cols
        let x_batch = [
            1.0f32, 0.5, 2.0, // batch 0
            0.0f32, 1.0, 3.0, // batch 1
        ];
        let mut y_batch = [0.0f32; 4]; // 2 batches * 2 rows

        let w_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4)
        };
        matmat(w_bytes, GGML_TYPE_F32, 2, 3, &x_batch, 2, &mut y_batch);

        // Check against manual calculations
        assert_eq!(y_batch[0], 8.0);
        assert_eq!(y_batch[1], 18.5);
        assert_eq!(y_batch[2], 11.0);
        assert_eq!(y_batch[3], 23.0);

        // Check against individual matvec calls
        let mut y_single_0 = [0.0f32; 2];
        let mut y_single_1 = [0.0f32; 2];
        matvec(w_bytes, GGML_TYPE_F32, 2, 3, &x_batch[0..3], &mut y_single_0);
        matvec(w_bytes, GGML_TYPE_F32, 2, 3, &x_batch[3..6], &mut y_single_1);
        assert_eq!(y_batch[0..2], y_single_0);
        assert_eq!(y_batch[2..4], y_single_1);
    }

    #[test]
    fn test_cpp_gemm_q4_k() {
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3c; // d = 1.0
        block[4] = 1;    // sc = 1, m = 0
        block[16] = 0x35; // qs[0] = 5 low, 3 high

        let batch_size = 3;
        let cols = 256;
        let rows = 1;
        let mut x_batch = vec![0.0f32; batch_size * cols];
        x_batch[0] = 2.0;            // batch 0
        x_batch[cols] = 4.0;         // batch 1
        x_batch[2 * cols] = 10.0;     // batch 2

        let mut y_batch = vec![0.0f32; batch_size * rows];
        matmat(&block, GGML_TYPE_Q4_K, rows, cols, &x_batch, batch_size, &mut y_batch);

        for b in 0..batch_size {
            let mut y_single = vec![0.0f32; rows];
            let x_slice = &x_batch[b * cols..(b + 1) * cols];
            matvec(&block, GGML_TYPE_Q4_K, rows, cols, x_slice, &mut y_single);
            for i in 0..rows {
                let diff = (y_batch[b * rows + i] - y_single[i]).abs();
                assert!(diff < 1e-2, "batch {b} mismatch at row {i}: got {} vs {}", y_batch[b * rows + i], y_single[i]);
            }
        }
    }

    #[test]
    fn test_cpp_gemv_q8_0() {
        // block_q8_0: 2 bytes fp16 scale + 32 bytes int8
        let mut block = vec![0u8; 34];
        block[0] = 0x00;
        block[1] = 0x3c; // d = 1.0 (fp16)
        block[2] = 5;    // qs[0] = 5
        block[3] = -2i8 as u8; // qs[1] = -2

        let cols = 32;
        let rows = 1;
        let mut x = vec![0.0f32; cols];
        x[0] = 3.0;
        x[1] = 4.0;

        let mut y = vec![0.0f32; rows];
        matvec(&block, GGML_TYPE_Q8_0, rows, cols, &x, &mut y);

        // Expected: 5 * 3.0 + (-2) * 4.0 = 15 - 8 = 7.0
        let diff = (y[0] - 7.0).abs();
        assert!(diff < 0.1, "Q8_0 mismatch: got {} vs 7.0", y[0]);
    }

    #[test]
    fn test_cpp_gemm_q6_k() {
        let mut block = vec![0u8; 210];
        block[128 + 64] = 2; // sc[0] = 2
        block[208] = 0x00;
        block[209] = 0x3c;   // d = 1.0
        block[0] = 5;
        block[128] = 2;

        let batch_size = 4;
        let cols = 256;
        let rows = 1;
        let mut x_batch = vec![0.0f32; batch_size * cols];
        x_batch[0] = 3.0;            // batch 0
        x_batch[cols] = 5.0;         // batch 1
        x_batch[2 * cols] = 7.0;     // batch 2
        x_batch[3 * cols] = 9.0;     // batch 3

        let mut y_batch = vec![0.0f32; batch_size * rows];
        matmat(&block, GGML_TYPE_Q6_K, rows, cols, &x_batch, batch_size, &mut y_batch);

        for b in 0..batch_size {
            let mut y_single = vec![0.0f32; rows];
            let x_slice = &x_batch[b * cols..(b + 1) * cols];
            matvec(&block, GGML_TYPE_Q6_K, rows, cols, x_slice, &mut y_single);
            assert!(
                (y_batch[b * rows] - y_single[0]).abs() < 1e-4,
                "batch {b} mismatch: got {} vs {}",
                y_batch[b * rows],
                y_single[0]
            );
        }
    }

    #[test]
    fn test_rope_shift_exact_equivalence() {
        let n_heads = 2;
        let head_dim = 64;
        let freq_base = 10000.0f32;
        let orig_pos = 120usize;
        let delta = 45usize;
        let target_pos = orig_pos - delta;

        let mut v_shifted = vec![0.0f32; n_heads * head_dim];
        let mut v_direct = vec![0.0f32; n_heads * head_dim];
        for i in 0..v_shifted.len() {
            let val = ((i * 17 + 3) % 100) as f32 / 10.0;
            v_shifted[i] = val;
            v_direct[i] = val;
        }

        // Method 1: Embed at orig_pos, then shift by -delta
        rope(&mut v_shifted, n_heads, head_dim, orig_pos, freq_base);
        rope_shift(&mut v_shifted, n_heads, head_dim, -(delta as i64), freq_base);

        // Method 2: Directly embed at target_pos
        rope(&mut v_direct, n_heads, head_dim, target_pos, freq_base);

        for i in 0..v_shifted.len() {
            let diff = (v_shifted[i] - v_direct[i]).abs();
            assert!(
                diff < 5e-5,
                "mismatch at index {i}: shifted {} vs direct {}, diff {diff}",
                v_shifted[i],
                v_direct[i]
            );
        }
    }

    #[test]
    fn test_kv_compact_layers() {
        let n_layers = 2;
        let n_kv_heads = 2;
        let head_dim = 32;
        let max_seq_len = 64;
        let current_seq_len = 40;
        let sink_tokens = 4;
        let drop_tokens = 10;
        let freq_base = 10000.0f32;

        let total_size = n_layers * n_kv_heads * max_seq_len * head_dim;
        let mut k_cache = vec![0.0f32; total_size];
        let mut v_cache = vec![0.0f32; total_size];

        for l in 0..n_layers {
            for kv_h in 0..n_kv_heads {
                let base = (l * n_kv_heads + kv_h) * max_seq_len * head_dim;
                for p in 0..current_seq_len {
                    for d in 0..head_dim {
                        let idx = base + p * head_dim + d;
                        v_cache[idx] = (p * 100 + d) as f32;
                        k_cache[idx] = (p * 100 + d) as f32;
                    }
                    let k_slice = &mut k_cache[base + p * head_dim..base + (p + 1) * head_dim];
                    rope(k_slice, 1, head_dim, p, freq_base);
                }
            }
        }

        kv_compact(
            &mut k_cache,
            &mut v_cache,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            current_seq_len,
            sink_tokens,
            drop_tokens,
            freq_base,
        );

        let new_len = current_seq_len - drop_tokens;
        assert_eq!(new_len, 30);

        for l in 0..n_layers {
            for kv_h in 0..n_kv_heads {
                let base = (l * n_kv_heads + kv_h) * max_seq_len * head_dim;
                // 1. Sink tokens (0..4) should remain intact
                for p in 0..sink_tokens {
                    for d in 0..head_dim {
                        let idx = base + p * head_dim + d;
                        assert_eq!(v_cache[idx], (p * 100 + d) as f32, "sink V mismatch at p={p}, d={d}");
                    }
                }
                // 2. Shifted tokens (4..30) should correspond to original (14..40)
                for new_p in sink_tokens..new_len {
                    let orig_p = new_p + drop_tokens;
                    for d in 0..head_dim {
                        let idx = base + new_p * head_dim + d;
                        assert_eq!(v_cache[idx], (orig_p * 100 + d) as f32, "shifted V mismatch at new_p={new_p}");
                    }
                    let mut direct_k = vec![0.0f32; head_dim];
                    for d in 0..head_dim {
                        direct_k[d] = (orig_p * 100 + d) as f32;
                    }
                    rope(&mut direct_k, 1, head_dim, new_p, freq_base);
                    for d in 0..head_dim {
                        let idx = base + new_p * head_dim + d;
                        let rel_diff = (k_cache[idx] - direct_k[d]).abs() / (direct_k[d].abs().max(1.0));
                        assert!(rel_diff < 1e-4, "compacted K mismatch at new_p={new_p}, d={d}: got {} vs {}, rel_diff={rel_diff}", k_cache[idx], direct_k[d]);
                    }
                }
            }
        }
    }

    /// The x4-position attention inner loops (QK dot + weighted V sum) must be
    /// bit-identical to the original per-position loops — covers plain, SWA,
    /// and SWA-with-sinks shapes (the C harness runs 6 configs per call).
    #[test]
    fn test_attention_x4_bit_exact() {
        assert_eq!(attention_x4_parity(4, 2, 32, 64), 0, "small SWA shapes diverged");
        assert_eq!(attention_x4_parity(16, 8, 128, 256), 0, "Qwen3-0.6B attention shapes diverged");
        assert_eq!(attention_x4_parity(6, 2, 48, 100), 0, "odd head_dim/seq shapes diverged");
    }

    /// The NEON softmax replaces per-score scalar expf with a magic-rounded
    /// degree-5 poly exp — not bit-identical to libm, but at ≤ 5e-7 relative
    /// it sits orders of magnitude inside greedy-decode argmax margins. The
    /// C harness sweeps score-row sizes and edge ranges (deep-underflow
    /// terms, exact zeros) against a double-precision reference.
    #[test]
    fn test_softmax_vec_rel_gate() {
        assert_eq!(softmax_vec_parity(), 0, "NEON softmax out of rel tolerance vs scalar reference");
    }

    /// The FA2 tiled online-softmax kernel must track the two-pass reference
    /// within rel 1e-4 across plain, SWA, and SWA-with-sinks shapes — keeping
    /// seq_len past the 128-block boundary so multi-block rescale is exercised.
    #[test]
    fn test_attention_fa2_rel_gates() {
        assert_eq!(attention_fa2_parity(4, 2, 32, 64), 0, "small SWA shapes out of tolerance");
        assert_eq!(attention_fa2_parity(16, 8, 128, 300), 0, "multi-block Qwen shapes out of tolerance");
        assert_eq!(attention_fa2_parity(6, 2, 48, 259), 0, "odd head_dim/seq shapes out of tolerance");
    }

    /// The RoPE sincos table must be bit-identical to the inline
    /// powf/cosf/sinf path across both RoPE styles and freq bases.
    #[test]
    fn test_rope_table_bit_exact() {
        assert_eq!(rope_table_parity(4, 32, 64), 0, "small rope shapes diverged");
        assert_eq!(rope_table_parity(16, 128, 264), 0, "Qwen3-0.6B rope shapes diverged");
    }

    /// The residual-fused and dual-pair GEMM epilogues (Phase-1 glue fusion)
    /// must be bit-identical to `matmat` + separate add / two matmats.
    #[test]
    fn test_gemm_resid_pair_bit_exact() {
        let _guard = tile_test_guard();
        unsafe {
            assert_eq!(sofuu_test_gemm_resid_pair_parity(64, 2048, 8), 0, "fused residual/pair paths diverged");
            assert_eq!(sofuu_test_gemm_resid_pair_parity(128, 1024, 16), 0, "second shape diverged");
        }
    }

    /// The Rust wrappers `matmat_residual` / `matmat_pair` must actually take
    /// the fused tile path (return true) for standard decoder shapes once the
    /// tiles are warmed — a regression here silently reverts to the unfused
    /// GEMM + separate residual-add, which is correct but slower.
    #[test]
    fn test_matmat_fused_wrappers_engage() {
        let _guard = tile_test_guard();
        let rows = 1024usize;
        let cols = 2048usize;
        let batch = 64usize;
        let mut w = vec![0u8; rows * (cols / 256) * 144];
        for (i, b) in w.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        tile_warm(w.as_ptr(), rows, cols, GGML_TYPE_Q4_K);
        let x = vec![0.35f32; batch * cols];
        let mut acc = vec![0.5f32; batch * rows];
        let fused = matmat_residual(
            &w,
            GGML_TYPE_Q4_K,
            rows,
            cols,
            &x,
            batch,
            &mut acc,
        );
        assert!(fused, "matmat_residual did not take the fused tile path");
        let mut g = vec![0f32; batch * rows];
        let mut u = vec![0f32; batch * rows];
        let xn = vec![0.1f32; batch * cols];
        let paired = matmat_pair(
            &w,
            &w,
            GGML_TYPE_Q4_K,
            cols / 2, // hidden-like rows must still be % 8 and % tile-friendly
            cols,
            &xn,
            batch,
            &mut g,
            &mut u,
        );
        assert!(paired, "matmat_pair did not take the fused tile path");
    }

    #[test]
    fn test_batched_attention_ext_swa() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 32;
        let max_seq_len = 64;
        let b_size = 2;
        let start_pos = 10;
        let layer_idx = 0;
        let sliding_window = 8;
        let sink_tokens = 2;

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q_batch = vec![0.1f32; b_size * q_dim];
        let k_cache = vec![0.2f32; max_seq_len * kv_dim];
        let v_cache = vec![0.3f32; max_seq_len * kv_dim];
        let mut head_out_batch = vec![0.0f32; b_size * q_dim];

        batched_attention_ext(
            &q_batch,
            &k_cache,
            &v_cache,
            &mut head_out_batch,
            b_size,
            start_pos,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq_len,
            sliding_window,
            sink_tokens,
        );

        for val in &head_out_batch {
            assert!(val.is_finite(), "expected finite output from attention");
            assert!(*val > 0.0, "expected positive attention output");
        }
    }

    /// Plain (no-window) prefill attention through the ext path with MORE
    /// rows than the 8-row QK group, so the batched-QK work domain (group
    /// boundaries, ragged per-row causal bounds) is exercised. The engine's
    /// softmax is the NEON vector exp (rel ~2e-6 peak), so the gate is
    /// peak-relative 1e-4 against an in-test scalar reference.
    #[test]
    fn test_batched_attention_ext_plain_rows8() {
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 32usize;
        let max_seq_len = 68usize;
        let b_size = 10usize;
        let start_pos = 54usize; // rows leave seqs 55..64, crossing a group of 8

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let n_rep = n_heads / n_kv_heads;
        let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();

        let mut rng = 91234u32;
        let mut next = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (((rng >> 9) % 4001) as i32 - 2000) as f32 / 512.0
        };
        let mut q_batch = vec![0f32; b_size * q_dim];
        let mut k_cache = vec![0f32; max_seq_len * kv_dim];
        let mut v_cache = vec![0f32; max_seq_len * kv_dim];
        for v in q_batch.iter_mut() {
            *v = next();
        }
        for v in k_cache.iter_mut() {
            *v = next();
        }
        for v in v_cache.iter_mut() {
            *v = next();
        }
        // one all-zero q row: exp(0-max) edge with an exact max tie at 0
        for v in q_batch[..q_dim].iter_mut() {
            *v = 0.0;
        }
        let mut head_out_batch = vec![0f32; b_size * q_dim];

        batched_attention_ext(
            &q_batch,
            &k_cache,
            &v_cache,
            &mut head_out_batch,
            b_size,
            start_pos,
            0,
            n_heads,
            n_kv_heads,
            head_dim,
            max_seq_len,
            0, // no sliding window → plain rows8 path
            0,
        );

        for b in 0..b_size {
            for h in 0..n_heads {
                let kv_h = h / n_rep;
                let seq = std::cmp::min(start_pos + b + 1, max_seq_len);
                let q_head = &q_batch[b * q_dim + h * head_dim..b * q_dim + (h + 1) * head_dim];
                let mut scores = vec![0f32; seq];
                for p in 0..seq {
                    // engine cache layout is head-major: [layer][kv_head][pos][dim]
                    let k = &k_cache[(kv_h * max_seq_len + p) * head_dim..(kv_h * max_seq_len + p + 1) * head_dim];
                    let mut d = 0f32;
                    for i in 0..head_dim {
                        d += q_head[i] * k[i];
                    }
                    scores[p] = d * inv_sqrt;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                let mut refo = vec![0f32; head_dim];
                for p in 0..seq {
                    let v = &v_cache[(kv_h * max_seq_len + p) * head_dim..(kv_h * max_seq_len + p + 1) * head_dim];
                    for j in 0..head_dim {
                        refo[j] += scores[p] * v[j];
                    }
                }
                let got = &head_out_batch[b * q_dim + h * head_dim..b * q_dim + (h + 1) * head_dim];
                let peak = refo.iter().fold(0f32, |a, x| a.max(x.abs())).max(1.0);
                let worst = got
                    .iter()
                    .zip(refo.iter())
                    .map(|(g, r)| ((g - r).abs()) as f64 / peak as f64)
                    .fold(0.0f64, f64::max);
                assert!(worst <= 1e-4, "rows8 plain attention diverged b={b} h={h}: {worst:.3e}");
            }
        }
    }

    /// Spark's fused-QKV prefill shape, exactly as spark.rs runs it: three
    /// zero-copy row-range matmats over one q4_K tensor, each warmed at its
    /// sub-range pointer (spark.rs:333) and dispatched with `matmat` —
    /// including the batch-78→80 pad-to-4 inside the GEMM. Guards the
    /// 9848995df Spark-corruption class: any tile/lookup/repack wrongness at
    /// these dims must show here as a reference mismatch.
    #[test]
    fn test_spark_qkv_cpp_gemm_parity() {
        let _guard = tile_test_guard();
        // variant knobs (env) for bisecting: 1 = warm only the q sub-range,
        // 2 = batch 8 (no pad), 3 = small dims
        let variant = std::env::var("SOFUU_SPARKTEST_VARIANT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        let dim = if variant == 3 { 512 } else { 2560usize };
        let q_dim = if variant == 3 { 64 } else { 4096usize };
        let kv_dim = if variant == 3 { 32 } else { 1024usize };
        let b_size = if variant == 2 {
            8
        } else if variant == 4 {
            80 // direct: tile_batch == batch_size, no pad branch
        } else if variant == 5 {
            6 // pad 6→8 at spark dims, single quantize group
        } else {
            78usize
        };
        let q_rows = dim / 256; // q4_K super-blocks per WEIGHT row (a row spans cols, not q_dim)
        let row_stride = q_rows * 144;
        let fused_rows = q_dim + 2 * kv_dim;

        let mut rng = 4242u32;
        let mut next_b = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 24) as u8
        };
        // fused tensor bytes; headers sanitized (finite f16 d/dmin)
        let mut w = vec![0u8; fused_rows * row_stride];
        for b in w.iter_mut() {
            *b = next_b();
        }
        for r in 0..fused_rows {
            for blk in 0..q_rows {
                let h = r * row_stride + blk * 144;
                let bits_d = ggml_fp32_to_fp16_bits(0.5f32 + 0.015625 * (r % 16) as f32);
                let bits_m = ggml_fp32_to_fp16_bits(0.0625f32 + 0.0078125 * ((r + blk) % 16) as f32);
                w[h..h + 2].copy_from_slice(&bits_d.to_le_bytes());
                w[h + 2..h + 4].copy_from_slice(&bits_m.to_le_bytes());
            }
        }

        // x activations, moderate magnitudes
        let mut x = vec![0f32; b_size * dim];
        for v in x.iter_mut() {
            *v = next_x(&mut rng);
        }

        // spark.rs warm: three sub-ranges at the exact pointers
        let base = w.as_ptr();
        tile_warm(base, q_dim, dim, GGML_TYPE_Q4_K);
        if variant != 1 {
            tile_warm(unsafe { base.add(q_dim * row_stride) }, kv_dim, dim, GGML_TYPE_Q4_K);
            tile_warm(
                unsafe { base.add((q_dim + kv_dim) * row_stride) },
                kv_dim,
                dim,
                GGML_TYPE_Q4_K,
            );
        }

        // dispatch exactly like project_qkv_batch
        let mut qy = vec![0f32; b_size * q_dim];
        let mut ky = vec![0f32; b_size * kv_dim];
        let mut vy = vec![0f32; b_size * kv_dim];
        matmat(&w[..], GGML_TYPE_Q4_K, q_dim, dim, &x, b_size, &mut qy);
        matmat(&w[q_dim * row_stride..], GGML_TYPE_Q4_K, kv_dim, dim, &x, b_size, &mut ky);
        matmat(
            &w[(q_dim + kv_dim) * row_stride..],
            GGML_TYPE_Q4_K,
            kv_dim,
            dim,
            &x,
            b_size,
            &mut vy,
        );

        // scalar reference: dequant each row, dot against x. The tile path
        // quantizes x to q8_K before the dot, so the reference dot on RAW x
        // differs by the act-quant rounding: the error is absolute-scale
        // (√dim × dx/2 × w-rms ≈ tens at this fixture) and only looks large
        // RELATIVE on near-cancelling dots — 8.6% at the smallest while the
        // same error is ~0.4% of a typical dot. Gate peak-relative per batch
        // column with an absolute floor; the 9848995df Spark corruption (27%
        // structural, 3893 absolute at b=0) stays far above it.
        let mut rowq = vec![0f32; dim];
        let mut got = vec![0f32; fused_rows * b_size];
        let mut refd = vec![0f32; fused_rows * b_size];
        for r in 0..fused_rows {
            dequantize_q4_k_row(&w[r * row_stride..], &mut rowq);
            for b in 0..b_size {
                let mut d = 0f32;
                for (i, rv) in rowq.iter().enumerate() {
                    d += rv * x[b * dim + i];
                }
                refd[r * b_size + b] = d;
                got[r * b_size + b] = if r < q_dim {
                    qy[b * q_dim + r]
                } else if r < q_dim + kv_dim {
                    ky[b * kv_dim + (r - q_dim)]
                } else {
                    vy[b * kv_dim + (r - q_dim - kv_dim)]
                };
            }
        }
        for b in 0..b_size {
            let colmax = (0..fused_rows)
                .map(|r| refd[r * b_size + b].abs())
                .fold(0f32, f32::max);
            for r in 0..fused_rows {
                let d = refd[r * b_size + b];
                let g = got[r * b_size + b];
                assert!(
                    (g - d).abs() <= 2e-2 * colmax + 150.0,
                    "qkv sub-range gemm diverged row={r} b={b}: {g:.6} vs {d:.6} colmax={colmax:.6}"
                );
            }
        }
    }

    fn next_x(rng: &mut u32) -> f32 {
        *rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (((*rng >> 9) % 2001) as i32 - 1000) as f32 / 512.0
    }

    fn ggml_fp32_to_fp16_bits(v: f32) -> u16 {
        // f16 bits via the std half rounding used by the engine's helpers
        let v = v as f64;
        let s = if v < 0.0 { 0x8000u16 } else { 0 };
        let a = v.abs();
        if a.is_infinite() || a >= 65504.0 {
            return s | 0x7C00;
        }
        if a == 0.0 {
            return s;
        }
        // exact for our test values (small dyadic rationals)
        let e = a.log2().floor();
        let m = (a / (2f64).powf(e - 10.0)).round();
        let re = (e + 15.0) as u16;
        let mm = (m as u16) & 0x3FF;
        s | (re << 10) | mm
    }

    #[test]
    fn test_ptq1_0_dequant_known_block() {
        // Hand-built one-block PTQ1_0 row: block = qs[24] | qh[2] | f16 d.
        // The reference quantizer packs 5 (qs) / 4 (qh) base-3 trits per byte
        // and pre-scales to a byte with ceiling division:
        //   qs byte = (q*256 + 242) / 243, qh byte = (q*3*256 + 242) / 243
        // (one extra *=3 shifts qh's 4 trits up). At stage level n the kernel
        // recovers trit floor(q/3^(4-n)) % 3 (qs) / floor(q/3^(3-n)) % 3 (qh)
        // as value (trit - 1) * d: the trit decomposition of the pre-scale q,
        // NOT of the byte itself (byte 3 encodes q=2, i.e. trits 2,0,0,0,0
        // msb-first, while base-3(3) would read 0,1,0,0,0).
        const D: f32 = 0.5;
        let mut block = [0u8; 28];
        block[27] = 0x38; // d = f16 0x3800 = 0.5 (little-endian, last 2 bytes)

        // Element m + n*c of a c-wide chunk carries trit (4-n) of its byte's
        // pre-scale q: stage 1 covers elements 0..80 (c=16), stage 2 covers
        // 80..120 (c=8, byte offset 16), qh pairs elements 120..128.
        //   q=0   (byte 0)   -> trits all 0      -> -d at 0,16,32,48,64
        //   q=242 (byte 255) -> trits all 2      -> +d at 1,17,33,49,65
        //   q=114 (byte 121) -> trits 1,1,0,2,0  -> 0, 0, -d, +d, -d
        //                                            at 2,18,34,50,66
        //   q=229 (byte 242) -> trits 2,2,1,1,1  -> +d, +d, 0, 0, 0
        //                                            at 80,88,96,104,112
        //   qh[0]=193 (qH=61)  -> trits 2,0,2,1  -> +d, -d, +d, 0
        //                                            at 120,122,124,126
        //   qh[1]=0   (qH=0)                     -> -d at 121,123,125,127
        block[0] = 0;
        block[1] = ((242 * 256 + 242) / 243) as u8; // 255
        block[2] = ((114 * 256 + 242) / 243) as u8; // 121
        block[16] = ((229 * 256 + 242) / 243) as u8; // 242
        block[24] = ((61 * 3 * 256 + 242) / 243) as u8; // 193

        let mut out = [0f32; 128];
        ptq1_0_dequant_row(&block, &mut out);

        // The modular extraction rule must reproduce the base-3 digits of the
        // byte's pre-image q at significance 3^(4-n) for a few bytes/levels:
        let pow3 = [1u8, 3, 9, 27, 81];
        let pow3hi = [81u32, 27, 9, 3, 1];
        for &(b, q) in &[(0u8, 0u32), (255, 242), (121, 114), (242, 229)] {
            for n in 0..5usize {
                let xi = ((b.wrapping_mul(pow3[n]) as u16) * 3 >> 8) as i32 - 1;
                let want = ((q / pow3hi[n]) % 3) as i32 - 1;
                assert_eq!(xi, want, "byte {b} trit {n}");
            }
        }
        let pow3h = [1u8, 3, 9, 27];
        let pow3hih = [27u32, 9, 3, 1];
        for n in 0..4usize {
            let xi = ((193u8.wrapping_mul(pow3h[n]) as u16) * 3 >> 8) as i32 - 1;
            let want = ((61 / pow3hih[n]) % 3) as i32 - 1;
            assert_eq!(xi, want, "qh byte 193 trit {n}");
        }

        // Full-row check against the hand-decoded values (every position not
        // listed decodes a zero byte -> -d):
        let mut expected = [-D; 128];
        for &i in &[1usize, 17, 33, 49, 65] {
            expected[i] = D;
        }
        for &(i, v) in &[
            (2usize, 0.0f32),
            (18, 0.0),
            (34, -D),
            (50, D),
            (66, -D),
            (80, D),
            (88, D),
            (96, 0.0),
            (104, 0.0),
            (112, 0.0),
            (120, D),
            (122, -D),
            (124, D),
            (126, 0.0),
        ] {
            expected[i] = v;
        }
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, expected[i], "element {i}");
        }
    }

    #[test]
    fn test_prefetch_streaming_and_temporal_safety() {
        // Null and zero bytes
        prefetch_streaming_bytes(std::ptr::null(), 0);
        prefetch_streaming_bytes(std::ptr::null(), 1024);
        prefetch_temporal_bytes(std::ptr::null(), 0);
        prefetch_temporal_bytes(std::ptr::null(), 1024);

        let empty: [u8; 0] = [];
        prefetch_streaming_slice(&empty);
        prefetch_temporal_slice(&empty);

        // Small buffer (< 64 bytes)
        let small = [42u8; 16];
        prefetch_streaming_slice(&small);
        prefetch_temporal_slice(&small);

        // Multi-page buffer (> 64 KB, exercises capping and multi-cacheline stepping)
        let buffer = vec![0x5au8; 128 * 1024];
        prefetch_streaming_slice(&buffer);
        prefetch_temporal_slice(&buffer);

        // Typed f32 slice
        let f32_buf = vec![1.234f32; 4096];
        prefetch_temporal_slice(&f32_buf);
    }
}

/// Batched prefill GEMM entries for the shared C++ kernels — the same ones
/// the GGUF backend's fast prefill uses — exposed for the other backends
/// (block_graph / gptx3 WeightMatrix). w is [rows][cols] row-major little-
/// endian; out is [batch][rows] token-major. Additive wrappers only.
pub(crate) fn gemm_batched_f32(
    w: &[u8], x: &[f32], out: &mut [f32], rows: usize, cols: usize, batch: usize,
) {
    unsafe {
        sofuu_gemm_f32(w.as_ptr() as *const f32, x.as_ptr(), out.as_mut_ptr(), rows, cols, batch);
    }
}

pub(crate) fn gemm_batched_f16(
    w: &[u8], x: &[f32], out: &mut [f32], rows: usize, cols: usize, batch: usize,
) {
    unsafe {
        sofuu_gemm_f16(w.as_ptr() as *const u16, x.as_ptr(), out.as_mut_ptr(), rows, cols, batch);
    }
}
