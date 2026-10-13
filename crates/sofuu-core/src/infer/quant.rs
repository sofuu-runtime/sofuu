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

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;

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

extern "C" {
    fn sofuu_set_threads(n: i32);
    fn sofuu_get_threads() -> i32;
    fn sofuu_dequantize_row_q4_0(src: *const u8, dst: *mut f32, k: usize);
    fn sofuu_dequantize_row_q8_0(src: *const u8, dst: *mut f32, k: usize);
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
    fn sofuu_gemm_f32(weight_bytes: *const f32, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_gemm_f16(weight_bytes: *const u16, x: *const f32, out: *mut f32, rows: usize, cols: usize, batch_size: usize);
    fn sofuu_rmsnorm(out: *mut f32, x: *const f32, weight: *const f32, n: usize, eps: f32);
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

    // Apple Silicon Metal Unified GPU Engine
    pub fn sofuu_metal_available() -> bool;
    pub fn sofuu_metal_init() -> *mut std::ffi::c_void;
    pub fn sofuu_metal_free(ctx: *mut std::ffi::c_void);
    pub fn sofuu_metal_buffer_create_zerocopy(
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

/// Out-place Root Mean Square Normalization (RMSNorm) via C++ SIMD.
#[inline]
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len());
    assert_eq!(out.len(), weight.len());
    unsafe {
        sofuu_rmsnorm(out.as_mut_ptr(), x.as_ptr(), weight.as_ptr(), x.len(), eps);
    }
}

/// In-place Root Mean Square Normalization (RMSNorm) via C++ SIMD.
#[inline]
pub fn rmsnorm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len());
    unsafe {
        sofuu_rmsnorm(x.as_mut_ptr(), x.as_ptr(), weight.as_ptr(), x.len(), eps);
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
}

