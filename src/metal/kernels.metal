#include <metal_stdlib>
using namespace metal;

static inline uint read_u32(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    if ((uintptr_t(p) & 3) == 0) {
        return *(device const uint*)p;
    } else {
        return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    }
}

/* bf16 -> f32 is a shift and a bit-cast, and spelling it out keeps this
 * shader clear of the `bfloat` builtin. That matters on iOS: a dispatch
 * whose kernel uses `bfloat` compiles but fails at execution on the
 * simulator's GPU (MTLCommandBufferErrorInternal), while this form runs
 * there and returns the same values — there is no other way to convert a
 * bf16 to a float32. */
static inline float read_bf16(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    ushort u;
    if ((uintptr_t(p) & 1) == 0) {
        u = *(device const ushort*)p;
    } else {
        u = ushort(p[0]) | (ushort(p[1]) << 8);
    }
    return as_type<float>((uint)u << 16);
}

static inline half read_f16(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    if ((uintptr_t(p) & 1) == 0) {
        return *(device const half*)p;
    } else {
        ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
        return as_type<half>(u);
    }
}

kernel void gemv_bf16(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint row_byte = w_rem + row * cols * 2;

    if (((uintptr_t(w_base) + w_rem) & 1) == 0) {
        device const ushort *row_w = (device const ushort*)(w_base + row_byte);
        uint i = lane * 4;
        for (; i + 4 <= cols; i += 32 * 4) {
            sum += as_type<float>((uint)row_w[i + 0] << 16) * x[i + 0];
            sum += as_type<float>((uint)row_w[i + 1] << 16) * x[i + 1];
            sum += as_type<float>((uint)row_w[i + 2] << 16) * x[i + 2];
            sum += as_type<float>((uint)row_w[i + 3] << 16) * x[i + 3];
        }
        for (; i < cols; ++i) {
            sum += as_type<float>((uint)row_w[i] << 16) * x[i];
        }
    } else {
        for (uint i = lane; i < cols; i += 32) {
            sum += read_bf16(w_base, row_byte + i * 2) * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_f16(
    device const uchar *w_base          [[buffer(0)]],
    device const float *x               [[buffer(1)]],
    device float       *y               [[buffer(2)]],
    constant uint      &cols            [[buffer(3)]],
    constant uint      &w_rem           [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint row_byte = w_rem + row * cols * 2;

    if (((uintptr_t(w_base) + w_rem) & 1) == 0) {
        device const half *row_w = (device const half*)(w_base + row_byte);
        uint i = lane * 4;
        for (; i + 4 <= cols; i += 32 * 4) {
            sum += float(row_w[i + 0]) * x[i + 0];
            sum += float(row_w[i + 1]) * x[i + 1];
            sum += float(row_w[i + 2]) * x[i + 2];
            sum += float(row_w[i + 3]) * x[i + 3];
        }
        for (; i < cols; ++i) {
            sum += float(row_w[i]) * x[i];
        }
    } else {
        for (uint i = lane; i < cols; i += 32) {
            sum += float(read_f16(w_base, row_byte + i * 2)) * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_f32(
    device const uchar *w_base          [[buffer(0)]],
    device const float *x               [[buffer(1)]],
    device float       *y               [[buffer(2)]],
    constant uint      &cols            [[buffer(3)]],
    constant uint      &w_rem           [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint row_byte = w_rem + row * cols * 4;

    if (((uintptr_t(w_base) + w_rem) & 3) == 0) {
        device const float *row_w = (device const float*)(w_base + row_byte);
        uint i = lane * 4;
        for (; i + 4 <= cols; i += 32 * 4) {
            sum += row_w[i + 0] * x[i + 0];
            sum += row_w[i + 1] * x[i + 1];
            sum += row_w[i + 2] * x[i + 2];
            sum += row_w[i + 3] * x[i + 3];
        }
        for (; i < cols; ++i) {
            sum += row_w[i] * x[i];
        }
    } else {
        for (uint i = lane; i < cols; i += 32) {
            float val = as_type<float>(read_u32(w_base, row_byte + i * 4));
            sum += val * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void rmsnorm(
    device const float  *x              [[buffer(0)]],
    device const uchar  *w_base         [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &dim            [[buffer(3)]],
    constant float      &eps            [[buffer(4)]],
    constant uint       &w_rem          [[buffer(5)]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        float val = x[i];
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(dim) + eps);

    for (uint i = lane; i < dim; i += 32) {
        y[i] = x[i] * rms * read_bf16(w_base, w_rem + i * 2);
    }
}

// Elementwise ops used by BOTH the per-op path (internal scratch x/y buffers,
// where an unguarded tail is harmless) and the batch path (slots inside a
// shared slab, where a tail would clobber the next live slot) — hence the
// `n` guard. Grid rounding stops at the guard; it never spills.
kernel void swiglu(
    device float       *g               [[buffer(0)]],
    device const float *u               [[buffer(1)]],
    constant uint      &n               [[buffer(2)]],
    uint id                             [[thread_position_in_grid]]
) {
    if (id >= n) { return; }
    float val = g[id];
    float silu = val / (1.0f + exp(-val));
    g[id] = silu * u[id];
}

kernel void add_inplace(
    device float       *acc             [[buffer(0)]],
    device const float *add             [[buffer(1)]],
    constant uint      &n               [[buffer(2)]],
    uint id                             [[thread_position_in_grid]]
) {
    if (id >= n) { return; }
    acc[id] += add[id];
}

kernel void conv1d_3(
    device const uchar *conv_w_base     [[buffer(0)]],
    device const float *state           [[buffer(1)]],
    device float       *out             [[buffer(2)]],
    constant uint      &w_rem           [[buffer(3)]],
    uint c                              [[thread_position_in_grid]]
) {
    uint base = w_rem + c * 3 * 2;
    float w0 = read_bf16(conv_w_base, base + 0);
    float w1 = read_bf16(conv_w_base, base + 2);
    float w2 = read_bf16(conv_w_base, base + 4);

    device const float *st = state + c * 3;
    float sum = w0 * st[0] + w1 * st[1] + w2 * st[2];
    out[c] = sum;
}

kernel void gemv_mlx4_affine_f16(
    device const uchar  *w_base         [[buffer(0)]],
    device const uchar  *scales_base    [[buffer(1)]],
    device const uchar  *biases_base    [[buffer(2)]],
    device const float  *x              [[buffer(3)]],
    device float        *y              [[buffer(4)]],
    constant uint       &cols           [[buffer(5)]],
    constant uint       &group_size     [[buffer(6)]],
    constant uint4      &rems           [[buffer(7)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint u32_cols = cols / 8;
    uint n_groups = cols / group_size;

    uint qw_row_byte = rems.x + row * (u32_cols * 4);
    uint s_row_byte  = rems.y + row * (n_groups * 2);
    uint b_row_byte  = rems.z + row * (n_groups * 2);

    for (uint col_u32 = lane; col_u32 < u32_cols; col_u32 += 32) {
        uint val = read_u32(w_base, qw_row_byte + col_u32 * 4);
        uint c_base = col_u32 * 8;
        uint g_idx = c_base / group_size;
        float s = float(read_f16(scales_base, s_row_byte + g_idx * 2));
        float b = float(read_f16(biases_base, b_row_byte + g_idx * 2));

        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint nibble = (val >> (i * 4)) & 0x0F;
            float w_val = float(nibble) * s + b;
            sum += w_val * x[c_base + i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_mlx4_affine_bf16(
    device const uchar  *w_base         [[buffer(0)]],
    device const uchar  *scales_base    [[buffer(1)]],
    device const uchar  *biases_base    [[buffer(2)]],
    device const float  *x              [[buffer(3)]],
    device float        *y              [[buffer(4)]],
    constant uint       &cols           [[buffer(5)]],
    constant uint       &group_size     [[buffer(6)]],
    constant uint4      &rems           [[buffer(7)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint u32_cols = cols / 8;
    uint n_groups = cols / group_size;

    uint qw_row_byte = rems.x + row * (u32_cols * 4);
    uint s_row_byte  = rems.y + row * (n_groups * 2);
    uint b_row_byte  = rems.z + row * (n_groups * 2);

    for (uint col_u32 = lane; col_u32 < u32_cols; col_u32 += 32) {
        uint val = read_u32(w_base, qw_row_byte + col_u32 * 4);
        uint c_base = col_u32 * 8;
        uint g_idx = c_base / group_size;
        float s = read_bf16(scales_base, s_row_byte + g_idx * 2);
        float b = read_bf16(biases_base, b_row_byte + g_idx * 2);

        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint nibble = (val >> (i * 4)) & 0x0F;
            float w_val = float(nibble) * s + b;
            sum += w_val * x[c_base + i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void lfm2_conv_gate(
    device const float *bcx           [[buffer(0)]],
    device const uchar *conv_w_base   [[buffer(1)]],
    device float       *state         [[buffer(2)]],
    device float       *out           [[buffer(3)]],
    constant uint      &dim           [[buffer(4)]],
    constant uint      &w_rem         [[buffer(5)]],
    uint c                              [[thread_position_in_grid]]
) {
    float b = bcx[c];
    float gate = bcx[dim + c];
    float xv = bcx[2 * dim + c];
    float u = b * xv;

    float s0 = state[2 * c + 0];
    float s1 = state[2 * c + 1];

    uint base = w_rem + c * 3 * 2;
    float v = s0 * read_bf16(conv_w_base, base + 0)
            + s1 * read_bf16(conv_w_base, base + 2)
            + u  * read_bf16(conv_w_base, base + 4);

    state[2 * c + 0] = s1;
    state[2 * c + 1] = u;
    out[c] = gate * v;
}

// Per-KV-head RMSNorm + split-half RoPE + KV-cache append, one threadgroup
// (32 threads) per kv head. Runs AFTER the qkv projections and BEFORE the
// attention kernel inside the same batch. k is normalized in place (the CPU
// path does the same scratch overwrite), then RoPE'd, then the finished k/v
// rows are stored at position `pos` of this layer's cache slice.
//   Norm halves and rope halves cross lanes (j and j+head_dim/2 usually live
//   in different lanes), so device-memory barriers separate the three phases.
kernel void lfm2_kv_write(
    device float       *k              [[buffer(0)]],
    device const float *v              [[buffer(1)]],
    device float       *k_cache_layer  [[buffer(2)]],
    device float       *v_cache_layer  [[buffer(3)]],
    device const uchar *k_ln_base      [[buffer(4)]],
    constant uint      &k_ln_rem       [[buffer(5)]],
    constant uint      &pos            [[buffer(6)]],
    constant uint      &head_dim       [[buffer(7)]],
    constant uint      &n_kv_heads     [[buffer(8)]],
    constant uint      &max_seq        [[buffer(9)]],
    constant float     &eps            [[buffer(10)]],
    constant float     &rope_base      [[buffer(11)]],
    uint kv                             [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    device float *k_head = k + kv * head_dim;

    float sum_sq = 0.0f;
    for (uint i = lane; i < head_dim; i += 32) {
        sum_sq += k_head[i] * k_head[i];
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(head_dim) + eps);
    for (uint i = lane; i < head_dim; i += 32) {
        k_head[i] = k_head[i] * rms * read_bf16(k_ln_base, k_ln_rem + i * 2);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint j = lane; j < head_dim / 2; j += 32) {
        float theta = float(pos) / pow(rope_base, (2.0f * float(j)) / float(head_dim));
        float cosv = cos(theta);
        float sinv = sin(theta);
        float k0 = k_head[j];
        float k1 = k_head[j + head_dim / 2];
        k_head[j]                = k0 * cosv - k1 * sinv;
        k_head[j + head_dim / 2] = k1 * cosv + k0 * sinv;
    }

    // Cache copy may read dims another lane just rope'd.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    device float *k_dst = k_cache_layer + ((size_t)kv * max_seq + pos) * head_dim;
    device float *v_dst = v_cache_layer + ((size_t)kv * max_seq + pos) * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        k_dst[i] = k_head[i];
        v_dst[i] = v[kv * head_dim + i];
    }
}

// GQA decode attention, one threadgroup (32 threads) per q head:
//   q <- head-RMSNorm(q) * q_ln_w, RoPE(q), then
//   softmax(q . K_cache) * V_cache over t in [0, pos].
// k/v must already be normed+rope'd and stored by lfm2_kv_write (barriered by
// the encoder between the two dispatches — same contract as chained GEMVs).
// q is overwritten in place, same scratch contract as the CPU path.
// Scores live in threadgroup memory: max_seq <= 4096 -> 16 KiB, inside the
// 32 KiB automatic threadgroup-storage budget of Apple silicon GPUs.
kernel void lfm2_attn(
    device float       *q               [[buffer(0)]],
    device const float *k_cache_layer   [[buffer(1)]],
    device const float *v_cache_layer   [[buffer(2)]],
    device const uchar *q_ln_base       [[buffer(3)]],
    device float       *head_out        [[buffer(4)]],
    constant uint      &q_ln_rem        [[buffer(5)]],
    constant uint      &pos             [[buffer(6)]],
    constant uint      &head_dim        [[buffer(7)]],
    constant uint      &n_heads         [[buffer(8)]],
    constant uint      &n_kv_heads      [[buffer(9)]],
    constant uint      &max_seq         [[buffer(10)]],
    constant float     &eps             [[buffer(11)]],
    constant float     &rope_base       [[buffer(12)]],
    uint h                              [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    device float *q_head = q + h * head_dim;
    float scale = rsqrt(float(head_dim));

    // Head RMSNorm in place.
    float sum_sq = 0.0f;
    for (uint i = lane; i < head_dim; i += 32) {
        sum_sq += q_head[i] * q_head[i];
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(head_dim) + eps);
    for (uint i = lane; i < head_dim; i += 32) {
        q_head[i] = q_head[i] * rms * read_bf16(q_ln_base, q_ln_rem + i * 2);
    }

    // Split-half RoPE in place.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint j = lane; j < head_dim / 2; j += 32) {
        float theta = float(pos) / pow(rope_base, (2.0f * float(j)) / float(head_dim));
        float cosv = cos(theta);
        float sinv = sin(theta);
        float q0 = q_head[j];
        float q1 = q_head[j + head_dim / 2];
        q_head[j]                = q0 * cosv - q1 * sinv;
        q_head[j + head_dim / 2] = q1 * cosv + q0 * sinv;
    }

    // Attention reads every dim of q; rope partners crossed lanes.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint kv = h / (n_heads / n_kv_heads);
    device const float *kbase = k_cache_layer + (size_t)kv * max_seq * head_dim;
    device const float *vbase = v_cache_layer + (size_t)kv * max_seq * head_dim;
    uint seq = pos + 1;

    threadgroup float scores[4096];

    // Scores: lane t-strides the sequence, full dot per owned t.
    float lmax = -INFINITY;
    for (uint t = lane; t < seq; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        float score = dot * scale;
        scores[t] = score;
        lmax = max(lmax, score);
    }
    float max_score = simd_max(lmax);

    // Softmax with the same max-subtraction as the CPU path.
    float lsum = 0.0f;
    for (uint t = lane; t < seq; t += 32) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        lsum += e;
    }
    float inv = 1.0f / simd_sum(lsum);

    // Weighted V: lane owns output dims lane, lane+32, ... (head_dim <= 128
    // -> at most 4 accumulators), walking t serially — no further syncs.
    float acc[4];
    uint n_acc = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        acc[n_acc++] = 0.0f;
    }
    for (uint t = 0; t < seq; ++t) {
        float w = scores[t] * inv;
        device const float *vt = vbase + (size_t)t * head_dim;
        n_acc = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[n_acc++] += w * vt[d];
        }
    }
    n_acc = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        head_out[h * head_dim + d] = acc[n_acc++];
    }
}

// ── G2: GGUF dequant-dot GEMV kernels (q4_0 / q8_0 / q4_K / q6_K) ──────────
// One row per 32-thread simdgroup, blocks lane-strided, one simd_sum at the
// end (the parallel partial order is tolerance-gated exactly like the landed
// mlx4 kernels; the DEQUANT itself is integer-exact so values match the CPU
// bit-for-bit before summation). rems: x = w byte remainder, y/z unused
// placeholders (0) to keep the uint4 binding shape of the mlx4 kernels.
// Rows arrive via rems.w for the dispatch grid edge guard.

kernel void gemv_q8_0(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // rows dispatched as row/8 threadgroups of 8 simds? No: one row per
    // threadgroup of 32 lanes; blocks of 32 int8 walked lane-strided.
    float sum = 0.0f;
    uint row_byte = w_rem + row * (cols / 32) * 34;
    uint nb = cols / 32;
    for (uint b = lane; b < nb; b += 32) {
        float d = float(read_f16(w_base, row_byte + b * 34));
        device const uchar *qs = w_base + row_byte + b * 34 + 2;
        float acc = 0.0f;
        for (uint j = 0; j < 32; ++j) {
            float w = float((char)qs[j]) * d;
            acc += w * x[b * 32 + j];
        }
        sum += acc;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_q4_0(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum = 0.0f;
    uint row_byte = w_rem + row * (cols / 32) * 18;
    uint nb = cols / 32;
    for (uint b = lane; b < nb; b += 32) {
        float d = float(read_f16(w_base, row_byte + b * 18));
        device const uchar *qs = w_base + row_byte + b * 18 + 2;
        float acc = 0.0f;
        for (uint j = 0; j < 16; ++j) {
            float w0 = (float)((char)((qs[j] & 0x0F) - 8)) * d;
            float w1 = (float)((char)((qs[j] >> 4) - 8)) * d;
            acc += w0 * x[b * 32 + j] + w1 * x[b * 32 + j + 16];
        }
        sum += acc;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

static inline void q4k_scale_min(device const uchar *scales, uint j, thread uint8_t *d, thread uint8_t *m) {
    if (j < 4) {
        *d = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

kernel void gemv_q4_k(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // 144 bytes per 256-value block: f16 d + f16 dmin + 12 scale bytes
    // (8 4-bit lows + 8 3-bit highs over 12 bytes) + 128 nibble bytes.
    // Dequant mirrors sofuu_dequantize_row_q4_k exactly: w = nib*dsc - dmin*m.
    float sum = 0.0f;
    uint row_byte = w_rem + row * (cols / 256) * 144;
    uint nb = cols / 256;
    for (uint b = lane; b < nb; b += 32) {
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        float acc = 0.0f;
        uint is = 0;
        // Literal transcription of the C scalar reference
        // (sofuu_dequantize_row_q4_k, #else branch): for l in 0..32,
        //   y[j+l]      = d1*(q[l] & 0x0F) - m1
        //   y[j+32+l]   = d2*(q[l] >> 4)   - m2
        // with q the 32-byte quant span of the 64-value group (two values
        // per byte) and (d1,m1)/(d2,m2) the scales at is/is+1.
        for (uint j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, is + 0, &sc1, &m1);
            float d1 = d * (float)sc1;
            float mm1 = dmin * (float)m1;
            q4k_scale_min(scales, is + 1, &sc2, &m2);
            float d2 = d * (float)sc2;
            float mm2 = dmin * (float)m2;
            uint qb = j / 2;
            uint xb = b * 256;   // x offset of this block
            for (uint l = 0; l < 32; ++l) {
                acc += d1 * (float)(qs[qb + l] & 0x0F) * x[xb + j + l] - mm1 * x[xb + j + l];
                acc += d2 * (float)(qs[qb + l] >> 4) * x[xb + j + 32 + l] - mm2 * x[xb + j + 32 + l];
            }
            is += 2;
        }
        sum += acc;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

kernel void gemv_q6_k(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // 210 bytes per 256-value block: ql[128] low nibbles, qh[64] upper 2
    // bits, 16 int8 scales, f16 d. Mirrors sofuu_dequantize_row_q6_k's
    // scalar reference exactly: per half n (128 values) and l in 0..32,
    //   q1 = ql[64n+l]&F    | qh bits0-1  at y[l+0],   scale sc[8n + l/16 + 0]
    //   q2 = ql[64n+l+32]&F | qh bits2-3  at y[l+32],  scale sc[8n + l/16 + 2]
    //   q3 = ql[64n+l]>>4   | qh bits4-5  at y[l+64],  scale sc[8n + l/16 + 4]
    //   q4 = ql[64n+l+32]>>4| qh bits6-7  at y[l+96],  scale sc[8n + l/16 + 6]
    // value = (6-bit - 32) * (d * scale); dot against x.
    float sum = 0.0f;
    uint row_byte = w_rem + row * (cols / 256) * 210;
    uint nb = cols / 256;
    for (uint b = lane; b < nb; b += 32) {
        // Layout: ql[128] @0, qh[64] @128, scales[16] @192, d f16 @208.
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = w_base + row_byte + b * 210 + 128;
        device const uchar *sc = w_base + row_byte + b * 210 + 192;
        float acc = 0.0f;
        uint xb6 = b * 256;   // x offset of this block
        for (uint n = 0; n < 2; ++n) {
            uint qlb = n * 64;
            uint qhb = n * 32;
            uint vb = n * 128;
            for (uint l = 0; l < 32; ++l) {
                uchar qh8 = qh[qhb + l];
                uint isn = 8 * n + l / 16;
                float s0 = d * (float)(signed char)sc[isn + 0];
                float s2 = d * (float)(signed char)sc[isn + 2];
                float s4 = d * (float)(signed char)sc[isn + 4];
                float s6 = d * (float)(signed char)sc[isn + 6];
                float q1 = (float)((ql[qlb + l] & 0x0F) | (((qh8 >> 0) & 3) << 4)) - 32.0f;
                float q2 = (float)((ql[qlb + l + 32] & 0x0F) | (((qh8 >> 2) & 3) << 4)) - 32.0f;
                float q3 = (float)((ql[qlb + l] >> 4) | (((qh8 >> 4) & 3) << 4)) - 32.0f;
                float q4 = (float)((ql[qlb + l + 32] >> 4) | (((qh8 >> 6) & 3) << 4)) - 32.0f;
                acc += q1 * s0 * x[xb6 + vb + l]
                     + q2 * s2 * x[xb6 + vb + l + 32]
                     + q3 * s4 * x[xb6 + vb + l + 64]
                     + q4 * s6 * x[xb6 + vb + l + 96];
            }
        }
        sum += acc;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

// ── GPU3 helper: per-head f32 RMSNorm on a [heads x head_dim] slot ─────────
// (q-norm for the GGUF shape; done as its own dispatch so the chunked
// attention kernel reads a fully normalized q with no intra-dispatch race).
kernel void head_rmsnorm(
    device float       *heads           [[buffer(0)]],
    device const float *w               [[buffer(1)]],
    constant uint      &w_rem           [[buffer(2)]],
    constant uint      &head_dim        [[buffer(3)]],
    constant uint      &n_heads         [[buffer(4)]],
    constant float     &eps             [[buffer(5)]],
    constant uint      &has_norm        [[buffer(6)]],
    constant uint      &pos             [[buffer(7)]],
    constant float     &rope_base       [[buffer(8)]],
    constant uint      &rope_mode       [[buffer(9)]],
    uint h                              [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // Per-head f32 RMSNorm (q-norm, when present) + RoPE — one dispatch so
    // the chunked attention reads fully processed q with no internal race.
    if (h >= n_heads) { return; }
    device float *row = heads + (size_t)h * head_dim;
    if (has_norm != 0u) {
        float ss = 0.0f;
        for (uint i = lane; i < head_dim; i += 32) {
            ss += row[i] * row[i];
        }
        ss = simd_sum(ss);
        float rms = rsqrt(ss / float(head_dim) + eps);
        for (uint i = lane; i < head_dim; i += 32) {
            row[i] = row[i] * rms * w[w_rem + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = lane; j < head_dim / 2; j += 32) {
        float theta = float(pos) / pow(rope_base, (2.0f * float(j)) / float(head_dim));
        float c = cos(theta);
        float s = sin(theta);
        if (rope_mode == 0u) {
            float r0 = row[j];
            float r1 = row[j + head_dim / 2];
            row[j]                = r0 * c - r1 * s;
            row[j + head_dim / 2] = r1 * c + r0 * s;
        } else {
            float r0 = row[2 * j];
            float r1 = row[2 * j + 1];
            row[2 * j]     = r0 * c - r1 * s;
            row[2 * j + 1] = r1 * c + r0 * s;
        }
    }
}

// ── GPU3: chunked decode attention (replaces serial per-head llama_attn) ───
// One threadgroup per (q head, 128-position chunk): lanes stride the chunk's
// positions with 128-wide unrolled dots (short dependency chains, real
// memory parallelism), softmax-normalized PARTIAL weighted-V sums and
// (chunk max, chunk sum) go to meta; a finish kernel rescales and sums the
// chunk partials per head — deterministic, same two-pass softmax semantics
// as the CPU (global max over chunks), tolerance-gated like every kernel.
// meta layout (f32): [h][c][CHUNK] partial V-accum (128 dims per chunk),
//                    [head_off] = heads*chunks*CHUNK → per-head {max, sum}.
#define ATTN_CHUNK 128

kernel void llama_attn_part(
    device float       *q               [[buffer(0)]],
    device const float *k_cache_layer   [[buffer(1)]],
    device const float *v_cache_layer   [[buffer(2)]],
    device const float *q_ln            [[buffer(3)]],
    device float       *meta            [[buffer(4)]],
    constant uint      &q_ln_rem        [[buffer(5)]],
    constant uint      &has_norm        [[buffer(6)]],
    constant uint      &pos             [[buffer(7)]],
    constant uint      &head_dim        [[buffer(8)]],
    constant uint      &n_heads         [[buffer(9)]],
    constant uint      &n_kv_heads      [[buffer(10)]],
    constant uint      &max_seq         [[buffer(11)]],
    constant float     &eps             [[buffer(12)]],
    constant float     &rope_base       [[buffer(13)]],
    constant uint      &rope_mode       [[buffer(14)]],
    uint g                              [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // NOTE: this kernel requires n_heads % 32 == 0 handled by grid math and
    // head_dim <= 128; threadgroup = 32 lanes; chunk = 128 positions.
    uint n_chunks = (pos + 1 + ATTN_CHUNK - 1) / ATTN_CHUNK;
    uint h = g / n_chunks;
    uint c = g % n_chunks;
    if (h >= n_heads) { return; }
    device float *q_head = q + h * head_dim;
    float scale = rsqrt(float(head_dim));
    if (c == 0 && has_norm != 0u) {
        float ss = 0.0f;
        for (uint i = lane; i < head_dim; i += 32) {
            ss += q_head[i] * q_head[i];
        }
        ss = simd_sum(ss);
        float rms = rsqrt(ss / float(head_dim) + eps);
        for (uint i = lane; i < head_dim; i += 32) {
            q_head[i] = q_head[i] * rms * q_ln[q_ln_rem + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint kv = h / (n_heads / n_kv_heads);
    device const float *kbase = k_cache_layer + (size_t)kv * max_seq * head_dim;
    device const float *vbase = v_cache_layer + (size_t)kv * max_seq * head_dim;
    uint t0 = c * ATTN_CHUNK;
    uint t_end = min(t0 + ATTN_CHUNK, pos + 1);

    // Pass 1: one lane per position (lane owns positions lane, lane+32, ...)
    // dot over head_dim, softmax numerator relative to the CHUNK max.
    float mx = -INFINITY;
    for (uint t = t0 + lane; t < t_end; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        if (dot * scale > mx) { mx = dot * scale; }
    }
    float cmax = simd_max(mx);

    float lsum = 0.0f;
    for (uint t = t0 + lane; t < t_end; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        lsum += exp(dot * scale - cmax);
    }
    float csum = simd_sum(lsum);

    if (lane == 0) {
        meta[(size_t)h * n_chunks * 2 + c * 2 + 0] = cmax;
        meta[(size_t)h * n_chunks * 2 + c * 2 + 1] = csum;
    }

    // Pass 2: stage per-position weights in threadgroup memory (one write).
    threadgroup float tw[ATTN_CHUNK];
    for (uint t = t0 + lane; t < t_end; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        tw[t - t0] = exp(dot * scale - cmax);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: lane owns dims lane, lane+32, ... — coalesced V reads, one
    // load+FMA per position; normalized partial for the finish rescale.
    size_t parts = (size_t)h * n_chunks * head_dim;
    float inv = 1.0f / csum;
    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (uint t = t0; t < t_end; ++t) {
            acc += tw[t - t0] * vbase[(size_t)t * head_dim + d];
        }
        meta[parts + (size_t)c * head_dim + d] = acc * inv;
    }
}

kernel void llama_attn_fin(
    device float       *meta            [[buffer(0)]],
    device float       *head_out        [[buffer(1)]],
    constant uint      &n_heads         [[buffer(2)]],
    constant uint      &head_dim        [[buffer(3)]],
    constant uint      &pos             [[buffer(4)]],
    uint g                              [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // out[d] = (1/gsum) * sum_c csum_c * exp(cmax_c - gmax) * partial[c][d]
    uint n_chunks = (pos + 1 + ATTN_CHUNK - 1) / ATTN_CHUNK;
    uint h = g;
    if (h >= n_heads) { return; }
    size_t mbase = (size_t)h * n_chunks * 2;
    float gmax = -INFINITY;
    for (uint c = 0; c < n_chunks; ++c) {
        gmax = max(gmax, meta[mbase + c * 2 + 0]);
    }
    float gsum = 0.0f;
    for (uint c = 0; c < n_chunks; ++c) {
        gsum += meta[mbase + c * 2 + 1] * exp(meta[mbase + c * 2 + 0] - gmax);
    }
    float inv = 1.0f / gsum;
    size_t parts = (size_t)h * n_chunks * head_dim;
    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (uint c = 0; c < n_chunks; ++c) {
            float coef = exp(meta[mbase + c * 2 + 0] - gmax) * meta[mbase + c * 2 + 1];
            acc += meta[parts + (size_t)c * head_dim + d] * coef;
        }
        head_out[h * head_dim + d] = acc * inv;
    }
}

// ── GPU2.1: one-commit-per-layer glue for the llama/GGUF shape ─────────────
// RMSNorm with F32 weights (GGUF norm tensors are f32), slot addresses via
// 16-byte-aligned buffer offsets; weight element remainder passed as a
// constant (offsets are 4-byte aligned, verified at arm time).

kernel void rmsnorm_f32w(
    device const float *x              [[buffer(0)]],
    device const float *w              [[buffer(1)]],
    device float       *y              [[buffer(2)]],
    constant uint      &dim            [[buffer(3)]],
    constant float     &eps            [[buffer(4)]],
    constant uint      &w_rem          [[buffer(5)]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float sum_sq = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = lane; i < dim; i += 32) {
        y[i] = x[i] * rms * w[w_rem + i];
    }
}

// Per-kv-head: optional head-RMSNorm (f32), split-half (mode 0) or
// interleaved (mode 1) RoPE on k, then append k (processed) and v (raw)
// at position `pos` of this layer's cache slice.
kernel void llama_kv_write(
    device float       *k              [[buffer(0)]],
    device const float *v              [[buffer(1)]],
    device float       *k_cache_layer  [[buffer(2)]],
    device float       *v_cache_layer  [[buffer(3)]],
    device const float *k_ln           [[buffer(4)]],
    constant uint      &k_ln_rem       [[buffer(5)]],
    constant uint      &has_norm       [[buffer(6)]],
    constant uint      &pos            [[buffer(7)]],
    constant uint      &head_dim       [[buffer(8)]],
    constant uint      &n_kv_heads     [[buffer(9)]],
    constant uint      &max_seq        [[buffer(10)]],
    constant float     &eps            [[buffer(11)]],
    constant float     &rope_base      [[buffer(12)]],
    constant uint      &rope_mode      [[buffer(13)]],
    uint kv                             [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    device float *k_head = k + kv * head_dim;
    if (has_norm != 0u) {
        float ss = 0.0f;
        for (uint i = lane; i < head_dim; i += 32) {
            ss += k_head[i] * k_head[i];
        }
        ss = simd_sum(ss);
        float rms = rsqrt(ss / float(head_dim) + eps);
        for (uint i = lane; i < head_dim; i += 32) {
            k_head[i] = k_head[i] * rms * k_ln[k_ln_rem + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    for (uint j = lane; j < head_dim / 2; j += 32) {
        float theta = float(pos) / pow(rope_base, (2.0f * float(j)) / float(head_dim));
        float c = cos(theta);
        float s = sin(theta);
        if (rope_mode == 0u) {
            float k0 = k_head[j];
            float k1 = k_head[j + head_dim / 2];
            k_head[j]                = k0 * c - k1 * s;
            k_head[j + head_dim / 2] = k1 * c + k0 * s;
        } else {
            float k0 = k_head[2 * j];
            float k1 = k_head[2 * j + 1];
            k_head[2 * j]     = k0 * c - k1 * s;
            k_head[2 * j + 1] = k1 * c + k0 * s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    device float *k_dst = k_cache_layer + ((size_t)kv * max_seq + pos) * head_dim;
    device float *v_dst = v_cache_layer + ((size_t)kv * max_seq + pos) * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        k_dst[i] = k_head[i];
        v_dst[i] = v[kv * head_dim + i];
    }
}

// Per-q-head GQA decode attention: optional head-norm + RoPE on q, softmax
// over the FULL cache prefix (full-attention layers only — windowed layers
// keep the 3-commit fallback), weighted V into head_out.
kernel void llama_attn(
    device float       *q               [[buffer(0)]],
    device const float *k_cache_layer   [[buffer(1)]],
    device const float *v_cache_layer   [[buffer(2)]],
    device const float *q_ln            [[buffer(3)]],
    device float       *head_out        [[buffer(4)]],
    constant uint      &q_ln_rem        [[buffer(5)]],
    constant uint      &has_norm        [[buffer(6)]],
    constant uint      &pos             [[buffer(7)]],
    constant uint      &head_dim        [[buffer(8)]],
    constant uint      &n_heads         [[buffer(9)]],
    constant uint      &n_kv_heads      [[buffer(10)]],
    constant uint      &max_seq         [[buffer(11)]],
    constant float     &eps             [[buffer(12)]],
    constant float     &rope_base       [[buffer(13)]],
    constant uint      &rope_mode       [[buffer(14)]],
    uint h                              [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    device float *q_head = q + h * head_dim;
    float scale = rsqrt(float(head_dim));
    if (has_norm != 0u) {
        float ss = 0.0f;
        for (uint i = lane; i < head_dim; i += 32) {
            ss += q_head[i] * q_head[i];
        }
        ss = simd_sum(ss);
        float rms = rsqrt(ss / float(head_dim) + eps);
        for (uint i = lane; i < head_dim; i += 32) {
            q_head[i] = q_head[i] * rms * q_ln[q_ln_rem + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    for (uint j = lane; j < head_dim / 2; j += 32) {
        float theta = float(pos) / pow(rope_base, (2.0f * float(j)) / float(head_dim));
        float c = cos(theta);
        float s = sin(theta);
        if (rope_mode == 0u) {
            float q0 = q_head[j];
            float q1 = q_head[j + head_dim / 2];
            q_head[j]                = q0 * c - q1 * s;
            q_head[j + head_dim / 2] = q1 * c + q0 * s;
        } else {
            float q0 = q_head[2 * j];
            float q1 = q_head[2 * j + 1];
            q_head[2 * j]     = q0 * c - q1 * s;
            q_head[2 * j + 1] = q1 * c + q0 * s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint kv = h / (n_heads / n_kv_heads);
    device const float *kbase = k_cache_layer + (size_t)kv * max_seq * head_dim;
    device const float *vbase = v_cache_layer + (size_t)kv * max_seq * head_dim;
    uint seq = pos + 1;
    threadgroup float scores[4096];

    float lmax = -INFINITY;
    for (uint t = lane; t < seq; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        float score = dot * scale;
        scores[t] = score;
        lmax = max(lmax, score);
    }
    float max_score = simd_max(lmax);

    float lsum = 0.0f;
    for (uint t = lane; t < seq; t += 32) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        lsum += e;
    }
    float inv = 1.0f / simd_sum(lsum);

    float acc_v[4];
    uint n_acc = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        acc_v[n_acc++] = 0.0f;
    }
    for (uint t = 0; t < seq; ++t) {
        float w = scores[t] * inv;
        device const float *vt = vbase + (size_t)t * head_dim;
        n_acc = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc_v[n_acc++] += w * vt[d];
        }
    }
    n_acc = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        head_out[h * head_dim + d] = acc_v[n_acc++];
    }
}
