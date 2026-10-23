#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// Hoisted above the chunked-attention kernels that use it (defined again
// further down; the identical redefinition is benign).
#define ATTN_CHUNK 128

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
    // 2D-split dispatch: linearization that is bit-identical when the grid fits
    // one x row of 1024 threadgroups and still legal above it (prefill
    // b=128 × 12288-hidden swiglu = 6144 threadgroups > the 1024 one-dim cap).
    uint3 tg                            [[threadgroup_position_in_grid]],
    uint ti                             [[thread_index_in_threadgroup]]
) {
    uint id = (tg.y * 1024u + tg.x) * 256u + ti;
    if (id >= n) { return; }
    float val = g[id];
    float silu = val / (1.0f + exp(-val));
    g[id] = silu * u[id];
}

kernel void add_inplace(
    device float       *acc             [[buffer(0)]],
    device const float *add             [[buffer(1)]],
    constant uint      &n               [[buffer(2)]],
    uint3 tg                            [[threadgroup_position_in_grid]],
    uint ti                             [[thread_index_in_threadgroup]]
) {
    uint id = (tg.y * 1024u + tg.x) * 256u + ti;
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

// ── GPU3: q4_K batched GEMV, B=4 tokens per weight pass ───────────────────
// Same mapping and dispatch family as gemv_q4_k_v2 (1 row per 32-lane
// simdgroup, 16-byte-aligned uint4 group loads, static unrolls, algebraic
// scale fold) but the weight stream is amortized over FOUR activation
// vectors, token-major x[4][cols] -> y[4][rows] (the engine's x_batch shape).
// Serves prefill chunks and multi-token speculative verification.

kernel void gemv_q4_k_b4(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],   // [4][cols]
    device float        *y              [[buffer(2)]],   // [4][rows]
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &rows           [[buffer(5)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // v1-clone arithmetic with the weight work SHARED across the 4 token
    // accumulators: the nibble byte is loaded/decoded once per l and the
    // d*nibble products are shared, so the instruction count per weight
    // byte stays ~v1+3 FMAs instead of 4 separate v1 bodies (which measured
    // exactly 4x a single stream — issue-bound). Per-token f32 order is
    // IDENTICAL to v1 (independent accumulator chains), so b4 stays
    // bit-exact against v1 per row. x token-major [4][cols] (x_batch shape).
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    device const float *x0b = x;
    device const float *x1b = x + (size_t)1 * cols;
    device const float *x2b = x + (size_t)2 * cols;
    device const float *x3b = x + (size_t)3 * cols;
    uint row_byte = w_rem + row * (cols / 256) * 144;
    uint nb = cols / 256;
    for (uint b = lane; b < nb; b += 32) {
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        uint is = 0;
        for (uint j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, is + j / 32, &sc1, &m1);
            q4k_scale_min(scales, is + j / 32 + 1, &sc2, &m2);
            float d1 = d * (float)sc1;
            float mm1 = dmin * (float)m1;
            float d2 = d * (float)sc2;
            float mm2 = dmin * (float)m2;
            uint qb = j / 2;
#pragma clang loop unroll(full)
            for (uint l = 0; l < 32; ++l) {
                uint q = (uint)qs[qb + l];
                float c1 = d1 * (float)(q & 0x0F);
                float c2 = d2 * (float)(q >> 4);
                // all 8 x-loads in flight before any FMA — issue-bound
                // loops serialize when each FMA waits on its own load
                float x00 = x0b[b * 256 + j + l];
                float x001 = x0b[b * 256 + j + 32 + l];
                float x10 = x1b[b * 256 + j + l];
                float x101 = x1b[b * 256 + j + 32 + l];
                float x20 = x2b[b * 256 + j + l];
                float x201 = x2b[b * 256 + j + 32 + l];
                float x30 = x3b[b * 256 + j + l];
                float x301 = x3b[b * 256 + j + 32 + l];
                acc0 += c1 * x00 - mm1 * x00;
                acc0 += c2 * x001 - mm2 * x001;
                acc1 += c1 * x10 - mm1 * x10;
                acc1 += c2 * x101 - mm2 * x101;
                acc2 += c1 * x20 - mm1 * x20;
                acc2 += c2 * x201 - mm2 * x201;
                acc3 += c1 * x30 - mm1 * x30;
                acc3 += c2 * x301 - mm2 * x301;
            }
        }
        is += 8;
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    y[row] = acc0;
    y[rows + row] = acc1;
    y[2 * rows + row] = acc2;
    y[3 * rows + row] = acc3;
}

// SINGLE-row variant of the b4c2 geometry (the serial client's actual op
// shape): 8 lanes x 16B uint4 spans cover one superblock, 4 superblocks per
// wavefront, float4 x loads + one float4 accumulator, horizontal sum once.
// gemv_q4_k_c (the byte-lane version) measured only 24 GB/s single-row —
// the 16B-span geometry is what carries the class.
kernel void gemv_q4_k_c2(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float4 acc = 0.0f;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 144;
    uint sub = lane / 8;
    uint l8 = lane % 8;
    uint g = l8 / 2;
    uint k0 = (l8 % 2) * 16;
    for (uint b0 = 0; b0 < nb; b0 += 4) {
        uint b = b0 + sub;
        if (b >= nb) { break; }
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        uint8_t sc1, m1, sc2, m2;
        q4k_scale_min(scales, g * 2, &sc1, &m1);
        q4k_scale_min(scales, g * 2 + 1, &sc2, &m2);
        float s1 = d * (float)sc1;
        float s2 = d * (float)sc2;
        float m1f = dmin * (float)m1;
        float m2f = dmin * (float)m2;
        uint4 q4 = *(device const uint4 *)&qs[l8 * 16];
        uint xbase = b * 256 + g * 64 + k0;
#pragma clang loop unroll(full)
        for (uint c = 0; c < 4; ++c) {
            uint comp;
            if (c == 0) { comp = q4.x; }
            else if (c == 1) { comp = q4.y; }
            else if (c == 2) { comp = q4.z; }
            else { comp = q4.w; }
            float4 lo4 = float4(float(comp & 15), float((comp >> 8) & 15),
                                float((comp >> 16) & 15), float((comp >> 24) & 15));
            float4 hi4 = float4(float((comp >> 4) & 15), float((comp >> 12) & 15),
                                float((comp >> 20) & 15), float((comp >> 28) & 15));
            float4 wlo = float4(s1) * lo4 - float4(m1f);
            float4 whi = float4(s2) * hi4 - float4(m2f);
            float4 xlo = *(device const float4 *)&x[xbase + 4 * c];
            float4 xhi = *(device const float4 *)&x[xbase + 32 + 4 * c];
            acc += wlo * xlo + whi * xhi;
        }
    }
    float v = (acc.x + acc.y) + (acc.z + acc.w);
    v = simd_sum(v);
    if (lane == 0) {
        y[row] = v;
    }
}

// B-row RMSNorm (f32 weights): one threadgroup per row, rows are
// contiguous [B][dim] — collapses the spec chain's 4 per-row norm
// dispatches into one (the elementwise swiglu/add need no variant: their
// flat kernels already accept n = B*n over the contiguous [B][n] slabs).
kernel void rmsnorm_f32w_b4(
    device const float *x              [[buffer(0)]],
    device const float *w              [[buffer(1)]],
    device float       *y              [[buffer(2)]],
    constant uint      &dim            [[buffer(3)]],
    constant float     &eps            [[buffer(4)]],
    constant uint      &w_rem          [[buffer(5)]],
    constant uint      &rows           [[buffer(6)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint lane                          [[thread_index_in_simdgroup]]
) {
    if (row >= rows) { return; }
    device const float *xr = x + row * dim;
    device float *yr = y + row * dim;
    float sum_sq = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        float v = xr[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = lane; i < dim; i += 32) {
        yr[i] = xr[i] * rms * w[w_rem + i];
    }
}

#define ATTN_CHUNK 128

// ── MULTI-ROW (B≤4 chain) attention family ──────────────────────────────────
// The spec chain pass dispatches kv_write/head_rmsnorm/attn_part/attn_fin
// once PER ROW today (~576 extra dispatches per token); these variants take
// the row count and a positions[B] u32 buffer (per-row RoPE position and
// attention bound) and process all rows in grid = rows × <heads> thread
// groups. Chain causality: kv_write_b4 lands every row's KV BEFORE any
// attention reads, and attn_part_b4 bounds row r's softmax strictly to
// positions[r] — later rows' cache content is excluded by construction, so
// the per-row semantics equal the per-row dispatch sequence exactly.
// Row slots are contiguous [B][...] (the b4 GEMV outputs), meta regions are
// meta_stride apart per row (the client passes meta_per_layer).
kernel void llama_kv_write_b4(
    device float       *k              [[buffer(0)]],
    device const float *v              [[buffer(1)]],
    device float       *k_cache_layer  [[buffer(2)]],
    device float       *v_cache_layer  [[buffer(3)]],
    device const float *k_ln           [[buffer(4)]],
    constant uint      &k_ln_rem       [[buffer(5)]],
    constant uint      &has_norm       [[buffer(6)]],
    device const uint  *positions      [[buffer(7)]],
    constant uint      &head_dim       [[buffer(8)]],
    constant uint      &n_kv_heads     [[buffer(9)]],
    constant uint      &max_seq        [[buffer(10)]],
    constant float     &eps            [[buffer(11)]],
    constant float     &rope_base      [[buffer(12)]],
    constant uint      &rope_mode      [[buffer(13)]],
    device const float *rope_tab      [[buffer(14)]],
    constant uint      &has_table      [[buffer(15)]],
    uint3 kvtg                         [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    uint kv = kvtg.y * 1024u + kvtg.x;
    uint row = kv / n_kv_heads;
    uint kh  = kv % n_kv_heads;
    uint pos = positions[row];
    // rope table (CPU-built, sofuu_rope_fill_rows-exact): [row] = cos(hd/2)|sin(hd/2).
    // has_table=0 keeps the legacy on-device theta path (decode/spec callers).
    uint tab_r = has_table != 0u ? row * head_dim : 0u;
    device float *k_head = k + ((size_t)row * n_kv_heads + kh) * head_dim;
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
        float c;
        float s;
        if (has_table != 0u) {
            c = rope_tab[tab_r + j];
            s = rope_tab[tab_r + head_dim / 2 + j];
        } else {
            float theta = float(pos) / pow(rope_base, (2.0f * float(j)) / float(head_dim));
            c = cos(theta);
            s = sin(theta);
        }
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
    device float *k_dst = k_cache_layer + ((size_t)kh * max_seq + pos) * head_dim;
    device float *v_dst = v_cache_layer + ((size_t)kh * max_seq + pos) * head_dim;
    for (uint i = lane; i < head_dim; i += 32) {
        k_dst[i] = k_head[i];
        v_dst[i] = v[(size_t)row * n_kv_heads * head_dim + kh * head_dim + i];
    }
}

kernel void head_rmsnorm_b4(
    device float       *heads           [[buffer(0)]],
    device const float *w               [[buffer(1)]],
    constant uint      &w_rem           [[buffer(2)]],
    constant uint      &head_dim        [[buffer(3)]],
    constant uint      &n_heads         [[buffer(4)]],
    constant float     &eps             [[buffer(5)]],
    constant uint      &has_norm        [[buffer(6)]],
    device const uint  *positions       [[buffer(7)]],
    constant float     &rope_base       [[buffer(8)]],
    constant uint      &rope_mode       [[buffer(9)]],
    device const float *rope_tab      [[buffer(10)]],
    constant uint      &has_table      [[buffer(11)]],
    uint3 htg                          [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint h = htg.y * 1024u + htg.x;
    uint row = h / n_heads;
    uint hh  = h % n_heads;
    uint pos = positions[row];
    device float *rp = heads + ((size_t)row * n_heads + hh) * head_dim;
    if (has_norm != 0u) {
        float ss = 0.0f;
        for (uint i = lane; i < head_dim; i += 32) {
            ss += rp[i] * rp[i];
        }
        ss = simd_sum(ss);
        float rms = rsqrt(ss / float(head_dim) + eps);
        for (uint i = lane; i < head_dim; i += 32) {
            rp[i] = rp[i] * rms * w[w_rem + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint tab_r = has_table != 0u ? row * head_dim : 0u;
    for (uint j = lane; j < head_dim / 2; j += 32) {
        float c;
        float s;
        if (has_table != 0u) {
            c = rope_tab[tab_r + j];
            s = rope_tab[tab_r + head_dim / 2 + j];
        } else {
            float theta = float(pos) / pow(rope_base, (2.0f * float(j)) / float(head_dim));
            c = cos(theta);
            s = sin(theta);
        }
        if (rope_mode == 0u) {
            float r0 = rp[j];
            float r1 = rp[j + head_dim / 2];
            rp[j]                = r0 * c - r1 * s;
            rp[j + head_dim / 2] = r1 * c + r0 * s;
        } else {
            float r0 = rp[2 * j];
            float r1 = rp[2 * j + 1];
            rp[2 * j]     = r0 * c - r1 * s;
            rp[2 * j + 1] = r1 * c + r0 * s;
        }
    }
}

kernel void llama_attn_part_b4(
    device float       *q               [[buffer(0)]],
    device const float *k_cache_layer   [[buffer(1)]],
    device const float *v_cache_layer   [[buffer(2)]],
    device const float *q_ln            [[buffer(3)]],
    device float       *meta            [[buffer(4)]],
    device const uint  *positions       [[buffer(5)]],
    constant uint      &head_dim        [[buffer(6)]],
    constant uint      &n_heads         [[buffer(7)]],
    constant uint      &n_kv_heads      [[buffer(8)]],
    constant uint      &max_seq         [[buffer(9)]],
    constant uint      &n_chunks_max    [[buffer(10)]],
    constant uint      &meta_stride     [[buffer(11)]],
    uint3 gtg                           [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint g = gtg.y * 1024u + gtg.x;
    // grid = rows*heads*n_chunks_max; the q-norm already ran (head_rmsnorm_b4)
    uint row = g / (n_heads * n_chunks_max);
    uint rem = g % (n_heads * n_chunks_max);
    uint h = rem / n_chunks_max;
    uint c = rem % n_chunks_max;
    uint pos = positions[row];
    uint n_chunks = (pos + 1 + ATTN_CHUNK - 1) / ATTN_CHUNK;
    if (c >= n_chunks) { return; }
    device float *q_head = q + ((size_t)row * n_heads + h) * head_dim;
    float scale = rsqrt(float(head_dim));
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint kv = h / (n_heads / n_kv_heads);
    device const float *kbase = k_cache_layer + (size_t)kv * max_seq * head_dim;
    device const float *vbase = v_cache_layer + (size_t)kv * max_seq * head_dim;
    device float *m = meta + (size_t)row * meta_stride;
    uint t0 = c * ATTN_CHUNK;
    uint t_end = min(t0 + ATTN_CHUNK, pos + 1);

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
        m[(size_t)h * n_chunks_max * 2 + c * 2 + 0] = cmax;
        m[(size_t)h * n_chunks_max * 2 + c * 2 + 1] = csum;
    }

    threadgroup float tw[ATTN_CHUNK];
    for (uint t = t0 + lane; t < t_end; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        tw[t - t0] = exp(dot * scale - cmax);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    size_t parts = (size_t)n_heads * n_chunks_max * 2 + (size_t)h * n_chunks_max * head_dim;
    float inv = 1.0f / csum;
    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (uint t = t0; t < t_end; ++t) {
            acc += tw[t - t0] * vbase[(size_t)t * head_dim + d];
        }
        m[parts + (size_t)c * head_dim + d] = acc * inv;
    }
}

kernel void llama_attn_fin_b4(
    device float       *meta            [[buffer(0)]],
    device float       *head_out        [[buffer(1)]],
    device const uint  *positions       [[buffer(2)]],
    constant uint      &n_heads         [[buffer(3)]],
    constant uint      &head_dim        [[buffer(4)]],
    constant uint      &n_chunks_max    [[buffer(5)]],
    constant uint      &meta_stride     [[buffer(6)]],
    uint3 gtg                           [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint g = gtg.y * 1024u + gtg.x;
    uint row = g / n_heads;
    uint h   = g % n_heads;
    uint pos = positions[row];
    uint n_chunks = (pos + 1 + ATTN_CHUNK - 1) / ATTN_CHUNK;
    device float *m = meta + (size_t)row * meta_stride;
    size_t mbase = (size_t)h * n_chunks_max * 2;
    float gmax = -INFINITY;
    for (uint c = 0; c < n_chunks; ++c) {
        gmax = max(gmax, m[mbase + c * 2 + 0]);
    }
    float gsum = 0.0f;
    for (uint c = 0; c < n_chunks; ++c) {
        gsum += m[mbase + c * 2 + 1] * exp(m[mbase + c * 2 + 0] - gmax);
    }
    float inv = 1.0f / gsum;
    size_t parts = (size_t)n_heads * n_chunks_max * 2 + (size_t)h * n_chunks_max * head_dim;
    device float *out = head_out + ((size_t)row * n_heads + h) * head_dim;
    for (uint d = lane; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (uint c = 0; c < n_chunks; ++c) {
            acc += m[mbase + c * 2 + 1] * exp(m[mbase + c * 2 + 0] - gmax)
                 * m[parts + (size_t)c * head_dim + d];
        }
        out[d] = acc * inv;
    }
}

// ── COALESCED-LANE q4_K GEMVs (the ~150GB/s kernel-class attack) ────────────
// Lane l reads qs byte l of each 32-byte group span, so 32 lanes cover a
// 64-value group's weight bytes in ONE coalesced wavefront (the shipped
// per-superblock-per-lane layout strides lanes 144B apart and issues ~800
// scalar ops per lane per superblock; here ~105, dominated by fully
// coalesced 32B weight waves and 128B x waves), and x loads coalesce
// across the wavefront too. Scale bytes and d/dmin are lane-uniform; every
// lane decodes them redundantly (same addresses, no divergence). The f32
// summation ORDER changes (per-lane partials accumulated across the row,
// one simd_sum per token at the end) — a DIFFERENT parity class from
// gemv_q4_k/b4: gated separately at vec_dot_raw_q4_k 1e-2-rel, and the
// client must run ONE q4_K kernel class per model run (serial and spec
// both flipped by the same env-gated PSO choice) so text parity holds
// internally.
kernel void gemv_q4_k_c(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float acc = 0.0f;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * (cols / 256) * 144;
    // ALL lanes of the threadgroup walk the same superblock: byte lane + a
    // 32-lane stride covers each 64-value group's qs span per wavefront.
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        // j = value offset 0/64/128/192; scale pair = (2j, 2j+1)
        for (uint j = 0; j < 4; ++j) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, j * 2, &sc1, &m1);
            q4k_scale_min(scales, j * 2 + 1, &sc2, &m2);
            float p1 = d * (float)sc1;
            float q1 = dmin * (float)m1;
            float p2 = d * (float)sc2;
            float q2 = dmin * (float)m2;
            uint q = (uint)qs[j * 32 + lane];
            uint xoff = b * 256 + j * 64 + lane;
            float xw0 = x[xoff];
            float xw1 = x[xoff + 32];
            acc += (p1 * (float)(q & 0x0F) - q1) * xw0
                 + (p2 * (float)(q >> 4) - q2) * xw1;
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}

// B=4 batched variant: same coalesced weight wavefronts, 4 token
// accumulators; x token-major [4][cols], y token-major [4][rows].
kernel void gemv_q4_k_b4c(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],   // [4][cols]
    device float        *y              [[buffer(2)]],   // [4][rows]
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &rows           [[buffer(5)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * (cols / 256) * 144;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        for (uint j = 0; j < 4; ++j) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, j * 2, &sc1, &m1);
            q4k_scale_min(scales, j * 2 + 1, &sc2, &m2);
            float p1 = d * (float)sc1;
            float q1 = dmin * (float)m1;
            float p2 = d * (float)sc2;
            float q2 = dmin * (float)m2;
            uint q = (uint)qs[j * 32 + lane];
            uint xoff = b * 256 + j * 64 + lane;
            uint xoff1 = xoff + 32;
            float w1 = p1 * (float)(q & 0x0F) - q1;
            float w2 = p2 * (float)(q >> 4) - q2;
            acc0 += w1 * x[xoff] + w2 * x[xoff1];
            acc1 += w1 * x[cols + xoff] + w2 * x[cols + xoff1];
            acc2 += w1 * x[2 * cols + xoff] + w2 * x[2 * cols + xoff1];
            acc3 += w1 * x[3 * cols + xoff] + w2 * x[3 * cols + xoff1];
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    y[row] = acc0;
    y[rows + row] = acc1;
    y[2 * rows + row] = acc2;
    y[3 * rows + row] = acc3;
}

// VEC variant of the coalesced class. Geometry: sub = lane/8 picks the
// superblock b = b_base+sub (4 superblocks per wavefront); l8 = lane%8
// picks the lane's 16B span of the superblock's 128 qs bytes — one 16B-
// aligned uint4 load (stride 144 = 9x16, span base l8*16). Byte
// w = l8*16 + 4c + j (component c, byte j) holds the value pair
// (g*64+k, g*64+32+k) with g = l8/2, k = (l8%2)*16 + 4c + j — component c
// covers the 4 consecutive pairs k0+4c..k0+4c+3, so its lo nibbles map to
// ONE float4 x run at xbase+4c (16B-aligned: k0%4==0) and its hi nibbles
// to the run at +32. Per-value weight = d*sc*nib - dmin*m (m is NOT
// nibble-scaled). float4 accumulators, one horizontal sum per token at
// the end + one simd_sum: per-lane-partials class, gated at
// vec_dot_raw_q4_k 1e-2-rel.
kernel void gemv_q4_k_b4c2(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],   // [4][cols]
    device float        *y              [[buffer(2)]],   // [4][rows]
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &rows           [[buffer(5)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * (cols / 256) * 144;
    uint sub = lane / 8;
    uint l8 = lane % 8;
    uint g = l8 / 2;
    uint k0 = (l8 % 2) * 16;
    for (uint b0 = 0; b0 < nb; b0 += 4) {
        uint b = b0 + sub;
        if (b >= nb) { break; }
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        uint8_t sc1, m1, sc2, m2;
        q4k_scale_min(scales, g * 2, &sc1, &m1);
        q4k_scale_min(scales, g * 2 + 1, &sc2, &m2);
        float s1 = d * (float)sc1;
        float s2 = d * (float)sc2;
        float m1f = dmin * (float)m1;
        float m2f = dmin * (float)m2;
        uint4 q4 = *(device const uint4 *)&qs[l8 * 16];
        uint xbase = b * 256 + g * 64 + k0;
        // component c covers byte pairs k0+4c..+3; unrolled so the vector
        // component access stays compile-time (dynamic indexing spills)
#pragma clang loop unroll(full)
        for (uint c = 0; c < 4; ++c) {
            uint comp;
            if (c == 0) { comp = q4.x; }
            else if (c == 1) { comp = q4.y; }
            else if (c == 2) { comp = q4.z; }
            else { comp = q4.w; }
            float lo0 = float(comp & 15);
            float lo1 = float((comp >> 8) & 15);
            float lo2 = float((comp >> 16) & 15);
            float lo3 = float((comp >> 24) & 15);
            float hi0 = float((comp >> 4) & 15);
            float hi1 = float((comp >> 12) & 15);
            float hi2 = float((comp >> 20) & 15);
            float hi3 = float((comp >> 28) & 15);
            float4 wlo = float4(s1) * float4(lo0, lo1, lo2, lo3) - float4(m1f);
            float4 whi = float4(s2) * float4(hi0, hi1, hi2, hi3) - float4(m2f);
            float4 x0lo = *(device const float4 *)&x[xbase + 4 * c];
            float4 x0hi = *(device const float4 *)&x[xbase + 32 + 4 * c];
            float4 x1lo = *(device const float4 *)&x[cols + xbase + 4 * c];
            float4 x1hi = *(device const float4 *)&x[cols + xbase + 32 + 4 * c];
            float4 x2lo = *(device const float4 *)&x[2 * cols + xbase + 4 * c];
            float4 x2hi = *(device const float4 *)&x[2 * cols + xbase + 32 + 4 * c];
            float4 x3lo = *(device const float4 *)&x[3 * cols + xbase + 4 * c];
            float4 x3hi = *(device const float4 *)&x[3 * cols + xbase + 32 + 4 * c];
            acc0 += wlo * x0lo + whi * x0hi;
            acc1 += wlo * x1lo + whi * x1hi;
            acc2 += wlo * x2lo + whi * x2hi;
            acc3 += wlo * x3lo + whi * x3hi;
        }
    }
    float v0 = (acc0.x + acc0.y) + (acc0.z + acc0.w);
    float v1 = (acc1.x + acc1.y) + (acc1.z + acc1.w);
    float v2 = (acc2.x + acc2.y) + (acc2.z + acc2.w);
    float v3 = (acc3.x + acc3.y) + (acc3.z + acc3.w);
    v0 = simd_sum(v0); v1 = simd_sum(v1);
    v2 = simd_sum(v2); v3 = simd_sum(v3);
    y[row] = v0;
    y[rows + row] = v1;
    y[2 * rows + row] = v2;
    y[3 * rows + row] = v3;
}
// RESID-ADD EPILOGUE variant: after the GEMV, x_acc[t*acc_dim + row] +=
// y[t*rows + row] for t in 0..4 — one writer per element (threadgroup row),
// removing the separate batch_add dispatch (and its barrier) per layer.
// Only valid when rows == acc_dim (the o/down projections: out width ==
// residual width, both dim).
kernel void gemv_q4_k_b4c2r(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],   // [4][cols]
    device float        *y              [[buffer(2)]],   // [4][rows]
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &rows           [[buffer(5)]],
    device float        *x_acc          [[buffer(6)]],
    constant uint       &acc_dim        [[buffer(7)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * (cols / 256) * 144;
    uint sub = lane / 8;
    uint l8 = lane % 8;
    uint g = l8 / 2;
    uint k0 = (l8 % 2) * 16;
    for (uint b0 = 0; b0 < nb; b0 += 4) {
        uint b = b0 + sub;
        if (b >= nb) { break; }
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        uint8_t sc1, m1, sc2, m2;
        q4k_scale_min(scales, g * 2, &sc1, &m1);
        q4k_scale_min(scales, g * 2 + 1, &sc2, &m2);
        float s1 = d * (float)sc1;
        float s2 = d * (float)sc2;
        float m1f = dmin * (float)m1;
        float m2f = dmin * (float)m2;
        uint4 q4 = *(device const uint4 *)&qs[l8 * 16];
        uint xbase = b * 256 + g * 64 + k0;
        // component c covers byte pairs k0+4c..+3; unrolled so the vector
        // component access stays compile-time (dynamic indexing spills)
#pragma clang loop unroll(full)
        for (uint c = 0; c < 4; ++c) {
            uint comp;
            if (c == 0) { comp = q4.x; }
            else if (c == 1) { comp = q4.y; }
            else if (c == 2) { comp = q4.z; }
            else { comp = q4.w; }
            float lo0 = float(comp & 15);
            float lo1 = float((comp >> 8) & 15);
            float lo2 = float((comp >> 16) & 15);
            float lo3 = float((comp >> 24) & 15);
            float hi0 = float((comp >> 4) & 15);
            float hi1 = float((comp >> 12) & 15);
            float hi2 = float((comp >> 20) & 15);
            float hi3 = float((comp >> 28) & 15);
            float4 wlo = float4(s1) * float4(lo0, lo1, lo2, lo3) - float4(m1f);
            float4 whi = float4(s2) * float4(hi0, hi1, hi2, hi3) - float4(m2f);
            float4 x0lo = *(device const float4 *)&x[xbase + 4 * c];
            float4 x0hi = *(device const float4 *)&x[xbase + 32 + 4 * c];
            float4 x1lo = *(device const float4 *)&x[cols + xbase + 4 * c];
            float4 x1hi = *(device const float4 *)&x[cols + xbase + 32 + 4 * c];
            float4 x2lo = *(device const float4 *)&x[2 * cols + xbase + 4 * c];
            float4 x2hi = *(device const float4 *)&x[2 * cols + xbase + 32 + 4 * c];
            float4 x3lo = *(device const float4 *)&x[3 * cols + xbase + 4 * c];
            float4 x3hi = *(device const float4 *)&x[3 * cols + xbase + 32 + 4 * c];
            acc0 += wlo * x0lo + whi * x0hi;
            acc1 += wlo * x1lo + whi * x1hi;
            acc2 += wlo * x2lo + whi * x2hi;
            acc3 += wlo * x3lo + whi * x3hi;
        }
    }
    float v0 = (acc0.x + acc0.y) + (acc0.z + acc0.w);
    float v1 = (acc1.x + acc1.y) + (acc1.z + acc1.w);
    float v2 = (acc2.x + acc2.y) + (acc2.z + acc2.w);
    float v3 = (acc3.x + acc3.y) + (acc3.z + acc3.w);
    v0 = simd_sum(v0); v1 = simd_sum(v1);
    v2 = simd_sum(v2); v3 = simd_sum(v3);
    y[row] = v0;
    y[rows + row] = v1;
    y[2 * rows + row] = v2;
    y[3 * rows + row] = v3;
    x_acc[row] += v0;
    x_acc[acc_dim + row] += v1;
    x_acc[2 * acc_dim + row] += v2;
    x_acc[3 * acc_dim + row] += v3;
}

// ── FUSED 1-chunk attention (pos<128): part+fin in ONE dispatch ─────────────
// Grid = rows × heads; the full softmax fits in one chunk, so the two-phase
// meta protocol (part writes partials, fin rescales — two dispatches + the
// drain between them) collapses into direct weighted-V output. Per-row
// positions from the buffer; softmax bounded by each row's own position
// (chain causality, same contract as attn_part_b4). vec_dot oracle-class.
kernel void llama_attn_1c_b4(
    device float       *q               [[buffer(0)]],
    device const float *k_cache_layer   [[buffer(1)]],
    device const float *v_cache_layer   [[buffer(2)]],
    device float       *head_out        [[buffer(3)]],
    device const uint  *positions       [[buffer(4)]],
    constant uint      &head_dim        [[buffer(5)]],
    constant uint      &n_heads         [[buffer(6)]],
    constant uint      &n_kv_heads      [[buffer(7)]],
    constant uint      &max_seq         [[buffer(8)]],
    device const float *rope_tab      [[buffer(9)]],
    constant uint      &has_table      [[buffer(10)]],
    uint3 gtg                           [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint g = gtg.y * 1024u + gtg.x;
    uint row = g / n_heads;
    uint h   = g % n_heads;
    uint pos = positions[row];
    device float *q_head = q + ((size_t)row * n_heads + h) * head_dim;
    float scale = rsqrt(float(head_dim));
    uint kv = h / (n_heads / n_kv_heads);
    device const float *kbase = k_cache_layer + (size_t)kv * max_seq * head_dim;
    device const float *vbase = v_cache_layer + (size_t)kv * max_seq * head_dim;
    uint t_end = pos + 1;

    float mx = -INFINITY;
    for (uint t = lane; t < t_end; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        if (dot * scale > mx) { mx = dot * scale; }
    }
    float gmax = simd_max(mx);
    float lsum = 0.0f;
    for (uint t = lane; t < t_end; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        lsum += exp(dot * scale - gmax);
    }
    float gsum = simd_sum(lsum);
    float inv = 1.0f / gsum;

    threadgroup float tw[128];
    for (uint t = lane; t < t_end; t += 32) {
        float dot = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            dot += q_head[i] * kbase[(size_t)t * head_dim + i];
        }
        tw[t] = exp(dot * scale - gmax);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out = head_out + ((size_t)row * n_heads + h) * head_dim;
    float inv_scale_w = inv;
    for (uint d = lane; d < head_dim; d += 32) {
        float accw = 0.0f;
        for (uint t = 0; t < t_end; ++t) {
            accw += tw[t] * vbase[(size_t)t * head_dim + d];
        }
        out[d] = accw * inv_scale_w;
    }
}

// COALESCED q6_K GEMV (single row) — the lm_head class fix: the shipped
// scalar q6_K runs the ~514MB vocab GEMV at ~6GB/s (~80ms of every 8B
// token; the layer-mimic ablation prices the head at ~+456ms/pass). Same
// wavefront idea as gemv_q4_k_c: 32 lanes cover one 210B superblock per
// wavefront — each lane serves (n,p) = (lane/16 gives p; n looped 0..1)
// and l = lane%16, reading ONE ql byte + ONE qh byte (both coalesced
// across lanes) and emitting 4 value products into x runs at +0/+32/+64/
// +96 (runs coalesce across consecutive lanes). q6_K semantics exactly
// as sofuu_vec_dot_q6_k: value v = d*sc[8n+p+{0,2,1,3}]*
// ((lo4 | hi2<<4) - 32) with hi2 from qh bit-pairs (0,1)/(2,3)/(4,5)/(6,7)
// for the four +0/+32/+64/+96 runs. int8 scales (char cast). Per-lane
// partials + one simd_sum: per-lane-partials class, gated at
// vec_dot_raw_q6_k 1e-2-rel.
kernel void gemv_q6_k_c(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float acc = 0.0f;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 210;
    uint p = lane / 16;
    uint l = lane % 16;
    for (uint b = 0; b < nb; ++b) {
        device const uchar *blk = w_base + row_byte + b * 210;
        float d = float(read_f16(blk, 208));
        for (uint n = 0; n < 2; ++n) {
            uint qh = (uint)blk[128 + 32 * n + 16 * p + l];
            uint qlb = 64 * n + 16 * p + l;
            uint lo = (uint)blk[qlb];
            uint hi = (uint)blk[qlb + 32];
            uint isb = 192 + 8 * n + p;
            // sofuu_vec_dot_q6_k scale order: runs +0/+32/+64/+96 use
            // sc[8n+part + 0/+2/+4/+6] (sc strides 8 per half here)
            float s0 = d * (float)(char)blk[isb + 0];
            float s2 = d * (float)(char)blk[isb + 2];
            float s1 = d * (float)(char)blk[isb + 4];
            float s3 = d * (float)(char)blk[isb + 6];
            // runs: +0 (sc0, qh bits 0-1), +32 (sc2, bits 2-3),
            // +64 (sc1, bits 4-5), +96 (sc3, bits 6-7); -32 signed bias
            int q1 = (int)((lo & 15u) | ((qh & 3u) << 4)) - 32;
            int q2 = (int)(((hi & 15u) | (((qh >> 2) & 3u) << 4)) & 63u) - 32;
            int q3 = (int)(((lo >> 4) | (((qh >> 4) & 3u) << 4)) & 63u) - 32;
            int q4 = (int)(((hi >> 4) | (((qh >> 6) & 3u) << 4)) & 63u) - 32;
            uint xoff = b * 256 + 128 * n + 16 * p + l;
            acc += s0 * (float)q1 * x[xoff]
                 + s2 * (float)q2 * x[xoff + 32]
                 + s1 * (float)q3 * x[xoff + 64]
                 + s3 * (float)q4 * x[xoff + 96];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}

// ── GPU3 helper: per-head f32 RMSNorm on a [heads x head_dim] slot ─────────
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
    // partials live AFTER all {max,sum} pairs (heads * chunks * 2 floats)
    size_t parts = (size_t)n_heads * n_chunks * 2 + (size_t)h * n_chunks * head_dim;
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
    size_t parts = (size_t)n_heads * n_chunks * 2 + (size_t)h * n_chunks * head_dim;
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

// ── PREFILL GEMM class (P1 of PLAN-LLAMA-CPP-PARITY) ───────────────────────
// y[t][row] = dot(W[row], X[t]) for a B-token tile. One threadgroup per
// (row, 8-token subtile); lanes stride the 256-value quant blocks exactly
// like gemv_q4_k, and EVERY weight byte is reused across the 8 tokens of the
// subtile (one dequant, 8 FMAs). The token loop is STATICALLY unrolled with
// compile-time indices (t0 computed from threadgroup coords, t runtime only
// in the MASK) — no runtime-indexed accumulator arrays, the measured spill
// trap. Grid: (rows, ceil(B/GEMM_TS)); y is [B x rows] row-major, x is
// [B x cols] row-major.
#define GEMM_TS 8

kernel void gemm_q4_k(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    // Lane = VALUE INDEX inside the 64-value group (the q6_K/q8_0 mapping),
    // not the block index. The old mapping gave each lane its own 144-byte
    // block, so a warp's 32 lanes touched 32 addresses 144 B apart: one
    // 32-byte sector fetched per lane per instruction (8x over-fetch) and no
    // cross-lane reuse of the x stream. Measured at fleet shapes that class
    // ran 20x slower than gemm_q6_k on the SAME grid (0.02 vs 0.44 TFLOP/s,
    // bench_gemm_rate_bands). Here a group read is 32 consecutive nibble
    // bytes and the two values it encodes are the two 32-wide halves of the
    // group — so weights, scales and x are all lane-contiguous.
    uint row = tg.x;
    uint t0 = tg.y * GEMM_TS;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 144;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    for (uint b = 0; b < nb; ++b) {
        device const uchar *blk = w_base + row_byte + b * 144;
        float d    = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *scales = blk + 4;
        device const uchar *qs = blk + 16;
        // Same value order as sofuu_dequantize_row_q4_k / gemv_q4_k:
        // group g covers values [64g, 64g+64); byte l carries value 64g+l in
        // its low nibble (scale 2g) and value 64g+32+l in its high nibble
        // (scale 2g+1).
        for (uint g = 0; g < 4; ++g) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, 2 * g + 0, &sc1, &m1);
            q4k_scale_min(scales, 2 * g + 1, &sc2, &m2);
            float d1 = d * (float)sc1;
            float mm1 = dmin * (float)m1;
            float d2 = d * (float)sc2;
            float mm2 = dmin * (float)m2;
            uchar byte = qs[g * 32 + lane];
            float wlo = d1 * (float)(byte & 0x0F) - mm1;
            float whi = d2 * (float)(byte >> 4) - mm2;
            uint k = b * 256 + g * 64 + lane;
            if (v0) { acc0 += wlo * x0[k] + whi * x0[k + 32]; }
            if (v1) { acc1 += wlo * x1[k] + whi * x1[k + 32]; }
            if (v2) { acc2 += wlo * x2[k] + whi * x2[k + 32]; }
            if (v3) { acc3 += wlo * x3[k] + whi * x3[k + 32]; }
            if (v4) { acc4 += wlo * x4[k] + whi * x4[k + 32]; }
            if (v5) { acc5 += wlo * x5[k] + whi * x5[k + 32]; }
            if (v6) { acc6 += wlo * x6[k] + whi * x6[k + 32]; }
            if (v7) { acc7 += wlo * x7[k] + whi * x7[k + 32]; }
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    if (lane == 0) {
        if (v0) y[(size_t)(t0 + 0) * rows + row] = acc0;
        if (v1) y[(size_t)(t0 + 1) * rows + row] = acc1;
        if (v2) y[(size_t)(t0 + 2) * rows + row] = acc2;
        if (v3) y[(size_t)(t0 + 3) * rows + row] = acc3;
        if (v4) y[(size_t)(t0 + 4) * rows + row] = acc4;
        if (v5) y[(size_t)(t0 + 5) * rows + row] = acc5;
        if (v6) y[(size_t)(t0 + 6) * rows + row] = acc6;
        if (v7) y[(size_t)(t0 + 7) * rows + row] = acc7;
    }
}

// ── P1 op-diet: multi-tensor GEMM (fused qkv / fused gate_up) ──────────────
// One dispatch computes y = concat(W_0 x, ..., W_{n-1} x) over B tokens —
// grid (total_rows, ceil(B/8)); each threadgroup resolves its section from
// the row span table (<=4 sections: {weight byte-offset, row count} pairs),
// then dots exactly like gemm_q4_k. Sections live in ONE registered weight
// buffer, so the offsets are bytes. Cuts 3 qkv GEMVs + 2 FFN GEMVs per layer
// into 2 dispatches.
#define GEMM_MAX_SECTIONS 4

struct GemmSection {
    uint off;       // 256-aligned byte offset into the weight buffer
    uint rem;       // byte remainder (=<255) folded into the dequant cursor
    uint rows_end;  // cumulative row end (prefix sum); host pads to 4 with
                    // duplicate ends so empty spans never match
    uint y_off;     // ABSOLUTE byte offset of this section's y region in the
                    // shared slab (q/k/v land in their own slots from ONE
                    // dispatch; the glue kernels keep reading per-region)
    uint y_stride;  // row stride of that region in floats (region rows)
};

kernel void gemm_q4_k_multi(
    device const uchar      *w_base   [[buffer(0)]],
    device const float      *x        [[buffer(1)]],
    device float            *y        [[buffer(2)]],
    constant uint           &cols     [[buffer(3)]],
    constant uint           &B        [[buffer(4)]],
    constant uint           &rows_tot [[buffer(5)]],
    constant GemmSection    *secs     [[buffer(6)]],
    uint2 tg                          [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    uint ts = tg.y;
    if (row >= rows_tot) { return; }
    uint t0 = ts * GEMM_TS;

    // resolve the section by row span (constant-space runtime indexing is a
    // plain load — not the thread-array spill trap); the weight cursor is
    // per-section: off + rem + local_row * nb * 144
    uint prev_end = 0u;
    uint sec_off = 0u;
    uint sec_rem = 0u;
    uint sec_yoff = 0u;
    uint sec_ystr = 0u;
    for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
        uint end_i = secs[i].rows_end;
        if (row >= prev_end && row < end_i) {
            sec_off = secs[i].off;
            sec_rem = secs[i].rem;
            sec_yoff = secs[i].y_off;
            sec_ystr = secs[i].y_stride;
            break;
        }
        prev_end = end_i;
    }
    uint local_row = row - prev_end;

    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);

    uint nb = cols / 256;
    uint row_byte = sec_off + sec_rem + local_row * nb * 144;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const uchar *blk = w_base + row_byte + b * 144;
        float d    = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *scales = blk + 4;
        device const uchar *qs = blk + 16;
        // Lane = value index in the 64-value group (the coalesced mapping of
        // gemm_q4_k / gemm_q6_k): byte l carries value 64g+l (scale 2g) in
        // its low nibble and 64g+32+l (scale 2g+1) in its high nibble.
        for (uint g = 0; g < 4; ++g) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, 2 * g + 0, &sc1, &m1);
            q4k_scale_min(scales, 2 * g + 1, &sc2, &m2);
            float d1 = d * (float)sc1;
            float mm1 = dmin * (float)m1;
            float d2 = d * (float)sc2;
            float mm2 = dmin * (float)m2;
            uchar byte = qs[g * 32 + lane];
            float wlo = d1 * (float)(byte & 0x0F) - mm1;
            float whi = d2 * (float)(byte >> 4) - mm2;
            uint k = b * 256 + g * 64 + lane;
            if (v0) { acc0 += wlo * x0[k] + whi * x0[k + 32]; }
            if (v1) { acc1 += wlo * x1[k] + whi * x1[k + 32]; }
            if (v2) { acc2 += wlo * x2[k] + whi * x2[k + 32]; }
            if (v3) { acc3 += wlo * x3[k] + whi * x3[k + 32]; }
            if (v4) { acc4 += wlo * x4[k] + whi * x4[k + 32]; }
            if (v5) { acc5 += wlo * x5[k] + whi * x5[k + 32]; }
            if (v6) { acc6 += wlo * x6[k] + whi * x6[k + 32]; }
            if (v7) { acc7 += wlo * x7[k] + whi * x7[k + 32]; }
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    device float *yr = y + sec_yoff / 4 + local_row;
    if (lane == 0) {
        if (v0) yr[(size_t)(t0 + 0) * sec_ystr] = acc0;
        if (v1) yr[(size_t)(t0 + 1) * sec_ystr] = acc1;
        if (v2) yr[(size_t)(t0 + 2) * sec_ystr] = acc2;
        if (v3) yr[(size_t)(t0 + 3) * sec_ystr] = acc3;
        if (v4) yr[(size_t)(t0 + 4) * sec_ystr] = acc4;
        if (v5) yr[(size_t)(t0 + 5) * sec_ystr] = acc5;
        if (v6) yr[(size_t)(t0 + 6) * sec_ystr] = acc6;
        if (v7) yr[(size_t)(t0 + 7) * sec_ystr] = acc7;
    }
}

// ── P1 wave 2: resid-add epilogue GEMM (o-projection / down-projection) ────
// Same dot as gemm_q4_k, but the store READS the residual region and adds:
//   y[t*stride + row] = dot(W[row], X[t]) + res[t*res_stride + row]
// with a single writer per element (rows == residual width) — the residual
// add fuses into the projection, removing 2 flat adds per layer.
kernel void gemm_q4_k_acc(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device const float  *res            [[buffer(2)]],
    device float        *y              [[buffer(3)]],
    constant uint       &cols           [[buffer(4)]],
    constant uint       &w_rem          [[buffer(5)]],
    constant uint       &B              [[buffer(6)]],
    constant uint       &res_stride     [[buffer(7)]],
    constant uint       &rows           [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    uint ts = tg.y;
    if (row >= rows) { return; }
    uint t0 = ts * GEMM_TS;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 144;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const uchar *blk = w_base + row_byte + b * 144;
        float d    = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *scales = blk + 4;
        device const uchar *qs = blk + 16;
        // Lane = value index in the 64-value group (the coalesced mapping of
        // gemm_q4_k / gemm_q6_k): byte l carries value 64g+l (scale 2g) in
        // its low nibble and 64g+32+l (scale 2g+1) in its high nibble.
        for (uint g = 0; g < 4; ++g) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, 2 * g + 0, &sc1, &m1);
            q4k_scale_min(scales, 2 * g + 1, &sc2, &m2);
            float d1 = d * (float)sc1;
            float mm1 = dmin * (float)m1;
            float d2 = d * (float)sc2;
            float mm2 = dmin * (float)m2;
            uchar byte = qs[g * 32 + lane];
            float wlo = d1 * (float)(byte & 0x0F) - mm1;
            float whi = d2 * (float)(byte >> 4) - mm2;
            uint k = b * 256 + g * 64 + lane;
            if (v0) { acc0 += wlo * x0[k] + whi * x0[k + 32]; }
            if (v1) { acc1 += wlo * x1[k] + whi * x1[k + 32]; }
            if (v2) { acc2 += wlo * x2[k] + whi * x2[k + 32]; }
            if (v3) { acc3 += wlo * x3[k] + whi * x3[k + 32]; }
            if (v4) { acc4 += wlo * x4[k] + whi * x4[k + 32]; }
            if (v5) { acc5 += wlo * x5[k] + whi * x5[k + 32]; }
            if (v6) { acc6 += wlo * x6[k] + whi * x6[k + 32]; }
            if (v7) { acc7 += wlo * x7[k] + whi * x7[k + 32]; }
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    if (lane == 0) {
        device float *yr = y + (size_t)row;
        device const float *rr = res + (size_t)row;
        if (v0) yr[(size_t)(t0 + 0) * rows] = acc0 + rr[(size_t)(t0 + 0) * res_stride];
        if (v1) yr[(size_t)(t0 + 1) * rows] = acc1 + rr[(size_t)(t0 + 1) * res_stride];
        if (v2) yr[(size_t)(t0 + 2) * rows] = acc2 + rr[(size_t)(t0 + 2) * res_stride];
        if (v3) yr[(size_t)(t0 + 3) * rows] = acc3 + rr[(size_t)(t0 + 3) * res_stride];
        if (v4) yr[(size_t)(t0 + 4) * rows] = acc4 + rr[(size_t)(t0 + 4) * res_stride];
        if (v5) yr[(size_t)(t0 + 5) * rows] = acc5 + rr[(size_t)(t0 + 5) * res_stride];
        if (v6) yr[(size_t)(t0 + 6) * rows] = acc6 + rr[(size_t)(t0 + 6) * res_stride];
        if (v7) yr[(size_t)(t0 + 7) * rows] = acc7 + rr[(size_t)(t0 + 7) * res_stride];
    }
}

// ── P2 decode diet: multi-section q4_K b4c GEMV ────────────────────────────
// Same dot shape as gemv_q4_k_b4c (byte-lane, all 4 token rows per
// threadgroup — no GEMM subtiles, the lanes stay busy at B<=4), with the
// multi-section table: grid = rows_total; each threadgroup resolves its
// section, reads its own weight span, and stores into that section's y
// region with region-row stride. Fuses qkv (3 tensors -> 1 dispatch) and
// gate_up (2 -> 1) in the decode chain without the GEMM's lane waste.
kernel void gemv_q4_k_b4multi(
    device const uchar      *w_base   [[buffer(0)]],
    device const float      *x        [[buffer(1)]],
    device float            *y        [[buffer(2)]],
    constant uint           &cols     [[buffer(3)]],
    constant uint           &rows_tot [[buffer(4)]],
    constant GemmSection    *secs     [[buffer(5)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    if (row >= rows_tot) { return; }
    uint prev_end = 0u;
    uint sec_off = 0u;
    uint sec_rem = 0u;
    uint sec_yoff = 0u;
    uint sec_ystr = 0u;
    for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
        uint end_i = secs[i].rows_end;
        if (row >= prev_end && row < end_i) {
            sec_off = secs[i].off;
            sec_rem = secs[i].rem;
            sec_yoff = secs[i].y_off;
            sec_ystr = secs[i].y_stride;
            break;
        }
        prev_end = end_i;
    }
    uint local_row = row - prev_end;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint nb = cols / 256;
    uint row_byte = sec_off + sec_rem + local_row * nb * 144;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        for (uint j = 0; j < 4; ++j) {
            uint8_t sc1, m1, sc2, m2;
            q4k_scale_min(scales, j * 2, &sc1, &m1);
            q4k_scale_min(scales, j * 2 + 1, &sc2, &m2);
            float p1 = d * (float)sc1;
            float q1 = dmin * (float)m1;
            float p2 = d * (float)sc2;
            float q2 = dmin * (float)m2;
            uint q = (uint)qs[j * 32 + lane];
            uint xoff = b * 256 + j * 64 + lane;
            uint xoff1 = xoff + 32;
            float w1 = p1 * (float)(q & 0x0F) - q1;
            float w2 = p2 * (float)(q >> 4) - q2;
            acc0 += w1 * x[xoff] + w2 * x[xoff1];
            acc1 += w1 * x[cols + xoff] + w2 * x[cols + xoff1];
            acc2 += w1 * x[2 * cols + xoff] + w2 * x[2 * cols + xoff1];
            acc3 += w1 * x[3 * cols + xoff] + w2 * x[3 * cols + xoff1];
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    device float *yr = y + sec_yoff / 4 + local_row;
    yr[0] = acc0;
    yr[sec_ystr] = acc1;
    yr[2 * sec_ystr] = acc2;
    yr[3 * sec_ystr] = acc3;
}

// ── P1/2: multi-section q6_K b4 GEMV (attn_v is Q6_K in Q4_K_M/i1 files;
// MiniCPM5-2B's whole FFN is Q6_K_XL) ──────────────────────────────────────
// Same section table as the q4_K b4multi. Lane = the l index (0..31) so the
// four x reads per (b, n, lane) are warp-consecutive — coalesced where the
// v1 kernel was lane-strided-over-blocks (uncoalesced x). 210-byte blocks:
// ql[128] @0, qh[64] @128, int8 scales[16] @192, d f16 @208; value =
// (6-bit - 32) * (d * sc[8n + l/16 + {0,2,4,6}]). All 4 token rows ride one
// threadgroup (4 accumulators), stores go to the section's y region.
kernel void gemv_q6_k_b4multi(
    device const uchar      *w_base   [[buffer(0)]],
    device const float      *x        [[buffer(1)]],
    device float            *y        [[buffer(2)]],
    constant uint           &cols     [[buffer(3)]],
    constant uint           &rows_tot [[buffer(4)]],
    constant GemmSection    *secs     [[buffer(5)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    if (row >= rows_tot) { return; }
    uint prev_end = 0u;
    uint sec_off = 0u;
    uint sec_rem = 0u;
    uint sec_yoff = 0u;
    uint sec_ystr = 0u;
    for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
        uint end_i = secs[i].rows_end;
        if (row >= prev_end && row < end_i) {
            sec_off = secs[i].off;
            sec_rem = secs[i].rem;
            sec_yoff = secs[i].y_off;
            sec_ystr = secs[i].y_stride;
            break;
        }
        prev_end = end_i;
    }
    uint local_row = row - prev_end;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint nb = cols / 256;
    uint row_byte = sec_off + sec_rem + local_row * nb * 210;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = w_base + row_byte + b * 210 + 128;
        device const uchar *sc = w_base + row_byte + b * 210 + 192;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint n = 0; n < 2; ++n) {
            uint l = lane;
            uchar qh8 = qh[n * 32 + l];
            uint isn = 8 * n + l / 16;
            float s0 = d * (float)(signed char)sc[isn + 0];
            float s2 = d * (float)(signed char)sc[isn + 2];
            float s4 = d * (float)(signed char)sc[isn + 4];
            float s6 = d * (float)(signed char)sc[isn + 6];
            float q1 = (float)((ql[n * 64 + l] & 0x0F) | (((qh8 >> 0) & 3) << 4)) - 32.0f;
            float q2 = (float)((ql[n * 64 + l + 32] & 0x0F) | (((qh8 >> 2) & 3) << 4)) - 32.0f;
            float q3 = (float)((ql[n * 64 + l] >> 4) | (((qh8 >> 4) & 3) << 4)) - 32.0f;
            float q4 = (float)((ql[n * 64 + l + 32] >> 4) | (((qh8 >> 6) & 3) << 4)) - 32.0f;
            uint xb6 = b * 256 + n * 128 + l;
            a0 += q1 * s0 * x[xb6]
                + q2 * s2 * x[xb6 + 32]
                + q3 * s4 * x[xb6 + 64]
                + q4 * s6 * x[xb6 + 96];
            a1 += q1 * s0 * x[cols + xb6]
                + q2 * s2 * x[cols + xb6 + 32]
                + q3 * s4 * x[cols + xb6 + 64]
                + q4 * s6 * x[cols + xb6 + 96];
            a2 += q1 * s0 * x[2 * cols + xb6]
                + q2 * s2 * x[2 * cols + xb6 + 32]
                + q3 * s4 * x[2 * cols + xb6 + 64]
                + q4 * s6 * x[2 * cols + xb6 + 96];
            a3 += q1 * s0 * x[3 * cols + xb6]
                + q2 * s2 * x[3 * cols + xb6 + 32]
                + q3 * s4 * x[3 * cols + xb6 + 64]
                + q4 * s6 * x[3 * cols + xb6 + 96];
        }
        acc0 += a0; acc1 += a1; acc2 += a2; acc3 += a3;
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    device float *yr = y + sec_yoff / 4 + local_row;
    if (lane == 0) {
        yr[0] = acc0;
        yr[sec_ystr] = acc1;
        yr[2 * sec_ystr] = acc2;
        yr[3 * sec_ystr] = acc3;
    }
}

// ── P2: multi-section q8_0 b4 GEMV (1B fleet + MiniCPM's q8_0 tensors) ─────
// q8_0 block = f16 d + 32 int8 (34 bytes / 32 values; d FIRST). Lane = value index
// (warp-consecutive x reads); every lane accumulates its position's
// contributions across ALL blocks, so ONE simd_sum per token closes it.
kernel void gemv_q8_0_b4multi(
    device const uchar      *w_base   [[buffer(0)]],
    device const float      *x        [[buffer(1)]],
    device float            *y        [[buffer(2)]],
    constant uint           &cols     [[buffer(3)]],
    constant uint           &rows_tot [[buffer(4)]],
    constant GemmSection    *secs     [[buffer(5)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    if (row >= rows_tot) { return; }
    uint prev_end = 0u;
    uint sec_off = 0u;
    uint sec_rem = 0u;
    uint sec_yoff = 0u;
    uint sec_ystr = 0u;
    for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
        uint end_i = secs[i].rows_end;
        if (row >= prev_end && row < end_i) {
            sec_off = secs[i].off;
            sec_rem = secs[i].rem;
            sec_yoff = secs[i].y_off;
            sec_ystr = secs[i].y_stride;
            break;
        }
        prev_end = end_i;
    }
    uint local_row = row - prev_end;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    uint nb = cols / 32;
    uint row_byte = sec_off + sec_rem + local_row * nb * 34;
    for (uint b = 0; b < nb; ++b) {
        // q8_0 = [f16 d][32 int8] — d FIRST (GGML layout)
        float d = float(read_f16(w_base, row_byte + b * 34));
        float q = (float)(as_type<int8_t>(w_base[row_byte + b * 34 + 2 + lane]));
        uint xb = b * 32 + lane;
        acc0 += d * q * x[xb];
        acc1 += d * q * x[cols + xb];
        acc2 += d * q * x[2 * cols + xb];
        acc3 += d * q * x[3 * cols + xb];
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    device float *yr = y + sec_yoff / 4 + local_row;
    if (lane == 0) {
        yr[0] = acc0;
        yr[sec_ystr] = acc1;
        yr[2 * sec_ystr] = acc2;
        yr[3 * sec_ystr] = acc3;
    }
}

// ── P2 rate wave: q8_0 b4 GEMV, FOUR WEIGHT-ROWS per threadgroup ───────────
// The b4multi shape leaves x:weight traffic ~14:1 (each row's threadgroup
// re-reads all 4 token rows of x). Grouping 4 weight rows per threadgroup
// shares each x load across 4 rows (x traffic / 4). Lane = value index;
// per-row section resolution (a group may span a section boundary).
#define Q8_R4 4

kernel void gemv_q8_0_b4r4(
    device const uchar      *w_base   [[buffer(0)]],
    device const float      *x        [[buffer(1)]],
    device float            *y        [[buffer(2)]],
    constant uint           &cols     [[buffer(3)]],
    constant uint           &rows_tot [[buffer(4)]],
    constant GemmSection    *secs     [[buffer(5)]],
    uint g                            [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    // resolve 4 rows (may straddle a section boundary)
    uint sec_off[Q8_R4], sec_rem[Q8_R4], sec_yoff[Q8_R4], sec_ystr[Q8_R4], lrow[Q8_R4];
    bool valid[Q8_R4];
    for (uint r = 0u; r < Q8_R4; ++r) {
        uint row = g * Q8_R4 + r;
        valid[r] = row < rows_tot;
        uint prev_end = 0u;
        sec_off[r] = 0u; sec_rem[r] = 0u; sec_yoff[r] = 0u; sec_ystr[r] = 0u; lrow[r] = 0u;
        for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
            uint end_i = secs[i].rows_end;
            if (row >= prev_end && row < end_i) {
                sec_off[r] = secs[i].off;
                sec_rem[r] = secs[i].rem;
                sec_yoff[r] = secs[i].y_off;
                sec_ystr[r] = secs[i].y_stride;
                lrow[r] = row - prev_end;
                break;
            }
            prev_end = end_i;
        }
}
    float acc[Q8_R4][4];
    for (uint r = 0u; r < Q8_R4; ++r) {
        acc[r][0] = 0.0f; acc[r][1] = 0.0f; acc[r][2] = 0.0f; acc[r][3] = 0.0f;
    }
    uint nb = cols / 32;
    uint rb[Q8_R4];
    for (uint r = 0u; r < Q8_R4; ++r) {
        rb[r] = sec_off[r] + sec_rem[r] + lrow[r] * nb * 34;
    }
    for (uint b = 0u; b < nb; ++b) {
        float xv0 = x[b * 32 + lane];
        float xv1 = x[cols + b * 32 + lane];
        float xv2 = x[2 * cols + b * 32 + lane];
        float xv3 = x[3 * cols + b * 32 + lane];
        for (uint r = 0u; r < Q8_R4; ++r) {
            float d = float(read_f16(w_base, rb[r] + b * 34));
            float q = (float)(as_type<int8_t>(w_base[rb[r] + b * 34 + 2 + lane]));
            acc[r][0] += d * q * xv0;
            acc[r][1] += d * q * xv1;
            acc[r][2] += d * q * xv2;
            acc[r][3] += d * q * xv3;
        }
    }
    for (uint r = 0u; r < Q8_R4; ++r) {
        float s0 = simd_sum(acc[r][0]);
        float s1 = simd_sum(acc[r][1]);
        float s2 = simd_sum(acc[r][2]);
        float s3 = simd_sum(acc[r][3]);
        if (lane == 0 && valid[r]) {
            device float *yr = y + sec_yoff[r] / 4 + lrow[r];
            yr[0] = s0;
            yr[sec_ystr[r]] = s1;
            yr[2 * sec_ystr[r]] = s2;
            yr[3 * sec_ystr[r]] = s3;
        }
    }
}

// ── P2: q8_0 coalesced single-row class (the serial-decode GEMV) ───────────
// v1 walked blocks LANE-STRIDED (warp x reads scattered across 32 blocks per
// instruction). v2: ALL lanes iterate the SAME block — qs[b*34+2+lane] = one
// 32B sector across the warp, x[b*32+lane] = one 128B wavefront; each lane
// accumulates its value-position across blocks; ONE simd_sum per row.
kernel void gemv_q8_0_c2(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    uint row                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    float acc = 0.0f;
    uint nb = cols / 32;
    uint row_byte = w_rem + row * nb * 34;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 34));
        float q = (float)(as_type<int8_t>(w_base[row_byte + b * 34 + 2 + lane]));
        acc += d * q * x[b * 32 + lane];
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = acc;
    }
}

// ── Phase 3 compute-rate: real GEMM kernels for q8_0 and q6_K ──────────────
// Before this, batch_gguf_gemm accepted q4_K only — every q8_0/q6_K tensor
// in the prefill GEMM route rode the per-token GEMV class (B dispatches per
// tensor per layer: the 1B's all-q8_0 prefill was 7×B×layers dispatches).
// These copy gemm_q4_k's shape (grid (rows, ceil(B/GEMM_TS)), 32 lanes,
// 8-token accumulators) with the DECODE kernels' receipted dequant cores
// (gemv_q8_0_b4multi / gemv_q6_k_b4multi) — per-accumulator order matches
// the decode sequence exactly.

kernel void gemm_q8_0(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    uint t0 = tg.y * GEMM_TS;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    uint nb = cols / 32;
    uint row_byte = w_rem + row * nb * 34;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    // lane = value index (32 consecutive int8 = one coalesced block read)
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 34));
        float q = (float)(as_type<int8_t>(w_base[row_byte + b * 34 + 2 + lane]));
        uint k = b * 32 + lane;
        if (v0) { acc0 += d * q * x0[k]; }
        if (v1) { acc1 += d * q * x1[k]; }
        if (v2) { acc2 += d * q * x2[k]; }
        if (v3) { acc3 += d * q * x3[k]; }
        if (v4) { acc4 += d * q * x4[k]; }
        if (v5) { acc5 += d * q * x5[k]; }
        if (v6) { acc6 += d * q * x6[k]; }
        if (v7) { acc7 += d * q * x7[k]; }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    if (lane == 0) {
        if (v0) y[(size_t)(t0 + 0) * rows + row] = acc0;
        if (v1) y[(size_t)(t0 + 1) * rows + row] = acc1;
        if (v2) y[(size_t)(t0 + 2) * rows + row] = acc2;
        if (v3) y[(size_t)(t0 + 3) * rows + row] = acc3;
        if (v4) y[(size_t)(t0 + 4) * rows + row] = acc4;
        if (v5) y[(size_t)(t0 + 5) * rows + row] = acc5;
        if (v6) y[(size_t)(t0 + 6) * rows + row] = acc6;
        if (v7) y[(size_t)(t0 + 7) * rows + row] = acc7;
    }
}

kernel void gemm_q6_k(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    uint t0 = tg.y * GEMM_TS;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 210;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = w_base + row_byte + b * 210 + 128;
        device const uchar *sc = w_base + row_byte + b * 210 + 192;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        for (uint n = 0; n < 2; ++n) {
            uchar qh8 = qh[n * 32 + lane];
            uint isn = 8 * n + lane / 16;
            float s0 = d * (float)(signed char)sc[isn + 0];
            float s2 = d * (float)(signed char)sc[isn + 2];
            float s4 = d * (float)(signed char)sc[isn + 4];
            float s6 = d * (float)(signed char)sc[isn + 6];
            float q1 = (float)((ql[n * 64 + lane] & 0x0F) | (((qh8 >> 0) & 3) << 4)) - 32.0f;
            float q2 = (float)((ql[n * 64 + lane + 32] & 0x0F) | (((qh8 >> 2) & 3) << 4)) - 32.0f;
            float q3 = (float)((ql[n * 64 + lane] >> 4) | (((qh8 >> 4) & 3) << 4)) - 32.0f;
            float q4 = (float)((ql[n * 64 + lane + 32] >> 4) | (((qh8 >> 6) & 3) << 4)) - 32.0f;
            uint xb6 = b * 256 + n * 128 + lane;
            if (v0) { a0 += q1 * s0 * x0[xb6] + q2 * s2 * x0[xb6 + 32] + q3 * s4 * x0[xb6 + 64] + q4 * s6 * x0[xb6 + 96]; }
            if (v1) { a1 += q1 * s0 * x1[xb6] + q2 * s2 * x1[xb6 + 32] + q3 * s4 * x1[xb6 + 64] + q4 * s6 * x1[xb6 + 96]; }
            if (v2) { a2 += q1 * s0 * x2[xb6] + q2 * s2 * x2[xb6 + 32] + q3 * s4 * x2[xb6 + 64] + q4 * s6 * x2[xb6 + 96]; }
            if (v3) { a3 += q1 * s0 * x3[xb6] + q2 * s2 * x3[xb6 + 32] + q3 * s4 * x3[xb6 + 64] + q4 * s6 * x3[xb6 + 96]; }
            if (v4) { a4 += q1 * s0 * x4[xb6] + q2 * s2 * x4[xb6 + 32] + q3 * s4 * x4[xb6 + 64] + q4 * s6 * x4[xb6 + 96]; }
            if (v5) { a5 += q1 * s0 * x5[xb6] + q2 * s2 * x5[xb6 + 32] + q3 * s4 * x5[xb6 + 64] + q4 * s6 * x5[xb6 + 96]; }
            if (v6) { a6 += q1 * s0 * x6[xb6] + q2 * s2 * x6[xb6 + 32] + q3 * s4 * x6[xb6 + 64] + q4 * s6 * x6[xb6 + 96]; }
            if (v7) { a7 += q1 * s0 * x7[xb6] + q2 * s2 * x7[xb6 + 32] + q3 * s4 * x7[xb6 + 64] + q4 * s6 * x7[xb6 + 96]; }
        }
        acc0 += a0; acc1 += a1; acc2 += a2; acc3 += a3;
        acc4 += a4; acc5 += a5; acc6 += a6; acc7 += a7;
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    if (lane == 0) {
        if (v0) y[(size_t)(t0 + 0) * rows + row] = acc0;
        if (v1) y[(size_t)(t0 + 1) * rows + row] = acc1;
        if (v2) y[(size_t)(t0 + 2) * rows + row] = acc2;
        if (v3) y[(size_t)(t0 + 3) * rows + row] = acc3;
        if (v4) y[(size_t)(t0 + 4) * rows + row] = acc4;
        if (v5) y[(size_t)(t0 + 5) * rows + row] = acc5;
        if (v6) y[(size_t)(t0 + 6) * rows + row] = acc6;
        if (v7) y[(size_t)(t0 + 7) * rows + row] = acc7;
    }
}

// Multi-section variants (the gemm_fused! qkv/gate_up path) with per-section
// GemmSection resolution exactly like gemm_q4_k_multi.
kernel void gemm_q8_0_multi(
    device const uchar      *w_base   [[buffer(0)]],
    device const float      *x        [[buffer(1)]],
    device float            *y        [[buffer(2)]],
    constant uint           &cols     [[buffer(3)]],
    constant uint           &B        [[buffer(4)]],
    constant uint           &rows_tot [[buffer(5)]],
    constant GemmSection    *secs     [[buffer(6)]],
    uint2 tg                          [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    if (row >= rows_tot) { return; }
    uint prev_end = 0u, sec_off = 0u, sec_rem = 0u, sec_yoff = 0u, sec_ystr = 0u;
    for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
        uint end_i = secs[i].rows_end;
        if (row >= prev_end && row < end_i) {
            sec_off = secs[i].off; sec_rem = secs[i].rem;
            sec_yoff = secs[i].y_off; sec_ystr = secs[i].y_stride;
            break;
        }
        prev_end = end_i;
    }
    uint local_row = row - prev_end;
    uint t0 = tg.y * GEMM_TS;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    uint nb = cols / 32;
    uint row_byte = sec_off + sec_rem + local_row * nb * 34;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 34));
        float q = (float)(as_type<int8_t>(w_base[row_byte + b * 34 + 2 + lane]));
        uint k = b * 32 + lane;
        if (v0) { acc0 += d * q * x0[k]; }
        if (v1) { acc1 += d * q * x1[k]; }
        if (v2) { acc2 += d * q * x2[k]; }
        if (v3) { acc3 += d * q * x3[k]; }
        if (v4) { acc4 += d * q * x4[k]; }
        if (v5) { acc5 += d * q * x5[k]; }
        if (v6) { acc6 += d * q * x6[k]; }
        if (v7) { acc7 += d * q * x7[k]; }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    device float *yr = y + sec_yoff / 4 + local_row;
    if (lane == 0) {
        if (v0) yr[0] = acc0;
        if (v1) yr[sec_ystr] = acc1;
        if (v2) yr[2 * sec_ystr] = acc2;
        if (v3) yr[3 * sec_ystr] = acc3;
        if (v4) yr[4 * sec_ystr] = acc4;
        if (v5) yr[5 * sec_ystr] = acc5;
        if (v6) yr[6 * sec_ystr] = acc6;
        if (v7) yr[7 * sec_ystr] = acc7;
    }
}

kernel void gemm_q6_k_multi(
    device const uchar      *w_base   [[buffer(0)]],
    device const float      *x        [[buffer(1)]],
    device float            *y        [[buffer(2)]],
    constant uint           &cols     [[buffer(3)]],
    constant uint           &B        [[buffer(4)]],
    constant uint           &rows_tot [[buffer(5)]],
    constant GemmSection    *secs     [[buffer(6)]],
    uint2 tg                          [[threadgroup_position_in_grid]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    if (row >= rows_tot) { return; }
    uint prev_end = 0u, sec_off = 0u, sec_rem = 0u, sec_yoff = 0u, sec_ystr = 0u;
    for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
        uint end_i = secs[i].rows_end;
        if (row >= prev_end && row < end_i) {
            sec_off = secs[i].off; sec_rem = secs[i].rem;
            sec_yoff = secs[i].y_off; sec_ystr = secs[i].y_stride;
            break;
        }
        prev_end = end_i;
    }
    uint local_row = row - prev_end;
    uint t0 = tg.y * GEMM_TS;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    uint nb = cols / 256;
    uint row_byte = sec_off + sec_rem + local_row * nb * 210;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = w_base + row_byte + b * 210 + 128;
        device const uchar *sc = w_base + row_byte + b * 210 + 192;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        for (uint n = 0; n < 2; ++n) {
            uchar qh8 = qh[n * 32 + lane];
            uint isn = 8 * n + lane / 16;
            float s0 = d * (float)(signed char)sc[isn + 0];
            float s2 = d * (float)(signed char)sc[isn + 2];
            float s4 = d * (float)(signed char)sc[isn + 4];
            float s6 = d * (float)(signed char)sc[isn + 6];
            float q1 = (float)((ql[n * 64 + lane] & 0x0F) | (((qh8 >> 0) & 3) << 4)) - 32.0f;
            float q2 = (float)((ql[n * 64 + lane + 32] & 0x0F) | (((qh8 >> 2) & 3) << 4)) - 32.0f;
            float q3 = (float)((ql[n * 64 + lane] >> 4) | (((qh8 >> 4) & 3) << 4)) - 32.0f;
            float q4 = (float)((ql[n * 64 + lane + 32] >> 4) | (((qh8 >> 6) & 3) << 4)) - 32.0f;
            uint xb6 = b * 256 + n * 128 + lane;
            if (v0) { a0 += q1 * s0 * x0[xb6] + q2 * s2 * x0[xb6 + 32] + q3 * s4 * x0[xb6 + 64] + q4 * s6 * x0[xb6 + 96]; }
            if (v1) { a1 += q1 * s0 * x1[xb6] + q2 * s2 * x1[xb6 + 32] + q3 * s4 * x1[xb6 + 64] + q4 * s6 * x1[xb6 + 96]; }
            if (v2) { a2 += q1 * s0 * x2[xb6] + q2 * s2 * x2[xb6 + 32] + q3 * s4 * x2[xb6 + 64] + q4 * s6 * x2[xb6 + 96]; }
            if (v3) { a3 += q1 * s0 * x3[xb6] + q2 * s2 * x3[xb6 + 32] + q3 * s4 * x3[xb6 + 64] + q4 * s6 * x3[xb6 + 96]; }
            if (v4) { a4 += q1 * s0 * x4[xb6] + q2 * s2 * x4[xb6 + 32] + q3 * s4 * x4[xb6 + 64] + q4 * s6 * x4[xb6 + 96]; }
            if (v5) { a5 += q1 * s0 * x5[xb6] + q2 * s2 * x5[xb6 + 32] + q3 * s4 * x5[xb6 + 64] + q4 * s6 * x5[xb6 + 96]; }
            if (v6) { a6 += q1 * s0 * x6[xb6] + q2 * s2 * x6[xb6 + 32] + q3 * s4 * x6[xb6 + 64] + q4 * s6 * x6[xb6 + 96]; }
            if (v7) { a7 += q1 * s0 * x7[xb6] + q2 * s2 * x7[xb6 + 32] + q3 * s4 * x7[xb6 + 64] + q4 * s6 * x7[xb6 + 96]; }
        }
        acc0 += a0; acc1 += a1; acc2 += a2; acc3 += a3;
        acc4 += a4; acc5 += a5; acc6 += a6; acc7 += a7;
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    device float *yr = y + sec_yoff / 4 + local_row;
    if (lane == 0) {
        if (v0) yr[0] = acc0;
        if (v1) yr[sec_ystr] = acc1;
        if (v2) yr[2 * sec_ystr] = acc2;
        if (v3) yr[3 * sec_ystr] = acc3;
        if (v4) yr[4 * sec_ystr] = acc4;
        if (v5) yr[5 * sec_ystr] = acc5;
        if (v6) yr[6 * sec_ystr] = acc6;
        if (v7) yr[7 * sec_ystr] = acc7;
    }
}

// Residual-add epilogue variants (o/down projections: batch_gguf_gemm_acc),
// dequant cores as above, y = acc + res[token*res_stride + row].
kernel void gemm_q8_0_acc(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device const float  *res            [[buffer(2)]],
    device float        *y              [[buffer(3)]],
    constant uint       &cols           [[buffer(4)]],
    constant uint       &w_rem          [[buffer(5)]],
    constant uint       &B              [[buffer(6)]],
    constant uint       &res_stride     [[buffer(7)]],
    constant uint       &rows           [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    if (row >= rows) { return; }
    uint t0 = tg.y * GEMM_TS;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    uint nb = cols / 32;
    uint row_byte = w_rem + row * nb * 34;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 34));
        float q = (float)(as_type<int8_t>(w_base[row_byte + b * 34 + 2 + lane]));
        uint k = b * 32 + lane;
        if (v0) { acc0 += d * q * x0[k]; }
        if (v1) { acc1 += d * q * x1[k]; }
        if (v2) { acc2 += d * q * x2[k]; }
        if (v3) { acc3 += d * q * x3[k]; }
        if (v4) { acc4 += d * q * x4[k]; }
        if (v5) { acc5 += d * q * x5[k]; }
        if (v6) { acc6 += d * q * x6[k]; }
        if (v7) { acc7 += d * q * x7[k]; }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    device float *yr = y + row;
    if (lane == 0) {
        if (v0) yr[(size_t)(t0 + 0) * rows] = acc0 + res[(size_t)(t0 + 0) * res_stride + row];
        if (v1) yr[(size_t)(t0 + 1) * rows] = acc1 + res[(size_t)(t0 + 1) * res_stride + row];
        if (v2) yr[(size_t)(t0 + 2) * rows] = acc2 + res[(size_t)(t0 + 2) * res_stride + row];
        if (v3) yr[(size_t)(t0 + 3) * rows] = acc3 + res[(size_t)(t0 + 3) * res_stride + row];
        if (v4) yr[(size_t)(t0 + 4) * rows] = acc4 + res[(size_t)(t0 + 4) * res_stride + row];
        if (v5) yr[(size_t)(t0 + 5) * rows] = acc5 + res[(size_t)(t0 + 5) * res_stride + row];
        if (v6) yr[(size_t)(t0 + 6) * rows] = acc6 + res[(size_t)(t0 + 6) * res_stride + row];
        if (v7) yr[(size_t)(t0 + 7) * rows] = acc7 + res[(size_t)(t0 + 7) * res_stride + row];
    }
}

kernel void gemm_q6_k_acc(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device const float  *res            [[buffer(2)]],
    device float        *y              [[buffer(3)]],
    constant uint       &cols           [[buffer(4)]],
    constant uint       &w_rem          [[buffer(5)]],
    constant uint       &B              [[buffer(6)]],
    constant uint       &res_stride     [[buffer(7)]],
    constant uint       &rows           [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    if (row >= rows) { return; }
    uint t0 = tg.y * GEMM_TS;
    bool v0 = t0 + 0 < B, v1 = t0 + 1 < B, v2 = t0 + 2 < B, v3 = t0 + 3 < B;
    bool v4 = t0 + 4 < B, v5 = t0 + 5 < B, v6 = t0 + 6 < B, v7 = t0 + 7 < B;
    device const float *x0 = x + (v0 ? (t0 + 0) * cols : 0);
    device const float *x1 = x + (v1 ? (t0 + 1) * cols : 0);
    device const float *x2 = x + (v2 ? (t0 + 2) * cols : 0);
    device const float *x3 = x + (v3 ? (t0 + 3) * cols : 0);
    device const float *x4 = x + (v4 ? (t0 + 4) * cols : 0);
    device const float *x5 = x + (v5 ? (t0 + 5) * cols : 0);
    device const float *x6 = x + (v6 ? (t0 + 6) * cols : 0);
    device const float *x7 = x + (v7 ? (t0 + 7) * cols : 0);
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 210;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = w_base + row_byte + b * 210 + 128;
        device const uchar *sc = w_base + row_byte + b * 210 + 192;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        for (uint n = 0; n < 2; ++n) {
            uchar qh8 = qh[n * 32 + lane];
            uint isn = 8 * n + lane / 16;
            float s0 = d * (float)(signed char)sc[isn + 0];
            float s2 = d * (float)(signed char)sc[isn + 2];
            float s4 = d * (float)(signed char)sc[isn + 4];
            float s6 = d * (float)(signed char)sc[isn + 6];
            float q1 = (float)((ql[n * 64 + lane] & 0x0F) | (((qh8 >> 0) & 3) << 4)) - 32.0f;
            float q2 = (float)((ql[n * 64 + lane + 32] & 0x0F) | (((qh8 >> 2) & 3) << 4)) - 32.0f;
            float q3 = (float)((ql[n * 64 + lane] >> 4) | (((qh8 >> 4) & 3) << 4)) - 32.0f;
            float q4 = (float)((ql[n * 64 + lane + 32] >> 4) | (((qh8 >> 6) & 3) << 4)) - 32.0f;
            uint xb6 = b * 256 + n * 128 + lane;
            if (v0) { a0 += q1 * s0 * x0[xb6] + q2 * s2 * x0[xb6 + 32] + q3 * s4 * x0[xb6 + 64] + q4 * s6 * x0[xb6 + 96]; }
            if (v1) { a1 += q1 * s0 * x1[xb6] + q2 * s2 * x1[xb6 + 32] + q3 * s4 * x1[xb6 + 64] + q4 * s6 * x1[xb6 + 96]; }
            if (v2) { a2 += q1 * s0 * x2[xb6] + q2 * s2 * x2[xb6 + 32] + q3 * s4 * x2[xb6 + 64] + q4 * s6 * x2[xb6 + 96]; }
            if (v3) { a3 += q1 * s0 * x3[xb6] + q2 * s2 * x3[xb6 + 32] + q3 * s4 * x3[xb6 + 64] + q4 * s6 * x3[xb6 + 96]; }
            if (v4) { a4 += q1 * s0 * x4[xb6] + q2 * s2 * x4[xb6 + 32] + q3 * s4 * x4[xb6 + 64] + q4 * s6 * x4[xb6 + 96]; }
            if (v5) { a5 += q1 * s0 * x5[xb6] + q2 * s2 * x5[xb6 + 32] + q3 * s4 * x5[xb6 + 64] + q4 * s6 * x5[xb6 + 96]; }
            if (v6) { a6 += q1 * s0 * x6[xb6] + q2 * s2 * x6[xb6 + 32] + q3 * s4 * x6[xb6 + 64] + q4 * s6 * x6[xb6 + 96]; }
            if (v7) { a7 += q1 * s0 * x7[xb6] + q2 * s2 * x7[xb6 + 32] + q3 * s4 * x7[xb6 + 64] + q4 * s6 * x7[xb6 + 96]; }
        }
        acc0 += a0; acc1 += a1; acc2 += a2; acc3 += a3;
        acc4 += a4; acc5 += a5; acc6 += a6; acc7 += a7;
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5); acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);
    device float *yr = y + row;
    if (lane == 0) {
        if (v0) yr[(size_t)(t0 + 0) * rows] = acc0 + res[(size_t)(t0 + 0) * res_stride + row];
        if (v1) yr[(size_t)(t0 + 1) * rows] = acc1 + res[(size_t)(t0 + 1) * res_stride + row];
        if (v2) yr[(size_t)(t0 + 2) * rows] = acc2 + res[(size_t)(t0 + 2) * res_stride + row];
        if (v3) yr[(size_t)(t0 + 3) * rows] = acc3 + res[(size_t)(t0 + 3) * res_stride + row];
        if (v4) yr[(size_t)(t0 + 4) * rows] = acc4 + res[(size_t)(t0 + 4) * res_stride + row];
        if (v5) yr[(size_t)(t0 + 5) * rows] = acc5 + res[(size_t)(t0 + 5) * res_stride + row];
        if (v6) yr[(size_t)(t0 + 6) * rows] = acc6 + res[(size_t)(t0 + 6) * res_stride + row];
        if (v7) yr[(size_t)(t0 + 7) * rows] = acc7 + res[(size_t)(t0 + 7) * res_stride + row];
    }
}

// ── Phase 3b: tile-32 GEMM class (lane = token) ────────────────────────────
// The GEMM_TS=8 shape re-reads every weight byte once per 8-token tile —
// at B=128 that is 16x weight traffic and the measured stream rate DROPS
// with B (q8_0 2048x2048: 3.1 / 1.4 / 0.7 GB/s at B=5/32/128). This shape
// assigns the 32 lanes to the 32 TOKENS of one tile: weight bytes are read
// ONCE per threadgroup (broadcast to all lanes), x stays per-lane, and
// each lane carries one complete dot — B=128 costs ceil(B/32) = 4 weight
// passes. Summation order per output = value-serial inside each
// quant-block, blocks in order — a plain scalar loop, matching
// vec_dot_raw_* association exactly.

kernel void gemm_q8_0_t32(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    uint t = tg.y * 32u + lane;
    bool v = t < B;
    uint nb = cols / 32;
    uint row_byte = w_rem + row * nb * 34;
    float acc = 0.0f;
    device const float *xt = x + (v ? t * cols : 0);
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 34));
        uint xb = b * 32;
        float s = 0.0f;
        for (uint j = 0; j < 32; ++j) {
            float q = (float)(as_type<int8_t>(w_base[row_byte + b * 34 + 2 + j]));
            s += q * xt[xb + j];
        }
        acc += d * s;
    }
    if (v) { y[(size_t)t * rows + row] = acc; }
}

kernel void gemm_q6_k_t32(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint row = tg.x;
    uint t = tg.y * 32u + lane;
    bool v = t < B;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 210;
    float acc = 0.0f;
    device const float *xt = x + (v ? t * cols : 0);
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = w_base + row_byte + b * 210 + 128;
        device const uchar *sc = w_base + row_byte + b * 210 + 192;
        float s = 0.0f;
        // value order == the decode b4multi kernel: half-blocks n=0,1; the
        // scale pair base = 8n + (l/16); d folded into the scales exactly
        // as decode; four 32-stride nibble groups added in the same sequence
        for (uint n = 0; n < 2; ++n) {
            for (uint l = 0; l < 32; ++l) {
                uint ix = 8 * n + (l >> 4);
                uchar qh8 = qh[n * 32 + l];
                float s0 = d * (float)(signed char)sc[ix + 0];
                float s2 = d * (float)(signed char)sc[ix + 2];
                float s4 = d * (float)(signed char)sc[ix + 4];
                float s6 = d * (float)(signed char)sc[ix + 6];
                float q1 = (float)((ql[n * 64 + l] & 0x0F) | (((qh8 >> 0) & 3) << 4)) - 32.0f;
                float q2 = (float)((ql[n * 64 + l + 32] & 0x0F) | (((qh8 >> 2) & 3) << 4)) - 32.0f;
                float q3 = (float)((ql[n * 64 + l] >> 4) | (((qh8 >> 4) & 3) << 4)) - 32.0f;
                float q4v = (float)((ql[n * 64 + l + 32] >> 4) | (((qh8 >> 6) & 3) << 4)) - 32.0f;
                uint xb6 = b * 256 + n * 128 + l;
                s += q1 * s0 * xt[xb6]
                   + q2 * s2 * xt[xb6 + 32]
                   + q3 * s4 * xt[xb6 + 64]
                   + q4v * s6 * xt[xb6 + 96];
            }
        }
        acc += s;
    }
    if (v) { y[(size_t)t * rows + row] = acc; }
}

// ── 8x4-tile MMQ mirror (SOFUU_GPU_TILE_MMQ=1) ──────────────────────────────
// Bit-identical twin of the CPU default-tile q4_K path (a76_repack.h):
// activations quantized per-256 q8_K exactly like ggml_quantize_mat_q8_K_4x4
// (signed-max iscale = -127/max, magic-number round-to-nearest-even): then
// the gemm mirrors ggml_gemm_q4_K_8x4_q8_K_p's per-(row,token) combination:
// per 64-chunk the exact integer sc_lo*acc_lo + sc_hi*acc_hi is fma'd into a
// running f32 acc (fused, single rounding), the bsum*min int32 bias
// accumulates across the 4 chunks and is subtracted fused at block tail
// (vmlsq = single-rounded). Integers are order-free; the only order-sensitive
// ops are explicit fma() calls.
#pragma clang fp contract(off)

struct TileQ8K {
    float d;
    char  qs[256];
};

// grid: (B, cols/256)
kernel void quantize_q8_k(
    device const float *x    [[buffer(0)]],   // [B x cols] row-major
    device TileQ8K     *y    [[buffer(1)]],   // [B * (cols/256)] blocks
    constant uint      &cols [[buffer(2)]],
    uint2 gid            [[thread_position_in_grid]]
) {
    uint nb = cols / 256;
    uint t  = gid.x;
    uint b  = gid.y;
    device const float *xb = x + (size_t)t * cols + b * 256;
    float amax = 0.0f, max = 0.0f;
    for (uint j = 0; j < 256; ++j) {
        float ax = fabs(xb[j]);
        if (ax > amax) { amax = ax; max = xb[j]; }
    }
    device TileQ8K *o = y + (size_t)t * nb + b;
    if (amax == 0.0f) {
        o->d = 0.0f;
        for (uint j = 0; j < 256; ++j) o->qs[j] = 0;
        return;
    }
    float iscale = -127.0f / max;
    o->d = 1.0f / iscale;
    for (uint j = 0; j < 256; ++j) {
        // round-to-nearest-even (the CPU's f32 magic-add trick); rint is the
        // same rounding and cannot be contracted away by the MSL compiler
        o->qs[j] = (char)rint(xb[j] * iscale);
    }
}

// grid: (rows, B)
kernel void gemm_q4_k_mmq(
    device const uchar   *w_base [[buffer(0)]],
    device const TileQ8K *xq     [[buffer(1)]],
    device float         *y      [[buffer(2)]],
    constant uint        &cols   [[buffer(3)]],
    constant uint        &w_rem  [[buffer(4)]],
    constant uint        &B      [[buffer(5)]],
    constant uint        &rows   [[buffer(6)]],
    uint2 gid              [[thread_position_in_grid]]   // (row, token)
) {
    uint row = gid.x;
    uint t   = gid.y;
    uint nb  = cols / 256;
    uint row_byte = w_rem + row * nb * 144;
    device const TileQ8K *x8 = xq + (size_t)t * nb;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d  = float(read_f16(w_base, row_byte + b * 144));
        float dmn = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        float dx = x8[b].d;
        float sbd_scale = d * dx;
        float sbd_min = dmn * dx;
        device const char *q8 = x8[b].qs;
        int bias = 0;
        for (uint sb = 0; sb < 4; ++sb) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            q4k_scale_min(scales, sb * 2 + 0, &sc_lo, &m_lo);
            q4k_scale_min(scales, sb * 2 + 1, &sc_hi, &m_hi);
            uint q_off = sb * 64;
            uint b_off = sb * 32;
            int acc_lo = 0, acc_hi = 0;
            int bsum_lo = 0, bsum_hi = 0;
            for (uint k = 0; k < 32; ++k) {
                uchar wq = qs[b_off + k];
                int ql = (int)q8[q_off + k];
                int qh = (int)q8[q_off + 32 + k];
                acc_lo  += (int)(wq & 0x0F) * ql;
                acc_hi  += (int)(wq >> 4)   * qh;
                bsum_lo += ql;
                bsum_hi += qh;
            }
            int sum_int = (int)sc_lo * acc_lo + (int)sc_hi * acc_hi;
            acc = fma(sbd_scale, (float)sum_int, acc);
            bias += (int)m_lo * bsum_lo + (int)m_hi * bsum_hi;
        }
        acc = fma(-sbd_min, (float)bias, acc);
    }
    y[(size_t)t * rows + row] = acc;
}

// q6_K MMQ mirror (SOFUU_GPU_TILE_MMQ=1, type 14): bit-twin of the CPU q6_K
// tile pipeline (ggml_gemm_q6_K_8x4_q8_K): same act-quant, per-block int32
// accumulation of scale*dot over (half, 32-group), ONE fused fma at block
// tail. Element e = n*128 + sb*32 + j; value = 6-bit | qh bits − 32; scale =
// sc[n*8 + j/16 + sb*2] (canonical sofuu_dequantize_row_q6_k indexing read
// from the ORIGINAL block bytes).
struct TileQ6K {
    float d;
    char  rest[208];   // ql[128] | qh[64] | scales[16]
};

// grid: (rows, B)
kernel void gemm_q6_k_mmq(
    device const uchar   *w_base [[buffer(0)]],
    device const TileQ8K *xq     [[buffer(1)]],
    device float         *y      [[buffer(2)]],
    constant uint        &cols   [[buffer(3)]],
    constant uint        &w_rem  [[buffer(4)]],
    constant uint        &B      [[buffer(5)]],
    constant uint        &rows   [[buffer(6)]],
    uint2 gid              [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint t   = gid.y;
    uint nb  = cols / 256;
    uint row_byte = w_rem + row * nb * 210;
    device const TileQ8K *x8 = xq + (size_t)t * nb;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        float dx = x8[b].d;
        float sbd = d * dx;
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = ql + 128;
        device const uchar *sc = qh + 64;
        device const char *q8 = x8[b].qs;
        int acc_int = 0;
        for (uint n = 0; n < 2; ++n) {
            device const uchar *qlh = ql + n * 64;
            device const uchar *qhh = qh + n * 32;
            for (uint sb = 0; sb < 4; ++sb) {
                int acc_l = 0, acc_h = 0;
                for (uint j = 0; j < 32; ++j) {
                    uint m = sb * 32 + j;
                    uint l = j;
                    uint qn = sb;
                    uchar qlb = ((qn & 1) == 0) ? qlh[l] : qlh[l + 32];
                    uchar nib = (qn < 2) ? (qlb & 0x0F) : (qlb >> 4);
                    int v = (int)((nib | (((qhh[l] >> (qn * 2)) & 3) << 4))) - 32;
                    int qv = (int)q8[n * 128 + m];
                    if (j < 16) {
                        acc_l += v * qv;
                    } else {
                        acc_h += v * qv;
                    }
                }
                acc_int += (int)(signed char)sc[n * 8 + 0 + sb * 2] * acc_l
                         + (int)(signed char)sc[n * 8 + 1 + sb * 2] * acc_h;
            }
        }
        acc = fma(sbd, (float)acc_int, acc);
    }
    y[(size_t)t * rows + row] = acc;
}

// ── Per-32 a76 act-quant decode mirrors (SOFUU_DECODE_MMQ=1) ────────────────
// The CPU serial matvec (sofuu_gemv_q4_k/q6_k dotprod branches) quantizes the
// activation per-32 to sofuu_q8_block_ext {d, sum_x, qs[32]} (POSITIVE d =
// amax/127, round-nearest-even) and dots with integer sdot + explicit
// (d_sc*dx)*acc − (m_sc*dx)*sum_x f32 ops in the CPU's sequential order.
// These kernels mirror that bit-for-bit.
struct Q8Ext { float d; int sum_x; char qs[32]; };

// grid: (B*cols/32)
kernel void quantize_q8_ext(
    device const float *x   [[buffer(0)]],
    device Q8Ext       *y   [[buffer(1)]],
    uint gid             [[thread_position_in_grid]]
) {
    device const float *xb = x + (size_t)gid * 32;
    float amax = 0.0f;
    for (uint i = 0; i < 32; ++i) { float ax = fabs(xb[i]); if (ax > amax) amax = ax; }
    float dd = amax / 127.0f;
    float id = (dd > 0.0f) ? (127.0f / amax) : 0.0f;
    device Q8Ext *o = y + gid;
    o->d = dd;
    int sum = 0;
    for (uint i = 0; i < 32; ++i) {
        int q = (int)rint(xb[i] * id);
        o->qs[i] = (char)q;
        sum += q;
    }
    o->sum_x = sum;
}

// grid: (rows,) — q4_K
kernel void gemv_q4_k_mmq32(
    device const uchar  *w_base [[buffer(0)]],
    device const Q8Ext  *xq     [[buffer(1)]],
    device float        *y      [[buffer(2)]],
    constant uint       &cols   [[buffer(3)]],
    constant uint       &w_rem  [[buffer(4)]],
    uint gid              [[thread_position_in_grid]]
) {
    uint row = gid;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 144;
    float total = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmn = float(read_f16(w_base, row_byte + b * 144 + 2));
        device const uchar *scales = w_base + row_byte + b * 144 + 4;
        device const uchar *qs = w_base + row_byte + b * 144 + 16;
        for (uint j = 0; j < 4; ++j) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            q4k_scale_min(scales, j * 2 + 0, &sc_lo, &m_lo);
            q4k_scale_min(scales, j * 2 + 1, &sc_hi, &m_hi);
            float d1 = d * (float)sc_lo;
            float m1 = dmn * (float)m_lo;
            float d2 = d * (float)sc_hi;
            float m2 = dmn * (float)m_hi;
            device const Q8Ext *xl = xq + (size_t)b * 8 + j * 2;
            device const Q8Ext *xh = xq + (size_t)b * 8 + j * 2 + 1;
            int acc_lo = 0, acc_hi = 0;
            for (uint k = 0; k < 32; ++k) {
                uchar wq = qs[j * 32 + k];
                acc_lo += (int)(wq & 0x0F) * (int)xl->qs[k];
                acc_hi += (int)(wq >> 4)   * (int)xh->qs[k];
            }
            total += (d1 * xl->d) * (float)acc_lo - (m1 * xl->d) * (float)xl->sum_x;
            total += (d2 * xh->d) * (float)acc_hi - (m2 * xh->d) * (float)xh->sum_x;
        }
    }
    y[row] = total;
}

// grid: (rows,) — q6_K
kernel void gemv_q6_k_mmq32(
    device const uchar  *w_base [[buffer(0)]],
    device const Q8Ext  *xq     [[buffer(1)]],
    device float        *y      [[buffer(2)]],
    constant uint       &cols   [[buffer(3)]],
    constant uint       &w_rem  [[buffer(4)]],
    uint gid              [[thread_position_in_grid]]
) {
    uint row = gid;
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 210;
    float total = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float d = float(read_f16(w_base, row_byte + b * 210 + 208));
        device const uchar *ql = w_base + row_byte + b * 210;
        device const uchar *qh = ql + 128;
        device const uchar *sc = qh + 64;
        for (uint hh = 0; hh < 2; ++hh) {
            for (uint part = 0; part < 2; ++part) {
                uint l_base = part * 16;
                float a[4]; int32_t acc4[4];
                for (uint qn = 0; qn < 4; ++qn) acc4[qn] = 0;
                for (uint j = 0; j < 16; ++j) {
                    uint m = l_base + j;
                    uchar ql0 = ql[hh * 64 + l_base + j];
                    uchar ql2 = ql[hh * 64 + l_base + j + 32];
                    uchar qhb = qh[hh * 32 + l_base + j];
                    int v1 = (int)((ql0 & 0x0F) | (((qhb >> 0) & 3) << 4)) - 32;
                    int v2 = (int)((ql2 & 0x0F) | (((qhb >> 2) & 3) << 4)) - 32;
                    int v3 = (int)(((ql0 >> 4) | (((qhb >> 4) & 3) << 4))) - 32;
                    int v4 = (int)(((ql2 >> 4) | (((qhb >> 6) & 3) << 4))) - 32;
                    device const Q8Ext *xg0 = xq + (size_t)b * 8 + hh * 4 + 0;
                    device const Q8Ext *xg1 = xq + (size_t)b * 8 + hh * 4 + 1;
                    device const Q8Ext *xg2 = xq + (size_t)b * 8 + hh * 4 + 2;
                    device const Q8Ext *xg3 = xq + (size_t)b * 8 + hh * 4 + 3;
                    acc4[0] += v1 * (int)xg0->qs[m];
                    acc4[1] += v2 * (int)xg1->qs[m];
                    acc4[2] += v3 * (int)xg2->qs[m];
                    acc4[3] += v4 * (int)xg3->qs[m];
                }
                uint hs = hh * 8 + part;
                float s0 = d * (float)(signed char)sc[hs + 0];
                float s2 = d * (float)(signed char)sc[hs + 2];
                float s4 = d * (float)(signed char)sc[hs + 4];
                float s6 = d * (float)(signed char)sc[hs + 6];
                device const Q8Ext *xg0 = xq + (size_t)b * 8 + hh * 4 + 0;
                device const Q8Ext *xg1 = xq + (size_t)b * 8 + hh * 4 + 1;
                device const Q8Ext *xg2 = xq + (size_t)b * 8 + hh * 4 + 2;
                device const Q8Ext *xg3 = xq + (size_t)b * 8 + hh * 4 + 3;
                total += (s0 * xg0->d) * (float)acc4[0];
                total += (s2 * xg1->d) * (float)acc4[1];
                total += (s4 * xg2->d) * (float)acc4[2];
                total += (s6 * xg3->d) * (float)acc4[3];
            }
        }
    }
    y[row] = total;
}

// Positive-d row variant for the GEMV class (sofuu_quantize_row_q8_K: d =
// amax/127, id = 127/amax, round-nearest-even) — the CPU tile GEMV consumes
// this convention, NOT the mat quantizer's negative signed-max one. Same
// TileQ8K block layout, same gemm_q4_k_mmq consumer.
kernel void quantize_q8_k_pos(
    device const float *x    [[buffer(0)]],
    device TileQ8K     *y    [[buffer(1)]],
    constant uint      &cols [[buffer(2)]],
    uint2 gid            [[thread_position_in_grid]]
) {
    uint nb = cols / 256;
    uint t  = gid.x;
    uint b  = gid.y;
    device const float *xb = x + (size_t)t * cols + b * 256;
    float amax = 0.0f;
    for (uint j = 0; j < 256; ++j) {
        float ax = fabs(xb[j]);
        if (ax > amax) amax = ax;
    }
    device TileQ8K *o = y + (size_t)t * nb + b;
    if (amax == 0.0f) {
        o->d = 0.0f;
        for (uint j = 0; j < 256; ++j) o->qs[j] = 0;
        return;
    }
    float id = 127.0f / amax;
    o->d = amax / 127.0f;
    for (uint j = 0; j < 256; ++j) {
        o->qs[j] = (char)rint(xb[j] * id);
    }
}

// ── P4 arm-d: threadgroup-tiled q4_K GEMM ──────────────────────────────────
// The 8-token grid re-reads every weight row B/8 times (16x at chunk-128) and
// re-reads the whole x tile once per output row; measured at 0.3 TFLOP/s that
// class cannot beat the CPU tile path (0.7 TFLOP/s) let alone llama.cpp's
// Metal batched GEMM (the bar implies 3-4 TFLOP/s).
//
// Here one threadgroup owns a 32-row x 64-token output tile and streams k in
// 32-value slices: the weights of those 32 rows and the 64 token vectors are
// dequantized into threadgroup memory ONCE and then reused by every thread in
// the group (weight traffic drops 8x at B=128; x is read once per row-block
// instead of once per row). Each k-slice is exactly one q4_K sub-block, so
// every staged value shares one (scale, min) pair — the dequant is hoisted out
// of the value loop.
//
// Thread map: 128 threads = 32 rows x 4 value-chunks in the staging phase;
// 16x8 in the compute phase (each thread owns a 4x4 output tile).
// Grid: (ceil(rows/32), ceil(B/64)); y is [B x rows] row-major.
#define TILED_R 32
#define TILED_T 64
#define TILED_BK 32
#define TILED_PAD 33
#define TILED_THREADS 128

kernel void gemm_q4_k_tiled(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float ws[TILED_R * TILED_PAD];
    threadgroup float xs[TILED_T * TILED_PAD];

    uint r0 = tg.x * TILED_R;
    uint t0 = tg.y * TILED_T;
    uint nb = cols / 256;

    float acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
    }

    uint nbk = cols / TILED_BK;
    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * TILED_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 32 rows x 32 values (one sub-block each) ──
        {
            uint r = tid >> 2;             // 0..31
            uint v0 = (tid & 3) * 8;       // 0/8/16/24
            uint grow = r0 + r;
            float wv[8];
            if (grow < rows) {
                uint row_byte = w_rem + grow * nb * 144;
                device const uchar *blk = w_base + row_byte + b * 144;
                float d = float(read_f16(w_base, row_byte + b * 144));
                float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
                uint8_t sc, m;
                q4k_scale_min(blk + 4, sub, &sc, &m);
                float ds = d * (float)sc;
                float mm = dmin * (float)m;
                device const uchar *qrow = blk + 16 + 32 * (sub >> 1);
                for (uint u = 0; u < 8; ++u) {
                    uchar byte = qrow[v0 + u];
                    uint nib = (sub & 1) ? (uint)(byte >> 4) : (uint)(byte & 0x0F);
                    wv[u] = ds * (float)nib - mm;
                }
            } else {
                for (uint u = 0; u < 8; ++u) { wv[u] = 0.0f; }
            }
            for (uint u = 0; u < 8; ++u) { ws[r * TILED_PAD + v0 + u] = wv[u]; }
        }
        // ── stage 64 tokens x 32 values ──
        {
            uint t = tid >> 1;             // 0..63
            uint v0 = (tid & 1) * 16;      // 0 or 16
            uint gt = t0 + t;
            float xv[16];
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xv[u] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xv[u] = 0.0f; }
            }
            for (uint u = 0; u < 16; ++u) { xs[t * TILED_PAD + v0 + u] = xv[u]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── compute: 16 threads in token x 8 threads in row ──
        uint tx = tid & 15;                // 0..15 -> tokens tx*4 .. +3
        uint ry = tid >> 4;                // 0..7  -> rows  ry*4 .. +3
        uint tbase = tx * 4;
        uint rbase = ry * 4;
        for (uint kk = 0; kk < TILED_BK; ++kk) {
            float xv0 = xs[(tbase + 0) * TILED_PAD + kk];
            float xv1 = xs[(tbase + 1) * TILED_PAD + kk];
            float xv2 = xs[(tbase + 2) * TILED_PAD + kk];
            float xv3 = xs[(tbase + 3) * TILED_PAD + kk];
            float wv0 = ws[(rbase + 0) * TILED_PAD + kk];
            float wv1 = ws[(rbase + 1) * TILED_PAD + kk];
            float wv2 = ws[(rbase + 2) * TILED_PAD + kk];
            float wv3 = ws[(rbase + 3) * TILED_PAD + kk];
            acc[0][0] += wv0 * xv0; acc[0][1] += wv0 * xv1;
            acc[0][2] += wv0 * xv2; acc[0][3] += wv0 * xv3;
            acc[1][0] += wv1 * xv0; acc[1][1] += wv1 * xv1;
            acc[1][2] += wv1 * xv2; acc[1][3] += wv1 * xv3;
            acc[2][0] += wv2 * xv0; acc[2][1] += wv2 * xv1;
            acc[2][2] += wv2 * xv2; acc[2][3] += wv2 * xv3;
            acc[3][0] += wv3 * xv0; acc[3][1] += wv3 * xv1;
            acc[3][2] += wv3 * xv2; acc[3][3] += wv3 * xv3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── epilogue ──
    uint tx2 = tid & 15;
    uint ry2 = tid >> 4;
    uint tbase2 = t0 + tx2 * 4;
    uint rbase2 = r0 + ry2 * 4;
    for (uint i = 0; i < 4; ++i) {
        uint grow = rbase2 + i;
        if (grow >= rows) { continue; }
        for (uint j = 0; j < 4; ++j) {
            uint gt = tbase2 + j;
            if (gt >= B) { continue; }
            y[(size_t)gt * rows + grow] = acc[i][j];
        }
    }
}

// ── P4 arm-d: tiled fused GEMM (multi-section + optional resid epilogue) ────
// The prefill route's dispatches are all fused (qkv = 3 sections, gate_up = 2,
// o/down = resid epilogue), so the tiled kernel above must speak the same
// section table to be usable there. Per-row section resolution happens ONCE
// per thread before the k loop (weight cursor) and once in the epilogue (y
// cursor): a 32-row tile may straddle a section boundary, so both are per row,
// never per threadgroup. res_stride == 0 disables the epilogue add — with one
// synthesized section the same kernel is the plain tiled GEMM.
kernel void gemm_q4_k_tiled_fused(
    device const uchar      *w_base     [[buffer(0)]],
    device const float      *x          [[buffer(1)]],
    device const float      *res        [[buffer(2)]],
    device float            *y          [[buffer(3)]],
    constant uint           &cols       [[buffer(4)]],
    constant uint           &B          [[buffer(5)]],
    constant uint           &rows_tot   [[buffer(6)]],
    constant uint           &res_stride [[buffer(7)]],
    constant GemmSection    *secs       [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float ws[TILED_R * TILED_PAD];
    threadgroup float xs[TILED_T * TILED_PAD];

    uint r0 = tg.x * TILED_R;
    uint t0 = tg.y * TILED_T;
    uint nb = cols / 256;

    float acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
    }

    // staging cursor for this thread's row — resolved ONCE (it is
    // section-dependent and k-independent). d/dmin are per 256-value block,
    // so they stay inside the k loop below.
    uint srow = r0 + (tid >> 2);
    device const uchar *s_blk = w_base;
    bool s_live = false;
    {
        uint prev_end = 0u, sec_off = 0u, sec_rem = 0u;
        for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
            uint end_i = secs[i].rows_end;
            if (srow >= prev_end && srow < end_i) {
                sec_off = secs[i].off;
                sec_rem = secs[i].rem;
                break;
            }
            prev_end = end_i;
        }
        if (srow < rows_tot) {
            s_blk = w_base + sec_off + sec_rem + (srow - prev_end) * nb * 144;
            s_live = true;
        }
    }
    uint sv0 = (tid & 3) * 8;

    uint nbk = cols / TILED_BK;
    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * TILED_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 32 rows x 32 values ──
        {
            float wv[8];
            if (s_live) {
                device const uchar *blk = s_blk + b * 144;
                float d = float(read_f16(s_blk, b * 144));
                float dmin = float(read_f16(s_blk, b * 144 + 2));
                uint8_t sc, m;
                q4k_scale_min(blk + 4, sub, &sc, &m);
                float ds = d * (float)sc;
                float mm = dmin * (float)m;
                device const uchar *qrow = blk + 16 + 32 * (sub >> 1);
                for (uint u = 0; u < 8; ++u) {
                    uchar byte = qrow[sv0 + u];
                    uint nib = (sub & 1) ? (uint)(byte >> 4) : (uint)(byte & 0x0F);
                    wv[u] = ds * (float)nib - mm;
                }
            } else {
                for (uint u = 0; u < 8; ++u) { wv[u] = 0.0f; }
            }
            for (uint u = 0; u < 8; ++u) { ws[(tid >> 2) * TILED_PAD + sv0 + u] = wv[u]; }
        }
        // ── stage 64 tokens x 32 values ──
        {
            uint t = tid >> 1;
            uint v0 = (tid & 1) * 16;
            uint gt = t0 + t;
            float xv[16];
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xv[u] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xv[u] = 0.0f; }
            }
            for (uint u = 0; u < 16; ++u) { xs[t * TILED_PAD + v0 + u] = xv[u]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tx = tid & 15;
        uint ry = tid >> 4;
        uint tbase = tx * 4;
        uint rbase = ry * 4;
        for (uint kk = 0; kk < TILED_BK; ++kk) {
            float xv0 = xs[(tbase + 0) * TILED_PAD + kk];
            float xv1 = xs[(tbase + 1) * TILED_PAD + kk];
            float xv2 = xs[(tbase + 2) * TILED_PAD + kk];
            float xv3 = xs[(tbase + 3) * TILED_PAD + kk];
            float wv0 = ws[(rbase + 0) * TILED_PAD + kk];
            float wv1 = ws[(rbase + 1) * TILED_PAD + kk];
            float wv2 = ws[(rbase + 2) * TILED_PAD + kk];
            float wv3 = ws[(rbase + 3) * TILED_PAD + kk];
            acc[0][0] += wv0 * xv0; acc[0][1] += wv0 * xv1;
            acc[0][2] += wv0 * xv2; acc[0][3] += wv0 * xv3;
            acc[1][0] += wv1 * xv0; acc[1][1] += wv1 * xv1;
            acc[1][2] += wv1 * xv2; acc[1][3] += wv1 * xv3;
            acc[2][0] += wv2 * xv0; acc[2][1] += wv2 * xv1;
            acc[2][2] += wv2 * xv2; acc[2][3] += wv2 * xv3;
            acc[3][0] += wv3 * xv0; acc[3][1] += wv3 * xv1;
            acc[3][2] += wv3 * xv2; acc[3][3] += wv3 * xv3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── epilogue: per-row section (y cursor) + optional residual ──
    uint ey = tid >> 4;
    uint et = tid & 15;
    uint gt0 = t0 + et * 4;
    for (uint i = 0; i < 4; ++i) {
        uint grow = r0 + ey * 4 + i;
        if (grow >= rows_tot) { continue; }
        uint prev_end = 0u, y_off = 0u, y_str = 0u;
        for (uint s = 0u; s < GEMM_MAX_SECTIONS; ++s) {
            uint end_i = secs[s].rows_end;
            if (grow >= prev_end && grow < end_i) {
                y_off = secs[s].y_off;
                y_str = secs[s].y_stride;
                break;
            }
            prev_end = end_i;
        }
        uint local_row = grow - prev_end;
        device float *yr = y + (y_off >> 2) + local_row;
        device const float *rr = res + (size_t)grow;
        for (uint j = 0; j < 4; ++j) {
            uint gt = gt0 + j;
            if (gt >= B) { continue; }
            float v = acc[i][j];
            if (res_stride != 0u) { v += rr[(size_t)gt * res_stride]; }
            yr[(size_t)gt * y_str] = v;
        }
    }
}

// ── P4 arm-d: tiled fused GEMM, q6_K weights ───────────────────────────────
// Same 32-row x 64-token threadgroup tile and section-table ABI as
// gemm_q4_k_tiled_fused; only the weight staging differs (q6_K super-block:
// ql[128] + qh[64] + scales[16] + f16 d, 210 bytes per 256 values).
kernel void gemm_q6_k_tiled_fused(
    device const uchar      *w_base     [[buffer(0)]],
    device const float      *x          [[buffer(1)]],
    device const float      *res        [[buffer(2)]],
    device float            *y          [[buffer(3)]],
    constant uint           &cols       [[buffer(4)]],
    constant uint           &B          [[buffer(5)]],
    constant uint           &rows_tot   [[buffer(6)]],
    constant uint           &res_stride [[buffer(7)]],
    constant GemmSection    *secs       [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float ws[TILED_R * TILED_PAD];
    threadgroup float xs[TILED_T * TILED_PAD];

    uint r0 = tg.x * TILED_R;
    uint t0 = tg.y * TILED_T;
    uint nb = cols / 256;

    float acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
    }

    // staging cursor for this thread's row — resolved ONCE (it is
    // section-dependent and k-independent). d/dmin are per 256-value block,
    // so they stay inside the k loop below.
    uint srow = r0 + (tid >> 2);
    device const uchar *s_blk = w_base;
    bool s_live = false;
    {
        uint prev_end = 0u, sec_off = 0u, sec_rem = 0u;
        for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
            uint end_i = secs[i].rows_end;
            if (srow >= prev_end && srow < end_i) {
                sec_off = secs[i].off;
                sec_rem = secs[i].rem;
                break;
            }
            prev_end = end_i;
        }
        if (srow < rows_tot) {
            s_blk = w_base + sec_off + sec_rem + (srow - prev_end) * nb * 210;
            s_live = true;
        }
    }
    uint sv0 = (tid & 3) * 8;

    uint nbk = cols / TILED_BK;
    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * TILED_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 32 rows x 32 values ──
        {
            float wv[8];
            if (s_live) {
                device const uchar *blk = s_blk + b * 210;
                float d = float(read_f16(s_blk, b * 210 + 208));
                uint hsel = sub >> 2;              // 0/1 -> 128-value half
                uint quart = sub & 3;              // quarter inside the half
                uint ql_off = 64 * hsel + ((quart & 1) ? 32 : 0);
                uint qh_off = 128 + 32 * hsel;
                uint shift = 2 * quart;
                uint nib_hi = (quart >= 2) ? 1u : 0u;
                uint si = 8 * hsel + 2 * quart + (sv0 >> 4);
                int sc = (int)blk[192 + si];
                if (sc > 127) { sc -= 256; }       // int8 scale
                float ds = d * (float)sc;
                for (uint u = 0; u < 8; ++u) {
                    uint l = sv0 + u;
                    uint b6 = (uint)blk[ql_off + l];
                    uint nib = nib_hi ? (b6 >> 4) : (b6 & 0x0F);
                    uint hi = ((uint)blk[qh_off + l] >> shift) & 3u;
                    int qv = (int)(nib | (hi << 4)) - 32;
                    wv[u] = ds * (float)qv;
                }
            } else {
                for (uint u = 0; u < 8; ++u) { wv[u] = 0.0f; }
            }
            for (uint u = 0; u < 8; ++u) { ws[(tid >> 2) * TILED_PAD + sv0 + u] = wv[u]; }
        }
        // ── stage 64 tokens x 32 values ──
        {
            uint t = tid >> 1;
            uint v0 = (tid & 1) * 16;
            uint gt = t0 + t;
            float xv[16];
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xv[u] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xv[u] = 0.0f; }
            }
            for (uint u = 0; u < 16; ++u) { xs[t * TILED_PAD + v0 + u] = xv[u]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tx = tid & 15;
        uint ry = tid >> 4;
        uint tbase = tx * 4;
        uint rbase = ry * 4;
        for (uint kk = 0; kk < TILED_BK; ++kk) {
            float xv0 = xs[(tbase + 0) * TILED_PAD + kk];
            float xv1 = xs[(tbase + 1) * TILED_PAD + kk];
            float xv2 = xs[(tbase + 2) * TILED_PAD + kk];
            float xv3 = xs[(tbase + 3) * TILED_PAD + kk];
            float wv0 = ws[(rbase + 0) * TILED_PAD + kk];
            float wv1 = ws[(rbase + 1) * TILED_PAD + kk];
            float wv2 = ws[(rbase + 2) * TILED_PAD + kk];
            float wv3 = ws[(rbase + 3) * TILED_PAD + kk];
            acc[0][0] += wv0 * xv0; acc[0][1] += wv0 * xv1;
            acc[0][2] += wv0 * xv2; acc[0][3] += wv0 * xv3;
            acc[1][0] += wv1 * xv0; acc[1][1] += wv1 * xv1;
            acc[1][2] += wv1 * xv2; acc[1][3] += wv1 * xv3;
            acc[2][0] += wv2 * xv0; acc[2][1] += wv2 * xv1;
            acc[2][2] += wv2 * xv2; acc[2][3] += wv2 * xv3;
            acc[3][0] += wv3 * xv0; acc[3][1] += wv3 * xv1;
            acc[3][2] += wv3 * xv2; acc[3][3] += wv3 * xv3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── epilogue: per-row section (y cursor) + optional residual ──
    uint ey = tid >> 4;
    uint et = tid & 15;
    uint gt0 = t0 + et * 4;
    for (uint i = 0; i < 4; ++i) {
        uint grow = r0 + ey * 4 + i;
        if (grow >= rows_tot) { continue; }
        uint prev_end = 0u, y_off = 0u, y_str = 0u;
        for (uint s = 0u; s < GEMM_MAX_SECTIONS; ++s) {
            uint end_i = secs[s].rows_end;
            if (grow >= prev_end && grow < end_i) {
                y_off = secs[s].y_off;
                y_str = secs[s].y_stride;
                break;
            }
            prev_end = end_i;
        }
        uint local_row = grow - prev_end;
        device float *yr = y + (y_off >> 2) + local_row;
        device const float *rr = res + (size_t)grow;
        for (uint j = 0; j < 4; ++j) {
            uint gt = gt0 + j;
            if (gt >= B) { continue; }
            float v = acc[i][j];
            if (res_stride != 0u) { v += rr[(size_t)gt * res_stride]; }
            yr[(size_t)gt * y_str] = v;
        }
    }
}


// ── P4 arm-d: tiled fused GEMM, q8_0 weights ───────────────────────────────
// q8_0 blocks are 34 bytes (f16 d + 32 int8), so one 32-value k-slice is
// exactly one block and the dequant is a single multiply per value.
kernel void gemm_q8_0_tiled_fused(
    device const uchar      *w_base     [[buffer(0)]],
    device const float      *x          [[buffer(1)]],
    device const float      *res        [[buffer(2)]],
    device float            *y          [[buffer(3)]],
    constant uint           &cols       [[buffer(4)]],
    constant uint           &B          [[buffer(5)]],
    constant uint           &rows_tot   [[buffer(6)]],
    constant uint           &res_stride [[buffer(7)]],
    constant GemmSection    *secs       [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float ws[TILED_R * TILED_PAD];
    threadgroup float xs[TILED_T * TILED_PAD];

    uint r0 = tg.x * TILED_R;
    uint t0 = tg.y * TILED_T;
    uint nb = cols / 256;

    float acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
    }

    // staging cursor for this thread's row — resolved ONCE (it is
    // section-dependent and k-independent). d/dmin are per 256-value block,
    // so they stay inside the k loop below.
    uint srow = r0 + (tid >> 2);
    device const uchar *s_blk = w_base;
    bool s_live = false;
    {
        uint prev_end = 0u, sec_off = 0u, sec_rem = 0u;
        for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
            uint end_i = secs[i].rows_end;
            if (srow >= prev_end && srow < end_i) {
                sec_off = secs[i].off;
                sec_rem = secs[i].rem;
                break;
            }
            prev_end = end_i;
        }
        if (srow < rows_tot) {
            s_blk = w_base + sec_off + sec_rem + (srow - prev_end) * nb * 8 * 34;
            s_live = true;
        }
    }
    uint sv0 = (tid & 3) * 8;

    uint nbk = cols / TILED_BK;
    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * TILED_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 32 rows x 32 values ──
        {
            float wv[8];
            if (s_live) {
                device const uchar *blk = s_blk + kb * 34;
                float d = float(read_f16(s_blk, kb * 34));
                for (uint u = 0; u < 8; ++u) {
                    int qv = (int)blk[2 + sv0 + u];
                    if (qv > 127) { qv -= 256; }   // int8 quants
                    wv[u] = d * (float)qv;
                }
            } else {
                for (uint u = 0; u < 8; ++u) { wv[u] = 0.0f; }
            }
            for (uint u = 0; u < 8; ++u) { ws[(tid >> 2) * TILED_PAD + sv0 + u] = wv[u]; }
        }
        // ── stage 64 tokens x 32 values ──
        {
            uint t = tid >> 1;
            uint v0 = (tid & 1) * 16;
            uint gt = t0 + t;
            float xv[16];
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xv[u] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xv[u] = 0.0f; }
            }
            for (uint u = 0; u < 16; ++u) { xs[t * TILED_PAD + v0 + u] = xv[u]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tx = tid & 15;
        uint ry = tid >> 4;
        uint tbase = tx * 4;
        uint rbase = ry * 4;
        for (uint kk = 0; kk < TILED_BK; ++kk) {
            float xv0 = xs[(tbase + 0) * TILED_PAD + kk];
            float xv1 = xs[(tbase + 1) * TILED_PAD + kk];
            float xv2 = xs[(tbase + 2) * TILED_PAD + kk];
            float xv3 = xs[(tbase + 3) * TILED_PAD + kk];
            float wv0 = ws[(rbase + 0) * TILED_PAD + kk];
            float wv1 = ws[(rbase + 1) * TILED_PAD + kk];
            float wv2 = ws[(rbase + 2) * TILED_PAD + kk];
            float wv3 = ws[(rbase + 3) * TILED_PAD + kk];
            acc[0][0] += wv0 * xv0; acc[0][1] += wv0 * xv1;
            acc[0][2] += wv0 * xv2; acc[0][3] += wv0 * xv3;
            acc[1][0] += wv1 * xv0; acc[1][1] += wv1 * xv1;
            acc[1][2] += wv1 * xv2; acc[1][3] += wv1 * xv3;
            acc[2][0] += wv2 * xv0; acc[2][1] += wv2 * xv1;
            acc[2][2] += wv2 * xv2; acc[2][3] += wv2 * xv3;
            acc[3][0] += wv3 * xv0; acc[3][1] += wv3 * xv1;
            acc[3][2] += wv3 * xv2; acc[3][3] += wv3 * xv3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── epilogue: per-row section (y cursor) + optional residual ──
    uint ey = tid >> 4;
    uint et = tid & 15;
    uint gt0 = t0 + et * 4;
    for (uint i = 0; i < 4; ++i) {
        uint grow = r0 + ey * 4 + i;
        if (grow >= rows_tot) { continue; }
        uint prev_end = 0u, y_off = 0u, y_str = 0u;
        for (uint s = 0u; s < GEMM_MAX_SECTIONS; ++s) {
            uint end_i = secs[s].rows_end;
            if (grow >= prev_end && grow < end_i) {
                y_off = secs[s].y_off;
                y_str = secs[s].y_stride;
                break;
            }
            prev_end = end_i;
        }
        uint local_row = grow - prev_end;
        device float *yr = y + (y_off >> 2) + local_row;
        device const float *rr = res + (size_t)grow;
        for (uint j = 0; j < 4; ++j) {
            uint gt = gt0 + j;
            if (gt >= B) { continue; }
            float v = acc[i][j];
            if (res_stride != 0u) { v += rr[(size_t)gt * res_stride]; }
            yr[(size_t)gt * y_str] = v;
        }
    }
}

// ── P5 arm-b: flash-style batched prefill attention ────────────────────────
// The shipped llama_attn_1c_b4 recomputes every Q·K dot THREE times (max
// pass, sum pass, weight table), reads K with a 512-byte lane stride, and
// gives one 32-lane threadgroup to a single (row, head) — so 128 query rows
// each re-stream the same K/V for that head. Measured 43 GFLOP/s on the 0.6B
// (4.9 GFLOP of dots in 283 ms, 40% of the whole prefill pass).
//
// This kernel: one threadgroup = one head x ATTN_FQROWS query rows (one
// simdgroup per row), lanes stride the head_dim (K/V reads coalesce, q
// broadcasts), each score is computed once, and the softmax runs online
// (flash) in ATTN_FTB-position blocks — so one dispatch covers any context
// length, replacing attn_1c for <=128 and the part+fin pair beyond it.
//
// Chain causality: row r attends to 0..positions[r]; within a prefill chunk
// the positions are consecutive and increasing, so the tile's last row bounds
// the block loop and per-row masking zeroes the future columns.
#define ATTN_FQROWS 4
#define ATTN_FTB 64
#define ATTN_FMAXD 256

kernel void llama_attn_flash_b4(
    device const float *q               [[buffer(0)]],
    device const float *k_cache_layer   [[buffer(1)]],
    device const float *v_cache_layer   [[buffer(2)]],
    device float       *head_out        [[buffer(3)]],
    device const uint  *positions       [[buffer(4)]],
    constant uint      &head_dim        [[buffer(5)]],
    constant uint      &n_heads         [[buffer(6)]],
    constant uint      &n_kv_heads      [[buffer(7)]],
    constant uint      &max_seq         [[buffer(8)]],
    device const float *rope_tab        [[buffer(9)]],
    constant uint      &has_table       [[buffer(10)]],
    constant uint      &rows            [[buffer(11)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float s_tile[ATTN_FQROWS][ATTN_FTB];
    threadgroup float o_acc[ATTN_FQROWS][ATTN_FMAXD];
    threadgroup float m_run[ATTN_FQROWS];
    threadgroup float l_run[ATTN_FQROWS];

    uint r = tid >> 5;                 // simdgroup index == row slot
    uint lane = tid & 31;
    uint h = tg.y;
    uint row0 = tg.x * ATTN_FQROWS;
    uint row = row0 + r;
    bool live = row < rows;
    float scale = rsqrt(float(head_dim));
    uint kv = h / (n_heads / n_kv_heads);

    uint pos_row = live ? positions[row] : 0u;
    uint last_row = min(row0 + ATTN_FQROWS - 1u, rows - 1u);
    uint tmax = positions[last_row] + 1u;

    device const float *q_head = q + ((size_t)row * n_heads + h) * head_dim;
    device const float *kbase = k_cache_layer + (size_t)kv * max_seq * head_dim;
    device const float *vbase = v_cache_layer + (size_t)kv * max_seq * head_dim;

    if (lane == 0) { m_run[r] = -INFINITY; l_run[r] = 0.0f; }
    for (uint d = lane; d < head_dim; d += 32) { o_acc[r][d] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tb = 0; tb < tmax; tb += ATTN_FTB) {
        uint tn = min((uint)ATTN_FTB, tmax - tb);

        // ── scores (one pass; lanes stride head_dim) ──
        for (uint j = 0; j < tn; ++j) {
            uint t = tb + j;
            float part = 0.0f;
            if (live && t <= pos_row) {
                device const float *kt = kbase + (size_t)t * head_dim;
                for (uint d = lane; d < head_dim; d += 32) { part += q_head[d] * kt[d]; }
                part = simd_sum(part) * scale;
            } else {
                part = -INFINITY;
            }
            if (lane == 0) { s_tile[r][j] = part; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── online softmax: rescale the running state, exp into weights ──
        float bmx = -INFINITY;
        for (uint j = lane; j < tn; j += 32) { bmx = max(bmx, s_tile[r][j]); }
        bmx = simd_max(bmx);
        float m_old = m_run[r];
        float m_new = max(m_old, bmx);
        float alpha = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        float lpart = 0.0f;
        for (uint j = lane; j < tn; j += 32) {
            float sc = s_tile[r][j];
            float w = (sc == -INFINITY) ? 0.0f : exp(sc - m_new);
            s_tile[r][j] = w;
            lpart += w;
        }
        lpart = simd_sum(lpart);
        for (uint d = lane; d < head_dim; d += 32) { o_acc[r][d] *= alpha; }
        if (lane == 0) {
            m_run[r] = m_new;
            l_run[r] = l_run[r] * alpha + lpart;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── weighted V (broadcast weight, coalesced V) ──
        for (uint d = lane; d < head_dim; d += 32) {
            float acc = o_acc[r][d];
            for (uint j = 0; j < tn; ++j) {
                acc += s_tile[r][j] * vbase[(size_t)(tb + j) * head_dim + d];
            }
            o_acc[r][d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (live) {
        float inv = 1.0f / l_run[r];
        device float *out = head_out + ((size_t)row * n_heads + h) * head_dim;
        for (uint d = lane; d < head_dim; d += 32) { out[d] = o_acc[r][d] * inv; }
    }
}

// ── P5 arm-c: simdgroup-matrix (MMA) tiled q4_K GEMM ───────────────────────
// The SIMT tile (gemm_q4_k_tiled) issues one FMA per lane per cycle: 128
// MAC/cycle per threadgroup. The matrix units do 8x8x8 per instruction, so the
// same threadgroup can reach 4x that. Layout: a 64-row x 64-token tile, 4
// simdgroups each owning 16 rows x 64 tokens (2 x 8 accumulator fragments),
// weights and activations dequantized to half once per 32-value k-slice in
// threadgroup memory (the arithmetic class llama.cpp's mul_mm uses).
// ABI-identical to gemm_q4_k_tiled so the two classes can be A/B'd.
#define MMA_R 32
#define MMA_T 64
#define MMA_BK 32
#define MMA_PAD 36

kernel void gemm_q4_k_mma(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup half ws[MMA_R][MMA_PAD];
    threadgroup half xs[MMA_T][MMA_PAD];
    threadgroup float oc[4][8][8];      // per-simdgroup epilogue scratch

    uint r0 = tg.x * MMA_R;
    uint t0 = tg.y * MMA_T;
    uint nb = cols / 256;
    uint sg = tid >> 5;                 // simdgroup 0..3 -> rows sg*16..+15
    uint lane = tid & 31;

    simdgroup_matrix<float, 8, 8> acc[1][8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 1; ++i) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint nbk = cols / MMA_BK;
    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * MMA_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 32 rows x 32 values (128 threads, one pass) ──
        {
            uint slot = tid;
            uint r = slot >> 2;
            uint v0 = (slot & 3) * 8;
            uint grow = r0 + r;
            float wv[8];
            if (grow < rows) {
                uint row_byte = w_rem + grow * nb * 144;
                device const uchar *blk = w_base + row_byte + b * 144;
                float d = float(read_f16(w_base, row_byte + b * 144));
                float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
                uchar sc, m;
                q4k_scale_min(blk + 4, sub, &sc, &m);
                float ds = d * (float)sc;
                float mm = dmin * (float)m;
                device const uchar *qrow = blk + 16 + 32 * (sub >> 1);
                for (uint u = 0; u < 8; ++u) {
                    uchar byte = qrow[v0 + u];
                    uint nib = (sub & 1) ? (uint)(byte >> 4) : (uint)(byte & 0x0F);
                    wv[u] = ds * (float)nib - mm;
                }
            } else {
                for (uint u = 0; u < 8; ++u) { wv[u] = 0.0f; }
            }
            for (uint u = 0; u < 8; ++u) { ws[r][v0 + u] = half(wv[u]); }
        }
        // ── stage 64 tokens x 32 values (128 threads, one pass) ──
        {
            uint slot = tid;
            uint t = slot >> 1;
            uint v0 = (slot & 1) * 16;
            uint gt = t0 + t;
            float xv[16];
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xv[u] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xv[u] = 0.0f; }
            }
            for (uint u = 0; u < 16; ++u) { xs[t][v0 + u] = half(xv[u]); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 4 k-blocks of 8: A rows are row-major, B is loaded transposed ──
        uint rb = sg * 8;
        for (uint kk = 0; kk < 4; ++kk) {
            uint koff = kk * 8;
            simdgroup_matrix<half, 8, 8> afr[1];
            simdgroup_matrix<half, 8, 8> bfr[8];
            simdgroup_load(afr[0], &ws[rb][koff], MMA_PAD, ulong2(0, 0), false);
            for (uint j = 0; j < 8; ++j) {
                simdgroup_load(bfr[j], &xs[j * 8][koff], MMA_PAD, ulong2(0, 0), true);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 1; ++i) {
#pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], afr[i], bfr[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── epilogue: y is token-major [B][rows] — store via guard-safe scratch ──
    for (uint i = 0; i < 1; ++i) {
        for (uint j = 0; j < 8; ++j) {
            simdgroup_store(acc[i][j], &oc[sg][0][0], 8, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint dr = e >> 3;
                uint dt = e & 7;
                uint grow = r0 + sg * 8 + i * 8 + dr;
                uint gt = t0 + j * 8 + dt;
                if (grow < rows && gt < B) {
                    y[(size_t)gt * rows + grow] = oc[sg][dr][dt];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ── arm-E: 64-row tiled q4_K GEMM (x staged once per 64 rows) ──────────────
// The 32-row tile is sized for weight reuse (rows small, B large: decode, o
// and down projections). For the ffn gate/up shape (rows = 6144, B = 128) the
// weights are 3.5 MB while the activation tile was re-staged once per 32-row
// block: 192 row-tiles x 2 token-tiles x 256 KB = 98 MB of x traffic per call,
// and that shape measured the SLOWEST of the four (0.39-0.64 TFLOP/s).
// This tile doubles the row block to 64 and gives each thread an 8x4 register
// tile (8 rows x 4 tokens), so x traffic per call drops 2x and the FMA:load
// ratio goes 2.0 -> 2.67. 128 threads = 8 row-groups x 16 token-groups.
#define WIDE_R 64
#define WIDE_T 64
#define WIDE_BK 32
#define WIDE_PAD 33

kernel void gemm_q4_k_tiled_wide(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float ws[WIDE_R * WIDE_PAD];
    threadgroup float xs[WIDE_T * WIDE_PAD];

    uint r0 = tg.x * WIDE_R;
    uint t0 = tg.y * WIDE_T;
    uint nb = cols / 256;

    float acc[8][4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
    }

    uint nbk = cols / WIDE_BK;
    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * WIDE_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 64 rows x 32 values (256 slots, 2 passes) ──
        for (uint pass = 0; pass < 2; ++pass) {
            uint slot = tid + pass * 128;
            uint r = slot >> 2;
            uint v0 = (slot & 3) * 8;
            uint grow = r0 + r;
            float wv[8];
            if (grow < rows) {
                uint row_byte = w_rem + grow * nb * 144;
                device const uchar *blk = w_base + row_byte + b * 144;
                float d = float(read_f16(w_base, row_byte + b * 144));
                float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
                uchar sc, m;
                q4k_scale_min(blk + 4, sub, &sc, &m);
                float ds = d * (float)sc;
                float mm = dmin * (float)m;
                device const uchar *qrow = blk + 16 + 32 * (sub >> 1);
                for (uint u = 0; u < 8; ++u) {
                    uchar byte = qrow[v0 + u];
                    uint nib = (sub & 1) ? (uint)(byte >> 4) : (uint)(byte & 0x0F);
                    wv[u] = ds * (float)nib - mm;
                }
            } else {
                for (uint u = 0; u < 8; ++u) { wv[u] = 0.0f; }
            }
            for (uint u = 0; u < 8; ++u) { ws[r * WIDE_PAD + v0 + u] = wv[u]; }
        }
        // ── stage 64 tokens x 32 values (128 slots, one pass) ──
        {
            uint t = tid >> 1;
            uint v0 = (tid & 1) * 16;
            uint gt = t0 + t;
            float xv[16];
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xv[u] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xv[u] = 0.0f; }
            }
            for (uint u = 0; u < 16; ++u) { xs[t * WIDE_PAD + v0 + u] = xv[u]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── compute: 8 rows x 4 tokens per thread ──
        uint tx = tid & 15;
        uint ry = tid >> 4;
        uint tb = tx * 4;
        uint rb = ry * 8;
        for (uint kk = 0; kk < WIDE_BK; ++kk) {
            float xv0 = xs[(tb + 0) * WIDE_PAD + kk];
            float xv1 = xs[(tb + 1) * WIDE_PAD + kk];
            float xv2 = xs[(tb + 2) * WIDE_PAD + kk];
            float xv3 = xs[(tb + 3) * WIDE_PAD + kk];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                float wv = ws[(rb + i) * WIDE_PAD + kk];
                acc[i][0] += wv * xv0;
                acc[i][1] += wv * xv1;
                acc[i][2] += wv * xv2;
                acc[i][3] += wv * xv3;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── epilogue ──
    uint tx2 = tid & 15;
    uint ry2 = tid >> 4;
    uint tbase = t0 + tx2 * 4;
    uint rbase = r0 + ry2 * 8;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        uint grow = rbase + i;
        if (grow >= rows) { continue; }
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            uint gt = tbase + j;
            if (gt >= B) { continue; }
            y[(size_t)gt * rows + grow] = acc[i][j];
        }
    }
}

// ── arm-E: 64-row tiled FUSED q4_K GEMM (section table + resid epilogue) ───
// Same section-table ABI as gemm_q4_k_tiled_fused, same 64-row geometry and
// 8x4 register tile as gemm_q4_k_tiled_wide. Section resolution is per ROW
// (a 64-row tile can straddle a section boundary): once per staging pass for
// the weight cursor, once in the epilogue for the y cursor.
kernel void gemm_q4_k_tiled_wide_fused(
    device const uchar      *w_base     [[buffer(0)]],
    device const float      *x          [[buffer(1)]],
    device const float      *res        [[buffer(2)]],
    device float            *y          [[buffer(3)]],
    constant uint           &cols       [[buffer(4)]],
    constant uint           &B          [[buffer(5)]],
    constant uint           &rows_tot   [[buffer(6)]],
    constant uint           &res_stride [[buffer(7)]],
    constant GemmSection    *secs       [[buffer(8)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float ws[WIDE_R * WIDE_PAD];
    threadgroup float xs[WIDE_T * WIDE_PAD];

    uint r0 = tg.x * WIDE_R;
    uint t0 = tg.y * WIDE_T;
    uint nb = cols / 256;

    float acc[8][4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) { acc[i][j] = 0.0f; }
    }

    // this thread's two staging rows (one per pass): resolve both section
    // cursors up front — they are section-dependent and k-independent.
    device const uchar *s_blk0 = w_base;
    device const uchar *s_blk1 = w_base;
    bool s_live0 = false;
    bool s_live1 = false;
    uint sv0 = (tid & 3) * 8;
    {
        for (uint pass = 0; pass < 2; ++pass) {
            uint srow = r0 + ((tid + pass * 128) >> 2);
            uint prev_end = 0u, sec_off = 0u, sec_rem = 0u;
            for (uint i = 0u; i < GEMM_MAX_SECTIONS; ++i) {
                uint end_i = secs[i].rows_end;
                if (srow >= prev_end && srow < end_i) {
                    sec_off = secs[i].off;
                    sec_rem = secs[i].rem;
                    break;
                }
                prev_end = end_i;
            }
            if (srow < rows_tot) {
                device const uchar *cursor = w_base + sec_off + sec_rem + (srow - prev_end) * nb * 144;
                if (pass == 0) { s_blk0 = cursor; s_live0 = true; }
                else { s_blk1 = cursor; s_live1 = true; }
            }
        }
    }

    uint nbk = cols / WIDE_BK;
    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * WIDE_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 64 rows x 32 values (256 slots, 2 passes) ──
        for (uint pass = 0; pass < 2; ++pass) {
            uint slot = tid + pass * 128;
            uint r = slot >> 2;
            device const uchar *s_blk = (pass == 0) ? s_blk0 : s_blk1;
            bool s_live = (pass == 0) ? s_live0 : s_live1;
            float wv[8];
            if (s_live) {
                device const uchar *blk = s_blk + b * 144;
                float d = float(read_f16(s_blk, b * 144));
                float dmin = float(read_f16(s_blk, b * 144 + 2));
                uint8_t sc, m;
                q4k_scale_min(blk + 4, sub, &sc, &m);
                float ds = d * (float)sc;
                float mm = dmin * (float)m;
                device const uchar *qrow = blk + 16 + 32 * (sub >> 1);
                for (uint u = 0; u < 8; ++u) {
                    uchar byte = qrow[sv0 + u];
                    uint nib = (sub & 1) ? (uint)(byte >> 4) : (uint)(byte & 0x0F);
                    wv[u] = ds * (float)nib - mm;
                }
            } else {
                for (uint u = 0; u < 8; ++u) { wv[u] = 0.0f; }
            }
            for (uint u = 0; u < 8; ++u) { ws[r * WIDE_PAD + sv0 + u] = wv[u]; }
        }
        // ── stage 64 tokens x 32 values (128 slots, one pass) ──
        {
            uint t = tid >> 1;
            uint v0 = (tid & 1) * 16;
            uint gt = t0 + t;
            float xv[16];
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xv[u] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xv[u] = 0.0f; }
            }
            for (uint u = 0; u < 16; ++u) { xs[t * WIDE_PAD + v0 + u] = xv[u]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── compute: 8 rows x 4 tokens per thread ──
        uint tx = tid & 15;
        uint ry = tid >> 4;
        uint tb = tx * 4;
        uint rb = ry * 8;
        for (uint kk = 0; kk < WIDE_BK; ++kk) {
            float xv0 = xs[(tb + 0) * WIDE_PAD + kk];
            float xv1 = xs[(tb + 1) * WIDE_PAD + kk];
            float xv2 = xs[(tb + 2) * WIDE_PAD + kk];
            float xv3 = xs[(tb + 3) * WIDE_PAD + kk];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                float wv = ws[(rb + i) * WIDE_PAD + kk];
                acc[i][0] += wv * xv0;
                acc[i][1] += wv * xv1;
                acc[i][2] += wv * xv2;
                acc[i][3] += wv * xv3;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── epilogue: per-row section (y cursor) + optional residual ──
    uint ey = tid >> 4;
    uint et = tid & 15;
    uint gt0 = t0 + et * 4;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        uint grow = r0 + ey * 8 + i;
        if (grow >= rows_tot) { continue; }
        uint prev_end = 0u, y_off = 0u, y_str = 0u;
        for (uint s = 0u; s < GEMM_MAX_SECTIONS; ++s) {
            uint end_i = secs[s].rows_end;
            if (grow >= prev_end && grow < end_i) {
                y_off = secs[s].y_off;
                y_str = secs[s].y_stride;
                break;
            }
            prev_end = end_i;
        }
        uint local_row = grow - prev_end;
        device float *yr = y + (y_off >> 2) + local_row;
        device const float *rr = res + (size_t)grow;
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            uint gt = gt0 + j;
            if (gt >= B) { continue; }
            float v = acc[i][j];
            if (res_stride != 0u) { v += rr[(size_t)gt * res_stride]; }
            yr[(size_t)gt * y_str] = v;
        }
    }
}

// ── item 2b: 8x8 register tile — the threadgroup-bandwidth fix ──────────────
// The probe split the 32-row tile: staging 36%, and the FMA loop alone at 1.41
// TFLOP/s = 25% of peak. The loop moves 8 floats of threadgroup memory per 16
// FMAs (2 B/FMA); a core's threadgroup pipe moves 64 B/cycle against 128
// FLOP/cycle of FMA, so peak needs <= 1 B/FMA and a 4x4 thread tile cannot
// reach it no matter how it is scheduled — which is why double-buffering,
// 256 threads and the 8x4 variant all measured the same.
//
// Bytes/FMA = (R + T) / (R*T) for an RxT thread tile: 2.0 at 4x4, 1.5 at 8x4,
// 1.0 at 8x8. This kernel is the 8x8 case, and the tiles are stored TRANSPOSED
// (ws[k][r], xs[k][t]) so each thread's 8 rows and 8 tokens are contiguous and
// come back as two float4 loads instead of eight scalar ones.
// Tile: 128 rows x 64 tokens, 128 threads, thread tile 8x8.
#define R88_R 128
#define R88_T 64
#define R88_BK 32
#define R88_PAD_R 132
#define R88_PAD_T 68
#define R88_THREADS 128

kernel void gemm_q4_k_tiled_r88(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    // declared as float4 so the buffers are 16-byte aligned and the compute
    // loop's vector loads are well defined (both strides are multiples of 4)
    threadgroup float4 ws4[(R88_BK * R88_PAD_R) / 4];
    threadgroup float4 xs4[(R88_BK * R88_PAD_T) / 4];
    threadgroup float *ws = (threadgroup float *)ws4;
    threadgroup float *xs = (threadgroup float *)xs4;

    uint r0 = tg.x * R88_R;
    uint t0 = tg.y * R88_T;
    uint nb = cols / 256;
    uint nbk = cols / R88_BK;

    float acc[8][8];
    for (uint i = 0; i < 8; ++i) {
        for (uint j = 0; j < 8; ++j) { acc[i][j] = 0.0f; }
    }

    uint lane_r = tid & 15;            // 16 groups of 8 rows   -> 128 rows
    uint lane_t = tid >> 4;            // 8 groups of 8 tokens  -> 64 tokens
    uint rb = lane_r * 8;
    uint tb = lane_t * 8;

    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * R88_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        // ── stage 128 rows x 32 values: one row per thread, one sub-block ──
        {
            uint grow = r0 + tid;
            if (grow < rows) {
                uint row_byte = w_rem + grow * nb * 144;
                device const uchar *blk = w_base + row_byte + b * 144;
                float d = float(read_f16(w_base, row_byte + b * 144));
                float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
                uint8_t sc, m;
                q4k_scale_min(blk + 4, sub, &sc, &m);
                float ds = d * (float)sc;
                float mm = dmin * (float)m;
                device const uchar *qrow = blk + 16 + 32 * (sub >> 1);
                bool hi = (sub & 1) != 0;
                for (uint u = 0; u < 32; ++u) {
                    uint byte = (uint)qrow[u];
                    uint nib = hi ? (byte >> 4) : (byte & 0x0F);
                    ws[u * R88_PAD_R + tid] = ds * (float)nib - mm;
                }
            } else {
                for (uint u = 0; u < 32; ++u) { ws[u * R88_PAD_R + tid] = 0.0f; }
            }
        }
        // ── stage 64 tokens x 32 values: 2 threads per token, 16 values each ──
        {
            uint t = tid >> 1;
            uint v0 = (tid & 1) * 16;
            uint gt = t0 + t;
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + v0;
                for (uint u = 0; u < 16; ++u) { xs[(v0 + u) * R88_PAD_T + t] = xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xs[(v0 + u) * R88_PAD_T + t] = 0.0f; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── compute: 8 rows x 8 tokens from registers ──
        for (uint kk = 0; kk < R88_BK; ++kk) {
            threadgroup float *wp = ws + kk * R88_PAD_R + rb;
            threadgroup float *xp = xs + kk * R88_PAD_T + tb;
            float4 w0 = *(threadgroup float4 *)wp;
            float4 w1 = *(threadgroup float4 *)(wp + 4);
            float4 x0 = *(threadgroup float4 *)xp;
            float4 x1 = *(threadgroup float4 *)(xp + 4);
            float wv[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
            float xv[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
#pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    acc[i][j] += wv[i] * xv[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint tbase = t0 + tb;
    uint rbase = r0 + rb;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        uint grow = rbase + i;
        if (grow >= rows) { continue; }
#pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            uint gt = tbase + j;
            if (gt >= B) { continue; }
            y[(size_t)gt * rows + grow] = acc[i][j];
        }
    }
}

// ── peak probes: what can this GPU actually do? ────────────────────────────
// Every GEMM schedule tried (4x4, 8x4, 8x8 register tiles, 128/256 threads,
// single/double buffered staging, 32/64/128-row tiles, scalar and float4
// threadgroup reads) lands at 1.2-1.3 TFLOP/s, while llama.cpp measures 4.0 on
// the same box. These two kernels have no memory traffic at all: they measure
// the fp32 FMA pipe and the simdgroup_matrix pipe so the ceiling is a fact
// rather than an inference.
kernel void peak_fma(device float *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a0 = (float)(tid & 7) * 1e-3f + 1.0f;
    float a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
    float b = 1.0000001f, c = 1e-9f;
    for (uint i = 0; i < 8192; ++i) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 12345.0f) { out[tid] = a0; }
}

kernel void peak_fma16(device float *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float a[16];
    for (uint i = 0; i < 16; ++i) { a[i] = 1.0f + (float)((tid + i) & 7) * 1e-3f; }
    float b = 1.0000001f, c = 1e-9f;
    for (uint i = 0; i < 4096; ++i) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 16; ++j) { a[j] = fma(a[j], b, c); }
    }
    float s = 0.0f;
    for (uint j = 0; j < 16; ++j) { s += a[j]; }
    if (s == 12345.0f) { out[tid] = s; }
}

kernel void peak_mma(device float *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    simdgroup_matrix<half, 8, 8> A, B;
    simdgroup_matrix<half, 8, 8> C[4];
    half v = (half)((tid & 7) * 0.001f + 1.0f);
    simdgroup_matrix<half, 8, 8> Z;
    Z = simdgroup_matrix<half, 8, 8>(v);
    A = Z; B = Z;
    C[0] = Z; C[1] = Z; C[2] = Z; C[3] = Z;
    for (uint i = 0; i < 8192; ++i) {
        simdgroup_multiply_accumulate(C[0], A, B, C[0]);
        simdgroup_multiply_accumulate(C[1], A, B, C[1]);
        simdgroup_multiply_accumulate(C[2], A, B, C[2]);
        simdgroup_multiply_accumulate(C[3], A, B, C[3]);
    }
    threadgroup half tmp[64];
    simdgroup_store(C[0], tmp, 8, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tmp[0] == (half)12345.0f) { out[tid] = (float)tmp[1]; }
}

// ── the parity path: q4_K GEMM through the matrix pipe ─────────────────────
// LEDGER #7: fp32 tops out at 2.1 TFLOP/s on this GPU while simdgroup_matrix
// measures 18.8, and llama.cpp's 4.0 TFLOP/s is above the fp32 pipe entirely.
// So prefill parity needs the matrix pipe: dequantize a k-slice into fp16
// threadgroup tiles, then accumulate 8x8x8 fragments (fp16 multiply, fp32
// accumulate - the same arithmetic class llama.cpp's mul_mm uses).
//
// Tile 64 rows x 64 tokens, 128 threads = 4 simdgroups, one per 16-row band.
// Each simdgroup holds 2x8 = 16 accumulator fragments (32 registers/lane).
// Fragment access follows gemm_q4_k_mma exactly (the one MMA kernel with a
// passing parity test): BOTH tiles are stored [row][k] with k contiguous, A is
// loaded with pointer arithmetic and transpose=false, B with transpose=true so
// it reads as (k, token), and output goes through a threadgroup scratch rather
// than a transposed direct store.
#define MM_R 64
#define MM_T 64
#define MM_BK 32
#define MM_PADK 40
#define MM_THREADS 128

kernel void gemm_q4_k_mma64(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &B              [[buffer(5)]],
    constant uint       &rows           [[buffer(6)]],
    uint2 tg                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sgid                           [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    threadgroup half ws[MM_R][MM_PADK];
    threadgroup half xs[MM_T][MM_PADK];
    threadgroup float oc[4][64];

    uint r0 = tg.x * MM_R;
    uint t0 = tg.y * MM_T;
    uint nb = cols / 256;
    uint nbk = cols / MM_BK;

    simdgroup_matrix<float, 8, 8> acc[2][8];
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 8; ++j) { acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f); }
    }

    uint srow = tid >> 1;          // 64 rows, 2 threads each
    uint sk = (tid & 1) * 16;      // 16 of the 32 k values

    for (uint kb = 0; kb < nbk; ++kb) {
        uint k0 = kb * MM_BK;
        uint b = k0 / 256;
        uint sub = (k0 % 256) / 32;

        {
            uint grow = r0 + srow;
            if (grow < rows) {
                uint row_byte = w_rem + grow * nb * 144;
                device const uchar *blk = w_base + row_byte + b * 144;
                float d = float(read_f16(w_base, row_byte + b * 144));
                float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
                uint8_t sc, m;
                q4k_scale_min(blk + 4, sub, &sc, &m);
                float ds = d * (float)sc;
                float mm = dmin * (float)m;
                device const uchar *qrow = blk + 16 + 32 * (sub >> 1);
                bool hi = (sub & 1) != 0;
                for (uint u = 0; u < 16; ++u) {
                    uint byte = (uint)qrow[sk + u];
                    uint nib = hi ? (byte >> 4) : (byte & 0x0F);
                    ws[srow][sk + u] = (half)(ds * (float)nib - mm);
                }
            } else {
                for (uint u = 0; u < 16; ++u) { ws[srow][sk + u] = (half)0.0f; }
            }
        }
        {
            uint gt = t0 + srow;
            if (gt < B) {
                device const float *xt = x + (size_t)gt * cols + k0 + sk;
                for (uint u = 0; u < 16; ++u) { xs[srow][sk + u] = (half)xt[u]; }
            } else {
                for (uint u = 0; u < 16; ++u) { xs[srow][sk + u] = (half)0.0f; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint rl = sgid * 16;
        for (uint kk = 0; kk < MM_BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> afr[2];
            simdgroup_matrix<half, 8, 8> bfr[8];
            simdgroup_load(afr[0], &ws[rl + 0][kk], MM_PADK, ulong2(0, 0), false);
            simdgroup_load(afr[1], &ws[rl + 8][kk], MM_PADK, ulong2(0, 0), false);
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                simdgroup_load(bfr[j], &xs[j * 8][kk], MM_PADK, ulong2(0, 0), true);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
#pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], afr[i], bfr[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // epilogue: acc is (rows x tokens); y is token-major [B][rows]. Go through a
    // threadgroup scratch (start of the loop, i.e. epilogue only) so the layout
    // conversion is plain stores instead of a transposed simdgroup_store.
    uint tbase = t0 + (sgid & 3);   // unused banding, kept simple below
    (void)tbase;
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 8; ++j) {
            simdgroup_store(acc[i][j], &oc[sgid][0], 8, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint grow = r0 + sgid * 16 + i * 8 + (lane >> 3);
            uint gt = t0 + j * 8 + (lane & 7);
            if (grow < rows && gt < B) {
                y[(size_t)gt * rows + grow] = oc[sgid][lane];
            }
            // each lane holds row (lane>>3), token (lane&7); cover the other
            // half of the 8x8 tile (rows 4..7) with a second pass
            uint grow2 = r0 + sgid * 16 + i * 8 + ((lane >> 3) + 4);
            if (grow2 < rows && gt < B) {
                y[(size_t)gt * rows + grow2] = oc[sgid][lane + 32];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ── decode GEMV v2: 8 rows per threadgroup, one warp per row ───────────────
// LEDGER #6/arm-D measured the shipped q4_K decode GEMV at 17-21 GB/s on the
// 1.8 MB shapes while Q8_0 reaches 91 on the same path, so the gap is the
// kernel's instruction mix, not DRAM. gemv_q4_k_c gives every row its own
// 32-thread threadgroup and reads one qs BYTE per lane per iteration; this one
// gives a warp one row, reads 8 bytes of qs (uint2) and 8 floats of x
// (two float4) per block, and hoists the (scale, min) pair for the lane's own
// 32-value sub-block out of the value loop. Lane L owns sub-block L/4 and
// values 8*(L%4) .. +7 of it, so 32 lanes cover exactly the block's 256 values.
#define GV2_ROWS 8
#define GV2_THREADS 256

kernel void gemv_q4_k_v2(
    device const uchar  *w_base         [[buffer(0)]],
    device const float  *x              [[buffer(1)]],
    device float        *y              [[buffer(2)]],
    constant uint       &cols           [[buffer(3)]],
    constant uint       &w_rem          [[buffer(4)]],
    constant uint       &rows           [[buffer(5)]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint wid                            [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * GV2_ROWS + wid;
    if (row >= rows) { return; }
    uint nb = cols / 256;
    uint row_byte = w_rem + row * nb * 144;
    uint sub = lane >> 2;                 // 0..7
    uint vbase = (lane & 3) * 8;           // 0/8/16/24 inside the sub-block
    uint qoff = 16 + 32 * (sub >> 1) + vbase;
    bool high = (sub & 1) != 0;
    uint xbase = 32 * sub + vbase;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const uchar *blk = w_base + row_byte + b * 144;
        float d = float(read_f16(w_base, row_byte + b * 144));
        float dmin = float(read_f16(w_base, row_byte + b * 144 + 2));
        uint8_t sc, m;
        q4k_scale_min(blk + 4, sub, &sc, &m);
        float ds = d * (float)sc;
        float mm = dmin * (float)m;
        uint q0 = *(device const uint *)(blk + qoff);
        uint q1 = *(device const uint *)(blk + qoff + 4);
        uint xo = b * 256 + xbase;
        device const float4 *xf = (device const float4 *)(x + xo);
        float4 xa = xf[0];
        float4 xb = xf[1];
        uint sh = high ? 4u : 0u;
        acc += (ds * (float)((q0 >> (0u + sh)) & 0xFu) - mm) * xa.x;
        acc += (ds * (float)((q0 >> (8u + sh)) & 0xFu) - mm) * xa.y;
        acc += (ds * (float)((q0 >> (16u + sh)) & 0xFu) - mm) * xa.z;
        acc += (ds * (float)((q0 >> (24u + sh)) & 0xFu) - mm) * xa.w;
        acc += (ds * (float)((q1 >> (0u + sh)) & 0xFu) - mm) * xb.x;
        acc += (ds * (float)((q1 >> (8u + sh)) & 0xFu) - mm) * xb.y;
        acc += (ds * (float)((q1 >> (16u + sh)) & 0xFu) - mm) * xb.z;
        acc += (ds * (float)((q1 >> (24u + sh)) & 0xFu) - mm) * xb.w;
    }
    acc = simd_sum(acc);
    if (lane == 0) { y[row] = acc; }
}
