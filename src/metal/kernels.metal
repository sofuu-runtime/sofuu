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

kernel void swiglu(
    device float       *g               [[buffer(0)]],
    device const float *u               [[buffer(1)]],
    uint id                             [[thread_position_in_grid]]
) {
    float val = g[id];
    float silu = val / (1.0f + exp(-val));
    g[id] = silu * u[id];
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
