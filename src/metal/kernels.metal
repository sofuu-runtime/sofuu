#include <metal_stdlib>
using namespace metal;

// ── Apple Silicon Unified GPU Engine (Metal Shading Language) ─────────
// Zero-copy compute shaders with arbitrary alignment offset rebasing for
// Edge LLMs & Liquid Foundation Models.

static inline uint read_u32(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    if ((uintptr_t(p) & 3) == 0) {
        return *(device const uint*)p;
    } else {
        return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    }
}

static inline bfloat read_bf16(device const uchar* base, uint byte_offset) {
    device const uchar* p = base + byte_offset;
    if ((uintptr_t(p) & 1) == 0) {
        return *(device const bfloat*)p;
    } else {
        ushort u = ushort(p[0]) | (ushort(p[1]) << 8);
        return as_type<bfloat>(u);
    }
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

// ── GEMV (BF16 weights, Float32 activations) ──────────────────────────
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
        device const bfloat *row_w = (device const bfloat*)(w_base + row_byte);
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
            sum += float(read_bf16(w_base, row_byte + i * 2)) * x[i];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

// ── GEMV (FP16 weights, Float32 activations) ──────────────────────────
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

// ── GEMV (FP32 weights, Float32 activations) ──────────────────────────
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

// ── RMSNorm Kernel (Float32) ──────────────────────────────────────────
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
        y[i] = x[i] * rms * float(read_bf16(w_base, w_rem + i * 2));
    }
}

// ── SwiGLU Kernel (Elementwise Fused SiLU * Multiply) ─────────────────
kernel void swiglu(
    device float       *g               [[buffer(0)]],
    device const float *u               [[buffer(1)]],
    uint id                             [[thread_position_in_grid]]
) {
    float val = g[id];
    float silu = val / (1.0f + exp(-val));
    g[id] = silu * u[id];
}

// ── Conv1D Kernel for Liquid Foundation Models (LFM-2) ────────────────
kernel void conv1d_3(
    device const uchar *conv_w_base     [[buffer(0)]],
    device const float *state           [[buffer(1)]],
    device float       *out             [[buffer(2)]],
    constant uint      &w_rem           [[buffer(3)]],
    uint c                              [[thread_position_in_grid]]
) {
    uint base = w_rem + c * 3 * 2;
    float w0 = float(read_bf16(conv_w_base, base + 0));
    float w1 = float(read_bf16(conv_w_base, base + 2));
    float w2 = float(read_bf16(conv_w_base, base + 4));

    device const float *st = state + c * 3;
    float sum = w0 * st[0] + w1 * st[1] + w2 * st[2];
    out[c] = sum;
}

// ── MLX 4-Bit Affine Quantized GEMV ──────────────────────────────────
kernel void gemv_mlx4_affine(
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
        float s = float(read_bf16(scales_base, s_row_byte + g_idx * 2));
        float b = float(read_bf16(biases_base, b_row_byte + g_idx * 2));

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
