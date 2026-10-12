#include <metal_stdlib>
using namespace metal;

// ── Apple Silicon Unified GPU Engine (Metal Shading Language) ─────────
// Zero-copy compute shaders with 256-byte alignment offset rebasing for
// Edge LLMs & Liquid Foundation Models.

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
    device const bfloat *w = (device const bfloat*)(w_base + w_rem);
    float sum = 0.0f;
    device const bfloat *row_w = w + row * cols;

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
    device const half *w = (device const half*)(w_base + w_rem);
    float sum = 0.0f;
    device const half *row_w = w + row * cols;

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
    device const float *w = (device const float*)(w_base + w_rem);
    float sum = 0.0f;
    device const float *row_w = w + row * cols;

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
    device const bfloat *w = (device const bfloat*)(w_base + w_rem);
    float sum_sq = 0.0f;
    for (uint i = lane; i < dim; i += 32) {
        float val = x[i];
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);
    float rms = rsqrt(sum_sq / float(dim) + eps);

    for (uint i = lane; i < dim; i += 32) {
        y[i] = x[i] * rms * float(w[i]);
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
    device const bfloat *conv_w = (device const bfloat*)(conv_w_base + w_rem);
    device const bfloat *cw = conv_w + c * 3;
    device const float  *st = state + c * 3;
    float sum = float(cw[0]) * st[0] + float(cw[1]) * st[1] + float(cw[2]) * st[2];
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
    device const uint   *q_w    = (device const uint*)(w_base + rems.x);
    device const bfloat *scales = (device const bfloat*)(scales_base + rems.y);
    device const bfloat *biases = (device const bfloat*)(biases_base + rems.z);

    float sum = 0.0f;
    uint u32_cols = cols / 8;
    uint n_groups = cols / group_size;

    device const uint   *row_qw = q_w + row * u32_cols;
    device const bfloat *row_s  = scales + row * n_groups;
    device const bfloat *row_b  = biases + row * n_groups;

    for (uint col_u32 = lane; col_u32 < u32_cols; col_u32 += 32) {
        uint val = row_qw[col_u32];
        uint c_base = col_u32 * 8;
        uint g_idx = c_base / group_size;
        float s = float(row_s[g_idx]);
        float b = float(row_b[g_idx]);

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
