#include <metal_stdlib>
using namespace metal;

// ── Apple Silicon Unified GPU Engine (Metal Shading Language) ─────────
// Zero-copy compute shaders for Edge LLMs & Liquid Foundation Models.

// ── GEMV (BF16 weights, Float32 activations) ──────────────────────────
kernel void gemv_bf16(
    device const bfloat *w          [[buffer(0)]],
    device const float  *x          [[buffer(1)]],
    device float        *y          [[buffer(2)]],
    constant uint       &cols       [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
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
    device const half  *w           [[buffer(0)]],
    device const float *x           [[buffer(1)]],
    device float       *y           [[buffer(2)]],
    constant uint      &cols        [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
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
    device const float *w           [[buffer(0)]],
    device const float *x           [[buffer(1)]],
    device float       *y           [[buffer(2)]],
    constant uint      &cols        [[buffer(3)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
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
    device const float  *x          [[buffer(0)]],
    device const bfloat *w          [[buffer(1)]],
    device float        *y          [[buffer(2)]],
    constant uint       &dim        [[buffer(3)]],
    constant float      &eps        [[buffer(4)]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
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
    device float  *g                [[buffer(0)]],
    device const float *u           [[buffer(1)]],
    uint id                         [[thread_position_in_grid]]
) {
    float val = g[id];
    float silu = val / (1.0f + exp(-val));
    g[id] = silu * u[id];
}

// ── Conv1D Kernel for Liquid Foundation Models (LFM-2) ────────────────
// 1D depthwise short convolution with kernel_size = 3 across channels
kernel void conv1d_3(
    device const bfloat *conv_w     [[buffer(0)]], // [channels, 3]
    device const float  *state      [[buffer(1)]], // [channels, 3] (history buffer)
    device float        *out        [[buffer(2)]], // [channels]
    uint c                          [[thread_position_in_grid]]
) {
    device const bfloat *cw = conv_w + c * 3;
    device const float  *st = state + c * 3;
    float sum = float(cw[0]) * st[0] + float(cw[1]) * st[1] + float(cw[2]) * st[2];
    out[c] = sum;
}
