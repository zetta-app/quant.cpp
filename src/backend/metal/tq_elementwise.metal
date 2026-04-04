/**
 * TurboQuant -- Element-wise Metal compute shaders
 *
 * Provides GPU kernels for operations between matmuls that would
 * otherwise force GPU->CPU->GPU round-trips:
 *   - RMSNorm (with threadgroup reduction)
 *   - SiLU activation
 *   - Element-wise multiply
 *   - Vector add
 */
#include <metal_stdlib>
using namespace metal;

/* ============================================================
 * SIMD-group sum reduction (matches tq_polar.metal helpers)
 * ============================================================ */

inline float simd_reduce_sum_ew(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

/* ============================================================
 * RMSNorm kernel
 *
 * out[i] = (x[i] / rms(x)) * weight[i]
 * rms(x) = sqrt(mean(x^2) + eps)
 *
 * Two-phase design:
 *   Phase 1: Parallel reduction to compute sum of squares.
 *   Phase 2: Each thread normalizes and scales its element(s).
 *
 * Dispatch: one threadgroup per row (n elements).
 * Threadgroup size: 256 threads (8 SIMD groups of 32).
 * Each thread handles ceil(n / tgsize) elements.
 * ============================================================ */
kernel void rmsnorm(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgsize     [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]])
{
    /* Scratch for cross-SIMD-group reduction (max 8 SIMD groups for TG=256) */
    threadgroup float scratch[8];

    /* Phase 1: accumulate sum of squares */
    float ss = 0.0f;
    for (uint i = tid; i < n; i += tgsize) {
        float v = x[i];
        ss += v * v;
    }

    /* SIMD-group reduction */
    ss = simd_reduce_sum_ew(ss);
    uint num_simd_groups = (tgsize + 31) / 32;

    if (simd_lane == 0) {
        scratch[simd_gid] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Final reduction in first SIMD group */
    if (simd_gid == 0) {
        float val = (tid < num_simd_groups) ? scratch[tid] : 0.0f;
        val = simd_reduce_sum_ew(val);
        if (tid == 0) {
            scratch[0] = rsqrt(val / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Phase 2: normalize and scale */
    float inv_rms = scratch[0];
    for (uint i = tid; i < n; i += tgsize) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/* ============================================================
 * SiLU (Sigmoid Linear Unit) activation
 *
 * out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
 *
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void silu(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        float v = x[tid];
        out[tid] = v / (1.0f + exp(-v));
    }
}

/* ============================================================
 * Element-wise multiply
 *
 * out[i] = a[i] * b[i]
 *
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void mul_elementwise(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        out[tid] = a[tid] * b[tid];
    }
}

/* ============================================================
 * Vector add
 *
 * out[i] = a[i] + b[i]
 *
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void add_vectors(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}
