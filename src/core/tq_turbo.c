/**
 * TurboQuant composite — PolarQuant + QJL residual correction
 *
 * Quantizes keys using PolarQuant first, then computes the residual
 * and quantizes it with QJL for additional correction.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>

/* External references to polar and qjl functions */
extern void tq_polar_quantize_ref(const float* src, void* dst, int n);
extern void tq_polar_dequantize_ref(const void* src, float* dst, int n);
extern void tq_qjl_quantize_ref(const float* src, void* dst, int n);
extern void tq_qjl_dequantize_ref(const void* src, float* dst, int n);

/* ---------- TurboQuant quantize (reference) ---------- */

void tq_turbo_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo* block = (block_tq_turbo*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Stage 1: PolarQuant */
    tq_polar_quantize_ref(src, &block->polar, dim);

    /* Compute residual = original - polar_reconstruction */
    float recon[TQ_BK];
    float residual[TQ_BK];
    tq_polar_dequantize_ref(&block->polar, recon, dim);

    for (int i = 0; i < dim; i++) {
        residual[i] = src[i] - recon[i];
    }
    for (int i = dim; i < TQ_BK; i++) {
        residual[i] = 0.0f;
    }

    /* Stage 2: QJL on residual */
    tq_qjl_quantize_ref(residual, &block->residual, dim);
}

/* ---------- TurboQuant dequantize (reference) ---------- */

void tq_turbo_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo* block = (const block_tq_turbo*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Reconstruct from polar */
    tq_polar_dequantize_ref(&block->polar, dst, dim);

    /* Add QJL residual correction */
    float residual[TQ_BK];
    tq_qjl_dequantize_ref(&block->residual, residual, dim);

    for (int i = 0; i < dim; i++) {
        dst[i] += residual[i];
    }
}

/* ---------- TurboQuant attention (reference) ---------- */

void tq_turbo_attention_ref(const float* query, const void* kv_cache,
                            float* scores, int seq_len, int head_dim) {
    const block_tq_turbo* blocks = (const block_tq_turbo*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    float dequant[TQ_BK];

    for (int s = 0; s < seq_len; s++) {
        tq_turbo_dequantize_ref(&blocks[s], dequant, dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            dot += query[d] * dequant[d];
        }
        scores[s] = dot;
    }
}
