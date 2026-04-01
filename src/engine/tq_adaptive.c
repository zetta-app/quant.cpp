/**
 * tq_adaptive.c -- Adaptive compression utilities
 *
 * Implements:
 *   - Per-layer bit allocation recommendation (kurtosis-based)
 *   - Attention entropy computation
 *   - Online Lloyd-Max codebook calibration
 */

#include "turboquant/tq_engine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ============================================================
 * Per-layer bit allocation recommendation
 *
 * Layers with high post-RHT kurtosis (> threshold) have heavier
 * tails and need more bits to avoid quantization error.
 * Layers with low kurtosis can use aggressive 1-bit quantization.
 * ============================================================ */

void tq_recommend_layer_bits(const float* kurtosis_values, int n_layers,
                             int* recommended_bits, float* avg_bits)
{
    if (!kurtosis_values || !recommended_bits || n_layers <= 0) return;

    /* Threshold: kurtosis > 6.0 -> 3-bit, otherwise 1-bit.
     * Normal distribution has kurtosis = 3.0.
     * Post-RHT values typically range from 3.9 to 7.9.
     * Higher kurtosis = heavier tails = more precision needed. */
    const float kurtosis_threshold = 6.0f;

    int total_bits = 0;
    for (int l = 0; l < n_layers; l++) {
        if (kurtosis_values[l] > kurtosis_threshold) {
            recommended_bits[l] = 3;  /* turbo_kv_3b */
        } else {
            recommended_bits[l] = 1;  /* turbo_kv_1b */
        }
        total_bits += recommended_bits[l];
    }

    if (avg_bits) {
        *avg_bits = (float)total_bits / (float)n_layers;
    }
}

/* ============================================================
 * Attention entropy
 *
 * Measures how "spread out" the attention distribution is.
 * H = -sum(p_i * log2(p_i))
 * Low entropy = sharp attention (few tokens dominate)
 * High entropy = diffuse attention (many tokens contribute)
 * ============================================================ */

float tq_attention_entropy(const float* probs, int seq_len)
{
    if (!probs || seq_len <= 0) return 0.0f;

    double entropy = 0.0;
    for (int i = 0; i < seq_len; i++) {
        float p = probs[i];
        if (p > 1e-10f) {
            entropy -= (double)p * log2((double)p);
        }
    }
    return (float)entropy;
}

/* ============================================================
 * Online Lloyd-Max codebook calibration
 *
 * Given empirical data, iteratively finds optimal centroids
 * that minimize MSE for scalar quantization.
 *
 * Algorithm:
 *   1. Initialize centroids uniformly in data range
 *   2. For each iteration:
 *      a. Assign each sample to nearest centroid
 *      b. Update centroids as mean of assigned samples
 *      c. Update boundaries as midpoints between centroids
 * ============================================================ */

float tq_calibrate_codebook(const float* data, int n_samples,
                            int n_levels, int iterations,
                            float* centroids, float* boundaries)
{
    if (!data || !centroids || n_samples <= 0 || n_levels <= 1 || iterations <= 0) {
        return -1.0f;
    }

    /* Find data range */
    float dmin = data[0], dmax = data[0];
    for (int i = 1; i < n_samples; i++) {
        if (data[i] < dmin) dmin = data[i];
        if (data[i] > dmax) dmax = data[i];
    }

    /* Initialize centroids uniformly in data range */
    for (int c = 0; c < n_levels; c++) {
        centroids[c] = dmin + (dmax - dmin) * ((float)c + 0.5f) / (float)n_levels;
    }

    /* Allocate workspace for centroid sums and counts */
    double* c_sum = (double*)calloc((size_t)n_levels, sizeof(double));
    int* c_count = (int*)calloc((size_t)n_levels, sizeof(int));
    float* bounds = (float*)calloc((size_t)(n_levels - 1), sizeof(float));
    if (!c_sum || !c_count || !bounds) {
        free(c_sum);
        free(c_count);
        free(bounds);
        return -1.0f;
    }

    /* Lloyd-Max iterations */
    for (int iter = 0; iter < iterations; iter++) {
        /* Compute decision boundaries (midpoints between centroids) */
        for (int b = 0; b < n_levels - 1; b++) {
            bounds[b] = (centroids[b] + centroids[b + 1]) * 0.5f;
        }

        /* Reset accumulators */
        memset(c_sum, 0, (size_t)n_levels * sizeof(double));
        memset(c_count, 0, (size_t)n_levels * sizeof(int));

        /* Assign each sample to nearest centroid and accumulate */
        for (int i = 0; i < n_samples; i++) {
            float val = data[i];
            /* Binary search for the correct bin */
            int bin = 0;
            for (int b = 0; b < n_levels - 1; b++) {
                if (val > bounds[b]) bin = b + 1;
            }
            c_sum[bin] += (double)val;
            c_count[bin]++;
        }

        /* Update centroids */
        for (int c = 0; c < n_levels; c++) {
            if (c_count[c] > 0) {
                centroids[c] = (float)(c_sum[c] / (double)c_count[c]);
            }
        }
    }

    /* Copy final boundaries if requested */
    if (boundaries) {
        for (int b = 0; b < n_levels - 1; b++) {
            boundaries[b] = (centroids[b] + centroids[b + 1]) * 0.5f;
        }
    }

    /* Compute final MSE */
    double mse = 0.0;
    for (int i = 0; i < n_samples; i++) {
        float val = data[i];
        /* Find nearest centroid */
        float best_dist = fabsf(val - centroids[0]);
        for (int c = 1; c < n_levels; c++) {
            float dist = fabsf(val - centroids[c]);
            if (dist < best_dist) best_dist = dist;
        }
        mse += (double)(best_dist * best_dist);
    }
    mse /= (double)n_samples;

    free(c_sum);
    free(c_count);
    free(bounds);

    return (float)mse;
}
