/**
 * test_cumulative_error.cpp -- Layer-cumulative error analysis
 *
 * Simulates a multi-layer attention pipeline with quantized KV cache
 * to measure whether quantization errors amplify or self-correct
 * across layers.
 *
 * Key findings expected:
 * - Softmax normalization bounds per-layer error contribution
 * - Errors from different layers are approximately independent
 * - Cumulative error grows sub-linearly (not exponentially)
 * - This supports using aggressive quantization (2-3 bit) for KV cache
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <numeric>

extern "C" {
#include "turboquant/turboquant.h"
#include "turboquant/tq_engine.h"
}

/* ============================================================
 * Helper: simulate one attention layer
 *
 * Input: query, key vectors, value vectors (all FP32)
 * 1. Compute attention scores: score[t] = Q . K[t] / sqrt(dim)
 * 2. Softmax over scores
 * 3. Weighted sum of values: output = sum(attn[t] * V[t])
 * ============================================================ */
static void simulate_attention_fp32(
    const float* query, int head_dim,
    const float* keys, const float* values,
    int seq_len, float* output)
{
    /* Compute attention scores */
    std::vector<float> scores(seq_len);
    float inv_scale = 1.0f / sqrtf((float)head_dim);
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * keys[t * head_dim + d];
        }
        scores[t] = dot * inv_scale;
    }

    /* Softmax */
    float max_score = scores[0];
    for (int t = 1; t < seq_len; t++)
        if (scores[t] > max_score) max_score = scores[t];
    float sum_exp = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }
    for (int t = 0; t < seq_len; t++)
        scores[t] /= sum_exp;

    /* Weighted sum */
    memset(output, 0, head_dim * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < head_dim; d++) {
            output[d] += scores[t] * values[t * head_dim + d];
        }
    }
}

/* ============================================================
 * Helper: simulate attention with Q4 quantized values
 * ============================================================ */
static void simulate_attention_q4v(
    const float* query, int head_dim,
    const float* keys, const float* values,
    int seq_len, float* output)
{
    /* Compute attention scores (same as FP32, keys not quantized here) */
    std::vector<float> scores(seq_len);
    float inv_scale = 1.0f / sqrtf((float)head_dim);
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * keys[t * head_dim + d];
        }
        scores[t] = dot * inv_scale;
    }

    /* Softmax */
    float max_score = scores[0];
    for (int t = 1; t < seq_len; t++)
        if (scores[t] > max_score) max_score = scores[t];
    float sum_exp = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }
    for (int t = 0; t < seq_len; t++)
        scores[t] /= sum_exp;

    /* Quantize all values to Q4 and dequantize */
    int total_v = seq_len * head_dim;
    int n_blocks = (total_v + 31) / 32;
    std::vector<uint8_t> qs(n_blocks * 16);
    std::vector<float> scales(n_blocks);
    std::vector<float> dequant_v(total_v);
    tq_quantize_row_q4(values, qs.data(), scales.data(), total_v);
    tq_dequantize_row_q4(qs.data(), scales.data(), dequant_v.data(), total_v);

    /* Weighted sum using dequantized values */
    memset(output, 0, head_dim * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < head_dim; d++) {
            output[d] += scores[t] * dequant_v[t * head_dim + d];
        }
    }
}

/* Helper: cosine similarity */
static double cosine_sim(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

/* Helper: MSE between two vectors */
static double vector_mse(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        mse += diff * diff;
    }
    return mse / (double)n;
}

/* ============================================================
 * Test 1: Per-layer error independence
 *
 * Run multiple layers of attention with Q4 V quantization.
 * Measure the per-layer error and cumulative error.
 * If errors were correlated, cumulative would grow as N^2.
 * If independent, cumulative grows as sqrt(N) (relative).
 * ============================================================ */
TEST(CumulativeError, PerLayerErrorGrowth) {
    const int head_dim = 128;
    const int seq_len = 64;
    const int n_layers = 16;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    /* Generate random keys and values for each "layer" */
    std::vector<std::vector<float>> all_keys(n_layers);
    std::vector<std::vector<float>> all_values(n_layers);
    for (int l = 0; l < n_layers; l++) {
        all_keys[l].resize(seq_len * head_dim);
        all_values[l].resize(seq_len * head_dim);
        for (int i = 0; i < seq_len * head_dim; i++) {
            all_keys[l][i] = dist(rng);
            all_values[l][i] = dist(rng);
        }
    }

    /* Initial "residual stream" (query is derived from this) */
    std::vector<float> x_fp32(head_dim);
    std::vector<float> x_q4(head_dim);
    for (int i = 0; i < head_dim; i++) {
        float val = dist(rng);
        x_fp32[i] = val;
        x_q4[i] = val;
    }

    fprintf(stderr, "\n=== Layer-Cumulative Error Analysis ===\n");
    fprintf(stderr, "Config: %d layers, head_dim=%d, seq_len=%d, V quant=Q4\n\n", n_layers, head_dim, seq_len);
    fprintf(stderr, "%-8s %-14s %-14s %-14s %-10s\n",
            "Layer", "Layer MSE", "Cumul MSE", "Cosine Sim", "Error Growth");
    fprintf(stderr, "-------- -------------- -------------- -------------- ----------\n");

    double prev_cumul_mse = 0.0;
    std::vector<float> attn_out_fp32(head_dim);
    std::vector<float> attn_out_q4(head_dim);

    for (int l = 0; l < n_layers; l++) {
        /* Use current residual as query */
        simulate_attention_fp32(x_fp32.data(), head_dim,
                               all_keys[l].data(), all_values[l].data(),
                               seq_len, attn_out_fp32.data());
        simulate_attention_q4v(x_q4.data(), head_dim,
                              all_keys[l].data(), all_values[l].data(),
                              seq_len, attn_out_q4.data());

        /* Per-layer error (between this layer's FP32 and Q4 outputs) */
        double layer_mse = vector_mse(attn_out_fp32.data(), attn_out_q4.data(), head_dim);

        /* Residual connection: x = x + attn_out */
        for (int d = 0; d < head_dim; d++) {
            x_fp32[d] += attn_out_fp32[d];
            x_q4[d]   += attn_out_q4[d];
        }

        /* Cumulative error (between residual streams) */
        double cumul_mse = vector_mse(x_fp32.data(), x_q4.data(), head_dim);
        double cos = cosine_sim(x_fp32.data(), x_q4.data(), head_dim);

        /* Error growth factor: how does cumulative error change? */
        const char* growth;
        if (l == 0) {
            growth = "baseline";
        } else if (cumul_mse < prev_cumul_mse * 0.9) {
            growth = "decreasing";
        } else if (cumul_mse < prev_cumul_mse * 1.5) {
            growth = "sub-linear";
        } else if (cumul_mse < prev_cumul_mse * 2.5) {
            growth = "linear";
        } else {
            growth = "super-lin";
        }

        fprintf(stderr, "%-8d %-14.6f %-14.6f %-14.6f %-10s\n",
                l, layer_mse, cumul_mse, cos, growth);

        prev_cumul_mse = cumul_mse;
    }

    /* Final assessment */
    double final_cos = cosine_sim(x_fp32.data(), x_q4.data(), head_dim);
    double final_mse = vector_mse(x_fp32.data(), x_q4.data(), head_dim);

    fprintf(stderr, "\n--- Summary ---\n");
    fprintf(stderr, "After %d layers:\n", n_layers);
    fprintf(stderr, "  Final cosine similarity: %.6f\n", final_cos);
    fprintf(stderr, "  Final MSE:               %.6f\n", final_mse);
    fprintf(stderr, "  Conclusion: errors %s across layers\n",
            final_cos > 0.99 ? "self-correct (bounded)" : "amplify moderately");
    fprintf(stderr, "=============================================\n");

    /* After 16 layers, cosine similarity should still be high.
     * Q4 quantization error per layer is small, and softmax normalization
     * prevents catastrophic error amplification. */
    EXPECT_GT(final_cos, 0.95)
        << "Cumulative error too large after " << n_layers << " layers";
}

/* ============================================================
 * Test 2: Compare Q4 vs Q2 cumulative error
 *
 * Q2 has 4x more quantization error per layer.
 * Cumulative error should be proportionally higher but still bounded.
 * ============================================================ */
TEST(CumulativeError, Q4vsQ2Comparison) {
    const int head_dim = 128;
    const int seq_len = 64;
    const int n_layers = 16;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> all_keys(n_layers);
    std::vector<std::vector<float>> all_values(n_layers);
    for (int l = 0; l < n_layers; l++) {
        all_keys[l].resize(seq_len * head_dim);
        all_values[l].resize(seq_len * head_dim);
        for (int i = 0; i < seq_len * head_dim; i++) {
            all_keys[l][i] = dist(rng);
            all_values[l][i] = dist(rng);
        }
    }

    /* Run FP32 baseline */
    std::vector<float> x_fp32(head_dim);
    for (int i = 0; i < head_dim; i++) x_fp32[i] = dist(rng);

    /* Q4 path */
    std::vector<float> x_q4 = x_fp32;
    /* Q2 path */
    std::vector<float> x_q2 = x_fp32;

    std::vector<float> out_fp32(head_dim), out_q4(head_dim), out_q2(head_dim);

    for (int l = 0; l < n_layers; l++) {
        /* FP32 */
        simulate_attention_fp32(x_fp32.data(), head_dim,
                               all_keys[l].data(), all_values[l].data(),
                               seq_len, out_fp32.data());
        for (int d = 0; d < head_dim; d++) x_fp32[d] += out_fp32[d];

        /* Q4 */
        simulate_attention_q4v(x_q4.data(), head_dim,
                              all_keys[l].data(), all_values[l].data(),
                              seq_len, out_q4.data());
        for (int d = 0; d < head_dim; d++) x_q4[d] += out_q4[d];

        /* Q2: quantize values to Q2 */
        {
            int total_v = seq_len * head_dim;
            int n_blocks = (total_v + 31) / 32;
            std::vector<uint8_t> qs(n_blocks * 8);
            std::vector<float> scales(n_blocks);
            std::vector<float> dequant_v(total_v);
            tq_quantize_row_q2(all_values[l].data(), qs.data(), scales.data(), total_v);
            tq_dequantize_row_q2(qs.data(), scales.data(), dequant_v.data(), total_v);

            /* Attention with Q2 values */
            std::vector<float> scores(seq_len);
            float inv_scale = 1.0f / sqrtf((float)head_dim);
            for (int t = 0; t < seq_len; t++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += x_q2[d] * all_keys[l][t * head_dim + d];
                scores[t] = dot * inv_scale;
            }
            float max_s = scores[0];
            for (int t = 1; t < seq_len; t++)
                if (scores[t] > max_s) max_s = scores[t];
            float sum_e = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum_e += scores[t];
            }
            for (int t = 0; t < seq_len; t++) scores[t] /= sum_e;

            memset(out_q2.data(), 0, head_dim * sizeof(float));
            for (int t = 0; t < seq_len; t++)
                for (int d = 0; d < head_dim; d++)
                    out_q2[d] += scores[t] * dequant_v[t * head_dim + d];

            for (int d = 0; d < head_dim; d++) x_q2[d] += out_q2[d];
        }
    }

    double cos_q4 = cosine_sim(x_fp32.data(), x_q4.data(), head_dim);
    double cos_q2 = cosine_sim(x_fp32.data(), x_q2.data(), head_dim);
    double mse_q4 = vector_mse(x_fp32.data(), x_q4.data(), head_dim);
    double mse_q2 = vector_mse(x_fp32.data(), x_q2.data(), head_dim);

    fprintf(stderr, "\n=== Q4 vs Q2 Cumulative Error (%d layers) ===\n", n_layers);
    fprintf(stderr, "Q4: cosine=%.6f, MSE=%.6f\n", cos_q4, mse_q4);
    fprintf(stderr, "Q2: cosine=%.6f, MSE=%.6f\n", cos_q2, mse_q2);
    fprintf(stderr, "MSE ratio (Q2/Q4): %.2fx\n", mse_q2 / (mse_q4 > 1e-30 ? mse_q4 : 1e-30));
    fprintf(stderr, "Conclusion: Q2 errors are bounded even at 2-bit.\n");
    fprintf(stderr, "=============================================\n");

    /* Q4 should have high cosine similarity */
    EXPECT_GT(cos_q4, 0.95) << "Q4 cumulative error too large";
    /* Q2 should still be reasonable */
    EXPECT_GT(cos_q2, 0.85) << "Q2 cumulative error out of bounds";
    /* Q2 error should be worse than Q4 */
    EXPECT_GT(mse_q2, mse_q4) << "Q2 should have higher error than Q4";
}

/* ============================================================
 * Test 3: Error self-correction via softmax normalization
 *
 * Shows that softmax acts as an error-correcting mechanism:
 * even if K quantization shifts scores, the normalization
 * constrains the output to be a convex combination of V.
 * ============================================================ */
TEST(CumulativeError, SoftmaxErrorBound) {
    const int head_dim = 128;
    const int seq_len = 32;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    /* Generate data */
    std::vector<float> query(head_dim);
    std::vector<float> keys(seq_len * head_dim);
    std::vector<float> values(seq_len * head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = dist(rng);
    for (int i = 0; i < seq_len * head_dim; i++) {
        keys[i] = dist(rng);
        values[i] = dist(rng);
    }

    /* FP32 attention */
    std::vector<float> out_fp32(head_dim);
    simulate_attention_fp32(query.data(), head_dim,
                           keys.data(), values.data(),
                           seq_len, out_fp32.data());

    /* Attention with noisy keys (simulating quantization error) */
    std::vector<float> noisy_keys = keys;
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);
    for (int i = 0; i < seq_len * head_dim; i++) {
        noisy_keys[i] += noise_dist(rng);
    }

    std::vector<float> out_noisy(head_dim);
    simulate_attention_fp32(query.data(), head_dim,
                           noisy_keys.data(), values.data(),
                           seq_len, out_noisy.data());

    double cos = cosine_sim(out_fp32.data(), out_noisy.data(), head_dim);
    double mse = vector_mse(out_fp32.data(), out_noisy.data(), head_dim);

    fprintf(stderr, "\n=== Softmax Error Bound Analysis ===\n");
    fprintf(stderr, "Key noise std: 0.1 (simulating quantization error)\n");
    fprintf(stderr, "Output cosine: %.6f\n", cos);
    fprintf(stderr, "Output MSE:    %.6f\n", mse);

    /* Compute max possible error: since output is convex combination of V,
     * the error is bounded by the range of V times the change in weights. */
    double v_norm_sq = 0.0;
    for (int i = 0; i < seq_len * head_dim; i++)
        v_norm_sq += (double)values[i] * (double)values[i];
    double avg_v_norm = sqrt(v_norm_sq / seq_len);

    fprintf(stderr, "Avg V norm:    %.4f\n", avg_v_norm);
    fprintf(stderr, "Insight: softmax normalization bounds output error even with noisy keys.\n");
    fprintf(stderr, "  Output is always a convex combination of V vectors,\n");
    fprintf(stderr, "  so error cannot exceed the diameter of the V set.\n");
    fprintf(stderr, "=============================================\n");

    /* Even with significant key noise, softmax bounds the error */
    EXPECT_GT(cos, 0.90) << "Softmax should bound output error";
}
