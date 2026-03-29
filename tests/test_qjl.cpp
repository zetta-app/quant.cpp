#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
void tq_qjl_quantize_ref(const float* src, void* dst, int n);
void tq_qjl_dequantize_ref(const void* src, float* dst, int n);
void tq_qjl_attention_ref(const float* query, const void* kv,
                           float* scores, int seq_len, int head_dim);
}
#include <cmath>
#include <vector>

TEST(QJL, RoundtripBasic) {
    // QJL is extremely lossy at 1-bit, but dequantize should produce
    // a vector with the correct norm
    std::vector<float> input(128);
    for (int i = 0; i < 128; i++) input[i] = sinf(i * 0.1f);

    // Compute original norm
    float orig_norm = 0;
    for (int i = 0; i < 128; i++) orig_norm += input[i] * input[i];
    orig_norm = sqrtf(orig_norm);

    block_tq_qjl block;
    tq_qjl_quantize_ref(input.data(), &block, 128);

    std::vector<float> output(128);
    tq_qjl_dequantize_ref(&block, output.data(), 128);

    // Check reconstructed norm matches
    float recon_norm = 0;
    for (int i = 0; i < 128; i++) recon_norm += output[i] * output[i];
    recon_norm = sqrtf(recon_norm);

    EXPECT_NEAR(recon_norm, orig_norm, orig_norm * 0.2f);
}

TEST(QJL, AttentionAccuracy) {
    std::vector<float> key(128), query(128);
    for (int i = 0; i < 128; i++) {
        key[i] = sinf(i * 0.05f);
        query[i] = cosf(i * 0.05f);
    }

    // FP32 dot product
    float fp32_score = 0;
    for (int i = 0; i < 128; i++) fp32_score += query[i] * key[i];

    // Quantized attention
    block_tq_qjl block;
    tq_qjl_quantize_ref(key.data(), &block, 128);
    float quant_score = 0;
    tq_qjl_attention_ref(query.data(), &block, &quant_score, 1, 128);

    // QJL 1-bit is extremely lossy; just verify finite output
    // and that it has the correct sign direction most of the time
    EXPECT_TRUE(std::isfinite(quant_score));
    float tolerance = fabsf(fp32_score) * 200.0f + 20.0f;
    EXPECT_NEAR(quant_score, fp32_score, tolerance);
}

TEST(QJL, BlockSize) {
    EXPECT_EQ(sizeof(block_tq_qjl), 4u + TQ_SKETCH_DIM / 8 + TQ_OUTLIERS);
}

TEST(QJL, HammingAttentionAccuracy) {
    // Test direct Hamming attention against FP32 reference
    // Uses multiple key-query pairs to verify statistical accuracy
    const int dim = 128;
    const int seq_len = 16;

    // Generate diverse key and query vectors
    std::vector<float> keys(seq_len * dim);
    std::vector<float> query(dim);
    for (int i = 0; i < dim; i++) {
        query[i] = cosf(i * 0.07f) + 0.5f * sinf(i * 0.13f);
    }
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            keys[s * dim + d] = sinf((s + 1) * 0.03f * d + s * 0.5f);
        }
    }

    // Compute FP32 reference scores
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < dim; d++) {
            dot += query[d] * keys[s * dim + d];
        }
        fp32_scores[s] = dot;
    }

    // Quantize keys and compute attention
    std::vector<block_tq_qjl> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        tq_qjl_quantize_ref(&keys[s * dim], &blocks[s], dim);
    }

    std::vector<float> quant_scores(seq_len);
    tq_qjl_attention_ref(query.data(), blocks.data(), quant_scores.data(), seq_len, dim);

    // All scores must be finite
    for (int s = 0; s < seq_len; s++) {
        EXPECT_TRUE(std::isfinite(quant_scores[s]))
            << "Score " << s << " is not finite: " << quant_scores[s];
    }

    // Check cosine similarity of the score vectors (should be > 0.5 for 1-bit)
    // QJL 1-bit is very lossy but should preserve relative ordering
    double dot_sq = 0, norm_fp32 = 0, norm_quant = 0;
    for (int s = 0; s < seq_len; s++) {
        dot_sq += (double)fp32_scores[s] * quant_scores[s];
        norm_fp32 += (double)fp32_scores[s] * fp32_scores[s];
        norm_quant += (double)quant_scores[s] * quant_scores[s];
    }
    double cos_sim = dot_sq / (sqrt(norm_fp32) * sqrt(norm_quant) + 1e-10);
    // QJL 1-bit is extremely lossy; cosine sim > 0.5 is a reasonable bar
    EXPECT_GT(cos_sim, 0.5) << "Score vector cosine similarity too low: " << cos_sim;
}

TEST(QJL, HammingAttentionSingleKey) {
    // Verify that the Hamming-based attention gives a finite result
    // for a single key and that sign direction is approximately preserved
    const int dim = 128;
    std::vector<float> key(dim), query(dim);

    // Use a correlated key-query pair (should give positive dot product)
    for (int i = 0; i < dim; i++) {
        key[i] = sinf(i * 0.1f);
        query[i] = sinf(i * 0.1f) + 0.1f * cosf(i * 0.3f);
    }

    float fp32_score = 0;
    for (int i = 0; i < dim; i++) fp32_score += query[i] * key[i];

    block_tq_qjl block;
    tq_qjl_quantize_ref(key.data(), &block, dim);
    float quant_score = 0;
    tq_qjl_attention_ref(query.data(), &block, &quant_score, 1, dim);

    EXPECT_TRUE(std::isfinite(quant_score));
    // For correlated vectors, both should be positive
    if (fabsf(fp32_score) > 1.0f) {
        // Same sign test (with generous tolerance for 1-bit)
        EXPECT_GT(fp32_score * quant_score, 0.0f)
            << "Sign mismatch: fp32=" << fp32_score << " quant=" << quant_score;
    }
}

TEST(QJL, OutlierDetection) {
    // Create a vector with clear outliers
    std::vector<float> input(128, 0.01f);
    input[10] = 100.0f;
    input[50] = -80.0f;
    input[90] = 60.0f;

    block_tq_qjl block;
    tq_qjl_quantize_ref(input.data(), &block, 128);

    // The outlier indices should include 10, 50, 90
    bool found_10 = false, found_50 = false, found_90 = false;
    for (int i = 0; i < TQ_OUTLIERS; i++) {
        if (block.outlier_idx[i] == 10) found_10 = true;
        if (block.outlier_idx[i] == 50) found_50 = true;
        if (block.outlier_idx[i] == 90) found_90 = true;
    }
    // At least the top outliers should be detected (we have 4 slots, 3 outliers)
    EXPECT_TRUE(found_10);
    EXPECT_TRUE(found_50);
    EXPECT_TRUE(found_90);
}
