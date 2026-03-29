#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
void tq_polar_attention_ref(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);
}
#include <cmath>
#include <vector>

TEST(PolarQuant, RoundtripBasic) {
    // Create a test vector of TQ_BK floats
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_polar block;
    tq_polar_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_polar_dequantize_ref(&block, output.data(), TQ_BK);

    // Check roundtrip MSE
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.5); // Lossy, but bounded
}

TEST(PolarQuant, AttentionAccuracy) {
    std::vector<float> key(128), query(128);
    for (int i = 0; i < 128; i++) {
        key[i] = sinf(i * 0.05f);
        query[i] = cosf(i * 0.05f);
    }

    // FP32 dot product
    float fp32_score = 0;
    for (int i = 0; i < 128; i++) fp32_score += query[i] * key[i];

    // Quantized attention
    block_tq_polar block;
    tq_polar_quantize_ref(key.data(), &block, 128);
    float quant_score = 0;
    tq_polar_attention_ref(query.data(), &block, &quant_score, 1, 128);

    // Polar quantization is lossy; verify the score is finite and
    // in a reasonable range (within 5x magnitude + offset)
    EXPECT_NEAR(quant_score, fp32_score, fabsf(fp32_score) * 5.0f + 1.0f);
}

TEST(PolarQuant, DirectAttentionAccuracy) {
    // Test that LUT-based direct attention matches dequantize+dot approach
    const int dim = 128;
    const int seq_len = 8;

    std::vector<float> keys(seq_len * dim);
    std::vector<float> query(dim);
    for (int i = 0; i < dim; i++) {
        query[i] = cosf(i * 0.07f) + 0.3f * sinf(i * 0.2f);
    }
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            keys[s * dim + d] = sinf((s + 1) * 0.05f * d + s * 0.7f);
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

    // Quantize and compute attention via direct LUT method
    std::vector<block_tq_polar> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        tq_polar_quantize_ref(&keys[s * dim], &blocks[s], dim);
    }

    std::vector<float> quant_scores(seq_len);
    tq_polar_attention_ref(query.data(), blocks.data(), quant_scores.data(), seq_len, dim);

    // Also compute via dequantize+dot to verify they match exactly
    for (int s = 0; s < seq_len; s++) {
        float dequant[TQ_BK];
        tq_polar_dequantize_ref(&blocks[s], dequant, dim);
        float dot = 0;
        for (int d = 0; d < dim; d++) {
            dot += query[d] * dequant[d];
        }
        // The LUT method should produce identical results to dequantize+dot
        // since they use the same math, just reordered
        EXPECT_NEAR(quant_scores[s], dot, fabsf(dot) * 1e-5f + 1e-5f)
            << "LUT vs dequantize mismatch at position " << s;
    }

    // Check cosine similarity of score vectors vs FP32 reference
    double dot_sq = 0, norm_fp32 = 0, norm_quant = 0;
    for (int s = 0; s < seq_len; s++) {
        dot_sq += (double)fp32_scores[s] * quant_scores[s];
        norm_fp32 += (double)fp32_scores[s] * fp32_scores[s];
        norm_quant += (double)quant_scores[s] * quant_scores[s];
    }
    double cos_sim = dot_sq / (sqrt(norm_fp32) * sqrt(norm_quant) + 1e-10);
    EXPECT_GT(cos_sim, 0.99) << "Score vector cosine similarity too low: " << cos_sim;
}

TEST(PolarQuant, DirectAttentionSinglePair) {
    // Verify single key-query direct attention
    const int dim = 128;
    std::vector<float> key(dim), query(dim);
    for (int i = 0; i < dim; i++) {
        key[i] = sinf(i * 0.05f);
        query[i] = cosf(i * 0.05f);
    }

    float fp32_score = 0;
    for (int i = 0; i < dim; i++) fp32_score += query[i] * key[i];

    block_tq_polar block;
    tq_polar_quantize_ref(key.data(), &block, dim);
    float quant_score = 0;
    tq_polar_attention_ref(query.data(), &block, &quant_score, 1, dim);

    // PolarQuant 4-bit should be fairly accurate
    EXPECT_NEAR(quant_score, fp32_score, fabsf(fp32_score) * 0.5f + 1.0f)
        << "Direct attention score too far from FP32: quant=" << quant_score
        << " fp32=" << fp32_score;
}

TEST(PolarQuant, ZeroVector) {
    std::vector<float> input(TQ_BK, 0.0f);
    block_tq_polar block;
    tq_polar_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_polar_dequantize_ref(&block, output.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(output[i], 0.0f, 0.01f);
    }
}

TEST(PolarQuant, BlockSize) {
    // Verify the block struct size is as expected
    EXPECT_EQ(sizeof(block_tq_polar), 8u + TQ_BK / 2);
}
