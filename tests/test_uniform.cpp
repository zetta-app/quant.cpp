#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
}
#include <cmath>
#include <vector>

TEST(Uniform4B, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_uniform_4b block;
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, output.data(), TQ_BK);

    // 4-bit uniform should have low MSE
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.01); // Very low MSE for 4-bit on [-1, 1] range
}

TEST(Uniform4B, ExtremeValues) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = (float)i / TQ_BK * 100.0f - 50.0f;

    block_tq_uniform_4b block;
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, output.data(), TQ_BK);

    // MSE scales with range^2 / (16^2); range=100 -> step~6.67 -> MSE~3.7
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 5.0); // Wider range = higher MSE but still bounded
}

TEST(Uniform4B, BlockSize) {
    EXPECT_EQ(sizeof(block_tq_uniform_4b), 4u + TQ_BK / 2);
}

TEST(Uniform2B, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_uniform_2b block;
    tq_uniform_2b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_2b_dequantize_ref(&block, output.data(), TQ_BK);

    // 2-bit is more lossy than 4-bit
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.15); // Higher MSE for 2-bit, but still bounded
}

TEST(Uniform2B, BlockSize) {
    EXPECT_EQ(sizeof(block_tq_uniform_2b), 4u + TQ_BK / 4);
}

TEST(Uniform4B, ConstantInput) {
    // All same value should roundtrip perfectly
    std::vector<float> input(TQ_BK, 3.14f);

    block_tq_uniform_4b block;
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, output.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(output[i], 3.14f, 0.1f);
    }
}

TEST(Uniform4B, Attention) {
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    // Create query vector
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    // Create key vectors and quantize them
    std::vector<block_tq_uniform_4b> blocks(seq_len);
    std::vector<std::vector<float>> keys(seq_len, std::vector<float>(head_dim));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 1.0f + d * 0.1f);
        tq_uniform_4b_quantize_ref(keys[s].data(), &blocks[s], head_dim);
    }

    // Compute attention scores via the new function
    std::vector<float> scores(seq_len);
    tq_uniform_4b_attention_ref(query.data(), blocks.data(), scores.data(),
                                 seq_len, head_dim);

    // Compare with FP32 dot product on dequantized keys
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_uniform_4b_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float fp32_dot = 0;
        for (int d = 0; d < head_dim; d++) fp32_dot += query[d] * deq[d];
        EXPECT_NEAR(scores[s], fp32_dot, 1e-4f);
    }
}

TEST(Uniform2B, Attention) {
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    // Create query vector
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    // Create key vectors and quantize them
    std::vector<block_tq_uniform_2b> blocks(seq_len);
    std::vector<std::vector<float>> keys(seq_len, std::vector<float>(head_dim));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d++)
            keys[s][d] = sinf(s * 1.0f + d * 0.1f);
        tq_uniform_2b_quantize_ref(keys[s].data(), &blocks[s], head_dim);
    }

    // Compute attention scores via the new function
    std::vector<float> scores(seq_len);
    tq_uniform_2b_attention_ref(query.data(), blocks.data(), scores.data(),
                                 seq_len, head_dim);

    // Compare with FP32 dot product on dequantized keys
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_uniform_2b_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float fp32_dot = 0;
        for (int d = 0; d < head_dim; d++) fp32_dot += query[d] * deq[d];
        EXPECT_NEAR(scores[s], fp32_dot, 1e-4f);
    }
}
