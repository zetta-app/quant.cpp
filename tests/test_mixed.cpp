#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
void tq_mixed_4b8_quantize_ref(const float* src, void* dst, int n);
void tq_mixed_4b8_dequantize_ref(const void* src, float* dst, int n);
void tq_mixed_4b8_attention_ref(const float* query, const void* kv,
                                 float* scores, int seq_len, int head_dim);
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
}
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

/* ============================================================
 * Block size verification
 * ============================================================ */

TEST(Mixed4B8, BlockSize) {
    /* 2(scale) + 2(zero) + 4(outlier_idx) + 8(outlier_vals) + 64(qs) = 80 */
    EXPECT_EQ(sizeof(block_tq_mixed_4b8), 80u);
}

/* ============================================================
 * Basic roundtrip — smooth data (no strong outliers)
 * ============================================================ */

TEST(Mixed4B8, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);

    block_tq_mixed_4b8 block;
    tq_mixed_4b8_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_mixed_4b8_dequantize_ref(&block, output.data(), TQ_BK);

    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.01); /* comparable or better than uniform_4b */
}

/* ============================================================
 * Outlier detection: top-4 channels by |value| are found
 * ============================================================ */

TEST(Mixed4B8, OutlierDetection) {
    std::vector<float> input(TQ_BK, 0.0f);

    /* Plant 4 known outliers at specific positions */
    input[10] =  50.0f;
    input[33] = -40.0f;
    input[77] =  30.0f;
    input[120] = -20.0f;

    /* Fill rest with small values */
    for (int i = 0; i < TQ_BK; i++) {
        if (i != 10 && i != 33 && i != 77 && i != 120)
            input[i] = sinf(i * 0.05f) * 0.5f;
    }

    block_tq_mixed_4b8 block;
    tq_mixed_4b8_quantize_ref(input.data(), &block, TQ_BK);

    /* Verify the 4 outlier indices are exactly {10, 33, 77, 120} (any order) */
    std::vector<int> detected(TQ_MIXED_OUTLIERS);
    for (int o = 0; o < TQ_MIXED_OUTLIERS; o++) detected[o] = block.outlier_idx[o];
    std::sort(detected.begin(), detected.end());
    EXPECT_EQ(detected[0], 10);
    EXPECT_EQ(detected[1], 33);
    EXPECT_EQ(detected[2], 77);
    EXPECT_EQ(detected[3], 120);
}

/* ============================================================
 * Quality: MSE is LOWER than uniform_4b on outlier-heavy data
 * ============================================================ */

TEST(Mixed4B8, BetterThanUniformOnOutlierData) {
    /* Create data that mimics real KV cache: mostly small, a few large */
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f) * 0.5f;

    /* Inject outliers */
    input[5]   =  25.0f;
    input[42]  = -30.0f;
    input[99]  =  20.0f;
    input[110] = -15.0f;

    /* Quantize with mixed_4b8 */
    block_tq_mixed_4b8 mixed_block;
    tq_mixed_4b8_quantize_ref(input.data(), &mixed_block, TQ_BK);
    std::vector<float> mixed_out(TQ_BK);
    tq_mixed_4b8_dequantize_ref(&mixed_block, mixed_out.data(), TQ_BK);

    double mixed_mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - mixed_out[i];
        mixed_mse += d * d;
    }
    mixed_mse /= TQ_BK;

    /* Quantize with uniform_4b */
    block_tq_uniform_4b uni_block;
    tq_uniform_4b_quantize_ref(input.data(), &uni_block, TQ_BK);
    std::vector<float> uni_out(TQ_BK);
    tq_uniform_4b_dequantize_ref(&uni_block, uni_out.data(), TQ_BK);

    double uni_mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - uni_out[i];
        uni_mse += d * d;
    }
    uni_mse /= TQ_BK;

    /* Mixed should be significantly better */
    EXPECT_LT(mixed_mse, uni_mse * 0.5)
        << "mixed_mse=" << mixed_mse << " uni_mse=" << uni_mse;
}

/* ============================================================
 * Quality: Attention cosine similarity is HIGHER than uniform_4b
 * ============================================================ */

TEST(Mixed4B8, AttentionCosineBetterThanUniform) {
    const int head_dim = TQ_BK;

    /* Query vector */
    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    /* Key vector with outliers */
    std::vector<float> key(head_dim);
    for (int i = 0; i < head_dim; i++) key[i] = sinf(i * 0.1f) * 0.5f;
    key[3]   = 20.0f;
    key[50]  = -18.0f;
    key[88]  = 15.0f;
    key[127] = -12.0f;

    /* FP32 reference dot product */
    double fp32_dot = 0;
    for (int d = 0; d < head_dim; d++) fp32_dot += query[d] * key[d];

    /* Mixed_4b8 dot product */
    block_tq_mixed_4b8 mixed_block;
    tq_mixed_4b8_quantize_ref(key.data(), &mixed_block, head_dim);
    std::vector<float> mixed_deq(head_dim);
    tq_mixed_4b8_dequantize_ref(&mixed_block, mixed_deq.data(), head_dim);
    double mixed_dot = 0;
    for (int d = 0; d < head_dim; d++) mixed_dot += query[d] * mixed_deq[d];

    /* Uniform_4b dot product */
    block_tq_uniform_4b uni_block;
    tq_uniform_4b_quantize_ref(key.data(), &uni_block, head_dim);
    std::vector<float> uni_deq(head_dim);
    tq_uniform_4b_dequantize_ref(&uni_block, uni_deq.data(), head_dim);
    double uni_dot = 0;
    for (int d = 0; d < head_dim; d++) uni_dot += query[d] * uni_deq[d];

    /* Mixed should be closer to fp32 than uniform */
    double mixed_err = fabs(mixed_dot - fp32_dot);
    double uni_err   = fabs(uni_dot   - fp32_dot);
    EXPECT_LT(mixed_err, uni_err)
        << "mixed_err=" << mixed_err << " uni_err=" << uni_err;
}

/* ============================================================
 * Attention function consistency: scores match dequantize+dot
 * ============================================================ */

TEST(Mixed4B8, AttentionConsistency) {
    const int head_dim = TQ_BK;
    const int seq_len = 4;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    std::vector<block_tq_mixed_4b8> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> key(head_dim);
        for (int d = 0; d < head_dim; d++)
            key[d] = sinf(s * 1.0f + d * 0.1f) + ((d == s * 30) ? 10.0f : 0.0f);
        tq_mixed_4b8_quantize_ref(key.data(), &blocks[s], head_dim);
    }

    /* Compute via attention function */
    std::vector<float> scores(seq_len);
    tq_mixed_4b8_attention_ref(query.data(), blocks.data(), scores.data(),
                                seq_len, head_dim);

    /* Compare with manual dequantize + dot */
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> deq(head_dim);
        tq_mixed_4b8_dequantize_ref(&blocks[s], deq.data(), head_dim);
        float fp32_dot = 0;
        for (int d = 0; d < head_dim; d++) fp32_dot += query[d] * deq[d];
        EXPECT_NEAR(scores[s], fp32_dot, 1e-4f);
    }
}

/* ============================================================
 * Constant input: all same value
 * ============================================================ */

TEST(Mixed4B8, ConstantInput) {
    std::vector<float> input(TQ_BK, 3.14f);

    block_tq_mixed_4b8 block;
    tq_mixed_4b8_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_mixed_4b8_dequantize_ref(&block, output.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(output[i], 3.14f, 0.1f);
    }
}

/* ============================================================
 * Traits table registration
 * ============================================================ */

TEST(Mixed4B8, TraitsRegistered) {
    EXPECT_STREQ(TQ_TRAITS[TQ_TYPE_MIXED_4B8].name, "mixed_4b8");
    EXPECT_EQ(TQ_TRAITS[TQ_TYPE_MIXED_4B8].block_size, (size_t)TQ_BK);
    EXPECT_EQ(TQ_TRAITS[TQ_TYPE_MIXED_4B8].type_size, sizeof(block_tq_mixed_4b8));
    EXPECT_GT(TQ_TRAITS[TQ_TYPE_MIXED_4B8].bpe, 0.0f);
    EXPECT_NE(TQ_TRAITS[TQ_TYPE_MIXED_4B8].quantize, nullptr);
    EXPECT_NE(TQ_TRAITS[TQ_TYPE_MIXED_4B8].dequantize, nullptr);
    EXPECT_NE(TQ_TRAITS[TQ_TYPE_MIXED_4B8].attention, nullptr);
}

/* ============================================================
 * Type lookup by name
 * ============================================================ */

TEST(Mixed4B8, TypeFromName) {
    EXPECT_EQ(tq_type_from_name("mixed_4b8"), TQ_TYPE_MIXED_4B8);
}

/* ============================================================
 * Format spec
 * ============================================================ */

TEST(Mixed4B8, FormatSpec) {
    tq_format_spec_t spec = tq_get_format_spec(TQ_TYPE_MIXED_4B8);
    EXPECT_EQ(spec.algorithm, TQ_ALG_MIXED);
    EXPECT_EQ(spec.key_bits, 4);
    EXPECT_EQ(spec.outlier_count, TQ_MIXED_OUTLIERS);
    EXPECT_EQ(spec.block_size, TQ_BK);
}

/* ============================================================
 * Extreme outlier data: verify 10x+ MSE improvement
 * ============================================================ */

TEST(Mixed4B8, ExtremeOutliers) {
    std::vector<float> input(TQ_BK);
    /* Most values in [-1, 1], four extreme outliers */
    for (int i = 0; i < TQ_BK; i++) input[i] = sinf(i * 0.1f);
    input[0]  =  100.0f;
    input[64] = -100.0f;
    input[32] =  80.0f;
    input[96] = -80.0f;

    /* Mixed */
    block_tq_mixed_4b8 mixed_block;
    tq_mixed_4b8_quantize_ref(input.data(), &mixed_block, TQ_BK);
    std::vector<float> mixed_out(TQ_BK);
    tq_mixed_4b8_dequantize_ref(&mixed_block, mixed_out.data(), TQ_BK);

    double mixed_mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - mixed_out[i];
        mixed_mse += d * d;
    }
    mixed_mse /= TQ_BK;

    /* Uniform */
    block_tq_uniform_4b uni_block;
    tq_uniform_4b_quantize_ref(input.data(), &uni_block, TQ_BK);
    std::vector<float> uni_out(TQ_BK);
    tq_uniform_4b_dequantize_ref(&uni_block, uni_out.data(), TQ_BK);

    double uni_mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - uni_out[i];
        uni_mse += d * d;
    }
    uni_mse /= TQ_BK;

    /* With extreme outliers, mixed should be massively better */
    EXPECT_LT(mixed_mse, uni_mse * 0.1)
        << "mixed_mse=" << mixed_mse << " uni_mse=" << uni_mse
        << " (expected 10x+ improvement)";
}
