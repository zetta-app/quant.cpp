/**
 * Edge case tests for TurboQuant — BUG-4, BUG-7 verification
 */
#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
}
#include <climits>
#include <vector>
#include <cmath>

class EdgeCaseFixture : public ::testing::Test {
protected:
    tq_context_t* ctx = nullptr;

    void SetUp() override {
        ASSERT_EQ(TQ_OK, tq_init(&ctx, TQ_BACKEND_CPU));
    }

    void TearDown() override {
        tq_free(ctx);
    }
};

/* ---- BUG-7: seq_len == 0 is a no-op ---- */

TEST_F(EdgeCaseFixture, SeqLenZero) {
    float query[128] = {};
    float scores[1]  = {-999.0f};
    block_tq_uniform_4b block = {};

    tq_status st = tq_attention(ctx, query, &block, /*seq_len=*/0, 128,
                                TQ_TYPE_UNIFORM_4B, scores);
    EXPECT_EQ(TQ_OK, st);
    /* scores must be untouched */
    EXPECT_FLOAT_EQ(-999.0f, scores[0]);
}

/* ---- BUG-7: NULL input pointers ---- */

TEST_F(EdgeCaseFixture, NullInput) {
    block_tq_uniform_4b block = {};
    tq_status st = tq_quantize_keys(ctx, /*keys=*/NULL, 1, 128,
                                    TQ_TYPE_UNIFORM_4B, &block, sizeof(block));
    EXPECT_EQ(TQ_ERR_NULL_PTR, st);
}

TEST_F(EdgeCaseFixture, NullOutput) {
    float keys[128] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 128,
                                    TQ_TYPE_UNIFORM_4B, /*out=*/NULL, 1024);
    EXPECT_EQ(TQ_ERR_NULL_PTR, st);
}

/* ---- BUG-4: buffer too small ---- */

TEST_F(EdgeCaseFixture, BufferTooSmall) {
    float keys[128] = {};
    for (int i = 0; i < 128; i++) keys[i] = sinf((float)i * 0.1f);

    block_tq_uniform_4b block = {};
    /* Provide a buffer smaller than required */
    tq_status st = tq_quantize_keys(ctx, keys, 1, 128,
                                    TQ_TYPE_UNIFORM_4B, &block, /*out_size=*/1);
    EXPECT_EQ(TQ_ERR_BUFFER_TOO_SMALL, st);
}

/* ---- BUG-4: overflow size ---- */

TEST(EdgeCase, OverflowSize) {
    size_t sz = tq_quantize_keys_size(INT_MAX, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz);
}

TEST(EdgeCase, NegativeN) {
    size_t sz = tq_quantize_keys_size(-1, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz);
}

TEST(EdgeCase, ZeroHeadDim) {
    size_t sz = tq_quantize_keys_size(1, 0, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz);
}

/* ---- BUG-7: odd head_dim for PolarQuant ---- */

TEST_F(EdgeCaseFixture, OddHeadDimPolar) {
    float keys[65] = {};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 65,
                                    TQ_TYPE_POLAR_3B, buf, sizeof(buf));
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

TEST_F(EdgeCaseFixture, OddHeadDimTurbo) {
    float keys[65] = {};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 65,
                                    TQ_TYPE_TURBO_3B, buf, sizeof(buf));
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- BUG-7: head_dim < 2 ---- */

TEST_F(EdgeCaseFixture, HeadDimOne) {
    float keys[1] = {1.0f};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 1, 1,
                                    TQ_TYPE_UNIFORM_4B, buf, sizeof(buf));
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- BUG-7: head_dim == 2 works correctly ---- */

TEST_F(EdgeCaseFixture, HeadDimTwo) {
    float keys[2] = {0.5f, -0.5f};
    /* head_dim=2 with block_size=128: 1 block, need type_size bytes */
    size_t needed = tq_quantize_keys_size(1, 2, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(needed, 0u);

    std::vector<uint8_t> buf(needed, 0);
    tq_status st = tq_quantize_keys(ctx, keys, 1, 2,
                                    TQ_TYPE_UNIFORM_4B, buf.data(), buf.size());
    EXPECT_EQ(TQ_OK, st);
}

/* ---- BUG-7: n == 0 is a no-op for quantize ---- */

TEST_F(EdgeCaseFixture, QuantizeZeroKeys) {
    float keys[128] = {};
    uint8_t buf[4096] = {};
    tq_status st = tq_quantize_keys(ctx, keys, 0, 128,
                                    TQ_TYPE_UNIFORM_4B, buf, sizeof(buf));
    EXPECT_EQ(TQ_OK, st);
}

/* ---- BUG-7: attention with head_dim < 2 ---- */

TEST_F(EdgeCaseFixture, AttentionHeadDimOne) {
    float query[1] = {1.0f};
    float scores[1] = {};
    uint8_t kv[256] = {};
    tq_status st = tq_attention(ctx, query, kv, 1, 1,
                                TQ_TYPE_UNIFORM_4B, scores);
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- BUG-7: attention odd head_dim for PolarQuant ---- */

TEST_F(EdgeCaseFixture, AttentionOddHeadDimPolar) {
    float query[65] = {};
    float scores[1] = {};
    uint8_t kv[4096] = {};
    tq_status st = tq_attention(ctx, query, kv, 1, 65,
                                TQ_TYPE_POLAR_3B, scores);
    EXPECT_EQ(TQ_ERR_INVALID_DIM, st);
}

/* ---- New error code string ---- */

TEST(EdgeCase, BufferTooSmallString) {
    const char* s = tq_status_string(TQ_ERR_BUFFER_TOO_SMALL);
    EXPECT_STREQ("buffer too small", s);
}

/* ---- MaxSeqLen boundary ---- */

TEST(EdgeCase, MaxSeqLenBoundary) {
    /* Exactly at TQ_MAX_SEQ_LEN should work (non-zero return) */
    size_t sz = tq_quantize_keys_size(TQ_MAX_SEQ_LEN, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_GT(sz, 0u);

    /* One over should fail */
    size_t sz2 = tq_quantize_keys_size(TQ_MAX_SEQ_LEN + 1, 128, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(0u, sz2);
}
