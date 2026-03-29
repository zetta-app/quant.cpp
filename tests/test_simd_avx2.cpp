/**
 * AVX2 vs reference consistency tests
 *
 * Verifies that AVX2-optimized quantize/dequantize produce the same
 * output as the reference scalar implementations.
 * Only compiled and run on x86 platforms with AVX2 support.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstring>

extern "C" {
#include "turboquant/turboquant.h"

/* Reference implementations */
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);

#ifdef __AVX2__
/* AVX2 implementations */
void tq_uniform_4b_quantize_avx2(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_avx2(const void* src, float* dst, int n);
#endif
}

#ifdef __AVX2__

class AVX2ConsistencyTest : public ::testing::Test {
protected:
    std::vector<float> make_input(int n, float scale = 1.0f) {
        std::vector<float> v(n);
        for (int i = 0; i < n; i++) {
            v[i] = sinf(i * 0.1f) * scale;
        }
        return v;
    }
};

TEST_F(AVX2ConsistencyTest, Uniform4B_QuantizeBitExact) {
    auto input = make_input(TQ_BK, 2.0f);

    block_tq_uniform_4b block_ref;
    block_tq_uniform_4b block_avx;
    memset(&block_ref, 0, sizeof(block_ref));
    memset(&block_avx, 0, sizeof(block_avx));

    tq_uniform_4b_quantize_ref(input.data(), &block_ref, TQ_BK);
    tq_uniform_4b_quantize_avx2(input.data(), &block_avx, TQ_BK);

    EXPECT_EQ(block_ref.scale, block_avx.scale) << "Scale mismatch";
    EXPECT_EQ(block_ref.zero_point, block_avx.zero_point) << "Zero point mismatch";

    for (int i = 0; i < TQ_BK / 2; i++) {
        EXPECT_EQ(block_ref.qs[i], block_avx.qs[i])
            << "Mismatch at byte " << i;
    }
}

TEST_F(AVX2ConsistencyTest, Uniform4B_DequantizeConsistent) {
    auto input = make_input(TQ_BK, 5.0f);

    block_tq_uniform_4b block;
    memset(&block, 0, sizeof(block));
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> out_ref(TQ_BK);
    std::vector<float> out_avx(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, out_ref.data(), TQ_BK);
    tq_uniform_4b_dequantize_avx2(&block, out_avx.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(out_ref[i], out_avx[i], 1e-5f)
            << "Dequantize mismatch at element " << i;
    }
}

TEST_F(AVX2ConsistencyTest, Uniform4B_RoundtripAVX2) {
    auto input = make_input(TQ_BK, 1.0f);

    block_tq_uniform_4b block;
    memset(&block, 0, sizeof(block));
    tq_uniform_4b_quantize_avx2(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_avx2(&block, output.data(), TQ_BK);

    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.01) << "AVX2 roundtrip MSE too high: " << mse;
}

TEST_F(AVX2ConsistencyTest, Uniform4B_ZeroInput) {
    std::vector<float> input(TQ_BK, 0.0f);

    block_tq_uniform_4b block_ref, block_avx;
    memset(&block_ref, 0, sizeof(block_ref));
    memset(&block_avx, 0, sizeof(block_avx));

    tq_uniform_4b_quantize_ref(input.data(), &block_ref, TQ_BK);
    tq_uniform_4b_quantize_avx2(input.data(), &block_avx, TQ_BK);

    std::vector<float> out_ref(TQ_BK), out_avx(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block_ref, out_ref.data(), TQ_BK);
    tq_uniform_4b_dequantize_avx2(&block_avx, out_avx.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(out_ref[i], out_avx[i], 1e-5f);
    }
}

#else /* !__AVX2__ */

TEST(AVX2Skipped, NotAvailable) {
    GTEST_SKIP() << "AVX2 not available on this platform";
}

#endif /* __AVX2__ */
