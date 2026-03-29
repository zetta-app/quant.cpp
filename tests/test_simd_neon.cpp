/**
 * NEON vs reference consistency tests
 *
 * Verifies that NEON-optimized quantize/dequantize produce the same
 * output as the reference scalar implementations.
 * Only compiled and run on ARM platforms with NEON support.
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
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
void tq_qjl_quantize_ref(const float* src, void* dst, int n);

#ifdef __ARM_NEON
/* NEON implementations */
void tq_uniform_4b_quantize_neon(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_neon(const void* src, float* dst, int n);
void tq_polar_quantize_neon(const float* src, void* dst, int n);
void tq_polar_dequantize_neon(const void* src, float* dst, int n);
void tq_qjl_quantize_neon(const float* src, void* dst, int n);
#endif
}

#ifdef __ARM_NEON

class NeonConsistencyTest : public ::testing::Test {
protected:
    std::vector<float> make_input(int n, float scale = 1.0f) {
        std::vector<float> v(n);
        for (int i = 0; i < n; i++) {
            v[i] = sinf(i * 0.1f) * scale;
        }
        return v;
    }
};

/* ----------------------------------------------------------------
 * Uniform 4-bit: NEON quantize should produce identical packed bytes
 * ---------------------------------------------------------------- */

TEST_F(NeonConsistencyTest, Uniform4B_QuantizeBitExact) {
    auto input = make_input(TQ_BK, 2.0f);

    block_tq_uniform_4b block_ref;
    block_tq_uniform_4b block_neon;
    memset(&block_ref, 0, sizeof(block_ref));
    memset(&block_neon, 0, sizeof(block_neon));

    tq_uniform_4b_quantize_ref(input.data(), &block_ref, TQ_BK);
    tq_uniform_4b_quantize_neon(input.data(), &block_neon, TQ_BK);

    /* Scale and zero_point should match */
    EXPECT_EQ(block_ref.scale, block_neon.scale)
        << "Scale mismatch: ref=" << block_ref.scale
        << " neon=" << block_neon.scale;
    EXPECT_EQ(block_ref.zero_point, block_neon.zero_point)
        << "Zero point mismatch";

    /* Packed quantized values should match */
    for (int i = 0; i < TQ_BK / 2; i++) {
        EXPECT_EQ(block_ref.qs[i], block_neon.qs[i])
            << "Mismatch at byte " << i;
    }
}

TEST_F(NeonConsistencyTest, Uniform4B_DequantizeConsistent) {
    auto input = make_input(TQ_BK, 5.0f);

    /* Quantize with reference */
    block_tq_uniform_4b block;
    memset(&block, 0, sizeof(block));
    tq_uniform_4b_quantize_ref(input.data(), &block, TQ_BK);

    /* Dequantize with both */
    std::vector<float> out_ref(TQ_BK);
    std::vector<float> out_neon(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block, out_ref.data(), TQ_BK);
    tq_uniform_4b_dequantize_neon(&block, out_neon.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(out_ref[i], out_neon[i], 1e-5f)
            << "Dequantize mismatch at element " << i;
    }
}

TEST_F(NeonConsistencyTest, Uniform4B_RoundtripNeon) {
    auto input = make_input(TQ_BK, 1.0f);

    block_tq_uniform_4b block;
    memset(&block, 0, sizeof(block));
    tq_uniform_4b_quantize_neon(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_4b_dequantize_neon(&block, output.data(), TQ_BK);

    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    EXPECT_LT(mse, 0.01) << "NEON roundtrip MSE too high: " << mse;
}

/* ----------------------------------------------------------------
 * Polar: NEON quantize should produce similar results (atan2 approx)
 * Note: NEON uses polynomial atan2 approximation, so results may
 * differ slightly from the reference atan2f. We allow small tolerance.
 * ---------------------------------------------------------------- */

TEST_F(NeonConsistencyTest, Polar_QuantizeConsistent) {
    auto input = make_input(TQ_BK, 3.0f);

    block_tq_polar block_ref;
    block_tq_polar block_neon;
    memset(&block_ref, 0, sizeof(block_ref));
    memset(&block_neon, 0, sizeof(block_neon));

    tq_polar_quantize_ref(input.data(), &block_ref, TQ_BK);
    tq_polar_quantize_neon(input.data(), &block_neon, TQ_BK);

    /* Dequantize both and compare results */
    std::vector<float> out_ref(TQ_BK);
    std::vector<float> out_neon(TQ_BK);
    tq_polar_dequantize_ref(&block_ref, out_ref.data(), TQ_BK);
    tq_polar_dequantize_ref(&block_neon, out_neon.data(), TQ_BK);

    /* Compute MSE between the two reconstructions */
    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = out_ref[i] - out_neon[i];
        mse += d * d;
    }
    mse /= TQ_BK;

    /* Allow slightly higher tolerance due to atan2 approximation */
    EXPECT_LT(mse, 0.5)
        << "Polar NEON vs ref MSE too high: " << mse;
}

TEST_F(NeonConsistencyTest, Polar_RoundtripNeon) {
    auto input = make_input(TQ_BK, 2.0f);

    block_tq_polar block;
    memset(&block, 0, sizeof(block));
    tq_polar_quantize_neon(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_polar_dequantize_neon(&block, output.data(), TQ_BK);

    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    /* Polar 4-bit with atan2 approx: higher MSE is expected */
    EXPECT_LT(mse, 2.0) << "Polar NEON roundtrip MSE too high: " << mse;
}

/* ----------------------------------------------------------------
 * QJL: NEON quantize should produce bit-exact hash output
 * (since it uses same random entry function and same logic)
 * ---------------------------------------------------------------- */

TEST_F(NeonConsistencyTest, QJL_QuantizeBitExact) {
    auto input = make_input(TQ_BK_QJL, 1.0f);

    block_tq_qjl block_ref;
    block_tq_qjl block_neon;
    memset(&block_ref, 0, sizeof(block_ref));
    memset(&block_neon, 0, sizeof(block_neon));

    tq_qjl_quantize_ref(input.data(), &block_ref, TQ_BK_QJL);
    tq_qjl_quantize_neon(input.data(), &block_neon, TQ_BK_QJL);

    /* Norms should be very close (FP16 rounding may differ slightly) */
    EXPECT_EQ(block_ref.norm, block_neon.norm) << "QJL norm mismatch";
    EXPECT_EQ(block_ref.outlier_norm, block_neon.outlier_norm)
        << "QJL outlier norm mismatch";

    /* Outlier indices should match */
    for (int i = 0; i < TQ_OUTLIERS; i++) {
        EXPECT_EQ(block_ref.outlier_idx[i], block_neon.outlier_idx[i])
            << "QJL outlier index mismatch at " << i;
    }

    /* Hash bits should match exactly
     * (Same random entries, same accumulation, same sign extraction) */
    for (int i = 0; i < TQ_SKETCH_DIM / 8; i++) {
        EXPECT_EQ(block_ref.hash[i], block_neon.hash[i])
            << "QJL hash mismatch at byte " << i;
    }
}

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST_F(NeonConsistencyTest, Uniform4B_ZeroInput) {
    std::vector<float> input(TQ_BK, 0.0f);

    block_tq_uniform_4b block_ref, block_neon;
    memset(&block_ref, 0, sizeof(block_ref));
    memset(&block_neon, 0, sizeof(block_neon));

    tq_uniform_4b_quantize_ref(input.data(), &block_ref, TQ_BK);
    tq_uniform_4b_quantize_neon(input.data(), &block_neon, TQ_BK);

    /* Both should handle zero input gracefully */
    std::vector<float> out_ref(TQ_BK), out_neon(TQ_BK);
    tq_uniform_4b_dequantize_ref(&block_ref, out_ref.data(), TQ_BK);
    tq_uniform_4b_dequantize_neon(&block_neon, out_neon.data(), TQ_BK);

    for (int i = 0; i < TQ_BK; i++) {
        EXPECT_NEAR(out_ref[i], out_neon[i], 1e-5f);
    }
}

TEST_F(NeonConsistencyTest, Uniform4B_LargeValues) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++) {
        input[i] = (float)i / TQ_BK * 1000.0f - 500.0f;
    }

    block_tq_uniform_4b block_ref, block_neon;
    memset(&block_ref, 0, sizeof(block_ref));
    memset(&block_neon, 0, sizeof(block_neon));

    tq_uniform_4b_quantize_ref(input.data(), &block_ref, TQ_BK);
    tq_uniform_4b_quantize_neon(input.data(), &block_neon, TQ_BK);

    EXPECT_EQ(block_ref.scale, block_neon.scale);
    EXPECT_EQ(block_ref.zero_point, block_neon.zero_point);

    for (int i = 0; i < TQ_BK / 2; i++) {
        EXPECT_EQ(block_ref.qs[i], block_neon.qs[i]);
    }
}

#else /* !__ARM_NEON */

TEST(NeonSkipped, NotAvailable) {
    GTEST_SKIP() << "ARM NEON not available on this platform";
}

#endif /* __ARM_NEON */
