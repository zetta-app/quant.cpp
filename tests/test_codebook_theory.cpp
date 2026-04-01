/**
 * test_codebook_theory.cpp -- Lloyd-Max codebook verification
 *
 * Verifies that the hardcoded codebook centroids match published
 * optimal Lloyd-Max values for N(0,1), checks symmetry, and
 * measures actual MSE against theoretical optimum.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>

extern "C" {
#include "turboquant/turboquant.h"

/* Codebook API */
const float* tq_codebook_centroids(int bits);
int tq_codebook_levels(int bits);
void tq_codebook_quantize(const float* src, uint8_t* dst_indices,
                           int n, int bits, float inv_std);
void tq_codebook_dequantize(const uint8_t* indices, float* dst,
                             int n, int bits, float inv_std);
}

/* ============================================================
 * Test 1: 2-bit centroids match N(0,1) Lloyd-Max optimal
 * Literature values: [-1.5104, -0.4528, 0.4528, 1.5104]
 * ============================================================ */

TEST(CodebookTheory, TwoBitCentroidsMatchLiterature) {
    const float expected[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
    const float* actual = tq_codebook_centroids(2);
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(tq_codebook_levels(2), 4);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(actual[i], expected[i], 0.001f)
            << "2-bit centroid[" << i << "] mismatch";
    }
}

/* ============================================================
 * Test 2: 3-bit centroids match N(0,1) Lloyd-Max optimal
 * ============================================================ */

TEST(CodebookTheory, ThreeBitCentroidsMatchLiterature) {
    const float expected[8] = {
        -2.1520f, -1.3440f, -0.7560f, -0.2451f,
         0.2451f,  0.7560f,  1.3440f,  2.1520f
    };
    const float* actual = tq_codebook_centroids(3);
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(tq_codebook_levels(3), 8);

    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(actual[i], expected[i], 0.001f)
            << "3-bit centroid[" << i << "] mismatch";
    }
}

/* ============================================================
 * Test 3: Codebook symmetry: centroid[i] = -centroid[n-1-i]
 * ============================================================ */

TEST(CodebookTheory, SymmetryProperty) {
    for (int bits = 1; bits <= 4; bits++) {
        const float* c = tq_codebook_centroids(bits);
        int n = tq_codebook_levels(bits);
        ASSERT_NE(c, nullptr);
        ASSERT_GT(n, 0);

        for (int i = 0; i < n / 2; i++) {
            EXPECT_NEAR(c[i], -c[n - 1 - i], 1e-5f)
                << "Symmetry violated for " << bits << "-bit codebook at index " << i;
        }
    }
}

/* ============================================================
 * Test 4: Actual MSE of 2-bit codebook on N(0,1) samples
 * Theoretical optimal MSE for 2-bit on N(0,1): ~0.1175
 * Verify actual MSE is within 1.20x of theory.
 * ============================================================ */

TEST(CodebookTheory, TwoBitMSEWithinTheoretical) {
    const int N = 100000;
    const float theoretical_mse = 0.1175f;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> samples(N);
    for (int i = 0; i < N; i++) {
        samples[i] = dist(rng);
    }

    /* Quantize with inv_std = 1.0 (already N(0,1)) */
    std::vector<uint8_t> indices(N);
    tq_codebook_quantize(samples.data(), indices.data(), N, 2, 1.0f);

    /* Dequantize */
    std::vector<float> reconstructed(N);
    tq_codebook_dequantize(indices.data(), reconstructed.data(), N, 2, 1.0f);

    /* Compute MSE */
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double d = (double)samples[i] - (double)reconstructed[i];
        mse += d * d;
    }
    mse /= N;

    printf("  2-bit codebook MSE: %.6f (theoretical: %.4f, ratio: %.3fx)\n",
           mse, theoretical_mse, mse / theoretical_mse);

    /* MSE should be within 1.20x of theoretical optimal */
    EXPECT_LT(mse, theoretical_mse * 1.20)
        << "2-bit MSE " << mse << " exceeds 1.20x theoretical " << theoretical_mse;

    /* MSE should not be lower than theoretical (sanity check) */
    EXPECT_GT(mse, theoretical_mse * 0.90)
        << "2-bit MSE " << mse << " is suspiciously lower than theoretical";
}

/* ============================================================
 * Test 5: 3-bit MSE verification
 * Theoretical optimal MSE for 3-bit on N(0,1): ~0.0344
 * ============================================================ */

TEST(CodebookTheory, ThreeBitMSEWithinTheoretical) {
    const int N = 100000;
    const float theoretical_mse = 0.0344f;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> samples(N);
    for (int i = 0; i < N; i++) {
        samples[i] = dist(rng);
    }

    std::vector<uint8_t> indices(N);
    tq_codebook_quantize(samples.data(), indices.data(), N, 3, 1.0f);

    std::vector<float> reconstructed(N);
    tq_codebook_dequantize(indices.data(), reconstructed.data(), N, 3, 1.0f);

    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double d = (double)samples[i] - (double)reconstructed[i];
        mse += d * d;
    }
    mse /= N;

    printf("  3-bit codebook MSE: %.6f (theoretical: %.4f, ratio: %.3fx)\n",
           mse, theoretical_mse, mse / theoretical_mse);

    EXPECT_LT(mse, theoretical_mse * 1.20)
        << "3-bit MSE " << mse << " exceeds 1.20x theoretical " << theoretical_mse;
}
