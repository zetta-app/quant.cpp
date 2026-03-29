/**
 * Tests for Random Hadamard Transform (RHT)
 *
 * Verifies:
 * 1. RHT roundtrip identity (transform + inverse = original)
 * 2. RHT improves quantization MSE vs raw quantization
 * 3. RHT improves attention cosine similarity
 * 4. Works with various power-of-2 dimensions
 * 5. Handles non-power-of-2 dimensions gracefully
 */

#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
}
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>

/* ---------- Helpers ---------- */

static uint32_t test_rng_state = 12345;
static float test_rand() {
    test_rng_state = test_rng_state * 1664525u + 1013904223u;
    return ((float)(test_rng_state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

static double compute_mse(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    return mse / n;
}

static double cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

/* ============================================================
 * Test: RHT roundtrip identity
 * ============================================================ */

TEST(RHT, RoundtripIdentity) {
    const int n = 128;
    const uint32_t seed = 42;

    std::vector<float> data(n);
    std::vector<float> original(n);

    /* Fill with known data */
    for (int i = 0; i < n; i++) {
        data[i] = sinf(i * 0.1f) + 0.5f * cosf(i * 0.3f);
        original[i] = data[i];
    }

    /* Forward transform */
    tq_rht_transform(data.data(), n, seed);

    /* The transformed data should be different from original */
    bool all_same = true;
    for (int i = 0; i < n; i++) {
        if (std::fabs(data[i] - original[i]) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "RHT should change the data";

    /* Inverse transform */
    tq_rht_inverse(data.data(), n, seed);

    /* Should recover original within floating point tolerance */
    double mse = compute_mse(original.data(), data.data(), n);
    EXPECT_LT(mse, 1e-10) << "RHT roundtrip MSE should be near-zero";

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(data[i], original[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

/* ============================================================
 * Test: RHT roundtrip with different seeds
 * ============================================================ */

TEST(RHT, DifferentSeedsProduceDifferentResults) {
    const int n = 128;
    std::vector<float> data1(n), data2(n);

    for (int i = 0; i < n; i++) {
        data1[i] = sinf(i * 0.1f);
        data2[i] = data1[i];
    }

    tq_rht_transform(data1.data(), n, 42);
    tq_rht_transform(data2.data(), n, 99);

    /* Different seeds should produce different transforms */
    bool all_same = true;
    for (int i = 0; i < n; i++) {
        if (std::fabs(data1[i] - data2[i]) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Different seeds should produce different results";
}

/* ============================================================
 * Test: RHT improves quantization MSE
 * ============================================================ */

TEST(RHT, ImprovesMSE) {
    const int head_dim = 128;
    const int n_vectors = 32;
    const uint32_t rht_seed = 42;

    tq_context_t* ctx = nullptr;
    ASSERT_EQ(TQ_OK, tq_init(&ctx, TQ_BACKEND_CPU));

    /* Generate realistic data with per-channel variance differences.
     * This simulates real LLM KV cache where some channels have much
     * larger magnitudes than others. */
    test_rng_state = 54321;
    std::vector<float> keys(n_vectors * head_dim);
    for (int i = 0; i < n_vectors; i++) {
        for (int d = 0; d < head_dim; d++) {
            /* Create per-channel variance: some channels 10x larger */
            float channel_scale = 1.0f + 9.0f * ((d % 8 == 0) ? 1.0f : 0.0f);
            keys[i * head_dim + d] = test_rand() * channel_scale;
        }
    }

    size_t qsize = tq_quantize_keys_size(n_vectors, head_dim, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(qsize, 0u);

    std::vector<uint8_t> qbuf_raw(qsize);
    std::vector<uint8_t> qbuf_rht(qsize);
    std::vector<float> recon_raw(n_vectors * head_dim);
    std::vector<float> recon_rht(n_vectors * head_dim);

    /* Raw quantization (no RHT) */
    ASSERT_EQ(TQ_OK, tq_quantize_keys(ctx, keys.data(), n_vectors, head_dim,
                                        TQ_TYPE_UNIFORM_4B, qbuf_raw.data(), qsize));
    ASSERT_EQ(TQ_OK, tq_dequantize_keys(ctx, qbuf_raw.data(), n_vectors, head_dim,
                                          TQ_TYPE_UNIFORM_4B, recon_raw.data()));

    /* RHT + quantization */
    ASSERT_EQ(TQ_OK, tq_quantize_keys_rht(ctx, keys.data(), n_vectors, head_dim,
                                            TQ_TYPE_UNIFORM_4B, rht_seed,
                                            qbuf_rht.data(), qsize));
    ASSERT_EQ(TQ_OK, tq_dequantize_keys_rht(ctx, qbuf_rht.data(), n_vectors, head_dim,
                                              TQ_TYPE_UNIFORM_4B, rht_seed,
                                              recon_rht.data()));

    /* Compute MSE for both */
    double mse_raw = compute_mse(keys.data(), recon_raw.data(), n_vectors * head_dim);
    double mse_rht = compute_mse(keys.data(), recon_rht.data(), n_vectors * head_dim);

    printf("  Raw uniform_4b MSE:       %.6f\n", mse_raw);
    printf("  RHT + uniform_4b MSE:     %.6f\n", mse_rht);
    printf("  Improvement ratio:        %.2fx\n", mse_raw / mse_rht);

    /* RHT should improve MSE when data has non-uniform channel variances */
    EXPECT_LT(mse_rht, mse_raw)
        << "RHT should reduce MSE for data with per-channel variance";

    tq_free(ctx);
}

/* ============================================================
 * Test: RHT improves attention cosine similarity
 * ============================================================ */

TEST(RHT, ImprovesAttention) {
    const int head_dim = 128;
    const int seq_len = 32;
    const uint32_t rht_seed = 42;

    tq_context_t* ctx = nullptr;
    ASSERT_EQ(TQ_OK, tq_init(&ctx, TQ_BACKEND_CPU));

    /* Generate data with per-channel variance differences */
    test_rng_state = 98765;
    std::vector<float> keys(seq_len * head_dim);
    std::vector<float> query(head_dim);

    for (int d = 0; d < head_dim; d++) {
        query[d] = test_rand();
    }

    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < head_dim; d++) {
            float channel_scale = 1.0f + 9.0f * ((d % 8 == 0) ? 1.0f : 0.0f);
            keys[s * head_dim + d] = test_rand() * channel_scale;
        }
    }

    size_t qsize = tq_quantize_keys_size(seq_len, head_dim, TQ_TYPE_UNIFORM_4B);
    std::vector<uint8_t> qbuf_raw(qsize);
    std::vector<uint8_t> qbuf_rht(qsize);
    std::vector<float> recon_raw(seq_len * head_dim);
    std::vector<float> recon_rht(seq_len * head_dim);

    /* Compute FP32 attention scores */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * keys[s * head_dim + d];
        }
        fp32_scores[s] = dot;
    }

    /* Raw quantization scores */
    ASSERT_EQ(TQ_OK, tq_quantize_keys(ctx, keys.data(), seq_len, head_dim,
                                        TQ_TYPE_UNIFORM_4B, qbuf_raw.data(), qsize));
    ASSERT_EQ(TQ_OK, tq_dequantize_keys(ctx, qbuf_raw.data(), seq_len, head_dim,
                                          TQ_TYPE_UNIFORM_4B, recon_raw.data()));

    std::vector<float> raw_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * recon_raw[s * head_dim + d];
        }
        raw_scores[s] = dot;
    }

    /* RHT quantization scores */
    ASSERT_EQ(TQ_OK, tq_quantize_keys_rht(ctx, keys.data(), seq_len, head_dim,
                                            TQ_TYPE_UNIFORM_4B, rht_seed,
                                            qbuf_rht.data(), qsize));
    ASSERT_EQ(TQ_OK, tq_dequantize_keys_rht(ctx, qbuf_rht.data(), seq_len, head_dim,
                                              TQ_TYPE_UNIFORM_4B, rht_seed,
                                              recon_rht.data()));

    std::vector<float> rht_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * recon_rht[s * head_dim + d];
        }
        rht_scores[s] = dot;
    }

    double cos_raw = cosine_similarity(fp32_scores.data(), raw_scores.data(), seq_len);
    double cos_rht = cosine_similarity(fp32_scores.data(), rht_scores.data(), seq_len);

    printf("  Raw uniform_4b attention cosine:  %.6f\n", cos_raw);
    printf("  RHT + uniform_4b attention cosine: %.6f\n", cos_rht);

    /* RHT should produce higher or comparable cosine similarity */
    EXPECT_GE(cos_rht, cos_raw - 0.01)
        << "RHT should not significantly degrade attention cosine";

    tq_free(ctx);
}

/* ============================================================
 * Test: Power-of-2 dimensions
 * ============================================================ */

TEST(RHT, PowerOfTwo) {
    const uint32_t seed = 42;
    int dims[] = {64, 128, 256, 512};

    for (int di = 0; di < 4; di++) {
        int n = dims[di];
        std::vector<float> data(n);
        std::vector<float> original(n);

        for (int i = 0; i < n; i++) {
            data[i] = sinf(i * 0.1f);
            original[i] = data[i];
        }

        tq_rht_transform(data.data(), n, seed);
        tq_rht_inverse(data.data(), n, seed);

        double mse = compute_mse(original.data(), data.data(), n);
        EXPECT_LT(mse, 1e-10)
            << "Roundtrip failed for dim=" << n << " (MSE=" << mse << ")";
    }
}

/* ============================================================
 * Test: Non-power-of-2 dimensions handled gracefully
 * ============================================================ */

TEST(RHT, NonPowerOfTwo) {
    const uint32_t seed = 42;
    /* 130 rounds down to 128 internally */
    const int n = 130;
    std::vector<float> data(n);
    std::vector<float> original(n);

    for (int i = 0; i < n; i++) {
        data[i] = sinf(i * 0.1f);
        original[i] = data[i];
    }

    tq_rht_transform(data.data(), n, seed);
    tq_rht_inverse(data.data(), n, seed);

    /* First 128 elements should roundtrip perfectly */
    double mse = compute_mse(original.data(), data.data(), 128);
    EXPECT_LT(mse, 1e-10) << "Power-of-2 portion should roundtrip";

    /* Elements beyond 128 should be unchanged */
    EXPECT_FLOAT_EQ(data[128], original[128]);
    EXPECT_FLOAT_EQ(data[129], original[129]);
}

/* ============================================================
 * Test: Edge cases
 * ============================================================ */

TEST(RHT, EdgeCases) {
    /* n=0 should not crash */
    tq_rht_transform(nullptr, 0, 42);
    tq_rht_inverse(nullptr, 0, 42);

    /* n=1 (power of 2 = 1): should just scale and sign-flip */
    float val = 3.0f;
    float orig = val;
    tq_rht_transform(&val, 1, 42);
    tq_rht_inverse(&val, 1, 42);
    EXPECT_NEAR(val, orig, 1e-5f);

    /* n=2: smallest non-trivial WHT */
    float data2[2] = {1.0f, 2.0f};
    float orig2[2] = {1.0f, 2.0f};
    tq_rht_transform(data2, 2, 42);
    tq_rht_inverse(data2, 2, 42);
    EXPECT_NEAR(data2[0], orig2[0], 1e-5f);
    EXPECT_NEAR(data2[1], orig2[1], 1e-5f);
}

/* ============================================================
 * Test: Full pipeline via high-level API
 * ============================================================ */

TEST(RHT, FullPipeline) {
    const int head_dim = 128;
    const int n = 8;
    const uint32_t rht_seed = 12345;

    tq_context_t* ctx = nullptr;
    ASSERT_EQ(TQ_OK, tq_init(&ctx, TQ_BACKEND_CPU));

    std::vector<float> keys(n * head_dim);
    for (int i = 0; i < n * head_dim; i++) {
        keys[i] = sinf(i * 0.01f);
    }

    size_t qsize = tq_quantize_keys_size(n, head_dim, TQ_TYPE_UNIFORM_4B);
    ASSERT_GT(qsize, 0u);

    std::vector<uint8_t> qbuf(qsize);
    std::vector<float> recon(n * head_dim);

    /* RHT quantize + dequantize */
    ASSERT_EQ(TQ_OK, tq_quantize_keys_rht(ctx, keys.data(), n, head_dim,
                                            TQ_TYPE_UNIFORM_4B, rht_seed,
                                            qbuf.data(), qsize));
    ASSERT_EQ(TQ_OK, tq_dequantize_keys_rht(ctx, qbuf.data(), n, head_dim,
                                              TQ_TYPE_UNIFORM_4B, rht_seed,
                                              recon.data()));

    /* Should be close to original (quantization loss only, no transform loss) */
    double mse = compute_mse(keys.data(), recon.data(), n * head_dim);
    printf("  Full pipeline MSE: %.6f\n", mse);
    EXPECT_LT(mse, 0.05) << "Full RHT pipeline MSE should be low";

    tq_free(ctx);
}

/* ============================================================
 * Test: WHT preserves L2 norm (Parseval's theorem)
 * ============================================================ */

TEST(RHT, PreservesNorm) {
    const int n = 128;
    const uint32_t seed = 42;

    std::vector<float> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = sinf(i * 0.1f);
    }

    /* Compute original L2 norm */
    double norm_before = 0.0;
    for (int i = 0; i < n; i++) {
        norm_before += (double)data[i] * (double)data[i];
    }
    norm_before = std::sqrt(norm_before);

    /* Transform */
    tq_rht_transform(data.data(), n, seed);

    /* Compute transformed L2 norm */
    double norm_after = 0.0;
    for (int i = 0; i < n; i++) {
        norm_after += (double)data[i] * (double)data[i];
    }
    norm_after = std::sqrt(norm_after);

    /* RHT (with normalization) should preserve L2 norm */
    EXPECT_NEAR(norm_after, norm_before, 1e-4)
        << "RHT should preserve L2 norm (Parseval's theorem)";
}
