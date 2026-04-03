/**
 * test_head_sensitivity.cpp -- Unit tests for head-level mixed precision
 *
 * Tests the entropy-based head sensitivity profiling and mixed-precision
 * quantization concept without requiring a real model. Uses synthetic
 * attention distributions and key vectors.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>

extern "C" {
#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
}

/* ============================================================
 * Utility: cosine similarity
 * ============================================================ */
static double cosine_sim(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    na = sqrt(na); nb = sqrt(nb);
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / (na * nb);
}

/* ============================================================
 * Test 1: tq_attention_entropy computes correct values
 * ============================================================ */
TEST(HeadSensitivity, EntropyUniform) {
    /* Uniform distribution over N items: entropy = log2(N) */
    const int N = 64;
    std::vector<float> probs(N, 1.0f / N);
    float ent = tq_attention_entropy(probs.data(), N);
    float expected = log2f((float)N);
    EXPECT_NEAR(ent, expected, 0.01f);
}

TEST(HeadSensitivity, EntropyDirac) {
    /* All attention on one token: entropy = 0 */
    const int N = 100;
    std::vector<float> probs(N, 0.0f);
    probs[42] = 1.0f;
    float ent = tq_attention_entropy(probs.data(), N);
    EXPECT_NEAR(ent, 0.0f, 0.001f);
}

TEST(HeadSensitivity, EntropySharp) {
    /* Sharp distribution: most weight on 2 tokens */
    const int N = 100;
    std::vector<float> probs(N, 0.001f / (N - 2));
    probs[10] = 0.6f;
    probs[20] = 0.399f;
    /* Normalize */
    float sum = 0.0f;
    for (auto p : probs) sum += p;
    for (auto& p : probs) p /= sum;

    float ent = tq_attention_entropy(probs.data(), N);
    /* Sharp distribution should have low entropy (< 2 bits) */
    EXPECT_LT(ent, 2.0f);
    EXPECT_GT(ent, 0.0f);
}

TEST(HeadSensitivity, EntropyDiffuse) {
    /* Diffuse: approximately uniform */
    const int N = 100;
    std::vector<float> probs(N);
    srand(42);
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        probs[i] = 0.5f + 0.5f * (float)rand() / RAND_MAX;
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    float ent = tq_attention_entropy(probs.data(), N);
    float max_ent = log2f((float)N);
    /* Near-uniform should have high entropy (close to log2(N)) */
    EXPECT_GT(ent, max_ent * 0.9f);
}

/* ============================================================
 * Test 2: Mixed precision gives better cosine than uniform 2-bit
 *         for low-entropy (sensitive) heads
 * ============================================================ */
TEST(HeadSensitivity, MixedPrecisionSensitiveHeads) {
    /* Generate a key vector with large dynamic range (typical of sensitive heads) */
    const int dim = 128;
    std::vector<float> key(dim);
    srand(123);
    for (int i = 0; i < dim; i++) {
        /* Large outliers make quantization harder */
        key[i] = (float)(rand() % 1000 - 500) / 100.0f;
    }
    /* Add some large spikes */
    key[0] = 50.0f;
    key[1] = -40.0f;

    /* Quantize at 4-bit */
    std::vector<uint8_t> qbuf_4b(4096);
    std::vector<float> deq_4b(dim);
    tq_uniform_4b_quantize_ref(key.data(), qbuf_4b.data(), dim);
    tq_uniform_4b_dequantize_ref(qbuf_4b.data(), deq_4b.data(), dim);
    double cos_4b = cosine_sim(key.data(), deq_4b.data(), dim);

    /* Quantize at 2-bit */
    std::vector<uint8_t> qbuf_2b(4096);
    std::vector<float> deq_2b(dim);
    tq_uniform_2b_quantize_ref(key.data(), qbuf_2b.data(), dim);
    tq_uniform_2b_dequantize_ref(qbuf_2b.data(), deq_2b.data(), dim);
    double cos_2b = cosine_sim(key.data(), deq_2b.data(), dim);

    /* 4-bit should be significantly better than 2-bit for spiky data */
    EXPECT_GT(cos_4b, cos_2b);
    EXPECT_GT(cos_4b, 0.95);
}

/* ============================================================
 * Test 3: Insensitive heads (smooth data) tolerate 2-bit well
 * ============================================================ */
TEST(HeadSensitivity, InsensitiveHeadsTolerate2Bit) {
    /* Generate a smooth, low-dynamic-range key vector */
    const int dim = 128;
    std::vector<float> key(dim);
    for (int i = 0; i < dim; i++) {
        /* Smooth sinusoidal — low dynamic range, no outliers */
        key[i] = sinf((float)i * 0.1f) * 0.5f + 0.1f * cosf((float)i * 0.3f);
    }

    /* 2-bit should still give decent cosine for smooth data */
    std::vector<uint8_t> qbuf(4096);
    std::vector<float> deq(dim);
    tq_uniform_2b_quantize_ref(key.data(), qbuf.data(), dim);
    tq_uniform_2b_dequantize_ref(qbuf.data(), deq.data(), dim);
    double cos_2b = cosine_sim(key.data(), deq.data(), dim);

    /* For smooth data, 2-bit should preserve direction reasonably */
    EXPECT_GT(cos_2b, 0.80);
}

/* ============================================================
 * Test 4: Entropy classification consistency
 * ============================================================ */
TEST(HeadSensitivity, EntropyMonotonic) {
    /* Progressively sharper distributions should have decreasing entropy */
    const int N = 64;
    float prev_ent = 1e30f;

    for (float sharpness = 0.1f; sharpness <= 10.0f; sharpness += 0.5f) {
        std::vector<float> probs(N);
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            probs[i] = expf(-sharpness * fabsf((float)i - N / 2.0f));
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;

        float ent = tq_attention_entropy(probs.data(), N);
        /* Sharper distributions should have lower entropy */
        EXPECT_LE(ent, prev_ent + 0.01f);  /* small tolerance for float */
        prev_ent = ent;
    }
}

/* ============================================================
 * Test 5: Attention score preservation under mixed quantization
 * ============================================================ */
TEST(HeadSensitivity, AttentionScorePreservation) {
    const int dim = 128;
    const int seq_len = 32;

    /* Generate query and keys */
    std::vector<float> query(dim);
    std::vector<float> keys(seq_len * dim);
    srand(777);
    for (int d = 0; d < dim; d++) {
        query[d] = (float)(rand() % 200 - 100) / 100.0f;
    }
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < dim; d++) {
            keys[t * dim + d] = (float)(rand() % 200 - 100) / 100.0f;
        }
    }

    /* Compute FP32 reference scores */
    std::vector<float> scores_ref(seq_len);
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            dot += query[d] * keys[t * dim + d];
        }
        scores_ref[t] = dot;
    }

    /* Compute 4-bit quantized scores */
    std::vector<float> scores_4b(seq_len);
    std::vector<uint8_t> qbuf(4096);
    std::vector<float> deq(dim);
    for (int t = 0; t < seq_len; t++) {
        tq_uniform_4b_quantize_ref(&keys[t * dim], qbuf.data(), dim);
        tq_uniform_4b_dequantize_ref(qbuf.data(), deq.data(), dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) dot += query[d] * deq[d];
        scores_4b[t] = dot;
    }

    /* Compute 2-bit quantized scores */
    std::vector<float> scores_2b(seq_len);
    for (int t = 0; t < seq_len; t++) {
        tq_uniform_2b_quantize_ref(&keys[t * dim], qbuf.data(), dim);
        tq_uniform_2b_dequantize_ref(qbuf.data(), deq.data(), dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) dot += query[d] * deq[d];
        scores_2b[t] = dot;
    }

    /* Attention score correlation should be high for both */
    double corr_4b = cosine_sim(scores_ref.data(), scores_4b.data(), seq_len);
    double corr_2b = cosine_sim(scores_ref.data(), scores_2b.data(), seq_len);

    EXPECT_GT(corr_4b, 0.99);  /* 4-bit preserves attention well */
    EXPECT_GT(corr_2b, 0.90);  /* 2-bit preserves direction */
    EXPECT_GT(corr_4b, corr_2b);  /* 4-bit should be better */
}

/* ============================================================
 * Test 6: Entropy edge cases
 * ============================================================ */
TEST(HeadSensitivity, EntropyZeroLength) {
    float ent = tq_attention_entropy(nullptr, 0);
    EXPECT_NEAR(ent, 0.0f, 0.001f);
}

TEST(HeadSensitivity, EntropySingleElement) {
    float p = 1.0f;
    float ent = tq_attention_entropy(&p, 1);
    EXPECT_NEAR(ent, 0.0f, 0.001f);
}
