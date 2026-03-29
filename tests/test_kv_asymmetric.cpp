#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
}
#include <cmath>
#include <cstdio>
#include <vector>

/* Helper: compute cosine similarity between two float arrays */
static double cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

class KVAsymmetricTest : public ::testing::Test {
protected:
    tq_context_t* ctx = nullptr;

    void SetUp() override {
        ASSERT_EQ(tq_init(&ctx, TQ_BACKEND_CPU), TQ_OK);
    }

    void TearDown() override {
        tq_free(ctx);
    }
};

TEST_F(KVAsymmetricTest, Key4Value2) {
    const int n = 4;
    const int head_dim = TQ_BK; /* 128 */
    const tq_type key_type = TQ_TYPE_UNIFORM_4B;
    const tq_type val_type = TQ_TYPE_UNIFORM_2B;

    /* Generate test data */
    std::vector<float> keys(n * head_dim);
    std::vector<float> values(n * head_dim);
    for (int i = 0; i < n * head_dim; i++) {
        keys[i]   = sinf(i * 0.07f);
        values[i] = cosf(i * 0.05f);
    }

    /* Compute buffer sizes */
    size_t key_size = tq_quantize_kv_key_size(n, head_dim, key_type);
    size_t val_size = tq_quantize_kv_value_size(n, head_dim, val_type);
    ASSERT_GT(key_size, 0u);
    ASSERT_GT(val_size, 0u);

    std::vector<uint8_t> key_buf(key_size);
    std::vector<uint8_t> val_buf(val_size);

    /* Quantize */
    tq_status status = tq_quantize_kv(ctx,
        keys.data(), values.data(), n, head_dim,
        key_type, val_type,
        key_buf.data(), key_size,
        val_buf.data(), val_size);
    ASSERT_EQ(status, TQ_OK);

    /* Dequantize and check key quality */
    std::vector<float> key_deq(n * head_dim);
    ASSERT_EQ(tq_dequantize_keys(ctx, key_buf.data(), n, head_dim, key_type, key_deq.data()), TQ_OK);

    double key_cos_sum = 0;
    for (int i = 0; i < n; i++) {
        double cs = cosine_similarity(keys.data() + i * head_dim,
                                       key_deq.data() + i * head_dim, head_dim);
        key_cos_sum += cs;
    }
    double key_cos_avg = key_cos_sum / n;
    EXPECT_GT(key_cos_avg, 0.99) << "Key cosine similarity too low";

    /* Dequantize and check value quality */
    std::vector<float> val_deq(n * head_dim);
    ASSERT_EQ(tq_dequantize_keys(ctx, val_buf.data(), n, head_dim, val_type, val_deq.data()), TQ_OK);

    double val_cos_sum = 0;
    for (int i = 0; i < n; i++) {
        double cs = cosine_similarity(values.data() + i * head_dim,
                                       val_deq.data() + i * head_dim, head_dim);
        val_cos_sum += cs;
    }
    double val_cos_avg = val_cos_sum / n;
    EXPECT_GT(val_cos_avg, 0.85) << "Value cosine similarity too low";

    /* Print average bits: key_bpe and value_bpe */
    float key_bpe = tq_type_bpe(key_type);
    float val_bpe = tq_type_bpe(val_type);
    printf("  K/V Asymmetric K4V2: key_bpe=%.2f, val_bpe=%.2f, avg=%.2f bit\n",
           key_bpe, val_bpe, (key_bpe + val_bpe) / 2.0f);
    printf("  Key cosine=%.6f, Value cosine=%.6f\n", key_cos_avg, val_cos_avg);
}

TEST_F(KVAsymmetricTest, Key4Value4) {
    const int n = 4;
    const int head_dim = TQ_BK;
    const tq_type key_type = TQ_TYPE_UNIFORM_4B;
    const tq_type val_type = TQ_TYPE_UNIFORM_4B;

    std::vector<float> keys(n * head_dim);
    std::vector<float> values(n * head_dim);
    for (int i = 0; i < n * head_dim; i++) {
        keys[i]   = sinf(i * 0.07f);
        values[i] = cosf(i * 0.05f);
    }

    size_t key_size = tq_quantize_kv_key_size(n, head_dim, key_type);
    size_t val_size = tq_quantize_kv_value_size(n, head_dim, val_type);
    ASSERT_GT(key_size, 0u);
    ASSERT_GT(val_size, 0u);

    std::vector<uint8_t> key_buf(key_size);
    std::vector<uint8_t> val_buf(val_size);

    tq_status status = tq_quantize_kv(ctx,
        keys.data(), values.data(), n, head_dim,
        key_type, val_type,
        key_buf.data(), key_size,
        val_buf.data(), val_size);
    ASSERT_EQ(status, TQ_OK);

    /* Both 4-bit: both should have high cosine */
    std::vector<float> key_deq(n * head_dim);
    std::vector<float> val_deq(n * head_dim);
    ASSERT_EQ(tq_dequantize_keys(ctx, key_buf.data(), n, head_dim, key_type, key_deq.data()), TQ_OK);
    ASSERT_EQ(tq_dequantize_keys(ctx, val_buf.data(), n, head_dim, val_type, val_deq.data()), TQ_OK);

    for (int i = 0; i < n; i++) {
        double kcs = cosine_similarity(keys.data() + i * head_dim,
                                        key_deq.data() + i * head_dim, head_dim);
        double vcs = cosine_similarity(values.data() + i * head_dim,
                                        val_deq.data() + i * head_dim, head_dim);
        EXPECT_GT(kcs, 0.99);
        EXPECT_GT(vcs, 0.99);
    }

    float key_bpe = tq_type_bpe(key_type);
    float val_bpe = tq_type_bpe(val_type);
    printf("  K/V Symmetric K4V4: key_bpe=%.2f, val_bpe=%.2f, avg=%.2f bit\n",
           key_bpe, val_bpe, (key_bpe + val_bpe) / 2.0f);
}

TEST_F(KVAsymmetricTest, NullInputs) {
    const int n = 4;
    const int head_dim = TQ_BK;
    std::vector<float> data(n * head_dim, 1.0f);
    std::vector<uint8_t> buf(4096);

    /* NULL context */
    EXPECT_EQ(tq_quantize_kv(nullptr,
        data.data(), data.data(), n, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        buf.data(), buf.size(), buf.data(), buf.size()),
        TQ_ERR_NULL_PTR);

    /* NULL keys */
    EXPECT_EQ(tq_quantize_kv(ctx,
        nullptr, data.data(), n, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        buf.data(), buf.size(), buf.data(), buf.size()),
        TQ_ERR_NULL_PTR);

    /* NULL values */
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), nullptr, n, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        buf.data(), buf.size(), buf.data(), buf.size()),
        TQ_ERR_NULL_PTR);

    /* NULL key_out */
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), data.data(), n, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        nullptr, buf.size(), buf.data(), buf.size()),
        TQ_ERR_NULL_PTR);

    /* NULL val_out */
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), data.data(), n, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        buf.data(), buf.size(), nullptr, buf.size()),
        TQ_ERR_NULL_PTR);

    /* Invalid key type */
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), data.data(), n, head_dim,
        (tq_type)99, TQ_TYPE_UNIFORM_2B,
        buf.data(), buf.size(), buf.data(), buf.size()),
        TQ_ERR_INVALID_TYPE);

    /* Invalid value type */
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), data.data(), n, head_dim,
        TQ_TYPE_UNIFORM_4B, (tq_type)99,
        buf.data(), buf.size(), buf.data(), buf.size()),
        TQ_ERR_INVALID_TYPE);

    /* Buffer too small for keys */
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), data.data(), n, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        buf.data(), 1, buf.data(), buf.size()),
        TQ_ERR_BUFFER_TOO_SMALL);

    /* Buffer too small for values */
    size_t key_size = tq_quantize_kv_key_size(n, head_dim, TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), data.data(), n, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        buf.data(), key_size, buf.data(), 1),
        TQ_ERR_BUFFER_TOO_SMALL);

    /* n=0 should succeed (no-op) */
    EXPECT_EQ(tq_quantize_kv(ctx,
        data.data(), data.data(), 0, head_dim,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
        buf.data(), buf.size(), buf.data(), buf.size()),
        TQ_OK);
}

TEST_F(KVAsymmetricTest, MixedTypes) {
    const int n = 4;
    const int head_dim = TQ_BK;
    const tq_type key_type = TQ_TYPE_POLAR_4B;
    const tq_type val_type = TQ_TYPE_UNIFORM_2B;

    std::vector<float> keys(n * head_dim);
    std::vector<float> values(n * head_dim);
    for (int i = 0; i < n * head_dim; i++) {
        keys[i]   = sinf(i * 0.07f);
        values[i] = cosf(i * 0.05f);
    }

    size_t key_size = tq_quantize_kv_key_size(n, head_dim, key_type);
    size_t val_size = tq_quantize_kv_value_size(n, head_dim, val_type);
    ASSERT_GT(key_size, 0u);
    ASSERT_GT(val_size, 0u);

    std::vector<uint8_t> key_buf(key_size);
    std::vector<uint8_t> val_buf(val_size);

    tq_status status = tq_quantize_kv(ctx,
        keys.data(), values.data(), n, head_dim,
        key_type, val_type,
        key_buf.data(), key_size,
        val_buf.data(), val_size);
    ASSERT_EQ(status, TQ_OK);

    /* Dequantize and verify both streams */
    std::vector<float> key_deq(n * head_dim);
    std::vector<float> val_deq(n * head_dim);
    ASSERT_EQ(tq_dequantize_keys(ctx, key_buf.data(), n, head_dim, key_type, key_deq.data()), TQ_OK);
    ASSERT_EQ(tq_dequantize_keys(ctx, val_buf.data(), n, head_dim, val_type, val_deq.data()), TQ_OK);

    double key_cos_sum = 0, val_cos_sum = 0;
    for (int i = 0; i < n; i++) {
        key_cos_sum += cosine_similarity(keys.data() + i * head_dim,
                                          key_deq.data() + i * head_dim, head_dim);
        val_cos_sum += cosine_similarity(values.data() + i * head_dim,
                                          val_deq.data() + i * head_dim, head_dim);
    }

    /* Polar 4B keys should have reasonable quality, uniform 2B values lower */
    EXPECT_GT(key_cos_sum / n, 0.90) << "Mixed: Polar4B key cosine too low";
    EXPECT_GT(val_cos_sum / n, 0.85) << "Mixed: Uniform2B value cosine too low";

    float key_bpe = tq_type_bpe(key_type);
    float val_bpe = tq_type_bpe(val_type);
    printf("  K/V Mixed (Polar4B + Uniform2B): key_bpe=%.2f, val_bpe=%.2f, avg=%.2f bit\n",
           key_bpe, val_bpe, (key_bpe + val_bpe) / 2.0f);
    printf("  Key cosine=%.6f, Value cosine=%.6f\n", key_cos_sum / n, val_cos_sum / n);
}
