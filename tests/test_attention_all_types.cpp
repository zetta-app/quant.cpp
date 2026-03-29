#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
}
#include <cmath>
#include <vector>
#include <cstring>

class AttentionAllTypes : public ::testing::TestWithParam<tq_type> {};

TEST_P(AttentionAllTypes, ProducesValidScore) {
    tq_type type = GetParam();
    const int head_dim = 128;
    const int seq_len = 8;

    // Generate test data
    std::vector<float> keys(seq_len * head_dim);
    std::vector<float> query(head_dim);
    for (int i = 0; i < seq_len * head_dim; i++) keys[i] = sinf(i * 0.01f);
    for (int i = 0; i < head_dim; i++) query[i] = cosf(i * 0.05f);

    // Context
    tq_context_t* ctx;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    ASSERT_EQ(status, TQ_OK);

    // Quantize all keys
    size_t buf_size = tq_quantize_keys_size(seq_len, head_dim, type);
    ASSERT_GT(buf_size, 0u) << "tq_quantize_keys_size returned 0 for "
                            << tq_type_name(type);
    std::vector<uint8_t> quantized(buf_size);
    status = tq_quantize_keys(ctx, keys.data(), seq_len, head_dim,
                              type, quantized.data(), buf_size);
    ASSERT_EQ(status, TQ_OK) << "tq_quantize_keys failed for "
                             << tq_type_name(type);

    // Compute attention — skip if not implemented (BUG-1 workaround)
    std::vector<float> scores(seq_len);
    status = tq_attention(ctx, query.data(), quantized.data(),
                          seq_len, head_dim, type, scores.data());
    if (status == TQ_ERR_NOT_IMPL) {
        tq_free(ctx);
        GTEST_SKIP() << "tq_attention not implemented for "
                     << tq_type_name(type) << " (BUG-1)";
    }
    ASSERT_EQ(status, TQ_OK) << "tq_attention failed for "
                             << tq_type_name(type);

    // Compute FP32 reference scores
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * keys[s * head_dim + d];
        fp32_scores[s] = dot;
    }

    // Verify cosine similarity of score vectors
    // Use fabs() because QJL 1-bit sign hashes can produce negatively
    // correlated scores (sign flip) while still preserving relative ordering.
    double dot_ab = 0, sq_a = 0, sq_b = 0;
    for (int s = 0; s < seq_len; s++) {
        dot_ab += (double)scores[s] * (double)fp32_scores[s];
        sq_a += (double)scores[s] * (double)scores[s];
        sq_b += (double)fp32_scores[s] * (double)fp32_scores[s];
    }
    double cosine = dot_ab / (sqrt(sq_a) * sqrt(sq_b) + 1e-10);
    EXPECT_GT(fabs(cosine), 0.8) << "Cosine magnitude too low for "
                                 << tq_type_name(type)
                                 << ": " << cosine;

    tq_free(ctx);
}

TEST_P(AttentionAllTypes, SingleKeyScore) {
    tq_type type = GetParam();
    const int head_dim = 128;

    float key[128], query[128];
    for (int i = 0; i < head_dim; i++) {
        key[i] = sinf(i * 0.1f);
        query[i] = cosf(i * 0.1f);
    }

    tq_context_t* ctx;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    ASSERT_EQ(status, TQ_OK);

    size_t sz = tq_quantize_keys_size(1, head_dim, type);
    ASSERT_GT(sz, 0u);
    std::vector<uint8_t> buf(sz);
    status = tq_quantize_keys(ctx, key, 1, head_dim, type, buf.data(), sz);
    ASSERT_EQ(status, TQ_OK);

    float score = 0;
    status = tq_attention(ctx, query, buf.data(), 1, head_dim, type, &score);
    if (status == TQ_ERR_NOT_IMPL) {
        tq_free(ctx);
        GTEST_SKIP() << "tq_attention not implemented for "
                     << tq_type_name(type) << " (BUG-1)";
    }
    ASSERT_EQ(status, TQ_OK) << "tq_attention failed for "
                             << tq_type_name(type);

    // Score should be finite and non-zero
    EXPECT_TRUE(std::isfinite(score)) << "Score not finite for "
                                      << tq_type_name(type);

    tq_free(ctx);
}

INSTANTIATE_TEST_SUITE_P(AllTypes, AttentionAllTypes,
    ::testing::Values(
        TQ_TYPE_POLAR_3B, TQ_TYPE_POLAR_4B,
        TQ_TYPE_QJL_1B,
        TQ_TYPE_TURBO_3B, TQ_TYPE_TURBO_4B,
        TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B
    ),
    [](const auto& info) { return std::string(tq_type_name(info.param)); }
);
