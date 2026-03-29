/**
 * Progressive compression tests
 *
 * Tests tier transition logic, FP16/FP32 hot tier retention,
 * and recompression from tier 1 to tier 2.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "turboquant/turboquant.h"

/* Forward declarations for progressive compression API */
typedef struct tq_progressive tq_progressive_t;

tq_status tq_progressive_create(tq_progressive_t** out,
                                const tq_progressive_config_t* config,
                                int head_dim, int max_tokens);
void tq_progressive_free(tq_progressive_t* p);
tq_status tq_progressive_append(tq_progressive_t* p,
                                const float* key, int head_dim);
int tq_progressive_get_tier(const tq_progressive_t* p, int position);
tq_status tq_progressive_get(const tq_progressive_t* p, int position,
                             const void** data, int* tier);
int tq_progressive_count(const tq_progressive_t* p);
const tq_progressive_config_t* tq_progressive_get_config(const tq_progressive_t* p);
tq_status tq_progressive_attention(const tq_progressive_t* p,
                                   const float* query,
                                   float* scores, int head_dim);
}

class ProgressiveTest : public ::testing::Test {
protected:
    static constexpr int HEAD_DIM = 128;

    std::vector<float> make_key(int seed) {
        std::vector<float> key(HEAD_DIM);
        for (int i = 0; i < HEAD_DIM; i++) {
            key[i] = sinf((float)(seed * 17 + i) * 0.1f);
        }
        return key;
    }
};

TEST_F(ProgressiveTest, CreateAndFree) {
    tq_progressive_config_t config = {};
    config.residual_window = 4;
    config.warm_window = 8;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;
    config.enable_recompression = 1;

    tq_progressive_t* prog = nullptr;
    tq_status st = tq_progressive_create(&prog, &config, HEAD_DIM, 100);
    ASSERT_EQ(st, TQ_OK);
    ASSERT_NE(prog, nullptr);

    EXPECT_EQ(tq_progressive_count(prog), 0);
    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, NullInputs) {
    tq_progressive_config_t config = {};
    config.residual_window = 4;
    config.warm_window = 8;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;

    tq_progressive_t* prog = nullptr;
    EXPECT_EQ(tq_progressive_create(nullptr, &config, HEAD_DIM, 100), TQ_ERR_NULL_PTR);
    EXPECT_EQ(tq_progressive_create(&prog, nullptr, HEAD_DIM, 100), TQ_ERR_NULL_PTR);
}

TEST_F(ProgressiveTest, AppendSingleToken) {
    tq_progressive_config_t config = {};
    config.residual_window = 4;
    config.warm_window = 8;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    auto key = make_key(0);
    EXPECT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    EXPECT_EQ(tq_progressive_count(prog), 1);

    /* Single token should be in hot tier (tier 0) */
    EXPECT_EQ(tq_progressive_get_tier(prog, 0), 0);

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, RecentTokensStayInFP32) {
    tq_progressive_config_t config = {};
    config.residual_window = 4;
    config.warm_window = 8;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;
    config.enable_recompression = 1;

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    /* Append 4 tokens (fills the residual window exactly) */
    for (int i = 0; i < 4; i++) {
        auto key = make_key(i);
        ASSERT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    }

    /* All 4 should be in tier 0 (hot) */
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(tq_progressive_get_tier(prog, i), 0)
            << "Token " << i << " should be in tier 0";
    }

    /* Verify FP32 data integrity for the most recent token */
    const void* data = nullptr;
    int tier = -1;
    ASSERT_EQ(tq_progressive_get(prog, 3, &data, &tier), TQ_OK);
    EXPECT_EQ(tier, 0);

    /* Check that the FP32 data matches what we put in */
    auto expected = make_key(3);
    const float* fp_data = (const float*)data;
    for (int d = 0; d < HEAD_DIM; d++) {
        EXPECT_FLOAT_EQ(fp_data[d], expected[d])
            << "FP32 data mismatch at dimension " << d;
    }

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, TierTransitionHotToWarm) {
    tq_progressive_config_t config = {};
    config.residual_window = 4;
    config.warm_window = 8;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;
    config.enable_recompression = 1;

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    /* Append 5 tokens: token 0 should transition to warm */
    for (int i = 0; i < 5; i++) {
        auto key = make_key(i);
        ASSERT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    }

    /* Token 0 (age=4) should have transitioned to tier 1 (warm) */
    const void* data = nullptr;
    int tier = -1;
    ASSERT_EQ(tq_progressive_get(prog, 0, &data, &tier), TQ_OK);
    EXPECT_EQ(tier, 1) << "Oldest token should be in warm tier";

    /* Tokens 1-4 should still be in tier 0 (hot) */
    for (int i = 1; i < 5; i++) {
        EXPECT_EQ(tq_progressive_get_tier(prog, i), 0)
            << "Token " << i << " should still be in hot tier";
    }

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, TierTransitionWarmToCold) {
    tq_progressive_config_t config = {};
    config.residual_window = 2;
    config.warm_window = 3;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;
    config.enable_recompression = 1;

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    /* Append 7 tokens:
     * residual_window=2, warm_window=3
     * age = count - 1 - position
     * Token 0 age=6 -> tier 2 (cold, age >= 2+3=5)
     * Token 1 age=5 -> tier 2 (cold)
     * Token 2 age=4 -> tier 1 (warm, 2 <= age < 5)
     * Token 3 age=3 -> tier 1 (warm)
     * Token 4 age=2 -> tier 1 (warm)
     * Token 5 age=1 -> tier 0 (hot, age < 2)
     * Token 6 age=0 -> tier 0 (hot)
     */
    for (int i = 0; i < 7; i++) {
        auto key = make_key(i);
        ASSERT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    }

    EXPECT_EQ(tq_progressive_get_tier(prog, 0), 2) << "Token 0 should be cold";
    EXPECT_EQ(tq_progressive_get_tier(prog, 1), 2) << "Token 1 should be cold";
    EXPECT_EQ(tq_progressive_get_tier(prog, 2), 1) << "Token 2 should be warm";
    EXPECT_EQ(tq_progressive_get_tier(prog, 3), 1) << "Token 3 should be warm";
    EXPECT_EQ(tq_progressive_get_tier(prog, 4), 1) << "Token 4 should be warm";
    EXPECT_EQ(tq_progressive_get_tier(prog, 5), 0) << "Token 5 should be hot";
    EXPECT_EQ(tq_progressive_get_tier(prog, 6), 0) << "Token 6 should be hot";

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, RecompressionDisabled) {
    tq_progressive_config_t config = {};
    config.residual_window = 2;
    config.warm_window = 2;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;
    config.enable_recompression = 0;  /* Disabled */

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    /* Append 5 tokens */
    for (int i = 0; i < 5; i++) {
        auto key = make_key(i);
        ASSERT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    }

    /* Token 0 age=4 => target tier=2, but recompression disabled
     * It should have gone to tier 1 first (from tier 0) and stayed there
     * because recompression from 1->2 is disabled */
    const void* data = nullptr;
    int tier = -1;
    ASSERT_EQ(tq_progressive_get(prog, 0, &data, &tier), TQ_OK);
    /* It stays at tier 1 because recompression from warm to cold is disabled */
    EXPECT_EQ(tier, 1) << "Token should stay in warm tier when recompression disabled";

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, MixedPrecisionAttention) {
    tq_progressive_config_t config = {};
    config.residual_window = 2;
    config.warm_window = 3;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;
    config.enable_recompression = 1;

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    /* Append 6 tokens (mix of tiers) */
    for (int i = 0; i < 6; i++) {
        auto key = make_key(i);
        ASSERT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    }

    /* Create a query vector */
    std::vector<float> query(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; i++) {
        query[i] = cosf(i * 0.05f);
    }

    std::vector<float> scores(6);
    tq_status st = tq_progressive_attention(prog, query.data(),
                                            scores.data(), HEAD_DIM);
    ASSERT_EQ(st, TQ_OK);

    /* Verify scores are finite and reasonable */
    for (int i = 0; i < 6; i++) {
        EXPECT_TRUE(std::isfinite(scores[i]))
            << "Score " << i << " is not finite: " << scores[i];
    }

    /* The hot-tier tokens should have the most accurate scores.
     * Compute reference scores for the hot-tier tokens (4 and 5) */
    for (int idx : {4, 5}) {
        auto key = make_key(idx);
        float ref_dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            ref_dot += query[d] * key[d];
        }
        /* Hot tier should be exact (FP32) */
        EXPECT_NEAR(scores[idx], ref_dot, 1e-5f)
            << "Hot tier score mismatch for token " << idx;
    }

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, CapacityExceeded) {
    tq_progressive_config_t config = {};
    config.residual_window = 2;
    config.warm_window = 2;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;

    tq_progressive_t* prog = nullptr;
    /* Only capacity for 3 tokens */
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 3), TQ_OK);

    for (int i = 0; i < 3; i++) {
        auto key = make_key(i);
        ASSERT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    }

    /* Fourth append should fail */
    auto key = make_key(3);
    EXPECT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_ERR_OUT_OF_MEM);

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, RecompressionWithPolarWarmType) {
    tq_progressive_config_t config = {};
    config.residual_window = 2;
    config.warm_window = 3;
    config.warm_type = TQ_TYPE_POLAR_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;
    config.enable_recompression = 1;

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    /* Append enough tokens so that the oldest transitions through tier 1 to tier 2.
     * residual_window=2, warm_window=3 => cold boundary at age >= 5
     * We need 7 tokens so token 0 (age=6) and token 1 (age=5) reach cold tier.
     */
    for (int i = 0; i < 7; i++) {
        auto key = make_key(i);
        ASSERT_EQ(tq_progressive_append(prog, key.data(), HEAD_DIM), TQ_OK);
    }

    /* Token 0 (age=6) should be in cold tier (tier 2) */
    const void* data = nullptr;
    int tier = -1;
    ASSERT_EQ(tq_progressive_get(prog, 0, &data, &tier), TQ_OK);
    EXPECT_EQ(tier, 2) << "Token 0 should have been recompressed to cold tier";
    EXPECT_NE(data, nullptr) << "Cold tier data should not be null";

    /* Token 1 (age=5) should also be cold */
    ASSERT_EQ(tq_progressive_get(prog, 1, &data, &tier), TQ_OK);
    EXPECT_EQ(tier, 2) << "Token 1 should be in cold tier";

    /* Verify the cold-tier data is valid by dequantizing it */
    tq_dequantize_fn dqfn = TQ_TRAITS[TQ_TYPE_UNIFORM_2B].dequantize;
    ASSERT_NE(dqfn, nullptr);
    std::vector<float> recovered(HEAD_DIM);
    dqfn(data, recovered.data(), HEAD_DIM);

    /* Check that recovered values are finite (valid data, not garbage) */
    for (int d = 0; d < HEAD_DIM; d++) {
        EXPECT_TRUE(std::isfinite(recovered[d]))
            << "Recovered value at dim " << d << " is not finite";
    }

    /* Compute MSE against original key for token 1.
     * After polar_4b -> uniform_2b recompression, expect some loss but MSE bounded. */
    auto orig = make_key(1);
    float mse = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
        float diff = recovered[d] - orig[d];
        mse += diff * diff;
    }
    mse /= HEAD_DIM;
    /* Recompression (polar -> dequant -> uniform_2b) has loss, but should be bounded */
    EXPECT_LT(mse, 1.0f) << "MSE after recompression is too high: " << mse;

    tq_progressive_free(prog);
}

TEST_F(ProgressiveTest, InvalidGetTier) {
    tq_progressive_config_t config = {};
    config.residual_window = 4;
    config.warm_window = 8;
    config.warm_type = TQ_TYPE_UNIFORM_4B;
    config.cold_type = TQ_TYPE_UNIFORM_2B;

    tq_progressive_t* prog = nullptr;
    ASSERT_EQ(tq_progressive_create(&prog, &config, HEAD_DIM, 100), TQ_OK);

    /* No tokens yet, any position is invalid */
    EXPECT_EQ(tq_progressive_get_tier(prog, 0), -1);
    EXPECT_EQ(tq_progressive_get_tier(prog, -1), -1);
    EXPECT_EQ(tq_progressive_get_tier(nullptr, 0), -1);

    tq_progressive_free(prog);
}
