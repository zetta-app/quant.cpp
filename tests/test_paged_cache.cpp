#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
}
#include <cmath>
#include <vector>

TEST(PagedCache, CreateAndFree) {
    tq_cache_t* cache = nullptr;
    tq_status status = tq_cache_create(&cache, 16, 64, 4, 128,
                                        TQ_TYPE_UNIFORM_4B);
    ASSERT_EQ(status, TQ_OK);
    ASSERT_NE(cache, nullptr);

    // Initial seq_len should be 0
    for (int h = 0; h < 4; h++) {
        EXPECT_EQ(tq_cache_seq_len(cache, h), 0);
    }

    tq_cache_free(cache);
}

TEST(PagedCache, AppendAndSeqLen) {
    tq_cache_t* cache = nullptr;
    tq_status status = tq_cache_create(&cache, 16, 64, 2, 128,
                                        TQ_TYPE_UNIFORM_4B);
    ASSERT_EQ(status, TQ_OK);

    std::vector<float> key(128), value(128);
    for (int i = 0; i < 128; i++) {
        key[i] = sinf(i * 0.1f);
        value[i] = cosf(i * 0.1f);
    }

    // Append a few tokens to head 0
    for (int t = 0; t < 5; t++) {
        status = tq_cache_append(cache, 0, key.data(), value.data(), 128);
        EXPECT_EQ(status, TQ_OK);
    }

    EXPECT_EQ(tq_cache_seq_len(cache, 0), 5);
    EXPECT_EQ(tq_cache_seq_len(cache, 1), 0);

    tq_cache_free(cache);
}

TEST(PagedCache, GetBlock) {
    tq_cache_t* cache = nullptr;
    tq_status status = tq_cache_create(&cache, 16, 64, 1, 128,
                                        TQ_TYPE_UNIFORM_4B);
    ASSERT_EQ(status, TQ_OK);

    std::vector<float> key(128);
    for (int i = 0; i < 128; i++) key[i] = sinf(i * 0.1f);

    status = tq_cache_append(cache, 0, key.data(), nullptr, 128);
    ASSERT_EQ(status, TQ_OK);

    const void* data = nullptr;
    tq_type type;
    status = tq_cache_get_block(cache, 0, 0, &data, &type);
    EXPECT_EQ(status, TQ_OK);
    EXPECT_NE(data, nullptr);
    EXPECT_EQ(type, TQ_TYPE_UNIFORM_4B);

    tq_cache_free(cache);
}

TEST(PagedCache, InvalidHead) {
    tq_cache_t* cache = nullptr;
    tq_cache_create(&cache, 16, 64, 2, 128, TQ_TYPE_UNIFORM_4B);

    std::vector<float> key(128, 1.0f);

    // Invalid head index
    tq_status status = tq_cache_append(cache, 5, key.data(), nullptr, 128);
    EXPECT_NE(status, TQ_OK);

    EXPECT_EQ(tq_cache_seq_len(cache, -1), 0);
    EXPECT_EQ(tq_cache_seq_len(cache, 99), 0);

    tq_cache_free(cache);
}

TEST(PagedCache, NullPointer) {
    tq_status status = tq_cache_create(nullptr, 16, 64, 1, 128,
                                        TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(status, TQ_ERR_NULL_PTR);

    EXPECT_EQ(tq_cache_seq_len(nullptr, 0), 0);

    tq_cache_free(nullptr); // Should not crash
}

TEST(PagedCache, CopyOnWriteBasic) {
    tq_cache_t* cache = nullptr;
    tq_status status = tq_cache_create(&cache, 16, 64, 1, 128,
                                        TQ_TYPE_UNIFORM_4B);
    ASSERT_EQ(status, TQ_OK);

    // Append a key to create block 0
    std::vector<float> key1(128);
    for (int i = 0; i < 128; i++) key1[i] = sinf(i * 0.1f);
    status = tq_cache_append(cache, 0, key1.data(), nullptr, 128);
    ASSERT_EQ(status, TQ_OK);

    // Verify initial ref_count is 1
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 1);

    // Share block 0 (simulates beam search fork)
    status = tq_cache_share_block(cache, 0, 0);
    ASSERT_EQ(status, TQ_OK);
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 2);

    // Capture the shared block's data before modification
    const void* shared_data = nullptr;
    tq_type shared_type;
    status = tq_cache_get_block(cache, 0, 0, &shared_data, &shared_type);
    ASSERT_EQ(status, TQ_OK);
    ASSERT_NE(shared_data, nullptr);

    // Save a copy of the shared block's content
    size_t type_size = TQ_TRAITS[TQ_TYPE_UNIFORM_4B].type_size;
    std::vector<uint8_t> shared_snapshot(type_size);
    memcpy(shared_snapshot.data(), shared_data, type_size);

    // Append a different key — this triggers CoW since ref_count > 1
    // The block_idx for seq_len=1 with block_size=16 is still 0
    // But seq_len / block_size = 0, so it writes to block 0 again
    std::vector<float> key2(128);
    for (int i = 0; i < 128; i++) key2[i] = cosf(i * 0.3f);
    status = tq_cache_append(cache, 0, key2.data(), nullptr, 128);
    ASSERT_EQ(status, TQ_OK);

    // After CoW, ref_count should be back to 1 (new private copy)
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 1);

    // The block pointer should have changed (new allocation)
    const void* new_data = nullptr;
    tq_type new_type;
    status = tq_cache_get_block(cache, 0, 0, &new_data, &new_type);
    ASSERT_EQ(status, TQ_OK);

    // The new block should be different from the old shared pointer
    // (CoW allocates a new block)
    EXPECT_NE(new_data, shared_data);

    tq_cache_free(cache);
}

TEST(PagedCache, ValueStorage) {
    tq_cache_t* cache = nullptr;
    tq_status status = tq_cache_create(&cache, 16, 64, 1, 128,
                                        TQ_TYPE_UNIFORM_4B);
    ASSERT_EQ(status, TQ_OK);

    // Create key and value vectors
    std::vector<float> key(128), value(128);
    for (int i = 0; i < 128; i++) {
        key[i] = sinf(i * 0.1f);
        value[i] = cosf(i * 0.1f);
    }

    // Append with both key and value
    status = tq_cache_append(cache, 0, key.data(), value.data(), 128);
    ASSERT_EQ(status, TQ_OK);

    // Retrieve the stored value block
    const void* val_data = nullptr;
    status = tq_cache_get_value(cache, 0, 0, &val_data);
    ASSERT_EQ(status, TQ_OK);
    ASSERT_NE(val_data, nullptr);

    // Dequantize the value and compare with original
    tq_dequantize_fn dqfn = TQ_TRAITS[TQ_TYPE_UNIFORM_4B].dequantize;
    ASSERT_NE(dqfn, nullptr);
    std::vector<float> recovered(128);
    dqfn(val_data, recovered.data(), 128);

    // Compute MSE between original and recovered value
    float mse = 0.0f;
    for (int i = 0; i < 128; i++) {
        float diff = recovered[i] - value[i];
        mse += diff * diff;
    }
    mse /= 128.0f;

    EXPECT_LT(mse, 0.1f) << "Value storage MSE too high: " << mse;

    // Verify all recovered values are finite
    for (int i = 0; i < 128; i++) {
        EXPECT_TRUE(std::isfinite(recovered[i]))
            << "Recovered value at index " << i << " is not finite";
    }

    tq_cache_free(cache);
}

TEST(PagedCache, RefCountLifecycle) {
    tq_cache_t* cache = nullptr;
    tq_status status = tq_cache_create(&cache, 16, 64, 1, 128,
                                        TQ_TYPE_UNIFORM_4B);
    ASSERT_EQ(status, TQ_OK);

    // Append to create block 0
    std::vector<float> key(128, 1.0f);
    status = tq_cache_append(cache, 0, key.data(), nullptr, 128);
    ASSERT_EQ(status, TQ_OK);

    // Initial ref_count = 1
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 1);

    // Share the block twice (ref_count -> 3)
    EXPECT_EQ(tq_cache_share_block(cache, 0, 0), TQ_OK);
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 2);

    EXPECT_EQ(tq_cache_share_block(cache, 0, 0), TQ_OK);
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 3);

    // Free once -> ref_count = 2
    EXPECT_EQ(tq_cache_free_block(cache, 0, 0), TQ_OK);
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 2);

    // Block data should still be valid (not freed yet)
    const void* data = nullptr;
    tq_type type;
    EXPECT_EQ(tq_cache_get_block(cache, 0, 0, &data, &type), TQ_OK);
    EXPECT_NE(data, nullptr);

    // Free again -> ref_count = 1
    EXPECT_EQ(tq_cache_free_block(cache, 0, 0), TQ_OK);
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 1);

    // Free last reference -> block data freed (ref_count = 0)
    EXPECT_EQ(tq_cache_free_block(cache, 0, 0), TQ_OK);
    EXPECT_EQ(tq_cache_block_ref_count(cache, 0, 0), 0);

    // Error cases
    EXPECT_NE(tq_cache_share_block(cache, 99, 0), TQ_OK);  // invalid head
    EXPECT_NE(tq_cache_free_block(cache, 99, 0), TQ_OK);   // invalid head
    EXPECT_EQ(tq_cache_block_ref_count(nullptr, 0, 0), 0); // null cache

    tq_cache_free(cache);
}
