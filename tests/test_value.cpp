#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
}
#include <vector>

TEST(ValueQuant, SizeCalculation4B) {
    size_t size = tq_quantize_values_size(1, 128, 4);
    // 128 elements, block size 128, one block of uniform_4b
    EXPECT_EQ(size, sizeof(block_tq_uniform_4b));
}

TEST(ValueQuant, SizeCalculation2B) {
    size_t size = tq_quantize_values_size(1, 128, 2);
    EXPECT_EQ(size, sizeof(block_tq_uniform_2b));
}

TEST(ValueQuant, SizeCalculationMultipleKeys) {
    size_t size = tq_quantize_values_size(4, 128, 4);
    EXPECT_EQ(size, 4 * sizeof(block_tq_uniform_4b));
}

TEST(ValueQuant, InvalidBits) {
    size_t size = tq_quantize_values_size(1, 128, 3);
    EXPECT_EQ(size, 0u);
}
