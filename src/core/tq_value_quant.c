/**
 * Value quantization — reference C implementation
 *
 * Simple uniform quantization for value vectors (2-bit or 4-bit).
 * Reuses the uniform quantize/dequantize functions.
 */

#include "turboquant/turboquant.h"
#include <stdlib.h>
#include <string.h>

/* These are defined in tq_uniform.c */
extern void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);

size_t tq_quantize_values_size(int n, int head_dim, int bits) {
    int total = n * head_dim;
    int block_size = TQ_BK;
    int num_blocks = (total + block_size - 1) / block_size;

    if (bits == 4) {
        return (size_t)num_blocks * sizeof(block_tq_uniform_4b);
    } else if (bits == 2) {
        return (size_t)num_blocks * sizeof(block_tq_uniform_2b);
    }
    return 0;
}
