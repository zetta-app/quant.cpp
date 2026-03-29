/**
 * TurboQuant -- Metal host utilities (Objective-C)
 *
 * Provides MTLDevice/MTLCommandQueue management, compute pipeline
 * caching, and buffer helpers for the Metal GPU backend.
 */
#ifndef TQ_METAL_COMMON_H
#define TQ_METAL_COMMON_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Pipeline identifiers for compute kernels
 * ============================================================ */

typedef enum {
    TQ_METAL_PIPE_POLAR_QUANTIZE = 0,
    TQ_METAL_PIPE_POLAR_ATTENTION,
    TQ_METAL_PIPE_QJL_QUANTIZE,
    TQ_METAL_PIPE_QJL_ATTENTION,
    TQ_METAL_PIPE_TURBO_QUANTIZE,
    TQ_METAL_PIPE_TURBO_ATTENTION,
    TQ_METAL_PIPE_FUSED_POLAR_CACHE,
    TQ_METAL_PIPE_FUSED_UNIFORM_CACHE,
    TQ_METAL_PIPE_VALUE_QUANTIZE_4B,
    TQ_METAL_PIPE_VALUE_QUANTIZE_2B,
    TQ_METAL_PIPE_VALUE_DEQUANT_MATMUL,
    TQ_METAL_PIPE_COUNT
} tq_metal_pipeline_id;

/* ============================================================
 * Metal backend state (opaque pointer for C consumers)
 * ============================================================ */

typedef struct tq_metal_context tq_metal_context_t;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/**
 * Initialize the Metal backend.
 * Returns 0 on success, -1 if no Metal device is available.
 */
int tq_init_metal_backend(void);

/**
 * Shut down the Metal backend and release all resources.
 */
void tq_shutdown_metal_backend(void);

/**
 * Returns 1 if Metal is available on this system, 0 otherwise.
 */
int tq_metal_is_available(void);

/**
 * Returns the name of the Metal device (e.g., "Apple M2 Pro").
 */
const char* tq_metal_device_name(void);

/* ============================================================
 * Buffer management
 * ============================================================ */

/**
 * Allocate a Metal buffer with shared storage mode (CPU/GPU visible).
 * Returns an opaque handle, or NULL on failure.
 */
void* tq_metal_alloc_shared(size_t size);

/**
 * Free a Metal buffer.
 */
void tq_metal_free(void* buffer);

/**
 * Get the CPU-accessible pointer for a shared Metal buffer.
 */
void* tq_metal_buffer_contents(void* buffer);

/* ============================================================
 * Kernel dispatch
 * ============================================================ */

/**
 * Dispatch a compute kernel by pipeline ID.
 * buffers/buffer_sizes arrays must have num_buffers entries.
 * grid_size and threadgroup_size specify the dispatch dimensions.
 */
void tq_metal_dispatch_kernel(
    tq_metal_pipeline_id pipeline,
    void** buffers,
    size_t* buffer_sizes,
    int num_buffers,
    int grid_x, int grid_y, int grid_z,
    int tg_x, int tg_y, int tg_z);

/**
 * Wait for all submitted Metal commands to complete.
 */
void tq_metal_synchronize(void);

/* ============================================================
 * High-level quantize/attention wrappers
 * ============================================================ */

void tq_polar_quantize_metal(const float* src, void* dst, int n);
void tq_polar_attention_metal(const float* query, const void* kv_cache,
                               float* scores, int seq_len, int head_dim);

void tq_qjl_quantize_metal(const float* src, void* dst, int n);
void tq_qjl_attention_metal(const float* query, const void* kv_cache,
                              float* scores, int seq_len, int head_dim);

void tq_turbo_quantize_metal(const float* src, void* dst, int n);
void tq_turbo_attention_metal(const float* query, const void* kv_cache,
                               float* scores, int seq_len, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* TQ_METAL_COMMON_H */
