/**
 * TurboQuant context and high-level API implementation
 *
 * Thread safety: a pthread mutex protects quantize and attention calls.
 */

#include "turboquant/turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>

struct tq_context {
    tq_backend backend;
    pthread_mutex_t mutex;
};

tq_status tq_init(tq_context_t** ctx, tq_backend backend) {
    if (!ctx) return TQ_ERR_NULL_PTR;

    tq_context_t* c = (tq_context_t*)calloc(1, sizeof(tq_context_t));
    if (!c) return TQ_ERR_OUT_OF_MEM;

    c->backend = (backend == TQ_BACKEND_AUTO) ? TQ_BACKEND_CPU : backend;

    if (pthread_mutex_init(&c->mutex, NULL) != 0) {
        free(c);
        return TQ_ERR_BACKEND;
    }

    *ctx = c;
    return TQ_OK;
}

void tq_free(tq_context_t* ctx) {
    if (!ctx) return;
    pthread_mutex_destroy(&ctx->mutex);
    free(ctx);
}

tq_backend tq_get_backend(const tq_context_t* ctx) {
    if (!ctx) return TQ_BACKEND_CPU;
    return ctx->backend;
}

size_t tq_quantize_keys_size(int n, int head_dim, tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0;
    if (n <= 0 || head_dim <= 0) return 0;
    if (n > TQ_MAX_SEQ_LEN) return 0;

    size_t block_size = TQ_TRAITS[type].block_size;
    size_t type_size  = TQ_TRAITS[type].type_size;
    /* Each key vector of head_dim elements uses ceil(head_dim/block_size) blocks */
    size_t blocks_per_key = ((size_t)head_dim + block_size - 1) / block_size;

    /* Check multiplication overflow: n * blocks_per_key * type_size */
    size_t nb = (size_t)n * blocks_per_key;
    if (blocks_per_key != 0 && nb / blocks_per_key != (size_t)n) return 0;
    size_t result = nb * type_size;
    if (type_size != 0 && result / type_size != nb) return 0;

    return result;
}

tq_status tq_quantize_keys(tq_context_t* ctx,
                           const float* keys, int n, int head_dim,
                           tq_type type,
                           void* out, size_t out_size) {
    if (!ctx || !keys || !out) return TQ_ERR_NULL_PTR;
    if (type < 0 || type >= TQ_TYPE_COUNT) return TQ_ERR_INVALID_TYPE;
    if (n == 0) return TQ_OK;
    if (head_dim < 2) return TQ_ERR_INVALID_DIM;

    /* PolarQuant and TurboQuant require even head_dim (polar coordinate pairs) */
    if (type == TQ_TYPE_POLAR_3B || type == TQ_TYPE_POLAR_4B ||
        type == TQ_TYPE_TURBO_3B || type == TQ_TYPE_TURBO_4B) {
        if (head_dim % 2 != 0) return TQ_ERR_INVALID_DIM;
    }

    size_t needed = tq_quantize_keys_size(n, head_dim, type);
    if (needed == 0) return TQ_ERR_INVALID_DIM;
    if (out_size < needed) return TQ_ERR_BUFFER_TOO_SMALL;

    tq_quantize_fn qfn = TQ_TRAITS[type].quantize;
    if (!qfn) return TQ_ERR_NOT_IMPL;

    pthread_mutex_lock(&ctx->mutex);

    size_t type_size = TQ_TRAITS[type].type_size;
    uint8_t* dst = (uint8_t*)out;

    for (int i = 0; i < n; i++) {
        qfn(keys + i * head_dim, dst, head_dim);
        dst += type_size;
    }

    pthread_mutex_unlock(&ctx->mutex);

    return TQ_OK;
}

tq_status tq_dequantize_keys(tq_context_t* ctx,
                             const void* quantized, int n, int head_dim,
                             tq_type type,
                             float* out) {
    if (!ctx || !quantized || !out) return TQ_ERR_NULL_PTR;
    if (type < 0 || type >= TQ_TYPE_COUNT) return TQ_ERR_INVALID_TYPE;
    if (head_dim <= 0) return TQ_ERR_INVALID_DIM;

    tq_dequantize_fn dfn = TQ_TRAITS[type].dequantize;
    if (!dfn) return TQ_ERR_NOT_IMPL;

    size_t type_size = TQ_TRAITS[type].type_size;
    const uint8_t* src = (const uint8_t*)quantized;

    for (int i = 0; i < n; i++) {
        dfn(src, out + i * head_dim, head_dim);
        src += type_size;
    }

    return TQ_OK;
}

tq_status tq_quantize_values(tq_context_t* ctx,
                             const float* values, int n, int head_dim,
                             int bits,
                             void* out, size_t out_size) {
    if (!ctx || !values || !out) return TQ_ERR_NULL_PTR;
    if (bits != 2 && bits != 4) return TQ_ERR_INVALID_TYPE;

    tq_type type = (bits == 4) ? TQ_TYPE_UNIFORM_4B : TQ_TYPE_UNIFORM_2B;
    tq_quantize_fn qfn = TQ_TRAITS[type].quantize;
    if (!qfn) return TQ_ERR_NOT_IMPL;

    size_t type_size = TQ_TRAITS[type].type_size;
    uint8_t* dst = (uint8_t*)out;

    for (int i = 0; i < n; i++) {
        qfn(values + i * head_dim, dst, head_dim);
        dst += type_size;
    }

    return TQ_OK;
}

tq_status tq_attention(tq_context_t* ctx,
                       const float* query,
                       const void* kv_cache,
                       int seq_len, int head_dim,
                       tq_type type,
                       float* scores) {
    if (!ctx || !query || !kv_cache || !scores) return TQ_ERR_NULL_PTR;
    if (type < 0 || type >= TQ_TYPE_COUNT) return TQ_ERR_INVALID_TYPE;
    if (seq_len == 0) return TQ_OK;
    if (head_dim < 2) return TQ_ERR_INVALID_DIM;

    /* PolarQuant and TurboQuant require even head_dim */
    if (type == TQ_TYPE_POLAR_3B || type == TQ_TYPE_POLAR_4B ||
        type == TQ_TYPE_TURBO_3B || type == TQ_TYPE_TURBO_4B) {
        if (head_dim % 2 != 0) return TQ_ERR_INVALID_DIM;
    }

    tq_attention_fn afn = TQ_TRAITS[type].attention;
    if (!afn) return TQ_ERR_NOT_IMPL;

    pthread_mutex_lock(&ctx->mutex);
    afn(query, kv_cache, scores, seq_len, head_dim);
    pthread_mutex_unlock(&ctx->mutex);

    return TQ_OK;
}

tq_type tq_recommend_strategy(int head_dim, int target_bits,
                              float quality_threshold) {
    (void)head_dim;
    (void)quality_threshold;
    if (target_bits <= 1) return TQ_TYPE_QJL_1B;
    if (target_bits <= 3) return TQ_TYPE_TURBO_3B;
    return TQ_TYPE_TURBO_4B;
}
