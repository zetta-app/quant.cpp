/**
 * Progressive compression engine
 *
 * Implements tiered compression for KV cache:
 *   Tier 0 (Hot)  — FP16 / FP32, recent tokens within residual_window
 *   Tier 1 (Warm) — 4-bit quantized, tokens within warm_window
 *   Tier 2 (Cold) — 3-bit quantized, older tokens
 *
 * As new tokens are appended, older tokens are progressively
 * compressed from higher precision to lower precision tiers.
 */

#include "turboquant/turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================
 * Internal structures
 * ================================================================ */

/* Per-token storage: either raw FP32 or quantized block */
typedef struct {
    int    tier;          /* 0 = FP32 hot, 1 = warm quantized, 2 = cold quantized */
    void*  data;          /* FP32 buffer or quantized block */
    int    head_dim;      /* dimension of the stored vector */
    tq_type stored_type;  /* quantization type used for this slot's data */
} tq_progressive_slot_t;

/* Progressive compression engine state */
typedef struct tq_progressive {
    tq_progressive_config_t config;
    tq_progressive_slot_t*  slots;      /* array of token slots */
    int                     capacity;   /* total slot capacity */
    int                     count;      /* number of tokens stored */
    int                     head_dim;   /* head dimension for all tokens */
} tq_progressive_t;

/* ================================================================
 * Lifecycle
 * ================================================================ */

tq_status tq_progressive_create(tq_progressive_t** out,
                                const tq_progressive_config_t* config,
                                int head_dim, int max_tokens) {
    if (!out || !config) return TQ_ERR_NULL_PTR;
    if (head_dim <= 0 || max_tokens <= 0) return TQ_ERR_INVALID_DIM;

    tq_progressive_t* p = (tq_progressive_t*)calloc(1, sizeof(tq_progressive_t));
    if (!p) return TQ_ERR_OUT_OF_MEM;

    p->config   = *config;
    p->head_dim = head_dim;
    p->capacity = max_tokens;
    p->count    = 0;

    /* Set defaults if not specified */
    if (p->config.residual_window <= 0) p->config.residual_window = 128;
    if (p->config.warm_window <= 0)     p->config.warm_window = 256;

    p->slots = (tq_progressive_slot_t*)calloc((size_t)max_tokens,
                                              sizeof(tq_progressive_slot_t));
    if (!p->slots) {
        free(p);
        return TQ_ERR_OUT_OF_MEM;
    }

    *out = p;
    return TQ_OK;
}

void tq_progressive_free(tq_progressive_t* p) {
    if (!p) return;
    if (p->slots) {
        for (int i = 0; i < p->count; i++) {
            free(p->slots[i].data);
        }
        free(p->slots);
    }
    free(p);
}

/* ================================================================
 * Internal: compress a slot from one tier to another
 * ================================================================ */

static tq_status compress_slot(tq_progressive_slot_t* slot,
                               tq_type target_type, int head_dim) {
    if (!slot || !slot->data) return TQ_ERR_NULL_PTR;

    int current_tier = slot->tier;

    /* Get the quantize function from traits */
    if (target_type < 0 || target_type >= TQ_TYPE_COUNT)
        return TQ_ERR_INVALID_TYPE;
    tq_quantize_fn qfn = TQ_TRAITS[target_type].quantize;
    if (!qfn) return TQ_ERR_NOT_IMPL;

    if (current_tier == 0) {
        /* Tier 0 (FP32) -> quantized tier */
        float* fp_data = (float*)slot->data;
        size_t block_size = TQ_TRAITS[target_type].type_size;
        void* qblock = calloc(1, block_size);
        if (!qblock) return TQ_ERR_OUT_OF_MEM;

        qfn(fp_data, qblock, head_dim);

        free(slot->data);
        slot->data = qblock;
    } else {
        /* Tier 1 (warm quantized) -> Tier 2 (cold quantized)
         * Must dequantize first, then re-quantize to colder type */
        tq_type src_type = slot->stored_type;
        if (src_type < 0 || src_type >= TQ_TYPE_COUNT)
            return TQ_ERR_INVALID_TYPE;

        tq_dequantize_fn dqfn = TQ_TRAITS[src_type].dequantize;
        if (!dqfn) return TQ_ERR_NOT_IMPL;

        /* Dequantize to temporary FP32 buffer */
        float* tmp = (float*)calloc((size_t)head_dim, sizeof(float));
        if (!tmp) return TQ_ERR_OUT_OF_MEM;

        dqfn(slot->data, tmp, head_dim);

        /* Quantize to target type */
        size_t block_size = TQ_TRAITS[target_type].type_size;
        void* qblock = calloc(1, block_size);
        if (!qblock) {
            free(tmp);
            return TQ_ERR_OUT_OF_MEM;
        }

        qfn(tmp, qblock, head_dim);

        free(tmp);
        free(slot->data);
        slot->data = qblock;
    }

    return TQ_OK;
}

/* ================================================================
 * Determine which tier a position falls into
 * ================================================================ */

int tq_progressive_get_tier(const tq_progressive_t* p, int position) {
    if (!p) return -1;
    if (position < 0 || position >= p->count) return -1;

    int age = p->count - 1 - position;  /* 0 = newest */

    if (age < p->config.residual_window) {
        return 0;  /* Hot tier: FP32 */
    } else if (age < p->config.residual_window + p->config.warm_window) {
        return 1;  /* Warm tier */
    } else {
        return 2;  /* Cold tier */
    }
}

/* ================================================================
 * Append a new token and trigger compression of older tokens
 * ================================================================ */

tq_status tq_progressive_append(tq_progressive_t* p,
                                const float* key, int head_dim) {
    if (!p || !key) return TQ_ERR_NULL_PTR;
    if (head_dim != p->head_dim) return TQ_ERR_INVALID_DIM;
    if (p->count >= p->capacity) return TQ_ERR_OUT_OF_MEM;

    /* Store new token as FP32 (Tier 0) */
    int idx = p->count;
    float* data = (float*)calloc((size_t)head_dim, sizeof(float));
    if (!data) return TQ_ERR_OUT_OF_MEM;
    memcpy(data, key, (size_t)head_dim * sizeof(float));

    p->slots[idx].data        = data;
    p->slots[idx].head_dim    = head_dim;
    p->slots[idx].tier        = 0;
    p->slots[idx].stored_type = TQ_TYPE_COUNT; /* No quant type for FP32 tier */
    p->count++;

    /* Check if any tokens need tier transitions */
    for (int i = 0; i < p->count; i++) {
        int target_tier = tq_progressive_get_tier(p, i);
        int current_tier = p->slots[i].tier;

        if (target_tier > current_tier) {
            /* Need to compress this slot */
            tq_type target_type;
            if (target_tier == 1) {
                target_type = p->config.warm_type;
            } else {
                target_type = p->config.cold_type;
                if (!p->config.enable_recompression && current_tier == 1) {
                    /* Recompression disabled, skip tier 1 -> tier 2 */
                    continue;
                }
            }

            tq_status st = compress_slot(&p->slots[i], target_type, head_dim);
            if (st == TQ_OK) {
                p->slots[i].tier = target_tier;
                p->slots[i].stored_type = target_type;
            }
            /* If compression fails, keep current tier (graceful degradation) */
        }
    }

    return TQ_OK;
}

/* ================================================================
 * Get token data and its current tier
 * ================================================================ */

tq_status tq_progressive_get(const tq_progressive_t* p, int position,
                             const void** data, int* tier) {
    if (!p || !data || !tier) return TQ_ERR_NULL_PTR;
    if (position < 0 || position >= p->count) return TQ_ERR_INVALID_DIM;

    *data = p->slots[position].data;
    *tier = p->slots[position].tier;
    return TQ_OK;
}

/* ================================================================
 * Get count and config
 * ================================================================ */

int tq_progressive_count(const tq_progressive_t* p) {
    if (!p) return 0;
    return p->count;
}

const tq_progressive_config_t* tq_progressive_get_config(const tq_progressive_t* p) {
    if (!p) return NULL;
    return &p->config;
}

/* ================================================================
 * Mixed-precision attention across tiers
 *
 * For each token, dequantize from its tier and compute dot product
 * with the query vector.
 * ================================================================ */

tq_status tq_progressive_attention(const tq_progressive_t* p,
                                   const float* query,
                                   float* scores, int head_dim) {
    if (!p || !query || !scores) return TQ_ERR_NULL_PTR;
    if (head_dim != p->head_dim) return TQ_ERR_INVALID_DIM;

    float dequant_buf[TQ_BK_QJL]; /* Large enough for any block */

    for (int i = 0; i < p->count; i++) {
        int tier = p->slots[i].tier;
        float dot = 0.0f;

        if (tier == 0) {
            /* FP32: direct dot product */
            const float* fp_data = (const float*)p->slots[i].data;
            for (int d = 0; d < head_dim; d++) {
                dot += query[d] * fp_data[d];
            }
        } else {
            /* Quantized: dequantize then dot product */
            tq_type qtype = (tier == 1) ? p->config.warm_type : p->config.cold_type;
            tq_dequantize_fn dqfn = TQ_TRAITS[qtype].dequantize;
            if (dqfn) {
                dqfn(p->slots[i].data, dequant_buf, head_dim);
                for (int d = 0; d < head_dim; d++) {
                    dot += query[d] * dequant_buf[d];
                }
            }
        }

        scores[i] = dot;
    }

    return TQ_OK;
}
