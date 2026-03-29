/**
 * Paged KV cache — manages blocks of quantized key/value data
 *
 * Each head has an array of block pointers and tracks its sequence length.
 * tq_cache_append quantizes incoming keys and stores them.
 *
 * Copy-on-Write: blocks can be shared via tq_cache_share_block().
 * When appending to a block with ref_count > 1, the block is copied first.
 */

#include "turboquant/turboquant.h"
#include <stdlib.h>
#include <string.h>

/* Per-head cache state */
typedef struct {
    void**  blocks;        /* Array of block data pointers (max_blocks) */
    void**  value_blocks;  /* Array of value block data pointers (max_blocks) */
    tq_type* block_types;  /* Type of each block */
    int*    ref_counts;    /* Reference count for each block */
    int     seq_len;       /* Current sequence length for this head */
    int     num_blocks;    /* Number of allocated blocks */
} tq_head_cache_t;

/* Main cache structure */
struct tq_cache {
    int              block_size;   /* tokens per block */
    int              max_blocks;   /* max blocks per head */
    int              num_heads;
    int              head_dim;
    tq_type          default_type;
    tq_head_cache_t* heads;        /* Array of per-head caches */
};

tq_status tq_cache_create(tq_cache_t** cache,
                          int block_size, int max_blocks,
                          int num_heads, int head_dim,
                          tq_type default_type) {
    if (!cache) return TQ_ERR_NULL_PTR;
    if (block_size <= 0 || max_blocks <= 0 || num_heads <= 0 || head_dim <= 0)
        return TQ_ERR_INVALID_DIM;
    if (default_type < 0 || default_type >= TQ_TYPE_COUNT)
        return TQ_ERR_INVALID_TYPE;

    tq_cache_t* c = (tq_cache_t*)calloc(1, sizeof(tq_cache_t));
    if (!c) return TQ_ERR_OUT_OF_MEM;

    c->block_size   = block_size;
    c->max_blocks   = max_blocks;
    c->num_heads    = num_heads;
    c->head_dim     = head_dim;
    c->default_type = default_type;

    c->heads = (tq_head_cache_t*)calloc((size_t)num_heads, sizeof(tq_head_cache_t));
    if (!c->heads) {
        free(c);
        return TQ_ERR_OUT_OF_MEM;
    }

    for (int h = 0; h < num_heads; h++) {
        c->heads[h].blocks = (void**)calloc((size_t)max_blocks, sizeof(void*));
        c->heads[h].value_blocks = (void**)calloc((size_t)max_blocks, sizeof(void*));
        c->heads[h].block_types = (tq_type*)calloc((size_t)max_blocks, sizeof(tq_type));
        c->heads[h].ref_counts = (int*)calloc((size_t)max_blocks, sizeof(int));
        if (!c->heads[h].blocks || !c->heads[h].value_blocks ||
            !c->heads[h].block_types || !c->heads[h].ref_counts) {
            /* Cleanup on failure */
            for (int j = 0; j <= h; j++) {
                free(c->heads[j].blocks);
                free(c->heads[j].value_blocks);
                free(c->heads[j].block_types);
                free(c->heads[j].ref_counts);
            }
            free(c->heads);
            free(c);
            return TQ_ERR_OUT_OF_MEM;
        }
        c->heads[h].seq_len    = 0;
        c->heads[h].num_blocks = 0;
    }

    *cache = c;
    return TQ_OK;
}

tq_status tq_cache_append(tq_cache_t* cache,
                          int head_idx,
                          const float* key, const float* value,
                          int head_dim) {
    if (!cache || !key) return TQ_ERR_NULL_PTR;
    if (head_idx < 0 || head_idx >= cache->num_heads)
        return TQ_ERR_INVALID_DIM;
    if (head_dim != cache->head_dim)
        return TQ_ERR_INVALID_DIM;

    tq_head_cache_t* hc = &cache->heads[head_idx];

    /* Check if we need a new block */
    int block_idx = hc->seq_len / cache->block_size;
    if (block_idx >= cache->max_blocks) return TQ_ERR_OUT_OF_MEM;

    size_t type_size = TQ_TRAITS[cache->default_type].type_size;

    /* Allocate block if needed */
    if (!hc->blocks[block_idx]) {
        hc->blocks[block_idx] = calloc(1, type_size);
        if (!hc->blocks[block_idx]) return TQ_ERR_OUT_OF_MEM;
        hc->block_types[block_idx] = cache->default_type;
        hc->ref_counts[block_idx] = 1;
        hc->num_blocks = block_idx + 1;
    }

    /* Copy-on-Write: if block is shared, copy before writing */
    if (hc->ref_counts[block_idx] > 1) {
        void* new_block = malloc(type_size);
        if (!new_block) return TQ_ERR_OUT_OF_MEM;
        memcpy(new_block, hc->blocks[block_idx], type_size);

        /* Decrement old block's ref_count */
        hc->ref_counts[block_idx]--;

        /* Install the new copy */
        hc->blocks[block_idx] = new_block;
        hc->ref_counts[block_idx] = 1;
    }

    /* Quantize the key into the current block.
       For simplicity, we quantize one key vector per block.
       A full implementation would pack multiple tokens per block. */
    tq_quantize_fn qfn = TQ_TRAITS[cache->default_type].quantize;
    if (qfn) {
        qfn(key, hc->blocks[block_idx], head_dim);
    }

    /* Quantize and store value if provided */
    if (value) {
        size_t val_type_size = TQ_TRAITS[TQ_TYPE_UNIFORM_4B].type_size;
        if (!hc->value_blocks[block_idx]) {
            hc->value_blocks[block_idx] = calloc(1, val_type_size);
            if (!hc->value_blocks[block_idx]) return TQ_ERR_OUT_OF_MEM;
        }
        tq_quantize_fn vqfn = TQ_TRAITS[TQ_TYPE_UNIFORM_4B].quantize;
        if (vqfn) {
            vqfn(value, hc->value_blocks[block_idx], head_dim);
        }
    }

    hc->seq_len++;

    return TQ_OK;
}

tq_status tq_cache_get_block(const tq_cache_t* cache,
                             int head_idx, int block_idx,
                             const void** data, tq_type* type) {
    if (!cache || !data || !type) return TQ_ERR_NULL_PTR;
    if (head_idx < 0 || head_idx >= cache->num_heads)
        return TQ_ERR_INVALID_DIM;
    if (block_idx < 0 || block_idx >= cache->heads[head_idx].num_blocks)
        return TQ_ERR_INVALID_DIM;

    *data = cache->heads[head_idx].blocks[block_idx];
    *type = cache->heads[head_idx].block_types[block_idx];
    return TQ_OK;
}

int tq_cache_seq_len(const tq_cache_t* cache, int head_idx) {
    if (!cache) return 0;
    if (head_idx < 0 || head_idx >= cache->num_heads) return 0;
    return cache->heads[head_idx].seq_len;
}

tq_status tq_cache_share_block(tq_cache_t* cache, int head_idx, int block_idx) {
    if (!cache) return TQ_ERR_NULL_PTR;
    if (head_idx < 0 || head_idx >= cache->num_heads)
        return TQ_ERR_INVALID_DIM;
    if (block_idx < 0 || block_idx >= cache->heads[head_idx].num_blocks)
        return TQ_ERR_INVALID_DIM;
    if (!cache->heads[head_idx].blocks[block_idx])
        return TQ_ERR_NULL_PTR;

    cache->heads[head_idx].ref_counts[block_idx]++;
    return TQ_OK;
}

tq_status tq_cache_free_block(tq_cache_t* cache, int head_idx, int block_idx) {
    if (!cache) return TQ_ERR_NULL_PTR;
    if (head_idx < 0 || head_idx >= cache->num_heads)
        return TQ_ERR_INVALID_DIM;
    if (block_idx < 0 || block_idx >= cache->heads[head_idx].num_blocks)
        return TQ_ERR_INVALID_DIM;

    tq_head_cache_t* hc = &cache->heads[head_idx];
    if (!hc->blocks[block_idx]) return TQ_OK;

    hc->ref_counts[block_idx]--;
    if (hc->ref_counts[block_idx] <= 0) {
        free(hc->blocks[block_idx]);
        hc->blocks[block_idx] = NULL;
        hc->ref_counts[block_idx] = 0;
    }
    return TQ_OK;
}

int tq_cache_block_ref_count(const tq_cache_t* cache, int head_idx, int block_idx) {
    if (!cache) return 0;
    if (head_idx < 0 || head_idx >= cache->num_heads) return 0;
    if (block_idx < 0 || block_idx >= cache->heads[head_idx].num_blocks) return 0;
    return cache->heads[head_idx].ref_counts[block_idx];
}

tq_status tq_cache_get_value(const tq_cache_t* cache, int head_idx, int block_idx,
                             const void** data) {
    if (!cache || !data) return TQ_ERR_NULL_PTR;
    if (head_idx < 0 || head_idx >= cache->num_heads)
        return TQ_ERR_INVALID_DIM;
    if (block_idx < 0 || block_idx >= cache->heads[head_idx].num_blocks)
        return TQ_ERR_INVALID_DIM;

    *data = cache->heads[head_idx].value_blocks[block_idx];
    if (!*data) return TQ_ERR_NULL_PTR;
    return TQ_OK;
}

void tq_cache_free(tq_cache_t* cache) {
    if (!cache) return;
    for (int h = 0; h < cache->num_heads; h++) {
        for (int b = 0; b < cache->heads[h].num_blocks; b++) {
            /* Only free if ref_count indicates ownership.
               In a full implementation, shared blocks would be tracked
               globally. Here we free unconditionally since cache_free
               tears down the entire cache. */
            free(cache->heads[h].blocks[b]);
            free(cache->heads[h].value_blocks[b]);
        }
        free(cache->heads[h].blocks);
        free(cache->heads[h].value_blocks);
        free(cache->heads[h].block_types);
        free(cache->heads[h].ref_counts);
    }
    free(cache->heads);
    free(cache);
}
