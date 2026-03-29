/**
 * TurboQuant -- QJL Metal compute shaders
 *
 * 1-bit sign hash quantization using random projections,
 * and Hamming distance-based attention scoring on Apple GPU.
 * Uses Metal popcount builtin for efficient bit counting.
 */
#include <metal_stdlib>
#include <metal_math>
using namespace metal;

/* ============================================================
 * Constants
 * ============================================================ */

constant int TQ_SKETCH_DIM = 256;
constant int TQ_HASH_BYTES = 32;    /* TQ_SKETCH_DIM / 8 */
constant int TQ_OUTLIERS   = 4;
constant int TQ_EMB_MAX    = 256;

/* ============================================================
 * Block structure
 * ============================================================ */

struct block_tq_qjl {
    ushort norm;
    ushort outlier_norm;
    uchar  hash[32];       /* TQ_SKETCH_DIM / 8 */
    uchar  outlier_idx[4]; /* TQ_OUTLIERS */
};

/* ============================================================
 * FP16 helpers
 * ============================================================ */

inline float fp16_to_float(ushort h) {
    return as_type<half>(h);
}

inline ushort float_to_fp16(float f) {
    return as_type<ushort>(half(f));
}

/* ============================================================
 * Deterministic pseudo-random projection (Rademacher +1/-1)
 * Must match the CPU reference implementation exactly.
 * ============================================================ */

inline float random_entry(int dim_idx, int sketch_idx) {
    uint h = uint(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h & 1u) ? 1.0f : -1.0f;
}

/* ============================================================
 * SIMD-group reductions
 * ============================================================ */

inline float simd_reduce_sum(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

inline int simd_reduce_sum_int(int val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

inline float threadgroup_reduce_sum_f(
    float val,
    uint tid,
    uint simd_lane,
    uint simd_group_id,
    uint num_simd_groups,
    threadgroup float* scratch)
{
    val = simd_reduce_sum(val);
    if (simd_lane == 0) scratch[simd_group_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < num_simd_groups) val = scratch[tid];
    else val = 0.0f;
    if (simd_group_id == 0) val = simd_reduce_sum(val);
    return val;
}

/* ============================================================
 * QJL quantize kernel
 *
 * Threadgroup: (32, 8, 1) -- 32 lanes x 8 SIMD groups
 * Grid:       (num_keys, sketch_chunks, 1)
 *
 * Each SIMD group computes 8 sketch projections (1 byte of hash).
 * The 32 lanes cooperatively reduce the dot product over emb_dim.
 * ============================================================ */
kernel void tq_qjl_quantize(
    device const float*      keys       [[buffer(0)]],
    device block_tq_qjl*     output     [[buffer(1)]],
    constant uint&           num_keys   [[buffer(2)]],
    constant uint&           emb_dim    [[buffer(3)]],
    uint2 tgid   [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sgid   [[simdgroup_index_in_threadgroup]])
{
    uint key_idx     = tgid.x;
    uint sketch_base = tgid.y * 8 * 8; /* 8 SIMD groups * 8 bits per group */

    if (key_idx >= num_keys) return;

    /* Load key into threadgroup memory */
    threadgroup float tg_key[TQ_EMB_MAX];
    uint tid_flat = sgid * 32 + lane;
    uint stride = 8 * 32;
    for (uint i = tid_flat; i < emb_dim; i += stride) {
        tg_key[i] = keys[key_idx * emb_dim + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Each SIMD group handles 8 sketch dimensions (one byte) */
    uint sketch_group_base = sketch_base + sgid * 8;
    uchar packed_byte = 0;

    for (int bit = 0; bit < 8; bit++) {
        uint sketch_idx = sketch_group_base + uint(bit);
        if (sketch_idx >= uint(TQ_SKETCH_DIM)) break;

        /* Partial dot product across lanes */
        float partial = 0.0f;
        for (uint d = lane; d < emb_dim; d += 32) {
            partial += tg_key[d] * random_entry(int(d), int(sketch_idx));
        }

        float dot = simd_reduce_sum(partial);
        if (lane == 0 && dot >= 0.0f) {
            packed_byte |= (1u << bit);
        }
    }

    /* Lane 0 of each SIMD group stores its byte */
    if (lane == 0 && sketch_group_base < uint(TQ_SKETCH_DIM)) {
        uint byte_idx = sketch_group_base / 8;
        output[key_idx].hash[byte_idx] = packed_byte;
    }

    /* First threadgroup chunk computes norm and outliers */
    if (tgid.y == 0 && sgid == 0) {
        float norm_partial = 0.0f;
        for (uint d = lane; d < emb_dim; d += 32) {
            float v = tg_key[d];
            norm_partial += v * v;
        }
        float norm_sq = simd_reduce_sum(norm_partial);

        if (lane == 0) {
            output[key_idx].norm = float_to_fp16(sqrt(norm_sq));

            /* Outlier detection (serial, only 4 outliers) */
            float abs_max[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
            uchar max_idx[4] = {0, 0, 0, 0};
            float abs_vals[TQ_EMB_MAX];

            for (uint d = 0; d < emb_dim && d < uint(TQ_EMB_MAX); d++) {
                abs_vals[d] = abs(tg_key[d]);
            }

            for (int o = 0; o < TQ_OUTLIERS; o++) {
                int best = 0; float best_val = -1.0f;
                for (uint d = 0; d < emb_dim && d < uint(TQ_EMB_MAX); d++) {
                    if (abs_vals[d] > best_val) {
                        best_val = abs_vals[d];
                        best = int(d);
                    }
                }
                max_idx[o] = uchar(best < 256 ? best : 255);
                abs_vals[best] = -1.0f;
                abs_max[o] = best_val;
            }

            float outlier_norm_sq = 0.0f;
            for (int o = 0; o < TQ_OUTLIERS; o++) {
                output[key_idx].outlier_idx[o] = max_idx[o];
                float v = tg_key[max_idx[o]];
                outlier_norm_sq += v * v;
            }
            output[key_idx].outlier_norm = float_to_fp16(sqrt(outlier_norm_sq));
        }
    }
}

/* ============================================================
 * QJL attention kernel (Hamming distance)
 *
 * Threadgroup: (32, 1, 1) -- one SIMD group per token
 * Grid:       (seq_len, 1, 1)
 *
 * Each threadgroup:
 *  1. Loads query into threadgroup memory
 *  2. Computes query projections for each sketch byte
 *  3. XOR with key hash, popcount to count disagreements
 *  4. Converts to estimated cosine similarity
 * ============================================================ */
kernel void tq_qjl_attention(
    device const float*         query    [[buffer(0)]],
    device const block_tq_qjl*  keys     [[buffer(1)]],
    device float*               scores   [[buffer(2)]],
    constant uint&              seq_len  [[buffer(3)]],
    constant uint&              head_dim [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint s = tgid;
    if (s >= seq_len) return;

    /* Load query into threadgroup memory */
    threadgroup float tg_query[TQ_EMB_MAX];
    for (uint d = lane; d < head_dim; d += 32) {
        tg_query[d] = query[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Each lane handles one or more hash bytes */
    int agree_count = 0;
    for (uint b = lane; b < uint(TQ_HASH_BYTES); b += 32) {
        uchar q_hash = 0;
        for (int bit = 0; bit < 8; bit++) {
            uint sketch_idx = b * 8 + uint(bit);
            if (sketch_idx >= uint(TQ_SKETCH_DIM)) break;

            float proj = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                proj += tg_query[d] * random_entry(int(d), int(sketch_idx));
            }
            if (proj >= 0.0f) {
                q_hash |= (1u << bit);
            }
        }

        uchar key_hash = keys[s].hash[b];
        uchar xored = q_hash ^ key_hash;
        int disagree = popcount(uint(xored));
        agree_count += (8 - disagree);
    }

    /* SIMD-group reduction of agreement count */
    agree_count = simd_reduce_sum_int(agree_count);

    if (lane == 0) {
        float key_norm = fp16_to_float(keys[s].norm);

        float q_norm_sq = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            q_norm_sq += tg_query[d] * tg_query[d];
        }
        float q_norm = sqrt(q_norm_sq);

        float frac = float(agree_count) / float(TQ_SKETCH_DIM);
        float cos_est = cos(M_PI_F * (1.0f - frac));

        scores[s] = cos_est * q_norm * key_norm;
    }
}
