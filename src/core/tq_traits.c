#include "turboquant/turboquant.h"
#include <string.h>

/* Forward declarations — defined in individual algorithm files */
extern void tq_polar_quantize_ref(const float* src, void* dst, int n);
extern void tq_polar_dequantize_ref(const void* src, float* dst, int n);
extern void tq_polar_attention_ref(const float* query, const void* kv,
                                   float* scores, int seq_len, int head_dim);

extern void tq_qjl_quantize_ref(const float* src, void* dst, int n);
extern void tq_qjl_dequantize_ref(const void* src, float* dst, int n);
extern void tq_qjl_attention_ref(const float* query, const void* kv,
                                 float* scores, int seq_len, int head_dim);

extern void tq_turbo_quantize_ref(const float* src, void* dst, int n);
extern void tq_turbo_dequantize_ref(const void* src, float* dst, int n);
extern void tq_turbo_attention_ref(const float* query, const void* kv,
                                   float* scores, int seq_len, int head_dim);

extern void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_4b_attention_ref(const float* query, const void* kv,
                                         float* scores, int seq_len, int head_dim);
extern void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_2b_attention_ref(const float* query, const void* kv,
                                         float* scores, int seq_len, int head_dim);

const tq_type_traits_t TQ_TRAITS[TQ_TYPE_COUNT] = {
    [TQ_TYPE_POLAR_3B] = {
        .name       = "polar_3b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_polar),
        .bpe        = 4.5f,
        .quantize   = tq_polar_quantize_ref,
        .dequantize = tq_polar_dequantize_ref,
        .attention  = tq_polar_attention_ref,
        .residual_type = TQ_TYPE_COUNT, /* none */
    },
    [TQ_TYPE_POLAR_4B] = {
        .name       = "polar_4b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_polar),
        .bpe        = 4.5f,
        .quantize   = tq_polar_quantize_ref,
        .dequantize = tq_polar_dequantize_ref,
        .attention  = tq_polar_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_QJL_1B] = {
        .name       = "qjl_1b",
        .block_size = TQ_BK_QJL,
        .type_size  = sizeof(block_tq_qjl),
        .bpe        = 1.25f,
        .quantize   = tq_qjl_quantize_ref,
        .dequantize = tq_qjl_dequantize_ref,
        .attention  = tq_qjl_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_TURBO_3B] = {
        .name       = "turbo_3b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo),
        .bpe        = 5.75f,
        .quantize   = tq_turbo_quantize_ref,
        .dequantize = tq_turbo_dequantize_ref,
        .attention  = tq_turbo_attention_ref,
        .residual_type = TQ_TYPE_QJL_1B,
    },
    [TQ_TYPE_TURBO_4B] = {
        .name       = "turbo_4b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_turbo),
        .bpe        = 5.75f,
        .quantize   = tq_turbo_quantize_ref,
        .dequantize = tq_turbo_dequantize_ref,
        .attention  = tq_turbo_attention_ref,
        .residual_type = TQ_TYPE_QJL_1B,
    },
    [TQ_TYPE_UNIFORM_4B] = {
        .name       = "uniform_4b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_uniform_4b),
        .bpe        = 4.25f,
        .quantize   = tq_uniform_4b_quantize_ref,
        .dequantize = tq_uniform_4b_dequantize_ref,
        .attention  = tq_uniform_4b_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
    [TQ_TYPE_UNIFORM_2B] = {
        .name       = "uniform_2b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_uniform_2b),
        .bpe        = 2.25f,
        .quantize   = tq_uniform_2b_quantize_ref,
        .dequantize = tq_uniform_2b_dequantize_ref,
        .attention  = tq_uniform_2b_attention_ref,
        .residual_type = TQ_TYPE_COUNT,
    },
};

const char* tq_type_name(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return "unknown";
    return TQ_TRAITS[type].name;
}

float tq_type_bpe(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0.0f;
    return TQ_TRAITS[type].bpe;
}

size_t tq_type_block_size(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0;
    return TQ_TRAITS[type].block_size;
}

size_t tq_type_type_size(tq_type type) {
    if (type < 0 || type >= TQ_TYPE_COUNT) return 0;
    return TQ_TRAITS[type].type_size;
}

const char* tq_status_string(tq_status status) {
    switch (status) {
        case TQ_OK:               return "OK";
        case TQ_ERR_NULL_PTR:     return "null pointer";
        case TQ_ERR_INVALID_TYPE: return "invalid type";
        case TQ_ERR_INVALID_DIM:  return "invalid dimension";
        case TQ_ERR_OUT_OF_MEM:   return "out of memory";
        case TQ_ERR_NOT_IMPL:     return "not implemented";
        case TQ_ERR_BACKEND:      return "backend error";
        default:                  return "unknown error";
    }
}

tq_format_spec_t tq_get_format_spec(tq_type type) {
    tq_format_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.spec_version = TQ_SPEC_VERSION;
    spec.block_size = (uint16_t)TQ_TRAITS[type].block_size;
    switch (type) {
        case TQ_TYPE_POLAR_3B:
            spec.algorithm = TQ_ALG_POLAR; spec.key_bits = 3; break;
        case TQ_TYPE_POLAR_4B:
            spec.algorithm = TQ_ALG_POLAR; spec.key_bits = 4; break;
        case TQ_TYPE_QJL_1B:
            spec.algorithm = TQ_ALG_QJL; spec.key_bits = 1;
            spec.sketch_dim = TQ_SKETCH_DIM; spec.outlier_count = TQ_OUTLIERS; break;
        case TQ_TYPE_TURBO_3B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 3;
            spec.sketch_dim = TQ_SKETCH_DIM; spec.outlier_count = TQ_OUTLIERS;
            spec.flags = TQ_FLAG_HAS_RESIDUAL; break;
        case TQ_TYPE_TURBO_4B:
            spec.algorithm = TQ_ALG_TURBO; spec.key_bits = 4;
            spec.sketch_dim = TQ_SKETCH_DIM; spec.outlier_count = TQ_OUTLIERS;
            spec.flags = TQ_FLAG_HAS_RESIDUAL; break;
        case TQ_TYPE_UNIFORM_4B:
            spec.algorithm = TQ_ALG_UNIFORM; spec.key_bits = 4; break;
        case TQ_TYPE_UNIFORM_2B:
            spec.algorithm = TQ_ALG_UNIFORM; spec.key_bits = 2; break;
        default: break;
    }
    return spec;
}
