/**
 * TurboQuant.cpp -- llama.cpp KV cache backend integration
 *
 * This file provides the glue between TurboQuant's quantization types and
 * llama.cpp's GGML type system. It registers custom GGML type IDs for each
 * TurboQuant quantization format and provides from_float / to_float / vec_dot
 * wrappers that delegate to the TurboQuant C API.
 *
 * Build: compile alongside llama.cpp, linking against libturboquant.
 *   g++ -std=c++17 -c tq_kv_cache.cpp -I../../include
 *
 * Usage in llama.cpp:
 *   1. Call tq_ggml_register_types() once at startup.
 *   2. Use --kv-cache-type turbo3 (or turbo4, polar3, qjl1) on CLI.
 */

#include "tq_ggml_type.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

/* ============================================================
 * Hypothetical GGML type IDs for TurboQuant types.
 * In a real integration these would be added to ggml.h's
 * ggml_type enum. We define them as offsets from a base ID
 * that sits above the last upstream GGML type.
 * ============================================================ */

#define GGML_TYPE_TQ_BASE      256

enum {
    GGML_TYPE_TQ_POLAR_3B  = GGML_TYPE_TQ_BASE + 0,
    GGML_TYPE_TQ_POLAR_4B  = GGML_TYPE_TQ_BASE + 1,
    GGML_TYPE_TQ_QJL_1B    = GGML_TYPE_TQ_BASE + 2,
    GGML_TYPE_TQ_TURBO_3B  = GGML_TYPE_TQ_BASE + 3,
    GGML_TYPE_TQ_TURBO_4B  = GGML_TYPE_TQ_BASE + 4,
    GGML_TYPE_TQ_UNIFORM_4B= GGML_TYPE_TQ_BASE + 5,
    GGML_TYPE_TQ_UNIFORM_2B= GGML_TYPE_TQ_BASE + 6,
    GGML_TYPE_TQ_COUNT     = 7,
};

/* ============================================================
 * GGML type-trait function wrappers
 *
 * GGML expects three core callbacks per quantized type:
 *   from_float(src_fp32, dst_quant, n_elements)
 *   to_float  (src_quant, dst_fp32, n_elements)
 *   vec_dot   (n, result, src_quant, src_fp32)
 *
 * We map these to TurboQuant's quantize / dequantize / attention
 * functions via the TQ_TRAITS dispatch table.
 * ============================================================ */

/* --- from_float wrappers (FP32 -> quantized) --- */

static void tq_ggml_from_float_polar_3b(const float* src, void* dst, int64_t n) {
    tq_quantize_fn qfn = TQ_TRAITS[TQ_TYPE_POLAR_3B].quantize;
    if (qfn) qfn(src, dst, (int)n);
}

static void tq_ggml_from_float_polar_4b(const float* src, void* dst, int64_t n) {
    tq_quantize_fn qfn = TQ_TRAITS[TQ_TYPE_POLAR_4B].quantize;
    if (qfn) qfn(src, dst, (int)n);
}

static void tq_ggml_from_float_qjl_1b(const float* src, void* dst, int64_t n) {
    tq_quantize_fn qfn = TQ_TRAITS[TQ_TYPE_QJL_1B].quantize;
    if (qfn) qfn(src, dst, (int)n);
}

static void tq_ggml_from_float_turbo_3b(const float* src, void* dst, int64_t n) {
    tq_quantize_fn qfn = TQ_TRAITS[TQ_TYPE_TURBO_3B].quantize;
    if (qfn) qfn(src, dst, (int)n);
}

static void tq_ggml_from_float_turbo_4b(const float* src, void* dst, int64_t n) {
    tq_quantize_fn qfn = TQ_TRAITS[TQ_TYPE_TURBO_4B].quantize;
    if (qfn) qfn(src, dst, (int)n);
}

static void tq_ggml_from_float_uniform_4b(const float* src, void* dst, int64_t n) {
    tq_quantize_fn qfn = TQ_TRAITS[TQ_TYPE_UNIFORM_4B].quantize;
    if (qfn) qfn(src, dst, (int)n);
}

static void tq_ggml_from_float_uniform_2b(const float* src, void* dst, int64_t n) {
    tq_quantize_fn qfn = TQ_TRAITS[TQ_TYPE_UNIFORM_2B].quantize;
    if (qfn) qfn(src, dst, (int)n);
}

/* --- to_float wrappers (quantized -> FP32) --- */

static void tq_ggml_to_float_polar_3b(const void* src, float* dst, int64_t n) {
    tq_dequantize_fn dfn = TQ_TRAITS[TQ_TYPE_POLAR_3B].dequantize;
    if (dfn) dfn(src, dst, (int)n);
}

static void tq_ggml_to_float_polar_4b(const void* src, float* dst, int64_t n) {
    tq_dequantize_fn dfn = TQ_TRAITS[TQ_TYPE_POLAR_4B].dequantize;
    if (dfn) dfn(src, dst, (int)n);
}

static void tq_ggml_to_float_qjl_1b(const void* src, float* dst, int64_t n) {
    tq_dequantize_fn dfn = TQ_TRAITS[TQ_TYPE_QJL_1B].dequantize;
    if (dfn) dfn(src, dst, (int)n);
}

static void tq_ggml_to_float_turbo_3b(const void* src, float* dst, int64_t n) {
    tq_dequantize_fn dfn = TQ_TRAITS[TQ_TYPE_TURBO_3B].dequantize;
    if (dfn) dfn(src, dst, (int)n);
}

static void tq_ggml_to_float_turbo_4b(const void* src, float* dst, int64_t n) {
    tq_dequantize_fn dfn = TQ_TRAITS[TQ_TYPE_TURBO_4B].dequantize;
    if (dfn) dfn(src, dst, (int)n);
}

static void tq_ggml_to_float_uniform_4b(const void* src, float* dst, int64_t n) {
    tq_dequantize_fn dfn = TQ_TRAITS[TQ_TYPE_UNIFORM_4B].dequantize;
    if (dfn) dfn(src, dst, (int)n);
}

static void tq_ggml_to_float_uniform_2b(const void* src, float* dst, int64_t n) {
    tq_dequantize_fn dfn = TQ_TRAITS[TQ_TYPE_UNIFORM_2B].dequantize;
    if (dfn) dfn(src, dst, (int)n);
}

/* --- vec_dot wrappers (quantized key . FP32 query -> scalar) --- */
/* GGML vec_dot signature: void vec_dot(int n, float* s, const void* x, const float* y)
 * We compute a single dot product by dequantizing x and dotting with y. */

static void tq_ggml_vec_dot_generic(tq_type type, int n, float* result,
                                     const void* x, const float* y) {
    tq_dequantize_fn dfn = TQ_TRAITS[type].dequantize;
    if (!dfn) {
        *result = 0.0f;
        return;
    }
    /* Stack-allocate temp buffer for typical head_dim sizes (up to 512) */
    float tmp[512];
    float* buf = (n <= 512) ? tmp : (float*)malloc((size_t)n * sizeof(float));
    if (!buf) { *result = 0.0f; return; }

    dfn(x, buf, n);

    float dot = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += buf[i] * y[i];
    }
    *result = dot;

    if (buf != tmp) free(buf);
}

static void tq_ggml_vec_dot_polar_3b(int n, float* s, const void* x, const float* y) {
    tq_ggml_vec_dot_generic(TQ_TYPE_POLAR_3B, n, s, x, y);
}

static void tq_ggml_vec_dot_polar_4b(int n, float* s, const void* x, const float* y) {
    tq_ggml_vec_dot_generic(TQ_TYPE_POLAR_4B, n, s, x, y);
}

static void tq_ggml_vec_dot_qjl_1b(int n, float* s, const void* x, const float* y) {
    tq_ggml_vec_dot_generic(TQ_TYPE_QJL_1B, n, s, x, y);
}

static void tq_ggml_vec_dot_turbo_3b(int n, float* s, const void* x, const float* y) {
    tq_ggml_vec_dot_generic(TQ_TYPE_TURBO_3B, n, s, x, y);
}

static void tq_ggml_vec_dot_turbo_4b(int n, float* s, const void* x, const float* y) {
    tq_ggml_vec_dot_generic(TQ_TYPE_TURBO_4B, n, s, x, y);
}

static void tq_ggml_vec_dot_uniform_4b(int n, float* s, const void* x, const float* y) {
    tq_ggml_vec_dot_generic(TQ_TYPE_UNIFORM_4B, n, s, x, y);
}

static void tq_ggml_vec_dot_uniform_2b(int n, float* s, const void* x, const float* y) {
    tq_ggml_vec_dot_generic(TQ_TYPE_UNIFORM_2B, n, s, x, y);
}

/* ============================================================
 * GGML type trait table for TurboQuant types
 *
 * In a real llama.cpp integration this struct would match
 * ggml_type_traits_t. We define a compatible struct here.
 * ============================================================ */

typedef void (*ggml_from_float_fn)(const float* src, void* dst, int64_t n);
typedef void (*ggml_to_float_fn)(const void* src, float* dst, int64_t n);
typedef void (*ggml_vec_dot_fn)(int n, float* s, const void* x, const float* y);

struct tq_ggml_type_trait {
    const char*       type_name;
    int               ggml_type_id;
    size_t            type_size;       /* bytes per block */
    size_t            block_size;      /* elements per block */
    float             bpe;             /* bits per element */
    ggml_from_float_fn from_float;
    ggml_to_float_fn   to_float;
    ggml_vec_dot_fn    vec_dot;
};

static const tq_ggml_type_trait TQ_GGML_TRAITS[GGML_TYPE_TQ_COUNT] = {
    {
        "tq_polar_3b", GGML_TYPE_TQ_POLAR_3B,
        sizeof(block_tq_polar), TQ_BK, 4.5f,
        tq_ggml_from_float_polar_3b,
        tq_ggml_to_float_polar_3b,
        tq_ggml_vec_dot_polar_3b,
    },
    {
        "tq_polar_4b", GGML_TYPE_TQ_POLAR_4B,
        sizeof(block_tq_polar), TQ_BK, 4.5f,
        tq_ggml_from_float_polar_4b,
        tq_ggml_to_float_polar_4b,
        tq_ggml_vec_dot_polar_4b,
    },
    {
        "tq_qjl_1b", GGML_TYPE_TQ_QJL_1B,
        sizeof(block_tq_qjl), TQ_BK_QJL, 1.25f,
        tq_ggml_from_float_qjl_1b,
        tq_ggml_to_float_qjl_1b,
        tq_ggml_vec_dot_qjl_1b,
    },
    {
        "tq_turbo_3b", GGML_TYPE_TQ_TURBO_3B,
        sizeof(block_tq_turbo), TQ_BK, 5.75f,
        tq_ggml_from_float_turbo_3b,
        tq_ggml_to_float_turbo_3b,
        tq_ggml_vec_dot_turbo_3b,
    },
    {
        "tq_turbo_4b", GGML_TYPE_TQ_TURBO_4B,
        sizeof(block_tq_turbo), TQ_BK, 5.75f,
        tq_ggml_from_float_turbo_4b,
        tq_ggml_to_float_turbo_4b,
        tq_ggml_vec_dot_turbo_4b,
    },
    {
        "tq_uniform_4b", GGML_TYPE_TQ_UNIFORM_4B,
        sizeof(block_tq_uniform_4b), TQ_BK, 4.25f,
        tq_ggml_from_float_uniform_4b,
        tq_ggml_to_float_uniform_4b,
        tq_ggml_vec_dot_uniform_4b,
    },
    {
        "tq_uniform_2b", GGML_TYPE_TQ_UNIFORM_2B,
        sizeof(block_tq_uniform_2b), TQ_BK, 2.25f,
        tq_ggml_from_float_uniform_2b,
        tq_ggml_to_float_uniform_2b,
        tq_ggml_vec_dot_uniform_2b,
    },
};

/* ============================================================
 * Registration
 * ============================================================ */

static int g_tq_registered = 0;

tq_status tq_ggml_register_types(void) {
    if (g_tq_registered) return TQ_OK;

    /*
     * In a real llama.cpp integration, this function would call:
     *
     *   ggml_register_custom_type(GGML_TYPE_TQ_POLAR_3B,
     *       TQ_GGML_TRAITS[0].type_name,
     *       TQ_GGML_TRAITS[0].type_size,
     *       TQ_GGML_TRAITS[0].block_size,
     *       TQ_GGML_TRAITS[0].from_float,
     *       TQ_GGML_TRAITS[0].to_float,
     *       TQ_GGML_TRAITS[0].vec_dot);
     *
     * for each type in TQ_GGML_TRAITS.
     *
     * Since ggml_register_custom_type() is not available without
     * linking against llama.cpp, we validate the trait table here
     * and mark registration as complete.
     */

    for (int i = 0; i < GGML_TYPE_TQ_COUNT; i++) {
        const tq_ggml_type_trait* t = &TQ_GGML_TRAITS[i];
        if (!t->type_name || !t->from_float || !t->to_float || !t->vec_dot) {
            fprintf(stderr, "tq_ggml_register_types: trait %d incomplete\n", i);
            return TQ_ERR_NOT_IMPL;
        }
        if (t->type_size == 0 || t->block_size == 0) {
            fprintf(stderr, "tq_ggml_register_types: trait %d has zero size\n", i);
            return TQ_ERR_INVALID_DIM;
        }
    }

    g_tq_registered = 1;
    fprintf(stderr, "[TurboQuant] Registered %d GGML types (IDs %d..%d)\n",
            GGML_TYPE_TQ_COUNT, GGML_TYPE_TQ_BASE,
            GGML_TYPE_TQ_BASE + GGML_TYPE_TQ_COUNT - 1);

    return TQ_OK;
}

/* ============================================================
 * CLI option parsing helper
 *
 * Parses --kv-cache-type argument and returns the corresponding
 * TurboQuant type. Returns TQ_TYPE_COUNT on unrecognized input.
 * ============================================================ */

tq_type tq_parse_kv_cache_type(const char* arg) {
    if (!arg) return TQ_TYPE_COUNT;

    struct { const char* name; tq_type type; } map[] = {
        { "turbo3",    TQ_TYPE_TURBO_3B   },
        { "turbo_3b",  TQ_TYPE_TURBO_3B   },
        { "turbo4",    TQ_TYPE_TURBO_4B   },
        { "turbo_4b",  TQ_TYPE_TURBO_4B   },
        { "polar3",    TQ_TYPE_POLAR_3B   },
        { "polar_3b",  TQ_TYPE_POLAR_3B   },
        { "polar4",    TQ_TYPE_POLAR_4B   },
        { "polar_4b",  TQ_TYPE_POLAR_4B   },
        { "qjl1",      TQ_TYPE_QJL_1B     },
        { "qjl_1b",    TQ_TYPE_QJL_1B     },
        { "uniform4",  TQ_TYPE_UNIFORM_4B },
        { "uniform_4b",TQ_TYPE_UNIFORM_4B },
        { "uniform2",  TQ_TYPE_UNIFORM_2B },
        { "uniform_2b",TQ_TYPE_UNIFORM_2B },
    };

    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (strcmp(arg, map[i].name) == 0) {
            return map[i].type;
        }
    }

    return TQ_TYPE_COUNT;
}

/* ============================================================
 * KV cache integration helpers
 *
 * These functions wrap TurboQuant's context-based API for use
 * within llama.cpp's KV cache management code.
 * ============================================================ */

/**
 * Create a TurboQuant context suitable for llama.cpp integration.
 * Caller must free with tq_free().
 */
tq_context_t* tq_llamacpp_create_context(void) {
    tq_context_t* ctx = nullptr;
    tq_status status = tq_init(&ctx, TQ_BACKEND_AUTO);
    if (status != TQ_OK) {
        fprintf(stderr, "[TurboQuant] Failed to create context: %s\n",
                tq_status_string(status));
        return nullptr;
    }
    return ctx;
}

/**
 * Quantize a batch of key vectors into the KV cache.
 *
 * @param ctx        TurboQuant context
 * @param keys       Input FP32 keys [n_tokens x head_dim]
 * @param n_tokens   Number of tokens to quantize
 * @param head_dim   Dimension per attention head
 * @param type       TurboQuant quantization type
 * @param out        Output buffer (pre-allocated by llama.cpp cache manager)
 * @param out_bytes  Size of output buffer
 * @return           TQ_OK on success
 */
tq_status tq_llamacpp_quantize_keys(tq_context_t* ctx,
                                     const float* keys,
                                     int n_tokens, int head_dim,
                                     tq_type type,
                                     void* out, size_t out_bytes) {
    return tq_quantize_keys(ctx, keys, n_tokens, head_dim, type, out, out_bytes);
}

/**
 * Quantize a batch of value vectors into the KV cache.
 *
 * @param ctx        TurboQuant context
 * @param values     Input FP32 values [n_tokens x head_dim]
 * @param n_tokens   Number of tokens
 * @param head_dim   Dimension per head
 * @param bits       Quantization bits (2 or 4)
 * @param out        Output buffer
 * @param out_bytes  Buffer size
 * @return           TQ_OK on success
 */
tq_status tq_llamacpp_quantize_values(tq_context_t* ctx,
                                       const float* values,
                                       int n_tokens, int head_dim,
                                       int bits,
                                       void* out, size_t out_bytes) {
    return tq_quantize_values(ctx, values, n_tokens, head_dim, bits, out, out_bytes);
}

/**
 * Compute attention scores from quantized KV cache.
 *
 * @param ctx       TurboQuant context
 * @param query     Query vector [head_dim]
 * @param kv_cache  Quantized key cache
 * @param seq_len   Sequence length (cached tokens)
 * @param head_dim  Head dimension
 * @param type      Quantization type
 * @param scores    Output scores [seq_len]
 * @return          TQ_OK on success
 */
tq_status tq_llamacpp_attention(tq_context_t* ctx,
                                 const float* query,
                                 const void* kv_cache,
                                 int seq_len, int head_dim,
                                 tq_type type,
                                 float* scores) {
    return tq_attention(ctx, query, kv_cache, seq_len, head_dim, type, scores);
}

/**
 * Compute the quantized KV cache memory per token for a given type.
 * Useful for llama.cpp's memory estimation in --kv-cache-type mode.
 *
 * @param head_dim   Dimension per head
 * @param key_type   TurboQuant key quantization type
 * @param value_bits Value quantization bits (2 or 4), 0 for no value quant
 * @return           Bytes per token per head (key + value)
 */
size_t tq_llamacpp_bytes_per_token(int head_dim, tq_type key_type, int value_bits) {
    size_t key_bytes = tq_quantize_keys_size(1, head_dim, key_type);
    size_t val_bytes = 0;
    if (value_bits > 0) {
        val_bytes = tq_quantize_values_size(1, head_dim, value_bits);
    } else {
        /* Default: store values as FP16 (2 bytes per element) */
        val_bytes = (size_t)head_dim * 2;
    }
    return key_bytes + val_bytes;
}

/**
 * Print a summary of memory savings for a given configuration.
 * Useful for llama.cpp's startup info output.
 */
void tq_llamacpp_print_config(tq_type key_type, int value_bits,
                               int n_heads, int head_dim, int max_seq_len) {
    size_t fp16_per_token = (size_t)head_dim * 2 * 2;  /* key + value, FP16 */
    size_t tq_per_token = tq_llamacpp_bytes_per_token(head_dim, key_type, value_bits);
    double ratio = (double)fp16_per_token / (double)tq_per_token;

    size_t fp16_total = fp16_per_token * (size_t)n_heads * (size_t)max_seq_len;
    size_t tq_total   = tq_per_token   * (size_t)n_heads * (size_t)max_seq_len;

    fprintf(stderr, "[TurboQuant] KV cache config:\n");
    fprintf(stderr, "  Key type:        %s\n", tq_type_name(key_type));
    fprintf(stderr, "  Value bits:      %d\n", value_bits > 0 ? value_bits : 16);
    fprintf(stderr, "  Heads:           %d x %d\n", n_heads, head_dim);
    fprintf(stderr, "  Max seq length:  %d\n", max_seq_len);
    fprintf(stderr, "  FP16 KV memory:  %.1f MB\n", (double)fp16_total / (1024.0 * 1024.0));
    fprintf(stderr, "  TQ KV memory:    %.1f MB\n", (double)tq_total / (1024.0 * 1024.0));
    fprintf(stderr, "  Compression:     %.2fx\n", ratio);
}
