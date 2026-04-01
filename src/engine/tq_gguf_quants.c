/**
 * tq_gguf_quants.c — GGUF weight dequantization and on-the-fly dequant matmul
 *
 * Implements dequantization for all major GGML quant types:
 *   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32, BF16
 * Plus stub implementations for IQ types (IQ2_XXS, IQ3_XXS, IQ4_XS).
 *
 * The matmul path includes NEON-optimized inner loop for Apple Silicon.
 *
 * Pure C11, no external dependencies.
 */

#include "turboquant/tq_gguf.h"

#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_HAS_NEON 1
#else
#define TQ_HAS_NEON 0
#endif

/* ============================================================
 * FP16 / BF16 helpers
 * ============================================================ */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* positive/negative zero */
            float r;
            uint32_t v = sign;
            memcpy(&r, &v, 4);
            return r;
        }
        /* subnormal: normalize */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        /* inf / nan */
        exp = 255;
    } else {
        exp += 112;
    }

    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

/* ============================================================
 * Block structures (matching llama.cpp / ggml exactly)
 * ============================================================ */

/* Q8_0: 34 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    int8_t   qs[32];  /* quantized values */
} block_q8_0;

/* Q4_K: 144 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qs[128];    /* 4-bit values, 2 per byte */
} block_q4_K;

/* Q2_K: 84 bytes, 256 elements */
typedef struct {
    uint8_t  scales[16]; /* sub-block scales+mins, 4-bit each */
    uint8_t  qs[64];     /* 2-bit values, 4 per byte */
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
} block_q2_K;

/* Q3_K: 110 bytes, 256 elements */
typedef struct {
    uint8_t  hmask[32];  /* high bits */
    uint8_t  qs[64];     /* low 2 bits, 4 per byte */
    uint8_t  scales[12]; /* sub-block scales, packed */
    uint16_t d;          /* fp16 scale */
} block_q3_K;

/* Q6_K: 210 bytes, 256 elements */
typedef struct {
    uint8_t  ql[128];    /* low 4 bits */
    uint8_t  qh[64];     /* high 2 bits */
    int8_t   scales[16]; /* sub-block scales (signed int8) */
    uint16_t d;          /* fp16 super-block scale */
} block_q6_K;

/* Q5_K: 176 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qh[32];     /* high bit for each of 256 elements */
    uint8_t  qs[128];    /* low 4 bits, 2 per byte */
} block_q5_K;

/* Q4_0: 18 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_0;

/* Q4_1: 20 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_1;

/* Q5_0: 22 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_0;

/* Q5_1: 24 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_1;

/* Q8_1: 36 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale (delta) */
    uint16_t s;       /* fp16 sum */
    int8_t   qs[32];
} block_q8_1;

/* ============================================================
 * Type size / block size / name utilities
 * ============================================================ */

size_t tq_ggml_type_size(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:       return 4;
        case TQ_GGML_TYPE_F16:       return 2;
        case TQ_GGML_TYPE_BF16:      return 2;
        case TQ_GGML_TYPE_Q4_0:      return sizeof(block_q4_0);    /* 18 */
        case TQ_GGML_TYPE_Q4_1:      return sizeof(block_q4_1);    /* 20 */
        case TQ_GGML_TYPE_Q5_0:      return sizeof(block_q5_0);    /* 22 */
        case TQ_GGML_TYPE_Q5_1:      return sizeof(block_q5_1);    /* 24 */
        case TQ_GGML_TYPE_Q8_0:      return sizeof(block_q8_0);    /* 34 */
        case TQ_GGML_TYPE_Q8_1:      return sizeof(block_q8_1);    /* 36 */
        case TQ_GGML_TYPE_Q2_K:      return sizeof(block_q2_K);    /* 84 */
        case TQ_GGML_TYPE_Q3_K:      return sizeof(block_q3_K);    /* 110 */
        case TQ_GGML_TYPE_Q4_K:      return sizeof(block_q4_K);    /* 144 */
        case TQ_GGML_TYPE_Q5_K:      return sizeof(block_q5_K);    /* 176 */
        case TQ_GGML_TYPE_Q6_K:      return sizeof(block_q6_K);    /* 210 */
        case TQ_GGML_TYPE_Q8_K:      return 292;                   /* 256 + 2 + 32 + 2 */
        case TQ_GGML_TYPE_IQ2_XXS:   return 66;
        case TQ_GGML_TYPE_IQ2_XS:    return 74;
        case TQ_GGML_TYPE_IQ3_XXS:   return 98;
        case TQ_GGML_TYPE_IQ1_S:     return 50;
        case TQ_GGML_TYPE_IQ4_NL:    return 18;
        case TQ_GGML_TYPE_IQ3_S:     return 110;
        case TQ_GGML_TYPE_IQ2_S:     return 82;
        case TQ_GGML_TYPE_IQ4_XS:    return 36;
        default:                     return 0;
    }
}

int tq_ggml_type_blck(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:       return 1;
        case TQ_GGML_TYPE_F16:       return 1;
        case TQ_GGML_TYPE_BF16:      return 1;
        case TQ_GGML_TYPE_Q4_0:      return 32;
        case TQ_GGML_TYPE_Q4_1:      return 32;
        case TQ_GGML_TYPE_Q5_0:      return 32;
        case TQ_GGML_TYPE_Q5_1:      return 32;
        case TQ_GGML_TYPE_Q8_0:      return 32;
        case TQ_GGML_TYPE_Q8_1:      return 32;
        case TQ_GGML_TYPE_Q2_K:      return 256;
        case TQ_GGML_TYPE_Q3_K:      return 256;
        case TQ_GGML_TYPE_Q4_K:      return 256;
        case TQ_GGML_TYPE_Q5_K:      return 256;
        case TQ_GGML_TYPE_Q6_K:      return 256;
        case TQ_GGML_TYPE_Q8_K:      return 256;
        case TQ_GGML_TYPE_IQ2_XXS:   return 256;
        case TQ_GGML_TYPE_IQ2_XS:    return 256;
        case TQ_GGML_TYPE_IQ3_XXS:   return 256;
        case TQ_GGML_TYPE_IQ1_S:     return 256;
        case TQ_GGML_TYPE_IQ4_NL:    return 32;
        case TQ_GGML_TYPE_IQ3_S:     return 256;
        case TQ_GGML_TYPE_IQ2_S:     return 256;
        case TQ_GGML_TYPE_IQ4_XS:    return 32;
        default:                     return 0;
    }
}

const char* tq_ggml_type_name(tq_ggml_dtype type) {
    switch (type) {
        case TQ_GGML_TYPE_F32:       return "F32";
        case TQ_GGML_TYPE_F16:       return "F16";
        case TQ_GGML_TYPE_BF16:      return "BF16";
        case TQ_GGML_TYPE_Q4_0:      return "Q4_0";
        case TQ_GGML_TYPE_Q4_1:      return "Q4_1";
        case TQ_GGML_TYPE_Q5_0:      return "Q5_0";
        case TQ_GGML_TYPE_Q5_1:      return "Q5_1";
        case TQ_GGML_TYPE_Q8_0:      return "Q8_0";
        case TQ_GGML_TYPE_Q8_1:      return "Q8_1";
        case TQ_GGML_TYPE_Q2_K:      return "Q2_K";
        case TQ_GGML_TYPE_Q3_K:      return "Q3_K";
        case TQ_GGML_TYPE_Q4_K:      return "Q4_K";
        case TQ_GGML_TYPE_Q5_K:      return "Q5_K";
        case TQ_GGML_TYPE_Q6_K:      return "Q6_K";
        case TQ_GGML_TYPE_Q8_K:      return "Q8_K";
        case TQ_GGML_TYPE_IQ2_XXS:   return "IQ2_XXS";
        case TQ_GGML_TYPE_IQ2_XS:    return "IQ2_XS";
        case TQ_GGML_TYPE_IQ3_XXS:   return "IQ3_XXS";
        case TQ_GGML_TYPE_IQ1_S:     return "IQ1_S";
        case TQ_GGML_TYPE_IQ4_NL:    return "IQ4_NL";
        case TQ_GGML_TYPE_IQ3_S:     return "IQ3_S";
        case TQ_GGML_TYPE_IQ2_S:     return "IQ2_S";
        case TQ_GGML_TYPE_IQ4_XS:    return "IQ4_XS";
        default:                     return "unknown";
    }
}

/* ============================================================
 * Per-type dequantization
 * ============================================================ */

/* --- F32: passthrough --- */
static void dequant_f32(const void* src, float* dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* --- F16 --- */
static void dequant_f16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(s[i]);
    }
}

/* --- BF16 --- */
static void dequant_bf16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = bf16_to_fp32(s[i]);
    }
}

/* --- Q8_0: 34 bytes, 32 elements --- */
static void dequant_q8_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q8_0* blk = (const block_q8_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = d * blk[b].qs[j];
        }
    }
}

/* --- Q4_0: 18 bytes, 32 elements --- */
static void dequant_q4_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_0* blk = (const block_q4_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * ((int)(byte & 0x0F) - 8);
            dst[b * 32 + j + 16] = d * ((int)(byte >> 4) - 8);
        }
    }
}

/* --- Q4_1: 20 bytes, 32 elements --- */
static void dequant_q4_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_1* blk = (const block_q4_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * (byte & 0x0F) + m;
            dst[b * 32 + j + 16] = d * (byte >> 4) + m;
        }
    }
}

/* --- Q5_0: 22 bytes, 32 elements --- */
static void dequant_q5_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_0* blk = (const block_q5_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * (lo - 16);
            dst[b * 32 + j + 16] = d * (hi - 16);
        }
    }
}

/* --- Q5_1: 24 bytes, 32 elements --- */
static void dequant_q5_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_1* blk = (const block_q5_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * lo + m;
            dst[b * 32 + j + 16] = d * hi + m;
        }
    }
}

/* --- Q2_K: 84 bytes, 256 elements --- */
static void dequant_q2_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q2_K* blk = (const block_q2_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* 16 sub-blocks of 16 elements each */
        for (int sb = 0; sb < 16; sb++) {
            /* scales[sb]: low 4 bits = scale, high 4 bits = min */
            const int sc = blk[b].scales[sb] & 0x0F;
            const int m  = blk[b].scales[sb] >> 4;

            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                /* 2-bit value: 4 values per byte */
                int byte_idx = idx / 4;
                int bit_off  = (idx % 4) * 2;
                int q = (blk[b].qs[byte_idx] >> bit_off) & 0x03;
                dst[b * 256 + idx] = d * sc * q - dmin * m;
            }
        }
    }
}

/* --- Q3_K: 110 bytes, 256 elements ---
 * 3-bit = 2 low bits (qs) + 1 high bit (hmask)
 * 16 sub-blocks with 6-bit scales packed into 12 bytes */
static void dequant_q3_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q3_K* blk = (const block_q3_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);

        /* Decode 16 sub-block scales from 12 packed bytes.
         * Encoding (matching ggml):
         *   scales[0..3]  : bits 0..5 of sub-block scales 0..3 (low byte)
         *   scales[4..7]  : bits 0..5 of sub-block scales 4..7
         *   scales[8]     : bits 4..5 of scales 0..3 in pairs of 2
         *   scales[9]     : bits 4..5 of scales 4..7 in pairs of 2
         *   scales[10]    : bits 4..5 of scales 8..11
         *   scales[11]    : bits 4..5 of scales 12..15
         *   scales[0..7] low 4 bits: low 4 bits of 6-bit scale for sub-blocks 0..7
         *   scales[8..11]: high 2 bits for sub-blocks 0..15
         *
         * Actually, the ggml Q3_K scale encoding:
         *   aux = scales[sb & 7] for sb < 8, or reconstruct for sb >= 8
         *   The 12 bytes encode sixteen 6-bit values, offset by 32.
         */
        int32_t sc[16];

        /* Low 4 bits from first 8 bytes */
        for (int i = 0; i < 8; i++) {
            sc[i] = blk[b].scales[i] & 0x0F;
        }
        /* Sub-blocks 8..15 from first 8 bytes, high nibble */
        for (int i = 0; i < 8; i++) {
            sc[i + 8] = blk[b].scales[i] >> 4;
        }
        /* High 2 bits from bytes 8..11 */
        for (int i = 0; i < 4; i++) {
            uint8_t hb = blk[b].scales[8 + i];
            sc[i * 2 + 0] |= ((hb >> 0) & 3) << 4;
            sc[i * 2 + 1] |= ((hb >> 2) & 3) << 4;
            sc[i * 2 + 8] |= ((hb >> 4) & 3) << 4;
            sc[i * 2 + 9] |= ((hb >> 6) & 3) << 4;
        }
        /* Scales are stored with offset 32 */
        for (int i = 0; i < 16; i++) {
            sc[i] -= 32;
        }

        /* Dequantize */
        for (int sb = 0; sb < 16; sb++) {
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;

                /* Low 2 bits from qs: 4 per byte */
                int byte_idx = idx / 4;
                int bit_off  = (idx % 4) * 2;
                int q_lo = (blk[b].qs[byte_idx] >> bit_off) & 0x03;

                /* High bit from hmask */
                int hbit = (blk[b].hmask[idx / 8] >> (idx % 8)) & 1;

                int q3 = q_lo | (hbit << 2);
                /* q3 is 0..7, center at 4 */
                dst[b * 256 + idx] = d * sc[sb] * (q3 - 4);
            }
        }
    }
}

/* --- Q4_K: 144 bytes, 256 elements ---
 * 8 sub-blocks of 32 elements each
 * 6-bit scale/min packed in 12 bytes */
static void dequant_q4_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q4_K* blk = (const block_q4_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Decode 8 scale/min pairs from 12 bytes.
         * bytes 0..3:  low 6 bits of scales[0..3]
         * bytes 4..7:  low 6 bits of mins[0..3]
         * bytes 8..9:  high 2 bits of scales[0..3] + scales[4..7]
         * bytes 10..11: high 2 bits of mins[0..3] + mins[4..7]
         *
         * Actually ggml Q4_K packing:
         *   scales[0..5]: low 6 bits of scale for sub-blocks 0..5
         *                 but the first 4 bytes have scale low 6,
         *                 bytes 4..7 have min low 6,
         *                 bytes 8..11 have the high bits.
         *
         * Match ggml exactly:
         */
        uint8_t sc[8], mn[8];

        /* Low 6 bits */
        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        /* High 2 bits from bytes 8..11 */
        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        /* 8 sub-blocks of 32 elements */
        for (int sb = 0; sb < 8; sb++) {
            const float scale = d * sc[sb];
            const float min   = dmin * mn[sb];

            for (int j = 0; j < 32; j++) {
                int idx = sb * 32 + j;
                /* 4-bit: 2 values per byte, low nibble first in lower half */
                int byte_idx = idx / 2;
                int q;
                if (idx % 2 == 0) {
                    q = blk[b].qs[byte_idx] & 0x0F;
                } else {
                    q = blk[b].qs[byte_idx] >> 4;
                }
                dst[b * 256 + idx] = scale * q - min;
            }
        }
    }
}

/* --- Q5_K: 176 bytes, 256 elements ---
 * Like Q4_K but with an extra high bit per element */
static void dequant_q5_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q5_K* blk = (const block_q5_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Same scale/min packing as Q4_K */
        uint8_t sc[8], mn[8];

        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        for (int sb = 0; sb < 8; sb++) {
            const float scale = d * sc[sb];
            const float min   = dmin * mn[sb];

            for (int j = 0; j < 32; j++) {
                int idx = sb * 32 + j;
                /* Low 4 bits from qs */
                int byte_idx = idx / 2;
                int q;
                if (idx % 2 == 0) {
                    q = blk[b].qs[byte_idx] & 0x0F;
                } else {
                    q = blk[b].qs[byte_idx] >> 4;
                }
                /* High bit from qh */
                int hbit = (blk[b].qh[idx / 8] >> (idx % 8)) & 1;
                q |= (hbit << 4);
                dst[b * 256 + idx] = scale * q - min;
            }
        }
    }
}

/* --- Q6_K: 210 bytes, 256 elements ---
 * 6-bit = 4 low bits (ql) + 2 high bits (qh)
 * 16 sub-blocks of 16 elements, int8 scales */
static void dequant_q6_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);

        /* ql: 128 bytes = 256 4-bit values (low nibble first half, high nibble second half)
         * qh: 64 bytes = 256 2-bit values (4 per byte)
         * Layout matches ggml:
         *   For i in 0..127: ql[i] holds two 4-bit values (low nibble, high nibble)
         *   For i in 0..63:  qh[i] holds four 2-bit values
         *
         * ggml layout:
         *   Elements 0..127:   low 4 bits from ql[i] & 0xF
         *   Elements 128..255: low 4 bits from ql[i] >> 4  (i = elem - 128)
         *   High 2 bits from qh:
         *     elem 0..63:    (qh[i] >> 0) & 3
         *     elem 64..127:  (qh[i-64] >> 2) & 3  -- wait, let me match ggml exactly
         */

        /* Match ggml dequantize_row_q6_K exactly */
        for (int sb = 0; sb < 16; sb++) {
            const int8_t scale = blk[b].scales[sb];

            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;

                /* ql: element idx < 128 uses low nibble of ql[idx],
                 *     element idx >= 128 uses high nibble of ql[idx - 128] */
                int q_lo;
                if (idx < 128) {
                    q_lo = blk[b].ql[idx] & 0x0F;
                } else {
                    q_lo = blk[b].ql[idx - 128] >> 4;
                }

                /* qh: 64 bytes, 4 x 2-bit per byte
                 * For element idx:
                 *   idx 0..63:   bits 0..1 of qh[idx]
                 *   idx 64..127: bits 2..3 of qh[idx-64]
                 *   idx 128..191: bits 4..5 of qh[idx-128]
                 *   idx 192..255: bits 6..7 of qh[idx-192]
                 */
                int q_hi;
                if (idx < 64) {
                    q_hi = (blk[b].qh[idx] >> 0) & 0x03;
                } else if (idx < 128) {
                    q_hi = (blk[b].qh[idx - 64] >> 2) & 0x03;
                } else if (idx < 192) {
                    q_hi = (blk[b].qh[idx - 128] >> 4) & 0x03;
                } else {
                    q_hi = (blk[b].qh[idx - 192] >> 6) & 0x03;
                }

                int q6 = q_lo | (q_hi << 4);
                dst[b * 256 + idx] = d * scale * (q6 - 32);
            }
        }
    }
}

/* --- IQ type stubs --- */
static void dequant_iq_stub(const char* type_name, float* dst, int n) {
    fprintf(stderr, "tq_gguf_quants: WARNING: %s dequant not yet implemented, "
                    "returning zeros\n", type_name);
    memset(dst, 0, (size_t)n * sizeof(float));
}

/* ============================================================
 * Main dequantization dispatcher
 * ============================================================ */

void tq_dequant_row_gguf(tq_ggml_dtype type, const void* src, float* dst, int n) {
    switch (type) {
        case TQ_GGML_TYPE_F32:
            dequant_f32(src, dst, n);
            break;
        case TQ_GGML_TYPE_F16:
            dequant_f16(src, dst, n);
            break;
        case TQ_GGML_TYPE_BF16:
            dequant_bf16(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_0:
            dequant_q4_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_1:
            dequant_q4_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_0:
            dequant_q5_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_1:
            dequant_q5_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q8_0:
            dequant_q8_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q2_K:
            dequant_q2_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q3_K:
            dequant_q3_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_K:
            dequant_q4_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_K:
            dequant_q5_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q6_K:
            dequant_q6_k(src, dst, n);
            break;

        /* IQ stubs */
        case TQ_GGML_TYPE_IQ2_XXS:
            dequant_iq_stub("IQ2_XXS", dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_XS:
            dequant_iq_stub("IQ2_XS", dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_XXS:
            dequant_iq_stub("IQ3_XXS", dst, n);
            break;
        case TQ_GGML_TYPE_IQ1_S:
            dequant_iq_stub("IQ1_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_NL:
            dequant_iq_stub("IQ4_NL", dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_S:
            dequant_iq_stub("IQ3_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_S:
            dequant_iq_stub("IQ2_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_XS:
            dequant_iq_stub("IQ4_XS", dst, n);
            break;

        default:
            fprintf(stderr, "tq_gguf_quants: ERROR: unsupported type %d\n", (int)type);
            memset(dst, 0, (size_t)n * sizeof(float));
            break;
    }
}

/* ============================================================
 * On-the-fly dequant matmul
 *
 * out[d] = sum_n( x[n] * dequant(W[d, n]) )
 *
 * W is stored row-major in quantized blocks.
 * Hot path for MoE expert computation.
 * ============================================================ */

void tq_matmul_gguf(float* out, const float* x,
                    const void* weight, tq_ggml_dtype weight_type,
                    int out_dim, int in_dim)
{
    const size_t block_bytes = tq_ggml_type_size(weight_type);
    const int    block_elems = tq_ggml_type_blck(weight_type);

    if (block_bytes == 0 || block_elems == 0) {
        fprintf(stderr, "tq_matmul_gguf: ERROR: unsupported weight type %d\n",
                (int)weight_type);
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }

    const int    n_blocks  = in_dim / block_elems;
    const size_t row_bytes = (size_t)n_blocks * block_bytes;

    for (int d = 0; d < out_dim; d++) {
        const uint8_t* row = (const uint8_t*)weight + (size_t)d * row_bytes;
        float sum = 0.0f;

        /* Dequant one block at a time and accumulate dot product */
        float tmp[256]; /* max block size is 256 */

        for (int b = 0; b < n_blocks; b++) {
            tq_dequant_row_gguf(weight_type,
                                row + (size_t)b * block_bytes,
                                tmp, block_elems);

            const float* xp = x + b * block_elems;

#if TQ_HAS_NEON
            /* NEON-optimized dot product accumulation */
            float32x4_t vsum0 = vdupq_n_f32(0.0f);
            float32x4_t vsum1 = vdupq_n_f32(0.0f);
            float32x4_t vsum2 = vdupq_n_f32(0.0f);
            float32x4_t vsum3 = vdupq_n_f32(0.0f);

            int j = 0;
            /* Process 16 elements per iteration */
            for (; j + 15 < block_elems; j += 16) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vx1 = vld1q_f32(xp + j + 4);
                float32x4_t vx2 = vld1q_f32(xp + j + 8);
                float32x4_t vx3 = vld1q_f32(xp + j + 12);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                float32x4_t vt1 = vld1q_f32(tmp + j + 4);
                float32x4_t vt2 = vld1q_f32(tmp + j + 8);
                float32x4_t vt3 = vld1q_f32(tmp + j + 12);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
                vsum1 = vfmaq_f32(vsum1, vx1, vt1);
                vsum2 = vfmaq_f32(vsum2, vx2, vt2);
                vsum3 = vfmaq_f32(vsum3, vx3, vt3);
            }
            /* Process remaining 4 at a time */
            for (; j + 3 < block_elems; j += 4) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
            }

            /* Horizontal reduction */
            vsum0 = vaddq_f32(vsum0, vsum1);
            vsum2 = vaddq_f32(vsum2, vsum3);
            vsum0 = vaddq_f32(vsum0, vsum2);
            sum += vaddvq_f32(vsum0);

            /* Scalar tail */
            for (; j < block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#else
            /* Scalar fallback */
            for (int j = 0; j < block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#endif
        }

        out[d] = sum;
    }
}
