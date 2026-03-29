#ifndef TQ_SPEC_H
#define TQ_SPEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Format specification — version-aware, ONNX-inspired */

#define TQ_SPEC_VERSION 1

#define TQ_ALG_POLAR    0
#define TQ_ALG_QJL      1
#define TQ_ALG_TURBO    2
#define TQ_ALG_UNIFORM  3

#define TQ_FLAG_HAS_ZERO_POINT  (1 << 0)
#define TQ_FLAG_SYMMETRIC       (1 << 1)
#define TQ_FLAG_HAS_RESIDUAL    (1 << 2)

typedef struct {
    uint8_t  spec_version;     /* TQ_SPEC_VERSION                  */
    uint8_t  algorithm;        /* TQ_ALG_POLAR / QJL / TURBO / ... */
    uint8_t  key_bits;         /* total bits for key quantization   */
    uint8_t  value_bits;       /* bits for value quantization (0=none) */
    uint16_t block_size;       /* elements per block                */
    uint16_t sketch_dim;       /* QJL sketch dimension (0 if N/A)   */
    uint8_t  outlier_count;    /* QJL outlier count (0 if N/A)      */
    uint8_t  flags;            /* TQ_FLAG_* bitmask                 */
} tq_format_spec_t;

#ifdef __cplusplus
}
#endif

#endif /* TQ_SPEC_H */
