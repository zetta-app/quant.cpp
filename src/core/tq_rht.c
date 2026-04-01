/**
 * Random Hadamard Transform (RHT) — core implementation
 *
 * The Walsh-Hadamard Transform (WHT) rotates input vectors to remove
 * inter-coordinate correlation, making scalar quantization near-optimal.
 * Combined with random sign flipping, it becomes the Random Hadamard
 * Transform (RHT).
 *
 * Key properties:
 * - WHT is self-inverse: H * H = n * I
 * - O(n log n) butterfly computation, no matrix storage needed
 * - Random signs decorrelate channels across different blocks
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ---------- Random sign generation from seed ---------- */

static int random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t)idx;
    h = h * 2654435761u;  /* Knuth multiplicative hash */
    return (h & 1) ? 1 : -1;
}

/* ---------- In-place Walsh-Hadamard Transform: O(n log n) ----------
 * n must be power of 2.
 * This is self-inverse up to scaling: WHT(WHT(x)) = n * x.
 * The butterfly pattern applies log2(n) stages of add/subtract pairs. */

static void walsh_hadamard(float* data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
#ifdef __ARM_NEON
            if (len >= 4) {
                int j = 0;
                for (; j + 3 < len; j += 4) {
                    float32x4_t u = vld1q_f32(data + i + j);
                    float32x4_t v = vld1q_f32(data + i + j + len);
                    vst1q_f32(data + i + j,       vaddq_f32(u, v));
                    vst1q_f32(data + i + j + len,  vsubq_f32(u, v));
                }
                for (; j < len; j++) {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j]       = u + v;
                    data[i + j + len] = u - v;
                }
            } else
#endif
            {
                for (int j = 0; j < len; j++) {
                    float u = data[i + j];
                    float v = data[i + j + len];
                    data[i + j]       = u + v;
                    data[i + j + len] = u - v;
                }
            }
        }
    }
}

/* ---------- RHT: Random sign flip + Walsh-Hadamard + normalize ---------- */

void tq_rht_transform(float* data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    /* Round down to nearest power of 2 */
    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Random sign flip (diagonal D matrix) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)random_sign(seed, i);
    }

    /* Step 2: Walsh-Hadamard butterfly */
    walsh_hadamard(data, n2);

    /* Step 3: Normalize by 1/sqrt(n) to make it an orthogonal transform */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
}

/* ---------- Inverse RHT ----------
 * Since H is self-inverse (up to scaling) and D*D = I:
 *   RHT     = (1/sqrt(n)) * H * D
 *   RHT^-1  = (1/sqrt(n)) * D * H
 * So: scale first, then WHT, then same sign flip. */

void tq_rht_inverse(float* data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Normalize by 1/sqrt(n) */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }

    /* Step 2: Walsh-Hadamard (self-inverse up to scaling) */
    walsh_hadamard(data, n2);

    /* Step 3: Same random sign flip (D * D = I, so applying same signs inverts) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)random_sign(seed, i);
    }
}
