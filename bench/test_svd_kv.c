/**
 * SVD / Low-Rank KV Cache Compression — Concept Validation
 *
 * Three approaches tested:
 *   1. Offline SVD (upper bound): full SVD of K[seq_len x head_dim], truncate to rank r
 *   2. Random Projection (JL-style): fixed random matrix R[rank x head_dim], no adaptation
 *   3. Online Incremental PCA: maintain rank-r basis, update as tokens arrive
 *
 * All use synthetic keys with realistic RoPE + residual structure.
 * Reports cosine similarity and bpe for rank = 4, 8, 16, 32.
 *
 * Build:
 *   cc -O2 -I include bench/test_svd_kv.c -lm -o build/test_svd_kv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ========== RNG (xoshiro128+) ========== */

static uint32_t rng_state[4] = {0x12345678, 0x9ABCDEF0, 0xDEADBEEF, 0xCAFEBABE};

static uint32_t rotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t xoshiro128p(void) {
    uint32_t result = rng_state[0] + rng_state[3];
    uint32_t t = rng_state[1] << 9;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 11);
    return result;
}

static float rand_uniform(void) {
    return (float)(xoshiro128p() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static float rand_normal(void) {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void seed_rng(uint32_t seed) {
    rng_state[0] = seed;
    rng_state[1] = seed * 2654435761u;
    rng_state[2] = seed * 340573321u;
    rng_state[3] = seed * 1013904223u;
    for (int i = 0; i < 20; i++) xoshiro128p();
}

/* ========== Metrics ========== */

static double cosine_sim(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    na = sqrt(na); nb = sqrt(nb);
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / (na * nb);
}

static double vec_norm(const double* v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrt(s);
}

/* ========== Generate Realistic Correlated Key Sequences ========== */

static void generate_correlated_keys(float* keys, int seq_len, int head_dim) {
    float base[512];
    for (int d = 0; d < head_dim; d++) {
        base[d] = rand_normal() * 0.5f;
    }

    float freqs[256];
    for (int i = 0; i < head_dim / 2; i++) {
        freqs[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
    }

    for (int t = 0; t < seq_len; t++) {
        float* k = keys + t * head_dim;
        memcpy(k, base, head_dim * sizeof(float));

        for (int i = 0; i < head_dim / 2; i++) {
            float theta = (float)t * freqs[i];
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float k0 = k[2 * i];
            float k1 = k[2 * i + 1];
            k[2 * i]     = k0 * cos_t - k1 * sin_t;
            k[2 * i + 1] = k0 * sin_t + k1 * cos_t;
        }

        for (int d = 0; d < head_dim; d++) {
            k[d] += rand_normal() * 0.05f;
        }

        if (t % 10 == 0) {
            for (int d = 0; d < head_dim; d++) {
                base[d] += rand_normal() * 0.02f;
            }
        }
    }
}

/* ========== Linear Algebra Helpers (double precision) ========== */

/* Matrix-vector multiply: y = A[m x n] * x[n] */
static void matvec(const double* A, const double* x, double* y, int m, int n) {
    for (int i = 0; i < m; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) {
            s += A[i * n + j] * x[j];
        }
        y[i] = s;
    }
}

/* Matrix-transpose-vector multiply: y = A^T[m x n] * x[m]  => y[n] */
static void matvec_t(const double* A, const double* x, double* y, int m, int n) {
    for (int j = 0; j < n; j++) y[j] = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            y[j] += A[i * n + j] * x[i];
        }
    }
}

/* Dot product */
static double dot_d(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* Scale vector */
static void scale_vec(double* v, double s, int n) {
    for (int i = 0; i < n; i++) v[i] *= s;
}

/* axpy: y += a * x */
static void axpy(double* y, double a, const double* x, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

/* ========== Offline SVD via Power Iteration + Deflation ========== */

/**
 * Compute top-r singular vectors of K[seq_len x head_dim] via power iteration.
 * Returns:
 *   U[seq_len x r], S[r], V[r x head_dim] such that K ~ U * diag(S) * V
 *
 * Algorithm:
 *   For each singular vector k = 0..r-1:
 *     1. Random init v_k
 *     2. Repeat: u = K * v; v = K^T * u; normalize
 *     3. sigma = ||K * v||
 *     4. Deflate: K -= sigma * u * v^T
 */
static void compute_svd_topk(const double* K, int seq_len, int head_dim, int rank,
                              double* U, double* S, double* V) {
    int m = seq_len, n = head_dim;

    /* Work on a copy since we deflate */
    double* Kw = (double*)malloc((size_t)m * n * sizeof(double));
    memcpy(Kw, K, (size_t)m * n * sizeof(double));

    double* u = (double*)malloc(m * sizeof(double));
    double* v = (double*)malloc(n * sizeof(double));
    double* tmp = (double*)malloc((m > n ? m : n) * sizeof(double));

    for (int k = 0; k < rank; k++) {
        /* Random init for v */
        seed_rng(42 + k * 137);
        for (int j = 0; j < n; j++) v[j] = rand_normal();
        double nrm = vec_norm(v, n);
        scale_vec(v, 1.0 / nrm, n);

        /* Power iteration: 100 steps is plenty for well-separated singular values */
        for (int iter = 0; iter < 100; iter++) {
            /* u = Kw * v */
            matvec(Kw, v, u, m, n);
            nrm = vec_norm(u, m);
            if (nrm < 1e-15) break;
            scale_vec(u, 1.0 / nrm, m);

            /* v = Kw^T * u */
            matvec_t(Kw, u, v, m, n);
            nrm = vec_norm(v, n);
            if (nrm < 1e-15) break;
            scale_vec(v, 1.0 / nrm, n);
        }

        /* Compute singular value: sigma = ||Kw * v|| */
        matvec(Kw, v, tmp, m, n);
        double sigma = vec_norm(tmp, m);
        S[k] = sigma;

        /* Compute u = Kw * v / sigma */
        if (sigma > 1e-15) {
            scale_vec(tmp, 1.0 / sigma, m);
            memcpy(u, tmp, m * sizeof(double));
        }

        /* Store u_k, v_k */
        for (int i = 0; i < m; i++) U[i * rank + k] = u[i];
        for (int j = 0; j < n; j++) V[k * n + j] = v[j];

        /* Deflate: Kw -= sigma * u * v^T */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Kw[i * n + j] -= sigma * u[i] * v[j];
            }
        }
    }

    free(Kw);
    free(u);
    free(v);
    free(tmp);
}

/* ========== Offline SVD Test ========== */

static void test_offline_svd(const float* keys, int seq_len, int head_dim,
                              int rank, double* out_avg_cos, double* out_min_cos) {
    int m = seq_len, n = head_dim;

    /* Convert to double */
    double* K = (double*)malloc((size_t)m * n * sizeof(double));
    for (int i = 0; i < m * n; i++) K[i] = (double)keys[i];

    double* U = (double*)calloc((size_t)m * rank, sizeof(double));
    double* S = (double*)calloc(rank, sizeof(double));
    double* V = (double*)calloc((size_t)rank * n, sizeof(double));

    compute_svd_topk(K, m, n, rank, U, S, V);

    /* Reconstruct: K_approx[i] = sum_k U[i,k] * S[k] * V[k,:] */
    float* recon = (float*)calloc((size_t)m * n, sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int k = 0; k < rank; k++) {
            double coeff = U[i * rank + k] * S[k];
            for (int j = 0; j < n; j++) {
                recon[i * n + j] += (float)(coeff * V[k * n + j]);
            }
        }
    }

    /* Measure quality */
    double cos_sum = 0.0, cos_min = 1.0;
    for (int t = 0; t < m; t++) {
        double c = cosine_sim(keys + t * n, recon + t * n, n);
        cos_sum += c;
        if (c < cos_min) cos_min = c;
    }
    *out_avg_cos = cos_sum / m;
    *out_min_cos = cos_min;

    /* Print singular value spectrum */
    if (rank <= 32) {
        printf("  Singular values: ");
        for (int k = 0; k < rank && k < 8; k++) {
            printf("%.2f ", S[k]);
        }
        if (rank > 8) printf("...");
        printf("\n");

        /* Energy captured */
        double total_energy = 0.0;
        {
            /* Need full spectrum for energy calculation - use K^T K eigenvalues */
            /* Approximation: sum of squares of all elements */
            double total_sq = 0.0;
            for (int i = 0; i < m * n; i++) total_sq += K[i] * K[i];
            double captured = 0.0;
            for (int k = 0; k < rank; k++) captured += S[k] * S[k];
            total_energy = total_sq;
            printf("  Energy captured: %.2f%% (%.2f / %.2f)\n",
                   100.0 * captured / total_energy, captured, total_energy);
        }
    }

    free(K);
    free(U);
    free(S);
    free(V);
    free(recon);
}

/* ========== Random Projection Test ========== */

/**
 * Random projection: R[rank x head_dim] is a fixed random matrix.
 * coeff[t] = R * key[t]  (rank coefficients per token)
 * recon[t] = R^T * coeff[t] / head_dim  (pseudo-inverse for orthogonal R)
 *
 * Actually, for R with iid N(0,1/rank) entries, R^T * R ~ I, so
 * recon = R^T * R * key ~ key with error decreasing as rank/head_dim.
 *
 * This is the JL approach. No adaptation, trivially parallel.
 */
static void test_random_projection(const float* keys, int seq_len, int head_dim,
                                    int rank, double* out_avg_cos, double* out_min_cos) {
    int m = seq_len, n = head_dim;

    /* Generate random projection matrix R[rank x head_dim]
     * Each entry ~ N(0, 1/rank) for approximate isometry */
    seed_rng(12345);
    double* R = (double*)malloc((size_t)rank * n * sizeof(double));
    for (int i = 0; i < rank * n; i++) {
        R[i] = rand_normal() / sqrt((double)rank);
    }

    /* Orthogonalize R via modified Gram-Schmidt for better reconstruction */
    for (int i = 0; i < rank; i++) {
        /* Orthogonalize against previous rows */
        for (int j = 0; j < i; j++) {
            double dot = dot_d(R + i * n, R + j * n, n);
            axpy(R + i * n, -dot, R + j * n, n);
        }
        /* Normalize */
        double nrm = vec_norm(R + i * n, n);
        if (nrm > 1e-15) {
            scale_vec(R + i * n, 1.0 / nrm, n);
        }
    }

    /* Project and reconstruct each token */
    double* coeff = (double*)malloc(rank * sizeof(double));
    float* recon_vec = (float*)malloc(n * sizeof(float));

    double cos_sum = 0.0, cos_min = 1.0;

    for (int t = 0; t < m; t++) {
        /* coeff = R * key[t] */
        for (int k = 0; k < rank; k++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += R[k * n + j] * (double)keys[t * n + j];
            }
            coeff[k] = s;
        }

        /* recon = R^T * coeff (since R is orthonormal rows, R^T * R = I on subspace) */
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < rank; k++) {
                s += R[k * n + j] * coeff[k];
            }
            recon_vec[j] = (float)s;
        }

        double c = cosine_sim(keys + t * n, recon_vec, n);
        cos_sum += c;
        if (c < cos_min) cos_min = c;
    }

    *out_avg_cos = cos_sum / m;
    *out_min_cos = cos_min;

    free(R);
    free(coeff);
    free(recon_vec);
}

/* ========== Online Incremental PCA ========== */

/**
 * Incremental PCA: maintain a rank-r orthonormal basis V[rank x head_dim]
 * and running mean mu[head_dim].
 *
 * For each new key k_t:
 *   1. Update running mean: mu = (t * mu + k_t) / (t+1)
 *   2. Centered: c = k_t - mu
 *   3. Project: coeff = V * c  [rank coefficients]
 *   4. Residual: r = c - V^T * coeff
 *   5. If ||r|| > threshold and we haven't filled rank yet, or
 *      if ||r|| > smallest singular direction, replace it
 *   6. Store coeff for this token
 *
 * Reconstruction: recon[t] = mu_final + V_final^T * coeff[t]
 * NOTE: using mu_final and V_final is cheating slightly since they evolve.
 *       A truly online system would need to store basis snapshots.
 *       We test both: "cheating" (final basis) and "honest" (basis at time t).
 */
static void test_online_ipca(const float* keys, int seq_len, int head_dim,
                              int rank, int honest_mode,
                              double* out_avg_cos, double* out_min_cos) {
    int n = head_dim;

    /* Basis V[rank x head_dim] (orthonormal rows) */
    double* V = (double*)calloc((size_t)rank * n, sizeof(double));
    int active_rank = 0;  /* how many basis vectors are active */

    /* Running mean */
    double* mu = (double*)calloc(n, sizeof(double));

    /* Store all coefficients (for later reconstruction) */
    double* all_coeff = (double*)calloc((size_t)seq_len * rank, sizeof(double));

    /* For honest mode: store basis snapshots */
    double* V_snapshots = NULL;
    double* mu_snapshots = NULL;
    if (honest_mode) {
        V_snapshots = (double*)calloc((size_t)seq_len * rank * n, sizeof(double));
        mu_snapshots = (double*)calloc((size_t)seq_len * n, sizeof(double));
    }

    /* Singular value estimates (for deciding which directions to keep) */
    double* sv_estimates = (double*)calloc(rank, sizeof(double));

    double* centered = (double*)malloc(n * sizeof(double));
    double* coeff = (double*)malloc(rank * sizeof(double));
    double* residual = (double*)malloc(n * sizeof(double));

    for (int t = 0; t < seq_len; t++) {
        const float* key = keys + t * n;

        /* 1. Snapshot basis BEFORE update (for honest reconstruction) */
        if (honest_mode) {
            memcpy(V_snapshots + (size_t)t * rank * n, V, (size_t)rank * n * sizeof(double));
            memcpy(mu_snapshots + (size_t)t * n, mu, n * sizeof(double));
        }

        /* 2. Center using CURRENT mean (before update) */
        for (int j = 0; j < n; j++) {
            centered[j] = (double)key[j] - mu[j];
        }

        /* 3. Project onto current basis (BEFORE any basis update) */
        for (int k = 0; k < active_rank; k++) {
            coeff[k] = dot_d(V + k * n, centered, n);
        }
        for (int k = active_rank; k < rank; k++) {
            coeff[k] = 0.0;
        }

        /* 4. Store coefficients (computed with pre-update basis) */
        memcpy(all_coeff + t * rank, coeff, rank * sizeof(double));

        /* 5. Compute residual for basis update decision */
        memcpy(residual, centered, n * sizeof(double));
        for (int k = 0; k < active_rank; k++) {
            axpy(residual, -coeff[k], V + k * n, n);
        }

        double res_norm = vec_norm(residual, n);

        /* 6. Update running mean AFTER projection */
        for (int j = 0; j < n; j++) {
            mu[j] = (mu[j] * t + (double)key[j]) / (t + 1);
        }

        /* 7. Update basis if residual is significant */
        if (res_norm > 1e-8) {
            if (active_rank < rank) {
                /* Add new basis direction */
                scale_vec(residual, 1.0 / res_norm, n);
                memcpy(V + active_rank * n, residual, n * sizeof(double));
                sv_estimates[active_rank] = res_norm;
                active_rank++;
            } else {
                /* Replace smallest direction if residual is larger */
                int min_idx = 0;
                for (int k = 1; k < rank; k++) {
                    if (sv_estimates[k] < sv_estimates[min_idx]) min_idx = k;
                }

                /* Exponential moving average of direction importance */
                for (int k = 0; k < rank; k++) {
                    sv_estimates[k] = sv_estimates[k] * 0.99 + fabs(coeff[k]) * 0.01;
                }

                if (res_norm > sv_estimates[min_idx] * 0.5) {
                    /* Replace the weakest direction */
                    scale_vec(residual, 1.0 / res_norm, n);
                    memcpy(V + min_idx * n, residual, n * sizeof(double));
                    sv_estimates[min_idx] = res_norm;

                    /* Re-orthogonalize (important for stability) */
                    for (int pass = 0; pass < 2; pass++) {
                        for (int k = 0; k < rank; k++) {
                            for (int j = 0; j < k; j++) {
                                double d = dot_d(V + k * n, V + j * n, n);
                                axpy(V + k * n, -d, V + j * n, n);
                            }
                            double nrm = vec_norm(V + k * n, n);
                            if (nrm > 1e-15) scale_vec(V + k * n, 1.0 / nrm, n);
                        }
                    }
                } else {
                    /* Just update importance estimates */
                    for (int k = 0; k < rank; k++) {
                        sv_estimates[k] = sv_estimates[k] * 0.99 + fabs(coeff[k]) * 0.01;
                    }
                }
            }
        }
    }

    /* Reconstruct and measure quality.
     * Skip first 'rank' tokens for honest mode since basis is bootstrapping. */
    float* recon_vec = (float*)malloc(n * sizeof(float));
    double cos_sum = 0.0, cos_min = 1.0;
    int start_t = (honest_mode) ? rank : 0;
    int count = 0;

    for (int t = start_t; t < seq_len; t++) {
        double* basis = honest_mode ? V_snapshots + (size_t)t * rank * n : V;
        double* mean = honest_mode ? mu_snapshots + (size_t)t * n : mu;

        for (int j = 0; j < n; j++) {
            double s = mean[j];
            for (int k = 0; k < rank; k++) {
                s += basis[k * n + j] * all_coeff[t * rank + k];
            }
            recon_vec[j] = (float)s;
        }

        double c = cosine_sim(keys + t * n, recon_vec, n);
        cos_sum += c;
        if (c < cos_min) cos_min = c;
        count++;
    }

    *out_avg_cos = (count > 0) ? cos_sum / count : 0.0;
    *out_min_cos = cos_min;

    free(V);
    free(mu);
    free(all_coeff);
    free(sv_estimates);
    free(centered);
    free(coeff);
    free(residual);
    free(recon_vec);
    if (V_snapshots) free(V_snapshots);
    if (mu_snapshots) free(mu_snapshots);
}

/* ========== Quantized Coefficient Test ========== */

/**
 * Test random projection with 4-bit quantized coefficients.
 * This is the practical scenario: we store rank coefficients per token,
 * each quantized to 4 bits with min-max uniform quantization.
 */
static void test_random_proj_quantized(const float* keys, int seq_len, int head_dim,
                                        int rank, double* out_avg_cos, double* out_min_cos) {
    int m = seq_len, n = head_dim;

    /* Generate and orthogonalize random projection matrix */
    seed_rng(12345);
    double* R = (double*)malloc((size_t)rank * n * sizeof(double));
    for (int i = 0; i < rank * n; i++) {
        R[i] = rand_normal() / sqrt((double)rank);
    }
    for (int i = 0; i < rank; i++) {
        for (int j = 0; j < i; j++) {
            double dot = dot_d(R + i * n, R + j * n, n);
            axpy(R + i * n, -dot, R + j * n, n);
        }
        double nrm = vec_norm(R + i * n, n);
        if (nrm > 1e-15) scale_vec(R + i * n, 1.0 / nrm, n);
    }

    /* Compute all coefficients first to find global min/max per component */
    double* all_coeff = (double*)malloc((size_t)m * rank * sizeof(double));
    for (int t = 0; t < m; t++) {
        for (int k = 0; k < rank; k++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += R[k * n + j] * (double)keys[t * n + j];
            }
            all_coeff[t * rank + k] = s;
        }
    }

    /* Quantize coefficients to 4-bit per component (min-max uniform) */
    /* Use per-component min/max over all tokens */
    double* comp_min = (double*)malloc(rank * sizeof(double));
    double* comp_max = (double*)malloc(rank * sizeof(double));
    for (int k = 0; k < rank; k++) {
        comp_min[k] = all_coeff[k];
        comp_max[k] = all_coeff[k];
        for (int t = 1; t < m; t++) {
            double v = all_coeff[t * rank + k];
            if (v < comp_min[k]) comp_min[k] = v;
            if (v > comp_max[k]) comp_max[k] = v;
        }
    }

    /* Quantize and dequantize */
    for (int t = 0; t < m; t++) {
        for (int k = 0; k < rank; k++) {
            double v = all_coeff[t * rank + k];
            double range = comp_max[k] - comp_min[k];
            if (range < 1e-12) range = 1e-12;
            /* 4-bit: 16 levels */
            int q = (int)(((v - comp_min[k]) / range) * 15.0 + 0.5);
            if (q < 0) q = 0;
            if (q > 15) q = 15;
            all_coeff[t * rank + k] = comp_min[k] + (double)q / 15.0 * range;
        }
    }

    /* Reconstruct */
    float* recon_vec = (float*)malloc(n * sizeof(float));
    double cos_sum = 0.0, cos_min = 1.0;

    for (int t = 0; t < m; t++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < rank; k++) {
                s += R[k * n + j] * all_coeff[t * rank + k];
            }
            recon_vec[j] = (float)s;
        }
        double c = cosine_sim(keys + t * n, recon_vec, n);
        cos_sum += c;
        if (c < cos_min) cos_min = c;
    }

    *out_avg_cos = cos_sum / m;
    *out_min_cos = cos_min;

    free(R);
    free(all_coeff);
    free(comp_min);
    free(comp_max);
    free(recon_vec);
}

/* ========== BPE Calculation ========== */

static void print_bpe_analysis(int rank, int head_dim, int seq_len) {
    /* Coefficients stored per token: rank values */
    /* At FP32: rank * 4 bytes per token */
    /* At 4-bit: rank * 0.5 bytes per token */
    /* Basis storage: rank * head_dim * 4 bytes (FP32), shared across all tokens */
    /* Basis amortized per token: rank * head_dim * 4 / seq_len bytes */

    double coeff_bytes_fp32 = (double)rank * 4.0;
    double coeff_bytes_4bit = (double)rank * 0.5;
    double basis_bytes = (double)rank * head_dim * 4.0;
    double basis_amortized = basis_bytes / seq_len;

    double total_fp32 = coeff_bytes_fp32 + basis_amortized;
    double total_4bit = coeff_bytes_4bit + basis_amortized;

    double fp16_per_token = (double)head_dim * 2.0;

    /* BPE = bits per element (per dim) */
    double bpe_fp32 = (total_fp32 * 8.0) / head_dim;
    double bpe_4bit = (total_4bit * 8.0) / head_dim;

    printf("    Coefficient storage (FP32): %.1f bytes/token = %.2f bpe\n",
           coeff_bytes_fp32, bpe_fp32);
    printf("    Coefficient storage (4-bit): %.1f bytes/token = %.2f bpe\n",
           coeff_bytes_4bit, bpe_4bit);
    printf("    Basis storage (amortized over %d tokens): %.1f bytes/token\n",
           seq_len, basis_amortized);
    printf("    Total with 4-bit coeffs: %.1f bytes/token (vs %.0f FP16)\n",
           total_4bit, fp16_per_token);
    printf("    Compression ratio (4-bit coeffs): %.1fx\n",
           fp16_per_token / total_4bit);
}

/* ========== Main ========== */

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    int head_dim = 64;
    int seq_len  = 200;

    if (argc > 1) head_dim = atoi(argv[1]);
    if (argc > 2) seq_len  = atoi(argv[2]);
    if (head_dim < 8) head_dim = 8;
    if (head_dim > 512) head_dim = 512;
    if (seq_len < 16) seq_len = 16;

    printf("=============================================================\n");
    printf("  SVD / Low-Rank KV Cache Compression — Concept Validation\n");
    printf("=============================================================\n");
    printf("head_dim=%d  seq_len=%d\n\n", head_dim, seq_len);

    /* Generate realistic keys */
    seed_rng(42);
    float* keys = (float*)calloc((size_t)seq_len * head_dim, sizeof(float));
    generate_correlated_keys(keys, seq_len, head_dim);

    int ranks[] = {4, 8, 16, 32};
    int n_ranks = 4;

    /* ====== Test 1: Offline SVD (upper bound on quality) ====== */
    printf("=== 1. OFFLINE SVD (Upper Bound) ===\n");
    printf("Full SVD of K[%d x %d], truncated to rank r.\n", seq_len, head_dim);
    printf("This is the BEST possible rank-r approximation (Eckart-Young theorem).\n\n");

    printf("rank | avg_cosine | min_cosine | energy%%\n");
    printf("-----|------------|------------|--------\n");

    for (int ri = 0; ri < n_ranks; ri++) {
        int r = ranks[ri];
        if (r > head_dim) continue;

        double avg_cos, min_cos;
        test_offline_svd(keys, seq_len, head_dim, r, &avg_cos, &min_cos);

        printf("  %2d | %.6f   | %.6f   | (see above)\n", r, avg_cos, min_cos);
    }

    /* ====== Test 2: Random Projection (no adaptation) ====== */
    printf("\n=== 2. RANDOM PROJECTION (JL-style, no adaptation) ===\n");
    printf("Fixed orthogonal random matrix R[rank x %d].\n", head_dim);
    printf("No data dependence — trivially parallel, no update cost.\n\n");

    printf("rank | avg_cosine | min_cosine\n");
    printf("-----|------------|----------\n");

    for (int ri = 0; ri < n_ranks; ri++) {
        int r = ranks[ri];
        if (r > head_dim) continue;

        double avg_cos, min_cos;
        test_random_projection(keys, seq_len, head_dim, r, &avg_cos, &min_cos);

        printf("  %2d | %.6f   | %.6f\n", r, avg_cos, min_cos);
    }

    /* ====== Test 3: Online Incremental PCA (cheating: final basis) ====== */
    printf("\n=== 3. ONLINE INCREMENTAL PCA (final basis — optimistic) ===\n");
    printf("Basis adapts as tokens arrive. Reconstruction uses FINAL basis.\n");
    printf("This overestimates quality since early tokens were encoded with early basis.\n\n");

    printf("rank | avg_cosine | min_cosine\n");
    printf("-----|------------|----------\n");

    for (int ri = 0; ri < n_ranks; ri++) {
        int r = ranks[ri];
        if (r > head_dim) continue;

        double avg_cos, min_cos;
        test_online_ipca(keys, seq_len, head_dim, r, 0, &avg_cos, &min_cos);

        printf("  %2d | %.6f   | %.6f\n", r, avg_cos, min_cos);
    }

    /* ====== Test 4: Online Incremental PCA (honest: basis at encode time) ====== */
    printf("\n=== 4. ONLINE INCREMENTAL PCA (honest — basis at encode time) ===\n");
    printf("Each token reconstructed using the basis that existed when it was encoded.\n");
    printf("This is what a real online system would achieve.\n\n");

    printf("rank | avg_cosine | min_cosine\n");
    printf("-----|------------|----------\n");

    for (int ri = 0; ri < n_ranks; ri++) {
        int r = ranks[ri];
        if (r > head_dim) continue;

        double avg_cos, min_cos;
        test_online_ipca(keys, seq_len, head_dim, r, 1, &avg_cos, &min_cos);

        printf("  %2d | %.6f   | %.6f\n", r, avg_cos, min_cos);
    }

    /* ====== Test 5: Random Projection with 4-bit quantized coefficients ====== */
    printf("\n=== 5. RANDOM PROJECTION + 4-BIT QUANTIZED COEFFICIENTS ===\n");
    printf("Practical scenario: random proj + 4-bit uniform quantization of coefficients.\n\n");

    printf("rank | avg_cosine | min_cosine\n");
    printf("-----|------------|----------\n");

    for (int ri = 0; ri < n_ranks; ri++) {
        int r = ranks[ri];
        if (r > head_dim) continue;

        double avg_cos, min_cos;
        test_random_proj_quantized(keys, seq_len, head_dim, r, &avg_cos, &min_cos);

        printf("  %2d | %.6f   | %.6f\n", r, avg_cos, min_cos);
    }

    /* ====== BPE Analysis ====== */
    printf("\n=== BPE / COMPRESSION ANALYSIS ===\n");
    printf("FP16 baseline: %d dims x 2 bytes = %d bytes/token = 16.0 bpe\n\n",
           head_dim, head_dim * 2);

    for (int ri = 0; ri < n_ranks; ri++) {
        int r = ranks[ri];
        if (r > head_dim) continue;
        printf("--- rank=%d ---\n", r);
        print_bpe_analysis(r, head_dim, seq_len);
        printf("\n");
    }

    /* ====== Per-position cosine for rank=8 (offline SVD) ====== */
    printf("\n=== PER-POSITION COSINE — OFFLINE SVD (rank=8) ===\n");
    printf("Shows quality variation across sequence positions.\n\n");

    {
        int r = 8;
        if (r <= head_dim) {
            int m = seq_len, n2 = head_dim;
            double* K = (double*)malloc((size_t)m * n2 * sizeof(double));
            for (int i = 0; i < m * n2; i++) K[i] = (double)keys[i];

            double* U = (double*)calloc((size_t)m * r, sizeof(double));
            double* S = (double*)calloc(r, sizeof(double));
            double* Vb = (double*)calloc((size_t)r * n2, sizeof(double));
            compute_svd_topk(K, m, n2, r, U, S, Vb);

            float* recon_svd = (float*)calloc((size_t)m * n2, sizeof(float));
            for (int i = 0; i < m; i++) {
                for (int k = 0; k < r; k++) {
                    double c = U[i * r + k] * S[k];
                    for (int j = 0; j < n2; j++) {
                        recon_svd[i * n2 + j] += (float)(c * Vb[k * n2 + j]);
                    }
                }
            }

            printf("pos  | cosine\n");
            printf("-----|--------\n");
            for (int t = 0; t < seq_len; t += 10) {
                double c_svd = cosine_sim(keys + t * n2, recon_svd + t * n2, n2);
                printf("%4d | %.6f\n", t, c_svd);
            }

            free(K); free(U); free(S); free(Vb); free(recon_svd);
        }
    }

    /* ====== Summary ====== */
    printf("\n=============================================================\n");
    printf("  SUMMARY\n");
    printf("=============================================================\n\n");

    printf("Method comparison at rank=8, head_dim=%d, seq_len=%d:\n\n", head_dim, seq_len);

    {
        double avg1, min1, avg2, min2, avg3, min3, avg4, min4, avg5, min5;
        int r = 8;
        if (r <= head_dim) {
            test_offline_svd(keys, seq_len, head_dim, r, &avg1, &min1);
            test_random_projection(keys, seq_len, head_dim, r, &avg2, &min2);
            test_online_ipca(keys, seq_len, head_dim, r, 0, &avg3, &min3);
            test_online_ipca(keys, seq_len, head_dim, r, 1, &avg4, &min4);
            test_random_proj_quantized(keys, seq_len, head_dim, r, &avg5, &min5);

            printf("  Offline SVD:          avg_cos=%.6f  min_cos=%.6f  (upper bound)\n", avg1, min1);
            printf("  Random Projection:    avg_cos=%.6f  min_cos=%.6f  (no adaptation)\n", avg2, min2);
            printf("  IPCA (final basis):   avg_cos=%.6f  min_cos=%.6f  (optimistic)\n", avg3, min3);
            printf("  IPCA (honest):        avg_cos=%.6f  min_cos=%.6f  (realistic)\n", avg4, min4);
            printf("  RandProj + 4-bit:     avg_cos=%.6f  min_cos=%.6f  (practical)\n", avg5, min5);

            printf("\n  Storage at rank=%d with 4-bit coefficients:\n", r);
            double coeff_4bit = (double)r * 0.5;
            double basis_amort = (double)r * head_dim * 4.0 / seq_len;
            double total = coeff_4bit + basis_amort;
            double fp16_ref = (double)head_dim * 2.0;
            printf("    %.1f bytes/token (%.1f coeff + %.1f basis amortized)\n",
                   total, coeff_4bit, basis_amort);
            printf("    vs FP16: %.0f bytes/token\n", fp16_ref);
            printf("    Compression: %.1fx\n", fp16_ref / total);
            printf("    Effective bpe: %.2f (vs 16.0 FP16)\n", total * 8.0 / head_dim);

            printf("\n");
            if (avg1 > 0.99) {
                printf("  Offline SVD achieves >0.99 cosine at rank=%d.\n", r);
                printf("  This proves the key matrix IS low-rank.\n");
            } else if (avg1 > 0.95) {
                printf("  Offline SVD achieves >0.95 cosine at rank=%d.\n", r);
                printf("  Moderate low-rank structure exists.\n");
            } else {
                printf("  Offline SVD cosine %.4f at rank=%d is mediocre.\n", avg1, r);
                printf("  The key matrix may not be strongly low-rank.\n");
            }

            if (avg2 > 0.95) {
                printf("  Random projection works well — no SVD update needed!\n");
            } else if (avg2 < avg1 - 0.1) {
                printf("  Random projection is much worse than SVD — data-dependent basis matters.\n");
            }

            if (avg4 > 0.95) {
                printf("  Online IPCA is practical with honest reconstruction.\n");
            } else if (avg4 < avg1 - 0.05) {
                printf("  Online IPCA loses quality vs offline — basis evolution is a problem.\n");
            }
        }
    }

    /* ====== Scaling test: longer sequences ====== */
    printf("\n=== SCALING: Offline SVD at rank=8, varying seq_len ===\n");
    printf("Shows how compression ratio improves as basis is amortized over more tokens.\n\n");
    printf("seq_len | avg_cos  | min_cos  | bpe    | compression\n");
    printf("--------|----------|----------|--------|------------\n");

    {
        int test_lens[] = {50, 100, 200, 500, 1000};
        int n_lens = 5;
        int r = 8;

        for (int li = 0; li < n_lens; li++) {
            int sl = test_lens[li];
            seed_rng(42);
            float* kk = (float*)calloc((size_t)sl * head_dim, sizeof(float));
            generate_correlated_keys(kk, sl, head_dim);

            double avg_c, min_c;
            test_offline_svd(kk, sl, head_dim, r, &avg_c, &min_c);

            double coeff_b = (double)r * 0.5;
            double basis_b = (double)r * head_dim * 4.0 / sl;
            double total_b = coeff_b + basis_b;
            double fp16_b = (double)head_dim * 2.0;
            double bpe = total_b * 8.0 / head_dim;

            printf("  %5d | %.6f | %.6f | %.2f   | %.1fx\n",
                   sl, avg_c, min_c, bpe, fp16_b / total_b);

            free(kk);
        }
    }

    /* ====== Test with head_dim=128 (common in larger models) ====== */
    printf("\n=== HEAD_DIM=128 TEST (Offline SVD) ===\n");
    printf("Larger head dimension means more room for low-rank structure.\n\n");

    {
        int hd = 128;
        int sl = 200;
        seed_rng(42);
        float* kk128 = (float*)calloc((size_t)sl * hd, sizeof(float));
        generate_correlated_keys(kk128, sl, hd);

        printf("rank | avg_cos  | min_cos  | bpe (4-bit) | compression\n");
        printf("-----|----------|----------|-------------|------------\n");

        for (int ri = 0; ri < n_ranks; ri++) {
            int r = ranks[ri];
            double avg_c, min_c;
            test_offline_svd(kk128, sl, hd, r, &avg_c, &min_c);

            double coeff_b = (double)r * 0.5;
            double basis_b = (double)r * hd * 4.0 / sl;
            double total_b = coeff_b + basis_b;
            double fp16_b = (double)hd * 2.0;
            double bpe = total_b * 8.0 / hd;

            printf("  %2d | %.6f | %.6f | %.2f        | %.1fx\n",
                   r, avg_c, min_c, bpe, fp16_b / total_b);
        }

        free(kk128);
    }

    free(keys);
    return 0;
}
