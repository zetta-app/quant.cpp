# REAL KV Compression Results (FP32 key cache eliminated)

All measurements use the REAL dequant path — no FP32 fallback.
Keys stored ONLY in quantized cache. Attention dequantizes per-query.

## uniform_4b K + Q4 V = 3.8x compression, PPL < 1%

| Model | Params | Baseline PPL | K4+VQ4 PPL | Delta | Tokens |
|-------|--------|-------------|-----------|-------|--------|
| SmolLM2 1.7B | 1.7B | 9.51 | 9.36 | **-1.6%** | 814 |
| Qwen3.5 0.8B | 752M | 153.6 | 155.1 | **+0.9%** | 810 |
| Qwen3.5 4B | 4B | 19.63 | 19.75 | **+0.6%** | 810 |

## All KV configs tested (SmolLM2 1.7B)

| Config | PPL | Delta | K+V Memory (32K) | Compression |
|--------|-----|-------|-------------------|-------------|
| FP16 K+V | 9.51 | — | 6.44 GB | 1.0x |
| uniform_4b K + FP16 V | 9.51 | +0.0% | 4.03 GB | 1.6x |
| **uniform_4b K + Q4 V** | **9.36** | **-1.6%** | **1.71 GB** | **3.8x** |
| uniform_4b K + Q2 V | 12.95 | +36% | 1.41 GB | 4.6x |
| turbo_kv_4b K + FP16 V | 10.07 | +5.9% | ~4 GB | ~1.6x |
| turbo_kv_3b K + FP16 V | 22.45 | +136% | ~3.8 GB | ~1.7x |
| turbo_kv_1b K + FP16 V | 1294.8 | catastrophic | ~3.5 GB | ~1.8x |
| uniform_2b K + FP16 V | 1618.6 | catastrophic | ~3.3 GB | ~2.0x |

## Key findings

1. **4-bit K is lossless.** uniform_4b gives exactly +0.00% PPL delta.
2. **Q4 V adds minimal noise.** Combined K4+VQ4 is within ±2% of baseline.
3. **Below 4-bit K: quality cliff.** 3-bit and below show significant degradation.
4. **Below Q4 V: noticeable degradation.** Q2 V adds +36% PPL.
5. **RHT-based types (turbo_kv_*) underperform uniform at head_dim=64.**
   turbo_kv_4b PPL is worse than uniform_4b despite same bit count.
