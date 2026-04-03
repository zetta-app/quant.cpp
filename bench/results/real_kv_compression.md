# REAL KV Compression Results (FP32 key cache eliminated)

All measurements use the REAL dequant path — no FP32 fallback.
Keys stored ONLY in quantized cache. Attention dequantizes per-query.

## Best Results: Delta Compression + Quantization

### SmolLM2 1.7B, 999 tokens (ppl_test_1k.txt)

| Config | K bpe | PPL | vs FP32 | Status |
|--------|-------|-----|---------|--------|
| FP32 baseline | 32 | 14.58 | — | reference |
| **delta + 4b K + Q4 V** | ~4 | **12.80** | **-12.2%** | **best quality** |
| **delta + 3b K + Q4 V** | ~3.5 | **14.11** | **-3.2%** | **best compression** |
| delta + 3b K + FP16 V | ~3.5 | 14.67 | +0.6% | near-lossless |
| uniform_4b K + Q4 V | 4 | 13.44 | -7.8% | proven |
| uniform_4b K + FP16 V | 4 | 14.58 | +0.0% | lossless |
| uniform_3b K + Q4 V | 3 | 23.62 | +62% | needs delta |
| uniform_4b K + Q2 V | 4 | 22.85 | +57% | V too aggressive |
| delta + 2b K + Q4 V | ~2.5 | 33.90 | +132% | drift too fast |
| uniform_2b K | 2 | 291.0 | catastrophic | — |

### Cross-model (4b K + Q4 V, earlier dataset, 810-814 tokens)

| Model | Params | Baseline PPL | K4+VQ4 PPL | Delta | Tokens |
|-------|--------|-------------|-----------|-------|--------|
| SmolLM2 1.7B | 1.7B | 9.51 | 9.36 | **-1.6%** | 814 |
| Qwen3.5 0.8B | 752M | 153.6 | 155.1 | **+0.9%** | 810 |
| Qwen3.5 4B | 4B | 19.63 | 19.75 | **+0.6%** | 810 |

## Delta Compression: How It Works

Adjacent keys in a transformer differ by ~30% of their absolute range.
Delta compression stores `key[t] - reconstruct(key[t-1])` instead of `key[t]`.
Periodic FP32 I-frames (every 64 tokens) anchor reconstruction and bound drift.

This is analogous to I/P-frames in video compression applied to KV cache.

**Result:** 3-bit without delta gives PPL +62%. With delta, it gives **-3.2%**.

## Age-Based Progressive K Compression

Tested: store recent N keys at FP32, old keys at 2-bit quantized.
Old tokens receive negligible attention weight, so 2-bit noise should not matter.

| Window | PPL | vs FP32 | Notes |
|--------|-----|---------|-------|
| 256 | 19.45 | +33% | 25% of sequence at FP32 |
| 128 | 29.72 | +104% | |
| 64 | 45.63 | +213% | |
| 32 | 53.82 | +269% | |
| 0 (pure 2-bit) | 291.0 | catastrophic | |

Finding: helps dramatically (291 -> 19.4) but 2-bit K is too destructive even
for "old" tokens. The accumulated noise from hundreds of 2-bit keys still
corrupts the attention distribution.

## Head-Level Mixed Precision

Per-head attention entropy profiling on SmolLM2 1.7B (32 heads x 24 layers):
- Entropy range: 0.0003 (L10 H28, very sharp) to 5.01 (L0 H0, near-uniform)
- Early layers (L0-L1): nearly all insensitive (high entropy, diffuse attention)
- Deep layers: mixed sensitivity

50/50 split (sensitive=4-bit, insensitive=2-bit) at 3.0 effective bpe:
- Attention score correlation: 0.9986 (vs 0.9998 for uniform 4-bit)
- Marginal improvement over uniform allocation — not worth the complexity.

## Online SVD / Low-Rank Approximation

Tested offline SVD, random projection, online incremental PCA.

Offline SVD at rank=8 (head_dim=64): avg cosine = 0.934, 87% energy captured.
The key matrix is NOT strongly low-rank — SVD approach is not competitive
with direct quantization. **Discarded.**

## 2-bit Research: All Approaches Tested

### Per-delta cosine (individual, dim=256, 199 deltas)
| Method | Cosine | Notes |
|--------|--------|-------|
| 2-bit uniform delta | 0.9975 | baseline |
| 2-bit + error feedback | 0.9963 | slightly worse |
| 2-bit NF2 (non-uniform) | 0.9959 | worse |
| 3-bit uniform delta | 0.9993 | reference |

### Accumulated cosine (200 steps, drift)
| Method | Avg cosine | Notes |
|--------|-----------|-------|
| Standard delta+2-bit | 0.885 | drift accumulation |
| Norm-corrected delta+2-bit | 0.877 | worse (distorts direction) |

### Conclusion
2-bit drift over 200 tokens (cos 0.997 -> 0.885) is the fundamental barrier.
No tested approach (error feedback, NF2, norm correction, 2nd-order prediction,
age-based windowing, head-mixed precision, online SVD) overcomes it.
**3-bit + delta is the practical minimum for KV cache keys.**
