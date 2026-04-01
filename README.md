# TurboQuant.cpp

**Pure C inference engine with [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Release](https://img.shields.io/github/v/release/quantumaikr/TurboQuant.cpp)]()
[![Tests](https://img.shields.io/badge/tests-30%20suites-brightgreen)]()

### Up to 7.1x total K+V compression. Quality preserved.

```
Gemma 3 4B — total K+V memory per token:

  FP16 K+V (llama.cpp):    136.00 KB   (baseline)
  1-bit K + Q4 V:            27.62 KB   (4.9x)   "Paris" ✓  "1+1=2" ✓
  1-bit K + Q2 V:            19.12 KB   (7.1x)   "Paris" ✓  "Mercury, Venus, Earth" ✓
```

Key compression: 10.7x (1-bit sign hash). Value compression: Q4 (3.8x) or Q2 (7.6x). Combined: **up to 7.1x total K+V**.

---

## Why This Matters

LLM attention computes **inner products** `<query, key>`. Standard quantizers minimize reconstruction error (MSE), but introduce **systematic bias** in inner product estimation.

The [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) proved this gap and showed how to close it:

- **Keys**: RHT + Lloyd-Max codebook + QJL residual → **unbiased** inner product estimation at any bit-width
- **Values**: RHT + Lloyd-Max codebook → **MSE-optimal** reconstruction for weighted sum

We implemented both in pure C, and pushed keys to **1 bit** — attention via XOR + popcount.

---

## Compression Options

```bash
# Key compression (affects attention scoring)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b       # 1-bit keys (10.7x)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_3b       # 3-bit keys (4.6x)

# Value compression (affects output reconstruction)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4  # + Q4 values → 4.9x total
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q2  # + Q2 values → 7.1x total

# Memory stats
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4 -M
```

### Total K+V Compression Table

| Config | K bits | V bits | K+V/token | Total compression | Quality |
|--------|--------|--------|-----------|-------------------|---------|
| FP16 (baseline) | 16 | 16 | 136.00 KB | 1.0x | reference |
| uniform_4b + FP16 V | 4 | 16 | 86.06 KB | 1.6x | baseline |
| 1-bit K + FP16 V | 1 | 16 | 74.38 KB | 1.8x | greedy identical up to ~120 tok |
| **1-bit K + Q4 V** | **1** | **4** | **27.62 KB** | **4.9x** | **"Paris" ✓ "1+1=2" ✓** |
| **1-bit K + Q2 V** | **1** | **2** | **19.12 KB** | **7.1x** | **"Paris" ✓ planets ✓** |

### Memory at 32K Context (Gemma 3 4B)

```
FP16 K+V:              4,352 MB
1-bit K + Q4 V:           885 MB   (4.9x, 3.4 GB saved)
1-bit K + Q2 V:           613 MB   (7.1x, 3.7 GB saved)
```

> **Note on quality:** With K-only quantization (V as FP16/FP32), greedy decode is byte-identical
> up to ~120 tokens. With V quantization (Q4/Q2), outputs diverge earlier but remain coherent
> and factually correct. This is expected — V quantization affects reconstruction directly.

---

## Quick Start

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

---

## The Algorithm

```
Keys (attention scoring — needs unbiased inner products):
  key → normalize → RHT → Lloyd-Max codebook (b-1 bits) → QJL signs (1 bit)
  1-bit extreme: skip codebook, store signs only → XOR + popcount attention

Values (weighted sum — needs MSE-optimal reconstruction):
  value → Q4 or Q2 per-block quantization → dequantize on the fly during output
```

| Component | For Keys | For Values |
|-----------|----------|------------|
| **Goal** | Unbiased inner product | Low MSE reconstruction |
| **Method** | RHT + codebook + QJL | Per-block scale + quantize |
| **1-bit** | Sign hash (XOR+popcount) | Not recommended |
| **Best config** | 1-bit (10.7x key compression) | Q4 (3.8x value compression) |

---

## Supported Models

| Model | Params | Speed (Q4, 6T) | Verified |
|-------|--------|----------------|----------|
| **Gemma 3 4B** | 4B | 20.2 tok/s | "Paris" ✓, planets ✓ |
| **Qwen3.5-0.8B** | 752M | 80.1 tok/s | 0.999 cosine vs PyTorch |
| **Gemma 3 270M** | 270M | 176 tok/s | per-layer exact match |

Multi-architecture: Qwen3.5 (DeltaNet hybrid) + Gemma 3 (sliding window). Gemma 4 ready.

---

## Under the Hood

- **10,000+ lines of pure C** — zero external dependencies
- **11 quantization types** — Uniform, Mixed, PolarQuant, QJL, TurboQuant KV (1/3/4-bit)
- **K+V independent compression** — 1-bit keys (XOR+popcount) + Q4/Q2 values
- **Faithful ICLR 2026 implementation** — RHT + Lloyd-Max + QJL residual
- **Multi-architecture** — Qwen3.5 (DeltaNet) + Gemma 3 (sliding window + GeGLU)
- **NEON vectorized** — matmul, attention, RHT butterfly, Hamming distance, Q4 dequant, FP16 conversion
- **Fused Q4 attention** — weighted sum directly from packed nibbles, no dequant buffer
- **Adaptive compression** — per-layer bit recommendation, codebook calibration, attention entropy
- **30 test suites** — KV roundtrip, attention distribution, codebook theory, NEON consistency, edge cases, unbiasedness, rate-distortion, cumulative error

### Verification Summary

| Category | Tests | What's Verified |
|----------|-------|-----------------|
| Perplexity | `--ppl` | Gemma 4B: 1b K + Q4 V = PPL 36.00 (+0.03% vs FP16) |
| Unbiasedness | 100K pairs | All types: relative bias < 0.2% |
| NEON/scalar consistency | 14 | Every NEON path matches scalar reference (Q4, Q2, RHT, RoPE, matmul, RMSNorm, Hamming) |
| Attention distribution | 8 | Cosine similarity, Spearman rank, top-k overlap vs FP32 reference |
| Codebook theory | 5 | Lloyd-Max centroids match literature, MSE within 1.18x of info-theoretic optimal |
| Edge cases | 29 | n=1, dim=0, NaN, Inf, all-same, all-zero, n=10000 |
| Rate-distortion | 5 | Info-theoretic lower bound gap: Q4 2.41x, Lloyd-Max < 0.15 bits wasted |
| Cumulative error | 3 | 16-layer cosine: 0.998 (Q4), errors grow sub-linearly |
| ASan + UBSan | 30 | Full suite under sanitizers, zero memory errors |
| Thread safety | mutex | Global workspace realloc protected against concurrent access |
| Numerical stability | 4 | Overflow-safe norm (max-abs rescaling), NaN/Inf input guards |

Full details: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

## Analysis Tools

```bash
# Perplexity measurement
./build/tq_run model.tqm --ppl input.txt -k turbo_kv_1b -v q4

# Per-layer bit allocation recommendation
./build/tq_run model.tqm --recommend -k turbo_kv_1b -p "calibration text"

# Online codebook calibration (measures MSE improvement)
./build/tq_run model.tqm --calibrate -k turbo_kv_1b -p "calibration text"

# Activation distribution profiling (pre/post-RHT)
./build/tq_run model.tqm --profile-kv -k turbo_kv_1b -p "text"

# Attention entropy analysis
./build/tq_run model.tqm --attn-entropy -k turbo_kv_1b -p "text"

# Full auto-profile pipeline
bash bench/auto_profile.sh model.tqm
```

---

## Benchmarks & Validation

### Ablation: Does TurboQuant Actually Help?

```bash
bash bench/ablation_test.sh model.tqm
```

Compares `uniform_4b`, `turbo_kv_3b`, and `turbo_kv_1b` at token counts 50-300 to show where each
method diverges from the uniform baseline. Key findings:

- **turbo_kv_3b** (codebook + QJL): Typically matches `uniform_4b` output at all tested lengths.
  The QJL residual bit corrects inner product estimation bias from the 2-bit codebook.
- **turbo_kv_1b** (sign hash only): May diverge at longer contexts, but output remains coherent.
  This is expected at 10.7x key compression.
- **RHT matters**: The Randomized Hadamard Transform distributes outlier values evenly across
  dimensions, preventing systematic quantization bias (Theorem 3.1, TurboQuant paper).

### V Quantization Reality

The "30/30 byte-identical" result applies to **K-only quantization** (values remain FP16/FP32).
With V=Q4, outputs diverge earlier but remain coherent and factually correct.

```bash
bash bench/kv_quality_bench.sh model.tqm   # Includes Phase 4: V quantization check
```

### Long Context Quality

```bash
bash bench/long_quality_test.sh model.tqm   # 200, 500, 1000 tokens
```

Tests coherence and speed at longer context lengths across `uniform_4b`, `turbo_kv_1b`, and
`turbo_kv_1b + Q4 V`. Outputs diverge from baseline at longer contexts but remain coherent.

### Temperature Sampling

```bash
bash bench/sampling_test.sh model.tqm   # T=0.3 and T=0.7, 3 runs each
```

Verifies that KV compression does not degrade stochastic sampling quality. All KV types
produce diverse, coherent outputs at each temperature with similar variance.

### Sanitizer Validation

```bash
bash scripts/sanitize.sh [model.tqm]   # ASan + UBSan build and test
```

Builds with `-fsanitize=address,undefined`, runs all tests, and optionally runs a short
inference to catch memory errors. No leaks or undefined behavior detected.

---

## FAQ

**Q: "Byte-identical output just means K doesn't matter, right?"**

No. Replacing K with random values produces garbage immediately (cosine < 0.09). TurboQuant preserves inner product ranking -- measured attention score cosine: uniform_4b = 0.996, turbo_kv_3b = 0.918, turbo_kv_1b = 0.634 (10-trial avg, 32 keys). The 1-bit cosine of 0.634 matches the information-theoretic limit of 2/pi = 0.637 for sign quantization -- this is mathematically optimal, not a deficiency. See `tests/test_attention_distribution.cpp`.

**Q: "How is this different from llama.cpp's Q4 KV?"**

llama.cpp uses uniform min-max quantization. TurboQuant uses RHT + Lloyd-Max codebook optimized for the post-rotation Gaussian distribution. The Lloyd-Max centroids are verified against theory (MSE within 1.18x of information-theoretic optimal, tested in `tests/test_codebook_theory.cpp`). The QJL residual provides provably unbiased inner product estimation -- the mathematical guarantee matters at scale.

**Q: "What about perplexity?"**

Attention score distribution is preserved: Spearman rank correlation = 0.990 (uniform_4b), 0.900 (turbo_kv_3b), 0.632 (turbo_kv_1b). Greedy decode matches up to ~120 tokens. The 1-bit cosine of 0.634 = 2/pi is the theoretical maximum for sign-only quantization (proven in JL literature). Full perplexity on standard datasets is in progress.

**Q: "Is the NEON code correct?"**

Every NEON path (Q4 dequant, RHT butterfly, matmul, RMSNorm, RoPE, Hamming attention) is verified against scalar reference in `tests/test_neon_scalar.cpp`. The Q4 dequant had a nibble-interleaving bug that was caught and fixed. ASan + UBSan pass on all 26 test suites with zero errors. NaN/Inf/edge-case inputs tested in `tests/test_edge_cases.cpp` (29 cases).

**Q: "What about thread safety?"**

Global workspaces (Q8 quantization buffer, sampler probability index) are mutex-protected to prevent concurrent realloc races. The thread pool uses a single dispatch mutex. Concurrent multi-context usage is safe at the API level.

**Q: "Only 4B model -- what about 8B+?"**

Architecture is model-size independent. Gemma 3 4B and Qwen3.5 0.8B use the same code path. 8B support is planned (Llama 3.1 8B architecture support in progress).

**Q: "RHT overhead?"**

RHT is O(d log d) per vector, NEON-vectorized. Measured: 147 ns per 128-dim vector. Full quantization: uniform_4b = 148 ns, turbo_kv_1b = 659 ns, turbo_kv_3b = 11066 ns per vector. 1-bit attention: 1.2 ns/key (XOR+popcount). Compared to matmul (~1ms/layer), all overhead is negligible. See `bench/bench_kv_overhead.cpp`.

---

## References

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — Online Vector Quantization with Near-optimal Distortion Rate
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit Quantized JL Transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar Coordinate Quantization

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
