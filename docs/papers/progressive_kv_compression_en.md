# Progressive KV Cache Compression in a Single-Header C Engine: Independent Validation, Negative Results, and Practical Deployment

**Authors:** QuantumAI Research

**Abstract.** We report an independent empirical validation of recency-based KV cache compression — keeping the last 128 tokens at FP32 while compressing the rest to 4-bit — in a minimal, single-header C inference engine. On Llama 3.2 3B at 3,970 tokens, this achieves PPL −0.1% vs. FP32 at 3× compression and +13% speed. While the recency-window approach has been explored concurrently by KVTC [7] and PM-KVQ [8], we contribute: (1) a demonstration that RHT normalization eliminates the need for per-layer calibration (max ~0.9% benefit from optimal per-layer allocation), (2) an honest negative result showing that 2-bit compression with a 512-token window, which appeared Pareto-dominant at 957 tokens (53% FP32), collapses to +36.7% PPL at honest evaluation lengths — an artifact we publicly retracted, (3) a context-length invariance measurement showing the same 3.2 pp improvement at both 957 tokens (13.4% FP32) and 3,970 tokens (3.2% FP32), and (4) a complete open-source implementation in 16K lines of C with zero dependencies, installable via `pip install quantcpp`. We also report an O(n log n) BPE tokenizer fix that was necessary to enable honest long-context evaluation.

---

## 1. Introduction

KV cache compression is an active area of research, with methods ranging from uniform quantization (llama.cpp Q4_0/Q8_0), per-channel calibration (KIVI [1], KVQuant [2]), attention-saliency-based adaptive precision (ZipCache [3]), to recent transform-coding approaches (KVTC [7]). A common finding across this literature is that **recent tokens require higher precision** — KVTC [7] keeps the 128 most recent tokens uncompressed, PM-KVQ [8] progressively lowers bit-width for older entries, and ZipCache [3] assigns more bits to attention-salient tokens.

We arrived at the same finding independently through a Karpathy-loop optimization process on quant.cpp, a single-header C inference engine. Rather than claiming novelty for the recency-window approach itself, we contribute the following:

### Contributions

1. **RHT eliminates per-layer calibration.** We show that after Random Hadamard Transform normalization, post-RHT kurtosis across 28 layers of Llama 3.2 3B ranges only 2.64–3.81 (mean 3.04, std 0.25). The theoretical maximum benefit from optimal per-layer bit allocation is ~0.9% PPL. This means the optimization landscape for KV compression is fundamentally **temporal** (which tokens), not **spatial** (which layers) — a finding that simplifies method design.

2. **Honest negative result.** We initially claimed that 2-bit + 512-token FP32 window "Pareto-dominates" flat 4-bit. This was measured at 957 tokens where 53.5% of tokens were FP32 — misleading. At 3,970 tokens (12.9% FP32), 2-bit PPL degraded to +36.7%. We retracted the claim and report it here as a cautionary example of short-context evaluation artifacts.

3. **Context-length invariance.** We measure the quality improvement from the 128-token window at two scales:
   - 957 tokens (k128 = 13.4% FP32): +3.8% → +0.6% (improvement: 3.2 pp)
   - 3,970 tokens (k128 = 3.2% FP32): +3.1% → −0.1% (improvement: 3.2 pp)
   
   The same 3.2 percentage point improvement with 4× less FP32 ratio suggests the result extends to arbitrary context lengths.

4. **Practical deployment.** The entire method — including RHT, Lloyd-Max codebooks, progressive window, infinite scrollback (automatic context shift), and KV cache persistence (save/load to disk) — is implemented in a single C header file (16K LOC, 654 KB) with zero dependencies, distributed via PyPI (`pip install quantcpp`) and as a 193 KB WASM binary.

---

## 2. Related Work

### 2.1 Uniform KV Quantization

llama.cpp offers Q4_0 and Q8_0 KV types with per-block min-max scaling, achieving ~2× compression at +10.6% PPL (Q4_0). KIVI [1] applies asymmetric 2-bit quantization with per-channel key and per-token value precision. KVQuant [2] adds pre-RoPE quantization, non-uniform per-layer datatypes, and per-vector dense-and-sparse quantization, achieving <0.1% PPL at 3-bit.

### 2.2 Non-Uniform Per-Token Precision

**ZipCache** [3] identifies attention-salient tokens and assigns them higher bit-width, achieving 4.98× compression at 0.38% accuracy drop. This is the first per-token adaptive approach, using saliency (attention score magnitude) rather than recency as the allocation criterion.

**KVTC** [7] (NVIDIA, ICLR 2026) keeps the 128 most recent tokens and 4 "attention sink" tokens uncompressed while applying PCA + entropy coding to the rest. This is structurally the closest prior work to our method — the 128-token recency window is identical.

**PM-KVQ** [8] (Tsinghua, 2025) designs a progressive quantization strategy that gradually lowers bit-width of older KV cache entries with block-wise memory allocation.

**"More Tokens, Lower Precision"** [9] (EMNLP 2025) demonstrates that storing 4× more tokens at 4-bit outperforms 1× tokens at 16-bit, directly supporting the temporal compression thesis.

### 2.3 Transform-Based Normalization

**HIGGS** [4] introduces RHT + MSE-optimal grid quantization for weight compression. **TurboQuant** [5] applies the same pattern to KV caches with a 1-bit QJL residual. Our implementation builds on TurboQuant's RHT + Lloyd-Max structure, with the QJL residual dropped through ablation (it contributed ~zero to attention scores).

---

## 3. Method

### 3.1 Progressive KV Compression

We partition KV cache tokens into two tiers using a single parameter $W$:
- **Hot tier** (last $W$ tokens): Keys at FP32
- **Cold tier** (all other tokens): Keys at 4-bit (RHT + 16-level Lloyd-Max codebook)
- **All tiers**: Values at FP16

The additional memory for the hot tier at $W$=128, Llama 3.2 3B: 14.7 MB (0.6% of the 32K cold-tier cache).

### 3.2 Attention-Aligned Rationale

The total weighted quantization error is $E = \sum_t \alpha_t \cdot \text{MSE}(K_t, \hat{K}_t)$, where $\alpha_t$ is the attention weight at position $t$. Causal attention concentrates $\alpha$ on recent tokens, so allocating full precision to the high-$\alpha$ region minimizes $E$. This is the same rationale behind KVTC [7] and ZipCache [3]'s saliency-based allocation.

Our contribution is not the rationale itself but the empirical measurement of its **context-length invariance** and the **RHT-based spatial analysis** that eliminates the per-layer dimension.

### 3.3 RHT Eliminates Per-Layer Variation

Post-RHT kurtosis across 28 layers: mean 3.04, std 0.25 (range 2.64–3.81), compared to pre-RHT range 4.13–20.62. The variance of $\log_2(\sigma)$ across layers is 0.0177 → theoretical max MSE improvement from per-layer allocation: ~1.8% → ~0.9% PPL. This is below measurement noise.

**Implication:** Methods that invest complexity in per-layer calibration (KIVI, KVQuant) gain little benefit when RHT normalization is applied. The optimization landscape is purely temporal.

---

## 4. Experiments

**Model:** Llama 3.2 3B Instruct (Q8_0 weights)
**Hardware:** Apple M1 Pro, 16 GB RAM, 8 threads, CPU-only
**Evaluation:** Teacher-forced perplexity on English text (957 tokens and 3,970 tokens)
**Tokenizer:** Custom BPE with O(n log n) heap-based merge

### 4.1 Progressive Compression Quality

**At 3,970 tokens** (k128 = 3.2% FP32 — honest condition):

| Configuration | PPL | vs FP32 |
|---|---:|---:|
| FP32 (baseline) | 19.41 | — |
| **4-bit + k128** | **19.39** | **−0.1%** |
| 4-bit flat | 20.02 | +3.1% |
| 2-bit + k512 | 26.53 | +36.7% |

### 4.2 Context-Length Invariance

| Eval Length | k128 FP32 Ratio | Improvement vs Flat |
|---:|---:|---:|
| 957 tokens | 13.4% | 3.2 pp |
| 3,970 tokens | 3.2% | 3.2 pp |

### 4.3 Window Size Saturation

| $W$ | PPL (957 tok) | vs FP32 |
|---:|---:|---:|
| 0 | 14.08 | +3.8% |
| 64 | 13.71 | +1.1% |
| **128** | **13.64** | **+0.6%** |
| 256 | 13.64 | +0.6% |

### 4.4 Memory and Speed (32K context)

| Config | KV Memory | Speed |
|---|---:|---:|
| FP32 | 7.17 GB | 6.9 tok/s |
| 4-bit + k128 | 2.33 GB | 7.8 tok/s (+13%) |

### 4.5 Negative Result: 2-bit Compression

At 957 tokens, 2-bit + k512 showed PPL +4.3% (k512 = 53.5% FP32). We initially claimed this "Pareto-dominated" flat 4-bit. At 3,970 tokens (k512 = 12.9% FP32), PPL collapsed to +36.7%.

**Root cause:** At 957 tokens, the 512-token FP32 window covered more than half the evaluation, masking the 2-bit degradation. This is a general hazard of evaluating KV compression at short context lengths with large FP32 windows.

### 4.6 Layer-Adaptive Analysis (Negative Result)

Post-RHT kurtosis variation is insufficient for per-layer adaptation to provide meaningful benefit (~0.9% max). This is a positive finding for method simplicity.

---

## 5. Engineering Contributions

### 5.1 Single-Header Implementation

The complete method — RHT, Lloyd-Max codebooks, progressive window, infinite scrollback, KV persistence, NEON/AVX2 SIMD kernels — is implemented in `quant.h` (16K LOC, 654 KB) with zero dependencies beyond libc.

### 5.2 O(n log n) BPE Tokenizer

The standard BPE merge algorithm is O(n²). For GPT-style byte-level BPE, a 17K-character text produces ~17K initial tokens, making naive merging impractical (~289M operations). We implemented a max-heap with lazy deletion, reducing merge complexity to O(n log n). This was necessary to enable the 3,970-token evaluation that caught the 2-bit artifact.

### 5.3 Distribution

- **PyPI:** `pip install quantcpp` (pre-built wheels for Linux x86_64/aarch64, macOS arm64)
- **WASM:** 193 KB browser demo with IndexedDB model caching
- **Model registry:** Auto-download from HuggingFace (`Model.from_pretrained("Llama-3.2-1B")`)

---

## 6. Discussion

### 6.1 Relationship to KVTC

KVTC [7] uses the same 128-token sliding window but adds PCA dimensionality reduction and entropy coding for the compressed region. Our approach is simpler (binary FP32/4-bit) and achieves comparable quality. We view this as convergent evidence that recency-based precision allocation is a robust principle.

### 6.2 Self-Correction as Methodology

Our project maintains a public correction log (10 self-found corrections, 0 external reports). The 2-bit Pareto claim (#10) was caught by our own evaluation infrastructure improvements (BPE fix → longer eval → honest measurement). We believe systematic self-validation — measuring, doubting, re-measuring at harder conditions — is as important as the algorithmic contribution.

### 6.3 Limitations

1. Single model (Llama 3.2 3B). Multi-model validation needed.
2. CPU-only speed measurements. GPU behavior may differ.
3. Maximum evaluated context: 3,970 tokens. 32K+ validation pending.
4. V cache not progressively compressed (FP16 throughout).

---

## 7. Conclusion

We provide independent empirical evidence that recency-based KV cache precision allocation — keeping 128 recent tokens at FP32 — achieves FP32 quality at 3× compression, confirming findings from KVTC [7] and PM-KVQ [8] in a simpler setting. Our additional contributions — the RHT spatial analysis, the retracted 2-bit result, the context-length invariance measurement, and the single-header open-source implementation — complement the existing literature with practical validation and honest methodology.

---

## References

[1] Z. Liu et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." ICML 2024. arXiv:2402.02750.

[2] C. Hooper et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." NeurIPS 2024. arXiv:2401.18079.

[3] Y. He et al. "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification." NeurIPS 2024. arXiv:2405.14256.

[4] V. Malinovskii et al. "HIGGS: Pushing the Limits of Large Language Model Quantization via the Linearity Theorem." NAACL 2025. arXiv:2411.17525.

[5] A. Zandieh et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. arXiv:2504.19874.

[6] G. Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024. arXiv:2309.17453.

[7] KVTC. "KV Cache Transform Coding." ICLR 2026. arXiv:2511.01815.

[8] Liu et al. "PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs." 2025. arXiv:2505.18610.

[9] "More Tokens, Lower Precision: Towards the Optimal Token-Precision Trade-off in KV Cache Compression." EMNLP 2025. arXiv:2412.12706.

---

**Reproducibility.** All code: [github.com/quantumaikr/quant.cpp](https://github.com/quantumaikr/quant.cpp). Install: `pip install quantcpp`. Benchmark artifacts in `bench/results/`.

**Correction log.** 10 self-corrections documented in [CHANGELOG.md](https://github.com/quantumaikr/quant.cpp/blob/main/CHANGELOG.md). Correction #10 (2-bit Pareto retraction) is discussed in Section 4.5.
