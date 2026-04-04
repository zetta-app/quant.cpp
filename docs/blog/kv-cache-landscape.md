# KV Cache Compression: A Practical Guide to Fitting More Context in Less Memory

*2026-04-05 -- quantumaikr/quant.cpp*

---

Someone on Reddit asked how quant.cpp relates to Titans, MLA, and the dozen other approaches to managing KV cache memory. It is a fair question. The field is moving fast and papers use different terminology for overlapping ideas. This post is our attempt at a practical map of the landscape -- what works, what the tradeoffs are, and where each approach fits.

## The problem: KV cache is the memory wall

When a transformer generates text, it stores a key and value vector for every token, at every layer, at every attention head. This is the KV cache. It exists so the model does not recompute attention over the entire history at each step.

The memory cost scales linearly with context length, and the constants are large. For Llama 3.1 8B at FP16 precision:

- 32 layers x 8 KV heads x 128 head_dim x 2 (K+V) x 2 bytes = 131 KB per token
- At 32K context: **4.0 GB** of KV cache alone
- The Q4_K_M model weights occupy roughly 4.5 GB

So at 32K context, the KV cache is almost as large as the model. On a 16 GB laptop, this means you hit OOM well before the model's theoretical context window. On a 8 GB machine, you may not get past 8K tokens. The model can handle longer context. Your hardware cannot store it.

This is the bottleneck that every approach below is trying to solve.

## Three categories of solutions

The KV cache memory problem has attracted a lot of research. The approaches cluster into three distinct categories, each with fundamentally different tradeoffs.

### 1. Eviction: drop tokens you probably do not need

**Key papers:** StreamingLLM (MIT, 2023), H2O (2023), Scissors (2024), SnapKV (2024)

The idea: not all tokens contribute equally to attention. Some positions (typically the first few tokens and the most recent tokens) receive most of the attention weight. Eviction methods identify "unimportant" tokens and remove them from the cache entirely.

StreamingLLM keeps a fixed window of recent tokens plus a small set of "attention sink" tokens from the beginning of the sequence. H2O (Heavy-Hitter Oracle) dynamically evicts tokens that receive the least cumulative attention. Scissors and SnapKV refine the selection criteria with different importance metrics.

**The upside:** dramatic memory savings. You can maintain a fixed-size cache regardless of sequence length. StreamingLLM can process infinite-length streams with constant memory.

**The downside:** information loss is permanent. If the model needs to refer back to an evicted token -- because the user asks "what did you say about X earlier" or a reasoning chain depends on a detail from paragraph three -- that information is gone. There is no way to reconstruct it. Eviction works well for streaming and summarization where older context fades in importance. It works poorly for tasks that require random access to the full history, like multi-turn conversation or long-document QA.

### 2. Architecture: redesign the model to use less KV memory

**Key papers:** Titans (Google, 2025), MLA/DeepSeek-V3 (2024), GQA (Ainslie et al., 2023), MQA (Shazeer, 2019), Linear Attention (Katharopoulos et al., 2020), Mamba/SSMs (Gu & Dao, 2023)

These approaches change the model architecture itself so that KV memory grows more slowly -- or not at all -- with sequence length.

**Grouped Query Attention (GQA)** shares KV heads across multiple query heads. Llama 3 uses GQA with 8 KV heads for 32 query heads, cutting KV memory by 4x compared to standard multi-head attention. **Multi-Query Attention (MQA)** takes this further with a single KV head shared across all queries, though quality often suffers.

**Multi-head Latent Attention (MLA)**, used in DeepSeek-V3, compresses KV into a low-rank latent space. Instead of storing full key and value vectors, it stores a compressed representation that is expanded at attention time. This achieves KV compression ratios comparable to MQA while preserving quality closer to full MHA.

**Titans** goes a different direction entirely: it augments the transformer with a learned memory module that compresses past context into a fixed-size state. The model itself learns what to remember and what to forget, rather than having the decision imposed by a fixed heuristic.

**State-space models** like Mamba sidestep the KV cache entirely by replacing attention with a recurrence that maintains a fixed-size hidden state. No KV cache, no memory growth.

**The upside:** these are the best results in the field. When you design the model around efficient memory from the start, you avoid the compression-quality tradeoff entirely. MLA and GQA are already standard in production models.

**The downside:** you need to train (or retrain) the model with the new architecture. You cannot take an existing Llama 3 checkpoint and retroactively add MLA or Titans memory. This makes architectural approaches inaccessible if you are running existing open-weight models, which is most of the local LLM community.

### 3. Compression: keep all tokens, store them in fewer bits

**Key papers:** TurboQuant (ICLR 2026), KIVI (2024), KVQuant (2024), Coupled Quantization (2024), QJL (AAAI 2025)

Compression methods keep every token in the cache but store the key and value vectors in a lower-precision format. Instead of FP16 (16 bits per element), you store 4-bit, 3-bit, or even 2-bit representations. The full sequence is preserved -- nothing is evicted -- but each element uses fewer bits.

**KIVI** applies per-channel quantization with different bit widths for keys (2-bit) and values (2-bit), using residual quantization to improve accuracy. **KVQuant** uses per-channel quantization with non-uniform codebooks calibrated on the distribution of each channel. **Coupled Quantization** jointly optimizes the key and value quantization to minimize the impact on attention scores. **TurboQuant** combines polar coordinate quantization for keys with QJL sign hashing and delta compression.

**The upside:** works on any existing model. Download a GGUF, enable compression, get 4-7x longer context. No retraining, no architecture changes, no fine-tuning. The model file is untouched.

**The downside:** there is a quality floor. You cannot compress to arbitrarily few bits without degradation. In our testing, 4-bit is effectively lossless, 3-bit with delta compression costs about +1.3% perplexity, and anything below 3-bit degrades rapidly. The compression ratio ceiling is around 8-10x before quality becomes unacceptable. Eviction and architectural methods can achieve higher effective ratios for their respective use cases.

## Comparison table

| Approach | Method | Memory Savings | Quality Impact | Requires Retraining? | Works on Existing Models? |
|:---------|:-------|:---------------|:---------------|:---------------------|:--------------------------|
| StreamingLLM | Eviction (sink + window) | Fixed cache size | Loses distant context | No | Yes |
| H2O | Eviction (heavy hitter) | 5-10x | Loses low-attention tokens | No | Yes |
| GQA | Shared KV heads | 4x (Llama 3) | Minimal | Yes (at pretraining) | Only GQA models |
| MLA (DeepSeek) | Low-rank KV latent | ~5x | Minimal | Yes (at pretraining) | Only MLA models |
| Titans | Learned memory module | Sublinear growth | Minimal | Yes (at pretraining) | Only Titans models |
| Mamba/SSMs | No KV cache | No KV at all | Different tradeoffs | Yes (at pretraining) | Only SSM models |
| KIVI | Per-channel 2-bit K+V | 8x | Moderate (+2-5% PPL) | No | Yes |
| KVQuant | Non-uniform codebooks | 4-8x | Low | No (needs calibration) | Yes |
| **quant.cpp** | Per-block min-max + delta | **4-8.5x** | **+0.0% at 4b, +1.3% at 3b** | **No** | **Yes** |

## Where quant.cpp fits

quant.cpp is a compression approach. It implements several techniques from the TurboQuant paper (ICLR 2026) in pure C:

**Per-block quantization.** Keys are quantized in 128-element blocks, each with its own min/max scale factors. This is critical for quality. Keys have high kurtosis -- outliers that destroy per-tensor or per-channel quantization. Per-block isolation means an outlier in one block does not affect any other. This is why quant.cpp achieves +0.0% PPL at 4-bit while llama.cpp's per-tensor Q4_0 KV gives +10.6%.

**Delta compression.** Adjacent key vectors in the same layer/head are highly correlated (average cosine similarity ~0.70). Like P-frames in video codecs, quant.cpp stores full-precision anchor keys every 64 tokens and encodes only the difference for tokens in between. This lets 3-bit deltas achieve what would require 4 bits without delta encoding.

**QK-norm auto-detection.** Models like Gemma 4 normalize keys to the unit sphere via QK-norm. These keys have extremely sparse distributions that quantization destroys (cosine drops to 0.62 at 4-bit). quant.cpp detects QK-norm at load time and automatically keeps keys in FP32 while compressing only values -- still achieving 3.5x savings.

**Independent K/V treatment.** Keys and values have different statistical properties and different impacts on attention quality. quant.cpp applies different quantization methods to each, rather than the same scheme for both.

## When to use what

**Use eviction** when you know which tokens matter. Summarization, streaming chat where old messages lose relevance, or any task where the model primarily attends to recent context. StreamingLLM is the simplest and most battle-tested option.

**Use architectural approaches** when training from scratch or choosing a base model. If you are deciding between Llama 3 and DeepSeek-V3, MLA gives DeepSeek a structural advantage in KV memory. If Titans models ship as open weights, they will be the best option for long-context tasks. GQA is already standard -- every Llama 3 model has it built in.

**Use compression** when you want to extend context on existing models without any changes. You have a GGUF model that works well for your task, but you run out of memory at 8K tokens and need 32K. Compression gets you there. This is the category quant.cpp targets.

These approaches are also composable. GQA already reduces KV heads at the architecture level. Compression can stack on top: Llama 3's GQA gives 4x KV reduction, and quant.cpp's 4-bit compression gives another 4x, for a combined ~16x reduction with negligible quality loss. If Titans models ship as GGUF, quant.cpp could compress their (already smaller) KV cache further.

## References

- **TurboQuant** -- Zhang et al., "TurboQuant: Online KV Cache Quantization via Adaptive Polar Quantization," ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **QJL** -- Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead," AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **StreamingLLM** -- Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- **H2O** -- Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- **KIVI** -- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **KVQuant** -- Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," 2024. [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- **Titans** -- Behrouz et al., "Titans: Learning to Memorize at Test Time," Google, 2025. [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **MLA / DeepSeek-V3** -- Liu et al., "DeepSeek-V3 Technical Report," 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **GQA** -- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **PolarQuant** -- Zhang et al., "PolarQuant: Polar Coordinate Quantization for KV Cache Compression," AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)

---

*[quant.cpp](https://github.com/quantumaikr/quant.cpp) -- LLM inference with 7x longer context. Pure C, zero dependencies.*
