# Reddit Comment Responses — "Same 4 bits" post (2026-04-06)

Copy-paste ready.

---

## @Emotional-Breath-838 — "why is llama.cpp faster?"

Good question. Three reasons:

1. **Hand-tuned SIMD kernels.** llama.cpp has years of hand-optimized NEON/AVX2/AVX-512 assembly for every quantized matmul variant (Q4_K_M, Q8_0, IQ2, etc.). quant.cpp has NEON kernels for the common formats but relies on compiler autovectorization for the rest. This alone accounts for ~2x.

2. **Metal/CUDA GPU offload.** llama.cpp offloads the entire forward pass to GPU. quant.cpp has Metal shaders but GPU dispatch is still basic — most of the work stays on CPU. On Apple Silicon, this is the biggest gap.

3. **Code maturity.** llama.cpp has 250K+ LOC and hundreds of contributors optimizing hot paths. quant.cpp is 72K LOC — deliberately smaller, which means easier to read and embed, but fewer micro-optimizations.

The tradeoff is intentional. We optimized for **memory** (KV compression) and **simplicity** (embeddable, single header) rather than raw tok/s. For a 3B model on M1, quant.cpp does ~10 tok/s vs llama.cpp's ~30 tok/s — slower, but fast enough to read in real time. The advantage shows up when llama.cpp hits OOM at 50K context and quant.cpp keeps going to 350K.

That said, speed improvements are on the roadmap — better Metal offload and more SIMD kernels would close the gap significantly without sacrificing the simplicity.

---

## @audioen — "This is not even correct. llama.cpp Q4_0 is per-block..."

You're right on several points and I should correct the post.

**What I got wrong:** llama.cpp Q4_0 *is* per-block (32 elements per block, 1 FP16 scale), not per-tensor. And llama.cpp can apply separate quant types to K and V — that's not a quant.cpp-only feature. The original wording overstated the difference. I'll fix it.

**What is different:**

- **Block size**: Q4_0 uses 32-element blocks. quant.cpp uses 128-element blocks with both min and max (effectively Q4_1-style at wider blocks). The larger block amortizes scale overhead better (4.25 bits/element vs Q4_0's 4.5 or Q4_1's 5.0), but the quality difference comes more from the min-max vs zero-point approach on key distributions specifically.

- **Delta compression**: This is the part llama.cpp genuinely doesn't have. Storing `key[t] - key[t-1]` instead of absolute keys reduces the dynamic range by ~70%, which is why 3-bit works at +1.3% PPL where absolute 3-bit gives +62%. This is the novel contribution from the TurboQuant paper, not the 4-bit uniform quantization itself.

- **The PPL +10.6% number**: This was measured with Q4_0 on both K and V using the default llama.cpp KV quant path. You're right that Q8_0 K + Q4_0 V (or Q5_0 V) would be significantly better. I should benchmark that specific config and update the comparison to be fair.

Fair criticism. The honest comparison is: at the **same total bit budget**, quant.cpp's approach preserves more quality. But the original post made it sound like llama.cpp's quantization is fundamentally broken, which isn't true — it's just a different tradeoff with coarser granularity.

---

## @Look_0ver_There — "Q8_0 K + Q5_0 V best tradeoff"

That makes sense — keeping K at higher precision is exactly the right call since attention scores are more sensitive to key quantization error than value quantization error. Q8_0 K + Q5_0 V gives you ~1.6x compression with minimal quality loss.

quant.cpp's pitch at that point becomes: if 1.6x is enough, use llama.cpp — it's faster. If you need 4-7x (extending 50K context to 200K+), that's where 4-bit K + Q4 V and delta compression come in. Different operating points on the compression-quality curve.

I should add this nuance to the comparison. Thanks for bringing up the KV rotation work — haven't benchmarked against it yet.

---

## @Pixer--- — "llamacpp recently implemented rotating kv caching"

Yes, KV cache rotation (ring buffer) is a different but complementary approach. Rotation recycles old KV slots so the cache never grows beyond a fixed size — great for streaming/chat where old context can be dropped.

quant.cpp does something different: it keeps all tokens but stores them in fewer bits. So rotation saves memory by *evicting* old tokens, compression saves memory by *shrinking* all tokens.

You could combine both — rotate a compressed cache for maximum context. Haven't benchmarked against the rotation PR yet, but it's on the list. Thanks for bringing it up.

---

## @putrasherni — "for larger context, better to try quant.cpp?"

Depends on how much longer you need:

- **1.5-2x more context** → llama.cpp with Q8_0 K + Q5_0 V. It's faster and the quality tradeoff is minimal.
- **4-7x more context** (e.g. 50K → 350K on 16GB) → that's where quant.cpp helps. 4-bit K + Q4 V gives 3.8x at +0.0% PPL, delta 3-bit pushes to 4.3x at +1.3%.

If you're already running llama.cpp and just want a bit more room, their built-in KV quant is probably enough. If you're hitting hard OOM walls and need to push significantly further, give quant.cpp a try.

---

## @chimpera — "relationship to delta-compress-llm?"

No relationship. I'm not familiar with that project — just looked at the repo and it appears to be a different approach (applying delta compression to model weights rather than KV cache).

quant.cpp compresses the **KV cache** at runtime — the key and value vectors that accumulate during inference. The model weights themselves are loaded from standard GGUF files and used as-is. Delta compression in our case means storing `key[t] - key[t-1]` between adjacent tokens in the same attention head, not compressing the weight tensors.

The underlying idea (delta encoding of correlated vectors) is the same, but applied to completely different data.
