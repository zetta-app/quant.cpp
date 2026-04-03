# quant.cpp

Minimal C inference engine for local LLM. 33K LOC. Zero dependencies.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/quant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-33%20pass-brightgreen)]()

---

## 4x longer context, same hardware

Delta KV compression fits 4x more context into the same GPU/CPU memory with no quality loss.

| Hardware | Model | Before | After | Gain |
|----------|-------|--------|-------|------|
| 8GB Laptop | Llama 8B (Q4) | 16K tokens | 61K tokens | 3.8x |
| 16GB Mac Air | SmolLM2 1.7B | 78K tokens | 298K tokens | 3.8x |
| 24GB RTX 3090 | Llama 8B (Q4) | 147K tokens | 559K tokens | 3.8x |

```bash
./quant model.gguf -p "hello" --compress
```

---

## Why quant.cpp

|  | quant.cpp | llama.cpp |
|--|-----------|-----------|
| Codebase | 33K LOC, pure C | 250K+ LOC, C++ |
| KV compression quality | PPL -3.2% (better than FP32) | PPL +10.6% |
| Dependencies | zero (libc/libm only) | - |
| Design goal | readable, hackable | feature-complete |

Same model (SmolLM2 1.7B), same benchmark. Their Q4_0 KV degrades quality. Ours improves it.

---

## Quick Start

```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run inference
./build/quant model.gguf -p "hello"

# With KV compression (4-bit K + Q4 V, 3.8x)
./build/quant model.gguf -p "hello" -k uniform_4b -v q4

# With delta compression (3-bit K + Q4 V, 4.3x)
./build/quant model.gguf -p "hello" -k uniform_3b -v q4 --delta

# Measure perplexity
./build/quant model.gguf --ppl input.txt -k uniform_4b -v q4
```

---

## KV Cache Compression

### Compression modes

| Config | Compression | PPL vs FP32 | Use case |
|--------|-------------|-------------|----------|
| delta + 3b K + Q4 V | ~4.3x | -3.2% | Maximum compression |
| delta + 4b K + Q4 V | ~3.8x | -12.2% | Best quality |
| uniform 4b K + Q4 V | 3.8x | -7.8% | Simple, no delta overhead |
| uniform 4b K + FP16 V | 1.6x | +0.0% | Lossless |

### How delta compression works

Standard KV caching stores each key vector as-is. Delta compression stores the *difference* between adjacent keys -- like video P-frames vs I-frames.

Adjacent keys in a transformer differ by ~30% of their absolute range. This smaller dynamic range means 3-bit quantization is enough. Without delta, 3-bit gives PPL +62%. With delta, the same 3-bit gives PPL -3.2%.

Every 64 tokens, an absolute key is stored as an FP32 I-frame to anchor accumulated deltas and prevent drift.

### Full PPL results (SmolLM2 1.7B, 999 tokens)

| Config | PPL | vs FP32 | Notes |
|--------|-----|---------|-------|
| FP32 baseline | 14.58 | -- | reference |
| delta + 4b K + Q4 V | 12.80 | -12.2% | best quality |
| delta + 3b K + Q4 V | 14.11 | -3.2% | best compression |
| uniform 4b K + Q4 V | 13.44 | -7.8% | proven |
| uniform 3b K + Q4 V (no delta) | 23.62 | +62% | delta is essential |

### Cross-model validation (4b K + Q4 V)

| Model | PPL delta |
|-------|-----------|
| SmolLM2 1.7B | -1.6% |
| Qwen3.5 0.8B | +0.9% |
| Qwen3.5 4B | +0.6% |

---

## Supported Models

| Model | Architecture | Params | KV Verified |
|-------|-------------|--------|-------------|
| SmolLM2-1.7B | Llama | 1.7B | PPL -1.6% |
| Qwen3.5-0.8B | Qwen3.5 (DeltaNet) | 752M | PPL +0.9% |
| Qwen3.5-4B | Qwen3.5 (DeltaNet) | 4B | PPL +0.6% |
| Qwen3.5-35B-A3B | Qwen2-MoE | 35B (3B active) | 4-bit K verified |
| Gemma 3 270M | Gemma 3 | 270M | 4-bit K verified |
| Gemma 4 E2B | Gemma 4 | 2B | WIP |
| Gemma 4 26B-A4B | Gemma 4 MoE | 26B (4B active) | WIP |

5 architectures: Llama, Gemma 3, Gemma 4, Qwen3.5 (DeltaNet), Qwen2-MoE.

---

## Backends

| Backend | Platform | Status |
|---------|----------|--------|
| NEON | ARM CPU | Production |
| AVX2 | x86 CPU | Production |
| Metal | Apple Silicon | Verified |
| CUDA | NVIDIA GPU | Compiles |
| Vulkan | Cross-platform | Compiles |
| ROCm/HIP | AMD GPU | Compiles |

---

## FAQ

**How does delta compression work?**

Instead of storing each key vector directly, delta mode stores `key[t] - reconstruct(key[t-1])`. Adjacent keys in a transformer are highly correlated, so the deltas have ~30% the dynamic range of absolute keys. This enables 3-bit quantization with no quality loss. Every 64 tokens, a full-precision I-frame is stored to prevent drift accumulation.

**How is this different from llama.cpp?**

quant.cpp is a standalone inference engine (33K LOC, pure C) -- not a fork or wrapper. The key difference in KV compression: llama.cpp's Q4_0 KV gives PPL +10.6% on SmolLM2 1.7B. quant.cpp's 4-bit K gives PPL +0.0% on the same model. We quantize K and V independently with type-appropriate methods.

**What about sub-3-bit?**

We tested extensively: 2-bit with delta, sub-block scaling, multi-hash sign quantization, error feedback, NF2 codebooks, online SVD, and more. None achieved acceptable quality. The fundamental barrier: per-step cosine similarity of 0.997 compounds to 0.885 after 200 steps. 3-bit with delta is the practical minimum.

---

**[QuantumAI](https://quantumai.kr)** | [GitHub](https://github.com/quantumaikr/quant.cpp)

[![Star History Chart](https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date)](https://star-history.com/#quantumaikr/quant.cpp&Date)
