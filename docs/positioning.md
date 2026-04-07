# quant.cpp Positioning

> *Updated April 2026 — after Google's TurboQuant publication at ICLR 2026.*

## TL;DR

**quant.cpp is the single-header C reference implementation of TurboQuant and related KV cache quantization research.** We are not competing with Google. We are not competing with llama.cpp. We are filling a gap nobody else can fill: running modern KV-quantized inference *anywhere a C compiler runs*.

## The Landscape (April 2026)

### What changed

In March–April 2026 the KV cache quantization landscape transformed:

1. **Google published TurboQuant** at ICLR 2026 (Zandieh, Daliri, Hadian, Mirrokni). [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
2. **PolarQuant** appeared at AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
3. Multiple competing OSS implementations sprang up in weeks: Rust, PyTorch, several llama.cpp forks
4. llama.cpp Discussion #20969 has 6+ independent fork implementations, **none merged**, no convergence

The "TurboQuant" name is now a Google research brand. Our project must carefully position around it.

### Where we stand

quant.cpp predates the Google publication. We were independently exploring the same algorithmic ideas (PolarQuant rotation, QJL sketch). When the official paper appeared, our codebase already had working implementations of the building blocks. **We are now repositioning as the canonical embedded/portable C implementation, not as a competitor to the algorithm authors.**

## Our Position in One Sentence

> The single-header C implementation of Google TurboQuant — for iPhone, Android, browser, microcontrollers, game engines, and every place a Rust crate or Python package can't go.

## What We Are

| | |
|--|--|
| **Engine class** | Single-header reference C implementation of published KV quantization research |
| **Audience** | App developers, mobile, embedded, browser, game engine, research |
| **Core artifact** | `quant.h` — 628KB single header, 15.7K LOC, libc + libm only |
| **License** | Apache 2.0 |
| **Algorithms shipped** | TurboQuant (Polar+QJL), PolarQuant, QJL, Uniform 4b/2b, TurboKV 1b/3b/4b |
| **Inference scope** | End-to-end: GGUF loader → tokenizer → forward pass → sampling → text |
| **Architectures** | Llama, Llama 3, Qwen, Qwen3.5 hybrid, Gemma 3, Gemma 4 MoE, SmolLM, DeltaNet |
| **Backends** | CPU (NEON, AVX2, generic), Metal (partial), CUDA (compiles), WASM, MSVC |
| **What proves the moat** | The fact that `embed_minimal` links only against `libSystem` — no library, no framework, no runtime |

## What We Are NOT

| | Why we don't compete |
|--|--|
| ❌ The fastest GPU inference engine | llama.cpp owns this with full Metal/CUDA tensor graphs |
| ❌ The highest-throughput batch server | vLLM owns this |
| ❌ The original TurboQuant authors | Google Research owns the algorithm |
| ❌ The most features | We deliberately stay minimal |
| ❌ A training framework | Use PyTorch/JAX |
| ❌ Production-grade for 100+ models | We verify 7 architectures end-to-end |

## Competitive Matrix

### vs other TurboQuant implementations

| Implementation | Lang | Size | Mobile | WASM | Embedded | End-to-end |
|---|---|---|---|---|---|---|
| **quant.cpp** | C11 | 628KB single header | ✅ | ✅ 192KB | ✅ | ✅ |
| RecursiveIntell/turbo-quant | Rust | Cargo crate | ❌ | ❌ | ❌ | kernel only |
| tonbistudio/turboquant-pytorch | Python | pip + Torch | ❌ | ❌ | ❌ | kernel only |
| OnlyTerp/turboquant | Python | pip | ❌ | ❌ | ❌ | kernel only |
| scos-lab/turboquant | Python | research | ❌ | ❌ | ❌ | kernel only |
| llama.cpp forks (#20969) | C++ | ggml fork | partial | ❌ | ❌ | depends on llama.cpp |

### vs production engines

| Engine | KV quant | Size | Read-in-an-afternoon | Embeddable | Best for |
|---|---|---|---|---|---|
| **quant.cpp** | TurboQuant + 6 schemes | 72K LOC | ✅ | ✅ single header | Embedded / mobile / WASM / education |
| llama.cpp | Q8_0/Q5_0 (~2x) | 250K+ LOC | ❌ | library | Workstation speed |
| vLLM | none | 100K+ LOC | ❌ | framework | Batch serving |
| MLX | none | 50K+ LOC | ❌ | framework | Apple Silicon |
| ONNX RT | none | 500K+ LOC | ❌ | framework | Multi-platform serving |

## Strategic Pillars

### Pillar 1 — Be the canonical reference C implementation
- Implement Google TurboQuant precisely per the ICLR 2026 paper
- Verify our numbers reproduce the paper's published results within ±1%
- Cite the paper authors prominently in every README and docs page
- Submit to llama.cpp Discussion #20969 with a clean ggml type registration

### Pillar 2 — Own the embedded niche
- iOS demo app (Xcode project)
- Android NDK build guide
- WASM npm package
- Unity C# binding
- Unreal C++ integration
- Microcontroller (Cortex-M4 with FlexRAM) feasibility study

### Pillar 3 — Stay readable
- Hard cap: forward pass in one file (`tq_transformer.c`)
- Hard cap: KV quantization plugin via 3 functions
- Hard cap: zero new dependencies in core
- Every PR that adds a feature must also add a unit test

### Pillar 4 — Honest benchmarks
- Always disclose: model, dataset, baseline, methodology
- Never claim "lossless" without PPL Δ on a specific dataset
- Always link to a reproducible script
- Match Google's published benchmarks (LongBench, NIH, ZeroSCROLLS, RULER, L-Eval) where feasible

## Naming Hygiene

| Term | What it means | Where to use |
|---|---|---|
| **TurboQuant** | Google's algorithm (Zandieh et al., ICLR 2026) | Always cite + link to arXiv |
| **PolarQuant** | The rotation + polar quantization step | Cite arXiv:2502.02617 |
| **QJL** | Quantized Johnson-Lindenstrauss residual sketch | Cite arXiv:2406.03482 |
| **quant.cpp** | This project — a C implementation | Project / repo name |
| **`TQ_TURBO_*`** | Our internal type identifiers (predates Google publication) | Code only — docs must clarify lineage |

## Goals (next 6 months)

| Goal | Metric | Owner |
|---|---|---|
| Repository stars | 1000+ | community |
| GitHub citations | 5+ academic | community |
| llama.cpp PR merged or formally reviewed | 1 | core |
| iOS demo app on App Store / TestFlight | shipped | core |
| npm @quantcpp/wasm package | published | core |
| arXiv tech report | submitted | core |
| Reproduce TurboQuant paper benchmarks | within ±1% | core |

## What success looks like

In 6 months, when someone googles "TurboQuant llama.cpp" or "TurboQuant iOS" or "KV cache compression embedded", quant.cpp is the first or second result. The Google paper is the theoretical reference; quant.cpp is the practical implementation everyone reaches for when they need to actually ship something.
