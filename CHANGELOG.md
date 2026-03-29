# Changelog

All notable changes to TurboQuant.cpp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-29

### Added

#### Core Library
- Public C API (`turboquant.h`) with context lifecycle, quantization, attention,
  and cache management functions.
- Type system (`tq_types.h`) with block structures and static size assertions
  for all quantization types.
- Format specification (`tq_spec.h`) with version-aware format metadata.
- PolarQuant 3-bit and 4-bit key quantization (`tq_polar.c`):
  - Polar coordinate transformation (cartesian to polar and back).
  - Reference quantize, dequantize, and attention implementations.
- QJL 1-bit sign hash quantization (`tq_qjl.c`):
  - Random projection matrix generation with seeded PRNG.
  - 1-bit sign quantization with outlier detection.
  - Hamming distance based attention score computation.
- TurboQuant composite quantization (`tq_turbo.c`):
  - Two-stage: PolarQuant for primary + QJL for residual correction.
  - Combined attention score computation.
- Uniform min-max baseline quantization (`tq_uniform.c`):
  - 2-bit and 4-bit variants with ONNX LSB-first bit packing.
- Value cache quantization (`tq_value_quant.c`):
  - Group-wise min-max quantization for value vectors.
- Type traits dispatch table (`tq_traits.c`):
  - O(1) function pointer dispatch for all quantization types.
  - Type metadata: name, block_size, type_size, bits-per-element.

#### Cache Management
- Paged quantized cache (`tq_paged_cache.c`):
  - Block-based KV cache with configurable block size.
  - Dynamic block allocation and deallocation.
  - Copy-on-write block support.
  - Per-head sequence tracking.

#### Build System
- CMake build system with C11/C++17 support.
- Platform detection (Linux, macOS, Windows).
- SIMD detection (AVX2, NEON, SVE).
- Build options: `TQ_BUILD_TESTS`, `TQ_BUILD_BENCH`, `TQ_BUILD_CUDA`, `TQ_BUILD_METAL`.
- Google Test integration for unit testing.

#### Tests
- Unit tests for all quantization types:
  - `test_polar.cpp`: PolarQuant roundtrip and attention tests.
  - `test_qjl.cpp`: QJL hash and attention tests.
  - `test_turbo.cpp`: TurboQuant composite tests.
  - `test_uniform.cpp`: Uniform quantization tests.
  - `test_value.cpp`: Value quantization tests.
  - `test_paged_cache.cpp`: Cache management tests.

#### Benchmarks
- Performance benchmark (`tq_bench.cpp`):
  - Quantize throughput, attention throughput, compression ratio.
- Quality benchmark (`tq_quality.cpp`):
  - Roundtrip MSE, attention cosine similarity, cross-platform determinism.
- Memory benchmark (`bench_memory.cpp`):
  - KV cache memory comparison across all types and sequence lengths.
- Latency benchmark (`bench_latency.cpp`):
  - Per-operation latency for quantize, dequantize, and attention.
- LongBench accuracy benchmark (`run_longbench.py`):
  - F1 score comparison across LongBench tasks.
- Needle-in-a-Haystack benchmark (`run_niah.py`):
  - Retrieval accuracy across context lengths and depths.

#### Integrations
- llama.cpp integration (`integrations/llamacpp/`):
  - GGML type registration for all TurboQuant types.
  - from_float, to_float, vec_dot wrappers.
  - CLI option parser for `--kv-cache-type`.
  - Integration guide.
- vLLM integration guide (`integrations/vllm/README.md`):
  - Custom CacheEngine documentation.
  - Usage examples with kv_cache_dtype.

#### Python Bindings
- ctypes-based Python package (`bindings/python/turboquant/`):
  - TurboQuantContext class wrapping C API.
  - quantize_keys(), quantize_values(), dequantize_keys(), attention() methods.
  - NumPy array support.
  - Module-level convenience functions.
- pip-installable package with setup.py.
- Python binding tests.

#### Documentation
- Product Requirements Document (PRD v0.1).
- Work Breakdown Structure (WBS v0.1).
- Benchmark results documentation.
- Integration guides for llama.cpp and vLLM.
- Format specification documents.

### Supported Quantization Types

| Type           | Key Bits | Algorithm              | Block Size |
|----------------|----------|------------------------|------------|
| TQ_POLAR_3B    | 3        | PolarQuant             | 128        |
| TQ_POLAR_4B    | 4        | PolarQuant             | 128        |
| TQ_QJL_1B      | 1        | QJL sign hash          | 256        |
| TQ_TURBO_3B    | 3        | PolarQuant 2b + QJL 1b | 128        |
| TQ_TURBO_4B    | 4        | PolarQuant 3b + QJL 1b | 128        |
| TQ_UNIFORM_4B  | 4        | Min-Max uniform        | 128        |
| TQ_UNIFORM_2B  | 2        | Min-Max uniform        | 128        |

### Supported Platforms

| Platform       | Architecture | Status     |
|----------------|-------------|------------|
| Linux          | x86_64      | Supported  |
| macOS          | arm64       | Supported  |
| macOS          | x86_64      | Supported  |
| Windows        | x86_64      | Planned    |

### Known Limitations

- CUDA backend is not yet implemented (stubs only).
- Metal backend is not yet implemented (stubs only).
- SIMD (AVX2/NEON) optimizations are not yet implemented.
- Progressive compression is partially implemented.
- Python bindings require pre-built shared library.

## [Unreleased]

### Planned
- AVX2 and NEON SIMD-optimized kernels.
- CUDA backend for GPU acceleration.
- Metal backend for Apple Silicon.
- Progressive compression with automatic tier transitions.
- Multi-threaded quantization.
- Full llama.cpp integration with upstream patches.
- Native vLLM cache engine plugin.
