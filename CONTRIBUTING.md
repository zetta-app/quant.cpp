# Contributing to quant.cpp

Thank you for your interest in contributing! Here's how to get started.

## Quick Setup

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

Or with Docker:

```bash
docker build -t quant .
docker run quant models/model.gguf -p "Hello"
```

## Running Tests

```bash
# All tests
ctest --test-dir build --output-on-failure

# Specific test
./build/test_polar
./build/test_qjl

# With scoring harness (5-dimension evaluation)
bash score.sh              # Full evaluation
bash score.sh --quick      # Build + correctness only
bash score.sh --bench      # Performance benchmarks
bash score.sh --quality    # Quantization quality metrics
```

## What to Work On

Check [Issues](https://github.com/quantumaikr/quant.cpp/issues) for tasks labeled `good first issue` or `help wanted`.

**High-impact areas:**
- New model architectures (Llama, Phi, Gemma)
- AVX2/AVX-512 SIMD kernels for x86
- Metal GPU compute shaders
- Long context benchmarks (8K, 32K, 128K tokens)

## Adding a New Model Architecture

1. Add the model config struct to `include/turboquant/tq_engine.h`
2. Implement the forward pass in `src/engine/` (one file per architecture)
3. Register the architecture in `tq_load_model()` in `src/engine/tq_model.c`
4. Add a test in `tests/` and an example in `examples/`
5. Verify with `bash score.sh --quick`

## Adding a New KV Cache Type

1. Define the type enum in `include/turboquant/tq_types.h` (append to `tq_type` enum)
2. Add block struct + `static_assert` size check in `include/turboquant/tq_spec.h`
3. Implement `quantize`/`dequantize`/`attention` in `src/core/tq_<name>.c`
4. Register in the dispatch table in `src/core/tq_traits.c`
5. Add unit tests in `tests/test_<name>.cpp`
6. Update `tools/quant.c` to accept the new type name in `parse_kv_type()`

## Code Standards

- **C11** for core library (`src/`), **C++17** for tests and CUDA/Metal wrappers
- No external dependencies in core (libc/libm/pthread only)
- Every block struct must have `static_assert` size verification
- Every public function needs a unit test
- ONNX LSB-first bit-packing convention for all quantized formats
- Use `refs/` code as algorithm reference -- port to C, don't wrap Python

## Module Ownership

Each module has exclusive files to prevent merge conflicts:

| Module | Files |
|--------|-------|
| `polar` | `src/core/tq_polar.*`, `tests/test_polar.*` |
| `qjl` | `src/core/tq_qjl.*`, `tests/test_qjl.*` |
| `turbo` | `src/core/tq_turbo.*`, `tests/test_turbo.*` |
| `uniform` | `src/core/tq_uniform.*`, `src/core/tq_value_quant.*` |
| `engine` | `src/engine/*` |
| `cache` | `src/cache/*` |
| `simd` | `src/backend/cpu/*` |

## Cross-Platform Checklist

Before submitting, verify:
- [ ] NEON intrinsics are inside `#ifdef __ARM_NEON` guards
- [ ] No GCC warnings (`-Wall -Wextra -Wpedantic`)
- [ ] Scalar fallback exists for all SIMD code paths
- [ ] Function pointer types match their typedefs

## Pull Request Process

1. Fork and create a feature branch
2. Make your changes
3. Ensure all tests pass and no new warnings
4. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
