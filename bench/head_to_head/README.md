# Head-to-Head: quant.cpp vs llama.cpp

Reproducible, automated benchmark comparing KV cache quantization between
quant.cpp and llama.cpp. Both engines run on the same model, same text,
same hardware, same thread count.

## Quick Start

```bash
# 1. Build quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

# 2. Build llama.cpp (pinned to b5200 for reproducibility)
bash bench/head_to_head/setup_llamacpp.sh

# 3. Run the benchmark (requires a GGUF model)
bash bench/head_to_head/run_benchmark.sh models/SmolLM2-1.7B-Q8_0.gguf
```

## What It Measures

For each configuration:

| Metric | Method |
|--------|--------|
| **Perplexity (PPL)** | Teacher-forced evaluation on `bench/data/ppl_4k.txt` (4K tokens) |
| **Generation speed** | 128-token generation from a fixed prompt, reported as tok/s |
| **Peak RSS memory** | Maximum resident set size during generation |

## Test Matrix

| Engine | Config | K bits | V bits |
|--------|--------|--------|--------|
| llama.cpp | FP16 KV (baseline) | 16 | 16 |
| llama.cpp | Q8_0 K + Q5_0 V | 8.5 | 5.5 |
| llama.cpp | Q4_0 KV | 4.5 | 4.5 |
| quant.cpp | FP16 KV (baseline) | 16 | 16 |
| quant.cpp | uniform_4b K + Q4 V | 4 | 4 |
| quant.cpp | delta_3b K + Q4 V | 3 | 4 |

## Output

- **stdout**: Summary table with PPL, tok/s, and RSS for all configs
- **CSV**: `bench/head_to_head/results/results_<date>.csv`

## Reproducibility

- llama.cpp is pinned to commit `f472633e` (tag b5200, 2025-03-30)
- `setup_llamacpp.sh` skips the build if already at the correct commit
- The same PPL text file is used for both engines
- Thread count defaults to all available cores; override with the second argument

## Interpreting Results

The key comparison is at similar bit rates:

- **4-bit**: llama.cpp Q4_0 KV vs quant.cpp uniform_4b + Q4 V
- **3-bit vs 4-bit**: llama.cpp Q4_0 KV vs quant.cpp delta_3b + Q4 V

If quant.cpp's 3-bit delta KV matches llama.cpp's 4-bit Q4_0 in PPL,
that represents ~30% more KV cache compression at equal quality.

## Adding New Configs

Edit the `CONFIGS` array in `run_benchmark.sh`. Format:

```
"engine|label|ppl_extra_args|gen_extra_args"
```

For llama.cpp, use `--cache-type-k TYPE --cache-type-v TYPE`.
For quant.cpp, use the `-k TYPE` and `-v TYPE` short flags.
