# Speed Comparison

SmolLM2-1.7B-Instruct-Q8_0, Apple M3, 4 threads, 100 tokens generated.

## quant.cpp speed by KV config

| Config | tok/s | Overhead vs baseline |
|--------|-------|---------------------|
| 4-bit K + FP16 V (baseline) | 25.3 | -- |
| 4-bit K + Q4 V | 21.8 | -14% |
| delta + 3-bit K + Q4 V | 7.2 | -72% |

Delta compression has significant speed overhead due to sequential key
reconstruction (accumulate deltas from last I-frame for each attention query).
This is the expected cost of trading compute for memory.

## quant.cpp vs llama.cpp (not controlled)

We have not performed a controlled speed comparison with llama.cpp yet.
Informal observation: llama.cpp with Metal on M3 achieves ~35 tok/s on
SmolLM2 1.7B Q8_0 vs our ~25 tok/s CPU-only.

This is NOT an apples-to-apples comparison:
- llama.cpp uses Metal GPU, quant.cpp uses CPU only
- Different weight quantization paths
- llama.cpp has more mature SIMD optimization

quant.cpp does not aim to match llama.cpp on throughput.
The value proposition is memory efficiency (longer context) and
code simplicity (33K LOC, readable, embeddable).

## Tradeoff summary

| Metric | 4-bit K | delta + 3-bit K |
|--------|---------|-----------------|
| PPL (WikiText-2) | +0.0% | +1.3% |
| Speed | 25.3 tok/s | 7.2 tok/s |
| KV compression | 4x | 4.3x |
| Use case | General | Memory-constrained only |

Delta mode trades speed for memory. Use it when context length matters
more than generation speed (e.g., long-document summarization on a laptop).
