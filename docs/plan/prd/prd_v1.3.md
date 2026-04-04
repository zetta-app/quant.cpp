# PRD v1.3 — Full GPU Offload (Metal/Apple Silicon)

## Overview

현재 quant.cpp의 추론은 CPU에서 실행됩니다 (AMX 가속 포함, 35 tok/s on M3).
ollama+MLX는 전체 forward pass를 Apple GPU에서 실행하여 50-100+ tok/s를 달성합니다.

v1.3의 목표: **전체 transformer forward pass를 Metal GPU에서 실행.**

## Target Performance

| Metric | Current (CPU+AMX) | Target (Metal GPU) | Reference (ollama+MLX) |
|--------|-------------------|--------------------|-----------------------|
| SmolLM2 1.7B tok/s | 35 | **80+** | ~100 |
| Qwen3.5 4B tok/s | 5.4 | **20+** | ~40 |
| Latency per token | 28ms | **<15ms** | ~10ms |
| GPU utilization | 0% | **>80%** | ~90% |

## Why This Is Achievable

Apple Silicon의 **통합 메모리**가 핵심 이점:
- CPU와 GPU가 같은 메모리를 공유 — 데이터 복사 불필요
- mmap된 모델 가중치를 GPU에서 직접 읽기 가능
- llama.cpp Metal과 동일한 접근 방식

## Architecture

```
Current:
  token → [CPU] embed → [CPU] attn_norm → [CPU] QKV matmul → [CPU] attention
       → [CPU] FFN matmul → [CPU] output_proj → logits

Target:
  token → [GPU] embed → [GPU] attn_norm → [GPU] QKV matmul → [GPU] attention
       → [GPU] FFN matmul → [GPU] output_proj → [CPU] sampling → next token
```

### Metal Compute Shaders Needed

| Shader | Input | Output | Priority |
|--------|-------|--------|----------|
| `matmul_q4_f32` | Q4 weights + FP32 vec | FP32 vec | P0 (90% of compute) |
| `matmul_f32` | FP32 weights + FP32 vec | FP32 vec | P0 |
| `rmsnorm` | FP32 vec + FP32 weights | FP32 vec | P1 |
| `rope` | FP32 Q/K + position | FP32 Q/K | P1 |
| `silu_elementwise` | FP32 gate + FP32 up | FP32 | P1 |
| `softmax` | FP32 scores | FP32 probs | P1 |
| `attention_fwd` | Q, K cache, V cache | FP32 output | P2 (fused) |
| `add_residual` | FP32 + FP32 | FP32 | P2 |

### Pipeline Design

```
1개 Command Buffer per token (최소 동기화):

  encoder.setComputePipelineState(matmul_q4_pipeline)
  encoder.setBuffer(weights_q,  0)  // Q projection weights (mmap)
  encoder.setBuffer(input,      1)  // normalized input
  encoder.setBuffer(output_q,   2)  // Q output
  encoder.dispatchThreadgroups(...)

  // ... K, V projection, RoPE, attention, FFN ...

  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()  // 1회만, 토큰당
```

## Key Design Decisions

1. **Single command buffer per token** — 셰이더 간 동기화 최소화
2. **가중치는 mmap 그대로** — 통합 메모리이므로 GPU가 직접 접근
3. **KV cache는 GPU 버퍼** — `MTLBuffer` with `storageModeShared`
4. **Sampling만 CPU** — top-p sampling은 GPU에서 비효율적
5. **Q4 dequant는 GPU에서** — matmul과 fused하여 대역폭 절약

## Scope & Non-Goals

### In Scope
- Metal compute shaders for all forward pass ops
- Apple Silicon (M1-M5) 지원
- Q4_K_M, Q8_0 가중치 형식
- 단일 시퀀스 추론 (batch=1)

### Out of Scope (v1.3)
- CUDA/Vulkan GPU offload (별도 버전)
- Batched inference
- Flash Attention
- Continuous batching
- Speculative decoding

## Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Per-dispatch overhead > compute gain (small models) | Medium | 큰 모델에서만 GPU 활성화 (dim >= 2048) |
| Q4 dequant shader 정확도 | Low | llama.cpp Metal shader 참고 |
| Command buffer 동기화 병목 | Medium | Double buffering, async commit |

## Success Criteria

1. SmolLM2 1.7B에서 **60+ tok/s** (현재 35)
2. Qwen3.5 4B에서 **15+ tok/s** (현재 5.4)
3. PPL 변화 없음 (GPU 계산 정확도 = CPU)
4. 기존 CPU 경로 유지 (GPU 없는 환경 폴백)
5. quant.h에는 영향 없음 (GPU는 full build only)
