# WBS v1.3 — Full GPU Offload (Metal)

## Phase 1: Core Metal Matmul (P0)

추론의 90%를 차지하는 matmul을 GPU로 이동.

- [ ] **1.1** Metal matmul shader: FP32 weight × FP32 vector
  - `kernel void matmul_f32(device float* w, device float* x, device float* out, uint n, uint d)`
  - Threadgroup: 각 output element를 1개 threadgroup이 계산
  - Shared memory로 input vector 캐시
  - File: `src/backend/metal/shaders/matmul.metal`

- [ ] **1.2** Metal matmul shader: Q4 weight × FP32 vector (fused dequant)
  - Q4_K_M block → FP32 dequant → dot product, 셰이더 내에서 융합
  - llama.cpp `ggml-metal.metal` 참고
  - File: `src/backend/metal/shaders/matmul_q4.metal`

- [ ] **1.3** Metal dispatch wrapper for matmul
  - `tq_metal_matmul(out, x, w, n, d)` — CPU/GPU 자동 선택
  - GPU 버퍼 관리 (weights는 mmap shared, activations는 managed)
  - File: `src/backend/metal/tq_metal_compute.m`

- [ ] **1.4** tq_ops.c에서 matmul GPU 경로 연결
  - `tq_matmul()`, `tq_matmul_q4()` → Metal dispatch 조건부 호출
  - dim >= 1024일 때만 GPU (작은 matmul은 CPU가 빠름)

- [ ] **1.5** 벤치마크: matmul-only GPU vs CPU
  - SmolLM2 1.7B: matmul 시간 비교
  - 목표: matmul 2x+ 속도 향상

## Phase 2: Element-wise Ops (P1)

matmul 사이의 ops를 GPU에서 실행하여 CPU↔GPU 동기화 제거.

- [ ] **2.1** RMSNorm Metal shader
  - L2 norm 계산 (reduction) + elementwise scale
  - Atomic or parallel reduction

- [ ] **2.2** RoPE Metal shader
  - Per-head rotation: cos/sin computation + complex multiply
  - Position encoding을 uniform buffer로 전달

- [ ] **2.3** SiLU/GELU activation Metal shader
  - Elementwise: `silu(x) = x * sigmoid(x)`
  - Gate × Up projection 결과에 적용

- [ ] **2.4** Softmax Metal shader
  - Reduction for max → subtract → exp → reduction for sum → divide
  - Attention scores에 적용

- [ ] **2.5** Add/Residual Metal shader
  - Elementwise add (trivial but needed to stay on GPU)

## Phase 3: Full Forward Pass on GPU (P2)

모든 ops를 연결하여 1 command buffer per token.

- [ ] **3.1** GPU-side KV cache
  - `MTLBuffer` (storageModeShared)로 KV cache 할당
  - Key/Value 저장 + attention lookup 모두 GPU에서

- [ ] **3.2** Forward pass orchestrator
  - `tq_forward_metal()` — 1개 command buffer에 모든 연산 인코딩
  - CPU fallback: Metal 미지원 환경에서 자동 CPU 경로

- [ ] **3.3** Embedding lookup on GPU
  - Token ID → embedding vector (GPU side gather)

- [ ] **3.4** Output projection + sampling handoff
  - Logit 계산까지 GPU → CPU로 결과 전송 → sampling

- [ ] **3.5** 통합 벤치마크
  - E2E tok/s: SmolLM2 1.7B, Qwen3.5 4B
  - GPU utilization monitoring
  - PPL 검증 (CPU와 동일해야 함)

## Phase 4: 최적화 (P3)

- [ ] **4.1** Double buffering — 이전 토큰 처리 중 다음 토큰 준비
- [ ] **4.2** Fused attention kernel — QK matmul + softmax + V weighted sum 1개 셰이더
- [ ] **4.3** Batched embedding dequant — 여러 행을 한 번에 dequant

## Milestone 정의

| Milestone | 목표 | 기준 |
|-----------|------|------|
| M1 (Phase 1) | matmul GPU 동작 | SmolLM2 matmul 2x 빠름 |
| M2 (Phase 2) | 전체 ops GPU | CPU↔GPU 전환 0회/token |
| M3 (Phase 3) | E2E GPU forward | 60+ tok/s on SmolLM2 |
| M4 (Phase 4) | 최적화 | 80+ tok/s on SmolLM2 |
