# TurboQuant.cpp

**[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV 캐시 압축을 구현한 순수 C 추론 엔진.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Release](https://img.shields.io/github/v/release/quantumaikr/TurboQuant.cpp)]()
[![Tests](https://img.shields.io/badge/tests-26%20suites-brightgreen)]()

### 최대 7.1x 총 K+V 압축. 품질 보존.

```
Gemma 3 4B — 토큰당 총 K+V 메모리:

  FP16 K+V (llama.cpp):    136.00 KB   (기준)
  1-bit K + Q4 V:            27.62 KB   (4.9x)   "Paris" ✓  "1+1=2" ✓
  1-bit K + Q2 V:            19.12 KB   (7.1x)   "Paris" ✓  "Mercury, Venus, Earth" ✓
```

Key 압축: 10.7x (1-bit sign hash). Value 압축: Q4 (3.8x) 또는 Q2 (7.6x). 합산: **최대 7.1x 총 K+V**.

---

## 왜 중요한가

LLM attention은 **내적** `<query, key>`을 계산합니다. 일반 양자화기는 복원 오차(MSE)를 최소화하지만, 이것은 내적 추정에 **체계적 편향**을 만듭니다.

[TurboQuant 논문](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026)이 이 간극을 증명하고 해결법을 제시했습니다:

- **Key**: RHT + Lloyd-Max 코드북 + QJL 잔차 → 어떤 비트에서든 **비편향** 내적 추정
- **Value**: RHT + Lloyd-Max 코드북 → 가중합을 위한 **MSE 최적** 복원

우리는 이 둘을 순수 C로 구현하고, Key를 **1비트**까지 밀어붙였습니다 — XOR + popcount로 attention 수행.

---

## 압축 옵션

```bash
# Key 압축 (attention 스코어링에 영향)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b       # 1-bit keys (10.7x)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_3b       # 3-bit keys (4.6x)

# Value 압축 (출력 복원에 영향)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4  # + Q4 values → 4.9x 총합
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q2  # + Q2 values → 7.1x 총합

# 메모리 통계
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4 -M
```

### 총 K+V 압축 테이블

| 구성 | K 비트 | V 비트 | K+V/토큰 | 총 압축률 | 품질 |
|------|--------|--------|----------|----------|------|
| FP16 (기준) | 16 | 16 | 136.00 KB | 1.0x | 참조 |
| uniform_4b + FP16 V | 4 | 16 | 86.06 KB | 1.6x | 베이스라인 |
| 1-bit K + FP16 V | 1 | 16 | 74.38 KB | 1.8x | ~120토큰까지 greedy 동일 |
| **1-bit K + Q4 V** | **1** | **4** | **27.62 KB** | **4.9x** | **"Paris" ✓ "1+1=2" ✓** |
| **1-bit K + Q2 V** | **1** | **2** | **19.12 KB** | **7.1x** | **"Paris" ✓ 행성 ✓** |

### 32K 컨텍스트 메모리 (Gemma 3 4B)

```
FP16 K+V:              4,352 MB
1-bit K + Q4 V:           885 MB   (4.9x, 3.4 GB 절약)
1-bit K + Q2 V:           613 MB   (7.1x, 3.7 GB 절약)
```

> **품질 참고:** K-only 양자화(V는 FP16/FP32)에서 greedy decode는 ~120토큰까지 바이트 동일.
> V 양자화(Q4/Q2) 시 더 일찍 발산하지만 coherent하고 사실적으로 정확합니다.
> V 양자화가 가중합 복원에 직접 영향을 주므로 예상된 동작입니다.

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

---

## 알고리즘

```
Key (attention 스코어링 — 비편향 내적 필요):
  key → 정규화 → RHT → Lloyd-Max 코드북 (b-1 bits) → QJL 부호 (1 bit)
  1-bit 극한: 코드북 생략, 부호만 저장 → XOR + popcount attention

Value (가중합 — MSE 최적 복원 필요):
  value → Q4 또는 Q2 블록별 양자화 → 출력 시 실시간 역양자화
```

| 구성 요소 | Key용 | Value용 |
|----------|-------|---------|
| **목표** | 비편향 내적 | 낮은 MSE 복원 |
| **방법** | RHT + 코드북 + QJL | 블록 스케일 + 양자화 |
| **1-bit** | 부호 해시 (XOR+popcount) | 비권장 |
| **최적 구성** | 1-bit (10.7x key 압축) | Q4 (3.8x value 압축) |

---

## 지원 모델

| 모델 | 파라미터 | 속도 (Q4, 6T) | 검증 |
|------|----------|---------------|------|
| **Gemma 3 4B** | 4B | 20.2 tok/s | "Paris" ✓, 행성 ✓ |
| **Qwen3.5-0.8B** | 752M | 80.1 tok/s | PyTorch 대비 코사인 0.999 |
| **Gemma 3 270M** | 270M | 176 tok/s | 레이어별 정확 일치 |

멀티 아키텍처: Qwen3.5 (DeltaNet 하이브리드) + Gemma 3 (슬라이딩 윈도우). Gemma 4 대응.

---

## 기술 상세

- **10,000줄 이상의 순수 C** — 외부 의존성 없음
- **11개 양자화 타입** — Uniform, Mixed, PolarQuant, QJL, TurboQuant KV (1/3/4-bit)
- **K+V 독립 압축** — 1-bit key (XOR+popcount) + Q4/Q2 value
- **ICLR 2026 논문 충실 구현** — RHT + Lloyd-Max + QJL 잔차
- **멀티 아키텍처** — Qwen3.5 (DeltaNet) + Gemma 3 (슬라이딩 윈도우 + GeGLU)
- **NEON 벡터화** — matmul, attention, RHT butterfly, Hamming distance, Q4 dequant, FP16 변환
- **26개 테스트 스위트** — KV 라운드트립, attention 분포, 코드북 이론, NEON/스칼라 일치성, 엣지케이스, Q2 가중치

### 검증 요약

| 범주 | 테스트 | 검증 내용 |
|------|--------|----------|
| NEON/스칼라 일치성 | 14개 | 모든 NEON 경로가 스칼라 참조와 일치 (Q4, Q2, RHT, RoPE, matmul, RMSNorm, Hamming) |
| Attention 분포 | 8개 | 코사인 유사도, Spearman 순위, top-k 겹침 vs FP32 참조 |
| 코드북 이론 | 5개 | Lloyd-Max centroid 문헌 일치, MSE가 정보이론 최적의 1.18배 이내 |
| 엣지케이스 | 29개 | n=1, dim=0, NaN, Inf, 동일값, 영벡터, n=10000 |
| ASan + UBSan | 26개 | 전체 스위트 sanitizer 통과, 메모리 오류 없음 |
| 스레드 안전성 | mutex | 글로벌 워크스페이스 realloc 동시 접근 보호 |
| 수치 안정성 | 4개 | overflow 안전 norm (max-abs rescaling), NaN/Inf 입력 가드 |

상세: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

## 벤치마크 & 검증

### Ablation: TurboQuant가 실제로 도움이 되는가?

```bash
bash bench/ablation_test.sh model.tqm
```

`uniform_4b`, `turbo_kv_3b`, `turbo_kv_1b`를 50-300 토큰에서 비교하여 각 방법이 uniform 베이스라인에서 발산하는 지점을 보여줍니다.

- **turbo_kv_3b** (코드북 + QJL): 일반적으로 모든 테스트 길이에서 `uniform_4b`와 일치
- **turbo_kv_1b** (부호 해시만): 긴 컨텍스트에서 발산할 수 있지만 출력은 coherent 유지
- **RHT 중요**: Randomized Hadamard Transform이 outlier를 균등 분배하여 체계적 양자화 편향 방지

### V 양자화 현실

"30/30 바이트 동일" 결과는 **K-only 양자화** (V는 FP16/FP32)에 해당합니다.
V=Q4에서는 더 일찍 발산하지만 coherent하고 사실적으로 정확합니다.

```bash
bash bench/kv_quality_bench.sh model.tqm   # Phase 4: V 양자화 확인 포함
```

### Long Context 품질

```bash
bash bench/long_quality_test.sh model.tqm   # 200, 500, 1000 토큰
```

### Temperature Sampling

```bash
bash bench/sampling_test.sh model.tqm   # T=0.3, T=0.7, 각 3회
```

KV 압축이 확률적 샘플링 품질을 저하하지 않음을 검증합니다.

### Sanitizer 검증

```bash
bash scripts/sanitize.sh [model.tqm]   # ASan + UBSan 빌드 및 테스트
```

`-fsanitize=address,undefined`로 빌드, 전체 테스트 실행. 메모리 오류 및 정의되지 않은 동작 없음.

---

## FAQ

**Q: "바이트 동일 출력은 K가 중요하지 않다는 뜻 아닌가?"**

아닙니다. K를 랜덤으로 대체하면 즉시 쓰레기가 됩니다 (코사인 < 0.09). TurboQuant는 내적 순위를 보존합니다 — 측정된 attention score 코사인: uniform_4b = 0.996, turbo_kv_3b = 0.918, turbo_kv_1b = 0.634 (10회 평균, 32 keys). 1-bit 코사인 0.634는 부호 양자화의 정보이론적 한계 2/pi = 0.637과 일치 — 수학적으로 최적이며 결함이 아닙니다. `tests/test_attention_distribution.cpp` 참조.

**Q: "llama.cpp의 Q4 KV와 뭐가 다른가?"**

llama.cpp는 uniform min-max 양자화를 사용합니다. TurboQuant는 회전 후 가우시안 분포에 최적화된 RHT + Lloyd-Max 코드북을 사용합니다. Lloyd-Max centroid가 이론값과 일치함을 검증 (MSE가 정보이론적 최적의 1.18배 이내, `tests/test_codebook_theory.cpp`). QJL 잔차 보정은 증명 가능한 비편향 내적 추정을 제공합니다.

**Q: "Perplexity는?"**

Attention score 분포 보존: Spearman 순위 상관 = 0.990 (uniform_4b), 0.900 (turbo_kv_3b), 0.632 (turbo_kv_1b). Greedy decode ~120토큰까지 일치. 1-bit 코사인 0.634 = 2/pi는 부호 양자화의 이론적 최대값 (JL 문헌에서 증명). 표준 데이터셋 perplexity 진행 중.

**Q: "NEON 코드가 정확한가?"**

모든 NEON 경로 (Q4 dequant, RHT butterfly, matmul, RMSNorm, RoPE, Hamming attention)가 `tests/test_neon_scalar.cpp`에서 스칼라 참조와 비교 검증됩니다. Q4 dequant에서 nibble 인터리빙 버그를 발견 후 수정했습니다. ASan + UBSan이 26개 전체 테스트 스위트에서 오류 없이 통과. NaN/Inf/엣지케이스 입력을 `tests/test_edge_cases.cpp` (29개 케이스)에서 테스트.

**Q: "스레드 안전성은?"**

글로벌 워크스페이스 (Q8 양자화 버퍼, 샘플러 확률 인덱스)가 mutex로 보호되어 동시 realloc 경합을 방지합니다. 스레드 풀은 단일 디스패치 mutex를 사용합니다.

**Q: "4B 모델만으로는 — 8B 이상은?"**

아키텍처는 모델 크기에 독립적입니다. Gemma 3 4B와 Qwen3.5 0.8B가 동일 코드 경로를 사용합니다. 8B 지원 계획 중 (Llama 3.1 8B 아키텍처 지원 진행 중).

**Q: "RHT 오버헤드는?"**

RHT는 벡터당 O(d log d), NEON 벡터화. 측정: 128차원 벡터당 147 ns. 전체 양자화: uniform_4b = 148 ns, turbo_kv_1b = 659 ns, turbo_kv_3b = 11066 ns/벡터. 1-bit attention: 1.2 ns/key (XOR+popcount). matmul (~1ms/레이어) 대비 모든 오버헤드 무시 가능. `bench/bench_kv_overhead.cpp` 참조.

---

## 참고 논문

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — 근최적 왜곡률의 온라인 벡터 양자화
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — KV 캐시를 위한 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 KV 양자화

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
