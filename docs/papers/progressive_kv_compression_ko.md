# 단일 헤더 C 엔진에서의 점진적 KV 캐시 압축: 독립 검증, 부정적 결과, 실용적 배포

**저자:** QuantumAI Research

**초록.** 본 논문은 최근성(recency) 기반 KV 캐시 압축 — 최근 128 토큰의 key를 FP32로 유지하고 나머지를 4비트로 압축 — 을 최소한의 단일 헤더 C 추론 엔진에서 독립적으로 실증한 결과를 보고한다. Llama 3.2 3B에서 3,970 토큰 평가 시, FP32 대비 PPL −0.1%, 3배 압축, 13% 속도 향상을 달성했다. 이 최근성 윈도우 접근법은 KVTC [7]와 PM-KVQ [8]에서 병행적으로 탐구되었으며, 본 연구의 기여는 다음과 같다: (1) RHT 정규화가 레이어별 캘리브레이션의 필요성을 제거함을 보임 (최적 레이어별 할당의 최대 이점 ~0.9%), (2) 2비트 압축 + 512-토큰 윈도우가 957 토큰(53% FP32)에서 Pareto 우위로 보였으나 정직한 평가 길이(3,970 토큰)에서 PPL +36.7%로 붕괴하는 부정적 결과를 보고하고 공개 철회, (3) 동일한 3.2 pp 개선이 957 토큰(13.4% FP32)과 3,970 토큰(3.2% FP32) 모두에서 관찰되는 context-length 불변성 측정, (4) 16K LOC, 의존성 0의 완전한 오픈소스 구현을 `pip install quantcpp`로 배포.

---

## 1. 서론

KV 캐시 압축은 활발한 연구 분야로, 균일 양자화(llama.cpp Q4_0/Q8_0), 채널별 캘리브레이션(KIVI [1], KVQuant [2]), 어텐션 현저성 기반 적응적 정밀도(ZipCache [3]), 최근의 변환 코딩(KVTC [7])까지 다양한 방법이 존재한다. 이 문헌들의 공통 발견은 **최근 토큰이 더 높은 정밀도를 필요로 한다**는 것이다 — KVTC [7]는 최근 128 토큰을 미압축 유지하고, PM-KVQ [8]는 점진적으로 비트를 낮추며, ZipCache [3]는 어텐션 현저 토큰에 더 많은 비트를 할당한다.

우리는 quant.cpp라는 단일 헤더 C 추론 엔진에서 카파시 루프(Karpathy-loop) 최적화 과정을 통해 동일한 발견에 독립적으로 도달했다. 최근성 윈도우 접근법 자체의 독창성을 주장하는 대신, 다음의 기여를 제시한다.

### 기여

1. **RHT가 레이어별 캘리브레이션을 불필요하게 만든다.** Llama 3.2 3B의 28개 레이어에서 post-RHT 첨도(kurtosis) 범위 2.64–3.81 (평균 3.04, 표준편차 0.25). 최적 레이어별 비트 할당의 이론적 최대 이점은 ~0.9% PPL에 불과. **KV 압축의 최적화 공간은 시간적(어떤 토큰)이지, 공간적(어떤 레이어)이 아니다.**

2. **정직한 부정적 결과.** 2비트 + 512-토큰 FP32 윈도우가 flat 4비트를 "Pareto 지배"한다고 초기 주장. 957 토큰에서 53.5%가 FP32인 조건에서 측정된 결과의 오류. 3,970 토큰(12.9% FP32)에서 PPL +36.7%로 붕괴. 주장 공개 철회.

3. **Context-length 불변성.** 128-토큰 윈도우의 품질 개선이 두 스케일에서 동일:
   - 957 토큰 (k128 = 13.4% FP32): flat 대비 3.2 pp 개선
   - 3,970 토큰 (k128 = 3.2% FP32): flat 대비 3.2 pp 개선

4. **실용적 배포.** RHT, Lloyd-Max 코드북, progressive 윈도우, 무한 스크롤백, KV 캐시 저장/복원을 포함한 전체 구현이 단일 C 헤더 파일(16K LOC, 654 KB)에 의존성 0으로 포함. PyPI + WASM(193 KB)으로 배포.

---

## 2. 관련 연구

### 2.1 균일 KV 양자화

llama.cpp의 Q4_0은 블록별 min-max 스케일링으로 ~7× 압축, +10.6% PPL. KIVI [1]는 비대칭 2비트 양자화를 적용 (key: 채널별, value: 토큰별). KVQuant [2]는 pre-RoPE 양자화, 비균일 레이어별 데이터타입, 벡터별 dense-and-sparse 양자화를 추가하여 3비트에서 <0.1% PPL 달성.

### 2.2 비균일 토큰별 정밀도

**ZipCache** [3]는 어텐션 현저 토큰을 식별하여 더 높은 비트폭을 할당. 4.98× 압축, 0.38% 정확도 하락. 현저성(어텐션 스코어 크기)을 기준으로 한 최초의 토큰별 적응적 접근법.

**KVTC** [7] (NVIDIA, ICLR 2026)는 최근 128 토큰과 4개의 "어텐션 싱크" 토큰을 미압축 유지하며, 나머지에 PCA + 엔트로피 코딩 적용. 본 연구와 구조적으로 가장 가까운 선행 연구.

**PM-KVQ** [8] (칭화대, 2025)는 점진적 양자화 전략을 설계하여 오래된 KV 캐시 항목의 비트를 블록별 메모리 할당과 함께 점차 낮춤.

**"More Tokens, Lower Precision"** [9] (EMNLP 2025)는 4비트로 4배 더 많은 토큰을 저장하는 것이 16비트로 1배 저장보다 우수함을 증명. 시간적 압축 논제를 직접 지지.

### 2.3 변환 기반 정규화

**HIGGS** [4]는 가중치 압축에 RHT + MSE-최적 그리드 양자화를 도입. **TurboQuant** [5]는 동일 패턴을 KV 캐시에 적용하며 1비트 QJL 잔차 추가. 본 구현은 TurboQuant의 RHT + Lloyd-Max 구조를 기반으로 하되, QJL 잔차를 어블레이션으로 제거 (어텐션 스코어에 ~0 기여).

---

## 3. 방법

### 3.1 점진적 KV 압축

단일 파라미터 $W$ (highres 윈도우)로 KV 캐시 토큰을 두 계층으로 분할:
- **핫 계층** (최근 $W$ 토큰): Key를 FP32로 저장
- **콜드 계층** (나머지 모든 토큰): Key를 4비트로 저장 (RHT + 16-레벨 Lloyd-Max 코드북)
- **전체**: Value는 FP16

$W$=128, Llama 3.2 3B에서 핫 계층의 추가 메모리: **14.7 MB** (32K 콜드 계층 캐시의 0.6%).

### 3.2 RHT가 레이어간 변이를 제거

Post-RHT 첨도: 평균 3.04, 표준편차 0.25 (범위 2.64–3.81). Pre-RHT 범위 4.13–20.62과 대조적. $\log_2(\sigma)$의 레이어간 분산 0.0177 → 최적 레이어별 할당의 이론적 최대 MSE 개선: ~1.8% → ~0.9% PPL.

**의미:** 레이어별 캘리브레이션에 복잡도를 투자하는 방법(KIVI, KVQuant)은 RHT 정규화 적용 시 이점이 거의 없다.

---

## 4. 실험

**모델:** Llama 3.2 3B Instruct (Q8_0 가중치)
**하드웨어:** Apple M1 Pro, 16 GB RAM, 8 스레드, CPU 전용
**평가:** 영어 텍스트에 대한 teacher-forced PPL (957 토큰 및 3,970 토큰)

### 4.1 점진적 압축 품질

**3,970 토큰** (k128 = 3.2% FP32):

| 설정 | PPL | vs FP32 |
|---|---:|---:|
| FP32 (기준) | 19.41 | — |
| **4비트 + k128** | **19.39** | **−0.1%** |
| 4비트 flat | 20.02 | +3.1% |
| 2비트 + k512 | 26.53 | +36.7% |

### 4.2 Context-Length 불변성

| 평가 길이 | k128 FP32 비율 | Flat 대비 개선 |
|---:|---:|---:|
| 957 토큰 | 13.4% | 3.2 pp |
| 3,970 토큰 | 3.2% | 3.2 pp |

### 4.3 윈도우 크기 포화

| $W$ | PPL (957 tok) | vs FP32 |
|---:|---:|---:|
| 0 | 14.08 | +3.8% |
| 64 | 13.71 | +1.1% |
| **128** | **13.64** | **+0.6%** |
| 256 | 13.64 | +0.6% |

### 4.4 메모리 및 속도 (32K context)

| 설정 | KV 메모리 | 속도 |
|---|---:|---:|
| FP32 | 7.17 GB | 6.9 tok/s |
| 4비트 + k128 | 2.33 GB | 7.8 tok/s (+13%) |

### 4.5 부정적 결과: 2비트 압축

957 토큰에서 2비트 + k512는 PPL +4.3% (53.5% FP32). "Pareto 지배"를 주장했으나, 3,970 토큰(12.9% FP32)에서 PPL +36.7%로 붕괴.

**근본 원인:** 957 토큰에서 512-토큰 FP32 윈도우가 평가의 절반 이상을 차지하여 2비트 열화를 은폐. 짧은 context 평가에서 큰 FP32 윈도우를 사용할 때의 일반적 위험.

### 4.6 레이어 적응 분석 (부정적 결과)

Post-RHT 첨도 변이가 레이어별 적응의 의미 있는 이점을 제공하기에 불충분 (~0.9% 최대). 방법 단순화에 긍정적인 발견.

---

## 5. 공학적 기여

### 5.1 단일 헤더 구현

전체 방법 — RHT, Lloyd-Max 코드북, progressive 윈도우, 무한 스크롤백, KV 저장/복원, NEON/AVX2 SIMD 커널 — 이 `quant.h` (16K LOC, 654 KB)에 libc 외 의존성 0으로 구현.

### 5.2 O(n log n) BPE 토크나이저

표준 BPE merge 알고리즘은 O(n²). GPT 스타일 바이트 레벨 BPE에서 17K 문자 텍스트는 ~17K 초기 토큰을 생성하여 naive merge가 비실용적 (~289M 연산). lazy deletion이 있는 max-heap으로 구현하여 merge 복잡도를 O(n log n)으로 감소. 이 수정이 2비트 artifact를 잡은 3,970 토큰 평가를 가능하게 했다.

### 5.3 배포

- **PyPI:** `pip install quantcpp` (Linux x86_64/aarch64, macOS arm64 사전 빌드 wheel)
- **WASM:** 193 KB 브라우저 데모, IndexedDB 모델 캐싱
- **모델 레지스트리:** HuggingFace 자동 다운로드 (`Model.from_pretrained("Llama-3.2-1B")`)

---

## 6. 논의

### 6.1 KVTC와의 관계

KVTC [7]는 동일한 128-토큰 슬라이딩 윈도우를 사용하지만 압축 영역에 PCA 차원 축소 + 엔트로피 코딩을 추가한다. 우리의 접근은 더 단순하며 (이진 FP32/4비트) 유사한 품질을 달성한다. 이것은 최근성 기반 정밀도 할당이 강건한 원리임을 보여주는 수렴적 증거로 본다.

### 6.2 방법론으로서의 자기 정정

본 프로젝트는 공개 정정 기록을 유지한다 (10건 자체 발견, 0건 외부 신고). 2비트 Pareto 주장(#10)은 자체 평가 인프라 개선(BPE 수정 → 더 긴 eval → 정직한 측정)으로 포착했다. 체계적 자기 검증 — 측정하고, 의심하고, 더 어려운 조건에서 재측정 — 이 알고리즘 기여만큼 중요하다고 믿는다.

### 6.3 한계

1. 단일 모델 (Llama 3.2 3B). 다중 모델 검증 필요.
2. CPU 전용 속도 측정. GPU 동작은 다를 수 있음.
3. 최대 평가 context: 3,970 토큰. 32K+ 검증 보류.
4. V 캐시는 점진적 압축 미적용 (FP16 유지).

---

## 7. 결론

최근성 기반 KV 캐시 정밀도 할당 — 최근 128 토큰을 FP32로 유지 — 이 FP32 품질을 3× 압축에서 달성함을 독립적으로 실증하여, KVTC [7]와 PM-KVQ [8]의 발견을 더 단순한 설정에서 확인했다. 추가 기여인 RHT 공간 분석, 철회된 2비트 결과, context-length 불변성 측정, 그리고 단일 헤더 오픈소스 구현은 기존 문헌을 실용적 검증과 정직한 방법론으로 보완한다.

---

## 참고문헌

[1] Z. Liu et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." ICML 2024. arXiv:2402.02750.

[2] C. Hooper et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." NeurIPS 2024. arXiv:2401.18079.

[3] Y. He et al. "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification." NeurIPS 2024. arXiv:2405.14256.

[4] V. Malinovskii et al. "HIGGS: Pushing the Limits of Large Language Model Quantization via the Linearity Theorem." NAACL 2025. arXiv:2411.17525.

[5] A. Zandieh et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. arXiv:2504.19874.

[6] G. Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024. arXiv:2309.17453.

[7] KVTC. "KV Cache Transform Coding." ICLR 2026. arXiv:2511.01815.

[8] Liu et al. "PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs." 2025. arXiv:2505.18610.

[9] "More Tokens, Lower Precision: Towards the Optimal Token-Precision Trade-off in KV Cache Compression." EMNLP 2025. arXiv:2412.12706.

---

**재현성.** 전체 코드: [github.com/quantumaikr/quant.cpp](https://github.com/quantumaikr/quant.cpp). 설치: `pip install quantcpp`. 벤치마크 아티팩트: `bench/results/`.

**정정 기록.** 10건의 자체 정정이 [CHANGELOG.md](https://github.com/quantumaikr/quant.cpp/blob/main/CHANGELOG.md)에 문서화. 정정 #10 (2비트 Pareto 철회)은 4.5절에서 논의.
