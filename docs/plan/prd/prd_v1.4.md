# PRD v1.4 — Community Trust & Killer Demo

## Background

v1.3까지 기술적 완성도 99.2%, 234 stars. Reddit/HN 커뮤니티 노출에서 핵심 피드백:

1. **비교가 공정하지 않다** (audioen) — llama.cpp Q8K+Q5V와 비교해야 함
2. **말로 하지 말고 보여줘** (hauhau901, MelodicRecognition7) — 데모/벤치마크로 증명
3. **속도가 느리다** (Emotional-Breath-838) — 속도 열세가 첫인상을 지배
4. **킬러 유즈케이스가 없다** (putrasherni) — "7x longer context"가 수치지 경험이 아님
5. **접근 장벽이 높다** (MimosaTen, Languages_Learner) — C 빌드가 진입 장벽

## Goal

**"직접 써봐"가 가능한 환경** — 수치 증명에서 체험 증명으로 전환.

## Target Metrics

| Metric | Current | Target | 의미 |
|--------|---------|--------|------|
| GitHub Stars | 234 | **500+** | 1차 마일스톤 |
| "직접 돌려봤다" 리포트 | 0 | **10+** | 실사용자 확보 |
| llama.cpp H2H 벤치마크 | 부정확 | **재현 가능** | 기술적 신뢰 확보 |
| pip install 지원 | 없음 | **동작** | Python 유저 접근 |
| 킬러 데모 | 없음 | **Book-in-a-Chat** | 바이럴 콘텐츠 |

## Scope

### Phase 1: 신뢰 확보 (P0)

**1.1 Head-to-head 벤치마크 자동화**

llama.cpp 최신 빌드와 동일 조건 비교. 재현 가능한 스크립트.

- 모델: SmolLM2 1.7B Q8_0
- 하드웨어: M1 Pro 16GB, 8 threads
- 설정:
  - llama.cpp: FP16 KV / Q8_0 K + Q5_0 V / Q4_0 K+V
  - quant.cpp: FP16 / uniform_4b K + Q4 V / delta_3b K + Q4 V
- 측정: PPL (WikiText-2 4K tokens), max context, tok/s, peak memory
- 출력: bench/head_to_head/ 에 스크립트 + CSV + README

**1.2 llama.cpp delta compression PR**

delta KV compression을 llama.cpp에 upstream PR 제출.
- ggml type 등록 (GGML_TYPE_TQ_DELTA_3B)
- kv_cache.cpp에 delta encode/decode 경로 추가
- 벤치마크 포함 PR description 작성

### Phase 2: 킬러 데모 (P0)

**2.1 Book-in-a-Chat 데모**

소설 한 권을 컨텍스트에 넣고 대화하는 시연.
- 데이터: Alice's Adventures in Wonderland (27K tokens, public domain)
- 비교: llama.cpp FP16 KV (OOM at ~50K) vs quant.cpp (전체 로드 + Q&A)
- 출력 형태:
  - CLI 세션 로그 (README에 포함)
  - 30초 asciinema 녹화 (GIF 변환)
  - bench/demo/book_chat.sh 재현 스크립트
- 핵심 질문: "7장에서 체셔 고양이가 뭐라고 했어?" → 정확한 인용 응답

**2.2 WASM 라이브 데모 개선**

GitHub Pages 데모를 실사용 가능 수준으로 개선.
- 모델 자동 다운로드 (SmolLM2 135M, 270MB)
- KV 압축 ON/OFF 토글 + 메모리 사용량 실시간 표시
- URL: quantumaikr.github.io/quant.cpp/

### Phase 3: 접근성 (P1)

**3.1 pip install quantcpp**

Python 패키지로 빌드 없이 사용 가능.
- cffi 또는 ctypes 기반 (외부 빌드 도구 최소화)
- `from quantcpp import Model; m = Model("model.gguf"); print(m.ask("hello"))`
- PyPI 배포, pip install quantcpp 한 줄
- macOS (ARM64), Linux (x86_64) wheel 사전 빌드

**3.2 Windows 프리빌트 바이너리**

GitHub Releases에 Windows x64 바이너리 포함.
- MSVC Release build (quant.exe + quant-server.exe)
- GitHub Actions CI에서 자동 빌드
- Languages_Learner 피드백 직접 대응

**3.3 모델 호환성 CI 매트릭스**

GitHub Actions에서 주요 모델 자동 테스트.
- SmolLM2 135M (빠른 CI용), Llama 3.2 1B, Qwen3.5 0.8B
- 테스트: 로드 + 10토큰 생성 + PPL 5개 (smoke test)
- README에 "Verified Models" 배지 자동 업데이트

## Non-Goals

- GPU 속도 경쟁 (v1.3에서 별도 진행)
- 70B+ 모델 지원 (메모리 제약)
- 학습/파인튜닝 기능
- vLLM 완전 통합 (별도 마일스톤)

## Success Criteria

1. bench/head_to_head/ 벤치마크가 llama.cpp 최신과 공정하게 비교
2. Book-in-a-Chat 데모가 README 최상단에서 30초 안에 임팩트 전달
3. pip install quantcpp가 macOS/Linux에서 동작
4. GitHub Stars 500 도달
