# WBS v1.4 — Community Trust & Killer Demo

기반 문서: [PRD v1.4](../prd/prd_v1.4.md)

---

## Phase 1: 신뢰 확보 (P0)

### 1.1 Head-to-head 벤치마크

- [ ] **1.1.1** llama.cpp 최신 빌드 스크립트
  - `bench/head_to_head/setup_llamacpp.sh`
  - git clone + cmake build + Metal 활성화
  - 특정 커밋 hash 고정 (재현성)

- [ ] **1.1.2** 벤치마크 자동화 스크립트
  - `bench/head_to_head/run_benchmark.sh`
  - 입력: 모델 경로, 반복 횟수
  - 테스트 매트릭스:
    - llama.cpp: FP16 KV / Q8_0K+Q5_0V / Q4_0 K+V
    - quant.cpp: FP16 / uniform_4b+Q4V / delta_3b+Q4V
  - 측정: PPL (4K tokens), max context before OOM, tok/s, peak RSS
  - 출력: CSV + JSON + 요약 테이블

- [ ] **1.1.3** 결과 분석 및 README 반영
  - `bench/head_to_head/README.md` — 재현 방법 + 결과
  - README.md 비교 섹션을 실측 데이터로 교체
  - "이 스크립트를 돌려서 직접 확인하세요" 문구 추가

### 1.2 llama.cpp upstream PR 준비

- [ ] **1.2.1** llama.cpp 코드 스타일 분석
  - ggml type 등록 패턴 파악 (GGML_TYPE_*)
  - kv_cache 양자화 경로 분석 (llama-kv-cache.cpp)
  - coding convention: snake_case, indent style, comment style

- [ ] **1.2.2** delta KV compression 패치 작성
  - ggml_type 추가: GGML_TYPE_DELTA_Q3
  - delta encode: key[t] - reconstruct(key[t-1])
  - delta decode: reconstruct + dequantize
  - I-frame 매 64 토큰 (FP32 anchor)
  - integrations/llamacpp/delta_kv.patch

- [ ] **1.2.3** PR description + 벤치마크 결과 포함
  - docs/pr/llamacpp-upstream-pr.md
  - 제목, 요약, 벤치마크 테이블, 테스트 방법

---

## Phase 2: 킬러 데모 (P0)

### 2.1 Book-in-a-Chat 데모

- [ ] **2.1.1** 데모 데이터 준비
  - `bench/demo/alice.txt` — Alice in Wonderland (Project Gutenberg)
  - `bench/demo/prepare_book.py` — 텍스트 정리 + 토큰 수 측정
  - 목표: ~27K 토큰 (Llama tokenizer 기준)

- [ ] **2.1.2** 데모 스크립트
  - `bench/demo/book_chat.sh`
  - Step 1: llama.cpp로 전체 책 로드 시도 → OOM 표시
  - Step 2: quant.cpp로 전체 책 로드 (KV 압축) → 성공
  - Step 3: 질문 3개 자동 실행:
    1. "Chapter 7에서 Mad Hatter가 Alice에게 한 수수께끼는?"
    2. "Queen of Hearts의 첫 등장 장면을 인용해줘"
    3. "Cheshire Cat이 사라지는 장면을 요약해줘"
  - 출력: 정리된 세션 로그

- [ ] **2.1.3** 녹화 및 README 반영
  - asciinema 또는 script + GIF 변환
  - README.md 최상단 "See it in action" 섹션 추가
  - "같은 Mac에서 한쪽은 OOM, 한쪽은 책 전체를 기억한다"

### 2.2 WASM 데모 개선

- [ ] **2.2.1** 모델 자동 다운로드 UI
  - index.html에 모델 선택 드롭다운 (SmolLM2 135M 기본)
  - fetch()로 HuggingFace에서 직접 다운로드
  - 프로그레스 바 표시

- [ ] **2.2.2** KV 압축 토글 + 메모리 표시
  - 실시간 메모리 사용량 (WASM heap) 표시
  - KV 압축 ON/OFF 토글 → 메모리 차이 시각화
  - "Compressed: 12MB / Uncompressed: 48MB" 실시간 표시

- [ ] **2.2.3** GitHub Pages 배포
  - .github/workflows/pages.yml
  - quantumaikr.github.io/quant.cpp/ 라이브 URL 확인

---

## Phase 3: 접근성 (P1)

### 3.1 pip install quantcpp

- [ ] **3.1.1** Python 패키지 구조
  - `bindings/python/quantcpp/__init__.py` — Model, Context 클래스
  - `bindings/python/setup.py` — cffi 빌드 설정
  - `bindings/python/pyproject.toml` — PEP 517 빌드

- [ ] **3.1.2** C 라이브러리 빌드 통합
  - setup.py에서 cmake 자동 호출 → libturboquant.a 빌드
  - cffi 또는 ctypes로 .so/.dylib 로드
  - fallback: quant.h 단일 헤더 직접 컴파일

- [ ] **3.1.3** API 구현
  - `Model(path)` — quant_load() 래핑
  - `Model.ask(prompt)` — quant_ask() 래핑
  - `Model.generate(prompt, callback)` — 스트리밍
  - `Model.config(temperature, top_p, kv_compress)` — 설정

- [ ] **3.1.4** 테스트 및 PyPI 배포
  - `tests/test_python.py` — 로드 + 생성 + 해제 테스트
  - macOS ARM64, Linux x86_64 wheel 빌드
  - `pip install quantcpp` 동작 확인

### 3.2 Windows 프리빌트 바이너리

- [ ] **3.2.1** GitHub Actions Windows CI
  - `.github/workflows/release-windows.yml`
  - MSVC 2022 Release build
  - 출력: quant.exe, quant-server.exe

- [ ] **3.2.2** GitHub Release 자동 첨부
  - Release 태그 생성 시 자동 빌드 + 업로드
  - README에 "Download Windows binary" 링크 추가

### 3.3 모델 호환성 CI

- [ ] **3.3.1** CI 워크플로 작성
  - `.github/workflows/model-compat.yml`
  - 모델: SmolLM2 135M Q8_0 (HuggingFace에서 다운로드)
  - 테스트: 로드 + 10토큰 생성 + greedy 출력 일치 검증

- [ ] **3.3.2** README 배지 자동 업데이트
  - CI 결과를 배지로 표시 (pass/fail)
  - "Verified on: SmolLM2 135M, ..." 자동 생성

---

## 완료 기준

| 항목 | 기준 | 검증 방법 |
|------|------|-----------|
| H2H 벤치마크 | 스크립트 1개로 전체 재현 | `bash bench/head_to_head/run_benchmark.sh model.gguf` |
| Book-in-a-Chat | 30초 GIF로 임팩트 전달 | README 최상단 확인 |
| pip install | macOS/Linux에서 설치 → ask() 동작 | `pip install quantcpp && python -c "..."` |
| Windows 바이너리 | Release 페이지에서 다운로드 → 실행 | GitHub Releases 확인 |
| 모델 CI | PR마다 자동 검증 | GitHub Actions 배지 |

---

## 의존성 그래프

```
Phase 1 (신뢰)
  1.1 H2H 벤치마크 ──────┐
  1.2 llama.cpp PR ──────┤
                          ▼
Phase 2 (데모)           README 업데이트
  2.1 Book-in-a-Chat ────┤
  2.2 WASM 개선 ─────────┘
                          
Phase 3 (접근성)         (독립)
  3.1 pip install ────── PyPI
  3.2 Windows binary ─── GitHub Release
  3.3 모델 CI ─────────── GitHub Actions
```

Phase 1과 2는 순차 (벤치마크 결과가 데모와 README에 반영), Phase 3은 독립 병렬 가능.
