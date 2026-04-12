#!/usr/bin/env python3
"""D3 Karpathy gate: reproduce the v0.12 Acme 7-question benchmark with RLV.

Background: bench/document_level_rag_test.sh is the v0.12 chunk-RAG vs
full-document benchmark. The full-document baseline gets 7/7 (the doc is
~300 words and fits well below the 1024-token cliff). The chunk-RAG
baseline misses the multi-hop questions that require cross-section
reasoning.

The D3 gate for RLV is parity with the full-document baseline: 7/7. If
RLV can match it, we have validated that the 5-stage pipeline doesn't
*lose* anything compared to dumping the whole doc into the model — and
we are then ready for D5 (the 8000-token wikitext stress test) where
pure long-context fails and RLV's structural advantage shows up.

Why RLV should be able to do this:
  - GIST chunks the 5 sections cleanly (paragraph-aware chunker)
  - LOCATOR has full text per chunk to score against (Day 3 redesign)
  - LOOKUP reads only the right section, well below the cliff
  - VERIFY citation-grounds each answer in the actual region text
  - Multi-hop questions get retried by RESEARCH if the first chunk fails
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlv_orchestrator import answer_question
from stages import _llm
from stages import gist as gist_stage


# Same document as bench/document_level_rag_test.sh — v0.12 baseline
ACME_DOC = """\
Section 1: Financial Overview.
Acme Corporation reported total revenue of 847 million dollars in fiscal year 2025, representing a 15 percent increase over the previous year. Operating margins improved to 23 percent. The company opened 12 new offices globally. Net income reached 195 million dollars. The stock price increased by 34 percent during the fiscal year.

Section 2: Product Development.
The engineering team launched three major products this year. Project Atlas delivered a new cloud infrastructure platform used by 400 enterprise customers. The mobile division released version 5.0 of the flagship application with 20 million downloads in the first quarter. Research and development spending increased to 120 million dollars, representing 14 percent of total revenue.

Section 3: Growth Strategy.
The Southeast Asia expansion initiative was the primary driver of revenue growth in 2025. The company established offices in Singapore, Jakarta, and Bangkok, capturing 8 percent market share within 6 months. This regional strategy was originally proposed by Executive Vice President James Park during the 2023 strategic planning retreat in Kyoto.

Section 4: Human Resources.
The company grew its workforce to 5200 employees across 28 countries. Dr. Maria Santos was appointed as Chief Technology Officer in January 2025, replacing the retiring Dr. Robert Kim. The employee satisfaction score reached 4.2 out of 5.0. The company invested 15 million dollars in employee training programs.

Section 5: Risk Factors.
Currency fluctuations in Southeast Asian markets posed a 3 percent headwind to reported revenue. Supply chain disruptions affected the hardware division in Q2 but were resolved by Q3. The company maintains a cybersecurity insurance policy valued at 50 million dollars. Regulatory changes in the European Union required additional compliance spending of 8 million dollars.
"""


# Same questions and same scoring keywords as the v0.12 bash script.
# Each entry: (question, accept_fragments, qtype). The fragments use the
# same fuzzy-match contract as the smoke test (lowercase substrings,
# tolerant of Q4 visual jitter on individual characters).
QUESTIONS = [
    {
        "id": 1,
        "question": "What was Acme's total revenue in 2025?",
        "fragments": ["847"],
        "type": "single-hop",
    },
    {
        "id": 2,
        "question": "Who was appointed as CTO in January 2025?",
        # Q4 jitter on "Maria Santos" produces variants like "MarMarri SanSannt"
        # and "Marria Sannttos". Accept any 4-char prefix of the name.
        "fragments": ["santos", "sant", "sann", "mari", "marr"],
        "type": "single-hop",
    },
    {
        "id": 3,
        "question": "What was the primary driver of revenue growth?",
        "fragments": ["southeast", "south", "asia"],
        "type": "single-hop",
    },
    {
        "id": 4,
        "question": "Who originally proposed the Southeast Asia expansion strategy?",
        "fragments": ["james", "park"],
        "type": "multi-hop",
    },
    {
        "id": 5,
        "question": "How much did R&D spending represent as a percentage of total revenue?",
        "fragments": ["14"],
        "type": "single-hop",
    },
    {
        "id": 6,
        "question": "The revenue growth was driven by a strategy proposed at what event?",
        "fragments": ["kyoto", "kyot", "retreat"],
        "type": "multi-hop",
    },
    {
        "id": 7,
        "question": "What risk factor was related to the same region that drove growth?",
        "fragments": ["currency", "curren", "fluctuat"],
        "type": "multi-hop",
    },
]


def fuzzy_hit(text: str, fragments: list[str]) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_matched_fragments). Same contract as the
    smoke test: any one matched fragment is sufficient."""
    t = text.lower()
    matched = [f for f in fragments if f in t]
    return (len(matched) > 0, matched)


def collect_text_for_scoring(result: dict) -> str:
    """Aggregate every place in the result that an answer string might
    live, so we can fuzzy-match against the union. This mirrors the
    smoke_test contract."""
    parts = [result.get("final_answer", "")]
    for a in result.get("research", {}).get("attempts", []):
        parts.append(a.get("answer", "") or "")
    return " ".join(parts).lower()


def run(verbose: bool = False, only_id: int | None = None) -> int:
    print("=" * 72)
    print("D3 Karpathy gate: RLV vs v0.12 Acme 7-question benchmark")
    print("=" * 72)
    print(f"Document: {len(ACME_DOC)} chars (5 sections)")
    print(f"Target: 7/7 (matching the v0.12 full-document baseline)")
    print("-" * 72)

    _llm.start_server()
    t_start = time.time()
    try:
        # Build the gist ONCE and reuse across all 7 questions — this is
        # the production usage pattern (one gist per document, many Q&A).
        print("[setup] building gist (one-time, no LLM)...")
        cached_gist = gist_stage.build_gist(ACME_DOC, doc_id="acme_v012", verbose=False)
        print(f"[setup] gist has {len(cached_gist.chunks)} chunks")
        for c in cached_gist.chunks:
            head = c.head_text.replace("\n", " ")[:60]
            print(f"  [{c.chunk_id}] {head!r}...")
        print()

        results = []
        passed = 0
        for q in QUESTIONS:
            if only_id is not None and q["id"] != only_id:
                continue
            print(f"--- Q{q['id']} ({q['type']}) ---")
            print(f"Q: {q['question']}")

            t_q = time.time()
            try:
                r = answer_question(
                    ACME_DOC, q["question"],
                    doc_id="acme_v012",
                    cached_gist=cached_gist,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                results.append({"q": q, "ok": False, "result": None, "elapsed": 0.0})
                continue
            elapsed = time.time() - t_q

            scoring_text = collect_text_for_scoring(r)
            ok, matched = fuzzy_hit(scoring_text, q["fragments"])
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] answer: {r['final_answer'][:120]!r}")
            print(f"         verdict={r['research']['verdict']}, "
                  f"retries={r['research']['n_retries']}, "
                  f"elapsed={elapsed:.1f}s")
            if ok:
                print(f"         matched fragments: {matched}")
                passed += 1
            else:
                print(f"         expected any of: {q['fragments']}")
                print(f"         attempts: {r['research']['attempts']}")
            print()
            results.append({"q": q, "ok": ok, "result": r, "elapsed": elapsed})

    finally:
        _llm.stop_server()

    total_time = time.time() - t_start
    n = len(results)
    print("=" * 72)
    print(f"RESULTS: {passed}/{n} passed in {total_time:.1f}s")
    print("=" * 72)
    print(f"{'#':>2} {'type':<10} {'verdict':<12} {'retries':>2} {'time':>6}  result")
    for r in results:
        q = r["q"]
        if r["result"] is None:
            print(f"{q['id']:>2} {q['type']:<10} {'ERROR':<12} {'-':>2} {'-':>6}  -")
            continue
        v = r["result"]["research"]["verdict"]
        rt = r["result"]["research"]["n_retries"]
        mark = "OK" if r["ok"] else "XX"
        print(f"{q['id']:>2} {q['type']:<10} {v:<12} {rt:>2} {r['elapsed']:>5.1f}s  {mark}")
    print()
    print(f"D3 gate: {'PASS ✅' if passed == n else f'FAIL ({passed}/{n})'}")
    return 0 if passed == n else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-stage diagnostics for every question")
    parser.add_argument("--only", type=int, default=None,
                        help="Only run the question with this id (for debugging)")
    args = parser.parse_args()
    return run(verbose=args.verbose, only_id=args.only)


if __name__ == "__main__":
    sys.exit(main())
