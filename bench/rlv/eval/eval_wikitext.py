#!/usr/bin/env python3
"""D5 Karpathy gate: 8000-token wikitext stress test.

Purpose: prove that on cliff-overflow documents (>1024 tokens, ~6× over
the 3B Q4 working memory limit), RLV beats both pure long-context
inference (which hits the cliff) and a vector-RAG-style chunk retriever
(which fails on cross-chunk and ambiguous queries).

The Acme benchmark in eval_acme.py used a sub-cliff document (1895
chars, ~500 tokens) — that's the regime where pure long-context is
strongest. Day 3 demonstrated parity in that regime. Day 4-5 demonstrates
the *value proposition*: when documents grow past the cliff, RLV's
chunked-and-routed approach is the only thing that still works.

Document: bench/data/ppl_8k.txt (~9000 tokens, three Wikipedia articles
concatenated — Robert Boulter, Du Fu, One Direction "Kiss You").

Three systems compared:
  1. RLV — full 5-stage pipeline (gist → locator → lookup → verify → research)
  2. long-context — entire 9000-token doc dumped into a single prompt
     (cliff overflow expected; we disable the cliff check via
     enforce_budget=False to actually run the call).
  3. vector-rag — chunk-based keyword search picks top-1 chunk, then a
     single LLM call answers the question on that chunk.

Scoring: keyword fragment match (same contract as eval_acme.py).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlv_orchestrator import answer_question
from stages import _llm
from stages import gist as gist_stage
from stages.locator import _question_keywords, _term_in_text, _normalize


# ----------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------
DOC_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ppl_8k.txt"


# ----------------------------------------------------------------------------
# Questions — 10 across 3 articles, mix of single-hop and multi-hop
# ----------------------------------------------------------------------------
# The wikitext doc has no paragraph breaks (it's wikitext-2 raw format), so
# every sentence is on a single line. Questions target facts that are
# uniquely answerable from the source.
QUESTIONS = [
    # === Robert Boulter (chars 0-~6500) ===
    {
        "id": 1, "topic": "boulter", "type": "single-hop",
        "question": "What is Robert Boulter's nationality?",
        "fragments": ["english"],
    },
    {
        "id": 2, "topic": "boulter", "type": "single-hop",
        "question": "Who wrote the play Herons that Boulter starred in at the Royal Court Theatre?",
        "fragments": ["simon stephens", "stephens"],
    },
    {
        "id": 3, "topic": "boulter", "type": "single-hop",
        "question": "Who directed the production of Mercury Fur in which Boulter appeared?",
        "fragments": ["john tiffany", "tiffany"],
    },
    {
        "id": 4, "topic": "boulter", "type": "single-hop",
        "question": "Who directed the 2008 film Donkey Punch?",
        "fragments": ["olly blackburn", "blackburn"],
    },

    # === Du Fu (chars ~7000-~26000) ===
    {
        "id": 5, "topic": "dufu", "type": "single-hop",
        "question": "In what year did Du Fu first meet Li Bai?",
        "fragments": ["744"],
    },
    {
        "id": 6, "topic": "dufu", "type": "single-hop",
        "question": "When did the An Lushan Rebellion begin?",
        "fragments": ["december 755", "755"],
    },
    {
        "id": 7, "topic": "dufu", "type": "single-hop",
        "question": "What position was Du Fu demoted to in Huazhou in 758?",
        "fragments": ["commissioner of education", "commissioner", "education"],
    },
    {
        "id": 8, "topic": "dufu", "type": "multi-hop",
        "question": "According to Hung, why did Du Fu fail his civil service exam?",
        "fragments": ["dense", "obscure", "prose style"],
    },

    # === One Direction "Kiss You" (chars ~28000-end) ===
    {
        "id": 9, "topic": "kiss_you", "type": "single-hop",
        "question": "How many million views did the Kiss You music video receive in the first 24 hours?",
        "fragments": ["10", "10.4"],
    },
    {
        "id": 10, "topic": "kiss_you", "type": "single-hop",
        "question": "Who directed the Kiss You music video?",
        "fragments": ["vaughan arnell", "arnell"],
    },
]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def fuzzy_hit(text: str, fragments: list[str]) -> tuple[bool, list[str]]:
    """Same scoring contract as eval_acme.py — any one matched fragment passes."""
    t = text.lower()
    matched = [f for f in fragments if f in t]
    return (len(matched) > 0, matched)


# ----------------------------------------------------------------------------
# Baseline 1: long-context (entire doc in one prompt)
# ----------------------------------------------------------------------------
LONG_CONTEXT_PROMPT_TEMPLATE = """Document:
{doc}

Question: {question}

Answer in one short sentence."""


def run_long_context(question: str, doc_text: str, *, verbose: bool = False) -> str:
    """Dump the entire doc into a single LLM call. Cliff check disabled
    so we can actually run the cliff-overflow regime."""
    prompt = LONG_CONTEXT_PROMPT_TEMPLATE.format(doc=doc_text, question=question)
    if verbose:
        est = _llm.estimate_tokens(prompt)
        print(f"  [long-context] prompt ~{est} tokens (cliff=1024, "
              f"overflow={est/1024:.1f}x)")
    result = _llm.llm_call(prompt, max_tokens=128, enforce_budget=False)
    return result.text.strip()


# ----------------------------------------------------------------------------
# Baseline 2: vector-RAG (keyword TF retrieval over chunks + single answer call)
# ----------------------------------------------------------------------------
VECTOR_RAG_PROMPT_TEMPLATE = """Context:
{chunk}

Question: {question}

Answer in one short sentence."""


def _vector_rag_score(question: str, chunk_full_text: str) -> float:
    """Naive TF score: sum of weighted question keyword matches in chunk."""
    weighted = _question_keywords(question)
    text_norm = _normalize(chunk_full_text)
    score = 0.0
    for term, weight in weighted:
        if _term_in_text(term, text_norm):
            score += weight
    return score


def run_vector_rag(
    question: str,
    doc_text: str,
    cached_gist,
    *,
    verbose: bool = False,
) -> str:
    """Pick top-1 chunk by keyword score, then a single LLM call to answer.

    This is a "fair" RAG baseline: it uses the same gist chunking as RLV
    so chunk size is identical, but it skips the locator's title-bonus
    weighting and the verifier/research stages — it's "retrieve-then-answer"
    in one shot.
    """
    best = max(cached_gist.chunks, key=lambda c: _vector_rag_score(question, c.full_text))
    if verbose:
        print(f"  [vector-rag] picked chunk {best.chunk_id} "
              f"({best.head_text[:60]!r}...)")
    chunk_text = doc_text[best.char_start:best.char_end]
    prompt = VECTOR_RAG_PROMPT_TEMPLATE.format(chunk=chunk_text, question=question)
    result = _llm.llm_call(prompt, max_tokens=128)
    return result.text.strip()


# ----------------------------------------------------------------------------
# Baseline 3: RLV (full pipeline)
# ----------------------------------------------------------------------------
def run_rlv(
    question: str,
    doc_text: str,
    cached_gist,
    *,
    verbose: bool = False,
) -> tuple[str, dict]:
    """Run the full RLV pipeline."""
    r = answer_question(
        doc_text, question,
        doc_id="wikitext_8k",
        cached_gist=cached_gist,
        verbose=verbose,
    )
    # Aggregate every place the answer might live (matches eval_acme.py contract)
    parts = [r["final_answer"]]
    for a in r["research"]["attempts"]:
        parts.append(a.get("answer", "") or "")
    return " ".join(parts), r


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only", type=int, default=None,
                        help="Only run the question with this id")
    parser.add_argument("--systems", default="rlv,long-context,vector-rag",
                        help="Comma-separated list of systems to run")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",")]

    doc_text = DOC_PATH.read_text(encoding="utf-8", errors="replace")
    print("=" * 76)
    print("D5 stress test: RLV vs long-context vs vector-RAG on 8000-token wikitext")
    print("=" * 76)
    print(f"Document: {DOC_PATH.name}")
    print(f"  chars: {len(doc_text)}")
    print(f"  est tokens: ~{_llm.estimate_tokens(doc_text)}")
    print(f"  cliff:      1024 (3B Q8 model)")
    print(f"  overflow:   {_llm.estimate_tokens(doc_text)/1024:.1f}x cliff")
    print(f"Systems: {systems}")
    print("-" * 76)

    _llm.start_server()
    t_start = time.time()
    try:
        # Build the gist ONCE — RLV and vector-RAG share it for fair comparison.
        print("[setup] building gist (one-time, no LLM)...")
        cached_gist = gist_stage.build_gist(doc_text, doc_id="wikitext_8k", verbose=False)
        print(f"[setup] gist has {len(cached_gist.chunks)} chunks")
        if args.verbose:
            for c in cached_gist.chunks:
                print(f"  [{c.chunk_id}] start={c.char_start} end={c.char_end} "
                      f"({c.char_end-c.char_start} chars) head={c.head_text[:60]!r}")
        print()

        results = {q["id"]: {"q": q, "rlv": None, "long-context": None, "vector-rag": None,
                              "rlv_t": 0.0, "long-context_t": 0.0, "vector-rag_t": 0.0,
                              "rlv_ok": False, "long-context_ok": False, "vector-rag_ok": False}
                   for q in QUESTIONS}

        for q in QUESTIONS:
            if args.only is not None and q["id"] != args.only:
                continue
            print(f"--- Q{q['id']} ({q['topic']}, {q['type']}) ---")
            print(f"Q: {q['question']}")
            print(f"  expected fragments: {q['fragments']}")

            if "rlv" in systems:
                t0 = time.time()
                try:
                    text, r = run_rlv(q["question"], doc_text, cached_gist, verbose=args.verbose)
                except Exception as e:
                    text = f"[ERROR: {type(e).__name__}: {e}]"
                    r = None
                elapsed = time.time() - t0
                ok, matched = fuzzy_hit(text, q["fragments"])
                results[q["id"]]["rlv"] = text
                results[q["id"]]["rlv_t"] = elapsed
                results[q["id"]]["rlv_ok"] = ok
                mark = "PASS" if ok else "FAIL"
                final = r["final_answer"] if r else text
                print(f"  [RLV {mark}] ({elapsed:.1f}s) {final[:120]!r}")
                if ok:
                    print(f"        matched: {matched}")

            if "long-context" in systems:
                t0 = time.time()
                try:
                    text = run_long_context(q["question"], doc_text, verbose=args.verbose)
                except Exception as e:
                    text = f"[ERROR: {type(e).__name__}: {e}]"
                elapsed = time.time() - t0
                ok, matched = fuzzy_hit(text, q["fragments"])
                results[q["id"]]["long-context"] = text
                results[q["id"]]["long-context_t"] = elapsed
                results[q["id"]]["long-context_ok"] = ok
                mark = "PASS" if ok else "FAIL"
                print(f"  [LC  {mark}] ({elapsed:.1f}s) {text[:120]!r}")
                if ok:
                    print(f"        matched: {matched}")

            if "vector-rag" in systems:
                t0 = time.time()
                try:
                    text = run_vector_rag(q["question"], doc_text, cached_gist, verbose=args.verbose)
                except Exception as e:
                    text = f"[ERROR: {type(e).__name__}: {e}]"
                elapsed = time.time() - t0
                ok, matched = fuzzy_hit(text, q["fragments"])
                results[q["id"]]["vector-rag"] = text
                results[q["id"]]["vector-rag_t"] = elapsed
                results[q["id"]]["vector-rag_ok"] = ok
                mark = "PASS" if ok else "FAIL"
                print(f"  [VR  {mark}] ({elapsed:.1f}s) {text[:120]!r}")
                if ok:
                    print(f"        matched: {matched}")
            print()

    finally:
        _llm.stop_server()

    total_time = time.time() - t_start

    rows = [r for r in results.values() if r["rlv"] is not None or r["long-context"] is not None or r["vector-rag"] is not None]

    print("=" * 76)
    print(f"RESULTS — total {total_time:.1f}s")
    print("=" * 76)
    print(f"{'Q':>2} {'topic':<10} {'type':<10}  {'RLV':<6} {'LC':<6} {'VR':<6}")
    rlv_pass = lc_pass = vr_pass = 0
    for r in rows:
        q = r["q"]
        rlv_m = "OK" if r["rlv_ok"] else ("--" if r["rlv"] is None else "XX")
        lc_m  = "OK" if r["long-context_ok"] else ("--" if r["long-context"] is None else "XX")
        vr_m  = "OK" if r["vector-rag_ok"] else ("--" if r["vector-rag"] is None else "XX")
        if r["rlv_ok"]: rlv_pass += 1
        if r["long-context_ok"]: lc_pass += 1
        if r["vector-rag_ok"]: vr_pass += 1
        print(f"{q['id']:>2} {q['topic']:<10} {q['type']:<10}  {rlv_m:<6} {lc_m:<6} {vr_m:<6}")

    n = len(rows)
    print()
    print(f"  RLV         : {rlv_pass}/{n}")
    print(f"  long-context: {lc_pass}/{n}")
    print(f"  vector-rag  : {vr_pass}/{n}")
    print()
    if rlv_pass > lc_pass and rlv_pass > vr_pass:
        print(f"  D5 gate: RLV WINS ✅  (RLV={rlv_pass}/{n}, LC={lc_pass}/{n}, VR={vr_pass}/{n})")
    else:
        print(f"  D5 gate: TIE/LOSS  (RLV={rlv_pass}/{n}, LC={lc_pass}/{n}, VR={vr_pass}/{n})")
    return 0 if rlv_pass >= max(lc_pass, vr_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
