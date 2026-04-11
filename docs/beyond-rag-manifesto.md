# Beyond RAG: A Manifesto

## TL;DR

> **Chunking RAG was a workaround for small context windows. The workaround became dogma. Now context windows are big enough that we don't need the workaround. Welcome to Beyond RAG.**

## Where We Are

In 2023, every "production AI" stack looked like this:

```
[document] → [chunker] → [embedder] → [vector DB]
                                            ↓
[user query] → [embedder] → [retriever] → [reranker] → [LLM] → [answer]
```

Six moving parts. Four of them exist solely because the LLM at the end couldn't fit your whole document in its context window.

This was a reasonable engineering compromise. Llama 1 had 2K context. GPT-3.5 had 4K. You had to chunk.

Then context windows grew. Llama 3.2 has 128K. Claude 3 has 200K. Gemini 1.5 has 2M. The compromise should have started disappearing.

It didn't. The infrastructure became dogma. The vector DB companies became billion-dollar valuations. The "RAG pipeline" became something every AI engineer was expected to build, regardless of whether their use case actually needed one.

## What We Measured

We tested chunk-RAG vs full-document context on a 5-section synthetic document with 7 questions, using Llama 3.2 3B Q8_0:

| Method | Accuracy |
|---|---:|
| Chunk-RAG (wrong section retrieved) | **0/7** |
| Full Document (FP32 KV) | **7/7** |
| Full Document (6.4× compressed KV) | **7/7** |

When chunk-RAG retrieved the wrong section, the model didn't say "I don't know." It made up answers:

| Question | Chunk-RAG hallucination | Truth |
|---|---|---|
| Who is the CTO? | "John Smith" | Maria Santos |
| What is the revenue? | "$1,000,000" | 847 million |
| What % is R&D? | "15% of net income" | 14% of revenue |

This is the failure mode no one is monitoring. **Your dashboards show 100% uptime. Your users get plausible-sounding lies.**

## Why This Happens

When you give an LLM a partial context and ask a question whose answer isn't in that context, two things can happen:

1. The model says "I don't know based on the provided context."
2. The model fills in the gap with the most likely-sounding answer.

Modern instruction-tuned models do **#2** by default. Their training rewards "give a confident answer" more than "admit uncertainty." Combined with RAG's silent retrieval failures, this creates a system that confidently lies whenever its retriever misses.

You can mitigate with prompt engineering, confidence thresholds, fine-tuning. None of them fix the root cause: **the LLM only sees a fragment**.

## The Beyond RAG Pattern

When the document fits in the context window, the entire stack collapses:

```
[document] ───────────────────────────────────────→ [LLM] → [answer]
                                                   (full context)
```

Three steps become one. The hallucination failure mode disappears because **the LLM has all the information**. There's nothing for it to hallucinate.

This isn't theoretical. It's just engineering: you need a context window big enough to fit your document, and you need it to fit on hardware you have.

That's where KV cache compression comes in. quant.cpp's 6.4× compression means a 128K-token context for a 3B model fits in **9.5 GB on a 16GB Mac**. Llama 3.2 3B + your full company manual + the user's question, all running locally, no cloud, no vector DB, no retriever to fail silently.

## When Beyond RAG Wins

| Use case | Best approach |
|---|---|
| Chat with one document (manual, paper, novel) | **Beyond RAG** |
| Codebase analysis (single repo) | **Beyond RAG** |
| Customer support over a product manual | **Beyond RAG** |
| Long conversation memory | **Beyond RAG** |
| Search across 100K product reviews | RAG (still) |
| Search across all of Wikipedia | RAG (still) |
| Multi-tenant systems with millions of docs | Hybrid: RAG + Beyond RAG |

The right question isn't "RAG or no RAG." It's **"is my entire context small enough to fit?"** If yes, skip the chunker. If no, use RAG to narrow the candidates, then load the survivors fully.

This is **document-level RAG**: retrieval at the document level, not the chunk level. You still get the recall of search. You still get the precision of full context. You don't get the hallucination from chunking.

## What Beyond RAG Is Not

- **Not "RAG is dead."** RAG is essential when your corpus exceeds context. We're saying: stop pretending it's the only tool.
- **Not "use Gemini 1.5 Pro for everything."** Cloud LLMs cost money per token, leak data, and require internet. Beyond RAG runs locally.
- **Not "vector DBs are obsolete."** They're great for what they are. They're just often misused as a hammer for non-nail problems.
- **Not a finished idea.** This is v1. We measured 7 questions on 1 model. Real validation needs LongBench, NIAH, multiple models, real corpora. We're going there.

## What We're Asking

If you're building production RAG, run our 5-minute benchmark on your own data:

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
# Adapt bench/document_level_rag_test.sh to your document + questions
bash bench/document_level_rag_test.sh
```

When chunk-RAG fails on your data, see what your users would have seen.

If the hallucinations bother you — and they should — try the alternative:

```python
pip install quantcpp
```

```python
from quantcpp import Model
m = Model.from_pretrained("Llama-3.2-3B", aggressive=True)
m.ask(open("your_document.txt").read() + "\n\nQuestion: ...")
```

No vector DB. No chunker. No retriever. No silent failure.

Just the model and the document.

## The Goal

Five years from now, "RAG" should mean "retrieve documents to load into context" — the way we use the word "search" today. It shouldn't mean "chunk-and-embed-and-pray."

We're not the only ones thinking this. Anthropic's contextual retrieval, Gemini's 2M context, the long-context benchmark community — everyone is moving toward the same insight from different directions.

quant.cpp is one tool: the one that makes Beyond RAG practical on consumer hardware. There will be others. Together, we move past the workaround.

> **Welcome to Beyond RAG. Bring your documents.**

---

## Honest Disclaimers

- This is a v1 finding. 5 sections, 7 questions, 1 model. We're not claiming a paper. We're starting a conversation.
- Q4 weight quantization produces visual artifacts ("Santos" → "SanSannt"). Semantically correct, visually noisy. Use Q8 weights for production.
- 1B models lack reliable instruction-following for QA. Use 3B+.
- Beyond RAG only works when the document fits. For large corpora, hybrid is needed.
- We'll update this manifesto with v2 evidence (LongBench, real corpora) when it's ready.

## Track Record

quant.cpp has **11 self-found, publicly-corrected claims** in its honest correction track. We don't ship vibes; we ship measurements. When this manifesto is wrong, we'll correct it and tell you what we got wrong.

## Sign On

If you've shipped a RAG system that hallucinated in production, we'd love to hear what failure mode it was. Open an issue or DM the maintainers. Real-world data > synthetic benchmarks.

If you want to validate Beyond RAG on a real benchmark, we'd love a PR.

If you think this manifesto is wrong, even better. Tell us why.

> *Written 2026-04-11. v1. We'll be wrong about something. We'll fix it in public.*
