# r/LocalLLM Comment Responses — "TurboQuant.cpp — 1-bit KV cache" (2026-04-05)

Thread: https://www.reddit.com/r/LocalLLM/comments/.../turboquantcpp_1bit_kv_cache/

Copy-paste ready. Each section = one comment.

---

## @Big_River_ — (both comments)

(Skip. No actionable content to respond to.)

---

## @snapo84 — "test on long outputs / thinking models / KLD decrease"

Great point — this is the right test and we haven't done it yet.

Our current benchmarks are short: 101-token and 999-token perplexity runs, plus greedy output matching on short prompts. That's enough to validate the basic quantization math, but it doesn't stress-test the failure mode you're describing: accumulated drift over thousands of tokens in a thinking chain.

The concern is real. 1-bit key reconstruction has cosine similarity ~0.634 (the information-theoretic limit of 2/pi). Over a long chain-of-thought, small attention errors compound — token 3000 is conditioned on every previous softmax distribution, so per-step error accumulates multiplicatively.

In fact, after our initial post we found a bug where an FP32 fallback was masking the true 1-bit quality. Once fixed, 1-bit is not practically usable for production. What does work:

- **4-bit K + Q4 V**: PPL +0.0% on WikiText-2 (genuinely lossless, even on longer sequences)
- **Delta 3-bit K + Q4 V**: PPL +1.3% with I-frames every 64 tokens to prevent drift

For a proper long-output test like you're describing — same seed, quantized vs unquantized, measuring token-level divergence over a full thinking trace — that's on the roadmap. If you have a specific thinking model + prompt pair you'd want tested, happy to run it.
