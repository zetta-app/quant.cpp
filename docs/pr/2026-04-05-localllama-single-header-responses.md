# r/LocalLLaMA Comment Responses — "LLM inference in a single C header file" (2026-04-05)

Thread: https://www.reddit.com/r/LocalLLaMA/comments/.../llm_inference_in_a_single_c_header_file/

Copy-paste ready. Each section = one comment.

---

## @MelodicRecognition7 — "15k lines of Claude vomit"

Yes, every commit has a Co-Authored-By tag because I use Claude Code and don't hide it. Same way people use Copilot or Cursor — it's a development tool, not the developer.

The architecture, the algorithm choices, the quantization types — those are mine. When we had a bug where an FP32 key cache was silently bypassing the quantized path, making our 1-bit results look impossibly good, Claude didn't catch it. I found it by reading the attention loop, pulled every claim based on that measurement, and rewrote the benchmarks from scratch.

The code compiles, the tests pass, the PPL numbers are reproducible. `./quant model.gguf --ppl input.txt -k uniform_4b -v q4` — verify it yourself.

---

## @Live-Crab3086 — "compilers rather than hand-coding assembly"

Ha — exactly. The output is what matters. `cc demo.c -lm -lpthread`, run the PPL benchmark, read the source if something looks wrong. The tools don't invalidate the results.

---

## @No_Pilot_1974 — "more potential for embedded solutions"

That's exactly the use case we're targeting. The single-header `quant.h` was designed for this — drop it into an iOS app, a game engine, an IoT device, anywhere you have a C compiler but no room for a full inference framework.

We also have a WASM build (192KB binary) that runs entirely in the browser — same idea, inference as a library call rather than a server dependency.

What kind of embedded platform are you thinking about?
