# r/LocalLLaMA Comment Responses — "LLM inference in a single C header file" (2026-04-06)

Copy-paste ready. Each section = one comment.

---

## @Revolutionalredstone — "dependencies are the difference between bad and good software"

Thank you! That's exactly the philosophy. The entire dependency list is libc + pthreads — things your OS already has. No package manager, no version conflicts, no "it works on my machine."

If you want a good reading path: start with the 6-function API at the top of `quant.h`, then follow `quant_generate()` into the forward pass. The attention loop is the most interesting part — you can see exactly how KV compression slots in without changing the matmul logic. Enjoy the read!

---

## @Languages_Learner — "Still waiting for Windows binary"

Windows is supported! Two options:

**Single header (easiest):**
```
cl app.c /O2 /link /out:app.exe
```
Or with MinGW: `gcc app.c -o app.exe -lm -lpthread`

**Full build:**
```
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

We added MSVC compatibility recently — `CreateFileMapping`/`MapViewOfFile` for mmap, `_aligned_malloc` for alignment, etc. If you hit any compile issue, please file an issue — we treat Windows build failures as bugs.

We don't ship prebuilt binaries yet, but that's a fair request. I'll add it to the next release.
