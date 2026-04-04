#define QUANT_IMPLEMENTATION
#include "../quant.h"
#include <stdio.h>

static void print_token(const char* text, void* ud) {
    (void)ud;
    printf("%s", text);
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }
    const char* prompt = argc > 2 ? argv[2] : "Hello!";

    quant_model* m = quant_load(argv[1]);
    if (!m) { fprintf(stderr, "Failed to load model\n"); return 1; }

    quant_config cfg = {
        .temperature = 0.7f,
        .top_p = 0.9f,
        .max_tokens = 100,
        .n_threads = 4,
        .kv_compress = 1  // 4-bit KV compression
    };
    quant_ctx* c = quant_new(m, &cfg);

    printf("Prompt: %s\n---\n", prompt);
    quant_generate(c, prompt, print_token, NULL);
    printf("\n---\n");

    quant_free_ctx(c);
    quant_free_model(m);
    return 0;
}
