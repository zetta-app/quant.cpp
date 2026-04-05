/**
 * embed_minimal.c — Smallest possible quant.cpp integration
 *
 * Shows how to add LLM to any C project with quant.h.
 * Compile: cc embed_minimal.c -o chat -lm -lpthread
 * Run:     ./chat model.gguf
 */

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
        printf("Usage: %s <model.gguf> [prompt]\n", argv[0]);
        printf("Example: %s smollm2-135m.gguf \"Tell me a joke\"\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = argc > 2 ? argv[2] : "Hello!";

    /* Load model */
    printf("Loading %s...\n", model_path);
    quant_model* model = quant_load(model_path);
    if (!model) {
        printf("Error: failed to load model\n");
        return 1;
    }

    /* Configure: KV compression for 7x longer context */
    quant_config cfg = {
        .temperature = 0.7f,
        .top_p       = 0.9f,
        .max_tokens  = 256,
        .n_threads   = 4,
        .kv_compress = 1,  /* 4-bit KV cache compression */
    };

    quant_ctx* ctx = quant_new(model, &cfg);
    if (!ctx) {
        printf("Error: failed to create context\n");
        quant_free_model(model);
        return 1;
    }

    /* Generate with streaming */
    printf("\n> %s\n", prompt);
    quant_generate(ctx, prompt, print_token, NULL);
    printf("\n");

    /* Cleanup */
    quant_free_ctx(ctx);
    quant_free_model(model);
    return 0;
}
