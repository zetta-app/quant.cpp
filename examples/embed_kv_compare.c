/**
 * embed_kv_compare.c — KV compression comparison demo
 *
 * Runs the same prompt with different KV compression levels
 * and shows memory savings + quality comparison.
 *
 * Compile: cc embed_kv_compare.c -o kv_compare -lm -lpthread
 * Run:     ./kv_compare model.gguf
 */

#define QUANT_IMPLEMENTATION
#include "../quant.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    quant_model* model = quant_load(argv[1]);
    if (!model) { printf("Failed to load model\n"); return 1; }

    const char* prompt = "What is the capital of France?";

    printf("Prompt: %s\n", prompt);
    printf("==========================================\n\n");

    /* Test with different KV compression levels */
    int kv_modes[] = { 0, 1, 2 };
    const char* kv_names[] = { "FP32 (no compression)", "4-bit K + Q4 V", "Delta 3-bit + Q4 V" };

    for (int m = 0; m < 3; m++) {
        quant_config cfg = {
            .temperature = 0.0f,  /* greedy for reproducibility */
            .top_p       = 1.0f,
            .max_tokens  = 64,
            .n_threads   = 4,
            .kv_compress = kv_modes[m],
        };

        quant_ctx* ctx = quant_new(model, &cfg);
        if (!ctx) continue;

        printf("[%s]\n", kv_names[m]);

        char* result = quant_ask(ctx, prompt);
        if (result) {
            printf("  Output: %s\n", result);
            free(result);
        }
        printf("\n");

        quant_free_ctx(ctx);
    }

    quant_free_model(model);
    return 0;
}
