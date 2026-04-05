/**
 * embed_chat.c — Interactive chat with quant.h
 *
 * A complete chat application in ~60 lines.
 * Compile: cc embed_chat.c -o chat -lm -lpthread
 * Run:     ./chat model.gguf
 */

#define QUANT_IMPLEMENTATION
#include "../quant.h"
#include <stdio.h>
#include <string.h>

static void print_token(const char* text, void* ud) {
    (void)ud;
    printf("%s", text);
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    quant_model* model = quant_load(argv[1]);
    if (!model) { printf("Failed to load model\n"); return 1; }

    quant_config cfg = {
        .temperature = 0.7f,
        .top_p       = 0.9f,
        .max_tokens  = 512,
        .n_threads   = 4,
        .kv_compress = 1,
    };

    printf("Model loaded. Type your message (Ctrl+C to exit).\n\n");

    char input[4096];
    while (1) {
        printf("> ");
        if (!fgets(input, sizeof(input), stdin)) break;

        /* Remove trailing newline */
        size_t len = strlen(input);
        if (len > 0 && input[len-1] == '\n') input[len-1] = '\0';
        if (strlen(input) == 0) continue;

        /* Create fresh context for each turn */
        quant_ctx* ctx = quant_new(model, &cfg);
        if (!ctx) continue;

        printf("\n");
        quant_generate(ctx, input, print_token, NULL);
        printf("\n\n");

        quant_free_ctx(ctx);
    }

    quant_free_model(model);
    return 0;
}
