#ifndef TQ_SERVER_H
#define TQ_SERVER_H

/**
 * tq_server.h -- Minimal OpenAI-compatible HTTP server for quant.cpp
 *
 * Exposes:
 *   POST /v1/chat/completions  (streaming + non-streaming)
 *   GET  /v1/models
 *   GET  /health
 *
 * Pure C, zero external dependencies (libc + pthreads only).
 */

#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Server configuration
 * ============================================================ */

typedef struct {
    int            port;             /* Listen port (default: 8080)           */
    int            max_connections;  /* Max concurrent connections (default: 8) */
    const char*    host;             /* Bind address (default: "0.0.0.0")     */

    /* Model */
    tq_model_t*    model;            /* Loaded model (required)               */
    tq_tokenizer_t* tokenizer;       /* Loaded tokenizer (required)           */
    const char*    model_id;         /* Model name for /v1/models (e.g. "quant.cpp/Qwen2.5-0.5B") */

    /* Inference defaults */
    tq_type        kv_type;          /* Default KV cache type                 */
    int            value_quant_bits; /* Default V quant bits (0/2/4)          */
    int            n_threads;        /* Threads per inference (default: 4)    */
    int            delta_kv;         /* Enable delta KV compression           */
} tq_server_config_t;

/**
 * Return a config with sensible defaults.
 * Caller must set model, tokenizer, and model_id before starting.
 */
tq_server_config_t tq_server_default_config(void);

/* ============================================================
 * Server lifecycle
 * ============================================================ */

typedef struct tq_server tq_server_t;

/**
 * Create and start the HTTP server. Blocks until tq_server_stop() is called
 * from another thread, or a signal is received.
 * Returns 0 on clean shutdown, -1 on error.
 */
int tq_server_start(tq_server_t** out, const tq_server_config_t* config);

/**
 * Signal the server to stop accepting connections and shut down.
 * Safe to call from a signal handler.
 */
void tq_server_stop(tq_server_t* server);

/**
 * Free server resources. Call after tq_server_start() returns.
 */
void tq_server_free(tq_server_t* server);

#ifdef __cplusplus
}
#endif

#endif /* TQ_SERVER_H */
