"""KV Cache pre-build for RLV (#83).

Pre-computes KV caches for each document chunk during indexing.
At query time, load_context restores the KV state instantly,
eliminating the expensive prefill step.

Uses quant.h directly via ctypes (same as phi35_server.py).
"""
import ctypes
import os
import time
from pathlib import Path

# Reuse the shared library from phi35_server
LIB_PATH = "/tmp/libquant_phi3.dylib"
REPO = Path(__file__).resolve().parent.parent.parent.parent


def _get_lib():
    """Get or build the quant.h shared library."""
    if not os.path.exists(LIB_PATH):
        src = REPO / "quant.h"
        impl = "/tmp/_quant_kv_impl.c"
        with open(impl, "w") as f:
            f.write(f'#define QUANT_IMPLEMENTATION\n#include "{src}"\n')
        rc = os.system(f'cc -O3 -shared -fPIC -o {LIB_PATH} {impl} -lm -lpthread')
        if rc != 0:
            raise RuntimeError("Failed to build quant.h shared library")

    lib = ctypes.CDLL(LIB_PATH)

    lib.quant_load.argtypes = [ctypes.c_char_p]
    lib.quant_load.restype = ctypes.c_void_p

    class QuantConfig(ctypes.Structure):
        _fields_ = [
            ("temperature", ctypes.c_float),
            ("top_p", ctypes.c_float),
            ("max_tokens", ctypes.c_int),
            ("n_threads", ctypes.c_int),
            ("kv_compress", ctypes.c_int),
            ("context_length", ctypes.c_int),
            ("k_highres_window", ctypes.c_int),
        ]

    lib.quant_new.argtypes = [ctypes.c_void_p, ctypes.POINTER(QuantConfig)]
    lib.quant_new.restype = ctypes.c_void_p

    ON_TOKEN = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
    lib.quant_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ON_TOKEN, ctypes.c_void_p]
    lib.quant_generate.restype = ctypes.c_int

    lib.quant_save_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.quant_save_context.restype = ctypes.c_int

    lib.quant_load_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.quant_load_context.restype = ctypes.c_int

    lib.quant_free_ctx.argtypes = [ctypes.c_void_p]
    lib.quant_free_model.argtypes = [ctypes.c_void_p]

    return lib, QuantConfig, ON_TOKEN


_lib = None
_QuantConfig = None
_ON_TOKEN = None
_model = None
_model_path = None


def _ensure_model(model_path: str, n_threads: int = 8):
    """Load model once, reuse across calls."""
    global _lib, _QuantConfig, _ON_TOKEN, _model, _model_path
    if _lib is None:
        _lib, _QuantConfig, _ON_TOKEN = _get_lib()
    if _model is None or _model_path != model_path:
        if _model:
            _lib.quant_free_model(_model)
        _model = _lib.quant_load(model_path.encode())
        _model_path = model_path
        if not _model:
            raise RuntimeError(f"Failed to load model: {model_path}")


def build_kv_cache(
    chunks: list,
    model_path: str,
    cache_dir: str,
    *,
    n_threads: int = 8,
    verbose: bool = True,
) -> dict:
    """Pre-build KV caches for all chunks.

    Returns dict: {chunk_id: cache_file_path}
    """
    _ensure_model(model_path, n_threads)

    os.makedirs(cache_dir, exist_ok=True)
    cache_map = {}
    null_cb = _ON_TOKEN(lambda *a: None)

    t_start = time.time()
    for i, chunk in enumerate(chunks):
        cache_file = os.path.join(cache_dir, f"chunk_{chunk.chunk_id}.kv")

        if os.path.exists(cache_file):
            cache_map[chunk.chunk_id] = cache_file
            continue

        text = chunk.full_text or chunk.head_text
        if not text.strip():
            continue

        cfg = _QuantConfig()
        cfg.temperature = 0.0
        cfg.top_p = 1.0
        cfg.max_tokens = 1  # generate 1 token after prefill
        cfg.n_threads = n_threads

        ctx = _lib.quant_new(_model, ctypes.byref(cfg))
        if not ctx:
            continue

        # Use quant_chat (NOT quant_generate) for prefill.
        # quant_generate creates internal state and discards it.
        # quant_chat preserves KV state in ctx → save_context works.
        _lib.quant_chat.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _ON_TOKEN, ctypes.c_void_p]
        _lib.quant_chat.restype = ctypes.c_int
        _lib.quant_chat(ctx, text.encode("utf-8"), null_cb, None)

        # Save KV state — now contains ALL prefill tokens
        rc = _lib.quant_save_context(ctx, cache_file.encode())
        _lib.quant_free_ctx(ctx)

        if rc == 0:
            cache_map[chunk.chunk_id] = cache_file
            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  [kv-cache] {i+1}/{len(chunks)} chunks indexed ({elapsed:.0f}s)")
        else:
            if verbose:
                print(f"  [kv-cache] WARN: failed to save chunk {chunk.chunk_id}")

    if verbose:
        elapsed = time.time() - t_start
        size_mb = sum(os.path.getsize(f) for f in cache_map.values()) / 1024 / 1024
        print(f"  [kv-cache] done: {len(cache_map)} caches, {size_mb:.1f}MB, {elapsed:.0f}s")

    return cache_map


def lookup_with_cache(
    question: str,
    chunk_id: int,
    cache_map: dict,
    model_path: str,
    *,
    max_tokens: int = 24,
    n_threads: int = 8,
) -> str:
    """Answer a question using pre-built KV cache (no prefill needed).

    Uses quant_chat which appends to existing KV cache instead of
    resetting it like quant_generate does.
    """
    _ensure_model(model_path, n_threads)

    cache_file = cache_map.get(chunk_id)
    if not cache_file or not os.path.exists(cache_file):
        return None

    cfg = _QuantConfig()
    cfg.temperature = 0.0
    cfg.top_p = 1.0
    cfg.max_tokens = max_tokens
    cfg.n_threads = n_threads

    ctx = _lib.quant_new(_model, ctypes.byref(cfg))
    if not ctx:
        return None

    rc = _lib.quant_load_context(ctx, cache_file.encode())
    if rc != 0:
        _lib.quant_free_ctx(ctx)
        return None

    # Append question to existing KV context via quant_chat.
    # The chunk text is already in the KV cache from prefill.
    # We only send the question — no need to repeat the document.
    prompt = f"\n\nBased on the text above, answer this question in one sentence.\nQuestion: {question}\nAnswer:"
    tokens = []

    def on_token(text_ptr, ud):
        if text_ptr:
            tokens.append(text_ptr.decode("utf-8", errors="replace"))

    cb = _ON_TOKEN(on_token)

    # Check if quant_chat is available
    if hasattr(_lib, 'quant_chat'):
        _lib.quant_chat.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _ON_TOKEN, ctypes.c_void_p]
        _lib.quant_chat.restype = ctypes.c_int
        _lib.quant_chat(ctx, prompt.encode("utf-8"), cb, None)
    else:
        # Fallback: use generate (will reset KV but still works, just slower)
        _lib.quant_generate(ctx, prompt.encode("utf-8"), cb, None)

    _lib.quant_free_ctx(ctx)
    return "".join(tokens)
