"""
TurboQuant -- Cross-platform KV cache compression for LLM inference.

Python bindings using ctypes to interface with the C library.

Usage:
    import turboquant

    ctx = turboquant.TurboQuantContext()
    quantized = ctx.quantize_keys(keys_np, type=turboquant.TURBO_3B)
    scores = ctx.attention(query_np, quantized, type=turboquant.TURBO_3B)
    ctx.close()
"""

__version__ = "0.1.0"
__author__ = "TurboQuant Contributors"

from turboquant._core import (
    TurboQuantContext,
    TurboQuantError,
    quantize_keys,
    quantize_values,
    dequantize_keys,
    attention,
    type_name,
    type_bpe,
    get_library_path,
)

# Quantization type constants
POLAR_3B   = 0
POLAR_4B   = 1
QJL_1B     = 2
TURBO_3B   = 3
TURBO_4B   = 4
UNIFORM_4B = 5
UNIFORM_2B = 6

# Backend constants
BACKEND_CPU   = 0
BACKEND_CUDA  = 1
BACKEND_METAL = 2
BACKEND_AUTO  = 99

__all__ = [
    "TurboQuantContext",
    "TurboQuantError",
    "quantize_keys",
    "quantize_values",
    "dequantize_keys",
    "attention",
    "type_name",
    "type_bpe",
    "get_library_path",
    "POLAR_3B",
    "POLAR_4B",
    "QJL_1B",
    "TURBO_3B",
    "TURBO_4B",
    "UNIFORM_4B",
    "UNIFORM_2B",
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_METAL",
    "BACKEND_AUTO",
]
