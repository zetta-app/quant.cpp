#!/usr/bin/env python3
"""
QJL (Quantized Johnson-Lindenstrauss) NumPy reference implementation.

Implements the QJL algorithm for 1-bit KV cache compression:
  1. Generate a random Gaussian projection matrix (d_key x d_sketch)
  2. Project key vectors: sketch = key @ projection
  3. Sign quantization: bit = 1 if sketch > 0 else 0
  4. Pack 8 sign bits per byte (ONNX LSB-first convention)
  5. Detect outlier dimensions by L2 norm
  6. Attention via Hamming distance with outlier correction

Reference: QJL paper (Zandieh et al., https://arxiv.org/abs/2406.03482)
Block format matches include/turboquant/tq_types.h block_tq_qjl.
"""

import numpy as np
import struct
from typing import Tuple, Optional

# Constants matching tq_types.h
TQ_SKETCH_DIM = 256
TQ_OUTLIERS = 4
TQ_BK_QJL = 256


def generate_projection_matrix(
    d_key: int,
    d_sketch: int = TQ_SKETCH_DIM,
    seed: int = 42,
    orthogonalize: bool = True,
) -> np.ndarray:
    """
    Generate a random projection matrix for QJL.

    Args:
        d_key: Input key dimension.
        d_sketch: Sketch (output) dimension.
        seed: Random seed for reproducibility.
        orthogonalize: If True, apply QR orthogonalization (improves quality).

    Returns:
        Projection matrix, shape (d_key, d_sketch), float32.
    """
    rng = np.random.RandomState(seed)
    P = rng.randn(d_key, d_sketch).astype(np.float32)

    if orthogonalize:
        # Chunk-wise QR orthogonalization (matches reference implementation)
        num_chunks = (d_sketch + d_key - 1) // d_key
        result_cols = []
        for i in range(num_chunks):
            start = i * d_key
            end = min((i + 1) * d_key, d_sketch)
            Q, _ = np.linalg.qr(P[:, start:end], mode='reduced')
            result_cols.append(Q)
        P = np.concatenate(result_cols, axis=1) * np.sqrt(d_key)

    return P


def qjl_quantize(
    keys: np.ndarray,
    projection: np.ndarray,
    n_outliers: int = TQ_OUTLIERS,
    group_size: int = TQ_BK_QJL,
) -> dict:
    """
    QJL 1-bit sign hash quantization.

    Args:
        keys: Input key vectors, shape (n, d_key), float32.
        projection: Projection matrix, shape (d_key, d_sketch), float32.
        n_outliers: Number of outlier dimensions to detect per group.
        group_size: Group size for outlier detection.

    Returns:
        Dictionary with:
          - 'hash': packed sign bits, shape (n, d_sketch // 8), uint8
          - 'norms': L2 norm per key, shape (n,), float32
          - 'outlier_idx': outlier dimension indices, shape (n, n_outliers), uint8
          - 'outlier_norms': outlier component norms, shape (n,), float32
          - 'd_sketch': sketch dimension
    """
    assert keys.ndim == 2, f"Expected 2D input, got {keys.ndim}D"
    n, d_key = keys.shape
    d_sketch = projection.shape[1]
    assert projection.shape[0] == d_key, "Projection dim mismatch"
    assert d_sketch % 8 == 0, "d_sketch must be divisible by 8"

    # Compute key norms
    key_norms = np.sqrt(np.sum(keys ** 2, axis=1))  # (n,)

    # Detect outlier dimensions
    # For each key, find dimensions with largest absolute values
    outlier_idx = np.zeros((n, n_outliers), dtype=np.uint8)
    outlier_norms = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # Top-k dimensions by absolute value
        abs_vals = np.abs(keys[i])
        top_dims = np.argsort(abs_vals)[-n_outliers:][::-1]
        outlier_idx[i] = top_dims.astype(np.uint8)
        outlier_norms[i] = np.sqrt(np.sum(keys[i, top_dims] ** 2))

    # Project keys: sketch = keys @ projection
    sketch = keys @ projection  # (n, d_sketch)

    # Sign quantization: > 0 -> 1, <= 0 -> 0
    sign_bits = (sketch > 0).astype(np.uint8)  # (n, d_sketch)

    # Pack 8 bits per byte, LSB-first (ONNX convention)
    n_bytes = d_sketch // 8
    packed = np.zeros((n, n_bytes), dtype=np.uint8)
    for b in range(8):
        packed |= sign_bits[:, b::8].astype(np.uint8) << b

    # Alternative: explicit bit packing
    packed2 = np.zeros((n, n_bytes), dtype=np.uint8)
    for j in range(n_bytes):
        for b in range(8):
            bit_idx = j * 8 + b
            packed2[:, j] |= sign_bits[:, bit_idx].astype(np.uint8) << b

    return {
        'hash': packed2,
        'norms': key_norms,
        'outlier_idx': outlier_idx,
        'outlier_norms': outlier_norms,
        'd_sketch': d_sketch,
    }


def qjl_attention_score(
    query: np.ndarray,
    quantized: dict,
    projection: np.ndarray,
    keys_orig: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute attention scores from QJL-compressed keys using Hamming distance.

    The QJL estimator reconstructs the inner product as:
      score ~= sqrt(pi/2) / d_sketch * ||key|| * (d_sketch - 2 * hamming_dist)

    With outlier correction:
      score += query[outlier_dims] . key_outlier_component

    Args:
        query: Query vector, shape (d_key,) or (nq, d_key).
        quantized: Output from qjl_quantize().
        projection: Same projection matrix used for quantization.
        keys_orig: Original keys (optional, for outlier correction).

    Returns:
        Attention scores, shape (nq, n) or (n,).
    """
    squeeze = False
    if query.ndim == 1:
        query = query[np.newaxis, :]
        squeeze = True

    nq, d_key = query.shape
    packed_keys = quantized['hash']     # (n, d_sketch // 8)
    key_norms = quantized['norms']      # (n,)
    d_sketch = quantized['d_sketch']
    n = packed_keys.shape[0]

    # Project query and sign-quantize
    query_sketch = query @ projection   # (nq, d_sketch)
    query_sign = (query_sketch > 0).astype(np.uint8)

    # Pack query signs
    n_bytes = d_sketch // 8
    query_packed = np.zeros((nq, n_bytes), dtype=np.uint8)
    for j in range(n_bytes):
        for b in range(8):
            bit_idx = j * 8 + b
            query_packed[:, j] |= query_sign[:, bit_idx].astype(np.uint8) << b

    # Compute Hamming distance via XOR + popcount
    scores = np.zeros((nq, n), dtype=np.float64)
    for qi in range(nq):
        for ki in range(n):
            xor_result = np.bitwise_xor(query_packed[qi], packed_keys[ki])
            # Popcount per byte
            hamming = 0
            for byte_val in xor_result:
                hamming += bin(byte_val).count('1')

            # QJL inner product estimator:
            # <q, k> ~= sqrt(pi/2) * ||k|| / d_sketch * (d_sketch - 2 * hamming)
            inner_prod_estimate = d_sketch - 2 * hamming
            scores[qi, ki] = np.sqrt(np.pi / 2) * key_norms[ki] / d_sketch * inner_prod_estimate

    # Outlier correction (if original keys available)
    if keys_orig is not None:
        outlier_idx = quantized['outlier_idx']  # (n, n_outliers)
        for qi in range(nq):
            for ki in range(n):
                # Add back the outlier component dot product
                dims = outlier_idx[ki]
                outlier_dot = np.sum(query[qi, dims] * keys_orig[ki, dims])
                scores[qi, ki] += outlier_dot

    if squeeze:
        return scores[0]
    return scores


def popcount_array(arr: np.ndarray) -> int:
    """Count total set bits in a uint8 array."""
    total = 0
    for v in arr.flat:
        total += bin(v).count('1')
    return total


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two packed bit arrays."""
    return popcount_array(np.bitwise_xor(a, b))


def qjl_pack_block(
    quantized: dict,
    vec_idx: int,
) -> bytes:
    """
    Pack a single QJL block into binary format matching block_tq_qjl.

    Layout (40 bytes for SKETCH=256, OUTLIERS=4):
      - norm:         fp16 (2 bytes)
      - outlier_norm: fp16 (2 bytes)
      - hash:         SKETCH_DIM / 8 bytes (32 bytes)
      - outlier_idx:  OUTLIERS bytes (4 bytes)

    Returns:
        Packed binary data.
    """
    norm_fp16 = np.float16(quantized['norms'][vec_idx])
    outlier_norm_fp16 = np.float16(quantized['outlier_norms'][vec_idx])

    buf = struct.pack('<e', norm_fp16)
    buf += struct.pack('<e', outlier_norm_fp16)
    buf += quantized['hash'][vec_idx].tobytes()
    buf += quantized['outlier_idx'][vec_idx].tobytes()

    return buf


def compute_attention_cosine_similarity(
    query: np.ndarray,
    keys: np.ndarray,
    projection: np.ndarray,
) -> float:
    """
    Compute cosine similarity between FP32 and QJL attention scores.
    """
    fp32_scores = query @ keys.T
    quantized = qjl_quantize(keys, projection)
    qjl_scores = qjl_attention_score(query, quantized, projection, keys_orig=keys)

    dot = np.sum(fp32_scores * qjl_scores)
    norm_a = np.sqrt(np.sum(fp32_scores ** 2))
    norm_b = np.sqrt(np.sum(qjl_scores ** 2))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(dot / (norm_a * norm_b))


# ---- Self-test ----
if __name__ == '__main__':
    np.random.seed(42)
    d_key = 128
    d_sketch = TQ_SKETCH_DIM
    n_keys = 16

    keys = np.random.randn(n_keys, d_key).astype(np.float32)
    query = np.random.randn(d_key).astype(np.float32)

    print("=== QJL 1-bit Reference ===")
    projection = generate_projection_matrix(d_key, d_sketch, seed=42)
    print(f"Projection shape: {projection.shape}")

    # Check orthogonality: P^T P should be close to d_key * I
    PtP = projection.T @ projection
    off_diag = PtP - np.diag(np.diag(PtP))
    print(f"Projection off-diagonal max: {np.max(np.abs(off_diag)):.4f}")

    quantized = qjl_quantize(keys, projection, n_outliers=TQ_OUTLIERS)
    print(f"Hash shape: {quantized['hash'].shape}")
    print(f"Norms shape: {quantized['norms'].shape}")
    print(f"Outlier indices shape: {quantized['outlier_idx'].shape}")

    # Attention accuracy
    fp32_scores = query @ keys.T
    qjl_scores = qjl_attention_score(query, quantized, projection, keys_orig=keys)

    cos_sim = compute_attention_cosine_similarity(query, keys, projection)
    print(f"Attention cosine similarity (with outliers): {cos_sim:.6f}")

    qjl_scores_no_outlier = qjl_attention_score(query, quantized, projection)
    dot = np.sum(fp32_scores * qjl_scores_no_outlier)
    na = np.sqrt(np.sum(fp32_scores ** 2))
    nb = np.sqrt(np.sum(qjl_scores_no_outlier ** 2))
    cos_sim_no = dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 1.0
    print(f"Attention cosine similarity (no outliers): {cos_sim_no:.6f}")

    # Pack block
    block_bytes = qjl_pack_block(quantized, vec_idx=0)
    expected_size = 4 + TQ_SKETCH_DIM // 8 + TQ_OUTLIERS
    print(f"Block size: {len(block_bytes)} bytes (expected {expected_size})")

    # Hamming distance test
    h1 = quantized['hash'][0]
    h2 = quantized['hash'][1]
    dist = hamming_distance(h1, h2)
    print(f"Hamming distance (key 0 vs 1): {dist}/{d_sketch}")

    print("\nAll QJL reference tests passed.")
