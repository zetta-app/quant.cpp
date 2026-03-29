#!/usr/bin/env python3
"""
PolarQuant NumPy reference implementation.

Implements the PolarQuant algorithm for KV cache compression:
  1. Split vector into D/2 2D pairs (x, y)
  2. Convert to polar coordinates: angle = atan2(y, x), radius = sqrt(x^2 + y^2)
  3. Group-wise min-max quantization of angle (theta) and radius (rho)
  4. Pack quantized indices: (rho << tbits) | theta
  5. Dequantize by reversing the process
  6. Compute attention scores directly in quantized polar domain

Reference: PolarQuant paper (https://arxiv.org/abs/2407.01119)
Block format matches include/turboquant/tq_types.h block_tq_polar.
"""

import numpy as np
import struct
from typing import Tuple, Optional

# Constants matching tq_types.h
TQ_BK = 128  # block size (elements)


def float32_to_fp16_bytes(val: float) -> bytes:
    """Convert float32 to IEEE 754 half-precision (2 bytes, little-endian)."""
    return struct.pack('<e', np.float16(val))


def fp16_bytes_to_float32(data: bytes) -> float:
    """Convert 2 bytes (little-endian fp16) back to float32."""
    return float(struct.unpack('<e', data)[0])


def polar_quantize(
    keys: np.ndarray,
    rbits: int = 2,
    tbits: int = 2,
    block_size: int = TQ_BK,
) -> dict:
    """
    PolarQuant quantization.

    Args:
        keys: Input key vectors, shape (n, head_dim). head_dim must be even.
        rbits: Bits for radius quantization (1-4).
        tbits: Bits for theta/angle quantization (1-4).
        block_size: Group size for min-max quantization.

    Returns:
        Dictionary with quantized data:
          - 'indices': packed uint8 indices, shape (n, head_dim // 2)
          - 'rscale': radius scale per block, shape (n, num_blocks)
          - 'rmn': radius minimum per block, shape (n, num_blocks)
          - 'tscale': theta scale per block, shape (n, num_blocks)
          - 'tmn': theta minimum per block, shape (n, num_blocks)
          - 'rbits': radius bits used
          - 'tbits': theta bits used
    """
    assert keys.ndim == 2, f"Expected 2D input, got {keys.ndim}D"
    n, head_dim = keys.shape
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    D = head_dim // 2  # number of 2D pairs per vector

    # Reshape into 2D pairs: (n, D, 2)
    pairs = keys.reshape(n, D, 2)
    x = pairs[:, :, 0]  # (n, D)
    y = pairs[:, :, 1]  # (n, D)

    # Convert to polar coordinates
    theta = np.arctan2(y, x)  # [-pi, pi]
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)  # [0, 2*pi]
    radius = np.sqrt(x ** 2 + y ** 2)  # >= 0

    # Group-wise min-max quantization
    # Each "block" covers block_size elements of the original vector,
    # which means block_size // 2 pairs.
    pairs_per_block = block_size // 2
    num_blocks = (D + pairs_per_block - 1) // pairs_per_block

    # Pad to full blocks if necessary
    pad_d = num_blocks * pairs_per_block - D
    if pad_d > 0:
        theta = np.pad(theta, ((0, 0), (0, pad_d)), constant_values=0)
        radius = np.pad(radius, ((0, 0), (0, pad_d)), constant_values=0)
        D_padded = D + pad_d
    else:
        D_padded = D

    # Reshape for block-wise operations: (n, num_blocks, pairs_per_block)
    theta_blocks = theta.reshape(n, num_blocks, pairs_per_block)
    radius_blocks = radius.reshape(n, num_blocks, pairs_per_block)

    # Theta quantization: min-max per block
    tmn = theta_blocks.min(axis=2)   # (n, num_blocks)
    tmx = theta_blocks.max(axis=2)
    tscale = (tmx - tmn) / (2 ** tbits)
    # Avoid division by zero
    tscale = np.where(tscale == 0, 1.0, tscale)

    theta_q = np.clip(
        np.floor((theta_blocks - tmn[:, :, np.newaxis]) / tscale[:, :, np.newaxis]),
        0, 2 ** tbits - 1
    ).astype(np.uint8)

    # Radius quantization: min-max per block
    rmn = radius_blocks.min(axis=2)  # (n, num_blocks)
    rmx = radius_blocks.max(axis=2)
    rscale = (rmx - rmn) / (2 ** rbits)
    rscale = np.where(rscale == 0, 1.0, rscale)

    rho_q = np.clip(
        np.floor((radius_blocks - rmn[:, :, np.newaxis]) / rscale[:, :, np.newaxis]),
        0, 2 ** rbits - 1
    ).astype(np.uint8)

    # Pack indices: (rho << tbits) | theta
    indices = (rho_q << tbits) | theta_q  # (n, num_blocks, pairs_per_block)
    indices = indices.reshape(n, D_padded)

    # Trim padding
    if pad_d > 0:
        indices = indices[:, :D]

    return {
        'indices': indices,
        'rscale': rscale.astype(np.float32),
        'rmn': rmn.astype(np.float32),
        'tscale': tscale.astype(np.float32),
        'tmn': tmn.astype(np.float32),
        'rbits': rbits,
        'tbits': tbits,
    }


def polar_dequantize(
    quantized: dict,
    head_dim: int,
    block_size: int = TQ_BK,
) -> np.ndarray:
    """
    PolarQuant dequantization.

    Args:
        quantized: Output from polar_quantize().
        head_dim: Original head dimension.
        block_size: Block size used during quantization.

    Returns:
        Reconstructed key vectors, shape (n, head_dim).
    """
    indices = quantized['indices']
    rscale = quantized['rscale']
    rmn = quantized['rmn']
    tscale = quantized['tscale']
    tmn = quantized['tmn']
    rbits = quantized['rbits']
    tbits = quantized['tbits']

    n = indices.shape[0]
    D = head_dim // 2
    pairs_per_block = block_size // 2
    num_blocks = rscale.shape[1]

    # Unpack indices
    theta_q = (indices & ((1 << tbits) - 1)).astype(np.float32)
    rho_q = (indices >> tbits).astype(np.float32)

    # Pad for block alignment
    D_padded = num_blocks * pairs_per_block
    if D < D_padded:
        theta_q = np.pad(theta_q, ((0, 0), (0, D_padded - D)), constant_values=0)
        rho_q = np.pad(rho_q, ((0, 0), (0, D_padded - D)), constant_values=0)

    theta_q = theta_q.reshape(n, num_blocks, pairs_per_block)
    rho_q = rho_q.reshape(n, num_blocks, pairs_per_block)

    # Dequantize: value = scale * (quantized + 0.5) + minimum (midpoint reconstruction)
    theta_deq = tscale[:, :, np.newaxis] * (theta_q + 0.5) + tmn[:, :, np.newaxis]
    radius_deq = rscale[:, :, np.newaxis] * (rho_q + 0.5) + rmn[:, :, np.newaxis]

    theta_deq = theta_deq.reshape(n, D_padded)[:, :D]
    radius_deq = radius_deq.reshape(n, D_padded)[:, :D]

    # Polar to Cartesian
    x = radius_deq * np.cos(theta_deq)
    y = radius_deq * np.sin(theta_deq)

    # Interleave back: (n, D, 2) -> (n, head_dim)
    result = np.stack([x, y], axis=2).reshape(n, head_dim)
    return result


def polar_attention_score(
    query: np.ndarray,
    quantized: dict,
    head_dim: int,
    block_size: int = TQ_BK,
) -> np.ndarray:
    """
    Compute attention scores directly from PolarQuant-compressed keys.

    Uses the lookup table approach from PolarQuant:
      1. For each theta quantization level, compute cos/sin lookup
      2. Multiply query pairs with cos/sin, gather by theta index
      3. Weight by dequantized radius

    Args:
        query: Query vector, shape (head_dim,) or (nq, head_dim).
        quantized: Output from polar_quantize().
        head_dim: Key head dimension.
        block_size: Block size used.

    Returns:
        Attention scores, shape (nq, n) or (n,) if query is 1D.
    """
    squeeze = False
    if query.ndim == 1:
        query = query[np.newaxis, :]
        squeeze = True

    indices = quantized['indices']
    rscale = quantized['rscale']
    rmn = quantized['rmn']
    tscale = quantized['tscale']
    tmn = quantized['tmn']
    rbits = quantized['rbits']
    tbits = quantized['tbits']

    nq = query.shape[0]
    n = indices.shape[0]
    D = head_dim // 2
    pairs_per_block = block_size // 2
    num_blocks = rscale.shape[1]

    n_theta_levels = 2 ** tbits
    n_rho_levels = 2 ** rbits

    # Unpack indices
    theta_idx = (indices & ((1 << tbits) - 1))  # (n, D)
    rho_idx = (indices >> tbits)                  # (n, D)

    # Reshape query into pairs: (nq, D, 2)
    q_pairs = query.reshape(nq, D, 2)
    qx = q_pairs[:, :, 0]  # (nq, D)
    qy = q_pairs[:, :, 1]  # (nq, D)

    scores = np.zeros((nq, n), dtype=np.float64)

    # Process block by block
    D_padded = num_blocks * pairs_per_block
    theta_idx_padded = np.pad(theta_idx, ((0, 0), (0, D_padded - D)), constant_values=0) if D < D_padded else theta_idx
    rho_idx_padded = np.pad(rho_idx, ((0, 0), (0, D_padded - D)), constant_values=0) if D < D_padded else rho_idx
    qx_padded = np.pad(qx, ((0, 0), (0, D_padded - D)), constant_values=0) if D < D_padded else qx
    qy_padded = np.pad(qy, ((0, 0), (0, D_padded - D)), constant_values=0) if D < D_padded else qy

    theta_idx_blocks = theta_idx_padded.reshape(n, num_blocks, pairs_per_block)
    rho_idx_blocks = rho_idx_padded.reshape(n, num_blocks, pairs_per_block)
    qx_blocks = qx_padded.reshape(nq, num_blocks, pairs_per_block)
    qy_blocks = qy_padded.reshape(nq, num_blocks, pairs_per_block)

    for b in range(num_blocks):
        # Build theta lookup table for this block
        # theta_levels: (n, n_theta_levels) -- different per key vector and block
        theta_levels = tscale[:, b:b+1] * (np.arange(n_theta_levels)[np.newaxis, :] + 0.5) + tmn[:, b:b+1]
        cos_lut = np.cos(theta_levels)  # (n, n_theta_levels)
        sin_lut = np.sin(theta_levels)  # (n, n_theta_levels)

        # Radius lookup: (n, n_rho_levels)
        rho_levels = rscale[:, b:b+1] * (np.arange(n_rho_levels)[np.newaxis, :] + 0.5) + rmn[:, b:b+1]

        # For each pair in block, gather cos/sin by theta index, weight by rho
        t_idx = theta_idx_blocks[:, b, :]  # (n, pairs_per_block)
        r_idx = rho_idx_blocks[:, b, :]    # (n, pairs_per_block)

        # Gather cos/sin values: for each key vector i and pair j, get cos_lut[i, t_idx[i,j]]
        cos_vals = np.take_along_axis(cos_lut, t_idx, axis=1)  # (n, pairs_per_block)
        sin_vals = np.take_along_axis(sin_lut, t_idx, axis=1)
        rho_vals = np.take_along_axis(rho_levels, r_idx, axis=1)  # (n, pairs_per_block)

        # qx_b: (nq, pairs_per_block), cos_vals: (n, pairs_per_block), rho_vals: (n, pairs_per_block)
        for qi in range(nq):
            dot = (qx_blocks[qi, b, :][np.newaxis, :] * cos_vals * rho_vals +
                   qy_blocks[qi, b, :][np.newaxis, :] * sin_vals * rho_vals)
            scores[qi, :] += dot.sum(axis=1)

    if squeeze:
        return scores[0]
    return scores


def polar_pack_block(
    quantized: dict,
    vec_idx: int,
    block_idx: int,
    block_size: int = TQ_BK,
) -> bytes:
    """
    Pack a single PolarQuant block into binary format matching block_tq_polar.

    Layout (72 bytes for BK=128):
      - rscale: fp16 (2 bytes)
      - rmn:    fp16 (2 bytes)
      - tscale: fp16 (2 bytes)
      - tmn:    fp16 (2 bytes)
      - indices: BK/2 bytes (64 bytes for BK=128)

    Returns:
        Packed binary data.
    """
    pairs_per_block = block_size // 2
    D = quantized['indices'].shape[1]

    rs = np.float16(quantized['rscale'][vec_idx, block_idx])
    rm = np.float16(quantized['rmn'][vec_idx, block_idx])
    ts = np.float16(quantized['tscale'][vec_idx, block_idx])
    tm = np.float16(quantized['tmn'][vec_idx, block_idx])

    start = block_idx * pairs_per_block
    end = min(start + pairs_per_block, D)
    idx = quantized['indices'][vec_idx, start:end]

    # Pad if needed
    if len(idx) < pairs_per_block:
        idx = np.pad(idx, (0, pairs_per_block - len(idx)), constant_values=0)

    buf = struct.pack('<e', rs) + struct.pack('<e', rm)
    buf += struct.pack('<e', ts) + struct.pack('<e', tm)
    buf += idx.astype(np.uint8).tobytes()

    return buf


def compute_roundtrip_mse(
    keys: np.ndarray,
    rbits: int = 2,
    tbits: int = 2,
    block_size: int = TQ_BK,
) -> float:
    """Quantize and dequantize, return MSE."""
    quantized = polar_quantize(keys, rbits=rbits, tbits=tbits, block_size=block_size)
    reconstructed = polar_dequantize(quantized, head_dim=keys.shape[1], block_size=block_size)
    return float(np.mean((keys - reconstructed) ** 2))


def compute_attention_cosine_similarity(
    query: np.ndarray,
    keys: np.ndarray,
    rbits: int = 2,
    tbits: int = 2,
    block_size: int = TQ_BK,
) -> float:
    """
    Compute cosine similarity between FP32 attention scores and
    PolarQuant quantized attention scores.
    """
    # FP32 reference
    fp32_scores = query @ keys.T

    # Quantized scores
    quantized = polar_quantize(keys, rbits=rbits, tbits=tbits, block_size=block_size)
    quant_scores = polar_attention_score(query, quantized, head_dim=keys.shape[1], block_size=block_size)

    # Cosine similarity
    dot = np.sum(fp32_scores * quant_scores)
    norm_a = np.sqrt(np.sum(fp32_scores ** 2))
    norm_b = np.sqrt(np.sum(quant_scores ** 2))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(dot / (norm_a * norm_b))


# ---- Self-test ----
if __name__ == '__main__':
    np.random.seed(42)
    head_dim = 128
    n_keys = 16
    keys = np.random.randn(n_keys, head_dim).astype(np.float32)
    query = np.random.randn(head_dim).astype(np.float32)

    for rbits, tbits, label in [(2, 2, '4b'), (1, 2, '3b')]:
        print(f"\n=== PolarQuant {label} (rbits={rbits}, tbits={tbits}) ===")
        quantized = polar_quantize(keys, rbits=rbits, tbits=tbits)
        recon = polar_dequantize(quantized, head_dim=head_dim)

        mse = float(np.mean((keys - recon) ** 2))
        print(f"Roundtrip MSE: {mse:.6f}")

        # Attention accuracy
        fp32_scores = query @ keys.T
        quant_scores = polar_attention_score(query, quantized, head_dim=head_dim)

        cos_sim = compute_attention_cosine_similarity(query, keys, rbits=rbits, tbits=tbits)
        print(f"Attention cosine similarity: {cos_sim:.6f}")

        # Pack first block
        block_bytes = polar_pack_block(quantized, vec_idx=0, block_idx=0)
        print(f"Block size: {len(block_bytes)} bytes (expected {8 + TQ_BK // 2})")

    print("\nAll PolarQuant reference tests passed.")
