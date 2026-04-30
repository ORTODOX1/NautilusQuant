"""S6: Realistic KV generator — mimics Llama 3 distribution without torch."""

from __future__ import annotations

import os

import numpy as np


def generate_llama3_kv(
    n_layers: int = 32,
    n_heads: int = 32,
    dim: int = 128,
    seq_len: int = 256,
    outlier_dims_per_head: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic KV-cache with Llama 3-like statistics.

    Returns shape (n_layers, 2, n_heads, seq_len, dim) — 2 for K and V.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Heavy-tailed base: t-distribution with df=3
    base = rng.standard_t(df=3, size=(n_layers, 2, n_heads, seq_len, dim)).astype(np.float32)
    base *= 0.3  # scale to realistic magnitude

    # Bimodal channels: half positive-shifted, half negative-shifted
    half = dim // 2
    base[..., :half] += 0.5  # positive bias
    base[..., half:] -= 0.5  # negative bias

    # Outlier dimensions: ~6 per head with 10x magnitude
    for l in range(n_layers):
        for h in range(n_heads):
            outlier_cols = rng.integers(0, dim, size=outlier_dims_per_head)
            base[l, :, h, :, outlier_cols] *= 10.0

    return base


def main():
    rng = np.random.default_rng(42)
    kv = generate_llama3_kv(rng=rng)
    path = os.path.join(os.path.dirname(__file__), "data", "llama3_like_seq256.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, kv)
    print(f"Saved: {path}  shape={kv.shape}  dtype={kv.dtype}")


if __name__ == "__main__":
    main()
