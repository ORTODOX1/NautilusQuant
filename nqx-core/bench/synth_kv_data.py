#!/usr/bin/env python3
"""Generate synthetic KV-cache data with realistic outliers.

Usage:
    python bench/synth_kv_data.py --layers 32 --heads 8 --dim 128 --seq 4096 -o kv_data.npy
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def generate(
    n_layers: int,
    n_heads: int,
    dim: int,
    seq_len: int,
    seed: int = 42,
    outlier_frac: float = 0.05,
    outlier_scale: float = 30.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_layers, 2, n_heads, seq_len, dim)).astype(np.float16) * 0.5

    n_outlier_cols = max(1, int(dim * outlier_frac))
    outlier_cols = rng.choice(dim, n_outlier_cols, replace=False)
    for l in range(n_layers):
        for h in range(n_heads):
            n_mask = max(1, int(seq_len * 0.3))
            idx = rng.choice(seq_len, n_mask, replace=False)
            vals = rng.standard_normal((n_mask, n_outlier_cols)).astype(np.float16) * outlier_scale
            for ci, c in enumerate(outlier_cols):
                base[l, 0, h, idx, c] = vals[:, ci]
    return base


def main():
    p = argparse.ArgumentParser(description="Generate synthetic KV-cache data")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seq", "--seq-len", type=int, default=4096, dest="seq_len")
    p.add_argument("-o", "--output", default="kv_synthetic.npy")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = generate(args.layers, args.heads, args.dim, args.seq_len, seed=args.seed)
    shape = (args.layers, 2, args.heads, args.seq_len, args.dim)
    print(f"Generated KV-cache: {shape}, dtype={data.dtype}, size={data.nbytes / 1e9:.2f} GB")
    np.save(args.output, data)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
