#!/usr/bin/env python3
"""Step-by-step encode inspection for a single vector from .npy file.

Usage:
    python tools/debug/inspect_encode.py path/to/vector.npy
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def main():
    if len(sys.argv) < 2:
        print("Usage: inspect_encode.py <vector.npy>", file=sys.stderr)
        sys.exit(1)

    vec = np.load(sys.argv[1])
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    dim = vec.shape[-1]

    cfg = NQXConfig(dim=dim, bits=3)
    core = NQXCore(cfg)

    x = vec.astype(np.float32)
    _print_stage("Input vector", x[0], 8)

    # L1
    x, _ = core.gu.apply_layer(x, 0)
    _print_stage("After L1 rotation", x[0], 8)

    # L2
    x, _ = core.gu.apply_layer(x, 1)
    _print_stage("After L2 rotation", x[0], 8)

    # L3
    x, _ = core.gu.apply_layer(x, 2)
    _print_stage("After L3 rotation", x[0], 8)

    # Polar
    polar, _ = core.pu.to_polar(x)
    _print_stage("Polar (r)", polar[0, 0::2], 4)
    _print_stage("Polar (θ)", polar[0, 1::2], 4)

    # Quantize
    _, q_idx, mins, maxs, _ = core.qu.quantize(polar, cfg.bits)
    _print_stage("Quantized indices", q_idx[0], 8)

    # QJL sign
    corrected, sign_bits, _ = core.qjl.apply(polar, polar)
    _print_stage("Sign bits", sign_bits[0], 16)

    # Pack
    packed, _ = core.pk.pack3plus1(q_idx, sign_bits)
    hex_lines = []
    for i in range(0, min(len(packed), 32), 16):
        hex_lines.append(packed[i:i+16].hex())
    print(f"Packed bytes (first 32 of {len(packed)}):")
    for line in hex_lines:
        print(f"  {line}")


def _print_stage(label, data, count):
    vals = data.flatten()[:count]
    fmt = " ".join(f"{v:8.4f}" for v in vals)
    print(f"{label:25s} [{fmt}]")


if __name__ == "__main__":
    main()
