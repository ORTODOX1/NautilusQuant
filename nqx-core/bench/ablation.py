#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def hadamard_matrix(n: int) -> np.ndarray:
    assert n & (n - 1) == 0, "Hadamard requires n to be a power of two"
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.standard_normal((n, n)).astype(np.float64)
    q, r = np.linalg.qr(a)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    return (q * sign).astype(np.float32)


def synth_outliers(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    x = rng.standard_normal((n, dim)).astype(np.float32)
    n_out = max(1, n // 32)
    cols = rng.integers(0, dim, size=n_out)
    rows = rng.integers(0, n, size=n_out)
    x[rows, cols] += rng.choice([-1.0, 1.0], size=n_out) * 8.0
    return x


def quantize_min_max(x: np.ndarray, bits: int) -> np.ndarray:
    levels = 2**bits
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    norm = (x - mins) / ranges
    q = np.round(norm * (levels - 1)).clip(0, levels - 1)
    return ((q / (levels - 1)) * ranges + mins).astype(np.float32)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


ROTATIONS = ("phi", "random", "hadamard", "none")


def get_rotation(name: str, dim: int, rng: np.random.Generator) -> np.ndarray | None:
    if name == "phi":
        cfg = NQXConfig(dim=dim)
        core = NQXCore(cfg)
        return core.rotation_matrix().astype(np.float32)
    if name == "random":
        return random_orthogonal(rng, dim)
    if name == "hadamard":
        return hadamard_matrix(dim)
    if name == "none":
        return None
    raise ValueError(f"unknown rotation {name}")


def run(dims, bits_list, n_vec=2048, seed=0):
    rows = []
    for rotation_name in ROTATIONS:
        for dim in dims:
            rng = np.random.default_rng(seed + dim)
            x = synth_outliers(rng, n_vec, dim)
            T = get_rotation(rotation_name, dim, rng)
            if T is not None:
                rotated = x @ T
            else:
                rotated = x
            for bits in bits_list:
                q = quantize_min_max(rotated, bits)
                if T is not None:
                    back = q @ T.T
                else:
                    back = q
                rows.append(
                    {
                        "rotation": rotation_name,
                        "dim": dim,
                        "bits": bits,
                        "rmse": rmse(x, back),
                    }
                )
    return rows


def format_markdown(rows) -> str:
    dims = sorted({r["dim"] for r in rows})
    bits_list = sorted({r["bits"] for r in rows})
    lines = ["# Rotation × dim × bits — RMSE ablation", ""]
    lines.append(
        "Synthetic isotropic Gaussians with 1/32 outliers of magnitude 8σ. "
        "Per-axis min/max quantisation after rotation; rotation is identity for "
        "`none`. RMSE measured against the original FP32 input after the inverse "
        "rotation. Lower is better."
    )
    lines.append("")
    for dim in dims:
        lines.append(f"## dim = {dim}")
        header = "| rotation \\ bits | " + " | ".join(str(b) for b in bits_list) + " |"
        sep = "|---|" + "|".join(["---:"] * len(bits_list)) + "|"
        lines.append(header)
        lines.append(sep)
        for rotation in ROTATIONS:
            cells = []
            for bits in bits_list:
                row = next(
                    r
                    for r in rows
                    if r["rotation"] == rotation and r["dim"] == dim and r["bits"] == bits
                )
                cells.append(f"{row['rmse']:.4f}")
            lines.append(f"| `{rotation}` | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python bench/ablation.py --out bench/ablation.md")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--vectors", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench/ablation.md"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = run(args.dims, args.bits, n_vec=args.vectors, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(rows))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
