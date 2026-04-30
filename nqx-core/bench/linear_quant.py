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


def uniform_quant(x: np.ndarray, bits: int) -> np.ndarray:
    levels = 2**bits
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    norm = (x - mins) / ranges
    q = np.round(norm * (levels - 1)).clip(0, levels - 1)
    return ((q / (levels - 1)) * ranges + mins).astype(np.float32)


def lloyd_max_1d(samples: np.ndarray, bits: int, iters: int = 50) -> np.ndarray:
    levels = 2**bits
    lo, hi = float(samples.min()), float(samples.max())
    if hi - lo < 1e-9:
        return np.full(samples.shape, lo, dtype=np.float32)
    centroids = np.linspace(lo, hi, levels, dtype=np.float64)
    for _ in range(iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, samples)
        new = centroids.copy()
        for k in range(levels):
            mask = idx == k
            if mask.any():
                new[k] = samples[mask].mean()
        if np.allclose(new, centroids):
            centroids = new
            break
        centroids = new
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(boundaries, samples)
    return centroids[idx].astype(np.float32)


def lloyd_max_quant(x: np.ndarray, bits: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    for d in range(x.shape[-1]):
        out[..., d] = lloyd_max_1d(x[..., d], bits)
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def measure(rotation: str, dim: int, bits: int, n_vec: int, seed: int):
    rng = np.random.default_rng(seed + dim * 17 + bits)
    x = rng.standard_normal((n_vec, dim)).astype(np.float32)
    n_out = max(1, n_vec // 32)
    rows = rng.integers(0, n_vec, size=n_out)
    cols = rng.integers(0, dim, size=n_out)
    x[rows, cols] += rng.choice([-1.0, 1.0], size=n_out) * 8.0

    if rotation == "phi":
        cfg = NQXConfig(dim=dim)
        core = NQXCore(cfg)
        T = core.rotation_matrix().astype(np.float32)
        rotated = x @ T
    else:
        rotated = x.copy()

    uni = uniform_quant(rotated, bits)
    llm = lloyd_max_quant(rotated, bits)
    rmse_uni = rmse(rotated, uni)
    rmse_llm = rmse(rotated, llm)
    return {
        "rotation": rotation,
        "dim": dim,
        "bits": bits,
        "rmse_uniform": rmse_uni,
        "rmse_lloyd_max": rmse_llm,
        "delta_pct": 100.0 * (rmse_uni - rmse_llm) / max(rmse_llm, 1e-12),
    }


def run(dims, bits_list, n_vec=1024, seed=0):
    rows = []
    for dim in dims:
        for bits in bits_list:
            rows.append(measure("phi", dim, bits, n_vec, seed))
            rows.append(measure("none", dim, bits, n_vec, seed))
    return rows


def format_markdown(rows, dims, bits_list) -> str:
    lines = ["# Linear (uniform) vs Lloyd-Max quantisation after rotation", ""]
    lines.append(
        "If φ-rotation flattens the distribution to near-uniform, then the "
        "optimal Lloyd-Max quantiser collapses onto the simple linear "
        "quantiser. **A small δ = (RMSE_linear − RMSE_LM) / RMSE_LM after "
        "φ-rotation is direct evidence that Lloyd-Max codebooks are unnecessary "
        "in hardware** — we can replace the per-feature Lloyd-Max table with a "
        "single 1-cycle linear quantiser."
    )
    lines.append("")
    lines.append("## Per-feature RMSE on synthetic outlier-laden Gaussians")
    lines.append("")
    for rotation in ("phi", "none"):
        lines.append(f"### rotation = `{rotation}`")
        header = (
            "| dim \\ bits | " + " | ".join(f"{b}-bit (linear / LM / δ%)" for b in bits_list) + " |"
        )
        sep = "|---|" + "|".join(["---:"] * len(bits_list)) + "|"
        lines.append(header)
        lines.append(sep)
        for dim in dims:
            cells = []
            for bits in bits_list:
                row = next(
                    r
                    for r in rows
                    if r["rotation"] == rotation and r["dim"] == dim and r["bits"] == bits
                )
                cells.append(
                    f"{row['rmse_uniform']:.4f} / "
                    f"{row['rmse_lloyd_max']:.4f} / "
                    f"{row['delta_pct']:+.2f}%"
                )
            lines.append(f"| {dim} | " + " | ".join(cells) + " |")
        lines.append("")
    avg_phi_delta = np.mean([r["delta_pct"] for r in rows if r["rotation"] == "phi"])
    avg_none_delta = np.mean([r["delta_pct"] for r in rows if r["rotation"] == "none"])
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Average δ after φ-rotation: **{avg_phi_delta:+.2f}%**")
    lines.append(f"- Average δ without rotation: **{avg_none_delta:+.2f}%**")
    lines.append("")
    if avg_phi_delta < 5.0:
        verdict = (
            "**Hypothesis confirmed.** After φ-rotation, the linear quantiser "
            "is within 5% of Lloyd-Max — Lloyd-Max codebooks add no measurable "
            "value at this bit-width. Hardware can replace the QU.q stage's "
            "code-table lookup with a single linear `(x - min) * inv_range * "
            "(2^bits - 1)` mapping, removing one ROM port and ≈ 0.4 mm² die "
            "area (see `asic/floorplan.md` QU row)."
        )
    else:
        verdict = (
            f"**Hypothesis NOT confirmed at this distribution.** Linear quant "
            f"is {avg_phi_delta:+.1f}% worse than Lloyd-Max after φ-rotation; "
            f"without rotation the gap is {avg_none_delta:+.1f}%. The synthetic "
            "input is isotropic-Gaussian + outliers — Gaussian is rotationally "
            "invariant, so the per-feature marginal stays Gaussian and Lloyd-Max "
            "wins by allocating more codes near zero. φ-rotation still helps the "
            "outlier dimension via spreading (compare absolute RMSE_LM rows: phi "
            "vs none) but does not flatten the distribution to uniform. The QU.q "
            "stage in `nqx/cpu.py` is therefore retained as Lloyd-Max for the "
            "production pipeline. **Implication for the paper:** the rotation's "
            "value is in outlier dispersion + ROM elimination, *not* in making "
            "Lloyd-Max obsolete on Gaussian-like KV activations. Re-run on real "
            "KV-cache (heavy-tailed) to see if the gap shrinks below 5%."
        )
    lines.append(verdict)
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python bench/linear_quant.py --out bench/linear_quant.md")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--vectors", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench/linear_quant.md"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = run(args.dims, args.bits, n_vec=args.vectors, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(rows, args.dims, args.bits))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
