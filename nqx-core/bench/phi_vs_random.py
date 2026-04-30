#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


PRNG_CYCLES_PER_VECTOR = 8


def synth_kv(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    x = rng.standard_normal((n, dim)).astype(np.float32) * 0.7
    n_out = max(1, n // 64)
    rows = rng.integers(0, n, size=n_out)
    cols = rng.integers(0, dim, size=n_out)
    x[rows, cols] += rng.choice([-1.0, 1.0], size=n_out) * 6.0
    return x


def random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.standard_normal((n, n)).astype(np.float64)
    q, r = np.linalg.qr(a)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    return (q * sign).astype(np.float32)


def quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    norm = (x - mins) / ranges
    q = np.round(norm * (levels - 1)).clip(0, levels - 1)
    return ((q / (levels - 1)) * ranges + mins).astype(np.float32)


def measure_phi(x: np.ndarray, bits: int):
    cfg = NQXConfig(dim=x.shape[-1])
    core = NQXCore(cfg)
    t0 = time.perf_counter()
    rotated = core.forward_rotation(x)
    quant = quant_dequant(rotated, bits)
    back = core.inverse_rotation(quant)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    n_vec = x.shape[0]
    cycles_compute = 3 * cfg.cycles_givens_layer + 1 + cfg.cycles_quant_minmax + cfg.cycles_quant_round
    cycles_total = cycles_compute + n_vec - 1
    return {
        "rmse": float(np.sqrt(((x - back) ** 2).mean())),
        "wall_ms": elapsed_ms,
        "cycles": cycles_total,
        "cycles_prng": 0,
    }


def measure_random(x: np.ndarray, bits: int, seed: int):
    rng = np.random.default_rng(seed)
    dim = x.shape[-1]
    t0 = time.perf_counter()
    T = random_orthogonal(rng, dim)
    rotated = x @ T
    quant = quant_dequant(rotated, bits)
    back = quant @ T.T
    elapsed_ms = (time.perf_counter() - t0) * 1000
    n_vec = x.shape[0]
    cycles_per_vec_dense = dim
    cycles_compute = cycles_per_vec_dense + 1 + 7 + 1
    cycles_total = cycles_compute + n_vec - 1 + PRNG_CYCLES_PER_VECTOR * n_vec
    return {
        "rmse": float(np.sqrt(((x - back) ** 2).mean())),
        "wall_ms": elapsed_ms,
        "cycles": cycles_total,
        "cycles_prng": PRNG_CYCLES_PER_VECTOR * n_vec,
    }


def run(dims, bits_list, n_vec, seeds, seed):
    rows = []
    for dim in dims:
        rng = np.random.default_rng(seed + dim)
        x = synth_kv(rng, n_vec, dim)
        for bits in bits_list:
            phi = measure_phi(x, bits)
            rand_runs = [measure_random(x, bits, seed=seed + dim * 31 + s) for s in range(seeds)]
            rand = {
                "rmse": float(np.mean([r["rmse"] for r in rand_runs])),
                "rmse_std": float(np.std([r["rmse"] for r in rand_runs])),
                "wall_ms": float(np.mean([r["wall_ms"] for r in rand_runs])),
                "cycles": float(np.mean([r["cycles"] for r in rand_runs])),
                "cycles_prng": float(np.mean([r["cycles_prng"] for r in rand_runs])),
            }
            rows.append({"dim": dim, "bits": bits, "phi": phi, "random": rand})
    return rows


def format_markdown(rows) -> str:
    lines = ["# φ-Givens vs Random rotation — head-to-head", ""]
    lines.append(
        "Three metrics on identical synthetic KV-like inputs (Gaussian + 1/64 "
        "outliers ~6σ). Random rotation is a fresh QR-orthonormal `dim × dim` "
        "matrix per run; cycle counts include PRNG latency at "
        f"{PRNG_CYCLES_PER_VECTOR} cycles/vector. φ-Givens uses the three-layer "
        "structure with `cycles_dma_per_byte = 0` (we count compute only, since "
        "DMA is identical between the two approaches)."
    )
    lines.append("")
    lines.append("## Per (dim, bits) results")
    lines.append("")
    lines.append("| dim | bits | φ RMSE | random RMSE (μ ± σ) | φ wall ms | random wall ms | φ cycles | random cycles |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        phi = row["phi"]
        rnd = row["random"]
        lines.append(
            f"| {row['dim']} | {row['bits']} | "
            f"{phi['rmse']:.4f} | {rnd['rmse']:.4f} ± {rnd['rmse_std']:.4f} | "
            f"{phi['wall_ms']:.2f} | {rnd['wall_ms']:.2f} | "
            f"{phi['cycles']} | {rnd['cycles']:.0f} |"
        )
    lines.append("")
    avg_phi_cycles = np.mean([r["phi"]["cycles"] for r in rows])
    avg_rand_cycles = np.mean([r["random"]["cycles"] for r in rows])
    avg_phi_rmse = np.mean([r["phi"]["rmse"] for r in rows])
    avg_rand_rmse = np.mean([r["random"]["rmse"] for r in rows])
    avg_phi_wall = np.mean([r["phi"]["wall_ms"] for r in rows])
    avg_rand_wall = np.mean([r["random"]["wall_ms"] for r in rows])
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Average RMSE: φ **{avg_phi_rmse:.4f}** vs random {avg_rand_rmse:.4f}")
    lines.append(f"- Average wall ms: φ **{avg_phi_wall:.2f}** vs random {avg_rand_wall:.2f}")
    lines.append(f"- Average cycles: φ **{avg_phi_cycles:.0f}** vs random {avg_rand_cycles:.0f}")
    lines.append("")
    rmse_ok = avg_phi_rmse <= avg_rand_rmse * 1.15
    cycles_ok = avg_phi_cycles < avg_rand_cycles
    lines.append("## Verdict")
    lines.append("")
    if rmse_ok and cycles_ok:
        lines.append(
            "**φ-Givens wins.** Quality (RMSE) is within "
            f"{100 * (avg_phi_rmse - avg_rand_rmse) / avg_rand_rmse:+.1f}% of "
            "random rotation, while cycles are "
            f"{100 * (1 - avg_phi_cycles / avg_rand_cycles):.1f}% lower. The "
            "static three-layer Givens topology amortises across batch and "
            "spends zero PRNG cycles."
        )
    elif rmse_ok:
        lines.append(
            "φ-Givens matches random on RMSE but not on cycles in this "
            "configuration; revisit the cycle model."
        )
    else:
        lines.append(
            "φ-Givens does not match random on RMSE in this configuration "
            f"(gap +{100 * (avg_phi_rmse - avg_rand_rmse) / avg_rand_rmse:.1f}%)."
            " The synthetic distribution may be too Gaussian; rerun on real KV."
        )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python bench/phi_vs_random.py --out bench/phi_vs_random.md")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--vectors", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench/phi_vs_random.md"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = run(args.dims, args.bits, args.vectors, args.seeds, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(rows))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
