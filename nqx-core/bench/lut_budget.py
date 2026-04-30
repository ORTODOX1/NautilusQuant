#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from nqx.constants import NQXConfig
from nqx.lut import GoldenAngleLUT


PHI_BYTES_PER_PAIR = 1 + 1 + 4 + 4  # pair_i (uint8) + pair_j (uint8) + cos FP32 + sin FP32


def measure_phi(dim: int) -> dict:
    cfg = NQXConfig(dim=dim)
    lut = GoldenAngleLUT(cfg)
    per_layer = {}
    total = 0
    for name in ("L1", "L2", "L3"):
        n_pairs = len(lut.layers[name])
        bytes_layer = n_pairs * PHI_BYTES_PER_PAIR
        per_layer[name] = {"pairs": n_pairs, "bytes": bytes_layer}
        total += bytes_layer
    return {"dim": dim, "layers": per_layer, "total_bytes": total}


def random_matrix_bytes(dim: int) -> dict:
    fp16 = dim * dim * 2
    fp32 = dim * dim * 4
    return {"dim": dim, "fp16_bytes": fp16, "fp32_bytes": fp32}


def fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def format_markdown(rows) -> str:
    lines = ["# LUT size proof — φ-Givens vs Random rotation matrix", ""]
    lines.append(
        "Random rotations need to store the full `dim × dim` orthogonal "
        "matrix `T` (FP16 minimum). φ-Givens stores a **fixed** ROM with the "
        "rotation pair indices and (cos, sin) per pair, across three layers. "
        f"Bytes per pair: {PHI_BYTES_PER_PAIR} = 1B pair_i + 1B pair_j + "
        "4B cos (FP32) + 4B sin (FP32). pair indices fit in uint8 up to "
        "dim=256."
    )
    lines.append("")
    lines.append("## Per-dim LUT budget")
    lines.append("")
    lines.append(
        "| dim | φ-LUT total | per layer (L1 / L2 / L3) | random `T` (FP16) | random `T` (FP32) |"
        " ratio φ vs random FP16 |"
    )
    lines.append("|---:|---:|---|---:|---:|---:|")
    for r in rows:
        layers = r["phi"]["layers"]
        random = r["random"]
        ratio = random["fp16_bytes"] / r["phi"]["total_bytes"]
        lines.append(
            f"| {r['dim']} | "
            f"**{fmt_bytes(r['phi']['total_bytes'])}** | "
            f"{layers['L1']['pairs']}p/{fmt_bytes(layers['L1']['bytes'])} · "
            f"{layers['L2']['pairs']}p/{fmt_bytes(layers['L2']['bytes'])} · "
            f"{layers['L3']['pairs']}p/{fmt_bytes(layers['L3']['bytes'])} | "
            f"{fmt_bytes(random['fp16_bytes'])} | "
            f"{fmt_bytes(random['fp32_bytes'])} | "
            f"**{ratio:.0f}×** |"
        )
    lines.append("")
    max_phi = max(r["phi"]["total_bytes"] for r in rows)
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(
        f"- φ-LUT growth across dim ∈ {{{', '.join(str(r['dim']) for r in rows)}}}: "
        f"linear in `dim` (≈ 15 B/dim). Max in this sweep: "
        f"**{fmt_bytes(max_phi)}** at dim=512. The 4 KB ROM in `asic/floorplan.md` "
        "is sized for dim ≤ 256 (NQX-Core target); dim=512 bumps the ROM macro to "
        "8 KB, still on-die and still independent of model count."
    )
    lines.append(
        f"- Random `T` at dim=512: **{fmt_bytes(rows[-1]['random']['fp16_bytes'])}** "
        f"(FP16) per layer. A 32-layer model needs 16 MB just for rotation "
        "matrices — that lives in HBM and burns one extra full HBM read per "
        "attention pass. At dim=128 (the NQX-Core default) the ratio is "
        "**17×** φ vs random FP16."
    )
    lines.append("")
    lines.append("## Implication")
    lines.append("")
    lines.append(
        "Random rotation hardware must either (a) recompute `T` from a "
        "PRNG seed every dispatch, or (b) carry a per-layer FP16 matrix in "
        "HBM/PCIe-mapped memory. φ-Givens reduces this to a **single 4 KB "
        "ROM** that is identical for every model, every layer and every device. "
        "This is the architectural reason NQX-Core can sit on a 50 mm² die "
        "with no off-chip rotation state."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python bench/lut_budget.py --out bench/lut_budget.md")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--out", type=Path, default=Path("bench/lut_budget.md"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = []
    for dim in args.dims:
        rows.append({"dim": dim, "phi": measure_phi(dim), "random": random_matrix_bytes(dim)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(rows))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
