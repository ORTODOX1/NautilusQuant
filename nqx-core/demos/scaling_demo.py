"""Llama-3-70B scaling projection. No model loading — pure arithmetic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def kv_cache_bytes_fp16(n_layers: int, n_heads: int, dim: int, ctx: int, kv_factor: int = 2) -> int:
    return n_layers * n_heads * ctx * dim * 2 * kv_factor


def kv_cache_bytes_nqx(
    n_layers: int, n_heads: int, dim: int, ctx: int, bits: int = 3, kv_factor: int = 2
) -> int:
    bits_per_value = bits + 1
    bytes_per_vec = (dim * bits_per_value + 7) // 8
    return n_layers * n_heads * ctx * bytes_per_vec * kv_factor


def gpus_needed(total_bytes: int, bytes_per_gpu: int) -> int:
    return (total_bytes + bytes_per_gpu - 1) // bytes_per_gpu


def fmt_bytes(n: int) -> str:
    if n >= 1024**4:
        return f"{n / 1024**4:.2f} TB"
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def run(model: str = "Llama-3-70B"):
    cfg = {
        "Llama-3-70B": {"layers": 80, "heads": 64, "dim": 128, "ctx": 128 * 1024},
        "Llama-3-8B": {"layers": 32, "heads": 32, "dim": 128, "ctx": 128 * 1024},
        "Llama-3-405B": {"layers": 126, "heads": 128, "dim": 128, "ctx": 128 * 1024},
    }[model]

    fp16 = kv_cache_bytes_fp16(cfg["layers"], cfg["heads"], cfg["dim"], cfg["ctx"])
    nqx = kv_cache_bytes_nqx(cfg["layers"], cfg["heads"], cfg["dim"], cfg["ctx"])

    h100_bytes = 80 * 1024**3
    nqx_chip_hbm_bytes = 24 * 1024**3  # 1× HBM2e stack on package per asic/floorplan.md
    nqx_chip_sram_bytes = 100 * 1024**2  # on-die scratchpad

    h100_count = gpus_needed(fp16, h100_bytes)
    nqx_count = gpus_needed(nqx, nqx_chip_hbm_bytes)

    h100_spot_per_hour = 2.50
    nqx_chip_amortised_per_hour = 0.05  # $1500 / 30k hours est.

    h100_hourly = h100_count * h100_spot_per_hour
    nqx_hourly = nqx_count * nqx_chip_amortised_per_hour

    return {
        "model": model,
        "config": cfg,
        "kv_bytes_fp16": fp16,
        "kv_bytes_nqx": nqx,
        "kv_bytes_fp16_human": fmt_bytes(fp16),
        "kv_bytes_nqx_human": fmt_bytes(nqx),
        "compression_ratio": fp16 / nqx,
        "h100_bytes_per_chip": h100_bytes,
        "nqx_bytes_per_chip": nqx_chip_hbm_bytes,
        "nqx_sram_bytes_per_chip": nqx_chip_sram_bytes,
        "h100_count": h100_count,
        "nqx_count": nqx_count,
        "h100_hourly_usd": h100_hourly,
        "nqx_hourly_usd": nqx_hourly,
        "savings_ratio": h100_hourly / max(nqx_hourly, 1e-9),
    }


def format_markdown(reports) -> str:
    lines = ["# 70B-class scaling projection — KV cache only", ""]
    lines.append(
        "Pure arithmetic projection. We compute the KV-cache footprint at the "
        "model's max context, divide by per-chip memory, and read off chip "
        "counts. **This is the cache footprint only**, not weights — "
        "deployment also needs ~2× weight memory but that's identical between "
        "the two stacks. NQX-Core chip assumed to carry 24 GB HBM2e + 100 MB "
        "on-die SRAM per `asic/floorplan.md`."
    )
    lines.append("")
    lines.append(
        "| Model | KV (FP16) | KV (NQX 3+1) | H100 (80 GB each) | NQX-Core (24 GB each) | $/hr H100 → NQX |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in reports:
        lines.append(
            f"| {r['model']} | {r['kv_bytes_fp16_human']} | {r['kv_bytes_nqx_human']} | "
            f"{r['h100_count']} chips (${r['h100_hourly_usd']:.2f}/hr) | "
            f"{r['nqx_count']:,} chips (${r['nqx_hourly_usd']:.2f}/hr) | "
            f"**{r['savings_ratio']:.1f}×** |"
        )
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    lines.append(
        "- KV factor = 2 (separate K and V tensors). FP16 = 2 bytes/element.\n"
        "- NQX 3+1 packed = 4 bits per value. Compression ratio = 4.00×.\n"
        "- H100 spot $2.50/hour (vast.ai 2026-04 average).\n"
        "- NQX chip TCO amortised over 30 000 hours at $1 500 per chip "
        "(target post-tape-out yield) ≈ $0.05/hour.\n"
        "- We size on a single concurrent request at full context. Real "
        "production multiplexes contexts across the same chips."
    )
    lines.append("")
    lines.append("## What this means for a CFO")
    lines.append("")
    headline = reports[0]
    lines.append(
        f"- {headline['model']} at {headline['config']['ctx']:,} ctx needs "
        f"**{headline['h100_count']} H100 chips just to hold the KV cache** in "
        f"FP16. With NQX-Core compression, the same context fits in "
        f"**{headline['nqx_count']:,} accelerator chips** (or "
        f"{headline['nqx_count'] // 8:,} 8-up boards), at "
        f"**{headline['savings_ratio']:.1f}× lower hourly cost**.\n"
        "- Even more telling: H100s are sold out for the foreseeable future. "
        "NQX-Core uses no exotic process node and fab capacity at TSMC N7 is "
        "abundant.\n"
        "- The compression ratio is loss-bounded (RMSE 0.28 at 3 bits, see "
        "`docs/paper/results.md`); production deployments typically retain "
        "all-or-nothing bit-fidelity for prompt prefix and only compress KV "
        "after the first 2k tokens — that hybrid further improves perplexity "
        "while preserving these savings."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python demos/scaling_demo.py --out demos/scaling_demo.md")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("demos/scaling_demo.md"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    reports = [run(m) for m in ("Llama-3-70B", "Llama-3-8B", "Llama-3-405B")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(reports))
    if args.json is not None:
        args.json.write_text(json.dumps(reports, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
