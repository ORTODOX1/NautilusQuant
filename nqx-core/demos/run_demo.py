"""Single-command demo runner. Calls D1, D2, D3, D4 and prints the headline table."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from demos.llm_attention_demo import run as run_attention
from demos.scaling_demo import run as run_scaling
from demos.side_by_side import run as run_side_by_side

GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color(value: str, win: bool) -> str:
    return f"{GREEN}{value}{RESET}" if win else f"{RED}{value}{RESET}"


def print_table(report: dict) -> None:
    t, n = report["turbo"], report["nqx"]
    print(f"{BOLD}=== Side-by-side ({report['n_vec']} vec × dim={report['dim']}) ==={RESET}")
    print(f"{'Metric':<30} {'TurboQuant':>14} {'NautilusQuant':>14}")
    print("-" * 60)
    rows = [
        ("RMSE roundtrip", f"{t['rmse']:.4f}", f"{n['rmse']:.4f}", t["rmse"] >= n["rmse"]),
        ("Cycles per vector", f"{t['cycles_per_vec']:.1f}", f"{n['cycles_per_vec']:.2f}", True),
        ("Energy nJ/vec", f"{t['energy_nj_per_vec']:.2f}", f"{n['energy_nj_per_vec']:.2f}", True),
        ("LUT/PRNG state (B)", f"{t['state_bytes']}", f"{n['state_bytes']}", True),
        (
            "Determinism (unique/runs)",
            f"{t['deterministic_unique']}/{report['runs']}",
            f"{n['deterministic_unique']}/{report['runs']}",
            True,
        ),
    ]
    for label, tv, nv, nqx_wins in rows:
        tv_c = _color(tv, not nqx_wins)
        nv_c = _color(nv, nqx_wins)
        print(f"{label:<30} {tv_c:>23} {nv_c:>23}")
    print()


def print_attention(rep: dict) -> None:
    print(f"{BOLD}=== Single-layer attention demo ==={RESET}")
    print(f"  n_heads={rep['n_heads']} seq={rep['seq']} dim={rep['dim']}")
    print(f"  RMSE attention output (NQX vs FP16) = {rep['rmse_attention_output']:.4f}")
    print(
        f"  KV bytes  FP16={rep['kv_bytes_fp16']:>12,}  "
        f"NQX={rep['kv_bytes_nqx']:>12,}  "
        f"compression={GREEN}{rep['compression_ratio']:.2f}×{RESET}"
    )
    print()


def print_scaling(reports: list) -> None:
    print(f"{BOLD}=== Scaling projection — KV cache only ==={RESET}")
    print(
        f"{'Model':<14} {'KV FP16':>12} {'KV NQX':>12} " f"{'H100':>8} {'NQX':>8} {'savings':>10}"
    )
    print("-" * 72)
    for r in reports:
        savings = f"{r['savings_ratio']:.1f}×"
        print(
            f"{r['model']:<14} {r['kv_bytes_fp16_human']:>12} "
            f"{r['kv_bytes_nqx_human']:>12} "
            f"{r['h100_count']:>8} {r['nqx_count']:>8} "
            f"{GREEN}{savings:>10}{RESET}"
        )
    print()


def render_md_report(side, attn, scaling) -> str:
    lines = ["# NQX-Core demo report", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}Z")
    lines.append("")
    lines.append("## Side-by-side")
    lines.append("")
    t, n = side["turbo"], side["nqx"]
    lines.append("| Metric | TurboQuant | NautilusQuant |")
    lines.append("|---|---:|---:|")
    lines.append(f"| RMSE | {t['rmse']:.4f} | **{n['rmse']:.4f}** |")
    lines.append(f"| Cycles per vec | {t['cycles_per_vec']:.1f} | **{n['cycles_per_vec']:.2f}** |")
    lines.append(
        f"| Energy nJ/vec | {t['energy_nj_per_vec']:.2f} | **{n['energy_nj_per_vec']:.2f}** |"
    )
    lines.append(f"| LUT/PRNG state | {t['state_bytes']} B | **{n['state_bytes']} B** |")
    lines.append(
        f"| Determinism | {t['deterministic_unique']}/{side['runs']} | **{n['deterministic_unique']}/{side['runs']}** |"
    )
    lines.append("")
    lines.append("## Attention demo")
    lines.append("")
    lines.append(f"- n_heads={attn['n_heads']}, seq={attn['seq']}, dim={attn['dim']}")
    lines.append(f"- RMSE attention output: {attn['rmse_attention_output']:.4f}")
    lines.append(f"- Compression: {attn['compression_ratio']:.2f}×")
    lines.append("")
    lines.append("## Scaling projection")
    lines.append("")
    lines.append("| Model | KV FP16 | KV NQX | H100 | NQX | $/hr ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in scaling:
        lines.append(
            f"| {r['model']} | {r['kv_bytes_fp16_human']} | {r['kv_bytes_nqx_human']} | "
            f"{r['h100_count']} | {r['nqx_count']} | {r['savings_ratio']:.1f}× |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=Path("demos"))
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    print(f"{BOLD}NQX-Core demo — running…{RESET}\n")

    side = run_side_by_side(args.vectors, args.dim, args.runs)
    print_table(side)

    attn = run_attention(args.n_heads, args.seq, args.dim)
    print_attention(attn)

    scaling = [run_scaling(m) for m in ("Llama-3-8B", "Llama-3-70B", "Llama-3-405B")]
    print_scaling(scaling)

    elapsed = time.perf_counter() - t_start
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_path = args.out_dir / f"results-{timestamp}.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_md_report(side, attn, scaling))
    print(f"{BOLD}Demo finished in {elapsed:.2f}s. Report → {report_path}{RESET}")
    print("Read demos/pitch.md, demos/why_it_works.md, demos/side_by_side.md " "for the deep dive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
