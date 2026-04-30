"""ASCII visualisations for the demo suite. No matplotlib."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from demos.turboquant_emul import encode as turbo_encode


BAR = "█"


def histogram_bar(value: float, max_value: float, width: int = 40) -> str:
    if max_value <= 0:
        return ""
    n = int(round(width * value / max_value))
    return BAR * max(1, n)


def latency_jitter(n_vec: int = 256, dim: int = 128, runs: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_vec, dim)).astype(np.float32)
    cfg = NQXConfig(dim=dim)

    nqx_lat = []
    for _ in range(runs):
        core = NQXCore(cfg)
        t0 = time.perf_counter()
        core.encode(x)
        nqx_lat.append((time.perf_counter() - t0) * 1000)
    turbo_lat = []
    for s in range(runs):
        t0 = time.perf_counter()
        turbo_encode(x, seed=s)
        turbo_lat.append((time.perf_counter() - t0) * 1000)
    return {"nqx": nqx_lat, "turbo": turbo_lat}


def render_latency_hist(label: str, lat: list, bins: int = 12, width: int = 40) -> str:
    lo = min(lat)
    hi = max(lat)
    rng = max(hi - lo, 1e-9)
    counts = [0] * bins
    for v in lat:
        idx = min(bins - 1, int((v - lo) / rng * bins))
        counts[idx] += 1
    max_c = max(counts) if counts else 1
    out = [f"{label}  min {lo:.2f} ms  max {hi:.2f} ms  jitter {hi - lo:.2f} ms"]
    for i, c in enumerate(counts):
        edge_lo = lo + (i / bins) * rng
        out.append(f"  {edge_lo:6.2f} ms |{histogram_bar(c, max_c, width)} {c}")
    return "\n".join(out)


def render_cycle_breakdown(width: int = 30) -> str:
    cfg = NQXConfig(dim=128)
    nqx = {
        "load":   1,
        "rotate": 3 * cfg.cycles_givens_layer,
        "polar":  cfg.cycles_polar,
        "quant":  cfg.cycles_quant_minmax + cfg.cycles_quant_round,
        "qjl":    cfg.cycles_qjl,
        "pack":   cfg.cycles_pack,
        "store":  1,
    }
    turbo = {
        "load":   1,
        "prng":   4 * cfg.dim * cfg.dim // 100,  # display scale-down
        "rotate": cfg.dim,
        "polar":  cfg.cycles_polar,
        "quant":  cfg.cycles_quant_minmax + cfg.cycles_quant_round,
        "qjl":    cfg.cycles_qjl,
        "pack":   cfg.cycles_pack,
        "store":  1,
    }
    max_total = max(sum(nqx.values()), sum(turbo.values()))
    lines = ["Cycle breakdown by stage (lower is better)", ""]
    lines.append(f"{'Stage':<10} {'NQX':>6} {'Turbo':>6}  scaled bar")
    lines.append("-" * 60)
    stages = sorted(set(list(nqx.keys()) + list(turbo.keys())))
    for s in stages:
        n = nqx.get(s, 0)
        t = turbo.get(s, 0)
        bar_n = histogram_bar(n, max_total, 20)
        bar_t = histogram_bar(t, max_total, 20)
        lines.append(f"{s:<10} {n:>6} {t:>6}  N|{bar_n}")
        lines.append(f"{'':<10} {'':>6} {'':>6}  T|{bar_t}")
    lines.append("")
    lines.append(f"Total cycles  NQX = {sum(nqx.values())}  Turbo = {sum(turbo.values())} (scaled)")
    return "\n".join(lines)


def render_gantt(n_vec: int = 8) -> str:
    stages = ["LDV", "GVNS.L1", "GVNS.L2", "GVNS.L3", "POLAR", "QUANT", "QJL", "PACK", "STV"]
    lines = ["Gantt — pipelined NQX-Core encode", ""]
    width = n_vec + len(stages)
    header = "         " + "".join(f"{c:>1}" for c in [str(i % 10) for i in range(width)])
    lines.append(header)
    for vec in range(n_vec):
        row = "vec" + f"{vec:>2}".rjust(3) + "  "
        timeline = [" "] * width
        for k, _ in enumerate(stages):
            t = vec + k
            timeline[t] = "█"
        lines.append(row + "".join(timeline))
    lines.append("")
    lines.append(f"Stages: {' → '.join(stages)}")
    return "\n".join(lines)


def render_all(out_md: Path | None = None) -> str:
    parts = []
    jitter = latency_jitter()
    parts.append(render_latency_hist("NQX-Core encode latency:", jitter["nqx"]))
    parts.append("")
    parts.append(render_latency_hist("TurboQuant encode latency:", jitter["turbo"]))
    parts.append("")
    parts.append(render_cycle_breakdown())
    parts.append("")
    parts.append(render_gantt())
    text = "\n".join(parts)
    if out_md is not None:
        out_md.write_text("```\n" + text + "\n```\n")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("demos/viz.md"))
    args = ap.parse_args(argv)
    text = render_all(out_md=args.out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
