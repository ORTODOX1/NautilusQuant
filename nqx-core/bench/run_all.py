#!/usr/bin/env python3
"""Run all emulator configs on synthetic data and write results to audits/results/.

Usage:
    python bench/run_all.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS_DIR = os.path.join(ROOT, "audits", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.functional_units import AttentionUnit
from nqx.mx_unit import MXQuantizer, MX_FORMATS
from nqx.subbit_unit import SubBitUnit


def _make_polar(rng, n, dim):
    cart = rng.standard_normal((n, dim)).astype(np.float32)
    polar = np.zeros_like(cart)
    polar[..., 0::2] = np.sqrt(cart[..., 0::2] ** 2 + cart[..., 1::2] ** 2)
    polar[..., 1::2] = np.arctan2(cart[..., 1::2], cart[..., 0::2])
    return polar


def bench_config(dim: int, bits: int, n_vectors: int) -> dict:
    cfg = NQXConfig(dim=dim, bits=bits)
    core = NQXCore(cfg)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_vectors, dim)).astype(np.float32)

    t0 = time.perf_counter()
    enc = core.encode(x)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = core.decode(enc)
    t_dec = time.perf_counter() - t0

    rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
    return {
        "dim": dim,
        "bits": bits,
        "n": n_vectors,
        "enc_cycles": enc.cycles,
        "dec_cycles": dec.cycles,
        "enc_ms": t_enc * 1000,
        "dec_ms": t_dec * 1000,
        "energy_nj": core.energy.total_nj(),
        "rmse": rmse,
    }


def bench_attn_dot(batches):
    cfg = NQXConfig(dim=128)
    attn = AttentionUnit(cfg)
    rng = np.random.default_rng(0)
    results = []
    for n_q, n_k, label in batches:
        pq = _make_polar(rng, n_q, 128)
        pk = _make_polar(rng, n_k, 128)
        t0 = time.perf_counter()
        out, fu = attn.dot_polar(pq, pk)
        dt = time.perf_counter() - t0
        results.append((label, dt * 1000, fu.cycles, fu.energy_pj))
        print(f"  ATTN_DOT {label}: {dt*1000:.3f}ms cycles={fu.cycles} energy={fu.energy_pj:.0f}pJ")
    return results


def bench_mx_all(n_vectors, dim):
    cfg = NQXConfig(dim=dim)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    results = []
    for fmt in MX_FORMATS:
        mx = MXQuantizer(cfg, format_name=fmt)
        t0 = time.perf_counter()
        dequant, meta, fu = mx.quantize(x)
        dt = time.perf_counter() - t0
        rmse = float(np.sqrt(((dequant - x) ** 2).mean()))
        results.append((fmt, dt * 1000, meta["effective_bits"], rmse))
        print(f"  MX {fmt:6s}: {dt*1000:.1f}ms eff_bits={meta['effective_bits']:.1f} rmse={rmse:.4f}")
    return results


def bench_subbit(pairs, n_vectors, dim):
    cfg = NQXConfig(dim=dim)
    sb = SubBitUnit(cfg)
    rng = np.random.default_rng(0)
    polar = _make_polar(rng, n_vectors, dim)
    results = []
    for r_bits, a_bits in pairs:
        t0 = time.perf_counter()
        out, meta, fu = sb.encode(polar, r_bits, a_bits)
        dt = time.perf_counter() - t0
        rmse = float(np.sqrt(((polar - out) ** 2).mean()))
        results.append((r_bits, a_bits, dt * 1000, meta["compression_ratio"], rmse))
        print(f"  SUBBIT (r={r_bits},θ={a_bits}): {dt*1000:.1f}ms comp={meta['compression_ratio']:.1f}x rmse={rmse:.4f}")
    return results


def main():
    configs = [
        (32, 3, 1024),
        (64, 3, 1024),
        (128, 3, 1024),
        (256, 3, 1024),
        (128, 4, 1024),
        (128, 5, 1024),
    ]
    results = []
    for dim, bits, n in configs:
        r = bench_config(dim, bits, n)
        results.append(r)
        print(f"  dim={dim} bits={bits} vectors={n}: enc={r['enc_cycles']}cyc dec={r['dec_cycles']}cyc "
              f"{r['enc_ms']+r['dec_ms']:.1f}ms energy={r['energy_nj']:.0f}nJ rmse={r['rmse']:.4f}")

    print("\n=== Extended: ATTN_DOT ===")
    attn_batches = [(32, 32, "32x32"), (128, 128, "128x128"), (512, 512, "512x512")]
    attn_results = bench_attn_dot(attn_batches)

    print("\n=== Extended: MX quantize (4096x128) ===")
    mx_results = bench_mx_all(4096, 128)

    print("\n=== Extended: SubBit (4096x128) ===")
    subbit_results = bench_subbit([(3, 1), (3, 2)], 4096, 128)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"bench-{ts}.md")
    lines = [
        f"# Bench run {ts}",
        "",
        "| dim | bits | vectors | enc cycles | dec cycles | enc ms | dec ms | energy (nJ) | RMSE |",
        "|----:|-----:|--------:|-----------:|-----------:|-------:|-------:|------------:|-----:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dim']} | {r['bits']} | {r['n']} "
            f"| {r['enc_cycles']} | {r['dec_cycles']} "
            f"| {r['enc_ms']:.2f} | {r['dec_ms']:.2f} "
            f"| {r['energy_nj']:.0f} | {r['rmse']:.4f} |"
        )
    lines.append("")
    lines.append("")
    lines.append("### ATTN_DOT (dim=128)")
    lines.append("| batch | ms | cycles | energy (pJ) |")
    lines.append("|------:|----:|-------:|------------:|")
    for label, ms, cyc, epj in attn_results:
        lines.append(f"| {label} | {ms:.3f} | {cyc} | {epj:.0f} |")

    lines.append("")
    lines.append("### MX quantize (4096×128)")
    lines.append("| format | ms | eff bits | RMSE |")
    lines.append("|-------:|----:|---------:|-----:|")
    for fmt, ms, ebits, rmse in mx_results:
        lines.append(f"| {fmt} | {ms:.1f} | {ebits:.1f} | {rmse:.4f} |")

    lines.append("")
    lines.append("### SubBit (4096×128)")
    lines.append("| r bits | θ bits | ms | comp | RMSE |")
    lines.append("|-------:|-------:|----:|-----:|-----:|")
    for r, a, ms, comp, rmse in subbit_results:
        lines.append(f"| {r} | {a} | {ms:.1f} | {comp:.1f}x | {rmse:.4f} |")

    body = "\n".join(lines)
    with open(path, "w") as f:
        f.write(body)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
