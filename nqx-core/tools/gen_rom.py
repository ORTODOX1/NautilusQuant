#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from nqx.constants import NQXConfig
from nqx.lut import GoldenAngleLUT


def fp32_hex(value: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f"{bits:08x}"


def emit_rom(dim: int, out_path: Path) -> int:
    cfg = NQXConfig(dim=dim)
    lut = GoldenAngleLUT(cfg)
    lines = []
    lines.append(f"// golden_rom.mem  generated for dim={dim}, phi={cfg.phi:.18f}")
    lines.append("// layout per layer: pair_i, pair_j, cos(angle), sin(angle)")
    lines.append("// each value is a 32-bit hex word; FP32 (IEEE-754, little-endian byte stream)")
    addr = 0
    for layer_name in ("L1", "L2", "L3"):
        layer = lut.layers[layer_name]
        lines.append(f"// ---- {layer_name}: {len(layer)} pairs ----")
        for k in range(len(layer)):
            i, j = layer.pairs[k]
            c = layer.cos[k]
            s = layer.sin[k]
            lines.append(f"{i:08x}  // [{addr:04x}] {layer_name}.pair_i[{k}]")
            addr += 1
            lines.append(f"{j:08x}  // [{addr:04x}] {layer_name}.pair_j[{k}]")
            addr += 1
            lines.append(f"{fp32_hex(c)}  // [{addr:04x}] {layer_name}.cos[{k}] = {c:.10f}")
            addr += 1
            lines.append(f"{fp32_hex(s)}  // [{addr:04x}] {layer_name}.sin[{k}] = {s:.10f}")
            addr += 1
    out_path.write_text("\n".join(lines) + "\n")
    return addr


def verify(dim: int, mem_path: Path) -> None:
    cfg = NQXConfig(dim=dim)
    lut = GoldenAngleLUT(cfg)
    text = mem_path.read_text().splitlines()
    words = [
        int(line.split()[0], 16) for line in text if line.strip() and not line.startswith("//")
    ]
    idx = 0
    for layer_name in ("L1", "L2", "L3"):
        layer = lut.layers[layer_name]
        for k in range(len(layer)):
            i, j = layer.pairs[k]
            assert words[idx] == i, f"{layer_name}.pair_i[{k}]"
            idx += 1
            assert words[idx] == j, f"{layer_name}.pair_j[{k}]"
            idx += 1
            cos_packed = struct.pack("<I", words[idx])
            cos_val = struct.unpack("<f", cos_packed)[0]
            assert np.isclose(cos_val, layer.cos[k], atol=0)
            idx += 1
            sin_packed = struct.pack("<I", words[idx])
            sin_val = struct.unpack("<f", sin_packed)[0]
            assert np.isclose(sin_val, layer.sin[k], atol=0)
            idx += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Emit golden_rom.mem for RTL simulator.")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out", type=Path, default=repo_root / "rtl" / "golden_rom.mem")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_words = emit_rom(args.dim, args.out)
    print(f"Wrote {args.out} ({n_words} 32-bit words, dim={args.dim})")
    if args.verify:
        verify(args.dim, args.out)
        print("ROM verified vs GoldenAngleLUT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
