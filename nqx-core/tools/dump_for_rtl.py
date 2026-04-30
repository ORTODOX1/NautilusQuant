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
from nqx.cpu import NQXCore


def fp32_hex(x: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return f"{bits:08x}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = NQXConfig(dim=args.dim)
    core = NQXCore(cfg)
    rng = np.random.default_rng(args.seed)
    vec = rng.standard_normal((1, args.dim)).astype(np.float32)
    rotated = core.forward_rotation(vec)[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for v in rotated:
            f.write(fp32_hex(v) + "\n")
    print(f"Wrote {args.out} ({args.dim} hex words from rotated reference vector)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
