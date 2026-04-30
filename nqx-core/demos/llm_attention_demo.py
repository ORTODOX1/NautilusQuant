"""Synthetic single-head attention forward pass on NQX-compressed K/V."""
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


def synth_qkv(rng: np.random.Generator, n_heads: int, seq: int, dim: int):
    shape = (n_heads, seq, dim)
    q = rng.standard_normal(shape).astype(np.float32) * 0.7
    k = rng.standard_normal(shape).astype(np.float32) * 0.7
    v = rng.standard_normal(shape).astype(np.float32) * 0.7
    n_out = max(1, n_heads * seq // 128)
    for _ in range(n_out):
        h = int(rng.integers(0, n_heads))
        s = int(rng.integers(0, seq))
        c = int(rng.integers(0, dim))
        scale = float(rng.choice([-1.0, 1.0])) * 6.0
        k[h, s, c] += scale
        v[h, s, c] += scale
    return q, k, v


def attention_fp16(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.einsum("hsd,htd->hst", q, k) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("hst,htd->hsd", weights, v)


def attention_nqx(q, k, v, core: NQXCore):
    n_heads, seq, dim = q.shape
    k_recon = np.zeros_like(k)
    v_recon = np.zeros_like(v)
    for h in range(n_heads):
        enc_k = core.encode(k[h])
        enc_v = core.encode(v[h])
        dec_k = core.decode(enc_k)
        dec_v = core.decode(enc_v)
        k_recon[h] = dec_k.reconstructed.reshape(seq, dim)
        v_recon[h] = dec_v.reconstructed.reshape(seq, dim)
    return attention_fp16(q, k_recon, v_recon), k_recon, v_recon


def kv_bytes_fp16(n_heads: int, seq: int, dim: int) -> int:
    return n_heads * seq * dim * 2 * 2  # K + V, FP16


def kv_bytes_nqx(n_heads: int, seq: int, dim: int, bits: int = 3) -> int:
    bits_per_value = bits + 1
    bytes_per_vec = (dim * bits_per_value + 7) // 8
    return n_heads * seq * bytes_per_vec * 2  # K + V


def cycles_per_token_nqx(cfg: NQXConfig, n_heads: int, seq: int) -> int:
    depth = (
        3 * cfg.cycles_givens_layer
        + cfg.cycles_polar
        + cfg.cycles_quant_minmax
        + cfg.cycles_quant_round
        + cfg.cycles_qjl
        + cfg.cycles_pack
    )
    encode_cycles = (depth + seq - 1) * 2 * n_heads
    return encode_cycles + n_heads * seq * 2  # +pack/dispatch overhead


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def run(n_heads: int, seq: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q, k, v = synth_qkv(rng, n_heads, seq, dim)
    out_fp16 = attention_fp16(q, k, v)
    cfg = NQXConfig(dim=dim)
    core = NQXCore(cfg)
    out_nqx, k_recon, v_recon = attention_nqx(q, k, v, core)
    return {
        "n_heads": n_heads,
        "seq": seq,
        "dim": dim,
        "rmse_attention_output": rmse(out_fp16, out_nqx),
        "rmse_k_recon": rmse(k, k_recon),
        "rmse_v_recon": rmse(v, v_recon),
        "kv_bytes_fp16": kv_bytes_fp16(n_heads, seq, dim),
        "kv_bytes_nqx": kv_bytes_nqx(n_heads, seq, dim),
        "compression_ratio": kv_bytes_fp16(n_heads, seq, dim) / kv_bytes_nqx(n_heads, seq, dim),
        "cycles_per_token_nqx": cycles_per_token_nqx(cfg, n_heads, seq) // seq,
    }


def format_markdown(report: dict) -> str:
    lines = ["# End-to-end attention demo on NQX-compressed K/V", ""]
    lines.append(
        f"Synthetic single-layer attention with `n_heads={report['n_heads']}`, "
        f"`seq={report['seq']}`, `dim={report['dim']}`. K and V are encoded via "
        "`NQXCore.encode` and decoded before attention scoring. Q is kept FP32 "
        "for the comparison. Inputs include 1/128 outliers at 6σ to stress the "
        "rotation+quant pipeline."
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| RMSE attention output (NQX vs FP16) | {report['rmse_attention_output']:.4f} |")
    lines.append(f"| RMSE K reconstruction               | {report['rmse_k_recon']:.4f} |")
    lines.append(f"| RMSE V reconstruction               | {report['rmse_v_recon']:.4f} |")
    lines.append(f"| KV cache bytes (FP16)               | {report['kv_bytes_fp16']:,} B |")
    lines.append(f"| KV cache bytes (NQX 3+1)            | {report['kv_bytes_nqx']:,} B |")
    lines.append(f"| **Compression ratio**               | **{report['compression_ratio']:.2f}×** |")
    lines.append(f"| Cycles per decoded token (NQX)      | {report['cycles_per_token_nqx']:,} |")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python demos/llm_attention_demo.py --n-heads 32 --seq 2048 --dim 128")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("demos/llm_attention_demo.md"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    report = run(args.n_heads, args.seq, args.dim, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
