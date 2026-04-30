"""Pure-numpy TurboQuant baseline emulation. No torch."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from nqx.constants import NQXConfig


PRNG_CYCLES_PER_RANDOM = 4
PRNG_PJ_PER_BYTE = 0.4


@dataclass
class TurboEncoded:
    rotation: np.ndarray
    q: np.ndarray
    sign: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    shape: tuple
    bits: int
    qjl_alpha: float


def random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.standard_normal((n, n)).astype(np.float64)
    q, r = np.linalg.qr(a)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    return (q * sign).astype(np.float32)


def to_polar(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[..., 0::2] = np.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
    out[..., 1::2] = np.arctan2(x[..., 1::2], x[..., 0::2])
    return out


def from_polar(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p)
    out[..., 0::2] = p[..., 0::2] * np.cos(p[..., 1::2])
    out[..., 1::2] = p[..., 0::2] * np.sin(p[..., 1::2])
    return out


def quant_dequant_with_qjl(x: np.ndarray, bits: int, alpha: float):
    levels = 2 ** bits
    if x.ndim == 1:
        xb = x.reshape(1, -1)
    else:
        xb = x
    mins = xb.min(axis=0)
    maxs = xb.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    norm = (xb - mins) / ranges
    q = np.round(norm * (levels - 1)).clip(0, levels - 1).astype(np.uint8)
    dequant = (q.astype(np.float32) / (levels - 1)) * ranges + mins
    error = xb - dequant
    sign = (error >= 0).astype(np.uint8)
    corrected = dequant + np.sign(error) * np.abs(error) * alpha
    return corrected.reshape(x.shape), q.reshape(xb.shape), sign.reshape(xb.shape), mins, maxs


def encode(x: np.ndarray, bits: int = 3, qjl_alpha: float = 0.5, seed: int = 0) -> TurboEncoded:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    rng = np.random.default_rng(seed)
    dim = x.shape[-1]
    T = random_orthogonal(rng, dim)
    rotated = x @ T
    polar = to_polar(rotated)
    corrected, q, sign, mins, maxs = quant_dequant_with_qjl(polar, bits, qjl_alpha)
    return TurboEncoded(
        rotation=T, q=q, sign=sign, mins=mins, maxs=maxs,
        shape=x.shape, bits=bits, qjl_alpha=qjl_alpha,
    )


def decode(enc: TurboEncoded) -> np.ndarray:
    levels = 2 ** enc.bits
    ranges = np.maximum(enc.maxs - enc.mins, 1e-8)
    dequant = (enc.q.astype(np.float32) / (levels - 1)) * ranges + enc.mins
    cart = from_polar(dequant)
    return cart @ enc.rotation.T


def encode_cycles(cfg: NQXConfig, n_vec: int) -> int:
    dim = cfg.dim
    prng_cycles = PRNG_CYCLES_PER_RANDOM * dim * dim
    rotate_cycles = dim
    polar_cycles = cfg.cycles_polar
    quant_cycles = cfg.cycles_quant_minmax + cfg.cycles_quant_round
    qjl_cycles = cfg.cycles_qjl
    pack_cycles = cfg.cycles_pack
    pipeline_depth = (
        rotate_cycles + polar_cycles + quant_cycles + qjl_cycles + pack_cycles
    )
    return prng_cycles + pipeline_depth + n_vec - 1


def encode_energy_pj(cfg: NQXConfig, n_vec: int) -> dict:
    dim = cfg.dim
    matrix_bytes = dim * dim * 2
    prng_pj = matrix_bytes * PRNG_PJ_PER_BYTE
    matrix_fetch_pj = matrix_bytes * cfg.pj_hbm_byte
    rotate_pj = n_vec * (dim * dim * cfg.pj_fp32_mul + dim * (dim - 1) * cfg.pj_fp32_add)
    polar_pj = n_vec * (dim // 2) * (3 * cfg.pj_fp32_mul + 2 * cfg.pj_fp32_add) * 2
    quant_pj = n_vec * dim * (2 * cfg.pj_fp32_mul + 2 * cfg.pj_fp32_add)
    qjl_pj = n_vec * dim * (1 * cfg.pj_fp32_mul + 2 * cfg.pj_fp32_add)
    pack_pj = ((n_vec * dim * 4) + 7) // 8 * cfg.pj_sram_byte
    bytes_in_pj = n_vec * dim * 2 * cfg.pj_hbm_byte
    bytes_out_pj = ((n_vec * dim * 4) + 7) // 8 * cfg.pj_hbm_byte
    total_pj = (prng_pj + matrix_fetch_pj + rotate_pj + polar_pj + quant_pj
                + qjl_pj + pack_pj + bytes_in_pj + bytes_out_pj)
    return {
        "prng_pj": prng_pj,
        "matrix_fetch_pj": matrix_fetch_pj,
        "rotate_pj": rotate_pj,
        "polar_pj": polar_pj,
        "quant_pj": quant_pj,
        "qjl_pj": qjl_pj,
        "pack_pj": pack_pj,
        "memory_pj": bytes_in_pj + bytes_out_pj,
        "total_pj": total_pj,
        "total_nj": total_pj / 1000.0,
        "energy_nj_per_vec": total_pj / 1000.0 / n_vec,
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def state_size_bytes(cfg: NQXConfig) -> int:
    dim = cfg.dim
    return dim * dim * 2


if __name__ == "__main__":
    cfg = NQXConfig(dim=128)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1024, 128)).astype(np.float32)
    enc = encode(x)
    back = decode(enc)
    print(f"TurboQuant RMSE roundtrip = {rmse(x, back):.4f}")
    print(f"TurboQuant cycles (1024 vec) = {encode_cycles(cfg, 1024)}")
    energy = encode_energy_pj(cfg, 1024)
    print(f"TurboQuant energy nJ/vec  = {energy['energy_nj_per_vec']:.3f}")
    print(f"TurboQuant matrix bytes   = {state_size_bytes(cfg)}")
