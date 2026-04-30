from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from nqx.constants import NQXConfig
from nqx.functional_units import FUResult


def _scalar_quantize(
    x: np.ndarray, bits: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    levels = 2**bits
    if x.ndim == 1:
        xb = x.reshape(1, -1)
    else:
        xb = x
    mins = xb.min(axis=0)
    maxs = xb.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    normalized = (xb - mins) / ranges
    q = np.round(normalized * (levels - 1)).clip(0, levels - 1).astype(np.uint8)
    dequant = (q.astype(np.float32) / (levels - 1)) * ranges + mins
    return dequant.reshape(x.shape), q.reshape(x.shape if x.ndim > 1 else (1, x.size)), mins, maxs


class SubBitUnit:
    def __init__(self, config: NQXConfig):
        self.config = config

    def encode(
        self,
        polar: np.ndarray,
        radius_bits: int,
        angle_bits: int,
    ) -> Tuple[np.ndarray, Dict[str, Any], FUResult]:
        assert polar.shape[-1] % 2 == 0, "polar dim must be even"
        if polar.ndim == 1:
            polar = polar.reshape(1, -1)
        radii = polar[..., 0::2]
        angles = polar[..., 1::2]

        r_dequant, r_q, r_mins, r_maxs = _scalar_quantize(radii, radius_bits)
        a_dequant, a_q, a_mins, a_maxs = _scalar_quantize(angles, angle_bits)

        result = np.zeros_like(polar)
        result[..., 0::2] = r_dequant
        result[..., 1::2] = a_dequant

        meta: Dict[str, Any] = {
            "radius_bits": radius_bits,
            "angle_bits": angle_bits,
            "r_q": r_q,
            "a_q": a_q,
            "r_mins": r_mins,
            "r_maxs": r_maxs,
            "a_mins": a_mins,
            "a_maxs": a_maxs,
            "bits_per_value": (radius_bits + angle_bits) / 2,
            "compression_ratio": 16.0 / ((radius_bits + angle_bits) / 2),
        }

        n = polar.shape[0]
        d = polar.shape[-1]
        cycles = self.config.cycles_quant_minmax + self.config.cycles_quant_round
        energy = n * d * (2 * self.config.pj_fp32_mul + 2 * self.config.pj_fp32_add)
        return result, meta, FUResult(cycles=cycles, energy_pj=energy)

    def decode(self, meta: Dict[str, Any]) -> Tuple[np.ndarray, FUResult]:
        r_bits = meta["radius_bits"]
        a_bits = meta["angle_bits"]
        r_levels = 2**r_bits
        a_levels = 2**a_bits
        r_q = meta["r_q"]
        a_q = meta["a_q"]
        r_mins = meta["r_mins"]
        r_maxs = meta["r_maxs"]
        a_mins = meta["a_mins"]
        a_maxs = meta["a_maxs"]
        r_ranges = np.maximum(r_maxs - r_mins, 1e-8)
        a_ranges = np.maximum(a_maxs - a_mins, 1e-8)
        r_dequant = (r_q.astype(np.float32) / (r_levels - 1)) * r_ranges + r_mins
        a_dequant = (a_q.astype(np.float32) / (a_levels - 1)) * a_ranges + a_mins
        n = r_q.shape[0]
        d = r_q.shape[-1] + a_q.shape[-1]
        out = np.zeros((n, d), dtype=np.float32)
        out[..., 0::2] = r_dequant
        out[..., 1::2] = a_dequant
        cycles = self.config.cycles_quant_round
        energy = n * d * (1 * self.config.pj_fp32_mul + 1 * self.config.pj_fp32_add)
        return out, FUResult(cycles=cycles, energy_pj=energy)
