from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from nqx.constants import NQXConfig
from nqx.functional_units import FUResult

MX_FORMATS: Dict[str, Dict[str, int]] = {
    "MXFP4": {"mantissa_bits": 2, "exponent_bits": 1, "total": 4},
    "MXFP6": {"mantissa_bits": 3, "exponent_bits": 2, "total": 6},
    "MXFP8": {"mantissa_bits": 3, "exponent_bits": 4, "total": 8},
    "MXINT8": {"mantissa_bits": 7, "exponent_bits": 0, "total": 8},
}

MX_FORMAT_BY_INDEX = ["MXFP4", "MXFP6", "MXFP8", "MXINT8"]


def _levels(fmt: Dict[str, int]) -> int:
    if fmt["exponent_bits"] > 0:
        return 2 ** (fmt["mantissa_bits"] + 1)
    return 2 ** (fmt["total"] - 1)


class MXQuantizer:
    def __init__(
        self,
        config: NQXConfig,
        format_name: str = "MXFP4",
        block_size: int = 32,
    ):
        assert format_name in MX_FORMATS, f"unknown MX format {format_name!r}"
        assert block_size > 0
        self.config = config
        self.format_name = format_name
        self.format = MX_FORMATS[format_name]
        self.block_size = block_size
        self.levels = _levels(self.format)
        self.overhead_bits_per_value = 8.0 / block_size

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], FUResult]:
        original_shape = x.shape
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        n = flat.size
        pad = (self.block_size - n % self.block_size) % self.block_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        blocks = flat.reshape(-1, self.block_size)
        n_blocks = blocks.shape[0]

        block_max = np.abs(blocks).max(axis=1)
        block_max = np.maximum(block_max, 1e-30)
        shared_exp = np.floor(np.log2(block_max)).clip(-127, 127).astype(np.int8)

        scale = (2.0 ** shared_exp.astype(np.float32))[:, None]
        normalized = blocks / scale
        levels = self.levels
        q = np.round(normalized * levels).clip(-levels, levels).astype(np.int32)
        dequant_blocks = (q.astype(np.float32) / levels) * scale

        dequant = dequant_blocks.reshape(-1)[:n].reshape(original_shape)

        meta: Dict[str, Any] = {
            "format": self.format_name,
            "block_size": self.block_size,
            "levels": levels,
            "q": q,
            "shared_exp": shared_exp,
            "n_blocks": n_blocks,
            "n_elements": n,
            "pad": pad,
            "original_shape": original_shape,
            "overhead_bits_per_value": self.overhead_bits_per_value,
            "effective_bits": self.format["total"] + self.overhead_bits_per_value,
        }

        cycles = max(1, n_blocks)
        energy = n * (2 * self.config.pj_fp32_mul + 1 * self.config.pj_fp32_add)
        return dequant, meta, FUResult(cycles=cycles, energy_pj=energy)

    def dequantize(self, meta: Dict[str, Any]) -> Tuple[np.ndarray, FUResult]:
        q = meta["q"]
        shared_exp = meta["shared_exp"]
        levels = meta["levels"]
        original_shape = meta["original_shape"]
        n = meta["n_elements"]

        scale = (2.0 ** shared_exp.astype(np.float32))[:, None]
        dequant_blocks = (q.astype(np.float32) / levels) * scale
        out = dequant_blocks.reshape(-1)[:n].reshape(original_shape)

        cycles = max(1, q.shape[0])
        energy = n * (1 * self.config.pj_fp32_mul + 1 * self.config.pj_fp32_add)
        return out, FUResult(cycles=cycles, energy_pj=energy)

    def serialize(self, meta: Dict[str, Any]) -> bytes:
        q = meta["q"].astype(np.int16).tobytes()
        exp = meta["shared_exp"].astype(np.int8).tobytes()
        return exp + q

    def deserialize(
        self,
        data: bytes,
        n_blocks: int,
        original_shape,
        pad: int,
    ) -> Dict[str, Any]:
        block_size = self.block_size
        levels = self.levels
        exp_bytes = n_blocks
        shared_exp = np.frombuffer(data[:exp_bytes], dtype=np.int8).copy()
        q = np.frombuffer(data[exp_bytes:], dtype=np.int16).astype(np.int32).copy()
        q = q.reshape(n_blocks, block_size)
        n_elements = int(np.prod(original_shape))
        return {
            "format": self.format_name,
            "block_size": block_size,
            "levels": levels,
            "q": q,
            "shared_exp": shared_exp,
            "n_blocks": n_blocks,
            "n_elements": n_elements,
            "pad": pad,
            "original_shape": original_shape,
            "overhead_bits_per_value": self.overhead_bits_per_value,
            "effective_bits": self.format["total"] + self.overhead_bits_per_value,
        }
