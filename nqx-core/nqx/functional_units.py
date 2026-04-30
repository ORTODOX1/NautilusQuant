"""Functional units of NQX-Core. All math is numpy; cycles & energy reported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from nqx.constants import NQXConfig
from nqx.lut import GoldenAngleLUT


@dataclass
class FUResult:
    cycles: int
    energy_pj: float


class GivensUnit:
    def __init__(self, config: NQXConfig, lut: GoldenAngleLUT):
        self.config = config
        self.lut = lut

    def apply_layer(
        self, x: np.ndarray, layer_id: int, inverse: bool = False
    ) -> Tuple[np.ndarray, FUResult]:
        layer = self.lut.layer(layer_id)
        out = x.copy()
        n_pairs = len(layer)

        i_idx = layer.i_idx
        j_idx = layer.j_idx
        c = layer.cos_arr
        s = layer.sin_arr if not inverse else -layer.sin_arr

        a = out[..., i_idx]
        b = out[..., j_idx]
        out[..., i_idx] = a * c - b * s
        out[..., j_idx] = a * s + b * c

        cycles = self.config.cycles_givens_layer
        n_lanes_used = n_pairs
        ops_mul = 4 * n_lanes_used
        ops_add = 2 * n_lanes_used
        energy = (
            ops_mul * self.config.pj_fp32_mul
            + ops_add * self.config.pj_fp32_add
            + n_lanes_used * 2 * self.config.pj_rom_read
        )
        n_vectors = x.shape[0] if x.ndim > 1 else 1
        return out, FUResult(cycles=cycles, energy_pj=energy * n_vectors)


class PolarUnit:
    def __init__(self, config: NQXConfig):
        self.config = config

    def to_polar(self, x: np.ndarray) -> Tuple[np.ndarray, FUResult]:
        out = np.zeros_like(x)
        dim = self.config.dim
        for k in range(dim // 2):
            i, j = 2 * k, 2 * k + 1
            xi = x[..., i]
            yj = x[..., j]
            out[..., i] = np.sqrt(xi * xi + yj * yj)
            out[..., j] = np.arctan2(yj, xi)
        if dim % 2:
            out[..., -1] = x[..., -1]

        cycles = self.config.cycles_polar
        n_vectors = x.shape[0] if x.ndim > 1 else 1
        n_pairs = dim // 2
        energy = (
            n_pairs * (3 * self.config.pj_fp32_mul + 2 * self.config.pj_fp32_add) * 2 * n_vectors
        )
        return out, FUResult(cycles=cycles, energy_pj=energy)

    def from_polar(self, p: np.ndarray) -> Tuple[np.ndarray, FUResult]:
        out = np.zeros_like(p)
        dim = self.config.dim
        for k in range(dim // 2):
            i, j = 2 * k, 2 * k + 1
            r = p[..., i]
            theta = p[..., j]
            out[..., i] = r * np.cos(theta)
            out[..., j] = r * np.sin(theta)
        if dim % 2:
            out[..., -1] = p[..., -1]

        cycles = self.config.cycles_polar
        n_vectors = p.shape[0] if p.ndim > 1 else 1
        n_pairs = dim // 2
        energy = n_pairs * (2 * self.config.pj_fp32_mul) * 2 * n_vectors
        return out, FUResult(cycles=cycles, energy_pj=energy)


class QuantUnit:
    def __init__(self, config: NQXConfig):
        self.config = config

    def quantize(
        self, x: np.ndarray, bits: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FUResult]:
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

        n = xb.shape[0]
        cycles = (
            self.config.cycles_quant_minmax + self.config.cycles_quant_round
            if n > 1
            else 1 + self.config.cycles_quant_round
        )
        energy = n * self.config.dim * (2 * self.config.pj_fp32_mul + 2 * self.config.pj_fp32_add)
        return dequant, q, mins, maxs, FUResult(cycles=cycles, energy_pj=energy)

    def dequantize(
        self, q: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int
    ) -> Tuple[np.ndarray, FUResult]:
        levels = 2**bits
        ranges = np.maximum(maxs - mins, 1e-8)
        dequant = (q.astype(np.float32) / (levels - 1)) * ranges + mins

        n = q.shape[0] if q.ndim > 1 else 1
        cycles = self.config.cycles_quant_round
        energy = n * self.config.dim * (1 * self.config.pj_fp32_mul + 1 * self.config.pj_fp32_add)
        return dequant, FUResult(cycles=cycles, energy_pj=energy)


class QJLUnit:
    def __init__(self, config: NQXConfig):
        self.config = config

    def apply(
        self, original: np.ndarray, quantized: np.ndarray, alpha: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray, FUResult]:
        a = self.config.qjl_alpha if alpha is None else alpha
        error = original - quantized
        sign_bits = (error >= 0).astype(np.uint8)
        corrected = quantized + np.sign(error) * np.abs(error) * a

        n = original.shape[0] if original.ndim > 1 else 1
        cycles = self.config.cycles_qjl
        energy = n * self.config.dim * (1 * self.config.pj_fp32_mul + 2 * self.config.pj_fp32_add)
        return corrected, sign_bits, FUResult(cycles=cycles, energy_pj=energy)


class AttentionUnit:
    def __init__(self, config: NQXConfig):
        self.config = config

    def dot_polar(self, q_polar: np.ndarray, k_polar: np.ndarray) -> Tuple[np.ndarray, FUResult]:
        if q_polar.ndim == 1:
            q_polar = q_polar.reshape(1, -1)
        if k_polar.ndim == 1:
            k_polar = k_polar.reshape(1, -1)
        assert q_polar.shape[-1] == k_polar.shape[-1]
        dim = q_polar.shape[-1]
        n_pairs = dim // 2

        r_q = q_polar[:, 0::2]
        t_q = q_polar[:, 1::2]
        r_k = k_polar[:, 0::2]
        t_k = k_polar[:, 1::2]

        diff = t_q[:, None, :] - t_k[None, :, :]
        contrib = r_q[:, None, :] * r_k[None, :, :] * np.cos(diff)
        result = contrib.sum(axis=-1).astype(np.float32)

        if dim % 2:
            result = result + (q_polar[:, -1:][:, :, None] * k_polar[:, -1][None, None, :]).reshape(
                result.shape
            )

        n_q = q_polar.shape[0]
        n_k = k_polar.shape[0]
        cycles = max(1, n_pairs)
        ops_mul = 2 * n_pairs * n_q * n_k
        ops_add = n_pairs * n_q * n_k
        energy = ops_mul * self.config.pj_fp32_mul + ops_add * self.config.pj_fp32_add
        return result, FUResult(cycles=cycles, energy_pj=energy)


class PackUnit:
    def __init__(self, config: NQXConfig):
        self.config = config

    def pack3plus1(self, q: np.ndarray, sign_bits: np.ndarray) -> Tuple[bytes, FUResult]:
        if q.ndim == 1:
            q = q.reshape(1, -1)
            sign_bits = sign_bits.reshape(1, -1)
        n, d = q.shape
        bits = self.config.bits
        total_bits_per_value = bits + 1
        total_bits = n * d * total_bits_per_value
        n_bytes = (total_bits + 7) // 8

        mask_bits = (1 << bits) - 1
        combined = (q.astype(np.uint8) & mask_bits) | ((sign_bits.astype(np.uint8) & 1) << bits)
        flat = combined.reshape(-1)
        shifts = np.arange(total_bits_per_value, dtype=np.uint8)
        expanded = ((flat[:, None] >> shifts) & 1).astype(np.uint8)
        bit_stream = expanded.reshape(-1)
        pad = (8 - bit_stream.size % 8) % 8
        if pad:
            bit_stream = np.concatenate([bit_stream, np.zeros(pad, dtype=np.uint8)])
        packed = np.packbits(bit_stream, bitorder="little")
        out = packed.tobytes()
        if len(out) > n_bytes:
            out = out[:n_bytes]

        cycles = self.config.cycles_pack
        energy = n_bytes * self.config.pj_sram_byte
        return out, FUResult(cycles=cycles, energy_pj=energy)

    def unpack3plus1(self, data: bytes, n: int) -> Tuple[np.ndarray, np.ndarray, FUResult]:
        d = self.config.dim
        bits = self.config.bits
        total = bits + 1
        mask_bits = (1 << bits) - 1
        n_values = n * d

        bit_stream = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
        needed = n_values * total
        bit_stream = bit_stream[:needed].reshape(n_values, total)
        weights = 1 << np.arange(total, dtype=np.uint8)
        values = (bit_stream * weights).sum(axis=1).astype(np.uint8)
        q = (values & mask_bits).reshape(n, d)
        sign = ((values >> bits) & 1).reshape(n, d)

        cycles = self.config.cycles_pack
        energy = len(data) * self.config.pj_sram_byte
        return q, sign, FUResult(cycles=cycles, energy_pj=energy)
