"""Hardware constants and default configuration for NQX-Core."""

from __future__ import annotations

import math
from dataclasses import dataclass

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_ANGLE: float = 2.0 * math.pi / (PHI**2)


@dataclass(frozen=True)
class NQXConfig:
    dim: int = 128
    bits: int = 3
    qjl_alpha: float = 0.5

    n_vector_regs: int = 16
    n_scalar_regs: int = 8

    sram_in_bytes: int = 24 * 1024
    sram_out_bytes: int = 24 * 1024
    rom_lut_bytes: int = 4 * 1024
    hbm_bytes: int = 4 * 1024 * 1024 * 1024

    n_lanes: int = 64

    cycles_givens_layer: int = 1
    cycles_polar: int = 1
    cycles_quant_minmax: int = 7
    cycles_quant_round: int = 1
    cycles_qjl: int = 1
    cycles_pack: int = 1
    cycles_dma_per_byte: float = 0.0625

    pj_fp32_mul: float = 3.7
    pj_fp32_add: float = 0.9
    pj_hbm_byte: float = 5.0
    pj_sram_byte: float = 0.05
    pj_rom_read: float = 0.02

    phi: float = PHI

    def vrf_bytes(self) -> int:
        return self.n_vector_regs * self.dim * 4

    def golden_angle(self) -> float:
        return 2.0 * math.pi / (self.phi**2)

    def n_l1_pairs(self) -> int:
        return self.dim // 2

    def n_l2_pairs(self) -> int:
        return (self.dim - 1) // 2

    def l3_stride(self) -> int:
        return max(2, self.dim // 4)
