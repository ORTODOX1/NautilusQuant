"""Energy accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from nqx.constants import NQXConfig

PRNG_PJ_PER_BYTE = 0.4
RANDOM_T_FETCH_BYTES_PER_LAYER = 2  # FP16 per matrix entry


def random_rotation_energy_pj(
    cfg: NQXConfig, n_vec: int, n_layers: int = 1, store_in_hbm: bool = True
) -> Dict[str, float]:
    dim = cfg.dim
    ops_mul = n_vec * dim * dim * n_layers
    ops_add = n_vec * dim * (dim - 1) * n_layers
    compute_pj = ops_mul * cfg.pj_fp32_mul + ops_add * cfg.pj_fp32_add
    matrix_bytes = dim * dim * RANDOM_T_FETCH_BYTES_PER_LAYER * n_layers
    if store_in_hbm:
        memory_pj = matrix_bytes * cfg.pj_hbm_byte
    else:
        memory_pj = matrix_bytes * cfg.pj_sram_byte
    prng_pj = matrix_bytes * PRNG_PJ_PER_BYTE
    return {
        "compute_pj": compute_pj,
        "matrix_memory_pj": memory_pj,
        "prng_pj": prng_pj,
        "total_pj": compute_pj + memory_pj + prng_pj,
        "matrix_bytes": matrix_bytes,
    }


@dataclass
class EnergyModel:
    by_unit: Dict[str, float] = field(default_factory=dict)
    by_memory: Dict[str, float] = field(default_factory=dict)

    def add_unit(self, name: str, pj: float) -> None:
        self.by_unit[name] = self.by_unit.get(name, 0.0) + pj

    def add_memory(self, name: str, pj: float) -> None:
        self.by_memory[name] = self.by_memory.get(name, 0.0) + pj

    def total_pj(self) -> float:
        return sum(self.by_unit.values()) + sum(self.by_memory.values())

    def total_nj(self) -> float:
        return self.total_pj() / 1000.0

    def report(self) -> str:
        lines = ["Energy report:"]
        if self.by_unit:
            lines.append("  Functional units:")
            for k, v in sorted(self.by_unit.items()):
                lines.append(f"    {k:<10} {v / 1000:>10.3f} nJ")
        if self.by_memory:
            lines.append("  Memory:")
            for k, v in sorted(self.by_memory.items()):
                lines.append(f"    {k:<10} {v / 1000:>10.3f} nJ")
        lines.append(f"  TOTAL    {self.total_nj():>10.3f} nJ")
        return "\n".join(lines)
