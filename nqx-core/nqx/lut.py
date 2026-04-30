"""Golden angle ROM LUT.

Three layers of non-overlapping Givens rotation pair indices and
precomputed (cos, sin) values. Mirrors NautilusQuantPyTorch reference exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from nqx.constants import NQXConfig


@dataclass
class LayerLUT:
    name: str
    pairs: List[Tuple[int, int]]
    angles: List[float]
    cos: List[float] = field(init=False)
    sin: List[float] = field(init=False)

    def __post_init__(self) -> None:
        n = min(len(self.pairs), len(self.angles))
        self.pairs = self.pairs[:n]
        self.angles = self.angles[:n]
        self.cos = [math.cos(a) for a in self.angles]
        self.sin = [math.sin(a) for a in self.angles]
        self.i_idx = np.array([p[0] for p in self.pairs], dtype=np.int64)
        self.j_idx = np.array([p[1] for p in self.pairs], dtype=np.int64)
        self.cos_arr = np.array(self.cos, dtype=np.float32)
        self.sin_arr = np.array(self.sin, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pairs)


class GoldenAngleLUT:
    def __init__(self, config: NQXConfig):
        self.config = config
        self.layers: Dict[str, LayerLUT] = {}
        self._build()

    def _build(self) -> None:
        cfg = self.config
        ga = cfg.golden_angle()
        phi = cfg.phi
        dim = cfg.dim

        l1_pairs = [(2 * k, 2 * k + 1) for k in range(dim // 2)]
        l1_angles = [ga * (k + 1) for k in range(dim // 2)]
        self.layers["L1"] = LayerLUT("L1", l1_pairs, l1_angles)

        l2_pairs = [(2 * k + 1, 2 * k + 2) for k in range((dim - 1) // 2)]
        l2_angles = [ga * (k + 1) * phi for k in range((dim - 1) // 2)]
        self.layers["L2"] = LayerLUT("L2", l2_pairs, l2_angles)

        stride = cfg.l3_stride()
        l3_pairs: List[Tuple[int, int]] = []
        l3_angles: List[float] = []
        used: set[int] = set()
        for k in range(dim):
            i, j = k, (k + stride) % dim
            if i == j or i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            l3_pairs.append((i, j))
            l3_angles.append(ga * (k + 1) * phi * phi)
        self.layers["L3"] = LayerLUT("L3", l3_pairs, l3_angles)

    def layer(self, layer_id: int) -> LayerLUT:
        return self.layers[("L1", "L2", "L3")[layer_id]]

    def total_pairs(self) -> int:
        return sum(len(layer) for layer in self.layers.values())

    def rom_bytes(self) -> int:
        per_pair = 1 + 1 + 4 + 4
        return self.total_pairs() * per_pair

    def summary(self) -> str:
        lines = [f"GoldenAngleLUT(dim={self.config.dim}, phi={self.config.phi:.6f})"]
        for name, layer in self.layers.items():
            lines.append(
                f"  {name}: {len(layer)} pairs, "
                f"first_angle={math.degrees(layer.angles[0]):.3f}°"
            )
        lines.append(f"  ROM size: {self.rom_bytes()} bytes")
        return "\n".join(lines)
