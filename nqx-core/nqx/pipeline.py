"""Pipeline orchestration & cycle counting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CycleCounter:
    total: int = 0
    by_stage: dict = field(default_factory=dict)

    def tick(self, n: int = 1, stage: Optional[str] = None) -> None:
        self.total += n
        if stage is not None:
            self.by_stage[stage] = self.by_stage.get(stage, 0) + n

    def report(self) -> str:
        lines = [f"Cycles: {self.total}"]
        if self.by_stage:
            for k, v in sorted(self.by_stage.items()):
                lines.append(f"  {k:<14} {v:>8}")
        return "\n".join(lines)


@dataclass
class Pipeline:
    cycles: CycleCounter = field(default_factory=CycleCounter)
    pipeline_depth: int = 18

    def fused_encode_cycles(self, n_vectors: int) -> int:
        return n_vectors + self.pipeline_depth - 1

    def fused_decode_cycles(self, n_vectors: int) -> int:
        return n_vectors + self.pipeline_depth - 1
