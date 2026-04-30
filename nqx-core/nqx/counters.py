"""HW-style performance counters mirrored to the scalar register file."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


COUNTER_NAMES: List[str] = [
    "cycle_count",
    "stall_cycles",
    "gu_busy_cycles",
    "pu_busy_cycles",
    "qu_busy_cycles",
    "dma_in_bytes",
    "dma_out_bytes",
    "prng_cycles_baseline",
]

MMIO_BASE = 0x3000_0000
MMIO_STRIDE = 4
MMIO_ADDRESSES: Dict[str, int] = {
    name: MMIO_BASE + i * MMIO_STRIDE for i, name in enumerate(COUNTER_NAMES)
}


@dataclass
class PerfCounters:
    counts: Dict[str, int] = field(default_factory=lambda: {n: 0 for n in COUNTER_NAMES})

    def reset(self) -> None:
        for k in self.counts:
            self.counts[k] = 0

    def add(self, name: str, value: int = 1) -> None:
        if name not in self.counts:
            raise KeyError(f"unknown counter {name!r}")
        self.counts[name] += int(value)

    def read(self, name: str) -> int:
        return self.counts[name]

    def read_mmio(self, address: int) -> int:
        for name, addr in MMIO_ADDRESSES.items():
            if addr == address:
                return self.counts[name]
        raise KeyError(f"address 0x{address:x} not in performance counter region")

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counts)

    def write_to_srf(self, srf) -> None:
        for i, name in enumerate(COUNTER_NAMES):
            if i < srf.config.n_scalar_regs:
                srf.write(i, float(self.counts[name]))

    def report(self) -> str:
        lines = ["Performance counters:"]
        for name in COUNTER_NAMES:
            addr = MMIO_ADDRESSES[name]
            lines.append(f"  {name:<24} 0x{addr:08x}  {self.counts[name]:>12}")
        return "\n".join(lines)
