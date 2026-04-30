"""S10: Prometheus /metrics — counters and histogram, hand-written format."""

from __future__ import annotations

import time
from typing import Dict, List


class Counter:
    def __init__(self, name: str, help: str, labels: Dict[str, str] | None = None):
        self.name = name
        self.help = help
        self._labels = labels or {}
        self._value = 0

    def inc(self, amount: int = 1) -> None:
        self._value += amount

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        label_str = ",".join(f'{k}="{v}"' for k, v in self._labels.items())
        if label_str:
            lines.append(f"{self.name}{{{label_str}}} {self._value}")
        else:
            lines.append(f"{self.name} {self._value}")
        return lines


class Histogram:
    def __init__(self, name: str, help: str, buckets: List[float] | None = None):
        self.name = name
        self.help = help
        self._buckets = sorted(buckets or [0.1, 0.5, 1.0, 5.0, 10.0])
        self._counts = {b: 0 for b in self._buckets}
        self._counts[float("inf")] = 0
        self._sum = 0.0
        self._n = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._n += 1
        for b in self._buckets:
            if value <= b:
                self._counts[b] += 1
        self._counts[float("inf")] += 1

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for b in self._buckets:
            lines.append(f'{self.name}_bucket{{le="{b}"}} {self._counts[b]}')
        lines.append(f"{self.name}_bucket{{le=\"+Inf\"}} {self._counts[float('inf')]}")
        lines.append(f"{self.name}_count {self._n}")
        lines.append(f"{self.name}_sum {self._sum:.2f}")
        return lines


# Global metrics
encode_total = Counter("nqx_encode_total", "Total encodes processed", {"format": "3bit"})
decode_total = Counter("nqx_decode_total", "Total decodes processed", {"format": "3bit"})
encode_latency = Histogram("nqx_encode_latency_ms", "Encode latency in ms")
errors_total = Counter("nqx_errors_total", "Total errors by type", {"type": "internal"})


def render_metrics() -> str:
    lines: List[str] = []
    for m in (encode_total, decode_total, errors_total, encode_latency):
        lines.extend(m.render())
    return "\n".join(lines) + "\n"
