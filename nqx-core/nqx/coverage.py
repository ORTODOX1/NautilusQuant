"""Coverage tracker for NQ-ASM execution. Records opcodes, op-pair sequences,
and register reads/writes per program."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from nqx.isa import Instruction, Opcode


@dataclass
class Coverage:
    opcode_counts: Dict[str, int] = field(default_factory=dict)
    pair_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    reg_reads: Dict[int, int] = field(default_factory=dict)
    reg_writes: Dict[int, int] = field(default_factory=dict)
    n_programs: int = 0
    n_instructions: int = 0

    def merge(self, other: "Coverage") -> None:
        for k, v in other.opcode_counts.items():
            self.opcode_counts[k] = self.opcode_counts.get(k, 0) + v
        for k, v in other.pair_counts.items():
            self.pair_counts[k] = self.pair_counts.get(k, 0) + v
        for k, v in other.reg_reads.items():
            self.reg_reads[k] = self.reg_reads.get(k, 0) + v
        for k, v in other.reg_writes.items():
            self.reg_writes[k] = self.reg_writes.get(k, 0) + v
        self.n_programs += other.n_programs
        self.n_instructions += other.n_instructions

    def opcode_coverage_fraction(self) -> float:
        all_opcodes = {op.name for op in Opcode}
        seen = {k for k, v in self.opcode_counts.items() if v > 0}
        return len(seen & all_opcodes) / len(all_opcodes)

    def missing_opcodes(self) -> List[str]:
        all_opcodes = {op.name for op in Opcode}
        seen = {k for k, v in self.opcode_counts.items() if v > 0}
        return sorted(all_opcodes - seen)


_WRITE_OPS = {
    Opcode.LDV,
    Opcode.LDV_ASYNC,
    Opcode.MOV,
    Opcode.GVNS,
    Opcode.GVNS_INV,
    Opcode.POLAR,
    Opcode.IPOLAR,
    Opcode.QUANT,
    Opcode.DEQUANT,
    Opcode.QJL,
    Opcode.UNPACK3,
    Opcode.MXPACK,
    Opcode.MXUNPACK,
    Opcode.SUBBIT_ENC,
    Opcode.SUBBIT_DEC,
}

_READ_OPS = {
    Opcode.STV: ("rd",),
    Opcode.MOV: ("rs1",),
    Opcode.GVNS: ("rd",),
    Opcode.GVNS_INV: ("rd",),
    Opcode.POLAR: ("rd",),
    Opcode.IPOLAR: ("rd",),
    Opcode.QUANT: ("rd",),
    Opcode.QJL: ("rd", "rs1"),
    Opcode.PACK3: ("rd", "rs1"),
    Opcode.MXPACK: ("rd",),
    Opcode.SUBBIT_ENC: ("rd",),
    Opcode.ATTN_DOT: ("rs1", "rs2"),
}


def trace_program(program: Iterable[Instruction]) -> Coverage:
    cov = Coverage()
    prev_op: Optional[str] = None
    program = list(program)
    for ins in program:
        op_name = ins.opcode.name
        cov.opcode_counts[op_name] = cov.opcode_counts.get(op_name, 0) + 1
        cov.n_instructions += 1
        if prev_op is not None:
            key = (prev_op, op_name)
            cov.pair_counts[key] = cov.pair_counts.get(key, 0) + 1
        prev_op = op_name
        for field_name in _READ_OPS.get(ins.opcode, ()):
            reg = getattr(ins, field_name)
            cov.reg_reads[reg] = cov.reg_reads.get(reg, 0) + 1
        if ins.opcode in _WRITE_OPS:
            cov.reg_writes[ins.rd] = cov.reg_writes.get(ins.rd, 0) + 1
    cov.n_programs = 1
    return cov


def render_markdown(cov: Coverage, n_runs: Optional[int] = None) -> str:
    lines = ["# Coverage report", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"- Programs traced: {cov.n_programs}")
    lines.append(f"- Instructions executed: {cov.n_instructions:,}")
    lines.append(f"- Opcode coverage: **{cov.opcode_coverage_fraction() * 100:.1f}%**")
    lines.append("")
    lines.append("## Opcode counts")
    lines.append("")
    lines.append("| Opcode | Count |")
    lines.append("|---|---:|")
    for op in Opcode:
        c = cov.opcode_counts.get(op.name, 0)
        marker = "" if c > 0 else "  ⚠ uncovered"
        lines.append(f"| {op.name}{marker} | {c} |")
    lines.append("")
    if cov.missing_opcodes():
        lines.append(f"**Missing opcodes**: {', '.join(cov.missing_opcodes())}")
    else:
        lines.append("**All opcodes covered.**")
    lines.append("")
    lines.append("## Top opcode pairs")
    lines.append("")
    lines.append("| previous → next | count |")
    lines.append("|---|---:|")
    pairs = sorted(cov.pair_counts.items(), key=lambda x: -x[1])[:15]
    for (a, b), c in pairs:
        lines.append(f"| `{a}` → `{b}` | {c} |")
    lines.append("")
    lines.append("## Register usage")
    lines.append("")
    lines.append("| Reg | reads | writes |")
    lines.append("|---|---:|---:|")
    for r in range(16):
        lines.append(f"| V{r} | {cov.reg_reads.get(r, 0)} | {cov.reg_writes.get(r, 0)} |")
    return "\n".join(lines) + "\n"


def write_report(cov: Coverage, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    md = out_dir / f"coverage-{ts}.md"
    md.write_text(render_markdown(cov))
    js = out_dir / f"coverage-{ts}.json"
    js.write_text(
        json.dumps(
            {
                "n_programs": cov.n_programs,
                "n_instructions": cov.n_instructions,
                "opcode_counts": cov.opcode_counts,
                "pair_counts": {f"{a}->{b}": v for (a, b), v in cov.pair_counts.items()},
                "reg_reads": cov.reg_reads,
                "reg_writes": cov.reg_writes,
                "opcode_coverage": cov.opcode_coverage_fraction(),
            },
            indent=2,
        )
    )
    return md
