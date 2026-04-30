"""Disassemble NQ-ASM bytecode → human-readable assembly."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from nqx.isa import Instruction, Opcode, unpack_program
from nqx.mx_unit import MX_FORMAT_BY_INDEX


def _format_addr(addr: int) -> str:
    return f"[0x{addr:x}]"


def disassemble_one(ins: Instruction) -> str:
    op = ins.opcode

    if op in (Opcode.NOP, Opcode.BARRIER, Opcode.HALT):
        return op.name

    if op in (Opcode.LDV, Opcode.LDV_ASYNC):
        return f"{op.name} V{ins.rd}, {_format_addr(ins.imm)}"

    if op == Opcode.STV:
        return f"STV V{ins.rd}, {_format_addr(ins.imm)}"

    if op == Opcode.MOV:
        return f"MOV V{ins.rd}, V{ins.rs1}"

    if op in (Opcode.GVNS, Opcode.GVNS_INV):
        return f"{op.name} V{ins.rd}, {ins.rs1}"

    if op in (Opcode.POLAR, Opcode.IPOLAR, Opcode.UNPACK3, Opcode.SUBBIT_DEC):
        return f"{op.name} V{ins.rd}"

    if op in (Opcode.QUANT, Opcode.DEQUANT):
        return f"{op.name} V{ins.rd}, {ins.rs1}"

    if op in (Opcode.QJL, Opcode.UNQJL):
        alpha = ins.rs2 & 0xFF
        if alpha == 0x80:
            return f"{op.name} V{ins.rd}, V{ins.rs1}"
        return f"{op.name} V{ins.rd}, V{ins.rs1}, 0x{alpha:x}"

    if op == Opcode.PACK3:
        return f"PACK3 V{ins.rd}, V{ins.rs1}"

    if op in (Opcode.MXPACK, Opcode.MXUNPACK):
        fmt = (
            MX_FORMAT_BY_INDEX[ins.rs1] if 0 <= ins.rs1 < len(MX_FORMAT_BY_INDEX) else str(ins.rs1)
        )
        return f"{op.name} V{ins.rd}, {fmt}"

    if op == Opcode.SUBBIT_ENC:
        return f"SUBBIT_ENC V{ins.rd}, {ins.rs1}, {ins.rs2}"

    if op == Opcode.ATTN_DOT:
        return f"ATTN_DOT V{ins.rs1}, V{ins.rs2}"

    if op in (Opcode.ENC, Opcode.DEC):
        addr = ins.imm
        return f"{op.name} {_format_addr(addr)}, {_format_addr(addr)}, {ins.imm}"

    raise ValueError(f"unknown opcode {op}")


def disassemble(program: Iterable[Instruction]) -> str:
    return "\n".join(disassemble_one(ins) for ins in program)


def disassemble_bytes(blob: bytes) -> str:
    return disassemble(unpack_program(blob))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Disassemble NQ-ASM bytecode.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    blob = args.input.read_bytes()
    text = disassemble_bytes(blob)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
