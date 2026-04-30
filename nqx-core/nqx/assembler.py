"""NQ-ASM assembler.

Syntax:
  ; comment
  LDV V0, [0x0]
  GVNS V0, 0
  POLAR V0
  QUANT V0, 3
  QJL V0, V1, 0x80
  PACK3 V0, V1
  STV V0, [0x10000000]
  HALT
"""

from __future__ import annotations

import re
from typing import List

from nqx.isa import Opcode, Instruction
from nqx.mx_unit import MX_FORMAT_BY_INDEX


class AssemblyError(Exception):
    pass


_REG_RE = re.compile(r"^V(\d+)$", re.IGNORECASE)
_SREG_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)
_ADDR_RE = re.compile(r"^\[\s*(0x[0-9a-fA-F]+|\d+)\s*\]$")
_INT_RE = re.compile(r"^(0x[0-9a-fA-F]+|\d+)$")


def _parse_reg(token: str, line_no: int) -> int:
    m = _REG_RE.match(token)
    if not m:
        raise AssemblyError(f"line {line_no}: expected vector register V0..V15, got {token!r}")
    n = int(m.group(1))
    if not (0 <= n < 16):
        raise AssemblyError(f"line {line_no}: V{n} out of range")
    return n


def _parse_int(token: str, line_no: int) -> int:
    m = _INT_RE.match(token)
    if not m:
        raise AssemblyError(f"line {line_no}: expected integer, got {token!r}")
    s = m.group(1)
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _parse_addr(token: str, line_no: int) -> int:
    m = _ADDR_RE.match(token)
    if not m:
        raise AssemblyError(f"line {line_no}: expected [addr], got {token!r}")
    s = m.group(1)
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _split_operands(rest: str) -> List[str]:
    if not rest.strip():
        return []
    return [t.strip() for t in rest.split(",")]


def assemble(source: str) -> List[Instruction]:
    program: List[Instruction] = []
    for line_no, raw in enumerate(source.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        mnem = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""
        operands = _split_operands(rest)

        try:
            op = Opcode[mnem]
        except KeyError:
            raise AssemblyError(f"line {line_no}: unknown mnemonic {mnem!r}")

        ins = _assemble_one(op, operands, line_no)
        program.append(ins)
    return program


def _assemble_one(op: Opcode, operands: List[str], line_no: int) -> Instruction:
    if op in (Opcode.NOP, Opcode.BARRIER, Opcode.HALT):
        if operands:
            raise AssemblyError(f"line {line_no}: {op.name} takes no operands")
        return Instruction(opcode=op)

    if op in (Opcode.LDV, Opcode.LDV_ASYNC):
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: {op.name} expects vd, [addr]")
        rd = _parse_reg(operands[0], line_no)
        addr = _parse_addr(operands[1], line_no)
        return Instruction(opcode=op, rd=rd, imm=addr & 0xFFFF, extra={"addr": addr})

    if op == Opcode.STV:
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: STV expects vs, [addr]")
        rd = _parse_reg(operands[0], line_no)
        addr = _parse_addr(operands[1], line_no)
        return Instruction(opcode=op, rd=rd, imm=addr & 0xFFFF, extra={"addr": addr})

    if op == Opcode.MOV:
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: MOV expects vd, vs")
        return Instruction(
            opcode=op, rd=_parse_reg(operands[0], line_no), rs1=_parse_reg(operands[1], line_no)
        )

    if op in (Opcode.GVNS, Opcode.GVNS_INV):
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: {op.name} expects vd, layer")
        rd = _parse_reg(operands[0], line_no)
        layer = _parse_int(operands[1], line_no)
        if not (0 <= layer <= 2):
            raise AssemblyError(f"line {line_no}: layer must be 0..2")
        return Instruction(opcode=op, rd=rd, rs1=layer)

    if op in (Opcode.POLAR, Opcode.IPOLAR, Opcode.UNPACK3):
        if len(operands) != 1:
            raise AssemblyError(f"line {line_no}: {op.name} expects vd")
        return Instruction(opcode=op, rd=_parse_reg(operands[0], line_no))

    if op == Opcode.QUANT:
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: QUANT expects vd, bits")
        rd = _parse_reg(operands[0], line_no)
        bits = _parse_int(operands[1], line_no)
        return Instruction(opcode=op, rd=rd, rs1=bits)

    if op == Opcode.DEQUANT:
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: DEQUANT expects vd, bits")
        rd = _parse_reg(operands[0], line_no)
        bits = _parse_int(operands[1], line_no)
        return Instruction(opcode=op, rd=rd, rs1=bits)

    if op in (Opcode.QJL, Opcode.UNQJL):
        if len(operands) not in (2, 3):
            raise AssemblyError(f"line {line_no}: {op.name} expects vd_orig, vd_q[, alpha]")
        rd = _parse_reg(operands[0], line_no)
        rs1 = _parse_reg(operands[1], line_no)
        alpha_q = 0x80
        if len(operands) == 3:
            alpha_q = _parse_int(operands[2], line_no)
        return Instruction(opcode=op, rd=rd, rs1=rs1, rs2=alpha_q & 0xFF)

    if op == Opcode.PACK3:
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: PACK3 expects vd, vsign")
        return Instruction(
            opcode=op, rd=_parse_reg(operands[0], line_no), rs1=_parse_reg(operands[1], line_no)
        )

    if op == Opcode.SUBBIT_ENC:
        if len(operands) != 3:
            raise AssemblyError(f"line {line_no}: SUBBIT_ENC expects vd, r_bits, theta_bits")
        rd = _parse_reg(operands[0], line_no)
        r_bits = _parse_int(operands[1], line_no)
        a_bits = _parse_int(operands[2], line_no)
        if not (1 <= r_bits <= 8) or not (1 <= a_bits <= 8):
            raise AssemblyError(f"line {line_no}: bits must be 1..8")
        return Instruction(opcode=op, rd=rd, rs1=r_bits, rs2=a_bits)

    if op == Opcode.SUBBIT_DEC:
        if len(operands) != 1:
            raise AssemblyError(f"line {line_no}: SUBBIT_DEC expects vd")
        return Instruction(opcode=op, rd=_parse_reg(operands[0], line_no))

    if op == Opcode.ATTN_DOT:
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: ATTN_DOT expects vq, vk")
        return Instruction(
            opcode=op, rs1=_parse_reg(operands[0], line_no), rs2=_parse_reg(operands[1], line_no)
        )

    if op in (Opcode.MXPACK, Opcode.MXUNPACK):
        if len(operands) != 2:
            raise AssemblyError(f"line {line_no}: {op.name} expects vd, fmt")
        rd = _parse_reg(operands[0], line_no)
        fmt_token = operands[1].upper()
        if fmt_token in MX_FORMAT_BY_INDEX:
            fmt_idx = MX_FORMAT_BY_INDEX.index(fmt_token)
        else:
            fmt_idx = _parse_int(operands[1], line_no)
        if not (0 <= fmt_idx < len(MX_FORMAT_BY_INDEX)):
            raise AssemblyError(f"line {line_no}: fmt index {fmt_idx} out of range")
        return Instruction(opcode=op, rd=rd, rs1=fmt_idx)

    if op in (Opcode.ENC, Opcode.DEC):
        if len(operands) != 3:
            raise AssemblyError(f"line {line_no}: {op.name} expects [src], [dst], cnt")
        src = _parse_addr(operands[0], line_no)
        dst = _parse_addr(operands[1], line_no)
        cnt = _parse_int(operands[2], line_no)
        return Instruction(opcode=op, imm=cnt & 0xFFFF, extra={"src": src, "dst": dst, "cnt": cnt})

    raise AssemblyError(f"line {line_no}: opcode {op.name} not implemented in assembler")
