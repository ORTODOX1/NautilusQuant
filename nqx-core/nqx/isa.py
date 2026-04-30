"""NQ-ISA: opcodes, encoding, decoding."""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field


class Opcode(enum.IntEnum):
    NOP = 0x00
    LDV = 0x01
    STV = 0x02
    MOV = 0x03
    LDV_ASYNC = 0x04

    GVNS = 0x10
    GVNS_INV = 0x11

    POLAR = 0x20
    IPOLAR = 0x21

    QUANT = 0x30
    DEQUANT = 0x31

    QJL = 0x40
    UNQJL = 0x41

    PACK3 = 0x50
    UNPACK3 = 0x51
    MXPACK = 0x52
    MXUNPACK = 0x53
    SUBBIT_ENC = 0x54
    SUBBIT_DEC = 0x55

    ENC = 0x60
    DEC = 0x61

    BARRIER = 0x70
    HALT = 0x7F

    ATTN_DOT = 0x80


_FORM_R = {
    Opcode.NOP,
    Opcode.MOV,
    Opcode.GVNS,
    Opcode.GVNS_INV,
    Opcode.POLAR,
    Opcode.IPOLAR,
    Opcode.QUANT,
    Opcode.DEQUANT,
    Opcode.QJL,
    Opcode.UNQJL,
    Opcode.PACK3,
    Opcode.UNPACK3,
    Opcode.MXPACK,
    Opcode.MXUNPACK,
    Opcode.SUBBIT_ENC,
    Opcode.SUBBIT_DEC,
    Opcode.ATTN_DOT,
    Opcode.BARRIER,
    Opcode.HALT,
}

_FORM_I = {Opcode.LDV, Opcode.STV, Opcode.ENC, Opcode.DEC, Opcode.LDV_ASYNC}


@dataclass
class Instruction:
    opcode: Opcode
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        op = self.opcode.name
        return (
            f"{op:<8} rd={self.rd} rs1={self.rs1} rs2={self.rs2} "
            f"imm={self.imm} extra={self.extra}"
        )


def encode_instruction(ins: Instruction) -> int:
    op = int(ins.opcode) & 0xFF
    if ins.opcode in _FORM_I:
        word = (op << 24) | ((ins.rd & 0xFF) << 16) | (ins.imm & 0xFFFF)
    else:
        word = (op << 24) | ((ins.rd & 0xFF) << 16) | ((ins.rs1 & 0xFF) << 8) | (ins.rs2 & 0xFF)
    return word & 0xFFFFFFFF


def decode_instruction(word: int) -> Instruction:
    op_int = (word >> 24) & 0xFF
    try:
        op = Opcode(op_int)
    except ValueError as e:
        raise ValueError(f"unknown opcode 0x{op_int:02X}") from e

    if op in _FORM_I:
        rd = (word >> 16) & 0xFF
        imm = word & 0xFFFF
        return Instruction(opcode=op, rd=rd, imm=imm)
    rd = (word >> 16) & 0xFF
    rs1 = (word >> 8) & 0xFF
    rs2 = word & 0xFF
    return Instruction(opcode=op, rd=rd, rs1=rs1, rs2=rs2)


def pack_program(instructions) -> bytes:
    return b"".join(struct.pack("<I", encode_instruction(i)) for i in instructions)


def unpack_program(blob: bytes) -> list:
    out = []
    for off in range(0, len(blob), 4):
        word = struct.unpack("<I", blob[off : off + 4])[0]
        out.append(decode_instruction(word))
    return out
