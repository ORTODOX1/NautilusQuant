import pytest

from nqx.assembler import AssemblyError, assemble
from nqx.isa import (
    Instruction,
    Opcode,
    decode_instruction,
    encode_instruction,
    pack_program,
    unpack_program,
)


def test_encode_decode_roundtrip_r_form():
    ins = Instruction(opcode=Opcode.GVNS, rd=3, rs1=1, rs2=0)
    word = encode_instruction(ins)
    back = decode_instruction(word)
    assert back.opcode == Opcode.GVNS
    assert back.rd == 3
    assert back.rs1 == 1


def test_encode_decode_roundtrip_i_form():
    ins = Instruction(opcode=Opcode.LDV, rd=5, imm=0xCAFE)
    word = encode_instruction(ins)
    back = decode_instruction(word)
    assert back.opcode == Opcode.LDV
    assert back.rd == 5
    assert back.imm == 0xCAFE


def test_assembler_simple_encode_program():
    src = """
    LDV V0, [0x0]
    GVNS V0, 0
    GVNS V0, 1
    GVNS V0, 2
    POLAR V0
    QUANT V0, 3
    PACK3 V0, V1
    STV V0, [0x10000000]
    HALT
    """
    prog = assemble(src)
    assert prog[0].opcode == Opcode.LDV
    assert prog[1].opcode == Opcode.GVNS
    assert prog[1].rs1 == 0
    assert prog[3].opcode == Opcode.GVNS
    assert prog[3].rs1 == 2
    assert prog[-1].opcode == Opcode.HALT


def test_assembler_rejects_bad_register():
    with pytest.raises(AssemblyError):
        assemble("LDV V99, [0x0]")


def test_assembler_rejects_unknown_mnemonic():
    with pytest.raises(AssemblyError):
        assemble("FROBNICATE V0")


def test_pack_unpack_program():
    src = "LDV V0, [0x0]\nGVNS V0, 1\nHALT"
    prog = assemble(src)
    blob = pack_program(prog)
    back = unpack_program(blob)
    assert [i.opcode for i in back] == [i.opcode for i in prog]


def test_qjl_alpha_default():
    prog = assemble("QJL V0, V1")
    assert prog[0].rs2 == 0x80
