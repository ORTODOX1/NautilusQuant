import pytest

from nqx.assembler import assemble, AssemblyError
from nqx.isa import encode_instruction, decode_instruction, pack_program, unpack_program

ALL_MNEMONICS = [
    ("NOP", "NOP"),
    ("LDV V0, [0x0]", "LDV"),
    ("LDV_ASYNC V0, [0x100]", "LDV_ASYNC"),
    ("STV V0, [0x10000000]", "STV"),
    ("MOV V1, V2", "MOV"),
    ("GVNS V0, 0", "GVNS"),
    ("GVNS_INV V0, 1", "GVNS_INV"),
    ("POLAR V0", "POLAR"),
    ("IPOLAR V0", "IPOLAR"),
    ("QUANT V0, 3", "QUANT"),
    ("DEQUANT V0, 3", "DEQUANT"),
    ("QJL V0, V1", "QJL"),
    ("UNQJL V0, V1", "UNQJL"),
    ("PACK3 V0, V1", "PACK3"),
    ("UNPACK3 V0", "UNPACK3"),
    ("MXPACK V0, 0", "MXPACK"),
    ("MXPACK V0, MXFP4", "MXPACK"),
    ("MXUNPACK V0, 1", "MXUNPACK"),
    ("MXUNPACK V0, MXFP6", "MXUNPACK"),
    ("SUBBIT_ENC V0, 3, 1", "SUBBIT_ENC"),
    ("SUBBIT_DEC V0", "SUBBIT_DEC"),
    ("ATTN_DOT V0, V1", "ATTN_DOT"),
    ("BARRIER", "BARRIER"),
    ("HALT", "HALT"),
    ("ENC [0x0], [0x1000], 64", "ENC"),
    ("DEC [0x0], [0x1000], 64", "DEC"),
]


def test_all_mnemonics_roundtrip():
    for src_line, expected_op in ALL_MNEMONICS:
        prog = assemble(src_line)
        assert len(prog) == 1
        assert prog[0].opcode.name == expected_op

        word = encode_instruction(prog[0])
        decoded = decode_instruction(word)
        assert decoded.opcode == prog[0].opcode
        assert decoded.rd == prog[0].rd


def test_pack_unpack_roundtrip_all():
    lines = [line for line, _ in ALL_MNEMONICS]
    src = "\n".join(lines)
    prog = assemble(src)

    blob = pack_program(prog)
    back = unpack_program(blob)

    assert len(back) == len(prog)
    for a, b in zip(prog, back):
        assert a.opcode == b.opcode


def test_assembler_rejects_bad_layer():
    with pytest.raises(AssemblyError, match="layer"):
        assemble("GVNS V0, 5")


def test_assembler_rejects_bad_operand_count():
    with pytest.raises(AssemblyError):
        assemble("LDV V0")
    with pytest.raises(AssemblyError):
        assemble("POLAR V0, V1")
    with pytest.raises(AssemblyError):
        assemble("HALT V0")


def test_assembler_rejects_bad_format():
    with pytest.raises(AssemblyError):
        assemble("MXPACK V0, 99")


def test_assembler_rejects_ldv_async_bad_reg():
    with pytest.raises(AssemblyError):
        assemble("LDV_ASYNC V99, [0x0]")


def test_assembler_rejects_subbit_enc_bad_operands():
    with pytest.raises(AssemblyError, match="SUBBIT_ENC expects"):
        assemble("SUBBIT_ENC V0, 3")
    with pytest.raises(AssemblyError, match="bits must be"):
        assemble("SUBBIT_ENC V0, 0, 1")
    with pytest.raises(AssemblyError, match="bits must be"):
        assemble("SUBBIT_ENC V0, 3, 9")


def test_assembler_rejects_subbit_dec_with_operands():
    with pytest.raises(AssemblyError, match="SUBBIT_DEC expects"):
        assemble("SUBBIT_DEC V0, V1")


def test_assembler_rejects_attn_dot_bad_operands():
    with pytest.raises(AssemblyError, match="ATTN_DOT expects"):
        assemble("ATTN_DOT V0")
    with pytest.raises(AssemblyError, match="ATTN_DOT expects"):
        assemble("ATTN_DOT V0, V1, V2")


def test_assembler_comments_ignored():
    prog = assemble("LDV V0, [0x0]; this is a comment\nHALT")
    assert len(prog) == 2
    assert prog[0].opcode.name == "LDV"
    assert prog[1].opcode.name == "HALT"
