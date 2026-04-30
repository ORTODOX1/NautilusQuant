import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nqx.assembler import assemble
from nqx.disassembler import disassemble, disassemble_bytes, disassemble_one
from nqx.isa import Instruction, Opcode, encode_instruction, pack_program

SAMPLE_SRC = """
LDV V0, [0x0]
LDV_ASYNC V1, [0x100]
BARRIER
NOP
MOV V2, V0
GVNS V0, 0
GVNS_INV V0, 1
POLAR V0
IPOLAR V0
QUANT V0, 3
DEQUANT V0, 3
QJL V0, V1, 0x80
UNQJL V0, V1
PACK3 V0, V1
UNPACK3 V2
MXPACK V1, MXFP4
MXUNPACK V1, MXFP4
SUBBIT_ENC V1, 3, 1
SUBBIT_DEC V1
ATTN_DOT V0, V2
STV V0, [0x10000000]
HALT
"""


def test_assemble_disassemble_assemble_roundtrip_bit_identical():
    program = assemble(SAMPLE_SRC)
    text = disassemble(program)
    again = assemble(text)
    assert len(program) == len(again)
    for a, b in zip(program, again):
        assert encode_instruction(a) == encode_instruction(b), (
            f"diff: {a} vs {b}"
        )


def test_disassemble_each_opcode_at_least_once():
    program = assemble(SAMPLE_SRC)
    text = disassemble(program)
    seen_mnemonics = {line.split()[0] for line in text.splitlines() if line.strip()}
    expected = {"LDV", "LDV_ASYNC", "BARRIER", "NOP", "MOV", "GVNS", "GVNS_INV",
                "POLAR", "IPOLAR", "QUANT", "DEQUANT", "QJL", "UNQJL", "PACK3",
                "UNPACK3", "MXPACK", "MXUNPACK", "SUBBIT_ENC", "SUBBIT_DEC",
                "ATTN_DOT", "STV", "HALT"}
    assert expected.issubset(seen_mnemonics)


def test_disassemble_bytes_via_pack_program():
    program = assemble("LDV V0, [0x42]\nGVNS V0, 1\nPOLAR V0\nHALT")
    blob = pack_program(program)
    text = disassemble_bytes(blob)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("LDV V0, [0x")
    assert lines[1] == "GVNS V0, 1"
    assert lines[2] == "POLAR V0"
    assert lines[3] == "HALT"


def test_disassemble_qjl_default_alpha_omits_argument():
    ins = Instruction(opcode=Opcode.QJL, rd=0, rs1=1, rs2=0x80)
    assert disassemble_one(ins) == "QJL V0, V1"


def test_disassemble_mxpack_uses_format_name():
    ins = Instruction(opcode=Opcode.MXPACK, rd=3, rs1=2)  # MXFP8
    assert disassemble_one(ins) == "MXPACK V3, MXFP8"


def test_cli_prints_disassembly(tmp_path):
    src = "LDV V0, [0x10]\nGVNS V0, 0\nHALT"
    program = assemble(src)
    blob = pack_program(program)
    bin_path = tmp_path / "p.bin"
    bin_path.write_bytes(blob)
    result = subprocess.run(
        [sys.executable, "-m", "nqx.disassembler", str(bin_path)],
        capture_output=True, text=True, check=True,
    )
    assert "LDV V0" in result.stdout
    assert "GVNS V0, 0" in result.stdout
    assert "HALT" in result.stdout
