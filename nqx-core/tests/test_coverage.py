import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nqx.assembler import assemble
from nqx.coverage import Coverage, trace_program, write_report


def test_trace_records_opcode_counts():
    src = """
    LDV V0, [0x0]
    GVNS V0, 0
    POLAR V0
    QUANT V0, 3
    HALT
    """
    cov = trace_program(assemble(src))
    assert cov.opcode_counts["LDV"] == 1
    assert cov.opcode_counts["GVNS"] == 1
    assert cov.opcode_counts["POLAR"] == 1
    assert cov.opcode_counts["QUANT"] == 1
    assert cov.opcode_counts["HALT"] == 1


def test_trace_records_pairs():
    src = "LDV V0, [0x0]\nGVNS V0, 1\nPOLAR V0\nHALT"
    cov = trace_program(assemble(src))
    assert cov.pair_counts.get(("LDV", "GVNS")) == 1
    assert cov.pair_counts.get(("GVNS", "POLAR")) == 1


def test_coverage_merge_accumulates():
    a = Coverage()
    a.opcode_counts = {"LDV": 1}
    a.n_programs = 1
    b = Coverage()
    b.opcode_counts = {"LDV": 2, "GVNS": 1}
    b.n_programs = 1
    a.merge(b)
    assert a.opcode_counts["LDV"] == 3
    assert a.opcode_counts["GVNS"] == 1
    assert a.n_programs == 2


def test_full_coverage_after_running_full_pipeline():
    src = """
    LDV V0, [0x0]
    LDV_ASYNC V1, [0x100]
    BARRIER
    NOP
    MOV V2, V0
    GVNS V0, 0
    GVNS_INV V0, 0
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
    POLAR V1
    SUBBIT_ENC V1, 3, 1
    SUBBIT_DEC V1
    POLAR V2
    POLAR V0
    ATTN_DOT V0, V2
    STV V0, [0x10000000]
    ENC [0x0], [0x10000000], 4
    DEC [0x0], [0x10000000], 4
    HALT
    """
    cov = trace_program(assemble(src))
    assert cov.opcode_coverage_fraction() == 1.0


def test_write_report_creates_file(tmp_path):
    cov = Coverage()
    cov.opcode_counts["LDV"] = 1
    cov.n_programs = 1
    cov.n_instructions = 1
    md = write_report(cov, tmp_path)
    assert md.exists()
    assert "Coverage report" in md.read_text()
