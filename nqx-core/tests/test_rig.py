import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from tools.rig import generate_program, run_iterations


def test_generate_program_starts_with_ldv_ends_with_halt():
    rng = np.random.default_rng(1)
    src, _ = generate_program(rng, length=20)
    lines = [line for line in src.splitlines() if line.strip()]
    assert lines[0].startswith("LDV ")
    assert lines[-1] == "HALT"


def test_rig_50_iterations_no_crashes():
    report = run_iterations(50, length_min=10, length_max=40, dim=128, seed=2026)
    assert report["crash_count"] == 0, report["crashes"][:1]
    assert report["instructions_total"] >= 50 * 10


def test_rig_covers_main_opcodes_in_50_runs():
    report = run_iterations(50, length_min=20, length_max=40, dim=128, seed=7)
    cov = report["coverage"]
    for op in ("LDV", "GVNS", "POLAR", "QUANT", "PACK3"):
        assert cov.opcode_counts.get(op, 0) > 0, f"opcode {op} never generated in 50 runs"


def test_rig_reaches_full_opcode_coverage_in_100_runs():
    report = run_iterations(100, length_min=30, length_max=80, dim=128, seed=11)
    assert report["crash_count"] == 0
    assert report["coverage"].opcode_coverage_fraction() == 1.0
