import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pytest

from nqx.assembler import assemble
from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.jtag import (
    IR,
    JTAGDebugger,
    NQX_IDCODE,
    TAP,
    TAPState,
    walk_full_state_space,
)


def test_reset_returns_to_test_logic_reset():
    tap = TAP()
    tap.state = TAPState.RUN_TEST_IDLE
    tap.go_to_test_logic_reset()
    assert tap.state == TAPState.TEST_LOGIC_RESET


def test_walk_visits_all_16_states():
    seq = walk_full_state_space()
    visited = set(seq)
    assert visited == set(TAPState), f"missing: {set(TAPState) - visited}"


def test_idcode_returns_nqx_signature():
    dbg = JTAGDebugger()
    dbg.attach_dummy = lambda: None
    assert dbg.execute_ir(int(IR.IDCODE)) == NQX_IDCODE


def test_bypass_register_zero():
    dbg = JTAGDebugger()
    assert dbg.execute_ir(int(IR.BYPASS)) == 0


def test_breakpoint_set_and_clear():
    dbg = JTAGDebugger()
    dbg.execute_ir(int(IR.BREAKPOINT_SET), payload=0x100)
    assert 0x100 in dbg.breakpoints
    dbg.execute_ir(int(IR.BREAKPOINT_CLEAR), payload=0x100)
    assert 0x100 not in dbg.breakpoints


def test_single_step_advances_pc():
    dbg = JTAGDebugger()
    pc_before = dbg.pc
    pc_after = dbg.execute_ir(int(IR.SINGLE_STEP))
    assert pc_after == pc_before + 1


def test_attach_reads_vrf_and_csr():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    core.execute_program(assemble("LDV V0, [0x0]\nGVNS V0, 0\nHALT"))
    dbg = JTAGDebugger()
    dbg.attach(core)
    assert dbg.csr_snapshot is not None
    assert "cycle_count" in dbg.csr_snapshot
    assert dbg.execute_ir(int(IR.READ_CSR), payload=0) == dbg.csr_snapshot["cycle_count"]


def test_unknown_ir_raises():
    dbg = JTAGDebugger()
    with pytest.raises(ValueError):
        dbg.execute_ir(0x77)


def test_shift_in_outside_shift_state_raises():
    tap = TAP()
    tap.reset()
    with pytest.raises(RuntimeError):
        tap.shift_in(0xCAFE, 16)


def test_clock_with_invalid_states_does_not_escape_table():
    tap = TAP()
    tap.reset()
    for _ in range(50):
        tap.clock(tms=int(_ % 2))
    assert tap.state in TAPState
