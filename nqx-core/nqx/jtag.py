"""IEEE 1149.1 TAP controller state machine + simple debug commands."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class TAPState(enum.IntEnum):
    TEST_LOGIC_RESET = 0
    RUN_TEST_IDLE = 1
    SELECT_DR_SCAN = 2
    CAPTURE_DR = 3
    SHIFT_DR = 4
    EXIT1_DR = 5
    PAUSE_DR = 6
    EXIT2_DR = 7
    UPDATE_DR = 8
    SELECT_IR_SCAN = 9
    CAPTURE_IR = 10
    SHIFT_IR = 11
    EXIT1_IR = 12
    PAUSE_IR = 13
    EXIT2_IR = 14
    UPDATE_IR = 15


# next-state table indexed by [current_state][TMS bit]
_NEXT = {
    TAPState.TEST_LOGIC_RESET: (TAPState.RUN_TEST_IDLE, TAPState.TEST_LOGIC_RESET),
    TAPState.RUN_TEST_IDLE: (TAPState.RUN_TEST_IDLE, TAPState.SELECT_DR_SCAN),
    TAPState.SELECT_DR_SCAN: (TAPState.CAPTURE_DR, TAPState.SELECT_IR_SCAN),
    TAPState.CAPTURE_DR: (TAPState.SHIFT_DR, TAPState.EXIT1_DR),
    TAPState.SHIFT_DR: (TAPState.SHIFT_DR, TAPState.EXIT1_DR),
    TAPState.EXIT1_DR: (TAPState.PAUSE_DR, TAPState.UPDATE_DR),
    TAPState.PAUSE_DR: (TAPState.PAUSE_DR, TAPState.EXIT2_DR),
    TAPState.EXIT2_DR: (TAPState.SHIFT_DR, TAPState.UPDATE_DR),
    TAPState.UPDATE_DR: (TAPState.RUN_TEST_IDLE, TAPState.SELECT_DR_SCAN),
    TAPState.SELECT_IR_SCAN: (TAPState.CAPTURE_IR, TAPState.TEST_LOGIC_RESET),
    TAPState.CAPTURE_IR: (TAPState.SHIFT_IR, TAPState.EXIT1_IR),
    TAPState.SHIFT_IR: (TAPState.SHIFT_IR, TAPState.EXIT1_IR),
    TAPState.EXIT1_IR: (TAPState.PAUSE_IR, TAPState.UPDATE_IR),
    TAPState.PAUSE_IR: (TAPState.PAUSE_IR, TAPState.EXIT2_IR),
    TAPState.EXIT2_IR: (TAPState.SHIFT_IR, TAPState.UPDATE_IR),
    TAPState.UPDATE_IR: (TAPState.RUN_TEST_IDLE, TAPState.SELECT_DR_SCAN),
}


# IEEE 1149.1 standard instructions + NQX vendor IRs.
class IR(enum.IntEnum):
    BYPASS = 0xFF  # all-ones, JTAG-required
    IDCODE = 0x01
    SAMPLE_PRELOAD = 0x02
    EXTEST = 0x03
    READ_PC = 0x10
    READ_VRF = 0x11
    READ_SRF = 0x12
    READ_CSR = 0x13
    SINGLE_STEP = 0x20
    BREAKPOINT_SET = 0x21
    BREAKPOINT_CLEAR = 0x22


NQX_IDCODE = 0x4E_51_58_01  # 'NQX'+v01


@dataclass
class TAP:
    state: TAPState = TAPState.TEST_LOGIC_RESET
    ir: int = int(IR.IDCODE)
    dr: int = 0
    dr_width: int = 32
    visited: set = field(default_factory=set)

    def reset(self) -> None:
        self.state = TAPState.TEST_LOGIC_RESET
        self.ir = int(IR.IDCODE)
        self.dr = 0
        self.visited = {self.state}

    def clock(self, tms: int) -> TAPState:
        tms = 1 if tms else 0
        next_state = _NEXT[self.state][tms]
        self.state = next_state
        self.visited.add(self.state)
        return self.state

    def shift_in(self, bits: int, length: int) -> int:
        if self.state == TAPState.SHIFT_IR:
            self.ir = bits & ((1 << length) - 1)
            return self.ir
        if self.state == TAPState.SHIFT_DR:
            self.dr = bits & ((1 << length) - 1)
            return self.dr
        raise RuntimeError(f"shift_in not allowed in {self.state.name}")

    def update(self) -> None:
        if self.state == TAPState.UPDATE_IR:
            pass
        elif self.state == TAPState.UPDATE_DR:
            pass

    def go_to_test_logic_reset(self) -> None:
        for _ in range(5):
            self.clock(tms=1)


@dataclass
class JTAGDebugger:
    tap: TAP = field(default_factory=TAP)
    breakpoints: set = field(default_factory=set)
    pc: int = 0
    halted: bool = False
    vrf_snapshot: Optional[Dict[int, list]] = None
    srf_snapshot: Optional[Dict[int, list]] = None
    csr_snapshot: Optional[Dict[str, int]] = None

    def attach(self, core) -> None:
        self.vrf_snapshot = {
            i: core.vrf.read(i).reshape(-1).tolist() for i in range(core.config.n_vector_regs)
        }
        self.srf_snapshot = {
            i: core.srf.read(i).reshape(-1).tolist() for i in range(core.config.n_scalar_regs)
        }
        self.csr_snapshot = dict(core.perf.snapshot())

    def execute_ir(self, ir: int, payload: int = 0) -> int:
        if ir == int(IR.BYPASS):
            return 0
        if ir == int(IR.IDCODE):
            return NQX_IDCODE
        if ir == int(IR.READ_PC):
            return self.pc
        if ir == int(IR.READ_VRF):
            reg = payload & 0xF
            if self.vrf_snapshot is None:
                return 0
            data = self.vrf_snapshot.get(reg, [])
            return int(data[0] * 1e6) if data else 0
        if ir == int(IR.READ_SRF):
            reg = payload & 0x7
            if self.srf_snapshot is None:
                return 0
            data = self.srf_snapshot.get(reg, [])
            return int(data[0]) if data else 0
        if ir == int(IR.READ_CSR):
            if self.csr_snapshot is None:
                return 0
            keys = list(self.csr_snapshot.keys())
            idx = payload & 0x7
            return self.csr_snapshot.get(keys[idx], 0) if idx < len(keys) else 0
        if ir == int(IR.SINGLE_STEP):
            self.pc += 1
            return self.pc
        if ir == int(IR.BREAKPOINT_SET):
            self.breakpoints.add(payload & 0xFFFF)
            return 1
        if ir == int(IR.BREAKPOINT_CLEAR):
            self.breakpoints.discard(payload & 0xFFFF)
            return 1
        raise ValueError(f"unknown IR 0x{ir:x}")


def all_states_visited(tap: TAP) -> bool:
    return tap.visited == set(TAPState)


def walk_full_state_space() -> List[TAPState]:
    """Drive a TMS sequence that visits every state at least once."""
    tap = TAP()
    tap.reset()
    sequence: List[TAPState] = [tap.state]
    pattern = [
        # TLR -> RTI -> SDS -> CDR -> SDR -> E1DR -> PDR -> E2DR -> UDR
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        # UDR -> SDS -> SIS -> CIR -> SIR -> E1IR -> PIR -> E2IR -> UIR -> RTI
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        # final reset to TLR
        1,
        1,
        1,
        1,
        1,
    ]
    for tms in pattern:
        sequence.append(tap.clock(tms))
    return sequence
