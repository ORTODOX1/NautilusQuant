"""SDK11: Boot ROM test — assemble boot.nqasm and run on emulator."""

from __future__ import annotations

import os

import numpy as np

from nqx.assembler import assemble
from nqx.cpu import NQXCore
from nqx.constants import NQXConfig

BOOT_DIR = os.path.dirname(os.path.abspath(__file__))
BOOT_NQASM = os.path.join(BOOT_DIR, "boot.nqasm")


def test_boot_assembles_and_runs():
    with open(BOOT_NQASM) as f:
        source = f.read()
    program = assemble(source)
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    result = core.execute_program(program)
    assert result["halted"], "Boot program did not HALT"


def test_boot_clear_vrf():
    # After boot, V0 should be zeros (loaded from HBM addr 0)
    with open(BOOT_NQASM) as f:
        source = f.read()
    program = assemble(source)
    # Pre-fill V0 with non-zero to verify clear
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    core.vrf.write(0, np.full(128, 99.0, dtype=np.float32))
    core.execute_program(program)
    v0 = core.vrf.read(0)
    assert np.all(v0 == 0.0), "V0 not cleared after boot"
