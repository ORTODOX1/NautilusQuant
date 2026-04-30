import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pytest

from nqx.assembler import assemble
from nqx.constants import NQXConfig
from nqx.counters import COUNTER_NAMES, MMIO_ADDRESSES, MMIO_BASE, PerfCounters
from nqx.cpu import NQXCore


def test_counter_addresses_are_unique_and_in_region():
    addrs = list(MMIO_ADDRESSES.values())
    assert len(addrs) == len(set(addrs))
    for a in addrs:
        assert MMIO_BASE <= a < MMIO_BASE + 4 * len(COUNTER_NAMES)


def test_perf_counters_start_at_zero():
    p = PerfCounters()
    for name in COUNTER_NAMES:
        assert p.read(name) == 0


def test_unknown_counter_raises():
    p = PerfCounters()
    with pytest.raises(KeyError):
        p.add("not_a_counter")


def test_mmio_read_returns_value():
    p = PerfCounters()
    p.add("cycle_count", 42)
    assert p.read_mmio(MMIO_ADDRESSES["cycle_count"]) == 42


def test_run_program_populates_counters():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(0)
    vec = rng.standard_normal((4, 128)).astype(np.float32)
    core.load_vectors_to_hbm(0, vec)
    src = """
    LDV V0, [0x0]
    GVNS V0, 0
    GVNS V0, 1
    GVNS V0, 2
    POLAR V0
    QUANT V0, 3
    HALT
    """
    prog = assemble(src)
    prog[0].extra["count"] = 4
    core.execute_program(prog)
    assert core.perf.read("cycle_count") == core.cycles.total
    assert core.perf.read("gu_busy_cycles") > 0
    assert core.perf.read("pu_busy_cycles") > 0
    assert core.perf.read("qu_busy_cycles") > 0
    assert core.perf.read("dma_in_bytes") == 4 * 128 * 2
    assert core.perf.read("prng_cycles_baseline") == 4 * 128 * 128


def test_counters_mirror_into_scalar_register_file():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    core.perf.add("cycle_count", 7)
    core.perf.add("gu_busy_cycles", 9)
    core.perf.write_to_srf(core.srf)
    assert core.srf.read(0)[0] == 7.0
    assert core.srf.read(2)[0] == 9.0
