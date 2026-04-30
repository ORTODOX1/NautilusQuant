import numpy as np

from nqx.assembler import assemble
from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.isa import Instruction, Opcode, decode_instruction, encode_instruction


def _make_core_with_data(n_vec=4096, dim=128):
    cfg = NQXConfig(dim=dim)
    core = NQXCore(cfg)
    rng = np.random.default_rng(0)
    vec = rng.standard_normal((n_vec, dim)).astype(np.float32)
    core.load_vectors_to_hbm(0, vec)
    return core, n_vec


def test_isa_ldv_async_roundtrip():
    a = Instruction(opcode=Opcode.LDV_ASYNC, rd=3, imm=0x1234)
    b = decode_instruction(encode_instruction(a))
    assert b.opcode == Opcode.LDV_ASYNC and b.rd == 3 and b.imm == 0x1234


def test_assembler_ldv_async():
    src = "LDV_ASYNC V0, [0x100]\nHALT"
    prog = assemble(src)
    assert prog[0].opcode == Opcode.LDV_ASYNC
    assert prog[0].rd == 0
    assert prog[0].extra["addr"] == 0x100


def test_ldv_async_loads_data_immediately():
    core, n = _make_core_with_data()
    src = """
    LDV_ASYNC V0, [0x0]
    BARRIER
    HALT
    """
    prog = assemble(src)
    prog[0].extra["count"] = n
    core.execute_program(prog)
    assert core.vrf.read(0).shape == (n, 128)


def test_async_dma_overlaps_with_compute():
    core, n = _make_core_with_data(n_vec=512)
    src = """
    LDV_ASYNC V0, [0x0]
    GVNS V0, 0
    GVNS V0, 1
    GVNS V0, 2
    BARRIER
    HALT
    """
    prog = assemble(src)
    prog[0].extra["count"] = n
    core.execute_program(prog)
    overlap_total = core.cycles.total

    core2 = NQXCore(NQXConfig(dim=128))
    rng = np.random.default_rng(0)
    vec = rng.standard_normal((n, 128)).astype(np.float32)
    core2.load_vectors_to_hbm(0, vec)
    src2 = """
    LDV V0, [0x0]
    GVNS V0, 0
    GVNS V0, 1
    GVNS V0, 2
    HALT
    """
    prog2 = assemble(src2)
    prog2[0].extra["count"] = n
    core2.execute_program(prog2)
    sync_total = core2.cycles.total

    assert (
        overlap_total < sync_total
    ), f"async {overlap_total} should be less than sync {sync_total}"


def test_barrier_waits_when_compute_shorter_than_dma():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(1)
    n = 4096
    vec = rng.standard_normal((n, 128)).astype(np.float32)
    core.load_vectors_to_hbm(0, vec)
    src = """
    LDV_ASYNC V0, [0x0]
    GVNS V0, 0
    BARRIER
    HALT
    """
    prog = assemble(src)
    prog[0].extra["count"] = n
    core.execute_program(prog)
    bytes_moved = n * 128 * 2
    dma_cycles = max(1, int(cfg.cycles_dma_per_byte * bytes_moved))
    assert core.cycles.total >= dma_cycles
    assert core.cycles.by_stage.get("DMA_wait", 0) > 0


def test_barrier_no_wait_when_compute_long():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(2)
    n = 1
    vec = rng.standard_normal((n, 128)).astype(np.float32)
    core.load_vectors_to_hbm(0, vec)
    bytes_moved = n * 128 * 2
    dma_cycles = max(1, int(cfg.cycles_dma_per_byte * bytes_moved))
    n_quants = dma_cycles + 4

    instrs = ["LDV_ASYNC V0, [0x0]"]
    for _ in range(n_quants):
        instrs.append("QUANT V0, 3")
    instrs.append("BARRIER")
    instrs.append("HALT")
    src = "\n".join(instrs)
    prog = assemble(src)
    prog[0].extra["count"] = n
    core.execute_program(prog)
    assert core.cycles.by_stage.get("DMA_wait", 0) == 0
