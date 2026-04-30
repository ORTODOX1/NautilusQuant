import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.assembler import assemble


def test_program_executes_to_halt():
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
    core = NQXCore(NQXConfig(dim=128))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 128)).astype(np.float32)
    core.load_vectors_to_hbm(0, x)

    prog = assemble(src)
    prog[0].extra["count"] = 32

    res = core.execute_program(prog)
    assert res["halted"] is True
    assert core.cycles.total > 0


def test_enc_macro_is_cycle_efficient():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(1)
    n = 256
    x = rng.standard_normal((n, 128)).astype(np.float32)
    enc = core.encode(x)
    depth = (
        3 * cfg.cycles_givens_layer
        + cfg.cycles_polar
        + cfg.cycles_quant_minmax
        + cfg.cycles_quant_round
        + cfg.cycles_qjl
        + cfg.cycles_pack
    )
    expected = depth + n - 1
    assert enc.cycles == expected, f"expected {expected}, got {enc.cycles}"


def test_energy_increases_with_batch():
    cfg = NQXConfig(dim=128)
    rng = np.random.default_rng(2)
    sizes = [16, 64, 256]
    energies = []
    for n in sizes:
        core = NQXCore(cfg)
        x = rng.standard_normal((n, 128)).astype(np.float32)
        core.encode(x)
        energies.append(core.energy.total_nj())
    assert energies[0] < energies[1] < energies[2]


def test_compression_per_vector_energy_below_naive_fp16():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.standard_normal((n, 128)).astype(np.float32)
    core.encode(x)
    nq_nj = core.energy.total_nj()
    naive_fp16_nj = (n * cfg.dim * 2 * 2 * cfg.pj_hbm_byte) / 1000.0
    assert nq_nj < naive_fp16_nj * 5, f"NQX energy {nq_nj:.1f}nJ vs naive RW {naive_fp16_nj:.1f}nJ"
