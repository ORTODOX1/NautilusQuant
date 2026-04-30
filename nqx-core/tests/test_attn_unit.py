import numpy as np

from nqx.assembler import assemble
from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.functional_units import AttentionUnit
from nqx.isa import Opcode, Instruction, encode_instruction, decode_instruction


def _to_polar(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[..., 0::2] = np.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
    out[..., 1::2] = np.arctan2(x[..., 1::2], x[..., 0::2])
    return out


def test_attn_dot_matches_cartesian_dot():
    cfg = NQXConfig(dim=128)
    attn = AttentionUnit(cfg)
    rng = np.random.default_rng(33)
    q = rng.standard_normal((1, 128)).astype(np.float32)
    k = rng.standard_normal((1, 128)).astype(np.float32)

    pq = _to_polar(q)
    pk = _to_polar(k)
    out, _ = attn.dot_polar(pq, pk)
    expected = (q * k).sum(axis=-1)
    assert out.shape == (1, 1)
    assert np.allclose(out.flat[0], expected.flat[0], rtol=1e-4, atol=1e-4)


def test_attn_dot_batch_pairwise():
    cfg = NQXConfig(dim=128)
    attn = AttentionUnit(cfg)
    rng = np.random.default_rng(11)
    n_q, n_k = 4, 6
    q = rng.standard_normal((n_q, 128)).astype(np.float32)
    k = rng.standard_normal((n_k, 128)).astype(np.float32)

    pq = _to_polar(q)
    pk = _to_polar(k)
    out, _ = attn.dot_polar(pq, pk)
    expected = q @ k.T
    assert out.shape == (n_q, n_k)
    assert np.allclose(out, expected, rtol=1e-3, atol=1e-3)


def test_attn_dot_self_is_norm_squared():
    cfg = NQXConfig(dim=128)
    attn = AttentionUnit(cfg)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 128)).astype(np.float32)
    px = _to_polar(x)
    out, _ = attn.dot_polar(px, px)
    expected = (x * x).sum()
    assert np.isclose(out.flat[0], expected, rtol=1e-4)


def test_isa_attn_dot_roundtrip():
    a = Instruction(opcode=Opcode.ATTN_DOT, rs1=2, rs2=3)
    b = decode_instruction(encode_instruction(a))
    assert b.opcode == Opcode.ATTN_DOT and b.rs1 == 2 and b.rs2 == 3


def test_assembler_attn_dot():
    src = "ATTN_DOT V0, V1\nHALT"
    prog = assemble(src)
    assert prog[0].opcode == Opcode.ATTN_DOT
    assert prog[0].rs1 == 0
    assert prog[0].rs2 == 1


def test_cpu_attn_dot_stores_result():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(99)
    q = rng.standard_normal((1, 128)).astype(np.float32)
    k = rng.standard_normal((1, 128)).astype(np.float32)
    pq = _to_polar(q)
    pk = _to_polar(k)
    core.vrf.write(0, pq)
    core.vrf.write(1, pk)

    src = "ATTN_DOT V0, V1\nHALT"
    prog = assemble(src)
    core.execute_program(prog)

    expected = (q * k).sum()
    assert core.last_attn_dot is not None
    assert np.isclose(core.last_attn_dot.flat[0], expected, rtol=1e-4, atol=1e-4)


def test_attn_dot_opcode_value():
    assert int(Opcode.ATTN_DOT) == 0x80
