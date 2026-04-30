import numpy as np
import pytest

from nqx.assembler import assemble
from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.isa import Opcode, Instruction, encode_instruction, decode_instruction
from nqx.subbit_unit import SubBitUnit


def _ref_scalar_quantize(x: np.ndarray, bits: int) -> np.ndarray:
    levels = 2**bits
    if x.ndim == 1:
        xb = x.reshape(1, -1)
    else:
        xb = x
    mins = xb.min(axis=0)
    maxs = xb.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    norm = (xb - mins) / ranges
    q = np.round(norm * (levels - 1))
    return ((q / (levels - 1)) * ranges + mins).reshape(x.shape)


def _ref_subbit(polar: np.ndarray, r_bits: int, a_bits: int) -> np.ndarray:
    radii = polar[..., 0::2]
    angles = polar[..., 1::2]
    r_dq = _ref_scalar_quantize(radii, r_bits)
    a_dq = _ref_scalar_quantize(angles, a_bits)
    out = np.zeros_like(polar)
    out[..., 0::2] = r_dq
    out[..., 1::2] = a_dq
    return out


def _make_polar(rng, n=4, dim=128):
    cart = rng.standard_normal((n, dim)).astype(np.float32)
    polar = np.zeros_like(cart)
    polar[..., 0::2] = np.sqrt(cart[..., 0::2] ** 2 + cart[..., 1::2] ** 2)
    polar[..., 1::2] = np.arctan2(cart[..., 1::2], cart[..., 0::2])
    return polar


@pytest.mark.parametrize("r_bits,a_bits", [(3, 1), (3, 2), (2, 1), (2, 2), (4, 2)])
def test_subbit_bit_exact_vs_reference(r_bits, a_bits):
    cfg = NQXConfig(dim=128)
    sb = SubBitUnit(cfg)
    rng = np.random.default_rng(2026)
    polar = _make_polar(rng, n=8, dim=128)
    out, meta, _ = sb.encode(polar, r_bits, a_bits)
    ref = _ref_subbit(polar, r_bits, a_bits)
    assert np.array_equal(out, ref)
    assert meta["radius_bits"] == r_bits
    assert meta["angle_bits"] == a_bits


_RMSE_BUDGET = {
    (3, 1): 1.6,
    (3, 2): 1.0,
    (2, 1): 1.6,
    (2, 2): 1.0,
    (4, 2): 0.5,
}


@pytest.mark.parametrize("r_bits,a_bits", [(3, 1), (3, 2), (2, 1), (2, 2), (4, 2)])
def test_subbit_rmse_within_budget(r_bits, a_bits):
    cfg = NQXConfig(dim=128)
    sb = SubBitUnit(cfg)
    rng = np.random.default_rng(101)
    polar = _make_polar(rng, n=64, dim=128)
    out, _, _ = sb.encode(polar, r_bits, a_bits)
    rmse = float(np.sqrt(((polar - out) ** 2).mean()))
    budget = _RMSE_BUDGET[(r_bits, a_bits)]
    assert rmse < budget, f"({r_bits},{a_bits}) RMSE={rmse:.4f} budget={budget}"


def test_subbit_compression_vs_rmse_table():
    cfg = NQXConfig(dim=128)
    sb = SubBitUnit(cfg)
    rng = np.random.default_rng(101)
    polar = _make_polar(rng, n=64, dim=128)
    pairs = [(3, 1), (3, 2), (2, 1), (2, 2), (4, 2)]
    print("\n  r_bits | θ_bits | bits/val | comp | RMSE")
    print("  -------|--------|----------|------|------")
    for r, a in pairs:
        out, meta, _ = sb.encode(polar, r, a)
        rmse = float(np.sqrt(((polar - out) ** 2).mean()))
        bpv = meta["bits_per_value"]
        comp = meta["compression_ratio"]
        print(f"  {r:6d} | {a:6d} | {bpv:7.1f} | {comp:4.1f}x | {rmse:.4f}")


def test_subbit_decode_roundtrip():
    cfg = NQXConfig(dim=128)
    sb = SubBitUnit(cfg)
    rng = np.random.default_rng(0)
    polar = _make_polar(rng, n=4, dim=128)
    enc, meta, _ = sb.encode(polar, 3, 1)
    dec, _ = sb.decode(meta)
    assert np.array_equal(enc, dec)


def test_isa_subbit_enc_roundtrip():
    a = Instruction(opcode=Opcode.SUBBIT_ENC, rd=2, rs1=3, rs2=1)
    b = decode_instruction(encode_instruction(a))
    assert b.opcode == Opcode.SUBBIT_ENC and b.rd == 2 and b.rs1 == 3 and b.rs2 == 1


def test_assembler_subbit_enc():
    src = "SUBBIT_ENC V0, 3, 1\nHALT"
    prog = assemble(src)
    assert prog[0].opcode == Opcode.SUBBIT_ENC
    assert prog[0].rs1 == 3
    assert prog[0].rs2 == 1


def test_assembler_subbit_rejects_bad_bits():
    from nqx.assembler import AssemblyError

    with pytest.raises(AssemblyError):
        assemble("SUBBIT_ENC V0, 0, 1")
    with pytest.raises(AssemblyError):
        assemble("SUBBIT_ENC V0, 3")


def test_cpu_subbit_enc_then_dec():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(2)
    polar = _make_polar(rng, n=2, dim=128)
    core.vrf.write(0, polar)

    src = """
    SUBBIT_ENC V0, 3, 1
    SUBBIT_DEC V0
    HALT
    """
    prog = assemble(src)
    core.execute_program(prog)
    out = core.vrf.read(0)
    ref = _ref_subbit(polar, 3, 1)
    assert np.array_equal(out, ref)
