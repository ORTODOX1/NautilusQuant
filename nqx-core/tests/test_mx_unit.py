import numpy as np
import pytest

from nqx.assembler import assemble
from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.isa import Opcode, encode_instruction, decode_instruction, Instruction
from nqx.mx_unit import MXQuantizer, MX_FORMATS, MX_FORMAT_BY_INDEX


def _ref_mx_quantize(x: np.ndarray, format_name: str, block_size: int = 32) -> np.ndarray:
    fmt = MX_FORMATS[format_name]
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    n = flat.size
    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, block_size)

    block_max = np.maximum(np.abs(blocks).max(axis=1), 1e-30)
    shared_exp = np.floor(np.log2(block_max)).clip(-127, 127).astype(np.int8)
    scale = (2.0 ** shared_exp.astype(np.float32))[:, None]
    normalized = blocks / scale

    if fmt["exponent_bits"] > 0:
        levels = 2 ** (fmt["mantissa_bits"] + 1)
    else:
        levels = 2 ** (fmt["total"] - 1)

    q = np.round(normalized * levels).clip(-levels, levels)
    dequant = (q / levels) * scale
    return dequant.reshape(-1)[:n].reshape(x.shape)


@pytest.mark.parametrize("fmt", list(MX_FORMATS))
def test_mx_quantize_bit_exact_vs_reference(fmt):
    cfg = NQXConfig(dim=128)
    mx = MXQuantizer(cfg, format_name=fmt, block_size=32)
    rng = np.random.default_rng(123)
    x = rng.standard_normal((4, 128)).astype(np.float32) * 3.0
    out, meta, _ = mx.quantize(x)
    ref = _ref_mx_quantize(x, fmt, block_size=32)
    assert out.shape == x.shape
    assert np.array_equal(out, ref), f"MX {fmt} not bit-exact vs reference"


def test_mx_metadata_shapes():
    cfg = NQXConfig(dim=128)
    mx = MXQuantizer(cfg, format_name="MXFP4", block_size=32)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((10, 128)).astype(np.float32)
    _, meta, _ = mx.quantize(x)
    assert meta["block_size"] == 32
    assert meta["n_blocks"] == 10 * 128 // 32
    assert meta["q"].shape == (meta["n_blocks"], 32)
    assert meta["shared_exp"].shape == (meta["n_blocks"],)
    assert meta["shared_exp"].dtype == np.int8


def test_mx_quantize_then_dequantize_roundtrip():
    cfg = NQXConfig(dim=128)
    mx = MXQuantizer(cfg, format_name="MXFP4")
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    dequant, meta, _ = mx.quantize(x)
    again, _ = mx.dequantize(meta)
    assert np.array_equal(dequant, again)


def test_mx_serialize_deserialize_roundtrip():
    cfg = NQXConfig(dim=128)
    mx = MXQuantizer(cfg, format_name="MXFP4")
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, 128)).astype(np.float32)
    dequant, meta, _ = mx.quantize(x)
    blob = mx.serialize(meta)
    rebuilt = mx.deserialize(blob, meta["n_blocks"], meta["original_shape"], meta["pad"])
    assert np.array_equal(rebuilt["q"], meta["q"])
    assert np.array_equal(rebuilt["shared_exp"], meta["shared_exp"])
    again, _ = mx.dequantize(rebuilt)
    assert np.array_equal(dequant, again)


def test_isa_mxpack_mxunpack_roundtrip():
    a = Instruction(opcode=Opcode.MXPACK, rd=2, rs1=0)
    b = decode_instruction(encode_instruction(a))
    assert b.opcode == Opcode.MXPACK and b.rd == 2 and b.rs1 == 0

    a2 = Instruction(opcode=Opcode.MXUNPACK, rd=5, rs1=2)
    b2 = decode_instruction(encode_instruction(a2))
    assert b2.opcode == Opcode.MXUNPACK and b2.rd == 5 and b2.rs1 == 2


def test_assembler_mxpack_mxunpack():
    src = """
    LDV V0, [0x0]
    MXPACK V0, MXFP4
    MXUNPACK V0, MXFP4
    HALT
    """
    prog = assemble(src)
    assert prog[1].opcode == Opcode.MXPACK
    assert prog[1].rd == 0
    assert prog[1].rs1 == 0
    assert prog[2].opcode == Opcode.MXUNPACK
    assert prog[2].rs1 == 0


def test_assembler_mxpack_with_numeric_format():
    src = "MXPACK V3, 2\nHALT"
    prog = assemble(src)
    assert prog[0].opcode == Opcode.MXPACK
    assert prog[0].rs1 == 2


def test_cpu_mxpack_then_mxunpack():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(5)
    vec = rng.standard_normal((4, 128)).astype(np.float32)
    core.load_vectors_to_hbm(0, vec)

    src = """
    LDV V0, [0x0]
    MXPACK V0, MXFP4
    MXUNPACK V0, MXFP4
    HALT
    """
    prog = assemble(src)
    prog[0].extra["count"] = 4
    core.execute_program(prog)

    out = core.vrf.read(0)
    ref = _ref_mx_quantize(vec, "MXFP4", block_size=32)
    assert np.array_equal(out, ref)


def test_format_index_table():
    assert MX_FORMAT_BY_INDEX == ["MXFP4", "MXFP6", "MXFP8", "MXINT8"]
