import numpy as np

from nqx.constants import NQXConfig
from nqx.functional_units import QuantUnit


def _make_unit(dim: int = 128):
    return QuantUnit(NQXConfig(dim=dim))


def test_quant_constant_input():
    unit = _make_unit()
    x = np.full(128, 3.14, dtype=np.float32)

    dequant, q, mins, maxs, _ = unit.quantize(x, bits=3)

    assert np.all(q == 0)
    assert np.allclose(dequant, 3.14, atol=1e-4)


def test_quant_all_zeros():
    unit = _make_unit()
    x = np.zeros(128, dtype=np.float32)

    dequant, q, mins, maxs, _ = unit.quantize(x, bits=3)

    assert np.all(q == 0)
    assert np.allclose(dequant, 0.0, atol=1e-6)


def test_quant_single_outlier():
    unit = _make_unit()
    x = np.zeros((5, 128), dtype=np.float32)
    x[3, 64] = 100.0

    dequant, q, mins, maxs, _ = unit.quantize(x, bits=3)

    assert q[3, 64] == 7
    assert mins[64] == 0.0
    assert maxs[64] == 100.0


def test_quant_negative_outlier():
    unit = _make_unit()
    x = np.zeros((5, 128), dtype=np.float32)
    x[1, 0] = -50.0

    dequant, q, _, _, _ = unit.quantize(x, bits=3)

    assert q[1, 0] == 0
    assert np.allclose(dequant[1, 0], -50.0, atol=1e-3)


def test_quant_roundtrip_perfect():
    unit = _make_unit()
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, size=128).astype(np.float32)

    dequant, q, mins, maxs, _ = unit.quantize(x, bits=8)

    back, _ = unit.dequantize(q, mins, maxs, bits=8)
    assert np.allclose(back, dequant, atol=1e-6)
