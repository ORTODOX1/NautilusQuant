import numpy as np

from nqx.constants import NQXConfig
from nqx.functional_units import QJLUnit


def _make_unit(dim: int = 128):
    return QJLUnit(NQXConfig(dim=dim))


def test_qjl_sign_correction_positive():
    unit = _make_unit()
    orig = np.ones(128, dtype=np.float32) * 2.0
    quant = np.ones(128, dtype=np.float32)

    corrected, signs, _ = unit.apply(orig, quant, alpha=0.5)

    assert np.all(signs == 1)
    assert np.all(corrected > quant)


def test_qjl_sign_correction_negative():
    unit = _make_unit()
    orig = np.ones(128, dtype=np.float32)
    quant = np.ones(128, dtype=np.float32) * 2.0

    corrected, signs, _ = unit.apply(orig, quant, alpha=0.5)

    assert np.all(signs == 0)
    assert np.all(corrected < quant)


def test_qjl_alpha_zero():
    unit = _make_unit()
    rng = np.random.default_rng(42)
    orig = rng.normal(size=128).astype(np.float32)
    quant = rng.uniform(-1, 1, size=128).astype(np.float32)

    corrected, signs, _ = unit.apply(orig, quant, alpha=0.0)

    assert np.allclose(corrected, quant, atol=1e-6)


def test_qjl_alpha_half():
    unit = _make_unit()
    orig = np.ones(128, dtype=np.float32) * 2.0
    quant = np.zeros(128, dtype=np.float32)

    corrected, signs, _ = unit.apply(orig, quant, alpha=0.5)

    assert np.allclose(corrected, np.ones(128), atol=1e-6)


def test_qjl_alpha_one():
    unit = _make_unit()
    orig = np.ones(128, dtype=np.float32) * 2.0
    quant = np.zeros(128, dtype=np.float32)

    corrected, signs, _ = unit.apply(orig, quant, alpha=1.0)

    assert np.allclose(corrected, orig, atol=1e-6)


def test_qjl_mixed_signs():
    unit = _make_unit()
    rng = np.random.default_rng(7)
    orig = rng.normal(size=128).astype(np.float32)
    quant = rng.normal(size=128).astype(np.float32)

    corrected, signs, _ = unit.apply(orig, quant, alpha=0.5)

    error = orig - quant
    expected = quant + np.sign(error) * np.abs(error) * 0.5
    assert np.allclose(corrected, expected, atol=1e-6)
    assert np.array_equal(signs, (error >= 0).astype(np.uint8))
