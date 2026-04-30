import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.functional_units import PolarUnit


def _make_unit(dim: int = 128):
    return PolarUnit(NQXConfig(dim=dim))


@pytest.mark.parametrize("dim", [16, 32, 64, 128])
def test_polar_roundtrip_1d(dim):
    unit = _make_unit(dim)
    rng = np.random.default_rng(42)
    x = rng.normal(size=dim).astype(np.float32)

    polar, _ = unit.to_polar(x)
    back, _ = unit.from_polar(polar)

    assert np.allclose(back, x, atol=1e-5)


@pytest.mark.parametrize("dim", [16, 32, 64, 128])
def test_polar_roundtrip_2d(dim):
    unit = _make_unit(dim)
    rng = np.random.default_rng(99)
    x = rng.normal(size=(8, dim)).astype(np.float32)

    polar, _ = unit.to_polar(x)
    back, _ = unit.from_polar(polar)

    assert np.allclose(back, x, atol=1e-5)


def test_polar_odd_dim():
    unit = _make_unit(dim=17)
    rng = np.random.default_rng(13)
    x = rng.normal(size=17).astype(np.float32)

    polar, _ = unit.to_polar(x)
    back, _ = unit.from_polar(polar)

    assert np.allclose(back, x, atol=1e-5)
    assert polar[-1] == x[-1]
