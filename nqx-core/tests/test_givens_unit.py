import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.functional_units import GivensUnit
from nqx.lut import GoldenAngleLUT


def _make_unit(dim: int = 128):
    cfg = NQXConfig(dim=dim)
    lut = GoldenAngleLUT(cfg)
    return GivensUnit(cfg, lut)


@pytest.mark.parametrize("layer_id", [0, 1, 2])
def test_givens_forward_inverse_roundtrip_1d(layer_id):
    unit = _make_unit()
    rng = np.random.default_rng(42)
    x = rng.normal(size=unit.config.dim).astype(np.float32)

    fwd, _ = unit.apply_layer(x, layer_id)
    back, _ = unit.apply_layer(fwd, layer_id, inverse=True)

    assert np.allclose(back, x, atol=1e-5)


@pytest.mark.parametrize("layer_id", [0, 1, 2])
def test_givens_forward_inverse_roundtrip_2d(layer_id):
    unit = _make_unit()
    rng = np.random.default_rng(99)
    x = rng.normal(size=(8, unit.config.dim)).astype(np.float32)

    fwd, _ = unit.apply_layer(x, layer_id)
    back, _ = unit.apply_layer(fwd, layer_id, inverse=True)

    assert np.allclose(back, x, atol=1e-5)


def test_givens_l2_default_dim():
    unit = _make_unit(128)
    rng = np.random.default_rng(123)
    x = rng.normal(size=(4, 128)).astype(np.float32)

    fwd, _ = unit.apply_layer(x, 1)
    back, _ = unit.apply_layer(fwd, 1, inverse=True)

    assert np.allclose(back, x, atol=1e-5)


def test_givens_small_dim():
    unit = _make_unit(dim=16)
    rng = np.random.default_rng(77)
    x = rng.normal(size=16).astype(np.float32)

    for layer_id in range(3):
        fwd, _ = unit.apply_layer(x, layer_id)
        back, _ = unit.apply_layer(fwd, layer_id, inverse=True)
        assert np.allclose(back, x, atol=1e-5), f"Small dim failed at layer {layer_id}"
