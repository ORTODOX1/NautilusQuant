import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.functional_units import PackUnit


def _make_unit(dim: int = 128):
    return PackUnit(NQXConfig(dim=dim))


@pytest.mark.parametrize("q_val", range(8))
@pytest.mark.parametrize("sign_val", [0, 1])
def test_pack_unpack_each_combination(q_val, sign_val):
    unit = _make_unit(dim=16)
    q = np.full((1, 16), q_val, dtype=np.uint8)
    signs = np.full((1, 16), sign_val, dtype=np.uint8)

    blob, _ = unit.pack3plus1(q, signs)
    q_back, signs_back, _ = unit.unpack3plus1(blob, n=1)

    assert q_back[0, 0] == q_val
    assert signs_back[0, 0] == sign_val


def test_pack_unpack_roundtrip_exhaustive():
    unit = _make_unit(dim=128)
    rng = np.random.default_rng(42)

    for _ in range(100):
        q = rng.integers(0, 8, size=(4, 128), dtype=np.uint8)
        signs = rng.integers(0, 2, size=(4, 128), dtype=np.uint8)

        blob, _ = unit.pack3plus1(q, signs)
        q_back, signs_back, _ = unit.unpack3plus1(blob, n=4)

        assert np.array_equal(q_back, q)
        assert np.array_equal(signs_back, signs)


def test_pack_covers_all_byte_positions():
    unit = _make_unit(dim=128)
    q = np.arange(128, dtype=np.uint8).reshape(1, -1) % 8
    signs = np.zeros((1, 128), dtype=np.uint8)

    blob, _ = unit.pack3plus1(q, signs)
    q_back, signs_back, _ = unit.unpack3plus1(blob, n=1)

    assert np.array_equal(q_back[0], q[0])
    assert np.array_equal(signs_back[0], signs[0])
