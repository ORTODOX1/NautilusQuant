import math


from nqx.constants import NQXConfig, GOLDEN_ANGLE
from nqx.lut import GoldenAngleLUT


def test_layer_pair_counts_dim128():
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    assert len(lut.layers["L1"]) == 64
    assert len(lut.layers["L2"]) == 63
    assert len(lut.layers["L3"]) >= 32


def test_layer_pairs_non_overlapping():
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    for name, layer in lut.layers.items():
        used = set()
        for i, j in layer.pairs:
            assert i != j, f"{name}: identical indices {i}"
            assert i not in used, f"{name}: idx {i} reused"
            assert j not in used, f"{name}: idx {j} reused"
            used.add(i)
            used.add(j)


def test_first_l1_angle_is_golden_angle():
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    assert math.isclose(lut.layers["L1"].angles[0], GOLDEN_ANGLE, rel_tol=1e-12)
    assert math.isclose(math.degrees(lut.layers["L1"].angles[0]), 137.5077640500378, abs_tol=1e-9)


def test_cos_sin_satisfy_unity():
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    for layer in lut.layers.values():
        for c, s in zip(layer.cos, layer.sin):
            assert math.isclose(c * c + s * s, 1.0, abs_tol=1e-12)


def test_rom_size_under_2kb():
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    assert lut.rom_bytes() < 2048
