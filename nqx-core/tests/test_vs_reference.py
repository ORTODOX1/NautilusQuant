"""Bit-exact match between NQX and a pure-numpy port of NautilusQuantPyTorch math.

This avoids requiring PyTorch but still proves we encode identically.
"""

import math

import numpy as np

from nqx.constants import NQXConfig, GOLDEN_ANGLE, PHI
from nqx.cpu import NQXCore
from nqx.lut import GoldenAngleLUT


def _ref_layer_pairs_angles(dim: int):
    ga = GOLDEN_ANGLE
    l1 = ([(2 * k, 2 * k + 1) for k in range(dim // 2)], [ga * (k + 1) for k in range(dim // 2)])
    l2 = (
        [(2 * k + 1, 2 * k + 2) for k in range((dim - 1) // 2)],
        [ga * (k + 1) * PHI for k in range((dim - 1) // 2)],
    )
    stride = max(2, dim // 4)
    used = set()
    pairs, angles = [], []
    for k in range(dim):
        i, j = k, (k + stride) % dim
        if i == j or i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j))
        angles.append(ga * (k + 1) * PHI * PHI)
    l3 = (pairs, angles)
    return l1, l2, l3


def _ref_apply_layer(x: np.ndarray, pairs, angles, inverse=False) -> np.ndarray:
    out = x.copy()
    items = list(zip(pairs, angles))
    if inverse:
        items = list(reversed(items))
    for (i, j), theta in items:
        t = -theta if inverse else theta
        c, s = math.cos(t), math.sin(t)
        a = out[..., i].copy()
        b = out[..., j].copy()
        out[..., i] = a * c - b * s
        out[..., j] = a * s + b * c
    return out


def _ref_forward(x: np.ndarray, dim: int) -> np.ndarray:
    l1, l2, l3 = _ref_layer_pairs_angles(dim)
    out = _ref_apply_layer(x, *l1)
    out = _ref_apply_layer(out, *l2)
    out = _ref_apply_layer(out, *l3)
    return out


def test_lut_pairs_match_reference():
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    l1, l2, l3 = _ref_layer_pairs_angles(128)
    assert lut.layers["L1"].pairs == l1[0]
    assert lut.layers["L2"].pairs == l2[0]
    assert lut.layers["L3"].pairs == l3[0]


def test_lut_angles_match_reference():
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    l1, l2, l3 = _ref_layer_pairs_angles(128)
    for ref, name in [(l1, "L1"), (l2, "L2"), (l3, "L3")]:
        for nqx_a, ref_a in zip(lut.layers[name].angles, ref[1]):
            assert math.isclose(nqx_a, ref_a, rel_tol=1e-15)


def test_forward_rotation_matches_reference():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((20, 128)).astype(np.float32)

    nqx = core.forward_rotation(x)
    ref = _ref_forward(x.astype(np.float64), 128).astype(np.float32)
    diff = np.abs(nqx - ref).max()
    assert diff < 1e-4, f"NQX vs ref forward max diff: {diff}"


def test_inverse_rotation_matches_reference():
    cfg = NQXConfig(dim=128)
    core = NQXCore(cfg)
    rng = np.random.default_rng(8)
    x = rng.standard_normal((10, 128)).astype(np.float32)

    fwd = _ref_forward(x.astype(np.float64), 128).astype(np.float32)
    nqx_back = core.inverse_rotation(fwd)
    rmse = float(np.sqrt(((x - nqx_back) ** 2).mean()))
    assert rmse < 1e-4


def test_forward_dim64_dim256():
    for d in (64, 256):
        cfg = NQXConfig(dim=d)
        core = NQXCore(cfg)
        rng = np.random.default_rng(d)
        x = rng.standard_normal((8, d)).astype(np.float32)
        nqx = core.forward_rotation(x)
        ref = _ref_forward(x.astype(np.float64), d).astype(np.float32)
        diff = np.abs(nqx - ref).max()
        assert diff < 1e-4, f"dim={d}: max diff {diff}"
