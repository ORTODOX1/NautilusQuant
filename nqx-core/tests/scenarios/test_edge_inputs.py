"""S5: Edge inputs — empty, zeros, NaN, Inf, constant, max FP16."""

from __future__ import annotations

import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def test_empty_raises():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.empty((0, 128), dtype=np.float32)
    with pytest.raises(ValueError):
        core.encode(x)


def test_single_vector():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.random.default_rng(42).standard_normal((1, 128)).astype(np.float32)
    enc = core.encode(x)
    dec = core.decode(enc)
    rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
    assert rmse < 0.01


def test_all_zeros():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.zeros((10, 128), dtype=np.float32)
    enc = core.encode(x)
    dec = core.decode(enc)
    rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
    assert rmse < 1e-6


def test_constant_one():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.full((10, 128), 1.0, dtype=np.float32)
    enc = core.encode(x)
    dec = core.decode(enc)
    rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
    assert rmse < 0.01


def test_nan_raises():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.full((4, 128), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        core.encode(x)


def test_inf_raises():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.full((4, 128), np.inf, dtype=np.float32)
    with pytest.raises(ValueError):
        core.encode(x)


def test_max_fp16():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.full((4, 128), 65504.0, dtype=np.float32)
    enc = core.encode(x)
    dec = core.decode(enc)
    assert np.all(np.isfinite(dec.reconstructed))
