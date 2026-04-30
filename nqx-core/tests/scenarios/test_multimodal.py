"""S4: Mixed-modality KV — 50% text, 50% visual (outlier dims), SubBit encode."""

from __future__ import annotations

import numpy as np

from nqx.constants import NQXConfig
from nqx.subbit_unit import SubBitUnit


def test_mixed_modality_subbit():
    cfg = NQXConfig(dim=128)
    sb = SubBitUnit(cfg)
    rng = np.random.default_rng(42)

    n_text = 64
    n_visual = 64
    dim = 128

    text = rng.standard_normal((n_text, dim)).astype(np.float32)
    text_polar = _to_polar(text)

    visual = rng.standard_normal((n_visual, dim)).astype(np.float32)
    outlier_cols = rng.integers(0, dim // 2, size=10) * 2  # even indices (radii)
    visual[:, outlier_cols] *= 20.0
    visual_polar = _to_polar(visual)

    mixed_polar = np.concatenate([text_polar, visual_polar], axis=0)
    recovered, meta, _ = sb.encode(mixed_polar, radius_bits=3, angle_bits=2)

    rmse = float(np.sqrt(((mixed_polar - recovered) ** 2).mean()))
    assert rmse < 0.65, f"Mixed-modality RMSE too high: {rmse:.4f}"


def _to_polar(x):
    polar = np.zeros_like(x)
    polar[..., 0::2] = np.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
    polar[..., 1::2] = np.arctan2(x[..., 1::2], x[..., 0::2])
    return polar
