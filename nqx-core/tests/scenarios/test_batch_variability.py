"""S3: Variable batch sizes — compression ratio constant, RMSE stable."""

from __future__ import annotations

import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


@pytest.mark.parametrize("batch", [1, 8, 64, 256, 1024, 4096])
def test_compression_ratio(batch):
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((batch, 128)).astype(np.float32)
    enc = core.encode(x)

    raw_bytes = x.size * 2  # FP16
    ratio = raw_bytes / len(enc.packed_bytes)
    assert abs(ratio / 4.0 - 1.0) < 0.01, f"ratio {ratio:.2f}x, expected 4.00x"


def test_rmse_independent_of_batch():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)
    rmse_vals = {}
    for batch in [64, 256, 1024, 4096]:
        x = rng.standard_normal((batch, 128)).astype(np.float32)
        enc = core.encode(x)
        dec = core.decode(enc)
        rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
        rmse_vals[batch] = rmse

    ref = rmse_vals[64]
    for batch, rmse in rmse_vals.items():
        assert (
            abs(rmse / ref - 1.0) < 0.10
        ), f"batch {batch}: RMSE {rmse:.4f} deviates from ref {ref:.4f} by >10%"
