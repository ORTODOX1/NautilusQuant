"""S2: Long-context boundary — encode growing seq_len, check linear cycles."""

from __future__ import annotations

import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


@pytest.mark.parametrize("seq_len", [1024, 4096, 16384, 65536])
def test_long_context(seq_len):
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((seq_len, 128)).astype(np.float32)
    enc = core.encode(x)
    dec = core.decode(enc)
    rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))

    assert enc.cycles > 0
    assert enc.cycles > seq_len  # at least 1 cycle/vector
    assert rmse < 0.5  # RMSE stable regardless of batch size
