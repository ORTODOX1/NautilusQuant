"""S13: OOM behaviour — large batch should succeed or raise MemoryError."""

from __future__ import annotations

import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def test_large_batch_oom_graceful():
    cfg = NQXConfig(dim=512, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((10000, 512)).astype(np.float32)

    try:
        enc = core.encode(x)
        dec = core.decode(enc)
        rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
        assert rmse < 0.5
    except MemoryError:
        pass  # graceful OOM is acceptable
    except Exception as e:
        pytest.fail(f"unexpected error type: {type(e).__name__}: {e}")
