"""S7: Golden reference regression test — bit-exact encode protection."""

from __future__ import annotations

import os

import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "golden")


def test_encode_bit_exact():
    path = os.path.join(GOLDEN_DIR, "seed42_dim128_bits3.npz")
    if not os.path.exists(path):
        pytest.skip(
            f"golden snapshot missing: {path} — regenerate with tests/fixtures/gen_golden.py"
        )

    golden = np.load(path)
    x = golden["input"]

    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    enc = core.encode(x)
    dec = core.decode(enc)

    import hashlib

    digest = hashlib.sha256(dec.reconstructed.tobytes()).hexdigest()
    expected = str(golden["sha256"])
    assert digest == expected, f"sha256 mismatch: got {digest}, expected {expected}"
