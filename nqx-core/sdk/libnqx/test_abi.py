"""SDK9: libnqx ABI test — verify functions are exposed correctly."""

from __future__ import annotations

import numpy as np
import pytest

from sdk.libnqx.libnqx import nqx_open, nqx_encode, nqx_decode, nqx_close, nqx_version


def test_open_close():
    hid = nqx_open('{"dim": 128, "bits": 3}')
    assert hid > 0
    nqx_close(hid)
    # Re-open after close
    hid2 = nqx_open('{"dim": 64, "bits": 4}')
    assert hid2 > 0
    nqx_close(hid2)


def test_encode_decode_roundtrip():
    hid = nqx_open('{"dim": 128, "bits": 3}')
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 128)).astype(np.float32)

    enc = nqx_encode(hid, x)
    assert enc["packed_len"] > 0
    assert enc["n"] == 8
    assert enc["dim"] == 128
    assert enc["encode_ms"] >= 0

    dec = nqx_decode(
        hid,
        enc["packed"],
        enc["sign_bits"],
        enc["mins"],
        enc["maxs"],
        enc["n"],
        enc["dim"],
        enc["bits"],
    )
    assert dec["vectors"].shape == (8, 128)
    assert dec["decode_ms"] >= 0

    nqx_close(hid)


def test_nqx_version():
    v = nqx_version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_invalid_handle_raises():
    with pytest.raises(KeyError):
        nqx_encode(999, np.zeros((1, 128), dtype=np.float32))
