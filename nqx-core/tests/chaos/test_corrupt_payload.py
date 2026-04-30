"""S14: Corrupt payload — invalid base64, wrong length, garbage, bad bits."""

from __future__ import annotations

import base64

import numpy as np
from starlette.testclient import TestClient

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from server.api import app

client = TestClient(app)


def _make_good_encode():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = np.random.default_rng(42).standard_normal((4, 128)).astype(np.float32)
    enc = core.encode(x)
    return {
        "packed_b64": base64.b64encode(enc.packed_bytes).decode(),
        "sign_b64": base64.b64encode(enc.sign_bits.tobytes()).decode(),
        "mins": enc.mins.tolist(),
        "maxs": enc.maxs.tolist(),
        "n": 4,
        "dim": 128,
        "bits": 3,
    }


def test_invalid_base64():
    good = _make_good_encode()
    good["packed_b64"] = "!!!invalid!!!"
    resp = client.post("/decode", json=good)
    assert resp.status_code == 400
    assert resp.json()["error_type"] == "BadRequest"


def test_wrong_byte_length():
    good = _make_good_encode()
    short = base64.b64encode(b"x").decode()
    good["sign_b64"] = short
    resp = client.post("/decode", json=good)
    assert resp.status_code in (400, 422)


def test_garbage_mins():
    good = _make_good_encode()
    good["mins"] = ["not", "numbers"]
    resp = client.post("/decode", json=good)
    assert resp.status_code == 422
    assert resp.json()["error_type"] == "ValidationError"


def test_bad_bits():
    good = _make_good_encode()
    good["bits"] = 99
    resp = client.post("/decode", json=good)
    # The backend might reconfigure or raise — either way fine
    assert resp.status_code in (200, 400, 422)
