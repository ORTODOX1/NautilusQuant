"""S9: Access log middleware — 5 requests → 5 lines in access.jsonl."""

from __future__ import annotations

import json
import os

from server.middleware import LOG_PATH


def test_access_log_writes_lines():
    # Clean slate
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    from starlette.testclient import TestClient

    from server.api import app

    client = TestClient(app)

    # 5 requests hitting various endpoints
    client.get("/health")
    client.get("/info")
    client.get("/")

    import numpy as np

    from nqx.constants import NQXConfig
    from nqx.cpu import NQXCore

    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    core.encode(x)
    payload = {
        "vectors": x.tolist(),
        "dim": 128,
        "bits": 3,
    }
    client.post("/encode", json=payload)
    client.post("/encode", json=payload)

    assert os.path.exists(LOG_PATH), "access log not created"
    with open(LOG_PATH) as f:
        lines = [ln for ln in f if ln.strip()]

    assert len(lines) == 5, f"expected 5 log lines, got {len(lines)}"
    for ln in lines:
        rec = json.loads(ln)
        for key in ("request_id", "ts", "route", "latency_ms", "status", "payload_bytes"):
            assert key in rec, f"missing key {key} in {rec}"
