"""S10: Prometheus /metrics endpoint — counters and histogram."""

from __future__ import annotations

from server.api import app
from starlette.testclient import TestClient

client = TestClient(app)


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "nqx_encode_total" in body
    assert "nqx_decode_total" in body
    assert "nqx_encode_latency_ms" in body
    assert "nqx_errors_total" in body


def test_metrics_counters_increment():
    import numpy as np
    from nqx.constants import NQXConfig
    from nqx.cpu import NQXCore

    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    payload = {"vectors": x.tolist(), "dim": 128, "bits": 3}

    # Reset server metrics by importing fresh (singletons carry over; just do 2 encodes)
    client.post("/encode", json=payload)
    client.post("/encode", json=payload)

    resp = client.get("/metrics")
    body = resp.text

    # Find the encode_total line
    for line in body.splitlines():
        if line.startswith("nqx_encode_total{"):
            val = int(line.split()[-1])
            assert val >= 2, f"expected >=2 encodes, got {val}"
            break
    else:
        assert False, "nqx_encode_total not found in metrics"
