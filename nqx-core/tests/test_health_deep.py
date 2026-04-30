"""S11: Deep health check — LUT hash, orthogonality, errors, uptime."""

from __future__ import annotations

from server.api import app
from starlette.testclient import TestClient

client = TestClient(app)


def test_health_deep():
    resp = client.get("/health/deep")
    assert resp.status_code == 200
    body = resp.json()
    assert "lut_sha256" in body
    assert len(body["lut_sha256"]) == 64  # SHA-256 hex
    assert "orthogonality_err" in body
    assert body["orthogonality_err"] < 1e-5
    assert body["orthogonality_pass"] is True
    assert "errors_last_10" in body
    assert isinstance(body["errors_last_10"], list)
    assert "uptime_s" in body
    assert body["uptime_s"] >= 0
    assert "backend" in body
    assert "device" in body
