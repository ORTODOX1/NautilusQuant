"""S12: Structured error responses — 400 + correct JSON format."""

from __future__ import annotations

from starlette.testclient import TestClient

from server.api import app


def test_invalid_shape_structured_error():
    client = TestClient(app)
    resp = client.post("/encode", json={"vectors": "not_a_list", "dim": 128, "bits": 3})
    assert resp.status_code == 422
    body = resp.json()
    assert "error_type" in body
    assert body["error_type"] == "ValidationError"
    assert "detail" in body
    assert "request_id" in body


def test_bad_decode_payload():
    client = TestClient(app)
    resp = client.post(
        "/decode",
        json={
            "packed_b64": "!!!invalid!!!",
            "sign_b64": "AAAA",
            "n": 4,
            "dim": 128,
            "bits": 3,
            "mins": [0.0],
            "maxs": [1.0],
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error_type"] == "BadRequest"
    assert "request_id" in body
