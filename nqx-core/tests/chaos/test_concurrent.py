"""S15: Concurrent requests — 32 parallel encode via asyncio + httpx."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

try:
    import httpx
except ImportError:
    httpx = None

from server.api import app


@pytest.mark.skipif(httpx is None, reason="httpx not installed")
def test_concurrent_encode():
    async def run():
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        base_url = "http://test"
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 128)).astype(np.float32)
        payload = {"vectors": x.tolist(), "dim": 128, "bits": 3}

        async with AsyncClient(transport=transport, base_url=base_url) as client:
            async def do_encode():
                resp = await client.post("/encode", json=payload)
                assert resp.status_code == 200
                return resp

            results = await asyncio.gather(*[do_encode() for _ in range(32)])
            return results

    results = asyncio.run(run())
    assert len(results) == 32
    for r in results:
        assert r.status_code == 200
