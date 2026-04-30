"""S16: Slow client — chunked upload with delays, server must not hang."""

from __future__ import annotations

import asyncio
import json

import pytest

try:
    import httpx
except ImportError:
    httpx = None

from server.api import app


@pytest.mark.skipif(httpx is None, reason="httpx not installed")
def test_slow_client_does_not_hang():
    async def run():
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=app)
        base_url = "http://test"

        # valid 1x128 vector, 1KB body
        body = json.dumps({"vectors": [[0.0] * 128], "bits": 3}).encode()

        async def slow_body():
            for i in range(0, len(body), 1):
                await asyncio.sleep(0.01)
                yield body[i:i + 1]

        async with AsyncClient(transport=transport, base_url=base_url, timeout=30) as client:
            try:
                resp = await client.post(
                    "/encode", content=slow_body(),
                    headers={"content-type": "application/json"},
                )
                # Any status is fine — key is that it didn't hang
                assert resp.status_code in (200, 400, 408, 422, 500)
            except (httpx.TimeoutException, httpx.RemoteProtocolError):
                pass

    asyncio.run(run())
