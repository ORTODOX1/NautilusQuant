"""S9: Request logging middleware for FastAPI."""

from __future__ import annotations

import json
import os
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_PATH = os.path.join(LOGS_DIR, "access.jsonl")
_MAX_BYTES = 10 * 1024 * 1024


def _write(line: str) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) >= _MAX_BYTES:
        base, ext = os.path.splitext(LOG_PATH)
        os.rename(LOG_PATH, f"{base}.old{ext}")
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        ts = time.time()
        body = await request.body()
        payload_bytes = len(body)
        response = await call_next(request)
        latency_ms = (time.time() - ts) * 1000

        record = {
            "request_id": request_id,
            "ts": ts,
            "route": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "payload_bytes": payload_bytes,
        }
        _write(json.dumps(record, default=str))
        return response
