#!/usr/bin/env python3
"""Replay harness — replays encode requests from access log and compares results.

Usage:
    python tools/debug/replay.py < access.jsonl
    python tools/debug/replay.py server/logs/access.jsonl

If payload files are available as <request_id>.json alongside the log, they
are used. Without saved payloads, only GET routes are replayed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.api import app
from starlette.testclient import TestClient


def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        log_dir = os.path.dirname(log_path)
        with open(log_path) as f:
            lines = [ln for ln in f if ln.strip()]
    else:
        log_dir = "."
        lines = [ln for ln in sys.stdin if ln.strip()]

    client = TestClient(app)
    ok = 0
    fail = 0

    for ln in lines:
        rec = json.loads(ln)
        route = rec.get("route", "")
        method = rec.get("method", "GET")
        rid = rec.get("request_id", "?")

        payload = None
        payload_path = os.path.join(log_dir, f"{rid}.json")
        if os.path.exists(payload_path):
            with open(payload_path) as pf:
                payload = json.load(pf)

        if method == "GET":
            resp = client.get(route)
        elif method == "POST" and payload is not None:
            resp = client.post(route, json=payload)
        else:
            print(f"  SKIP {rid} {method} {route} (no payload)")
            continue

        status_ok = resp.status_code == rec.get("status", 200)
        if status_ok:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {rid} {method} {route}: got {resp.status_code}")

    print(f"Replay: {ok} ok, {fail} failed, {len(lines)} total")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
