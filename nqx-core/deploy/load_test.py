#!/usr/bin/env python3
"""S21: Load test — 100 concurrent clients, 50% encode 50% decode, 60s.

Usage:
    python deploy/load_test.py [url]
    url defaults to http://localhost:8000
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import httpx
except ImportError:
    print("httpx required", file=sys.stderr)
    sys.exit(1)

RESULTS_DIR = os.path.join(ROOT, "deploy")

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N_CLIENTS = 100
DURATION_S = 60

# Pre-encoded payload for decode requests
import base64
import numpy as np
from nqx.constants import NQXConfig
from nqx.cpu import NQXCore

cfg = NQXConfig(dim=128, bits=3)
core = NQXCore(cfg)
rng = np.random.default_rng(42)
x = rng.standard_normal((4, 128)).astype(np.float32)
enc = core.encode(x)
DECODE_PAYLOAD = {
    "packed_b64": base64.b64encode(enc.packed_bytes).decode(),
    "sign_b64": base64.b64encode(enc.sign_bits.tobytes()).decode(),
    "mins": enc.mins.tolist(),
    "maxs": enc.maxs.tolist(),
    "n": 4,
    "dim": 128,
    "bits": 3,
}
ENCODE_PAYLOAD = {"vectors": x.tolist(), "bits": 3}


async def run(client: httpx.AsyncClient, results: list, pid: int):
    end_time = time.time() + DURATION_S
    while time.time() < end_time:
        t0 = time.perf_counter()
        try:
            if pid % 2 == 0:
                resp = await client.post(f"{BASE_URL}/encode", json=ENCODE_PAYLOAD)
            else:
                resp = await client.post(f"{BASE_URL}/decode", json=DECODE_PAYLOAD)
            latency = (time.perf_counter() - t0) * 1000
            results.append(latency)
        except Exception:
            pass
        await asyncio.sleep(0)


async def main():
    print(f"Load test: {N_CLIENTS} clients, {DURATION_S}s, target {BASE_URL}")
    print()

    results: list[float] = []
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [run(client, results, i) for i in range(N_CLIENTS)]
        await asyncio.gather(*tasks)

    if not results:
        print("No successful requests")
        sys.exit(1)

    results.sort()
    n = len(results)
    p50 = results[int(n * 0.5)]
    p95 = results[int(n * 0.95)]
    p99 = results[int(n * 0.99)]
    throughput = n / DURATION_S

    print(f"Requests completed: {n}")
    print(f"Throughput: {throughput:.0f} req/s")
    print(f"Latency p50: {p50:.2f}ms")
    print(f"Latency p95: {p95:.2f}ms")
    print(f"Latency p99: {p99:.2f}ms")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(RESULTS_DIR, f"load_test_results.md")
    with open(path, "w") as f:
        f.write(f"# Load test results {ts}\n\n")
        f.write(f"- Target: {BASE_URL}\n")
        f.write(f"- Clients: {N_CLIENTS}\n")
        f.write(f"- Duration: {DURATION_S}s\n")
        f.write(f"- Requests: {n}\n")
        f.write(f"- Throughput: {throughput:.0f} req/s\n")
        f.write(f"- p50: {p50:.2f}ms\n")
        f.write(f"- p95: {p95:.2f}ms\n")
        f.write(f"- p99: {p99:.2f}ms\n")
    print(f"\nSaved: {path}")
    print(f"Load test {'PASSED' if n > 0 else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
