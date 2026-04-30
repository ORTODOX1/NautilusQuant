"""FastAPI app exposing NautilusQuant encode/decode."""

from __future__ import annotations

import base64
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import HTMLResponse

from server.backends import Backend, auto_select
from server.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    DecodeRequest,
    DecodeResponse,
    EncodeRequest,
    EncodeResponse,
    EncodeStats,
    HealthResponse,
    InfoResponse,
    VerifyResponse,
)

from server.middleware import AccessLogMiddleware
from server.metrics import encode_total, decode_total, encode_latency, errors_total, render_metrics
from server.health_deep import deep_health, record_error
from server.errors import structured_error_handler, validation_error_handler, register_status_code

import nqx

NQX_DIM = int(os.environ.get("NQX_DIM", "128"))
NQX_BITS = int(os.environ.get("NQX_BITS", "3"))
NQX_PREFER = os.environ.get("NQX_BACKEND", "auto").lower()

app = FastAPI(
    title="NQX-Server",
    description="HTTP API for NautilusQuant KV-cache compression",
    version=nqx.__version__,
)
app.add_middleware(AccessLogMiddleware)

app.add_exception_handler(StarletteHTTPException, structured_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
register_status_code(400, "BadRequest")
register_status_code(422, "ValidationError")
register_status_code(500, "InternalError")

_backend: Optional[Backend] = None


def get_backend() -> Backend:
    global _backend
    if _backend is None:
        _backend = auto_select(dim=NQX_DIM, bits=NQX_BITS, prefer=NQX_PREFER)
    return _backend


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>NQX-Server</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto;
       padding: 0 20px; line-height: 1.5; color: #222; }
h1 { font-size: 28px; margin-bottom: 0; }
h2 { color: #555; font-weight: normal; margin-top: 4px; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
pre { background: #1a1a1a; color: #eee; padding: 12px; border-radius: 6px; overflow-x: auto; }
.box { background: #f9f9f9; border-left: 4px solid #d4af37; padding: 12px 16px; margin: 16px 0; }
a { color: #c47700; }
</style></head><body>
<h1>NQX-Server</h1>
<h2>HTTP API for NautilusQuant — golden-ratio KV-cache quantization</h2>
<div class="box">
  <strong>Backend:</strong> <code id="backend">checking...</code><br>
  <strong>Device:</strong> <code id="device">checking...</code>
</div>
<h3>Endpoints</h3>
<ul>
  <li><code>GET /health</code> — backend status</li>
  <li><code>GET /info</code> — current dim/bits/φ/LUT info</li>
  <li><code>POST /encode</code> — JSON {vectors: [[...]]} → packed payload</li>
  <li><code>POST /decode</code> — packed payload → vectors</li>
  <li><code>POST /benchmark</code> — throughput on this hardware</li>
  <li><code>POST /verify</code> — orthogonality + roundtrip checks</li>
  <li><a href="/docs">/docs</a> — interactive Swagger UI</li>
</ul>
<h3>Quick test</h3>
<pre>curl -s http://HOST:8000/health | jq
curl -s -X POST http://HOST:8000/benchmark -H 'content-type: application/json' \\
  -d '{"n_vectors":4096,"dim":128,"bits":3}' | jq</pre>
<script>
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('backend').textContent = d.backend;
  document.getElementById('device').textContent = d.device;
});
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


@app.get("/health", response_model=HealthResponse)
def health():
    be = get_backend()
    return HealthResponse(
        status="ok",
        backend=be.name,
        device=be.device,
        nqx_version=nqx.__version__,
        config=be.info(),
    )


@app.get("/health/deep")
def health_deep():
    be = get_backend()
    return deep_health(be)


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(**get_backend().info())


@app.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    be = get_backend()
    try:
        x = np.asarray(req.vectors, dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"invalid vectors payload: {e}")
    if x.ndim != 2:
        raise HTTPException(400, f"expected shape [n, dim], got {x.shape}")

    res = be.encode(x, bits=req.bits)

    encode_total.inc()
    encode_latency.observe(res["encode_ms"])

    return EncodeResponse(
        packed_b64=base64.b64encode(res["packed"]).decode("ascii"),
        sign_b64=base64.b64encode(res["sign"].tobytes()).decode("ascii"),
        mins=res["mins"].astype(np.float32).tolist(),
        maxs=res["maxs"].astype(np.float32).tolist(),
        n=res["n"],
        dim=res["dim"],
        bits=res["bits"],
        stats=EncodeStats(
            encode_ms=res["encode_ms"],
            cycles=int(res.get("cycles", 0)),
            energy_nj=float(res.get("energy_nj", 0.0)),
            compression_ratio=(res["n"] * res["dim"] * 2) / max(len(res["packed"]), 1),
            polar_rmse=res["polar_rmse"],
        ),
    )


@app.post("/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest):
    be = get_backend()
    try:
        packed = base64.b64decode(req.packed_b64)
        sign_blob = base64.b64decode(req.sign_b64)
        mins = np.asarray(req.mins, dtype=np.float32)
        maxs = np.asarray(req.maxs, dtype=np.float32)
    except Exception as e:
        raise HTTPException(400, f"invalid payload: {e}")

    from nqx.functional_units import PackUnit

    pk = PackUnit(be.config)
    q, _, _ = pk.unpack3plus1(packed, req.n)
    try:
        sign = np.frombuffer(sign_blob, dtype=np.uint8).reshape(req.n, req.dim).copy()
    except Exception as e:
        raise HTTPException(400, f"invalid sign payload: {e}")

    res = be.decode(q, sign, mins, maxs, req.bits)

    decode_total.inc()

    return DecodeResponse(
        vectors=res["x"].astype(np.float32).tolist(),
        decode_ms=res["decode_ms"],
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest):
    be = get_backend()
    rng = np.random.default_rng(req.seed)
    x = rng.standard_normal((req.n_vectors, req.dim)).astype(np.float32) * 0.5
    for col in (0, 15, 31, 63, 95, 127):
        if col < req.dim:
            mask = rng.random(req.n_vectors) < 0.75
            x[mask, col] = rng.standard_normal(int(mask.sum())).astype(np.float32) * 30.0

    enc = be.encode(x, bits=req.bits)
    dec = be.decode(enc["q"], enc["sign"], enc["mins"], enc["maxs"], req.bits)
    rmse = float(np.sqrt(((x - dec["x"]) ** 2).mean()))
    encode_ms = enc["encode_ms"]
    decode_ms = dec["decode_ms"]
    total_s = (encode_ms + decode_ms) / 1000.0
    throughput = req.n_vectors / max(total_s, 1e-9)

    return BenchmarkResponse(
        backend=be.name,
        device=be.device,
        n_vectors=req.n_vectors,
        dim=req.dim,
        bits=req.bits,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        throughput_vec_per_sec=throughput,
        compression_ratio=(req.n_vectors * req.dim * 2) / max(len(enc["packed"]), 1),
        roundtrip_rmse=rmse,
        energy_nj_per_vec=enc.get("energy_nj", 0.0) / max(req.n_vectors, 1),
    )


@app.post("/verify", response_model=VerifyResponse)
def verify():
    be = get_backend()
    if not hasattr(be, "rotation_matrix"):
        from server.backends import CPUBackend

        be = CPUBackend(dim=NQX_DIM, bits=NQX_BITS)
    T = be.rotation_matrix()
    orth_err = float(np.abs(T.T @ T - np.eye(T.shape[0])).max())
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, T.shape[0])).astype(np.float32)
    y = be.forward_rotation(x)
    norm_err = float(abs(np.linalg.norm(x, axis=-1).mean() - np.linalg.norm(y, axis=-1).mean()))
    x_back = be.inverse_rotation(y)
    rt_no_q = float(np.sqrt(((x - x_back) ** 2).mean()))
    enc = be.encode(x)
    dec = be.decode(enc["q"], enc["sign"], enc["mins"], enc["maxs"], enc["bits"])
    rt_q = float(np.sqrt(((x - dec["x"]) ** 2).mean()))
    return VerifyResponse(
        orthogonality_err=orth_err,
        norm_preservation_err=norm_err,
        roundtrip_rmse_no_quant=rt_no_q,
        roundtrip_rmse_with_quant=rt_q,
        all_passed=(orth_err < 1e-4 and rt_no_q < 1e-4),
    )


@app.get("/metrics", include_in_schema=False)
def metrics():
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(render_metrics())
