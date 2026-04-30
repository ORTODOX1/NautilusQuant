"""S11: Deep health check — LUT hash, orthogonality self-test, errors, uptime."""

from __future__ import annotations

import hashlib
import time
from typing import List

import numpy as np

_ERROR_LOG: List[dict] = []
_START_TIME = time.time()


def record_error(error_type: str, detail: str) -> None:
    _ERROR_LOG.append({"ts": time.time(), "type": error_type, "detail": detail})
    if len(_ERROR_LOG) > 10:
        _ERROR_LOG.pop(0)


def deep_health(backend) -> dict:
    # LUT hash
    lut = backend.core.lut
    buf = bytearray()
    for name in ("L1", "L2", "L3"):
        layer = lut.layers[name]
        buf.extend(layer.cos_arr.tobytes())
        buf.extend(layer.sin_arr.tobytes())
    lut_hash = hashlib.sha256(buf).hexdigest()

    # Orthogonality self-test: T^T·T ≈ I
    T = backend.rotation_matrix().astype(np.float64)
    orth_err = float(np.abs(T.T @ T - np.eye(T.shape[0])).max())

    # Uptime
    uptime_s = time.time() - _START_TIME

    return {
        "lut_sha256": lut_hash,
        "orthogonality_err": orth_err,
        "orthogonality_pass": orth_err < 1e-5,
        "errors_last_10": list(_ERROR_LOG),
        "uptime_s": round(uptime_s, 1),
        "backend": backend.name,
        "device": backend.device,
    }
