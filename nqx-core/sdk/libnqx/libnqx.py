#!/usr/bin/env python3
"""libnqx Python implementation — C ABI prototype backed by NQXCore.

Usage:
    from libnqx import nqx_open, nqx_encode, nqx_decode, nqx_close
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore

_handles: dict[int, Any] = {}
_next_id = 1


def nqx_open(config: str) -> int:
    global _next_id
    try:
        cfg_dict = json.loads(config) if config else {}
        dim = cfg_dict.get("dim", 128)
        bits = cfg_dict.get("bits", 3)
        cfg = NQXConfig(dim=dim, bits=bits)
        core = NQXCore(cfg)
        hid = _next_id
        _next_id += 1
        _handles[hid] = core
        return hid
    except Exception as e:
        raise RuntimeError(f"nqx_open failed: {e}") from e


def nqx_encode(
    handle: int,
    vectors: np.ndarray,
    bits: int = 0,
) -> dict:
    core = _handles[handle]
    if bits == 0:
        bits = core.config.bits
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    import time
    t0 = time.perf_counter()
    enc = core.encode(vectors)
    dt = (time.perf_counter() - t0) * 1000

    dec = core.decode(enc)
    rmse = float(np.sqrt(((vectors - dec.reconstructed) ** 2).mean()))

    return {
        "packed": enc.packed_bytes,
        "packed_len": len(enc.packed_bytes),
        "sign_bits": enc.sign_bits.tobytes(),
        "sign_len": enc.sign_bits.nbytes,
        "mins": enc.mins.tolist(),
        "maxs": enc.maxs.tolist(),
        "dim": core.config.dim,
        "n": vectors.shape[0],
        "bits": core.config.bits,
        "encode_ms": dt,
        "rmse": rmse,
    }


def nqx_decode(
    handle: int,
    packed: bytes,
    sign_bits: bytes,
    mins: list[float],
    maxs: list[float],
    n: int,
    dim: int,
    bits: int,
) -> dict:
    from nqx.functional_units import PackUnit

    core = _handles[handle]
    pk = PackUnit(core.config)
    q, _, _ = pk.unpack3plus1(packed, n)
    sign = np.frombuffer(sign_bits, dtype=np.uint8).reshape(n, dim).copy()
    mins_arr = np.asarray(mins, dtype=np.float32)
    maxs_arr = np.asarray(maxs, dtype=np.float32)

    from nqx.functional_units import QuantUnit
    qu = QuantUnit(core.config)

    import time
    t0 = time.perf_counter()
    dequant, _ = qu.dequantize(q, mins_arr, maxs_arr, bits)
    core.pu.from_polar(dequant)
    # Full decode
    from nqx.cpu import EncodeResult, DecodeResult

    er = EncodeResult(
        quantized_indices=q, sign_bits=sign, mins=mins_arr, maxs=maxs_arr,
        packed_bytes=packed, polar=dequant, cycles=0, energy_nj=0,
    )
    dec = core.decode(er)
    dt = (time.perf_counter() - t0) * 1000

    return {
        "vectors": dec.reconstructed,
        "n": dec.reconstructed.shape[0],
        "dim": dec.reconstructed.shape[1],
        "decode_ms": dt,
        "rmse": 0.0,
    }


def nqx_close(handle: int) -> None:
    _handles.pop(handle, None)


def nqx_version() -> str:
    import nqx
    return nqx.__version__
