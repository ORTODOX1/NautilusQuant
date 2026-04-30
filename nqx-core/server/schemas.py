"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    backend: str
    device: str
    nqx_version: str
    config: dict


class InfoResponse(BaseModel):
    dim: int
    bits: int
    phi: float
    golden_angle_deg: float
    n_pairs_l1: int
    n_pairs_l2: int
    n_pairs_l3: int
    rom_bytes: int


class EncodeRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="Shape [n, dim] FP32 vectors")
    bits: Optional[int] = Field(None, description="Override quantization bits (default 3)")


class EncodeStats(BaseModel):
    encode_ms: float
    cycles: int
    energy_nj: float
    compression_ratio: float
    polar_rmse: float


class EncodeResponse(BaseModel):
    packed_b64: str
    sign_b64: str
    mins: List[float]
    maxs: List[float]
    n: int
    dim: int
    bits: int
    stats: EncodeStats


class DecodeRequest(BaseModel):
    packed_b64: str
    sign_b64: str
    mins: List[float]
    maxs: List[float]
    n: int
    dim: int
    bits: int


class DecodeResponse(BaseModel):
    vectors: List[List[float]]
    decode_ms: float


class BenchmarkRequest(BaseModel):
    n_vectors: int = 4096
    dim: int = 128
    bits: int = 3
    seed: int = 42


class BenchmarkResponse(BaseModel):
    backend: str
    device: str
    n_vectors: int
    dim: int
    bits: int
    encode_ms: float
    decode_ms: float
    throughput_vec_per_sec: float
    compression_ratio: float
    roundtrip_rmse: float
    energy_nj_per_vec: float


class VerifyResponse(BaseModel):
    orthogonality_err: float
    norm_preservation_err: float
    roundtrip_rmse_no_quant: float
    roundtrip_rmse_with_quant: float
    all_passed: bool
