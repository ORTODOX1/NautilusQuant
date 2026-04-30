"""S12: Structured error responses — unified JSON format."""

from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_ERROR_TYPE_MAP: dict[int, str] = {}


def register_status_code(status_code: int, error_type: str) -> None:
    _ERROR_TYPE_MAP[status_code] = error_type


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex[:8])


async def structured_error_handler(request: Request, exc: StarletteHTTPException):
    error_type = _ERROR_TYPE_MAP.get(exc.status_code, f"HTTP{exc.status_code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": error_type,
            "detail": str(exc.detail),
            "request_id": _get_request_id(request),
        },
    )


async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error_type": "ValidationError",
            "detail": str(exc.errors()),
            "request_id": _get_request_id(request),
        },
    )
