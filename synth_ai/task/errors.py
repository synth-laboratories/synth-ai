"""Error helpers used across Task App implementations."""

from __future__ import annotations

from typing import Any

from .json import to_jsonable


def error_payload(
    code: str, message: str, *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if extra:
        payload["error"].update(extra)
    return payload


def http_exception(
    status_code: int,
    code: str,
    message: str,
    *,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
):
    try:
        from fastapi import HTTPException  # type: ignore
    except Exception as exc:  # pragma: no cover - FastAPI not installed
        raise RuntimeError("fastapi must be installed to raise HTTPException") from exc

    payload = error_payload(code, message, extra=extra)
    return HTTPException(status_code=status_code, detail=to_jsonable(payload), headers=headers)


def json_error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
):
    try:
        from fastapi.responses import JSONResponse  # type: ignore
    except Exception as exc:  # pragma: no cover - FastAPI not installed
        raise RuntimeError("fastapi must be installed to build JSONResponse") from exc

    payload = error_payload(code, message, extra=extra)
    return JSONResponse(status_code=status_code, content=to_jsonable(payload), headers=headers)
