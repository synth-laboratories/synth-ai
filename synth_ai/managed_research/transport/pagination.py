"""Pagination helpers."""

from __future__ import annotations

from typing import Any


def build_query_params(**kwargs: Any) -> dict[str, Any]:
    """Drop unset values while preserving falsey-but-meaningful values."""

    params: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        params[key] = value
    return params


def extract_next_cursor(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("next_cursor", "cursor", "nextCursor"):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


__all__ = ["build_query_params", "extract_next_cursor"]
