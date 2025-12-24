"""Utility helpers for status commands."""

from __future__ import annotations

from typing import Any


def build_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key}


def ensure_status_ok(response) -> dict[str, Any]:
    if response.status_code >= 400:
        detail = ""
        try:
            payload = response.json()
            detail = payload.get("detail", "")
        except Exception:
            detail = response.text
        raise RuntimeError(detail or f"Request failed ({response.status_code})")
    return response.json()
