"""Vendor API key helpers shared by Task Apps."""

from __future__ import annotations

import os

from .errors import http_exception

_VENDOR_KEYS = {
    "OPENAI_API_KEY": ("dev_openai_api_key", "DEV_OPENAI_API_KEY"),
    "GROQ_API_KEY": ("dev_groq_api_key", "DEV_GROQ_API_KEY"),
}


def _mask(value: str, *, prefix: int = 4) -> str:
    if not value:
        return "<empty>"
    visible = value[:prefix]
    return f"{visible}{'…' if len(value) > prefix else ''}"


def _normalize_single(key: str) -> str | None:
    direct = os.getenv(key)
    if direct:
        return direct
    fallbacks = _VENDOR_KEYS.get(key, ())
    for env in fallbacks:
        candidate = os.getenv(env)
        if candidate:
            os.environ[key] = candidate
            print(
                f"[task:vendor] {key} set from {env} (prefix={_mask(candidate)})",
                flush=True,
            )
            return candidate
    return None


def normalize_vendor_keys() -> dict[str, str | None]:
    """Normalise known vendor keys from dev fallbacks and return the mapping."""

    resolved: dict[str, str | None] = {}
    for key in _VENDOR_KEYS:
        resolved[key] = _normalize_single(key)
    return resolved


def get_openai_key_or_503() -> str:
    key = _normalize_single("OPENAI_API_KEY")
    if not key:
        raise http_exception(503, "missing_openai_api_key", "OPENAI_API_KEY is not configured")
    return key


def get_groq_key_or_503() -> str:
    key = _normalize_single("GROQ_API_KEY")
    if not key:
        raise http_exception(503, "missing_groq_api_key", "GROQ_API_KEY is not configured")
    return key
