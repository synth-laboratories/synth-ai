"""Vendor API key helpers shared by Task Apps."""

from __future__ import annotations

import synth_ai_py

from .errors import http_exception


def normalize_vendor_keys() -> dict[str, str | None]:
    """Normalise known vendor keys from dev fallbacks and return the mapping."""

    return synth_ai_py.localapi_normalize_vendor_keys()


def get_openai_key_or_503() -> str:
    key = synth_ai_py.localapi_get_openai_key()
    if not key:
        raise http_exception(503, "missing_openai_api_key", "OPENAI_API_KEY is not configured")
    return key


def get_groq_key_or_503() -> str:
    key = synth_ai_py.localapi_get_groq_key()
    if not key:
        raise http_exception(503, "missing_groq_api_key", "GROQ_API_KEY is not configured")
    return key
