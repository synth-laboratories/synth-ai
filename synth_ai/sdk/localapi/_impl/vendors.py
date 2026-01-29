"""Vendor API key helpers shared by Task Apps."""

from __future__ import annotations

import os

import synth_ai_py

from .errors import http_exception


def normalize_vendor_keys() -> dict[str, str | None]:
    """Normalise known vendor keys from dev fallbacks and return the mapping."""

    fn = getattr(synth_ai_py, "localapi_normalize_vendor_keys", None)
    if callable(fn):
        return fn()
    # Fallback: return current env values without modification.
    return {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }


def get_openai_key_or_503() -> str:
    fn = getattr(synth_ai_py, "localapi_get_openai_key", None)
    key = fn() if callable(fn) else os.environ.get("OPENAI_API_KEY")
    if not key:
        raise http_exception(503, "missing_openai_api_key", "OPENAI_API_KEY is not configured")
    return key


def get_groq_key_or_503() -> str:
    fn = getattr(synth_ai_py, "localapi_get_groq_key", None)
    key = fn() if callable(fn) else os.environ.get("GROQ_API_KEY")
    if not key:
        raise http_exception(503, "missing_groq_api_key", "GROQ_API_KEY is not configured")
    return key
