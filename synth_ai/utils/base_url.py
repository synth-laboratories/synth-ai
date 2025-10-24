"""
Base URL resolution for learning-v2 and related backend APIs.

Default to production, allow overrides via environment variables:
- LEARNING_V2_BASE_URL (highest precedence)
- SYNTH_BASE_URL (legacy)
- SYNTH_LOCAL_BASE_URL
- SYNTH_DEV_BASE_URL
- SYNTH_PROD_BASE_URL (fallback if none provided)

Normalization: ensure the returned URL ends with "/api".
"""

import os
from typing import Literal

PROD_BASE_URL_DEFAULT = "https://agent-learning.onrender.com"


def _normalize_base(url: str) -> str:
    url = url.strip()
    if url.endswith("/v1"):
        url = url[:-3]
    url = url.rstrip("/")
    if not url.endswith("/api"):
        url = f"{url}/api"
    return url


def get_learning_v2_base_url(mode: Literal["dev", "prod"] = "prod") -> str:
    if mode == "prod":
        prod = os.getenv("SYNTH_PROD_BASE_URL") or PROD_BASE_URL_DEFAULT
        return _normalize_base(prod)
    env_url = os.getenv("LEARNING_V2_BASE_URL")
    if env_url:
        return _normalize_base(env_url)

    legacy = os.getenv("SYNTH_BASE_URL")
    if legacy:
        return _normalize_base(legacy)

    local = os.getenv("SYNTH_LOCAL_BASE_URL")
    if local:
        return _normalize_base(local)

    dev = os.getenv("SYNTH_DEV_BASE_URL")
    if dev:
        return _normalize_base(dev)

    raise ValueError("No base URL configured. Set one of: LEARNING_V2_BASE_URL, SYNTH_BASE_URL, SYNTH_LOCAL_BASE_URL, SYNTH_DEV_BASE_URL")


def _resolve_override_mode() -> str:
    ov = (os.getenv("SYNTH_BACKEND_URL_OVERRIDE", "") or "").strip().lower()
    if ov in {"local", "dev", "prod"}:
        return ov
    return "prod"


def get_backend_from_env() -> tuple[str, str]:
    direct_override = (os.getenv("BACKEND_OVERRIDE") or "").strip()
    if direct_override:
        base = direct_override.rstrip("/")
        if base.endswith("/api"):
            base = base[: -len("/api")]
        api_key = os.getenv("SYNTH_API_KEY", "").strip()
        return base, api_key

    mode = _resolve_override_mode()
    if mode == "local":
        base = os.getenv("LOCAL_BACKEND_URL", "http://localhost:8000")
        key = os.getenv("TESTING_LOCAL_SYNTH_API_KEY", "")
        return base.rstrip("/"), key
    if mode == "dev":
        base = os.getenv("DEV_BACKEND_URL", "") or "http://localhost:8000"
        key = os.getenv("DEV_SYNTH_API_KEY", "")
        return base.rstrip("/"), key
    base = os.getenv("PROD_BACKEND_URL", f"{PROD_BASE_URL_DEFAULT}")
    base = base.rstrip("/")
    if base.endswith("/api"):
        base = base[: -len("/api")]
    key = (
        os.getenv("PROD_SYNTH_API_KEY", "")
        or os.getenv("TESTING_PROD_SYNTH_API_KEY", "")
        or os.getenv("SYNTH_API_KEY", "")
    )
    return base, key


__all__ = [
    "PROD_BASE_URL_DEFAULT",
    "get_backend_from_env",
    "get_learning_v2_base_url",
]
