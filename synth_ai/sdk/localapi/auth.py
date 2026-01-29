"""Local API authentication helpers (Rust-backed with Python fallback)."""

from __future__ import annotations

import os
import secrets
from typing import Any

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for localapi.auth.") from exc

ENVIRONMENT_API_KEY_NAME = "ENVIRONMENT_API_KEY"
DEV_ENVIRONMENT_API_KEY_NAME = "DEV_ENVIRONMENT_API_KEY"
MAX_ENVIRONMENT_API_KEY_BYTES = 8 * 1024

__all__ = [
    "ENVIRONMENT_API_KEY_NAME",
    "DEV_ENVIRONMENT_API_KEY_NAME",
    "MAX_ENVIRONMENT_API_KEY_BYTES",
    "encrypt_for_backend",
    "ensure_localapi_auth",
    "mint_environment_api_key",
    "setup_environment_api_key",
]


def mint_environment_api_key() -> str:
    fn = getattr(synth_ai_py, "mint_environment_api_key", None)
    if callable(fn):
        return fn()
    # Fallback: generate a local-only API key
    return f"env_{secrets.token_urlsafe(24)}"


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    if isinstance(secret, bytes):
        secret = secret.decode("utf-8")
    fn = getattr(synth_ai_py, "encrypt_for_backend", None)
    if callable(fn):
        return fn(pubkey_b64, secret)
    raise RuntimeError("encrypt_for_backend requires synth_ai_py support")


def ensure_localapi_auth(
    backend_base: str | None = None,
    synth_api_key: str | None = None,
    *,
    upload: bool = True,
    persist: bool | None = None,
) -> str:
    fn = getattr(synth_ai_py, "ensure_localapi_auth", None)
    if callable(fn):
        return fn(backend_base, synth_api_key, upload, persist)

    # Fallback: use or mint a local environment key (no backend upload)
    existing = os.environ.get(ENVIRONMENT_API_KEY_NAME, "").strip()
    if existing:
        return existing
    key = mint_environment_api_key()
    os.environ[ENVIRONMENT_API_KEY_NAME] = key
    return key


def setup_environment_api_key(
    backend_base: str,
    synth_api_key: str,
    token: str | None = None,
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    fn = getattr(synth_ai_py, "setup_environment_api_key", None)
    if callable(fn):
        result = fn(backend_base, synth_api_key, token, timeout)
        return result if isinstance(result, dict) else {}
    raise RuntimeError("setup_environment_api_key requires synth_ai_py support")
