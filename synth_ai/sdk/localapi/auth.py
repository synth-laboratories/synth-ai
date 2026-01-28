"""Local API authentication helpers (Rust-backed)."""

from __future__ import annotations

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
    return synth_ai_py.mint_environment_api_key()


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    if isinstance(secret, bytes):
        secret = secret.decode("utf-8")
    return synth_ai_py.encrypt_for_backend(pubkey_b64, secret)


def ensure_localapi_auth(
    backend_base: str | None = None,
    synth_api_key: str | None = None,
    *,
    upload: bool = True,
    persist: bool | None = None,
) -> str:
    return synth_ai_py.ensure_localapi_auth(backend_base, synth_api_key, upload, persist)


def setup_environment_api_key(
    backend_base: str,
    synth_api_key: str,
    token: str | None = None,
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    result = synth_ai_py.setup_environment_api_key(backend_base, synth_api_key, token, timeout)
    return result if isinstance(result, dict) else {}
