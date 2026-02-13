"""Local API authentication helpers (Rust-backed with Python fallback)."""

from __future__ import annotations

import base64
import os
import secrets
from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover - optional in minimal runtime images
    synth_ai_py = None  # type: ignore[assignment]

ENVIRONMENT_API_KEY_NAME = "ENVIRONMENT_API_KEY"
DEV_ENVIRONMENT_API_KEY_NAME = "DEV_ENVIRONMENT_API_KEY"
MAX_ENVIRONMENT_API_KEY_BYTES = 8 * 1024

__all__ = [
    "ENVIRONMENT_API_KEY_NAME",
    "DEV_ENVIRONMENT_API_KEY_NAME",
    "MAX_ENVIRONMENT_API_KEY_BYTES",
    "encrypt_for_backend",
    "ensure_container_auth",
    "mint_environment_api_key",
    "setup_environment_api_key",
]


def mint_environment_api_key() -> str:
    fn = getattr(synth_ai_py, "mint_environment_api_key", None) if synth_ai_py else None
    if callable(fn):
        return fn()
    # Fallback: generate a local-only API key
    return f"env_{secrets.token_urlsafe(24)}"


def _normalize_backend_base(backend_base: str | None) -> str:
    if not backend_base:
        return ""
    base = backend_base.rstrip("/")
    for suffix in ("/api/v1", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def _backend_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _fetch_backend_env_key(
    backend_base: str,
    synth_api_key: str,
    *,
    timeout: float = 15.0,
) -> str | None:
    import httpx

    base = _normalize_backend_base(backend_base)
    if not base:
        return None
    url = f"{base}/api/v1/env-keys"
    resp = httpx.get(url, headers=_backend_headers(synth_api_key), timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    credentials = payload.get("credentials") if isinstance(payload, dict) else None
    if not credentials:
        return None
    record = credentials[0]
    if isinstance(record, dict):
        plaintext = record.get("plaintext")
        if isinstance(plaintext, str) and plaintext.strip():
            return plaintext.strip()
    return None


def _backend_env_key_exists(
    backend_base: str,
    synth_api_key: str,
    *,
    timeout: float = 10.0,
) -> bool:
    import httpx

    base = _normalize_backend_base(backend_base)
    if not base:
        return False
    url = f"{base}/api/v1/env-keys/verify"
    resp = httpx.get(url, headers=_backend_headers(synth_api_key), timeout=timeout)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    if isinstance(secret, bytes):
        secret = secret.decode("utf-8")
    fn = getattr(synth_ai_py, "encrypt_for_backend", None) if synth_ai_py else None
    if callable(fn):
        return fn(pubkey_b64, secret)
    try:
        from nacl.public import PublicKey, SealedBox  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "encrypt_for_backend requires synth_ai_py or PyNaCl (pip install pynacl)"
        ) from exc
    try:
        pubkey_raw = base64.b64decode(pubkey_b64, validate=True)
    except Exception as exc:  # pragma: no cover - invalid base64
        raise RuntimeError("Invalid backend public key (not base64)") from exc
    box = SealedBox(PublicKey(pubkey_raw))
    ciphertext = box.encrypt(secret.encode("utf-8"))
    return base64.b64encode(ciphertext).decode("utf-8")


def ensure_container_auth(
    backend_base: str | None = None,
    synth_api_key: str | None = None,
    *,
    upload: bool = True,
    persist: bool | None = None,
) -> str:
    fn = getattr(synth_ai_py, "ensure_container_auth", None) if synth_ai_py else None
    if callable(fn):
        return fn(backend_base, synth_api_key, upload, persist)

    existing = os.environ.get(ENVIRONMENT_API_KEY_NAME, "").strip()
    backend_base = _normalize_backend_base(backend_base)

    if upload and backend_base and synth_api_key:
        if not existing:
            # Prefer reusing backend-stored env key to avoid mismatches.
            try:
                backend_key = _fetch_backend_env_key(backend_base, synth_api_key)
            except Exception:
                backend_key = None
            if backend_key:
                os.environ[ENVIRONMENT_API_KEY_NAME] = backend_key
                return backend_key

        # If backend already has a key and we don't have one, use it.
        if not existing:
            try:
                if _backend_env_key_exists(backend_base, synth_api_key):
                    backend_key = _fetch_backend_env_key(backend_base, synth_api_key)
                    if backend_key:
                        os.environ[ENVIRONMENT_API_KEY_NAME] = backend_key
                        return backend_key
            except Exception:
                pass

    if existing:
        if upload and backend_base and synth_api_key:
            backend_key = None
            try:
                backend_key = _fetch_backend_env_key(backend_base, synth_api_key)
            except Exception:
                backend_key = None
            if backend_key != existing:
                # Ensure backend matches the explicit key if provided locally.
                setup_environment_api_key(backend_base, synth_api_key, token=existing)
        return existing

    key = mint_environment_api_key()
    os.environ[ENVIRONMENT_API_KEY_NAME] = key
    if upload and backend_base and synth_api_key:
        setup_environment_api_key(backend_base, synth_api_key, token=key)
    return key


def setup_environment_api_key(
    backend_base: str,
    synth_api_key: str,
    token: str | None = None,
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    fn = getattr(synth_ai_py, "setup_environment_api_key", None) if synth_ai_py else None
    if callable(fn):
        result = fn(backend_base, synth_api_key, token, timeout)
        return result if isinstance(result, dict) else {}
    import httpx

    base = _normalize_backend_base(backend_base)
    if not base:
        raise ValueError("backend_base is required")
    if not synth_api_key:
        raise ValueError("synth_api_key is required")

    env_key = token or mint_environment_api_key()
    pubkey_url = f"{base}/api/v1/crypto/public-key"
    env_key_url = f"{base}/api/v1/env-keys"

    resp = httpx.get(pubkey_url, headers=_backend_headers(synth_api_key), timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    pubkey_b64 = payload.get("public_key") if isinstance(payload, dict) else None
    if not isinstance(pubkey_b64, str) or not pubkey_b64:
        raise RuntimeError("Backend public key missing in response")

    ciphertext_b64 = encrypt_for_backend(pubkey_b64, env_key)
    upsert_payload = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ciphertext_b64}
    upsert_resp = httpx.post(
        env_key_url,
        headers=_backend_headers(synth_api_key),
        json=upsert_payload,
        timeout=timeout,
    )
    upsert_resp.raise_for_status()
    os.environ.setdefault(ENVIRONMENT_API_KEY_NAME, env_key)
    result = upsert_resp.json()
    return result if isinstance(result, dict) else {}
