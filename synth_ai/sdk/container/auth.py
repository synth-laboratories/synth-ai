"""Container auth helpers used by live container and eval workflows."""

from __future__ import annotations

import base64
import os
import secrets
from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover - optional in the pure-Python runtime
    synth_ai_py = None  # type: ignore[assignment]

__all__ = [
    "_fetch_backend_env_key",
    "ensure_container_auth",
    "encrypt_for_backend",
    "has_container_token_signing_key",
]


def _decode_base64_key_material(value: str) -> bytes | None:
    raw = value.strip()
    if not raw:
        return None
    variants = (raw, raw + "=" * ((4 - len(raw) % 4) % 4))
    for variant in variants:
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                return decoder(variant.encode("utf-8"))
            except Exception:
                continue
    return None


def _parse_signing_key_entry(entry: str) -> bytes | None:
    raw = entry.strip()
    if not raw:
        return None
    if ":" in raw:
        _kid, encoded = raw.split(":", 1)
        decoded = _decode_base64_key_material(encoded)
        if decoded is not None:
            return decoded
    return _decode_base64_key_material(raw)


def has_container_token_signing_key() -> bool:
    raw_list = os.environ.get("SYNTH_CONTAINER_AUTH_PRIVATE_KEYS", "")
    for part in raw_list.split(","):
        if _parse_signing_key_entry(part):
            return True
    single = os.environ.get("SYNTH_CONTAINER_AUTH_PRIVATE_KEY", "")
    return _parse_signing_key_entry(single) is not None


def _select_environment_key_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    entries = payload.get("credentials")
    if not isinstance(entries, list):
        entries = payload.get("items")
    if not isinstance(entries, list):
        return None

    normalized: list[dict[str, str]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        plaintext = raw.get("plaintext")
        if not isinstance(plaintext, str) or not plaintext.strip():
            continue
        normalized.append(
            {
                "name": str(raw.get("name") or ""),
                "plaintext": plaintext.strip(),
                "created_at": str(raw.get("created_at") or ""),
            }
        )
    if not normalized:
        return None

    env_named = [entry for entry in normalized if entry["name"] == "ENVIRONMENT_API_KEY"]
    candidates = env_named or normalized
    dated = [entry for entry in candidates if entry["created_at"]]
    if dated:
        dated.sort(key=lambda item: item["created_at"], reverse=True)
        return dated[0]["plaintext"]
    return candidates[0]["plaintext"]


def _fetch_backend_env_key(backend_base: str, synth_api_key: str) -> str | None:
    base = (backend_base or "").strip().rstrip("/")
    key = (synth_api_key or "").strip()
    if not base or not key:
        return None
    try:
        import httpx
    except Exception:
        return None

    headers = {"Authorization": f"Bearer {key}"}
    urls = (
        f"{base}/v1/credentials",
        f"{base}/api/v1/credentials",
    )
    for url in urls:
        try:
            response = httpx.get(url, headers=headers, timeout=10.0)
        except Exception:
            continue
        if int(getattr(response, "status_code", 0)) == 404:
            return None
        if int(getattr(response, "status_code", 0)) >= 400:
            continue
        fetched = _select_environment_key_from_payload(response.json())
        if fetched:
            return fetched
    return None


def ensure_container_auth(
    *,
    backend_base: str | None = None,
    synth_api_key: str | None = None,
    upload: bool = True,
) -> str:
    existing = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if existing:
        return existing

    dev_existing = (os.environ.get("DEV_ENVIRONMENT_API_KEY") or "").strip()
    if dev_existing:
        os.environ["ENVIRONMENT_API_KEY"] = dev_existing
        return dev_existing

    resolved_backend = (backend_base or os.environ.get("SYNTH_BACKEND_URL") or "").strip()
    resolved_api_key = (synth_api_key or os.environ.get("SYNTH_API_KEY") or "").strip()
    if upload and resolved_backend and resolved_api_key:
        fetched = _fetch_backend_env_key(resolved_backend, resolved_api_key)
        if fetched:
            os.environ["ENVIRONMENT_API_KEY"] = fetched
            return fetched

    fn = getattr(synth_ai_py, "ensure_container_auth", None) if synth_ai_py else None
    if callable(fn):
        minted = fn(resolved_backend, resolved_api_key, upload)
        if isinstance(minted, str) and minted.strip():
            os.environ["ENVIRONMENT_API_KEY"] = minted.strip()
            return minted.strip()

    minted = f"env_{secrets.token_urlsafe(24)}"
    os.environ["ENVIRONMENT_API_KEY"] = minted
    return minted


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    if isinstance(secret, bytes):
        secret = secret.decode("utf-8")
    fn = getattr(synth_ai_py, "encrypt_for_backend", None) if synth_ai_py else None
    if callable(fn):
        return fn(pubkey_b64, secret)
    try:
        from nacl.public import PublicKey, SealedBox
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
