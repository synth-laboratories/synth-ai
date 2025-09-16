from __future__ import annotations

"""Helpers for uploading RL environment credentials to the backend."""

import base64
import binascii
import json
from typing import Any, Dict
import os

import requests
from nacl.public import PublicKey, SealedBox

__all__ = ["encrypt_for_backend", "setup_environment_api_key", "MAX_ENVIRONMENT_API_KEY_BYTES"]

MAX_ENVIRONMENT_API_KEY_BYTES = 8 * 1024
_ALGORITHM = "libsodium.sealedbox.v1"


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    """Encrypt ``secret`` for storage by the backend using libsodium sealed boxes."""

    if not isinstance(pubkey_b64, str) or not pubkey_b64.strip():
        raise ValueError("public key must be a non-empty base64 string")

    try:
        key_bytes = base64.b64decode(pubkey_b64, validate=True)
    except binascii.Error as exc:  # pragma: no cover - defensive guard
        raise ValueError("public key must be base64-encoded") from exc

    if len(key_bytes) != 32:
        raise ValueError("public key must be 32 bytes for X25519")

    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        secret_bytes = secret
    else:  # pragma: no cover - type guard
        raise TypeError("secret must be str or bytes")

    if not secret_bytes:
        raise ValueError("secret must not be empty")

    box = SealedBox(PublicKey(key_bytes))
    ciphertext = box.encrypt(secret_bytes)
    return base64.b64encode(ciphertext).decode("ascii")


def setup_environment_api_key(
    backend_base: str,
    synth_api_key: str,
    token: str | None = None,
    *,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Upload an ENVIRONMENT_API_KEY to the backend."""

    backend = backend_base.rstrip("/")
    if not backend:
        raise ValueError("backend_base must be provided")
    if not synth_api_key:
        raise ValueError("synth_api_key must be provided")

    # Require caller-provided plaintext. If not provided, read from ENVIRONMENT_API_KEY.
    plaintext = token if token is not None else os.getenv("ENVIRONMENT_API_KEY", "").strip()
    if not plaintext:
        raise ValueError("ENVIRONMENT_API_KEY must be set (or pass token=...) to upload")
    if not isinstance(plaintext, str):  # pragma: no cover - defensive guard
        raise TypeError("token must be a string")

    token_bytes = plaintext.encode("utf-8")
    if not token_bytes:
        raise ValueError("ENVIRONMENT_API_KEY token must not be empty")
    if len(token_bytes) > MAX_ENVIRONMENT_API_KEY_BYTES:
        raise ValueError("ENVIRONMENT_API_KEY token exceeds 8 KiB limit")

    headers = {"Authorization": f"Bearer {synth_api_key}"}
    pub_url = f"{backend}/api/v1/crypto/public-key"
    response = requests.get(pub_url, headers=headers, timeout=timeout)
    _raise_with_detail(response)

    try:
        doc = response.json()
    except ValueError as exc:  # pragma: no cover - backend invariant
        raise RuntimeError("backend returned invalid JSON for public key") from exc

    if not isinstance(doc, dict):
        raise RuntimeError("backend public key response must be an object")

    pubkey = doc.get("public_key")
    if not isinstance(pubkey, str) or not pubkey:
        raise RuntimeError("backend response missing public_key")

    # The backend currently returns a single algorithm identifier; keep a guard in
    # case future versions change the value and we need to surface that to callers.
    alg = doc.get("alg")
    if alg is not None and alg != _ALGORITHM:
        raise RuntimeError(f"unsupported sealed box algorithm: {alg}")

    ciphertext_b64 = encrypt_for_backend(pubkey, token_bytes)

    body = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ciphertext_b64}
    post_url = f"{backend}/api/v1/env-keys"
    response2 = requests.post(post_url, headers={**headers, "Content-Type": "application/json"}, json=body, timeout=timeout)
    _raise_with_detail(response2)

    try:
        upload_doc = response2.json()
    except ValueError:
        upload_doc = {}

    if not isinstance(upload_doc, dict):
        upload_doc = {}

    return {
        "stored": True,
        "id": upload_doc.get("id"),
        "name": upload_doc.get("name"),
        "updated_at": upload_doc.get("updated_at"),
    }


def _raise_with_detail(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail_snippet: str | None = None
        try:
            detail = response.json()
            detail_snippet = json.dumps(detail, separators=(",", ":"))[:200]
        except Exception:
            body = response.text if response.text is not None else ""
            detail_snippet = body[:200] if body else None
        message = str(exc)
        if detail_snippet:
            message = f"{message} | body={detail_snippet}"
        raise requests.HTTPError(message, request=exc.request, response=exc.response) from None
