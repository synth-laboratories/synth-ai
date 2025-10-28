"""Helpers for uploading Environment credentials to the backend."""

from __future__ import annotations

import base64
import binascii
import json
import os
from typing import Any

import requests
from nacl.public import PublicKey, SealedBox

__all__ = ["encrypt_for_backend", "setup_environment_api_key", "MAX_ENVIRONMENT_API_KEY_BYTES"]

MAX_ENVIRONMENT_API_KEY_BYTES = 8 * 1024
_ALGORITHM = "libsodium.sealedbox.v1"


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    if not isinstance(pubkey_b64, str) or not pubkey_b64.strip():
        raise ValueError("public key must be a non-empty base64 string")

    try:
        key_bytes = base64.b64decode(pubkey_b64, validate=True)
    except binascii.Error as exc:
        raise ValueError("public key must be base64-encoded") from exc

    if len(key_bytes) != 32:
        raise ValueError("public key must be 32 bytes for X25519")

    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        secret_bytes = secret
    else:
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
) -> dict[str, Any]:
    backend = backend_base.rstrip("/")
    if not backend:
        raise ValueError("backend_base must be provided")
    if not synth_api_key:
        raise ValueError("synth_api_key must be provided")

    plaintext = token if token is not None else os.getenv("ENVIRONMENT_API_KEY", "").strip()
    if not plaintext:
        raise ValueError("ENVIRONMENT_API_KEY must be set (or pass token=...) to upload")
    if not isinstance(plaintext, str):
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
    except ValueError as exc:
        raise RuntimeError("backend returned invalid JSON for public key") from exc

    if not isinstance(doc, dict):
        raise RuntimeError("backend public key response must be an object")

    pubkey = doc.get("public_key")
    if not isinstance(pubkey, str) or not pubkey:
        raise RuntimeError("backend response missing public_key")

    alg = doc.get("alg")
    if alg is not None and alg != _ALGORITHM:
        raise RuntimeError(f"unsupported sealed box algorithm: {alg}")

    # Diagnostics: safe previews and hashes to correlate with backend logs
    try:
        import hashlib as _hash

        pk_bytes = base64.b64decode(pubkey, validate=True)
        pk_sha256 = _hash.sha256(pk_bytes).hexdigest()
        print(
            f"[env-keys] public_key: b64_len={len(pubkey)} sha256={pk_sha256} head={pubkey[:16]} tail={pubkey[-16:]}"
        )
        _plen = len(plaintext)
        _ppref = (plaintext[:6] + "…") if _plen > 10 else plaintext
        _psuf = ("…" + plaintext[-4:]) if _plen > 10 else ""
        _has_ws = any(ch.isspace() for ch in plaintext)
        print(
            f"[env-keys] plaintext: len={_plen} preview={_ppref}{_psuf} has_ws={bool(_has_ws)}"
        )
    except Exception:
        pass

    ciphertext_b64 = encrypt_for_backend(pubkey, token_bytes)

    body = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ciphertext_b64}
    post_url = f"{backend}/api/v1/env-keys"
    # Ciphertext diagnostics
    try:
        import hashlib as _hash

        _ct_bytes = base64.b64decode(ciphertext_b64, validate=True)
        _ct_sha = _hash.sha256(_ct_bytes).hexdigest()
        print(
            f"[env-keys] ciphertext: b64_len={len(ciphertext_b64)} sha256={_ct_sha} head={ciphertext_b64[:16]} tail={ciphertext_b64[-16:]}"
        )
    except Exception:
        pass

    response2 = requests.post(
        post_url,
        headers={**headers, "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
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
