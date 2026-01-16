"""Local API authentication helpers."""

import base64
import binascii
import json
import logging
import os
import secrets
from typing import Any

from synth_ai.core.urls import synth_env_keys_url, synth_public_key_url
from synth_ai.core.user_config import load_user_env, update_user_config

ENVIRONMENT_API_KEY_NAME = "ENVIRONMENT_API_KEY"
DEV_ENVIRONMENT_API_KEY_NAME = "DEV_ENVIRONMENT_API_KEY"
MAX_ENVIRONMENT_API_KEY_BYTES = 8 * 1024
_ALGORITHM = "libsodium.sealedbox.v1"

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
    """Mint a random ENVIRONMENT_API_KEY value."""

    return secrets.token_hex(32)


def ensure_localapi_auth(
    synth_user_key: str | None = None,
    *,
    upload: bool = True,
    persist: bool | None = None,
    synth_base_url: str | None = None,
) -> str:
    """Ensure ENVIRONMENT_API_KEY is present and optionally registered."""

    load_user_env(override=False)

    localapi_key = (os.environ.get(ENVIRONMENT_API_KEY_NAME) or "").strip()
    if not localapi_key:
        localapi_key = (os.environ.get(DEV_ENVIRONMENT_API_KEY_NAME) or "").strip()
        if localapi_key:
            os.environ[ENVIRONMENT_API_KEY_NAME] = localapi_key

    minted = False
    if not localapi_key:
        localapi_key = mint_environment_api_key()
        if not localapi_key:
            raise RuntimeError("Failed to mint ENVIRONMENT_API_KEY")
        os.environ[ENVIRONMENT_API_KEY_NAME] = localapi_key
        minted = True

    os.environ.setdefault(DEV_ENVIRONMENT_API_KEY_NAME, localapi_key)

    if persist is None:
        persist = os.environ.get("SYNTH_LOCALAPI_AUTH_PERSIST", "1") != "0"

    if minted and persist:
        update_user_config(
            {
                ENVIRONMENT_API_KEY_NAME: localapi_key,
                DEV_ENVIRONMENT_API_KEY_NAME: localapi_key,
            }
        )

    if upload:
        if synth_user_key is None:
            synth_user_key = os.environ.get("SYNTH_API_KEY")
        if synth_user_key:
            try:
                setup_environment_api_key(
                    synth_user_key=synth_user_key,
                    localapi_key=localapi_key,
                    synth_base_url=synth_base_url,
                )
            except Exception as exc:
                logger = logging.getLogger(__name__)
                logger.warning("Failed to upload ENVIRONMENT_API_KEY to backend: %s", exc)

    if not localapi_key:
        raise RuntimeError("ENVIRONMENT_API_KEY is required but missing")

    return localapi_key


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    from nacl.public import PublicKey, SealedBox

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
    synth_user_key: str,
    localapi_key: str | None = None,
    *,
    timeout: float = 15.0,
    synth_base_url: str | None = None,
) -> dict[str, Any]:
    import requests

    if not synth_user_key:
        raise ValueError("synth_user_key must be provided")

    plaintext = (
        localapi_key
        if localapi_key is not None
        else os.getenv(ENVIRONMENT_API_KEY_NAME, "").strip()
    )
    if not plaintext:
        raise ValueError("ENVIRONMENT_API_KEY must be set (or pass localapi_key=...) to upload")
    if not isinstance(plaintext, str):
        raise TypeError("localapi_key must be a string")

    token_bytes = plaintext.encode("utf-8")
    if not token_bytes:
        raise ValueError("ENVIRONMENT_API_KEY token must not be empty")
    if len(token_bytes) > MAX_ENVIRONMENT_API_KEY_BYTES:
        raise ValueError("ENVIRONMENT_API_KEY token exceeds 8 KiB limit")

    headers = {"Authorization": f"Bearer {synth_user_key}"}
    pub_url = synth_public_key_url(synth_base_url)
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

    ciphertext_b64 = encrypt_for_backend(pubkey, token_bytes)

    body = {"name": ENVIRONMENT_API_KEY_NAME, "ciphertext_b64": ciphertext_b64}
    post_url = synth_env_keys_url(synth_base_url)

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


def _raise_with_detail(response: Any) -> None:
    import requests

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
