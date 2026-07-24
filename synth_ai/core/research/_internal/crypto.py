"""Lightweight crypto helpers used by the SDK."""

from __future__ import annotations

import base64
import binascii

from nacl.public import PublicKey, SealedBox


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    """Encrypt a provider secret with the backend sealed-box public key."""

    secret_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")
    try:
        public_key = base64.b64decode(pubkey_b64, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise RuntimeError("Invalid backend public key (not base64)") from exc
    ciphertext = SealedBox(PublicKey(public_key)).encrypt(secret_bytes)
    return base64.b64encode(ciphertext).decode("utf-8")


__all__ = ["encrypt_for_backend"]
