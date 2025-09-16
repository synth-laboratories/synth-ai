from __future__ import annotations

"""Helpers for generating RL environment credentials."""

import secrets

__all__ = ["mint_environment_api_key"]


def mint_environment_api_key() -> str:
    """Mint a random ENVIRONMENT_API_KEY value.

    The current format is 64 hexadecimal characters (256 bits of entropy), which
    matches the shell helpers used by the RL examples. This keeps the token easy
    to copy while remaining suitably strong for authentication.
    """

    # secrets.token_hex(32) → 32 random bytes rendered as 64 hex characters.
    return secrets.token_hex(32)
