"""Helpers for generating RL environment credentials."""

from __future__ import annotations

import secrets

__all__ = ["mint_environment_api_key"]


def mint_environment_api_key() -> str:
    """Mint a random ENVIRONMENT_API_KEY value."""

    # Use a human-recognisable prefix so logs surface the key type while keeping the suffix random.
    # 16 random bytes → 32 hex characters; total length stays compact (39 chars with prefix).
    return f"sk_env_{secrets.token_hex(16)}"
