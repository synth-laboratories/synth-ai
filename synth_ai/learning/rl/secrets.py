"""Helpers for generating Environment credentials."""

from __future__ import annotations

import secrets

__all__ = ["mint_environment_api_key"]


def mint_environment_api_key() -> str:
    """Mint a random ENVIRONMENT_API_KEY value."""

    return secrets.token_hex(32)
