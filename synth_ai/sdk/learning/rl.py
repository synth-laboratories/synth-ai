"""Backward-compatible RL helpers for learning namespace."""

from __future__ import annotations

from synth_ai.sdk.localapi.auth import (  # noqa: F401
    mint_environment_api_key,
    setup_environment_api_key,
)

__all__ = [
    "mint_environment_api_key",
    "setup_environment_api_key",
]
