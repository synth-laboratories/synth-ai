"""User-facing helpers for API key resolution."""

import os
from collections.abc import Callable

from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE, normalize_base_url

Validator = Callable[[str], bool]


def get_or_mint_synth_api_key(
    *,
    backend_url: str | None = None,
    env_key: str = "SYNTH_API_KEY",
    ttl_hours: int = 4,
    allow_mint: bool = True,
    validator: Validator | None = None,
    set_env: bool = True,
) -> str:
    """Resolve a Synth API key from the environment or mint a demo key."""
    api_key = (os.environ.get(env_key) or "").strip()
    if api_key and (validator is None or validator(api_key)):
        if set_env:
            os.environ[env_key] = api_key
        return api_key

    if not allow_mint:
        raise RuntimeError(f"{env_key} is required but missing.")

    resolved_backend = normalize_base_url(backend_url or BACKEND_URL_BASE)
    api_key = mint_demo_api_key(backend_url=resolved_backend, ttl_hours=ttl_hours)
    if set_env:
        os.environ[env_key] = api_key
    return api_key


__all__ = ["get_or_mint_synth_api_key"]
