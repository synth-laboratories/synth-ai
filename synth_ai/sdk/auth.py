"""User-facing helpers for API key resolution."""

import os
from collections.abc import Callable

from synth_ai.core.env import mint_demo_api_key

Validator = Callable[[str], bool]


def get_or_mint_synth_user_key(
    *,
    env_key: str = "SYNTH_API_KEY",
    ttl_hours: int = 4,
    allow_mint: bool = True,
    validator: Validator | None = None,
    set_env: bool = True,
    synth_base_url: str | None = None,
) -> str:
    """Resolve a Synth API key from the environment or mint a demo key."""
    synth_user_key = (os.environ.get(env_key) or "").strip()
    if synth_user_key and (validator is None or validator(synth_user_key)):
        if set_env:
            os.environ[env_key] = synth_user_key
        return synth_user_key

    if not allow_mint:
        raise RuntimeError(f"{env_key} is required but missing.")

    synth_user_key = mint_demo_api_key(ttl_hours=ttl_hours, synth_base_url=synth_base_url)
    if set_env:
        os.environ[env_key] = synth_user_key
    return synth_user_key


__all__ = ["get_or_mint_synth_user_key"]
