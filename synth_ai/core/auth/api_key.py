"""API key resolution for SDK usage."""

from __future__ import annotations

import os
from collections.abc import Callable

from synth_ai.core.utils.env import mint_demo_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_base_url

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for api key resolution.") from exc

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
    if ttl_hours == 4:
        api_key = synth_ai_py.get_or_mint_api_key(
            backend_url=backend_url,
            allow_mint=allow_mint,
        )
    else:
        api_key = synth_ai_py.get_api_key(env_key)
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
