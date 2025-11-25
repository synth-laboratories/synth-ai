"""Deprecated: Use synth_ai.core.env instead.

This module is kept for backward compatibility.
"""

import warnings

warnings.warn(
    "synth_ai.config.base_url is deprecated. Use synth_ai.core.env instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location
from synth_ai.core.env import (
    PROD_BASE_URL,
    get_backend_from_env,
    get_backend_url,
)

# Legacy name
PROD_BASE_URL_DEFAULT = PROD_BASE_URL


def get_learning_v2_base_url(mode: str = "prod") -> str:
    """Deprecated: Use get_backend_url() instead."""
    url = get_backend_url(mode if mode in ("prod", "dev", "local") else "prod")  # type: ignore
    # Legacy behavior: return with /api suffix
    return f"{url}/api"


def _resolve_override_mode() -> str:
    """Deprecated: Use get_backend_from_env() instead."""
    import os
    ov = (os.getenv("SYNTH_BACKEND_URL_OVERRIDE", "") or "").strip().lower()
    if ov in {"local", "dev", "prod"}:
        return ov
    return "prod"


__all__ = [
    "PROD_BASE_URL_DEFAULT",
    "PROD_BASE_URL",
    "get_learning_v2_base_url",
    "get_backend_from_env",
    "_resolve_override_mode",
]
