"""Configuration utilities for the status command suite.

Provides helpers to resolve backend URLs, API keys, and request timeouts
from CLI options and environment variables.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_TIMEOUT = 30.0


def _load_backend_helpers() -> tuple[str, Callable[[], tuple[str, str]] | None]:
    """Attempt to load shared backend helpers from synth_ai.core.env."""
    try:
        module = importlib.import_module("synth_ai.core.env")
    except Exception:
        return "https://agent-learning.onrender.com", None

    default = getattr(module, "PROD_BASE_URL_DEFAULT", "https://agent-learning.onrender.com")
    getter = getattr(module, "get_backend_from_env", None)
    return str(default), getter if callable(getter) else None


PROD_BASE_URL_DEFAULT, _GET_BACKEND_FROM_ENV = _load_backend_helpers()


def _normalize_base_url(raw: str) -> str:
    """Ensure the configured base URL includes the /api/v1 prefix."""
    base = raw.rstrip("/") if raw else ""
    if not base:
        return raw
    if base.endswith("/api") or base.endswith("/api/v1") or "/api/" in base:
        return base
    return f"{base}/api/v1"


def _default_base_url() -> str:
    """Compute the default backend base URL using env vars or helper module."""
    for var in ("SYNTH_BACKEND_BASE_URL", "BACKEND_BASE_URL", "SYNTH_BASE_URL"):
        val = os.getenv(var)
        if val:
            return _normalize_base_url(val)
    if _GET_BACKEND_FROM_ENV:
        try:
            base, _ = _GET_BACKEND_FROM_ENV()
            return _normalize_base_url(base)
        except Exception:
            pass
    return _normalize_base_url(PROD_BASE_URL_DEFAULT)


def _resolve_api_key(cli_key: str | None) -> tuple[str | None, str | None]:
    """Resolve the API key from CLI input or known environment variables."""
    if cli_key:
        return cli_key, "--api-key"
    for var in ("SYNTH_BACKEND_API_KEY", "SYNTH_API_KEY", "DEFAULT_DEV_API_KEY"):
        val = os.getenv(var)
        if val:
            return val, var
    return None, None


@dataclass()
class BackendConfig:
    """Configuration bundle shared across status commands."""

    base_url: str
    api_key: str | None
    timeout: float = DEFAULT_TIMEOUT

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


def resolve_backend_config(
    *,
    base_url: str | None,
    api_key: str | None,
    timeout: float | None = None,
) -> BackendConfig:
    """Resolve the backend configuration from CLI options/environment."""
    resolved_url = _normalize_base_url(base_url) if base_url else _default_base_url()
    key, _ = _resolve_api_key(api_key)
    return BackendConfig(base_url=resolved_url, api_key=key, timeout=timeout or DEFAULT_TIMEOUT)
