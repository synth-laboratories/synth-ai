"""Environment helpers for langprobe integrations.

This centralizes backend/task app URL resolution so we can quickly toggle
between local development (localhost) and production (agent-learning).
"""

from __future__ import annotations

import os
from typing import Optional

from synth_ai.config.base_url import get_backend_from_env

# Canonical defaults for the blogpost experiments
DEFAULT_BACKENDS: dict[str, str] = {
    "local": "http://localhost:8000",
    "prod": "https://agent-learning.onrender.com",
}

# Known task app defaults (extend as new benchmarks are added)
DEFAULT_TASK_APP_URLS: dict[str, str] = {
    "iris": "http://127.0.0.1:8115",
}


def normalize_backend_url(url: str) -> str:
    """Strip trailing slashes and any trailing /api segment."""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/api"):
        cleaned = cleaned[: -len("/api")]
    return cleaned


def is_local_url(url: str) -> bool:
    """Return True if URL points to localhost/127.*."""
    lowered = url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def resolve_backend_url(
    backend_url: Optional[str] = None,
    backend_env: Optional[str] = None,
) -> str:
    """Resolve backend base URL with precedence: explicit > env hint > synth defaults.

    backend_env accepts local|prod (case-insensitive). When unset, falls back to:
    - BACKEND_BASE_URL (legacy override)
    - LANGPROBE_BACKEND_ENV
    - SYNTH_BACKEND_URL_OVERRIDE (local|dev|prod understood by synth config)
    - synth_ai.config.base_url.get_backend_from_env()
    """
    # Explicit env override that was historically used in scripts
    if not backend_url and os.getenv("BACKEND_BASE_URL"):
        backend_url = os.getenv("BACKEND_BASE_URL")

    if backend_url:
        return normalize_backend_url(backend_url)

    env_hint = (
        backend_env
        or os.getenv("LANGPROBE_BACKEND_ENV")
        or os.getenv("SYNTH_BACKEND_URL_OVERRIDE")
        or ""
    ).strip().lower()

    if env_hint in {"local", "localhost"}:
        return DEFAULT_BACKENDS["local"]
    if env_hint in {"prod", "production"}:
        return DEFAULT_BACKENDS["prod"]

    base, _ = get_backend_from_env()
    return normalize_backend_url(base)


def resolve_task_app_url(default: str, override: Optional[str] = None) -> str:
    """Resolve task app URL with optional overrides."""
    url = override or os.getenv("LANGPROBE_TASK_APP_URL") or default
    return url.rstrip("/")


def should_auto_tunnel(
    auto_tunnel: Optional[bool],
    backend_url: str,
    task_app_url: str,
) -> bool:
    """Decide whether to create a tunnel for the task app.

    Defaults to True when the backend is remote but the task app lives on localhost.
    Explicit True/False wins over detection.
    """
    if auto_tunnel is not None:
        return auto_tunnel

    backend_is_local = is_local_url(backend_url)
    task_app_is_local = is_local_url(task_app_url)
    return task_app_is_local and not backend_is_local
