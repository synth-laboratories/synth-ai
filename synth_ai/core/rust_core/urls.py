from __future__ import annotations

from synth_ai.core.utils.urls import normalize_base_url as _normalize_base_url

try:
    import synth_ai_py as _synth_ai_py
except Exception:  # pragma: no cover - optional rust bindings
    _synth_ai_py = None


def normalize_base_url(url: str) -> str:
    """Normalize backend base URL (no trailing /api or /v1)."""
    if _synth_ai_py is not None:
        return _synth_ai_py.normalize_backend_base(url)
    return _normalize_base_url(url)


def ensure_api_base(base_url: str) -> str:
    """Ensure the URL includes /api suffix exactly once."""
    normalized = normalize_base_url(base_url).rstrip("/")
    return f"{normalized}/api"


__all__ = ["normalize_base_url", "ensure_api_base"]
