"""Shared HTTP client pool for high-concurrency LLM calls."""

from __future__ import annotations

try:
    import synth_ai_py
except Exception:  # pragma: no cover
    synth_ai_py = None

_shared_client = None


def get_shared_http_client():
    """Get a shared httpx.AsyncClient configured for high concurrency."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "localapi_get_shared_http_client"):
        return synth_ai_py.localapi_get_shared_http_client()

    global _shared_client
    if _shared_client is None:
        import httpx

        _shared_client = httpx.AsyncClient(timeout=60.0)
    return _shared_client


def reset_shared_http_client() -> None:
    """Reset the shared HTTP client (mainly for testing)."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "localapi_reset_shared_http_client"):
        synth_ai_py.localapi_reset_shared_http_client()
        return

    global _shared_client
    _shared_client = None


__all__ = ["get_shared_http_client", "reset_shared_http_client"]
