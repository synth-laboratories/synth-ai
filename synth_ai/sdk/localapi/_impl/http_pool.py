"""Shared HTTP client pool for high-concurrency LLM calls."""

from __future__ import annotations

import synth_ai_py


def get_shared_http_client():
    """Get a shared httpx.AsyncClient configured for high concurrency."""

    return synth_ai_py.localapi_get_shared_http_client()


def reset_shared_http_client() -> None:
    """Reset the shared HTTP client (mainly for testing)."""

    synth_ai_py.localapi_reset_shared_http_client()


__all__ = ["get_shared_http_client", "reset_shared_http_client"]
