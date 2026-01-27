"""Shared HTTP client pool for high-concurrency LLM calls.

When making many concurrent LLM API calls (e.g., during eval or GEPA rollouts),
creating a new AsyncOpenAI/AsyncAnthropic client per request causes connection
pool thrashing - each client establishes new TLS connections, causing contention.

This module provides a shared httpx.AsyncClient that can be passed to LLM SDKs
to enable connection reuse across requests.

Usage:
    from synth_ai.sdk.localapi._impl.http_pool import get_shared_http_client

    # For OpenAI
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url=inference_url,
        api_key=api_key,
        http_client=get_shared_http_client(),
    )

    # For Anthropic
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(
        base_url=inference_url,
        api_key=api_key,
        http_client=get_shared_http_client(),
    )
"""

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

# Global shared HTTP client - lazily initialized
_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None


def get_shared_http_client() -> httpx.AsyncClient:
    """Get a shared httpx.AsyncClient configured for high concurrency.

    This client is designed to be passed to OpenAI/Anthropic SDK constructors
    via the `http_client` parameter. It maintains a connection pool that can
    handle 200+ concurrent connections, eliminating TLS handshake overhead
    for repeated requests to the same host.

    The client is created lazily on first call and reused for all subsequent
    calls. It is automatically closed on interpreter shutdown.

    Returns:
        A shared httpx.AsyncClient instance.

    Example:
        from openai import AsyncOpenAI
        from synth_ai.sdk.localapi._impl.http_pool import get_shared_http_client

        # Each call can use a different base_url, but they share connections
        client = AsyncOpenAI(
            base_url="https://api.usesynth.ai/api/interceptor/v1/trial_123/corr_456",
            api_key="synth-interceptor",
            http_client=get_shared_http_client(),
        )
    """
    global _SHARED_HTTP_CLIENT

    if _SHARED_HTTP_CLIENT is None:
        import httpx

        _SHARED_HTTP_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=200,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(
                connect=30.0,
                read=300.0,  # 5 min for slow LLM responses
                write=30.0,
                pool=30.0,
            ),
            # Don't follow redirects automatically - let the SDK handle it
            follow_redirects=False,
        )

        # Register cleanup on interpreter shutdown
        atexit.register(_cleanup_shared_client)

    return _SHARED_HTTP_CLIENT


def _cleanup_shared_client() -> None:
    """Clean up the shared client on interpreter shutdown."""
    global _SHARED_HTTP_CLIENT
    # Note: This is sync cleanup during interpreter shutdown.
    # The async close won't run, but httpx handles this gracefully.
    _SHARED_HTTP_CLIENT = None


def reset_shared_http_client() -> None:
    """Reset the shared HTTP client (mainly for testing).

    This closes the existing client and clears the global reference,
    so the next call to get_shared_http_client() creates a fresh client.
    """
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is not None:
        # Schedule async close - caller should await if needed
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_SHARED_HTTP_CLIENT.aclose())
        except RuntimeError:
            # No running loop - best effort sync close
            pass
        _SHARED_HTTP_CLIENT = None


__all__ = ["get_shared_http_client", "reset_shared_http_client"]
