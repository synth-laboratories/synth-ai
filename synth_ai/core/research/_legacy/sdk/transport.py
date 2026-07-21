"""HTTP transport helpers for the Managed Research SDK."""

from __future__ import annotations

from synth_ai.core.research._legacy.sdk.config import auth_headers
from synth_ai.core.research._legacy.transport.http import SmrHttpTransport


def build_http_transport(
    *,
    api_key: str,
    backend_base: str,
    timeout_seconds: float,
) -> SmrHttpTransport:
    return SmrHttpTransport(
        base_url=backend_base,
        headers=auth_headers(api_key),
        timeout=timeout_seconds,
    )


__all__ = ["build_http_transport"]
