"""Python-only front-door SDK clients."""

from __future__ import annotations

import os

from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk import (
    AsyncContainerPoolsClient,
    AsyncContainersClient,
    AsyncTunnelsClient,
    ContainerPoolsClient,
    ContainersClient,
    TunnelsClient,
)


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or os.getenv("SYNTH_API_KEY") or "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
    return resolved


def _resolve_base_url(base_url: str | None) -> str:
    return normalize_backend_base(base_url or BACKEND_URL_BASE)


class SynthClient:
    """Sync client for containers, tunnels, and container pools."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout
        self.containers = ContainersClient(
            api_key=self.api_key,
            backend_base=self.base_url,
        )
        self.tunnels = TunnelsClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout=self.timeout,
        )
        self.pools = ContainerPoolsClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout=self.timeout,
        )


class AsyncSynthClient:
    """Async client for containers, tunnels, and container pools."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout
        self.containers = AsyncContainersClient(
            ContainersClient(
                api_key=self.api_key,
                backend_base=self.base_url,
            )
        )
        self.tunnels = AsyncTunnelsClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout=self.timeout,
        )
        self.pools = AsyncContainerPoolsClient(
            ContainerPoolsClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout=self.timeout,
            )
        )


__all__ = [
    "AsyncContainerPoolsClient",
    "AsyncContainersClient",
    "AsyncSynthClient",
    "AsyncTunnelsClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "SynthClient",
    "TunnelsClient",
]
