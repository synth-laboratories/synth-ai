"""Python-only front-door SDK clients."""

from __future__ import annotations

import os

from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk import (
    AsyncContainerPoolsClient,
    AsyncContainersClient,
    AsyncHorizonsPrivateClient,
    AsyncManagedAgentsAnthropicClient,
    AsyncOpenAIAgentsSdkClient,
    AsyncTunnelsClient,
    ContainerPoolsClient,
    ContainersClient,
    HorizonsPrivateClient,
    ManagedAgentsAnthropicClient,
    OpenAIAgentsSdkClient,
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
    """Sync client for containers, tunnels, pools, and compat surfaces."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        openai_transport_mode: str = "auto",
        openai_organization: str | None = None,
        openai_project: str | None = None,
        openai_request_id: str | None = None,
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
        self.horizons_private = HorizonsPrivateClient(self.pools)
        self.managed_agents = ManagedAgentsAnthropicClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout=self.timeout,
        )
        self.openai_agents_sdk = OpenAIAgentsSdkClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout=self.timeout,
            transport_mode=openai_transport_mode,
            openai_organization=openai_organization,
            openai_project=openai_project,
            request_id=openai_request_id,
        )


class AsyncSynthClient:
    """Async client for containers, tunnels, pools, and compat surfaces."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        openai_transport_mode: str = "auto",
        openai_organization: str | None = None,
        openai_project: str | None = None,
        openai_request_id: str | None = None,
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
        self.horizons_private = AsyncHorizonsPrivateClient(
            HorizonsPrivateClient(self.pools.raw)
        )
        self.managed_agents = AsyncManagedAgentsAnthropicClient(
            ManagedAgentsAnthropicClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout=self.timeout,
            )
        )
        self.openai_agents_sdk = AsyncOpenAIAgentsSdkClient(
            OpenAIAgentsSdkClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout=self.timeout,
                transport_mode=openai_transport_mode,
                openai_organization=openai_organization,
                openai_project=openai_project,
                request_id=openai_request_id,
            )
        )


__all__ = [
    "AsyncContainerPoolsClient",
    "AsyncContainersClient",
    "AsyncHorizonsPrivateClient",
    "AsyncManagedAgentsAnthropicClient",
    "AsyncOpenAIAgentsSdkClient",
    "AsyncSynthClient",
    "AsyncTunnelsClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "HorizonsPrivateClient",
    "ManagedAgentsAnthropicClient",
    "OpenAIAgentsSdkClient",
    "SynthClient",
    "TunnelsClient",
]
