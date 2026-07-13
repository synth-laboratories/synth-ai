"""Python-only front-door SDK clients."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk import (
    AsyncContainerPoolsClient,
    AsyncContainersClient,
    AsyncHorizonsPrivateClient,
    AsyncManagedAgentsAnthropicClient,
    AsyncOpenAIAgentsSdkClient,
    AsyncOptimizersClient,
    AsyncSynthManagedAgents,
    AsyncTunnelsClient,
    AsyncWorkflowsClient,
    ContainerPoolsClient,
    ContainersClient,
    HorizonsPrivateClient,
    ManagedAgentsAnthropicClient,
    OpenAIAgentsSdkClient,
    OptimizersClient,
    SynthManagedAgents,
    TunnelsClient,
    WorkflowsClient,
)

if TYPE_CHECKING:
    from synth_ai.research.async_client import AsyncResearchClient
    from synth_ai.research.client import ResearchClient


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or get_api_key(required=False) or "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
    return resolved


def _resolve_base_url(base_url: str | None) -> str:
    return normalize_backend_base(base_url or BACKEND_URL_BASE)


class SynthClient:
    """Sync client for containers, tunnels, pools, and Managed Research.

    Use ``research`` for hosted runs, projects, limits, and Factory Tag.
    Infrastructure namespaces: ``containers``, ``tunnels``, ``pools``.
    """

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
            timeout_seconds=self.timeout,
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
        self.optimizers = OptimizersClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout_seconds=self.timeout,
        )
        self.workflows = WorkflowsClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout_seconds=self.timeout,
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
        self._research_client: ResearchClient | None = None
        self._horizons_private: HorizonsPrivateClient | None = None
        self._managed_agents: ManagedAgentsAnthropicClient | None = None
        self._managed_agents_anthropic: SynthManagedAgents | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "horizons_private":
            warnings.warn(
                "SynthClient.horizons_private is deprecated; use client.pools.rollouts instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self._horizons_private is None:
                self._horizons_private = HorizonsPrivateClient(self.pools)
            return self._horizons_private
        if name == "managed_agents":
            warnings.warn(
                "SynthClient.managed_agents is deprecated; import ManagedAgentsAnthropicClient explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self._managed_agents is None:
                self._managed_agents = ManagedAgentsAnthropicClient(
                    api_key=self.api_key,
                    backend_base=self.base_url,
                    timeout=self.timeout,
                )
            return self._managed_agents
        if name == "managed_agents_anthropic":
            warnings.warn(
                "SynthClient.managed_agents_anthropic is deprecated; import SynthManagedAgents explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self._managed_agents_anthropic is None:
                self._managed_agents_anthropic = SynthManagedAgents.from_transport(
                    ManagedAgentsAnthropicClient(
                        api_key=self.api_key,
                        backend_base=self.base_url,
                        timeout=self.timeout,
                    )
                )
            return self._managed_agents_anthropic
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def research(self) -> ResearchClient:
        """Managed Research hero namespace (projects, runs, limits, factories)."""
        if self._research_client is None:
            from synth_ai.research.client import ResearchClient

            self._research_client = ResearchClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout,
            )
        return self._research_client


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
                timeout_seconds=self.timeout,
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
        self._pools_sync = self.pools._sync_obj
        self.optimizers = AsyncOptimizersClient(
            OptimizersClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout_seconds=self.timeout,
            )
        )
        self.workflows = AsyncWorkflowsClient(
            WorkflowsClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout_seconds=self.timeout,
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
        self._research_client: ResearchClient | None = None
        self._async_research_client: AsyncResearchClient | None = None
        self._horizons_private: AsyncHorizonsPrivateClient | None = None
        self._managed_agents: AsyncManagedAgentsAnthropicClient | None = None
        self._managed_agents_anthropic: AsyncSynthManagedAgents | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "horizons_private":
            warnings.warn(
                "AsyncSynthClient.horizons_private is deprecated; use client.pools.rollouts instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self._horizons_private is None:
                self._horizons_private = AsyncHorizonsPrivateClient(
                    HorizonsPrivateClient(self._pools_sync)
                )
            return self._horizons_private
        if name == "managed_agents":
            warnings.warn(
                "AsyncSynthClient.managed_agents is deprecated; import AsyncManagedAgentsAnthropicClient explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self._managed_agents is None:
                self._managed_agents = AsyncManagedAgentsAnthropicClient(
                    ManagedAgentsAnthropicClient(
                        api_key=self.api_key,
                        backend_base=self.base_url,
                        timeout=self.timeout,
                    )
                )
            return self._managed_agents
        if name == "managed_agents_anthropic":
            warnings.warn(
                "AsyncSynthClient.managed_agents_anthropic is deprecated; import AsyncSynthManagedAgents explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self._managed_agents_anthropic is None:
                self._managed_agents_anthropic = AsyncSynthManagedAgents.from_transport(
                    ManagedAgentsAnthropicClient(
                        api_key=self.api_key,
                        backend_base=self.base_url,
                        timeout=self.timeout,
                    )
                )
            return self._managed_agents_anthropic
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def research(self) -> ResearchClient:
        """Sync Managed Research namespace (thread-offloaded from async client)."""
        if self._research_client is None:
            from synth_ai.research.client import ResearchClient

            self._research_client = ResearchClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout,
            )
        return self._research_client

    @property
    def async_research(self) -> AsyncResearchClient:
        """Async adapter over ``ResearchClient``."""
        if self._async_research_client is None:
            from synth_ai.research.async_client import AsyncResearchClient

            self._async_research_client = AsyncResearchClient(self.research)
        return self._async_research_client


__all__ = [
    "AsyncContainerPoolsClient",
    "AsyncContainersClient",
    "AsyncHorizonsPrivateClient",
    "AsyncManagedAgentsAnthropicClient",
    "AsyncOpenAIAgentsSdkClient",
    "AsyncOptimizersClient",
    "AsyncSynthClient",
    "AsyncSynthManagedAgents",
    "AsyncTunnelsClient",
    "AsyncWorkflowsClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "HorizonsPrivateClient",
    "ManagedAgentsAnthropicClient",
    "OpenAIAgentsSdkClient",
    "OptimizersClient",
    "SynthClient",
    "SynthManagedAgents",
    "TunnelsClient",
    "WorkflowsClient",
]
