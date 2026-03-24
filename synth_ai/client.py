"""Canonical front-door SDK clients.

# See: specs/README.md
# See: specs/sdk_logic.md
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from synth_ai.core.tunnels import TunnelBackend, TunneledContainer, TunnelProvider
from synth_ai.core.tunnels.errors import TunnelErrorCode, TunnelProviderError
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.container import ContainerClient, InProcessContainer, create_container
from synth_ai.sdk.container_pools import (
    AsyncContainerPoolsClient,
    CANONICAL_ROLLOUT_REQUEST_KEYS,
    ContainerPoolsClient,
    PoolTarget,
)
from synth_ai.sdk.containers import (
    Container as HostedContainer,
)
from synth_ai.sdk.containers import (
    ContainersClient as HostedContainersAPIClient,
)
from synth_ai.sdk.containers import (
    ContainerSpec as HostedContainerSpec,
)
from synth_ai.sdk.containers import (
    ContainerType as HostedContainerType,
)
from synth_ai.sdk.graphs.completions import (
    GraphCompletionsAsyncClient,
    GraphCompletionsSyncClient,
    VerifierClient,
)
from synth_ai.sdk.inference import InferenceClient, InferenceJobsClient
from synth_ai.sdk.optimization import (
    PolicyOptimizationOfflineJob,
    PolicyOptimizationOnlineSession,
    PolicyOptimizationSystem,
)
from synth_ai.sdk.optimization.utils import run_sync


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (api_key or os.getenv("SYNTH_API_KEY") or "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
    return resolved


def _resolve_base_url(base_url: str | None) -> str:
    return normalize_backend_base(base_url or BACKEND_URL_BASE)


def _resolve_tunnel_backend(
    *,
    backend: TunnelBackend | None,
    provider: TunnelProvider | str | None,
) -> TunnelBackend:
    if provider is None:
        return backend or TunnelBackend.SynthTunnel
    provider_raw = str(provider).strip()
    provider_lc = provider_raw.lower()
    provider_map = {
        "synthtunnel": TunnelBackend.SynthTunnel,
        "ngrok": TunnelBackend.NgrokManaged,
        "localhost": TunnelBackend.Localhost,
    }
    resolved_from_provider = provider_map.get(provider_lc)
    if resolved_from_provider is None:
        raise TunnelProviderError(
            f"provider must be one of SynthTunnel, Ngrok, or Localhost (got {provider!r})",
            code=TunnelErrorCode.PROVIDER_INVALID,
            provider=str(provider),
        )
    if backend is not None and backend != resolved_from_provider:
        raise TunnelProviderError(
            "backend and provider conflict; pass one selector or matching values.",
            code=TunnelErrorCode.PROVIDER_INVALID,
            provider=provider_raw,
        )
    return resolved_from_provider


def _is_proxy_namespace(value: Any) -> bool:
    if isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
            int,
            float,
            bool,
            type(None),
            dict,
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return False
    if isinstance(value, type):
        return False
    return hasattr(value, "__dict__")


class _AsyncThreadProxy:
    def __init__(self, sync_obj: Any) -> None:
        self._sync_obj = sync_obj
        self._proxy_cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        cached = self._proxy_cache.get(name)
        if cached is not None:
            return cached
        attr = getattr(self._sync_obj, name)
        if callable(attr):

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            self._proxy_cache[name] = _wrapped
            return _wrapped
        if _is_proxy_namespace(attr):
            proxy = _AsyncThreadProxy(attr)
            self._proxy_cache[name] = proxy
            return proxy
        return attr


@dataclass(slots=True)
class _SystemsSyncClient:
    _base_url: str
    _api_key: str
    _timeout: float

    def create(self, **kwargs: Any) -> PolicyOptimizationSystem:
        return PolicyOptimizationSystem.create(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    def get(self, system_id: str, **kwargs: Any) -> PolicyOptimizationSystem:
        return PolicyOptimizationSystem.get(
            system_id,
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    def list(self, **kwargs: Any) -> dict[str, Any]:
        return PolicyOptimizationSystem.list(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )


class _SystemsAsyncClient(_AsyncThreadProxy):
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        super().__init__(_SystemsSyncClient(base_url, api_key, timeout))


@dataclass(slots=True)
class _OfflineSyncClient:
    _base_url: str
    _api_key: str
    _timeout: float

    def create(self, **kwargs: Any) -> PolicyOptimizationOfflineJob:
        return PolicyOptimizationOfflineJob.create(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    def get(self, job_id: str, **kwargs: Any) -> PolicyOptimizationOfflineJob:
        return PolicyOptimizationOfflineJob.get(
            job_id,
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    def list(self, **kwargs: Any) -> dict[str, Any]:
        return PolicyOptimizationOfflineJob.list(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )


class _OfflineAsyncClient(_AsyncThreadProxy):
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        super().__init__(_OfflineSyncClient(base_url, api_key, timeout))


@dataclass(slots=True)
class _OnlineSyncClient:
    _base_url: str
    _api_key: str
    _timeout: float

    def create(self, **kwargs: Any) -> PolicyOptimizationOnlineSession:
        return PolicyOptimizationOnlineSession.create(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    def get(self, session_id: str, **kwargs: Any) -> PolicyOptimizationOnlineSession:
        return PolicyOptimizationOnlineSession.get(
            session_id,
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    def list(self, **kwargs: Any) -> dict[str, Any]:
        return PolicyOptimizationOnlineSession.list(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    def runtime_compatibility(self, **kwargs: Any) -> dict[str, Any]:
        return PolicyOptimizationOnlineSession.runtime_compatibility(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )


class _OnlineAsyncClient(_AsyncThreadProxy):
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        super().__init__(_OnlineSyncClient(base_url, api_key, timeout))


class _InferenceChatCompletionsSyncClient:
    def __init__(self, client: InferenceClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self._client.create_chat_completion(**kwargs), label="inference.chat.completions.create")


class _InferenceChatCompletionsAsyncClient:
    def __init__(self, client: InferenceClient) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.create_chat_completion(**kwargs)


@dataclass(slots=True)
class _InferenceChatSyncClient:
    completions: _InferenceChatCompletionsSyncClient


@dataclass(slots=True)
class _InferenceChatAsyncClient:
    completions: _InferenceChatCompletionsAsyncClient


class _InferenceJobsSyncClient:
    def __init__(self, client: InferenceJobsClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self._client.create_job(**kwargs), label="inference.jobs.create")

    def create_from_request(self, request: Any) -> dict[str, Any]:
        return run_sync(self._client.create_job_from_request(request), label="inference.jobs.create_from_request")

    def create_from_path(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self._client.create_job_from_path(**kwargs), label="inference.jobs.create_from_path")

    def get(self, job_id: str) -> dict[str, Any]:
        return run_sync(self._client.get_job(job_id), label="inference.jobs.get")

    def list_artifacts(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self._client.list_artifacts(job_id, **kwargs), label="inference.jobs.list_artifacts")

    def download_artifact(self, job_id: str, artifact_id: str, **kwargs: Any) -> bytes:
        return run_sync(
            self._client.download_artifact(job_id, artifact_id, **kwargs),
            label="inference.jobs.download_artifact",
        )


class _InferenceJobsAsyncClient:
    def __init__(self, client: InferenceJobsClient) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.create_job(**kwargs)

    async def create_from_request(self, request: Any) -> dict[str, Any]:
        return await self._client.create_job_from_request(request)

    async def create_from_path(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.create_job_from_path(**kwargs)

    async def get(self, job_id: str) -> dict[str, Any]:
        return await self._client.get_job(job_id)

    async def list_artifacts(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._client.list_artifacts(job_id, **kwargs)

    async def download_artifact(self, job_id: str, artifact_id: str, **kwargs: Any) -> bytes:
        return await self._client.download_artifact(job_id, artifact_id, **kwargs)


@dataclass(slots=True)
class _InferenceSyncClient:
    chat: _InferenceChatSyncClient
    jobs: _InferenceJobsSyncClient


@dataclass(slots=True)
class _InferenceAsyncClient:
    chat: _InferenceChatAsyncClient
    jobs: _InferenceJobsAsyncClient


class GraphsClient:
    """Canonical sync graph-completions namespace."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self._client = GraphCompletionsSyncClient(base_url=base_url, api_key=api_key, timeout=timeout)

    def run(self, **kwargs: Any) -> Any:
        return self._client.run(**kwargs)

    def run_output(self, **kwargs: Any) -> Any:
        return self._client.run_output(**kwargs)

    def list_graphs(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.list_graphs(**kwargs)


class AsyncGraphsClient:
    """Canonical async graph-completions namespace."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self._client = GraphCompletionsAsyncClient(base_url=base_url, api_key=api_key, timeout=timeout)

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.run(**kwargs)

    async def run_output(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.run_output(**kwargs)

    async def list_graphs(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.list_graphs(**kwargs)

    async def rlm_inference(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.rlm_inference(**kwargs)


class VerifiersClient:
    """Canonical sync verifier namespace."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self._client = VerifierClient(base_url=base_url, api_key=api_key, timeout=timeout)

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self._client.evaluate(**kwargs), label="verifiers.evaluate")


class AsyncVerifiersClient:
    """Canonical async verifier namespace."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self._client = VerifierClient(base_url=base_url, api_key=api_key, timeout=timeout)

    async def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.evaluate(**kwargs)


@dataclass(slots=True)
class _OptimizationSyncClient:
    """Canonical sync optimization namespace."""

    systems: _SystemsSyncClient
    offline: _OfflineSyncClient
    online: _OnlineSyncClient


@dataclass(slots=True)
class _OptimizationAsyncClient:
    """Canonical async optimization namespace."""

    systems: _SystemsAsyncClient
    offline: _OfflineAsyncClient
    online: _OnlineAsyncClient


class _ReservedNamespace:
    """Placeholder namespace for routes reserved in the canonical SDK surface.

    The docs and sync lints treat these namespaces as part of the public front-door
    shape even where the implementation is still intentionally thin.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        raise NotImplementedError(
            f"synth_ai.{self._name}.{attr} is not implemented in the canonical front-door SDK yet."
        )


# Back-compat for code or docs referring to synth_ai.client._CANONICAL_ROLLOUT_REQUEST_KEYS.
_CANONICAL_ROLLOUT_REQUEST_KEYS = CANONICAL_ROLLOUT_REQUEST_KEYS
PoolsClient = ContainerPoolsClient
AsyncPoolsClient = AsyncContainerPoolsClient


class ContainersClient:
    """Hosted containers API with canonical helpers."""

    Container = HostedContainer
    ContainerSpec = HostedContainerSpec
    ContainerType = HostedContainerType

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        raw_client: HostedContainersAPIClient | None = None,
    ) -> None:
        self._raw = raw_client or HostedContainersAPIClient(
            api_key=api_key,
            backend_base=base_url,
        )

    @property
    def raw(self) -> HostedContainersAPIClient:
        return self._raw

    def create(self, spec: HostedContainerSpec, *, timeout: float = 30.0) -> HostedContainer:
        return self._raw.create(spec, timeout=timeout)

    def get(self, container_id: str, *, timeout: float = 30.0) -> HostedContainer:
        return self._raw.get(container_id, timeout=timeout)

    def list(self, *, timeout: float = 30.0) -> list[HostedContainer]:
        return self._raw.list(timeout=timeout)

    def delete(self, container_id: str, *, timeout: float = 30.0) -> None:
        self._raw.delete(container_id, timeout=timeout)

    def wait_ready(
        self,
        container_id: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> HostedContainer:
        return self._raw.wait_ready(
            container_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def create_harbor_code(
        self,
        *,
        name: str,
        definition: dict[str, Any],
        environment_config: dict[str, Any] | None = None,
        internal_url: str | None = None,
        timeout: float = 30.0,
    ) -> HostedContainer:
        return self.create(
            HostedContainerSpec(
                name=name,
                task_type=HostedContainerType.harbor_code,
                definition=definition,
                environment_config=environment_config,
                internal_url=internal_url,
            ),
            timeout=timeout,
        )

    def create_harbor_browser(
        self,
        *,
        name: str,
        definition: dict[str, Any],
        environment_config: dict[str, Any] | None = None,
        internal_url: str | None = None,
        timeout: float = 30.0,
    ) -> HostedContainer:
        return self.create(
            HostedContainerSpec(
                name=name,
                task_type=HostedContainerType.harbor_browser,
                definition=definition,
                environment_config=environment_config,
                internal_url=internal_url,
            ),
            timeout=timeout,
        )

    def create_openenv(
        self,
        *,
        name: str,
        definition: dict[str, Any],
        environment_config: dict[str, Any] | None = None,
        internal_url: str | None = None,
        timeout: float = 30.0,
    ) -> HostedContainer:
        return self.create(
            HostedContainerSpec(
                name=name,
                task_type=HostedContainerType.openenv,
                definition=definition,
                environment_config=environment_config,
                internal_url=internal_url,
            ),
            timeout=timeout,
        )

    def create_horizons(
        self,
        *,
        name: str,
        definition: dict[str, Any],
        environment_config: dict[str, Any] | None = None,
        internal_url: str | None = None,
        timeout: float = 30.0,
    ) -> HostedContainer:
        return self.create(
            HostedContainerSpec(
                name=name,
                task_type=HostedContainerType.archipelago,
                definition=definition,
                environment_config=environment_config,
                internal_url=internal_url,
            ),
            timeout=timeout,
        )


class AsyncContainersClient(_AsyncThreadProxy):
    """Async adapter over ContainersClient and nested values."""


class _LocalContainersNamespace:
    InProcessContainer = InProcessContainer
    ContainerClient = ContainerClient

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Any:
        return create_container(*args, **kwargs)

    @staticmethod
    def connect(
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        retries: int = 3,
    ) -> ContainerClient:
        return ContainerClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
        )


class SynthTunnel:
    """Specialized first-class SynthTunnel abstraction."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def open_async(
        self,
        *,
        local_port: int,
        verify_dns: bool = True,
        progress: bool = False,
        reason: str | None = None,
        requested_ttl_seconds: int | None = None,
    ) -> TunneledContainer:
        return await TunneledContainer.create(
            local_port=local_port,
            backend=TunnelBackend.SynthTunnel,
            api_key=self._api_key,
            backend_url=self._base_url,
            verify_dns=verify_dns,
            progress=progress,
            reason=reason,
            requested_ttl_seconds=requested_ttl_seconds,
        )

    def open(self, **kwargs: Any) -> TunneledContainer:
        return run_sync(
            self.open_async(**kwargs),
            label="tunnels.synth.open",
        )

    async def open_for_app_async(
        self,
        *,
        app: Any,
        local_port: int | None = None,
        verify_dns: bool = True,
        progress: bool = False,
        requested_ttl_seconds: int | None = None,
    ) -> TunneledContainer:
        return await TunneledContainer.create_for_app(
            app=app,
            local_port=local_port,
            backend=TunnelBackend.SynthTunnel,
            api_key=self._api_key,
            backend_url=self._base_url,
            verify_dns=verify_dns,
            progress=progress,
            requested_ttl_seconds=requested_ttl_seconds,
        )

    def open_for_app(self, **kwargs: Any) -> TunneledContainer:
        return run_sync(
            self.open_for_app_async(**kwargs),
            label="tunnels.synth.open_for_app",
        )


class AsyncSynthTunnel:
    """Async-first SynthTunnel abstraction."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._sync = SynthTunnel(api_key=api_key, base_url=base_url)

    async def open(self, **kwargs: Any) -> TunneledContainer:
        return await self._sync.open_async(**kwargs)

    async def open_for_app(self, **kwargs: Any) -> TunneledContainer:
        return await self._sync.open_for_app_async(**kwargs)


class NgrokTunnel:
    """First-class Synth-managed ngrok-compatible tunnel abstraction."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def open_async(
        self,
        *,
        local_port: int,
        managed_ngrok_url: str | None = None,
        verify_dns: bool = True,
        progress: bool = False,
    ) -> TunneledContainer:
        return await TunneledContainer.create(
            local_port=local_port,
            backend=TunnelBackend.NgrokManaged,
            managed_ngrok_url=managed_ngrok_url,
            api_key=self._api_key,
            backend_url=self._base_url,
            verify_dns=verify_dns,
            progress=progress,
        )

    def open(self, **kwargs: Any) -> TunneledContainer:
        return run_sync(self.open_async(**kwargs), label="tunnels.ngrok.open")

    async def open_for_app_async(
        self,
        *,
        app: Any,
        local_port: int | None = None,
        managed_ngrok_url: str | None = None,
        verify_dns: bool = True,
        progress: bool = False,
    ) -> TunneledContainer:
        return await TunneledContainer.create_for_app(
            app=app,
            local_port=local_port,
            backend=TunnelBackend.NgrokManaged,
            managed_ngrok_url=managed_ngrok_url,
            api_key=self._api_key,
            backend_url=self._base_url,
            verify_dns=verify_dns,
            progress=progress,
        )

    def open_for_app(self, **kwargs: Any) -> TunneledContainer:
        return run_sync(self.open_for_app_async(**kwargs), label="tunnels.ngrok.open_for_app")


class AsyncNgrokTunnel:
    """Async-first Synth-managed ngrok-compatible abstraction."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._sync = NgrokTunnel(api_key=api_key, base_url=base_url)

    async def open(self, **kwargs: Any) -> TunneledContainer:
        return await self._sync.open_async(**kwargs)

    async def open_for_app(self, **kwargs: Any) -> TunneledContainer:
        return await self._sync.open_for_app_async(**kwargs)


class TunnelsClient:
    """First-class tunnel client with provider-first selection."""

    TunnelBackend = TunnelBackend
    TunnelProvider = TunnelProvider
    TunneledContainer = TunneledContainer
    SynthTunnel = SynthTunnel
    NgrokTunnel = NgrokTunnel

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self.synth = SynthTunnel(api_key=api_key, base_url=base_url)
        self.ngrok = NgrokTunnel(api_key=api_key, base_url=base_url)

    async def open_async(
        self,
        *,
        local_port: int,
        backend: TunnelBackend | None = None,
        provider: TunnelProvider | str | None = None,
        verify_dns: bool = True,
        progress: bool = False,
        reason: str | None = None,
        requested_ttl_seconds: int | None = None,
        managed_ngrok_url: str | None = None,
    ) -> TunneledContainer:
        resolved_backend = _resolve_tunnel_backend(backend=backend, provider=provider)
        return await TunneledContainer.create(
            local_port=local_port,
            backend=resolved_backend,
            provider=provider,
            managed_ngrok_url=managed_ngrok_url,
            api_key=self._api_key,
            backend_url=self._base_url,
            verify_dns=verify_dns,
            progress=progress,
            reason=reason,
            requested_ttl_seconds=requested_ttl_seconds,
        )

    def open(self, **kwargs: Any) -> TunneledContainer:
        return run_sync(
            self.open_async(**kwargs),
            label="tunnels.open",
        )

    def ngrok_managed(self, **kwargs: Any) -> TunneledContainer:
        return self.open(backend=TunnelBackend.NgrokManaged, **kwargs)

    def localhost(self, **kwargs: Any) -> TunneledContainer:
        return self.open(backend=TunnelBackend.Localhost, **kwargs)

    async def open_for_app_async(
        self,
        *,
        app: Any,
        local_port: int | None = None,
        backend: TunnelBackend | None = None,
        provider: TunnelProvider | str | None = None,
        verify_dns: bool = True,
        progress: bool = False,
        requested_ttl_seconds: int | None = None,
        managed_ngrok_url: str | None = None,
    ) -> TunneledContainer:
        resolved_backend = _resolve_tunnel_backend(backend=backend, provider=provider)
        return await TunneledContainer.create_for_app(
            app=app,
            local_port=local_port,
            backend=resolved_backend,
            provider=provider,
            managed_ngrok_url=managed_ngrok_url,
            api_key=self._api_key,
            backend_url=self._base_url,
            verify_dns=verify_dns,
            progress=progress,
            requested_ttl_seconds=requested_ttl_seconds,
        )

    def open_for_app(self, **kwargs: Any) -> TunneledContainer:
        return run_sync(
            self.open_for_app_async(**kwargs),
            label="tunnels.open_for_app",
        )


class AsyncTunnelsClient:
    """Async-first tunnel client."""

    TunnelBackend = TunnelBackend
    TunnelProvider = TunnelProvider
    TunneledContainer = TunneledContainer
    SynthTunnel = AsyncSynthTunnel
    NgrokTunnel = AsyncNgrokTunnel

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._sync = TunnelsClient(api_key=api_key, base_url=base_url)
        self.synth = AsyncSynthTunnel(api_key=api_key, base_url=base_url)
        self.ngrok = AsyncNgrokTunnel(api_key=api_key, base_url=base_url)

    async def open(self, **kwargs: Any) -> TunneledContainer:
        return await self._sync.open_async(**kwargs)

    async def ngrok_managed(self, **kwargs: Any) -> TunneledContainer:
        return await self.open(backend=TunnelBackend.NgrokManaged, **kwargs)

    async def localhost(self, **kwargs: Any) -> TunneledContainer:
        return await self.open(backend=TunnelBackend.Localhost, **kwargs)

    async def open_for_app(self, **kwargs: Any) -> TunneledContainer:
        return await self._sync.open_for_app_async(**kwargs)


class _ContainerNamespace:
    InProcessContainer = InProcessContainer
    ContainerClient = ContainerClient
    Container = HostedContainer
    ContainerSpec = HostedContainerSpec
    ContainerType = HostedContainerType
    TunnelBackend = TunnelBackend
    TunnelProvider = TunnelProvider
    TunneledContainer = TunneledContainer
    SynthTunnel = SynthTunnel
    NgrokTunnel = NgrokTunnel

    def __init__(
        self,
        *,
        hosted: Any,
        pools: Any,
        tunnels: Any,
    ) -> None:
        self.hosted = hosted
        self.local = _LocalContainersNamespace()
        self.pools = pools
        self.tunnels = tunnels
        self.synth_tunnel = tunnels.synth

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Any:
        return create_container(*args, **kwargs)

    @staticmethod
    def connect(
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        retries: int = 3,
    ) -> ContainerClient:
        return _LocalContainersNamespace.connect(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
        )


class SynthClient:
    """Canonical sync front-door client."""

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

        self.optimization = _OptimizationSyncClient(
            systems=_SystemsSyncClient(self.base_url, self.api_key, self.timeout),
            offline=_OfflineSyncClient(self.base_url, self.api_key, self.timeout),
            online=_OnlineSyncClient(self.base_url, self.api_key, self.timeout),
        )
        self.inference = _ReservedNamespace("inference")
        self.graphs = _ReservedNamespace("graphs")
        self.verifiers = _ReservedNamespace("verifiers")

        inference_client = InferenceClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        inference_jobs = InferenceJobsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.inference = _InferenceSyncClient(
            chat=_InferenceChatSyncClient(
                completions=_InferenceChatCompletionsSyncClient(inference_client),
            ),
            jobs=_InferenceJobsSyncClient(inference_jobs),
        )
        self.graphs = GraphsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.verifiers = VerifiersClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.pools = ContainerPoolsClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.tunnels = TunnelsClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        hosted = ContainersClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.container = _ContainerNamespace(
            hosted=hosted,
            pools=self.pools,
            tunnels=self.tunnels,
        )


class AsyncSynthClient:
    """Canonical async front-door client."""

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

        self.optimization = _OptimizationAsyncClient(
            systems=_SystemsAsyncClient(self.base_url, self.api_key, self.timeout),
            offline=_OfflineAsyncClient(self.base_url, self.api_key, self.timeout),
            online=_OnlineAsyncClient(self.base_url, self.api_key, self.timeout),
        )
        self.inference = _AsyncThreadProxy(_ReservedNamespace("inference"))
        self.graphs = _AsyncThreadProxy(_ReservedNamespace("graphs"))
        self.verifiers = _AsyncThreadProxy(_ReservedNamespace("verifiers"))

        inference_client = InferenceClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        inference_jobs = InferenceJobsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.inference = _InferenceAsyncClient(
            chat=_InferenceChatAsyncClient(
                completions=_InferenceChatCompletionsAsyncClient(inference_client),
            ),
            jobs=_InferenceJobsAsyncClient(inference_jobs),
        )
        self.graphs = AsyncGraphsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.verifiers = AsyncVerifiersClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        sync_pools = ContainerPoolsClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.pools = AsyncContainerPoolsClient(sync_pools)
        self.tunnels = AsyncTunnelsClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        hosted = AsyncContainersClient(
            ContainersClient(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        )
        self.container = _ContainerNamespace(
            hosted=hosted,
            pools=self.pools,
            tunnels=self.tunnels,
        )


__all__ = [
    "AsyncContainerPoolsClient",
    "AsyncContainersClient",
    "AsyncGraphsClient",
    "AsyncNgrokTunnel",
    "AsyncPoolsClient",
    "AsyncSynthTunnel",
    "AsyncSynthClient",
    "AsyncTunnelsClient",
    "AsyncVerifiersClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "GraphsClient",
    "NgrokTunnel",
    "PoolTarget",
    "PoolsClient",
    "SynthTunnel",
    "SynthClient",
    "TunnelsClient",
    "VerifiersClient",
]
