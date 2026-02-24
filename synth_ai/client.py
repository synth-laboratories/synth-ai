"""Canonical front-door SDK clients.

# See: specifications/tanha/current/documentation/docs_specification.md
# See: specifications/daily/feb24_2026/tinker_synth_final.md
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.container import ContainerClient, InProcessContainer, create_container
from synth_ai.sdk.container_pools import ContainerPoolsClient
from synth_ai.sdk.graphs.completions import (
    GraphCompletionsAsyncClient,
    GraphCompletionsSyncClient,
    VerifierAsyncClient,
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


@dataclass(slots=True)
class _SystemsAsyncClient:
    _base_url: str
    _api_key: str
    _timeout: float

    async def create(self, **kwargs: Any) -> PolicyOptimizationSystem:
        return await PolicyOptimizationSystem.create_async(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    async def get(self, system_id: str, **kwargs: Any) -> PolicyOptimizationSystem:
        return await PolicyOptimizationSystem.get_async(
            system_id,
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    async def list(self, **kwargs: Any) -> dict[str, Any]:
        return await PolicyOptimizationSystem.list_async(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )


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


@dataclass(slots=True)
class _OfflineAsyncClient:
    _base_url: str
    _api_key: str
    _timeout: float

    async def create(self, **kwargs: Any) -> PolicyOptimizationOfflineJob:
        return await PolicyOptimizationOfflineJob.create_async(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    async def get(self, job_id: str, **kwargs: Any) -> PolicyOptimizationOfflineJob:
        return await PolicyOptimizationOfflineJob.get_async(
            job_id,
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    async def list(self, **kwargs: Any) -> dict[str, Any]:
        return await PolicyOptimizationOfflineJob.list_async(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )


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


@dataclass(slots=True)
class _OnlineAsyncClient:
    _base_url: str
    _api_key: str
    _timeout: float

    async def create(self, **kwargs: Any) -> PolicyOptimizationOnlineSession:
        return await PolicyOptimizationOnlineSession.create_async(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    async def get(self, session_id: str, **kwargs: Any) -> PolicyOptimizationOnlineSession:
        return await PolicyOptimizationOnlineSession.get_async(
            session_id,
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )

    async def list(self, **kwargs: Any) -> dict[str, Any]:
        return await PolicyOptimizationOnlineSession.list_async(
            backend_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            **kwargs,
        )


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


class _InferenceChatCompletionsSyncClient:
    def __init__(self, client: InferenceClient) -> None:
        self._client = client

    def create(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        return run_sync(
            self._client.create_chat_completion(model=model, messages=messages, **kwargs),
            label="inference.chat.completions.create",
        )


class _InferenceChatCompletionsAsyncClient:
    def __init__(self, client: InferenceClient) -> None:
        self._client = client

    async def create(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        return await self._client.create_chat_completion(model=model, messages=messages, **kwargs)


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
        return run_sync(
            self._client.create_job_from_request(request),
            label="inference.jobs.create_from_request",
        )

    def create_from_path(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(
            self._client.create_job_from_path(**kwargs),
            label="inference.jobs.create_from_path",
        )

    def get(self, job_id: str) -> dict[str, Any]:
        return run_sync(self._client.get_job(job_id), label="inference.jobs.get")

    def list_artifacts(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        return run_sync(
            self._client.list_artifacts(job_id, **kwargs),
            label="inference.jobs.list_artifacts",
        )

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
        self._client = GraphCompletionsSyncClient(
            base_url=base_url, api_key=api_key, timeout=timeout
        )

    def run(self, **kwargs: Any) -> Any:
        return self._client.run(**kwargs)

    def run_output(self, **kwargs: Any) -> Any:
        return self._client.run_output(**kwargs)

    def list_graphs(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.list_graphs(**kwargs)


class AsyncGraphsClient:
    """Canonical async graph-completions namespace."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self._client = GraphCompletionsAsyncClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

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
        self._client = VerifierAsyncClient(base_url=base_url, api_key=api_key, timeout=timeout)

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return run_sync(self._client.evaluate(**kwargs), label="verifiers.evaluate")


class AsyncVerifiersClient:
    """Canonical async verifier namespace."""

    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self._client = VerifierAsyncClient(base_url=base_url, api_key=api_key, timeout=timeout)

    async def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client.evaluate(**kwargs)


class AsyncPoolsClient:
    """Async adapter over the synchronous pools client."""

    def __init__(self, sync_client: ContainerPoolsClient) -> None:
        self._sync_client = sync_client

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._sync_client, name)
        if not callable(attr):
            return attr

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _wrapped


class _ContainerNamespace:
    InProcessContainer = InProcessContainer
    ContainerClient = ContainerClient

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Any:
        return create_container(*args, **kwargs)


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
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
        )
        self.verifiers = VerifiersClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.pools = ContainerPoolsClient(
            api_key=self.api_key,
            backend_base=self.base_url,
            timeout=self.timeout,
        )
        self.container = _ContainerNamespace()


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
        self.pools = AsyncPoolsClient(
            ContainerPoolsClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout=self.timeout,
            )
        )
        self.container = _ContainerNamespace()


__all__ = [
    "AsyncGraphsClient",
    "AsyncPoolsClient",
    "AsyncSynthClient",
    "AsyncVerifiersClient",
    "GraphsClient",
    "SynthClient",
    "VerifiersClient",
]
