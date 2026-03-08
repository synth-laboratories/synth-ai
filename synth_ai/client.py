"""Canonical front-door SDK clients.

# See: specs/README.md
# See: specs/sdk_logic.md
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from synth_ai.core.tunnels import TunnelBackend, TunneledContainer, TunnelProvider
from synth_ai.core.tunnels.errors import TunnelErrorCode, TunnelProviderError
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.container import ContainerClient, InProcessContainer, create_container
from synth_ai.sdk.container_pools import ContainerPoolsClient
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
from synth_ai.sdk.opencode_skills import (
    install_all_packaged_opencode_skills,
    install_packaged_opencode_skill,
    list_packaged_opencode_skill_names,
    materialize_tui_opencode_config_dir,
)
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
            Path,
            Enum,
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


class PoolTarget(str, Enum):
    HARBOR = "harbor"
    OPENENV = "openenv"
    HORIZONS = "horizons"
    ARBITRARY = "arbitrary"


_POOL_TARGET_TEMPLATE: dict[PoolTarget, str] = {
    PoolTarget.HARBOR: "harbor",
    PoolTarget.OPENENV: "openenv",
    # Horizons routes through Archipelago-managed pools in the control plane.
    PoolTarget.HORIZONS: "archipelago",
}


def _require_field(payload: dict[str, Any], field: str, *, context: str) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise RuntimeError(f"{context} response missing required field: {field}")
    return value


_CANONICAL_ROLLOUT_REQUEST_KEYS = {
    "trace_correlation_id",
    "inference_url",
    "policy",
    "policy_config",
    "env",
    "metadata",
    "safety",
    "limits",
    "params",
    "seed",
}


def _validate_rollout_request(request: dict[str, Any], *, context: str) -> None:
    invalid_keys = sorted(key for key in request if key not in _CANONICAL_ROLLOUT_REQUEST_KEYS)
    if invalid_keys:
        invalid_list = ", ".join(invalid_keys)
        raise ValueError(
            f"{context} request contains unsupported keys ({invalid_list}); "
            "use canonical rollout fields only."
        )


class _PoolUploadsSyncClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def create(
        self,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        expires_in_seconds: int | None = None,
        auto_create_data_source: bool = False,
        auto_create_timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        return self._raw.create_upload(
            filename=filename,
            content_type=content_type,
            expires_in_seconds=expires_in_seconds,
            auto_create_data_source=auto_create_data_source,
            auto_create_timeout_sec=auto_create_timeout_sec,
        )

    def get(self, upload_id: str) -> dict[str, Any]:
        return self._raw.get_upload(upload_id)


class _PoolDataSourcesSyncClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw.create_data_source(request)

    def list(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.list_data_sources(state=state, limit=limit, cursor=cursor)

    def get(self, data_source_id: str) -> dict[str, Any]:
        return self._raw.get_data_source(data_source_id)

    def update(self, data_source_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw.update_data_source(data_source_id, request)

    def refresh(self, data_source_id: str) -> dict[str, Any]:
        return self._raw.refresh_data_source(data_source_id)


class _PoolAssembliesSyncClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def create(
        self,
        *,
        data_source_id: str,
        exclusion_patterns: list[str] | None = None,
        runtime_type: str = "custom_container",
        template_name: str | None = None,
        agent_model: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.create_assembly(
            data_source_id=data_source_id,
            exclusion_patterns=exclusion_patterns,
            runtime_type=runtime_type,  # type: ignore[arg-type]
            template_name=template_name,
            agent_model=agent_model,
        )

    def list(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.list_assemblies(state=state, limit=limit, cursor=cursor)

    def get(self, assembly_id: str) -> dict[str, Any]:
        return self._raw.get_assembly(assembly_id)

    def events(self, assembly_id: str, *, cursor: str | None = None) -> Iterable[dict[str, Any]]:
        return self._raw.stream_assembly_events(assembly_id, cursor=cursor)


class _PoolRolloutsSyncClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def create(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        _validate_rollout_request(request, context="pools.rollouts.create")
        return self._raw.create_rollout(pool_id, request)

    def get(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_rollout(pool_id, rollout_id)

    def list(
        self,
        pool_id: str,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.list_rollouts(pool_id, state=state, limit=limit, cursor=cursor)

    def cancel(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._raw.cancel_rollout(pool_id, rollout_id)

    def artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_rollout_artifacts(pool_id, rollout_id)

    def usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_rollout_usage(pool_id, rollout_id)

    def events(
        self,
        pool_id: str,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        return self._raw.stream_rollout_events(pool_id, rollout_id, cursor=cursor)


class _PoolTasksSyncClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def list(self, pool_id: str) -> dict[str, Any]:
        return self._raw._request("GET", f"/v1/pools/{pool_id}/tasks")

    def create(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw._request("POST", f"/v1/pools/{pool_id}/tasks", json_body=request)

    def update(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw._request("PUT", f"/v1/pools/{pool_id}/tasks/{task_id}", json_body=request)

    def delete(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._raw._request("DELETE", f"/v1/pools/{pool_id}/tasks/{task_id}")


class _PoolMetricsSyncClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def get(self, pool_id: str) -> dict[str, Any]:
        return self._raw._request("GET", f"/v1/pools/{pool_id}/metrics")


class _PoolSkillsSyncClient:
    """OpenCode skill helpers scoped under container pools."""

    def list(self) -> list[str]:
        return list_packaged_opencode_skill_names()

    def install(
        self,
        *,
        skill_name: str,
        dest_skills_dir: Path | None = None,
        force: bool = False,
    ) -> Path:
        return install_packaged_opencode_skill(
            skill_name=skill_name,
            dest_skills_dir=dest_skills_dir,
            force=force,
        )

    def install_all(
        self,
        *,
        dest_skills_dir: Path | None = None,
        force: bool = False,
    ) -> list[Path]:
        return install_all_packaged_opencode_skills(
            dest_skills_dir=dest_skills_dir,
            force=force,
        )

    def materialize_config_dir(
        self,
        *,
        dest_dir: Path | None = None,
        force: bool = True,
        include_packaged_skills: Iterable[str] | None = None,
    ) -> Path:
        return materialize_tui_opencode_config_dir(
            dest_dir=dest_dir,
            force=force,
            include_packaged_skills=include_packaged_skills,
        )


class _PoolTemplateSyncClient:
    def __init__(self, pools: PoolsClient, target: PoolTarget) -> None:
        self._pools = pools
        self.target = target

    def _runtime_config(self) -> tuple[str, str | None]:
        if self.target == PoolTarget.ARBITRARY:
            return ("custom_container", None)
        return ("managed_template", _POOL_TARGET_TEMPLATE[self.target])

    def create_from_data_source(
        self,
        *,
        data_source_id: str,
        exclusion_patterns: list[str] | None = None,
        agent_model: str | None = None,
        pool_request_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_type, template_name = self._runtime_config()
        assembly = self._pools.assemblies.create(
            data_source_id=data_source_id,
            exclusion_patterns=exclusion_patterns,
            runtime_type=runtime_type,
            template_name=template_name,
            agent_model=agent_model,
        )
        assembly_id = _require_field(
            assembly,
            "assembly_id",
            context=f"{self.target.value} assembly.create",
        )
        pool_request: dict[str, Any] = {"assembly_id": assembly_id}
        if pool_request_overrides:
            pool_request.update(pool_request_overrides)
        pool = self._pools.create(pool_request)
        return {
            "target": self.target.value,
            "assembly": assembly,
            "pool": pool,
        }

    def reassemble(
        self,
        pool_id: str,
        *,
        exclusion_patterns: list[str] | None = None,
        agent_model: str | None = None,
    ) -> dict[str, Any]:
        runtime_type, template_name = self._runtime_config()
        return self._pools.reassemble(
            pool_id,
            exclusion_patterns=exclusion_patterns,
            runtime_type=runtime_type,
            template_name=template_name,
            agent_model=agent_model,
        )


class PoolsClient:
    """First-class container pools client with target-specific abstractions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        raw_client: ContainerPoolsClient | None = None,
    ) -> None:
        self._raw = raw_client or ContainerPoolsClient(
            api_key=api_key,
            backend_base=base_url,
            timeout=timeout,
        )
        self.uploads = _PoolUploadsSyncClient(self._raw)
        self.data_sources = _PoolDataSourcesSyncClient(self._raw)
        self.assemblies = _PoolAssembliesSyncClient(self._raw)
        self.rollouts = _PoolRolloutsSyncClient(self._raw)
        self.tasks = _PoolTasksSyncClient(self._raw)
        self.metrics = _PoolMetricsSyncClient(self._raw)
        self.skills = _PoolSkillsSyncClient()
        self.harbor = _PoolTemplateSyncClient(self, PoolTarget.HARBOR)
        self.openenv = _PoolTemplateSyncClient(self, PoolTarget.OPENENV)
        self.horizons = _PoolTemplateSyncClient(self, PoolTarget.HORIZONS)
        self.arbitrary = _PoolTemplateSyncClient(self, PoolTarget.ARBITRARY)

    @property
    def raw(self) -> ContainerPoolsClient:
        return self._raw

    # Top-level core pool lifecycle
    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        if isinstance(request.get("assembly_id"), str) and request["assembly_id"].strip():
            return self._raw.create_pool(request)
        return self._raw._request("POST", "/v1/pools", json_body=request)

    def list(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.list_pools(state=state, limit=limit, cursor=cursor)

    def get(self, pool_id: str) -> dict[str, Any]:
        return self._raw.get_pool(pool_id)

    def update(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw.update_pool(pool_id, request)

    def delete(self, pool_id: str) -> dict[str, Any]:
        return self._raw.delete_pool(pool_id)

    def reassemble(
        self,
        pool_id: str,
        *,
        exclusion_patterns: list[str] | None = None,
        runtime_type: str = "custom_container",
        template_name: str | None = None,
        agent_model: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.reassemble_pool(
            pool_id,
            exclusion_patterns=exclusion_patterns,
            runtime_type=runtime_type,  # type: ignore[arg-type]
            template_name=template_name,
            agent_model=agent_model,
        )

    # Convenience pass-throughs for direct flat usage
    def create_upload(self, **kwargs: Any) -> dict[str, Any]:
        return self.uploads.create(**kwargs)

    def get_upload(self, upload_id: str) -> dict[str, Any]:
        return self.uploads.get(upload_id)

    def create_data_source(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.data_sources.create(request)

    def list_data_sources(self, **kwargs: Any) -> dict[str, Any]:
        return self.data_sources.list(**kwargs)

    def get_data_source(self, data_source_id: str) -> dict[str, Any]:
        return self.data_sources.get(data_source_id)

    def update_data_source(self, data_source_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.data_sources.update(data_source_id, request)

    def refresh_data_source(self, data_source_id: str) -> dict[str, Any]:
        return self.data_sources.refresh(data_source_id)

    def create_assembly(self, **kwargs: Any) -> dict[str, Any]:
        return self.assemblies.create(**kwargs)

    def list_assemblies(self, **kwargs: Any) -> dict[str, Any]:
        return self.assemblies.list(**kwargs)

    def get_assembly(self, assembly_id: str) -> dict[str, Any]:
        return self.assemblies.get(assembly_id)

    def stream_assembly_events(
        self,
        assembly_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        return self.assemblies.events(assembly_id, cursor=cursor)

    def create_rollout(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.rollouts.create(pool_id, request)

    def get_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self.rollouts.get(pool_id, rollout_id)

    def list_rollouts(self, pool_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.rollouts.list(pool_id, **kwargs)

    def cancel_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self.rollouts.cancel(pool_id, rollout_id)

    def get_rollout_artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self.rollouts.artifacts(pool_id, rollout_id)

    def get_rollout_usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self.rollouts.usage(pool_id, rollout_id)

    def stream_rollout_events(
        self,
        pool_id: str,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        return self.rollouts.events(pool_id, rollout_id, cursor=cursor)

    def get_metrics(self, pool_id: str) -> dict[str, Any]:
        return self.metrics.get(pool_id)

    def list_tasks(self, pool_id: str) -> dict[str, Any]:
        return self.tasks.list(pool_id)

    def create_task(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.tasks.create(pool_id, request)

    def update_task(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.tasks.update(pool_id, task_id, request)

    def delete_task(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self.tasks.delete(pool_id, task_id)


class AsyncPoolsClient(_AsyncThreadProxy):
    """Async adapter over PoolsClient and its nested namespaces."""


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

        self.pools = PoolsClient(
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

        sync_pools = PoolsClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.pools = AsyncPoolsClient(sync_pools)
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
    "AsyncContainersClient",
    "AsyncNgrokTunnel",
    "AsyncPoolsClient",
    "AsyncSynthTunnel",
    "AsyncSynthClient",
    "AsyncTunnelsClient",
    "ContainersClient",
    "NgrokTunnel",
    "PoolTarget",
    "PoolsClient",
    "SynthTunnel",
    "SynthClient",
    "TunnelsClient",
]
