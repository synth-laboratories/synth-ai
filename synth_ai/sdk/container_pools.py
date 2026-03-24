"""Canonical container pools client.

Targets consolidated `/v1/pools/*` and related `/v1/rollouts` endpoints.
"""

from __future__ import annotations

import asyncio
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional

from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base
from synth_ai.sdk.opencode_skills import (
    install_all_packaged_opencode_skills,
    install_packaged_opencode_skill,
    list_packaged_opencode_skill_names,
    materialize_tui_opencode_config_dir,
)

# Mirrors backend/rhodes/contracts.py ROLLOUT_REQUEST_KEYS.
CANONICAL_ROLLOUT_REQUEST_KEYS: frozenset[str] = frozenset(
    {
        "context_overrides",
        "env",
        "inference_url",
        "instance_id",
        "limits",
        "messages",
        "metadata",
        "mode",
        "ops",
        "params",
        "policy",
        "policy_config",
        "pool_id",
        "record",
        "safety",
        "seed",
        "task_id",
        "trace_correlation_id",
        "override_bundle_id",
        "task_contract",
    }
)


def validate_pool_rollout_request(request: dict[str, Any], *, context: str) -> None:
    invalid_keys = sorted(key for key in request if key not in CANONICAL_ROLLOUT_REQUEST_KEYS)
    if invalid_keys:
        invalid_list = ", ".join(invalid_keys)
        raise ValueError(
            f"{context} request contains unsupported keys ({invalid_list}); "
            "use canonical rollout fields only."
        )


def _seed_query_params(seeds: Optional[list[int]]) -> Optional[list[tuple[str, str]]]:
    if seeds is None:
        return None
    return [("seed", str(value)) for value in seeds]


class PoolTarget(str, Enum):
    HARBOR = "harbor"
    OPENENV = "openenv"
    HORIZONS = "horizons"
    ARBITRARY = "arbitrary"


_POOL_TARGET_TEMPLATE: dict[PoolTarget, str] = {
    PoolTarget.HARBOR: "harbor",
    PoolTarget.OPENENV: "openenv",
    PoolTarget.HORIZONS: "archipelago",
}


def _require_field(payload: dict[str, Any], field: str, *, context: str) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise RuntimeError(f"{context} response missing required field: {field}")
    return value


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


class _PoolUploadsSyncClient:
    def __init__(self, raw: "ContainerPoolsClient") -> None:
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
    def __init__(self, raw: "ContainerPoolsClient") -> None:
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
    def __init__(self, raw: "ContainerPoolsClient") -> None:
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
    def __init__(self, raw: "ContainerPoolsClient") -> None:
        self._raw = raw

    def create(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        validate_pool_rollout_request(request, context="container_pools.rollouts.create")
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

    def summary(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_rollout_summary(pool_id, rollout_id)

    def events(
        self,
        pool_id: str,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        return self._raw.stream_rollout_events(pool_id, rollout_id, cursor=cursor)


class _PoolAgentRolloutsSyncClient:
    def __init__(self, raw: "ContainerPoolsClient") -> None:
        self._raw = raw

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        validate_pool_rollout_request(request, context="container_pools.agent_rollouts.create")
        return self._raw.create_global_rollout(request)

    def list(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.list_global_rollouts(state=state, limit=limit, cursor=cursor)

    def get(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout(rollout_id)

    def artifacts(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout_artifacts(rollout_id)

    def usage(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout_usage(rollout_id)

    def summary(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout_summary(rollout_id)

    def cancel(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.cancel_global_rollout(rollout_id)

    def events(
        self,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        return self._raw.stream_global_rollout_events(rollout_id, cursor=cursor)


class _PoolTasksSyncClient:
    def __init__(self, raw: "ContainerPoolsClient") -> None:
        self._raw = raw

    def list(self, pool_id: str) -> dict[str, Any]:
        return self._raw.list_tasks(pool_id)

    def create(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw.create_task(pool_id, request)

    def update(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw.update_task(pool_id, task_id, request)

    def patch(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._raw.patch_task(pool_id, task_id, request)

    def delete(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._raw.delete_task(pool_id, task_id)


class _PoolMetricsSyncClient:
    def __init__(self, raw: "ContainerPoolsClient") -> None:
        self._raw = raw

    def get(self, pool_id: str) -> dict[str, Any]:
        return self._raw.get_pool_metrics(pool_id)


class _PoolSkillsSyncClient:
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
    def __init__(self, pools: "ContainerPoolsClient", target: PoolTarget) -> None:
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


class ContainerPoolsClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        backend_base: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.getenv("SYNTH_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        resolved_base = backend_base or base_url or BACKEND_URL_BASE
        self._backend_base = normalize_backend_base(resolved_base)
        self._timeout = timeout
        self.uploads = _PoolUploadsSyncClient(self)
        self.data_sources = _PoolDataSourcesSyncClient(self)
        self.assemblies = _PoolAssembliesSyncClient(self)
        self.rollouts = _PoolRolloutsSyncClient(self)
        self.agent_rollouts = _PoolAgentRolloutsSyncClient(self)
        self.tasks = _PoolTasksSyncClient(self)
        self.metrics = _PoolMetricsSyncClient(self)
        self.skills = _PoolSkillsSyncClient()
        self.harbor = _PoolTemplateSyncClient(self, PoolTarget.HARBOR)
        self.openenv = _PoolTemplateSyncClient(self, PoolTarget.OPENENV)
        self.horizons = _PoolTemplateSyncClient(self, PoolTarget.HORIZONS)
        self.arbitrary = _PoolTemplateSyncClient(self, PoolTarget.ARBITRARY)

    @property
    def raw(self) -> "ContainerPoolsClient":
        return self

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        params_list: Optional[list[tuple[str, str]]] = None,
    ) -> Any:
        import httpx

        request_params: Any = params_list if params_list is not None else params
        resp = httpx.request(
            method,
            join_url(self._backend_base, path),
            headers=self._headers(),
            json=json_body,
            params=request_params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        if not resp.content:
            return {}
        data = resp.json()
        return data

    # Uploads
    def create_upload(
        self,
        *,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
        auto_create_data_source: bool = False,
        auto_create_timeout_sec: Optional[int] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "auto_create_data_source": auto_create_data_source,
        }
        if filename is not None:
            payload["filename"] = filename
        if content_type is not None:
            payload["content_type"] = content_type
        if expires_in_seconds is not None:
            payload["expires_in_seconds"] = expires_in_seconds
        if auto_create_timeout_sec is not None:
            payload["auto_create_timeout_sec"] = auto_create_timeout_sec
        return self._request("POST", "/v1/pools/uploads", json_body=payload)

    def get_upload(self, upload_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/uploads/{upload_id}")

    # Data sources
    def create_data_source(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/pools/data-sources", json_body=request)

    def list_data_sources(
        self,
        *,
        state: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", "/v1/pools/data-sources", params=params)

    def get_data_source(self, data_source_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/data-sources/{data_source_id}")

    def update_data_source(self, data_source_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/v1/pools/data-sources/{data_source_id}", json_body=request)

    def refresh_data_source(self, data_source_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/data-sources/{data_source_id}/refresh", json_body={})

    # Assemblies
    def create_assembly(
        self,
        *,
        data_source_id: str,
        exclusion_patterns: Optional[list[str]] = None,
        runtime_type: Literal["custom_container", "managed_template"] = "custom_container",
        template_name: Optional[str] = None,
        agent_model: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "data_source_id": data_source_id,
            "runtime_type": runtime_type,
        }
        if exclusion_patterns is not None:
            payload["exclusion_patterns"] = exclusion_patterns
        if agent_model is not None:
            payload["agent_model"] = agent_model
        if runtime_type == "managed_template" and template_name:
            payload["template_name"] = template_name
        return self._request("POST", "/v1/pools/assemblies", json_body=payload)

    def list_assemblies(
        self,
        *,
        state: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", "/v1/pools/assemblies", params=params)

    def get_assembly(self, assembly_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/assemblies/{assembly_id}")

    def stream_assembly_events(self, assembly_id: str, *, cursor: Optional[str] = None) -> Iterator[dict[str, Any]]:
        import httpx

        params = {"cursor": cursor} if cursor is not None else None
        with httpx.stream(
            "GET",
            join_url(self._backend_base, f"/v1/pools/assemblies/{assembly_id}/events"),
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if text.startswith("data:"):
                    payload = text[5:].strip()
                    if payload:
                        yield json.loads(payload)

    # Pools
    def create_pool(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/pools", json_body=request)

    def list_pools(self, *, state: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", "/v1/pools", params=params)

    def get_pool(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}")

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.create_pool(request)

    def list(
        self,
        *,
        state: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        return self.list_pools(state=state, limit=limit, cursor=cursor)

    def get(self, pool_id: str) -> dict[str, Any]:
        return self.get_pool(pool_id)

    def replace_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/v1/pools/{pool_id}", json_body=request)

    def update_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/v1/pools/{pool_id}", json_body=request)

    def update(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.update_pool(pool_id, request)

    def replace(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.replace_pool(pool_id, request)

    def delete_pool(self, pool_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/pools/{pool_id}")

    def delete(self, pool_id: str) -> dict[str, Any]:
        return self.delete_pool(pool_id)

    def get_pool_urls(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/urls")

    def get_urls(self, pool_id: str) -> dict[str, Any]:
        return self.get_pool_urls(pool_id)

    def get_pool_metrics(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/metrics")

    def get_metrics(self, pool_id: str) -> dict[str, Any]:
        return self.get_pool_metrics(pool_id)

    def reassemble_pool(
        self,
        pool_id: str,
        *,
        exclusion_patterns: Optional[list[str]] = None,
        runtime_type: Literal["custom_container", "managed_template"] = "custom_container",
        template_name: Optional[str] = None,
        agent_model: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"runtime_type": runtime_type}
        if exclusion_patterns is not None:
            payload["exclusion_patterns"] = exclusion_patterns
        if agent_model is not None:
            payload["agent_model"] = agent_model
        if runtime_type == "managed_template" and template_name:
            payload["template_name"] = template_name
        return self._request("POST", f"/v1/pools/{pool_id}/assemblies", json_body=payload)

    def reassemble(
        self,
        pool_id: str,
        *,
        exclusion_patterns: Optional[list[str]] = None,
        runtime_type: str = "custom_container",
        template_name: str | None = None,
        agent_model: str | None = None,
    ) -> dict[str, Any]:
        return self.reassemble_pool(
            pool_id,
            exclusion_patterns=exclusion_patterns,
            runtime_type=runtime_type,  # type: ignore[arg-type]
            template_name=template_name,
            agent_model=agent_model,
        )

    # Tasks
    def list_tasks(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks")

    def create_task(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/tasks", json_body=request)

    def update_task(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/v1/pools/{pool_id}/tasks/{task_id}", json_body=request)

    def patch_task(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/v1/pools/{pool_id}/tasks/{task_id}", json_body=request)

    def delete_task(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/pools/{pool_id}/tasks/{task_id}")

    def materialize_task(self, pool_id: str, task_id: str, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/materialize",
            json_body=request or {},
        )

    def cleanup_task(self, pool_id: str, task_id: str, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/cleanup",
            json_body=request or {},
        )

    def warm_task(self, pool_id: str, task_id: str, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/warm",
            json_body=request or {},
        )

    def cool_task(self, pool_id: str, task_id: str, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/cool",
            json_body=request or {},
        )

    def list_task_instances(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/instances")

    def warm_task_instances(self, pool_id: str, task_id: str, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/instances/warm",
            json_body=request or {},
        )

    def delete_task_instance(self, pool_id: str, task_id: str, instance_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/pools/{pool_id}/tasks/{task_id}/instances/{instance_id}")

    # Runtime images
    def list_runtime_images(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/runtime-images")

    def create_runtime_image(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/runtime-images", json_body=request)

    def get_runtime_image(self, pool_id: str, release_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/runtime-images/{release_id}")

    def delete_runtime_image(self, pool_id: str, release_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/pools/{pool_id}/runtime-images/{release_id}")

    def bind_runtime_image(self, pool_id: str, release_id: str, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/runtime-images/{release_id}/bind",
            json_body=request or {},
        )

    def bind_runtime_image_to_task(
        self,
        pool_id: str,
        task_id: str,
        release_id: str,
        request: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/runtime-images/{release_id}/bind",
            json_body=request or {},
        )

    # Shared bundles
    def list_shared_bundles(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/shared-bundles")

    def create_shared_bundle(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/shared-bundles", json_body=request)

    def bind_shared_bundle(self, pool_id: str, release_id: str, request: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/shared-bundles/{release_id}/bind",
            json_body=request or {},
        )

    # Task bundles
    def list_task_bundles(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/task-bundles")

    def create_task_bundle(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/tasks/{task_id}/task-bundles", json_body=request)

    def bind_task_bundle(
        self,
        pool_id: str,
        task_id: str,
        release_id: str,
        request: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/task-bundles/{release_id}/bind",
            json_body=request or {},
        )

    # Instance bundles
    def list_instance_bundles(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/instance-bundles")

    def create_instance_bundle(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/tasks/{task_id}/instance-bundles", json_body=request)

    def bind_instance_bundle(
        self,
        pool_id: str,
        task_id: str,
        release_id: str,
        *,
        instance_key: Optional[str] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if instance_key is not None:
            body["instance_key"] = instance_key
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/instance-bundles/{release_id}/bind",
            json_body=body,
        )

    # Pool-scoped rollouts
    def create_rollout(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/rollouts", json_body=request)

    def get_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}")

    def list_rollouts(self, pool_id: str, *, state: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts", params=params)

    def cancel_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel", json_body={})

    def get_rollout_artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts")

    def get_rollout_usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/usage")

    def get_rollout_summary(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/summary")

    def stream_rollout_events(self, pool_id: str, rollout_id: str, *, cursor: Optional[str] = None) -> Iterator[dict[str, Any]]:
        import httpx

        params = {"cursor": cursor} if cursor is not None else None
        with httpx.stream(
            "GET",
            join_url(self._backend_base, f"/v1/pools/{pool_id}/rollouts/{rollout_id}/events"),
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if text.startswith("data:"):
                    payload = text[5:].strip()
                    if payload:
                        yield json.loads(payload)

    # Global rollouts (no pool_id in path)
    def create_global_rollout(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/rollouts", json_body=request)

    def list_global_rollouts(
        self,
        *,
        state: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", "/v1/rollouts", params=params)

    def get_global_rollout(self, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/rollouts/{rollout_id}")

    def cancel_global_rollout(self, rollout_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/rollouts/{rollout_id}/cancel", json_body={})

    def get_global_rollout_artifacts(self, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/rollouts/{rollout_id}/artifacts")

    def get_global_rollout_usage(self, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/rollouts/{rollout_id}/usage")

    def get_global_rollout_summary(self, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/rollouts/{rollout_id}/summary")

    def stream_global_rollout_events(self, rollout_id: str, *, cursor: Optional[str] = None) -> Iterator[dict[str, Any]]:
        import httpx

        params = {"cursor": cursor} if cursor is not None else None
        with httpx.stream(
            "GET",
            join_url(self._backend_base, f"/v1/rollouts/{rollout_id}/events"),
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if text.startswith("data:"):
                    payload = text[5:].strip()
                    if payload:
                        yield json.loads(payload)

    # Container probes (pool- and task-scoped)
    def get_pool_container_health(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/container/health")

    def get_task_container_health(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/health")

    def get_pool_container_info(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/container/info")

    def get_task_container_info(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/info")

    def get_pool_container_task_info(self, pool_id: str, *, seeds: Optional[list[int]] = None) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/pools/{pool_id}/container/task_info",
            params_list=_seed_query_params(seeds),
        )

    def get_task_container_task_info(self, pool_id: str, task_id: str, *, seeds: Optional[list[int]] = None) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/pools/{pool_id}/tasks/{task_id}/container/task_info",
            params_list=_seed_query_params(seeds),
        )

    def get_pool_container_metadata(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/container/metadata")

    def get_task_container_metadata(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/metadata")

    def execute_pool_container_rollout(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/container/rollout", json_body=request)

    def execute_task_container_rollout(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/tasks/{task_id}/container/rollout", json_body=request)

    def prompt_learning_evaluate_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/container/prompt-learning/evaluate", json_body=request)

    def prompt_learning_evaluate_task(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/container/prompt-learning/evaluate",
            json_body=request,
        )

    # Misc control-plane
    def get_queue_status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/queue/status")

    def get_capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/v1/capabilities")


class AsyncContainerPoolsClient(_AsyncThreadProxy):
    """Async adapter over ``ContainerPoolsClient`` and its namespaces."""


__all__ = [
    "AsyncContainerPoolsClient",
    "CANONICAL_ROLLOUT_REQUEST_KEYS",
    "ContainerPoolsClient",
    "PoolTarget",
    "validate_pool_rollout_request",
]
