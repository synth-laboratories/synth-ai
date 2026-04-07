"""Python-only container pools SDK."""

from __future__ import annotations

import asyncio
import json
import os
from enum import Enum
from typing import Any, Iterator

from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base

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


class PoolTarget(str, Enum):
    HARBOR = "harbor"
    OPENENV = "openenv"
    HORIZONS = "horizons"
    ARBITRARY = "arbitrary"


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
        return attr


class _PoolRolloutsClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def create(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        validate_pool_rollout_request(request, context="pools.rollouts.create")
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
    ) -> Iterator[dict[str, Any]]:
        return self._raw.stream_rollout_events(pool_id, rollout_id, cursor=cursor)


class _AgentRolloutsClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        validate_pool_rollout_request(request, context="rollouts.create")
        return self._raw.create_global_rollout(request)

    def get(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout(rollout_id)

    def list(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._raw.list_global_rollouts(state=state, limit=limit, cursor=cursor)

    def cancel(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.cancel_global_rollout(rollout_id)

    def artifacts(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout_artifacts(rollout_id)

    def usage(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout_usage(rollout_id)

    def summary(self, rollout_id: str) -> dict[str, Any]:
        return self._raw.get_global_rollout_summary(rollout_id)

    def events(self, rollout_id: str, *, cursor: str | None = None) -> Iterator[dict[str, Any]]:
        return self._raw.stream_global_rollout_events(rollout_id, cursor=cursor)


class _PoolTasksClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
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


class _PoolMetricsClient:
    def __init__(self, raw: ContainerPoolsClient) -> None:
        self._raw = raw

    def get(self, pool_id: str) -> dict[str, Any]:
        return self._raw.get_pool_metrics(pool_id)


class ContainerPoolsClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = (api_key or os.getenv("SYNTH_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        resolved_base = backend_base or base_url or BACKEND_URL_BASE
        self._backend_base = normalize_backend_base(resolved_base)
        self._timeout = timeout
        self.rollouts = _PoolRolloutsClient(self)
        self.agent_rollouts = _AgentRolloutsClient(self)
        self.tasks = _PoolTasksClient(self)
        self.metrics = _PoolMetricsClient(self)

    @property
    def raw(self) -> ContainerPoolsClient:
        return self

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        import httpx

        resp = httpx.request(
            method,
            join_url(self._backend_base, path),
            headers=self._headers(),
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        if not resp.content:
            return {}
        return resp.json()

    def _stream(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        import httpx

        with httpx.stream(
            "GET",
            join_url(self._backend_base, path),
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if payload:
                    yield json.loads(payload)

    def create_pool(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/pools", json_body=request)

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.create_pool(request)

    def list_pools(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params = {
            k: v
            for k, v in {"state": state, "limit": limit, "cursor": cursor}.items()
            if v is not None
        }
        return self._request("GET", "/v1/pools", params=params)

    def list(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self.list_pools(state=state, limit=limit, cursor=cursor)

    def get_pool(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}")

    def get(self, pool_id: str) -> dict[str, Any]:
        return self.get_pool(pool_id)

    def replace_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/v1/pools/{pool_id}", json_body=request)

    def replace(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.replace_pool(pool_id, request)

    def update_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/v1/pools/{pool_id}", json_body=request)

    def update(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.update_pool(pool_id, request)

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

    def create_rollout(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        validate_pool_rollout_request(request, context="pools.create_rollout")
        return self._request("POST", f"/v1/pools/{pool_id}/rollouts", json_body=request)

    def get_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}")

    def list_rollouts(
        self,
        pool_id: str,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params = {
            k: v
            for k, v in {"state": state, "limit": limit, "cursor": cursor}.items()
            if v is not None
        }
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts", params=params)

    def cancel_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel", json_body={}
        )

    def get_rollout_artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts")

    def get_rollout_usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/usage")

    def get_rollout_summary(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/summary")

    def stream_rollout_events(
        self,
        pool_id: str,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        params = {"cursor": cursor} if cursor is not None else None
        return self._stream(f"/v1/pools/{pool_id}/rollouts/{rollout_id}/events", params=params)

    def create_global_rollout(self, request: dict[str, Any]) -> dict[str, Any]:
        validate_pool_rollout_request(request, context="rollouts.create")
        return self._request("POST", "/v1/rollouts", json_body=request)

    def list_global_rollouts(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params = {
            k: v
            for k, v in {"state": state, "limit": limit, "cursor": cursor}.items()
            if v is not None
        }
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

    def stream_global_rollout_events(
        self,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        params = {"cursor": cursor} if cursor is not None else None
        return self._stream(f"/v1/rollouts/{rollout_id}/events", params=params)

    def get_pool_container_health(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/container/health")

    def get_task_container_health(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/health")

    def get_pool_container_info(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/container/info")

    def get_task_container_info(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/info")

    def get_pool_container_metadata(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/container/metadata")

    def get_task_container_metadata(self, pool_id: str, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/metadata")

    def execute_pool_container_rollout(
        self, pool_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/container/rollout", json_body=request)

    def execute_task_container_rollout(
        self, pool_id: str, task_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/pools/{pool_id}/tasks/{task_id}/container/rollout", json_body=request
        )

    def prompt_learning_evaluate_pool(
        self, pool_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/pools/{pool_id}/container/prompt-learning/evaluate", json_body=request
        )

    def prompt_learning_evaluate_task(
        self, pool_id: str, task_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/pools/{pool_id}/tasks/{task_id}/container/prompt-learning/evaluate",
            json_body=request,
        )

    def get_queue_status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/queue/status")

    def get_capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/v1/capabilities")


class AsyncContainerPoolsClient(_AsyncThreadProxy):
    """Async adapter over ``ContainerPoolsClient``."""


__all__ = [
    "AsyncContainerPoolsClient",
    "CANONICAL_ROLLOUT_REQUEST_KEYS",
    "ContainerPoolsClient",
    "PoolTarget",
    "validate_pool_rollout_request",
]
