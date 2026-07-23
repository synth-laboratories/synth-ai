"""Container pools, tasks, rollouts, metrics, artifacts, usage, and events.

Access this API through ``SynthClient().pools``. Prefer its task-oriented
namespaces:

* ``pools.rollouts`` for rollouts scoped to one pool.
* ``pools.agent_rollouts`` for global rollouts.
* ``pools.tasks`` for pool task definitions.
* ``pools.metrics`` for pool utilization.

The lower-level methods remain available on ``pools.raw``. Request and response
bodies are backend-owned JSON objects. ``pools.rollouts.create``,
``pools.agent_rollouts.create``, ``pools.raw.create_rollout``, and
``pools.raw.create_global_rollout`` validate their closed set of request keys.
Other raw methods pass backend contracts through without fabricating a second
schema authority.

Example:
    from synth_ai import SynthClient

    client = SynthClient()
    try:
        page = client.pools.list(limit=20)
        for pool in page.get("items", []):
            metrics = client.pools.metrics.get(pool["id"])
            print(pool["id"], metrics)
    finally:
        client.pools.close()

Expected output:
    Expected output is one line per pool containing its stable ID and the
    service-owned metrics object. An organization with no pools returns an
    empty ``items`` list.

Pagination:
    ``limit`` defaults to ``100``. Pagination is backend-owned: continue only
    when a response supplies a distinct continuation token such as
    ``next_cursor``. The current production service may omit a continuation or
    echo the supplied ``cursor``; stop in either case to avoid a loop.

Errors:
    The four rollout-creation entry points named above reject unsupported keys
    locally with ``ValueError``.
    Service failures raise ``httpx.HTTPStatusError``. Inspect
    ``error.response.status_code`` to distinguish authentication,
    authorization, validation, capacity, and transient service failures.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Iterator

from synth_ai.sdk.base import SynthBaseClient

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
    """Reject rollout payloads with non-canonical keys."""
    invalid_keys = sorted(key for key in request if key not in CANONICAL_ROLLOUT_REQUEST_KEYS)
    if invalid_keys:
        invalid_list = ", ".join(invalid_keys)
        raise ValueError(
            f"{context} request contains unsupported keys ({invalid_list}); "
            "use canonical rollout fields only."
        )


class PoolTarget(str, Enum):
    """Deployment target substrate for a pool."""

    HARBOR = "harbor"
    OPENENV = "openenv"
    HORIZONS = "horizons"
    HORIZONS_PRIVATE = "horizons_private"
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


class ContainerPoolsClient(SynthBaseClient):
    """Manage container pools, tasks, rollouts, and metrics.

    Args:
        api_key: Synth API key. Defaults to ``SYNTH_API_KEY``.
        backend_base: Preferred API base URL override.
        base_url: Compatibility alias for ``backend_base``.
        timeout: Default request timeout in seconds. Defaults to ``30``.
        timeout_seconds: Preferred timeout override. When supplied, it takes
            precedence over ``timeout``.

    Attributes:
        rollouts: Pool-scoped rollout operations.
        agent_rollouts: Global rollout operations.
        tasks: Pool task operations.
        metrics: Pool utilization reads.

    Raises:
        ValueError: The API key is missing or a rollout request contains a
            non-canonical key.
        httpx.HTTPStatusError: The service rejects an operation.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            backend_base=backend_base or base_url,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else timeout,
        )
        self.rollouts = _PoolRolloutsClient(self)
        self.agent_rollouts = _AgentRolloutsClient(self)
        self.tasks = _PoolTasksClient(self)
        self.metrics = _PoolMetricsClient(self)

    @property
    def raw(self) -> ContainerPoolsClient:
        """Return ``self`` for advanced raw HTTP access."""
        return self

    def create_pool(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create one container pool.

        Args:
            request: Backend-owned pool definition. Use fields documented by
                the active pool API contract.

        Returns:
            The created pool JSON object, including its service-assigned ID.

        Raises:
            httpx.HTTPStatusError: Authentication, authorization, validation,
                conflict, quota, or service errors.
        """
        return self._request("POST", "/v1/pools", json_body=request)

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create a container pool (alias for ``create_pool``)."""
        return self.create_pool(request)

    def list_pools(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List container pools with cursor pagination.

        Args:
            state: Optional service-owned lifecycle-state filter. Defaults to
                no state filter.
            limit: Maximum page size. Defaults to ``100``.
            cursor: Backend-owned continuation token. Defaults to ``None``.

        Returns:
            The backend page object. Stop if its continuation is missing or unchanged.

        Raises:
            httpx.HTTPStatusError: The request is rejected.
        """
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
        """List container pools (alias for ``list_pools``)."""
        return self.list_pools(state=state, limit=limit, cursor=cursor)

    def get_pool(self, pool_id: str) -> dict[str, Any]:
        """Retrieve one pool by ID.

        Args:
            pool_id: Stable ID returned by ``create_pool`` or ``list_pools``.

        Returns:
            The current pool JSON object.

        Raises:
            httpx.HTTPStatusError: The pool is unavailable or the request is
                rejected.
        """
        return self._request("GET", f"/v1/pools/{pool_id}")

    def get(self, pool_id: str) -> dict[str, Any]:
        """Retrieve a pool by id (alias for ``get_pool``)."""
        return self.get_pool(pool_id)

    def replace_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Replace a pool definition."""
        return self._request("PUT", f"/v1/pools/{pool_id}", json_body=request)

    def replace(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Replace a pool definition (alias for ``replace_pool``)."""
        return self.replace_pool(pool_id, request)

    def update_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Patch-update a pool definition."""
        return self._request("PATCH", f"/v1/pools/{pool_id}", json_body=request)

    def update(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Patch-update a pool (alias for ``update_pool``)."""
        return self.update_pool(pool_id, request)

    def delete_pool(self, pool_id: str) -> dict[str, Any]:
        """Delete a pool by id."""
        return self._request("DELETE", f"/v1/pools/{pool_id}")

    def delete(self, pool_id: str) -> dict[str, Any]:
        """Delete a pool by id (alias for ``delete_pool``)."""
        return self.delete_pool(pool_id)

    def get_pool_urls(self, pool_id: str) -> dict[str, Any]:
        """Return rollout URLs for a pool."""
        return self._request("GET", f"/v1/pools/{pool_id}/urls")

    def get_urls(self, pool_id: str) -> dict[str, Any]:
        """Return rollout URLs (alias for ``get_pool_urls``)."""
        return self.get_pool_urls(pool_id)

    def get_pool_metrics(self, pool_id: str) -> dict[str, Any]:
        """Return utilization metrics for a pool."""
        return self._request("GET", f"/v1/pools/{pool_id}/metrics")

    def list_tasks(self, pool_id: str) -> dict[str, Any]:
        """List tasks configured on a pool."""
        return self._request("GET", f"/v1/pools/{pool_id}/tasks")

    def create_task(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Create a task on a pool."""
        return self._request("POST", f"/v1/pools/{pool_id}/tasks", json_body=request)

    def update_task(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Replace a pool task definition."""
        return self._request("PUT", f"/v1/pools/{pool_id}/tasks/{task_id}", json_body=request)

    def patch_task(self, pool_id: str, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Patch-update a pool task."""
        return self._request("PATCH", f"/v1/pools/{pool_id}/tasks/{task_id}", json_body=request)

    def delete_task(self, pool_id: str, task_id: str) -> dict[str, Any]:
        """Delete a pool task."""
        return self._request("DELETE", f"/v1/pools/{pool_id}/tasks/{task_id}")

    def create_rollout(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Start one rollout on a pool.

        Args:
            pool_id: Target pool ID.
            request: Rollout request using only
                `CANONICAL_ROLLOUT_REQUEST_KEYS`.

        Returns:
            The created rollout JSON object.

        Raises:
            ValueError: ``request`` contains a non-canonical key.
            httpx.HTTPStatusError: The pool cannot accept the rollout or the
                request is rejected.
        """
        validate_pool_rollout_request(request, context="pools.create_rollout")
        return self._request("POST", f"/v1/pools/{pool_id}/rollouts", json_body=request)

    def get_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        """Fetch rollout state for a pool."""
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}")

    def list_rollouts(
        self,
        pool_id: str,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List rollouts for one pool.

        Args:
            pool_id: Target pool ID.
            state: Optional service-owned lifecycle-state filter. Defaults to
                no state filter.
            limit: Maximum page size. Defaults to ``100``.
            cursor: Backend-owned continuation token. Defaults to ``None``.

        Returns:
            The backend page object. Stop if its continuation is missing or unchanged.

        Raises:
            httpx.HTTPStatusError: The request is rejected.
        """
        params = {
            k: v
            for k, v in {"state": state, "limit": limit, "cursor": cursor}.items()
            if v is not None
        }
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts", params=params)

    def cancel_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        """Cancel an in-flight pool rollout."""
        return self._request(
            "POST", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel", json_body={}
        )

    def get_rollout_artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        """List artifacts produced by a pool rollout."""
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts")

    def get_rollout_usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        """Return usage totals for a pool rollout."""
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/usage")

    def get_rollout_summary(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        """Return a summary document for a pool rollout."""
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/summary")

    def stream_rollout_events(
        self,
        pool_id: str,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream rollout events for a pool."""
        params = {"cursor": cursor} if cursor is not None else None
        return self._stream(f"/v1/pools/{pool_id}/rollouts/{rollout_id}/events", params=params)

    def create_global_rollout(self, request: dict[str, Any]) -> dict[str, Any]:
        """Start a global (non-pool-scoped) rollout."""
        validate_pool_rollout_request(request, context="rollouts.create")
        return self._request("POST", "/v1/rollouts", json_body=request)

    def list_global_rollouts(
        self,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List global rollouts."""
        params = {
            k: v
            for k, v in {"state": state, "limit": limit, "cursor": cursor}.items()
            if v is not None
        }
        return self._request("GET", "/v1/rollouts", params=params)

    def get_global_rollout(self, rollout_id: str) -> dict[str, Any]:
        """Fetch a global rollout by id."""
        return self._request("GET", f"/v1/rollouts/{rollout_id}")

    def cancel_global_rollout(self, rollout_id: str) -> dict[str, Any]:
        """Cancel a global rollout."""
        return self._request("POST", f"/v1/rollouts/{rollout_id}/cancel", json_body={})

    def get_global_rollout_artifacts(self, rollout_id: str) -> dict[str, Any]:
        """List artifacts for a global rollout."""
        return self._request("GET", f"/v1/rollouts/{rollout_id}/artifacts")

    def get_global_rollout_usage(self, rollout_id: str) -> dict[str, Any]:
        """Return usage totals for a global rollout."""
        return self._request("GET", f"/v1/rollouts/{rollout_id}/usage")

    def get_global_rollout_summary(self, rollout_id: str) -> dict[str, Any]:
        """Return a summary for a global rollout."""
        return self._request("GET", f"/v1/rollouts/{rollout_id}/summary")

    def stream_global_rollout_events(
        self,
        rollout_id: str,
        *,
        cursor: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream events for a global rollout."""
        params = {"cursor": cursor} if cursor is not None else None
        return self._stream(f"/v1/rollouts/{rollout_id}/events", params=params)

    def get_pool_container_health(self, pool_id: str) -> dict[str, Any]:
        """Return container health for a pool."""
        return self._request("GET", f"/v1/pools/{pool_id}/container/health")

    def get_task_container_health(self, pool_id: str, task_id: str) -> dict[str, Any]:
        """Return container health for a pool task."""
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/health")

    def get_pool_container_info(self, pool_id: str) -> dict[str, Any]:
        """Return container info for a pool."""
        return self._request("GET", f"/v1/pools/{pool_id}/container/info")

    def get_task_container_info(self, pool_id: str, task_id: str) -> dict[str, Any]:
        """Return container info for a pool task."""
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/info")

    def get_pool_container_metadata(self, pool_id: str) -> dict[str, Any]:
        """Return container metadata for a pool."""
        return self._request("GET", f"/v1/pools/{pool_id}/container/metadata")

    def get_task_container_metadata(self, pool_id: str, task_id: str) -> dict[str, Any]:
        """Return container metadata for a pool task."""
        return self._request("GET", f"/v1/pools/{pool_id}/tasks/{task_id}/container/metadata")

    def execute_pool_container_rollout(
        self, pool_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a one-off container rollout on a pool."""
        return self._request("POST", f"/v1/pools/{pool_id}/container/rollout", json_body=request)

    def execute_task_container_rollout(
        self, pool_id: str, task_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a container rollout for a specific task."""
        return self._request(
            "POST", f"/v1/pools/{pool_id}/tasks/{task_id}/container/rollout", json_body=request
        )

    def get_queue_status(self) -> dict[str, Any]:
        """Return global queue saturation status."""
        return self._request("GET", "/v1/queue/status")

    def get_capabilities(self) -> dict[str, Any]:
        """Return backend capability flags for pools."""
        return self._request("GET", "/v1/capabilities")


class AsyncContainerPoolsClient(_AsyncThreadProxy):
    """Async adapter over ``ContainerPoolsClient``."""


PoolsClient = ContainerPoolsClient


__all__ = [
    "AsyncContainerPoolsClient",
    "CANONICAL_ROLLOUT_REQUEST_KEYS",
    "ContainerPoolsClient",
    "PoolsClient",
    "PoolTarget",
    "validate_pool_rollout_request",
]
