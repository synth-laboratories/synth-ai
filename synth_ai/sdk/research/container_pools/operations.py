"""Container-pool operation registry.

Deliberately separate from ``sdk/research/operations.py``. That module is gated
against the vendored ``openapi/research-v1.json`` by an exact set comparison
(``testing/scripts/check_research_openapi_contract.py`` walks every
``_operation(...)`` call in that one file), and container pools are a different
backend API — ``/v1/pools``, not ``/smr/*``. Adding these entries there would
fail the gate as ``extra`` operations forever.

The backend does not yet publish a curated pools contract to vendor here:
``GET /v1/pools/openapi.json`` returns the whole FastAPI application schema.
Until it does, this registry is the source of truth and has no drift gate.
"""

from __future__ import annotations

from synth_ai.core.http.request import HttpMethod, OperationId, OperationMetadata


def _operation(
    operation_id: str,
    method: HttpMethod,
    path: str,
    *,
    mutation: bool = False,
    idempotent: bool = False,
) -> OperationMetadata:
    return OperationMetadata(OperationId(operation_id), method, path, mutation, idempotent)


CONTAINER_POOL_OPERATIONS = {
    operation.operation_id: operation
    for operation in (
        # Pools
        _operation("list_container_pools", HttpMethod.GET, "/v1/pools"),
        _operation("create_container_pool", HttpMethod.POST, "/v1/pools", mutation=True),
        _operation("get_container_pool", HttpMethod.GET, "/v1/pools/{pool_id}"),
        _operation(
            "delete_container_pool",
            HttpMethod.DELETE,
            "/v1/pools/{pool_id}",
            mutation=True,
            idempotent=True,
        ),
        # Tasks
        _operation("list_container_pool_tasks", HttpMethod.GET, "/v1/pools/{pool_id}/tasks"),
        _operation(
            "create_container_pool_task",
            HttpMethod.POST,
            "/v1/pools/{pool_id}/tasks",
            mutation=True,
        ),
        _operation(
            "update_container_pool_task",
            HttpMethod.PATCH,
            "/v1/pools/{pool_id}/tasks/{task_id}",
            mutation=True,
        ),
        # Runtime image releases — the deployment primitive.
        _operation(
            "list_container_pool_runtime_image_releases",
            HttpMethod.GET,
            "/v1/pools/{pool_id}/runtime_image_releases",
        ),
        _operation(
            "create_container_pool_runtime_image_release",
            HttpMethod.POST,
            "/v1/pools/{pool_id}/runtime_image_releases",
            mutation=True,
        ),
        _operation(
            "get_container_pool_runtime_image_release",
            HttpMethod.GET,
            "/v1/pools/{pool_id}/runtime_image_releases/{release_id}",
        ),
        # Bind is what actually builds the snapshot; it is not idempotent today.
        _operation(
            "bind_container_pool_runtime_image_release",
            HttpMethod.POST,
            "/v1/pools/{pool_id}/runtime_image_releases/{release_id}/bind",
            mutation=True,
        ),
        # Rollouts
        _operation(
            "create_container_pool_rollout",
            HttpMethod.POST,
            "/v1/pools/{pool_id}/rollouts",
            mutation=True,
        ),
        _operation("list_container_pool_rollouts", HttpMethod.GET, "/v1/pools/{pool_id}/rollouts"),
        _operation(
            "get_container_pool_rollout",
            HttpMethod.GET,
            "/v1/pools/{pool_id}/rollouts/{rollout_id}",
        ),
        _operation(
            "cancel_container_pool_rollout",
            HttpMethod.POST,
            "/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel",
            mutation=True,
            idempotent=True,
        ),
        _operation(
            "list_container_pool_rollout_artifacts",
            HttpMethod.GET,
            "/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts",
        ),
        _operation(
            "get_container_pool_rollout_usage",
            HttpMethod.GET,
            "/v1/pools/{pool_id}/rollouts/{rollout_id}/usage",
        ),
    )
}


def container_pool_operation(operation_id: str) -> OperationMetadata:
    try:
        return CONTAINER_POOL_OPERATIONS[OperationId(operation_id)]
    except KeyError as error:
        raise ValueError(f"unknown container pool operation_id {operation_id!r}") from error


__all__ = ["CONTAINER_POOL_OPERATIONS", "container_pool_operation"]
