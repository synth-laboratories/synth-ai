"""Container pools: create, deploy a release, bind it, run rollouts.

The deployment primitive is the **runtime image release**, not the pool. A pool
holds tasks; a release holds the image (and, for the Harbor subtype, the task
bundle); binding a release to a task is what builds and attaches a snapshot.
Several tasks in one pool may bind the same release, which is what makes a
taskset cost one build.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.container_pools.contracts import (
    HARBOR_CONTAINER_SUBTYPE,
    Pool,
    PoolId,
    PoolTask,
    Rollout,
    RolloutArtifact,
    RolloutId,
    RuntimeImageRelease,
    RuntimeImageReleaseId,
)
from synth_ai.sdk.research.container_pools.operations import container_pool_operation
from synth_ai.sdk.research.container_pools.packaging import build_harbor_bundle_archive
from synth_ai.sdk.research.contracts.managed_inference import ManagedInference

#: Poll interval while waiting on a rollout. Harbor rollouts run for minutes to
#: hours, so a tight loop buys nothing and costs request quota.
DEFAULT_POLL_INTERVAL_SECONDS = 5.0


class RolloutTimeoutError(TimeoutError):
    """A rollout did not reach a terminal status inside the wait budget."""


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
    timeout_seconds: float | None = None,
) -> HttpRequest:
    return HttpRequest(
        container_pool_operation(operation_id),
        path,
        query=query or {},
        body=body,
        timeout_seconds=timeout_seconds,
    )


def _seg(value: str, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return quote(text, safe="")


def _items(value: JsonValue, *keys: str) -> list[JsonValue]:
    """Unwrap a list response that may be bare or wrapped under a key."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in keys:
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
    return []


def _json_object(value: Mapping[str, Any] | None) -> JsonObject:
    return {str(key): item for key, item in dict(value or {}).items()}


class ContainerPoolsAPI:
    """Pools, their tasks, their releases, and their rollouts."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    # -- pools ----------------------------------------------------------

    def list(self) -> tuple[Pool, ...]:
        value = self._transport.execute(_request("list_container_pools", "/v1/pools"))
        return tuple(Pool.from_wire(item) for item in _items(value, "pools", "data"))

    def create(
        self,
        *,
        pool_id: str,
        backend: str = "arbitrary",
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> Pool:
        """Create a pool.

        ``backend`` defaults to ``arbitrary``: runtime image releases are only
        accepted on ``arbitrary`` and ``smr_session`` pools, so a Harbor task
        belongs on an arbitrary pool with ``container_subtype="harbor"`` on its
        release — not on a ``backend="harbor"`` pool.
        """
        body: JsonObject = {"pool_id": pool_id, "backend": backend}
        if name:
            body["name"] = name
        if metadata:
            body["metadata"] = _json_object(metadata)
        body.update(_json_object(extra))
        return Pool.from_wire(
            self._transport.execute(_request("create_container_pool", "/v1/pools", body=body))
        )

    def get(self, pool_id: PoolId | str) -> Pool:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
        return Pool.from_wire(self._transport.execute(_request("get_container_pool", path)))

    def delete(self, pool_id: PoolId | str) -> None:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
        self._transport.execute(_request("delete_container_pool", path))

    # -- tasks ----------------------------------------------------------

    def list_tasks(self, pool_id: PoolId | str) -> tuple[PoolTask, ...]:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}/tasks"
        value = self._transport.execute(_request("list_container_pool_tasks", path))
        return tuple(PoolTask.from_wire(item) for item in _items(value, "tasks", "data"))

    def create_task(
        self,
        pool_id: PoolId | str,
        *,
        task_id: str,
        backend: str = "arbitrary",
        task_config: Mapping[str, Any] | None = None,
        task_metadata: Mapping[str, Any] | None = None,
        inference: ManagedInference | None = None,
    ) -> PoolTask:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}/tasks"
        body: JsonObject = {"task_id": task_id, "backend": backend}
        normalized_task_config = _json_object(task_config)
        if inference is not None:
            normalized_task_config["inference"] = inference.to_wire()
        if normalized_task_config:
            body["task_config"] = normalized_task_config
        if task_metadata:
            body["task_metadata"] = _json_object(task_metadata)
        return PoolTask.from_wire(
            self._transport.execute(_request("create_container_pool_task", path, body=body))
        )

    # -- releases -------------------------------------------------------

    def list_releases(self, pool_id: PoolId | str) -> tuple[RuntimeImageRelease, ...]:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}/runtime_image_releases"
        value = self._transport.execute(
            _request("list_container_pool_runtime_image_releases", path)
        )
        return tuple(
            RuntimeImageRelease.from_wire(item)
            for item in _items(value, "runtime_image_releases", "releases", "data")
        )

    def create_release(
        self,
        pool_id: PoolId | str,
        *,
        source_kind: str,
        archive_base64: str | None = None,
        image_ref: str | None = None,
        compute_provider: str | None = None,
        name: str | None = None,
        dockerfile_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        upload_timeout_seconds: float = 600.0,
    ) -> RuntimeImageRelease:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}/runtime_image_releases"
        body: JsonObject = {"source_kind": source_kind}
        if archive_base64:
            body["archive_base64"] = archive_base64
        if image_ref:
            body["image_ref"] = image_ref
        if compute_provider:
            body["compute_provider"] = compute_provider
        if name:
            body["name"] = name
        if dockerfile_path:
            body["dockerfile_path"] = dockerfile_path
        if metadata:
            body["metadata"] = _json_object(metadata)
        return RuntimeImageRelease.from_wire(
            self._transport.execute(
                _request(
                    "create_container_pool_runtime_image_release",
                    path,
                    body=body,
                    # A bundle upload is a single large request; the default
                    # transport timeout is sized for control-plane calls.
                    timeout_seconds=upload_timeout_seconds,
                )
            )
        )

    def create_harbor_release(
        self,
        pool_id: PoolId | str,
        *,
        bundle_dir: str | Path,
        model_name: str,
        reasoning_effort: str = "medium",
        harness_subtype: str = "codex",
        compute_provider: str = "daytona",
        name: str | None = None,
        dockerfile_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_credential_files: bool = False,
        collect_phase_logs: bool | None = None,
    ) -> RuntimeImageRelease:
        """Package a Harbor task bundle and publish it as a release.

        Rejects a bundle that cannot deploy before uploading it — see
        :mod:`.packaging` for the build-context and credential rules.
        """
        bundle = build_harbor_bundle_archive(
            bundle_dir,
            dockerfile_path=dockerfile_path or "environment/Dockerfile",
            allow_credential_files=allow_credential_files,
        )
        release_metadata: JsonObject = {
            "container_subtype": HARBOR_CONTAINER_SUBTYPE,
            "container_harness_subtype": harness_subtype,
            "model_name": model_name,
            "harness_kwargs": {"reasoning_effort": reasoning_effort},
        }
        if dockerfile_path:
            release_metadata["harbor_dockerfile_path"] = dockerfile_path
        if collect_phase_logs is not None:
            # Whether the backend collects /logs from the sandbox into rollout
            # artifacts. On by default; a failed rollout otherwise discards the
            # only evidence of why the agent phase failed, and every distinct
            # failure reports the same exit code.
            release_metadata["collect_phase_logs"] = bool(collect_phase_logs)
        release_metadata.update(_json_object(metadata))
        return self.create_release(
            pool_id,
            source_kind="docker_context",
            archive_base64=bundle.archive_base64,
            compute_provider=compute_provider,
            name=name or bundle.task_name,
            dockerfile_path=dockerfile_path,
            metadata=release_metadata,
        )

    def get_release(
        self, pool_id: PoolId | str, release_id: RuntimeImageReleaseId | str
    ) -> RuntimeImageRelease:
        path = (
            f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
            f"/runtime_image_releases/{_seg(release_id, field_name='release_id')}"
        )
        return RuntimeImageRelease.from_wire(
            self._transport.execute(_request("get_container_pool_runtime_image_release", path))
        )

    def bind_release(
        self,
        pool_id: PoolId | str,
        release_id: RuntimeImageReleaseId | str,
        *,
        bind_timeout_seconds: float = 1800.0,
    ) -> JsonValue:
        """Bind a release to the pool's tasks, building the snapshot.

        This is the slow call: it builds an image. Pool-level bind currently
        requires the pool to hold exactly one active task.
        """
        path = (
            f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
            f"/runtime_image_releases/{_seg(release_id, field_name='release_id')}/bind"
        )
        return self._transport.execute(
            _request(
                "bind_container_pool_runtime_image_release",
                path,
                timeout_seconds=bind_timeout_seconds,
            )
        )

    # -- rollouts -------------------------------------------------------

    def create_rollout(
        self,
        pool_id: PoolId | str,
        *,
        task_id: str | None = None,
        seed: int | None = None,
        policy: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        env: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> Rollout:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}/rollouts"
        body: JsonObject = {}
        if task_id:
            body["task_id"] = task_id
        if seed is not None:
            body["seed"] = int(seed)
        if policy:
            # The backend 422s when policy.model disagrees with the task-pinned
            # harbor_agent.model_name, so leaving this unset is usually right.
            body["policy"] = _json_object(policy)
        if metadata:
            body["metadata"] = _json_object(metadata)
        if env:
            body["env"] = _json_object(env)
        body.update(_json_object(extra))
        return Rollout.from_wire(
            self._transport.execute(_request("create_container_pool_rollout", path, body=body))
        )

    def get_rollout(self, pool_id: PoolId | str, rollout_id: RolloutId | str) -> Rollout:
        path = (
            f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
            f"/rollouts/{_seg(rollout_id, field_name='rollout_id')}"
        )
        return Rollout.from_wire(
            self._transport.execute(_request("get_container_pool_rollout", path))
        )

    def list_rollouts(self, pool_id: PoolId | str) -> tuple[Rollout, ...]:
        path = f"/v1/pools/{_seg(pool_id, field_name='pool_id')}/rollouts"
        value = self._transport.execute(_request("list_container_pool_rollouts", path))
        return tuple(Rollout.from_wire(item) for item in _items(value, "rollouts", "data"))

    def cancel_rollout(self, pool_id: PoolId | str, rollout_id: RolloutId | str) -> JsonValue:
        path = (
            f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
            f"/rollouts/{_seg(rollout_id, field_name='rollout_id')}/cancel"
        )
        return self._transport.execute(_request("cancel_container_pool_rollout", path))

    def list_rollout_artifacts(
        self, pool_id: PoolId | str, rollout_id: RolloutId | str
    ) -> tuple[RolloutArtifact, ...]:
        path = (
            f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
            f"/rollouts/{_seg(rollout_id, field_name='rollout_id')}/artifacts"
        )
        value = self._transport.execute(_request("list_container_pool_rollout_artifacts", path))
        return tuple(RolloutArtifact.from_wire(item) for item in _items(value, "artifacts", "data"))

    def get_rollout_usage(self, pool_id: PoolId | str, rollout_id: RolloutId | str) -> JsonValue:
        path = (
            f"/v1/pools/{_seg(pool_id, field_name='pool_id')}"
            f"/rollouts/{_seg(rollout_id, field_name='rollout_id')}/usage"
        )
        return self._transport.execute(_request("get_container_pool_rollout_usage", path))

    def wait_for_rollout(
        self,
        pool_id: PoolId | str,
        rollout_id: RolloutId | str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> Rollout:
        """Poll until the rollout is terminal, or raise :class:`RolloutTimeoutError`.

        Raising rather than returning a non-terminal rollout keeps a timeout
        from being mistaken for a failed run: the rollout is still executing,
        and its result is still retrievable later.
        """
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            rollout = self.get_rollout(pool_id, rollout_id)
            if rollout.is_terminal:
                return rollout
            if time.monotonic() >= deadline:
                raise RolloutTimeoutError(
                    f"rollout {rollout.rollout_id} still {rollout.status!r} after "
                    f"{timeout_seconds}s; it is still running and can be polled again"
                )
            time.sleep(poll_interval_seconds)


__all__ = ["ContainerPoolsAPI", "DEFAULT_POLL_INTERVAL_SECONDS", "RolloutTimeoutError"]
