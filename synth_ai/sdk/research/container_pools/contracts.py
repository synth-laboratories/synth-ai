"""Typed wire contracts for container pools.

Only the fields the SDK depends on are decoded strictly; the untouched wire
payload is kept on ``raw`` so callers can reach backend fields this package has
not modelled yet without waiting for an SDK release.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import (
    object_value,
    optional_text,
    required_text,
)


class PoolId(str):
    """Backend-owned container pool identifier."""


class PoolTaskId(str):
    """Caller-chosen task identifier, unique within a pool."""


class RuntimeImageReleaseId(str):
    """Backend-owned runtime image release identifier."""


class RolloutId(str):
    """Backend-owned rollout identifier."""


#: Rollout states that will not change again.
TERMINAL_ROLLOUT_STATUSES = frozenset({"completed", "failed", "cancelled"})

HARBOR_CONTAINER_SUBTYPE = "harbor"


def _optional_object(payload: JsonObject, name: str) -> JsonObject:
    value = payload.get(name)
    return value if isinstance(value, dict) else {}


def _optional_float(payload: JsonObject, name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True, slots=True)
class Pool:
    pool_id: PoolId
    backend: str
    status: str
    name: str | None
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_wire(cls, value: JsonValue) -> Pool:
        payload = object_value(value, operation_id="container_pool")
        return cls(
            pool_id=PoolId(required_text(payload, "pool_id")),
            backend=required_text(payload, "backend"),
            status=str(payload.get("status") or ""),
            name=optional_text(payload, "name"),
            raw=payload,
        )


@dataclass(frozen=True, slots=True)
class PoolTask:
    task_id: PoolTaskId
    backend: str | None
    status: str | None
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_wire(cls, value: JsonValue) -> PoolTask:
        payload = object_value(value, operation_id="container_pool_task")
        return cls(
            task_id=PoolTaskId(required_text(payload, "task_id")),
            backend=optional_text(payload, "backend"),
            status=optional_text(payload, "status"),
            raw=payload,
        )


@dataclass(frozen=True, slots=True)
class RuntimeImageRelease:
    """A deployable image plus, for the Harbor subtype, its task bundle.

    This is the object many tasks share: binding the same release to several
    tasks in a pool is what makes a taskset cost one build rather than N.
    """

    release_id: RuntimeImageReleaseId
    source_kind: str
    compute_provider: str | None
    container_subtype: str | None
    resolved_digest: str | None
    raw: JsonObject = field(repr=False)

    @property
    def is_harbor(self) -> bool:
        return self.container_subtype == HARBOR_CONTAINER_SUBTYPE

    @classmethod
    def from_wire(cls, value: JsonValue) -> RuntimeImageRelease:
        payload = object_value(value, operation_id="container_pool_runtime_image_release")
        identifier = payload.get("release_id") or payload.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("runtime image release must carry release_id or id")
        metadata = _optional_object(payload, "release_metadata") or _optional_object(
            payload, "metadata"
        )
        subtype = metadata.get("container_subtype")
        return cls(
            release_id=RuntimeImageReleaseId(identifier.strip()),
            source_kind=str(payload.get("source_kind") or ""),
            compute_provider=optional_text(payload, "compute_provider"),
            container_subtype=subtype.strip().lower()
            if isinstance(subtype, str) and subtype.strip()
            else None,
            resolved_digest=optional_text(payload, "resolved_digest"),
            raw=payload,
        )


@dataclass(frozen=True, slots=True)
class Rollout:
    rollout_id: RolloutId
    status: str
    #: Present only once the rollout reaches a terminal status.
    outcome_reward: float | None
    error: str | None
    metrics: JsonObject = field(repr=False)
    raw: JsonObject = field(repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_ROLLOUT_STATUSES

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and not self.error

    @classmethod
    def from_wire(cls, value: JsonValue) -> Rollout:
        payload = object_value(value, operation_id="container_pool_rollout")
        identifier = payload.get("rollout_id") or payload.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("rollout must carry rollout_id or id")
        metrics = _optional_object(payload, "metrics")
        reward = _optional_float(metrics, "outcome_reward")
        if reward is None:
            reward = _optional_float(payload, "score")
        error = payload.get("error")
        return cls(
            rollout_id=RolloutId(identifier.strip()),
            status=str(payload.get("status") or ""),
            outcome_reward=reward,
            error=error.strip() if isinstance(error, str) and error.strip() else None,
            metrics=metrics,
            raw=payload,
        )


@dataclass(frozen=True, slots=True)
class RolloutArtifact:
    artifact_type: str
    artifact_name: str | None
    size_bytes: int | None
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_wire(cls, value: JsonValue) -> RolloutArtifact:
        payload = object_value(value, operation_id="container_pool_rollout_artifact")
        size = payload.get("size_bytes")
        return cls(
            artifact_type=str(payload.get("artifact_type") or ""),
            artifact_name=optional_text(payload, "artifact_name"),
            size_bytes=int(size) if isinstance(size, int) and not isinstance(size, bool) else None,
            raw=payload,
        )


__all__ = [
    "HARBOR_CONTAINER_SUBTYPE",
    "TERMINAL_ROLLOUT_STATUSES",
    "Pool",
    "PoolId",
    "PoolTask",
    "PoolTaskId",
    "Rollout",
    "RolloutArtifact",
    "RolloutId",
    "RuntimeImageRelease",
    "RuntimeImageReleaseId",
]
