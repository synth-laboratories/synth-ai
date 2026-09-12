"""Container-pool rollouts a swarm launched.

Mirrors backend ``SwarmRolloutPage`` (``app/api/v1/managed_research/schemas/
swarm_rollouts.py``). A rollout appears here only when its verified budget
parent is the swarm; caller-supplied metadata cannot place it in this list.

Each rollout links run → pool/task → logical intent → executor launch (lease or
explicit lease disposition, executed image/source) → rollout artifacts. The
execution record is written by the backend executor from what it launched;
``None`` means unknown (legacy row or not yet started), never "no image".
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"swarm rollout {key} is required")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"swarm rollout {key} must be a string or null")


def _optional_datetime(payload: Mapping[str, object], key: str) -> datetime | None:
    value = _optional_text(payload, key)
    return datetime.fromisoformat(value) if value is not None else None


def _integer(payload: Mapping[str, object], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"swarm rollout {key} must be an integer >= {minimum}")
    return value


def _optional_digest(payload: Mapping[str, object], key: str) -> str | None:
    value = _optional_text(payload, key)
    if value is not None and not _SHA256.fullmatch(value):
        raise ValueError(f"swarm rollout {key} must be a sha256 digest or null")
    return value


def _mapping(payload: object, subject: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"swarm rollout {subject} must be an object")
    return payload


@dataclass(frozen=True)
class SwarmRolloutReleaseCoordinates:
    """Accepted release snapshot; not an attestation of the executed native image.

    See: backend/notes/specifications/tanha/current/systems/platform/intern_resource_inventory.md.
    """

    runtime_image_release_id: str | None = None
    resolved_image_digest: str | None = None
    shared_bundle_release_id: str | None = None
    task_bundle_release_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutReleaseCoordinates:
        if not isinstance(payload, Mapping):
            raise ValueError("swarm rollout release coordinates must be an object")
        return cls(
            runtime_image_release_id=_optional_text(payload, "runtime_image_release_id"),
            resolved_image_digest=_optional_text(payload, "resolved_image_digest"),
            shared_bundle_release_id=_optional_text(payload, "shared_bundle_release_id"),
            task_bundle_release_id=_optional_text(payload, "task_bundle_release_id"),
        )


@dataclass(frozen=True)
class SwarmRolloutLogicalIntent:
    """The one intended evaluation this rollout executes.

    ``accepted_submissions`` counts submissions (request keys) that attached to
    this rollout instead of paying for another. ``intent_id`` is set only for
    an explicit caller intent.
    """

    intent_ref: str
    source: str
    scope_kind: str
    accepted_submissions: int
    intent_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutLogicalIntent:
        payload = _mapping(payload, "logical intent")
        source = _text(payload, "source")
        scope_kind = _text(payload, "scope_kind")
        intent_id = _optional_text(payload, "intent_id")
        if source not in {"explicit", "derived"}:
            raise ValueError("swarm rollout logical intent source is unknown")
        if scope_kind not in {"smr_run", "intern_runtime", "organization", "intern_delegation"}:
            raise ValueError("swarm rollout logical intent scope is unknown")
        if (source == "explicit") != (intent_id is not None):
            raise ValueError("only an explicit logical intent carries an intent_id")
        return cls(
            intent_ref=_text(payload, "intent_ref"),
            source=source,
            scope_kind=scope_kind,
            accepted_submissions=_integer(payload, "accepted_submissions", minimum=1),
            intent_id=intent_id,
        )


@dataclass(frozen=True)
class SwarmRolloutLease:
    lease_id: str

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutLease:
        return cls(lease_id=_text(_mapping(payload, "lease"), "lease_id"))


@dataclass(frozen=True)
class SwarmRolloutLeaseDisposition:
    """Why ``lease`` is or is not set; the backend never infers a nearby lease."""

    status: str
    reason: str

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutLeaseDisposition:
        payload = _mapping(payload, "lease disposition")
        status = _text(payload, "status")
        if status not in {"recorded", "not_applicable"}:
            raise ValueError("swarm rollout lease disposition status is unknown")
        return cls(status=status, reason=_text(payload, "reason"))


@dataclass(frozen=True)
class SwarmRolloutExecutionLaunch:
    """One provider sandbox launch (Harbor: agent, then optional verifier)."""

    sequence: int
    image_attestation: str
    role: str | None = None
    provider: str | None = None
    launch_id: str | None = None
    executed_image_digest: str | None = None
    recorded_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutExecutionLaunch:
        payload = _mapping(payload, "execution launch")
        return cls(
            sequence=_integer(payload, "sequence"),
            image_attestation=_text(payload, "image_attestation"),
            role=_optional_text(payload, "role"),
            provider=_optional_text(payload, "provider"),
            launch_id=_optional_text(payload, "launch_id"),
            executed_image_digest=_optional_digest(payload, "executed_image_digest"),
            recorded_at=_optional_datetime(payload, "recorded_at"),
        )


@dataclass(frozen=True)
class SwarmRolloutExecution:
    """What the backend executor actually launched for the rollout.

    ``executed_image_digest`` is set only when the launch reference was
    digest-pinned or the provider resolved the exact image ID; otherwise
    ``image_attestation`` says why it is unknown. ``lease`` is always stated:
    ``None`` comes with a ``not_applicable`` disposition and its reason.
    """

    lease_disposition: SwarmRolloutLeaseDisposition
    deployment_binding: str
    image_attestation: str
    lease: SwarmRolloutLease | None = None
    recorded_by: str | None = None
    interface: str | None = None
    interface_mode: str | None = None
    provider: str | None = None
    launch_id: str | None = None
    executed_image_digest: str | None = None
    executed_source_digest: str | None = None
    launches: tuple[SwarmRolloutExecutionLaunch, ...] = ()
    recorded_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutExecution:
        payload = _mapping(payload, "execution")
        if "lease" not in payload:
            raise ValueError("swarm rollout execution must state its lease (null when not used)")
        lease = SwarmRolloutLease.from_wire(payload["lease"]) if payload["lease"] is not None else None
        disposition = SwarmRolloutLeaseDisposition.from_wire(payload.get("lease_disposition"))
        if (disposition.status == "recorded") != (lease is not None):
            raise ValueError("swarm rollout lease and lease disposition disagree")
        launches = payload.get("launches", [])
        if not isinstance(launches, list):
            raise ValueError("swarm rollout execution launches must be a list")
        return cls(
            lease_disposition=disposition,
            deployment_binding=_text(payload, "deployment_binding"),
            image_attestation=_text(payload, "image_attestation"),
            lease=lease,
            recorded_by=_optional_text(payload, "recorded_by"),
            interface=_optional_text(payload, "interface"),
            interface_mode=_optional_text(payload, "interface_mode"),
            provider=_optional_text(payload, "provider"),
            launch_id=_optional_text(payload, "launch_id"),
            executed_image_digest=_optional_digest(payload, "executed_image_digest"),
            executed_source_digest=_optional_digest(payload, "executed_source_digest"),
            launches=tuple(SwarmRolloutExecutionLaunch.from_wire(item) for item in launches),
            recorded_at=_optional_datetime(payload, "recorded_at"),
        )


@dataclass(frozen=True)
class SwarmRolloutArtifact:
    """A rollout-owned retained artifact (identity only; storage stays private)."""

    artifact_id: str
    artifact_type: str
    size_bytes: int
    content_type: str | None = None
    created_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutArtifact:
        payload = _mapping(payload, "artifact")
        return cls(
            artifact_id=_text(payload, "artifact_id"),
            artifact_type=_text(payload, "artifact_type"),
            size_bytes=_integer(payload, "size_bytes"),
            content_type=_optional_text(payload, "content_type"),
            created_at=_optional_datetime(payload, "created_at"),
        )


@dataclass(frozen=True)
class SwarmRollout:
    rollout_id: str
    pool_id: str
    adapter: str
    status: str
    seed: int
    trace_correlation_id: str
    budget_parent_run_id: str
    task_id: str | None = None
    success: bool | None = None
    score: float | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    release_coordinates: SwarmRolloutReleaseCoordinates | None = None
    logical_intent: SwarmRolloutLogicalIntent | None = None
    execution: SwarmRolloutExecution | None = None
    artifacts: tuple[SwarmRolloutArtifact, ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRollout:
        if not isinstance(payload, Mapping):
            raise ValueError("swarm rollout must be an object")
        seed = payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("swarm rollout seed must be an integer")
        success = payload.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError("swarm rollout success must be a boolean or null")
        score = payload.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise ValueError("swarm rollout score must be a number or null")
        artifacts = payload.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("swarm rollout artifacts must be a list")
        return cls(
            rollout_id=_text(payload, "rollout_id"),
            pool_id=_text(payload, "pool_id"),
            adapter=_text(payload, "adapter"),
            status=_text(payload, "status"),
            seed=seed,
            trace_correlation_id=_text(payload, "trace_correlation_id"),
            budget_parent_run_id=_text(payload, "budget_parent_run_id"),
            task_id=_optional_text(payload, "task_id"),
            success=success,
            score=float(score) if score is not None else None,
            error=_optional_text(payload, "error"),
            created_at=_optional_datetime(payload, "created_at"),
            started_at=_optional_datetime(payload, "started_at"),
            completed_at=_optional_datetime(payload, "completed_at"),
            cancelled_at=_optional_datetime(payload, "cancelled_at"),
            release_coordinates=(
                SwarmRolloutReleaseCoordinates.from_wire(payload["release_coordinates"])
                if payload.get("release_coordinates") is not None else None
            ),
            logical_intent=(
                SwarmRolloutLogicalIntent.from_wire(payload["logical_intent"])
                if payload.get("logical_intent") is not None else None
            ),
            execution=(
                SwarmRolloutExecution.from_wire(payload["execution"])
                if payload.get("execution") is not None else None
            ),
            artifacts=tuple(SwarmRolloutArtifact.from_wire(item) for item in artifacts),
        )


def swarm_rollouts_from_wire(payload: object, *, swarm_id: str) -> tuple[SwarmRollout, ...]:
    if not isinstance(payload, Mapping) or payload.get("run_id") != swarm_id:
        raise ValueError("swarm rollout page identity drifted")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("swarm rollout page items must be a list")
    rollouts = tuple(SwarmRollout.from_wire(item) for item in items)
    if any(rollout.budget_parent_run_id != swarm_id for rollout in rollouts):
        raise ValueError("swarm rollout budget parent drifted")
    return rollouts


__all__ = [
    "SwarmRollout",
    "SwarmRolloutArtifact",
    "SwarmRolloutExecution",
    "SwarmRolloutExecutionLaunch",
    "SwarmRolloutLease",
    "SwarmRolloutLeaseDisposition",
    "SwarmRolloutLogicalIntent",
    "SwarmRolloutReleaseCoordinates",
    "swarm_rollouts_from_wire",
]
