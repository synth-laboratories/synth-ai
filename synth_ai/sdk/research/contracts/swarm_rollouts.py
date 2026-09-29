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

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRolloutReleaseCoordinates

    coordinates = SwarmRolloutReleaseCoordinates.from_wire({})
    assert coordinates.resolved_image_digest is None
    ```
    """

    #: Accepted runtime-image release identifier; not an execution attestation.
    runtime_image_release_id: str | None = None
    #: Accepted resolved-image digest, when recorded; actual execution is separate.
    resolved_image_digest: str | None = None
    #: Accepted shared-bundle release identifier, when recorded.
    shared_bundle_release_id: str | None = None
    #: Accepted task-bundle release identifier, when recorded.
    task_bundle_release_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutReleaseCoordinates:
        """Parse the backend wire representation of SwarmRolloutReleaseCoordinates.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRolloutReleaseCoordinates with backend values preserved.

        Raises:
            ValueError: The payload is not an object or a release coordinate is not a string or null.
        """
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

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRolloutLogicalIntent

    intent = SwarmRolloutLogicalIntent.from_wire({
        "intent_ref": "evaluation-1", "source": "derived",
        "scope_kind": "smr_run", "accepted_submissions": 1,
    })
    assert intent.intent_id is None
    ```
    """

    #: Reference grouping accepted submissions for one logical evaluation.
    intent_ref: str
    #: Whether intent identity is explicit or derived.
    source: str
    #: Intent scope: smr_run, intern_runtime, organization, or intern_delegation.
    scope_kind: str
    #: Positive count of submissions attached to this logical intent.
    accepted_submissions: int
    #: Explicit intent ID; present exactly when source is explicit.
    intent_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutLogicalIntent:
        """Parse the backend wire representation of SwarmRolloutLogicalIntent.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRolloutLogicalIntent with backend values preserved.

        Raises:
            ValueError: The source, scope, explicit intent ID, or positive submission count is invalid.
        """
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
    """Recorded executor lease identity, separate from accepted release coordinates.

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRolloutLease

    lease = SwarmRolloutLease.from_wire({"lease_id": "lease-1"})
    assert lease.lease_id == "lease-1"
    ```
    """
    #: Recorded executor lease identifier.
    lease_id: str

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutLease:
        """Parse the backend wire representation of SwarmRolloutLease.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRolloutLease with backend values preserved.
        """
        return cls(lease_id=_text(_mapping(payload, "lease"), "lease_id"))


@dataclass(frozen=True)
class SwarmRolloutLeaseDisposition:
    """Why ``lease`` is or is not set; the backend never infers a nearby lease.

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRolloutLeaseDisposition

    disposition = SwarmRolloutLeaseDisposition.from_wire({
        "status": "not_applicable", "reason": "Executor does not use leases",
    })
    assert disposition.status == "not_applicable"
    ```
    """

    #: Whether a lease is recorded or explicitly not applicable.
    status: str
    #: Required backend explanation of the lease disposition.
    reason: str

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutLeaseDisposition:
        """Parse the backend wire representation of SwarmRolloutLeaseDisposition.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRolloutLeaseDisposition with backend values preserved.

        Raises:
            ValueError: The status is neither recorded nor not_applicable, or the reason is missing.
        """
        payload = _mapping(payload, "lease disposition")
        status = _text(payload, "status")
        if status not in {"recorded", "not_applicable"}:
            raise ValueError("swarm rollout lease disposition status is unknown")
        return cls(status=status, reason=_text(payload, "reason"))


@dataclass(frozen=True)
class SwarmRolloutExecutionLaunch:
    """One provider sandbox launch (Harbor: agent, then optional verifier).

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRolloutExecutionLaunch

    launch = SwarmRolloutExecutionLaunch.from_wire({
        "sequence": 0, "image_attestation": "unknown",
    })
    assert launch.executed_image_digest is None
    ```
    """

    #: Non-negative execution-launch sequence number.
    sequence: int
    #: Explanation of the launch image identity or why it is unknown.
    image_attestation: str
    #: Launch role, such as agent or verifier, when recorded.
    role: str | None = None
    #: Execution provider recorded for this launch.
    provider: str | None = None
    #: Provider launch identifier, when recorded.
    launch_id: str | None = None
    #: Actual executed image SHA-256 digest, or null when unknown.
    executed_image_digest: str | None = None
    #: Timestamp when the launch evidence was recorded, when supplied.
    recorded_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutExecutionLaunch:
        """Parse the backend wire representation of SwarmRolloutExecutionLaunch.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRolloutExecutionLaunch with backend values preserved.
        """
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

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRolloutExecution

    execution = SwarmRolloutExecution.from_wire({
        "lease": None, "lease_disposition": {"status": "not_applicable",
            "reason": "Executor does not use leases"},
        "deployment_binding": "binding-1", "image_attestation": "unknown",
    })
    assert execution.executed_image_digest is None
    ```
    """

    #: Explicit disposition explaining whether an executor lease is present.
    lease_disposition: SwarmRolloutLeaseDisposition
    #: Recorded deployment binding for the executor.
    deployment_binding: str
    #: Explanation of executed image identity or why it is unknown.
    image_attestation: str
    #: Executor lease; null only with a not_applicable disposition.
    lease: SwarmRolloutLease | None = None
    #: Source that recorded execution evidence, when supplied.
    recorded_by: str | None = None
    #: Executor interface name, when supplied.
    interface: str | None = None
    #: Recorded executor interface mode, when supplied.
    interface_mode: str | None = None
    #: Execution provider, when supplied.
    provider: str | None = None
    #: Provider launch identifier, when supplied.
    launch_id: str | None = None
    #: Actual executed image SHA-256 digest; null means unknown.
    executed_image_digest: str | None = None
    #: Actual executed source SHA-256 digest; null means unknown.
    executed_source_digest: str | None = None
    #: Ordered launch evidence, including separate agent/verifier launches when recorded.
    launches: tuple[SwarmRolloutExecutionLaunch, ...] = ()
    #: Timestamp when execution evidence was recorded.
    recorded_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutExecution:
        """Parse the backend wire representation of SwarmRolloutExecution.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRolloutExecution with backend values preserved.

        Raises:
            ValueError: Lease presence disagrees with its disposition, launches is not a list, or execution fields are malformed.
        """
        payload = _mapping(payload, "execution")
        if "lease" not in payload:
            raise ValueError("swarm rollout execution must state its lease (null when not used)")
        lease = (
            SwarmRolloutLease.from_wire(payload["lease"]) if payload["lease"] is not None else None
        )
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
    """A rollout-owned retained artifact (identity only; storage stays private).

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRolloutArtifact

    artifact = SwarmRolloutArtifact.from_wire({
        "artifact_id": "artifact-1", "artifact_type": "trace", "size_bytes": 12,
    })
    assert artifact.size_bytes == 12
    ```
    """

    #: Identity of a retained rollout artifact.
    artifact_id: str
    #: Backend artifact type.
    artifact_type: str
    #: Non-negative artifact size in bytes.
    size_bytes: int
    #: Artifact MIME type, when recorded.
    content_type: str | None = None
    #: Artifact creation timestamp, when recorded.
    created_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRolloutArtifact:
        """Parse the backend wire representation of SwarmRolloutArtifact.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRolloutArtifact with backend values preserved.
        """
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
    """Container-pool rollout attributed to a verified Swarm budget parent.

    ```python
    from synth_ai.sdk.research.contracts.swarm_rollouts import SwarmRollout

    rollout = SwarmRollout.from_wire({
        "rollout_id": "rollout-1", "pool_id": "pool-1", "adapter": "example",
        "status": "queued", "seed": 0, "trace_correlation_id": "trace-1",
        "budget_parent_run_id": "run-1",
    })
    assert rollout.execution is None
    ```
    """
    #: Identity of this container-pool rollout.
    rollout_id: str
    #: Container pool used for this rollout.
    pool_id: str
    #: Rollout adapter selected by the backend.
    adapter: str
    #: Backend rollout status.
    status: str
    #: Integer evaluation seed; booleans are rejected.
    seed: int
    #: Correlation key linking rollout and trace evidence.
    trace_correlation_id: str
    #: Verified parent run charged for this rollout; checked against the requested Swarm.
    budget_parent_run_id: str
    #: Task identifier within the pool, when recorded.
    task_id: str | None = None
    #: Evaluation success flag, or null when no result is available.
    success: bool | None = None
    #: Numeric evaluation score, or null when unavailable.
    score: float | None = None
    #: Recorded rollout error, when present.
    error: str | None = None
    #: Rollout creation timestamp, when recorded.
    created_at: datetime | None = None
    #: Rollout start timestamp, when recorded.
    started_at: datetime | None = None
    #: Rollout completion timestamp, when recorded.
    completed_at: datetime | None = None
    #: Rollout cancellation timestamp, when recorded.
    cancelled_at: datetime | None = None
    #: Accepted release snapshot; separate from actual execution evidence.
    release_coordinates: SwarmRolloutReleaseCoordinates | None = None
    #: Logical evaluation identity and accepted-submission count, when recorded.
    logical_intent: SwarmRolloutLogicalIntent | None = None
    #: Actual executor evidence; null means unavailable, not absence of execution.
    execution: SwarmRolloutExecution | None = None
    #: Retained artifact identities; private storage coordinates are not exposed.
    artifacts: tuple[SwarmRolloutArtifact, ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRollout:
        """Parse the backend wire representation of SwarmRollout.

        Args:
            payload: Backend response object to validate and convert.

        Returns:
            The parsed SwarmRollout with backend values preserved.

        Raises:
            ValueError: The seed, success flag, score, artifact list, or nested rollout fields are malformed.
        """
        payload = _mapping(payload, "swarm rollout")
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
                if payload.get("release_coordinates") is not None
                else None
            ),
            logical_intent=(
                SwarmRolloutLogicalIntent.from_wire(payload["logical_intent"])
                if payload.get("logical_intent") is not None
                else None
            ),
            execution=(
                SwarmRolloutExecution.from_wire(payload["execution"])
                if payload.get("execution") is not None
                else None
            ),
            artifacts=tuple(SwarmRolloutArtifact.from_wire(item) for item in artifacts),
        )


def swarm_rollouts_from_wire(payload: object, *, swarm_id: str) -> tuple[SwarmRollout, ...]:
    """Parse a rollout page and verify its Swarm budget-parent boundary.

    Args:
        payload: Backend rollout-page object with run_id and items.
        swarm_id: Requested Swarm ID; page and rollout parent IDs must match it.

    Returns:
        Parsed rollouts attributed to the requested Swarm.

    Raises:
        ValueError: The page or a rollout crosses the requested run boundary, or items is not a list.
    """
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
