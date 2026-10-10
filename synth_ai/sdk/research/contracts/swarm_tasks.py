"""Typed task, dependency, review-verdict and retry reads for one Swarm.

# See: openapi/research-v1.json SmrRunTaskResponse, SmrRunTaskEventsResponse

Tasks come from ``list_run_tasks`` (dependency edges and retry lineage are
authoritative task fields). Review verdicts and same-task repair retries are
derived only from the backend's typed task-state transitions in
``list_project_run_task_events``; no summary text is interpreted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import (
    array_value,
    object_value,
    optional_datetime,
    optional_text,
    required_datetime,
    required_text,
)
from synth_ai.sdk.research.contracts.common import ProjectId, SwarmId
from synth_ai.sdk.research.task_execution_reads import public_task_execution


def _bounded(
    payload: JsonObject, *, label: str, required: frozenset[str], allowed: frozenset[str]
) -> None:
    missing, extra = required - payload.keys(), payload.keys() - allowed
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )


def _state(payload: JsonObject, name: str) -> str | None:
    value = optional_text(payload, name)
    return None if value is None else value.lower()


class TaskReviewState(StrEnum):
    """Task states that bracket a reviewer decision."""

    REVIEW_REQUIRED = "review_required"
    REPAIR_REQUIRED = "repair_required"


ACCEPTED_TASK_STATES = frozenset({"done", "completed", "succeeded"})
FAILED_TASK_STATES = frozenset({"failed"})
ACTIVE_TASK_STATES = frozenset({"ready", "assigned", "queued", "running", "executing"})


class ReviewVerdict(StrEnum):
    ACCEPTED = "accepted"
    REPAIR_REQUESTED = "repair_requested"
    REJECTED_TERMINAL = "rejected_terminal"


class RetryKind(StrEnum):
    NEW_ATTEMPT = "new_attempt"
    """A new task superseding a prior one (``retry_of`` lineage)."""
    REPAIR = "repair"
    """The same task resumed after a reviewer requested repair."""


_TASK_REQUIRED = frozenset(
    {
        "task_id",
        "run_id",
        "org_id",
        "project_id",
        "task_key",
        "kind",
        "public_task_state",
        "execution_owner",
        "created_at",
        "updated_at",
    }
)
_TASK_ALLOWED = _TASK_REQUIRED | {
    "agent_goal_assignment",
    "claimed_by",
    "depends_on_task_keys",
    "diagnostics",
    "finished_at",
    "input",
    "last_heartbeat_at",
    "lease_expires_at",
    "output",
    "retry_of",
    "started_at",
    "task_dispatch",
    "task_state",
    "worker_pool",
    "execution",
}


@dataclass(frozen=True, slots=True)
class SwarmTask:
    task_id: str
    task_key: str
    swarm_id: SwarmId
    project_id: ProjectId
    kind: str
    state: str
    depends_on_task_keys: tuple[str, ...]
    retry_of: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    execution: JsonObject | None = None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmTask:
        payload = object_value(value, operation_id="list_run_tasks item")
        _bounded(payload, label="swarm task", required=_TASK_REQUIRED, allowed=_TASK_ALLOWED)
        depends = payload.get("depends_on_task_keys")
        keys = [] if depends is None else array_value(depends, operation_id="depends_on_task_keys")
        if not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError("depends_on_task_keys must contain non-empty strings")
        return cls(
            required_text(payload, "task_id"),
            required_text(payload, "task_key"),
            SwarmId(required_text(payload, "run_id")),
            ProjectId(required_text(payload, "project_id")),
            required_text(payload, "kind"),
            required_text(payload, "public_task_state").lower(),
            tuple(str(key).strip() for key in keys),
            optional_text(payload, "retry_of"),
            required_datetime(payload, "created_at"),
            required_datetime(payload, "updated_at"),
            optional_datetime(payload, "started_at"),
            optional_datetime(payload, "finished_at"),
            public_task_execution(
                payload.get("execution"),
                organization_id=required_text(payload, "org_id"),
                project_id=required_text(payload, "project_id"),
                run_id=required_text(payload, "run_id"),
                task_id=required_text(payload, "task_id"),
            ),
        )


_EVENT_REQUIRED = frozenset(
    {"event_id", "event_kind", "display_kind", "summary", "source", "occurred_at"}
)
_EVENT_ALLOWED = _EVENT_REQUIRED | {
    "actor_id",
    "actor_label",
    "event_state",
    "participant_role",
    "payload",
    "previous_event_state",
    "source_row_id",
    "source_row_kind",
    "task_id",
    "task_key",
}


@dataclass(frozen=True, slots=True)
class SwarmTaskEvent:
    event_id: str
    event_kind: str
    occurred_at: datetime
    summary: str
    source: str
    task_id: str | None = None
    task_key: str | None = None
    previous_state: str | None = None
    state: str | None = None
    actor_id: str | None = None
    participant_role: str | None = None

    @property
    def task_ref(self) -> str | None:
        return self.task_key or self.task_id

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmTaskEvent:
        payload = object_value(value, operation_id="task event")
        _bounded(
            payload, label="swarm task event", required=_EVENT_REQUIRED, allowed=_EVENT_ALLOWED
        )
        summary = payload.get("summary")
        if not isinstance(summary, str):
            raise ValueError("summary must be a string")
        return cls(
            required_text(payload, "event_id"),
            required_text(payload, "event_kind"),
            required_datetime(payload, "occurred_at"),
            summary,
            required_text(payload, "source"),
            optional_text(payload, "task_id"),
            optional_text(payload, "task_key"),
            _state(payload, "previous_event_state"),
            _state(payload, "event_state"),
            optional_text(payload, "actor_id"),
            optional_text(payload, "participant_role"),
        )


@dataclass(frozen=True, slots=True)
class SwarmTaskEvents:
    swarm_id: SwarmId
    project_id: ProjectId
    generated_at: datetime
    events: tuple[SwarmTaskEvent, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmTaskEvents:
        payload = object_value(value, operation_id="list_project_run_task_events")
        _bounded(
            payload,
            label="swarm task events",
            required=frozenset({"project_id", "run_id", "generated_at"}),
            allowed=frozenset(
                {"schema_version", "project_id", "run_id", "generated_at", "events", "cursor"}
            ),
        )
        version = payload.get("schema_version", "smr_run_task_events.v1")
        if version != "smr_run_task_events.v1":
            raise ValueError(f"unsupported task events schema_version {version!r}")
        events = payload.get("events")
        items = [] if events is None else array_value(events, operation_id="task events")
        return cls(
            SwarmId(required_text(payload, "run_id")),
            ProjectId(required_text(payload, "project_id")),
            required_datetime(payload, "generated_at"),
            tuple(SwarmTaskEvent.from_wire(item) for item in items),
        )


@dataclass(frozen=True, slots=True)
class TaskDependency:
    upstream_task_key: str
    downstream_task_key: str


@dataclass(frozen=True, slots=True)
class TaskReviewDecision:
    task_ref: str
    verdict: ReviewVerdict
    from_state: str
    to_state: str
    event_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class TaskRetry:
    task_ref: str
    kind: RetryKind
    retry_of: str | None = None
    event_id: str | None = None
    occurred_at: datetime | None = None


def _verdict(from_state: str, to_state: str) -> ReviewVerdict | None:
    if from_state != TaskReviewState.REVIEW_REQUIRED:
        return None
    if to_state == TaskReviewState.REPAIR_REQUIRED:
        return ReviewVerdict.REPAIR_REQUESTED
    if to_state in ACCEPTED_TASK_STATES:
        return ReviewVerdict.ACCEPTED
    if to_state in FAILED_TASK_STATES:
        return ReviewVerdict.REJECTED_TERMINAL
    return None


def _transitions(
    events: tuple[SwarmTaskEvent, ...],
) -> Iterator[tuple[str, SwarmTaskEvent, str, str]]:
    """Yield (task_ref, event, from_state, to_state) in occurrence order."""
    last: dict[str, str] = {}
    for event in sorted(events, key=lambda item: (item.occurred_at, item.event_id)):
        ref, state = event.task_ref, event.state
        if ref is None or state is None:
            continue
        previous = event.previous_state or last.get(ref)
        last[ref] = state
        if previous is not None and previous != state:
            yield ref, event, previous, state


@dataclass(frozen=True, slots=True)
class SwarmWorkGraph:
    """Tasks, dependency edges, reviewer verdicts and retries for one Swarm."""

    swarm_id: SwarmId
    tasks: tuple[SwarmTask, ...]
    events: tuple[SwarmTaskEvent, ...]
    dependencies: tuple[TaskDependency, ...]
    review_decisions: tuple[TaskReviewDecision, ...]
    retries: tuple[TaskRetry, ...]

    @classmethod
    def build(
        cls, swarm_id: SwarmId, tasks: tuple[SwarmTask, ...], events: tuple[SwarmTaskEvent, ...]
    ) -> SwarmWorkGraph:
        if any(task.swarm_id != swarm_id for task in tasks):
            raise ValueError("task belongs to a different swarm")
        dependencies = tuple(
            TaskDependency(upstream, task.task_key)
            for task in tasks
            for upstream in task.depends_on_task_keys
        )
        decisions: list[TaskReviewDecision] = []
        retries = [
            TaskRetry(task.task_key, RetryKind.NEW_ATTEMPT, retry_of=task.retry_of)
            for task in tasks
            if task.retry_of is not None
        ]
        for ref, event, before, after in _transitions(events):
            verdict = _verdict(before, after)
            if verdict is not None:
                decisions.append(
                    TaskReviewDecision(
                        ref, verdict, before, after, event.event_id, event.occurred_at
                    )
                )
            if before == TaskReviewState.REPAIR_REQUIRED and (
                after in ACTIVE_TASK_STATES or after == TaskReviewState.REVIEW_REQUIRED
            ):
                retries.append(
                    TaskRetry(ref, RetryKind.REPAIR, None, event.event_id, event.occurred_at)
                )
        return cls(swarm_id, tasks, events, dependencies, tuple(decisions), tuple(retries))

    def task(self, ref: str) -> SwarmTask | None:
        return next((t for t in self.tasks if ref in (t.task_key, t.task_id)), None)

    def _keys(self, ref: str) -> set[str]:
        task = self.task(ref)
        return {ref} if task is None else {task.task_key, task.task_id}

    def dependents_of(self, ref: str) -> tuple[SwarmTask, ...]:
        keys = self._keys(ref)
        downstream = {
            d.downstream_task_key for d in self.dependencies if d.upstream_task_key in keys
        }
        return tuple(t for t in self.tasks if t.task_key in downstream)

    def decisions_for(self, ref: str) -> tuple[TaskReviewDecision, ...]:
        keys = self._keys(ref)
        return tuple(d for d in self.review_decisions if d.task_ref in keys)

    def retries_of(self, ref: str) -> tuple[TaskRetry, ...]:
        keys = self._keys(ref)
        return tuple(r for r in self.retries if r.task_ref in keys or r.retry_of in keys)


def swarm_tasks_from_wire(value: JsonValue, swarm_id: SwarmId) -> tuple[SwarmTask, ...]:
    tasks = tuple(
        SwarmTask.from_wire(item) for item in array_value(value, operation_id="list_run_tasks")
    )
    if any(task.swarm_id != swarm_id for task in tasks):
        raise ValueError("list_run_tasks returned a task for a different swarm")
    return tasks


__all__ = [
    "ReviewVerdict",
    "RetryKind",
    "SwarmTask",
    "SwarmTaskEvent",
    "SwarmTaskEvents",
    "SwarmWorkGraph",
    "TaskDependency",
    "TaskReviewDecision",
    "TaskReviewState",
    "TaskRetry",
    "swarm_tasks_from_wire",
]
