"""Typed bounded execution activity for Research swarms.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._activity_wire import (
    exact_object,
    non_negative_int,
    optional_datetime,
    optional_non_negative_int,
)
from synth_ai.core.research.contracts._wire import (
    array_value,
    optional_text,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.activity_items import (
    ActivityActor,
    ActivityActorState,
    ActivityEvent,
    ActivityMessage,
    ActivityTask,
    ActivityTaskState,
    ActivityVisibility,
    ActivityWorkProduct,
)
from synth_ai.core.research.contracts.common import ActivityEventId, ProjectId, SwarmId
from synth_ai.core.research.contracts.swarms import SwarmState


class SwarmLivenessPhase(StrEnum):
    ADMITTED = "admitted"
    PLANNING = "planning"
    TASK_RUNNING = "task_running"
    REVIEWER_REQUIRED = "reviewer_required"
    BLOCKED = "blocked"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    TERMINAL = "terminal"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class ActivityWindow:
    """Requested upper bounds for one consistent activity snapshot."""

    event_limit: int = 100
    actor_limit: int = 50
    task_limit: int = 100
    message_limit: int = 50
    work_product_limit: int = 50

    def __post_init__(self) -> None:
        limits = (
            ("event_limit", self.event_limit, 500),
            ("actor_limit", self.actor_limit, 200),
            ("task_limit", self.task_limit, 250),
            ("message_limit", self.message_limit, 200),
            ("work_product_limit", self.work_product_limit, 200),
        )
        for name, value, maximum in limits:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 through {maximum}")

    def to_query(self) -> JsonObject:
        return {
            "event_limit": self.event_limit,
            "actor_limit": self.actor_limit,
            "task_limit": self.task_limit,
            "message_limit": self.message_limit,
            "work_product_limit": self.work_product_limit,
        }


@dataclass(frozen=True, slots=True)
class ActivityWindowReceipt:
    requested: ActivityWindow
    event_count: int
    actor_count: int
    task_count: int
    message_count: int
    work_product_count: int

    def __post_init__(self) -> None:
        counts = (
            ("event_count", self.event_count, self.requested.event_limit),
            ("actor_count", self.actor_count, self.requested.actor_limit),
            ("task_count", self.task_count, self.requested.task_limit),
            ("message_count", self.message_count, self.requested.message_limit),
            (
                "work_product_count",
                self.work_product_count,
                self.requested.work_product_limit,
            ),
        )
        for name, value, limit in counts:
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError(f"{name} must be an integer from 0 through {limit}")

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityWindowReceipt:
        fields = frozenset(
            {
                "event_limit",
                "actor_limit",
                "task_limit",
                "message_limit",
                "work_product_limit",
                "event_count",
                "actor_count",
                "task_count",
                "message_count",
                "work_product_count",
            }
        )
        payload = exact_object(value, label="swarm activity window", fields=fields)
        requested = ActivityWindow(
            event_limit=non_negative_int(payload, "event_limit"),
            actor_limit=non_negative_int(payload, "actor_limit"),
            task_limit=non_negative_int(payload, "task_limit"),
            message_limit=non_negative_int(payload, "message_limit"),
            work_product_limit=non_negative_int(payload, "work_product_limit"),
        )
        return cls(
            requested=requested,
            event_count=non_negative_int(payload, "event_count"),
            actor_count=non_negative_int(payload, "actor_count"),
            task_count=non_negative_int(payload, "task_count"),
            message_count=non_negative_int(payload, "message_count"),
            work_product_count=non_negative_int(payload, "work_product_count"),
        )

    def to_wire(self) -> JsonObject:
        return {
            **self.requested.to_query(),
            "event_count": self.event_count,
            "actor_count": self.actor_count,
            "task_count": self.task_count,
            "message_count": self.message_count,
            "work_product_count": self.work_product_count,
        }


@dataclass(frozen=True, slots=True)
class ActivityCursor:
    latest_node_id: ActivityEventId | None
    latest_event_seq: int | None
    transcript_cursor: str | None
    generated_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityCursor:
        payload = exact_object(
            value,
            label="swarm activity cursor",
            fields=frozenset(
                {
                    "latest_node_id",
                    "latest_event_seq",
                    "transcript_cursor",
                    "generated_at",
                }
            ),
        )
        latest_node_id = optional_text(payload, "latest_node_id")
        return cls(
            latest_node_id=(
                ActivityEventId(latest_node_id)
                if latest_node_id is not None
                else None
            ),
            latest_event_seq=optional_non_negative_int(payload, "latest_event_seq"),
            transcript_cursor=optional_text(payload, "transcript_cursor"),
            generated_at=required_datetime(payload, "generated_at"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "latest_node_id": self.latest_node_id,
            "latest_event_seq": self.latest_event_seq,
            "transcript_cursor": self.transcript_cursor,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ActivityRun:
    state: SwarmState
    liveness_phase: SwarmLivenessPhase | None
    terminal_outcome: str | None
    started_at: datetime | None
    completed_at: datetime | None
    latest_summary: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityRun:
        payload = exact_object(
            value,
            label="swarm activity run",
            fields=frozenset(
                {
                    "public_state",
                    "liveness_phase",
                    "terminal_outcome",
                    "started_at",
                    "completed_at",
                    "latest_summary",
                }
            ),
        )
        liveness_phase = optional_text(payload, "liveness_phase")
        return cls(
            state=SwarmState(required_text(payload, "public_state")),
            liveness_phase=(
                SwarmLivenessPhase(liveness_phase)
                if liveness_phase is not None
                else None
            ),
            terminal_outcome=optional_text(payload, "terminal_outcome"),
            started_at=optional_datetime(payload, "started_at"),
            completed_at=optional_datetime(payload, "completed_at"),
            latest_summary=optional_text(payload, "latest_summary"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "public_state": self.state.value,
            "liveness_phase": (
                self.liveness_phase.value if self.liveness_phase is not None else None
            ),
            "terminal_outcome": self.terminal_outcome,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "latest_summary": self.latest_summary,
        }


@dataclass(frozen=True, slots=True)
class SwarmActivity:
    swarm_id: SwarmId
    project_id: ProjectId
    generated_at: datetime
    run: ActivityRun
    cursor: ActivityCursor
    actors: tuple[ActivityActor, ...]
    tasks: tuple[ActivityTask, ...]
    messages: tuple[ActivityMessage, ...]
    events: tuple[ActivityEvent, ...]
    work_products: tuple[ActivityWorkProduct, ...]
    window: ActivityWindowReceipt

    def __post_init__(self) -> None:
        expected = (
            ("actor_count", self.window.actor_count, len(self.actors)),
            ("task_count", self.window.task_count, len(self.tasks)),
            ("message_count", self.window.message_count, len(self.messages)),
            ("event_count", self.window.event_count, len(self.events)),
            (
                "work_product_count",
                self.window.work_product_count,
                len(self.work_products),
            ),
        )
        for name, received, actual in expected:
            if received != actual:
                raise ValueError(f"{name} must equal the returned collection length")

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmActivity:
        payload = exact_object(
            value,
            label="retrieve_swarm_activity",
            fields=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "project_id",
                    "generated_at",
                    "run",
                    "cursor",
                    "actors",
                    "tasks",
                    "messages",
                    "events",
                    "work_products",
                    "window",
                }
            ),
        )
        if payload["schema_version"] != 1:
            raise ValueError("retrieve_swarm_activity.schema_version must be 1")
        return cls(
            swarm_id=SwarmId(required_text(payload, "run_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            generated_at=required_datetime(payload, "generated_at"),
            run=ActivityRun.from_wire(payload["run"]),
            cursor=ActivityCursor.from_wire(payload["cursor"]),
            actors=tuple(
                ActivityActor.from_wire(item)
                for item in array_value(
                    payload["actors"], operation_id="retrieve_swarm_activity.actors"
                )
            ),
            tasks=tuple(
                ActivityTask.from_wire(item)
                for item in array_value(
                    payload["tasks"], operation_id="retrieve_swarm_activity.tasks"
                )
            ),
            messages=tuple(
                ActivityMessage.from_wire(item)
                for item in array_value(
                    payload["messages"],
                    operation_id="retrieve_swarm_activity.messages",
                )
            ),
            events=tuple(
                ActivityEvent.from_wire(item)
                for item in array_value(
                    payload["events"], operation_id="retrieve_swarm_activity.events"
                )
            ),
            work_products=tuple(
                ActivityWorkProduct.from_wire(item)
                for item in array_value(
                    payload["work_products"],
                    operation_id="retrieve_swarm_activity.work_products",
                )
            ),
            window=ActivityWindowReceipt.from_wire(payload["window"]),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": 1,
            "run_id": self.swarm_id,
            "project_id": self.project_id,
            "generated_at": self.generated_at.isoformat(),
            "run": self.run.to_wire(),
            "cursor": self.cursor.to_wire(),
            "actors": [actor.to_wire() for actor in self.actors],
            "tasks": [task.to_wire() for task in self.tasks],
            "messages": [message.to_wire() for message in self.messages],
            "events": [event.to_wire() for event in self.events],
            "work_products": [item.to_wire() for item in self.work_products],
            "window": self.window.to_wire(),
        }


__all__ = [
    "ActivityActor",
    "ActivityActorState",
    "ActivityCursor",
    "ActivityEvent",
    "ActivityMessage",
    "ActivityRun",
    "ActivityTask",
    "ActivityTaskState",
    "ActivityVisibility",
    "ActivityWindow",
    "ActivityWindowReceipt",
    "ActivityWorkProduct",
    "SwarmActivity",
    "SwarmLivenessPhase",
]
