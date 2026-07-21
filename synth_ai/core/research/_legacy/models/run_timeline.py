"""Typed Managed Research logical timeline and checkpoint-branching models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synth_ai.core.research._legacy.models.run_authority import (
        ManagedResearchAuthorityTask,
        ManagedResearchRuntimeAuthority,
    )


def _require_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    text = _require_text(str(value) if value is not None else None, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime") from exc


def _coerce_string_dict(value: Any, *, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        normalized[str(key)] = str(item)
    return normalized


def _coerce_any_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _coerce_optional_any_dict(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _coerce_any_dict(value, field_name=field_name)


def _coerce_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [str(item) for item in value]


class SmrBranchMode(StrEnum):
    EXACT = "exact"
    WITH_MESSAGE = "with_message"


@dataclass(frozen=True, slots=True)
class SmrRunBranchRequest:
    checkpoint_id: str | None = None
    checkpoint_record_id: str | None = None
    checkpoint_uri: str | None = None
    mode: SmrBranchMode = SmrBranchMode.EXACT
    message: str | None = None
    reason: str | None = None
    title: str | None = None
    source_node_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", _optional_text(self.checkpoint_id))
        object.__setattr__(self, "checkpoint_record_id", _optional_text(self.checkpoint_record_id))
        object.__setattr__(self, "checkpoint_uri", _optional_text(self.checkpoint_uri))
        object.__setattr__(self, "message", _optional_text(self.message))
        object.__setattr__(self, "reason", _optional_text(self.reason))
        object.__setattr__(self, "title", _optional_text(self.title))
        object.__setattr__(self, "source_node_id", _optional_text(self.source_node_id))
        reference_count = sum(
            1
            for value in (
                self.checkpoint_id,
                self.checkpoint_record_id,
                self.checkpoint_uri,
            )
            if value is not None
        )
        if reference_count != 1:
            raise ValueError(
                "exactly one of checkpoint_id, checkpoint_record_id, or checkpoint_uri is required"
            )
        if self.mode == SmrBranchMode.WITH_MESSAGE and self.message is None:
            raise ValueError("message is required when mode is with_message")
        if self.mode == SmrBranchMode.EXACT and self.message is not None:
            raise ValueError("message must be omitted when mode is exact")

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode.value}
        for key in (
            "checkpoint_id",
            "checkpoint_record_id",
            "checkpoint_uri",
            "message",
            "reason",
            "title",
            "source_node_id",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True, slots=True)
class SmrRunBranchResponse:
    accepted: bool
    parent_run_id: str
    child_run_id: str
    source_checkpoint_id: str
    source_checkpoint_record_id: str | None = None
    source_node_id: str | None = None
    branch_message_id: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parent_run_id", _require_text(self.parent_run_id, field_name="parent_run_id")
        )
        object.__setattr__(
            self, "child_run_id", _require_text(self.child_run_id, field_name="child_run_id")
        )
        object.__setattr__(
            self,
            "source_checkpoint_id",
            _require_text(self.source_checkpoint_id, field_name="source_checkpoint_id"),
        )
        object.__setattr__(
            self,
            "source_checkpoint_record_id",
            _optional_text(self.source_checkpoint_record_id),
        )
        object.__setattr__(self, "source_node_id", _optional_text(self.source_node_id))
        object.__setattr__(self, "branch_message_id", _optional_text(self.branch_message_id))

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunBranchResponse:
        return cls(
            accepted=bool(payload.get("accepted")),
            parent_run_id=_require_text(payload.get("parent_run_id"), field_name="parent_run_id"),
            child_run_id=_require_text(payload.get("child_run_id"), field_name="child_run_id"),
            source_checkpoint_id=_require_text(
                payload.get("source_checkpoint_id"), field_name="source_checkpoint_id"
            ),
            source_checkpoint_record_id=_optional_text(payload.get("source_checkpoint_record_id")),
            source_node_id=_optional_text(payload.get("source_node_id")),
            branch_message_id=_optional_text(payload.get("branch_message_id")),
            created_at=_parse_datetime(payload.get("created_at"), field_name="created_at")
            if payload.get("created_at") is not None
            else None,
        )


@dataclass(frozen=True, slots=True)
class SmrLogicalTimelineNode:
    node_id: str
    run_id: str
    created_at: datetime
    logical_index: int
    kind: str
    source: str
    title: str
    summary: str
    state: str | None = None
    actor_id: str | None = None
    actor_type: str | None = None
    participant_role: str | None = None
    task_id: str | None = None
    task_key: str | None = None
    worker_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint_record_id: str | None = None
    checkpoint_uri: str | None = None
    checkpoint_boundary_kind: str | None = None
    message_id: str | None = None
    delivery_id: str | None = None
    artifact_id: str | None = None
    launch_id: str | None = None
    parent_node_id: str | None = None
    branch_parent_run_id: str | None = None
    branch_child_run_id: str | None = None
    branchable: bool = False
    steerable: bool = False
    live: bool = False
    detail: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_text(self.node_id, field_name="node_id"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "kind", _require_text(self.kind, field_name="kind"))
        object.__setattr__(self, "source", _require_text(self.source, field_name="source"))
        object.__setattr__(self, "title", _require_text(self.title, field_name="title"))
        object.__setattr__(self, "summary", _require_text(self.summary, field_name="summary"))
        object.__setattr__(self, "state", _optional_text(self.state))
        object.__setattr__(self, "actor_id", _optional_text(self.actor_id))
        object.__setattr__(self, "actor_type", _optional_text(self.actor_type))
        object.__setattr__(self, "participant_role", _optional_text(self.participant_role))
        object.__setattr__(self, "task_id", _optional_text(self.task_id))
        object.__setattr__(self, "task_key", _optional_text(self.task_key))
        object.__setattr__(self, "worker_id", _optional_text(self.worker_id))
        object.__setattr__(self, "checkpoint_id", _optional_text(self.checkpoint_id))
        object.__setattr__(self, "checkpoint_record_id", _optional_text(self.checkpoint_record_id))
        object.__setattr__(self, "checkpoint_uri", _optional_text(self.checkpoint_uri))
        object.__setattr__(
            self,
            "checkpoint_boundary_kind",
            _optional_text(self.checkpoint_boundary_kind),
        )
        object.__setattr__(self, "message_id", _optional_text(self.message_id))
        object.__setattr__(self, "delivery_id", _optional_text(self.delivery_id))
        object.__setattr__(self, "artifact_id", _optional_text(self.artifact_id))
        object.__setattr__(self, "launch_id", _optional_text(self.launch_id))
        object.__setattr__(self, "parent_node_id", _optional_text(self.parent_node_id))
        object.__setattr__(self, "branch_parent_run_id", _optional_text(self.branch_parent_run_id))
        object.__setattr__(self, "branch_child_run_id", _optional_text(self.branch_child_run_id))
        object.__setattr__(self, "detail", _coerce_string_dict(self.detail, field_name="detail"))
        if self.logical_index < 0:
            raise ValueError("logical_index must be >= 0")

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrLogicalTimelineNode:
        return cls(
            node_id=_require_text(payload.get("node_id"), field_name="node_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            created_at=_parse_datetime(payload.get("created_at"), field_name="created_at"),
            logical_index=int(payload.get("logical_index") or 0),
            kind=_require_text(payload.get("kind"), field_name="kind"),
            source=_require_text(payload.get("source"), field_name="source"),
            title=_require_text(payload.get("title"), field_name="title"),
            summary=_require_text(payload.get("summary"), field_name="summary"),
            state=_optional_text(payload.get("state")),
            actor_id=_optional_text(payload.get("actor_id")),
            actor_type=_optional_text(payload.get("actor_type")),
            participant_role=_optional_text(payload.get("participant_role")),
            task_id=_optional_text(payload.get("task_id")),
            task_key=_optional_text(payload.get("task_key")),
            worker_id=_optional_text(payload.get("worker_id")),
            checkpoint_id=_optional_text(payload.get("checkpoint_id")),
            checkpoint_record_id=_optional_text(payload.get("checkpoint_record_id")),
            checkpoint_uri=_optional_text(payload.get("checkpoint_uri")),
            checkpoint_boundary_kind=_optional_text(payload.get("checkpoint_boundary_kind")),
            message_id=_optional_text(payload.get("message_id")),
            delivery_id=_optional_text(payload.get("delivery_id")),
            artifact_id=_optional_text(payload.get("artifact_id")),
            launch_id=_optional_text(payload.get("launch_id")),
            parent_node_id=_optional_text(payload.get("parent_node_id")),
            branch_parent_run_id=_optional_text(payload.get("branch_parent_run_id")),
            branch_child_run_id=_optional_text(payload.get("branch_child_run_id")),
            branchable=bool(payload.get("branchable")),
            steerable=bool(payload.get("steerable")),
            live=bool(payload.get("live")),
            detail=_coerce_string_dict(payload.get("detail"), field_name="detail"),
        )


@dataclass(frozen=True, slots=True)
class SmrLogicalTimeline:
    project_id: str
    run_id: str
    generated_at: datetime
    run_state: str
    latest_node_id: str | None = None
    nodes: list[SmrLogicalTimelineNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id", _require_text(self.project_id, field_name="project_id")
        )
        object.__setattr__(self, "run_id", _require_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "run_state", _require_text(self.run_state, field_name="run_state"))
        object.__setattr__(self, "latest_node_id", _optional_text(self.latest_node_id))

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrLogicalTimeline:
        raw_nodes = payload.get("nodes")
        if raw_nodes is None:
            nodes: list[SmrLogicalTimelineNode] = []
        elif isinstance(raw_nodes, list):
            nodes = [
                SmrLogicalTimelineNode.from_wire(item)
                for item in raw_nodes
                if isinstance(item, dict)
            ]
        else:
            raise ValueError("nodes must be a list")
        return cls(
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            generated_at=_parse_datetime(payload.get("generated_at"), field_name="generated_at"),
            run_state=_require_text(payload.get("run_state"), field_name="run_state"),
            latest_node_id=_optional_text(payload.get("latest_node_id")),
            nodes=nodes,
        )


@dataclass(frozen=True, slots=True)
class SmrRunEventLogEntry:
    event_log_id: str
    project_id: str
    run_id: str
    occurred_at: datetime
    source: str
    event_kind: str
    title: str
    summary: str
    status: str | None = None
    logical_timeline_node_id: str | None = None
    context_event_id: str | None = None
    disposition_id: str | None = None
    task_id: str | None = None
    task_key: str | None = None
    actor_id: str | None = None
    participant_role: str | None = None
    detail: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunEventLogEntry:
        return cls(
            event_log_id=_require_text(payload.get("event_log_id"), field_name="event_log_id"),
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            occurred_at=_parse_datetime(payload.get("occurred_at"), field_name="occurred_at"),
            source=_require_text(payload.get("source"), field_name="source"),
            event_kind=_require_text(payload.get("event_kind"), field_name="event_kind"),
            title=_require_text(payload.get("title"), field_name="title"),
            summary=_require_text(payload.get("summary"), field_name="summary"),
            status=_optional_text(payload.get("status")),
            logical_timeline_node_id=_optional_text(payload.get("logical_timeline_node_id")),
            context_event_id=_optional_text(payload.get("context_event_id")),
            disposition_id=_optional_text(payload.get("disposition_id")),
            task_id=_optional_text(payload.get("task_id")),
            task_key=_optional_text(payload.get("task_key")),
            actor_id=_optional_text(payload.get("actor_id")),
            participant_role=_optional_text(payload.get("participant_role")),
            detail=_coerce_string_dict(payload.get("detail"), field_name="detail"),
        )


@dataclass(frozen=True, slots=True)
class SmrRunEventLog:
    project_id: str
    run_id: str
    generated_at: datetime
    sources: list[str] = field(default_factory=list)
    event_kinds: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    entries: list[SmrRunEventLogEntry] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunEventLog:
        raw_entries = payload.get("entries")
        if raw_entries is None:
            entries: list[SmrRunEventLogEntry] = []
        elif isinstance(raw_entries, list):
            entries = [
                SmrRunEventLogEntry.from_wire(item)
                for item in raw_entries
                if isinstance(item, dict)
            ]
        else:
            raise ValueError("entries must be a list")
        return cls(
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            generated_at=_parse_datetime(payload.get("generated_at"), field_name="generated_at"),
            sources=_coerce_string_list(payload.get("sources"), field_name="sources"),
            event_kinds=_coerce_string_list(payload.get("event_kinds"), field_name="event_kinds"),
            statuses=_coerce_string_list(payload.get("statuses"), field_name="statuses"),
            entries=entries,
        )


@dataclass(frozen=True, slots=True)
class SmrAuthorityReadouts:
    project_id: str
    run_id: str
    generated_at: datetime
    source_authority_version: str
    runtime_authority_source_version: str | None = None
    public_status: dict[str, Any] | None = None
    operator_control: dict[str, Any] = field(default_factory=dict)
    diagnostic: dict[str, Any] = field(default_factory=dict)
    runtime_authority: dict[str, Any] | None = None
    compatibility: dict[str, Any] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)

    @property
    def typed_runtime_authority(self) -> ManagedResearchRuntimeAuthority | None:
        """Parse the optional backend authority payload without using compatibility data."""

        if self.runtime_authority is None:
            return None
        from synth_ai.core.research._legacy.models.run_authority import (
            ManagedResearchRuntimeAuthority,
        )

        authority = ManagedResearchRuntimeAuthority.from_wire(self.runtime_authority)
        if authority.project_id != self.project_id or authority.run_id != self.run_id:
            raise ValueError("runtime authority identity does not match its project/run readout")
        if self.runtime_authority_source_version is None:
            raise ValueError(
                "runtime_authority_source_version is required when runtime authority is included"
            )
        if authority.source_authority_version != self.runtime_authority_source_version:
            raise ValueError("runtime authority version does not match its project/run readout")
        return authority

    @property
    def authority_tasks(self) -> tuple[ManagedResearchAuthorityTask, ...]:
        """Return canonical typed task rows from the backend authority readout."""

        authority = self.typed_runtime_authority
        return () if authority is None else authority.tasks

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrAuthorityReadouts:
        return cls(
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            generated_at=_parse_datetime(payload.get("generated_at"), field_name="generated_at"),
            source_authority_version=_require_text(
                payload.get("source_authority_version"),
                field_name="source_authority_version",
            ),
            runtime_authority_source_version=_optional_text(
                payload.get("runtime_authority_source_version")
            ),
            public_status=_coerce_optional_any_dict(
                payload.get("public_status"), field_name="public_status"
            ),
            operator_control=_coerce_any_dict(
                payload.get("operator_control"), field_name="operator_control"
            ),
            diagnostic=_coerce_any_dict(payload.get("diagnostic"), field_name="diagnostic"),
            runtime_authority=_coerce_optional_any_dict(
                payload.get("runtime_authority"), field_name="runtime_authority"
            ),
            compatibility=_coerce_any_dict(
                payload.get("compatibility"), field_name="compatibility"
            ),
            links=_coerce_string_dict(payload.get("links"), field_name="links"),
        )

    def to_wire(self) -> dict[str, Any]:
        typed_runtime_authority = self.typed_runtime_authority
        return {
            "project_id": self.project_id,
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "source_authority_version": self.source_authority_version,
            "runtime_authority_source_version": self.runtime_authority_source_version,
            "public_status": (dict(self.public_status) if self.public_status is not None else None),
            "operator_control": dict(self.operator_control),
            "diagnostic": dict(self.diagnostic),
            "runtime_authority": (
                typed_runtime_authority.to_wire() if typed_runtime_authority is not None else None
            ),
            "compatibility": dict(self.compatibility),
            "links": dict(self.links),
        }


__all__ = [
    "SmrAuthorityReadouts",
    "SmrBranchMode",
    "SmrLogicalTimeline",
    "SmrLogicalTimelineNode",
    "SmrRunEventLog",
    "SmrRunEventLogEntry",
    "SmrRunBranchRequest",
    "SmrRunBranchResponse",
]
