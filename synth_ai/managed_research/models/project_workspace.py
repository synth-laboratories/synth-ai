"""Typed project workspace projection models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

ProjectWorkspaceTruthStatus = Literal["accepted", "proposed", "needs_review", "stale", "missing"]


class ProjectWorkspaceActorControlStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    INTERRUPT_REQUESTED = "interrupt_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ProjectWorkspaceObjectiveStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    REVIEW_PENDING = "review_pending"
    COMPLETE = "complete"
    FAILED = "failed"
    WITHDRAWN = "withdrawn"


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _require_string(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_string(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _int_value(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _bool_value(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string_list(payload: object, *, label: str) -> list[str]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be an array when provided")
    values: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError(f"{label} entries must be strings")
        normalized = item.strip()
        if normalized:
            values.append(normalized)
    return values


def _mapping_list(payload: object, *, label: str) -> list[Mapping[str, object]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be an array when provided")
    return [_require_mapping(item, label=f"{label}[]") for item in payload]


def _truth_status(payload: Mapping[str, object], key: str) -> ProjectWorkspaceTruthStatus:
    value = _require_string(payload, key, label=key)
    if value not in {"accepted", "proposed", "needs_review", "stale", "missing"}:
        raise ValueError(f"{key} has unknown truth status {value!r}")
    return value  # type: ignore[return-value]


def _actor_control_status(
    payload: Mapping[str, object], key: str
) -> ProjectWorkspaceActorControlStatus:
    value = _require_string(payload, key, label=key)
    try:
        return ProjectWorkspaceActorControlStatus(value)
    except ValueError as exc:
        raise ValueError(f"{key} has unknown actor control status {value!r}") from exc


def _objective_status(payload: Mapping[str, object], key: str) -> ProjectWorkspaceObjectiveStatus:
    value = _optional_string(payload, key) or ProjectWorkspaceObjectiveStatus.ACTIVE.value
    return ProjectWorkspaceObjectiveStatus(value)


@dataclass(frozen=True)
class ProjectWorkspaceAuthority:
    projection_authority: str
    canon_policy: str
    durable_truth_writer: str
    proposal_sources: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceAuthority:
        mapping = _require_mapping(payload, label="project workspace authority")
        return cls(
            projection_authority=_require_string(
                mapping, "projection_authority", label="authority.projection_authority"
            ),
            canon_policy=_require_string(mapping, "canon_policy", label="authority.canon_policy"),
            durable_truth_writer=_require_string(
                mapping, "durable_truth_writer", label="authority.durable_truth_writer"
            ),
            proposal_sources=_string_list(
                mapping.get("proposal_sources"), label="authority.proposal_sources"
            ),
        )


@dataclass(frozen=True)
class ProjectWorkspaceSummary:
    project_id: str
    name: str
    timezone: str
    phase: str
    readiness: str
    mission: str | None = None
    active_run_id: str | None = None
    latest_run_id: str | None = None
    active_run_count: int = 0
    actor_count: int = 0
    paused_actor_count: int = 0
    event_count: int = 0
    objective_count: int = 0
    experiment_count: int = 0
    pending_review_count: int = 0
    accepted_knowledge: bool = False

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceSummary:
        mapping = _require_mapping(payload, label="project workspace summary")
        return cls(
            project_id=_require_string(mapping, "project_id", label="summary.project_id"),
            name=_require_string(mapping, "name", label="summary.name"),
            timezone=_require_string(mapping, "timezone", label="summary.timezone"),
            phase=_require_string(mapping, "phase", label="summary.phase"),
            readiness=_require_string(mapping, "readiness", label="summary.readiness"),
            mission=_optional_string(mapping, "mission"),
            active_run_id=_optional_string(mapping, "active_run_id"),
            latest_run_id=_optional_string(mapping, "latest_run_id"),
            active_run_count=_int_value(mapping, "active_run_count"),
            actor_count=_int_value(mapping, "actor_count"),
            paused_actor_count=_int_value(mapping, "paused_actor_count"),
            event_count=_int_value(mapping, "event_count"),
            objective_count=_int_value(mapping, "objective_count"),
            experiment_count=_int_value(mapping, "experiment_count"),
            pending_review_count=_int_value(mapping, "pending_review_count"),
            accepted_knowledge=_bool_value(mapping, "accepted_knowledge"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceRun:
    run_id: str
    public_state: str
    truth_status: ProjectWorkspaceTruthStatus
    created_at: str
    updated_at: str
    href: str
    started_at: str | None = None
    finished_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceRun:
        mapping = _require_mapping(payload, label="project workspace run")
        return cls(
            run_id=_require_string(mapping, "run_id", label="run.run_id"),
            public_state=_require_string(mapping, "public_state", label="run.public_state"),
            truth_status=_truth_status(mapping, "truth_status"),
            created_at=_require_string(mapping, "created_at", label="run.created_at"),
            updated_at=_require_string(mapping, "updated_at", label="run.updated_at"),
            href=_require_string(mapping, "href", label="run.href"),
            started_at=_optional_string(mapping, "started_at"),
            finished_at=_optional_string(mapping, "finished_at"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceActor:
    actor_id: str
    run_id: str
    actor_key: str
    actor_type: str
    state: str
    pause_state: str
    control_status: ProjectWorkspaceActorControlStatus
    updated_at: str
    href: str
    last_activity_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceActor:
        mapping = _require_mapping(payload, label="project workspace actor")
        return cls(
            actor_id=_require_string(mapping, "actor_id", label="actor.actor_id"),
            run_id=_require_string(mapping, "run_id", label="actor.run_id"),
            actor_key=_require_string(mapping, "actor_key", label="actor.actor_key"),
            actor_type=_require_string(mapping, "actor_type", label="actor.actor_type"),
            state=_require_string(mapping, "state", label="actor.state"),
            pause_state=_require_string(mapping, "pause_state", label="actor.pause_state"),
            control_status=_actor_control_status(mapping, "control_status"),
            updated_at=_require_string(mapping, "updated_at", label="actor.updated_at"),
            href=_require_string(mapping, "href", label="actor.href"),
            last_activity_at=_optional_string(mapping, "last_activity_at"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceEvent:
    event_id: str
    event_kind: str
    source_family: str
    status: str
    review_policy: str
    truth_status: ProjectWorkspaceTruthStatus
    observed_at: str
    summary: str | None = None
    run_id: str | None = None
    actor_id: str | None = None
    href: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceEvent:
        mapping = _require_mapping(payload, label="project workspace event")
        return cls(
            event_id=_require_string(mapping, "event_id", label="event.event_id"),
            event_kind=_require_string(mapping, "event_kind", label="event.event_kind"),
            source_family=_require_string(mapping, "source_family", label="event.source_family"),
            status=_require_string(mapping, "status", label="event.status"),
            review_policy=_require_string(mapping, "review_policy", label="event.review_policy"),
            truth_status=_truth_status(mapping, "truth_status"),
            observed_at=_require_string(mapping, "observed_at", label="event.observed_at"),
            summary=_optional_string(mapping, "summary"),
            run_id=_optional_string(mapping, "run_id"),
            actor_id=_optional_string(mapping, "actor_id"),
            href=_optional_string(mapping, "href"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceContextPack:
    status: str
    note: str
    slice_count: int = 0
    accepted_slice_count: int = 0
    tentative_slice_count: int = 0
    stale_slice_count: int = 0
    preview: str | None = None
    missing_inputs: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceContextPack:
        mapping = _require_mapping(payload, label="project workspace context pack")
        return cls(
            status=_require_string(mapping, "status", label="context_pack.status"),
            note=_require_string(mapping, "note", label="context_pack.note"),
            slice_count=_int_value(mapping, "slice_count"),
            accepted_slice_count=_int_value(mapping, "accepted_slice_count"),
            tentative_slice_count=_int_value(mapping, "tentative_slice_count"),
            stale_slice_count=_int_value(mapping, "stale_slice_count"),
            preview=_optional_string(mapping, "preview"),
            missing_inputs=_string_list(
                mapping.get("missing_inputs"), label="context_pack.missing_inputs"
            ),
        )


@dataclass(frozen=True)
class ProjectWorkspaceCanonChange:
    change_id: str
    change_kind: str
    title: str
    source_run_id: str | None = None
    accepted_at: str | None = None
    href: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceCanonChange:
        mapping = _require_mapping(payload, label="project workspace canon change")
        return cls(
            change_id=_require_string(mapping, "change_id", label="canon_change.change_id"),
            change_kind=_require_string(mapping, "change_kind", label="canon_change.change_kind"),
            title=_require_string(mapping, "title", label="canon_change.title"),
            source_run_id=_optional_string(mapping, "source_run_id"),
            accepted_at=_optional_string(mapping, "accepted_at"),
            href=_optional_string(mapping, "href"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceChangeSet:
    changeset_id: str
    title: str
    status: str
    truth_status: ProjectWorkspaceTruthStatus
    review_policy: str
    updated_at: str
    href: str
    item_count: int = 0
    run_id: str | None = None
    decided_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceChangeSet:
        mapping = _require_mapping(payload, label="project workspace changeset")
        return cls(
            changeset_id=_require_string(mapping, "changeset_id", label="changeset.changeset_id"),
            title=_require_string(mapping, "title", label="changeset.title"),
            status=_require_string(mapping, "status", label="changeset.status"),
            truth_status=_truth_status(mapping, "truth_status"),
            review_policy=_require_string(
                mapping, "review_policy", label="changeset.review_policy"
            ),
            updated_at=_require_string(mapping, "updated_at", label="changeset.updated_at"),
            href=_require_string(mapping, "href", label="changeset.href"),
            item_count=_int_value(mapping, "item_count"),
            run_id=_optional_string(mapping, "run_id"),
            decided_at=_optional_string(mapping, "decided_at"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceNextAction:
    action_id: str
    priority: str
    title: str
    reason: str
    href: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceNextAction:
        mapping = _require_mapping(payload, label="project workspace next action")
        return cls(
            action_id=_require_string(mapping, "action_id", label="next_action.action_id"),
            priority=_require_string(mapping, "priority", label="next_action.priority"),
            title=_require_string(mapping, "title", label="next_action.title"),
            reason=_require_string(mapping, "reason", label="next_action.reason"),
            href=_optional_string(mapping, "href"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceObjective:
    objective_id: str
    objective_kind: str
    title: str
    body: str
    status: ProjectWorkspaceObjectiveStatus
    evaluation_state: str
    truth_status: ProjectWorkspaceTruthStatus
    updated_at: str
    review_summary: str | None = None
    reviewed_at: str | None = None
    run_id: str | None = None
    progress_count: int = 0
    achievement_count: int = 0
    percent_complete: float | None = None
    active_task_count: int = 0
    pending_claim_count: int = 0
    related_run_count: int = 0
    budget_max_cost_cents: int | None = None
    budget_spent_cost_cents: int | None = None
    budget_max_tokens: int | None = None
    budget_spent_tokens: int | None = None
    linked_artifact_ids: list[str] = field(default_factory=list)
    linked_entry_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceObjective:
        mapping = _require_mapping(payload, label="project workspace objective")
        return cls(
            objective_id=_require_string(mapping, "objective_id", label="objective.objective_id"),
            objective_kind=_require_string(
                mapping, "objective_kind", label="objective.objective_kind"
            ),
            title=_require_string(mapping, "title", label="objective.title"),
            body=_require_string(mapping, "body", label="objective.body"),
            status=_objective_status(mapping, "status"),
            evaluation_state=_require_string(
                mapping, "evaluation_state", label="objective.evaluation_state"
            ),
            truth_status=_truth_status(mapping, "truth_status"),
            updated_at=_require_string(mapping, "updated_at", label="objective.updated_at"),
            review_summary=_optional_string(mapping, "review_summary"),
            reviewed_at=_optional_string(mapping, "reviewed_at"),
            run_id=_optional_string(mapping, "run_id"),
            progress_count=_int_value(mapping, "progress_count"),
            achievement_count=_int_value(mapping, "achievement_count"),
            percent_complete=(
                float(mapping["percent_complete"])
                if mapping.get("percent_complete") is not None
                else None
            ),
            active_task_count=_int_value(mapping, "active_task_count"),
            pending_claim_count=_int_value(mapping, "pending_claim_count"),
            related_run_count=_int_value(mapping, "related_run_count"),
            budget_max_cost_cents=(
                _int_value(mapping, "budget_max_cost_cents")
                if mapping.get("budget_max_cost_cents") is not None
                else None
            ),
            budget_spent_cost_cents=(
                _int_value(mapping, "budget_spent_cost_cents")
                if mapping.get("budget_spent_cost_cents") is not None
                else None
            ),
            budget_max_tokens=(
                _int_value(mapping, "budget_max_tokens")
                if mapping.get("budget_max_tokens") is not None
                else None
            ),
            budget_spent_tokens=(
                _int_value(mapping, "budget_spent_tokens")
                if mapping.get("budget_spent_tokens") is not None
                else None
            ),
            linked_artifact_ids=_string_list(
                mapping.get("linked_artifact_ids"), label="objective.linked_artifact_ids"
            ),
            linked_entry_ids=_string_list(
                mapping.get("linked_entry_ids"), label="objective.linked_entry_ids"
            ),
        )


@dataclass(frozen=True)
class ProjectWorkspaceExperiment:
    experiment_id: str
    title: str
    hypothesis: str
    status: str
    truth_status: ProjectWorkspaceTruthStatus
    updated_at: str
    summary: str | None = None
    recommendation: str | None = None
    disposition: str | None = None
    run_id: str | None = None
    linked_artifact_ids: list[str] = field(default_factory=list)
    linked_entry_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceExperiment:
        mapping = _require_mapping(payload, label="project workspace experiment")
        return cls(
            experiment_id=_require_string(
                mapping, "experiment_id", label="experiment.experiment_id"
            ),
            title=_require_string(mapping, "title", label="experiment.title"),
            hypothesis=_require_string(mapping, "hypothesis", label="experiment.hypothesis"),
            status=_require_string(mapping, "status", label="experiment.status"),
            truth_status=_truth_status(mapping, "truth_status"),
            updated_at=_require_string(mapping, "updated_at", label="experiment.updated_at"),
            summary=_optional_string(mapping, "summary"),
            recommendation=_optional_string(mapping, "recommendation"),
            disposition=_optional_string(mapping, "disposition"),
            run_id=_optional_string(mapping, "run_id"),
            linked_artifact_ids=_string_list(
                mapping.get("linked_artifact_ids"), label="experiment.linked_artifact_ids"
            ),
            linked_entry_ids=_string_list(
                mapping.get("linked_entry_ids"), label="experiment.linked_entry_ids"
            ),
        )


@dataclass(frozen=True)
class ProjectWorkspaceKnowledge:
    status: ProjectWorkspaceTruthStatus
    source: str
    note: str
    content_preview: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceKnowledge:
        mapping = _require_mapping(payload, label="project workspace knowledge")
        return cls(
            status=_truth_status(mapping, "status"),
            source=_require_string(mapping, "source", label="knowledge.source"),
            note=_require_string(mapping, "note", label="knowledge.note"),
            content_preview=_optional_string(mapping, "content_preview"),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceReviewItem:
    review_id: str
    review_kind: str
    title: str
    reason: str
    truth_status: ProjectWorkspaceTruthStatus
    source_run_id: str | None = None
    href: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceReviewItem:
        mapping = _require_mapping(payload, label="project workspace review item")
        return cls(
            review_id=_require_string(mapping, "review_id", label="review.review_id"),
            review_kind=_require_string(mapping, "review_kind", label="review.review_kind"),
            title=_require_string(mapping, "title", label="review.title"),
            reason=_require_string(mapping, "reason", label="review.reason"),
            truth_status=_truth_status(mapping, "truth_status"),
            source_run_id=_optional_string(mapping, "source_run_id"),
            href=_optional_string(mapping, "href"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceReport:
    report_id: str
    title: str
    mode: str
    truth_status: ProjectWorkspaceTruthStatus
    run_id: str
    created_at: str
    source_entry_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceReport:
        mapping = _require_mapping(payload, label="project workspace report")
        return cls(
            report_id=_require_string(mapping, "report_id", label="report.report_id"),
            title=_require_string(mapping, "title", label="report.title"),
            mode=_require_string(mapping, "mode", label="report.mode"),
            truth_status=_truth_status(mapping, "truth_status"),
            run_id=_require_string(mapping, "run_id", label="report.run_id"),
            created_at=_require_string(mapping, "created_at", label="report.created_at"),
            source_entry_ids=_string_list(
                mapping.get("source_entry_ids"), label="report.source_entry_ids"
            ),
        )


@dataclass(frozen=True)
class ProjectWorkspaceLaunchRisk:
    risk_id: str
    severity: str
    status: str
    title: str
    detail: str

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceLaunchRisk:
        mapping = _require_mapping(payload, label="project workspace launch risk")
        return cls(
            risk_id=_require_string(mapping, "risk_id", label="risk.risk_id"),
            severity=_require_string(mapping, "severity", label="risk.severity"),
            status=_require_string(mapping, "status", label="risk.status"),
            title=_require_string(mapping, "title", label="risk.title"),
            detail=_require_string(mapping, "detail", label="risk.detail"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceLinks:
    project: str
    runs: str
    experiments: str
    knowledge: str
    review: str
    reports: str
    settings: str

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceLinks:
        mapping = _require_mapping(payload, label="project workspace links")
        return cls(
            project=_require_string(mapping, "project", label="links.project"),
            runs=_require_string(mapping, "runs", label="links.runs"),
            experiments=_require_string(mapping, "experiments", label="links.experiments"),
            knowledge=_require_string(mapping, "knowledge", label="links.knowledge"),
            review=_require_string(mapping, "review", label="links.review"),
            reports=_require_string(mapping, "reports", label="links.reports"),
            settings=_require_string(mapping, "settings", label="links.settings"),
        )


@dataclass(frozen=True)
class ProjectWorkspaceProjection:
    project_id: str
    generated_at: str
    authority: ProjectWorkspaceAuthority
    summary: ProjectWorkspaceSummary
    knowledge: ProjectWorkspaceKnowledge
    context_pack: ProjectWorkspaceContextPack
    links: ProjectWorkspaceLinks
    runs: list[ProjectWorkspaceRun] = field(default_factory=list)
    actors: list[ProjectWorkspaceActor] = field(default_factory=list)
    events: list[ProjectWorkspaceEvent] = field(default_factory=list)
    objectives: list[ProjectWorkspaceObjective] = field(default_factory=list)
    experiments: list[ProjectWorkspaceExperiment] = field(default_factory=list)
    changesets: list[ProjectWorkspaceChangeSet] = field(default_factory=list)
    canon_changes: list[ProjectWorkspaceCanonChange] = field(default_factory=list)
    next_actions: list[ProjectWorkspaceNextAction] = field(default_factory=list)
    review_queue: list[ProjectWorkspaceReviewItem] = field(default_factory=list)
    reports: list[ProjectWorkspaceReport] = field(default_factory=list)
    launch_risks: list[ProjectWorkspaceLaunchRisk] = field(default_factory=list)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectWorkspaceProjection:
        mapping = _require_mapping(payload, label="project workspace projection")
        return cls(
            project_id=_require_string(mapping, "project_id", label="workspace.project_id"),
            generated_at=_require_string(mapping, "generated_at", label="workspace.generated_at"),
            authority=ProjectWorkspaceAuthority.from_wire(mapping.get("authority")),
            summary=ProjectWorkspaceSummary.from_wire(mapping.get("summary")),
            knowledge=ProjectWorkspaceKnowledge.from_wire(mapping.get("knowledge")),
            context_pack=ProjectWorkspaceContextPack.from_wire(mapping.get("context_pack")),
            links=ProjectWorkspaceLinks.from_wire(mapping.get("links")),
            runs=[
                ProjectWorkspaceRun.from_wire(item)
                for item in _mapping_list(mapping.get("runs"), label="workspace.runs")
            ],
            actors=[
                ProjectWorkspaceActor.from_wire(item)
                for item in _mapping_list(mapping.get("actors"), label="workspace.actors")
            ],
            events=[
                ProjectWorkspaceEvent.from_wire(item)
                for item in _mapping_list(mapping.get("events"), label="workspace.events")
            ],
            objectives=[
                ProjectWorkspaceObjective.from_wire(item)
                for item in _mapping_list(mapping.get("objectives"), label="workspace.objectives")
            ],
            experiments=[
                ProjectWorkspaceExperiment.from_wire(item)
                for item in _mapping_list(mapping.get("experiments"), label="workspace.experiments")
            ],
            changesets=[
                ProjectWorkspaceChangeSet.from_wire(item)
                for item in _mapping_list(mapping.get("changesets"), label="workspace.changesets")
            ],
            canon_changes=[
                ProjectWorkspaceCanonChange.from_wire(item)
                for item in _mapping_list(
                    mapping.get("canon_changes"), label="workspace.canon_changes"
                )
            ],
            next_actions=[
                ProjectWorkspaceNextAction.from_wire(item)
                for item in _mapping_list(
                    mapping.get("next_actions"), label="workspace.next_actions"
                )
            ],
            review_queue=[
                ProjectWorkspaceReviewItem.from_wire(item)
                for item in _mapping_list(
                    mapping.get("review_queue"), label="workspace.review_queue"
                )
            ],
            reports=[
                ProjectWorkspaceReport.from_wire(item)
                for item in _mapping_list(mapping.get("reports"), label="workspace.reports")
            ],
            launch_risks=[
                ProjectWorkspaceLaunchRisk.from_wire(item)
                for item in _mapping_list(
                    mapping.get("launch_risks"), label="workspace.launch_risks"
                )
            ],
            raw=dict(mapping),
        )


__all__ = [
    "ProjectWorkspaceActor",
    "ProjectWorkspaceActorControlStatus",
    "ProjectWorkspaceAuthority",
    "ProjectWorkspaceCanonChange",
    "ProjectWorkspaceChangeSet",
    "ProjectWorkspaceContextPack",
    "ProjectWorkspaceEvent",
    "ProjectWorkspaceNextAction",
    "ProjectWorkspaceExperiment",
    "ProjectWorkspaceKnowledge",
    "ProjectWorkspaceLaunchRisk",
    "ProjectWorkspaceLinks",
    "ProjectWorkspaceObjective",
    "ProjectWorkspaceObjectiveStatus",
    "ProjectWorkspaceProjection",
    "ProjectWorkspaceReport",
    "ProjectWorkspaceReviewItem",
    "ProjectWorkspaceRun",
    "ProjectWorkspaceSummary",
    "ProjectWorkspaceTruthStatus",
]
