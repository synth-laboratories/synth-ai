"""Typed contracts for the Effort-primary Intern research program.

Effort is the Async organizer. The Intern objectives, milestones, tasks,
progress claims, and links below are **subordinate** to an Effort: every row
carries a non-null ``effort_id``, and the Effort is the path segment on every
create route, so it cannot be omitted.

The Intern program store is a separate store from the SMR objective store. It
shares the objective/milestone/task *logic* but never the rows, and it never
references the SMR run task graph -- the Intern kicks a swarm run off and polls
it, and SMR owns the resulting DAG.

Backend remains the contract authority.

# See: backend packages/smr/contracts/public_api/v1/intern_program.py (WP6 SS4/SS5)
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictInternProgramContract(BaseModel):
    """Extra fields are drift, not forward compatibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_wire(cls, value: object) -> Self:
        return cls.model_validate(value)


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


class InternObjectiveKind(StrEnum):
    """The two objective shapes an Intern may open under an Effort."""

    OPEN_ENDED_QUESTION = "open_ended_question"
    DIRECTED_EFFORT_OUTCOME = "directed_effort_outcome"


class InternObjectiveStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    REVIEW_PENDING = "review_pending"
    COMPLETE = "complete"
    FAILED = "failed"
    WITHDRAWN = "withdrawn"


class InternMilestoneState(StrEnum):
    PLANNED = "planned"
    READY = "ready"
    ACTIVE = "active"
    VALIDATION_PENDING = "validation_pending"
    VALIDATING = "validating"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"
    FAILED = "failed"
    STOPPED = "stopped"


class InternEffortTaskState(StrEnum):
    PLANNED = "planned"
    READY = "ready"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW_REQUIRED = "review_required"
    REPAIR_REQUIRED = "repair_required"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"
    SUPERSEDED = "superseded"


InternMilestoneKind = Literal["subquestion", "suboutcome"]
InternProgressClaimKind = Literal["progress", "achievement"]
InternProgressClaimStatus = Literal["pending", "review_pending", "accepted", "rejected"]

#: Link targets are SMR-owned rows. A link is a *reference*: creating one never
#: writes the target, and never grants the Intern authority over it.
InternObjectiveLinkKind = Literal[
    "smr_run",
    "smr_open_ended_question",
    "smr_directed_effort_outcome",
    "smr_work_product",
    "smr_report",
    "smr_experiment",
]
InternObjectiveLinkRole = Literal[
    "primary",
    "supporting",
    "reviewer",
    "blocker",
    "out_of_scope",
]

InternEffortStatus = Literal[
    "active",
    "paused",
    "waiting",
    "blocked",
    "ready_for_review",
    "archived_reference",
]
InternEffortType = Literal["research", "eval_factory", "optimizer"]

_CRITERIA_MAX = 12


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class InternObjectiveCreateRequest(_StrictInternProgramContract):
    """Open one Intern objective under an Effort.

    ``effort_id`` is deliberately absent: the Effort is the path segment, so the
    binding cannot be omitted or contradicted.
    """

    objective_kind: InternObjectiveKind
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=20_000)
    scope: str = Field(min_length=1, max_length=4_000)
    question_text: str | None = Field(default=None, max_length=20_000)
    evidence_requirements: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    resolution_criteria: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    outcome_text: str | None = Field(default=None, max_length=20_000)
    success_criteria: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    deliverable_requirements: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    max_evaluation_iterations: int = Field(default=1, ge=1, le=10)
    idempotency_key: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def require_body_for_kind(self) -> InternObjectiveCreateRequest:
        if self.objective_kind is InternObjectiveKind.OPEN_ENDED_QUESTION:
            if not (self.question_text or "").strip():
                raise ValueError("open_ended_question requires question_text")
            if self.outcome_text is not None:
                raise ValueError("open_ended_question does not accept outcome_text")
        else:
            if not (self.outcome_text or "").strip():
                raise ValueError("directed_effort_outcome requires outcome_text")
            if self.question_text is not None:
                raise ValueError("directed_effort_outcome does not accept question_text")
        return self


class InternObjectivePatchRequest(_StrictInternProgramContract):
    """Revise one Intern objective.

    ``objective_kind`` selects the backing table and is required. ``effort_id``
    is not patchable: the binding is immutable, so moving an objective between
    Efforts is a new objective, not an update.
    """

    objective_kind: InternObjectiveKind
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, min_length=1, max_length=20_000)
    scope: str | None = Field(default=None, min_length=1, max_length=4_000)
    question_text: str | None = Field(default=None, max_length=20_000)
    outcome_text: str | None = Field(default=None, max_length=20_000)
    status: InternObjectiveStatus | None = None
    evaluation_state: str | None = Field(default=None, max_length=200)
    review_summary: str | None = Field(default=None, max_length=20_000)
    success_criteria: tuple[str, ...] | None = Field(default=None, max_length=_CRITERIA_MAX)
    resolution_criteria: tuple[str, ...] | None = Field(default=None, max_length=_CRITERIA_MAX)
    evidence_requirements: tuple[str, ...] | None = Field(default=None, max_length=_CRITERIA_MAX)
    deliverable_requirements: tuple[str, ...] | None = Field(default=None, max_length=_CRITERIA_MAX)

    @model_validator(mode="after")
    def require_one_change(self) -> InternObjectivePatchRequest:
        changes = self.model_dump(exclude_none=True)
        changes.pop("objective_kind", None)
        if not changes:
            raise ValueError("intern objective patch changes no fields")
        return self


class InternObjectiveResponse(_StrictInternProgramContract):
    objective_id: str
    objective_kind: InternObjectiveKind
    effort_id: str
    project_id: str
    status: InternObjectiveStatus
    evaluation_state: str
    title: str
    description: str
    scope: str
    question_text: str | None = None
    outcome_text: str | None = None
    evidence_requirements: tuple[str, ...] = ()
    resolution_criteria: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    deliverable_requirements: tuple[str, ...] = ()
    evaluation_iteration: int = Field(default=0, ge=0)
    max_evaluation_iterations: int = Field(default=1, ge=1)
    review_summary: str | None = None
    milestone_count: int = Field(default=0, ge=0)
    task_count: int = Field(default=0, ge=0)
    open_task_count: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Milestones
# ---------------------------------------------------------------------------


class InternMilestoneCreateRequest(_StrictInternProgramContract):
    parent_kind: InternObjectiveKind
    parent_id: str
    milestone_kind: InternMilestoneKind
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=8_000)
    acceptance_criteria: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    position: int = Field(default=0, ge=0)
    idempotency_key: str | None = Field(default=None, max_length=200)


class InternMilestoneTransitionRequest(_StrictInternProgramContract):
    next_state: InternMilestoneState
    reason: str | None = Field(default=None, max_length=2_000)


class InternMilestoneResponse(_StrictInternProgramContract):
    milestone_id: str
    effort_id: str
    project_id: str
    parent_kind: InternObjectiveKind
    parent_id: str
    milestone_kind: InternMilestoneKind
    title: str
    objective: str
    state: InternMilestoneState
    allowed_next_states: tuple[InternMilestoneState, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    evidence_artifact_ids: tuple[str, ...] = ()
    evidence_entry_ids: tuple[str, ...] = ()
    position: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Tasks (the Intern planner checklist -- not the SMR run task graph)
# ---------------------------------------------------------------------------


class InternEffortTaskCreateRequest(_StrictInternProgramContract):
    """Create one Intern planner task under an Effort.

    This is not a swarm task. It never becomes a node in an SMR run DAG, and
    the Intern never plans that DAG.
    """

    title: str = Field(min_length=1, max_length=200)
    body: str | None = Field(default=None, max_length=8_000)
    milestone_id: str | None = None
    objective_kind: InternObjectiveKind | None = None
    objective_id: str | None = None
    acceptance_criteria: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    dependency_task_ids: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    position: int = Field(default=0, ge=0)
    idempotency_key: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def require_objective_pair(self) -> InternEffortTaskCreateRequest:
        if (self.objective_kind is None) != (self.objective_id is None):
            raise ValueError("objective_kind and objective_id travel together")
        return self


class InternEffortTaskPatchRequest(_StrictInternProgramContract):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    body: str | None = Field(default=None, max_length=8_000)
    state: InternEffortTaskState | None = None
    acceptance_criteria: tuple[str, ...] | None = Field(default=None, max_length=_CRITERIA_MAX)
    artifact_ids: tuple[str, ...] | None = Field(default=None, max_length=_CRITERIA_MAX)
    assigned_actor_key: str | None = Field(default=None, max_length=512)
    assignment_reason: str | None = Field(default=None, max_length=2_000)
    position: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_one_change(self) -> InternEffortTaskPatchRequest:
        if not self.model_dump(exclude_none=True):
            raise ValueError("intern task patch changes no fields")
        return self


class InternEffortTaskResponse(_StrictInternProgramContract):
    intern_task_id: str
    effort_id: str
    project_id: str
    milestone_id: str | None = None
    objective_kind: InternObjectiveKind | None = None
    objective_id: str | None = None
    title: str
    body: str | None = None
    state: InternEffortTaskState
    allowed_next_states: tuple[InternEffortTaskState, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    dependency_task_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    assigned_actor_key: str | None = None
    assignment_reason: str | None = None
    position: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Progress claims
# ---------------------------------------------------------------------------


class InternProgressClaimCreateRequest(_StrictInternProgramContract):
    """Fold a result back into an objective.

    ``smr_run_id`` is the run whose results the claim folds in. It is a
    read-only reference to an SMR-owned row.
    """

    objective_kind: InternObjectiveKind
    objective_id: str
    title: str | None = Field(default=None, max_length=200)
    summary: str = Field(min_length=1, max_length=8_000)
    claim_kind: InternProgressClaimKind = "progress"
    percent_complete: float | None = Field(default=None, ge=0, le=100)
    intern_task_id: str | None = None
    milestone_id: str | None = None
    smr_run_id: str | None = None
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    expected_remaining_work: tuple[str, ...] = Field(default=(), max_length=_CRITERIA_MAX)
    idempotency_key: str | None = Field(default=None, max_length=200)


class InternProgressClaimResponse(_StrictInternProgramContract):
    progress_claim_id: str
    effort_id: str
    project_id: str
    objective_kind: InternObjectiveKind
    objective_id: str
    intern_task_id: str | None = None
    milestone_id: str | None = None
    title: str | None = None
    summary: str
    claim_kind: InternProgressClaimKind
    percent_complete: float | None = None
    status: InternProgressClaimStatus
    evidence_refs: tuple[str, ...] = ()
    expected_remaining_work: tuple[str, ...] = ()
    smr_run_id: str | None = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Objective links (references only)
# ---------------------------------------------------------------------------


class InternObjectiveLinkCreateRequest(_StrictInternProgramContract):
    """Record that an Intern objective relates to an SMR-owned row.

    Creating a link writes one Intern row and nothing else. It never writes the
    target table.
    """

    objective_kind: InternObjectiveKind
    link_kind: InternObjectiveLinkKind
    target_id: str
    link_role: InternObjectiveLinkRole = "primary"
    note: str | None = Field(default=None, max_length=2_000)


class InternObjectiveLinkResponse(_StrictInternProgramContract):
    objective_link_id: str
    effort_id: str
    objective_kind: InternObjectiveKind
    objective_id: str
    link_kind: InternObjectiveLinkKind
    target_id: str
    link_role: InternObjectiveLinkRole
    note: str | None = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Effort board and Effort detail rollup
# ---------------------------------------------------------------------------


class InternEffortRef(_StrictInternProgramContract):
    """The Effort block shared by the board row and the detail rollup.

    Sourced from ``smr_efforts``; a projection of the SMR Effort response, not a
    second authority over it.
    """

    effort_id: str
    factory_id: str
    project_id: str
    name: str
    hypothesis_or_topic: str | None = None
    status: InternEffortStatus
    effort_type: InternEffortType
    decision_needed: bool = False
    decision_note: str | None = None
    next_wake_at: datetime | None = None
    latest_run_id: str | None = None
    latest_report_id: str | None = None
    latest_work_product_id: str | None = None
    created_at: datetime
    updated_at: datetime


class InternEffortSummary(InternEffortRef):
    """One Effort board row: the Effort block plus open-work counts.

    The board carries no progress/results/experiments/knowledge bodies. The
    caller fetches detail for the selected Effort.
    """

    open_objective_count: int = Field(default=0, ge=0)
    open_task_count: int = Field(default=0, ge=0)
    open_question_count: int = Field(default=0, ge=0)
    latest_claim_at: datetime | None = None
    blocked: bool = False


class InternEffortBoardResponse(_StrictInternProgramContract):
    schema_version: Literal["smr.intern-effort-board.v1"] = "smr.intern-effort-board.v1"
    efforts: tuple[InternEffortSummary, ...] = ()
    next_cursor: str | None = None


class InternEffortObjectiveProgress(_StrictInternProgramContract):
    objective_id: str
    objective_kind: InternObjectiveKind
    title: str
    status: str
    evaluation_state: str
    evaluation_iteration: int = Field(default=0, ge=0)
    max_evaluation_iterations: int = Field(default=1, ge=1)
    milestone_count: int = Field(default=0, ge=0)
    task_count: int = Field(default=0, ge=0)
    open_task_count: int = Field(default=0, ge=0)
    latest_claim_at: datetime | None = None
    percent_complete: float | None = None


class InternEffortMilestoneProgress(_StrictInternProgramContract):
    milestone_id: str
    title: str
    state: InternMilestoneState
    milestone_kind: InternMilestoneKind
    parent_kind: InternObjectiveKind
    parent_id: str
    position: int = Field(default=0, ge=0)
    allowed_next_states: tuple[InternMilestoneState, ...] = ()


class InternEffortTaskProgress(_StrictInternProgramContract):
    intern_task_id: str
    title: str
    state: InternEffortTaskState
    milestone_id: str | None = None
    objective_id: str | None = None
    position: int = Field(default=0, ge=0)


class InternEffortClaimProgress(_StrictInternProgramContract):
    progress_claim_id: str
    objective_id: str
    objective_kind: InternObjectiveKind
    claim_kind: InternProgressClaimKind
    summary: str
    percent_complete: float | None = None
    status: InternProgressClaimStatus
    smr_run_id: str | None = None
    created_at: datetime


class InternEffortOpenQuestion(_StrictInternProgramContract):
    """One parked ask-and-continue question for this Effort.

    An open question does not freeze the board; the Intern keeps working other
    Efforts while it waits.
    """

    interaction_id: str
    question: str
    asked_at: datetime | None = None


class InternEffortProgressRollup(_StrictInternProgramContract):
    objective_counts: dict[str, int] = Field(default_factory=dict)
    objectives: tuple[InternEffortObjectiveProgress, ...] = ()
    milestone_counts: dict[str, int] = Field(default_factory=dict)
    milestones: tuple[InternEffortMilestoneProgress, ...] = ()
    task_counts: dict[str, int] = Field(default_factory=dict)
    tasks: tuple[InternEffortTaskProgress, ...] = ()
    recent_claims: tuple[InternEffortClaimProgress, ...] = ()
    open_questions: tuple[InternEffortOpenQuestion, ...] = ()


class InternEffortWorkProduct(_StrictInternProgramContract):
    work_product_id: str
    title: str | None = None
    kind: str | None = None
    status: str | None = None
    run_id: str | None = None
    url: str | None = None
    #: Whether this row reached the Effort through the Effort binding itself or
    #: through an Intern objective link.
    link_source: Literal["effort_binding", "intern_objective_link"] = "effort_binding"
    created_at: datetime | None = None


class InternEffortWorkSummary(_StrictInternProgramContract):
    work_summary_id: str
    summary: str
    #: False when the summary belongs to the Intern rather than to this Effort,
    #: so a board never presents org-wide work as Effort-specific.
    effort_scoped: bool = False
    created_at: datetime | None = None
    runtime_kind: str | None = None


class InternEffortReportRef(_StrictInternProgramContract):
    report_id: str
    title: str | None = None
    status: str | None = None
    created_at: datetime | None = None


class InternEffortResultsRollup(_StrictInternProgramContract):
    work_products: tuple[InternEffortWorkProduct, ...] = ()
    work_summaries: tuple[InternEffortWorkSummary, ...] = ()
    latest_report: InternEffortReportRef | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class InternEffortLinkedRun(_StrictInternProgramContract):
    """One SMR run this Effort kicked off or linked to.

    Read-only. The Intern polls runs; it does not own or edit their task graph.
    """

    run_id: str
    status: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    is_nonterminal: bool = False
    link_role: InternObjectiveLinkRole = "primary"
    link_source: Literal["effort_binding", "intern_objective_link"] = "effort_binding"


class InternEffortExperiment(_StrictInternProgramContract):
    experiment_id: str
    name: str | None = None
    status: str | None = None
    created_at: datetime | None = None


class InternEffortExperimentsRollup(_StrictInternProgramContract):
    linked_runs: tuple[InternEffortLinkedRun, ...] = ()
    experiments: tuple[InternEffortExperiment, ...] = ()
    counts: dict[str, int] = Field(default_factory=dict)


class InternEffortMemoryHit(_StrictInternProgramContract):
    hit_id: str
    kind: str | None = None
    title: str | None = None
    snippet: str | None = None
    occurred_at: datetime | None = None


class InternEffortLogEntry(_StrictInternProgramContract):
    entry_id: str
    title: str | None = None
    created_at: datetime | None = None


class InternEffortEvidenceRef(_StrictInternProgramContract):
    artifact_id: str
    kind: str | None = None
    created_at: datetime | None = None


class InternEffortKnowledgeRollup(_StrictInternProgramContract):
    memory_hits: tuple[InternEffortMemoryHit, ...] = ()
    experiment_log_entries: tuple[InternEffortLogEntry, ...] = ()
    evidence_refs: tuple[InternEffortEvidenceRef, ...] = ()
    counts: dict[str, int] = Field(default_factory=dict)


class InternEffortSpend(_StrictInternProgramContract):
    """Day and month burn against the ceilings that bind Async work."""

    daily_cents: int = Field(default=0, ge=0)
    daily_ceiling_cents: int = Field(default=0, ge=0)
    monthly_cents: int = Field(default=0, ge=0)
    monthly_ceiling_cents: int = Field(default=0, ge=0)
    blocked_reason: str | None = None


class InternEffortStickyHost(_StrictInternProgramContract):
    """The persistent host lease backing Async work.

    Pausing frees the lease; resuming reacquires it.
    """

    host_kind: str | None = None
    lease_id: str | None = None
    bound: bool = False


class InternEffortRuntimeStrip(_StrictInternProgramContract):
    """Secondary ops chrome. Cycle and wake are debug detail, not the product."""

    async_status: str | None = None
    cycle_number: int = Field(default=0, ge=0)
    next_wake_at: datetime | None = None
    durable_cursor: str | None = None
    parked_question_count: int = Field(default=0, ge=0)
    #: The Effort holding the current cycle slot, which need not be this one.
    active_effort_id: str | None = None
    context_snapshot_at: datetime | None = None
    #: How the Effort inventory fared against its context sub-budget on the last
    #: compile. A nonzero ``omitted`` says Efforts were trimmed, which is the
    #: signal that the board is reading a partial inventory.
    effort_inventory_available: int = Field(default=0, ge=0)
    effort_inventory_included: int = Field(default=0, ge=0)
    effort_inventory_omitted: int = Field(default=0, ge=0)
    spend: InternEffortSpend = Field(default_factory=InternEffortSpend)
    sticky_host: InternEffortStickyHost = Field(default_factory=InternEffortStickyHost)


class InternEffortDetailResponse(_StrictInternProgramContract):
    """The Effort detail rollup.

    Every block is aggregated from Postgres. Nothing here is read back out of
    workflow history.
    """

    schema_version: Literal["smr.intern-effort-detail.v1"] = "smr.intern-effort-detail.v1"
    generated_at: datetime
    effort: InternEffortRef
    progress: InternEffortProgressRollup = Field(default_factory=InternEffortProgressRollup)
    results: InternEffortResultsRollup = Field(default_factory=InternEffortResultsRollup)
    experiments: InternEffortExperimentsRollup = Field(
        default_factory=InternEffortExperimentsRollup
    )
    knowledge: InternEffortKnowledgeRollup = Field(default_factory=InternEffortKnowledgeRollup)
    runtime: InternEffortRuntimeStrip = Field(default_factory=InternEffortRuntimeStrip)


# ---------------------------------------------------------------------------
# Intern memory retrieval
# ---------------------------------------------------------------------------


class InternMemoryHitKind(StrEnum):
    """What kind of record a memory hit came from."""

    EVENT = "event"
    HANDOFF = "handoff"
    MESSAGE = "message"
    SEGMENT = "segment"


class InternMemoryHit(_StrictInternProgramContract):
    """One bounded keyword hit over an Intern's own history.

    ``hit_id`` is the stable ``kind:id`` handle search returns; passing it back
    to the retrieval endpoint fetches the same record.
    """

    hit_id: str
    kind: InternMemoryHitKind
    snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class InternMemorySearchResponse(_StrictInternProgramContract):
    """Search results, shaped exactly like the agent-facing tool result.

    The HTTP surface and the MCP tool describe the same retrieval with the same
    fields and no wrapper, so an operator reading memory sees what the Intern
    read.
    """

    query: str
    limit: int
    kinds: tuple[str, ...] = ()
    hits: tuple[InternMemoryHit, ...] = ()
    hit_count: int = Field(default=0, ge=0)


class InternMemoryHitResponse(_StrictInternProgramContract):
    hit: InternMemoryHit


__all__ = [
    "InternEffortBoardResponse",
    "InternEffortClaimProgress",
    "InternEffortDetailResponse",
    "InternEffortEvidenceRef",
    "InternEffortExperiment",
    "InternEffortExperimentsRollup",
    "InternEffortKnowledgeRollup",
    "InternEffortLinkedRun",
    "InternEffortLogEntry",
    "InternEffortMemoryHit",
    "InternEffortMilestoneProgress",
    "InternEffortObjectiveProgress",
    "InternEffortOpenQuestion",
    "InternEffortProgressRollup",
    "InternEffortRef",
    "InternEffortReportRef",
    "InternEffortResultsRollup",
    "InternEffortRuntimeStrip",
    "InternEffortSpend",
    "InternEffortStickyHost",
    "InternEffortSummary",
    "InternEffortTaskCreateRequest",
    "InternEffortTaskPatchRequest",
    "InternEffortTaskProgress",
    "InternEffortTaskResponse",
    "InternEffortTaskState",
    "InternEffortWorkProduct",
    "InternEffortWorkSummary",
    "InternMemoryHit",
    "InternMemoryHitKind",
    "InternMemoryHitResponse",
    "InternMemorySearchResponse",
    "InternMilestoneCreateRequest",
    "InternMilestoneResponse",
    "InternMilestoneState",
    "InternMilestoneTransitionRequest",
    "InternObjectiveCreateRequest",
    "InternObjectiveKind",
    "InternObjectiveLinkCreateRequest",
    "InternObjectiveLinkResponse",
    "InternObjectivePatchRequest",
    "InternObjectiveResponse",
    "InternObjectiveStatus",
    "InternProgressClaimCreateRequest",
    "InternProgressClaimResponse",
]
