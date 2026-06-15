"""Pydantic v2 schemas for the Open Research v1 surface.

Field names mirror the HTTP contract verbatim
(``open_research_http_contract_v1.md``). The MCP tool layer accepts
loose ``dict`` arguments (JSON-RPC) but coerces them through these
models so contract drift is a parse error rather than a silent shape
mismatch.

No fallbacks. No legacy-shape parsing. ``model_config`` is
``ConfigDict(extra='ignore')`` for forward compatibility with additive
contract fields, but every documented field is typed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MetricOperator = Literal[">=", "<=", "=="]
DeoKind = Literal["open_ended_discovery"]
SubmissionStatus = Literal["review_pending", "approved", "rejected", "launched", "duplicate"]
ReviewClass = Literal["ok", "safety", "validity", "novelty", "theme_fit"]
ExperimentStatus = Literal["running", "done", "failed", "canceled"]
ExperimentSource = Literal["computer_use", "mcp", "signed_in_web"]
WorkProductState = Literal["blocked", "ready", "viewable"]
ExperimentStatusFilter = Literal["running", "done", "failed", "all"]


class _OpenResearchModel(BaseModel):
    """Common config: allow additive contract fields without breaking."""

    model_config = ConfigDict(extra="ignore", frozen=False)


class MetricTarget(_OpenResearchModel):
    name: str = Field(min_length=1, max_length=256)
    operator: MetricOperator
    value: float


class SubmitterSchema(_OpenResearchModel):
    handle: str = Field(min_length=1, max_length=128)
    fingerprint: str | None = None


# ---- request models ---------------------------------------------------


class SubmitQuestionArgs(_OpenResearchModel):
    """Arguments accepted by ``open_research_submit_question``.

    The MCP tool peels off ``submitter_fingerprint`` and routes it into
    either the ``X-OR-Fingerprint`` header (anonymous) or the request
    body's ``submitter.fingerprint`` (the contract says the backend
    mirrors the header into the field for anonymous callers; we send
    both for parity with the public composer).
    """

    project_slug: str = Field(min_length=1, max_length=128)
    queue_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=2000)
    hypothesis: str = Field(default="", max_length=1000)
    metric_target: MetricTarget
    deo_kind: DeoKind
    rubric_acknowledged: bool
    submitter_handle: str = Field(min_length=1, max_length=128)
    submitter_fingerprint: str | None = None

    @field_validator("rubric_acknowledged")
    @classmethod
    def _rubric_must_be_true(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("rubric_acknowledged must be true to submit a question")
        return value


class ListExperimentsArgs(_OpenResearchModel):
    project_slug: str | None = None
    status: ExperimentStatusFilter | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    cursor: str | None = None


# ---- response models --------------------------------------------------


class ProjectSummary(_OpenResearchModel):
    slug: str
    name: str
    tagline: str
    challenge_url: str
    baseline_score: float | None = None
    current_best_score: float | None = None
    best_experiment_id: str | None = None
    default_queue_id: str
    supported_queue_ids: list[str]


class ListProjectsResponse(_OpenResearchModel):
    projects: list[ProjectSummary]


class ProjectBestSnapshot(_OpenResearchModel):
    experiment_id: str | None = None
    score: float | None = None
    submitter_handle: str | None = None


class ProjectPaperRef(_OpenResearchModel):
    title: str
    url: str


class ProjectPriorReceipt(_OpenResearchModel):
    experiment_id: str
    score: float | None = None


class ProjectResources(_OpenResearchModel):
    code_url: str | None = None
    eval_harness_url: str | None = None
    papers: list[ProjectPaperRef] = Field(default_factory=list)
    prior_receipts: list[ProjectPriorReceipt] = Field(default_factory=list)


class ProjectDetail(_OpenResearchModel):
    slug: str
    name: str
    tagline: str
    challenge_statement_md: str
    baseline_score: float | None = None
    current_best: ProjectBestSnapshot | None = None
    rubric_md: str
    resources: ProjectResources
    default_queue_id: str
    supported_queue_ids: list[str]


class QueueAdmission(_OpenResearchModel):
    open: bool
    depth: int = Field(ge=0)
    estimated_wait_seconds: int | None = Field(default=None, ge=0)
    dispatch_pool_capacity_ok: bool
    runtime_readiness_ok: bool
    resource_availability_ok: bool


class QueueSummary(_OpenResearchModel):
    id: str
    project_slug: str
    horizon_seconds: int = Field(ge=1)
    work_mode: str
    entitlement: str
    unsigned_in_allowed: bool
    admission: QueueAdmission


class ListQueuesResponse(_OpenResearchModel):
    queues: list[QueueSummary]


class SubmissionResponse(_OpenResearchModel):
    submission_id: str
    status: SubmissionStatus
    review_verdict: Any | None = None
    duplicate: bool = False
    idempotency_key: str | None = None
    experiment_id: str | None = None


class ReviewVerdict(_OpenResearchModel):
    review_class: ReviewClass = Field(alias="class")
    message: str
    actionable: str
    reviewed_at: str

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SubmissionDetail(_OpenResearchModel):
    submission_id: str
    project_slug: str
    queue_id: str
    status: SubmissionStatus
    review_verdict: ReviewVerdict | None = None
    experiment_id: str | None = None
    objective_id: str | None = None
    submitted_at: str
    launched_at: str | None = None


class ExperimentSummary(_OpenResearchModel):
    experiment_id: str
    project_slug: str
    submitter_handle: str | None = None
    submission_id: str | None = None
    source: ExperimentSource
    status: ExperimentStatus
    score: float | None = None
    objective_id: str | None = None
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    wallclock_seconds_used: int | None = None
    wallclock_seconds_budget: int | None = None


class ListExperimentsResponse(_OpenResearchModel):
    experiments: list[ExperimentSummary]
    next_cursor: str | None = None


class ExperimentStatusDetail(_OpenResearchModel):
    smr_state: str
    work_product_state: WorkProductState
    blockers: list[str] = Field(default_factory=list)


class RewardSeriesPoint(_OpenResearchModel):
    t: float
    reward: float


class AchievementRow(_OpenResearchModel):
    name: str
    first_step: int | None = None
    count: int = 0


class ScoreRow(_OpenResearchModel):
    metric: str
    value: float


class RolloutSummary(_OpenResearchModel):
    rollout_id: str
    thumbnail_url: str | None = None
    frames_url: str | None = None
    clip_url: str | None = None
    reward: float | None = None
    achievements_count: int | None = None


class ExperimentDetail(_OpenResearchModel):
    experiment_id: str
    project_slug: str
    submitter_handle: str | None = None
    submission_id: str | None = None
    source: ExperimentSource
    status: ExperimentStatus
    status_detail: ExperimentStatusDetail
    score: float | None = None
    metric_target: MetricTarget | None = None
    metric_target_hit: bool | None = None
    objective_id: str | None = None
    reward_series: list[RewardSeriesPoint] = Field(default_factory=list)
    achievements: list[AchievementRow] = Field(default_factory=list)
    score_table: list[ScoreRow] = Field(default_factory=list)
    rollouts: list[RolloutSummary] = Field(default_factory=list)
    artifacts_url: str
    receipt_url: str


class ReceiptPayload(_OpenResearchModel):
    experiment_id: str
    project_slug: str
    receipt_md: str
    receipt_url: str
    bundle_url: str
    score: float | None = None
    metric_target_hit: bool | None = None
    submitted_at: str | None = None
    finished_at: str | None = None


class BundleDownloadResult(_OpenResearchModel):
    """Return shape for ``open_research_download_bundle``."""

    experiment_id: str
    output_path: str
    bytes_written: int = Field(ge=0)
    sha256: str
    content_type: str | None = None


__all__ = [
    "AchievementRow",
    "BundleDownloadResult",
    "DeoKind",
    "ExperimentDetail",
    "ExperimentSource",
    "ExperimentStatus",
    "ExperimentStatusDetail",
    "ExperimentStatusFilter",
    "ExperimentSummary",
    "ListExperimentsArgs",
    "ListExperimentsResponse",
    "ListProjectsResponse",
    "ListQueuesResponse",
    "MetricOperator",
    "MetricTarget",
    "ProjectBestSnapshot",
    "ProjectDetail",
    "ProjectPaperRef",
    "ProjectPriorReceipt",
    "ProjectResources",
    "ProjectSummary",
    "QueueAdmission",
    "QueueSummary",
    "ReceiptPayload",
    "ReviewClass",
    "ReviewVerdict",
    "RewardSeriesPoint",
    "RolloutSummary",
    "ScoreRow",
    "SubmissionDetail",
    "SubmissionResponse",
    "SubmissionStatus",
    "SubmitQuestionArgs",
    "SubmitterSchema",
    "WorkProductState",
]
