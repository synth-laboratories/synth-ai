# Generated from backend native evidence producer; do not edit.
# Source SHA256: 1da88eeedfa3ab17d556ae9cc9ffd88495ad874c98eafaad801db3c25207f504
"""Native-owned runtime evidence attached to exact Forge Experiment revisions.

See notes/specifications/tanha/core/forge-native-attachments.md and Decision0002.
These are native execution/result records, never Forge Trial or Result shadows.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from .forge.contracts import (
    ExactReference,
    Producer,
    Scope,
    contract_digest,
    canonical_bytes,
)
from .forge.operations import WriteOperation, Receipt
from .native_result_outcome import (
    ResearchResultOutcome,
    require_result_measurement,
)


class NativeResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    kind: Literal["native_result"] = "native_result"
    evaluation_mode: Literal["live", "mock", "fixture", "offline_fallback"]
    scientific_status: Literal["scientific", "non_scientific_smoke"]
    intervention_receipt: dict[str, JsonValue]
    lifecycle_success: bool | None = None
    verifier_success: bool | None = None
    run_id: Optional[str] = None
    experiment_run_id: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    experiment_run_ref: ExactReference | None = None
    container_run_id: Optional[str] = None
    candidate_id: Optional[str] = None
    candidate_kind: Optional[str] = None
    candidate_label: Optional[str] = None
    metric: str = Field(..., min_length=1, max_length=255)
    metric_direction: str = Field(
        default="higher_is_better", min_length=1, max_length=64
    )
    value: float | None = Field(strict=True, allow_inf_nan=False)
    outcome: ResearchResultOutcome = ResearchResultOutcome.MEASURED
    comparison_cohort_key: str | None = None
    comparison_missing_dimensions: list[str] = Field(default_factory=list)
    baseline_value: Optional[float] = Field(default=None, strict=True)
    delta: Optional[float] = Field(default=None, strict=True)
    dataset_or_task_set_id: Optional[str] = None
    dataset_or_task_set_version: Optional[str] = None
    container_name: Optional[str] = None
    container_version: Optional[str] = None
    container_digest: Optional[str] = None
    eval_profile_id: Optional[str] = None
    eval_profile_version: Optional[str] = None
    verifier_or_scorer_id: Optional[str] = None
    verifier_or_scorer_version: Optional[str] = None
    taskset_seed: Optional[int] = None
    task_ids: List[str] = Field(default_factory=list)
    example_ids: List[str] = Field(default_factory=list)
    sample_size: Optional[int] = Field(default=None, ge=0)
    seed_set: List[int] = Field(default_factory=list)
    split_name: Optional[str] = None
    scorer_config_digest: Optional[str] = None
    per_example_artifact_id: Optional[str] = None
    summary_artifact_id: Optional[str] = None
    per_example_artifact_path: Optional[str] = None
    summary_artifact_path: Optional[str] = None
    cost_cents: Optional[int] = Field(default=None, ge=0)
    tokens: Optional[int] = Field(default=None, ge=0)
    wall_time_seconds: Optional[float] = Field(default=None, ge=0, strict=True)
    evidence_grade: Optional[str] = Field(default=None, max_length=64)
    truth_status: str = Field(default="observed", min_length=1, max_length=64)
    caveats: Optional[str] = None
    reviewer_notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def explicit_result_outcome(self):
        require_result_measurement(self.value, self.outcome)
        return self


class NativeRunLinkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["native_run_link"] = "native_run_link"
    run_id: str = Field(min_length=1, max_length=128)
    role: str = Field(default="primary", min_length=1, max_length=64)
    source_role: str = Field(min_length=1, max_length=64)
    notes: str | None = Field(default=None, max_length=8000)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


NativePayload = Annotated[
    NativeRunLinkPayload | NativeResultPayload, Field(discriminator="kind")
]


class NativeScientificAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["smr.forge_native_attachment.v1"] = (
        "smr.forge_native_attachment.v1"
    )
    attachment_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    scope: Scope
    experiment: ExactReference
    producer: Producer
    payload: NativePayload

    @model_validator(mode="after")
    def exact_scientific_basis(self):
        if (self.experiment.authority, self.experiment.kind) != ("forge", "experiment"):
            raise ValueError(
                "native attachment requires exact canonical Forge Experiment basis"
            )
        if len(canonical_bytes(self)) > 65536:
            raise ValueError("native attachment exceeds 65536 byte bound")
        return self

    def reference(self) -> ExactReference:
        return ExactReference(
            authority="smr",
            kind=self.payload.kind,
            record_id=self.attachment_id,
            revision="1",
            digest_sha256=contract_digest(self),
        )


class NativeAttachmentIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    argument_digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attachment: NativeScientificAttachment
    operation: WriteOperation

    @model_validator(mode="after")
    def exact_intent_binding(self):
        attachment = self.attachment
        if (
            self.operation.scope != attachment.scope
            or self.operation.producer != attachment.producer
            or self.operation.record_id != attachment.experiment.record_id
            or self.operation.expected_revision != int(attachment.experiment.revision)
            or self.operation.payload.kind != "experiment"
            or tuple(self.operation.payload.sources)
            != (attachment.experiment, attachment.reference())
        ):
            raise ValueError(
                "native evidence intent differs from its canonical basis/citation/producer"
            )
        return self


class NativeAttachmentRead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attachment: NativeScientificAttachment
    status: Literal["pending", "attached"]
    receipt: Receipt | None = None

    @model_validator(mode="after")
    def first_class_delivery(self):
        if (self.status == "attached") != (self.receipt is not None):
            raise ValueError(
                "native attachment delivery status requires exact canonical receipt"
            )
        return self


class NativeAttachmentPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attachments: tuple[NativeAttachmentRead, ...] = Field(max_length=1000)
    truncated: bool = Field(strict=True)
    next_cursor: str | None = None
