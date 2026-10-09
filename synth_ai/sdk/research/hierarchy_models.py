"""Project hierarchy wires from backend a28b28d and Sublinear 646d2e4.

# See: testing/specifications/sdk/hierarchy_owner.md
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from synth_ai.core.contracts.json_value import JsonObject


def _identity(value: str) -> str:
    if str(UUID(value)) != value or UUID(value).int == 0:
        raise ValueError("hierarchy identity must be a canonical nonzero UUID")
    return value


EntityId = Annotated[str, Field(min_length=36, max_length=36), AfterValidator(_identity)]
Name = Annotated[str, Field(min_length=1, max_length=256, pattern=r"[!-~]+\z")]
OperationId = Annotated[str, Field(min_length=64, max_length=64, pattern=r"[0-9a-f]{64}\z")]
Digest = Annotated[str, Field(min_length=71, max_length=71, pattern=r"sha256:[0-9a-f]{64}\z")]
Positive = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
Natural = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
EntityKind = Literal[
    "research_program",
    "objective",
    "milestone",
    "answer",
    "answer_revision",
    "answer_source",
    "review",
    "progress_claim",
    "research_claim",
    "claim_revision",
    "claim_evidence",
    "claim_edge",
    "oeq_resolution",
    "run_scope",
    "objective_event",
    "task_link",
]
HierarchyPermission = Literal["read", "plan", "evidence", "review", "association", "transfer"]


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class HierarchyScope(Closed):
    organization_id: EntityId
    project_id: EntityId


class EntityKey(Closed):
    kind: EntityKind
    entity_id: Name


class HierarchyReference(Closed):
    owner: Name
    reference: Name
    schema_: Name = Field(alias="schema")
    sha256: Digest


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate original JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite original JSON: {value}")


class HierarchyOriginal(Closed):
    reference: HierarchyReference
    raw_json: Annotated[str, Field(min_length=1, max_length=16 * 1024 * 1024)]

    @model_validator(mode="after")
    def original(self) -> Self:
        raw = self.raw_json.encode("utf-8")
        if (
            len(raw) > 16 * 1024 * 1024
            or "sha256:" + hashlib.sha256(raw).hexdigest() != self.reference.sha256
        ):
            raise ValueError("hierarchy original digest mismatch")
        value = json.loads(
            self.raw_json, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
        if not isinstance(value, dict) or value.get("schema_version") != self.reference.schema_:
            raise ValueError("hierarchy original schema mismatch")
        # Preserve exact bytes. Server canonical codecs remain the authority,
        # including their float encoding; Python never reseals owner originals.
        return self

    def document(self) -> JsonObject:
        return json.loads(
            self.raw_json, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )


Text = Annotated[str, Field(strict=True, min_length=1, max_length=65536)]
Title = Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
Description = Annotated[str, Field(strict=True, max_length=65536)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
References = Annotated[tuple[HierarchyReference, ...], Field(max_length=128)]
Sources = Annotated[tuple[HierarchyReference, ...], Field(min_length=1, max_length=128)]
Criteria = Annotated[
    tuple[Annotated[str, Field(min_length=1, max_length=4096)], ...],
    Field(max_length=128),
]
ReviewDecision = Literal["accepted", "rejected", "needs_revision", "insufficient_evidence"]
Relation = Literal["supports", "contradicts", "qualifies", "independent_of"]
MilestoneState = Literal[
    "planned",
    "ready",
    "active",
    "validation_pending",
    "validating",
    "accepted",
    "blocked",
    "failed",
    "stopped",
]


class ResearchProgram(Closed):
    kind: Literal["research_program"]
    title: Title
    description: Description


class Objective(Closed):
    kind: Literal["objective"]
    objective_kind: Literal["oeq", "deo"]
    title: Title
    description: Description
    question_or_outcome: Text
    criteria: Criteria


class Milestone(Closed):
    kind: Literal["milestone"]
    title: Title
    description: Description
    acceptance_criteria: Criteria


class Answer(Closed):
    kind: Literal["answer"]
    title: Title


class Define(Closed):
    kind: Literal["define"]
    definition: Annotated[
        ResearchProgram | Objective | Milestone | Answer, Field(discriminator="kind")
    ]
    aliases: Annotated[tuple[Name, ...], Field(max_length=128)]
    parents: Annotated[tuple[EntityKey, ...], Field(max_length=128)]

    @model_validator(mode="after")
    def ordered(self):
        parents = [(key.kind, key.entity_id) for key in self.parents]
        if list(self.aliases) != sorted(set(self.aliases)) or parents != sorted(set(parents)):
            raise ValueError("hierarchy_command_alias_or_parent_order_invalid")
        return self


class EditMetadata(Closed):
    kind: Literal["edit_metadata"]
    title: Title
    description: Description


class MoveMilestone(Closed):
    kind: Literal["move_milestone"]
    next: MilestoneState

    @model_validator(mode="after")
    def separate_acceptance(self):
        if self.next == "accepted":
            raise ValueError("hierarchy_review_permission_required")
        return self


class AcceptMilestone(Closed):
    kind: Literal["accept_milestone"]
    validation: HierarchyReference
    review: HierarchyReference


class ProposeProgressClaim(Closed):
    kind: Literal["propose_progress_claim"]
    claim_kind: Literal["progress", "achievement"]
    summary: Text
    evidence: References


class ReviewProgressClaim(Closed):
    kind: Literal["review_progress_claim"]
    decision: Literal["accepted", "rejected"]
    summary: Text
    review: HierarchyReference


class PublishAnswerRevision(Closed):
    kind: Literal["publish_answer_revision"]
    answer: EntityKey
    objective: EntityKey
    answer_text: Annotated[str, Field(strict=True, min_length=1, max_length=1048576)]
    confidence: Confidence
    sources: Sources


class PublishScientificClaim(Closed):
    kind: Literal["publish_scientific_claim"]
    objective: EntityKey
    claim_kind: Literal["hypothesis", "finding", "limitation", "unknown"]
    statement: Text
    confidence: Confidence
    sources: Sources


class PublishClaimRevision(Closed):
    kind: Literal["publish_claim_revision"]
    claim: EntityKey
    statement: Text
    confidence: Confidence
    sources: Sources


class LinkClaimEvidence(Closed):
    kind: Literal["link_claim_evidence"]
    claim_revision: EntityKey
    source: HierarchyReference
    relation: Relation


class LinkAnswerSource(Closed):
    kind: Literal["link_answer_source"]
    answer_revision: EntityKey
    source: HierarchyReference


class LinkClaims(Closed):
    kind: Literal["link_claims"]
    source_revision: EntityKey
    target_revision: EntityKey
    relation: Relation


class ResolveOeq(Closed):
    kind: Literal["resolve_oeq"]
    objective: EntityKey
    answer_revision: EntityKey
    decision: ReviewDecision
    summary: Text
    review: HierarchyReference


class RecordObjectiveReview(Closed):
    kind: Literal["record_objective_review"]
    objective: EntityKey
    decision: ReviewDecision
    summary: Text
    review: HierarchyReference


class BindRun(Closed):
    kind: Literal["bind_run"]
    objective: EntityKey
    run_id: EntityId
    role: Literal["primary", "supporting", "reviewer", "blocker", "out_of_scope"]
    run_admission: HierarchyReference


class LinkTask(Closed):
    kind: Literal["link_task"]
    objective: EntityKey
    task: HierarchyReference


class RecordObjectiveEvent(Closed):
    kind: Literal["record_objective_event"]
    objective: EntityKey
    event_kind: Name
    evidence: Sources


Action = Annotated[
    Define
    | EditMetadata
    | MoveMilestone
    | AcceptMilestone
    | ProposeProgressClaim
    | ReviewProgressClaim
    | PublishAnswerRevision
    | PublishScientificClaim
    | PublishClaimRevision
    | LinkClaimEvidence
    | LinkAnswerSource
    | LinkClaims
    | ResolveOeq
    | RecordObjectiveReview
    | BindRun
    | LinkTask
    | RecordObjectiveEvent,
    Field(discriminator="kind"),
]


class EntityRead(Closed):
    kind: Literal["entity"]
    entity: EntityKey
    revision: Positive | None


class AliasRead(Closed):
    kind: Literal["alias"]
    entity_kind: EntityKind
    alias: Name


class ChildrenRead(Closed):
    kind: Literal["children"]
    parent: EntityKey
    after: EntityKey | None
    limit: Annotated[int, Field(strict=True, ge=1, le=128)]


class NotebookRead(Closed):
    kind: Literal["notebook"]
    objective: EntityKey
    after: EntityKey | None
    limit: Annotated[int, Field(strict=True, ge=1, le=128)]


class CurrentAnswerRead(Closed):
    kind: Literal["current_answer"]
    answer: EntityKey


class CurrentClaimRead(Closed):
    kind: Literal["current_claim"]
    claim: EntityKey


class OriginalRead(Closed):
    kind: Literal["original"]
    reference: HierarchyReference


class OperationRead(Closed):
    kind: Literal["operation"]
    operation_id: OperationId


class HeadRead(Closed):
    kind: Literal["head"]


class HierarchyRead(Closed):
    schema_version: Literal["sublinear.hierarchy-read.v1"]
    scope: HierarchyScope
    resource: Annotated[
        EntityRead
        | AliasRead
        | ChildrenRead
        | NotebookRead
        | CurrentAnswerRead
        | CurrentClaimRead
        | OriginalRead
        | OperationRead
        | HeadRead,
        Field(discriminator="kind"),
    ]


class HierarchyCommand(Closed):
    schema_version: Literal["sublinear.hierarchy-command.v1"] = "sublinear.hierarchy-command.v1"
    scope: HierarchyScope
    operation_id: OperationId
    expected_authority_epoch: Positive
    expected_owner_cursor: Natural
    target: EntityKey
    expected_entity_revision: Annotated[int, Field(strict=True, ge=0, lt=2**63 - 1)]
    action: Action


class HierarchyReadReply(Closed):
    schema_version: Literal["sublinear.hierarchy-read-reply.v1"]
    scope: HierarchyScope
    authority_epoch: Positive
    owner_committed_cursor: Natural
    ownership_phase: Literal["unselected", "prepared", "owned"]
    originals: tuple[HierarchyOriginal, ...]
    next_after: EntityKey | None
    current_child: HierarchyReference | None
    domain_revision: Positive | None

    @model_validator(mode="after")
    def child(self) -> Self:
        if (self.current_child is None) != (self.domain_revision is None):
            raise ValueError("hierarchy child/domain revision mismatch")
        if self.current_child is not None and self.current_child not in [
            x.reference for x in self.originals
        ]:
            raise ValueError("hierarchy current child original missing")
        return self


class HierarchyCommandReceipt(Closed):
    schema_version: Literal["sublinear.hierarchy-command-receipt.v1"]
    scope: HierarchyScope
    operation_id: OperationId
    authority_epoch: Positive
    owner_committed_cursor: Positive
    entity: EntityKey
    revision: Positive
    original: HierarchyReference
    produced: Annotated[tuple[HierarchyReference, ...], Field(min_length=1, max_length=2)]
    admitted_policy_use: HierarchyReference

    @model_validator(mode="after")
    def originals(self) -> Self:
        keys = [(r.owner, r.schema_, r.reference, r.sha256) for r in self.produced]
        if keys != sorted(set(keys)) or self.original not in self.produced:
            raise ValueError("hierarchy command original custody invalid")
        return self


class HierarchyTransferReceipt(Closed):
    schema_version: Literal["sublinear.hierarchy-transfer-receipt.v1"]
    scope: HierarchyScope
    operation_id: OperationId
    transfer_operation_id: OperationId
    phase: Literal["prepared", "staged", "owned"]
    authority_epoch: Positive
    owner_committed_cursor: Positive
    decision: HierarchyReference
    manifest: HierarchyReference
    preserved_record_count: Natural
    admitted_policy_use: HierarchyReference


class HierarchyTransferBundle(Closed):
    decision: HierarchyOriginal
    manifest: HierarchyOriginal
    chunks: tuple[HierarchyOriginal, ...]

    @model_validator(mode="after")
    def joined(self) -> Self:
        decision = self.decision.document()
        manifest = self.manifest.document()
        if (
            self.decision.reference.schema_ != "synth.planning-hierarchy-transfer-decision.v1"
            or self.manifest.reference.schema_ != "synth.planning-hierarchy-export.v1"
            or decision.get("scope") != manifest.get("scope")
            or decision.get("export_manifest") != self.manifest.reference.model_dump(by_alias=True)
            or decision.get("operation_id") != manifest.get("transfer_operation_id")
        ):
            raise ValueError("hierarchy transfer bundle join mismatch")
        return self
