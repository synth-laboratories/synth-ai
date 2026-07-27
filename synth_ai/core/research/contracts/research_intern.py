"""Typed contracts for one organization Research Intern and its resources.

Backend remains the contract authority. These models intentionally preserve
scoped policies and owner-authored evidence instead of flattening them into an
SDK-side source of truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_wire(cls, value: object) -> Self:
        return cls.model_validate(value)


class ResearchInternStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class MagiMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"
    SERAPH = "seraph"


class MagiCanonicalUser(StrEnum):
    CASPER = "Casper"
    MELCHIOR = "Melchior"
    BALTHASAR = "Balthasar"


MAGI_CANONICAL_USER_BY_MODE: dict[MagiMode, MagiCanonicalUser] = {
    MagiMode.SYNC: MagiCanonicalUser.CASPER,
    MagiMode.ASYNC: MagiCanonicalUser.MELCHIOR,
    MagiMode.SERAPH: MagiCanonicalUser.BALTHASAR,
}


class MagiDecisionKind(StrEnum):
    DELEGATE = "delegate"
    INSPECT = "inspect"
    PAUSE = "pause"
    INTERVENE = "intervene"
    RESUME = "resume"
    VERDICT = "verdict"
    REVISE = "revise"


class ResearchInternPolicySet(_StrictContract):
    organization: dict[str, Any] = Field(default_factory=dict)
    team: dict[str, Any] = Field(default_factory=dict)
    user: dict[str, Any] = Field(default_factory=dict)
    organization_policy_ref: str | None = None
    team_policy_ref: str | None = None
    user_policy_ref: str | None = None


class ResearchInternProvisionRequest(_StrictContract):
    display_name: str = Field(default="Research Intern", min_length=1, max_length=255)
    policies: ResearchInternPolicySet = Field(default_factory=ResearchInternPolicySet)
    attribution_team_id: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchInternPatchRequest(_StrictContract):
    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    status: ResearchInternStatus | None = None
    policies: ResearchInternPolicySet | None = None
    attribution_team_id: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_change(self) -> ResearchInternPatchRequest:
        if not self.model_fields_set:
            raise ValueError("ResearchInternPatchRequest must change at least one field")
        return self

    def to_wire(self) -> dict[str, Any]:
        """Preserve explicit nulls so attribution and metadata can be cleared."""
        return self.model_dump(mode="json", exclude_unset=True)


class ResearchInternResponse(_StrictContract):
    research_intern_id: str
    org_id: str
    display_name: str
    status: ResearchInternStatus
    policies: ResearchInternPolicySet
    attribution_user_id: str | None = None
    attribution_team_id: str | None = None
    state_generation: int = Field(ge=0)
    durable_state: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ResearchInternFactoryMembershipResponse(_StrictContract):
    research_intern_id: str
    org_id: str
    factory_id: str
    role: str
    attached_by_user_id: str | None = None
    attached_at: datetime


class MagiDecisionRequest(_StrictContract):
    mode: MagiMode
    decision_kind: MagiDecisionKind
    idempotency_key: str = Field(min_length=1, max_length=512)
    factory_id: str | None = None
    project_id: str | None = None
    experiment_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    state_patch: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=20_000)
    verdict: str | None = Field(default=None, max_length=255)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_seraph_verdict(self) -> MagiDecisionRequest:
        if self.decision_kind is MagiDecisionKind.VERDICT:
            if self.mode is not MagiMode.SERAPH or not self.verdict:
                raise ValueError("verdict decisions require Seraph mode and verdict")
        return self


class MagiDecisionReceiptResponse(_StrictContract):
    receipt_id: str
    research_intern_id: str
    org_id: str
    factory_id: str | None = None
    project_id: str | None = None
    experiment_id: str | None = None
    mode: MagiMode
    canonical_user: MagiCanonicalUser
    decision_kind: MagiDecisionKind
    idempotency_key: str
    state_generation: int = Field(ge=0)
    evidence_refs: list[str]
    state_patch: dict[str, Any]
    rationale: str
    verdict: str | None = None
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    decided_by_user_id: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def require_canonical_mode_user(self) -> MagiDecisionReceiptResponse:
        if self.canonical_user != MAGI_CANONICAL_USER_BY_MODE[self.mode]:
            raise ValueError("canonical_user does not match Magi mode")
        return self


class ProjectComputerLifecycle(StrEnum):
    PROVISIONING = "provisioning"
    READY = "ready"
    REPLACING = "replacing"
    RETIRED = "retired"
    ERROR = "error"


class ProjectComputerProvisionRequest(_StrictContract):
    factory_id: str
    provider_kind: str = Field(min_length=1, max_length=100)
    adapter_kind: str = Field(min_length=1, max_length=100)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectComputerResponse(_StrictContract):
    project_computer_id: str
    org_id: str
    research_intern_id: str
    factory_id: str
    project_id: str
    provider_kind: str
    adapter_kind: str
    provider_resource_ref: str | None = None
    source_repository_id: str
    source_revision: str
    snapshot_digest: str | None = None
    lifecycle: ProjectComputerLifecycle
    generation: int = Field(ge=0)
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ProjectComputerMaterializationReceipt(_StrictContract):
    """Adapter-authored proof of the code and snapshot present on a computer."""

    schema_version: Literal["smr.project-computer-materialization.v1"]
    owner: str = Field(min_length=1, max_length=255)
    provider_resource_ref: str = Field(min_length=1, max_length=1000)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_archive_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    workspace_path: str = Field(min_length=1, max_length=1000)
    materialized: bool
    receipt_ref: str = Field(min_length=1, max_length=1000)


class ProjectComputerRetirementReceipt(_StrictContract):
    schema_version: Literal["smr.project-computer-retirement.v1"]
    owner: str = Field(min_length=1, max_length=255)
    org_id: str
    factory_id: str
    project_id: str
    generation: int
    provider_kind: str
    adapter_kind: str
    provider_resource_ref: str
    disposition: Literal["retired", "already_absent"]
    pre_inventory: dict[str, Any]
    post_inventory: dict[str, Any]
    retired: bool
    receipt_ref: str = Field(min_length=1, max_length=1000)


class ProjectComputerCleanupRequest(_StrictContract):
    idempotency_key: str = Field(min_length=1, max_length=512)


class ProjectComputerCleanupReceiptResponse(_StrictContract):
    schema_version: Literal["smr.project-computer-cleanup.v1"]
    service_origin: Literal["urn:synth:research-intern:project-computer"]
    owner: Literal["research_intern_control_plane"]
    receipt_id: str
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    receipt_uri: str = Field(min_length=1, max_length=2000)
    org_id: str
    research_intern_id: str
    factory_id: str
    idempotency_key: str
    pre_inventory: list[ProjectComputerResponse]
    materialization_receipts: list[ProjectComputerMaterializationReceipt]
    retirement_receipts: list[ProjectComputerRetirementReceipt]
    post_inventory: list[ProjectComputerResponse]
    cleanup_complete: bool
    created_at: datetime

    @model_validator(mode="after")
    def validate_owner_receipt(self) -> ProjectComputerCleanupReceiptResponse:
        if self.receipt_id != self.content_digest:
            raise ValueError("cleanup receipt_id must equal its content_digest")
        if self.content_digest.removeprefix("sha256:") not in self.receipt_uri:
            raise ValueError("cleanup receipt URI must contain its content digest")
        if self.cleanup_complete != (not self.post_inventory):
            raise ValueError("cleanup_complete must match the post-cleanup inventory")
        if any(
            computer.factory_id != self.factory_id
            for computer in (*self.pre_inventory, *self.post_inventory)
        ):
            raise ValueError("cleanup inventory crossed its Factory boundary")
        if any(
            receipt.factory_id != self.factory_id
            for receipt in self.retirement_receipts
        ):
            raise ValueError("retirement receipt crossed its Factory boundary")
        if not all(receipt.materialized for receipt in self.materialization_receipts):
            raise ValueError("cleanup receipt contains unmaterialized computer evidence")
        return self


class ProjectComputerReplaceRequest(_StrictContract):
    factory_id: str
    provider_kind: str = Field(min_length=1, max_length=100)
    adapter_kind: str = Field(min_length=1, max_length=100)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataBindingCreateRequest(_StrictContract):
    factory_id: str
    name: str = Field(min_length=1, max_length=255)
    binding_kind: str = Field(min_length=1, max_length=100)
    authority_ref: str = Field(min_length=1, max_length=1000)
    access_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataBindingResponse(_StrictContract):
    data_binding_id: str
    org_id: str
    research_intern_id: str
    factory_id: str
    project_id: str
    name: str
    binding_kind: str
    authority_ref: str
    access_policy: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime


class DatasetRevisionCreateRequest(_StrictContract):
    revision_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_uri: str = Field(min_length=1, max_length=2000)
    schema_version: str = Field(min_length=1, max_length=255)
    parent_revision_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetRevisionResponse(_StrictContract):
    dataset_revision_id: str
    data_binding_id: str
    org_id: str
    factory_id: str
    project_id: str
    revision_digest: str
    manifest_uri: str
    schema_version: str
    parent_revision_id: str | None = None
    metadata: dict[str, Any]
    created_at: datetime

    @model_validator(mode="after")
    def validate_revision_digest(self) -> DatasetRevisionResponse:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.revision_digest) is None:
            raise ValueError("revision_digest must be a sha256 digest")
        return self


__all__ = [
    "DataBindingCreateRequest",
    "DataBindingResponse",
    "DatasetRevisionCreateRequest",
    "DatasetRevisionResponse",
    "MAGI_CANONICAL_USER_BY_MODE",
    "MagiCanonicalUser",
    "MagiDecisionKind",
    "MagiDecisionReceiptResponse",
    "MagiDecisionRequest",
    "MagiMode",
    "ProjectComputerCleanupReceiptResponse",
    "ProjectComputerCleanupRequest",
    "ProjectComputerLifecycle",
    "ProjectComputerMaterializationReceipt",
    "ProjectComputerProvisionRequest",
    "ProjectComputerReplaceRequest",
    "ProjectComputerResponse",
    "ProjectComputerRetirementReceipt",
    "ResearchInternFactoryMembershipResponse",
    "ResearchInternPatchRequest",
    "ResearchInternPolicySet",
    "ResearchInternProvisionRequest",
    "ResearchInternResponse",
    "ResearchInternStatus",
]
