"""Typed contracts for one organization Research Intern and its resources.

Backend remains the contract authority. These models intentionally preserve
scoped policies and owner-authored evidence instead of flattening them into an
SDK-side source of truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synth_ai.core.research.contracts.dataset_revisions import (
    DatasetRevisionCreateRequest,
    DatasetRevisionLifecycleRequest,
    DatasetRevisionResponse,
)
from synth_ai.core.research.contracts.project_runtime import ProjectComputerState
from synth_ai.core.research.contracts.project_workspace_evidence import (
    ProjectComputerWorkspaceSnapshot,
)


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
        if self.decision_kind is MagiDecisionKind.VERDICT and (
            self.mode is not MagiMode.SERAPH or not self.verdict
        ):
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


ProjectComputerLifecycle = ProjectComputerState


class ProjectComputerProvisionRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    cloud_deployment_id: str = Field(min_length=1, max_length=255)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    workspace_snapshot: ProjectComputerWorkspaceSnapshot | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_snapshot_binding(self) -> ProjectComputerProvisionRequest:
        if (self.snapshot_digest is None) != (self.workspace_snapshot is None):
            raise ValueError("snapshot_digest and workspace_snapshot must be supplied together")
        if self.workspace_snapshot is not None and (
            self.snapshot_digest != self.workspace_snapshot.manifest.manifest_digest
            or self.source_revision != self.workspace_snapshot.commit_sha
        ):
            raise ValueError("Project Computer source does not match its snapshot")
        return self


class ProjectComputerResponse(_StrictContract):
    project_computer_id: str
    org_id: str
    research_intern_id: str
    factory_id: str
    project_id: str
    cloud_deployment_id: str
    adapter_kind: str
    source_repository_id: str
    source_revision: str
    snapshot_digest: str | None = None
    lifecycle: ProjectComputerLifecycle
    generation: int = Field(ge=0)
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ProjectComputerProvisionReceipt(_StrictContract):
    org_id: str
    factory_id: str
    project_id: str
    generation: int
    cloud_deployment_id: str
    adapter_kind: str
    deployment_state: str = Field(min_length=1, max_length=100)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    ready: Literal[True]
    workspace_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    workspace_clean: Literal[True]
    workspace_restore_receipt: dict[str, Any] | None = None
    receipt_ref: str = Field(min_length=1, max_length=1000)


class ProjectComputerRestorationReceipt(_StrictContract):
    org_id: str
    factory_id: str
    project_id: str
    generation: int
    cloud_deployment_id: str
    adapter_kind: str
    deployment_state: str = Field(min_length=1, max_length=100)
    source_repository_id: str
    source_revision: str
    snapshot_digest: str
    restored: Literal[True]
    workspace_restore_receipt: dict[str, Any]
    previous_cloud_deployment_retired: Literal[True]
    previous_retirement_receipt_ref: str = Field(min_length=1, max_length=1000)
    receipt_ref: str = Field(min_length=1, max_length=1000)


class ProjectComputerRetirementReceipt(_StrictContract):
    org_id: str
    factory_id: str
    project_id: str
    generation: int
    cloud_deployment_id: str
    adapter_kind: str
    deployment_state: str = Field(min_length=1, max_length=100)
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
    workspace_materialization_receipts: list[
        ProjectComputerProvisionReceipt | ProjectComputerRestorationReceipt
    ]
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
        if any(receipt.factory_id != self.factory_id for receipt in self.retirement_receipts):
            raise ValueError("retirement receipt crossed its Factory boundary")
        return self


class ProjectComputerReplaceRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    cloud_deployment_id: str = Field(min_length=1, max_length=255)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    workspace_snapshot: ProjectComputerWorkspaceSnapshot
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_snapshot_binding(self) -> ProjectComputerReplaceRequest:
        if (
            self.snapshot_digest != self.workspace_snapshot.manifest.manifest_digest
            or self.source_revision != self.workspace_snapshot.commit_sha
        ):
            raise ValueError("replacement source does not match its snapshot")
        return self


class DataBindingCreateRequest(_StrictContract):
    factory_id: str
    name: str = Field(min_length=1, max_length=255)
    dataset_id: str = Field(min_length=1, max_length=255)
    data_contract_version: str = Field(min_length=1, max_length=255)
    binding_kind: str = Field(min_length=1, max_length=100)
    authority_ref: str = Field(min_length=1, max_length=1000)
    access_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataBindingResponse(_StrictContract):
    data_binding_id: UUID
    org_id: str
    research_intern_id: str
    factory_id: str
    project_id: str
    name: str
    dataset_id: str
    data_contract_version: str
    generation: int = Field(ge=1)
    binding_kind: str
    authority_ref: str
    access_policy: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime


__all__ = [
    "DataBindingCreateRequest",
    "DataBindingResponse",
    "DatasetRevisionCreateRequest",
    "DatasetRevisionLifecycleRequest",
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
    "ProjectComputerProvisionRequest",
    "ProjectComputerProvisionReceipt",
    "ProjectComputerReplaceRequest",
    "ProjectComputerResponse",
    "ProjectComputerRestorationReceipt",
    "ProjectComputerRetirementReceipt",
    "ProjectComputerWorkspaceSnapshot",
    "ResearchInternFactoryMembershipResponse",
    "ResearchInternPatchRequest",
    "ResearchInternPolicySet",
    "ResearchInternProvisionRequest",
    "ResearchInternResponse",
    "ResearchInternStatus",
]
