"""Immutable DatasetRevision and server-owned publication contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_ai.core.research.contracts.traces import validate_sha256_digest


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_wire(cls, value: object):
        return cls.model_validate(value)

    def to_wire(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_none=True)


class DatasetRevisionState(StrEnum):
    BUILDING = "building"
    SEALED = "sealed"
    REVOKED = "revoked"
    ABANDONED = "abandoned"
    TOMBSTONED = "tombstoned"


class DatasetRevisionSourceKind(StrEnum):
    ARTIFACT = "artifact"
    DATASET_REVISION = "dataset_revision"
    INTERNAL_GIT_REVISION = "internal_git_revision"
    UPLOAD = "upload"


class DatasetRevisionSourceRef(_FrozenContract):
    kind: DatasetRevisionSourceKind
    authority_id: str = Field(min_length=1, max_length=255)
    authority_version: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    content_digest: str

    _content_digest = field_validator("content_digest")(validate_sha256_digest)


class DatasetRevisionParentRef(_FrozenContract):
    dataset_revision_id: UUID
    binding_id: UUID
    binding_generation: int = Field(ge=1)
    revision_number: int = Field(ge=1)
    dataset_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    content_digest: str
    revision_digest: str

    _content_digest = field_validator("content_digest")(validate_sha256_digest)
    _revision_digest = field_validator("revision_digest")(validate_sha256_digest)


class DatasetRevisionLineage(_FrozenContract):
    parent: DatasetRevisionParentRef | None = None
    sources: tuple[DatasetRevisionSourceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> DatasetRevisionLineage:
        identities = [
            (source.kind, source.authority_id, source.authority_version)
            for source in self.sources
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("DatasetRevision source identities must be unique")
        return self


class DatasetRevisionContent(_FrozenContract):
    logical_path: str = Field(min_length=1, max_length=1024)
    content_digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=255)
    object_schema_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    _content_digest = field_validator("content_digest")(validate_sha256_digest)

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("logical_path must be revision-relative")
        return normalized


class DatasetRevisionDraft(_FrozenContract):
    schema_version: Literal["synth.dataset-revision-draft.v1"] = (
        "synth.dataset-revision-draft.v1"
    )
    state: Literal[DatasetRevisionState.BUILDING] = DatasetRevisionState.BUILDING
    dataset_revision_id: UUID
    revision_number: int = Field(ge=1)
    binding_id: UUID
    binding_generation: int = Field(ge=1)
    dataset_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    dataset_schema_version: str = Field(min_length=1, max_length=255)
    lineage: DatasetRevisionLineage
    contents: tuple[DatasetRevisionContent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope(self) -> DatasetRevisionDraft:
        scope = (self.org_id, self.factory_id, self.project_id)
        if any(
            (source.org_id, source.factory_id, source.project_id) != scope
            for source in self.lineage.sources
        ):
            raise ValueError("DatasetRevision source crossed its requested scope")
        parent = self.lineage.parent
        if parent is not None and (
            (parent.org_id, parent.factory_id, parent.project_id) != scope
            or parent.dataset_id != self.dataset_id
            or parent.binding_id != self.binding_id
            or parent.binding_generation != self.binding_generation
            or parent.revision_number >= self.revision_number
        ):
            raise ValueError("DatasetRevision parent does not precede the same binding")
        paths = [content.logical_path for content in self.contents]
        if len(paths) != len(set(paths)):
            raise ValueError("DatasetRevision logical paths must be unique")
        return self


class DatasetRevisionManifestReceipt(_FrozenContract):
    schema_version: Literal["synth.dataset-revision-manifest-receipt.v1"] = (
        "synth.dataset-revision-manifest-receipt.v1"
    )
    authority: Literal["factory_immutable_object_store"] = (
        "factory_immutable_object_store"
    )
    receipt_id: UUID
    authority_generation: int = Field(ge=1)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    manifest_uri: str = Field(min_length=1, max_length=2000)
    manifest_digest: str
    content_digest: str
    object_count: int = Field(ge=1)
    total_size_bytes: int = Field(ge=0)
    issued_at: datetime
    receipt_digest: str

    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)
    _content_digest = field_validator("content_digest")(validate_sha256_digest)
    _receipt_digest = field_validator("receipt_digest")(validate_sha256_digest)


class DatasetRevisionCreateRequest(_FrozenContract):
    draft: DatasetRevisionDraft
    manifest_receipt: DatasetRevisionManifestReceipt
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_receipt_join(self) -> DatasetRevisionCreateRequest:
        draft = self.draft
        receipt = self.manifest_receipt
        if (
            receipt.authority_generation != draft.binding_generation
            or (receipt.org_id, receipt.factory_id, receipt.project_id)
            != (draft.org_id, draft.factory_id, draft.project_id)
            or receipt.object_count != len(draft.contents)
            or receipt.total_size_bytes
            != sum(content.size_bytes for content in draft.contents)
        ):
            raise ValueError("manifest receipt does not join the DatasetRevision draft")
        return self


class SealedDatasetRevision(_FrozenContract):
    schema_version: Literal["synth.dataset-revision.v1"] = "synth.dataset-revision.v1"
    state: Literal[DatasetRevisionState.SEALED] = DatasetRevisionState.SEALED
    dataset_revision_id: UUID
    revision_number: int = Field(ge=1)
    binding_id: UUID
    binding_generation: int = Field(ge=1)
    dataset_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    dataset_schema_version: str = Field(min_length=1, max_length=255)
    lineage: DatasetRevisionLineage
    contents: tuple[DatasetRevisionContent, ...] = Field(min_length=1)
    content_digest: str
    manifest_digest: str
    storage_receipt_id: UUID
    storage_receipt_digest: str
    revision_digest: str
    sealed_at: datetime
    sealed_by_principal_id: str = Field(min_length=1, max_length=255)

    _content_digest = field_validator("content_digest")(validate_sha256_digest)
    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)
    _storage_receipt_digest = field_validator("storage_receipt_digest")(
        validate_sha256_digest
    )
    _revision_digest = field_validator("revision_digest")(validate_sha256_digest)


class DatasetRevisionVerificationReceipt(_FrozenContract):
    schema_version: Literal["synth.dataset-revision-verification-receipt.v1"] = (
        "synth.dataset-revision-verification-receipt.v1"
    )
    verified: Literal[True] = True
    dataset_revision_id: UUID
    binding_id: UUID
    binding_generation: int = Field(ge=1)
    org_id: str
    factory_id: str
    project_id: str
    content_digest: str
    manifest_digest: str
    storage_receipt_id: UUID
    storage_receipt_digest: str
    revision_digest: str

    _content_digest = field_validator("content_digest")(validate_sha256_digest)
    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)
    _storage_receipt_digest = field_validator("storage_receipt_digest")(
        validate_sha256_digest
    )
    _revision_digest = field_validator("revision_digest")(validate_sha256_digest)


class DatasetRevisionLifecycleRequest(_FrozenContract):
    expected_state: DatasetRevisionState
    target_state: DatasetRevisionState
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_target(self) -> DatasetRevisionLifecycleRequest:
        if self.target_state not in {
            DatasetRevisionState.REVOKED,
            DatasetRevisionState.TOMBSTONED,
        }:
            raise ValueError("lifecycle target must be revoked or tombstoned")
        return self


class DatasetRevisionResponse(_FrozenContract):
    dataset_revision_id: UUID
    data_binding_id: UUID
    org_id: str
    factory_id: str
    project_id: str
    binding_generation: int = Field(ge=1)
    revision_number: int = Field(ge=1)
    dataset_id: str
    content_digest: str
    manifest_digest: str
    storage_receipt_id: UUID
    storage_receipt_digest: str
    revision_digest: str
    manifest_uri: str
    schema_version: str
    parent_revision_id: UUID | None = None
    state: DatasetRevisionState
    lifecycle_generation: int = Field(ge=1)
    lifecycle_updated_at: datetime
    revoked_at: datetime | None = None
    tombstoned_at: datetime | None = None
    sealed_revision: SealedDatasetRevision
    manifest_receipt: DatasetRevisionManifestReceipt
    metadata: dict[str, object]
    created_at: datetime

    _content_digest = field_validator("content_digest")(validate_sha256_digest)
    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)
    _storage_receipt_digest = field_validator("storage_receipt_digest")(
        validate_sha256_digest
    )
    _revision_digest = field_validator("revision_digest")(validate_sha256_digest)

    @model_validator(mode="after")
    def validate_sealed_join(self) -> DatasetRevisionResponse:
        sealed = self.sealed_revision
        receipt = self.manifest_receipt
        if (
            sealed.dataset_revision_id != self.dataset_revision_id
            or sealed.binding_id != self.data_binding_id
            or sealed.binding_generation != self.binding_generation
            or (sealed.org_id, sealed.factory_id, sealed.project_id)
            != (self.org_id, self.factory_id, self.project_id)
            or sealed.revision_digest != self.revision_digest
            or sealed.content_digest != self.content_digest
            or sealed.manifest_digest != self.manifest_digest
            or sealed.storage_receipt_id != self.storage_receipt_id
            or sealed.storage_receipt_digest != self.storage_receipt_digest
            or receipt.authority_generation != self.binding_generation
            or (receipt.org_id, receipt.factory_id, receipt.project_id)
            != (self.org_id, self.factory_id, self.project_id)
            or receipt.manifest_digest != self.manifest_digest
            or receipt.content_digest != self.content_digest
        ):
            raise ValueError("DatasetRevision response evidence join drifted")
        return self


class DatasetRevisionPreparationState(StrEnum):
    PREPARED = "prepared"
    FINALIZED = "finalized"
    FAILED = "failed"


class DatasetRevisionPrepareRequest(_FrozenContract):
    schema_version: Literal["synth.dataset-revision-prepare.v1"] = (
        "synth.dataset-revision-prepare.v1"
    )
    idempotency_key: str = Field(min_length=1, max_length=255)
    draft: DatasetRevisionDraft
    upload_expires_in_seconds: int = Field(default=900, ge=60, le=3600)


class DatasetRevisionObjectUploadTarget(_FrozenContract):
    logical_path: str
    content_digest: str
    size_bytes: int = Field(ge=0)
    media_type: str
    object_uri: str
    upload_url: str
    required_headers: dict[str, str]

    _content_digest = field_validator("content_digest")(validate_sha256_digest)


class DatasetRevisionPreparationResponse(_FrozenContract):
    schema_version: Literal["synth.dataset-revision-preparation.v1"] = (
        "synth.dataset-revision-preparation.v1"
    )
    preparation_id: UUID
    idempotency_key: str
    request_digest: str
    state: DatasetRevisionPreparationState
    org_id: str
    factory_id: str
    project_id: str
    data_binding_id: UUID
    dataset_revision_id: UUID
    object_uploads: tuple[DatasetRevisionObjectUploadTarget, ...]

    _request_digest = field_validator("request_digest")(validate_sha256_digest)


class DatasetRevisionFinalizeRequest(_FrozenContract):
    schema_version: Literal["synth.dataset-revision-finalize.v1"] = (
        "synth.dataset-revision-finalize.v1"
    )
    idempotency_key: str = Field(min_length=1, max_length=255)


class DatasetRevisionFinalizeResponse(_FrozenContract):
    schema_version: Literal["synth.dataset-revision-finalize.v1"] = (
        "synth.dataset-revision-finalize.v1"
    )
    preparation_id: UUID
    state: Literal[DatasetRevisionPreparationState.FINALIZED] = (
        DatasetRevisionPreparationState.FINALIZED
    )
    sealed_revision: SealedDatasetRevision
    manifest_receipt: DatasetRevisionManifestReceipt
    verification_receipt: DatasetRevisionVerificationReceipt

    @model_validator(mode="after")
    def validate_receipt_join(self) -> DatasetRevisionFinalizeResponse:
        sealed = self.sealed_revision
        manifest = self.manifest_receipt
        verification = self.verification_receipt
        identity = (
            sealed.dataset_revision_id,
            sealed.binding_id,
            sealed.binding_generation,
            sealed.org_id,
            sealed.factory_id,
            sealed.project_id,
            sealed.content_digest,
            sealed.manifest_digest,
            sealed.storage_receipt_id,
            sealed.storage_receipt_digest,
            sealed.revision_digest,
        )
        verified = (
            verification.dataset_revision_id,
            verification.binding_id,
            verification.binding_generation,
            verification.org_id,
            verification.factory_id,
            verification.project_id,
            verification.content_digest,
            verification.manifest_digest,
            verification.storage_receipt_id,
            verification.storage_receipt_digest,
            verification.revision_digest,
        )
        if identity != verified:
            raise ValueError("finalization verification receipt drifted")
        if (
            manifest.authority_generation != sealed.binding_generation
            or (manifest.org_id, manifest.factory_id, manifest.project_id)
            != (sealed.org_id, sealed.factory_id, sealed.project_id)
            or manifest.content_digest != sealed.content_digest
            or manifest.manifest_digest != sealed.manifest_digest
        ):
            raise ValueError("finalization manifest receipt drifted")
        return self


__all__ = [
    "DatasetRevisionContent",
    "DatasetRevisionCreateRequest",
    "DatasetRevisionDraft",
    "DatasetRevisionFinalizeRequest",
    "DatasetRevisionFinalizeResponse",
    "DatasetRevisionLifecycleRequest",
    "DatasetRevisionLineage",
    "DatasetRevisionManifestReceipt",
    "DatasetRevisionObjectUploadTarget",
    "DatasetRevisionParentRef",
    "DatasetRevisionPreparationResponse",
    "DatasetRevisionPreparationState",
    "DatasetRevisionPrepareRequest",
    "DatasetRevisionResponse",
    "DatasetRevisionSourceKind",
    "DatasetRevisionSourceRef",
    "DatasetRevisionState",
    "DatasetRevisionVerificationReceipt",
    "SealedDatasetRevision",
]
