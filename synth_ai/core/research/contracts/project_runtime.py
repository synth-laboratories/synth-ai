"""Provider-neutral Project Computer operations and Factory storage authority."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_ai.core.research.contracts.traces import (
    TraceStoreLifecycleReceipt,
    validate_optional_sha256_digest,
    validate_sha256_digest,
)


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_wire(cls, value: object):
        return cls.model_validate(value)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class ProjectComputerState(StrEnum):
    REQUESTED = "requested"
    PROVISIONING = "provisioning"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    QUIESCING = "quiescing"
    DEGRADED = "degraded"
    REPLACING = "replacing"
    REMATERIALIZING = "rematerializing"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    RETIRING = "retiring"
    RETIRED = "retired"


class ProjectRuntimeOperation(StrEnum):
    PROVISION = "provision"
    INSPECT = "inspect"
    EXECUTE = "execute"
    SUSPEND = "suspend"
    RESUME = "resume"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"
    REPLACE = "replace"
    RETIRE = "retire"
    RETRY = "retry"
    CANCEL = "cancel"
    ADOPT = "adopt"
    TOMBSTONE = "tombstone"
    EXPORT = "export"
    ROTATE_CREDENTIAL = "rotate_credential"
    REVOKE_CREDENTIAL = "revoke_credential"
    RECONCILE_DELIVERY = "reconcile_delivery"


class ProjectRuntimeOperationOutcome(StrEnum):
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    RECONCILED = "reconciled"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class ProjectRuntimeLifecycleTransitionAuthority(StrEnum):
    CONTROL_PLANE = "control_plane"
    RUNTIME = "runtime"
    RESEARCH_CONTROL = "research_control"
    RECONCILER = "reconciler"
    OPERATOR = "operator"


class ProjectRuntimeFailureClass(StrEnum):
    VALIDATION = "validation"
    POLICY = "policy"
    AUTH = "auth"
    UNAVAILABLE = "unavailable"
    CONFLICT_FENCED = "conflict_fenced"
    INTERRUPTED = "interrupted"
    UNKNOWN_EXTERNAL_OUTCOME = "unknown_external_outcome"
    INTEGRITY = "integrity"
    PROVIDER_TERMINAL = "provider_terminal"


class ProjectRuntimeLifecycleFailure(_StrictContract):
    schema_version: Literal["synth.project-runtime-lifecycle-failure.v1"] = (
        "synth.project-runtime-lifecycle-failure.v1"
    )
    failure_class: ProjectRuntimeFailureClass
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    retryable: bool


class ProjectRuntimeExecutionEvidence(_StrictContract):
    exit_code: int
    stdout_digest: str
    stderr_digest: str
    stdout_bytes: int = Field(ge=0)
    stderr_bytes: int = Field(ge=0)
    stdout_truncated: bool
    stderr_truncated: bool

    _stdout_digest = field_validator("stdout_digest")(validate_sha256_digest)
    _stderr_digest = field_validator("stderr_digest")(validate_sha256_digest)


class ProjectRuntimeOperationReceipt(_StrictContract):
    schema_version: Literal["synth.project-runtime-operation-receipt.v1"] = (
        "synth.project-runtime-operation-receipt.v1"
    )
    receipt_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=1, max_length=512)
    operation: ProjectRuntimeOperation
    authority: ProjectRuntimeLifecycleTransitionAuthority
    resource_id: str = Field(min_length=1, max_length=255)
    resource_generation: int = Field(ge=1)
    outcome: ProjectRuntimeOperationOutcome
    input_digest: str
    output_digest: str | None = None
    execution_evidence: ProjectRuntimeExecutionEvidence | None = None
    failure: ProjectRuntimeLifecycleFailure | None = None
    started_at: datetime
    completed_at: datetime
    receipt_digest: str

    _input_digest = field_validator("input_digest")(validate_sha256_digest)
    _output_digest = field_validator("output_digest")(validate_optional_sha256_digest)
    _receipt_digest = field_validator("receipt_digest")(validate_sha256_digest)

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> ProjectRuntimeOperationReceipt:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("operation receipt timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        failed = self.outcome in {
            ProjectRuntimeOperationOutcome.FAILED,
            ProjectRuntimeOperationOutcome.UNSUPPORTED,
        }
        if failed != (self.failure is not None):
            raise ValueError("failed/unsupported outcomes require exactly one failure")
        if not failed and self.output_digest is None:
            raise ValueError("successful outcomes require output_digest")
        if self.execution_evidence is not None and self.operation is not ProjectRuntimeOperation.EXECUTE:
            raise ValueError("execution_evidence is only valid for execute")
        return self


class ProjectComputerInspectRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_generation: int = Field(ge=1)


class ProjectComputerExecuteRequest(ProjectComputerInspectRequest):
    lease_id: str = Field(min_length=1, max_length=255)
    fencing_token_digest: str
    argv: tuple[str, ...] = Field(min_length=1, max_length=256)
    relative_working_directory: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
    )
    timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    max_output_bytes: int = Field(default=256 * 1024, ge=1, le=1024 * 1024)

    _fencing_token_digest = field_validator("fencing_token_digest")(
        validate_sha256_digest
    )


class ProjectComputerOperationReconcileRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_generation: int = Field(ge=1)


class ProjectComputerLeaseAcquireRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    expected_generation: int = Field(ge=1)
    duration_seconds: int = Field(default=900, ge=30, le=3600)


class ProjectComputerLeaseRenewRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    expected_generation: int = Field(ge=1)
    fencing_token_digest: str
    duration_seconds: int = Field(default=900, ge=30, le=3600)

    _fencing_token_digest = field_validator("fencing_token_digest")(
        validate_sha256_digest
    )


class ProjectComputerLeaseReleaseRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    expected_generation: int = Field(ge=1)
    fencing_token_digest: str

    _fencing_token_digest = field_validator("fencing_token_digest")(
        validate_sha256_digest
    )


class ProjectComputerLeaseResponse(_StrictContract):
    project_computer_id: str
    org_id: str
    factory_id: str
    project_id: str
    generation: int = Field(ge=1)
    lease_id: str
    fencing_token_digest: str
    state: Literal["active", "released"]
    expires_at: datetime | None

    _fencing_token_digest = field_validator("fencing_token_digest")(
        validate_sha256_digest
    )


class FactoryStorageAuthorityRole(StrEnum):
    CONTROL_PLANE = "control_plane"
    IMMUTABLE_OBJECTS = "immutable_objects"
    REBUILDABLE_CATALOG = "rebuildable_catalog"


class FactoryStorageTechnology(StrEnum):
    POSTGRES = "postgres"
    S3 = "s3"
    TURSO = "turso"


_EXPECTED_STORAGE_TECHNOLOGY = {
    FactoryStorageAuthorityRole.CONTROL_PLANE: FactoryStorageTechnology.POSTGRES,
    FactoryStorageAuthorityRole.IMMUTABLE_OBJECTS: FactoryStorageTechnology.S3,
    FactoryStorageAuthorityRole.REBUILDABLE_CATALOG: FactoryStorageTechnology.TURSO,
}


class FactoryStorageAuthorityRef(_StrictContract):
    role: FactoryStorageAuthorityRole
    technology: FactoryStorageTechnology
    opaque_locator_ref: str = Field(min_length=1, max_length=512)
    secret_ref_id: str | None = Field(default=None, min_length=1, max_length=512)
    generation: int = Field(ge=1)
    disposable: bool

    @model_validator(mode="after")
    def validate_authority_boundary(self) -> FactoryStorageAuthorityRef:
        expected = _EXPECTED_STORAGE_TECHNOLOGY[self.role]
        if self.technology is not expected:
            raise ValueError(f"{self.role.value} must use {expected.value}")
        if self.role is FactoryStorageAuthorityRole.REBUILDABLE_CATALOG:
            if not self.disposable:
                raise ValueError("Turso catalog authority must be disposable")
        elif self.disposable:
            raise ValueError("Postgres/S3 authority cannot be disposable")
        locator = self.opaque_locator_ref.lower()
        if "://" in locator or any(
            marker in locator for marker in ("password=", "token=", "secret=", "apikey=")
        ):
            raise ValueError("Factory storage locator leaked a DSN or credential")
        return self


class FactoryStorageAuthorityDescriptor(_StrictContract):
    schema_version: Literal["synth.factory-storage-authority.v1"] = (
        "synth.factory-storage-authority.v1"
    )
    descriptor_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    generation: int = Field(ge=1)
    authorities: list[FactoryStorageAuthorityRef] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_cardinality(self) -> FactoryStorageAuthorityDescriptor:
        roles = [authority.role for authority in self.authorities]
        if len(set(roles)) != 3 or set(roles) != set(FactoryStorageAuthorityRole):
            raise ValueError("storage descriptor requires exactly one authority per role")
        return self


def _descriptor_digest(descriptor: FactoryStorageAuthorityDescriptor) -> str:
    encoded = json.dumps(
        descriptor.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class FactoryStorageAuthorityResponse(_StrictContract):
    descriptor: FactoryStorageAuthorityDescriptor
    descriptor_digest: str
    lifecycle_receipt: TraceStoreLifecycleReceipt

    _descriptor_digest_validator = field_validator("descriptor_digest")(
        validate_sha256_digest
    )

    @model_validator(mode="after")
    def validate_lifecycle_join(self) -> FactoryStorageAuthorityResponse:
        descriptor = self.descriptor
        receipt = self.lifecycle_receipt
        if self.descriptor_digest != _descriptor_digest(descriptor):
            raise ValueError("Factory storage descriptor digest does not verify")
        if (
            receipt.action != "provision"
            or receipt.org_id != descriptor.org_id
            or receipt.factory_id != descriptor.factory_id
            or receipt.generation != descriptor.generation
            or receipt.details.get("storage_authority_descriptor_id")
            != descriptor.descriptor_id
            or receipt.details.get("storage_authority_descriptor_digest")
            != self.descriptor_digest
        ):
            raise ValueError("Factory storage descriptor does not join its owner receipt")
        return self


__all__ = [
    "FactoryStorageAuthorityDescriptor",
    "FactoryStorageAuthorityRef",
    "FactoryStorageAuthorityResponse",
    "FactoryStorageAuthorityRole",
    "FactoryStorageTechnology",
    "ProjectComputerExecuteRequest",
    "ProjectComputerInspectRequest",
    "ProjectComputerLeaseAcquireRequest",
    "ProjectComputerLeaseReleaseRequest",
    "ProjectComputerLeaseRenewRequest",
    "ProjectComputerLeaseResponse",
    "ProjectComputerOperationReconcileRequest",
    "ProjectComputerState",
    "ProjectRuntimeExecutionEvidence",
    "ProjectRuntimeFailureClass",
    "ProjectRuntimeLifecycleFailure",
    "ProjectRuntimeLifecycleTransitionAuthority",
    "ProjectRuntimeOperation",
    "ProjectRuntimeOperationOutcome",
    "ProjectRuntimeOperationReceipt",
]
