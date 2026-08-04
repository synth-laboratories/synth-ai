"""Factory-scoped managed Trace V5 transport contracts.

Trace and evidence document schemas are backend-authored; these models cover
only the backend storage, publication, query, receipt, and download protocol.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_ai.core.contracts.json_value import JsonObject, JsonValue


class _TraceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_wire(cls, value: JsonValue) -> Any:
        return cls.model_validate(value)

    def to_wire(self) -> JsonObject:
        return self.model_dump(mode="json")


class TraceCatalogProvider(StrEnum):
    NONE = "none"
    TURSO = "turso"


class TraceStoreProvisioningStatus(StrEnum):
    READY = "ready"
    DELETING = "deleting"
    CREDENTIAL_INVALIDATED = "credential_invalidated"
    ERROR = "error"


class TraceBundlePublicationStatus(StrEnum):
    PENDING = "pending"
    APPLYING = "applying"
    COMMITTED = "committed"
    FAILED = "failed"


class TraceBundleObjectKind(StrEnum):
    TRACE = "trace"
    EVIDENCE = "evidence"
    BLOB = "blob"
    BINDING = "binding"
    SEGMENT = "segment"
    PROJECTION = "projection"
    RECEIPT = "receipt"


def validate_sha256_digest(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized.startswith("sha256:") or len(normalized) != 71:
        raise ValueError("digest must be sha256:<64 lowercase hexadecimal characters>")
    try:
        int(normalized[7:], 16)
    except ValueError as error:
        raise ValueError("digest contains non-hexadecimal characters") from error
    return normalized


def validate_optional_sha256_digest(value: str | None) -> str | None:
    return None if value is None else validate_sha256_digest(value)


def validate_sha256_digest_list(values: list[str]) -> list[str]:
    return [validate_sha256_digest(value) for value in values]


class TraceStoreDescriptor(_TraceContract):
    store_id: str
    org_id: str
    factory_id: str
    store_schema_version: str
    bundle_schema_version: str
    blob_provider: str
    blob_bucket: str
    blob_prefix: str
    blob_region: str | None = None
    blob_endpoint_profile: str | None = None
    catalog_provider: TraceCatalogProvider
    catalog_database_name: str | None = None
    catalog_group_name: str | None = None
    catalog_region: str | None = None
    provisioning_status: TraceStoreProvisioningStatus
    provisioning_version: int
    migration_generation: int
    retention_policy: dict[str, Any] = Field(default_factory=dict)
    visibility_policy: dict[str, Any] = Field(default_factory=dict)
    encryption_policy: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TraceBundleObjectDeclaration(_TraceContract):
    digest: str
    path: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(default="application/octet-stream", min_length=1)
    kind: TraceBundleObjectKind

    _digest = field_validator("digest")(validate_sha256_digest)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("path must be a non-empty bundle-relative path")
        return normalized


class TraceBundleSourceArtifactRef(_TraceContract):
    artifact_id: str = Field(min_length=1, max_length=255)
    locator_ref: str = Field(min_length=1, max_length=1024)
    archive_digest: str
    size_bytes: int = Field(ge=1)

    _archive_digest = field_validator("archive_digest")(validate_sha256_digest)


class TraceExperimentRevisionRef(_TraceContract):
    experiment_id: str = Field(min_length=1, max_length=255)
    revision: int = Field(ge=1)
    experiment_run_id: str | None = Field(default=None, min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=128)


class TraceEvidenceAuthorityRef(_TraceContract):
    authority_kind: str = Field(min_length=1, max_length=128)
    authority_id: str = Field(min_length=1, max_length=255)
    authority_version: str | None = Field(default=None, min_length=1, max_length=255)
    locator_ref: str = Field(min_length=1, max_length=1024)
    content_digest: str

    _content_digest = field_validator("content_digest")(validate_sha256_digest)


class TraceBundlePublicationLineage(_TraceContract):
    source_artifact: TraceBundleSourceArtifactRef
    task_id: str = Field(min_length=1, max_length=255)
    task_key: str = Field(min_length=1, max_length=255)
    capture_id: str = Field(min_length=1, max_length=255)
    trace_id: str = Field(min_length=1, max_length=255)
    actor_id: str = Field(min_length=1, max_length=255)
    actor_session_id: str = Field(min_length=1, max_length=255)
    turn_id: str | None = Field(default=None, min_length=1, max_length=255)
    thread_id: str | None = Field(default=None, min_length=1, max_length=255)
    experiment_revisions: list[TraceExperimentRevisionRef] = Field(default_factory=list)
    evidence_authorities: list[TraceEvidenceAuthorityRef] = Field(default_factory=list)


class TraceBundlePrepareRequest(_TraceContract):
    bundle_id: str = Field(min_length=1, max_length=255)
    manifest_digest: str
    manifest: dict[str, Any]
    objects: list[TraceBundleObjectDeclaration] = Field(default_factory=list)
    project_id: str | None = None
    run_id: str | None = None
    effort_id: str | None = None
    lineage: TraceBundlePublicationLineage | None = None
    experiment_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)


class TraceObjectUpload(_TraceContract):
    digest: str
    s3_uri: str
    upload_url: str | None = None
    required_headers: dict[str, str] = Field(default_factory=dict)
    already_present: bool = False

    _digest = field_validator("digest")(validate_sha256_digest)


class TraceBundlePublication(_TraceContract):
    publication_id: str
    store_id: str
    factory_id: str
    bundle_id: str
    manifest_digest: str
    manifest_uri: str
    status: TraceBundlePublicationStatus
    upload_objects: list[TraceObjectUpload] = Field(default_factory=list)
    trace_count: int = 0
    evidence_count: int = 0
    receipt_uri: str | None = None
    receipt_digest: str | None = None
    failure_code: str | None = None
    created_at: datetime
    committed_at: datetime | None = None

    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)
    _receipt_digest = field_validator("receipt_digest")(validate_optional_sha256_digest)


class TracePromotionReceipt(_TraceContract):
    schema_version: str = "synth.trace-promotion-receipt.v1"
    publication_id: str
    store_id: str
    factory_id: str
    bundle_id: str
    manifest_digest: str
    object_digests: list[str]
    trace_digests: list[str]
    evidence_digests: list[str]
    experiment_id: str | None = None
    experiment_revision: TraceExperimentRevisionRef | None = None
    catalog_provider: TraceCatalogProvider
    catalog_generation: int
    committed_at: datetime
    receipt_digest: str
    receipt_uri: str

    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)
    _object_digests = field_validator("object_digests")(validate_sha256_digest_list)
    _trace_digests = field_validator("trace_digests")(validate_sha256_digest_list)
    _evidence_digests = field_validator("evidence_digests")(validate_sha256_digest_list)
    _receipt_digest = field_validator("receipt_digest")(validate_sha256_digest)

    @model_validator(mode="after")
    def validate_experiment_revision_identity(self) -> TracePromotionReceipt:
        if self.experiment_id is None and self.experiment_revision is None:
            return self
        if self.experiment_id is None or self.experiment_revision is None:
            raise ValueError("experiment_id and experiment_revision must be supplied together")
        if self.experiment_revision.experiment_id != self.experiment_id:
            raise ValueError("experiment_revision does not cite the receipt experiment")
        if self.experiment_revision.experiment_run_id is None:
            raise ValueError("experiment_revision requires experiment_run_id")
        return self


class TraceRecordSummary(_TraceContract):
    trace_id: str
    trace_digest: str
    trace_kind: str
    schema_version: str
    lifecycle_status: str
    capture_status: str
    capture_id: str
    started_at: str | None = None
    ended_at: str | None = None
    actor_count: int = 0
    span_count: int = 0
    event_count: int = 0
    message_count: int = 0
    artifact_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    task_id: str | None = None
    run_id: str | None = None
    correlation_id: str | None = None
    publication_id: str
    manifest_digest: str

    _trace_digest = field_validator("trace_digest")(validate_sha256_digest)
    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)


class TraceStoreAccessReceipt(_TraceContract):
    schema_version: str = "synth.trace-store-access-receipt.v1"
    action: str
    store_id: str
    factory_id: str
    org_id: str
    principal_id: str | None = None
    request_digest: str
    result_digest: str
    occurred_at: datetime
    content_digest: str
    receipt_uri: str
    details: dict[str, Any] = Field(default_factory=dict)

    _request_digest = field_validator("request_digest")(validate_sha256_digest)
    _result_digest = field_validator("result_digest")(validate_sha256_digest)
    _content_digest = field_validator("content_digest")(validate_sha256_digest)


class TraceQueryResult(_TraceContract):
    factory_id: str
    count: int
    traces: list[TraceRecordSummary] = Field(default_factory=list)
    receipt: TraceStoreAccessReceipt


class TraceDownload(_TraceContract):
    trace_digest: str
    bytes_digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    s3_uri: str
    download_url: str
    expires_at: datetime
    receipt: TraceStoreAccessReceipt

    _trace_digest = field_validator("trace_digest")(validate_sha256_digest)
    _bytes_digest = field_validator("bytes_digest")(validate_sha256_digest)


class TraceBundleDownloadObject(_TraceContract):
    path: str
    digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    kind: TraceBundleObjectKind
    download_url: str
    expires_at: datetime

    _digest = field_validator("digest")(validate_sha256_digest)


class TraceBundleDownload(_TraceContract):
    publication_id: str
    bundle_id: str
    manifest_digest: str
    manifest: dict[str, Any]
    objects: list[TraceBundleDownloadObject]
    receipt: TraceStoreAccessReceipt

    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)


class TraceStoreLifecycleReceipt(_TraceContract):
    schema_version: str = "synth.trace-store-lifecycle-receipt.v1"
    action: str
    store_id: str
    factory_id: str
    org_id: str
    principal_id: str | None = None
    generation: int
    occurred_at: datetime
    content_digest: str
    receipt_uri: str
    details: dict[str, Any] = Field(default_factory=dict)

    _content_digest = field_validator("content_digest")(validate_sha256_digest)


class TraceStoreProvisionResult(_TraceContract):
    descriptor: TraceStoreDescriptor
    receipt: TraceStoreLifecycleReceipt


# The preflight trio keeps the backend contract names verbatim (unlike the
# older abridged names above) because the response and identity block are
# schema-versioned envelopes that sealed runners embed and re-verify by name.


class TraceStorePreflightRequest(_TraceContract):
    """Pre-launch trace-store readiness gate for a sealed run."""

    catalog_provider: TraceCatalogProvider = TraceCatalogProvider.NONE
    provision_if_missing: bool = True


class TraceStoreRunEnvelopeIdentity(_TraceContract):
    """Trace-store identity carried inside a sealed-run envelope.

    Sealed runners embed exactly this block (as ``trace_store``) in the run
    envelope so trace-inventory collection after grading can never 404 on an
    unprovisioned or ambiguous store.
    """

    schema_version: Literal["synth.trace-store-run-envelope-identity.v1"]
    trace_store_id: str
    org_id: str
    factory_id: str
    blob_bucket: str
    blob_prefix: str
    catalog_provider: TraceCatalogProvider
    catalog_database_name: str | None = None
    provisioning_status: TraceStoreProvisioningStatus
    provisioning_version: int


class TraceStorePreflightResponse(_TraceContract):
    """Typed provision-and-health result gating sealed-run launch."""

    schema_version: Literal["synth.trace-store-preflight.v1"]
    healthy: bool
    provisioned: Literal["existing", "created"]
    descriptor: TraceStoreDescriptor
    run_envelope_identity: TraceStoreRunEnvelopeIdentity
    health_conditions: list[str] = Field(default_factory=list)
    checked_at: datetime


__all__ = [
    "TraceBundleDownload",
    "TraceBundleDownloadObject",
    "TraceBundleObjectDeclaration",
    "TraceBundleObjectKind",
    "TraceBundlePrepareRequest",
    "TraceBundlePublicationLineage",
    "TraceBundlePublication",
    "TraceBundlePublicationStatus",
    "TraceBundleSourceArtifactRef",
    "TraceCatalogProvider",
    "TraceDownload",
    "TraceEvidenceAuthorityRef",
    "TraceExperimentRevisionRef",
    "TraceObjectUpload",
    "TracePromotionReceipt",
    "TraceQueryResult",
    "TraceRecordSummary",
    "TraceStoreAccessReceipt",
    "TraceStoreDescriptor",
    "TraceStoreLifecycleReceipt",
    "TraceStorePreflightRequest",
    "TraceStorePreflightResponse",
    "TraceStoreProvisionResult",
    "TraceStoreProvisioningStatus",
    "TraceStoreRunEnvelopeIdentity",
    "validate_optional_sha256_digest",
    "validate_sha256_digest",
    "validate_sha256_digest_list",
]
