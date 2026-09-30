"""Upload wire contracts mirrored from backend Artifact Platform; schema-parity tested."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

ARTIFACT_CONTRACT_SCHEMA_VERSION = "synth.artifact-platform.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
ArtifactDigest = Annotated[str, StringConstraints(strip_whitespace=True, pattern=SHA256_PATTERN)]
ArtifactUuid = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, pattern=UUID_PATTERN)
]


class ArtifactContract(BaseModel):
    """Base for closed, immutable Artifact Platform boundary payloads."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


ArtifactIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]
ArtifactOwnerNamespace = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=re.compile(r"^[a-z][a-z0-9_.-]{0,127}$"),
    ),
]


class ArtifactTenantKind(StrEnum):
    """Whether an artifact belongs to an organization or the platform."""
    ORG = "org"
    PLATFORM = "platform"


class ArtifactVisibility(StrEnum):
    """Current artifact disclosure audience."""
    PRIVATE = "private"
    ORG = "org"
    PUBLIC = "public"


class ArtifactResourceScope(ArtifactContract):
    """Stored tenancy and product-owner identity used for every decision."""

    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal[ARTIFACT_CONTRACT_SCHEMA_VERSION] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    #: Organization-owned or platform-owned tenancy boundary.
    tenant_kind: ArtifactTenantKind
    #: Organization identifier, required only for organization-owned tenancy.
    org_id: ArtifactIdentifier | None = None
    #: Product-owner namespace responsible for this resource.
    owner_namespace: ArtifactOwnerNamespace
    #: Identifier of the product-owner resource governing access.
    owner_resource_id: ArtifactIdentifier
    #: Current disclosure audience: private, organization or public.
    visibility: ArtifactVisibility
    #: Positive version of the stored authorization policy.
    policy_version: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]

    @model_validator(mode="after")
    def validate_tenant_shape(self) -> ArtifactResourceScope:
        if self.tenant_kind == ArtifactTenantKind.ORG and self.org_id is None:
            raise ValueError("org tenant requires org_id")
        if self.tenant_kind == ArtifactTenantKind.PLATFORM and self.org_id is not None:
            raise ValueError("platform tenant forbids org_id")
        if self.visibility == ArtifactVisibility.ORG and self.tenant_kind != ArtifactTenantKind.ORG:
            raise ValueError("org visibility requires an org tenant")
        return self


class ArtifactCollectionResponse(ArtifactContract):
    """Allocated collection with its tenancy scope and private storage namespace."""
    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal[ARTIFACT_CONTRACT_SCHEMA_VERSION] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    #: UUID of the artifact collection receiving this publication.
    collection_id: ArtifactUuid
    #: Stored tenancy, owner and visibility scope used for authorization.
    scope: ArtifactResourceScope
    #: UUID of the backing storage namespace; not a provider URL or credential.
    storage_namespace_id: ArtifactUuid


class ArtifactObjectDeclaration(ArtifactContract):
    """One immutable manifest object addressed only by logical path and digest."""

    #: Normalized relative manifest path; absolute paths and traversal are refused.
    logical_path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
    ]
    #: Lowercase SHA-256 digest expected for the declared object bytes.
    digest_sha256: ArtifactDigest
    #: Exact declared object size in bytes.
    size_bytes: Annotated[
        StrictInt,
        Field(ge=0, le=9_223_372_036_854_775_807),
    ]
    #: Declared media type of the stored object.
    media_type: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
    ]

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        if value.startswith("/") or value.endswith("/"):
            raise ValueError("logical_path must be a relative file path")
        parts = value.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("logical_path must be normalized and traversal-safe")
        if "\\" in value or "\x00" in value:
            raise ValueError("logical_path contains a forbidden character")
        return value


class ArtifactUploadTarget(ArtifactContract):
    """Create-only upload instructions without a provider storage locator."""

    #: Normalized relative manifest path; absolute paths and traversal are refused.
    logical_path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
    ]
    #: Lowercase SHA-256 digest expected for the declared object bytes.
    digest_sha256: ArtifactDigest
    #: Prepared create-only transfer URL; never include it in public evidence.
    upload_url: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=16_384),
    ]
    #: Headers required for the prepared upload; preserve confidentiality.
    required_headers: dict[str, str] = Field(max_length=16)

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        ArtifactObjectDeclaration(
            logical_path=value,
            digest_sha256="0" * 64,
            size_bytes=0,
            media_type="application/octet-stream",
        )
        return value


class ArtifactPublicationPrepareResponse(ArtifactContract):
    """Prepared create-only object uploads for one publication."""
    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal["synth.artifact-platform.v1"] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    #: UUID of this artifact publication attempt.
    publication_id: ArtifactUuid
    #: UUID of the artifact collection receiving this publication.
    collection_id: ArtifactUuid
    #: Positive artifact publication revision within its collection.
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    #: SHA-256 digest of the exact canonical manifest bytes.
    manifest_digest: ArtifactDigest
    #: Bounded create-only transfer instructions for the declared objects.
    upload_targets: tuple[ArtifactUploadTarget, ...] = Field(max_length=100_000)


class ArtifactPublicationStatus(StrEnum):
    """Storage publication state, separate from Contribution review acceptance."""
    BUILDING = "building"
    COMMITTED = "committed"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class ArtifactPublicationResponse(ArtifactContract):
    """Recorded publication identity and current storage status."""
    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal["synth.artifact-platform.v1"] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    #: UUID of this artifact publication attempt.
    publication_id: ArtifactUuid
    #: UUID of the artifact collection receiving this publication.
    collection_id: ArtifactUuid
    #: Positive artifact publication revision within its collection.
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    #: SHA-256 digest of the exact canonical manifest bytes.
    manifest_digest: ArtifactDigest
    #: Current artifact storage publication state.
    status: ArtifactPublicationStatus


class ArtifactManifest(ArtifactContract):
    """Closed canonical manifest; see Artifact Platform manifest specification."""

    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal[ARTIFACT_CONTRACT_SCHEMA_VERSION] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    #: Schema identifier describing the canonical manifest payload.
    manifest_schema_version: ArtifactIdentifier
    #: UUID of this artifact publication attempt.
    publication_id: ArtifactUuid
    #: UUID of the artifact collection receiving this publication.
    collection_id: ArtifactUuid
    #: Positive artifact publication revision within its collection.
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    #: Immutable object declarations addressed by logical path and digest.
    objects: tuple[ArtifactObjectDeclaration, ...] = Field(max_length=100_000)
    #: SHA-256 digest of the exact canonical manifest bytes.
    manifest_digest: ArtifactDigest

    @model_validator(mode="after")
    def validate_objects(self) -> ArtifactManifest:
        paths = tuple(item.logical_path for item in self.objects)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("manifest objects must have sorted, unique logical paths")
        return self


class ArtifactPublicationPrepare(ArtifactContract):
    """Prepare exact declared objects for one create-only artifact publication."""
    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal[ARTIFACT_CONTRACT_SCHEMA_VERSION] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    #: UUID of this artifact publication attempt.
    publication_id: ArtifactUuid
    #: UUID of the artifact collection receiving this publication.
    collection_id: ArtifactUuid
    #: Positive artifact publication revision within its collection.
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    #: Schema identifier describing the canonical manifest payload.
    manifest_schema_version: ArtifactIdentifier
    #: Immutable object declarations addressed by logical path and digest.
    objects: tuple[ArtifactObjectDeclaration, ...] = Field(max_length=100_000)

    @model_validator(mode="after")
    def validate_unique_logical_paths(self) -> ArtifactPublicationPrepare:
        logical_paths = tuple(item.logical_path for item in self.objects)
        if len(logical_paths) != len(set(logical_paths)):
            raise ValueError("object logical paths must be unique")
        declarations_by_digest: dict[str, tuple[int, str]] = {}
        for item in self.objects:
            declaration_shape = (item.size_bytes, item.media_type)
            previous_shape = declarations_by_digest.setdefault(
                item.digest_sha256,
                declaration_shape,
            )
            if previous_shape != declaration_shape:
                raise ValueError("one object digest cannot declare multiple sizes or media types")
        if sum(item.size_bytes for item in self.objects) > 9_223_372_036_854_775_807:
            raise ValueError("publication size exceeds signed 64-bit persistence limit")
        return self
