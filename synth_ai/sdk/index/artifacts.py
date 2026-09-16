"""Upload wire contracts mirrored from backend Artifact Platform; schema-parity tested."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, field_validator

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


class ArtifactObjectDeclaration(ArtifactContract):
    """One immutable manifest object addressed only by logical path and digest."""

    logical_path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
    ]
    digest_sha256: ArtifactDigest
    size_bytes: Annotated[
        StrictInt,
        Field(ge=0, le=9_223_372_036_854_775_807),
    ]
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

    logical_path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
    ]
    digest_sha256: ArtifactDigest
    upload_url: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=16_384),
    ]
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
    schema_version: Literal["synth.artifact-platform.v1"] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    publication_id: ArtifactUuid
    collection_id: ArtifactUuid
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    manifest_digest: ArtifactDigest
    upload_targets: tuple[ArtifactUploadTarget, ...] = Field(max_length=100_000)


class ArtifactPublicationStatus(StrEnum):
    BUILDING = "building"
    COMMITTED = "committed"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class ArtifactPublicationFinalize(ArtifactContract):
    """Finalize request: names the prepared publication whose bytes to verify."""

    publication_id: ArtifactUuid
    schema_version: Literal["synth.artifact-platform.v1"] = ARTIFACT_CONTRACT_SCHEMA_VERSION


class ArtifactPublicationResponse(ArtifactContract):
    schema_version: Literal["synth.artifact-platform.v1"] = ARTIFACT_CONTRACT_SCHEMA_VERSION
    publication_id: ArtifactUuid
    collection_id: ArtifactUuid
    revision: Annotated[StrictInt, Field(ge=1, le=2_147_483_647)]
    manifest_digest: ArtifactDigest
    status: ArtifactPublicationStatus
