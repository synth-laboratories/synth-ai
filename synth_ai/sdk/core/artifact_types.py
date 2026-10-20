"""Project artifacts preserve the platform's exact canonical manifest identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    StrictInt,
    model_serializer,
    model_validator,
)

from synth_ai.core.local_files import open_regular_file
from synth_ai.errors import ContractError
from synth_ai.types.base import (
    Digest,
    Identifier,
    Model,
    ResponseIdentifier,
    ResponseModel,
)
from synth_ai.types.refs import Ref, ReferenceIdentifier
from synth_ai.types.unknown import UnknownValue
from synth_ai.types.uploads import validate_upload_headers

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
ArtifactKind = Literal[
    "file", "report", "answer", "dataset", "log", "figure", "policy", "workspace-snapshot", "other"
]
ArtifactMediaType = Literal[
    "text/plain",
    "text/markdown",
    "application/json",
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/octet-stream",
]
LabelKey = Annotated[
    str, Field(strict=True, min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")
]
LabelValue = Annotated[str, Field(strict=True, max_length=256)]
Labels = Annotated[dict[LabelKey, LabelValue], Field(max_length=32)]
# Observations decode what the server stored: a newer server may carry keys or
# values a request constructor would refuse. Requests keep the strict Labels.
ResponseLabels = dict[
    Annotated[str, Field(strict=True, min_length=1, max_length=512)],
    Annotated[str, Field(strict=True)],
]


class ArtifactDigestMismatch(ContractError):  # noqa: N818 - public name since 0.23.3
    service = "sdk"

    def __init__(self, expected, actual):
        self.expected, self.actual = expected, actual
        super().__init__(
            "Artifact digest verification failed",
            code="artifact_digest_mismatch",
            stage="artifact_verification",
        )


def logical_path(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 1024
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("Artifact path must be a bounded normalized relative path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Artifact path must be a bounded normalized relative path")
    return value


class ArtifactObject(Model):
    logical_path: str = Field(min_length=1, max_length=1024)
    digest_sha256: Digest
    size_bytes: StrictInt = Field(ge=0, le=MAX_ARTIFACT_BYTES)
    media_type: str = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        logical_path(self.logical_path)
        return self


class ArtifactManifest(Model):
    # Deliberately strict (extra="forbid"), unlike its tolerant Artifact envelope:
    # manifest_digest covers every manifest field, so an ignored unknown field
    # would silently make the canonical digest unverifiable. An unknown manifest
    # field must fail closed rather than be dropped.
    schema_version: Literal["synth.artifact-platform.v1"] = "synth.artifact-platform.v1"
    manifest_schema_version: str = Field(min_length=1, max_length=255)
    publication_id: ReferenceIdentifier
    collection_id: ReferenceIdentifier
    revision: StrictInt = Field(ge=1, le=2_147_483_647)
    objects: tuple[ArtifactObject, ...] = Field(min_length=1, max_length=1)
    manifest_digest: Digest

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        unsigned = self.model_dump(mode="json", exclude={"manifest_digest"})
        actual = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual != self.manifest_digest:
            raise ValueError("Artifact manifest differs from its canonical digest")
        return self


class Artifact(ResponseModel):
    artifact_id: ResponseIdentifier
    project_id: ResponseIdentifier
    kind: ArtifactKind | UnknownValue
    labels: ResponseLabels
    labels_revision: StrictInt = Field(ge=1)
    ref: Ref
    manifest: ArtifactManifest
    aliases: tuple[str, ...] = Field(default=(), max_length=200)
    aliases_complete: bool = True
    scan: Literal["accepted", "none"]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.ref != Ref.artifact(
            self.artifact_id, str(self.manifest.revision), self.manifest.manifest_digest
        ):
            raise ValueError("Artifact reference differs from its canonical publication manifest")
        return self


class ArtifactUploadManifest(Model):
    filename: str = Field(min_length=1, max_length=255)
    byte_size: StrictInt = Field(ge=0, le=MAX_ARTIFACT_BYTES)
    digest_sha256: Digest
    media_type: ArtifactMediaType

    @model_validator(mode="after")
    def validate_filename(self) -> Self:
        if (
            self.filename != self.filename.strip()
            or self.filename in {".", ".."}
            or any(character in self.filename for character in "/\\\x00\r\n")
        ):
            raise ValueError("Artifact filename is a label, not a path")
        return self


class ArtifactUploadRequest(Model):
    manifest: ArtifactUploadManifest
    kind: ArtifactKind = "file"
    labels: Labels = Field(default_factory=dict)
    artifact_id: Identifier | None = None

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        if self.artifact_id is None:
            value.pop("artifact_id", None)
        return value


class ArtifactUploadStatus(ResponseModel):
    upload_id: ResponseIdentifier
    project_id: ResponseIdentifier
    artifact_id: ResponseIdentifier
    revision: StrictInt = Field(ge=1, le=2_147_483_647)
    state: Literal["prepared", "received", "finalized", "expired", "refused"]
    manifest: ArtifactUploadManifest
    kind: ArtifactKind | UnknownValue
    transfer_url: str | None = Field(default=None, max_length=4096, repr=False)
    expires_at: AwareDatetime | None = None
    transfer_headers: dict[str, str] = Field(default_factory=dict, max_length=5, repr=False)

    @model_validator(mode="after")
    def validate_delivery(self) -> Self:
        validate_upload_headers(
            self.transfer_headers,
            size=self.manifest.byte_size,
            digest=self.manifest.digest_sha256,
            media_type=self.manifest.media_type,
        )
        if self.state == "prepared" and (self.transfer_url is None or self.expires_at is None):
            raise ValueError("Prepared artifact upload needs an expiring transfer grant")
        return self


class ArtifactUploadReceipt(ResponseModel):
    upload_id: ResponseIdentifier
    artifact: Artifact
    state: Literal["finalized"] = "finalized"


class ArtifactLabelsUpdate(Model):
    set: Labels = Field(default_factory=dict)
    remove: tuple[LabelKey, ...] = Field(default=(), max_length=32)
    expected_revision: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if len(self.remove) != len(set(self.remove)) or self.set.keys() & set(self.remove):
            raise ValueError("Label update keys must be distinct")
        return self


class ArtifactAliasSet(Model):
    ref: Ref
    expected_revision: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.ref.authority != "artifacts" or self.ref.kind != "artifact":
            raise ValueError("Alias must select an exact artifact revision")
        Ref.artifact(self.ref.record_id, self.ref.revision, self.ref.digest_sha256)
        return self


class ArtifactAlias(ResponseModel):
    project_id: ResponseIdentifier
    name: ReferenceIdentifier
    ref: Ref
    revision: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.ref.authority != "artifacts" or self.ref.kind != "artifact":
            raise ValueError("Artifact alias must resolve to an exact artifact revision")
        Ref.artifact(self.ref.record_id, self.ref.revision, self.ref.digest_sha256)
        return self


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    path: Path
    manifest: ArtifactUploadManifest

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        media_type: ArtifactMediaType,
        maximum_bytes: int = MAX_ARTIFACT_BYTES,
        filename: str | None = None,
    ) -> ArtifactSource:
        if type(maximum_bytes) is not int or not 0 <= maximum_bytes <= MAX_ARTIFACT_BYTES:
            raise ValueError("Artifact source bound must be in0..=64MiB")
        digest = hashlib.sha256()
        size = 0
        with open_regular_file(path) as source:
            while chunk := source.read(65536):
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("Artifact source exceeds the selected byte bound")
                digest.update(chunk)
        return cls(
            Path(path),
            ArtifactUploadManifest(
                filename=filename or path.name,
                byte_size=size,
                digest_sha256=digest.hexdigest(),
                media_type=media_type,
            ),
        )


class ArtifactDownloadReceipt(Model):
    ref: Ref
    path: str
    destination: Path
    byte_size: StrictInt = Field(ge=0)
    digest_sha256: Digest
