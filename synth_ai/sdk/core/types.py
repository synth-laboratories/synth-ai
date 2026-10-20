"""Repository observations and explicit clone/transfer bounds (core.v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    StrictInt,
    model_serializer,
    model_validator,
)

from synth_ai.types.base import Digest, Model, ResponseBase, ResponseIdentifier, ResponseModel
from synth_ai.types.refs import Ref, ReferenceIdentifier

GitSHA = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{40}$")]
GitMode = Literal["000000", "100644", "100755", "120000", "160000"]


def commit_ref(value: Ref, repo_id: str, sha: str | None = None) -> Ref:
    """A generic Forge reference becomes a Git commit only after owner checks."""
    from pydantic import TypeAdapter

    if value.authority != "git" or value.kind != "commit" or value.record_id != repo_id:
        raise ValueError("Expected an exact commit reference belonging to this repository")
    TypeAdapter(GitSHA).validate_python(value.revision)
    if sha is not None and value.revision != sha:
        raise ValueError("Commit reference revision does not match the selected SHA")
    return value


class RepositorySnapshot(ResponseModel):
    project_id: ResponseIdentifier
    repo_id: ReferenceIdentifier
    default_branch: str
    ref: Ref | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.ref is not None:
            commit_ref(self.ref, self.repo_id)
        return self

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        if self.ref is None:
            value.pop("ref", None)
        return value


class RepositoryRef(ResponseBase):
    name: str
    commit_sha: GitSHA
    ref: Ref

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        commit_ref(self.ref, self.ref.record_id, self.commit_sha)
        return self


class RepositoryCommit(ResponseModel):
    project_id: ResponseIdentifier
    repo_id: ReferenceIdentifier
    ref: Ref
    sha: GitSHA
    tree_sha: GitSHA
    parent_shas: tuple[GitSHA, ...]
    author_name: str
    author_email: str
    authored_at: AwareDatetime
    summary: str
    manifest_digest_sha256: Digest
    entries: StrictInt = Field(ge=0, le=100_000)
    bytes: StrictInt = Field(ge=0, le=2 * 1024 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        commit_ref(self.ref, self.repo_id, self.sha)
        if self.ref.digest_sha256 != self.manifest_digest_sha256:
            raise ValueError("Commit reference digest differs from its canonical manifest")
        return self


class RepositoryDiffFile(ResponseBase):
    path: str
    old_path: str | None = None
    old_mode: GitMode
    new_mode: GitMode
    status: str
    additions: StrictInt | None = Field(default=None, ge=0)
    deletions: StrictInt | None = Field(default=None, ge=0)

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


class RepositoryDiff(ResponseModel):
    project_id: ResponseIdentifier
    repo_id: ReferenceIdentifier
    base: Ref
    head: Ref
    files: tuple[RepositoryDiffFile, ...]
    patch: str | None = None
    truncated: bool

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        commit_ref(self.base, self.repo_id)
        commit_ref(self.head, self.repo_id)
        return self

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        if self.patch is None:
            value.pop("patch", None)
        return value


class CloneTokenRequest(Model):
    ttl_seconds: StrictInt = Field(default=300, ge=1, le=600)


class CloneToken(ResponseModel):
    project_id: ResponseIdentifier
    repo_id: ReferenceIdentifier
    clone_url: str
    auth_header: str = Field(repr=False, min_length=8, max_length=8192)
    expires_at: AwareDatetime
    scope: Literal["project"]

    @model_validator(mode="after")
    def validate_delivery(self) -> Self:
        url = urlsplit(self.clone_url)
        if (
            not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not (
                url.scheme == "https"
                or (url.scheme == "http" and url.hostname in {"127.0.0.1", "::1", "localhost"})
            )
        ):
            raise ValueError("Clone URL must be a customer-reachable HTTPS URL without credentials")
        if (
            not self.auth_header.startswith("Bearer ")
            or len(self.auth_header) > 8192
            or any(character.isspace() for character in self.auth_header[7:])
            or not self.auth_header[7:]
        ):
            raise ValueError("Clone authorization requires a bounded bearer capability")
        return self


class RepositoryArchiveOptions(Model):
    """Bounds cover the complete uncompressed tar and its safe extraction."""

    timeout_seconds: float = Field(default=900, gt=0, le=900)
    maximum_bytes: StrictInt = Field(default=512 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    maximum_files: StrictInt = Field(default=100_000, ge=1, le=100_000)
    # Includes implicit parent directories and gitlink directories. The upper
    # bound is derived from the authority's path-byte and file-count limits.
    maximum_directories: StrictInt = Field(default=100_000, ge=0, le=100_000 * 2048)


class RepositoryArchiveReceipt(Model):
    ref: Ref
    destination: Path
    byte_size: StrictInt = Field(ge=0)
    content_sha256: Digest
    entries: StrictInt = Field(ge=0)
