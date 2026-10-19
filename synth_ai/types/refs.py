"""Exact, immutable references and append-only provenance links (core.v1).

References preserve Forge ExactReference.v1 and Producer.v1 wire shapes. An
alias must be resolved before constructing a reference; it never means latest.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
)

from .base import Model

ReferenceIdentifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")]
ReferenceDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RefInvalid(ValueError):  # noqa: N818
    """A reference does not satisfy the exact-reference wire contract."""

    code = "ref_invalid"
    service = "sdk"
    stage = "reference_validation"


class LinkInvalid(ValueError):  # noqa: N818
    """A link does not satisfy the immutable provenance contract."""

    code = "link_invalid"
    service = "sdk"
    stage = "link_validation"


class ExactReference(Model):
    authority: ReferenceIdentifier
    kind: ReferenceIdentifier
    record_id: ReferenceIdentifier
    revision: ReferenceIdentifier
    digest_sha256: ReferenceDigest

    @classmethod
    def _exact(cls, authority: str, kind: str, record_id: str, revision: str, digest: str) -> Self:
        try:
            return cls(
                authority=authority,
                kind=kind,
                record_id=record_id,
                revision=revision,
                digest_sha256=digest,
            )
        except ValidationError as error:
            raise RefInvalid(f"Invalid {authority}/{kind} exact reference") from error

    @classmethod
    def artifact(cls, artifact_id: str, revision: str, digest_sha256: str) -> Self:
        import re

        if re.fullmatch(r"[1-9][0-9]*", revision) is None:
            raise RefInvalid("Artifact revision must be an exact positive revision number")
        return cls._exact("artifacts", "artifact", artifact_id, revision, digest_sha256)

    @classmethod
    def commit(cls, repo_id: str, commit_sha: str, digest_sha256: str) -> Self:
        import re

        if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
            raise RefInvalid("Commit revision must be a full lowercase SHA-1")
        return cls._exact("git", "commit", repo_id, commit_sha, digest_sha256)

    @classmethod
    def run(cls, run_id: str, digest_sha256: str) -> Self:
        return cls._exact("swarms", "run", run_id, "1", digest_sha256)


Ref = ExactReference


class Producer(Model):
    """Execution DTOs stay producer-owned; only exact opaque references live here."""

    authority: ReferenceIdentifier
    subject_id: ReferenceIdentifier
    admission: Ref | None = None
    budget: Ref | None = None
    context: Ref | None = None
    execution: Ref | None = None
    grant: Ref | None = None

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {key: value for key, value in handler(self).items() if value is not None}


class Relation(StrEnum):
    CONSUMED = "consumed"
    PRODUCED = "produced"
    BASED_ON = "based_on"
    CITES = "cites"
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"


class Link(Model):
    link_id: ReferenceIdentifier
    project_id: ReferenceIdentifier
    subject: Ref
    relation: Relation
    object: Ref
    producer: Producer
    created_at: AwareDatetime

    @classmethod
    def from_wire(cls, data: str | bytes) -> Self:
        try:
            return cls.model_validate_json(data)
        except ValidationError as error:
            raise LinkInvalid("Invalid immutable provenance link") from error
