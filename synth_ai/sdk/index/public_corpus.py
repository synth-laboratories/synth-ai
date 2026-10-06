"""Frozen public corpus and authorized lineage contracts.

See notes/specifications/synth-index/public-corpus-lineage.md. Frozen identities
are evidence bindings, never permissions; only complete currently readable
manifests are disclosed.
"""

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from .contracts import (
    ContributionKind,
    ContributionReference,
    Identifier,
    IndexContract,
    ResearchArea,
    ShortText,
    WorkflowStage,
    require_unique,
)
from .lifecycle import Digest, Title
from .package import CreditedContributor


class CorpusPin(IndexContract):
    reference: ContributionReference
    manifest_digest: Digest


class RegisteredCorpus(IndexContract):
    schema_version: Literal["synth.index.registered-corpus.v1"]
    corpus_id: Identifier
    title: Title
    source_receipt_sha256: Digest
    members: tuple[CorpusPin, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def distinct_contributions(self) -> Self:
        require_unique(
            tuple(item.reference.contribution_id for item in self.members),
            "corpus Contributions",
        )
        return self


class PublicCorpusSummary(IndexContract):
    corpus_id: Identifier
    title: Title
    manifest_digest: Digest
    source_receipt_sha256: Digest
    count: Annotated[StrictInt, Field(ge=1, le=256)]


class PublicCorpusMember(CorpusPin):
    title: Title
    kind: ContributionKind
    research_areas: tuple[ResearchArea, ...] = Field(max_length=11)
    workflow_stages: tuple[WorkflowStage, ...] = Field(max_length=6)
    tag_ids: tuple[Identifier, ...] = Field(max_length=10)


class PublicCorpus(PublicCorpusSummary):
    schema_version: Literal["synth.index.public-corpus.v1"] = "synth.index.public-corpus.v1"
    members: tuple[PublicCorpusMember, ...] = Field(min_length=1, max_length=256)


class PublicCorpora(IndexContract):
    schema_version: Literal["synth.index.public-corpora.v1"] = "synth.index.public-corpora.v1"
    items: tuple[PublicCorpusSummary, ...] = Field(max_length=32)


class PublicLineageRevision(CorpusPin):
    title: Title
    is_current: bool
    published_at: AwareDatetime | None = None
    parent_revision_id: Identifier | None = None
    change_reason: ShortText | None = None


class PublicLineageRelationship(IndexContract):
    kind: Literal[
        "upstream",
        "replication",
        "supersedes",
        "citation",
        "reuse",
        "derivation",
        "correction",
        "contradiction",
    ]
    source: ContributionReference
    target: ContributionReference
    reason: ShortText | None = None


class PublicLineage(IndexContract):
    schema_version: Literal["synth.index.public-lineage.v1"] = "synth.index.public-lineage.v1"
    contribution_id: Identifier
    current_revision_id: Identifier
    revisions: tuple[PublicLineageRevision, ...] = Field(min_length=1, max_length=256)
    relationships: tuple[PublicLineageRelationship, ...] = Field(max_length=384)
    contributors: tuple[CreditedContributor, ...] = Field(max_length=32)
