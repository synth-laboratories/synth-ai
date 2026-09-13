"""Portable, strict Contribution descriptor over the existing Artifact contract.

See sibling docs/drafts/synth-index-contribution-format-2026-09-12.md §3–6.
The descriptor is one Artifact object; final publication digest lives outside it.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StringConstraints, model_validator

from .artifacts import ArtifactObjectDeclaration
from .contracts import (
    ContributionAudience,
    ContributionKind,
    ContributionOrigin,
    ContributionReference,
    Identifier,
    IndexContract,
    ResearchArea,
    ShortText,
    WorkflowStage,
    require_unique,
)


class ResearchCard(IndexContract):
    reusable: ShortText
    how_to_use: ShortText
    observed: ShortText
    limitations: ShortText
    resources: ShortText


class ContributionAsset(IndexContract):
    asset_id: Identifier
    role: Literal[
        "summary",
        "report",
        "code",
        "data",
        "model",
        "environment",
        "evidence",
        "reproduce",
    ]
    object: ArtifactObjectDeclaration
    license: Annotated[str, StringConstraints(min_length=1, max_length=256)]


class EvidenceReference(IndexContract):
    evidence_id: Identifier
    kind: Literal["comparison", "experiment", "evaluation", "trace", "manifest", "conformance"]
    asset_id: Identifier
    locator: ShortText


class ContributionClaim(IndexContract):
    claim_id: Identifier
    kind: Literal["empirical", "composition", "conformance", "diagnostic"]
    statement: ShortText
    scope: ShortText
    author_assessment: Literal["supported", "disproven", "inconclusive", "not_tested"]
    supporting_evidence_ids: tuple[Identifier, ...] = Field(default=(), max_length=64)
    contradicting_evidence_ids: tuple[Identifier, ...] = Field(default=(), max_length=64)
    missing_evidence_reason: ShortText | None = None

    @model_validator(mode="after")
    def check_evidence_declaration(self) -> Self:
        require_unique(self.supporting_evidence_ids, "supporting_evidence_ids")
        require_unique(self.contradicting_evidence_ids, "contradicting_evidence_ids")
        if set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids):
            raise ValueError("one evidence item cannot both support and contradict the same claim")
        if (
            not self.supporting_evidence_ids
            and not self.contradicting_evidence_ids
            and (self.missing_evidence_reason is None or self.author_assessment != "not_tested")
        ):
            raise ValueError("absent evidence requires not_tested and an explicit explanation")
        return self


class CreditedContributor(IndexContract):
    principal_id: Identifier
    roles: tuple[
        Literal["research", "code", "data", "evaluation", "writing", "replication"], ...
    ] = Field(
        min_length=1,
        max_length=6,
    )


class ContributionProvenance(IndexContract):
    origin: ContributionOrigin
    contributors: tuple[CreditedContributor, ...] = Field(min_length=1, max_length=32)
    upstream: tuple[ContributionReference, ...] = Field(default=(), max_length=64)
    tools: tuple[ShortText, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def check_unique_lineage(self) -> Self:
        require_unique(tuple(item.principal_id for item in self.contributors), "contributors")
        require_unique(
            tuple((item.contribution_id, item.revision_id) for item in self.upstream),
            "upstream",
        )
        return self


class Reproduction(IndexContract):
    level: Literal["inspectable", "runnable", "reproduced", "not_applicable"]
    instructions: ShortText
    expected_outputs: ShortText
    restrictions: ShortText
    entrypoint_asset_id: Identifier | None = None
    environment_asset_ids: tuple[Identifier, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def check_runnable_dependencies(self) -> Self:
        if self.level in ("runnable", "reproduced") and (
            self.entrypoint_asset_id is None or not self.environment_asset_ids
        ):
            raise ValueError("runnable reproduction requires entrypoint and environment assets")
        return self


class ContributionPackage(IndexContract):
    schema_version: Literal["synth.contribution.v1"] = "synth.contribution.v1"
    contribution_id: Identifier
    revision_id: Identifier
    parent_revision_id: Identifier | None = None
    kind: ContributionKind
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    abstract: ShortText
    research_areas: tuple[ResearchArea, ...] = Field(min_length=1, max_length=11)
    workflow_stages: tuple[WorkflowStage, ...] = Field(min_length=1, max_length=6)
    tag_ids: tuple[Identifier, ...] = Field(default=(), max_length=10)
    card: ResearchCard
    assets: tuple[ContributionAsset, ...] = Field(min_length=1, max_length=1024)
    claims: tuple[ContributionClaim, ...] = Field(min_length=1, max_length=128)
    evidence: tuple[EvidenceReference, ...] = Field(default=(), max_length=512)
    provenance: ContributionProvenance
    reproduction: Reproduction
    requested_audience: ContributionAudience = ContributionAudience.PRIVATE
    rights_attested: StrictBool
    sensitive_data: Literal["none_declared", "declared", "unknown"]

    @model_validator(mode="after")
    def check_package_links(self) -> Self:
        for name in ("research_areas", "workflow_stages", "tag_ids"):
            require_unique(getattr(self, name), name)
        if self.parent_revision_id == self.revision_id:
            raise ValueError("a revision cannot parent itself")
        if any(item.contribution_id == self.contribution_id for item in self.provenance.upstream):
            raise ValueError(
                "same-Contribution history belongs in parent_revision_id, not reward lineage"
            )
        assets = {item.asset_id: item for item in self.assets}
        require_unique(tuple(item.asset_id for item in self.assets), "asset_ids")
        require_unique(tuple(item.object.logical_path for item in self.assets), "asset_paths")
        require_unique(tuple(item.claim_id for item in self.claims), "claim_ids")
        require_unique(tuple(item.evidence_id for item in self.evidence), "evidence_ids")
        evidence = {item.evidence_id: item for item in self.evidence}
        for item in self.evidence:
            if item.asset_id not in assets:
                raise ValueError("evidence refers to an undeclared asset")
        for claim in self.claims:
            references = claim.supporting_evidence_ids + claim.contradicting_evidence_ids
            if any(reference not in evidence for reference in references):
                raise ValueError("claim refers to undeclared evidence")
            if (
                claim.kind == "empirical"
                and claim.author_assessment != "not_tested"
                and not any(evidence[reference].kind == "comparison" for reference in references)
            ):
                raise ValueError("empirical conclusions require comparison evidence")
        reproduction_ids = self.reproduction.environment_asset_ids
        if self.reproduction.entrypoint_asset_id is not None:
            reproduction_ids += (self.reproduction.entrypoint_asset_id,)
        if any(reference not in assets for reference in reproduction_ids):
            raise ValueError("reproduction refers to an undeclared asset")
        if self.kind in (
            ContributionKind.RESEARCH_REPORT,
            ContributionKind.REPLICATION,
        ) and not any(asset.role == "report" for asset in self.assets):
            raise ValueError("research reports and replications require a report asset")
        if self.kind == ContributionKind.REPLICATION and not self.provenance.upstream:
            raise ValueError("replication requires an exact upstream Contribution revision")
        return self
