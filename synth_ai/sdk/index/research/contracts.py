"""Closed frozen-input contracts; no object here is an authorization grant.

See notes/specifications/synth-index/research-archive-release.md. Public release
contracts deliberately have no archive/session/recipe identities.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..artifacts import ArtifactObjectDeclaration
from ..contracts import (
    ContributionAudience,
    ContributionReference,
    Identifier,
    IndexContract,
    ShortText,
    require_unique,
)

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def canonical_bytes(value: IndexContract) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def contract_digest(value: IndexContract) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class FrozenObject(IndexContract):
    object_id: Identifier
    purpose: Literal[
        "code",
        "data",
        "environment",
        "protocol",
        "attempt",
        "analysis",
        "authored",
        "session",
    ]
    object: ArtifactObjectDeclaration


class SessionExport(IndexContract):
    schema_version: Literal["synth.research.session-export.v1"] = "synth.research.session-export.v1"
    source: Literal["codex", "swarms", "mlok"]
    native_session_id: ShortText
    parent_native_ids: tuple[ShortText, ...] = Field(default=(), max_length=64)
    event_start: int = Field(ge=0)
    event_end_exclusive: int = Field(ge=0)
    captured_at: AwareDatetime
    cutoff_at: AwareDatetime
    completeness: Literal["complete", "partial"]
    gaps: tuple[ShortText, ...] = Field(default=(), max_length=256)
    native_objects: tuple[FrozenObject, ...] = Field(min_length=1, max_length=1024)

    @field_validator("captured_at", "cutoff_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        if self.event_end_exclusive < self.event_start or self.cutoff_at > self.captured_at:
            raise ValueError("capture range or cutoff is reversed")
        if (self.completeness == "partial") != bool(self.gaps):
            raise ValueError(
                "partial exports require declared gaps; complete exports cannot have gaps"
            )
        if self.native_session_id in self.parent_native_ids:
            raise ValueError("session cannot parent itself")
        require_unique(self.parent_native_ids, "session parents")
        require_unique(tuple(item.object_id for item in self.native_objects), "session object IDs")
        if any(item.purpose != "session" for item in self.native_objects):
            raise ValueError("native exports must be session objects")
        return self


class Attempt(IndexContract):
    attempt_id: Identifier
    outcome: Literal["succeeded", "failed", "excluded", "cancelled", "abandoned"]
    input_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1024)
    observation_object_ids: tuple[Identifier, ...] = Field(default=(), max_length=1024)
    explanation: ShortText
    finding: Literal["positive", "negative", "null", "not_assessed"]


class ResearchSnapshot(IndexContract):
    schema_version: Literal["synth.research.snapshot.v1"] = "synth.research.snapshot.v1"
    snapshot_id: Identifier
    parent_snapshot_id: Identifier | None = None
    archive_collection_id: Identifier
    cutoff_at: AwareDatetime
    objects: tuple[FrozenObject, ...] = Field(min_length=1, max_length=4096)
    sessions: tuple[SessionExport, ...] = Field(default=(), max_length=256)
    attempts: tuple[Attempt, ...] = Field(min_length=1, max_length=100_000)

    @field_validator("cutoff_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_links(self) -> Self:
        if self.snapshot_id == self.parent_snapshot_id:
            raise ValueError("snapshot cannot parent itself")
        require_unique(tuple(item.object_id for item in self.objects), "snapshot object IDs")
        require_unique(tuple(item.object.logical_path for item in self.objects), "snapshot paths")
        require_unique(tuple(item.attempt_id for item in self.attempts), "attempt IDs")
        require_unique(
            tuple((item.source, item.native_session_id) for item in self.sessions),
            "sessions",
        )
        by_id = {item.object_id: item for item in self.objects}
        for attempt in self.attempts:
            for references in (
                attempt.input_object_ids,
                attempt.observation_object_ids,
            ):
                require_unique(references, "attempt references")
                if not set(references).issubset(by_id):
                    raise ValueError("attempt refers to an unfrozen object")
        for export in self.sessions:
            if export.cutoff_at > self.cutoff_at:
                raise ValueError("session cutoff exceeds snapshot cutoff")
            if any(by_id.get(item.object_id) != item for item in export.native_objects):
                raise ValueError("session object differs from snapshot")
        return self


class OutputBinding(IndexContract):
    release_asset_id: Identifier
    source_object_id: Identifier
    digest_sha256: Digest


class BuildRecipe(IndexContract):
    schema_version: Literal["synth.research.build-recipe.v1"] = "synth.research.build-recipe.v1"
    recipe_id: Identifier
    snapshot_digest_sha256: Digest
    builder_version: Literal["frozen-copy-v1"] = "frozen-copy-v1"
    environment_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    authored_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1024)
    outputs: tuple[OutputBinding, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        require_unique(self.environment_object_ids, "environment IDs")
        require_unique(self.authored_object_ids, "authored IDs")
        require_unique(tuple(item.release_asset_id for item in self.outputs), "recipe outputs")
        return self


class ApprovedRepresentation(IndexContract):
    representation_id: Identifier
    asset_id: Identifier
    digest_sha256: Digest
    parser_version: Identifier
    kind: Literal["utf8_text", "approved_caption"]


class DeliverableAsset(IndexContract):
    asset_id: Identifier
    digest_sha256: Digest


class ReleaseDisclosure(IndexContract):
    schema_version: Literal["synth.contribution.release-disclosure.v1"] = (
        "synth.contribution.release-disclosure.v1"
    )
    reference: ContributionReference
    release_collection_id: Identifier
    release_manifest_digest_sha256: Digest
    audience: ContributionAudience
    disclosure_version: int = Field(ge=1)
    classification_version: int = Field(ge=1)
    deliverable_assets: tuple[DeliverableAsset, ...] = Field(min_length=1, max_length=1024)
    representations: tuple[ApprovedRepresentation, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_representations(self) -> Self:
        require_unique(
            tuple(item.asset_id for item in self.deliverable_assets),
            "deliverable assets",
        )
        require_unique(
            tuple(item.representation_id for item in self.representations),
            "representations",
        )
        assets = {item.asset_id: item.digest_sha256 for item in self.deliverable_assets}
        if any(assets.get(item.asset_id) != item.digest_sha256 for item in self.representations):
            raise ValueError("indexable representation must bind an exact deliverable asset")
        return self


class ReproductionReceipt(IndexContract):
    schema_version: Literal["synth.research.reproduction-receipt.v1"] = (
        "synth.research.reproduction-receipt.v1"
    )
    receipt_id: Identifier
    snapshot_digest_sha256: Digest
    recipe_digest_sha256: Digest
    release_manifest_digest_sha256: Digest
    scope: Literal["artifact_reconstruction", "analysis_recomputation", "experimental_rerun"]
    outcome: Literal["passed", "failed", "not_applicable"]
    checked_at: AwareDatetime
    verifier_id: Identifier
    evidence_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1024)
    limitations: ShortText


class DerivationBinding(IndexContract):
    schema_version: Literal["synth.research.derivation.v1"] = "synth.research.derivation.v1"
    snapshot: ResearchSnapshot
    recipe: BuildRecipe
    disclosure: ReleaseDisclosure
    reproduction_receipts: tuple[ReproductionReceipt, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_exact_binding(self) -> Self:
        snapshot_digest = contract_digest(self.snapshot)
        recipe_digest = contract_digest(self.recipe)
        if self.recipe.snapshot_digest_sha256 != snapshot_digest:
            raise ValueError("recipe binds a different snapshot")
        if self.snapshot.archive_collection_id == self.disclosure.release_collection_id:
            raise ValueError("archive and release require separate collections")
        objects = {item.object_id: item for item in self.snapshot.objects}
        for references, purpose in (
            (self.recipe.environment_object_ids, "environment"),
            (self.recipe.authored_object_ids, "authored"),
        ):
            if any(
                identifier not in objects or objects[identifier].purpose != purpose
                for identifier in references
            ):
                raise ValueError("recipe has missing or incorrectly typed frozen inputs")
        outputs = {item.release_asset_id: item for item in self.recipe.outputs}
        for asset in self.disclosure.deliverable_assets:
            output = outputs.get(asset.asset_id)
            if output is None or output.digest_sha256 != asset.digest_sha256:
                raise ValueError("deliverable lacks an exact recipe output")
            source = objects.get(output.source_object_id)
            if source is None or source.object.digest_sha256 != output.digest_sha256:
                raise ValueError("output is not bound to frozen bytes")
        for receipt in self.reproduction_receipts:
            if (
                receipt.snapshot_digest_sha256,
                receipt.recipe_digest_sha256,
                receipt.release_manifest_digest_sha256,
            ) != (
                snapshot_digest,
                recipe_digest,
                self.disclosure.release_manifest_digest_sha256,
            ):
                raise ValueError("reproduction receipt binds different inputs or outputs")
            if not set(receipt.evidence_object_ids).issubset(objects):
                raise ValueError("receipt evidence is not frozen")
        return self
