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
    BeforeValidator,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from synth_ai.sdk.research.contracts.forge.operations import RevisionReference

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
    """Serialize the validated frozen contract without external effects.

    Args:
        value: Value serialized into canonical bytes or a private receipt.

    Returns:
        bytes: Sorted compact UTF-8 JSON bytes with non-finite numbers refused.

    Raises:
        ValueError: Canonical JSON serialization rejects an invalid value.

    Examples:
        result = canonical_bytes(value)
    """
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def contract_digest(value: IndexContract) -> str:
    """Serialize the validated frozen contract without external effects.

    Args:
        value: Value serialized into canonical bytes or a private receipt.

    Returns:
        str: Lowercase SHA-256 digest of the canonical contract bytes.

    Raises:
        ValueError: Canonical JSON serialization rejects an invalid value.

    Examples:
        result = contract_digest(value)
    """
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class FrozenObject(IndexContract):
    """One exact frozen artifact declaration with its role in the research archive."""

    #: Unique identity of this frozen object within the snapshot.
    object_id: Identifier
    #: Declared input role; native session exports must use the session purpose.
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
    #: Immutable artifact path, digest, byte count and media type.
    object: ArtifactObjectDeclaration


class SessionExport(IndexContract):
    """Bounded native session capture with an explicit cutoff and completeness gaps."""

    #: Exact wire schema identifier.
    schema_version: Literal["synth.research.session-export.v1"] = "synth.research.session-export.v1"
    #: Native session producer: Codex, Swarms or mlok.
    source: Literal["codex", "swarms", "mlok"]
    #: Private native session identity retained only in the archive.
    native_session_id: ShortText
    #: Distinct parent session identities; the session cannot parent itself.
    parent_native_ids: tuple[ShortText, ...] = Field(default=(), max_length=64)
    #: Inclusive nonnegative start of the captured native event interval.
    event_start: int = Field(ge=0)
    #: Exclusive end of the captured native event interval.
    event_end_exclusive: int = Field(ge=0)
    #: Timezone-aware capture timestamp normalized to UTC.
    captured_at: AwareDatetime
    #: Timezone-aware frozen cutoff; sessions cannot exceed the containing snapshot cutoff.
    cutoff_at: AwareDatetime
    #: Whether the export is complete or has explicitly declared gaps.
    completeness: Literal["complete", "partial"]
    #: Declared missing capture segments; required for partial exports and absent for complete exports.
    gaps: tuple[ShortText, ...] = Field(default=(), max_length=256)
    #: Exact frozen session objects represented by this export.
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
    """Retained research attempt, including unsuccessful and excluded work."""

    #: Unique research attempt identifier within the snapshot.
    attempt_id: Identifier
    #: Recorded result; failed, excluded, cancelled and abandoned attempts remain evidence.
    outcome: Literal["succeeded", "failed", "excluded", "cancelled", "abandoned"]
    #: Distinct frozen input object identities consumed by the attempt.
    input_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1024)
    #: Distinct frozen observations retained from the attempt.
    observation_object_ids: tuple[Identifier, ...] = Field(default=(), max_length=1024)
    #: Explanation of the attempt result or exclusion.
    explanation: ShortText
    #: Whether the finding was positive, negative, null or not assessed.
    finding: Literal["positive", "negative", "null", "not_assessed"]


class ResearchSnapshot(IndexContract):
    """Closed research input archive with exact objects, sessions and attempt ledger."""

    #: Exact wire schema identifier.
    schema_version: Literal["synth.research.snapshot.v1"] = "synth.research.snapshot.v1"
    #: Frozen snapshot identity.
    snapshot_id: Identifier
    #: Parent snapshot identity, or none for a root snapshot.
    parent_snapshot_id: Identifier | None = None
    #: Private retained-input collection, separate from the release collection.
    archive_collection_id: Identifier
    #: Timezone-aware frozen cutoff; sessions cannot exceed the containing snapshot cutoff.
    cutoff_at: AwareDatetime
    #: Closed frozen object set with unique object identities and logical paths.
    objects: tuple[FrozenObject, ...] = Field(min_length=1, max_length=4096)
    #: Bounded native session exports whose objects are present in the snapshot.
    sessions: tuple[SessionExport, ...] = Field(default=(), max_length=256)
    #: Explicit retained ledger of successful and unsuccessful research attempts.
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
    """Approved release asset bound to one frozen source object and digest."""

    #: Identifier of the deliverable asset produced by this recipe output.
    release_asset_id: Identifier
    #: Frozen object supplying the exact output bytes.
    source_object_id: Identifier
    #: Lowercase SHA-256 digest of the exact referenced bytes.
    digest_sha256: Digest


class BuildRecipe(IndexContract):
    """Deterministic frozen-copy recipe with explicit environment, authored inputs and outputs."""

    #: Exact wire schema identifier.
    schema_version: Literal["synth.research.build-recipe.v1"] = "synth.research.build-recipe.v1"
    #: Identifier of this deterministic build recipe.
    recipe_id: Identifier
    #: SHA-256 digest of the canonical frozen snapshot contract.
    snapshot_digest_sha256: Digest
    #: Frozen-copy builder identity; no arbitrary code execution is implied.
    builder_version: Literal["frozen-copy-v1"] = "frozen-copy-v1"
    #: Frozen objects with the environment purpose required by this recipe.
    environment_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    #: Frozen objects with the authored purpose required by this recipe.
    authored_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1024)
    #: Distinct release assets bound to exact frozen source object bytes.
    outputs: tuple[OutputBinding, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        require_unique(self.environment_object_ids, "environment IDs")
        require_unique(self.authored_object_ids, "authored IDs")
        require_unique(tuple(item.release_asset_id for item in self.outputs), "recipe outputs")
        return self


class ApprovedRepresentation(IndexContract):
    """Explicit indexable interpretation of an exact deliverable asset."""

    #: Unique identity of the approved indexable representation.
    representation_id: Identifier
    #: Identifier of the exact deliverable asset.
    asset_id: Identifier
    #: Lowercase SHA-256 digest of the exact referenced bytes.
    digest_sha256: Digest
    #: Pinned parser identity used to interpret the disclosed bytes.
    parser_version: Identifier
    #: Approved representation type: UTF-8 text or an approved caption.
    kind: Literal["utf8_text", "approved_caption"]


class DeliverableAsset(IndexContract):
    """Exact asset bytes approved for release disclosure."""

    #: Identifier of the exact deliverable asset.
    asset_id: Identifier
    #: Lowercase SHA-256 digest of the exact referenced bytes.
    digest_sha256: Digest


class ReleaseDisclosure(IndexContract):
    """Audience-bound release assets and representations; excludes private archive identities."""

    #: Exact wire schema identifier.
    schema_version: Literal["synth.contribution.release-disclosure.v1"] = (
        "synth.contribution.release-disclosure.v1"
    )
    #: Exact Contribution and revision authorized by this disclosure.
    reference: ContributionReference
    #: Released-output collection, distinct from the private input archive.
    release_collection_id: Identifier
    #: SHA-256 digest of the exact released manifest.
    release_manifest_digest_sha256: Digest
    #: Audience approved for this exact release disclosure.
    audience: ContributionAudience
    #: Positive version of the approved disclosure.
    disclosure_version: int = Field(ge=1)
    #: Positive classification version used by this disclosure.
    classification_version: int = Field(ge=1)
    #: Distinct exact assets approved for delivery to the specified audience.
    deliverable_assets: tuple[DeliverableAsset, ...] = Field(min_length=1, max_length=1024)
    #: Approved indexable representations bound to deliverable asset digests.
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
    """Verifier observation binding reproduction scope and outcome to exact inputs and outputs."""

    #: Exact wire schema identifier.
    schema_version: Literal["synth.research.reproduction-receipt.v1"] = (
        "synth.research.reproduction-receipt.v1"
    )
    #: Identifier of this verifier receipt.
    receipt_id: Identifier
    #: SHA-256 digest of the canonical frozen snapshot contract.
    snapshot_digest_sha256: Digest
    #: SHA-256 digest of the canonical recipe contract.
    recipe_digest_sha256: Digest
    #: SHA-256 digest of the exact released manifest.
    release_manifest_digest_sha256: Digest
    #: What was actually reproduced: artifact reconstruction, analysis recomputation or experimental rerun.
    scope: Literal["artifact_reconstruction", "analysis_recomputation", "experimental_rerun"]
    #: Recorded result; failed, excluded, cancelled and abandoned attempts remain evidence.
    outcome: Literal["passed", "failed", "not_applicable"]
    #: Timezone-aware verification timestamp.
    checked_at: AwareDatetime
    #: Identity of the verifier that produced this observation.
    verifier_id: Identifier
    #: Frozen evidence object identities supporting the receipt.
    evidence_object_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1024)
    #: Explicit limitations of reproduction scope or outcome.
    limitations: ShortText


class DerivationBinding(IndexContract):
    """Validated join of frozen archive, recipe, disclosure and reproduction evidence."""

    #: Exact wire schema identifier.
    schema_version: Literal["synth.research.derivation.v1"] = "synth.research.derivation.v1"
    #: Exact closed retained-input snapshot.
    snapshot: ResearchSnapshot
    #: Deterministic recipe bound to the canonical snapshot digest.
    recipe: BuildRecipe
    #: Exact audience-bound approved release projection.
    disclosure: ReleaseDisclosure
    #: Bounded reproduction observations matching snapshot, recipe and release manifest.
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


class ScientificSnapshot(ResearchSnapshot):
    """Human-capable frozen inventory; never invent execution evidence for a copy."""

    schema_version: Literal["synth.research.snapshot.v2"] = "synth.research.snapshot.v2"
    attempts: tuple[Attempt, ...] = Field(default=(), max_length=100_000)


class FrozenCopyRecipe(BuildRecipe):
    schema_version: Literal["synth.research.build-recipe.v2"] = "synth.research.build-recipe.v2"
    builder_version: Literal["frozen-copy-v2"] = "frozen-copy-v2"
    environment_object_ids: tuple[Identifier, ...] = Field(default=(), max_length=64)


class ScientificApprovedRepresentation(ApprovedRepresentation):
    """V2 also admits explicit structured UTF-8 scientific evidence."""

    kind: Literal["utf8_text", "approved_caption", "structured_text"]


class ScientificReleaseDisclosure(ReleaseDisclosure):
    """Full-credit reviewer exclusions, bound without exposing private source IDs."""

    schema_version: Literal["synth.contribution.release-disclosure.v2"] = (
        "synth.contribution.release-disclosure.v2"
    )
    credited_principal_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=32)
    representations: tuple[ScientificApprovedRepresentation, ...] = Field(
        min_length=1, max_length=1024
    )

    @model_validator(mode="after")
    def unique_credit(self):
        require_unique(self.credited_principal_ids, "credited principals")
        return self


class ForgeProjectionOutput(IndexContract):
    """An approved release output referenced by scientific provenance, not copied."""

    release_asset_id: Identifier
    digest_sha256: Digest
    projection: Literal["export_asset", "public_provenance"]


class ForgeProjectionRecipe(IndexContract):
    """Closed Forge v3 primary-archive projection; current registration is authority."""

    schema_version: Literal["synth.research.forge-projection.v1"] = (
        "synth.research.forge-projection.v1"
    )
    recipe_id: Identifier
    snapshot_digest_sha256: Digest
    builder_version: Literal["forge-export-projection-v1"] = "forge-export-projection-v1"
    source_digest_sha256: Digest
    source_revision_reference: RevisionReference
    scientific_archive_object_id: Identifier
    outputs: tuple[ForgeProjectionOutput, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def unique_outputs(self):
        require_unique(
            tuple(output.release_asset_id for output in self.outputs),
            "projection outputs",
        )
        return self


class ScientificDerivation(DerivationBinding):
    schema_version: Literal["synth.research.derivation.v2"] = "synth.research.derivation.v2"
    snapshot: ScientificSnapshot
    recipe: Annotated[
        FrozenCopyRecipe | ForgeProjectionRecipe, Field(discriminator="schema_version")
    ]
    disclosure: ScientificReleaseDisclosure

    @model_validator(mode="after")
    def validate_exact_binding(self) -> Self:
        if isinstance(self.recipe, FrozenCopyRecipe):
            return super().validate_exact_binding()
        if self.recipe.snapshot_digest_sha256 != contract_digest(self.snapshot):
            raise ValueError("projection recipe binds a different snapshot")
        if self.snapshot.archive_collection_id == self.disclosure.release_collection_id:
            raise ValueError("archive and release require separate collections")
        objects = {item.object_id: item for item in self.snapshot.objects}
        archive = objects.get(self.recipe.scientific_archive_object_id)
        if (
            archive is None
            or archive.purpose != "data"
            or archive.object.logical_path != "scientific-archive.json"
        ):
            raise ValueError("projection requires exact frozen scientific archive object")
        outputs = {item.release_asset_id: item.digest_sha256 for item in self.recipe.outputs}
        declared = {
            item.asset_id: item.digest_sha256 for item in self.disclosure.deliverable_assets
        }
        if outputs != declared:
            raise ValueError("projection outputs differ from exact disclosure")
        for receipt in self.reproduction_receipts:
            if (
                receipt.snapshot_digest_sha256,
                receipt.recipe_digest_sha256,
                receipt.release_manifest_digest_sha256,
            ) != (
                contract_digest(self.snapshot),
                contract_digest(self.recipe),
                self.disclosure.release_manifest_digest_sha256,
            ):
                raise ValueError("reproduction receipt binds different inputs or outputs")
            if not set(receipt.evidence_object_ids).issubset(objects):
                raise ValueError("receipt evidence is not frozen")
        return self


def _legacy_derivation_version(value):
    if isinstance(value, dict) and "schema_version" not in value:
        return {**value, "schema_version": "synth.research.derivation.v1"}
    return value


VersionedDerivation = Annotated[
    DerivationBinding | ScientificDerivation,
    Field(discriminator="schema_version"),
    BeforeValidator(_legacy_derivation_version),
]
_DERIVATION_ADAPTER = TypeAdapter(VersionedDerivation)


def decode_derivation(value: dict | bytes) -> DerivationBinding | ScientificDerivation:
    """Decode an explicit version, retaining v1's required attempts/environment."""
    if isinstance(value, bytes):
        return _DERIVATION_ADAPTER.validate_json(value)
    return _DERIVATION_ADAPTER.validate_python(value)


def _legacy_disclosure_version(value):
    if isinstance(value, dict) and "schema_version" not in value:
        return {**value, "schema_version": "synth.contribution.release-disclosure.v1"}
    return value


VersionedDisclosure = Annotated[
    ReleaseDisclosure | ScientificReleaseDisclosure,
    Field(discriminator="schema_version"),
    BeforeValidator(_legacy_disclosure_version),
]
_DISCLOSURE_ADAPTER = TypeAdapter(VersionedDisclosure)


def decode_disclosure(value: dict) -> ReleaseDisclosure | ScientificReleaseDisclosure:
    return _DISCLOSURE_ADAPTER.validate_python(value)
