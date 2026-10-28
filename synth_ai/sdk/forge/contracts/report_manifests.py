# Generated from Forge packages/forge/report_manifests.py (Forge dev 6d47a173).
# fmt: off
"""Report snapshots, evidence manifests and immutable versions. See SYN-5155.

Forge owns the scientific record and its report lineage. Bytes, immutable file
versions, grants and serving stay with Artifact Platform: a manifest cites
artifacts by exact ``artifact-platform`` references and content digests (the
backend#3337 ``synth.forge.evidence.v1`` locator is exactly an ``Artifact``
record with role ``evidence``). Published version documents are content
addressed, so they can live in that same custody; this module never chooses a
storage location and grants no access. Disclosure here is a sharing audience,
never an Index contribution: Index alone approves publication.

Lifecycle: a ``ReportVersion`` is canonical JSON whose sha256 is its identity.
Corrections, supersessions and retractions append a new version that pins the
previous version digest; old bytes are never rewritten and stay readable, and a
citation of an older version reads back as ``stale`` or ``retracted``.

Migration: v1 ``Report`` records and ``ScientificExport`` selections are read
unchanged and projected into a v1 manifest (``manifest_from_report`` and
``manifest_from_scientific_export``). Nothing is written back to the v0.24
records, and the v0.24 contracts keep validating as before.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, Field, model_validator

from .contracts import (
    Contract,
    Digest,
    ExactReference,
    ForgeError,
    Identifier,
    Producer,
    Scope,
    Text,
    canonical_bytes,
    contract_digest,
)
from .exports import Credit, ScientificExport
from .operations import RevisionReference, revision_reference
from .operations_v2 import ScientificRevision, parse_revision
from .records import Artifact, ArtifactDeliveryBinding, Report

REPORT_MANIFEST_V1 = "forge.report-manifest.v1"
REPORT_VERSION_V1 = "forge.report-version.v1"
VERSION_BYTES_MAX = 4 * 1024 * 1024  # backend#3337 evidence MAX_UPLOAD_BYTES

SourceRole = Literal[
    "source", "evaluation", "result", "snapshot", "export", "recipe", "exporter"
]
EvidenceState = Literal["absent", "unverified", "inaccessible"]


class ManifestSource(Contract):
    reference: ExactReference
    role: SourceRole


class ManifestArtifact(Contract):
    """One exact artifact version the report relies on, pinned by content digest."""

    artifact_id: Identifier
    object: ExactReference
    content_sha256: Digest
    role: Literal["input", "intermediate", "output", "evidence"]
    media_type: Text | None = None
    byte_count: int | None = Field(default=None, ge=0)
    record: ExactReference | None = None
    manifest: ExactReference | None = None
    delivery: ArtifactDeliveryBinding | None = None
    source: ExactReference | None = None

    @model_validator(mode="after")
    def artifact_custody(self):
        if self.object.authority == "forge" or (
            self.manifest is not None and self.manifest.authority == "forge"
        ):
            raise ValueError("artifact bytes belong to the artifact authority")
        if (
            self.object.authority == "artifact-platform"
            and self.object.digest_sha256 != self.content_sha256
        ):
            raise ValueError("artifact-platform object digest is the content digest")
        if self.delivery is not None and (
            self.manifest is None
            or self.manifest.authority != "artifact-platform"
            or self.manifest.record_id != str(self.delivery.publication_id)
        ):
            raise ValueError("delivery must pin the exact publication")
        if self.record is not None and (
            self.record.authority != "forge" or self.record.kind != "artifact"
        ):
            raise ValueError("artifact record must be an exact Forge artifact revision")
        return self


class MissingEvidence(Contract):
    """Evidence the report would need but cannot pin; never silently dropped."""

    description: Text
    state: EvidenceState
    reference: ExactReference | None = None


class Disclosure(Contract):
    """Who may be shown this version. Sharing is not an Index contribution."""

    audience: Literal["private", "organization", "invited"] = "private"
    index_contribution: Literal["none"] = "none"


class ManifestProvenance(Contract):
    report_revision: RevisionReference
    producer: Producer
    derived_from: Literal["forge.report.v1", "forge.scientific-export.v1"]
    export: ExactReference | None = None
    # Retained as evidence of the author's v0.24 proposal; grants nothing.
    index_disclosure_proposals: tuple[ExactReference, ...] = Field(
        default=(), max_length=256
    )


class ReportManifest(Contract):
    schema_version: Literal["forge.report-manifest.v1"] = REPORT_MANIFEST_V1
    scope: Scope
    report: ExactReference
    title: Text
    methodology: Text
    body_sha256: Digest
    claims: tuple[Text, ...] = Field(default=(), max_length=256)
    limitations: tuple[Text, ...] = Field(default=(), max_length=256)
    credits: tuple[Credit, ...] = Field(default=(), max_length=256)
    sources: tuple[ManifestSource, ...] = Field(default=(), max_length=2048)
    artifacts: tuple[ManifestArtifact, ...] = Field(default=(), max_length=1024)
    missing_evidence: tuple[MissingEvidence, ...] = Field(default=(), max_length=1024)
    disclosure: Disclosure = Disclosure()
    provenance: ManifestProvenance

    @model_validator(mode="after")
    def exact_lineage(self):
        if self.report.authority != "forge" or self.report.kind != "report":
            raise ValueError("manifest pins one exact Forge report revision")
        if self.provenance.report_revision.reference != self.report:
            raise ValueError("provenance must bind the pinned report revision")
        if len({item.artifact_id for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("artifact IDs must be unique")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("sources must be unique")
        pinned = {item.content_sha256 for item in self.artifacts}
        if any(
            item.reference is not None and item.reference.digest_sha256 in pinned
            for item in self.missing_evidence
        ):
            raise ValueError("evidence cannot be both pinned and missing")
        return self


class ReportVersionLink(Contract):
    version: int = Field(ge=1)
    version_sha256: Digest


class ReportVersion(Contract):
    schema_version: Literal["forge.report-version.v1"] = REPORT_VERSION_V1
    lineage_id: Identifier
    version: Annotated[int, Field(strict=True, ge=1, le=2147483647)]
    disposition: Literal["initial", "correction", "supersession", "retraction"]
    previous: ReportVersionLink | None = None
    reason: Text | None = None
    manifest_sha256: Digest
    manifest: ReportManifest
    published_at: AwareDatetime

    @model_validator(mode="after")
    def lifecycle(self):
        if self.manifest_sha256 != contract_digest(self.manifest):
            raise ValueError("manifest digest differs from manifest bytes")
        if (self.disposition == "initial") != (self.previous is None):
            raise ValueError("only the initial version omits a previous version")
        if (self.disposition == "initial") != (self.version == 1):
            raise ValueError("the initial version is version 1")
        if self.previous is not None and self.previous.version != self.version - 1:
            raise ValueError("versions append contiguously")
        if self.disposition != "initial" and self.reason is None:
            raise ValueError("corrections, supersessions and retractions need a reason")
        return self


def version_bytes(version: ReportVersion) -> bytes:
    return canonical_bytes(version)


def version_digest(version: ReportVersion) -> str:
    return hashlib.sha256(version_bytes(version)).hexdigest()


def read_version(content: bytes, expected_sha256: str) -> ReportVersion:
    """Independent readback: bytes hash to the cited digest and re-encode."""
    if len(content) > VERSION_BYTES_MAX:
        raise ForgeError("version_too_large", "version exceeds custody bound")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ForgeError("digest_mismatch", "report version bytes differ from citation")
    try:
        version = ReportVersion.model_validate(json.loads(content))
    except ValueError as error:
        raise ForgeError("version_invalid", "version does not validate") from error
    if version_bytes(version) != content:
        raise ForgeError("version_invalid", "report version is not canonical")
    return version


def publish_initial(manifest: ReportManifest, published_at) -> ReportVersion:
    return ReportVersion(
        lineage_id=manifest.report.record_id,
        version=1,
        disposition="initial",
        manifest_sha256=contract_digest(manifest),
        manifest=manifest,
        published_at=published_at,
    )


def publish_next(
    previous: ReportVersion,
    *,
    disposition: Literal["correction", "supersession", "retraction"],
    reason: str,
    published_at,
    manifest: ReportManifest | None = None,
) -> ReportVersion:
    """Append a version; the previous bytes stay as they were."""
    if previous.disposition == "retraction":
        raise ForgeError("lineage_retracted", "a retracted lineage takes no versions")
    if published_at <= previous.published_at:
        raise ForgeError("version_order", "versions are published in time order")
    if disposition == "retraction":
        if manifest is not None and manifest != previous.manifest:
            raise ForgeError("selection_invalid", "a retraction keeps the manifest")
        manifest = previous.manifest
    elif manifest is None or contract_digest(manifest) == previous.manifest_sha256:
        raise ForgeError("selection_invalid", f"a {disposition} changes the manifest")
    if manifest.scope != previous.manifest.scope:
        raise ForgeError("wrong_scope", "report lineage cannot change scope")
    if disposition == "correction" and (
        manifest.report.record_id != previous.manifest.report.record_id
    ):
        raise ForgeError(
            "reference_conflict", "a correction revises the same report record"
        )
    return ReportVersion(
        lineage_id=previous.lineage_id,
        version=previous.version + 1,
        disposition=disposition,
        previous=ReportVersionLink(
            version=previous.version, version_sha256=version_digest(previous)
        ),
        reason=reason,
        manifest_sha256=contract_digest(manifest),
        manifest=manifest,
        published_at=published_at,
    )


def verify_lineage(versions: Sequence[ReportVersion]) -> None:
    """Every version pins the exact bytes of its predecessor."""
    if not versions or versions[0].disposition != "initial":
        raise ForgeError("lineage_invalid", "lineage starts at its initial version")
    for prior, current in zip(versions, versions[1:], strict=False):
        if (
            current.lineage_id != prior.lineage_id
            or current.previous is None
            or current.previous.version != prior.version
            or current.previous.version_sha256 != version_digest(prior)
            or current.manifest.scope != prior.manifest.scope
        ):
            raise ForgeError("lineage_invalid", "version does not pin its predecessor")


class ContentStore(Protocol):
    """Content-addressed byte custody, e.g. backend#3337 ``forge-evidence``."""

    def put(self, digest: str, content: bytes, media_type: str) -> object: ...

    def get(self, digest: str) -> bytes: ...


class ReportVersionLedger:
    """Append-only lineage heads over a content-addressed store.

    The store keeps bytes; this ledger only refuses forks and rewrites. A
    hosted deployment keeps heads in Forge records and bytes in Artifact
    Platform; the logic is the same.
    """

    MEDIA_TYPE = "application/vnd.synth.forge.report-version.v1+json"

    def __init__(self, store: ContentStore):
        self._store = store
        self._heads: dict[tuple[Scope, str], list[str]] = {}

    def publish(self, version: ReportVersion) -> str:
        key = (version.manifest.scope, version.lineage_id)
        chain = self._heads.get(key, [])
        expected = chain[-1] if chain else None
        cited = version.previous.version_sha256 if version.previous else None
        if cited != expected or version.version != len(chain) + 1:
            raise ForgeError("revision_conflict", "version does not extend the head")
        content = version_bytes(version)
        if len(content) > VERSION_BYTES_MAX:
            raise ForgeError("version_too_large", "version exceeds custody bound")
        digest = hashlib.sha256(content).hexdigest()
        self._store.put(digest, content, self.MEDIA_TYPE)
        self._heads[key] = [*chain, digest]
        return digest

    def read(self, digest: str) -> ReportVersion:
        return read_version(self._store.get(digest), digest)

    def history(self, scope: Scope, lineage_id: str) -> tuple[ReportVersion, ...]:
        versions = tuple(
            self.read(item) for item in self._heads.get((scope, lineage_id), [])
        )
        verify_lineage(versions)
        return versions

    def citation(self, scope: Scope, lineage_id: str, digest: str) -> CitationStatus:
        return citation_status(
            self._heads.get((scope, lineage_id), []), digest, self.read
        )


def citation_status(
    chain: Sequence[str], digest: str, read: Callable[[str], ReportVersion]
) -> CitationStatus:
    """Resolve a cited version digest against its lineage chain of digests."""
    if digest not in chain:
        raise ForgeError("reference_unavailable", "version not in this lineage")
    head = read(chain[-1])
    cited = read(digest)
    if head.disposition == "retraction":
        state = "retracted"
    else:
        state = "current" if digest == chain[-1] else "stale"
    return CitationStatus(
        cited=ReportVersionLink(version=cited.version, version_sha256=digest),
        head=ReportVersionLink(version=head.version, version_sha256=chain[-1]),
        state=state,
    )


class CitationStatus(Contract):
    cited: ReportVersionLink
    head: ReportVersionLink
    state: Literal["current", "stale", "retracted"]


class ArtifactCheck(Contract):
    artifact_id: Identifier
    content_sha256: Digest
    state: Literal["matched", "digest_mismatch", "absent", "inaccessible"]
    observed_sha256: Digest | None = None
    error_code: Identifier | None = None


class ReadbackReport(Contract):
    version_sha256: Digest
    verdict: Literal["complete", "partial", "refused"]
    artifacts: tuple[ArtifactCheck, ...]
    missing_evidence: tuple[MissingEvidence, ...]


ArtifactReader = Callable[[ManifestArtifact], bytes]
_ABSENT = {"reference_unavailable", "not_found", "evidence_unavailable"}


def verify_readback(
    content: bytes, expected_sha256: str, read_artifact: ArtifactReader
) -> ReadbackReport:
    """Read a version independently and match every available artifact digest.

    ``read_artifact`` is the caller's authorized reader (Artifact Platform
    delivery or the evidence ``/content`` route); it raises ``ForgeError``
    when bytes are absent or denied. Absence stays explicit; a wrong digest
    refuses the whole readback.
    """
    version = read_version(content, expected_sha256)
    checks = []
    for item in version.manifest.artifacts:
        try:
            observed = hashlib.sha256(read_artifact(item)).hexdigest()
        except ForgeError as error:
            state = "absent" if error.code in _ABSENT else "inaccessible"
            checks.append(
                ArtifactCheck(
                    artifact_id=item.artifact_id,
                    content_sha256=item.content_sha256,
                    state=state,
                    error_code=error.code,
                )
            )
            continue
        checks.append(
            ArtifactCheck(
                artifact_id=item.artifact_id,
                content_sha256=item.content_sha256,
                state="matched"
                if observed == item.content_sha256
                else "digest_mismatch",
                observed_sha256=observed,
            )
        )
    states = {check.state for check in checks}
    if "digest_mismatch" in states:
        verdict = "refused"
    elif states - {"matched"} or version.manifest.missing_evidence:
        verdict = "partial"
    else:
        verdict = "complete"
    return ReadbackReport(
        version_sha256=expected_sha256,
        verdict=verdict,
        artifacts=tuple(checks),
        missing_evidence=version.manifest.missing_evidence,
    )


# --- v0.24 migration: read old records unchanged, write the v1 manifest ---


def _exact(record: ScientificRevision) -> ScientificRevision:
    record = parse_revision(record.model_dump(mode="json"))
    if (
        record.reference.authority != "forge"
        or record.reference.kind != record.payload.kind
        or record.reference.digest_sha256 != contract_digest(record.payload)
    ):
        raise ForgeError("reference_conflict", "record differs from exact source bytes")
    return record


def artifact_entry(record: ScientificRevision) -> ManifestArtifact:
    """Project one Forge ``Artifact`` record (e.g. a #3337 evidence locator)."""
    record = _exact(record)
    artifact = record.payload
    if not isinstance(artifact, Artifact):
        raise ForgeError("selection_invalid", "expected a Forge artifact record")
    return ManifestArtifact(
        artifact_id=record.reference.record_id,
        object=artifact.object,
        content_sha256=artifact.object.digest_sha256,
        role=artifact.role,
        media_type=artifact.media_type,
        byte_count=artifact.byte_count,
        record=record.reference,
        manifest=artifact.manifest,
        delivery=artifact.delivery,
    )


def _sources(
    references: Iterable[ExactReference], role: SourceRole
) -> list[ManifestSource]:
    roles: dict[str, SourceRole] = {
        "result": "result",
        "snapshot": "snapshot",
        "scientific_export": "export",
    }
    return [
        ManifestSource(reference=item, role=roles.get(item.kind, role))
        for item in references
    ]


def manifest_from_report(
    report_record: ScientificRevision,
    *,
    artifacts: Sequence[ScientificRevision] = (),
    evaluations: Sequence[ExactReference] = (),
    missing_evidence: Sequence[MissingEvidence] = (),
    disclosure: Disclosure = Disclosure(),
    credits: Sequence[Credit] = (),
) -> ReportManifest:
    """Read a v0.24 ``Report`` revision and write a v1 manifest.

    Cited artifact sources without a supplied, matching artifact record are
    declared ``unverified`` instead of being dropped or assumed present.
    """
    record = _exact(report_record)
    report = record.payload
    if not isinstance(report, Report):
        raise ForgeError("selection_invalid", "expected an exact Forge report")
    entries = [artifact_entry(item) for item in artifacts]
    pinned = {entry.record for entry in entries}
    missing = list(missing_evidence)
    for reference in report.sources:
        if reference.kind == "artifact" and reference not in pinned:
            missing.append(
                MissingEvidence(
                    description="cited artifact record was not supplied for pinning",
                    state="unverified",
                    reference=reference,
                )
            )
    sources = _sources(report.sources, "source")
    sources += [
        ManifestSource(reference=item, role="evaluation") for item in evaluations
    ]
    return ReportManifest(
        scope=record.scope,
        report=record.reference,
        title=report.title,
        methodology=report.methods,
        body_sha256=hashlib.sha256(report.body.encode("utf-8")).hexdigest(),
        claims=report.claims,
        limitations=report.limitations,
        credits=tuple(credits),
        sources=tuple(dict.fromkeys(sources)),
        artifacts=tuple(entries),
        missing_evidence=tuple(missing),
        disclosure=disclosure,
        provenance=ManifestProvenance(
            report_revision=revision_reference(record),
            producer=record.producer,
            derived_from="forge.report.v1",
        ),
    )


def manifest_from_scientific_export(
    export_record: ScientificRevision,
    report_record: ScientificRevision,
    *,
    artifacts: Sequence[ScientificRevision] = (),
    missing_evidence: Sequence[MissingEvidence] = (),
    disclosure: Disclosure = Disclosure(),
) -> ReportManifest:
    """Read a v0.24 ``ScientificExport`` plus its report and write a v1 manifest.

    Export assets become artifact entries pinned by their frozen digests; the
    export's public-source proposals are retained as provenance only.
    """
    export_rev = _exact(export_record)
    export = export_rev.payload
    if not isinstance(export, ScientificExport):
        raise ForgeError("selection_invalid", "expected an exact scientific export")
    report_rev = _exact(report_record)
    if report_rev.scope != export_rev.scope:
        raise ForgeError("wrong_scope", "report and export scopes differ")
    base = manifest_from_report(
        report_rev,
        artifacts=artifacts,
        missing_evidence=missing_evidence,
        disclosure=disclosure,
        credits=export.credits,
    )
    pinned = {item.content_sha256 for item in base.artifacts}
    assets = [
        ManifestArtifact(
            artifact_id=asset.asset_id,
            object=asset.source,
            content_sha256=asset.digest_sha256,
            role="output",
            source=asset.source,
        )
        for asset in export.assets
        if asset.digest_sha256 not in pinned
    ]
    sources = list(base.sources)
    sources += [ManifestSource(reference=export_rev.reference, role="export")]
    sources += _sources(export.snapshots, "snapshot")
    sources += _sources(export.sources, "source")
    sources += [
        ManifestSource(reference=export.recipe, role="recipe"),
        ManifestSource(reference=export.exporter, role="exporter"),
    ]
    migrated = base.model_copy(
        update={
            "sources": tuple(dict.fromkeys(sources)),
            "artifacts": (*base.artifacts, *assets),
            "provenance": base.provenance.model_copy(
                update={
                    "derived_from": "forge.scientific-export.v1",
                    "export": export_rev.reference,
                    "index_disclosure_proposals": export.proposed_public_sources,
                }
            ),
        }
    )
    # model_copy skips validation; re-validate the written manifest.
    return ReportManifest.model_validate(migrated.model_dump(mode="json"))
