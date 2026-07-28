"""Typed public contracts for first-class Research Visuals.

# See: Jstack/.jstack/daily_notes/2026-07-27/SPEC_visuals_resource.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.contracts.pagination import PageCursor, extract_next_cursor
from synth_ai.core.research.contracts._wire import (
    array_value,
    object_value,
    optional_datetime,
    optional_text,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    ArtifactId,
    ProjectId,
    SwarmId,
    WorkProductId,
)


class VisualStatus(StrEnum):
    """Lifecycle state of a Visual version."""

    BUILDING = "building"
    READY = "ready"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class VisualVisibility(StrEnum):
    """Access boundary for a Visual."""

    ORGANIZATION = "org"
    PUBLIC = "public"


class VisualBlobState(StrEnum):
    """Durable blob deletion state."""

    PRESENT = "present"
    PENDING_DELETE = "pending_delete"
    DELETED = "deleted"


def _optional_identifier(
    payload: JsonObject,
    name: str,
    identifier_type: type[ArtifactId] | type[SwarmId] | type[WorkProductId],
) -> ArtifactId | SwarmId | WorkProductId | None:
    value = optional_text(payload, name)
    return identifier_type(value) if value is not None else None


def _required_integer(payload: JsonObject, name: str, *, minimum: int) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_string(payload: JsonObject, name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string when provided")
    return value


@dataclass(frozen=True, slots=True)
class VisualEvidencePayload:
    """Org-scoped evidence attached to a Visual.

    Mirrors the backend ``SmrVisualEvidencePayload``. Evidence is served only
    on the authenticated org Visual surface; the unauthenticated public
    surface never carries it.
    """

    source_run_ids: tuple[SwarmId, ...] = ()
    evidence: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(cls, value: JsonValue) -> VisualEvidencePayload:
        payload = object_value(value, operation_id="visual.evidence")
        source_run_ids = tuple(
            SwarmId(required_text({"source_run_id": item}, "source_run_id"))
            for item in array_value(
                cast(JsonValue, payload.get("source_run_ids", [])),
                operation_id="visual.evidence.source_run_ids",
            )
        )
        evidence = payload.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("visual evidence must be an object")
        return cls(source_run_ids=source_run_ids, evidence=evidence)

    def to_wire(self) -> JsonObject:
        return {
            "source_run_ids": [str(item) for item in self.source_run_ids],
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class Visual:
    """One immutable Visual version returned by the backend authority."""

    hosted_artifact_id: ArtifactId
    root_artifact_id: ArtifactId
    project_id: ProjectId
    run_id: SwarmId | None
    built_by_run_id: SwarmId | None
    work_product_id: WorkProductId | None
    title: str
    visual_kind: str
    artifact_version: int
    visibility: VisualVisibility
    status: VisualStatus
    public_slug: str | None
    public_url: str | None
    preview_url: str | None
    canonical_url: str
    summary: str | None
    content_digest: str | None
    size_bytes: int
    blob_state: VisualBlobState
    source_run_ids: tuple[SwarmId, ...]
    superseded_by_id: ArtifactId | None
    deleted_at: datetime | None
    published_at: datetime | None
    created_at: datetime
    evidence: VisualEvidencePayload | None = None

    @classmethod
    def from_wire(cls, value: JsonValue) -> Visual:
        """Decode a Visual without admitting raw storage locations."""
        payload = object_value(value, operation_id="visual")
        source_run_ids = tuple(
            SwarmId(required_text({"source_run_id": item}, "source_run_id"))
            for item in array_value(
                cast(JsonValue, payload.get("source_run_ids", [])),
                operation_id="visual.source_run_ids",
            )
        )
        visual = cls(
            hosted_artifact_id=ArtifactId(required_text(payload, "hosted_artifact_id")),
            root_artifact_id=ArtifactId(required_text(payload, "root_artifact_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            run_id=cast(
                SwarmId | None,
                _optional_identifier(payload, "run_id", SwarmId),
            ),
            built_by_run_id=cast(
                SwarmId | None,
                _optional_identifier(payload, "built_by_run_id", SwarmId),
            ),
            work_product_id=cast(
                WorkProductId | None,
                _optional_identifier(payload, "work_product_id", WorkProductId),
            ),
            title=required_text(payload, "title"),
            visual_kind=required_text(payload, "visual_kind"),
            artifact_version=_required_integer(payload, "artifact_version", minimum=1),
            visibility=VisualVisibility(required_text(payload, "visibility")),
            status=VisualStatus(required_text(payload, "status")),
            public_slug=optional_text(payload, "public_slug"),
            public_url=optional_text(payload, "public_url"),
            preview_url=optional_text(payload, "preview_url"),
            canonical_url=required_text(payload, "canonical_url"),
            summary=_optional_string(payload, "summary"),
            content_digest=optional_text(payload, "content_digest"),
            size_bytes=_required_integer(payload, "size_bytes", minimum=0),
            blob_state=VisualBlobState(required_text(payload, "blob_state")),
            source_run_ids=source_run_ids,
            superseded_by_id=cast(
                ArtifactId | None,
                _optional_identifier(payload, "superseded_by_id", ArtifactId),
            ),
            deleted_at=optional_datetime(payload, "deleted_at"),
            published_at=optional_datetime(payload, "published_at"),
            created_at=required_datetime(payload, "created_at"),
            evidence=(
                VisualEvidencePayload.from_wire(cast(JsonValue, payload["evidence"]))
                if payload.get("evidence") is not None
                else None
            ),
        )
        if visual.visibility is VisualVisibility.PUBLIC:
            if visual.public_slug is None or visual.public_url is None:
                raise ValueError("public Visual responses require slug and a safe public URL")
        elif any(value is not None for value in (visual.public_slug, visual.public_url)):
            raise ValueError("organization Visual responses must not expose public URLs")
        return visual

    def to_wire(self) -> JsonObject:
        """Serialize the decoded Visual surface; evidence stays None off-org."""
        return {
            "hosted_artifact_id": str(self.hosted_artifact_id),
            "root_artifact_id": str(self.root_artifact_id),
            "project_id": str(self.project_id),
            "run_id": str(self.run_id) if self.run_id is not None else None,
            "built_by_run_id": (
                str(self.built_by_run_id) if self.built_by_run_id is not None else None
            ),
            "work_product_id": (
                str(self.work_product_id) if self.work_product_id is not None else None
            ),
            "title": self.title,
            "visual_kind": self.visual_kind,
            "artifact_version": self.artifact_version,
            "visibility": self.visibility.value,
            "status": self.status.value,
            "public_slug": self.public_slug,
            "public_url": self.public_url,
            "preview_url": self.preview_url,
            "canonical_url": self.canonical_url,
            "summary": self.summary,
            "content_digest": self.content_digest,
            "size_bytes": self.size_bytes,
            "blob_state": self.blob_state.value,
            "source_run_ids": [str(item) for item in self.source_run_ids],
            "superseded_by_id": (
                str(self.superseded_by_id) if self.superseded_by_id is not None else None
            ),
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at is not None else None,
            "published_at": (
                self.published_at.isoformat() if self.published_at is not None else None
            ),
            "created_at": self.created_at.isoformat(),
            "evidence": self.evidence.to_wire() if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class VisualPage:
    """One cursor-paginated page of Visuals."""

    visuals: tuple[Visual, ...]
    next_cursor: PageCursor | None = None

    @classmethod
    def from_wire(
        cls,
        value: JsonValue,
        *,
        operation_id: str,
    ) -> VisualPage:
        payload = object_value(value, operation_id=operation_id)
        visuals = tuple(
            Visual.from_wire(item)
            for item in array_value(
                cast(JsonValue, payload.get("visuals")),
                operation_id=f"{operation_id}.visuals",
            )
        )
        return cls(visuals=visuals, next_cursor=extract_next_cursor(payload))

    def to_wire(self) -> JsonObject:
        return {
            "visuals": [visual.to_wire() for visual in self.visuals],
            "next_cursor": str(self.next_cursor) if self.next_cursor is not None else None,
        }


@dataclass(frozen=True, slots=True)
class VisualVersions:
    """All durable versions sharing one stable Visual identity."""

    root_artifact_id: ArtifactId
    versions: tuple[Visual, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> VisualVersions:
        payload = object_value(value, operation_id="list_visual_versions")
        return cls(
            root_artifact_id=ArtifactId(required_text(payload, "root_artifact_id")),
            versions=tuple(
                Visual.from_wire(item)
                for item in array_value(
                    cast(JsonValue, payload.get("versions")),
                    operation_id="list_visual_versions.versions",
                )
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "root_artifact_id": str(self.root_artifact_id),
            "versions": [visual.to_wire() for visual in self.versions],
        }


@dataclass(frozen=True, slots=True)
class VisualPatch:
    """Editable Visual metadata; visibility changes use dedicated operations."""

    title: str | None = None
    summary: str | None = None
    visual_kind: str | None = None

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {}
        if self.title is not None:
            title = self.title.strip()
            if not title:
                raise ValueError("title must not be empty when provided")
            payload["title"] = title
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.visual_kind is not None:
            visual_kind = self.visual_kind.strip()
            if not visual_kind:
                raise ValueError("visual_kind must not be empty when provided")
            payload["visual_kind"] = visual_kind
        if not payload:
            raise ValueError("VisualPatch must change at least one field")
        return payload

    @classmethod
    def from_wire(cls, value: JsonValue) -> VisualPatch:
        payload = object_value(value, operation_id="visual_patch")
        return cls(
            title=optional_text(payload, "title"),
            summary=_optional_string(payload, "summary"),
            visual_kind=optional_text(payload, "visual_kind"),
        )


@dataclass(frozen=True, slots=True)
class VisualPromotion:
    """Requested stable public slug for a Visual."""

    slug: str

    def to_wire(self) -> JsonObject:
        slug = self.slug.strip()
        if not slug:
            raise ValueError("slug must not be empty")
        return {"slug": slug}

    @classmethod
    def from_wire(cls, value: JsonValue) -> VisualPromotion:
        return cls(slug=required_text(object_value(value, operation_id="visual_promotion"), "slug"))


ResearchVisual = Visual
ResearchVisualPage = VisualPage
ResearchVisualPatchRequest = VisualPatch
ResearchVisualPromotionRequest = VisualPromotion
ResearchVisualVersions = VisualVersions


__all__ = [
    "ResearchVisual",
    "ResearchVisualPage",
    "ResearchVisualPatchRequest",
    "ResearchVisualPromotionRequest",
    "ResearchVisualVersions",
    "Visual",
    "VisualBlobState",
    "VisualEvidencePayload",
    "VisualPage",
    "VisualPatch",
    "VisualPromotion",
    "VisualStatus",
    "VisualVisibility",
    "VisualVersions",
]
