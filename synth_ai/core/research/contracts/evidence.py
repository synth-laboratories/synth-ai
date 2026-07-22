"""Typed durable evidence contracts for Research swarms.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    array_value,
    object_value,
    optional_text,
    required_bool,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    ArtifactId,
    ProjectId,
    SwarmId,
    WorkProductId,
)


def _exact_object(value: JsonValue, *, label: str, fields: frozenset[str]) -> JsonObject:
    payload = object_value(value, operation_id=label)
    missing = fields - payload.keys()
    extra = payload.keys() - fields
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    return payload


def _non_negative_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class ContentDisposition(StrEnum):
    INLINE = "inline"
    ATTACHMENT = "attachment"


class WorkProductKind(StrEnum):
    REPORT = "report"
    MODEL = "model"
    CANDIDATE = "candidate"
    CONTAINER_EVAL = "container_eval"


class WorkProductStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"
    DELETED = "deleted"


class WorkProductReadiness(StrEnum):
    DOWNLOADABLE = "downloadable"
    IMPORTABLE = "importable"
    VIEWABLE = "viewable"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


class WorkProductArtifactRole(StrEnum):
    PRIMARY = "primary"
    EVIDENCE = "evidence"
    VISUAL = "visual"
    MEDIA = "media"
    SCORECARD = "scorecard"


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    artifact_id: ArtifactId
    artifact_type: str
    title: str | None
    digest: str | None
    created_at: datetime
    content_url: str
    download_url: str

    @classmethod
    def from_wire(cls, value: JsonValue) -> EvidenceArtifact:
        payload = _exact_object(
            value,
            label="swarm evidence artifact",
            fields=frozenset(
                {
                    "artifact_id",
                    "artifact_type",
                    "title",
                    "digest",
                    "created_at",
                    "content_url",
                    "download_url",
                }
            ),
        )
        return cls(
            artifact_id=ArtifactId(required_text(payload, "artifact_id")),
            artifact_type=required_text(payload, "artifact_type"),
            title=optional_text(payload, "title"),
            digest=optional_text(payload, "digest"),
            created_at=required_datetime(payload, "created_at"),
            content_url=required_text(payload, "content_url"),
            download_url=required_text(payload, "download_url"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "title": self.title,
            "digest": self.digest,
            "created_at": self.created_at.isoformat(),
            "content_url": self.content_url,
            "download_url": self.download_url,
        }


@dataclass(frozen=True, slots=True)
class WorkProductArtifactLink:
    artifact_id: ArtifactId
    role: WorkProductArtifactRole
    label: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> WorkProductArtifactLink:
        payload = _exact_object(
            value,
            label="swarm WorkProduct artifact link",
            fields=frozenset({"artifact_id", "role", "label"}),
        )
        return cls(
            artifact_id=ArtifactId(required_text(payload, "artifact_id")),
            role=WorkProductArtifactRole(required_text(payload, "role")),
            label=optional_text(payload, "label"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "artifact_id": self.artifact_id,
            "role": self.role.value,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class WorkProductBlocker:
    code: str
    message: str
    source: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> WorkProductBlocker:
        payload = _exact_object(
            value,
            label="swarm WorkProduct blocker",
            fields=frozenset({"code", "message", "source"}),
        )
        return cls(
            code=required_text(payload, "code"),
            message=required_text(payload, "message"),
            source=optional_text(payload, "source"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class EvidenceWorkProduct:
    work_product_id: WorkProductId
    kind: WorkProductKind
    subtype_kind: str | None
    title: str
    summary: str | None
    status: WorkProductStatus
    readiness: WorkProductReadiness
    artifact_id: ArtifactId | None
    artifact_links: tuple[WorkProductArtifactLink, ...]
    content_url: str
    blocker: WorkProductBlocker | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> EvidenceWorkProduct:
        payload = _exact_object(
            value,
            label="swarm evidence WorkProduct",
            fields=frozenset(
                {
                    "work_product_id",
                    "kind",
                    "subtype_kind",
                    "title",
                    "summary",
                    "status",
                    "readiness",
                    "artifact_id",
                    "artifact_links",
                    "content_url",
                    "blocker",
                    "created_at",
                    "updated_at",
                }
            ),
        )
        artifact_id = optional_text(payload, "artifact_id")
        blocker = payload["blocker"]
        return cls(
            work_product_id=WorkProductId(required_text(payload, "work_product_id")),
            kind=WorkProductKind(required_text(payload, "kind")),
            subtype_kind=optional_text(payload, "subtype_kind"),
            title=required_text(payload, "title"),
            summary=optional_text(payload, "summary"),
            status=WorkProductStatus(required_text(payload, "status")),
            readiness=WorkProductReadiness(required_text(payload, "readiness")),
            artifact_id=ArtifactId(artifact_id) if artifact_id is not None else None,
            artifact_links=tuple(
                WorkProductArtifactLink.from_wire(item)
                for item in array_value(
                    payload["artifact_links"],
                    operation_id="swarm evidence WorkProduct.artifact_links",
                )
            ),
            content_url=required_text(payload, "content_url"),
            blocker=(WorkProductBlocker.from_wire(blocker) if blocker is not None else None),
            created_at=required_datetime(payload, "created_at"),
            updated_at=required_datetime(payload, "updated_at"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "work_product_id": self.work_product_id,
            "kind": self.kind.value,
            "subtype_kind": self.subtype_kind,
            "title": self.title,
            "summary": self.summary,
            "status": self.status.value,
            "readiness": self.readiness.value,
            "artifact_id": self.artifact_id,
            "artifact_links": [link.to_wire() for link in self.artifact_links],
            "content_url": self.content_url,
            "blocker": self.blocker.to_wire() if self.blocker is not None else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class EvidenceFreshness:
    generated_at: datetime
    artifact_count: int
    work_product_count: int
    run_is_terminal: bool

    @classmethod
    def from_wire(cls, value: JsonValue) -> EvidenceFreshness:
        payload = _exact_object(
            value,
            label="swarm evidence freshness",
            fields=frozenset(
                {
                    "generated_at",
                    "artifact_count",
                    "work_product_count",
                    "run_is_terminal",
                }
            ),
        )
        return cls(
            generated_at=required_datetime(payload, "generated_at"),
            artifact_count=_non_negative_int(payload, "artifact_count"),
            work_product_count=_non_negative_int(payload, "work_product_count"),
            run_is_terminal=required_bool(payload, "run_is_terminal"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "generated_at": self.generated_at.isoformat(),
            "artifact_count": self.artifact_count,
            "work_product_count": self.work_product_count,
            "run_is_terminal": self.run_is_terminal,
        }


@dataclass(frozen=True, slots=True)
class SwarmEvidence:
    swarm_id: SwarmId
    project_id: ProjectId
    artifacts: tuple[EvidenceArtifact, ...]
    work_products: tuple[EvidenceWorkProduct, ...]
    freshness: EvidenceFreshness

    def __post_init__(self) -> None:
        if self.freshness.artifact_count != len(self.artifacts):
            raise ValueError("artifact_count must equal the number of artifacts")
        if self.freshness.work_product_count != len(self.work_products):
            raise ValueError("work_product_count must equal the number of WorkProducts")

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmEvidence:
        payload = _exact_object(
            value,
            label="retrieve_swarm_evidence",
            fields=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "project_id",
                    "artifacts",
                    "work_products",
                    "freshness",
                }
            ),
        )
        if payload["schema_version"] != 1:
            raise ValueError("retrieve_swarm_evidence.schema_version must be 1")
        return cls(
            swarm_id=SwarmId(required_text(payload, "run_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            artifacts=tuple(
                EvidenceArtifact.from_wire(item)
                for item in array_value(
                    payload["artifacts"],
                    operation_id="retrieve_swarm_evidence.artifacts",
                )
            ),
            work_products=tuple(
                EvidenceWorkProduct.from_wire(item)
                for item in array_value(
                    payload["work_products"],
                    operation_id="retrieve_swarm_evidence.work_products",
                )
            ),
            freshness=EvidenceFreshness.from_wire(payload["freshness"]),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": 1,
            "run_id": self.swarm_id,
            "project_id": self.project_id,
            "artifacts": [artifact.to_wire() for artifact in self.artifacts],
            "work_products": [item.to_wire() for item in self.work_products],
            "freshness": self.freshness.to_wire(),
        }


__all__ = [
    "ContentDisposition",
    "EvidenceArtifact",
    "EvidenceFreshness",
    "EvidenceWorkProduct",
    "SwarmEvidence",
    "WorkProductArtifactLink",
    "WorkProductArtifactRole",
    "WorkProductBlocker",
    "WorkProductKind",
    "WorkProductReadiness",
    "WorkProductStatus",
]
