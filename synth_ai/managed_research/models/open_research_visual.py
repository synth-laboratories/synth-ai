"""Open Research visual artifact contract models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

OPEN_RESEARCH_VISUAL_SCHEMA_VERSION = "open_research_visual.v1"


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


@dataclass(frozen=True)
class OpenResearchVisualClaim:
    id: str
    text: str
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> OpenResearchVisualClaim:
        return cls(
            id=str(payload["id"]),
            text=str(payload["text"]),
            evidence_refs=_string_list(payload.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class OpenResearchVisualManifest:
    schema_version: str
    kind: str
    track: str
    title: str
    summary: str
    data_path: str
    findings_path: str
    entrypoint_path: str | None = None
    preview_image_path: str | None = None
    claims: list[OpenResearchVisualClaim] = field(default_factory=list)
    grader_inputs: dict[str, object] = field(default_factory=dict)
    render: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> OpenResearchVisualManifest:
        schema_version = str(payload["schema_version"])
        if schema_version != OPEN_RESEARCH_VISUAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported Open Research visual schema: {schema_version}")
        raw_claims = payload.get("claims", [])
        claims = [
            OpenResearchVisualClaim.from_wire(item)
            for item in raw_claims
            if isinstance(item, Mapping)
        ]

        return cls(
            schema_version=schema_version,
            kind=str(payload["kind"]),
            track=str(payload["track"]),
            title=str(payload["title"]),
            summary=str(payload["summary"]),
            data_path=str(payload["data_path"]),
            findings_path=str(payload["findings_path"]),
            entrypoint_path=_optional_string(payload.get("entrypoint_path")),
            preview_image_path=_optional_string(payload.get("preview_image_path")),
            claims=claims,
            grader_inputs=dict(payload.get("grader_inputs", {}))  # type: ignore[arg-type]
            if isinstance(payload.get("grader_inputs"), Mapping)
            else {},
            render=dict(payload.get("render", {}))  # type: ignore[arg-type]
            if isinstance(payload.get("render"), Mapping)
            else {},
        )


def is_open_research_visual_manifest(payload: Mapping[str, Any]) -> bool:
    return payload.get("schema_version") == OPEN_RESEARCH_VISUAL_SCHEMA_VERSION


__all__ = [
    "OPEN_RESEARCH_VISUAL_SCHEMA_VERSION",
    "OpenResearchVisualClaim",
    "OpenResearchVisualManifest",
    "is_open_research_visual_manifest",
]
