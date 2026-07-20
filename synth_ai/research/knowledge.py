"""Project notes and org knowledge facade namespaces (``client.research``)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from synth_ai.managed_research.models.factories import _optional_datetime
from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _require_notes_mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload must be an object")
    return {str(key): value for key, value in payload.items()}


@dataclass(frozen=True)
class ProjectNotes:
    """Typed ``SmrProjectNotesResponse`` (project_id, notes, updated_at)."""

    project_id: str
    notes: str | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectNotes:
        mapping = _require_notes_mapping(payload, label="project notes")
        notes = mapping.get("notes")
        return cls(
            project_id=str(mapping.get("project_id") or ""),
            notes=str(notes) if notes is not None else None,
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class OrgKnowledge:
    """Typed ``SmrOrgKnowledgeResponse`` (org_id, content, updated_at)."""

    org_id: str
    content: str = ""
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> OrgKnowledge:
        mapping = _require_notes_mapping(payload, label="org knowledge")
        return cls(
            org_id=str(mapping.get("org_id") or ""),
            content=str(mapping.get("content") or ""),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


class ResearchProjectsNotesAPI:
    """Durable free-form project notes."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self, project_id: str) -> ProjectNotes:
        """Read project notes. Backend route: ``GET /smr/projects/{project_id}/notes``."""
        return ProjectNotes.from_wire(self._session.get_project_notes(project_id))

    def set(self, project_id: str, notes: str) -> ProjectNotes:
        """Replace project notes. Backend route: ``PUT /smr/projects/{project_id}/notes``."""
        return ProjectNotes.from_wire(self._session.set_project_notes(project_id, notes))

    def append(self, project_id: str, text: str) -> ProjectNotes:
        """Append to project notes.

        Backend route: ``POST /smr/projects/{project_id}/notes/append``.
        """
        return ProjectNotes.from_wire(self._session.append_project_notes(project_id, text))


class ResearchKnowledgeAPI:
    """Org-level durable knowledge document."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self) -> OrgKnowledge:
        """Read org knowledge. Backend route: ``GET /smr/org/knowledge``."""
        return OrgKnowledge.from_wire(self._session.get_org_knowledge())

    def set(self, content: str) -> OrgKnowledge:
        """Replace org knowledge. Backend route: ``PUT /smr/org/knowledge``."""
        return OrgKnowledge.from_wire(self._session.set_org_knowledge(content))


__all__ = [
    "OrgKnowledge",
    "ProjectNotes",
    "ResearchKnowledgeAPI",
    "ResearchProjectsNotesAPI",
]
