"""``client.research.wiki`` — project wiki reads plus proposal intake.

The session client has no wiki bindings today, so these methods call the
backend directly through the SDK's internal ``_request_json`` layer (same
transport, auth, and error mapping as every other session call).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from synth_ai.managed_research.models.factories import _optional_datetime
from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _require_wiki_mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload must be an object")
    return {str(key): value for key, value in payload.items()}


@dataclass(frozen=True)
class WikiPage:
    """One serialized wiki page row (unknown keys preserved in ``raw``)."""

    page_id: str
    project_id: str
    slug: str = ""
    title: str = ""
    summary: str | None = None
    status: str | None = None
    origin: str | None = None
    parent_page_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> WikiPage:
        mapping = _require_wiki_mapping(payload, label="wiki page")

        def _opt_str(key: str) -> str | None:
            value = mapping.get(key)
            return str(value) if value is not None else None

        return cls(
            page_id=str(mapping.get("page_id") or ""),
            project_id=str(mapping.get("project_id") or ""),
            slug=str(mapping.get("slug") or ""),
            title=str(mapping.get("title") or ""),
            summary=_opt_str("summary"),
            status=_opt_str("status"),
            origin=_opt_str("origin"),
            parent_page_id=_opt_str("parent_page_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class WikiEnvelope:
    """Common wiki response envelope: wiki state + wiki project + full payload."""

    state: dict[str, object] = field(default_factory=dict)
    wiki_project: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> WikiEnvelope:
        mapping = _require_wiki_mapping(payload, label="wiki response")
        state = mapping.get("state")
        wiki_project = mapping.get("wiki_project")
        return cls(
            state=_require_wiki_mapping(state, label="wiki state")
            if isinstance(state, Mapping)
            else {},
            wiki_project=_require_wiki_mapping(wiki_project, label="wiki project")
            if isinstance(wiki_project, Mapping)
            else {},
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class WikiPageList:
    """Page listing envelope with typed page rows."""

    state: dict[str, object] = field(default_factory=dict)
    wiki_project: dict[str, object] = field(default_factory=dict)
    pages: tuple[WikiPage, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> WikiPageList:
        base = WikiEnvelope.from_wire(payload)
        raw_pages = base.raw.get("pages")
        pages = tuple(WikiPage.from_wire(item) for item in list(raw_pages or []))
        return cls(
            state=base.state,
            wiki_project=base.wiki_project,
            raw=base.raw,
            pages=pages,
        )


class ResearchWikiPagesAPI:
    """List and read wiki pages."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(self, project_id: str) -> WikiPageList:
        """List wiki pages. Backend route: ``GET /smr/projects/{project_id}/wiki/pages``."""
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/wiki/pages",
        )
        return WikiPageList.from_wire(payload)

    def get(self, project_id: str, page_id_or_slug: str) -> WikiEnvelope:
        """Read one page (with sections/detail merged into ``raw``).

        Backend route: ``GET /smr/projects/{project_id}/wiki/pages/{page_id_or_slug}``.
        """
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/wiki/pages/{page_id_or_slug}",
        )
        return WikiEnvelope.from_wire(payload)


class ResearchWikiContextPackAPI:
    """Preview the retrieval context pack assembled from the wiki."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def preview(self, project_id: str, *, limit: int = 80) -> WikiEnvelope:
        """Preview the context pack (``limit`` 1-200).

        Backend route: ``GET /smr/projects/{project_id}/wiki/context-pack/preview``.
        """
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/wiki/context-pack/preview",
            params={"limit": int(limit)},
        )
        return WikiEnvelope.from_wire(payload)


class ResearchWikiProposalsAPI:
    """List and create wiki change proposals."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        state: str | None = None,
        limit: int = 50,
    ) -> WikiEnvelope:
        """List proposals (optional ``state`` filter, ``limit`` 1-200).

        Backend route: ``GET /smr/projects/{project_id}/wiki/proposals``.
        """
        params: dict[str, Any] = {"limit": int(limit)}
        if state is not None:
            params["state"] = str(state)
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/wiki/proposals",
            params=params,
        )
        return WikiEnvelope.from_wire(payload)

    def create(
        self,
        project_id: str,
        request: Mapping[str, Any] | dict[str, Any],
    ) -> WikiEnvelope:
        """Create a wiki proposal (``SmrWikiProposalCreateRequest`` wire shape).

        Backend route: ``POST /smr/projects/{project_id}/wiki/proposals``.
        """
        payload = self._session._request_json(
            "POST",
            f"/smr/projects/{project_id}/wiki/proposals",
            json_body=dict(request),
        )
        return WikiEnvelope.from_wire(payload)


class ResearchWikiAPI:
    """Project wiki namespace (read-first).

    The backend exposes GET routes for overview, pages, search, context-pack
    preview, and proposals; change-sets, evidence-links, and staleness-signals
    are write-only (POST) today and are intentionally not bound here.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._pages: ResearchWikiPagesAPI | None = None
        self._context_pack: ResearchWikiContextPackAPI | None = None
        self._proposals: ResearchWikiProposalsAPI | None = None

    @property
    def pages(self) -> ResearchWikiPagesAPI:
        """List and read wiki pages."""
        if self._pages is None:
            self._pages = ResearchWikiPagesAPI(self._session)
        return self._pages

    @property
    def context_pack(self) -> ResearchWikiContextPackAPI:
        """Preview the wiki-derived retrieval context pack."""
        if self._context_pack is None:
            self._context_pack = ResearchWikiContextPackAPI(self._session)
        return self._context_pack

    @property
    def proposals(self) -> ResearchWikiProposalsAPI:
        """List and create wiki change proposals."""
        if self._proposals is None:
            self._proposals = ResearchWikiProposalsAPI(self._session)
        return self._proposals

    def overview(self, project_id: str) -> WikiEnvelope:
        """Read the wiki overview. Backend route: ``GET /smr/projects/{project_id}/wiki``."""
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/wiki",
        )
        return WikiEnvelope.from_wire(payload)

    def search(
        self,
        project_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> WikiEnvelope:
        """Search the wiki (``q`` query text, ``limit`` 1-50).

        Backend route: ``GET /smr/projects/{project_id}/wiki/search``.
        """
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/wiki/search",
            params={"q": str(query), "limit": int(limit)},
        )
        return WikiEnvelope.from_wire(payload)


__all__ = [
    "ResearchWikiAPI",
    "ResearchWikiContextPackAPI",
    "ResearchWikiPagesAPI",
    "ResearchWikiProposalsAPI",
    "WikiEnvelope",
    "WikiPage",
    "WikiPageList",
]
