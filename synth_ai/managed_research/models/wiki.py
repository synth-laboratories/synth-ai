"""Typed synth-wiki project-memory models for Managed Research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    normalized = str(value).strip()
    return normalized or None


def _required_text(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer when provided")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer when provided") from exc


def _optional_mapping(payload: object, *, label: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return {str(key): value for key, value in payload.items()}


def _mapping_list(payload: object, *, label: str) -> list[Mapping[str, object]]:
    if payload is None:
        return []
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    items: list[Mapping[str, object]] = []
    for index, item in enumerate(payload):
        items.append(_require_mapping(item, label=f"{label}[{index}]"))
    return items


@dataclass(frozen=True, slots=True)
class WikiState:
    project_id: str
    org_id: str
    wiki_project_id: str
    lifecycle_status: str
    binding_mode: str | None = None
    provisioned: bool | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WikiState:
        mapping = _require_mapping(payload, label="wiki state")
        return cls(
            project_id=_required_text(mapping, "project_id", label="wiki.state.project_id"),
            org_id=_required_text(mapping, "org_id", label="wiki.state.org_id"),
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.state.wiki_project_id"
            ),
            lifecycle_status=_required_text(
                mapping, "lifecycle_status", label="wiki.state.lifecycle_status"
            ),
            binding_mode=_optional_text(mapping, "binding_mode"),
            provisioned=(
                bool(mapping["provisioned"]) if "provisioned" in mapping else None
            ),
        )


@dataclass(frozen=True, slots=True)
class WikiProject:
    wiki_project_id: str
    org_id: str
    project_id: str
    title: str
    lifecycle_status: str
    root_page_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WikiProject:
        mapping = _require_mapping(payload, label="wiki project")
        return cls(
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.project.wiki_project_id"
            ),
            org_id=_required_text(mapping, "org_id", label="wiki.project.org_id"),
            project_id=_required_text(
                mapping, "project_id", label="wiki.project.project_id"
            ),
            title=_required_text(mapping, "title", label="wiki.project.title"),
            lifecycle_status=_required_text(
                mapping, "lifecycle_status", label="wiki.project.lifecycle_status"
            ),
            root_page_id=_optional_text(mapping, "root_page_id"),
        )


@dataclass(frozen=True, slots=True)
class WikiPage:
    page_id: str
    wiki_project_id: str
    slug: str
    title: str
    status: str
    summary: str | None = None
    order_key: str | None = None
    proposal_count: int | None = None
    staleness_signal_count: int | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WikiPage:
        mapping = _require_mapping(payload, label="wiki page")
        return cls(
            page_id=_required_text(mapping, "page_id", label="wiki.page.page_id"),
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.page.wiki_project_id"
            ),
            slug=_required_text(mapping, "slug", label="wiki.page.slug"),
            title=_required_text(mapping, "title", label="wiki.page.title"),
            status=_required_text(mapping, "status", label="wiki.page.status"),
            summary=_optional_text(mapping, "summary"),
            order_key=_optional_text(mapping, "order_key"),
            proposal_count=_optional_int(mapping, "proposal_count"),
            staleness_signal_count=_optional_int(mapping, "staleness_signal_count"),
        )


@dataclass(frozen=True, slots=True)
class WikiSection:
    section_id: str
    page_id: str
    wiki_project_id: str
    heading: str
    body_markdown: str
    learning_family: str
    learning_kind: str
    status: str
    position: int | None = None
    accepted_at: str | None = None
    accepted_by_type: str | None = None
    accepted_by_id: str | None = None
    stale_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    linkouts: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiSection:
        mapping = _require_mapping(payload, label="wiki section")
        raw_linkouts = mapping.get("linkouts") or []
        linkouts: list[dict[str, str]] = []
        if isinstance(raw_linkouts, Sequence) and not isinstance(
            raw_linkouts, (str, bytes)
        ):
            for item in raw_linkouts:
                if isinstance(item, Mapping):
                    linkouts.append(
                        {
                            "kind": str(item.get("kind") or ""),
                            "id": str(item.get("id") or ""),
                            "path": str(item.get("path") or ""),
                        }
                    )
        return cls(
            section_id=_required_text(
                mapping, "section_id", label="wiki.section.section_id"
            ),
            page_id=_required_text(mapping, "page_id", label="wiki.section.page_id"),
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.section.wiki_project_id"
            ),
            heading=_required_text(mapping, "heading", label="wiki.section.heading"),
            body_markdown=str(mapping.get("body_markdown") or ""),
            learning_family=_required_text(
                mapping, "learning_family", label="wiki.section.learning_family"
            ),
            learning_kind=_required_text(
                mapping, "learning_kind", label="wiki.section.learning_kind"
            ),
            status=_required_text(mapping, "status", label="wiki.section.status"),
            position=_optional_int(mapping, "position"),
            accepted_at=_optional_text(mapping, "accepted_at"),
            accepted_by_type=_optional_text(mapping, "accepted_by_type"),
            accepted_by_id=_optional_text(mapping, "accepted_by_id"),
            stale_reason=_optional_text(mapping, "stale_reason"),
            metadata=_optional_mapping(mapping.get("metadata"), label="wiki.section.metadata"),
            linkouts=linkouts,
        )


@dataclass(frozen=True, slots=True)
class WikiChangeSet:
    changeset_id: str
    wiki_project_id: str
    title: str
    summary: str
    state: str
    source_kind: str
    decision: str | None = None
    decision_rationale: str | None = None
    reviewer_type: str | None = None
    reviewer_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WikiChangeSet:
        mapping = _require_mapping(payload, label="wiki change set")
        return cls(
            changeset_id=_required_text(
                mapping, "changeset_id", label="wiki.changeset.changeset_id"
            ),
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.changeset.wiki_project_id"
            ),
            title=_required_text(mapping, "title", label="wiki.changeset.title"),
            summary=str(mapping.get("summary") or ""),
            state=_required_text(mapping, "state", label="wiki.changeset.state"),
            source_kind=_required_text(
                mapping, "source_kind", label="wiki.changeset.source_kind"
            ),
            decision=_optional_text(mapping, "decision"),
            decision_rationale=_optional_text(mapping, "decision_rationale"),
            reviewer_type=_optional_text(mapping, "reviewer_type"),
            reviewer_id=_optional_text(mapping, "reviewer_id"),
        )


@dataclass(frozen=True, slots=True)
class WikiProposal:
    proposal_id: str
    changeset_id: str
    wiki_project_id: str
    operation: str
    target_kind: str
    review_state: str
    target_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_summary: str | None = None
    confidence: str | None = None
    linkouts: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiProposal:
        mapping = _require_mapping(payload, label="wiki proposal")
        raw_linkouts = mapping.get("linkouts") or []
        linkouts: list[dict[str, str]] = []
        if isinstance(raw_linkouts, Sequence) and not isinstance(
            raw_linkouts, (str, bytes)
        ):
            for item in raw_linkouts:
                if isinstance(item, Mapping):
                    linkouts.append(
                        {
                            "kind": str(item.get("kind") or ""),
                            "id": str(item.get("id") or ""),
                            "path": str(item.get("path") or ""),
                        }
                    )
        return cls(
            proposal_id=_required_text(
                mapping, "proposal_id", label="wiki.proposal.proposal_id"
            ),
            changeset_id=_required_text(
                mapping, "changeset_id", label="wiki.proposal.changeset_id"
            ),
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.proposal.wiki_project_id"
            ),
            operation=_required_text(
                mapping, "operation", label="wiki.proposal.operation"
            ),
            target_kind=_required_text(
                mapping, "target_kind", label="wiki.proposal.target_kind"
            ),
            review_state=_required_text(
                mapping, "review_state", label="wiki.proposal.review_state"
            ),
            target_id=_optional_text(mapping, "target_id"),
            payload=_optional_mapping(mapping.get("payload"), label="wiki.proposal.payload"),
            evidence_summary=_optional_text(mapping, "evidence_summary"),
            confidence=_optional_text(mapping, "confidence"),
            linkouts=linkouts,
        )


@dataclass(frozen=True, slots=True)
class WikiEvidenceLink:
    evidence_link_id: str
    wiki_project_id: str
    target_kind: str
    target_id: str
    source_kind: str
    source_id: str
    role: str
    evidence_ref_id: str | None = None
    quote: str | None = None
    note: str | None = None
    confidence: str | None = None
    linkouts: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiEvidenceLink:
        mapping = _require_mapping(payload, label="wiki evidence link")
        raw_linkouts = mapping.get("linkouts") or []
        linkouts: list[dict[str, str]] = []
        if isinstance(raw_linkouts, Sequence) and not isinstance(
            raw_linkouts, (str, bytes)
        ):
            for item in raw_linkouts:
                if isinstance(item, Mapping):
                    linkouts.append(
                        {
                            "kind": str(item.get("kind") or ""),
                            "id": str(item.get("id") or ""),
                            "path": str(item.get("path") or ""),
                        }
                    )
        return cls(
            evidence_link_id=_required_text(
                mapping, "evidence_link_id", label="wiki.evidence.evidence_link_id"
            ),
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.evidence.wiki_project_id"
            ),
            target_kind=_required_text(
                mapping, "target_kind", label="wiki.evidence.target_kind"
            ),
            target_id=_required_text(
                mapping, "target_id", label="wiki.evidence.target_id"
            ),
            source_kind=_required_text(
                mapping, "source_kind", label="wiki.evidence.source_kind"
            ),
            source_id=_required_text(
                mapping, "source_id", label="wiki.evidence.source_id"
            ),
            role=_required_text(mapping, "role", label="wiki.evidence.role"),
            evidence_ref_id=_optional_text(mapping, "evidence_ref_id"),
            quote=_optional_text(mapping, "quote"),
            note=_optional_text(mapping, "note"),
            confidence=_optional_text(mapping, "confidence"),
            linkouts=linkouts,
        )


@dataclass(frozen=True, slots=True)
class WikiStalenessSignal:
    staleness_signal_id: str
    wiki_project_id: str
    target_kind: str
    target_id: str
    signal_kind: str
    reason: str
    status: str
    source_kind: str
    source_id: str | None = None
    resolved_at: str | None = None
    linkouts: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiStalenessSignal:
        mapping = _require_mapping(payload, label="wiki staleness signal")
        raw_linkouts = mapping.get("linkouts") or []
        linkouts: list[dict[str, str]] = []
        if isinstance(raw_linkouts, Sequence) and not isinstance(
            raw_linkouts, (str, bytes)
        ):
            for item in raw_linkouts:
                if isinstance(item, Mapping):
                    linkouts.append(
                        {
                            "kind": str(item.get("kind") or ""),
                            "id": str(item.get("id") or ""),
                            "path": str(item.get("path") or ""),
                        }
                    )
        return cls(
            staleness_signal_id=_required_text(
                mapping,
                "staleness_signal_id",
                label="wiki.staleness.staleness_signal_id",
            ),
            wiki_project_id=_required_text(
                mapping, "wiki_project_id", label="wiki.staleness.wiki_project_id"
            ),
            target_kind=_required_text(
                mapping, "target_kind", label="wiki.staleness.target_kind"
            ),
            target_id=_required_text(
                mapping, "target_id", label="wiki.staleness.target_id"
            ),
            signal_kind=_required_text(
                mapping, "signal_kind", label="wiki.staleness.signal_kind"
            ),
            reason=_required_text(mapping, "reason", label="wiki.staleness.reason"),
            status=_required_text(mapping, "status", label="wiki.staleness.status"),
            source_kind=_required_text(
                mapping, "source_kind", label="wiki.staleness.source_kind"
            ),
            source_id=_optional_text(mapping, "source_id"),
            resolved_at=_optional_text(mapping, "resolved_at"),
            linkouts=linkouts,
        )


@dataclass(frozen=True, slots=True)
class WikiOverview:
    state: WikiState
    wiki_project: WikiProject
    page_count: int | None = None
    section_count: int | None = None
    open_proposal_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> WikiOverview:
        mapping = _require_mapping(payload, label="wiki overview")
        counts = mapping.get("counts")
        count_map = (
            _require_mapping(counts, label="wiki.overview.counts")
            if counts is not None
            else {}
        )
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            page_count=_optional_int(count_map, "pages"),
            section_count=_optional_int(count_map, "sections"),
            open_proposal_count=_optional_int(count_map, "open_proposals"),
            raw={str(key): value for key, value in mapping.items()},
        )


@dataclass(frozen=True, slots=True)
class WikiPageDetail:
    state: WikiState
    wiki_project: WikiProject
    page: WikiPage
    sections: list[WikiSection] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiPageDetail:
        mapping = _require_mapping(payload, label="wiki page detail")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            page=WikiPage.from_wire(mapping.get("page")),
            sections=[
                WikiSection.from_wire(item)
                for item in _mapping_list(mapping.get("sections"), label="wiki.page.sections")
            ],
        )


@dataclass(frozen=True, slots=True)
class WikiPagesResult:
    state: WikiState
    wiki_project: WikiProject
    pages: list[WikiPage] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiPagesResult:
        mapping = _require_mapping(payload, label="wiki pages result")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            pages=[
                WikiPage.from_wire(item)
                for item in _mapping_list(mapping.get("pages"), label="wiki.pages")
            ],
        )


@dataclass(frozen=True, slots=True)
class WikiSearchHit:
    kind: str
    score_source: str | None
    page: WikiPage | None = None
    section: WikiSection | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WikiSearchHit:
        mapping = _require_mapping(payload, label="wiki search hit")
        page_payload = mapping.get("page")
        section_payload = mapping.get("section")
        return cls(
            kind=_required_text(mapping, "kind", label="wiki.search.kind"),
            score_source=_optional_text(mapping, "score_source"),
            page=WikiPage.from_wire(page_payload) if page_payload is not None else None,
            section=(
                WikiSection.from_wire(section_payload)
                if section_payload is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class WikiSearchResult:
    state: WikiState
    wiki_project: WikiProject
    query: str
    items: list[WikiSearchHit] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiSearchResult:
        mapping = _require_mapping(payload, label="wiki search result")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            query=str(mapping.get("query") or ""),
            items=[
                WikiSearchHit.from_wire(item)
                for item in _mapping_list(mapping.get("items"), label="wiki.search.items")
            ],
        )


@dataclass(frozen=True, slots=True)
class WikiContextPack:
    state: WikiState
    wiki_project: WikiProject
    sections: list[WikiSection] = field(default_factory=list)
    warnings: list[Any] = field(default_factory=list)
    included_statuses: list[str] = field(default_factory=list)
    excluded_statuses: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> WikiContextPack:
        mapping = _require_mapping(payload, label="wiki context pack")
        included = mapping.get("included_statuses") or []
        excluded = mapping.get("excluded_statuses") or []
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            sections=[
                WikiSection.from_wire(item)
                for item in _mapping_list(
                    mapping.get("sections"), label="wiki.context.sections"
                )
            ],
            warnings=list(mapping.get("warnings") or []),
            included_statuses=[str(item) for item in included],
            excluded_statuses=[str(item) for item in excluded],
            raw={str(key): value for key, value in mapping.items()},
        )


@dataclass(frozen=True, slots=True)
class WikiProposalsResult:
    state: WikiState
    wiki_project: WikiProject
    proposals: list[WikiProposal] = field(default_factory=list)
    change_sets: list[WikiChangeSet] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WikiProposalsResult:
        mapping = _require_mapping(payload, label="wiki proposals result")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            proposals=[
                WikiProposal.from_wire(item)
                for item in _mapping_list(
                    mapping.get("proposals"), label="wiki.proposals"
                )
            ],
            change_sets=[
                WikiChangeSet.from_wire(item)
                for item in _mapping_list(
                    mapping.get("change_sets"), label="wiki.change_sets"
                )
            ],
        )


@dataclass(frozen=True, slots=True)
class WikiProposalMutationResult:
    state: WikiState
    wiki_project: WikiProject
    change_set: WikiChangeSet
    proposal: WikiProposal
    applied: dict[str, Any] | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WikiProposalMutationResult:
        mapping = _require_mapping(payload, label="wiki proposal mutation")
        applied = mapping.get("applied")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            change_set=WikiChangeSet.from_wire(mapping.get("change_set")),
            proposal=WikiProposal.from_wire(mapping.get("proposal")),
            applied=(
                _optional_mapping(applied, label="wiki.proposal.applied")
                if applied is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class WikiChangeSetCreateResult:
    state: WikiState
    wiki_project: WikiProject
    change_set: WikiChangeSet

    @classmethod
    def from_wire(cls, payload: object) -> WikiChangeSetCreateResult:
        mapping = _require_mapping(payload, label="wiki change set create")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            change_set=WikiChangeSet.from_wire(mapping.get("change_set")),
        )


@dataclass(frozen=True, slots=True)
class WikiEvidenceCreateResult:
    state: WikiState
    wiki_project: WikiProject
    evidence_link: WikiEvidenceLink

    @classmethod
    def from_wire(cls, payload: object) -> WikiEvidenceCreateResult:
        mapping = _require_mapping(payload, label="wiki evidence create")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            evidence_link=WikiEvidenceLink.from_wire(mapping.get("evidence_link")),
        )


@dataclass(frozen=True, slots=True)
class WikiStalenessCreateResult:
    state: WikiState
    wiki_project: WikiProject
    staleness_signal: WikiStalenessSignal

    @classmethod
    def from_wire(cls, payload: object) -> WikiStalenessCreateResult:
        mapping = _require_mapping(payload, label="wiki staleness create")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            staleness_signal=WikiStalenessSignal.from_wire(
                mapping.get("staleness_signal")
            ),
        )


@dataclass(frozen=True, slots=True)
class WikiStalenessAcceptResult:
    state: WikiState
    wiki_project: WikiProject
    staleness_signal: WikiStalenessSignal
    applied_target: dict[str, Any] | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WikiStalenessAcceptResult:
        mapping = _require_mapping(payload, label="wiki staleness accept")
        applied = mapping.get("applied_target")
        return cls(
            state=WikiState.from_wire(mapping.get("state")),
            wiki_project=WikiProject.from_wire(mapping.get("wiki_project")),
            staleness_signal=WikiStalenessSignal.from_wire(
                mapping.get("staleness_signal")
            ),
            applied_target=(
                _optional_mapping(applied, label="wiki.staleness.applied_target")
                if applied is not None
                else None
            ),
        )


__all__ = [
    "WikiChangeSet",
    "WikiChangeSetCreateResult",
    "WikiContextPack",
    "WikiEvidenceCreateResult",
    "WikiEvidenceLink",
    "WikiOverview",
    "WikiPage",
    "WikiPageDetail",
    "WikiPagesResult",
    "WikiProject",
    "WikiProposal",
    "WikiProposalMutationResult",
    "WikiProposalsResult",
    "WikiSearchHit",
    "WikiSearchResult",
    "WikiSection",
    "WikiStalenessAcceptResult",
    "WikiStalenessCreateResult",
    "WikiStalenessSignal",
    "WikiState",
]
