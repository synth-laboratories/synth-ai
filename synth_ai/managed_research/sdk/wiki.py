"""Project wiki SDK namespace over `/smr/projects/{id}/wiki/*`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.wiki import (
    WikiChangeSetCreateResult,
    WikiContextPack,
    WikiEvidenceCreateResult,
    WikiOverview,
    WikiPageDetail,
    WikiPagesResult,
    WikiProposalMutationResult,
    WikiProposalsResult,
    WikiSearchResult,
    WikiStalenessAcceptResult,
    WikiStalenessCreateResult,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


class WikiAPI(_ClientNamespace):
    def get(self, project_id: str) -> WikiOverview:
        return WikiOverview.from_wire(self._client.get_project_wiki(project_id))

    def list_pages(self, project_id: str) -> WikiPagesResult:
        return WikiPagesResult.from_wire(self._client.list_project_wiki_pages(project_id))

    def get_page(self, project_id: str, page_id_or_slug: str) -> WikiPageDetail:
        return WikiPageDetail.from_wire(
            self._client.get_project_wiki_page(project_id, page_id_or_slug)
        )

    def search(
        self,
        project_id: str,
        *,
        query: str,
        limit: int = 20,
    ) -> WikiSearchResult:
        return WikiSearchResult.from_wire(
            self._client.search_project_wiki(project_id, query=query, limit=limit)
        )

    def context_pack(
        self,
        project_id: str,
        *,
        limit: int = 80,
    ) -> WikiContextPack:
        return WikiContextPack.from_wire(
            self._client.preview_project_wiki_context_pack(project_id, limit=limit)
        )

    def list_proposals(
        self,
        project_id: str,
        *,
        state: str | None = None,
        limit: int = 50,
    ) -> WikiProposalsResult:
        return WikiProposalsResult.from_wire(
            self._client.list_project_wiki_proposals(
                project_id,
                state=state,
                limit=limit,
            )
        )

    def create_change_set(
        self,
        project_id: str,
        *,
        title: str,
        summary: str = "",
        source_kind: str = "manual",
        source_run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WikiChangeSetCreateResult:
        return WikiChangeSetCreateResult.from_wire(
            self._client.create_project_wiki_change_set(
                project_id,
                title=title,
                summary=summary,
                source_kind=source_kind,
                source_run_id=source_run_id,
                metadata=metadata,
            )
        )

    def create_proposal(
        self,
        project_id: str,
        *,
        operation: str,
        target_kind: str,
        payload: Mapping[str, Any],
        title: str = "Wiki change proposal",
        summary: str = "",
        source_kind: str = "manual",
        changeset_id: str | None = None,
        target_id: str | None = None,
        evidence_summary: str | None = None,
        confidence: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WikiProposalMutationResult:
        return WikiProposalMutationResult.from_wire(
            self._client.create_project_wiki_proposal(
                project_id,
                operation=operation,
                target_kind=target_kind,
                payload=payload,
                title=title,
                summary=summary,
                source_kind=source_kind,
                changeset_id=changeset_id,
                target_id=target_id,
                evidence_summary=evidence_summary,
                confidence=confidence,
                metadata=metadata,
            )
        )

    def accept_proposal(
        self,
        project_id: str,
        proposal_id: str,
        *,
        reviewer_type: str = "operator",
        reviewer_id: str | None = None,
        decision_rationale: str | None = None,
    ) -> WikiProposalMutationResult:
        return WikiProposalMutationResult.from_wire(
            self._client.accept_project_wiki_proposal(
                project_id,
                proposal_id,
                reviewer_type=reviewer_type,
                reviewer_id=reviewer_id,
                decision_rationale=decision_rationale,
            )
        )

    def reject_proposal(
        self,
        project_id: str,
        proposal_id: str,
        *,
        reviewer_type: str = "operator",
        reviewer_id: str | None = None,
        decision_rationale: str | None = None,
    ) -> WikiProposalMutationResult:
        return WikiProposalMutationResult.from_wire(
            self._client.reject_project_wiki_proposal(
                project_id,
                proposal_id,
                reviewer_type=reviewer_type,
                reviewer_id=reviewer_id,
                decision_rationale=decision_rationale,
            )
        )

    def attach_evidence(
        self,
        project_id: str,
        *,
        target_kind: str,
        target_id: str,
        evidence_ref_id: str | None = None,
        source_kind: str = "manual",
        source_id: str | None = None,
        role: str = "supporting",
        quote: str | None = None,
        note: str | None = None,
        confidence: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WikiEvidenceCreateResult:
        return WikiEvidenceCreateResult.from_wire(
            self._client.create_project_wiki_evidence_link(
                project_id,
                target_kind=target_kind,
                target_id=target_id,
                evidence_ref_id=evidence_ref_id,
                source_kind=source_kind,
                source_id=source_id,
                role=role,
                quote=quote,
                note=note,
                confidence=confidence,
                metadata=metadata,
            )
        )

    def mark_stale(
        self,
        project_id: str,
        *,
        target_kind: str,
        target_id: str,
        reason: str,
        signal_kind: str = "stale",
        source_kind: str = "manual",
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WikiStalenessCreateResult:
        return WikiStalenessCreateResult.from_wire(
            self._client.create_project_wiki_staleness_signal(
                project_id,
                target_kind=target_kind,
                target_id=target_id,
                reason=reason,
                signal_kind=signal_kind,
                source_kind=source_kind,
                source_id=source_id,
                metadata=metadata,
            )
        )

    def accept_staleness(
        self,
        project_id: str,
        staleness_signal_id: str,
        *,
        reviewer_type: str = "operator",
        reviewer_id: str | None = None,
    ) -> WikiStalenessAcceptResult:
        return WikiStalenessAcceptResult.from_wire(
            self._client.accept_project_wiki_staleness_signal(
                project_id,
                staleness_signal_id,
                reviewer_type=reviewer_type,
                reviewer_id=reviewer_id,
            )
        )


__all__ = ["WikiAPI"]
