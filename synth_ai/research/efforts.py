"""``client.research.efforts`` — graduate Runs into persistent Efforts (alpha)."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import List

from synth_ai.managed_research.models.factories import Effort, GraduationProposal
from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchEffortsProposalsAPI:
    """List Gardener-authored graduation proposals for a project."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(self, project_id: str) -> List[GraduationProposal]:
        """Return graduation proposals suggesting Runs to promote into Efforts.

        Args:
            project_id: Owning project id.

        Returns:
            List of ``GraduationProposal``, each pairing a suggested Effort name
            with the Run ids the Gardener recommends graduating.
        """
        return self._session.efforts.list_graduation_proposals(project_id)


class ResearchEffortsAPI:
    """Graduate runs into persistent Research Factory Efforts.

    Nested namespace: ``proposals`` (Gardener-authored graduation proposals).
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._proposals: ResearchEffortsProposalsAPI | None = None

    @property
    def proposals(self) -> ResearchEffortsProposalsAPI:
        """List graduation proposals for a project."""
        if self._proposals is None:
            self._proposals = ResearchEffortsProposalsAPI(self._session)
        return self._proposals

    def from_swarms(
        self,
        project_id: str,
        name: str,
        swarm_ids: Iterable[str],
        factory_id: str | None = None,
    ) -> Effort:
        """Graduate a set of Managed Swarms into a persistent Effort.

        Args:
            project_id: Project that owns the swarms and the new Effort.
            name: Human-readable Effort name.
            swarm_ids: Swarm ids to link to the new Effort.
            factory_id: Optional Factory to own the Effort.

        Returns:
            The created ``Effort``.

        Example:
            proposals = research.efforts.proposals.list(project_id)
            effort = research.efforts.from_swarms(
                project_id,
                proposals[0].suggested_name,
                proposals[0].run_ids,
            )
        """
        return self._session.efforts.from_runs(
            project_id=project_id,
            name=name,
            run_ids=swarm_ids,
            factory_id=factory_id,
        )

    def from_runs(
        self,
        project_id: str,
        name: str,
        run_ids: Iterable[str],
        factory_id: str | None = None,
    ) -> Effort:
        """Deprecated alias for :meth:`from_swarms` (the public noun is Swarm)."""
        warnings.warn(
            "efforts.from_runs is deprecated; use efforts.from_swarms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.from_swarms(
            project_id,
            name,
            run_ids,
            factory_id=factory_id,
        )


__all__ = ["ResearchEffortsAPI", "ResearchEffortsProposalsAPI"]
