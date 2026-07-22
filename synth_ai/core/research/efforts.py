"""``client.research.efforts`` — persistent Factory Efforts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, List

from synth_ai.core.research._legacy.models.factories import (
    Effort,
    EffortCreateRequest,
    GraduationProposal,
)
from synth_ai.core.research._legacy.sdk.client import (
    ManagedResearchClient as LegacyResearchSession,
)


class EffortProposals:
    """List Gardener-authored graduation proposals for a project."""

    def __init__(self, session: LegacyResearchSession) -> None:
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


class Efforts:
    """Persistent Research Factory Efforts.

    Nested namespace: ``proposals`` (Gardener-authored graduation proposals).
    Prefer ``research.factories.create_effort`` / ``list_efforts`` when the
    Factory is already known; use this namespace for Effort-id-first workflows.
    """

    def __init__(self, session: LegacyResearchSession) -> None:
        self._session = session
        self._proposals: EffortProposals | None = None

    @property
    def proposals(self) -> EffortProposals:
        """List graduation proposals for a project."""
        if self._proposals is None:
            self._proposals = EffortProposals(self._session)
        return self._proposals

    def create(
        self,
        request: EffortCreateRequest | Mapping[str, Any] | dict[str, Any],
    ) -> Effort:
        """Create an Effort from a typed request body.

        Args:
            request: ``EffortCreateRequest`` or equivalent mapping (must include
                ``factory_id`` / ``project_id`` as required by the backend).

        Returns:
            The created ``Effort``.
        """
        return self._session.efforts.create(request)

    def get(self, effort_id: str) -> Effort:
        """Fetch one Effort by id.

        Args:
            effort_id: Effort to fetch.

        Returns:
            ``Effort`` record.
        """
        return self._session.efforts.get(effort_id)

    def pause(self, effort_id: str) -> Effort:
        """Pause an Effort (stop new wakes for this Effort).

        Args:
            effort_id: Effort to pause.

        Returns:
            Updated ``Effort``.
        """
        return self._session.efforts.pause(effort_id)

    def resume(self, effort_id: str) -> Effort:
        """Resume a paused Effort.

        Args:
            effort_id: Effort to resume.

        Returns:
            Updated ``Effort``.
        """
        return self._session.efforts.resume(effort_id)

    def launch(
        self,
        effort_id: str,
        objective: str | None = None,
        *,
        run_kind: str = "research",
        **kwargs: Any,
    ):
        """Launch a Run bound to an Effort.

        Args:
            effort_id: Effort that owns the Run.
            objective: Optional objective override; defaults to the Effort
                hypothesis/name.
            run_kind: Run kind (default ``research``).
            **kwargs: Forwarded to the Runs start path.

        Returns:
            A run handle from the backing Managed Research client.
        """
        return self._session.efforts.launch(
            effort_id,
            objective,
            run_kind=run_kind,
            **kwargs,
        )

    def from_runs(
        self,
        project_id: str,
        name: str,
        run_ids: Iterable[str],
        factory_id: str | None = None,
    ) -> Effort:
        """Graduate a set of Runs into a persistent Effort.

        Args:
            project_id: Project that owns the runs and the new Effort.
            name: Human-readable Effort name.
            run_ids: Run ids to link to the new Effort.
            factory_id: Optional Factory to own the Effort.

        Returns:
            The created ``Effort``.

        Example:
            proposals = research.efforts.proposals.list(project_id)
            effort = research.efforts.from_runs(
                project_id,
                proposals[0].suggested_name,
                proposals[0].run_ids,
            )
        """
        return self._session.efforts.from_runs(
            project_id=project_id,
            name=name,
            run_ids=run_ids,
            factory_id=factory_id,
        )


__all__ = ["EffortProposals", "Efforts"]
