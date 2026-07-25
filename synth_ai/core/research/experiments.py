"""``client.research.experiments`` — typed experiment observability projections."""

from __future__ import annotations

from collections.abc import Iterable

from synth_ai.core.research.contracts.factory_operations import (
    ExperimentBundle,
    ExperimentComparison,
    ExperimentHistory,
)
from synth_ai.core.research.session.client import ResearchSession


class ResearchExperimentsAPI:
    """Owner-assembled experiment bundles, history, and comparisons.

    All projections are backend-owned; this namespace decodes the wire payloads
    into the frozen dataclasses in ``synth_ai.core.research.contracts.factory_operations``
    (each keeps the full payload in its ``raw`` field where applicable).
    """

    def __init__(self, session: ResearchSession) -> None:
        self._session = session

    def bundle(self, project_id: str, experiment_id: str) -> ExperimentBundle:
        """Read one experiment bundle.

        Backend route: ``GET /smr/projects/{project_id}/experiments/{experiment_id}/bundle``.
        """
        return ExperimentBundle.from_wire(
            self._session.get_experiment_bundle(project_id, experiment_id)
        )

    def history(self, project_id: str, *, limit: int = 50) -> ExperimentHistory:
        """Read experiment history with missing-evidence alerts.

        Backend route: ``GET /smr/projects/{project_id}/experiment-bundles``.
        """
        return ExperimentHistory.from_wire(
            self._session.get_experiment_history(project_id, limit=limit)
        )

    def compare(
        self,
        project_id: str,
        experiment_ids: Iterable[str],
    ) -> ExperimentComparison:
        """Compare accepted experiments across matching evidence dimensions.

        Backend route: ``GET /smr/projects/{project_id}/experiments/compare``
        (query ``experiment_ids``, 2-20 ids).
        """
        return ExperimentComparison.from_wire(
            self._session.compare_experiments(project_id, experiment_ids)
        )


__all__ = ["ResearchExperimentsAPI"]
