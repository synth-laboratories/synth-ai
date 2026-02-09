"""Adapter protocol definitions for GEPA compatibility."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

RolloutOutput = TypeVar("RolloutOutput")
Trajectory = TypeVar("Trajectory")
DataInst = TypeVar("DataInst")
Candidate = dict[str, str]


@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """Container for evaluating a proposed candidate on a batch of data."""

    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None
    objective_scores: list[dict[str, float]] | None = None


class ProposalFn(Protocol):
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Return a mapping from component names to new component text."""
        ...


class GEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):
    """Integration point between the system under optimization and GEPA."""

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """Run the program defined by candidate on a batch of data."""
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build a small dataset to guide instruction refinement."""
        ...

    propose_new_texts: ProposalFn | None = None
