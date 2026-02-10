"""GEPA result container for compatibility mode."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Generic

from .adapter import RolloutOutput
from .data_loader import DataId
from .state import ProgramIdx


@dataclass(frozen=True)
class GEPAResult(Generic[RolloutOutput, DataId]):
    """Immutable snapshot of a GEPA run with convenience accessors."""

    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[DataId, float]]
    per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
    discovery_eval_counts: list[int]
    val_aggregate_subscores: list[dict[str, float]] | None = None
    per_objective_best_candidates: dict[str, set[ProgramIdx]] | None = None
    objective_pareto_front: dict[str, float] | None = None
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 2

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def num_val_instances(self) -> int:
        return len(self.per_val_instance_best_candidates)

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda i: scores[i])

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [dict(cand.items()) for cand in self.candidates],
            "parents": self.parents,
            "val_aggregate_scores": self.val_aggregate_scores,
            "val_subscores": self.val_subscores,
            "best_outputs_valset": self.best_outputs_valset,
            "per_val_instance_best_candidates": {
                val_id: list(front)
                for val_id, front in self.per_val_instance_best_candidates.items()
            },
            "val_aggregate_subscores": self.val_aggregate_subscores,
            "per_objective_best_candidates": (
                {k: list(v) for k, v in self.per_objective_best_candidates.items()}
                if self.per_objective_best_candidates is not None
                else None
            ),
            "objective_pareto_front": self.objective_pareto_front,
            "discovery_eval_counts": self.discovery_eval_counts,
            "total_metric_calls": self.total_metric_calls,
            "num_full_val_evals": self.num_full_val_evals,
            "run_dir": self.run_dir,
            "seed": self.seed,
            "best_idx": self.best_idx,
            "validation_schema_version": GEPAResult._VALIDATION_SCHEMA_VERSION,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GEPAResult[RolloutOutput, DataId]:
        version = data.get("validation_schema_version") or 0
        if version > GEPAResult._VALIDATION_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported GEPAResult validation schema version "
                f"{version}; max supported is {GEPAResult._VALIDATION_SCHEMA_VERSION}"
            )
        if version <= 1:
            return GEPAResult._migrate_from_dict_v0(data)
        return GEPAResult._from_dict_v2(data)

    @staticmethod
    def _common_kwargs_from_dict(data: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidates": [dict(candidate) for candidate in data.get("candidates", [])],
            "parents": [list(parent_row) for parent_row in data.get("parents", [])],
            "val_aggregate_scores": list(data.get("val_aggregate_scores", [])),
            "discovery_eval_counts": list(data.get("discovery_eval_counts", [])),
            "total_metric_calls": data.get("total_metric_calls"),
            "num_full_val_evals": data.get("num_full_val_evals"),
            "run_dir": data.get("run_dir"),
            "seed": data.get("seed"),
        }

    @staticmethod
    def _migrate_from_dict_v0(data: dict[str, Any]) -> GEPAResult[RolloutOutput, DataId]:
        kwargs = GEPAResult._common_kwargs_from_dict(data)
        kwargs["val_subscores"] = [
            dict(enumerate(scores)) for scores in data.get("val_subscores", [])
        ]
        kwargs["per_val_instance_best_candidates"] = {
            idx: set(front)
            for idx, front in enumerate(data.get("per_val_instance_best_candidates", []))
        }
        best_outputs_valset = data.get("best_outputs_valset")
        if best_outputs_valset is not None:
            kwargs["best_outputs_valset"] = {
                idx: [(program_idx, output) for program_idx, output in outputs]
                for idx, outputs in enumerate(best_outputs_valset)
            }
        else:
            kwargs["best_outputs_valset"] = None
        return GEPAResult(**kwargs)

    @staticmethod
    def _from_dict_v2(data: dict[str, Any]) -> GEPAResult[RolloutOutput, DataId]:
        kwargs = GEPAResult._common_kwargs_from_dict(data)
        kwargs["val_subscores"] = [dict(scores) for scores in data.get("val_subscores", [])]
        per_val_instance_best_candidates_data = data.get("per_val_instance_best_candidates", {})
        kwargs["per_val_instance_best_candidates"] = {
            val_id: set(candidates_on_front)
            for val_id, candidates_on_front in per_val_instance_best_candidates_data.items()
        }
        best_outputs_valset = data.get("best_outputs_valset")
        if best_outputs_valset is not None:
            kwargs["best_outputs_valset"] = {
                val_id: [(program_idx, output) for program_idx, output in outputs]
                for val_id, outputs in best_outputs_valset.items()
            }
        else:
            kwargs["best_outputs_valset"] = None
        val_aggregate_subscores = data.get("val_aggregate_subscores")
        kwargs["val_aggregate_subscores"] = (
            [dict(scores) for scores in val_aggregate_subscores]
            if val_aggregate_subscores is not None
            else None
        )
        per_objective_best_candidates = data.get("per_objective_best_candidates")
        if per_objective_best_candidates is not None:
            kwargs["per_objective_best_candidates"] = {
                objective: set(program_indices)
                for objective, program_indices in per_objective_best_candidates.items()
            }
        else:
            kwargs["per_objective_best_candidates"] = None
        objective_pareto_front = data.get("objective_pareto_front")
        kwargs["objective_pareto_front"] = (
            dict(objective_pareto_front) if objective_pareto_front is not None else None
        )
        return GEPAResult(**kwargs)
