"""GEPA-compatible local optimizer facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from prompt_opt.adapters.synth_offline import SynthOfflineLearningAdapter

try:
    from synth_ai.gepa.core.result import GEPAResult as SynthGEPAResult
except Exception:  # pragma: no cover - allows standalone use
    SynthGEPAResult = None


class LocalGEPAAdapterProtocol(Protocol):
    """Protocol compatible with GEPA adapter-style calls."""

    def evaluate(
        self,
        batch: list[Mapping[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ...


@dataclass(frozen=True)
class LocalGEPAResult:
    """Minimal GEPA-style result when synth_ai is unavailable."""

    candidates: list[dict[str, str]]
    parents: list[list[int | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[int, float]]
    per_val_instance_best_candidates: dict[int, set[int]]
    discovery_eval_counts: list[int]
    total_metric_calls: int

    @property
    def best_idx(self) -> int:
        return max(range(len(self.val_aggregate_scores)), key=self.val_aggregate_scores.__getitem__)

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]


def _score_candidate(
    adapter: LocalGEPAAdapterProtocol,
    dataset: list[Mapping[str, Any]],
    candidate: dict[str, str],
) -> tuple[float, dict[int, float]]:
    eval_batch = adapter.evaluate(dataset, candidate, capture_traces=False)
    per_item = {idx: float(score) for idx, score in enumerate(eval_batch.scores)}
    aggregate = sum(per_item.values()) / max(1, len(per_item))
    return aggregate, per_item


def _mutate_candidate(base: dict[str, str], trial_idx: int) -> dict[str, str]:
    mutated = dict(base)
    for key, text in mutated.items():
        suffix = [
            "Be concise and factual.",
            "Reason step-by-step, then answer.",
            "Prefer deterministic output format.",
            "State assumptions explicitly.",
        ][trial_idx % 4]
        mutated[key] = f"{text.rstrip()}\n\n{suffix}"
    return mutated


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[Mapping[str, Any]],
    valset: list[Mapping[str, Any]] | None = None,
    adapter: LocalGEPAAdapterProtocol | None = None,
    task_lm: str | Any | None = None,
    evaluator: Any | None = None,
    reflection_lm: str | Any | None = None,
    max_metric_calls: int | None = None,
    stop_callbacks: Any | None = None,
    **_: Any,
) -> Any:
    """Local GEPA-compatible optimize function.

    This function is shaped for `gepa-ai` compatibility and can be used as a
    drop-in local replacement where a simple adapter-based optimization loop is
    sufficient.
    """
    del task_lm, evaluator, reflection_lm, stop_callbacks
    if not seed_candidate:
        raise ValueError("seed_candidate must contain at least one entry.")
    if not trainset:
        raise ValueError("trainset must contain at least one item.")
    if adapter is None:
        raise ValueError("adapter is required for local/offline mode.")

    eval_set: list[Mapping[str, Any]] = list(valset if valset is not None else trainset)
    budget = max(1, int(max_metric_calls or 8))
    candidates: list[dict[str, str]] = [dict(seed_candidate)]
    parents: list[list[int | None]] = [[None]]
    scores: list[float] = []
    subscores: list[dict[int, float]] = []
    eval_counts: list[int] = []
    metric_calls_total = 0

    base_score, base_subscores = _score_candidate(adapter, eval_set, candidates[0])
    scores.append(base_score)
    subscores.append(base_subscores)
    eval_counts.append(len(eval_set))
    metric_calls_total += len(eval_set)

    for idx in range(1, budget):
        candidate = _mutate_candidate(candidates[0], idx)
        score, per_item_scores = _score_candidate(adapter, eval_set, candidate)
        candidates.append(candidate)
        parents.append([0])
        scores.append(score)
        subscores.append(per_item_scores)
        eval_counts.append(len(eval_set))
        metric_calls_total += len(eval_set)

    per_instance_best: dict[int, set[int]] = {}
    for example_idx in range(len(eval_set)):
        best_score = max(item.get(example_idx, float("-inf")) for item in subscores)
        best = {
            candidate_idx
            for candidate_idx, item in enumerate(subscores)
            if item.get(example_idx, float("-inf")) == best_score
        }
        per_instance_best[example_idx] = best

    if SynthGEPAResult is not None:
        return SynthGEPAResult(
            candidates=candidates,
            parents=parents,
            val_aggregate_scores=scores,
            val_subscores=subscores,
            per_val_instance_best_candidates=per_instance_best,
            discovery_eval_counts=eval_counts,
            total_metric_calls=metric_calls_total,
        )

    return LocalGEPAResult(
        candidates=candidates,
        parents=parents,
        val_aggregate_scores=scores,
        val_subscores=subscores,
        per_val_instance_best_candidates=per_instance_best,
        discovery_eval_counts=eval_counts,
        total_metric_calls=metric_calls_total,
    )


__all__ = [
    "LocalGEPAAdapterProtocol",
    "LocalGEPAResult",
    "SynthOfflineLearningAdapter",
    "optimize",
]
