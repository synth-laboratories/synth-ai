"""Synth-compatible offline learning adapters for local use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable


ScoreFunction = Callable[[Mapping[str, Any], dict[str, str]], float]


@dataclass(frozen=True)
class LocalEvaluator:
    """Evaluates one dataset example against one candidate."""

    score_fn: ScoreFunction

    def score_batch(self, batch: Sequence[Mapping[str, Any]], candidate: dict[str, str]) -> list[float]:
        """Return per-example scores for the candidate."""
        return [float(self.score_fn(example, candidate)) for example in batch]


@dataclass(frozen=True)
class LocalEvaluationBatch:
    """Local batch output shaped to GEPA adapter expectations."""

    outputs: list[dict[str, Any]]
    scores: list[float]
    trajectories: list[dict[str, Any]] | None = None
    objective_scores: list[dict[str, float]] | None = None


class SynthOfflineLearningAdapter:
    """Local adapter with a GEPA-compatible evaluation shape.

    This adapter is intentionally lightweight so it can run completely offline
    without backend dependencies.
    """

    def __init__(self, evaluator: LocalEvaluator) -> None:
        self._evaluator = evaluator

    def evaluate(
        self,
        batch: list[Mapping[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> LocalEvaluationBatch:
        del capture_traces
        scores = self._evaluator.score_batch(batch, candidate)
        outputs = [{"candidate": candidate, "score": score} for score in scores]
        trajectories = [{"kind": "offline_local"} for _ in scores]
        objective_scores = [{"reward": score} for score in scores]
        return LocalEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: LocalEvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        reflective_rows: list[dict[str, Any]] = []
        for idx, (output, score) in enumerate(zip(eval_batch.outputs, eval_batch.scores)):
            reflective_rows.append(
                {
                    "index": idx,
                    "candidate": dict(candidate),
                    "score": float(score),
                    "output": output,
                }
            )
        return {
            component: tuple(reflective_rows)
            for component in components_to_update
        }
