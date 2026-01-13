"""Rubric evaluation helpers for events and outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import Criterion, Rubric


def _as_float(value: Any) -> float | None:
    """Safely convert value to float, returning None on failure."""
    try:
        return float(value)
    except Exception:
        return None


def _aggregate_rewards(
    criteria: Iterable[Criterion],
    values: dict[str, float],
    aggregation: str,
) -> dict[str, Any]:
    """Aggregate criterion rewards into a single value.

    Returns:
        Dict with aggregation method, aggregate reward, and per-criterion breakdown.
    """
    if aggregation == "inherit":
        aggregation = "weighted_sum"

    per_criterion: dict[str, dict[str, Any]] = {}
    total = 0.0
    total_weight = 0.0
    for criterion in criteria:
        reward = values.get(criterion.id, 0.0)
        per_criterion[criterion.id] = {
            "reward": reward,
            "weight": criterion.weight,
            "required": criterion.required,
        }
        if aggregation == "sum":
            total += reward
        elif aggregation == "weighted_sum":
            total += reward * criterion.weight
            total_weight += criterion.weight

    if aggregation == "weighted_sum" and total_weight > 0:
        total = total / total_weight

    if aggregation == "custom":
        total = None  # type: ignore[assignment]

    return {
        "aggregation": aggregation,
        "reward": total,
        "per_criterion": per_criterion,
    }


def evaluate_events_against_rubric(
    events: list[dict[str, Any]],
    rubric: Rubric | None,
) -> dict[str, Any]:
    """Evaluate a list of events against a rubric.

    Events should contain criterion_id/id/criterion and reward fields.
    """
    if rubric is None:
        return {"aggregation": "none", "reward": None, "per_criterion": {}}

    values: dict[str, float] = {}
    for event in events or []:
        if not isinstance(event, dict):
            continue
        cid = event.get("criterion_id") or event.get("id") or event.get("criterion")
        reward = _as_float(event.get("reward"))
        if cid and reward is not None:
            values[str(cid)] = reward

    return _aggregate_rewards(rubric.criteria, values, rubric.aggregation)


def evaluate_outcome_against_rubric(
    outcome: dict[str, Any],
    rubric: Rubric | None,
) -> dict[str, Any]:
    """Evaluate an outcome dict against a rubric.

    Outcome should map criterion IDs to rewards, optionally nested under "criteria".
    """
    if rubric is None:
        return {"aggregation": "none", "reward": None, "per_criterion": {}}

    values: dict[str, float] = {}
    if isinstance(outcome, dict):
        candidates = (
            outcome.get("criteria") if isinstance(outcome.get("criteria"), dict) else outcome
        )
        if isinstance(candidates, dict):
            for key, value in candidates.items():
                reward = _as_float(value)
                if reward is not None:
                    values[str(key)] = reward

    return _aggregate_rewards(rubric.criteria, values, rubric.aggregation)
