"""Rubric scoring utilities for events and outcomes."""

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


def _score(
    criteria: Iterable[Criterion], values: dict[str, float], aggregation: str
) -> dict[str, Any]:
    """Compute aggregate score from criterion values.
    
    Args:
        criteria: List of criteria defining scoring dimensions
        values: Map of criterion IDs to scores
        aggregation: How to aggregate ("sum", "weighted_sum", "custom")
        
    Returns:
        Dict with aggregation method, total score, and per-criterion breakdown
    """
    if aggregation == "inherit":
        aggregation = "weighted_sum"
    per_criterion: dict[str, dict[str, Any]] = {}
    total = 0.0
    total_weight = 0.0
    for criterion in criteria:
        score = values.get(criterion.id, 0.0)
        per_criterion[criterion.id] = {
            "score": score,
            "weight": criterion.weight,
            "required": criterion.required,
        }
        if aggregation == "sum":
            total += score
        elif aggregation == "weighted_sum":
            total += score * criterion.weight
            total_weight += criterion.weight
    if aggregation == "weighted_sum" and total_weight > 0:
        total = total / total_weight
    if aggregation == "custom":
        total = None  # type: ignore[assignment]
    return {
        "aggregation": aggregation,
        "score": total,
        "per_criterion": per_criterion,
    }


def score_events_against_rubric(
    events: list[dict[str, Any]], rubric: Rubric | None
) -> dict[str, Any]:
    """Score a list of evaluation events against a rubric.
    
    Events should contain criterion_id/id/criterion and score fields.
    
    Args:
        events: List of event dicts with scoring info
        rubric: Rubric defining criteria and aggregation
        
    Returns:
        Scoring result with total and per-criterion scores
    """
    if rubric is None:
        return {"aggregation": "none", "score": None, "per_criterion": {}}
    values: dict[str, float] = {}
    for event in events or []:
        if not isinstance(event, dict):
            continue
        cid = event.get("criterion_id") or event.get("id") or event.get("criterion")
        score = _as_float(event.get("score"))
        if cid and score is not None:
            values[str(cid)] = score
    return _score(rubric.criteria, values, rubric.aggregation)


def score_outcome_against_rubric(outcome: dict[str, Any], rubric: Rubric | None) -> dict[str, Any]:
    """Score a rollout outcome against a rubric.
    
    Outcome should be a dict mapping criterion IDs to scores, optionally
    nested under a "criteria" key.
    
    Args:
        outcome: Outcome dict with criterion scores
        rubric: Rubric defining criteria and aggregation
        
    Returns:
        Scoring result with total and per-criterion scores
    """
    if rubric is None:
        return {"aggregation": "none", "score": None, "per_criterion": {}}
    values: dict[str, float] = {}
    if isinstance(outcome, dict):
        candidates = (
            outcome.get("criteria") if isinstance(outcome.get("criteria"), dict) else outcome
        )
        if isinstance(candidates, dict):
            for key, value in candidates.items():
                score = _as_float(value)
                if score is not None:
                    values[str(key)] = score
    return _score(rubric.criteria, values, rubric.aggregation)




