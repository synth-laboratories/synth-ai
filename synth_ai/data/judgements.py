"""Judgement and rubric assignment data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CriterionScoreData:
    """Score for a single rubric criterion."""

    score: float
    reason: Optional[str] = None
    weight: float = 1.0


@dataclass
class RubricAssignment:
    """Rubric scores assigned for a run."""

    criterion_scores: Dict[str, CriterionScoreData] = field(default_factory=dict)
    total: float = 0.0
    rubric_ref: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class Judgement:
    """Judgement wrapper for rubric scores and annotations."""

    rubric_assignment: Optional[RubricAssignment] = None
    annotation: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CriterionScoreData",
    "RubricAssignment",
    "Judgement",
]
