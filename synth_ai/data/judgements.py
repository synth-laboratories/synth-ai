"""Judgement and rubric assignment data structures."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from . import rust as _rust_data
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.judgements.") from exc


@dataclass
class CriterionScoreData:
    """Score for a single rubric criterion."""

    score: float
    reason: Optional[str] = None
    weight: float = 1.0
    normalized_score: Optional[float] = None
    passed: Optional[bool] = None


@dataclass
class RubricAssignment:
    """Rubric scores assigned for a run."""

    criterion_scores: Dict[str, CriterionScoreData] = field(default_factory=dict)
    total: float = 0.0
    rubric_ref: Optional[str] = None
    summary: Optional[str] = None
    all_required_passed: Optional[bool] = None
    normalized_total: Optional[float] = None

    def __post_init__(self) -> None:
        coerced: Dict[str, CriterionScoreData] = {}
        for key, value in self.criterion_scores.items():
            if isinstance(value, dict):
                coerced[key] = CriterionScoreData(**value)
            else:
                coerced[key] = value
        self.criterion_scores = coerced


@dataclass
class Judgement:
    """Judgement wrapper for rubric scores and annotations."""

    rubric_assignment: Optional[RubricAssignment] = None
    annotation: Dict[str, Any] = field(default_factory=dict)
    passed: Optional[bool] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    judged_at: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.rubric_assignment, dict):
            self.rubric_assignment = RubricAssignment(**self.rubric_assignment)

    @classmethod
    def from_dict(cls, data: dict) -> Judgement:
        if _rust_data is not None:
            with contextlib.suppress(Exception):
                data = _rust_data.normalize_judgement(data)  # noqa: F811
        return cls(**data)


try:  # Require Rust-backed classes
    import synth_ai_py as _rust_models  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.judgements.") from exc

with contextlib.suppress(AttributeError):
    CriterionScoreData = _rust_models.CriterionScoreData  # noqa: F811
    RubricAssignment = _rust_models.RubricAssignment  # noqa: F811
    Judgement = _rust_models.Judgement  # noqa: F811


__all__ = [
    "CriterionScoreData",
    "RubricAssignment",
    "Judgement",
]
