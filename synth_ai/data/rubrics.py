"""Rubric and Criterion data models.

These are INPUT/DEFINITION structures that users create to define evaluation criteria.
For OUTPUT/RESULT structures (scores after evaluation), see judgements.py.
"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from . import rust as _rust_data
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.rubrics.") from exc


@dataclass
class CriterionExample:
    """Example snippet for a criterion with an expected score."""

    content: str
    expected_score: float
    explanation: Optional[str] = None


@dataclass
class Criterion:
    """Single evaluation criterion within a rubric.

    Flexible variant allowing weights > 1.0 and no normalization requirement.
    Used by task apps for general reward computation.

    Attributes:
        id: Unique identifier for this criterion
        description: Human-readable description of what this criterion evaluates
        weight: Relative weight for scoring (must be positive, default 1.0)
        required: Whether this criterion must be satisfied for success
    """

    id: str
    description: str
    weight: float = 1.0
    required: bool = False
    scale_max: Optional[float] = None
    examples: list[CriterionExample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("criterion id must be non-empty")
        if self.weight <= 0:
            raise ValueError("criterion weight must be positive")
        if self.scale_max is not None and self.scale_max <= 0:
            raise ValueError("criterion scale_max must be positive")
        coerced: list[CriterionExample] = []
        for example in self.examples:
            if isinstance(example, dict):
                coerced.append(CriterionExample(**example))
            else:
                coerced.append(example)
        self.examples = coerced

    def validate(self) -> None:
        """Validate the criterion configuration."""
        self.__post_init__()


@dataclass
class Rubric:
    """Rubric definition for evaluating task app outcomes.

    Supports flexible aggregation and blending. Criteria weights do not need
    to sum to 1.0, making this suitable for general task app usage.

    Attributes:
        version: Version string for this rubric definition
        goal_text: Optional human-readable description of the evaluation goal
        criteria: List of Criterion objects defining evaluation criteria
        aggregation: How to aggregate criterion scores ('sum', 'weighted_sum', 'custom', 'inherit')
    """

    version: str
    goal_text: str | None = None
    criteria: list[Criterion] = field(default_factory=list)
    aggregation: str = "weighted_sum"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce criteria from dicts
        coerced: list[Criterion] = []
        for criterion in self.criteria:
            if isinstance(criterion, dict):
                coerced.append(Criterion(**criterion))
            else:
                coerced.append(criterion)
        self.criteria = coerced

        # Validate aggregation
        allowed = {"sum", "weighted_sum", "mean", "weighted_mean", "custom", "inherit"}
        if self.aggregation not in allowed:
            raise ValueError(f"aggregation must be one of {sorted(allowed)}")

        # Validate no duplicate criterion IDs
        seen: set[str] = set()
        for criterion in self.criteria:
            if criterion.id in seen:
                raise ValueError(f"duplicate criterion id: {criterion.id}")
            seen.add(criterion.id)
            criterion.validate()

        if not self.criteria and self.aggregation != "inherit":
            raise ValueError(
                "rubric must have at least one criterion unless aggregation is inherit"
            )

    @classmethod
    def from_dict(cls, data: dict) -> Rubric:
        if _rust_data is not None:
            with contextlib.suppress(Exception):
                data = _rust_data.normalize_rubric(data)  # noqa: F811
        return cls(**data)


try:  # Require Rust-backed classes
    import synth_ai_py as _rust_models  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.rubrics.") from exc


def _is_constructible(cls: Any) -> bool:
    try:
        sig = inspect.signature(cls)
    except Exception:
        return False
    return bool(sig.parameters)


with contextlib.suppress(AttributeError):
    if _is_constructible(_rust_models.Criterion):
        CriterionExample = _rust_models.CriterionExample  # noqa: F811
        Criterion = _rust_models.Criterion  # noqa: F811
        Rubric = _rust_models.Rubric  # noqa: F811


__all__ = [
    "CriterionExample",
    "Criterion",
    "Rubric",
]
