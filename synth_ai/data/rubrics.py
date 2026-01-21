"""Rubric and Criterion data models.

These are INPUT/DEFINITION structures that users create to define evaluation criteria.
For OUTPUT/RESULT structures (scores after evaluation), see judgements.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("criterion weight must be positive")


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

    def __post_init__(self) -> None:
        # Validate aggregation
        allowed = {"sum", "weighted_sum", "custom", "inherit"}
        if self.aggregation not in allowed:
            raise ValueError(f"aggregation must be one of {sorted(allowed)}")

        # Validate no duplicate criterion IDs
        seen: set[str] = set()
        for criterion in self.criteria:
            if criterion.id in seen:
                raise ValueError(f"duplicate criterion id: {criterion.id}")
            seen.add(criterion.id)


__all__ = [
    "Criterion",
    "Rubric",
]
