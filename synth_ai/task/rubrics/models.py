"""Rubric and Criterion data models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Criterion(BaseModel):
    """Single scoring criterion within a rubric.
    
    Flexible variant allowing weights > 1.0 and no normalization requirement.
    Used by task apps for general rubric scoring.
    """

    id: str
    description: str
    weight: float = 1.0
    required: bool = False

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("criterion weight must be positive")
        return value


class Rubric(BaseModel):
    """Rubric definition for scoring task app outcomes.
    
    Supports flexible aggregation and blending. Criteria weights do not need
    to sum to 1.0, making this suitable for general task app usage.
    """

    version: str
    goal_text: str | None = None
    criteria: list[Criterion] = Field(default_factory=list)
    aggregation: str = "weighted_sum"

    @field_validator("aggregation")
    @classmethod
    def _validate_aggregation(cls, value: str) -> str:
        allowed = {"sum", "weighted_sum", "custom", "inherit"}
        if value not in allowed:
            raise ValueError(f"aggregation must be one of {sorted(allowed)}")
        return value

    @field_validator("criteria")
    @classmethod
    def _validate_criteria(cls, criteria: list[Criterion]) -> list[Criterion]:
        seen = set()
        for criterion in criteria:
            if criterion.id in seen:
                raise ValueError(f"duplicate criterion id: {criterion.id}")
            seen.add(criterion.id)
        return criteria

