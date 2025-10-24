"""Strict rubric validators for step-wise judges.

These validators enforce stricter constraints than the general-purpose rubrics:
- Weights must be ≤ 1.0 and sum to exactly 1.0
- Only weighted_sum aggregation is allowed
- All required fields must be non-empty

Used primarily for validation in judge configurations.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pydantic


class StrictCriterion(pydantic.BaseModel):
    """Single scoring criterion with strict validation.
    
    Enforces:
    - Weight ≤ 1.0 (for proper normalization)
    - Weight > 0.0 (positive)
    - Non-empty strings
    """

    id: str
    description: str
    weight: float
    scale: str | None = None

    @pydantic.field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight must be a finite number")
        if value <= 0.0:
            raise ValueError("weight must be positive")
        if value > 1.0:
            raise ValueError("weight must be <= 1.0")
        return value

    @pydantic.field_validator("id", "description", mode="before")
    @classmethod
    def _strip_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class StrictRubric(pydantic.BaseModel):
    """Strict rubric definition for step-wise judges.
    
    Enforces:
    - Weights must sum to 1.0
    - Only weighted_sum aggregation
    - Non-empty version and goal_text
    - At least one criterion
    """

    version: str
    goal_text: str
    aggregation: Literal["weighted_sum"]
    criteria: list[StrictCriterion]

    @pydantic.model_validator(mode="after")
    def _validate_weights(self) -> StrictRubric:
        if not self.criteria:
            raise ValueError("rubric must declare at least one criterion")
        total_weight = sum(criterion.weight for criterion in self.criteria)
        if not math.isclose(total_weight, 1.0, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError(
                f"criterion weights must sum to 1 (got {total_weight:.6f})"
            )
        return self

    @pydantic.field_validator("version")
    @classmethod
    def _non_empty_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("version string must not be empty")
        return value

    @pydantic.field_validator("goal_text")
    @classmethod
    def _non_empty_goal_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("goal_text must not be empty")
        return value


# Re-export pydantic's ValidationError for convenience
ValidationError = pydantic.ValidationError


def validate_rubric_dict(payload: dict[str, Any]) -> StrictRubric:
    """Validate an in-memory rubric payload with strict rules.
    
    Args:
        payload: Dictionary representing the rubric JSON
        
    Returns:
        Validated StrictRubric instance
        
    Raises:
        ValidationError: If payload is invalid or doesn't meet strict constraints
    """
    if not isinstance(payload, dict):
        raise TypeError("rubric payload must be a dictionary")
    return StrictRubric.model_validate(payload)


def _load_payload_from_file(path: Path) -> dict[str, Any]:
    """Load JSON rubric from file."""
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported rubric file type: {path}")
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def validate_rubric_file(path: Path) -> StrictRubric:
    """Load and validate a rubric file with strict rules.
    
    Args:
        path: Path to a JSON rubric document
        
    Returns:
        Validated StrictRubric instance
    """
    payload = _load_payload_from_file(path)
    return validate_rubric_dict(payload)


def validate_rubric_files(paths: Iterable[Path]) -> list[StrictRubric]:
    """Validate multiple rubric files with strict rules.
    
    Useful for bulk validation inside tests or CI checks.
    """
    validated: list[StrictRubric] = []
    for path in paths:
        validated.append(validate_rubric_file(path))
    return validated

