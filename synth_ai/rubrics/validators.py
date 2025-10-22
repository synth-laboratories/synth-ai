from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal

import pydantic


class RubricCriterion(pydantic.BaseModel):
    """Single scoring criterion within a rubric."""

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


class RubricSpec(pydantic.BaseModel):
    """High-level rubric definition used by step-wise judges."""

    version: str
    goal_text: str
    aggregation: Literal["weighted_sum"]
    criteria: list[RubricCriterion]

    @pydantic.model_validator(mode="after")
    def _validate_weights(self) -> "RubricSpec":
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


ValidationError = pydantic.ValidationError


def validate_rubric_dict(payload: dict[str, Any]) -> RubricSpec:
    """
    Validate an in-memory rubric payload and return the parsed model.

    Args:
        payload: Dictionary representing the rubric JSON.
    Returns:
        Validated RubricSpec instance.
    Raises:
        ValidationError: If the payload is missing required fields or contains
        invalid weights.
    """

    if not isinstance(payload, dict):
        raise TypeError("rubric payload must be a dictionary")
    return RubricSpec.model_validate(payload)


def _load_payload_from_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported rubric file type: {path}")
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def validate_rubric_file(path: Path) -> RubricSpec:
    """
    Load and validate a rubric file.

    Args:
        path: Path to a JSON rubric document.
    Returns:
        Validated RubricSpec instance.
    """

    payload = _load_payload_from_file(path)
    return validate_rubric_dict(payload)


def validate_rubric_files(paths: Iterable[Path]) -> list[RubricSpec]:
    """
    Validate multiple rubric files and return their parsed models.

    Useful for bulk validation inside tests or CI checks.
    """

    validated: list[RubricSpec] = []
    for path in paths:
        validated.append(validate_rubric_file(path))
    return validated
