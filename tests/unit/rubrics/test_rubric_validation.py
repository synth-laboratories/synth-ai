from __future__ import annotations

from pathlib import Path

import pytest

from synth_ai.task.rubrics import ValidationError, validate_rubric_dict, validate_rubric_file

RUBRICS_DIR = Path("examples/multi_step/rubrics")


@pytest.mark.parametrize(
    "rubric_path",
    sorted(RUBRICS_DIR.glob("*.json")),
    ids=lambda p: p.name,
)
def test_example_rubrics_are_valid(rubric_path: Path) -> None:
    """All curated example rubrics should satisfy the schema."""
    spec = validate_rubric_file(rubric_path)
    assert spec.goal_text, "goal text must be non-empty"
    assert abs(sum(c.weight for c in spec.criteria) - 1.0) <= 1e-6


def test_invalid_weight_sum_rejected() -> None:
    """Validator should reject rubrics whose weights do not sum to 1."""
    payload = {
        "version": "1",
        "goal_text": "Test rubric",
        "aggregation": "weighted_sum",
        "criteria": [
            {"id": "a", "description": "A", "weight": 0.6},
            {"id": "b", "description": "B", "weight": 0.6},
        ],
    }
    with pytest.raises(ValidationError):
        validate_rubric_dict(payload)

