

from pathlib import Path

import pytest

from synth_ai.task.rubrics import ValidationError, load_rubric, validate_rubric_dict, validate_rubric_file

RUBRICS_DIR = Path("examples/multi_step/rubrics")


def _get_task_app_rubrics():
    """Get rubrics in task app format (exclude backend judge rubrics)."""
    all_rubrics = sorted(RUBRICS_DIR.glob("*.json"))
    # Exclude backend judge rubrics (different format)
    return [p for p in all_rubrics if not p.name.endswith("_backend_judge.json")]


@pytest.mark.parametrize(
    "rubric_path",
    _get_task_app_rubrics(),
    ids=lambda p: p.name,
)
def test_example_rubrics_are_valid(rubric_path: Path) -> None:
    """All curated example rubrics in task app format should satisfy the schema."""
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


def _get_backend_judge_rubrics():
    """Get rubrics in backend judge format."""
    all_rubrics = sorted(RUBRICS_DIR.glob("*_backend_judge.json"))
    return all_rubrics


@pytest.mark.parametrize(
    "rubric_path",
    _get_backend_judge_rubrics(),
    ids=lambda p: p.name,
)
def test_backend_judge_rubrics_have_correct_format(rubric_path: Path) -> None:
    """Backend judge rubrics should have event/outcome structure."""
    import json
    
    with open(rubric_path) as f:
        rubric = json.load(f)
    
    # Backend judge rubrics must have 'event' and 'outcome' keys
    assert "event" in rubric, f"{rubric_path.name} must have 'event' key"
    assert "outcome" in rubric, f"{rubric_path.name} must have 'outcome' key"
    assert isinstance(rubric["event"], list), f"{rubric_path.name} 'event' must be a list"
    assert isinstance(rubric["outcome"], list), f"{rubric_path.name} 'outcome' must be a list"
    
    # Validate structure of criteria
    for criterion in rubric["event"] + rubric["outcome"]:
        assert "id" in criterion, "Each criterion must have 'id'"
        assert "description" in criterion, "Each criterion must have 'description'"
        assert "weight" in criterion, "Each criterion must have 'weight'"
        assert "scale" in criterion, "Each criterion must have 'scale'"
        assert isinstance(criterion["weight"], (int, float)), "Weight must be numeric"
        assert criterion["scale"] in ["bounded", "unbounded"], "Scale must be 'bounded' or 'unbounded'"


def test_load_rubric_rejects_backend_judge_format_with_helpful_error() -> None:
    """load_rubric should provide a clear error when given backend judge format."""
    backend_judge_rubric = {
        "event": [],
        "outcome": [
            {"id": "test", "description": "Test", "weight": 1.0, "scale": "bounded"}
        ]
    }
    
    with pytest.raises(ValueError) as exc_info:
        load_rubric(backend_judge_rubric)
    
    error_msg = str(exc_info.value)
    assert "backend judge format" in error_msg.lower()
    assert "event" in error_msg or "outcome" in error_msg
    assert "version" in error_msg or "goal_text" in error_msg or "criteria" in error_msg

