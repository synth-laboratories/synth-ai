from __future__ import annotations

from synth_ai.cli.commands.filter.validation import validate_filter_options


def test_validate_filter_options_normalizes_lists() -> None:
    options = {
        "splits": "train",
        "task_ids": ["a", " ", "b"],
        "models": ("model-a", "model-b"),
    }

    normalized = validate_filter_options(options)
    assert normalized["splits"] == ["train"]
    assert normalized["task_ids"] == ["a", "b"]
    assert normalized["models"] == ["model-a", "model-b"]


def test_validate_filter_options_normalizes_scores() -> None:
    options = {
        "min_official_score": "0.5",
        "max_official_score": "",
        "min_verifier_scores": {"quality": "0.7", "bad": "nan"},
        "max_verifier_scores": None,
    }

    normalized = validate_filter_options(options)
    assert normalized["min_official_score"] == 0.5
    assert normalized["max_official_score"] is None
    assert normalized["min_verifier_scores"] == {"quality": 0.7}
    assert normalized["max_verifier_scores"] == {}


def test_validate_filter_options_numeric_flags_and_shuffle() -> None:
    options = {
        "limit": "10",
        "offset": "",
        "shuffle": "true",
        "shuffle_seed": "42",
    }

    normalized = validate_filter_options(options)
    assert normalized["limit"] == 10
    assert normalized["offset"] is None
    assert normalized["shuffle"] is True
    assert normalized["shuffle_seed"] == 42
