from __future__ import annotations

from pathlib import Path

import pytest

from rubrics_dev.judge_eval import (
    _merge_summaries,
    _pearson,
    _prepare_options,
    _validate_and_aggregate_judge_response,
)


def test_merge_summaries_replaces_and_appends() -> None:
    existing = [
        {"session_id": "a", "x": 1},
        {"session_id": "b", "x": 2},
    ]
    new = [
        {"session_id": "b", "x": 3},
        {"session_id": "c", "x": 4},
    ]
    merged = _merge_summaries(existing, new)
    assert merged == [
        {"session_id": "a", "x": 1},
        {"session_id": "b", "x": 3},
        {"session_id": "c", "x": 4},
    ]


def test_prepare_options_overrides_only_provided() -> None:
    base = {"provider": "p", "model": "m", "rubric_id": "r", "event": True, "outcome": True}
    opts = _prepare_options(base, provider=None, model="m2", rubric_id=None, enable_event=False, enable_outcome=None)
    assert opts["provider"] == "p"
    assert opts["model"] == "m2"
    assert opts["rubric_id"] == "r"
    assert opts["event"] is False
    assert opts["outcome"] is True


def test_pearson_happy_path() -> None:
    xs = [1.0, 2.0, 3.0]
    ys = [1.0, 2.0, 3.0]
    c = _pearson(xs, ys)
    assert c is not None and c == pytest.approx(1.0)


def test_pearson_edge_cases() -> None:
    assert _pearson([], []) is None
    assert _pearson([1.0], [1.0]) is None
    with pytest.raises(ValueError):
        _pearson([1.0, 2.0], [1.0])


def test_validate_and_aggregate_contract_parsing() -> None:
    response = {
        "status": "ok",
        "event_reviews": [
            {"criteria": {}, "total": 0.0, "summary": None},
            {"criteria": {}, "total": 0.0, "summary": None},
        ],
        "outcome_review": {"criteria": {}, "total": 0.5, "summary": None},
        "event_totals": [0.25, 0.75],
        "details": {},
        "metadata": {"provider": "x"},
    }
    validated, event_reward, outcome_reward = _validate_and_aggregate_judge_response(response)
    assert event_reward == pytest.approx(1.0)
    assert outcome_reward == pytest.approx(0.5)


