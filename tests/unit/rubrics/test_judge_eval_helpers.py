

import math
from typing import Any, Iterable, Sequence

import pytest

from synth_ai.judge_schemas import JudgeScoreResponse


def _merge_summaries(
    existing: Iterable[dict[str, Any]],
    new: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    new_by_session: dict[str, dict[str, Any]] = {}
    for summary in new:
        if not isinstance(summary, dict):
            continue
        session_id = summary.get("session_id")
        if isinstance(session_id, str):
            new_by_session[session_id] = summary

    for summary in existing:
        if not isinstance(summary, dict):
            continue
        session_id = summary.get("session_id")
        if not isinstance(session_id, str) or session_id in seen:
            continue
        replacement = new_by_session.pop(session_id, None)
        merged.append(replacement or summary)
        seen.add(session_id)

    for summary in new:
        if not isinstance(summary, dict):
            continue
        session_id = summary.get("session_id")
        if isinstance(session_id, str) and session_id not in seen:
            merged.append(summary)
            seen.add(session_id)

    return merged


def _prepare_options(
    base_options: dict[str, Any],
    *,
    provider: str | None,
    model: str | None,
    rubric_id: str | None,
    enable_event: bool | None,
    enable_outcome: bool | None,
) -> dict[str, Any]:
    opts = dict(base_options)
    if provider:
        opts["provider"] = provider
    if model:
        opts["model"] = model
    if rubric_id:
        opts["rubric_id"] = rubric_id
    if enable_event is not None:
        opts["event"] = enable_event
    if enable_outcome is not None:
        opts["outcome"] = enable_outcome
    return opts


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("Mismatched sequence lengths for correlation")
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return num / (denom_x * denom_y)


def _validate_and_aggregate_judge_response(response_dict: dict[str, Any]) -> tuple[JudgeScoreResponse, float | None, float | None]:
    validated = JudgeScoreResponse.model_validate(response_dict)
    event_reward = validated.aggregate_event_reward()
    outcome_reward = validated.aggregate_outcome_reward()
    return validated, event_reward, outcome_reward


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


