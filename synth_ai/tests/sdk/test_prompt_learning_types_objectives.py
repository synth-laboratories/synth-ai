from __future__ import annotations

from synth_ai.sdk.learning.prompt_learning_types import (
    BestPromptEventData,
    Candidate,
    OptimizedCandidate,
    ValidationScoredEventData,
)


def test_candidate_from_dict_prefers_objectives() -> None:
    data = {"objectives": {"reward": 0.7}, "accuracy": 0.2}
    candidate = Candidate.from_dict(data)
    assert candidate.objectives == {"reward": 0.7}
    assert candidate.accuracy == 0.7


def test_optimized_candidate_from_dict_prefers_objectives() -> None:
    data = {"score": {"objectives": {"reward": 0.8}, "accuracy": 0.1}, "payload_kind": "transformation"}
    candidate = OptimizedCandidate.from_dict(data)
    assert candidate.score.objectives == {"reward": 0.8}
    assert candidate.score.accuracy == 0.8
    assert candidate.objectives == {"reward": 0.8}


def test_validation_scored_event_prefers_objectives() -> None:
    data = {"objectives": {"reward": 0.55}, "accuracy": 0.2}
    event = ValidationScoredEventData.from_dict(data)
    assert event.objectives == {"reward": 0.55}
    assert event.accuracy == 0.55


def test_best_prompt_event_prefers_objectives() -> None:
    data = {"objectives": {"reward": 0.9}, "best_score": 0.1, "best_prompt": {"messages": []}}
    event = BestPromptEventData.from_dict(data)
    assert event.best_objectives == {"reward": 0.9}
    assert event.best_score == 0.9
