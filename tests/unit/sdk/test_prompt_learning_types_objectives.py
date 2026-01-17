from __future__ import annotations

from synth_ai.sdk.learning.prompt_learning_types import (
    BestPromptEventData,
    Candidate,
    OptimizedCandidate,
    ValidationScoredEventData,
)


def test_candidate_from_dict_with_outcome_objectives() -> None:
    """Test that outcome_objectives is used to derive accuracy."""
    data = {"outcome_objectives": {"reward": 0.7}, "accuracy": 0.2}
    candidate = Candidate.from_dict(data)
    assert candidate.objectives == {"reward": 0.7}
    assert candidate.accuracy == 0.7  # derived from outcome_objectives


def test_candidate_from_dict_fallback_to_accuracy() -> None:
    """Test that accuracy field is used when no outcome_objectives/outcome_reward."""
    data = {"objectives": {"reward": 0.7}, "accuracy": 0.2}
    candidate = Candidate.from_dict(data)
    assert candidate.objectives == {"reward": 0.7}  # from objectives field
    assert candidate.accuracy == 0.2  # fallback to accuracy field


def test_optimized_candidate_from_dict_with_outcome_objectives() -> None:
    """Test OptimizedCandidate with outcome_objectives in score."""
    data = {
        "score": {"outcome_objectives": {"reward": 0.8}, "accuracy": 0.1},
        "payload_kind": "transformation",
    }
    candidate = OptimizedCandidate.from_dict(data)
    assert candidate.score.objectives == {"reward": 0.8}
    assert candidate.score.accuracy == 0.8  # derived from outcome_objectives
    assert candidate.objectives == {"reward": 0.8}


def test_optimized_candidate_from_dict_fallback() -> None:
    """Test OptimizedCandidate falls back to accuracy when no outcome_objectives."""
    data = {
        "score": {"objectives": {"reward": 0.8}, "accuracy": 0.1},
        "payload_kind": "transformation",
    }
    candidate = OptimizedCandidate.from_dict(data)
    assert candidate.score.objectives == {"reward": 0.8}
    assert candidate.score.accuracy == 0.1  # fallback to accuracy field


def test_validation_scored_event_with_outcome_objectives() -> None:
    """Test ValidationScoredEventData with outcome_objectives."""
    data = {"outcome_objectives": {"reward": 0.55}, "accuracy": 0.2}
    event = ValidationScoredEventData.from_dict(data)
    assert event.objectives == {"reward": 0.55}
    assert event.accuracy == 0.55  # derived from outcome_objectives


def test_validation_scored_event_fallback() -> None:
    """Test ValidationScoredEventData falls back to accuracy."""
    data = {"objectives": {"reward": 0.55}, "accuracy": 0.2}
    event = ValidationScoredEventData.from_dict(data)
    assert event.objectives == {"reward": 0.55}
    assert event.accuracy == 0.2  # fallback to accuracy field


def test_best_prompt_event_with_outcome_objectives() -> None:
    """Test BestPromptEventData with outcome_objectives."""
    data = {
        "outcome_objectives": {"reward": 0.9},
        "best_score": 0.1,
        "best_prompt": {"messages": []},
    }
    event = BestPromptEventData.from_dict(data)
    assert event.best_objectives == {"reward": 0.9}
    assert event.best_score == 0.9  # derived from outcome_objectives


def test_best_prompt_event_fallback() -> None:
    """Test BestPromptEventData falls back to best_score."""
    data = {"objectives": {"reward": 0.9}, "best_score": 0.1, "best_prompt": {"messages": []}}
    event = BestPromptEventData.from_dict(data)
    assert event.best_objectives == {"reward": 0.9}
    assert event.best_score == 0.1  # fallback to best_score field
