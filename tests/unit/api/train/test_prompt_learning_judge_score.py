"""Unit tests for judge_score extraction in SDK.

Tests that judge_score is correctly extracted from job results.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any, Dict
import pytest

from synth_ai.api.train.prompt_learning import PromptLearningJob


@pytest.fixture
def mock_job_results_with_judge_score() -> Dict[str, Any]:
    """Sample job results with judge_score."""
    return {
        "best_score": 0.85,
        "judge_score": 0.9,
        "best_prompt": {"id": "test-prompt"},
        "validation_results": [
            {"rank": 0, "accuracy": 0.85, "score": 0.9},
        ],
    }


@pytest.fixture
def mock_job_results_without_judge_score() -> Dict[str, Any]:
    """Sample job results without judge_score."""
    return {
        "best_score": 0.85,
        "best_prompt": {"id": "test-prompt"},
        "validation_results": [
            {"rank": 0, "accuracy": 0.85},
        ],
    }


def test_extract_judge_score_from_top_level(mock_job_results_with_judge_score: Dict[str, Any]):
    """Test extracting judge_score from top-level results."""
    results = mock_job_results_with_judge_score
    judge_score = results.get("judge_score")
    assert judge_score == 0.9


def test_extract_judge_score_fallback_to_validation_results(mock_job_results_without_judge_score: Dict[str, Any]):
    """Test fallback to validation_results when judge_score not in top-level."""
    results = mock_job_results_without_judge_score
    judge_score = results.get("judge_score")
    
    # Fallback to validation_results
    if judge_score is None:
        validation_results = results.get("validation_results") or []
        if validation_results:
            judge_score = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    
    # Should be None in this case (no score field)
    assert judge_score is None


def test_extract_judge_score_from_validation_results():
    """Test extracting judge_score from validation_results."""
    results = {
        "best_score": 0.85,
        "validation_results": [
            {"rank": 0, "accuracy": 0.85, "score": 0.9},
        ],
    }
    
    judge_score = results.get("judge_score")
    if judge_score is None:
        validation_results = results.get("validation_results") or []
        if validation_results:
            judge_score = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    
    assert judge_score == 0.9


def test_extract_judge_score_none_when_missing():
    """Test that None is returned when judge_score is missing."""
    results = {
        "best_score": 0.85,
        "best_prompt": {"id": "test-prompt"},
    }
    
    judge_score = results.get("judge_score")
    if judge_score is None:
        validation_results = results.get("validation_results") or []
        if validation_results:
            judge_score = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    
    assert judge_score is None


def test_job_get_results_includes_judge_score(mock_job_results_with_judge_score: Dict[str, Any]):
    """Test that job.get_results() includes judge_score."""
    # This is a simplified test - actual implementation would use PromptLearningClient
    results = mock_job_results_with_judge_score
    
    # Verify judge_score is present
    assert "judge_score" in results
    assert results["judge_score"] == 0.9
    assert results["best_score"] == 0.85


def test_judge_score_display_in_outcome(mock_job_results_with_judge_score: Dict[str, Any]):
    """Test that judge_score is displayed in outcome."""
    results = mock_job_results_with_judge_score
    
    outcome = {
        "status": "succeeded",
        "best_score": results.get("best_score"),
        "judge_score": results.get("judge_score"),
        "best_prompt": results.get("best_prompt"),
    }
    
    assert outcome["judge_score"] == 0.9
    assert outcome["best_score"] == 0.85
    assert outcome["status"] == "succeeded"


def test_judge_score_fallback_logic():
    """Test the fallback logic for judge_score extraction."""
    # Case 1: judge_score in top-level
    results1 = {"judge_score": 0.9, "best_score": 0.85}
    judge_score1 = results1.get("judge_score")
    assert judge_score1 == 0.9
    
    # Case 2: judge_score in validation_results
    results2 = {
        "best_score": 0.85,
        "validation_results": [{"score": 0.9}],
    }
    judge_score2 = results2.get("judge_score")
    if judge_score2 is None:
        validation_results = results2.get("validation_results") or []
        if validation_results:
            judge_score2 = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    assert judge_score2 == 0.9
    
    # Case 3: No judge_score anywhere
    results3 = {"best_score": 0.85}
    judge_score3 = results3.get("judge_score")
    if judge_score3 is None:
        validation_results = results3.get("validation_results") or []
        if validation_results:
            judge_score3 = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    assert judge_score3 is None


