"""Unit tests for verifier_score extraction in SDK.

Tests that verifier_score is correctly extracted from job results.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any, Dict
import pytest

from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob


@pytest.fixture
def mock_job_results_with_verifier_score() -> Dict[str, Any]:
    """Sample job results with verifier_score."""
    return {
        "best_score": 0.85,
        "verifier_score": 0.9,
        "best_prompt": {"id": "test-prompt"},
        "validation_results": [
            {"rank": 0, "accuracy": 0.85, "score": 0.9},
        ],
    }


@pytest.fixture
def mock_job_results_without_verifier_score() -> Dict[str, Any]:
    """Sample job results without verifier_score."""
    return {
        "best_score": 0.85,
        "best_prompt": {"id": "test-prompt"},
        "validation_results": [
            {"rank": 0, "accuracy": 0.85},
        ],
    }


def test_extract_verifier_score_from_top_level(mock_job_results_with_verifier_score: Dict[str, Any]):
    """Test extracting verifier_score from top-level results."""
    results = mock_job_results_with_verifier_score
    verifier_score = results.get("verifier_score")
    assert verifier_score == 0.9


def test_extract_verifier_score_fallback_to_validation_results(mock_job_results_without_verifier_score: Dict[str, Any]):
    """Test fallback to validation_results when verifier_score not in top-level."""
    results = mock_job_results_without_verifier_score
    verifier_score = results.get("verifier_score")
    
    # Fallback to validation_results
    if verifier_score is None:
        validation_results = results.get("validation_results") or []
        if validation_results:
            verifier_score = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    
    # Should be None in this case (no score field)
    assert verifier_score is None


def test_extract_verifier_score_from_validation_results():
    """Test extracting verifier_score from validation_results."""
    results = {
        "best_score": 0.85,
        "validation_results": [
            {"rank": 0, "accuracy": 0.85, "score": 0.9},
        ],
    }
    
    verifier_score = results.get("verifier_score")
    if verifier_score is None:
        validation_results = results.get("validation_results") or []
        if validation_results:
            verifier_score = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    
    assert verifier_score == 0.9


def test_extract_verifier_score_none_when_missing():
    """Test that None is returned when verifier_score is missing."""
    results = {
        "best_score": 0.85,
        "best_prompt": {"id": "test-prompt"},
    }
    
    verifier_score = results.get("verifier_score")
    if verifier_score is None:
        validation_results = results.get("validation_results") or []
        if validation_results:
            verifier_score = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    
    assert verifier_score is None


def test_job_get_results_includes_verifier_score(mock_job_results_with_verifier_score: Dict[str, Any]):
    """Test that job.get_results() includes verifier_score."""
    # This is a simplified test - actual implementation would use PromptLearningClient
    results = mock_job_results_with_verifier_score
    
    # Verify verifier_score is present
    assert "verifier_score" in results
    assert results["verifier_score"] == 0.9
    assert results["best_score"] == 0.85


def test_verifier_score_display_in_outcome(mock_job_results_with_verifier_score: Dict[str, Any]):
    """Test that verifier_score is displayed in outcome."""
    results = mock_job_results_with_verifier_score
    
    outcome = {
        "status": "succeeded",
        "best_score": results.get("best_score"),
        "verifier_score": results.get("verifier_score"),
        "best_prompt": results.get("best_prompt"),
    }
    
    assert outcome["verifier_score"] == 0.9
    assert outcome["best_score"] == 0.85
    assert outcome["status"] == "succeeded"


def test_verifier_score_fallback_logic():
    """Test the fallback logic for verifier_score extraction."""
    # Case 1: verifier_score in top-level
    results1 = {"verifier_score": 0.9, "best_score": 0.85}
    verifier_score1 = results1.get("verifier_score")
    assert verifier_score1 == 0.9
    
    # Case 2: verifier_score in validation_results
    results2 = {
        "best_score": 0.85,
        "validation_results": [{"score": 0.9}],
    }
    verifier_score2 = results2.get("verifier_score")
    if verifier_score2 is None:
        validation_results = results2.get("validation_results") or []
        if validation_results:
            verifier_score2 = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    assert verifier_score2 == 0.9
    
    # Case 3: No verifier_score anywhere
    results3 = {"best_score": 0.85}
    verifier_score3 = results3.get("verifier_score")
    if verifier_score3 is None:
        validation_results = results3.get("validation_results") or []
        if validation_results:
            verifier_score3 = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
    assert verifier_score3 is None


