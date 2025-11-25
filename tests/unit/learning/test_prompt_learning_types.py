"""Unit tests for prompt learning type definitions."""


import pytest
from synth_ai.sdk.learning.prompt_learning_types import (
    BestPromptEventData,
    Candidate,
    CandidateScore,
    FinalResultsEventData,
    OptimizedCandidate,
    PromptLearningEvent,
    PromptResults,
    PromptSection,
    TextReplacement,
    ValidationScoredEventData,
)

pytestmark = pytest.mark.unit


class TestTextReplacement:
    """Tests for TextReplacement dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating TextReplacement with minimal fields."""
        replacement = TextReplacement(new_text="Hello world")
        assert replacement.new_text == "Hello world"
        assert replacement.apply_to_role == "system"
        assert replacement.old_text is None
        assert replacement.position is None

    def test_create_full(self) -> None:
        """Test creating TextReplacement with all fields."""
        replacement = TextReplacement(
            new_text="New text",
            apply_to_role="user",
            old_text="Old text",
            position=5,
        )
        assert replacement.new_text == "New text"
        assert replacement.apply_to_role == "user"
        assert replacement.old_text == "Old text"
        assert replacement.position == 5


class TestCandidateScore:
    """Tests for CandidateScore dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating CandidateScore with minimal fields."""
        score = CandidateScore(accuracy=0.75)
        assert score.accuracy == 0.75
        assert score.prompt_length == 0
        assert score.tool_call_rate == 0.0
        assert score.instance_scores == []

    def test_create_full(self) -> None:
        """Test creating CandidateScore with all fields."""
        score = CandidateScore(
            accuracy=0.85,
            prompt_length=150,
            tool_call_rate=0.3,
            instance_scores=[0.8, 0.9, 0.85],
        )
        assert score.accuracy == 0.85
        assert score.prompt_length == 150
        assert score.tool_call_rate == 0.3
        assert score.instance_scores == [0.8, 0.9, 0.85]

    def test_instance_scores_default_factory(self) -> None:
        """Test that instance_scores uses default_factory (new list each time)."""
        score1 = CandidateScore(accuracy=0.5)
        score2 = CandidateScore(accuracy=0.6)
        score1.instance_scores.append(0.7)
        assert score1.instance_scores == [0.7]
        assert score2.instance_scores == []  # Should be separate list


class TestPromptSection:
    """Tests for PromptSection dataclass."""

    def test_create(self) -> None:
        """Test creating PromptSection."""
        section = PromptSection(role="system", content="You are a helpful assistant")
        assert section.role == "system"
        assert section.content == "You are a helpful assistant"


class TestCandidate:
    """Tests for Candidate dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating Candidate with minimal fields."""
        candidate = Candidate(accuracy=0.8)
        assert candidate.accuracy == 0.8
        assert candidate.prompt_length == 0
        assert candidate.tool_call_rate == 0.0
        assert candidate.instance_scores == []
        assert candidate.object is None

    def test_create_full(self) -> None:
        """Test creating Candidate with all fields."""
        obj = {"text_replacements": [{"new_text": "test"}]}
        candidate = Candidate(
            accuracy=0.9,
            prompt_length=200,
            tool_call_rate=0.5,
            instance_scores=[0.85, 0.95],
            object=obj,
        )
        assert candidate.accuracy == 0.9
        assert candidate.prompt_length == 200
        assert candidate.tool_call_rate == 0.5
        assert candidate.instance_scores == [0.85, 0.95]
        assert candidate.object == obj

    def test_from_dict_minimal(self) -> None:
        """Test Candidate.from_dict with minimal data."""
        data = {"accuracy": 0.75}
        candidate = Candidate.from_dict(data)
        assert candidate.accuracy == 0.75
        assert candidate.prompt_length == 0
        assert candidate.tool_call_rate == 0.0
        assert candidate.instance_scores == []
        assert candidate.object is None

    def test_from_dict_full(self) -> None:
        """Test Candidate.from_dict with full data."""
        data = {
            "accuracy": 0.85,
            "prompt_length": 150,
            "tool_call_rate": 0.3,
            "instance_scores": [0.8, 0.9],
            "object": {"key": "value"},
        }
        candidate = Candidate.from_dict(data)
        assert candidate.accuracy == 0.85
        assert candidate.prompt_length == 150
        assert candidate.tool_call_rate == 0.3
        assert candidate.instance_scores == [0.8, 0.9]
        assert candidate.object == {"key": "value"}

    def test_from_dict_missing_fields(self) -> None:
        """Test Candidate.from_dict with missing fields uses defaults."""
        data = {}
        candidate = Candidate.from_dict(data)
        assert candidate.accuracy == 0.0
        assert candidate.prompt_length == 0
        assert candidate.tool_call_rate == 0.0
        assert candidate.instance_scores == []


class TestOptimizedCandidate:
    """Tests for OptimizedCandidate dataclass."""

    def test_create(self) -> None:
        """Test creating OptimizedCandidate."""
        score = CandidateScore(accuracy=0.9, prompt_length=100)
        candidate = OptimizedCandidate(
            score=score,
            payload_kind="transformation",
            object={"data": {}},
            instance_scores=[0.85, 0.95],
        )
        assert candidate.score == score
        assert candidate.payload_kind == "transformation"
        assert candidate.object == {"data": {}}
        assert candidate.instance_scores == [0.85, 0.95]

    def test_from_dict_with_score_dict(self) -> None:
        """Test OptimizedCandidate.from_dict with score as dict."""
        data = {
            "score": {
                "accuracy": 0.85,
                "prompt_length": 120,
                "tool_call_rate": 0.2,
                "instance_scores": [0.8, 0.9],
            },
            "payload_kind": "transformation",
            "object": {"key": "value"},
            "instance_scores": [0.75, 0.85],
        }
        candidate = OptimizedCandidate.from_dict(data)
        assert candidate.score.accuracy == 0.85
        assert candidate.score.prompt_length == 120
        assert candidate.score.tool_call_rate == 0.2
        assert candidate.score.instance_scores == [0.8, 0.9]
        assert candidate.payload_kind == "transformation"
        assert candidate.object == {"key": "value"}
        assert candidate.instance_scores == [0.75, 0.85]

    def test_from_dict_with_score_not_dict(self) -> None:
        """Test OptimizedCandidate.from_dict when score is not a dict."""
        data = {
            "score": "invalid",
            "payload_kind": "template",
        }
        candidate = OptimizedCandidate.from_dict(data)
        assert candidate.score.accuracy == 0.0
        assert candidate.payload_kind == "template"

    def test_from_dict_missing_score(self) -> None:
        """Test OptimizedCandidate.from_dict with missing score."""
        data = {
            "payload_kind": "transformation",
        }
        candidate = OptimizedCandidate.from_dict(data)
        assert candidate.score.accuracy == 0.0
        assert candidate.payload_kind == "transformation"


class TestPromptLearningEvent:
    """Tests for PromptLearningEvent dataclass."""

    def test_create(self) -> None:
        """Test creating PromptLearningEvent."""
        event = PromptLearningEvent(
            type="prompt.learning.test",
            message="Test message",
            data={"key": "value"},
            seq=1,
            created_at="2024-01-01T00:00:00Z",
        )
        assert event.type == "prompt.learning.test"
        assert event.message == "Test message"
        assert event.data == {"key": "value"}
        assert event.seq == 1
        assert event.created_at == "2024-01-01T00:00:00Z"

    def test_from_dict_minimal(self) -> None:
        """Test PromptLearningEvent.from_dict with minimal data."""
        data = {
            "type": "prompt.learning.test",
            "message": "Test",
            "data": {},
            "seq": 1,
        }
        event = PromptLearningEvent.from_dict(data)
        assert event.type == "prompt.learning.test"
        assert event.message == "Test"
        assert event.data == {}
        assert event.seq == 1
        assert event.created_at is None

    def test_from_dict_full(self) -> None:
        """Test PromptLearningEvent.from_dict with full data."""
        data = {
            "type": "prompt.learning.test",
            "message": "Test message",
            "data": {"key": "value"},
            "seq": 5,
            "created_at": "2024-01-01T00:00:00Z",
        }
        event = PromptLearningEvent.from_dict(data)
        assert event.type == "prompt.learning.test"
        assert event.message == "Test message"
        assert event.data == {"key": "value"}
        assert event.seq == 5
        assert event.created_at == "2024-01-01T00:00:00Z"

    def test_from_dict_missing_fields(self) -> None:
        """Test PromptLearningEvent.from_dict with missing fields uses defaults."""
        data = {}
        event = PromptLearningEvent.from_dict(data)
        assert event.type == ""
        assert event.message == ""
        assert event.data == {}
        assert event.seq == 0
        assert event.created_at is None


class TestBestPromptEventData:
    """Tests for BestPromptEventData dataclass."""

    def test_create(self) -> None:
        """Test creating BestPromptEventData."""
        prompt = {"sections": [{"role": "system", "content": "You are helpful"}]}
        event_data = BestPromptEventData(best_score=0.95, best_prompt=prompt)
        assert event_data.best_score == 0.95
        assert event_data.best_prompt == prompt

    def test_from_dict(self) -> None:
        """Test BestPromptEventData.from_dict."""
        data = {
            "best_score": 0.9,
            "best_prompt": {"sections": []},
        }
        event_data = BestPromptEventData.from_dict(data)
        assert event_data.best_score == 0.9
        assert event_data.best_prompt == {"sections": []}

    def test_from_dict_missing_fields(self) -> None:
        """Test BestPromptEventData.from_dict with missing fields uses defaults."""
        data = {}
        event_data = BestPromptEventData.from_dict(data)
        assert event_data.best_score == 0.0
        assert event_data.best_prompt == {}


class TestFinalResultsEventData:
    """Tests for FinalResultsEventData dataclass."""

    def test_create(self) -> None:
        """Test creating FinalResultsEventData."""
        attempted = [{"accuracy": 0.8}]
        optimized = [{"score": {"accuracy": 0.9}}]
        event_data = FinalResultsEventData(
            attempted_candidates=attempted,
            optimized_candidates=optimized,
        )
        assert event_data.attempted_candidates == attempted
        assert event_data.optimized_candidates == optimized

    def test_from_dict(self) -> None:
        """Test FinalResultsEventData.from_dict."""
        data = {
            "attempted_candidates": [{"accuracy": 0.7}],
            "optimized_candidates": [{"score": {"accuracy": 0.8}}],
        }
        event_data = FinalResultsEventData.from_dict(data)
        assert len(event_data.attempted_candidates) == 1
        assert len(event_data.optimized_candidates) == 1

    def test_from_dict_missing_fields(self) -> None:
        """Test FinalResultsEventData.from_dict with missing fields uses defaults."""
        data = {}
        event_data = FinalResultsEventData.from_dict(data)
        assert event_data.attempted_candidates == []
        assert event_data.optimized_candidates == []


class TestValidationScoredEventData:
    """Tests for ValidationScoredEventData dataclass."""

    def test_create(self) -> None:
        """Test creating ValidationScoredEventData."""
        event_data = ValidationScoredEventData(
            accuracy=0.85,
            instance_scores=[0.8, 0.9],
            is_baseline=True,
        )
        assert event_data.accuracy == 0.85
        assert event_data.instance_scores == [0.8, 0.9]
        assert event_data.is_baseline is True

    def test_from_dict(self) -> None:
        """Test ValidationScoredEventData.from_dict."""
        data = {
            "accuracy": 0.75,
            "instance_scores": [0.7, 0.8],
            "is_baseline": False,
        }
        event_data = ValidationScoredEventData.from_dict(data)
        assert event_data.accuracy == 0.75
        assert event_data.instance_scores == [0.7, 0.8]
        assert event_data.is_baseline is False

    def test_from_dict_missing_fields(self) -> None:
        """Test ValidationScoredEventData.from_dict with missing fields uses defaults."""
        data = {"accuracy": 0.8}
        event_data = ValidationScoredEventData.from_dict(data)
        assert event_data.accuracy == 0.8
        assert event_data.instance_scores == []
        assert event_data.is_baseline is False


class TestPromptResults:
    """Tests for PromptResults dataclass."""

    def test_create_empty(self) -> None:
        """Test creating empty PromptResults."""
        results = PromptResults()
        assert results.best_prompt is None
        assert results.best_score is None
        assert results.top_prompts == []
        assert results.optimized_candidates == []
        assert results.attempted_candidates == []
        assert results.validation_results == []

    def test_create_full(self) -> None:
        """Test creating PromptResults with all fields."""
        best_prompt = {"sections": [{"role": "system", "content": "Hello"}]}
        top_prompts = [{"rank": 1, "full_text": "Prompt 1"}]
        optimized = [{"score": {"accuracy": 0.9}}]
        attempted = [{"accuracy": 0.8}]
        validation = [{"accuracy": 0.85}]

        results = PromptResults(
            best_prompt=best_prompt,
            best_score=0.95,
            top_prompts=top_prompts,
            optimized_candidates=optimized,
            attempted_candidates=attempted,
            validation_results=validation,
        )
        assert results.best_prompt == best_prompt
        assert results.best_score == 0.95
        assert results.top_prompts == top_prompts
        assert results.optimized_candidates == optimized
        assert results.attempted_candidates == attempted
        assert results.validation_results == validation

    def test_from_dict_minimal(self) -> None:
        """Test PromptResults.from_dict with minimal data."""
        data = {}
        results = PromptResults.from_dict(data)
        assert results.best_prompt is None
        assert results.best_score is None
        assert results.top_prompts == []
        assert results.optimized_candidates == []
        assert results.attempted_candidates == []
        assert results.validation_results == []

    def test_from_dict_full(self) -> None:
        """Test PromptResults.from_dict with full data."""
        data = {
            "best_prompt": {"sections": []},
            "best_score": 0.9,
            "top_prompts": [{"rank": 1}],
            "optimized_candidates": [{"score": {"accuracy": 0.85}}],
            "attempted_candidates": [{"accuracy": 0.8}],
            "validation_results": [{"accuracy": 0.75}],
        }
        results = PromptResults.from_dict(data)
        assert results.best_prompt == {"sections": []}
        assert results.best_score == 0.9
        assert len(results.top_prompts) == 1
        assert len(results.optimized_candidates) == 1
        assert len(results.attempted_candidates) == 1
        assert len(results.validation_results) == 1

    def test_default_factory_isolation(self) -> None:
        """Test that default_factory creates separate lists for each instance."""
        results1 = PromptResults()
        results2 = PromptResults()
        results1.top_prompts.append({"rank": 1})
        assert len(results1.top_prompts) == 1
        assert len(results2.top_prompts) == 0  # Should be separate list


