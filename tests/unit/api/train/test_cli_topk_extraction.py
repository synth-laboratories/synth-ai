"""
Unit tests for CLI top-K prompt extraction from events.

Tests the logic that parses optimized.scored and validation.scored events
to extract train/val accuracy pairs for display in final JSON output.
"""
from typing import Any, Dict, List, Optional

import pytest


class TopKPromptResult:
    """
    Represents a top-K prompt with train/val scores.
    This is the dataclass we should implement (currently just dict in CLI).
    """
    def __init__(self, rank: int, train_accuracy: float, 
                 val_accuracy: Optional[float], prompt_preview: str):
        self.rank = rank
        self.train_accuracy = train_accuracy
        self.val_accuracy = val_accuracy
        self.prompt_preview = prompt_preview
        
        # Assertions
        assert self.rank >= 0, f"rank must be >=0, got {self.rank}"
        assert 0.0 <= self.train_accuracy <= 1.0, \
            f"train_accuracy must be [0,1], got {self.train_accuracy}"
        if self.val_accuracy is not None:
            assert 0.0 <= self.val_accuracy <= 1.0, \
                f"val_accuracy must be [0,1], got {self.val_accuracy}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "prompt_preview": self.prompt_preview,
        }
    
    @classmethod
    def from_optimized_event(cls, event: Dict[str, Any]) -> Optional["TopKPromptResult"]:
        """
        Parse from optimized.scored event.
        Message format: "optimized[0] train_accuracy=0.2 len=86 N=5 val_accuracy=0.120"
        """
        msg = event.get("message", "")
        if "optimized[" not in msg:
            return None
        
        try:
            # Extract rank
            idx_str = msg.split("optimized[")[1].split("]")[0]
            idx = int(idx_str)
            
            # Extract train accuracy
            train_acc_str = msg.split("train_accuracy=")[1].split()[0]
            train_acc = float(train_acc_str)
            
            # Extract validation accuracy if present
            val_acc = None
            if "val_accuracy=" in msg:
                val_acc = float(msg.split("val_accuracy=")[1].split()[0])
            
            # Extract prompt preview
            prompt_preview = ""
            if "✨ TRANSFORMATION:" in msg:
                parts = msg.split("✨ TRANSFORMATION:")
                if len(parts) > 1:
                    prompt_preview = parts[1].strip()[:200]
            
            return cls(
                rank=idx,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                prompt_preview=prompt_preview,
            )
        except (IndexError, ValueError, AttributeError):
            return None


def extract_validation_scores(events: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Extract validation scores by rank from validation.scored events.
    Message format: "top[0] val_accuracy=0.620 (N=50) lift=0.0"
    """
    validation_scores = {}
    
    for event in events:
        if event.get("type") != "prompt.learning.validation.scored":
            continue
        
        msg = event.get("message", "")
        data = event.get("data", {})
        
        if "top[" in msg:
            try:
                rank_str = msg.split("top[")[1].split("]")[0]
                rank = int(rank_str)
                
                # Try data first, then parse from message
                val_acc = data.get("accuracy")
                if val_acc is None and "val_accuracy=" in msg:
                    val_acc = float(msg.split("val_accuracy=")[1].split()[0])
                
                if val_acc is not None:
                    validation_scores[rank] = float(val_acc)
            except (IndexError, ValueError, AttributeError, KeyError):
                pass  # Skip malformed events
    
    return validation_scores


def extract_top_k_prompts(events: List[Dict[str, Any]]) -> List[TopKPromptResult]:
    """
    Extract top-K prompts with train/val scores from event stream.
    This is the actual logic from handle_prompt_learning in cli.py.
    """
    # First pass: collect validation scores by rank
    validation_scores = extract_validation_scores(events)
    
    # Second pass: collect optimized candidates
    top_k_results = []
    
    for event in events:
        if event.get("type") != "prompt.learning.optimized.scored":
            continue
        
        result = TopKPromptResult.from_optimized_event(event)
        if result:
            # Backfill val_accuracy if not in message but available from validation events
            if result.val_accuracy is None and result.rank in validation_scores:
                result.val_accuracy = validation_scores[result.rank]
            
            top_k_results.append(result)
    
    # Sort by rank
    top_k_results.sort(key=lambda x: x.rank)
    
    return top_k_results


class TestTopKPromptResultParsing:
    """Test parsing individual optimized.scored events."""
    
    def test_parse_with_validation_score(self):
        """Parse event with both train and val accuracy."""
        event = {
            "type": "prompt.learning.optimized.scored",
            "message": "optimized[0] train_accuracy=0.2 len=86 N=5 val_accuracy=0.120",
        }
        
        result = TopKPromptResult.from_optimized_event(event)
        assert result is not None
        assert result.rank == 0
        assert result.train_accuracy == 0.2
        assert result.val_accuracy == 0.120
    
    def test_parse_without_validation_score(self):
        """Parse event with only train accuracy."""
        event = {
            "type": "prompt.learning.optimized.scored",
            "message": "optimized[1] train_accuracy=0.4 len=97 N=5",
        }
        
        result = TopKPromptResult.from_optimized_event(event)
        assert result is not None
        assert result.rank == 1
        assert result.train_accuracy == 0.4
        assert result.val_accuracy is None
    
    def test_parse_with_prompt_preview(self):
        """Parse event with prompt transformation preview."""
        event = {
            "type": "prompt.learning.optimized.scored",
            "message": """optimized[0] train_accuracy=0.2 len=86 N=5
  ✨ TRANSFORMATION:
    [SYSTEM]: Determine the single intent from the enumerated list...""",
        }
        
        result = TopKPromptResult.from_optimized_event(event)
        assert result is not None
        assert result.rank == 0
        assert "SYSTEM" in result.prompt_preview
        assert "Determine the single intent" in result.prompt_preview
    
    def test_parse_malformed_event_returns_none(self):
        """Malformed event should return None gracefully."""
        bad_events = [
            {"message": "not an optimized event"},
            {"message": "optimized[not_a_number] train_accuracy=0.5"},
            {"message": "optimized[0] train_accuracy=invalid"},
            {"message": "optimized[0]"},  # Missing train_accuracy
            {},  # Empty event
        ]
        
        for event in bad_events:
            result = TopKPromptResult.from_optimized_event(event)
            assert result is None, f"Should return None for: {event}"
    
    def test_parse_edge_case_accuracies(self):
        """Test edge case accuracy values."""
        # Perfect score
        event1 = {"message": "optimized[0] train_accuracy=1.0 len=50 N=5 val_accuracy=1.000"}
        result1 = TopKPromptResult.from_optimized_event(event1)
        assert result1.train_accuracy == 1.0
        assert result1.val_accuracy == 1.0
        
        # Zero score
        event2 = {"message": "optimized[1] train_accuracy=0.0 len=50 N=5 val_accuracy=0.000"}
        result2 = TopKPromptResult.from_optimized_event(event2)
        assert result2.train_accuracy == 0.0
        assert result2.val_accuracy == 0.0


class TestValidationScoreExtraction:
    """Test extraction of validation scores from validation.scored events."""
    
    def test_extract_single_validation_score(self):
        """Extract validation score from single event."""
        events = [
            {
                "type": "prompt.learning.validation.scored",
                "message": "top[0] val_accuracy=0.620 (N=50) lift=0.0",
                "data": {"accuracy": 0.620},
            }
        ]
        
        scores = extract_validation_scores(events)
        assert scores == {0: 0.620}
    
    def test_extract_multiple_validation_scores(self):
        """Extract validation scores from multiple events."""
        events = [
            {
                "type": "prompt.learning.validation.scored",
                "message": "top[0] val_accuracy=0.620 (N=50) lift=0.0",
                "data": {"accuracy": 0.620},
            },
            {
                "type": "prompt.learning.validation.scored",
                "message": "top[1] val_accuracy=0.640 (N=50) lift=2.0",
                "data": {"accuracy": 0.640},
            },
            {
                "type": "prompt.learning.validation.scored",
                "message": "top[2] val_accuracy=0.620 (N=50) lift=0.0",
                "data": {"accuracy": 0.620},
            },
        ]
        
        scores = extract_validation_scores(events)
        assert scores == {0: 0.620, 1: 0.640, 2: 0.620}
    
    def test_extract_baseline_validation_score(self):
        """Extract baseline validation score (no rank, or rank=-1)."""
        events = [
            {
                "type": "prompt.learning.validation.scored",
                "message": "baseline val_accuracy=0.620 (N=50)",
                "data": {"accuracy": 0.620, "is_baseline": True},
            }
        ]
        
        # Baseline is not in "top[N]" format, so won't be extracted by current logic
        scores = extract_validation_scores(events)
        assert scores == {}  # Expected behavior: baseline not indexed
    
    def test_extract_from_message_when_data_missing(self):
        """Fall back to parsing message if data.accuracy missing."""
        events = [
            {
                "type": "prompt.learning.validation.scored",
                "message": "top[0] val_accuracy=0.620 (N=50) lift=0.0",
                "data": {},  # No accuracy in data
            }
        ]
        
        scores = extract_validation_scores(events)
        assert scores == {0: 0.620}
    
    def test_skip_non_validation_events(self):
        """Non-validation events should be ignored."""
        events = [
            {"type": "prompt.learning.optimized.scored", "message": "optimized[0]"},
            {"type": "prompt.learning.progress", "message": "progress"},
            {"type": "prompt.learning.validation.scored", "message": "top[0] val_accuracy=0.5"},
        ]
        
        scores = extract_validation_scores(events)
        assert scores == {0: 0.5}
    
    def test_handle_malformed_validation_events(self):
        """Malformed validation events should be skipped gracefully."""
        events = [
            {"type": "prompt.learning.validation.scored", "message": "invalid"},
            {"type": "prompt.learning.validation.scored", "message": "top[not_a_number]"},
            {"type": "prompt.learning.validation.scored"},  # No message
        ]
        
        scores = extract_validation_scores(events)
        assert scores == {}


class TestTopKPromptExtraction:
    """Integration tests for full top-K extraction from event stream."""
    
    def test_extract_with_validation_scores(self):
        """Extract top-K with both train and val scores."""
        events = [
            # Validation events first
            {
                "type": "prompt.learning.validation.scored",
                "message": "top[0] val_accuracy=0.120 (N=50)",
                "data": {"accuracy": 0.120},
            },
            # Optimized events
            {
                "type": "prompt.learning.optimized.scored",
                "message": "optimized[0] train_accuracy=0.2 len=86 N=5 val_accuracy=0.120",
            },
        ]
        
        results = extract_top_k_prompts(events)
        assert len(results) == 1
        assert results[0].rank == 0
        assert results[0].train_accuracy == 0.2
        assert results[0].val_accuracy == 0.120
    
    def test_extract_backfills_validation_score(self):
        """Validation score from separate event should backfill if missing in message."""
        events = [
            # Validation event
            {
                "type": "prompt.learning.validation.scored",
                "message": "top[0] val_accuracy=0.120 (N=50)",
                "data": {"accuracy": 0.120},
            },
            # Optimized event WITHOUT val_accuracy in message
            {
                "type": "prompt.learning.optimized.scored",
                "message": "optimized[0] train_accuracy=0.2 len=86 N=5",  # No val_accuracy!
            },
        ]
        
        results = extract_top_k_prompts(events)
        assert len(results) == 1
        assert results[0].val_accuracy == 0.120  # Backfilled from validation event
    
    def test_extract_multiple_candidates(self):
        """Extract multiple top-K candidates."""
        events = [
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[0] train_accuracy=0.8 len=105 N=5 val_accuracy=0.620"},
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[1] train_accuracy=0.6 len=97 N=5 val_accuracy=0.640"},
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[2] train_accuracy=0.4 len=89 N=5 val_accuracy=0.620"},
        ]
        
        results = extract_top_k_prompts(events)
        assert len(results) == 3
        assert results[0].rank == 0
        assert results[1].rank == 1
        assert results[2].rank == 2
        assert results[1].val_accuracy == 0.640  # Best val score
    
    def test_extract_sorts_by_rank(self):
        """Results should be sorted by rank even if events are out of order."""
        events = [
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[2] train_accuracy=0.4 len=89 N=5"},
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[0] train_accuracy=0.8 len=105 N=5"},
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[1] train_accuracy=0.6 len=97 N=5"},
        ]
        
        results = extract_top_k_prompts(events)
        assert [r.rank for r in results] == [0, 1, 2]
    
    def test_extract_skips_non_optimized_events(self):
        """Only optimized.scored events should be extracted."""
        events = [
            {"type": "prompt.learning.proposal.scored", 
             "message": "proposal[0] train_accuracy=0.5"},
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[0] train_accuracy=0.8"},
            {"type": "prompt.learning.progress", 
             "message": "50% complete"},
        ]
        
        results = extract_top_k_prompts(events)
        assert len(results) == 1
        assert results[0].rank == 0
    
    def test_extract_empty_events_list(self):
        """Empty events list should return empty results."""
        results = extract_top_k_prompts([])
        assert results == []
    
    def test_extract_with_mixed_malformed_events(self):
        """Malformed events should be skipped, valid ones extracted."""
        events = [
            {"type": "prompt.learning.optimized.scored", "message": "malformed"},
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[0] train_accuracy=0.8 len=105 N=5"},
            {"type": "prompt.learning.optimized.scored", "message": ""},
            {"type": "prompt.learning.optimized.scored", 
             "message": "optimized[1] train_accuracy=0.6 len=97 N=5"},
        ]
        
        results = extract_top_k_prompts(events)
        assert len(results) == 2
        assert results[0].rank == 0
        assert results[1].rank == 1


class TestTopKPromptResultValidation:
    """Test validation and error handling in TopKPromptResult."""
    
    def test_negative_rank_raises(self):
        """Negative rank should raise assertion error."""
        with pytest.raises(AssertionError, match="rank must be >=0"):
            TopKPromptResult(rank=-1, train_accuracy=0.5, 
                           val_accuracy=None, prompt_preview="")
    
    def test_train_accuracy_out_of_range(self):
        """Train accuracy outside [0,1] should raise."""
        with pytest.raises(AssertionError, match="train_accuracy must be"):
            TopKPromptResult(rank=0, train_accuracy=1.5, 
                           val_accuracy=None, prompt_preview="")
        
        with pytest.raises(AssertionError, match="train_accuracy must be"):
            TopKPromptResult(rank=0, train_accuracy=-0.1, 
                           val_accuracy=None, prompt_preview="")
    
    def test_val_accuracy_out_of_range(self):
        """Val accuracy outside [0,1] should raise."""
        with pytest.raises(AssertionError, match="val_accuracy must be"):
            TopKPromptResult(rank=0, train_accuracy=0.5, 
                           val_accuracy=1.5, prompt_preview="")
    
    def test_none_val_accuracy_allowed(self):
        """None val_accuracy should be allowed."""
        result = TopKPromptResult(rank=0, train_accuracy=0.5, 
                                 val_accuracy=None, prompt_preview="")
        assert result.val_accuracy is None


class TestTopKPromptResultSerialization:
    """Test to_dict() serialization."""
    
    def test_to_dict_with_all_fields(self):
        """to_dict with all fields should produce complete dict."""
        result = TopKPromptResult(
            rank=0,
            train_accuracy=0.8,
            val_accuracy=0.62,
            prompt_preview="[SYSTEM]: Test prompt",
        )
        
        d = result.to_dict()
        assert d == {
            "rank": 0,
            "train_accuracy": 0.8,
            "val_accuracy": 0.62,
            "prompt_preview": "[SYSTEM]: Test prompt",
        }
    
    def test_to_dict_with_none_val_accuracy(self):
        """to_dict with None val_accuracy should include None."""
        result = TopKPromptResult(
            rank=0,
            train_accuracy=0.8,
            val_accuracy=None,
            prompt_preview="",
        )
        
        d = result.to_dict()
        assert d["val_accuracy"] is None
    
    def test_to_dict_json_serializable(self):
        """to_dict output should be JSON-serializable."""
        import json
        
        result = TopKPromptResult(
            rank=0,
            train_accuracy=0.8,
            val_accuracy=0.62,
            prompt_preview="Test",
        )
        
        d = result.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed == d


# Test fixtures
@pytest.fixture
def sample_event_stream():
    """A realistic event stream from a GEPA run."""
    return [
        # Progress events
        {"type": "prompt.learning.progress", "message": "10% complete"},
        
        # Proposal events (should be ignored for top-K)
        {"type": "prompt.learning.proposal.scored", 
         "message": "proposal[0] train_accuracy=0.2 len=100 N=5"},
        
        # Validation events
        {"type": "prompt.learning.validation.scored", 
         "message": "baseline val_accuracy=0.620 (N=50)", 
         "data": {"accuracy": 0.620, "is_baseline": True}},
        {"type": "prompt.learning.validation.scored", 
         "message": "top[0] val_accuracy=0.620 (N=50) lift=0.0", 
         "data": {"accuracy": 0.620}},
        {"type": "prompt.learning.validation.scored", 
         "message": "top[1] val_accuracy=0.640 (N=50) lift=2.0", 
         "data": {"accuracy": 0.640}},
        {"type": "prompt.learning.validation.scored", 
         "message": "top[2] val_accuracy=0.620 (N=50) lift=0.0", 
         "data": {"accuracy": 0.620}},
        
        # Optimized events (these are the top-K)
        {"type": "prompt.learning.optimized.scored", 
         "message": "optimized[0] train_accuracy=0.4 len=97 N=5 val_accuracy=0.620"},
        {"type": "prompt.learning.optimized.scored", 
         "message": "optimized[1] train_accuracy=0.4 len=97 N=5 val_accuracy=0.640"},
        {"type": "prompt.learning.optimized.scored", 
         "message": "optimized[2] train_accuracy=0.8 len=105 N=5 val_accuracy=0.620"},
        
        # Completion events
        {"type": "prompt.learning.gepa.complete", "message": "GEPA complete"},
    ]


class TestRealisticEventStream:
    """Integration test with realistic event stream."""
    
    def test_extract_from_realistic_stream(self, sample_event_stream):
        """Extract top-K from realistic event stream."""
        results = extract_top_k_prompts(sample_event_stream)
        
        assert len(results) == 3
        
        # Check rank 0
        assert results[0].rank == 0
        assert results[0].train_accuracy == 0.4
        assert results[0].val_accuracy == 0.620
        
        # Check rank 1 (best val score)
        assert results[1].rank == 1
        assert results[1].train_accuracy == 0.4
        assert results[1].val_accuracy == 0.640
        
        # Check rank 2 (best train score)
        assert results[2].rank == 2
        assert results[2].train_accuracy == 0.8
        assert results[2].val_accuracy == 0.620
    
    def test_validation_scores_extracted_separately(self, sample_event_stream):
        """Validation scores should be extractable independently."""
        val_scores = extract_validation_scores(sample_event_stream)
        
        # Should have 3 top-K validation scores (baseline not included)
        assert val_scores == {0: 0.620, 1: 0.640, 2: 0.620}


