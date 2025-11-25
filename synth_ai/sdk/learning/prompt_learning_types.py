"""Type definitions for prompt learning data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextReplacement:
    """A text replacement in a prompt transformation."""
    
    new_text: str
    apply_to_role: str = "system"
    old_text: Optional[str] = None
    position: Optional[int] = None


@dataclass
class CandidateScore:
    """Scoring information for a candidate prompt."""
    
    accuracy: float
    prompt_length: int = 0
    tool_call_rate: float = 0.0
    instance_scores: List[float] = field(default_factory=list)


@dataclass
class PromptSection:
    """A section of a prompt (e.g., system, user, assistant)."""
    
    role: str
    content: str


@dataclass
class Candidate:
    """A candidate prompt from the optimization process."""
    
    accuracy: float
    prompt_length: int = 0
    tool_call_rate: float = 0.0
    instance_scores: List[float] = field(default_factory=list)
    object: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Candidate:
        """Create a Candidate from a dictionary."""
        return cls(
            accuracy=data.get("accuracy", 0.0),
            prompt_length=data.get("prompt_length", 0),
            tool_call_rate=data.get("tool_call_rate", 0.0),
            instance_scores=data.get("instance_scores", []),
            object=data.get("object"),
        )


@dataclass
class OptimizedCandidate:
    """An optimized candidate from the Pareto frontier."""
    
    score: CandidateScore
    payload_kind: str  # "transformation" or "template"
    object: Optional[Dict[str, Any]] = None
    instance_scores: Optional[List[float]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OptimizedCandidate:
        """Create an OptimizedCandidate from a dictionary."""
        score_data = data.get("score", {})
        if isinstance(score_data, dict):
            score = CandidateScore(
                accuracy=score_data.get("accuracy", 0.0),
                prompt_length=score_data.get("prompt_length", 0),
                tool_call_rate=score_data.get("tool_call_rate", 0.0),
                instance_scores=score_data.get("instance_scores", []),
            )
        else:
            score = CandidateScore(accuracy=0.0)
        
        return cls(
            score=score,
            payload_kind=data.get("payload_kind", "unknown"),
            object=data.get("object"),
            instance_scores=data.get("instance_scores"),
        )


@dataclass
class PromptLearningEvent:
    """A generic prompt learning event."""
    
    type: str
    message: str
    data: Dict[str, Any]
    seq: int
    created_at: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptLearningEvent:
        """Create a PromptLearningEvent from a dictionary."""
        return cls(
            type=data.get("type", ""),
            message=data.get("message", ""),
            data=data.get("data", {}),
            seq=data.get("seq", 0),
            created_at=data.get("created_at"),
        )


@dataclass
class BestPromptEventData:
    """Data for prompt.learning.best.prompt event."""
    
    best_score: float
    best_prompt: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BestPromptEventData:
        """Create BestPromptEventData from a dictionary."""
        return cls(
            best_score=data.get("best_score", 0.0),
            best_prompt=data.get("best_prompt", {}),
        )


@dataclass
class FinalResultsEventData:
    """Data for prompt.learning.final.results event."""
    
    attempted_candidates: List[Dict[str, Any]]
    optimized_candidates: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FinalResultsEventData:
        """Create FinalResultsEventData from a dictionary."""
        return cls(
            attempted_candidates=data.get("attempted_candidates", []),
            optimized_candidates=data.get("optimized_candidates", []),
        )


@dataclass
class ValidationScoredEventData:
    """Data for prompt.learning.validation.scored event."""
    
    accuracy: float
    instance_scores: List[float] = field(default_factory=list)
    is_baseline: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationScoredEventData:
        """Create ValidationScoredEventData from a dictionary."""
        return cls(
            accuracy=data.get("accuracy", 0.0),
            instance_scores=data.get("instance_scores", []),
            is_baseline=data.get("is_baseline", False),
        )


@dataclass
class PromptResults:
    """Results from a completed prompt learning job."""
    
    best_prompt: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    top_prompts: List[Dict[str, Any]] = field(default_factory=list)
    optimized_candidates: List[Dict[str, Any]] = field(default_factory=list)
    attempted_candidates: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptResults:
        """Create PromptResults from a dictionary."""
        return cls(
            best_prompt=data.get("best_prompt"),
            best_score=data.get("best_score"),
            top_prompts=data.get("top_prompts", []),
            optimized_candidates=data.get("optimized_candidates", []),
            attempted_candidates=data.get("attempted_candidates", []),
            validation_results=data.get("validation_results", []),
        )


