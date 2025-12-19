"""
Verifier API Contract Schemas

These schemas define the expected structure for requests and responses
to the verifier scoring endpoint at POST /api/judge/v1/score. Zero-shot
verifier graphs use the same response format via POST /api/graphs/completions.

This is the canonical contract that the backend MUST conform to.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CriterionScorePayload(BaseModel):
    """Per-criterion score returned by the judge."""

    score: float = Field(..., description="Numeric score for this criterion")
    reason: str = Field(default="", description="Explanation for the score")
    weight: float = Field(default=1.0, description="Weight of this criterion")
    description: str = Field(default="", description="Description of the criterion")


class ReviewPayload(BaseModel):
    """Rubric review (event-level or outcome-level)."""

    criteria: dict[str, CriterionScorePayload] = Field(
        default_factory=dict,
        description="Map of criterion keys to their scores"
    )
    total: float = Field(default=0.0, description="Aggregated total score")
    summary: Optional[str] = Field(None, description="Optional text summary")


class JudgeScoreResponse(BaseModel):
    """
    Response body for POST /api/judge/v1/score.

    This is the canonical contract that judge backends MUST return and is
    also used as the zero-shot verifier graph output.
    """

    status: Literal["ok", "failed"] = Field(default="ok", description="Request status")
    event_reviews: list[ReviewPayload] = Field(
        default_factory=list,
        description="List of per-event rubric reviews (one per step)"
    )
    outcome_review: Optional[ReviewPayload] = Field(
        None,
        description="Optional outcome-level rubric review"
    )
    event_totals: list[float] = Field(
        default_factory=list,
        description="List of aggregated scores per event (matches event_reviews length)"
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details (provider, latency, etc.)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata (provider, options, etc.)"
    )

    def aggregate_event_reward(self) -> Optional[float]:
        """
        Aggregate all event totals into a single reward.
        
        Returns:
            Sum of all event_totals, or None if empty
        """
        if not self.event_totals:
            return None
        return sum(self.event_totals)

    def aggregate_outcome_reward(self) -> Optional[float]:
        """
        Extract outcome reward from outcome_review.
        
        Returns:
            outcome_review.total, or None if no outcome review
        """
        if self.outcome_review is None:
            return None
        return self.outcome_review.total


# Request schemas for completeness

class JudgeTaskApp(BaseModel):
    """Task application metadata."""
    
    id: str = Field(..., description="Task app identifier")
    base_url: Optional[str] = Field(None, description="Optional base URL for task app")


class JudgeOptions(BaseModel):
    """Judge provider and configuration options."""
    
    provider: Optional[str] = Field(None, description="Judge provider (e.g., 'openai', 'groq')")
    model: Optional[str] = Field(None, description="Model identifier")
    rubric_id: Optional[str] = Field(None, description="Rubric identifier")
    event: bool = Field(True, description="Enable event-level judging")
    outcome: bool = Field(True, description="Enable outcome-level judging")


class JudgeTracePayload(BaseModel):
    """Trace payload containing trajectory context."""
    
    event_history: list[dict[str, Any]] = Field(..., description="List of events/steps")
    markov_blanket_message_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional message history for context"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Trace metadata")


class JudgeScoreRequest(BaseModel):
    """Request body for POST /api/judge/v1/score."""
    
    policy_name: str = Field(..., description="Name of the policy being evaluated")
    task_app: JudgeTaskApp = Field(..., description="Task application metadata")
    trace: JudgeTracePayload = Field(..., description="Trajectory trace to evaluate")
    options: JudgeOptions = Field(default_factory=lambda: JudgeOptions(), description="Judge options")
    rubric: Optional[dict[str, Any]] = Field(None, description="Optional explicit rubric criteria")


# Verifier input validation schemas

class CalibrationExampleInput(BaseModel):
    """Input schema for a calibration example (few-shot verifier).
    
    Validates that the example has a valid trace and matching rewards.
    Uses synth_ai.data.rewards.CalibrationExample dataclass for structure.
    """
    
    session_trace: dict[str, Any] = Field(..., description="V3 SessionTrace format (validated separately)")
    event_rewards: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        ..., 
        description="List of rewards per event (0.0-1.0), must match number of events in trace"
    )
    outcome_reward: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., 
        description="Overall outcome reward (0.0-1.0)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    
    @model_validator(mode="after")
    def validate_rewards_match_trace(self) -> "CalibrationExampleInput":
        """Validate that event_rewards length matches trace events."""
        # Count events in trace
        trace_events = self._count_trace_events()
        if len(self.event_rewards) != trace_events:
            raise ValueError(
                f"event_rewards length ({len(self.event_rewards)}) doesn't match trace events ({trace_events}). "
                f"Each event in the trace must have a corresponding reward."
            )
        return self
    
    def _count_trace_events(self) -> int:
        """Count total events in session_trace."""
        count = 0
        # Try event_history first (V3 format)
        if isinstance(self.session_trace, dict):
            event_history = self.session_trace.get("event_history", [])
            if isinstance(event_history, list):
                return len(event_history)
            
            # Try session_time_steps format
            time_steps = self.session_trace.get("session_time_steps", [])
            if isinstance(time_steps, list):
                for step in time_steps:
                    if isinstance(step, dict):
                        events = step.get("events", [])
                        if isinstance(events, list):
                            count += len(events)
                return count
        return 0
    
    def to_dataclass(self) -> "CalibrationExample":
        """Convert to synth_ai.data.rewards.CalibrationExample dataclass."""
        from synth_ai.data.rewards import CalibrationExample
        return CalibrationExample(
            session_trace=self.session_trace,
            event_rewards=self.event_rewards,
            outcome_reward=self.outcome_reward,
            metadata=self.metadata,
        )


class GoldExampleInput(BaseModel):
    """Input schema for a gold example (contrastive verifier).
    
    Validates that the example has required fields with correct types.
    Uses synth_ai.data.rewards.GoldExample dataclass for structure.
    """
    
    summary: str = Field(..., min_length=1, description="Summary of the trace being evaluated")
    gold_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., 
        description="Gold-standard score (0.0-1.0)"
    )
    gold_reasoning: str = Field(..., min_length=1, description="Gold-standard reasoning/explanation")
    session_trace: Optional[dict[str, Any]] = Field(
        None, 
        description="Optional full trace (for richer evaluation)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    
    def to_dataclass(self) -> "GoldExample":
        """Convert to synth_ai.data.rewards.GoldExample dataclass."""
        from synth_ai.data.rewards import GoldExample
        return GoldExample(
            summary=self.summary,
            gold_score=self.gold_score,
            gold_reasoning=self.gold_reasoning,
            session_trace=self.session_trace,
            metadata=self.metadata,
        )
