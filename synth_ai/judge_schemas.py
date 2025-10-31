"""
Judge API Contract Schemas

These schemas define the expected structure for requests and responses
to the judge scoring endpoint at POST /api/judge/v1/score.

This is the canonical contract that the backend MUST conform to.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


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
    
    This is the canonical contract that judge backends MUST return.
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

