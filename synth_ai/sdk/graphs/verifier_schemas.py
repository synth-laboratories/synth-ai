"""
Verifier API Contract Schemas

These schemas define the expected structure for requests and responses
to the verifier scoring endpoint at POST /api/graphs/verifiers/completions.
Zero-shot verifier graphs use the same response format via POST /api/graphs/completions.

This is the canonical contract that the backend MUST conform to.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CriterionScorePayload(BaseModel):
    """Per-criterion score returned by the verifier."""

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


class VerifierScoreResponse(BaseModel):
    """
    Response body for POST /api/graphs/verifiers/completions.

    This is the canonical contract that verifier backends MUST return and is
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
    objectives: Dict[str, float] = Field(
        default_factory=dict,
        description="Canonical objectives dict (e.g., {'reward': score})",
    )
    outcome_objectives: Optional[Dict[str, float]] = Field(
        default=None,
        description="Outcome objectives (preferred over legacy objectives when present).",
    )
    event_objectives: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Per-event objectives aligned with trace events.",
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

class VerifierTaskApp(BaseModel):
    """Task application metadata."""
    
    id: str = Field(..., description="Task app identifier")
    base_url: Optional[str] = Field(None, description="Optional base URL for task app")


class VerifierOptions(BaseModel):
    """Verifier provider and configuration options."""
    
    provider: Optional[str] = Field(None, description="Verifier provider (e.g., 'openai', 'groq')")
    model: Optional[str] = Field(None, description="Model identifier")
    rubric_id: Optional[str] = Field(None, description="Rubric identifier")
    event: bool = Field(True, description="Enable event-level verification")
    outcome: bool = Field(True, description="Enable outcome-level verification")


class VerifierTracePayload(BaseModel):
    """Trace payload containing trajectory context."""
    
    event_history: list[dict[str, Any]] = Field(..., description="List of events/steps")
    markov_blanket_message_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional message history for context"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Trace metadata")


class VerifierScoreRequest(BaseModel):
    """Request body for POST /api/graphs/verifiers/completions."""
    
    policy_name: str = Field(..., description="Name of the policy being evaluated")
    task_app: VerifierTaskApp = Field(..., description="Task application metadata")
    trace: VerifierTracePayload = Field(..., description="Trajectory trace to evaluate")
    options: VerifierOptions = Field(default_factory=lambda: VerifierOptions(), description="Verifier options")
    rubric: Optional[dict[str, Any]] = Field(None, description="Optional explicit rubric criteria")


# Verifier input validation schemas

class CalibrationExampleInput(BaseModel):
    """Input schema for a calibration example (few-shot verifier).
    
    Validates that the example has a valid trace and matching rewards/objectives.
    Uses synth_ai.data.rewards.CalibrationExample dataclass for structure.
    """
    
    session_trace: dict[str, Any] = Field(
        ..., description="V3 SessionTrace format (validated separately)"
    )
    event_rewards: Optional[list[Annotated[float, Field(ge=0.0, le=1.0)]]] = Field(
        default=None,
        description="List of rewards per event (0.0-1.0), must match number of events in trace",
    )
    outcome_reward: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None,
        description="Overall outcome reward (0.0-1.0)",
    )
    event_objectives: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Optional per-event objectives aligned with trace events.",
    )
    outcome_objectives: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional outcome objectives (preferred over outcome_reward when present).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    @model_validator(mode="before")
    @classmethod
    def normalize_trace(cls, data: Any) -> Any:
        if isinstance(data, dict) and "session_trace" not in data and "trace" in data:
            data = dict(data)
            data["session_trace"] = data.pop("trace")
        return data
    
    @model_validator(mode="after")
    def validate_rewards_match_trace(self) -> "CalibrationExampleInput":
        """Validate that event rewards/objectives length matches trace events."""
        # Count events in trace
        trace_events = self._count_trace_events()
        event_rewards = self._resolve_event_rewards()
        if len(event_rewards) != trace_events:
            raise ValueError(
                f"event rewards length ({len(event_rewards)}) doesn't match trace events ({trace_events}). "
                f"Each event in the trace must have a corresponding reward objective."
            )
        _ = self._resolve_outcome_reward()
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

    def _resolve_event_rewards(self) -> list[float]:
        if self.event_objectives is not None:
            if not isinstance(self.event_objectives, list):
                raise ValueError("event_objectives must be a list")
            rewards: list[float] = []
            for idx, item in enumerate(self.event_objectives):
                if not isinstance(item, dict):
                    raise ValueError(f"event_objectives[{idx}] must be a mapping")
                objectives = item.get("objectives") if isinstance(item.get("objectives"), dict) else item
                reward_val = objectives.get("reward") if isinstance(objectives, dict) else None
                if not isinstance(reward_val, (int, float)) or isinstance(reward_val, bool):
                    raise ValueError(f"event_objectives[{idx}].reward must be a number")
                rewards.append(float(reward_val))
            return rewards

        if self.event_rewards is None:
            raise ValueError("event_rewards or event_objectives is required")
        if not isinstance(self.event_rewards, list):
            raise ValueError("event_rewards must be a list")
        return [float(r) for r in self.event_rewards]

    def _resolve_outcome_reward(self) -> float:
        if self.outcome_objectives is not None:
            if not isinstance(self.outcome_objectives, dict):
                raise ValueError("outcome_objectives must be a mapping")
            reward_val = self.outcome_objectives.get("reward")
            if not isinstance(reward_val, (int, float)) or isinstance(reward_val, bool):
                raise ValueError("outcome_objectives.reward must be a number")
            return float(reward_val)

        if self.outcome_reward is None:
            raise ValueError("outcome_reward or outcome_objectives is required")
        return float(self.outcome_reward)
    
    def to_dataclass(self) -> "CalibrationExample":
        """Convert to synth_ai.data.rewards.CalibrationExample dataclass."""
        from synth_ai.data.rewards import CalibrationExample
        return CalibrationExample(
            session_trace=self.session_trace,
            event_rewards=self._resolve_event_rewards(),
            outcome_reward=self._resolve_outcome_reward(),
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

    @model_validator(mode="before")
    @classmethod
    def normalize_trace(cls, data: Any) -> Any:
        if isinstance(data, dict) and "session_trace" not in data and "trace" in data:
            data = dict(data)
            data["session_trace"] = data.pop("trace")
        return data
    
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
