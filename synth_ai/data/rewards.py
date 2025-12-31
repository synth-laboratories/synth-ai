"""Reward data structures.

This module defines pure data types for representing rewards in training
and evaluation contexts. These are actual data records, not API abstractions.

Synth AI uses two primary reward scopes:

- **Event Rewards**: Fine-grained rewards attached to individual events within a session
  (e.g., each tool call, each LLM response). Use `EventRewardRecord` to annotate specific
  events with reward values.

- **Outcome Rewards**: Episode-level rewards that summarize the overall success of a
  complete session. Use `OutcomeRewardRecord` for aggregate metrics.

Example usage:

```python
from synth_ai.data.rewards import EventRewardRecord, OutcomeRewardRecord

# Annotate a specific event with a reward
event_reward = EventRewardRecord(
    event_id="evt_123",
    session_id="sess_abc",
    reward_value=0.8,
    reward_type="evaluator",
    annotation={"reason": "Correct tool selection"}
)

# Record episode-level outcome
outcome = OutcomeRewardRecord(
    session_id="sess_abc",
    total_reward=0.85,
    achievements_count=3,
    total_steps=10,
    metadata={"task": "code_generation"}
)
```

See Also:
- Event rewards SDK guide: /sdk/tracing/rewards/event-rewards
- Outcome rewards SDK guide: /sdk/tracing/rewards/outcome-rewards
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class RewardRecord:
    """A single reward observation.

    Represents a reward signal at a specific point in a trajectory,
    with metadata about its source and scope.

    Attributes:
        value: The numeric reward value (typically in range [0, 1] or unbounded).
        reward_type: Category of reward - "shaped" (dense), "sparse" (terminal only),
            "achievement" (milestone), "penalty" (negative signal), "evaluator"
            (from LLM verifier), or "human" (manual annotation).
        scope: Granularity level - "step" (per action), "event" (per significant event),
            or "outcome" (episode-level).
        source: Origin of the reward - "environment" (task env), "runner" (framework),
            "evaluator" (verifier), or "human" (annotator).
        key: Optional identifier like achievement name or rubric criterion ID.
        turn: Turn number within the session where reward was earned.
        timestamp: When the reward was recorded.
        metadata: Additional context (e.g., rubric scores, evaluation details).
    """

    value: float
    reward_type: Literal["shaped", "sparse", "achievement", "penalty", "evaluator", "human"] = "shaped"
    scope: Literal["step", "event", "outcome"] = "step"
    source: Literal["environment", "runner", "evaluator", "human"] | None = None
    key: str | None = None  # e.g., achievement name
    turn: int | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeRewardRecord:
    """Episode-level reward summary.

    Aggregates reward information for a complete episode/session,
    including total reward, achievements, and step counts. This is the
    primary data structure for outcome rewards used in training.

    Attributes:
        session_id: Unique identifier linking to the SessionTrace.
        total_reward: Aggregate reward for the entire episode (typically 0.0-1.0).
        achievements_count: Number of achievements/milestones reached.
        total_steps: Total number of steps in the episode.
        metadata: Task-specific metadata (e.g., {"task": "code_gen", "difficulty": "hard"}).
        annotation: Human or evaluator annotations explaining the score.
        created_at: When this record was created.

    Example:
        ```python
        outcome = OutcomeRewardRecord(
            session_id="sess_abc123",
            total_reward=0.75,
            achievements_count=2,
            total_steps=8,
            metadata={"task": "customer_support"},
            annotation={"evaluator": "Resolved issue but could improve tone"}
        )
        ```
    """

    session_id: str
    total_reward: float
    achievements_count: int = 0
    total_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    annotation: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class EventRewardRecord:
    """Event-level reward annotation.

    Links a reward to a specific event in a trace, with optional
    annotations and source information. Event rewards provide fine-grained
    feedback on individual actions or decisions within a session.

    Attributes:
        event_id: Unique identifier of the event being rewarded.
        session_id: Session containing this event.
        reward_value: Reward for this specific event (typically 0.0-1.0).
        reward_type: Category of reward (e.g., "tool_success", "reasoning", "progress").
        key: Rubric criterion or achievement key this reward relates to.
        turn_number: Turn/step within the session where event occurred.
        source: Origin of the reward ("environment", "evaluator", "human").
        annotation: Explanation or details about why this reward was given.
        created_at: When this record was created.

    Example:
        ```python
        event_reward = EventRewardRecord(
            event_id="evt_tool_call_5",
            session_id="sess_abc123",
            reward_value=1.0,
            reward_type="tool_success",
            turn_number=3,
            source="environment",
            annotation={"tool": "search", "result": "found_answer"}
        )
        ```
    """

    event_id: str
    session_id: str
    reward_value: float
    reward_type: str | None = None
    key: str | None = None
    turn_number: int | None = None
    source: str | None = None
    annotation: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class RewardAggregates:
    """Aggregated statistics for a set of rewards."""

    mean: float
    median: float = 0.0
    std: float = 0.0
    n: int = 0
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class CalibrationExample:
    """A calibration example for few-shot verifier evaluation.
    
    Contains a full execution trace with its ground truth rewards.
    Used to teach the verifier evaluation patterns through labeled examples.
    """
    
    session_trace: dict[str, Any]  # V3 SessionTrace format (validated separately)
    event_rewards: list[float]  # List of rewards per event (0.0-1.0), one per event in trace
    outcome_reward: float  # Overall outcome reward (0.0-1.0)
    metadata: dict[str, Any] = field(default_factory=dict)  # Optional metadata
    
    def __post_init__(self) -> None:
        """Validate reward values."""
        # Validate event_rewards are in range
        for idx, reward in enumerate(self.event_rewards):
            if not isinstance(reward, (int, float)) or reward < 0.0 or reward > 1.0:
                raise ValueError(
                    f"event_rewards[{idx}] must be float 0.0-1.0, got {reward} (type: {type(reward).__name__})"
                )
        
        # Validate outcome_reward is in range
        if not isinstance(self.outcome_reward, (int, float)) or self.outcome_reward < 0.0 or self.outcome_reward > 1.0:
            raise ValueError(
                f"outcome_reward must be float 0.0-1.0, got {self.outcome_reward} (type: {type(self.outcome_reward).__name__})"
            )
        
        # Convert to float for consistency
        self.event_rewards = [float(r) for r in self.event_rewards]
        self.outcome_reward = float(self.outcome_reward)


@dataclass
class GoldExample:
    """A gold-standard example for contrastive verifier evaluation.
    
    Contains a correctly scored trace example that the verifier's judgment
    will be compared against. Used to evaluate verifier consistency.
    """
    
    summary: str  # Summary of the trace being evaluated
    gold_score: float  # Gold-standard score (0.0-1.0)
    gold_reasoning: str  # Gold-standard reasoning/explanation
    session_trace: dict[str, Any] | None = None  # Optional full trace (for richer evaluation)
    metadata: dict[str, Any] = field(default_factory=dict)  # Optional metadata
    
    def __post_init__(self) -> None:
        """Validate score and fields."""
        if not self.summary or not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("summary must be a non-empty string")
        
        if not isinstance(self.gold_score, (int, float)) or self.gold_score < 0.0 or self.gold_score > 1.0:
            raise ValueError(
                f"gold_score must be float 0.0-1.0, got {self.gold_score} (type: {type(self.gold_score).__name__})"
            )
        
        if not self.gold_reasoning or not isinstance(self.gold_reasoning, str) or not self.gold_reasoning.strip():
            raise ValueError("gold_reasoning must be a non-empty string")
        
        # Convert to float for consistency
        self.gold_score = float(self.gold_score)


__all__ = [
    "RewardRecord",
    "OutcomeRewardRecord",
    "EventRewardRecord",
    "RewardAggregates",
    "CalibrationExample",
    "GoldExample",
]
