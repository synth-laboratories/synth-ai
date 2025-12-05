"""Dataclasses for GEPA progress tracking.

These provide type-safe access to event data and tracking state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutSample:
    """Sample rollout for a seed (max 3 per candidate, frontier-expanding only)."""

    seed: int
    query: str
    expected: str
    predicted: str
    correct: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RolloutSample:
        return cls(
            seed=d.get("seed", -1),
            query=d.get("query", ""),
            expected=d.get("expected", ""),
            predicted=d.get("predicted", ""),
            correct=d.get("correct", False),
        )


@dataclass
class CandidateInfo:
    """Info about a single candidate prompt."""

    candidate_id: str
    accuracy: float | None = None
    val_accuracy: float | None = None
    train_accuracy: float | None = None
    generation: int | None = None
    parent_id: str | None = None
    is_pareto: bool = False
    accepted: bool = False
    instance_scores: list[float] = field(default_factory=list)
    seeds_evaluated: list[int] = field(default_factory=list)
    prompt_summary: str | None = None
    transformation: dict[str, Any] | None = None
    mutation_type: str | None = None
    rollout_sample: list[RolloutSample] = field(default_factory=list)
    timestamp: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_event_data(cls, data: dict[str, Any], candidate_id: str | None = None) -> CandidateInfo:
        """Create from SSE event data payload."""
        cid = candidate_id or data.get("version_id") or data.get("candidate_id") or "unknown"

        # Extract instance scores
        instance_scores = data.get("instance_scores", [])
        if not instance_scores:
            seed_eval_info = data.get("seed_eval_info", {})
            if isinstance(seed_eval_info, dict):
                instance_scores = seed_eval_info.get("instance_scores", [])

        # Extract rollout samples
        rollout_samples = []
        for sample in data.get("rollout_sample", []):
            if isinstance(sample, dict):
                rollout_samples.append(RolloutSample.from_dict(sample))

        return cls(
            candidate_id=cid,
            accuracy=data.get("accuracy") or data.get("score"),
            val_accuracy=data.get("val_accuracy"),
            train_accuracy=data.get("train_accuracy"),
            generation=data.get("generation"),
            parent_id=data.get("parent_id"),
            is_pareto=data.get("is_pareto", False),
            accepted=data.get("accepted", False),
            instance_scores=instance_scores,
            seeds_evaluated=data.get("seeds_evaluated", []),
            prompt_summary=data.get("prompt_summary") or data.get("prompt_text"),
            transformation=data.get("transformation"),
            mutation_type=data.get("mutation_type") or data.get("operator"),
            rollout_sample=rollout_samples,
            timestamp=data.get("timestamp", 0.0),
            raw_data=data,
        )


@dataclass
class FrontierUpdate:
    """Pareto frontier update event."""

    timestamp: float
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    frontier: list[str] = field(default_factory=list)
    frontier_scores: dict[str, float] = field(default_factory=dict)
    frontier_size: int = 0
    optimistic_score: float | None = None
    generation: int | None = None

    @classmethod
    def from_event_data(cls, data: dict[str, Any], timestamp: float = 0.0) -> FrontierUpdate:
        """Create from SSE event data payload."""
        return cls(
            timestamp=timestamp,
            added=data.get("added", []),
            removed=data.get("removed", []),
            frontier=data.get("frontier", []),
            frontier_scores=data.get("frontier_scores", {}),
            frontier_size=data.get("frontier_size", len(data.get("frontier", []))),
            optimistic_score=data.get("optimistic_score") or data.get("best_score"),
            generation=data.get("generation"),
        )


@dataclass
class BaselineInfo:
    """Baseline prompt info."""

    accuracy: float | None = None
    instance_scores: list[float] = field(default_factory=list)
    seeds_evaluated: list[int] = field(default_factory=list)
    prompt: dict[str, Any] | None = None
    rollout_sample: list[RolloutSample] = field(default_factory=list)

    @classmethod
    def from_event_data(cls, data: dict[str, Any]) -> BaselineInfo:
        """Create from SSE event data payload."""
        # Extract rollout samples
        rollout_samples = []
        for sample in data.get("rollout_sample", []):
            if isinstance(sample, dict):
                rollout_samples.append(RolloutSample.from_dict(sample))

        return cls(
            accuracy=data.get("accuracy") or data.get("baseline_score") or data.get("baseline_accuracy"),
            instance_scores=data.get("instance_scores", []),
            seeds_evaluated=data.get("seeds_evaluated", []),
            prompt=data.get("prompt"),
            rollout_sample=rollout_samples,
        )


@dataclass
class GenerationSummary:
    """Summary of a single generation."""

    generation: int
    candidates_proposed: int = 0
    candidates_accepted: int = 0
    best_accuracy: float = 0.0
    frontier_size: int = 0
    children: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float | None = None
    timestamp: float = 0.0


@dataclass
class GEPAProgress:
    """Current progress snapshot."""

    phase: str = "init"  # "init", "optimization", "validation", "complete", "failed"
    rollouts_completed: int = 0
    rollouts_total: int = 0
    generations_completed: int = 0
    candidates_evaluated: int = 0
    best_score: float = 0.0
    baseline_score: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    finish_reason: str | None = None

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage based on rollouts."""
        if self.rollouts_total > 0:
            return (self.rollouts_completed / self.rollouts_total) * 100
        return 0.0

    @property
    def lift(self) -> float | None:
        """Calculate improvement over baseline."""
        if self.baseline_score is not None:
            return self.best_score - self.baseline_score
        return None
