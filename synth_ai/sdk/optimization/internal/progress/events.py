"""Centralized SSE event parsing with normalization.

This module handles the messy reality of SSE events:
- [MASKED] → gepa normalization
- Multiple event type patterns for the same logical event
- Extracting data from nested structures
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.progress.events.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "parse_optimization_event"):
        raise RuntimeError(
            "Rust core optimization event parser required; synth_ai_py is unavailable."
        )
    return synth_ai_py


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_instance_rewards(data: dict[str, Any]) -> list[float] | None:
    instance_objectives = data.get("instance_objectives")
    if not isinstance(instance_objectives, list):
        return None
    values: list[float] = []
    for item in instance_objectives:
        if isinstance(item, dict):
            objectives = item.get("objectives")
            if isinstance(objectives, dict):
                reward_val = _coerce_float(objectives.get("reward"))
            else:
                reward_val = _coerce_float(item.get("reward"))
        else:
            reward_val = None
        if reward_val is None:
            return None
        values.append(reward_val)
    return values if values else None


class EventCategory(Enum):
    """High-level event categories."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"
    FRONTIER = "frontier"
    PROGRESS = "progress"
    GENERATION = "generation"
    THROUGHPUT = "throughput"
    TERMINATION = "termination"
    COMPLETE = "complete"
    VALIDATION = "validation"
    USAGE = "usage"
    UNKNOWN = "unknown"


@dataclass
class ParsedEvent:
    """Base class for parsed events."""

    event_type: str  # Original event type (normalized)
    category: EventCategory
    data: dict[str, Any]
    seq: int | None = None
    timestamp_ms: int | None = None


@dataclass
class BaselineEvent(ParsedEvent):
    """Baseline evaluation event."""

    reward: float | None = None
    objectives: Optional[Dict[str, float]] = None
    instance_rewards: list[float] | None = None
    instance_objectives: Optional[List[Dict[str, float]]] = None
    prompt: dict[str, Any] | None = None


@dataclass
class CandidateEvent(ParsedEvent):
    """Candidate evaluation event."""

    candidate_id: str = ""
    reward: float | None = None
    objectives: Optional[Dict[str, float]] = None
    accepted: bool = False
    generation: int | None = None
    parent_id: str | None = None
    is_pareto: bool = False
    instance_rewards: list[float] | None = None
    instance_objectives: Optional[List[Dict[str, float]]] = None
    mutation_type: str | None = None


@dataclass
class FrontierEvent(ParsedEvent):
    """Pareto frontier update event."""

    frontier: list[str] | None = None
    added: list[str] | None = None
    removed: list[str] | None = None
    frontier_size: int = 0
    best_reward: float | None = None
    frontier_rewards: dict[str, float] | None = None
    frontier_objectives: Optional[List[Dict[str, float]]] = None


@dataclass
class ProgressEvent(ParsedEvent):
    """Progress update event."""

    rollouts_completed: int = 0
    rollouts_total: int | None = None
    trials_completed: int = 0
    best_reward: float | None = None
    baseline_reward: float | None = None


@dataclass
class GenerationEvent(ParsedEvent):
    """Generation complete event."""

    generation: int = 0
    best_reward: float = 0.0
    candidates_proposed: int = 0
    candidates_accepted: int = 0


@dataclass
class CompleteEvent(ParsedEvent):
    """Optimization complete event."""

    best_reward: float | None = None
    baseline_reward: float | None = None
    finish_reason: str | None = None
    total_candidates: int = 0


@dataclass
class TerminationEvent(ParsedEvent):
    """Termination triggered event."""

    reason: str = "unknown"


@dataclass
class UsageEvent(ParsedEvent):
    """Usage/cost event."""

    total_usd: float = 0.0
    tokens_usd: float = 0.0
    sandbox_usd: float = 0.0


class EventParser:
    """Centralized SSE event parsing with normalization.

    Handles:
    - [MASKED] → gepa normalization
    - Multiple event type patterns for the same logical event
    - Nested data extraction

    Example:
        parser = EventParser()
        event = parser.parse({"type": "prompt.learning.[MASKED].baseline", "data": {...}})
        if isinstance(event, BaselineEvent):
            print(f"Baseline reward: {event.reward}")
    """

    # Event type patterns for each category
    BASELINE_PATTERNS = (".baseline",)
    CANDIDATE_PATTERNS = (
        ".candidate.evaluated",
        ".candidate.new_best",
        ".proposal.scored",
        ".optimized.scored",
        ".candidate_scored",
    )
    FRONTIER_PATTERNS = (".frontier_updated",)
    PROGRESS_PATTERNS = (
        ".progress",
        ".rollouts_limit_progress",
        ".rollouts.progress",
        ".job.started",
        ".trial.started",
        ".trial.completed",
        ".iteration.started",
        ".iteration.completed",
    )
    GENERATION_PATTERNS = (".generation.complete", ".generation.completed", ".generation.started")
    THROUGHPUT_PATTERNS = (".throughput", ".rollout.concurrency", ".rollout_concurrency")
    TERMINATION_PATTERNS = (".termination.triggered",)
    COMPLETE_PATTERNS = (".complete",)  # But NOT .generation.complete
    VALIDATION_PATTERNS = (".validation.scored",)
    USAGE_PATTERNS = (".usage.recorded", ".billing.sandboxes")

    @staticmethod
    def normalize_type(event_type: str) -> str:
        """Normalize event types (handle [MASKED], variants, etc.)."""
        # Replace [MASKED] with gepa
        return event_type.replace("[MASKED]", "gepa")

    @classmethod
    def is_event(cls, event_type: str, *patterns: str) -> bool:
        """Check if event matches any pattern."""
        normalized = cls.normalize_type(event_type)
        return any(pattern in normalized or normalized.endswith(pattern) for pattern in patterns)

    @classmethod
    def get_category(cls, event_type: str) -> EventCategory:
        """Determine event category from type."""
        normalized = cls.normalize_type(event_type)

        # Check for generation.complete BEFORE checking for .complete
        # (since .generation.complete contains .complete)
        if cls.is_event(normalized, *cls.GENERATION_PATTERNS):
            return EventCategory.GENERATION

        if cls.is_event(normalized, *cls.BASELINE_PATTERNS):
            return EventCategory.BASELINE
        if cls.is_event(normalized, *cls.CANDIDATE_PATTERNS):
            return EventCategory.CANDIDATE
        if cls.is_event(normalized, *cls.FRONTIER_PATTERNS):
            return EventCategory.FRONTIER
        if cls.is_event(normalized, *cls.PROGRESS_PATTERNS):
            return EventCategory.PROGRESS
        if cls.is_event(normalized, *cls.THROUGHPUT_PATTERNS):
            return EventCategory.THROUGHPUT
        if cls.is_event(normalized, *cls.TERMINATION_PATTERNS):
            return EventCategory.TERMINATION
        if cls.is_event(normalized, *cls.COMPLETE_PATTERNS):
            return EventCategory.COMPLETE
        if cls.is_event(normalized, *cls.VALIDATION_PATTERNS):
            return EventCategory.VALIDATION
        if cls.is_event(normalized, *cls.USAGE_PATTERNS):
            return EventCategory.USAGE

        return EventCategory.UNKNOWN

    @classmethod
    def parse(cls, event: dict[str, Any]) -> ParsedEvent:
        """Parse raw SSE event into typed event.

        Args:
            event: Raw SSE event dict with 'type' and 'data' keys

        Returns:
            Typed ParsedEvent subclass based on event category
        """
        rust = _require_rust()
        parsed = rust.parse_optimization_event(event)
        if isinstance(parsed, dict):
            return cls._from_rust(parsed)

        event_type = cls.normalize_type(event.get("type", ""))
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
        return ParsedEvent(
            event_type=event_type,
            category=EventCategory.UNKNOWN,
            data=data,
            seq=event.get("seq"),
            timestamp_ms=event.get("timestamp_ms"),
        )

    @staticmethod
    def _from_rust(parsed: dict[str, Any]) -> ParsedEvent:
        event_type = parsed.get("event_type") or ""
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
        seq = parsed.get("seq")
        timestamp_ms = parsed.get("timestamp_ms")
        category_raw = parsed.get("category", EventCategory.UNKNOWN.value)
        try:
            category = EventCategory(category_raw)
        except ValueError:
            category = EventCategory.UNKNOWN

        if category == EventCategory.BASELINE:
            return BaselineEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                reward=parsed.get("reward") or parsed.get("accuracy"),
                objectives=parsed.get("objectives"),
                instance_rewards=parsed.get("instance_rewards") or parsed.get("instance_scores"),
                instance_objectives=parsed.get("instance_objectives"),
                prompt=parsed.get("prompt"),
            )

        if category == EventCategory.CANDIDATE:
            return CandidateEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                candidate_id=parsed.get("candidate_id") or "",
                reward=parsed.get("reward") or parsed.get("accuracy"),
                objectives=parsed.get("objectives"),
                accepted=parsed.get("accepted", False),
                generation=parsed.get("generation"),
                parent_id=parsed.get("parent_id"),
                is_pareto=parsed.get("is_pareto", False),
                instance_rewards=parsed.get("instance_rewards") or parsed.get("instance_scores"),
                instance_objectives=parsed.get("instance_objectives"),
                mutation_type=parsed.get("mutation_type"),
            )

        if category == EventCategory.FRONTIER:
            return FrontierEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                frontier=parsed.get("frontier"),
                added=parsed.get("added"),
                removed=parsed.get("removed"),
                frontier_size=parsed.get("frontier_size", 0),
                best_reward=parsed.get("best_reward") or parsed.get("best_score"),
                frontier_rewards=parsed.get("frontier_rewards") or parsed.get("frontier_scores"),
                frontier_objectives=parsed.get("frontier_objectives"),
            )

        if category == EventCategory.PROGRESS:
            return ProgressEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                rollouts_completed=parsed.get("rollouts_completed", 0),
                rollouts_total=parsed.get("rollouts_total"),
                trials_completed=parsed.get("trials_completed", 0),
                best_reward=parsed.get("best_reward") or parsed.get("best_score"),
                baseline_reward=parsed.get("baseline_reward") or parsed.get("baseline_score"),
            )

        if category == EventCategory.GENERATION:
            return GenerationEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                generation=parsed.get("generation", 0),
                best_reward=parsed.get("best_reward") or parsed.get("best_accuracy", 0.0),
                candidates_proposed=parsed.get("candidates_proposed", 0),
                candidates_accepted=parsed.get("candidates_accepted", 0),
            )

        if category == EventCategory.COMPLETE:
            return CompleteEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                best_reward=parsed.get("best_reward") or parsed.get("best_score"),
                baseline_reward=parsed.get("baseline_reward") or parsed.get("baseline_score"),
                finish_reason=parsed.get("finish_reason"),
                total_candidates=parsed.get("total_candidates", 0),
            )

        if category == EventCategory.TERMINATION:
            return TerminationEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                reason=parsed.get("reason", "unknown"),
            )

        if category == EventCategory.USAGE:
            return UsageEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                total_usd=parsed.get("total_usd", 0.0),
                tokens_usd=parsed.get("tokens_usd", 0.0),
                sandbox_usd=parsed.get("sandbox_usd", 0.0),
            )

        return ParsedEvent(
            event_type=event_type,
            category=category,
            data=data,
            seq=seq,
            timestamp_ms=timestamp_ms,
        )
