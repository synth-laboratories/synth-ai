"""Centralized SSE event parsing with normalization.

This module handles the messy reality of SSE events:
- [MASKED] → gepa normalization
- Multiple event type patterns for the same logical event
- Extracting data from nested structures
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


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

    accuracy: float | None = None
    instance_scores: list[float] | None = None
    prompt: dict[str, Any] | None = None


@dataclass
class CandidateEvent(ParsedEvent):
    """Candidate evaluation event."""

    candidate_id: str = ""
    accuracy: float | None = None
    accepted: bool = False
    generation: int | None = None
    parent_id: str | None = None
    is_pareto: bool = False
    instance_scores: list[float] | None = None
    mutation_type: str | None = None


@dataclass
class FrontierEvent(ParsedEvent):
    """Pareto frontier update event."""

    frontier: list[str] | None = None
    added: list[str] | None = None
    removed: list[str] | None = None
    frontier_size: int = 0
    best_score: float | None = None
    frontier_scores: dict[str, float] | None = None


@dataclass
class ProgressEvent(ParsedEvent):
    """Progress update event."""

    rollouts_completed: int = 0
    rollouts_total: int | None = None
    trials_completed: int = 0
    best_score: float | None = None
    baseline_score: float | None = None


@dataclass
class GenerationEvent(ParsedEvent):
    """Generation complete event."""

    generation: int = 0
    best_accuracy: float = 0.0
    candidates_proposed: int = 0
    candidates_accepted: int = 0


@dataclass
class CompleteEvent(ParsedEvent):
    """Optimization complete event."""

    best_score: float | None = None
    baseline_score: float | None = None
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
            print(f"Baseline accuracy: {event.accuracy}")
    """

    # Event type patterns for each category
    BASELINE_PATTERNS = (".baseline",)
    CANDIDATE_PATTERNS = (".candidate.evaluated", ".proposal.scored", ".optimized.scored", ".candidate_scored")
    FRONTIER_PATTERNS = (".frontier_updated",)
    PROGRESS_PATTERNS = (".progress", ".rollouts_limit_progress")
    GENERATION_PATTERNS = (".generation.complete",)
    THROUGHPUT_PATTERNS = (".throughput",)
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
        event_type_raw = event.get("type", "")
        event_type = cls.normalize_type(event_type_raw)
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
        seq = event.get("seq")
        timestamp_ms = event.get("timestamp_ms")

        category = cls.get_category(event_type)

        if category == EventCategory.BASELINE:
            return BaselineEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                accuracy=data.get("accuracy") or data.get("baseline_score") or data.get("baseline_accuracy"),
                instance_scores=data.get("instance_scores"),
                prompt=data.get("prompt"),
            )

        if category == EventCategory.CANDIDATE:
            return CandidateEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                candidate_id=data.get("version_id") or data.get("candidate_id") or "",
                accuracy=data.get("accuracy") or data.get("score"),
                accepted=data.get("accepted", False),
                generation=data.get("generation"),
                parent_id=data.get("parent_id"),
                is_pareto=data.get("is_pareto", False),
                instance_scores=data.get("instance_scores"),
                mutation_type=data.get("mutation_type") or data.get("operator"),
            )

        if category == EventCategory.FRONTIER:
            return FrontierEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                frontier=data.get("frontier"),
                added=data.get("added"),
                removed=data.get("removed"),
                frontier_size=data.get("frontier_size", len(data.get("frontier", []))),
                best_score=data.get("best_score"),
                frontier_scores=data.get("frontier_scores"),
            )

        if category == EventCategory.PROGRESS:
            return ProgressEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                rollouts_completed=data.get("rollouts_completed") or data.get("rollouts_executed") or 0,
                rollouts_total=data.get("rollouts_total") or data.get("total_rollouts"),
                trials_completed=data.get("trials_completed") or 0,
                best_score=data.get("best_score"),
                baseline_score=data.get("baseline_score"),
            )

        if category == EventCategory.GENERATION:
            return GenerationEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                generation=data.get("generation", 0),
                best_accuracy=data.get("best_accuracy", 0.0),
                candidates_proposed=data.get("candidates_proposed", 0),
                candidates_accepted=data.get("candidates_accepted", 0),
            )

        if category == EventCategory.COMPLETE:
            return CompleteEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                best_score=data.get("best_score"),
                baseline_score=data.get("baseline_score"),
                finish_reason=data.get("finish_reason") or data.get("reason_terminated"),
                total_candidates=data.get("total_candidates", 0),
            )

        if category == EventCategory.TERMINATION:
            return TerminationEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                reason=data.get("reason", "unknown"),
            )

        if category == EventCategory.USAGE:
            return UsageEvent(
                event_type=event_type,
                category=category,
                data=data,
                seq=seq,
                timestamp_ms=timestamp_ms,
                total_usd=data.get("total_usd", 0.0),
                tokens_usd=data.get("usd_tokens", 0.0),
                sandbox_usd=data.get("sandbox_usd", 0.0),
            )

        # Default: generic ParsedEvent
        return ParsedEvent(
            event_type=event_type,
            category=category,
            data=data,
            seq=seq,
            timestamp_ms=timestamp_ms,
        )
