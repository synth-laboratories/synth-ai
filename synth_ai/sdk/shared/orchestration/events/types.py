"""Event type definitions and enums for prompt learning jobs.

This module provides type-safe event types and enums used across:
- Backend event emission
- SDK event consumption
- Code generation for TypeScript/Rust

All enums use str as base to ensure JSON serialization produces string values.
"""

from __future__ import annotations

from enum import Enum
from typing import Set

# =============================================================================
# Job and Candidate Lifecycle Enums
# =============================================================================


class JobStatus(str, Enum):
    """Job lifecycle status values.

    IMPORTANT: Use these constants instead of string literals to prevent
    inconsistencies between backend/frontend (e.g., "succeeded" vs "completed").
    """

    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    IN_PROGRESS = "in_progress"  # Alias for RUNNING (OpenResponses convention)
    SUCCEEDED = "succeeded"
    COMPLETED = "completed"  # Alias for SUCCEEDED (OpenResponses convention)
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def terminal_statuses(cls) -> Set[JobStatus]:
        """Statuses that indicate the job has finished (success or failure)."""
        return {cls.SUCCEEDED, cls.COMPLETED, cls.FAILED, cls.CANCELLED}

    @classmethod
    def active_statuses(cls) -> Set[JobStatus]:
        """Statuses that indicate the job is still running or pending."""
        return {cls.QUEUED, cls.PENDING, cls.RUNNING, cls.IN_PROGRESS}


class Phase(str, Enum):
    """Optimization phases for GEPA and MIPRO.

    GEPA phases: bootstrap -> optimization -> validation -> complete
    MIPRO phases: bootstrap -> optimization -> complete (+ optional test)
    """

    BOOTSTRAP = "bootstrap"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    TEST = "test"
    COMPLETE = "complete"


class CandidateStatus(str, Enum):
    """Status of a candidate prompt during optimization.

    Used in candidate.evaluated events to indicate outcome.
    """

    # Evaluation states
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    IN_PROGRESS = "in_progress"  # OpenResponses convention

    # Terminal states
    COMPLETED = "completed"  # OpenResponses convention
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DOMINATED = "dominated"  # Removed from pareto frontier
    FAILED = "failed"


class MutationType(str, Enum):
    """Types of mutations/transformations applied to prompts.

    Used for tracking mutation effectiveness and frontend display.
    """

    INITIAL = "initial"
    INITIAL_POPULATION = "initial_population"
    SYNTH = "synth"
    LLM_GUIDED = "llm_guided"
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    TPE_SELECTED = "tpe_selected"  # MIPRO-specific
    OPTIMIZED = "optimized"
    UNKNOWN = "unknown"


class TerminationReason(str, Enum):
    """Reasons why optimization terminated.

    Used in job completion events to explain why the job ended.
    """

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_LIMIT = "time_limit"
    ROLLOUT_LIMIT = "rollout_limit"
    GENERATION_LIMIT = "max_generations_reached"
    NO_IMPROVEMENT = "no_improvement"
    CANCELLED = "cancelled"
    ERROR = "error"


class ErrorType(str, Enum):
    """Categories of errors that can occur during optimization.

    Used for error tracking and summary events.
    """

    TIMEOUT = "timeout"
    HTTP = "http"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    EXCEPTION = "exception"
    VALIDATION = "validation"


# =============================================================================
# Event Type Enum (Comprehensive Taxonomy)
# =============================================================================


class EventType(str, Enum):
    """All event types emitted during prompt learning jobs.

    Canonical naming convention: learning.policy.<algorithm>.<entity>.<action>

    The SSE broker maintains backwards compatibility by emitting both canonical
    and legacy event names via the alias system.

    Categories:
    - Lifecycle: job.queued, job.started, job.completed, job.failed
    - Progress: job.progress, phase.started
    - GEPA: learning.policy.gepa.* events
    - MIPRO: learning.policy.mipro.* events
    - Validation: validation.* events
    - Billing: billing.* events
    """

    # === Lifecycle Events (GEPA is the default algorithm for prompt learning) ===
    CREATED = "learning.policy.gepa.job.queued"
    STARTED = "learning.policy.gepa.job.started"
    COMPLETED = "learning.policy.gepa.job.completed"
    FAILED = "learning.policy.gepa.job.failed"

    # === Progress Events ===
    PROGRESS = "learning.policy.gepa.job.progress"
    PHASE_CHANGED = "learning.policy.gepa.phase.started"

    # === Candidate Events (shared) ===
    CANDIDATE_EVALUATION_STARTED = "learning.policy.gepa.candidate.started"
    PROPOSAL_SCORED = "learning.policy.gepa.candidate.evaluated"
    OPTIMIZED_SCORED = "learning.policy.gepa.candidate.evaluated"

    # === GEPA-specific Events ===
    GEPA_START = "learning.policy.gepa.job.started"
    GEPA_COMPLETE = "learning.policy.gepa.job.completed"
    GEPA_CANDIDATE_EVALUATED = "learning.policy.gepa.candidate.evaluated"
    GEPA_FRONTIER_UPDATED = "learning.policy.gepa.frontier.updated"
    GEPA_NEW_BEST = "learning.policy.gepa.candidate.new_best"
    GEPA_GENERATION_START = "learning.policy.gepa.generation.started"
    GEPA_GENERATION_COMPLETE = "learning.policy.gepa.generation.completed"
    GEPA_VARIATION_SCORE = "learning.policy.gepa.candidate.evaluated"
    GEPA_BASELINE = "learning.policy.gepa.candidate.evaluated"

    # === MIPRO-specific Events ===
    MIPRO_START = "learning.policy.mipro.job.started"
    MIPRO_COMPLETE = "learning.policy.mipro.job.completed"
    MIPRO_TRIAL_STARTED = "learning.policy.mipro.trial.started"
    MIPRO_TRIAL_COMPLETE = "learning.policy.mipro.trial.completed"
    MIPRO_NEW_INCUMBENT = "learning.policy.mipro.candidate.new_best"
    MIPRO_ITERATION_START = "learning.policy.mipro.iteration.started"
    MIPRO_ITERATION_COMPLETE = "learning.policy.mipro.iteration.completed"
    MIPRO_MINIBATCH_SCORE = "learning.policy.mipro.candidate.evaluated"
    MIPRO_FULL_SCORE = "learning.policy.mipro.candidate.evaluated"

    # === Validation Events ===
    VALIDATION_START = "learning.policy.gepa.validation.started"
    VALIDATION_SCORED = "learning.policy.gepa.validation.completed"
    VALIDATION_BASELINE_START = "learning.policy.gepa.validation.started"
    VALIDATION_TOPK_START = "learning.policy.gepa.validation.started"
    VALIDATION_TIMEOUT = "learning.policy.gepa.validation.failed"

    # === Rollout Events ===
    ROLLOUTS_START = "learning.policy.gepa.rollout.started"
    TRIAL_RESULTS = "learning.policy.gepa.rollout.completed"

    # === Billing Events (keep legacy format - not part of core event model) ===
    BILLING_START = "learning.policy.gepa.billing.started"
    BILLING_END = "learning.policy.gepa.billing.completed"
    BILLING_SANDBOXES = "learning.policy.gepa.billing.updated"
    BUDGET_REACHED = "learning.policy.gepa.job.failed"
    USAGE_RECORDED = "learning.policy.gepa.billing.updated"

    # === Error Events ===
    OPTIMIZATION_TIMEOUT = "learning.policy.gepa.job.failed"
    OPTIMIZATION_ERROR = "learning.policy.gepa.job.failed"

    # === Token Events ===
    POLICY_TOKENS = "learning.policy.gepa.job.progress"

    # === Results Events ===
    RESULTS_SAVED = "learning.policy.gepa.job.completed"

    # === Summary Events ===
    MUTATION_SUMMARY = "learning.policy.gepa.job.summary"
    SEED_ANALYSIS = "learning.policy.gepa.job.summary"

    # === Connection Lifecycle Events ===
    STREAM_CONNECTED = "learning.policy.gepa.job.started"


# =============================================================================
# Event Type Validation
# =============================================================================

# Cached set of all valid event type values
_ALL_EVENT_TYPES: Set[str] = {e.value for e in EventType}


def is_valid_event_type(event_type: str) -> bool:
    """Check if an event type string is a known event type."""
    return event_type in _ALL_EVENT_TYPES


def validate_event_type(event_type: str) -> str:
    """Validate and return the event type string.

    Raises ValueError if the event type is not recognized.
    """
    if event_type not in _ALL_EVENT_TYPES:
        raise ValueError(
            f"Unknown event type: {event_type!r}. "
            f"Use EventType enum or add new type to events/types.py"
        )
    return event_type


__all__ = [
    # Enums
    "JobStatus",
    "Phase",
    "CandidateStatus",
    "MutationType",
    "TerminationReason",
    "ErrorType",
    "EventType",
    # Validators
    "is_valid_event_type",
    "validate_event_type",
]
