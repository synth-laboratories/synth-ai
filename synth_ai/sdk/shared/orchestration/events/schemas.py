"""Event data schemas for prompt learning jobs.

This module defines structured event data that work across algorithms.
Algorithm-specific extensions can subclass BaseCandidateEventData.

Shared event types:
- prompt.learning.progress - Progress updates
- prompt.learning.completed - Job completion
- prompt.learning.started - Job started

Frontend expects (PromptNode interface in types.ts):
- version_id: string
- parent_id: string | null
- generation: number
- score/accuracy: number
- prompt_text: string
- mutation_type: string
- is_pareto: boolean

SDK compatibility:
- SDK parses `message` field for display (e.g., "proposal[0] train_accuracy=0.5")
- SDK uses `data` field for specific handlers
"""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# =============================================================================
# Size Limits for Event Payloads
# =============================================================================

MAX_INSTRUCTION_LENGTH = 4000  # Max chars per stage instruction
MAX_ROLLOUT_SAMPLES = 5  # Max rollout samples per candidate
MAX_SEED_INFO_COUNT = 50  # Max seeds to include in seed_info


# =============================================================================
# Summary Dataclasses
# =============================================================================


@dataclass
class MutationTypeStats:
    """Statistics for a single mutation type."""

    attempts: int
    acceptances: int
    acceptance_rate: float


@dataclass
class MutationSummary:
    """Summary of mutation effectiveness across all types."""

    by_mutation_type: Dict[str, MutationTypeStats]
    total_attempts: int
    total_acceptances: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for event emission."""
        return {
            "by_mutation_type": {
                k: {
                    "attempts": v.attempts,
                    "acceptances": v.acceptances,
                    "acceptance_rate": v.acceptance_rate,
                }
                for k, v in self.by_mutation_type.items()
            },
            "total_attempts": self.total_attempts,
            "total_acceptances": self.total_acceptances,
        }


@dataclass
class SeedAnalysis:
    """Analysis of seed difficulty and solve rates."""

    hard_seeds: List[int]
    easy_seeds: List[int]
    baseline_failures: List[int]
    total_seeds_evaluated: int
    total_candidates: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for event emission."""
        return {
            "hard_seeds": self.hard_seeds,
            "easy_seeds": self.easy_seeds,
            "baseline_failures": self.baseline_failures,
            "total_seeds_evaluated": self.total_seeds_evaluated,
            "total_candidates": self.total_candidates,
        }


@dataclass
class PhaseSummary:
    """Summary data for a completed phase."""

    phase: str
    duration_seconds: Optional[float] = None
    rollouts_completed: Optional[int] = None
    candidates_evaluated: Optional[int] = None
    best_score: Optional[float] = None  # Deprecated legacy field
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for event emission."""
        result: Dict[str, Any] = {"phase": self.phase}
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        if self.rollouts_completed is not None:
            result["rollouts_completed"] = self.rollouts_completed
        if self.candidates_evaluated is not None:
            result["candidates_evaluated"] = self.candidates_evaluated
        if self.best_score is not None:
            result["best_reward"] = self.best_score
            result["best_score"] = self.best_score  # Deprecated legacy field
        result.update(self.extra)
        return result


# =============================================================================
# Stage and Seed Information
# =============================================================================


@dataclass
class StageInfo:
    """Single stage of a multi-stage program candidate.

    A program consists of one or more stages, each with:
    - instruction: The prompt text for this stage
    - rules: Optional rules/constraints dict
    - temperature: Optional temperature override
    - prompts: Optional list of prompt variants (for multi-prompt stages)
    """

    instruction: str
    rules: Dict[str, Any] = field(default_factory=dict)
    temperature: Optional[float] = None
    prompts: Optional[List[str]] = None  # For multi-prompt stages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None/empty values."""
        result: Dict[str, Any] = {"instruction": self.instruction}
        if self.rules:
            result["rules"] = self.rules
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.prompts:
            result["prompts"] = self.prompts
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StageInfo:
        """Create from dict."""
        return cls(
            instruction=str(d.get("instruction", "")),
            rules=d.get("rules", {}),
            temperature=d.get("temperature"),
            prompts=d.get("prompts"),
        )


@dataclass
class SeedInfo:
    """Information about a single evaluated seed.

    Includes the seed ID, query text, expected output, and optionally
    the model's prediction and whether it was correct.
    """

    seed: int
    query: str = ""
    expected: str = ""
    predicted: Optional[str] = None
    correct: Optional[bool] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        result: Dict[str, Any] = {
            "seed": self.seed,
            "query": self.query,
            "expected": self.expected,
        }
        if self.predicted is not None:
            result["predicted"] = self.predicted
        if self.correct is not None:
            result["correct"] = self.correct
        if self.score is not None:
            result["score"] = self.score
        return result


@dataclass
class TokenUsage:
    """Token usage for a candidate evaluation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
        }


# =============================================================================
# Program Candidate (First-Class Programs)
# =============================================================================


@dataclass
class ProgramCandidate:
    """First-class representation of a program candidate.

    This is the core data structure for candidate events. Candidates are
    multi-stage PROGRAMS, not single prompts.

    In unified optimization mode, candidates can include both:
    - Prompt transformations (stages) - Interceptor-applied prompt deltas
    - Context overrides - Task-app context modifications (AGENTS.md, skills, etc.)
    """

    candidate_id: str
    generation: int
    stages: Dict[str, StageInfo]

    # Lineage
    parent_id: Optional[str] = None
    mutation_type: str = "unknown"
    mutation_params: Optional[Dict[str, Any]] = None

    # Scores
    accuracy: float = 0.0
    val_accuracy: Optional[float] = None
    minibatch_score: Optional[float] = None

    # Per-seed data
    seed_scores: Optional[List[Dict[str, Any]]] = None  # [{seed: int, score: float}, ...]
    seed_info: Optional[List[SeedInfo]] = None  # Full seed metadata
    instance_scores: Optional[List[Optional[float]]] = None
    objectives: Optional[Dict[str, float]] = None
    instance_objectives: Optional[List[Dict[str, float]]] = None
    newly_solved_seeds: Optional[List[int]] = None
    artifact_refs: Optional[List[Dict[str, Any]]] = None
    success_statuses: Optional[List[Dict[str, Any]]] = None

    # Token usage
    token_usage: Optional[TokenUsage] = None
    cost_usd: Optional[float] = None

    # Timing
    timestamp_ms: Optional[int] = None
    evaluation_duration_ms: Optional[int] = None

    # Transformation data (normalized)
    transformation: Optional[Dict[str, Any]] = None

    # Metadata
    prompt_length: Optional[int] = None
    status: str = "evaluated"

    # Context override fields (unified optimization)
    context_override_bundle_id: Optional[str] = None  # Stable ID for context override bundle
    context_overrides: Optional[List[Dict[str, Any]]] = None  # List of context override specs
    override_application_status: Optional[str] = None  # "applied", "failed", "not_requested"
    override_application_errors: Optional[List[Dict[str, Any]]] = None  # Structured errors
    context_snapshot_ref: Optional[str] = None  # Context snapshot artifact ID

    def get_prompt_summary(self, max_length: int = 500) -> str:
        """Derive prompt_summary from stages for backwards compatibility.

        Concatenates stage instructions, truncating if needed.
        """
        if not self.stages:
            return ""

        parts = []
        for stage_id in sorted(self.stages.keys()):
            stage = self.stages[stage_id]
            instruction = (
                stage.instruction if isinstance(stage, StageInfo) else stage.get("instruction", "")
            )
            if instruction:
                # Truncate individual stage if needed
                if len(instruction) > MAX_INSTRUCTION_LENGTH:
                    instruction = instruction[:MAX_INSTRUCTION_LENGTH] + "..."
                parts.append(f"[{stage_id.upper()}]: {instruction}")

        summary = "\n\n".join(parts)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for event payload."""
        result: Dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "stages": {
                sid: (s.to_dict() if isinstance(s, StageInfo) else s)
                for sid, s in self.stages.items()
            },
            "accuracy": self.accuracy,
            "mutation_type": self.mutation_type,
            "status": self.status,
        }

        # Add optional fields if present
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        if self.mutation_params:
            result["mutation_params"] = self.mutation_params
        if self.val_accuracy is not None:
            result["val_accuracy"] = self.val_accuracy
        if self.minibatch_score is not None:
            result["minibatch_score"] = self.minibatch_score
        if self.seed_scores:
            result["seed_scores"] = self.seed_scores
        if self.seed_info:
            result["seed_info"] = [s.to_dict() for s in self.seed_info[:MAX_SEED_INFO_COUNT]]
        if self.instance_scores:
            result["instance_scores"] = self.instance_scores
        if self.objectives:
            result["objectives"] = self.objectives
        if self.instance_objectives:
            result["instance_objectives"] = self.instance_objectives
        if self.newly_solved_seeds:
            result["newly_solved_seeds"] = self.newly_solved_seeds
        if self.artifact_refs:
            result["artifact_refs"] = self.artifact_refs
        if self.success_statuses:
            result["success_statuses"] = self.success_statuses
        if self.token_usage:
            result["token_usage"] = self.token_usage.to_dict()
        if self.cost_usd is not None:
            result["cost_usd"] = self.cost_usd
        if self.timestamp_ms is not None:
            result["timestamp_ms"] = self.timestamp_ms
        if self.evaluation_duration_ms is not None:
            result["evaluation_duration_ms"] = self.evaluation_duration_ms
        if self.transformation:
            result["transformation"] = self.transformation
        if self.prompt_length is not None:
            result["prompt_length"] = self.prompt_length

        # Context override fields (unified optimization)
        if self.context_override_bundle_id is not None:
            result["context_override_bundle_id"] = self.context_override_bundle_id
        if self.context_overrides:
            result["context_overrides"] = self.context_overrides
        if self.override_application_status:
            result["override_application_status"] = self.override_application_status
        if self.override_application_errors:
            result["override_application_errors"] = self.override_application_errors
        if self.context_snapshot_ref is not None:
            result["context_snapshot_ref"] = self.context_snapshot_ref

        # Derived field for backwards compatibility
        result["prompt_summary"] = self.get_prompt_summary()

        return result


# =============================================================================
# Base Candidate Event Data (for subclassing)
# =============================================================================


@dataclass
class BaseCandidateEventData(ABC):
    """Base data structure for ALL candidate evaluation events.

    Subclass this for algorithm-specific extensions:
    - GEPACandidateEventData (pareto, frontier fields)
    - MIPROCandidateEventData (trial, iteration fields)

    ## Canonical Field Names (prefer these):
    - `candidate_id` over `version_id`
    - `mean_reward` over `accuracy`/`reward`
    - `instance_rewards` over `instance_scores`

    ## Unified Optimization Fields (optional):
    - `artifact_refs`: List of artifact IDs (NOT full content)
    - `context_snapshot_ref`: Context snapshot artifact ID
    - `override_application_status`: "applied", "failed", "not_requested"
    """

    # === Required for frontend graph ===
    version_id: str  # DEPRECATED: Use candidate_id
    generation: int  # Generation number (0 = initial)
    accuracy: float  # DEPRECATED: Use mean_reward
    candidate_id: Optional[str] = None  # CANONICAL: Unique identifier for this candidate
    reward: Optional[float] = None  # DEPRECATED: Use mean_reward
    mean_reward: Optional[float] = None  # CANONICAL: Average reward across seeds

    # === Lineage ===
    parent_id: Optional[str] = None

    # === Display ===
    prompt_text: str = ""
    mutation_type: str = "unknown"

    # === Structured prompt data (for stage-level UI) ===
    stages: Optional[Dict[str, Dict[str, Any]]] = None

    # === Scores ===
    minibatch_score: Optional[float] = None  # Train score
    full_score: Optional[float] = None  # Validation score

    # === Per-seed scores ===
    seed_scores: Optional[List[Dict[str, Any]]] = None  # [{seed: int, score: float}, ...]
    instance_rewards: Optional[List[float]] = None  # CANONICAL: Per-seed rewards
    objectives: Optional[Dict[str, float]] = None
    instance_objectives: Optional[List[Dict[str, float]]] = None
    newly_solved_seeds: Optional[List[int]] = None
    artifact_refs: Optional[List[Dict[str, Any]]] = None
    success_statuses: Optional[List[Dict[str, Any]]] = None

    # === Metadata ===
    prompt_length: Optional[int] = None
    status: str = "evaluated"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values.

        Emits both canonical and legacy field names for backward compatibility.
        """
        result: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value

        # Ensure canonical fields are always populated
        if "candidate_id" not in result:
            result["candidate_id"] = self.version_id

        # mean_reward is canonical, but emit legacy aliases too
        canonical_reward = self.mean_reward if self.mean_reward is not None else self.accuracy
        result["mean_reward"] = canonical_reward
        result["reward"] = canonical_reward  # legacy alias
        result["accuracy"] = canonical_reward  # legacy alias

        # instance_rewards is canonical, emit as instance_scores too for legacy
        if self.instance_rewards is not None:
            result["instance_rewards"] = self.instance_rewards
            result["instance_scores"] = self.instance_rewards  # legacy alias

        if self.objectives is not None:
            result["objectives"] = self.objectives
        if self.instance_objectives is not None:
            result["instance_objectives"] = self.instance_objectives

        return result


__all__ = [
    # Constants
    "MAX_INSTRUCTION_LENGTH",
    "MAX_ROLLOUT_SAMPLES",
    "MAX_SEED_INFO_COUNT",
    # Summary classes
    "MutationTypeStats",
    "MutationSummary",
    "SeedAnalysis",
    "PhaseSummary",
    # Stage/seed classes
    "StageInfo",
    "SeedInfo",
    "TokenUsage",
    # Program candidate
    "ProgramCandidate",
    # Base event data
    "BaseCandidateEventData",
]
