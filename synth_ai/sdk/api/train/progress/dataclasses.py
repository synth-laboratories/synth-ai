"""Dataclasses for GEPA progress tracking.

These provide type-safe access to event data and tracking state.

## ProgramCandidate Model (First-Class Programs)

Candidates are multi-stage PROGRAMS, not single prompts. Each candidate has:
- `candidate_id`, `parent_id`, `generation`, `mutation_type`, `mutation_params`
- `stages: Dict[stage_id, StageInfo]` where StageInfo contains instruction, rules, temperature
- Optional: `seed_info`, `token_usage`, `val_accuracy`, `timing`

The `prompt_summary` field is DERIVED from stages for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Size Limits (match backend)
# =============================================================================
MAX_INSTRUCTION_LENGTH = 4000  # Max chars per stage instruction
MAX_ROLLOUT_SAMPLES = 5  # Max rollout samples per candidate
MAX_SEED_INFO_COUNT = 50  # Max seeds to include in seed_info


# =============================================================================
# StageInfo - Single stage of a multi-stage program
# =============================================================================
@dataclass
class StageInfo:
    """
    Single stage of a multi-stage program candidate.

    A program consists of one or more stages, each with:
    - instruction: The prompt text for this stage
    - rules: Optional rules/constraints dict
    - temperature: Optional temperature override
    - prompts: Optional list of prompt variants (for multi-prompt stages)
    """
    instruction: str
    rules: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    prompts: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None/empty values."""
        result: dict[str, Any] = {"instruction": self.instruction}
        if self.rules:
            result["rules"] = self.rules
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.prompts:
            result["prompts"] = self.prompts
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageInfo:
        """Create from dict."""
        return cls(
            instruction=str(d.get("instruction", "")),
            rules=d.get("rules", {}),
            temperature=d.get("temperature"),
            prompts=d.get("prompts"),
        )


# =============================================================================
# SeedInfo - Information about a single evaluated seed
# =============================================================================
@dataclass
class SeedInfo:
    """
    Information about a single evaluated seed.

    Includes the seed ID, query text, expected output, and optionally
    the model's prediction and whether it was correct.
    """
    seed: int
    query: str = ""
    expected: str = ""
    predicted: str | None = None
    correct: bool | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        result: dict[str, Any] = {"seed": self.seed, "query": self.query, "expected": self.expected}
        if self.predicted is not None:
            result["predicted"] = self.predicted
        if self.correct is not None:
            result["correct"] = self.correct
        if self.score is not None:
            result["score"] = self.score
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeedInfo:
        """Create from dict."""
        return cls(
            seed=d.get("seed", -1),
            query=d.get("query", ""),
            expected=d.get("expected", ""),
            predicted=d.get("predicted"),
            correct=d.get("correct"),
            score=d.get("score"),
        )


# =============================================================================
# TokenUsage - Per-candidate token usage
# =============================================================================
@dataclass
class TokenUsage:
    """Token usage for a candidate evaluation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenUsage:
        """Create from dict."""
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            reasoning_tokens=d.get("reasoning_tokens", 0),
            cached_tokens=d.get("cached_tokens", 0),
        )


# =============================================================================
# Size Limits (match backend)
# =============================================================================
MAX_INSTRUCTION_LENGTH = 4000  # Max chars per stage instruction
MAX_ROLLOUT_SAMPLES = 5  # Max rollout samples per candidate
MAX_SEED_INFO_COUNT = 50  # Max seeds to include in seed_info


# =============================================================================
# StageInfo - Single stage of a multi-stage program
# =============================================================================
@dataclass
class StageInfo:
    """
    Single stage of a multi-stage program candidate.

    A program consists of one or more stages, each with:
    - instruction: The prompt text for this stage
    - rules: Optional rules/constraints dict
    - temperature: Optional temperature override
    - prompts: Optional list of prompt variants (for multi-prompt stages)
    """
    instruction: str
    rules: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    prompts: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None/empty values."""
        result: dict[str, Any] = {"instruction": self.instruction}
        if self.rules:
            result["rules"] = self.rules
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.prompts:
            result["prompts"] = self.prompts
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageInfo:
        """Create from dict."""
        return cls(
            instruction=str(d.get("instruction", "")),
            rules=d.get("rules", {}),
            temperature=d.get("temperature"),
            prompts=d.get("prompts"),
        )


# =============================================================================
# SeedInfo - Information about a single evaluated seed
# =============================================================================
@dataclass
class SeedInfo:
    """
    Information about a single evaluated seed.

    Includes the seed ID, query text, expected output, and optionally
    the model's prediction and whether it was correct.
    """
    seed: int
    query: str = ""
    expected: str = ""
    predicted: str | None = None
    correct: bool | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        result: dict[str, Any] = {"seed": self.seed, "query": self.query, "expected": self.expected}
        if self.predicted is not None:
            result["predicted"] = self.predicted
        if self.correct is not None:
            result["correct"] = self.correct
        if self.score is not None:
            result["score"] = self.score
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeedInfo:
        """Create from dict."""
        return cls(
            seed=d.get("seed", -1),
            query=d.get("query", ""),
            expected=d.get("expected", ""),
            predicted=d.get("predicted"),
            correct=d.get("correct"),
            score=d.get("score"),
        )


# =============================================================================
# TokenUsage - Per-candidate token usage
# =============================================================================
@dataclass
class TokenUsage:
    """Token usage for a candidate evaluation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenUsage:
        """Create from dict."""
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            reasoning_tokens=d.get("reasoning_tokens", 0),
            cached_tokens=d.get("cached_tokens", 0),
        )


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
    """
    Info about a single candidate program.

    Candidates are multi-stage PROGRAMS, not single prompts. The `stages` field
    contains the structured program data, and `prompt_summary` is derived from stages.
    """

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

    # === First-class program structure ===
    stages: dict[str, StageInfo] = field(default_factory=dict)
    prompt_summary: str | None = None  # Derived from stages for backwards compatibility

    # === Mutation/lineage ===
    mutation_type: str | None = None
    mutation_params: dict[str, Any] | None = None
    transformation: dict[str, Any] | None = None

    # === Seed data ===
    seed_scores: list[dict[str, Any]] = field(default_factory=list)  # [{seed, score}, ...]
    seed_info: list[SeedInfo] = field(default_factory=list)  # Full seed metadata
    rollout_sample: list[RolloutSample] = field(default_factory=list)

    # === Token usage and cost ===
    token_usage: TokenUsage | None = None
    cost_usd: float | None = None

    # === Timing ===
    timestamp: float = 0.0
    timestamp_ms: int | None = None
    evaluation_duration_ms: int | None = None

    # === Evaluation details ===
    minibatch_scores: list[float] = field(default_factory=list)  # Per-minibatch scores during evaluation
    skip_reason: str | None = None  # Reason why candidate was skipped (e.g., "minibatch_rejected", "budget_exhausted")

    # === Raw data for debugging ===
    raw_data: dict[str, Any] = field(default_factory=dict)

    def get_prompt_summary(self, max_length: int = 500) -> str:
        """
        Derive prompt_summary from stages.

        Concatenates stage instructions, truncating if needed.
        """
        if self.prompt_summary:
            return self.prompt_summary

        if not self.stages:
            return ""

        parts = []
        for stage_id in sorted(self.stages.keys()):
            stage = self.stages[stage_id]
            instruction = stage.instruction
            if instruction:
                # Truncate individual stage if needed
                if len(instruction) > MAX_INSTRUCTION_LENGTH:
                    instruction = instruction[:MAX_INSTRUCTION_LENGTH] + "..."
                parts.append(f"[{stage_id.upper()}]: {instruction}")

        summary = "\n\n".join(parts)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary

    @classmethod
    def from_event_data(cls, data: dict[str, Any], candidate_id: str | None = None) -> CandidateInfo:
        """Create from SSE event data payload.

        Supports both legacy flat format and new program_candidate block format.
        """
        # Check for program_candidate block (preferred)
        program_candidate = data.get("program_candidate", {})
        if program_candidate:
            # Use program_candidate block as primary source
            data = {**data, **program_candidate}

        cid = candidate_id or data.get("version_id") or data.get("candidate_id") or "unknown"

        # Extract stages (first-class program structure)
        stages: dict[str, StageInfo] = {}
        stages_data = data.get("stages", {})
        if isinstance(stages_data, dict):
            for stage_id, stage_dict in stages_data.items():
                if isinstance(stage_dict, dict):
                    stages[stage_id] = StageInfo.from_dict(stage_dict)

        # Extract instance scores
        instance_scores = data.get("instance_scores", [])
        if not instance_scores:
            seed_eval_info = data.get("seed_eval_info", {})
            if isinstance(seed_eval_info, dict):
                instance_scores = seed_eval_info.get("instance_scores", [])

        # Extract seed_scores [{seed, score}, ...]
        seed_scores = data.get("seed_scores", [])

        # Extract seed_info [{seed, query, expected}, ...]
        seed_info: list[SeedInfo] = []
        seed_info_data = data.get("seed_info", [])
        if isinstance(seed_info_data, list):
            for info in seed_info_data:
                if isinstance(info, dict):
                    seed_info.append(SeedInfo.from_dict(info))

        # Extract rollout samples
        rollout_samples: list[RolloutSample] = []
        for sample in data.get("rollout_sample", []):
            if isinstance(sample, dict):
                rollout_samples.append(RolloutSample.from_dict(sample))

        # Extract token usage
        token_usage: TokenUsage | None = None
        token_usage_data = data.get("token_usage")
        if isinstance(token_usage_data, dict):
            token_usage = TokenUsage.from_dict(token_usage_data)

        # Derive prompt_summary from stages if not directly provided
        prompt_summary = data.get("prompt_summary") or data.get("prompt_text")
        if not prompt_summary and stages:
            parts = []
            for stage_id in sorted(stages.keys()):
                instruction = stages[stage_id].instruction
                if instruction:
                    parts.append(f"[{stage_id.upper()}]: {instruction}")
            prompt_summary = "\n".join(parts)

        # Extract minibatch_scores array (preferred) or fallback to single minibatch_score
        minibatch_scores = data.get("minibatch_scores", [])
        if not minibatch_scores and data.get("minibatch_score") is not None:
            # Fallback: convert single minibatch_score to array for backwards compatibility
            minibatch_scores = [data["minibatch_score"]]

        # Extract skip_reason
        skip_reason = data.get("skip_reason")

        return cls(
            candidate_id=cid,
            accuracy=data.get("accuracy") or data.get("score"),
            val_accuracy=data.get("val_accuracy") or data.get("full_score"),
            train_accuracy=data.get("train_accuracy") or data.get("minibatch_score"),
            generation=data.get("generation"),
            parent_id=data.get("parent_id"),
            is_pareto=data.get("is_pareto", False),
            accepted=data.get("accepted", False),
            instance_scores=instance_scores,
            seeds_evaluated=data.get("seeds_evaluated", []),
            stages=stages,
            prompt_summary=prompt_summary,
            mutation_type=data.get("mutation_type") or data.get("operator"),
            mutation_params=data.get("mutation_params"),
            transformation=data.get("transformation"),
            seed_scores=seed_scores,
            seed_info=seed_info,
            rollout_sample=rollout_samples,
            token_usage=token_usage,
            cost_usd=data.get("cost_usd"),
            timestamp=data.get("timestamp", 0.0),
            timestamp_ms=data.get("timestamp_ms"),
            evaluation_duration_ms=data.get("evaluation_duration_ms"),
            minibatch_scores=minibatch_scores,
            skip_reason=skip_reason,
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
    baseline_score: float | None = None
    timestamp_ms: int | None = None

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
            baseline_score=data.get("baseline_score"),
            timestamp_ms=data.get("timestamp_ms"),
        )


@dataclass
class BaselineInfo:
    """Baseline prompt info."""

    accuracy: float | None = None  # Training accuracy
    val_accuracy: float | None = None  # Validation accuracy
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
