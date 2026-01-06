"""Centralized enums for Synth AI SDK.

This module defines all enum types used across the SDK. Enums are pure data
with no dependencies on other modules.
"""

from __future__ import annotations

from enum import Enum


class JobType(str, Enum):
    """Types of training/optimization jobs."""

    PROMPT_LEARNING = "prompt_learning"
    SFT = "sft"
    RL = "rl"
    GSPO = "gspo"
    EVAL = "eval"
    RESEARCH_AGENT = "research_agent"


class JobStatus(str, Enum):
    """Status of a job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PromptLearningMethod(str, Enum):
    """Prompt optimization algorithms."""

    GEPA = "gepa"
    MIPRO = "mipro"


class RLMethod(str, Enum):
    """Reinforcement learning training algorithms."""

    PPO = "ppo"
    GRPO = "grpo"
    REINFORCE = "reinforce"


class SFTMethod(str, Enum):
    """Supervised fine-tuning approaches."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class InferenceMode(str, Enum):
    """Inference modes for policy evaluation."""

    SYNTH_HOSTED = "synth_hosted"


class ProviderName(str, Enum):
    """LLM provider names."""

    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


class RewardSource(str, Enum):
    """Source of reward signal for training."""

    TASK_APP = "task_app"
    VERIFIER = "verifier"
    FUSED = "fused"


class RewardType(str, Enum):
    """Category of reward signal."""

    SHAPED = "shaped"
    SPARSE = "sparse"
    PENALTY = "penalty"
    EVALUATOR = "evaluator"
    HUMAN = "human"
    ACHIEVEMENT = "achievement"
    ACHIEVEMENT_DELTA = "achievement_delta"
    UNIQUE_ACHIEVEMENT_DELTA = "unique_achievement_delta"


class RewardScope(str, Enum):
    """Granularity level for reward observations."""

    EVENT = "event"
    OUTCOME = "outcome"


class ObjectiveKey(str, Enum):
    """Canonical objective keys."""

    REWARD = "reward"
    LATENCY_MS = "latency_ms"
    COST_USD = "cost_usd"


class ObjectiveDirection(str, Enum):
    """Optimization direction for an objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class GraphType(str, Enum):
    """Graph categories for training and inference."""

    POLICY = "policy"
    VERIFIER = "verifier"
    RLM = "rlm"


class OptimizationMode(str, Enum):
    """Optimization strategy selection."""

    AUTO = "auto"
    GRAPH_ONLY = "graph_only"
    PROMPT_ONLY = "prompt_only"


class VerifierMode(str, Enum):
    """Verifier evaluation mode."""

    RUBRIC = "rubric"
    CONTRASTIVE = "contrastive"
    GOLD_EXAMPLES = "gold_examples"


class TrainingType(str, Enum):
    """Training/optimization category labels."""

    SFT = "sft"
    GEPA = "gepa"
    GRAPH_EVOLVE = "graph_evolve"
    GRAPHGEN = "graphgen"
    RL = "rl"
    GSPO = "gspo"


class AdaptiveCurriculumLevel(str, Enum):
    """Preset levels for adaptive pooling curriculum."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class AdaptiveBatchLevel(str, Enum):
    """Preset levels for adaptive batch curriculum (GEPA only)."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class SynthModelName(str, Enum):
    """Synth-hosted model names for routing through Synth backend."""

    SYNTH_SMALL = "synth-small"
    SYNTH_MEDIUM = "synth-medium"


# Legacy: list of model names for CLI validation
SYNTH_MODEL_NAMES: list[str] = [m.value for m in SynthModelName]


__all__ = [
    "JobType",
    "JobStatus",
    "PromptLearningMethod",
    "RLMethod",
    "SFTMethod",
    "InferenceMode",
    "ProviderName",
    "RewardSource",
    "RewardType",
    "RewardScope",
    "ObjectiveKey",
    "ObjectiveDirection",
    "GraphType",
    "OptimizationMode",
    "VerifierMode",
    "TrainingType",
    "AdaptiveCurriculumLevel",
    "AdaptiveBatchLevel",
    "SynthModelName",
    "SYNTH_MODEL_NAMES",
]
