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
    JUDGE = "judge"
    FUSED = "fused"


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
    "AdaptiveCurriculumLevel",
    "AdaptiveBatchLevel",
    "SynthModelName",
    "SYNTH_MODEL_NAMES",
]

