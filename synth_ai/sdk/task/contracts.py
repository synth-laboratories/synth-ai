"""Contracts for Task Apps.

Prefer synth_ai.sdk.localapi.contracts moving forward. This module remains for
backward compatibility during the naming transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RolloutMode(str, Enum):
    """Mode controls how rollout infrastructure processes inference URLs."""
    RL = "rl"
    EVAL = "eval"


class OutputMode(str, Enum):
    """Controls how the policy expects model outputs.

    - TOOL_CALLS: Use function/tool calling (default, current behavior)
    - TEXT: Plain text in message.content
    - STRUCTURED: JSON via response_format (OpenAI json_schema, Groq json_object, Gemini responseSchema)
    """
    TOOL_CALLS = "tool_calls"
    TEXT = "text"
    STRUCTURED = "structured"


class StructuredOutputConfig(BaseModel):
    """Configuration for structured output mode (OutputMode.STRUCTURED).

    Defines the JSON schema that the model must conform to when using structured outputs.
    This is normalized across providers:
    - OpenAI: response_format.json_schema
    - Groq: response_format.json_schema or json_object
    - Gemini: generationConfig.responseSchema
    """
    schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema for the expected response structure"
    )
    schema_name: str = Field(
        default="response",
        description="Name for the schema (required by some providers)"
    )
    strict: bool = Field(
        default=True,
        description="Whether to enforce strict schema validation (OpenAI strict mode)"
    )


@dataclass(frozen=True)
class TaskAppEndpoints:
    """Required Task App endpoints used by RL trainers and clients.

    Task Apps run as lightweight HTTP services (often on Modal) that expose these
    standard endpoints. Additional endpoints (proxies, debug routes) may be added
    by individual task apps as needed.
    """

    root: str = "/"
    health: str = "/health"
    info: str = "/info"
    task_info: str = "/task_info"
    rollout: str = "/rollout"


@dataclass(frozen=True)
class LocalAPIEndpoints(TaskAppEndpoints):
    """Alias for TaskAppEndpoints with LocalAPI naming."""


# --- Unified rollout schema used by Task App services and SDK utilities ---


class RolloutEnvSpec(BaseModel):
    env_id: str | None = None
    env_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None


class RolloutPolicySpec(BaseModel):
    policy_id: str | None = None
    policy_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    # Output mode configuration (defaults to tool_calls for backward compatibility)
    output_mode: OutputMode = Field(
        default=OutputMode.TOOL_CALLS,
        description="How the policy expects model outputs: tool_calls, text, or structured"
    )
    structured_config: StructuredOutputConfig | None = Field(
        default=None,
        description="Configuration for structured output mode (required if output_mode=STRUCTURED)"
    )


class RolloutRecordConfig(BaseModel):
    logprobs: bool = False
    value: bool = False
    return_trace: bool = False
    trace_format: Literal["compact", "full", "structured"] = "compact"


class RolloutSafetyConfig(BaseModel):
    max_time_s: float = 3600.0


class RolloutRequest(BaseModel):
    run_id: str
    env: RolloutEnvSpec
    policy: RolloutPolicySpec
    record: RolloutRecordConfig = RolloutRecordConfig()
    on_done: str = "reset"
    safety: RolloutSafetyConfig = RolloutSafetyConfig()
    training_session_id: str | None = None
    synth_base_url: str | None = None
    mode: RolloutMode = RolloutMode.RL  # Default to RL mode for training/optimization


class RolloutMetrics(BaseModel):
    episode_returns: list[float]
    mean_return: float
    num_steps: int
    num_episodes: int = 0
    outcome_score: float | None = None
    events_score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RolloutResponse(BaseModel):
    """Response from a rollout execution (trace-only)."""
    run_id: str
    branches: dict[str, list[str]] = Field(default_factory=dict)
    metrics: RolloutMetrics
    aborted: bool = False
    trace_correlation_id: str | None = None
    trace: dict[str, Any] | None = None
    pipeline_metadata: dict[str, Any] = Field(default_factory=dict)


class _ExtraAllowModel(BaseModel):
    """Base helper that preserves unknown keys while still exposing typed attributes."""

    model_config = ConfigDict(extra="allow")


class TaskDescriptor(_ExtraAllowModel):
    """Human-readable task identifiers shown in UIs and logs."""

    id: str
    name: str
    description: str | None = None
    version: str | None = None


class DatasetInfo(_ExtraAllowModel):
    """Metadata about the prompt/task dataset powering the environment."""

    id: str | None = None
    name: str | None = None
    version: str | None = None
    splits: list[str] | None = None
    default_split: str | None = None
    description: str | None = None


class RubricCriterion(_ExtraAllowModel):
    id: str
    description: str
    weight: float | None = None


class RubricSection(_ExtraAllowModel):
    name: str
    criteria: list[RubricCriterion] = Field(default_factory=list)


class RubricInfo(_ExtraAllowModel):
    """Outcome and event scoring definitions used by judges."""

    outcome: RubricSection | None = None
    events: RubricSection | None = None


class InferenceInfo(_ExtraAllowModel):
    """Recommended defaults for policy model routing."""

    model: str | None = None
    inference_url: str | None = None


class LimitsInfo(_ExtraAllowModel):
    """Operational limits the environment enforces."""

    max_turns: int | None = None
    max_response_tokens: int | None = None
    timeout_seconds: int | None = None


class TaskInfo(_ExtraAllowModel):
    """Static metadata describing the capabilities of a Task App task."""

    task: TaskDescriptor
    environment: str
    dataset: DatasetInfo
    rubric: RubricInfo
    inference: InferenceInfo
    limits: LimitsInfo
    task_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific extras (e.g. prompt version info, documentation links).",
    )
