"""Contracts for Task Apps.

Prefer synth_ai.sdk.localapi.contracts moving forward. This module remains for
backward compatibility during the naming transition.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    """Metrics from a rollout execution.

    ## Preferred Fields (New - Normalized)

    - `outcome_reward`: The reward for this rollout (PREFERRED)
    - `event_rewards`: Optional per-step rewards

    ## Legacy Fields (Backward Compatibility)

    - `episode_rewards`, `reward_mean`, `num_steps`: Still supported for backward
      compatibility. For new implementations, just use `outcome_reward`.
    - `outcome_score`: Alias for `outcome_reward` (deprecated)

    ## Example - Minimal (New Style)

        metrics = RolloutMetrics(
            outcome_reward=1.0,  # PREFERRED - just provide the reward
        )

    ## Example - Full (Backward Compatible)

        metrics = RolloutMetrics(
            episode_rewards=[1.0],
            reward_mean=1.0,
            num_steps=1,
            outcome_reward=1.0,  # PREFERRED
        )
    """

    # =========================================================================
    # PREFERRED FIELDS (New - Normalized)
    # =========================================================================
    outcome_objectives: Optional[Dict[str, float]] = Field(
        default=None,
        description="Canonical outcome objectives (e.g., {'reward': 0.9}).",
    )
    event_objectives: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Optional per-event objectives aligned to trace events.",
    )
    outcome_reward: float | None = Field(
        default=None,
        description="The reward for this rollout. PREFERRED field for scoring.",
    )
    event_rewards: list[float] | None = Field(
        default=None,
        description="Optional per-step/event rewards for multi-step tasks.",
    )

    # =========================================================================
    # LEGACY FIELDS (Backward Compatibility)
    # =========================================================================
    episode_rewards: list[float] = Field(
        default_factory=list,
        description="[LEGACY] Per-episode rewards. Use outcome_reward instead.",
    )
    reward_mean: float = Field(
        default=0.0,
        description="[LEGACY] Mean reward. Use outcome_reward instead.",
    )
    num_steps: int = Field(
        default=1,
        description="[LEGACY] Step count. Can be derived from event_rewards or trace.",
    )
    num_episodes: int = Field(
        default=1,
        description="[LEGACY] Episode count. Usually 1 for GEPA tasks.",
    )
    outcome_score: float | None = Field(
        default=None,
        description="[DEPRECATED] Alias for outcome_reward. Use outcome_reward instead.",
    )
    events_score: float | None = Field(
        default=None,
        description="[LEGACY] Aggregate event score. Use event_rewards instead.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata only. Do NOT use details.correct for rewards.",
    )


class RolloutResponse(BaseModel):
    """Response from a rollout execution.

    ## Key Fields

    - `run_id`: Echo from request (required)
    - `metrics`: Rollout metrics with `outcome_reward` (required)
    - `trace`: v3 trace payload (required for verifier scoring)

    ## Canonical Locations (Top-Level)

    - `trace_correlation_id`: Correlation ID for trace recovery (TOP-LEVEL CANONICAL)
    - `inference_url`: Inference URL used for this rollout (TOP-LEVEL CANONICAL)

    These fields SHOULD be at top-level. The monorepo parses from top-level first,
    with fallback to nested locations for backward compatibility.

    ## Example

        response = RolloutResponse(
            run_id=request.run_id,
            metrics=RolloutMetrics(outcome_reward=1.0),
            trace=trace_payload,
            trace_correlation_id="trace_abc123",
            inference_url="https://api.usesynth.ai/v1/trial-xyz",
        )
    """

    run_id: str
    metrics: RolloutMetrics
    trace: dict[str, Any] | None = None

    # =========================================================================
    # CANONICAL LOCATIONS (Top-Level - Preferred for Parsing)
    # =========================================================================
    trace_correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for trace recovery. TOP-LEVEL CANONICAL location.",
    )
    inference_url: str | None = Field(
        default=None,
        description="Inference URL used for this rollout. TOP-LEVEL CANONICAL location.",
    )

    # =========================================================================
    # LEGACY FIELDS (Backward Compatibility)
    # =========================================================================
    branches: dict[str, list[str]] = Field(
        default_factory=dict,
        description="[LEGACY] Branch tracking. Usually empty for single-path rollouts.",
    )
    aborted: bool = Field(
        default=False,
        description="Whether the rollout was aborted early.",
    )
    pipeline_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="[LEGACY] Additional metadata. Prefer top-level fields instead.",
    )

    @field_validator('trace_correlation_id')
    @classmethod
    def warn_missing_correlation_id(cls, v: str | None) -> str | None:
        """Warn if trace_correlation_id is None - this breaks trace hydration.

        When trace_correlation_id is None, the backend cannot correlate LLM traces
        with eval seeds, causing trace hydration to fail. This results in warnings like:
        "missing correlation_id" and prevents proper trace tracking.

        To fix: Call extract_trace_correlation_id() from
        synth_ai.sdk.task.trace_correlation_helpers and include the result
        in your RolloutResponse, or use build_rollout_response() helper.
        """
        if v is None:
            warnings.warn(
                "RolloutResponse.trace_correlation_id is None. "
                "Trace hydration will fail. "
                "See: https://docs.usesynth.ai/guides/local-api#trace-correlation "
                "or use build_rollout_response() helper from synth_ai.sdk.task",
                UserWarning,
                stacklevel=4
            )
        return v


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
    """Outcome and event scoring definitions used by verifiers."""

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
    dataset: DatasetInfo
    inference: InferenceInfo
    limits: LimitsInfo
    environment: str | None = Field(
        default=None,
        description="[DEPRECATED] Legacy field not read by server. Will be removed in future version.",
    )
    rubric: RubricInfo | None = Field(
        default=None,
        description="[DEPRECATED] Use LocalAPIConfig.rubrics (RubricBundle) instead. Server ignores this field.",
    )
    task_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific extras (e.g. prompt version info, documentation links).",
    )
