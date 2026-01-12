"""Contracts for Task Apps.

Prefer synth_ai.sdk.localapi.contracts moving forward. This module remains for
backward compatibility during the naming transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
        ..., description="JSON Schema for the expected response structure"
    )
    schema_name: str = Field(
        default="response", description="Name for the schema (required by some providers)"
    )
    strict: bool = Field(
        default=True, description="Whether to enforce strict schema validation (OpenAI strict mode)"
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
        description="How the policy expects model outputs: tool_calls, text, or structured",
    )
    structured_config: StructuredOutputConfig | None = Field(
        default=None,
        description="Configuration for structured output mode (required if output_mode=STRUCTURED)",
    )


class RolloutSafetyConfig(BaseModel):
    """Safety configuration for rollout execution.

    Note: max_time_s is set but enforcement depends on the task app implementation.
    """

    max_time_s: float = 3600.0


class RolloutRequest(BaseModel):
    """Request to execute a rollout.

    ## Identifier Fields

    - `trace_correlation_id`: REQUIRED - Single source of truth for rollout identification.
      Used for trace correlation, registry operations, seed derivation, and resource tracking.

    ## Context Override Fields (for unified prompt + context optimization)

    - `context_overrides`: Optional structured overrides (AGENTS.md, skills, workspace files)
    - `override_bundle_id`: Optional stable ID for the override bundle
    """

    trace_correlation_id: str = Field(
        ...,
        description="REQUIRED - Unique identifier for this rollout. Used for trace correlation, "
        "registry operations, seed derivation, and resource tracking. Single source of truth.",
    )
    env: RolloutEnvSpec
    policy: RolloutPolicySpec
    on_done: str = "reset"
    safety: RolloutSafetyConfig = RolloutSafetyConfig()
    training_session_id: str | None = None
    synth_base_url: str | None = None

    # Context override fields (for unified optimization)
    context_overrides: list[Any] | None = (
        Field(  # Will be list[ContextOverride] to avoid circular import
            default=None,
            description="Optional context overrides to apply before agent execution (AGENTS.md, skills, etc.)",
        )
    )
    override_bundle_id: str | None = Field(
        default=None,
        description="Optional stable ID for the override bundle (for tracking/debugging)",
    )


class RolloutMetrics(BaseModel):
    """Metrics from a rollout execution.

    ## Required Fields

    - `outcome_reward`: REQUIRED - The reward for this rollout

    ## Optional Fields

    - `event_rewards`: Per-step rewards for multi-step tasks
    - `outcome_objectives`: Multi-objective outcomes (e.g., {'reward': 0.9, 'latency': 0.5})
    - `event_objectives`: Per-event objectives aligned to trace events
    - `details`: Metadata only (not for scoring)

    ## Example - Minimal

        metrics = RolloutMetrics(outcome_reward=1.0)

    ## Example - Multi-objective

        metrics = RolloutMetrics(
            outcome_reward=0.85,
            outcome_objectives={"reward": 0.85, "latency": 0.7},
            event_rewards=[0.8, 0.9, 0.85],
        )
    """

    # =========================================================================
    # REQUIRED FIELD
    # =========================================================================
    outcome_reward: float = Field(
        ...,
        description="REQUIRED - The reward for this rollout. Single source of truth for scoring.",
    )

    # =========================================================================
    # OPTIONAL FIELDS
    # =========================================================================
    event_rewards: list[float] | None = Field(
        default=None,
        description="Optional per-step/event rewards for multi-step tasks.",
    )
    outcome_objectives: Optional[Dict[str, float]] = Field(
        default=None,
        description="Multi-objective outcomes (e.g., {'reward': 0.9, 'latency': 0.5}).",
    )
    event_objectives: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Optional per-event objectives aligned to trace events.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata only. Do NOT use details for reward computation.",
    )


class RolloutResponse(BaseModel):
    """Response from a rollout execution.

    ## Key Fields

    - `trace_correlation_id`: REQUIRED - Echo from request (single source of truth)
    - `metrics`: Rollout metrics with `outcome_reward` (required)
    - `trace`: v3 trace payload (required for verifier scoring)
    - `inference_url`: Inference URL used for this rollout
    - `artifact`: Optional list of artifacts produced by the workflow
    - `success_status`: Infrastructure/runtime success status (orthogonal to reward)
    - `status_detail`: Optional debug string for failure details
    - `override_application`: Optional result of applying context overrides

    ## Example - Basic

        response = RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            metrics=RolloutMetrics(outcome_reward=1.0),
            trace=trace_payload,
            inference_url="https://api.usesynth.ai/v1/trial-xyz",
        )

    ## Example - With Artifacts and Status

        from synth_ai.data import Artifact, SuccessStatus

        response = RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            metrics=RolloutMetrics(outcome_reward=0.95),
            trace=trace_payload,
            artifact=[Artifact(
                content=rust_code,
                content_type="rust_code",
                metadata={"file_path": "pokemon.rs"},
            )],
            success_status=SuccessStatus.SUCCESS,
        )
    """

    trace_correlation_id: str = Field(
        ...,
        description="REQUIRED - Correlation ID for trace recovery. Single source of truth. "
        "Echo from request.trace_correlation_id.",
    )
    metrics: RolloutMetrics
    trace: dict[str, Any] | None = None
    inference_url: str | None = Field(
        default=None,
        description="Inference URL used for this rollout.",
    )

    # Artifact and status fields (Phase 1 additions)
    artifact: list[Any] | None = Field(  # Will be list[Artifact] to avoid circular import
        default=None,
        description="Optional artifacts produced by the workflow (code, JSON, files). "
        "Stored separately and linked via trace_correlation_id.",
    )
    success_status: str | None = Field(  # Will be SuccessStatus to avoid circular import
        default=None,
        description="Infrastructure/runtime success status (orthogonal to reward). "
        "Never computed from reward - only from infra/runtime outcome.",
    )
    status_detail: str | None = Field(
        default=None,
        description="Optional freeform debug string for failure details (e.g., stderr excerpt).",
    )
    override_application: Any | None = (
        Field(  # Will be OverrideApplication to avoid circular import
            default=None,
            description="Optional result of applying context overrides (applied? errors? warnings?).",
        )
    )


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
