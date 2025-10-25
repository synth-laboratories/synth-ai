from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RolloutMode(str, Enum):
    """Mode controls how rollout infrastructure processes inference URLs."""
    RL = "rl"
    EVAL = "eval"


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


class RolloutRecordConfig(BaseModel):
    trajectories: bool = True
    logprobs: bool = False
    value: bool = False
    return_trace: bool = False
    trace_format: Literal["compact", "full", "structured"] = "compact"


class RolloutSafetyConfig(BaseModel):
    max_ops: int = 100000
    max_time_s: float = 3600.0


class RolloutRequest(BaseModel):
    run_id: str
    env: RolloutEnvSpec
    policy: RolloutPolicySpec
    ops: list[dict[str, Any]] | list[str]
    record: RolloutRecordConfig = RolloutRecordConfig()
    on_done: str = "reset"
    safety: RolloutSafetyConfig = RolloutSafetyConfig()
    training_session_id: str | None = None
    synth_base_url: str | None = None
    mode: RolloutMode  # Required: explicit RL vs EVAL mode


class RolloutStep(BaseModel):
    """Single step in a rollout trajectory.
    
    DEPRECATED: This is part of the legacy trajectory format. New code should
    consume v3 traces (RolloutResponse.trace) instead. See monorepo/trace_single_source.txt
    for migration plan.
    """
    obs: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    reward: float | None = None
    done: bool = False
    truncated: bool | None = None
    info: dict[str, Any] | None = None


class RolloutTrajectory(BaseModel):
    """Legacy trajectory format for rollout results.
    
    DEPRECATED: This format duplicates data already present in v3 traces and will
    be removed once training code migrates to consuming RolloutResponse.trace.
    
    Current state:
    - Task apps emit BOTH this format AND v3 traces (dual serialization)
    - Training code (GSPO) reads from this format
    - Eval/filter tools read from v3 traces
    
    Migration plan:
    - Phase 1: Training code learns to read from v3 traces (with fallback to this)
    - Phase 2: Make this field optional once training is migrated
    - Phase 3: Remove this field entirely and delete this class
    
    See: monorepo/trace_single_source.txt for full migration plan and timeline.
    
    Why v3 traces are better:
    - Single source of truth (no duplication/drift)
    - Richer data: token IDs, logprobs, reasoning, timing, images
    - Built-in audit trail and replay capability
    - Standard schema across all Synth AI tooling
    """
    env_id: str
    policy_id: str
    steps: list[RolloutStep]
    final: dict[str, Any] | None = None
    length: int
    
    # Required for trace correlation with inference mesh (optional initially for backward compat)
    # See: monorepo/INFERENCE_URL_REQUIREMENT_PLAN.md and trace_creation_and_judgement.txt
    inference_url: str
    
    decision_samples: list[dict[str, Any]] | None = None


class RolloutMetrics(BaseModel):
    episode_returns: list[float]
    mean_return: float
    num_steps: int
    num_episodes: int = 0
    outcome_score: float | None = None
    events_score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RolloutResponse(BaseModel):
    """Response from a rollout execution.
    
    Contains both legacy trajectory format (for backward compatibility) and
    modern v3 trace format (preferred going forward).
    """
    run_id: str
    
    # DEPRECATED: Legacy format maintained for training code compatibility.
    # Will be removed once training migrates to reading from `trace` field.
    # See: monorepo/trace_single_source.txt for migration plan.
    trajectories: list[RolloutTrajectory]
    
    branches: dict[str, list[str]] = Field(default_factory=dict)
    metrics: RolloutMetrics
    aborted: bool = False
    ops_executed: int = 0
    
    # OPTIONAL: correlation ID for linking rollout to inference traces
    # If not provided, trainer will infer it from trajectory.inference_url ?cid=... parameter
    trace_correlation_id: str | None = None
    
    # PREFERRED: v3 trace format (SessionTrace). This is the single source of truth
    # for rollout data and should be used by all new code. Contains richer data than
    # trajectories including token IDs, logprobs, timing, and multimodal content.
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
