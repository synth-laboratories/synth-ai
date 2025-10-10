from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class TaskAppEndpoints:
    """Canonical Task App endpoint shapes used by RL trainers.

    Task Apps run as lightweight HTTP services (often on Modal) that expose a
    consistent set of endpoints for health, metadata, environment lifecycle,
    rollouts, and optional proxy access to vendor models. The endpoint strings
    defined here act as defaults and documentation for clients.
    """

    root: str = "/"
    health: str = "/health"
    info: str = "/info"
    task_info: str = "/task_info"
    rollout: str = "/rollout"
    proxy_chat_completions: str = "/proxy/v1/chat/completions"
    proxy_groq_chat_completions: str = "/proxy/groq/v1/chat/completions"
    env_initialize: str = "/env/{env_name}/initialize"
    env_step: str = "/env/{env_name}/step"
    env_terminate: str = "/env/{env_name}/terminate"


@dataclass(frozen=True)
class TaskAppContract:
    """Requirements and expectations for a Task App used by RL trainers.

    - Auth: ENVIRONMENT_API_KEY must be set in the Task App environment; requests include X-API-Key.
    - Health: /health returns 200 and JSON; may verify X-API-Key header.
    - Env API: initialize/step/terminate are present for the target env (e.g., CrafterClassic).
    - Rollout API: optional; provides a single-call rollout for convenience/testing.
    - Inference routing: policy config passes an inference_url (Synth backend or OpenAI proxy).
    - URL: base must be reachable via HTTPS and should be under .modal.run in production.
    """

    base_url: str
    env_name: str | None = None
    requires_api_key_header: bool = True


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
    trace_format: Literal["compact", "full"] = "compact"


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


class RolloutStep(BaseModel):
    obs: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    reward: float | None = None
    done: bool = False
    truncated: bool | None = None
    info: dict[str, Any] | None = None


class RolloutTrajectory(BaseModel):
    env_id: str
    policy_id: str
    steps: list[RolloutStep]
    final: dict[str, Any] | None = None
    length: int


class RolloutMetrics(BaseModel):
    episode_returns: list[float]
    mean_return: float
    num_steps: int
    num_episodes: int = 0
    outcome_score: float | None = None
    events_score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RolloutResponse(BaseModel):
    run_id: str
    trajectories: list[RolloutTrajectory]
    branches: dict[str, list[str]] = Field(default_factory=dict)
    metrics: RolloutMetrics
    aborted: bool = False
    ops_executed: int = 0
    trace: dict[str, Any] | None = None


class TaskInfo(BaseModel):
    """Static metadata describing the capabilities of a Task App task."""

    task: dict[str, Any]
    environments: list[str]
    action_space: dict[str, Any]
    observation: dict[str, Any]
    dataset: dict[str, Any]
    rubric: dict[str, Any]
    inference: dict[str, Any]
    capabilities: dict[str, Any]
    limits: dict[str, Any]
