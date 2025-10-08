from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Literal
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
    env_name: Optional[str] = None
    requires_api_key_header: bool = True


# --- Unified rollout schema used by Task App services and SDK utilities ---


class RolloutEnvSpec(BaseModel):
    env_id: Optional[str] = None
    env_name: Optional[str] = None
    config: Dict[str, Any] = {}
    seed: Optional[int] = None


class RolloutPolicySpec(BaseModel):
    policy_id: Optional[str] = None
    policy_name: Optional[str] = None
    config: Dict[str, Any] = {}


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
    ops: List[Dict[str, Any]] | List[str]
    record: RolloutRecordConfig = RolloutRecordConfig()
    on_done: str = "reset"
    safety: RolloutSafetyConfig = RolloutSafetyConfig()
    training_session_id: Optional[str] = None
    synth_base_url: Optional[str] = None


class RolloutStep(BaseModel):
    obs: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    reward: Optional[float] = None
    done: bool = False
    truncated: Optional[bool] = None
    info: Optional[Dict[str, Any]] = None


class RolloutTrajectory(BaseModel):
    env_id: str
    policy_id: str
    steps: List[RolloutStep]
    final: Optional[Dict[str, Any]] = None
    length: int


class RolloutMetrics(BaseModel):
    episode_returns: List[float]
    mean_return: float
    num_steps: int
    num_episodes: int = 0
    outcome_score: Optional[float] = None
    events_score: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class RolloutResponse(BaseModel):
    run_id: str
    trajectories: List[RolloutTrajectory]
    branches: Dict[str, List[str]] = {}
    metrics: RolloutMetrics
    aborted: bool = False
    ops_executed: int = 0
    trace: Dict[str, Any] | None = None


class TaskInfo(BaseModel):
    """Static metadata describing the capabilities of a Task App task."""

    task: Dict[str, Any]
    environments: List[str]
    action_space: Dict[str, Any]
    observation: Dict[str, Any]
    dataset: Dict[str, Any]
    rubric: Dict[str, Any]
    inference: Dict[str, Any]
    capabilities: Dict[str, Any]
    limits: Dict[str, Any]
