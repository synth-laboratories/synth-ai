from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from pydantic import BaseModel


@dataclass(frozen=True)
class TaskAppEndpoints:
    """Canonical Task App endpoint shapes used by RL trainers.

    The Task App is an HTTP service (often deployed on Modal) that exposes:
    - Health: GET /health
      • Requires header X-API-Key (when ENVIRONMENT_API_KEY is configured)
      • Returns { healthy: true }
    - Environment lifecycle:
      • POST /env/{env_name}/initialize → { env_id, observation }
      • POST /env/{env_name}/step      → { observation, reward, done, info }
      • POST /env/{env_name}/terminate → { ok: true }
    - Rollout (optional, unified schema):
      • POST /rollout → { run_id, trajectories[], metrics, ... }
    - Proxy (optional):
      • POST /proxy/v1/chat/completions (for direct OpenAI calls from Task App)
    """

    health: str = "/health"
    rollout: str = "/rollout"
    proxy_chat_completions: str = "/proxy/v1/chat/completions"
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


class RolloutResponse(BaseModel):
    run_id: str
    trajectories: List[RolloutTrajectory]
    branches: Dict[str, List[str]] = {}
    metrics: RolloutMetrics
    aborted: bool = False
    ops_executed: int = 0

