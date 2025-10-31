from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import model_validator

from ..utils import load_toml
from .shared import AlgorithmConfig, ComputeConfig, ExtraModel, LoraConfig, PolicyConfig


class RLServicesConfig(ExtraModel):
    task_url: str
    judge_url: str | None = None


class ModelConfig(ExtraModel):
    source: str | None = None
    base: str | None = None
    trainer_mode: str
    label: str

    @model_validator(mode="after")
    def _ensure_exactly_one_source_or_base(self) -> ModelConfig:
        if bool(self.source) == bool(self.base):
            raise ValueError("Config must set exactly one of [model].source or [model].base")
        return self


class RolloutConfig(ExtraModel):
    env_name: str
    policy_name: str
    env_config: dict[str, Any] | None = None
    policy_config: dict[str, Any] | None = None
    max_turns: int
    episodes_per_batch: int
    max_concurrent_rollouts: int
    batches_per_step: int | None = None
    ops: list[str] | None = None


class WeightSyncConfig(ExtraModel):
    enable: bool | None = None
    targets: list[str] | None = None
    mode: str | None = None
    direct: bool | None = None
    verify_every_k: int | None = None


class RewardsConfig(ExtraModel):
    """Rewards configuration for RL training."""
    step_rewards_enabled: bool | None = None
    step_rewards_mode: str | None = None
    step_rewards_indicator_lambda: float | None = None
    step_rewards_beta: float | None = None
    step_rewards_strategy: str | None = None
    event_rewards_kind: str | None = None


class RLTrainingConfig(ExtraModel):
    num_epochs: int
    iterations_per_epoch: int
    gradient_accumulation_steps: int | None = None
    max_accumulated_minibatch: int | None = None
    max_turns: int
    batch_size: int
    group_size: int
    learning_rate: float
    log_interval: int | None = None
    weight_sync_interval: int | None = None
    # DEPRECATED: flat reward fields (use rewards.* instead)
    step_rewards_enabled: bool | None = None
    step_rewards_mode: str | None = None
    step_rewards_indicator_lambda: float | None = None
    step_rewards_beta: float | None = None
    step_rewards_strategy: str | None = None
    event_rewards_kind: str | None = None
    # NEW: nested configs
    weight_sync: WeightSyncConfig | None = None
    lora: LoraConfig | None = None
    rewards: RewardsConfig | None = None


class EvaluationConfig(ExtraModel):
    instances: int
    every_n_iters: int
    seeds: list[int]


class JudgeOptionsConfig(ExtraModel):
    event: bool | None = None
    outcome: bool | None = None
    provider: str | None = None
    model: str | None = None
    rubric_id: str | None = None
    rubric_overrides: dict[str, Any] | None = None
    tracks: list[str] | None = None
    weights: dict[str, float] | None = None
    max_concurrency: int | None = None


class RubricConfig(ExtraModel):
    """Rubric configuration for reward blending."""
    enabled: bool = False
    reward_blend: dict[str, float] | None = None  # env, event, outcome weights


class JudgeConfig(ExtraModel):
    type: str | None = None
    timeout_s: int | None = None
    enabled: bool | None = None  # Master switch for judge/rubric
    reward_blend: dict[str, float] | None = None  # NEW: nested reward blending (replaces rubric.weights)
    rubric: RubricConfig | None = None  # DEPRECATED: use flat fields instead
    options: JudgeOptionsConfig | None = None


class SmokeConfig(ExtraModel):
    """Configuration for local smoke testing (CLI only, ignored by trainer)."""
    # Test parameters
    task_url: str | None = None
    env_name: str | None = None
    policy_name: str | None = None
    max_steps: int | None = None
    policy: str | None = None  # mock, gpt-5-nano, openai, groq
    model: str | None = None
    mock_backend: str | None = None  # synthetic or openai
    mock_port: int | None = None
    return_trace: bool | None = None
    use_mock: bool | None = None
    
    # Task app auto-start configuration
    task_app_name: str | None = None  # Task app to serve (e.g., "grpo-crafter")
    task_app_port: int | None = None  # Port for task app (default: 8765)
    task_app_env_file: str | None = None  # Path to .env file for task app
    task_app_force: bool | None = None  # Use --force flag when serving
    
    # sqld auto-start configuration
    sqld_auto_start: bool | None = None  # Auto-start sqld server
    sqld_db_path: str | None = None  # Database path (default: ./traces/local.db)
    sqld_hrana_port: int | None = None  # Hrana WebSocket port (default: 8080)
    sqld_http_port: int | None = None  # HTTP API port (default: 8081)


class RLConfig(ExtraModel):
    algorithm: AlgorithmConfig
    services: RLServicesConfig
    compute: ComputeConfig | None = None
    topology: dict[str, Any] | None = None  # DEPRECATED: use compute.topology instead
    vllm: dict[str, Any] | None = None
    reference: dict[str, Any] | None = None  # DEPRECATED: use compute.topology.reference_placement instead
    model: ModelConfig | None = None  # DEPRECATED: use policy instead
    policy: PolicyConfig | None = None  # NEW: unified policy (preferred)
    lora: dict[str, Any] | None = None  # DEPRECATED: use training.lora instead
    rollout: RolloutConfig | None = None
    evaluation: EvaluationConfig | None = None
    training: RLTrainingConfig | None = None
    rubric: dict[str, Any] | None = None  # DEPRECATED: use judge.reward_blend and judge.enabled instead
    judge: JudgeConfig | None = None
    tags: dict[str, Any] | None = None
    smoke: SmokeConfig | None = None  # CLI-only: local smoke testing config (ignored by trainer)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RLConfig:
        """Load RL config from dict/TOML mapping."""
        return cls.model_validate(data)

    @classmethod
    def from_path(cls, path: Path) -> RLConfig:
        content = load_toml(path)
        return cls.from_mapping(content)


__all__ = [
    "EvaluationConfig",
    "JudgeConfig",
    "JudgeOptionsConfig",
    "ModelConfig",
    "RLConfig",
    "RLServicesConfig",
    "RLTrainingConfig",
    "RolloutConfig",
    "SmokeConfig",
    "WeightSyncConfig",
]
