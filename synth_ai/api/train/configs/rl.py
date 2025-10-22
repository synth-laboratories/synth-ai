from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import model_validator

from ..utils import load_toml
from .shared import AlgorithmConfig, ComputeConfig, ExtraModel


class RLServicesConfig(ExtraModel):
    task_url: str
    judge_url: str | None = None


class ModelConfig(ExtraModel):
    source: str | None = None
    base: str | None = None
    trainer_mode: str
    label: str

    @model_validator(mode="after")
    def _ensure_exactly_one_source_or_base(self) -> "ModelConfig":
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
    step_rewards_enabled: bool | None = None
    step_rewards_mode: str | None = None
    step_rewards_indicator_lambda: float | None = None
    step_rewards_beta: float | None = None
    step_rewards_strategy: str | None = None
    event_rewards_kind: str | None = None
    weight_sync: WeightSyncConfig | None = None


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


class JudgeConfig(ExtraModel):
    type: str | None = None
    timeout_s: int | None = None
    options: JudgeOptionsConfig | None = None


class RLConfig(ExtraModel):
    algorithm: AlgorithmConfig
    services: RLServicesConfig
    compute: ComputeConfig | None = None
    topology: dict[str, Any] | None = None
    vllm: dict[str, Any] | None = None
    reference: dict[str, Any] | None = None
    model: ModelConfig
    lora: dict[str, Any] | None = None
    rollout: RolloutConfig | None = None
    evaluation: EvaluationConfig | None = None
    training: RLTrainingConfig | None = None
    rubric: dict[str, Any] | None = None
    judge: JudgeConfig | None = None
    tags: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RLConfig":
        return cls.model_validate(dict(data))

    @classmethod
    def from_path(cls, path: Path) -> "RLConfig":
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
    "WeightSyncConfig",
]
