from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field

from ..utils import load_toml
from .shared import AlgorithmConfig, ComputeConfig, ExtraModel, LoraConfig, PolicyConfig


class JobConfig(ExtraModel):
    model: str
    data: str | None = None
    data_path: str | None = None
    poll_seconds: int | None = None


class SFTDataConfig(ExtraModel):
    topology: dict[str, Any] | None = None
    validation_path: str | None = None


class TrainingValidationConfig(ExtraModel):
    enabled: bool | None = None
    evaluation_strategy: str | None = None
    eval_steps: int | None = None
    save_best_model_at_end: bool | None = None
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None


class TrainingConfig(ExtraModel):
    mode: str | None = None
    use_qlora: bool | None = None
    validation: TrainingValidationConfig | None = None
    lora: LoraConfig | None = None  # NEW: nested LoRA config


class HyperparametersParallelism(ExtraModel):
    use_deepspeed: bool | None = None
    deepspeed_stage: int | None = None
    fsdp: bool | None = None
    bf16: bool | None = None
    fp16: bool | None = None
    activation_checkpointing: bool | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None


class HyperparametersConfig(ExtraModel):
    n_epochs: int = 1
    batch_size: int | None = None
    global_batch: int | None = None
    per_device_batch: int | None = None
    gradient_accumulation_steps: int | None = None
    sequence_length: int | None = None
    learning_rate: float | None = None
    warmup_ratio: float | None = None
    train_kind: str | None = None
    weight_decay: float | None = None
    parallelism: HyperparametersParallelism | None = None


class SFTConfig(ExtraModel):
    algorithm: AlgorithmConfig | None = None
    job: JobConfig
    policy: PolicyConfig | None = None  # NEW: unified policy section
    compute: ComputeConfig | None = None
    data: SFTDataConfig | None = None
    training: TrainingConfig | None = None
    hyperparameters: HyperparametersConfig = Field(default_factory=HyperparametersConfig)
    lora: dict[str, Any] | None = None  # DEPRECATED: use training.lora instead
    tags: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SFTConfig:
        """Load SFT config from dict/TOML mapping."""
        return cls.model_validate(data)

    @classmethod
    def from_path(cls, path: Path) -> SFTConfig:
        content = load_toml(path)
        return cls.from_mapping(content)


__all__ = [
    "HyperparametersConfig",
    "HyperparametersParallelism",
    "JobConfig",
    "SFTConfig",
    "SFTDataConfig",
    "TrainingConfig",
    "TrainingValidationConfig",
]
