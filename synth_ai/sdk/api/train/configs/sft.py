"""SFT (Supervised Fine-Tuning) configuration models.

This module defines the configuration schema for SFT training jobs.

Example TOML configuration:
    ```toml
    [algorithm]
    type = "offline"
    method = "sft"

    [job]
    model = "Qwen/Qwen3-4B"
    data_path = "training_data.jsonl"

    [compute]
    gpu_type = "H100"
    gpu_count = 1

    [training]
    mode = "lora"

    [training.lora]
    r = 16
    alpha = 32
    dropout = 0.1

    [hyperparameters]
    n_epochs = 3
    batch_size = 4
    learning_rate = 2e-5
    sequence_length = 2048
    ```

See Also:
    - Training reference: /training/sft
    - Quickstart: /quickstart/supervised-fine-tuning
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field

from ..utils import load_toml
from .shared import AlgorithmConfig, ComputeConfig, ExtraModel, LoraConfig, PolicyConfig


class JobConfig(ExtraModel):
    """Core job configuration for SFT.

    Attributes:
        model: Base model to fine-tune (e.g., "Qwen/Qwen3-4B", "meta-llama/Llama-3-8B").
        data: Dataset identifier (if using registered datasets).
        data_path: Path to JSONL training data file.
        poll_seconds: Polling interval for job status. Default: 10.
    """
    model: str
    data: str | None = None
    data_path: str | None = None
    poll_seconds: int | None = None


class SFTDataConfig(ExtraModel):
    """Data configuration for SFT training.

    Attributes:
        topology: Data loading topology configuration.
        validation_path: Path to validation JSONL file for eval during training.
    """
    topology: dict[str, Any] | None = None
    validation_path: str | None = None


class TrainingValidationConfig(ExtraModel):
    """Validation configuration during training.

    Attributes:
        enabled: Enable validation during training. Default: False.
        evaluation_strategy: When to evaluate - "steps" or "epoch".
        eval_steps: Evaluate every N steps (if strategy is "steps").
        save_best_model_at_end: Save only the best model checkpoint.
        metric_for_best_model: Metric to use for best model selection (e.g., "eval_loss").
        greater_is_better: Whether higher metric is better. Default: False for loss.
    """
    enabled: bool | None = None
    evaluation_strategy: str | None = None
    eval_steps: int | None = None
    save_best_model_at_end: bool | None = None
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None


class TrainingConfig(ExtraModel):
    """Training mode configuration.

    Attributes:
        mode: Training mode - "lora", "qlora", or "full".
        use_qlora: Enable QLoRA (4-bit quantized LoRA). Default: False.
        validation: Validation configuration.
        lora: LoRA hyperparameters (r, alpha, dropout, target_modules).
    """
    mode: str | None = None
    use_qlora: bool | None = None
    validation: TrainingValidationConfig | None = None
    lora: LoraConfig | None = None  # NEW: nested LoRA config


class HyperparametersParallelism(ExtraModel):
    """Parallelism configuration for distributed training.

    Attributes:
        use_deepspeed: Enable DeepSpeed. Default: False.
        deepspeed_stage: DeepSpeed ZeRO stage (1, 2, or 3).
        fsdp: Enable PyTorch FSDP. Default: False.
        bf16: Use bfloat16 precision. Default: True on supported hardware.
        fp16: Use float16 precision. Default: False.
        activation_checkpointing: Enable gradient checkpointing. Default: False.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_parallel_size: Pipeline parallelism degree.
    """
    use_deepspeed: bool | None = None
    deepspeed_stage: int | None = None
    fsdp: bool | None = None
    bf16: bool | None = None
    fp16: bool | None = None
    activation_checkpointing: bool | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None


class HyperparametersConfig(ExtraModel):
    """Training hyperparameters for SFT.

    Attributes:
        n_epochs: Number of training epochs. Default: 1.
        batch_size: Training batch size (alias for global_batch).
        global_batch: Global batch size across all GPUs.
        per_device_batch: Per-device batch size.
        gradient_accumulation_steps: Steps to accumulate gradients. Default: 1.
        sequence_length: Maximum sequence length. Default: 2048.
        learning_rate: Optimizer learning rate (e.g., 2e-5).
        warmup_ratio: Fraction of steps for LR warmup. Default: 0.1.
        train_kind: Training variant (advanced).
        weight_decay: Weight decay coefficient. Default: 0.01.
        parallelism: Distributed training configuration.
    """
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
    """Root configuration for SFT (Supervised Fine-Tuning) jobs.

    This is the top-level config loaded from a TOML file.

    Example:
        ```python
        from synth_ai.sdk.api.train.configs.sft import SFTConfig

        # Load from file
        config = SFTConfig.from_path("sft_config.toml")

        # Or from dict
        config = SFTConfig.from_mapping({
            "job": {"model": "Qwen/Qwen3-4B", "data_path": "data.jsonl"},
            "hyperparameters": {"n_epochs": 3, "learning_rate": 2e-5},
        })
        ```

    Attributes:
        algorithm: Algorithm configuration (type="offline", method="sft").
        job: Core job configuration (model, data_path).
        policy: Policy configuration (preferred over job.model).
        compute: GPU and compute configuration.
        data: Data loading configuration.
        training: Training mode (lora, full) and LoRA config.
        hyperparameters: Training hyperparameters.
        lora: Deprecated - use training.lora instead.
        tags: Optional metadata tags.

    Returns:
        After training completes, you receive a result dict:
        ```python
        {
            "status": "succeeded",
            "model_id": "ft:Qwen/Qwen3-4B:sft_abc123",
            "final_loss": 0.42,
            "checkpoints": [
                {"epoch": 1, "loss": 0.65, "path": "..."},
                {"epoch": 2, "loss": 0.52, "path": "..."},
                {"epoch": 3, "loss": 0.42, "path": "..."},
            ],
        }
        ```

    Events:
        During training, you'll receive streaming events:
        - `sft.created` - Job created
        - `sft.running` - Training started
        - `sft.epoch.complete` - Epoch finished with loss
        - `sft.checkpoint.saved` - Checkpoint saved
        - `sft.succeeded` / `sft.failed` - Terminal states
    """
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
        """Convert config to a dictionary."""
        return self.model_dump(mode="python", exclude_none=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SFTConfig:
        """Load SFT config from dict/TOML mapping.

        Args:
            data: Dictionary or TOML mapping with configuration.

        Returns:
            Validated SFTConfig instance.
        """
        return cls.model_validate(data)

    @classmethod
    def from_path(cls, path: Path) -> SFTConfig:
        """Load SFT config from a TOML file.

        Args:
            path: Path to the TOML configuration file.

        Returns:
            Validated SFTConfig instance.
        """
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
