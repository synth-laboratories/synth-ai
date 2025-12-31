"""RL (Reinforcement Learning) configuration models.

This module defines the configuration schema for RL training jobs using GSPO
(Group Sequence Policy Optimization) or other policy gradient methods.

Example TOML configuration:
    ```toml
    [algorithm]
    type = "online"
    method = "policy_gradient"
    variety = "gspo"

    [services]
    task_url = "https://your-tunnel.trycloudflare.com"

    [model]
    base = "Qwen/Qwen3-4B"
    trainer_mode = "lora"
    label = "my-rl-model"

    [rollout]
    env_name = "my-task"
    policy_name = "my-policy"
    max_turns = 10
    episodes_per_batch = 32
    max_concurrent_rollouts = 8

    [training]
    num_epochs = 1
    iterations_per_epoch = 20
    batch_size = 16
    group_size = 4
    learning_rate = 5e-5
    ```

See Also:
    - Training reference: /training/gspo
    - Job events: /sdk/jobs/rl
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import model_validator

from ..utils import load_toml
from .shared import AlgorithmConfig, ComputeConfig, ExtraModel, LoraConfig, PolicyConfig


class RLServicesConfig(ExtraModel):
    """Service URLs for RL training.

    Attributes:
        task_url: URL of your task app (typically a Cloudflare tunnel URL).
            Required for rollout execution.
        verifier_url: Optional URL for verifier service. Defaults to Synth's
            hosted verifier at https://synth-backend.onrender.com/api.
    """
    task_url: str
    verifier_url: str | None = None


class ModelConfig(ExtraModel):
    """Model configuration for RL training.

    Specify either `base` (for a new model) or `source` (to continue from
    a checkpoint), but not both.

    Attributes:
        source: Checkpoint ID to continue training from (e.g., "ft:job_abc123").
            Mutually exclusive with `base`.
        base: Base model to fine-tune (e.g., "Qwen/Qwen3-4B").
            Mutually exclusive with `source`.
        trainer_mode: Training mode - "lora", "qlora", or "full".
        label: Human-readable identifier for this model.
    """
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
    """Rollout configuration for episode collection.

    Controls how episodes are collected from the task app during training.

    Attributes:
        env_name: Environment/task name registered in your task app.
        policy_name: Policy identifier for the rollout.
        env_config: Optional environment-specific configuration dict.
        policy_config: Optional policy-specific configuration dict.
        max_turns: Maximum steps per episode before truncation.
        episodes_per_batch: Number of episodes to collect per training batch.
        max_concurrent_rollouts: Maximum parallel rollouts to the task app.
        batches_per_step: Batches to collect per training step. Default: 1.
    """
    env_name: str
    policy_name: str
    env_config: dict[str, Any] | None = None
    policy_config: dict[str, Any] | None = None
    max_turns: int
    episodes_per_batch: int
    max_concurrent_rollouts: int
    batches_per_step: int | None = None


class WeightSyncConfig(ExtraModel):
    """Weight synchronization configuration.

    Controls how model weights are synchronized between training and inference.

    Attributes:
        enable: Whether to enable weight sync. Default: True.
        targets: Sync targets, typically ["policy"].
        mode: Sync mode (advanced).
        direct: Use direct sync method.
        verify_every_k: Verify sync every K iterations.
    """
    enable: bool | None = None
    targets: list[str] | None = None
    mode: str | None = None
    direct: bool | None = None
    verify_every_k: int | None = None


class RewardsConfig(ExtraModel):
    """Rewards configuration for RL training.

    Controls step-level and event-level reward computation.

    Attributes:
        step_rewards_enabled: Enable step-level rewards. Default: False.
        step_rewards_mode: Reward mode - "off", "decision_stepwise", or "env_sparse".
        step_rewards_indicator_lambda: Lambda coefficient for indicator rewards.
        step_rewards_beta: Beta coefficient for step rewards.
        step_rewards_strategy: Reward computation strategy.
        event_rewards_kind: Event reward aggregation - "unique" or "absolute".
    """
    step_rewards_enabled: bool | None = None
    step_rewards_mode: str | None = None
    step_rewards_indicator_lambda: float | None = None
    step_rewards_beta: float | None = None
    step_rewards_strategy: str | None = None
    event_rewards_kind: str | None = None


class RLTrainingConfig(ExtraModel):
    """Training hyperparameters for RL.

    Attributes:
        num_epochs: Number of training epochs.
        iterations_per_epoch: Training iterations per epoch.
        gradient_accumulation_steps: Steps to accumulate gradients. Default: 1.
        max_accumulated_minibatch: Maximum accumulated minibatch size.
        max_turns: Maximum turns during training rollouts.
        batch_size: Training batch size.
        group_size: GSPO group size for advantage estimation.
        learning_rate: Optimizer learning rate (e.g., 5e-5).
        log_interval: Log metrics every N steps.
        weight_sync_interval: Sync weights every N steps.
        weight_sync: Nested weight sync configuration.
        lora: LoRA configuration (r, alpha, dropout, target_modules).
        rewards: Nested rewards configuration.
    """
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
    """Evaluation configuration during training.

    Attributes:
        instances: Number of evaluation instances to run.
        every_n_iters: Run evaluation every N training iterations.
        seeds: List of seeds for reproducible evaluation.
    """
    instances: int
    every_n_iters: int
    seeds: list[int]


class VerifierOptionsConfig(ExtraModel):
    """Verifier scoring options.

    Attributes:
        event: Enable event-level verification.
        outcome: Enable outcome-level verification.
        provider: Verifier provider - "synth" for Synth's hosted verifier.
        model: Verifier model identifier.
        rubric_id: Optional rubric identifier.
        rubric_overrides: Override specific rubric parameters.
        tracks: Tracks to evaluate.
        weights: Per-track scoring weights.
        max_concurrency: Maximum concurrent verifier API calls.
    """
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
    """Rubric configuration for reward blending.

    Attributes:
        enabled: Enable rubric-based scoring. Default: False.
        reward_blend: Weights for reward sources - {"env": 1.0, "event": 0.0, "outcome": 0.0}.
    """
    enabled: bool = False
    reward_blend: dict[str, float] | None = None  # env, event, outcome weights


class VerifierConfig(ExtraModel):
    """Verifier configuration for LLM-based reward scoring.

    Attributes:
        type: Verifier type - "synth" for Synth's hosted verifier.
        timeout_s: Timeout in seconds for verifier API calls.
        enabled: Master switch to enable/disable verifier scoring.
        reward_blend: Reward source weights - {"env": 1.0, "event": 0.0, "outcome": 0.0}.
        rubric: Deprecated - use reward_blend instead.
        options: Detailed verifier options.
    """
    type: str | None = None
    timeout_s: int | None = None
    enabled: bool | None = None  # Master switch for verifier/rubric
    reward_blend: dict[str, float] | None = None  # NEW: nested reward blending (replaces rubric.weights)
    rubric: RubricConfig | None = None  # DEPRECATED: use flat fields instead
    options: VerifierOptionsConfig | None = None


class SmokeConfig(ExtraModel):
    """Configuration for local smoke testing (CLI only, ignored by trainer).

    Use this section to configure quick local tests before submitting
    a full training job.

    Attributes:
        task_url: Override task app URL for testing.
        env_name: Environment name to test.
        policy_name: Policy name to test.
        max_steps: Maximum steps for smoke test.
        policy: Policy type - "mock", "gpt-5-nano", "openai", "groq".
        model: Model identifier for the policy.
        mock_backend: Mock backend type - "synthetic" or "openai".
        mock_port: Port for mock backend.
        return_trace: Include trace in response.
        use_mock: Use mock policy.
        task_app_name: Task app to auto-serve (e.g., "grpo-crafter").
        task_app_port: Port for auto-served task app. Default: 8765.
        task_app_env_file: Path to .env file for task app.
        task_app_force: Use --force flag when serving.
        sqld_auto_start: Auto-start sqld server.
        sqld_db_path: Database path. Default: ./traces/local.db.
        sqld_hrana_port: Hrana WebSocket port. Default: 8080.
        sqld_http_port: HTTP API port. Default: 8081.
    """
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
    """Root configuration for RL (Reinforcement Learning) training jobs.

    This is the top-level config loaded from a TOML file. Use `RLConfig.from_path()`
    to load from a file, or `RLConfig.from_mapping()` to load from a dict.

    Example:
        ```python
        from synth_ai.sdk.api.train.configs.rl import RLConfig

        # Load from file
        config = RLConfig.from_path("rl_config.toml")

        # Or from dict
        config = RLConfig.from_mapping({
            "algorithm": {"type": "online", "method": "policy_gradient", "variety": "gspo"},
            "services": {"task_url": "https://my-tunnel.trycloudflare.com"},
            "model": {"base": "Qwen/Qwen3-4B", "trainer_mode": "lora", "label": "my-model"},
            ...
        })
        ```

    Attributes:
        algorithm: Algorithm configuration (type, method, variety).
        services: Service URLs (task_url, verifier_url).
        compute: GPU and compute configuration.
        topology: Deprecated - use compute.topology.
        vllm: vLLM inference server configuration.
        reference: Deprecated - use compute.topology.reference_placement.
        model: Deprecated - use policy instead.
        policy: Policy/model configuration (preferred).
        lora: Deprecated - use training.lora.
        rollout: Rollout/episode collection configuration.
        evaluation: Evaluation configuration.
        training: Training hyperparameters.
        rubric: Deprecated - use verifier.reward_blend.
        verifier: Verifier/reward configuration.
        tags: Optional metadata tags.
        smoke: CLI-only smoke testing configuration.

    Returns:
        After training completes, you receive a result dict:
        ```python
        {
            "status": "succeeded",
            "final_reward": 0.85,
            "model_id": "ft:Qwen/Qwen3-0.6B:job_abc123",
            "checkpoints": [
                {"step": 100, "path": "..."},
                {"step": 200, "path": "..."},
            ],
        }
        ```

    Events:
        During training, you'll receive streaming events:
        - `rl.created` - Job created
        - `rl.running` - Training started
        - `rl.iteration.complete` - Iteration finished with metrics
        - `rl.evaluation.complete` - Evaluation finished with scores
        - `rl.succeeded` / `rl.failed` - Terminal states
    """
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
    rubric: dict[str, Any] | None = None  # DEPRECATED: use verifier.reward_blend and verifier.enabled instead
    verifier: VerifierConfig | None = None
    tags: dict[str, Any] | None = None
    smoke: SmokeConfig | None = None  # CLI-only: local smoke testing config (ignored by trainer)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary."""
        return self.model_dump(mode="python", exclude_none=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RLConfig:
        """Load RL config from dict/TOML mapping.

        Args:
            data: Dictionary or TOML mapping with configuration.

        Returns:
            Validated RLConfig instance.
        """
        return cls.model_validate(data)

    @classmethod
    def from_path(cls, path: Path) -> RLConfig:
        """Load RL config from a TOML file.

        Args:
            path: Path to the TOML configuration file.

        Returns:
            Validated RLConfig instance.
        """
        content = load_toml(path)
        return cls.from_mapping(content)


__all__ = [
    "EvaluationConfig",
    "VerifierConfig",
    "VerifierOptionsConfig",
    "ModelConfig",
    "RLConfig",
    "RLServicesConfig",
    "RLTrainingConfig",
    "RolloutConfig",
    "SmokeConfig",
    "WeightSyncConfig",
]
