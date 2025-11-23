"""Prompt Learning configuration models for MIPRO and GEPA."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from ..utils import load_toml
from .shared import ExtraModel


class InferenceMode(str, Enum):
    synth_hosted = "synth_hosted"


class ProviderName(str, Enum):
    openai = "openai"
    groq = "groq"
    google = "google"


class PromptLearningPolicyConfig(ExtraModel):
    """Policy configuration for prompt learning (model, provider, etc.)."""
    model: str
    provider: ProviderName
    inference_url: str | None = None  # Optional - trainer provides it in rollout requests (ignored if present)
    inference_mode: InferenceMode = InferenceMode.synth_hosted
    temperature: float = 0.0
    max_completion_tokens: int = 512
    policy_name: str | None = None
    
    @field_validator("inference_url", mode="before")
    @classmethod
    def _strip_inference_url(cls, v: str | None) -> str | None:
        """Strip whitespace from inference_url if provided."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            # Validate that URL starts with http:// or https:// if provided (non-empty)
            if v and not v.startswith(("http://", "https://")):
                raise ValueError("inference_url must start with http:// or https://")
            # Reject empty strings after stripping
            if not v:
                raise ValueError("inference_url must start with http:// or https://")
        return v


class MessagePatternConfig(ExtraModel):
    """Configuration for a single message pattern."""
    role: str
    pattern: str
    order: int = 0


class PromptPatternConfig(ExtraModel):
    """Initial prompt pattern configuration."""
    id: str | None = None
    name: str | None = None
    messages: list[MessagePatternConfig] = []
    wildcards: dict[str, str] = Field(default_factory=dict)


class MIPROMetaConfig(ExtraModel):
    """Configuration for the meta-model that proposes prompt updates."""
    model: str
    provider: str
    inference_url: str | None = None
    temperature: float = 0.8
    max_tokens: int = 1024


class MIPROStageConfig(ExtraModel):
    """Configuration for a single MIPRO stage inside a module.
    
    Each stage MUST have its own policy configuration. The policy field is required
    and must include 'model' and 'provider' fields.
    """
    stage_id: str
    baseline_instruction: str
    baseline_messages: list[dict[str, str]] = Field(default_factory=list)
    max_instruction_slots: int | None = None
    max_demo_slots: int | None = None
    policy: PromptLearningPolicyConfig | dict[str, Any] = Field(
        ...,
        description="Required per-stage policy configuration. Must include 'model' and 'provider' fields."
    )


class MIPROModuleConfig(ExtraModel):
    """Configuration for a single module in a MIPRO pipeline."""
    module_id: str
    stages: list[MIPROStageConfig] = Field(default_factory=list)


class MIPROSeedConfig(ExtraModel):
    """Seed pools used across bootstrap, optimization, and evaluation."""
    bootstrap: list[int] = Field(default_factory=list)
    online: list[int] = Field(default_factory=list)
    test: list[int] = Field(default_factory=list)
    reference: list[int] = Field(default_factory=list)


class PromptLearningJudgeConfig(ExtraModel):
    """Judge configuration shared by GEPA and MIPRO."""
    enabled: bool = False
    reward_source: Literal["task_app", "judge", "fused"] = "task_app"
    backend_base: str = ""
    backend_api_key_env: str = "SYNTH_API_KEY"
    backend_provider: str = ""
    backend_model: str = ""
    backend_rubric_id: str = ""
    backend_event_enabled: bool = True
    backend_outcome_enabled: bool = True
    backend_options: dict[str, Any] = Field(default_factory=dict)
    concurrency: int = 8
    timeout: float = 60.0
    weight_env: float = 1.0
    weight_event: float = 0.0
    weight_outcome: float = 0.0
    spec_path: str | None = None
    spec_max_tokens: int = 5000
    spec_context: str | None = None


class ProxyModelsConfig(ExtraModel):
    """Configuration for proxy usage on policy evaluations.
    
    Uses a low-fidelity (LO) model for most evaluations and a high-fidelity (HI) model
    for verification, with dynamic switching based on calibration and correlation.
    """
    hi_provider: str  # Provider for high-fidelity model
    hi_model: str  # High-fidelity model name
    lo_provider: str  # Provider for low-fidelity proxy model
    lo_model: str  # Low-fidelity proxy model name
    n_min_hi: int = 5  # Minimum HI evaluations before proxying
    r2_thresh: float = 0.5  # R² threshold for proxy calibration
    r2_stop: float = 0.2  # Stop proxying if R² drops below this
    sigma_max: float = 1e6  # Maximum sigma for proxy calibration
    sigma_stop: float = 1e9  # Stop proxying if sigma exceeds this
    verify_every: int = 0  # Verify proxy calibration every N LO calls (0 = no periodic verify)
    proxy_patience_usd: float = -100.0  # Stop proxying if net gain drops below this (USD)


class AdaptiveCurriculumLevel(str, Enum):
    """Preset levels for adaptive pooling curriculum."""
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class AdaptivePoolConfig(ExtraModel):
    """Configuration for adaptive pooling (dynamically adjusting evaluation pool size).
    
    Reduces evaluation costs by focusing on the most informative examples while
    maintaining optimization quality through informativeness-based selection.
    """
    level: AdaptiveCurriculumLevel = AdaptiveCurriculumLevel.LOW
    anchor_size: int = 30  # Frozen examples (always evaluated)
    pool_init_size: int | None = None  # Initial pool size (None = use all available)
    pool_min_size: int | None = None  # Target minimum after annealing
    warmup_iters: int = 5  # Don't start annealing until this iteration
    anneal_stop_iter: int = 20  # Reach min_size by this iteration
    pool_update_period: int = 3  # Update informativeness every N generations
    min_evals_per_example: int = 3  # Min evals per example for informativeness
    k_info_prompts: int = 10  # Number of prompts for informativeness
    info_buffer_factor: float = 0.9  # Buffer factor for informativeness
    info_epsilon: float = 1e-6  # Epsilon for informativeness
    anchor_selection_method: Literal["random", "clustering"] = "clustering"
    exploration_strategy: Literal["random", "diversity"] = "diversity"
    heatup_reserve_pool: list[int] | None = None  # Reserve pool for heat-up phase
    heatup_trigger: Literal["after_min_size", "immediate"] = "after_min_size"
    heatup_size: int = 20  # Size of heat-up pool
    heatup_cooldown_trials: int = 50  # Cooldown trials after heat-up
    heatup_schedule: Literal["repeat", "once"] = "repeat"
    
    @property
    def enabled(self) -> bool:
        """Whether adaptive pooling is enabled (level != NONE)."""
        return self.level != AdaptiveCurriculumLevel.NONE


class AdaptiveBatchLevel(str, Enum):
    """Preset levels for adaptive batch curriculum (GEPA only)."""
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class GEPAAdaptiveBatchConfig(ExtraModel):
    """Configuration for adaptive batch evaluation (GEPA only).
    
    Reduces evaluation costs by using smaller minibatches and subsampling validation.
    """
    level: AdaptiveBatchLevel = AdaptiveBatchLevel.MODERATE
    reflection_minibatch_size: int = 3  # Train examples per reflection step
    min_local_improvement: float = 0.0  # Threshold for accepting proposals
    val_evaluation_mode: Literal["full", "subsample"] = "subsample"  # Validation mode
    val_subsample_size: int = 64  # Subsample size when mode="subsample"
    candidate_selection_strategy: Literal["coverage", "random"] = "coverage"
    
    @property
    def enabled(self) -> bool:
        """Whether adaptive batch is enabled (level != NONE)."""
        return self.level != AdaptiveBatchLevel.NONE


# Default presets for adaptive pool (mirrors monorepo structure)
_ADAPTIVE_POOL_DEFAULTS: dict[AdaptiveCurriculumLevel, dict[str, Any]] = {
    AdaptiveCurriculumLevel.NONE: {
        "anchor_size": 0,
        "pool_init_size": None,
        "pool_min_size": None,
        "warmup_iters": 999_999,
        "anneal_stop_iter": 999_999,
        "pool_update_period": 999_999,
        "min_evals_per_example": 1,
        "k_info_prompts": 0,
        "info_buffer_factor": 1.0,
        "info_epsilon": 1e-6,
        "anchor_selection_method": "random",
        "exploration_strategy": "random",
        "heatup_reserve_pool": None,
        "heatup_trigger": "after_min_size",
        "heatup_size": 20,
        "heatup_cooldown_trials": 50,
        "heatup_schedule": "repeat",
    },
    AdaptiveCurriculumLevel.LOW: {
        "anchor_size": 50,
        "pool_init_size": 150,
        "pool_min_size": 100,
        "warmup_iters": 10,
        "anneal_stop_iter": 30,
        "pool_update_period": 2,
        "min_evals_per_example": 5,
        "k_info_prompts": 15,
        "info_buffer_factor": 0.95,
        "info_epsilon": 1e-6,
        "anchor_selection_method": "clustering",
        "exploration_strategy": "diversity",
        "heatup_reserve_pool": None,
        "heatup_trigger": "after_min_size",
        "heatup_size": 20,
        "heatup_cooldown_trials": 50,
        "heatup_schedule": "repeat",
    },
    AdaptiveCurriculumLevel.MODERATE: {
        "anchor_size": 30,
        "pool_init_size": 100,
        "pool_min_size": 50,
        "warmup_iters": 5,
        "anneal_stop_iter": 20,
        "pool_update_period": 3,
        "min_evals_per_example": 3,
        "k_info_prompts": 10,
        "info_buffer_factor": 0.9,
        "info_epsilon": 1e-6,
        "anchor_selection_method": "clustering",
        "exploration_strategy": "diversity",
        "heatup_reserve_pool": None,
        "heatup_trigger": "after_min_size",
        "heatup_size": 20,
        "heatup_cooldown_trials": 50,
        "heatup_schedule": "repeat",
    },
    AdaptiveCurriculumLevel.HIGH: {
        "anchor_size": 20,
        "pool_init_size": 60,
        "pool_min_size": 30,
        "warmup_iters": 3,
        "anneal_stop_iter": 10,
        "pool_update_period": 5,
        "min_evals_per_example": 2,
        "k_info_prompts": 5,
        "info_buffer_factor": 0.8,
        "info_epsilon": 1e-6,
        "anchor_selection_method": "clustering",
        "exploration_strategy": "diversity",
        "heatup_reserve_pool": None,
        "heatup_trigger": "after_min_size",
        "heatup_size": 20,
        "heatup_cooldown_trials": 50,
        "heatup_schedule": "repeat",
    },
}

# Default presets for adaptive batch (GEPA only)
_ADAPTIVE_BATCH_DEFAULTS: dict[AdaptiveBatchLevel, dict[str, Any]] = {
    AdaptiveBatchLevel.NONE: {
        "reflection_minibatch_size": 8,
        "min_local_improvement": 0.0,
        "val_evaluation_mode": "full",
        "val_subsample_size": 64,
        "candidate_selection_strategy": "random",
    },
    AdaptiveBatchLevel.LOW: {
        "reflection_minibatch_size": 5,
        "min_local_improvement": 0.0,
        "val_evaluation_mode": "subsample",
        "val_subsample_size": 80,
        "candidate_selection_strategy": "coverage",
    },
    AdaptiveBatchLevel.MODERATE: {
        "reflection_minibatch_size": 3,
        "min_local_improvement": 0.0,
        "val_evaluation_mode": "subsample",
        "val_subsample_size": 64,
        "candidate_selection_strategy": "coverage",
    },
    AdaptiveBatchLevel.HIGH: {
        "reflection_minibatch_size": 2,
        "min_local_improvement": 0.0,
        "val_evaluation_mode": "subsample",
        "val_subsample_size": 48,
        "candidate_selection_strategy": "coverage",
    },
}


def resolve_adaptive_pool_config(
    *,
    level: AdaptiveCurriculumLevel | str | None = None,
    overrides: dict[str, Any] | None = None,
    dev_pool_size: int | None = None,
) -> AdaptivePoolConfig:
    """Resolve adaptive pool config from level preset and overrides.
    
    Args:
        level: Preset level (NONE, LOW, MODERATE, HIGH). Defaults to MODERATE if None.
        overrides: Dict of field overrides to apply on top of level defaults.
        dev_pool_size: Optional dev pool size to cap pool_init_size if needed.
    
    Returns:
        AdaptivePoolConfig with resolved values.
    """
    # Normalize level
    if level is None:
        level = AdaptiveCurriculumLevel.LOW
    elif isinstance(level, str):
        try:
            level = AdaptiveCurriculumLevel[level.strip().upper()]
        except KeyError:
            valid_levels = ", ".join(l.name for l in AdaptiveCurriculumLevel)
            raise ValueError(f"Invalid adaptive pool level '{level}'. Must be one of: {valid_levels}")
    
    # Get defaults for level
    defaults = _ADAPTIVE_POOL_DEFAULTS[level].copy()
    
    # Apply overrides
    if overrides:
        defaults.update(overrides)
    
    # Handle pool_init_size and pool_min_size with dev_pool_size
    pool_init_size = defaults.get("pool_init_size")
    pool_min_size = defaults.get("pool_min_size")
    
    if pool_init_size is None:
        pool_init_size = dev_pool_size
    if pool_min_size is None:
        pool_min_size = dev_pool_size
    
    # Cap pool_init_size if dev_pool_size is provided
    if dev_pool_size is not None and pool_init_size is not None:
        if pool_init_size > dev_pool_size:
            pool_init_size = dev_pool_size
    
    # Handle heatup_reserve_pool (can be list, None, or single value)
    heatup_reserve = defaults.get("heatup_reserve_pool")
    if heatup_reserve is not None and not isinstance(heatup_reserve, (list, tuple)):
        # Convert single value or other types to list
        heatup_reserve = [heatup_reserve] if heatup_reserve else None
    
    # Create config with proper types
    config = AdaptivePoolConfig(
        level=level,
        anchor_size=int(defaults["anchor_size"]),
        pool_init_size=None if pool_init_size is None else int(pool_init_size),
        pool_min_size=None if pool_min_size is None else int(pool_min_size),
        warmup_iters=int(defaults["warmup_iters"]),
        anneal_stop_iter=int(defaults["anneal_stop_iter"]),
        pool_update_period=int(defaults["pool_update_period"]),
        min_evals_per_example=int(defaults["min_evals_per_example"]),
        k_info_prompts=int(defaults["k_info_prompts"]),
        info_buffer_factor=float(defaults["info_buffer_factor"]),
        info_epsilon=float(defaults["info_epsilon"]),
        anchor_selection_method=str(defaults["anchor_selection_method"]),
        exploration_strategy=str(defaults["exploration_strategy"]),
        heatup_reserve_pool=list(heatup_reserve) if heatup_reserve else None,
        heatup_trigger=str(defaults.get("heatup_trigger", "after_min_size")),
        heatup_size=int(defaults.get("heatup_size", 20)),
        heatup_cooldown_trials=int(defaults.get("heatup_cooldown_trials", 50)),
        heatup_schedule=str(defaults.get("heatup_schedule", "repeat")),
    )
    
    return config


def resolve_adaptive_batch_config(
    *,
    level: AdaptiveBatchLevel | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> GEPAAdaptiveBatchConfig:
    """Resolve adaptive batch config from level preset and overrides.
    
    Args:
        level: Preset level (NONE, LOW, MODERATE, HIGH). Defaults to MODERATE if None.
        overrides: Dict of field overrides to apply on top of level defaults.
    
    Returns:
        GEPAAdaptiveBatchConfig with resolved values.
    """
    # Normalize level
    if level is None:
        level = AdaptiveBatchLevel.MODERATE
    elif isinstance(level, str):
        try:
            level = AdaptiveBatchLevel[level.strip().upper()]
        except KeyError:
            valid_levels = ", ".join(l.name for l in AdaptiveBatchLevel)
            raise ValueError(f"Invalid adaptive batch level '{level}'. Must be one of: {valid_levels}")
    
    # Get defaults for level
    defaults = _ADAPTIVE_BATCH_DEFAULTS[level].copy()
    
    # Apply overrides
    if overrides:
        defaults.update(overrides)
    
    # Create config with proper types
    return GEPAAdaptiveBatchConfig(
        level=level,
        reflection_minibatch_size=int(defaults["reflection_minibatch_size"]),
        min_local_improvement=float(defaults["min_local_improvement"]),
        val_evaluation_mode=str(defaults["val_evaluation_mode"]),
        val_subsample_size=int(defaults["val_subsample_size"]),
        candidate_selection_strategy=str(defaults["candidate_selection_strategy"]),
    )


class MIPROConfig(ExtraModel):
    """MIPRO-specific configuration.
    
    MIPROv2 uses meta-learning with bootstrap phase, TPE optimization, and mini-batch evaluation
    to efficiently optimize prompts with fewer evaluations than genetic algorithms.
    """
    task_app_url: str | None = None
    task_app_api_key: str | None = None
    task_app_id: str | None = None
    num_iterations: int = 20
    num_evaluations_per_iteration: int = 5
    batch_size: int = 32
    max_concurrent: int = 20
    env_name: str = "banking77"
    env_config: dict[str, Any] | None = None
    meta_model: str = "gpt-4o-mini"
    meta_model_provider: str = "openai"
    meta_model_inference_url: str | None = None
    few_shot_score_threshold: float = 0.8
    results_file: str | None = None
    max_wall_clock_seconds: float | None = None
    max_total_tokens: int | None = None
    policy_config: dict[str, Any] | None = None
    meta: MIPROMetaConfig | dict[str, Any] | None = None
    modules: list[MIPROModuleConfig] | list[dict[str, Any]] | None = None
    seeds: MIPROSeedConfig | dict[str, Any] | None = None

    # Proposer configuration
    proposer_effort: Literal["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"] = "LOW"  # Effort level for proposer model selection
    proposer_output_tokens: Literal["RAPID", "FAST", "SLOW"] = "FAST"  # Output token limit: RAPID=3000 (gpt-oss-120b only), FAST=10000, SLOW=25000

    # Token and budget configuration (mirrors GEPA pattern)
    max_token_limit: int | None = None  # Total tokens across all rollouts (policy + proposer)
    max_spend_usd: float | None = None  # Maximum spend in USD
    token_counting_model: str = "gpt-4"  # Model for token estimation (tiktoken)
    enforce_token_limit: bool = True  # Halt optimization if limit exceeded
    
    # TPE configuration
    tpe: dict[str, Any] | None = None
    
    # Demo configuration
    demo: dict[str, Any] | None = None
    
    # Grounding configuration
    grounding: dict[str, Any] | None = None
    
    # Meta-update configuration
    meta_update: dict[str, Any] | None = None

    # Judge configuration (shared with GEPA)
    judge: PromptLearningJudgeConfig | dict[str, Any] | None = None
    
    # Proxy models configuration (optional, can also be at top-level)
    proxy_models: ProxyModelsConfig | dict[str, Any] | None = None
    
    # Adaptive pool configuration (optional)
    adaptive_pool: AdaptivePoolConfig | dict[str, Any] | None = None
    
    # System spec configuration
    spec_path: str | None = None  # Path to system spec JSON file
    spec_max_tokens: int = 5000  # Max tokens for spec context in meta-prompt
    spec_include_examples: bool = True  # Include examples from spec
    spec_priority_threshold: int | None = None  # Only include rules with priority >= threshold
    # Custom metaprompt (optional)
    metaprompt: str | None = None  # Custom metaprompt text to include in instruction generation prompts
    
    # Bootstrap seeds (for few-shot examples)
    bootstrap_train_seeds: list[int] | None = None
    
    # Online pool (for mini-batch evaluation)
    online_pool: list[int] | None = None
    
    # Test pool (held-out seeds)
    test_pool: list[int] | None = None
    
    # Reference pool (for dataset context in meta-prompt, must not overlap with train/test)
    reference_pool: list[int] | None = None

    @classmethod
    def simple(
        cls,
        *,
        task_app_url: str,
        task_app_api_key: str,
        env_name: str,
        rollout_budget: int,
        initial_prompt_messages: Sequence[Mapping[str, Any]] | Sequence[Any],
        task_app_id: str | None = None,
        bootstrap_seeds: list[int] | None = None,
        online_seeds: list[int] | None = None,
        test_seeds: list[int] | None = None,
        reference_pool: list[int] | None = None,
        env_config: dict[str, Any] | None = None,
        num_iterations: int | None = None,
        num_evaluations_per_iteration: int | None = None,
        batch_size: int | None = None,
        max_concurrent: int | None = None,
        meta_preset: Literal["fast", "balanced", "high_quality"] = "balanced",
        policy_model: str = "openai/gpt-oss-20b",
        policy_provider: str = "groq",
        policy_temperature: float = 1.0,
        policy_max_completion_tokens: int = 512,
        policy_name: str | None = None,
        meta_model: str | None = None,
        meta_provider: str | None = None,
        meta_inference_url: str | None = None,
    ) -> MIPROConfig:
        """Convenience constructor for single-stage MIPRO tasks.
        
        Automatically infers reasonable defaults for seeds, iterations, and module layout
        based on the rollout budget. This keeps simple benchmarks (e.g., Iris) readable
        while leaving the full constructor available for complex multi-stage pipelines.
        """
        if rollout_budget <= 0:
            raise ValueError("rollout_budget must be positive for MIPROConfig.simple()")
        normalized_messages = _normalize_messages(initial_prompt_messages)
        if not normalized_messages:
            raise ValueError("initial_prompt_messages must contain at least one message")
        
        bootstrap = bootstrap_seeds or _auto_calculate_bootstrap_seeds(rollout_budget)
        online = online_seeds or _auto_calculate_online_seeds(rollout_budget)
        tests = test_seeds or []
        reference = reference_pool or _auto_calculate_reference_pool(rollout_budget)
        
        iterations = num_iterations or _auto_calculate_iterations(rollout_budget)
        evals_per_iteration = (
            num_evaluations_per_iteration
            or _auto_calculate_evaluations_per_iteration(rollout_budget)
        )
        derived_batch_size = batch_size or max(1, min(len(online), 32))
        derived_max_concurrent = max_concurrent or 10
        
        baseline_instruction = _extract_baseline_instruction(normalized_messages)
        meta_config = _create_meta_config_from_preset(meta_preset)
        if meta_model:
            meta_config.model = meta_model
        if meta_provider:
            meta_config.provider = meta_provider
        if meta_inference_url is not None:
            meta_config.inference_url = meta_inference_url
        
        stage = MIPROStageConfig(
            stage_id="default_stage_0",
            baseline_instruction=baseline_instruction,
            baseline_messages=normalized_messages,
        )
        module = MIPROModuleConfig(
            module_id="default",
            stages=[stage],
        )
        seeds = MIPROSeedConfig(
            bootstrap=bootstrap,
            online=online,
            test=tests,
            reference=reference,
        )
        policy_config = {
            "model": policy_model,
            "provider": policy_provider,
            "temperature": policy_temperature,
            "max_completion_tokens": policy_max_completion_tokens,
        }
        if policy_name:
            policy_config["policy_name"] = policy_name
        
        return cls(
            task_app_url=task_app_url,
            task_app_api_key=task_app_api_key,
            task_app_id=task_app_id or env_name,
            env_name=env_name,
            env_config=env_config,
            seeds=seeds,
            num_iterations=iterations,
            num_evaluations_per_iteration=evals_per_iteration,
            batch_size=derived_batch_size,
            max_concurrent=derived_max_concurrent,
            policy_config=policy_config,
            meta=meta_config,
            modules=[module],
        )


def _auto_calculate_bootstrap_seeds(rollout_budget: int) -> list[int]:
    """Auto-calculate bootstrap seeds from rollout budget."""
    count = max(3, min(10, max(rollout_budget // 10, 1)))
    return list(range(count))


def _auto_calculate_online_seeds(rollout_budget: int) -> list[int]:
    """Auto-calculate online pool seeds from rollout budget."""
    count = max(5, min(50, max(rollout_budget // 3, 1)))
    return list(range(10, 10 + count))


def _auto_calculate_reference_pool(rollout_budget: int) -> list[int]:
    """Auto-calculate reference pool seeds from rollout budget."""
    count = max(5, min(30, max(rollout_budget // 5, 1)))
    return list(range(20, 20 + count))


def _auto_calculate_iterations(rollout_budget: int) -> int:
    """Auto-calculate number of optimization iterations."""
    online_pool_size = max(5, min(50, max(rollout_budget // 3, 1)))
    evals_per_iteration = max(3, min(10, max(rollout_budget // max(online_pool_size * 2, 1), 1)))
    iterations = max(5, min(20, max(rollout_budget // max(online_pool_size * evals_per_iteration, 1), 1)))
    return iterations


def _auto_calculate_evaluations_per_iteration(rollout_budget: int) -> int:
    """Auto-calculate number of evaluations per iteration."""
    online_pool_size = max(5, min(50, max(rollout_budget // 3, 1)))
    iterations = max(5, min(20, max(rollout_budget // max(online_pool_size * 5, 1), 1)))
    evals_per_iteration = max(3, min(10, max(rollout_budget // max(online_pool_size * iterations, 1), 1)))
    return evals_per_iteration


def _coerce_message_mapping(message: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert message objects or dicts into a mutable dict."""
    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        try:
            data = message.model_dump()
            if isinstance(data, dict):
                return data
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(message, "__dict__"):
        try:
            return {
                key: value
                for key, value in vars(message).items()
                if not key.startswith("_")
            }
        except Exception:  # pragma: no cover - defensive
            return {}
    return {}


def _extract_baseline_instruction(messages: Sequence[Mapping[str, str]] | Sequence[Any]) -> str:
    """Extract the baseline instruction string from message templates."""
    for raw in messages:
        msg = _coerce_message_mapping(raw)
        if msg.get("role", "user") == "system":
            text = (msg.get("content") or msg.get("pattern") or "").strip()
            if text:
                return text
    for raw in messages:
        msg = _coerce_message_mapping(raw)
        if msg.get("role", "user") == "user":
            text = (msg.get("content") or msg.get("pattern") or "").strip()
            if text:
                return text
    return "Complete the task."


def _normalize_messages(messages: Sequence[Mapping[str, str]] | Sequence[Any]) -> list[dict[str, str]]:
    """Normalize message dictionaries so downstream tools can rely on `content`."""
    normalized: list[dict[str, str]] = []
    for raw in messages:
        msg = _coerce_message_mapping(raw)
        role = msg.get("role", "user") or "user"
        content = msg.get("content") or msg.get("pattern") or ""
        normalized.append({"role": str(role), "content": str(content)})
    return normalized


def _create_meta_config_from_preset(preset: str) -> MIPROMetaConfig:
    """Create a meta config preset (fast/balanced/high_quality)."""
    preset_key = preset.lower().strip()
    presets: dict[str, MIPROMetaConfig] = {
        "fast": MIPROMetaConfig(
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.7,
            max_tokens=512,
            inference_url=None,
        ),
        "balanced": MIPROMetaConfig(
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.8,
            max_tokens=1024,
            inference_url=None,
        ),
        "high_quality": MIPROMetaConfig(
            model="gpt-4o",
            provider="openai",
            temperature=0.9,
            max_tokens=2048,
            inference_url=None,
        ),
    }
    return presets.get(preset_key, presets["balanced"])


# GEPA nested configs (mirroring RL structure)
class GEPARolloutConfig(ExtraModel):
    """GEPA rollout configuration (mirrors RL [rollout] section)."""
    budget: int | None = None  # Total rollout budget
    max_concurrent: int = 20  # Maximum concurrent rollouts
    minibatch_size: int = 8  # Minibatch size for evaluation


class GEPAEvaluationConfig(ExtraModel):
    """GEPA evaluation configuration (mirrors RL [evaluation] section)."""
    seeds: list[int] | None = None  # Evaluation seeds (training set)
    validation_seeds: list[int] | None = None  # Validation seeds (held-out)
    test_pool: list[int] | None = None  # Test pool (final evaluation)
    validation_pool: str | None = None  # Pool name for validation (e.g., "validation")
    validation_top_k: int | None = None  # Top-K prompts to validate


class GEPAMutationConfig(ExtraModel):
    """GEPA mutation configuration (LLM-guided mutation settings)."""
    rate: float = 0.3  # Mutation rate
    llm_model: str | None = None  # Model for generating mutations
    llm_provider: str = "groq"  # Provider for mutation LLM
    llm_inference_url: str | None = None  # Custom inference URL
    prompt: str | None = None  # Custom mutation prompt


class GEPAPopulationConfig(ExtraModel):
    """GEPA population configuration (evolution parameters)."""
    initial_size: int = 20  # Initial population size
    num_generations: int = 10  # Number of generations
    children_per_generation: int = 5  # Children generated per generation
    crossover_rate: float = 0.5  # Crossover rate
    selection_pressure: float = 1.0  # Pareto selection pressure
    patience_generations: int = 3  # Early stopping patience


class GEPAArchiveConfig(ExtraModel):
    """GEPA archive configuration (Pareto archive settings)."""
    size: int = 64  # Archive size
    pareto_set_size: int = 64  # Pareto set size
    pareto_eps: float = 1e-6  # Pareto epsilon
    feedback_fraction: float = 0.5  # Fraction of archive for feedback


class GEPATokenConfig(ExtraModel):
    """GEPA token and budget configuration."""
    max_limit: int | None = None  # Maximum tokens allowed in prompt
    counting_model: str = "gpt-4"  # Model for token counting
    enforce_pattern_limit: bool = True  # Enforce token limit on patterns
    max_spend_usd: float | None = None  # Maximum spend in USD


class GEPAModuleConfig(ExtraModel):
    """Configuration for a single GEPA pipeline module/stage (instruction-only).
    
    Each module MUST have its own policy configuration. The policy field is required
    and must include 'model' and 'provider' fields.
    """
    module_id: str
    max_instruction_slots: int = 3
    allowed_tools: list[str] | None = None
    max_tokens: int | None = None
    policy: PromptLearningPolicyConfig | dict[str, Any] = Field(
        ...,
        description="Required per-module policy configuration. Must include 'model' and 'provider' fields."
    )
    
    @field_validator("module_id")
    @classmethod
    def _validate_module_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("module_id cannot be empty")
        return v
    
    @field_validator("max_instruction_slots")
    @classmethod
    def _validate_slots(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_instruction_slots must be >= 1")
        return v
    
    @field_validator("policy", mode="before")
    @classmethod
    def _validate_policy(cls, v: Any) -> dict[str, Any]:
        """Validate that policy is a dict with required fields."""
        if v is None:
            raise ValueError("policy is required for each module/stage")
        if isinstance(v, dict):
            if not v.get("model"):
                raise ValueError("policy must include 'model' field")
            if not v.get("provider"):
                raise ValueError("policy must include 'provider' field")
            return v
        # If it's already a PromptLearningPolicyConfig, it will be validated by Pydantic
        return v


class GEPAConfig(ExtraModel):
    """GEPA-specific configuration with nested subsections."""
    # Top-level fields (for backwards compatibility)
    env_name: str = "banking77"
    env_config: dict[str, Any] | None = None
    rng_seed: int | None = None
    proposer_type: str = "dspy"  # "dspy" or "synth"
    proposer_effort: Literal["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"] = "LOW"  # Effort level for proposer model selection
    proposer_output_tokens: Literal["RAPID", "FAST", "SLOW"] = "FAST"  # Output token limit: RAPID=3000 (gpt-oss-120b only), FAST=10000, SLOW=25000
    # Custom metaprompt (optional)
    metaprompt: str | None = None  # Custom metaprompt text to include in mutation prompts
    
    # Multi-stage pipeline support
    modules: list[GEPAModuleConfig] | None = None
    
    # Nested subsections (preferred, mirrors RL structure)
    rollout: GEPARolloutConfig | None = None
    evaluation: GEPAEvaluationConfig | None = None
    mutation: GEPAMutationConfig | None = None
    population: GEPAPopulationConfig | None = None
    archive: GEPAArchiveConfig | None = None
    token: GEPATokenConfig | None = None
    judge: PromptLearningJudgeConfig | dict[str, Any] | None = None
    proxy_models: ProxyModelsConfig | dict[str, Any] | None = None  # Proxy models config (can be at top-level or gepa-specific)
    adaptive_pool: AdaptivePoolConfig | dict[str, Any] | None = None  # Adaptive pooling config
    adaptive_batch: GEPAAdaptiveBatchConfig | dict[str, Any] | None = None  # Adaptive batch config (GEPA only)
    
    # Backwards compatibility: flat fields (deprecated, prefer nested)
    # These will be flattened from nested configs if provided
    rollout_budget: int | None = None
    max_concurrent_rollouts: int | None = None
    minibatch_size: int | None = None
    evaluation_seeds: list[int] | None = None
    validation_seeds: list[int] | None = None
    test_pool: list[int] | None = None
    validation_pool: str | None = None
    validation_top_k: int | None = None
    mutation_rate: float | None = None
    mutation_llm_model: str | None = None
    mutation_llm_provider: str | None = None
    mutation_llm_inference_url: str | None = None
    mutation_prompt: str | None = None
    initial_population_size: int | None = None
    num_generations: int | None = None
    children_per_generation: int | None = None
    crossover_rate: float | None = None
    selection_pressure: float | None = None
    patience_generations: int | None = None
    archive_size: int | None = None
    pareto_set_size: int | None = None
    pareto_eps: float | None = None
    feedback_fraction: float | None = None
    max_token_limit: int | None = None
    token_counting_model: str | None = None
    enforce_pattern_token_limit: bool | None = None
    max_spend_usd: float | None = None
    
    def _get_rollout_budget(self) -> int | None:
        """Get rollout budget from nested or flat structure."""
        if self.rollout and self.rollout.budget is not None:
            return self.rollout.budget
        return self.rollout_budget
    
    def _get_max_concurrent_rollouts(self) -> int:
        """Get max concurrent rollouts from nested or flat structure."""
        if self.rollout and self.rollout.max_concurrent is not None:
            return self.rollout.max_concurrent
        return self.max_concurrent_rollouts or 20
    
    def _get_minibatch_size(self) -> int:
        """Get minibatch size from nested or flat structure."""
        if self.rollout and self.rollout.minibatch_size is not None:
            return self.rollout.minibatch_size
        return self.minibatch_size or 8
    
    def _get_evaluation_seeds(self) -> list[int] | None:
        """Get evaluation seeds from nested or flat structure."""
        if self.evaluation and self.evaluation.seeds is not None:
            return self.evaluation.seeds
        return self.evaluation_seeds
    
    def _get_validation_seeds(self) -> list[int] | None:
        """Get validation seeds from nested or flat structure."""
        if self.evaluation and self.evaluation.validation_seeds is not None:
            return self.evaluation.validation_seeds
        return self.validation_seeds
    
    def _get_test_pool(self) -> list[int] | None:
        """Get test pool from nested or flat structure."""
        if self.evaluation and self.evaluation.test_pool is not None:
            return self.evaluation.test_pool
        return self.test_pool
    
    def _get_mutation_rate(self) -> float:
        """Get mutation rate from nested or flat structure."""
        if self.mutation and self.mutation.rate is not None:
            return self.mutation.rate
        return self.mutation_rate or 0.3
    
    def _get_mutation_llm_model(self) -> str | None:
        """Get mutation LLM model from nested or flat structure."""
        if self.mutation and self.mutation.llm_model is not None:
            return self.mutation.llm_model
        return self.mutation_llm_model
    
    def _get_mutation_llm_provider(self) -> str:
        """Get mutation LLM provider from nested or flat structure."""
        if self.mutation and self.mutation.llm_provider is not None:
            return self.mutation.llm_provider
        return self.mutation_llm_provider or "groq"
    
    def _get_mutation_llm_inference_url(self) -> str | None:
        """Get mutation LLM inference URL from nested or flat structure."""
        if self.mutation and self.mutation.llm_inference_url is not None:
            return self.mutation.llm_inference_url
        return self.mutation_llm_inference_url
    
    def _get_mutation_prompt(self) -> str | None:
        """Get mutation prompt from nested or flat structure."""
        if self.mutation and self.mutation.prompt is not None:
            return self.mutation.prompt
        return self.mutation_prompt
    
    def _get_initial_population_size(self) -> int:
        """Get initial population size from nested or flat structure."""
        if self.population and self.population.initial_size is not None:
            return self.population.initial_size
        return self.initial_population_size or 20
    
    def _get_num_generations(self) -> int:
        """Get num generations from nested or flat structure."""
        if self.population and self.population.num_generations is not None:
            return self.population.num_generations
        return self.num_generations or 10
    
    def _get_children_per_generation(self) -> int:
        """Get children per generation from nested or flat structure."""
        if self.population and self.population.children_per_generation is not None:
            return self.population.children_per_generation
        return self.children_per_generation or 5
    
    def _get_crossover_rate(self) -> float:
        """Get crossover rate from nested or flat structure."""
        if self.population and self.population.crossover_rate is not None:
            return self.population.crossover_rate
        return self.crossover_rate or 0.5
    
    def _get_selection_pressure(self) -> float:
        """Get selection pressure from nested or flat structure."""
        if self.population and self.population.selection_pressure is not None:
            return self.population.selection_pressure
        return self.selection_pressure or 1.0
    
    def _get_patience_generations(self) -> int:
        """Get patience generations from nested or flat structure."""
        if self.population and self.population.patience_generations is not None:
            return self.population.patience_generations
        return self.patience_generations or 3
    
    def _get_archive_size(self) -> int:
        """Get archive size from nested or flat structure."""
        if self.archive and self.archive.size is not None:
            return self.archive.size
        return self.archive_size or 64
    
    def _get_pareto_set_size(self) -> int:
        """Get pareto set size from nested or flat structure."""
        if self.archive and self.archive.pareto_set_size is not None:
            return self.archive.pareto_set_size
        return self.pareto_set_size or 64
    
    def _get_pareto_eps(self) -> float:
        """Get pareto eps from nested or flat structure."""
        if self.archive and self.archive.pareto_eps is not None:
            return self.archive.pareto_eps
        return self.pareto_eps or 1e-6
    
    def _get_feedback_fraction(self) -> float:
        """Get feedback fraction from nested or flat structure."""
        if self.archive and self.archive.feedback_fraction is not None:
            return self.archive.feedback_fraction
        return self.feedback_fraction or 0.5
    
    def _get_max_token_limit(self) -> int | None:
        """Get max token limit from nested or flat structure."""
        if self.token and self.token.max_limit is not None:
            return self.token.max_limit
        return self.max_token_limit
    
    def _get_token_counting_model(self) -> str:
        """Get token counting model from nested or flat structure."""
        if self.token and self.token.counting_model is not None:
            return self.token.counting_model
        return self.token_counting_model or "gpt-4"
    
    def _get_enforce_pattern_token_limit(self) -> bool:
        """Get enforce pattern token limit from nested or flat structure."""
        if self.token and self.token.enforce_pattern_limit is not None:
            return self.token.enforce_pattern_limit
        return self.enforce_pattern_token_limit if self.enforce_pattern_token_limit is not None else True
    
    def _get_max_spend_usd(self) -> float | None:
        """Get max spend USD from nested or flat structure."""
        if self.token and self.token.max_spend_usd is not None:
            return self.token.max_spend_usd
        return self.max_spend_usd
    
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GEPAConfig:
        """Load GEPA config from dict/TOML, handling both nested and flat structures."""
        # Check for nested structure first
        nested_data = {}
        flat_data = {}
        
        for key, value in data.items():
            if key in ("rollout", "evaluation", "mutation", "population", "archive", "token", "modules", "proxy_models", "adaptive_pool", "adaptive_batch", "judge"):
                nested_data[key] = value
            else:
                flat_data[key] = value
        
        # If we have nested data, create nested configs
        if nested_data:
            if "rollout" in nested_data:
                nested_data["rollout"] = GEPARolloutConfig.model_validate(nested_data["rollout"])
            if "evaluation" in nested_data:
                nested_data["evaluation"] = GEPAEvaluationConfig.model_validate(nested_data["evaluation"])
            if "mutation" in nested_data:
                nested_data["mutation"] = GEPAMutationConfig.model_validate(nested_data["mutation"])
            if "population" in nested_data:
                nested_data["population"] = GEPAPopulationConfig.model_validate(nested_data["population"])
            if "archive" in nested_data:
                nested_data["archive"] = GEPAArchiveConfig.model_validate(nested_data["archive"])
            if "token" in nested_data:
                nested_data["token"] = GEPATokenConfig.model_validate(nested_data["token"])
            if "modules" in nested_data:
                modules_data = nested_data["modules"]
                if isinstance(modules_data, list):
                    nested_data["modules"] = [
                        GEPAModuleConfig.model_validate(m) if isinstance(m, dict) else m
                        for m in modules_data
                    ]
            if "proxy_models" in nested_data and isinstance(nested_data["proxy_models"], dict):
                nested_data["proxy_models"] = ProxyModelsConfig.model_validate(nested_data["proxy_models"])
            if "adaptive_pool" in nested_data and isinstance(nested_data["adaptive_pool"], dict):
                # Resolve adaptive pool config with level and overrides
                adaptive_pool_data = nested_data["adaptive_pool"]
                level = adaptive_pool_data.get("level")
                overrides = {k: v for k, v in adaptive_pool_data.items() if k != "level"}
                # Get dev_pool_size from evaluation.seeds if available
                dev_pool_size = None
                if "evaluation" in nested_data and isinstance(nested_data["evaluation"], dict):
                    eval_seeds = nested_data["evaluation"].get("seeds")
                    if isinstance(eval_seeds, list):
                        dev_pool_size = len(eval_seeds)
                nested_data["adaptive_pool"] = resolve_adaptive_pool_config(
                    level=level,
                    overrides=overrides if overrides else None,
                    dev_pool_size=dev_pool_size,
                )
            if "adaptive_batch" in nested_data and isinstance(nested_data["adaptive_batch"], dict):
                # Resolve adaptive batch config with level and overrides
                adaptive_batch_data = nested_data["adaptive_batch"]
                level = adaptive_batch_data.get("level")
                overrides = {k: v for k, v in adaptive_batch_data.items() if k != "level"}
                try:
                    nested_data["adaptive_batch"] = resolve_adaptive_batch_config(
                        level=level,
                        overrides=overrides if overrides else None,
                    )
                except Exception as exc:
                    # Re-raise with clearer context
                    raise ValueError(f"Failed to resolve adaptive_batch config: {exc}") from exc
        
        # Merge nested and flat data
        merged_data = {**flat_data, **nested_data}
        return cls.model_validate(merged_data)


class PromptLearningConfig(ExtraModel):
    """Top-level prompt learning configuration."""
    algorithm: str  # "mipro" or "gepa"
    task_app_url: str
    task_app_api_key: str | None = None
    task_app_id: str | None = None
    initial_prompt: PromptPatternConfig | None = None
    policy: PromptLearningPolicyConfig | None = None
    mipro: MIPROConfig | None = None
    gepa: GEPAConfig | None = None
    judge: PromptLearningJudgeConfig | dict[str, Any] | None = None
    proxy_models: ProxyModelsConfig | dict[str, Any] | None = None  # Proxy models config (can be at top-level or algorithm-specific)
    env_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for API payload."""
        result = self.model_dump(mode="python", exclude_none=True)
        # Ensure prompt_learning section wraps everything
        if "prompt_learning" not in result:
            pl_data = dict(result.items())
            result = {"prompt_learning": pl_data}
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PromptLearningConfig:
        """Load prompt learning config from dict/TOML mapping."""
        # Handle both [prompt_learning] section and flat structure
        pl_data = data.get("prompt_learning", {})
        if not pl_data:
            # If no prompt_learning section, assume top-level is prompt_learning
            pl_data = dict(data)
        
        # Handle proxy_models at top-level FIRST (takes precedence over algorithm-specific)
        # This ensures top-level proxy_models is available for algorithm configs to check
        top_level_proxy_models = None
        if "proxy_models" in pl_data and isinstance(pl_data["proxy_models"], dict):
            top_level_proxy_models = ProxyModelsConfig.model_validate(pl_data["proxy_models"])
            pl_data["proxy_models"] = top_level_proxy_models
        
        # Handle gepa config specially to support nested structure
        if "gepa" in pl_data and isinstance(pl_data["gepa"], dict):
            gepa_data = pl_data["gepa"]
            # If top-level proxy_models exists, remove gepa-specific proxy_models (top-level takes precedence)
            if top_level_proxy_models is not None and "proxy_models" in gepa_data:
                gepa_data.pop("proxy_models")
            pl_data["gepa"] = GEPAConfig.from_mapping(gepa_data)
            # Ensure gepa config uses top-level proxy_models if available
            if top_level_proxy_models is not None:
                # Note: gepa.proxy_models will be None, but top-level proxy_models will be used by backend
                pass
        
        # Handle mipro config - check for adaptive_pool
        if "mipro" in pl_data and isinstance(pl_data["mipro"], dict):
            mipro_data = pl_data["mipro"]
            # If top-level proxy_models exists, remove mipro-specific proxy_models (top-level takes precedence)
            if top_level_proxy_models is not None and "proxy_models" in mipro_data:
                mipro_data.pop("proxy_models")
            
            # Extract bootstrap_train_seeds and online_pool from top-level pl_data if not in mipro_data
            # These fields can be at top-level [prompt_learning] or nested [prompt_learning.mipro]
            if "bootstrap_train_seeds" not in mipro_data and "bootstrap_train_seeds" in pl_data:
                mipro_data["bootstrap_train_seeds"] = pl_data["bootstrap_train_seeds"]
            if "online_pool" not in mipro_data and "online_pool" in pl_data:
                mipro_data["online_pool"] = pl_data["online_pool"]
            if "test_pool" not in mipro_data and "test_pool" in pl_data:
                mipro_data["test_pool"] = pl_data["test_pool"]
            if "reference_pool" not in mipro_data and "reference_pool" in pl_data:
                mipro_data["reference_pool"] = pl_data["reference_pool"]
            
            # Handle adaptive_pool in mipro config
            if "adaptive_pool" in mipro_data and isinstance(mipro_data["adaptive_pool"], dict):
                adaptive_pool_data = mipro_data["adaptive_pool"]
                level = adaptive_pool_data.get("level")
                overrides = {k: v for k, v in adaptive_pool_data.items() if k != "level"}
                # Get dev_pool_size from online_pool if available
                dev_pool_size = None
                online_pool = mipro_data.get("online_pool") or (mipro_data.get("seeds") or {}).get("online", [])
                if isinstance(online_pool, list):
                    dev_pool_size = len(online_pool)
                try:
                    mipro_data["adaptive_pool"] = resolve_adaptive_pool_config(
                        level=level,
                        overrides=overrides if overrides else None,
                        dev_pool_size=dev_pool_size,
                    )
                except Exception as exc:
                    # Re-raise with clearer context
                    raise ValueError(f"Failed to resolve mipro.adaptive_pool config: {exc}") from exc
            
            # Handle proxy_models in mipro config
            if "proxy_models" in mipro_data and isinstance(mipro_data["proxy_models"], dict):
                mipro_data["proxy_models"] = ProxyModelsConfig.model_validate(mipro_data["proxy_models"])
        
        if "judge" in pl_data and isinstance(pl_data["judge"], dict):
            pl_data["judge"] = PromptLearningJudgeConfig.model_validate(pl_data["judge"])
        
        return cls.model_validate(pl_data)

    @classmethod
    def from_path(cls, path: Path) -> PromptLearningConfig:
        """Load prompt learning config from TOML file."""
        content = load_toml(path)
        return cls.from_mapping(content)


__all__ = [
    "GEPAConfig",
    "GEPAModuleConfig",
    "GEPARolloutConfig",
    "GEPAEvaluationConfig",
    "GEPAMutationConfig",
    "GEPAPopulationConfig",
    "GEPAArchiveConfig",
    "GEPATokenConfig",
    "GEPAAdaptiveBatchConfig",
    "MIPROConfig",
    "MIPROMetaConfig",
    "MIPROModuleConfig",
    "MIPROStageConfig",
    "MIPROSeedConfig",
    "MessagePatternConfig",
    "PromptLearningConfig",
    "PromptLearningPolicyConfig",
    "PromptPatternConfig",
    "PromptLearningJudgeConfig",
    "ProxyModelsConfig",
    "AdaptivePoolConfig",
    "AdaptiveCurriculumLevel",
    "AdaptiveBatchLevel",
    "resolve_adaptive_pool_config",
    "resolve_adaptive_batch_config",
]
