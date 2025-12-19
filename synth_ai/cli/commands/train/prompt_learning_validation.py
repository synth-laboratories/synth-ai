"""Validation for prompt learning (GEPA/MIPRO) configurations.

This module validates TOML configs and warns about:
1. Unrecognized/unknown sections/fields that won't be processed
2. Deprecated fields that should be migrated
3. Missing required fields
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

# Known top-level TOML sections
KNOWN_TOP_LEVEL_SECTIONS = {
    "prompt_learning",
    "display",  # Deprecated but recognized
    "termination_config",  # Supported - creates TerminationManager conditions
}

# Known fields in [prompt_learning] section
KNOWN_PROMPT_LEARNING_FIELDS = {
    "algorithm",
    "task_app_url",
    "task_app_api_key",
    "task_app_id",
    "initial_prompt",
    "policy",
    "mipro",
    "gepa",
    "judge",
    "proxy_models",
    "env_config",
    "env_name",
    "termination_config",  # Supported - backend uses TerminationManager
    "free_tier",  # Enable free tier mode with OSS models
    # Deprecated fields (still recognized for warnings)
    "results_folder",
    "env_file_path",
    # Seed pools (can be at top-level or nested)
    "bootstrap_train_seeds",
    "online_pool",
    "test_pool",
    "reference_pool",
}

# Known fields in [prompt_learning.policy] section
KNOWN_POLICY_FIELDS = {
    "model",
    "provider",
    "inference_url",
    "inference_mode",
    "temperature",
    "max_completion_tokens",
    "policy_name",
}

# Known fields in [prompt_learning.termination_config] section
# All of these are SUPPORTED by backend's TerminationManager
KNOWN_TERMINATION_CONFIG_FIELDS = {
    "max_cost_usd",  # BudgetTerminationCondition
    "max_trials",  # TrialLimitTerminationCondition
    "max_seconds",  # TimeLimitTerminationCondition
    "max_time_seconds",  # Alias for max_seconds
    "max_rollouts",  # RolloutLimitTerminationCondition
    "max_trials_without_improvement",  # NoImprovementTerminationCondition
    "pessimism_enabled",  # PessimismTerminationCondition
    "max_category_costs_usd",  # CategoryBudgetTerminationCondition
}

# Known fields in [prompt_learning.gepa] section
KNOWN_GEPA_FIELDS = {
    # Top-level GEPA fields
    "env_name",
    "env_config",
    "rng_seed",
    "proposer_type",
    "proposer_effort",
    "proposer_output_tokens",
    "metaprompt",
    "modules",
    # Nested subsections
    "rollout",
    "evaluation",
    "mutation",
    "population",
    "archive",
    "token",
    "judge",
    "proxy_models",
    "adaptive_pool",
    "adaptive_batch",
    # Backwards-compat flat fields (deprecated but recognized)
    "rollout_budget",
    "max_concurrent_rollouts",
    "minibatch_size",
    "evaluation_seeds",
    "validation_seeds",
    "test_pool",
    "validation_pool",
    "validation_top_k",
    "mutation_rate",
    "mutation_llm_model",
    "mutation_llm_provider",
    "mutation_llm_inference_url",
    "mutation_prompt",
    "initial_population_size",
    "num_generations",
    "children_per_generation",
    "crossover_rate",
    "selection_pressure",
    "patience_generations",
    "archive_size",
    "pareto_set_size",
    "pareto_eps",
    "feedback_fraction",
    "max_token_limit",
    "token_counting_model",
    "enforce_pattern_token_limit",
    "max_spend_usd",
}

# Known fields in [prompt_learning.gepa.rollout]
KNOWN_GEPA_ROLLOUT_FIELDS = {
    "budget",
    "max_concurrent",
    "minibatch_size",
}

# Known fields in [prompt_learning.gepa.evaluation]
KNOWN_GEPA_EVALUATION_FIELDS = {
    "seeds",
    "validation_seeds",
    "test_pool",
    "validation_pool",
    "validation_top_k",
}

# Known fields in [prompt_learning.gepa.mutation]
KNOWN_GEPA_MUTATION_FIELDS = {
    "rate",
    "llm_model",
    "llm_provider",
    "llm_inference_url",
    "prompt",
}

# Known fields in [prompt_learning.gepa.population]
KNOWN_GEPA_POPULATION_FIELDS = {
    "initial_size",
    "num_generations",
    "children_per_generation",
    "crossover_rate",
    "selection_pressure",
    "patience_generations",
}

# Known fields in [prompt_learning.gepa.archive]
KNOWN_GEPA_ARCHIVE_FIELDS = {
    "size",
    "pareto_set_size",
    "pareto_eps",
    "feedback_fraction",
}

# Known fields in [prompt_learning.gepa.token]
KNOWN_GEPA_TOKEN_FIELDS = {
    "max_limit",
    "counting_model",
    "enforce_pattern_limit",
    "max_spend_usd",
}

# Known fields in [prompt_learning.mipro] section
KNOWN_MIPRO_FIELDS = {
    "task_app_url",
    "task_app_api_key",
    "task_app_id",
    "num_iterations",
    "num_evaluations_per_iteration",
    "batch_size",
    "max_concurrent",
    "env_name",
    "env_config",
    "meta_model",
    "meta_model_provider",
    "meta_model_inference_url",
    "few_shot_score_threshold",
    "results_file",
    "max_wall_clock_seconds",
    "max_total_tokens",
    "policy_config",
    "meta",
    "modules",
    "seeds",
    "proposer_effort",
    "proposer_output_tokens",
    "max_token_limit",
    "max_spend_usd",
    "token_counting_model",
    "enforce_token_limit",
    "tpe",
    "demo",
    "grounding",
    "meta_update",
    "judge",
    "proxy_models",
    "adaptive_pool",
    "spec_path",
    "spec_max_tokens",
    "spec_include_examples",
    "spec_priority_threshold",
    "metaprompt",
    "bootstrap_train_seeds",
    "online_pool",
    "test_pool",
    "reference_pool",
    "min_bootstrap_demos",
}

# Known fields in [prompt_learning.judge]
KNOWN_JUDGE_FIELDS = {
    "enabled",
    "reward_source",
    "backend_base",
    "backend_api_key_env",
    "backend_provider",
    "backend_model",
    "synth_verifier_id",
    "backend_rubric_id",
    "backend_event_enabled",
    "backend_outcome_enabled",
    "backend_options",
    "concurrency",
    "timeout",
    "weight_env",
    "weight_event",
    "weight_outcome",
    "spec_path",
    "spec_max_tokens",
    "spec_context",
}

# Known fields in adaptive_pool config
KNOWN_ADAPTIVE_POOL_FIELDS = {
    "level",
    "anchor_size",
    "pool_init_size",
    "pool_min_size",
    "warmup_iters",
    "anneal_stop_iter",
    "pool_update_period",
    "min_evals_per_example",
    "k_info_prompts",
    "info_buffer_factor",
    "info_epsilon",
    "anchor_selection_method",
    "exploration_strategy",
    "heatup_reserve_pool",
    "heatup_trigger",
    "heatup_size",
    "heatup_cooldown_trials",
    "heatup_schedule",
}

# Known fields in adaptive_batch config (GEPA only)
KNOWN_ADAPTIVE_BATCH_FIELDS = {
    "level",
    "reflection_minibatch_size",
    "min_local_improvement",
    "val_evaluation_mode",
    "val_subsample_size",
    "candidate_selection_strategy",
}

# Known fields in proxy_models config
KNOWN_PROXY_MODELS_FIELDS = {
    "hi_provider",
    "hi_model",
    "lo_provider",
    "lo_model",
    "n_min_hi",
    "r2_thresh",
    "r2_stop",
    "sigma_max",
    "sigma_stop",
    "verify_every",
    "proxy_patience_usd",
}

# Deprecated fields with migration suggestions
DEPRECATED_FIELDS = {
    # Truly deprecated (backend ignores)
    "display": "The [display] section is deprecated and ignored by the backend. Remove it from your config.",
    "results_folder": "'results_folder' is deprecated and ignored by the backend. Remove it from your config.",
    "env_file_path": "'env_file_path' is deprecated. Use environment variables instead.",
    # Deprecated flat GEPA fields (prefer nested)
    "rollout_budget": "Use [prompt_learning.gepa.rollout].budget instead of flat rollout_budget.",
    "max_concurrent_rollouts": "Use [prompt_learning.gepa.rollout].max_concurrent instead.",
    "evaluation_seeds": "Use [prompt_learning.gepa.evaluation].seeds instead of flat evaluation_seeds.",
    "validation_seeds": "Use [prompt_learning.gepa.evaluation].validation_seeds instead.",
    "backend_rubric_id": "Use 'synth_verifier_id' instead of 'backend_rubric_id' in [prompt_learning.judge].",
}


class PromptLearningValidationResult:
    """Result of prompt learning config validation."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    @property
    def is_valid(self) -> bool:
        """Config is valid if there are no errors."""
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    def __str__(self) -> str:
        lines = []
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")
        if self.info:
            lines.append("INFO:")
            for i in self.info:
                lines.append(f"  ℹ️  {i}")
        return "\n".join(lines) if lines else "✅ Config is valid"


def _check_unknown_fields(
    config: dict[str, Any],
    known_fields: set[str],
    section_path: str,
    result: PromptLearningValidationResult,
) -> None:
    """Check for unknown fields in a config section."""
    for key in config:
        if key not in known_fields:
            result.add_warning(
                f"Unknown field '{key}' in [{section_path}]. "
                f"This field will be ignored. Check spelling or remove it."
            )


def _check_deprecated_fields(
    config: dict[str, Any],
    section_path: str,
    result: PromptLearningValidationResult,
) -> None:
    """Check for deprecated fields and suggest migrations."""
    for key in config:
        if key in DEPRECATED_FIELDS:
            result.add_warning(f"[{section_path}] {DEPRECATED_FIELDS[key]}")


def validate_prompt_learning_config(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> PromptLearningValidationResult:
    """Validate a prompt learning configuration.

    Args:
        config: Parsed TOML config dictionary
        config_path: Optional path for better error messages

    Returns:
        PromptLearningValidationResult with errors, warnings, and info
    """
    result = PromptLearningValidationResult()
    path_prefix = f"({config_path}) " if config_path else ""

    # Check top-level sections
    for key in config:
        if key not in KNOWN_TOP_LEVEL_SECTIONS:
            result.add_warning(
                f"{path_prefix}Unknown top-level section '[{key}]'. "
                f"Known sections: {', '.join(sorted(KNOWN_TOP_LEVEL_SECTIONS))}"
            )

    # Check for deprecated [display] section
    if "display" in config:
        result.add_warning(
            f"{path_prefix}The [display] section is deprecated and ignored by the backend. "
            "Remove it to clean up your config."
        )

    # Validate [prompt_learning] section
    pl_config = config.get("prompt_learning", {})
    if not pl_config:
        result.add_error(f"{path_prefix}Missing required [prompt_learning] section")
        return result

    # Check for unknown fields in prompt_learning
    _check_unknown_fields(pl_config, KNOWN_PROMPT_LEARNING_FIELDS, "prompt_learning", result)

    # Check for deprecated fields
    _check_deprecated_fields(pl_config, "prompt_learning", result)

    # Check required fields
    algorithm = pl_config.get("algorithm")
    if not algorithm:
        result.add_error(f"{path_prefix}Missing required 'algorithm' field in [prompt_learning]")
    elif algorithm not in ("gepa", "mipro"):
        result.add_error(
            f"{path_prefix}Invalid algorithm '{algorithm}'. Must be 'gepa' or 'mipro'"
        )

    if not pl_config.get("task_app_url"):
        result.add_error(f"{path_prefix}Missing required 'task_app_url' in [prompt_learning]")

    # Validate [prompt_learning.policy] if present
    policy = pl_config.get("policy")
    if policy and isinstance(policy, dict):
        _check_unknown_fields(policy, KNOWN_POLICY_FIELDS, "prompt_learning.policy", result)

    # Validate [prompt_learning.termination_config] if present
    termination = pl_config.get("termination_config")
    if termination and isinstance(termination, dict):
        _check_unknown_fields(
            termination,
            KNOWN_TERMINATION_CONFIG_FIELDS,
            "prompt_learning.termination_config",
            result,
        )
        # Info: termination_config IS supported
        result.add_info(
            "termination_config is supported and will create backend TerminationManager conditions"
        )

    # Validate [prompt_learning.judge] if present
    judge = pl_config.get("judge")
    if judge and isinstance(judge, dict):
        _check_unknown_fields(judge, KNOWN_JUDGE_FIELDS, "prompt_learning.judge", result)

    # Validate [prompt_learning.proxy_models] if present
    proxy_models = pl_config.get("proxy_models")
    if proxy_models and isinstance(proxy_models, dict):
        _check_unknown_fields(
            proxy_models, KNOWN_PROXY_MODELS_FIELDS, "prompt_learning.proxy_models", result
        )

    # Validate algorithm-specific sections
    if algorithm == "gepa":
        _validate_gepa_config(pl_config.get("gepa", {}), result, path_prefix)
    elif algorithm == "mipro":
        _validate_mipro_config(pl_config.get("mipro", {}), result, path_prefix)

    return result


def _validate_gepa_config(
    gepa: dict[str, Any],
    result: PromptLearningValidationResult,
    path_prefix: str,
) -> None:
    """Validate GEPA-specific configuration."""
    if not gepa:
        result.add_warning(f"{path_prefix}No [prompt_learning.gepa] section found for GEPA algorithm")
        return

    # Check for unknown fields
    _check_unknown_fields(gepa, KNOWN_GEPA_FIELDS, "prompt_learning.gepa", result)

    # Check for deprecated flat fields
    deprecated_flat = {
        "rollout_budget",
        "max_concurrent_rollouts",
        "evaluation_seeds",
        "validation_seeds",
    }
    for field in deprecated_flat:
        if field in gepa:
            result.add_info(
                f"Using flat '{field}' in [prompt_learning.gepa] - "
                "consider migrating to nested structure for clarity"
            )

    # Validate nested sections
    if "rollout" in gepa and isinstance(gepa["rollout"], dict):
        _check_unknown_fields(
            gepa["rollout"], KNOWN_GEPA_ROLLOUT_FIELDS, "prompt_learning.gepa.rollout", result
        )

    if "evaluation" in gepa and isinstance(gepa["evaluation"], dict):
        _check_unknown_fields(
            gepa["evaluation"],
            KNOWN_GEPA_EVALUATION_FIELDS,
            "prompt_learning.gepa.evaluation",
            result,
        )

    if "mutation" in gepa and isinstance(gepa["mutation"], dict):
        _check_unknown_fields(
            gepa["mutation"], KNOWN_GEPA_MUTATION_FIELDS, "prompt_learning.gepa.mutation", result
        )

    if "population" in gepa and isinstance(gepa["population"], dict):
        _check_unknown_fields(
            gepa["population"],
            KNOWN_GEPA_POPULATION_FIELDS,
            "prompt_learning.gepa.population",
            result,
        )

    if "archive" in gepa and isinstance(gepa["archive"], dict):
        _check_unknown_fields(
            gepa["archive"], KNOWN_GEPA_ARCHIVE_FIELDS, "prompt_learning.gepa.archive", result
        )

    if "token" in gepa and isinstance(gepa["token"], dict):
        _check_unknown_fields(
            gepa["token"], KNOWN_GEPA_TOKEN_FIELDS, "prompt_learning.gepa.token", result
        )

    if "adaptive_pool" in gepa and isinstance(gepa["adaptive_pool"], dict):
        _check_unknown_fields(
            gepa["adaptive_pool"],
            KNOWN_ADAPTIVE_POOL_FIELDS,
            "prompt_learning.gepa.adaptive_pool",
            result,
        )

    if "adaptive_batch" in gepa and isinstance(gepa["adaptive_batch"], dict):
        _check_unknown_fields(
            gepa["adaptive_batch"],
            KNOWN_ADAPTIVE_BATCH_FIELDS,
            "prompt_learning.gepa.adaptive_batch",
            result,
        )

    if "proxy_models" in gepa and isinstance(gepa["proxy_models"], dict):
        _check_unknown_fields(
            gepa["proxy_models"],
            KNOWN_PROXY_MODELS_FIELDS,
            "prompt_learning.gepa.proxy_models",
            result,
        )

    if "judge" in gepa and isinstance(gepa["judge"], dict):
        _check_unknown_fields(
            gepa["judge"], KNOWN_JUDGE_FIELDS, "prompt_learning.gepa.judge", result
        )


def _validate_mipro_config(
    mipro: dict[str, Any],
    result: PromptLearningValidationResult,
    path_prefix: str,
) -> None:
    """Validate MIPRO-specific configuration."""
    if not mipro:
        result.add_warning(
            f"{path_prefix}No [prompt_learning.mipro] section found for MIPRO algorithm"
        )
        return

    # Check for unknown fields
    _check_unknown_fields(mipro, KNOWN_MIPRO_FIELDS, "prompt_learning.mipro", result)

    # Validate nested sections
    if "judge" in mipro and isinstance(mipro["judge"], dict):
        _check_unknown_fields(
            mipro["judge"], KNOWN_JUDGE_FIELDS, "prompt_learning.mipro.judge", result
        )

    if "adaptive_pool" in mipro and isinstance(mipro["adaptive_pool"], dict):
        _check_unknown_fields(
            mipro["adaptive_pool"],
            KNOWN_ADAPTIVE_POOL_FIELDS,
            "prompt_learning.mipro.adaptive_pool",
            result,
        )

    if "proxy_models" in mipro and isinstance(mipro["proxy_models"], dict):
        _check_unknown_fields(
            mipro["proxy_models"],
            KNOWN_PROXY_MODELS_FIELDS,
            "prompt_learning.mipro.proxy_models",
            result,
        )


def validate_and_warn(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Validate config and print warnings/errors to stderr.

    This is a convenience function for CLI usage that prints validation
    results and raises an exception if there are errors.

    Args:
        config: Parsed TOML config dictionary
        config_path: Optional path for better error messages

    Raises:
        ValueError: If config has validation errors
    """
    result = validate_prompt_learning_config(config, config_path=config_path)

    # Print warnings (these don't stop execution)
    for warning in result.warnings:
        warnings.warn(warning, UserWarning, stacklevel=2)

    # Print info messages
    for info in result.info:
        print(f"ℹ️  {info}")

    # Raise on errors
    if result.errors:
        error_msg = "\n".join(f"  - {e}" for e in result.errors)
        raise ValueError(f"Config validation failed:\n{error_msg}")


__all__ = [
    "validate_prompt_learning_config",
    "validate_and_warn",
    "PromptLearningValidationResult",
]
