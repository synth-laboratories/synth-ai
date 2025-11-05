"""SDK-side validation for training configs - catch errors BEFORE sending to backend."""

from pathlib import Path
from typing import Any

import click


class ConfigValidationError(Exception):
    """Raised when a training config is invalid."""
    pass


def validate_prompt_learning_config(config_data: dict[str, Any], config_path: Path) -> None:
    """
    Validate prompt learning config BEFORE sending to backend.
    
    This catches common errors early with clear messages instead of cryptic backend errors.
    
    Args:
        config_data: Parsed TOML/JSON config
        config_path: Path to config file (for error messages)
    
    Raises:
        ConfigValidationError: If config is invalid
        click.ClickException: If validation fails (for CLI)
    """
    errors: list[str] = []
    
    # Check for prompt_learning section
    pl_section = config_data.get("prompt_learning")
    if not pl_section:
        errors.append(
            "Missing [prompt_learning] section in config. "
            "Expected: [prompt_learning] with algorithm, task_app_url, etc."
        )
        _raise_validation_errors(errors, config_path)
        return
    
    if not isinstance(pl_section, dict):
        errors.append(
            f"[prompt_learning] must be a table/dict, got {type(pl_section).__name__}"
        )
        _raise_validation_errors(errors, config_path)
        return
    
    # CRITICAL: Validate algorithm field
    algorithm = pl_section.get("algorithm")
    if not algorithm:
        errors.append(
            "Missing required field: prompt_learning.algorithm\n"
            "  Must be one of: 'gepa', 'mipro'\n"
            "  Example:\n"
            "    [prompt_learning]\n"
            "    algorithm = \"gepa\""
        )
    elif algorithm not in ("gepa", "mipro"):
        errors.append(
            f"Invalid algorithm: '{algorithm}'\n"
            f"  Must be one of: 'gepa', 'mipro' (Note: MIPRO not yet implemented)\n"
            f"  Got: '{algorithm}'"
        )
    
    # Validate task_app_url
    task_app_url = pl_section.get("task_app_url")
    if not task_app_url:
        errors.append(
            "Missing required field: prompt_learning.task_app_url\n"
            "  Example:\n"
            "    task_app_url = \"http://127.0.0.1:8102\""
        )
    elif not isinstance(task_app_url, str):
        errors.append(
            f"task_app_url must be a string, got {type(task_app_url).__name__}"
        )
    elif not task_app_url.startswith(("http://", "https://")):
        errors.append(
            f"task_app_url must start with http:// or https://, got: '{task_app_url}'"
        )
    
    # Validate initial_prompt if present
    initial_prompt = pl_section.get("initial_prompt")
    if initial_prompt:
        if not isinstance(initial_prompt, dict):
            errors.append(
                f"prompt_learning.initial_prompt must be a table/dict, got {type(initial_prompt).__name__}"
            )
        else:
            # Validate messages array
            messages = initial_prompt.get("messages")
            if messages is not None:
                if not isinstance(messages, list):
                    errors.append(
                        f"prompt_learning.initial_prompt.messages must be an array, got {type(messages).__name__}"
                    )
                elif len(messages) == 0:
                    errors.append(
                        "prompt_learning.initial_prompt.messages is empty (must have at least one message)"
                    )
    
    # Validate policy config
    policy = pl_section.get("policy")
    if not policy or not isinstance(policy, dict):
        errors.append("Missing [prompt_learning.policy] section or not a table")
    else:
        # Enforce inference_mode
        mode = str(policy.get("inference_mode", "")).strip().lower()
        if not mode:
            errors.append("Missing required field: prompt_learning.policy.inference_mode (must be 'synth_hosted')")
        elif mode != "synth_hosted":
            errors.append("prompt_learning.policy.inference_mode must be 'synth_hosted' (bring_your_own unsupported)")
        # Required fields for synth_hosted
        provider = (policy.get("provider") or "").strip().lower()
        model = (policy.get("model") or "").strip()
        inference_url = (policy.get("inference_url") or "").strip()
        if not provider:
            errors.append("Missing required field: prompt_learning.policy.provider")
        if not model:
            errors.append("Missing required field: prompt_learning.policy.model")
        if not inference_url:
            errors.append("Missing required field: prompt_learning.policy.inference_url")
        elif not isinstance(inference_url, str) or not inference_url.startswith(("http://", "https://")):
            errors.append(f"policy.inference_url must start with http:// or https://, got: '{inference_url}'")
    
    # Validate algorithm-specific config
    if algorithm == "gepa":
        gepa_config = pl_section.get("gepa")
        if not gepa_config or not isinstance(gepa_config, dict):
            errors.append("Missing [prompt_learning.gepa] section for GEPA algorithm")
        else:
            # Numeric sanity checks
            def _pos_int(name: str) -> None:
                val = gepa_config.get(name)
                if val is not None:
                    try:
                        ival = int(val)
                        if ival <= 0:
                            errors.append(f"prompt_learning.gepa.{name} must be > 0")
                    except Exception:
                        errors.append(f"prompt_learning.gepa.{name} must be an integer")
            for fld in ("initial_population_size", "num_generations", "children_per_generation", "max_concurrent_rollouts"):
                _pos_int(fld)
            
            # Require validation_seeds (even if empty list) for final baseline comparison
            evaluation_section = gepa_config.get("evaluation", {})
            if "validation_seeds" not in evaluation_section and "validation_seeds" not in gepa_config and "validation_seeds" not in pl_section:
                errors.append(
                    "prompt_learning.gepa.evaluation.validation_seeds is required (even if empty []).\n"
                    "  Validation seeds are a held-out set used ONLY for final comparison, never touched during optimization.\n"
                    "  Example:\n"
                    "    [prompt_learning.gepa.evaluation]\n"
                    "    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Training seeds (used by GEPA)\n"
                    "    validation_seeds = [10, 11, 12, 13, 14]  # Held-out seeds (baseline vs top-K comparison)\n"
                    "  Or use an empty list if you don't want validation:\n"
                    "    validation_seeds = []"
                )
            
            # Budget cap
            if "max_spend_usd" in gepa_config and gepa_config.get("max_spend_usd") is not None:
                try:
                    f = float(gepa_config.get("max_spend_usd"))
                    if f <= 0:
                        errors.append("prompt_learning.gepa.max_spend_usd must be > 0 when provided")
                except Exception:
                    errors.append("prompt_learning.gepa.max_spend_usd must be numeric")
    
    elif algorithm == "mipro":
        # MIPRO is not yet implemented in synth-ai
        errors.append(
            "MIPRO algorithm is not yet implemented in synth-ai.\n"
            "  Please use 'gepa' algorithm for prompt optimization.\n"
            "  MIPRO support is planned for a future release.\n"
            "  Example:\n"
            "    [prompt_learning]\n"
            "    algorithm = \"gepa\"\n"
            "    [prompt_learning.gepa]\n"
            "    # ... gepa configuration"
        )
    
    # Raise all errors at once for better UX
    if errors:
        _raise_validation_errors(errors, config_path)


def _raise_validation_errors(errors: list[str], config_path: Path) -> None:
    """Format and raise validation errors."""
    error_msg = (
        f"\n❌ Invalid prompt learning config: {config_path}\n\n"
        f"Found {len(errors)} error(s):\n\n"
    )
    
    for i, error in enumerate(errors, 1):
        # Indent multi-line errors
        indented_error = "\n  ".join(error.split("\n"))
        error_msg += f"{i}. {indented_error}\n\n"
    
    error_msg += (
        "📖 See example configs:\n"
        "  - examples/blog_posts/gepa/configs/banking77_gepa_local.toml\n"
        "  - examples/blog_posts/mipro/configs/banking77_mipro_local.toml\n"
    )
    
    raise click.ClickException(error_msg)


def validate_rl_config(config_data: dict[str, Any], config_path: Path) -> None:
    """
    Validate RL config BEFORE sending to backend.
    
    Args:
        config_data: Parsed TOML/JSON config
        config_path: Path to config file (for error messages)
    
    Raises:
        ConfigValidationError: If config is invalid
        click.ClickException: If validation fails (for CLI)
    """
    errors: list[str] = []
    
    # Check for rl section
    rl_section = config_data.get("rl") or config_data.get("online_rl")
    if not rl_section:
        errors.append(
            "Missing [rl] or [online_rl] section in config"
        )
        _raise_validation_errors(errors, config_path)
        return
    
    # Validate algorithm
    algorithm = rl_section.get("algorithm")
    if not algorithm:
        errors.append(
            "Missing required field: rl.algorithm\n"
            "  Must be one of: 'grpo', 'ppo', etc."
        )
    
    # Validate task_url
    task_url = rl_section.get("task_url")
    if not task_url:
        errors.append(
            "Missing required field: rl.task_url"
        )
    elif not isinstance(task_url, str):
        errors.append(
            f"task_url must be a string, got {type(task_url).__name__}"
        )
    
    if errors:
        _raise_validation_errors(errors, config_path)


def validate_sft_config(config_data: dict[str, Any], config_path: Path) -> None:
    """
    Validate SFT config BEFORE sending to backend.
    
    Args:
        config_data: Parsed TOML/JSON config
        config_path: Path to config file (for error messages)
    
    Raises:
        ConfigValidationError: If config is invalid
        click.ClickException: If validation fails (for CLI)
    """
    errors: list[str] = []
    
    # Check for sft section
    sft_section = config_data.get("sft")
    if not sft_section:
        errors.append(
            "Missing [sft] section in config"
        )
        _raise_validation_errors(errors, config_path)
        return
    
    # Validate model
    model = sft_section.get("model")
    if not model:
        errors.append(
            "Missing required field: sft.model"
        )
    
    if errors:
        _raise_validation_errors(errors, config_path)


__all__ = [
    "ConfigValidationError",
    "validate_prompt_learning_config",
    "validate_rl_config",
    "validate_sft_config",
]

