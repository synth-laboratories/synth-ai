"""SDK-side validation for training configs - catch errors BEFORE sending to backend."""

import re
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import click
import toml

# Import unknown field validation from CLI module
from synth_ai.cli.commands.train.prompt_learning_validation import (
    validate_prompt_learning_config as _validate_unknown_fields,
)
from synth_ai.core.telemetry import log_info


class ConfigValidationError(Exception):
    """Raised when a training config is invalid."""

    pass


# Supported models for prompt learning (GEPA)
# NOTE: gpt-5-pro is explicitly EXCLUDED - too expensive for prompt learning
OPENAI_SUPPORTED_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    # OpenAI Image Generation Models
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
    "chatgpt-image-latest",
    # Explicitly EXCLUDED: "gpt-5-pro" - too expensive
}

# Groq supported models - patterns and exact matches
# Models can be in format "model-name" or "provider/model-name" (e.g., "openai/gpt-oss-20b")
GROQ_SUPPORTED_PATTERNS = [
    re.compile(r"^(openai/)?gpt-oss-\d+b"),  # e.g., gpt-oss-20b, openai/gpt-oss-120b
    re.compile(r"^(llama-3\.3-70b|groq/llama-3\.3-70b)"),  # e.g., llama-3.3-70b-versatile
    re.compile(r"^(qwen.*32b|groq/qwen.*32b)"),  # e.g., qwen-32b, qwen3-32b, groq/qwen3-32b
]

GROQ_EXACT_MATCHES = {
    "llama-3.3-70b",
    "llama-3.1-8b-instant",
    "qwen-32b",
    "qwen3-32b",
}

# Google/Gemini supported models
GOOGLE_SUPPORTED_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-pro-gt200k",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image",  # Image generation model
    "gemini-3-pro-image-preview",  # Gemini 3 pro image generation model
}


def _is_supported_openai_model(model: str) -> bool:
    """Check if model is a supported OpenAI model."""
    model_lower = model.lower().strip()
    # Strip provider prefix if present (e.g., "openai/gpt-4o" -> "gpt-4o")
    if "/" in model_lower:
        model_lower = model_lower.split("/", 1)[1]
    return model_lower in {m.lower() for m in OPENAI_SUPPORTED_MODELS}


def _is_supported_groq_model(model: str) -> bool:
    """Check if model is a supported Groq model."""
    model_lower = model.lower().strip()

    # Remove provider prefix if present (e.g., "openai/gpt-oss-20b" -> "gpt-oss-20b")
    if "/" in model_lower:
        model_lower = model_lower.split("/", 1)[1]

    # Check exact matches first
    if model_lower in {m.lower() for m in GROQ_EXACT_MATCHES}:
        return True

    # Check patterns (patterns already handle provider prefix)
    return any(pattern.match(model.lower().strip()) for pattern in GROQ_SUPPORTED_PATTERNS)


def _is_supported_google_model(model: str) -> bool:
    """Check if model is a supported Google/Gemini model."""
    model_lower = model.lower().strip()
    # Strip provider prefix if present (e.g., "google/gemini-2.5-flash-lite" -> "gemini-2.5-flash-lite")
    if "/" in model_lower:
        model_lower = model_lower.split("/", 1)[1]
    return model_lower in {m.lower() for m in GOOGLE_SUPPORTED_MODELS}


def _validate_adaptive_pool_config(
    adaptive_pool_section: dict[str, Any],
    prefix: str,  # e.g., "gepa.adaptive_pool"
    errors: list[str],
) -> None:
    """Validate adaptive_pool configuration section.

    Validates all fields in adaptive_pool config including:
    - Level presets (NONE, LOW, MODERATE, HIGH)
    - Numeric fields with min/max constraints
    - Relationship constraints (pool_init_size >= pool_min_size >= anchor_size)
    - String enum fields (anchor_selection_method, exploration_strategy, etc.)
    - Heat-up phase configuration

    Args:
        adaptive_pool_section: Dict containing adaptive_pool config with fields:
            - level: Preset level (NONE, LOW, MODERATE, HIGH)
            - anchor_size: Number of anchor examples (always evaluated)
            - pool_init_size: Initial pool size
            - pool_min_size: Target minimum pool size after annealing
            - warmup_iters: Iterations before starting annealing
            - anneal_stop_iter: Iteration when pool reaches min_size
            - pool_update_period: Update informativeness every N generations
            - min_evals_per_example: Min evals before computing informativeness
            - k_info_prompts: Number of prompts for informativeness
            - info_buffer_factor: Buffer factor (0.0-1.0) for preserving info
            - info_epsilon: Epsilon for informativeness calculations
            - anchor_selection_method: "random" or "clustering"
            - exploration_strategy: "random" or "diversity"
            - heatup_trigger: "after_min_size", "immediate", or "every_N_trials_after_min"
            - heatup_schedule: "repeat" or "once"
            - heatup_size: Number of seeds to add during heat-up
            - heatup_cooldown_trials: Trials to wait before cooling down
            - heatup_reserve_pool: Optional list of seed IDs for heat-up
        prefix: Prefix for error messages (e.g., "gepa.adaptive_pool")
        errors: List to append validation errors to
    """
    if not isinstance(adaptive_pool_section, dict):
        errors.append(f"❌ {prefix} must be a table/dict when provided")
        return

    # Validate level
    level = adaptive_pool_section.get("level")
    if level is not None:
        valid_levels = {"NONE", "LOW", "MODERATE", "HIGH"}
        if str(level).upper() not in valid_levels:
            errors.append(f"❌ {prefix}.level must be one of {valid_levels}, got '{level}'")

    # Validate numeric fields
    for field, min_val in [
        ("anchor_size", 0),
        ("pool_init_size", 0),
        ("pool_min_size", 0),
        ("warmup_iters", 0),
        ("anneal_stop_iter", 0),
        ("pool_update_period", 1),
        ("min_evals_per_example", 1),
        ("k_info_prompts", 0),
    ]:
        val = adaptive_pool_section.get(field)
        if val is not None:
            try:
                ival = int(val)
                if ival < min_val:
                    errors.append(f"❌ {prefix}.{field} must be >= {min_val}, got {ival}")
            except (TypeError, ValueError):
                errors.append(f"❌ {prefix}.{field} must be an integer, got {type(val).__name__}")

    # Validate pool_init_size >= pool_min_size if both provided
    pool_init = adaptive_pool_section.get("pool_init_size")
    pool_min = adaptive_pool_section.get("pool_min_size")
    if pool_init is not None and pool_min is not None:
        try:
            pool_init_int = int(pool_init)
            pool_min_int = int(pool_min)
            if pool_init_int < pool_min_int:
                errors.append(
                    f"❌ {prefix}.pool_init_size ({pool_init}) must be >= pool_min_size ({pool_min})"
                )
        except (TypeError, ValueError):
            pass  # Already validated above

    # Validate pool_min_size >= anchor_size if both provided
    anchor_size = adaptive_pool_section.get("anchor_size")
    if pool_min is not None and anchor_size is not None:
        try:
            pool_min_int = int(pool_min)
            anchor_size_int = int(anchor_size)
            if pool_min_int < anchor_size_int:
                errors.append(
                    f"❌ {prefix}.pool_min_size ({pool_min}) must be >= anchor_size ({anchor_size})"
                )
        except (TypeError, ValueError):
            pass  # Already validated above

    # Validate info_buffer_factor and info_epsilon
    for field, min_val, max_val in [("info_buffer_factor", 0.0, 1.0), ("info_epsilon", 0.0, None)]:
        val = adaptive_pool_section.get(field)
        if val is not None:
            try:
                fval = float(val)
                if fval < min_val:
                    errors.append(f"❌ {prefix}.{field} must be >= {min_val}, got {fval}")
                if max_val is not None and fval > max_val:
                    errors.append(f"❌ {prefix}.{field} must be <= {max_val}, got {fval}")
            except (TypeError, ValueError):
                errors.append(f"❌ {prefix}.{field} must be numeric, got {type(val).__name__}")

    # Validate string fields
    anchor_method = adaptive_pool_section.get("anchor_selection_method")
    if anchor_method is not None and anchor_method not in ("random", "clustering"):
        errors.append(
            f"❌ {prefix}.anchor_selection_method must be 'random' or 'clustering', got '{anchor_method}'"
        )

    exploration_strategy = adaptive_pool_section.get("exploration_strategy")
    if exploration_strategy is not None and exploration_strategy not in ("random", "diversity"):
        errors.append(
            f"❌ {prefix}.exploration_strategy must be 'random' or 'diversity', got '{exploration_strategy}'"
        )

    # Validate heatup fields
    heatup_trigger = adaptive_pool_section.get("heatup_trigger")
    if heatup_trigger is not None and heatup_trigger not in (
        "after_min_size",
        "immediate",
        "every_N_trials_after_min",
    ):
        errors.append(
            f"❌ {prefix}.heatup_trigger must be 'after_min_size', 'immediate', or 'every_N_trials_after_min', got '{heatup_trigger}'"
        )

    heatup_schedule = adaptive_pool_section.get("heatup_schedule")
    if heatup_schedule is not None and heatup_schedule not in ("repeat", "once"):
        errors.append(
            f"❌ {prefix}.heatup_schedule must be 'repeat' or 'once', got '{heatup_schedule}'"
        )

    heatup_size = adaptive_pool_section.get("heatup_size")
    if heatup_size is not None:
        try:
            if int(heatup_size) <= 0:
                errors.append(f"❌ {prefix}.heatup_size must be > 0, got {heatup_size}")
        except (TypeError, ValueError):
            errors.append(
                f"❌ {prefix}.heatup_size must be an integer, got {type(heatup_size).__name__}"
            )

    heatup_cooldown_trials = adaptive_pool_section.get("heatup_cooldown_trials")
    if heatup_cooldown_trials is not None:
        try:
            if int(heatup_cooldown_trials) < 0:
                errors.append(
                    f"❌ {prefix}.heatup_cooldown_trials must be >= 0, got {heatup_cooldown_trials}"
                )
        except (TypeError, ValueError):
            errors.append(
                f"❌ {prefix}.heatup_cooldown_trials must be an integer, got {type(heatup_cooldown_trials).__name__}"
            )

    heatup_reserve_pool = adaptive_pool_section.get("heatup_reserve_pool")
    if heatup_reserve_pool is not None:
        if not isinstance(heatup_reserve_pool, list):
            errors.append(
                f"❌ {prefix}.heatup_reserve_pool must be a list, got {type(heatup_reserve_pool).__name__}"
            )
        elif not all(isinstance(s, int) for s in heatup_reserve_pool):
            errors.append(f"❌ {prefix}.heatup_reserve_pool must contain only integers")


def _validate_model_for_provider(
    model: str, provider: str, field_name: str, *, allow_nano: bool = False
) -> list[str]:
    """
    Validate that a model is supported for the given provider.

    Models can be specified with or without provider prefix (e.g., "gpt-4o" or "openai/gpt-4o").
    The provider prefix is stripped before validation.

    REJECTS gpt-5-pro explicitly (too expensive).
    REJECTS nano models for proposal/mutation models (unless allow_nano=True).

    Args:
        model: Model name to validate
        provider: Provider name (openai, groq, google)
        field_name: Field name for error messages (e.g., "prompt_learning.policy.model")
        allow_nano: If True, allow nano models (for policy models). If False, reject nano models.

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    if not model or not isinstance(model, str) or not model.strip():
        errors.append(f"Missing or empty {field_name}")
        return errors

    provider_lower = provider.lower().strip()
    model_lower = model.lower().strip()

    # Strip provider prefix if present (e.g., "openai/gpt-4o" -> "gpt-4o")
    model_without_prefix = model_lower.split("/", 1)[1] if "/" in model_lower else model_lower

    # Explicitly reject gpt-5-pro (too expensive)
    if model_without_prefix == "gpt-5-pro":
        errors.append(
            f"Model '{model}' is not supported for prompt learning (too expensive).\n"
            f"  gpt-5-pro is excluded due to high cost ($15/$120 per 1M tokens).\n"
            f"  Please use a supported model instead."
        )
        return errors

    # Reject nano models for proposal/mutation models (unless explicitly allowed)
    if not allow_nano and model_without_prefix.endswith("-nano"):
        errors.append(
            f"Model '{model}' is not supported for {field_name}.\n"
            f"  ❌ Nano models (e.g., gpt-4.1-nano, gpt-5-nano) are NOT allowed for proposal/mutation models.\n"
            f"  \n"
            f"  Why?\n"
            f"  Proposal and mutation models need to be SMART and capable of generating high-quality,\n"
            f"  creative prompt variations. Nano models are too small and lack the reasoning capability\n"
            f"  needed for effective prompt optimization.\n"
            f"  \n"
            f"  ✅ Use a larger model instead:\n"
            f"     - For OpenAI: gpt-4.1-mini, gpt-4o-mini, gpt-4o, or gpt-4.1\n"
            f"     - For Groq: openai/gpt-oss-120b, llama-3.3-70b-versatile\n"
            f"     - For Google: gemini-2.5-flash, gemini-2.5-pro\n"
            f"  \n"
            f"  Note: Nano models ARE allowed for policy models (task execution), but NOT for\n"
            f"  proposal/mutation models (prompt generation)."
        )
        return errors

    if provider_lower == "openai":
        if not _is_supported_openai_model(model_without_prefix):
            errors.append(
                f"Unsupported OpenAI model: '{model}'\n"
                f"  Supported OpenAI models for prompt learning:\n"
                f"    - gpt-4o\n"
                f"    - gpt-4o-mini\n"
                f"    - gpt-4.1, gpt-4.1-mini, gpt-4.1-nano\n"
                f"    - gpt-5, gpt-5-mini, gpt-5-nano\n"
                f"    - Image generation: gpt-image-1.5, gpt-image-1, gpt-image-1-mini, chatgpt-image-latest\n"
                f"  Note: gpt-5-pro is excluded (too expensive)\n"
                f"  Got: '{model}'"
            )
    elif provider_lower == "groq":
        # For Groq, check both with and without prefix since models can be "openai/gpt-oss-20b"
        if not _is_supported_groq_model(model_lower):
            errors.append(
                f"Unsupported Groq model: '{model}'\n"
                f"  Supported Groq models for prompt learning:\n"
                f"    - gpt-oss-Xb (e.g., gpt-oss-20b, openai/gpt-oss-120b)\n"
                f"    - llama-3.3-70b (and variants like llama-3.3-70b-versatile)\n"
                f"    - llama-3.1-8b-instant\n"
                f"    - qwen/qwen3-32b (and variants)\n"
                f"  Got: '{model}'"
            )
    elif provider_lower == "google":
        if not _is_supported_google_model(model_without_prefix):
            errors.append(
                f"Unsupported Google/Gemini model: '{model}'\n"
                f"  Supported Google models for prompt learning:\n"
                f"    - gemini-2.5-pro, gemini-2.5-pro-gt200k\n"
                f"    - gemini-2.5-flash\n"
                f"    - gemini-2.5-flash-lite\n"
                f"    - Image generation: gemini-2.5-flash-image, gemini-3-pro-image-preview\n"
                f"  Got: '{model}'"
            )
    else:
        errors.append(
            f"Unsupported provider: '{provider}'\n"
            f"  Supported providers for prompt learning: 'openai', 'groq', 'google'\n"
            f"  Got: '{provider}'"
        )

    return errors


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
    ctx: dict[str, Any] = {"config_path": str(config_path)}
    log_info("validate_prompt_learning_config invoked", ctx=ctx)
    errors: list[str] = []

    # Run unknown field validation (warnings only, doesn't raise)
    try:
        validation_result = _validate_unknown_fields(config_data, config_path=config_path)
        # Print warnings about unknown fields and deprecated sections
        for warning_msg in validation_result.warnings:
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
    except Exception:
        # Don't fail validation if unknown field check fails
        pass

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
        errors.append(f"[prompt_learning] must be a table/dict, got {type(pl_section).__name__}")
        _raise_validation_errors(errors, config_path)
        return

    # CRITICAL: Validate algorithm field
    algorithm = pl_section.get("algorithm")
    if not algorithm:
        errors.append(
            "Missing required field: prompt_learning.algorithm\n"
            "  Must be 'gepa'\n"
            "  Example:\n"
            "    [prompt_learning]\n"
            '    algorithm = "gepa"'
        )
    elif algorithm != "gepa":
        errors.append(f"Invalid algorithm: '{algorithm}'\n  Must be 'gepa'\n  Got: '{algorithm}'")

    # Validate task_app_url
    task_app_url = pl_section.get("task_app_url")
    if not task_app_url:
        errors.append(
            "Missing required field: prompt_learning.task_app_url\n"
            "  Example:\n"
            '    task_app_url = "http://127.0.0.1:8102"'
        )
    elif not isinstance(task_app_url, str):
        errors.append(f"task_app_url must be a string, got {type(task_app_url).__name__}")
    elif not task_app_url.startswith(("http://", "https://")):
        errors.append(f"task_app_url must start with http:// or https://, got: '{task_app_url}'")

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
            errors.append(
                "Missing required field: prompt_learning.policy.inference_mode (must be 'synth_hosted')"
            )
        elif mode != "synth_hosted":
            errors.append(
                "prompt_learning.policy.inference_mode must be 'synth_hosted' (bring_your_own unsupported)"
            )
        # Required fields for synth_hosted
        provider = (policy.get("provider") or "").strip()
        model = (policy.get("model") or "").strip()
        if not provider:
            errors.append("Missing required field: prompt_learning.policy.provider")
        if not model:
            errors.append("Missing required field: prompt_learning.policy.model")
        else:
            # Validate model is supported for the provider
            if provider:
                errors.extend(
                    _validate_model_for_provider(
                        model, provider, "prompt_learning.policy.model", allow_nano=True
                    )
                )
        # VALIDATION: Reject inference_url in config - trainer must provide it in rollout requests
        if "inference_url" in policy:
            errors.append(
                "inference_url must not be specified in [prompt_learning.policy]. "
                "The trainer provides the inference URL in rollout requests. "
                "Remove inference_url from your config file."
            )
        if "api_base" in policy:
            errors.append(
                "api_base must not be specified in [prompt_learning.policy]. "
                "The trainer provides the inference URL in rollout requests. "
                "Remove api_base from your config file."
            )
        if "base_url" in policy:
            errors.append(
                "base_url must not be specified in [prompt_learning.policy]. "
                "The trainer provides the inference URL in rollout requests. "
                "Remove base_url from your config file."
            )

    # Validate proxy_models config (can be at top-level or algorithm-specific)
    proxy_models_section = pl_section.get("proxy_models")
    if proxy_models_section:
        if not isinstance(proxy_models_section, dict):
            errors.append(
                f"prompt_learning.proxy_models must be a table/dict, got {type(proxy_models_section).__name__}"
            )
        else:
            required_fields = ["hi_provider", "hi_model", "lo_provider", "lo_model"]
            for field in required_fields:
                if not proxy_models_section.get(field):
                    errors.append(f"prompt_learning.proxy_models.{field} is required")
            # Validate numeric fields
            for field, min_val in [
                ("n_min_hi", 0),
                ("r2_thresh", 0.0),
                ("r2_stop", 0.0),
                ("sigma_max", 0.0),
                ("sigma_stop", 0.0),
                ("verify_every", 0),
            ]:
                val = proxy_models_section.get(field)
                if val is not None:
                    try:
                        if field in ("r2_thresh", "r2_stop"):
                            fval = float(val)
                            if not (0.0 <= fval <= 1.0):
                                errors.append(
                                    f"prompt_learning.proxy_models.{field} must be between 0.0 and 1.0, got {fval}"
                                )
                        elif field.startswith("sigma"):
                            fval = float(val)
                            if fval < min_val:
                                errors.append(
                                    f"prompt_learning.proxy_models.{field} must be >= {min_val}, got {fval}"
                                )
                        else:
                            ival = int(val)
                            if ival < min_val:
                                errors.append(
                                    f"prompt_learning.proxy_models.{field} must be >= {min_val}, got {ival}"
                                )
                    except (TypeError, ValueError):
                        errors.append(
                            f"prompt_learning.proxy_models.{field} must be numeric, got {type(val).__name__}"
                        )
            # Validate provider/model combinations
            if proxy_models_section.get("hi_provider") and proxy_models_section.get("hi_model"):
                hi_errors = _validate_model_for_provider(
                    proxy_models_section["hi_model"],
                    proxy_models_section["hi_provider"],
                    "prompt_learning.proxy_models.hi_model",
                    allow_nano=True,
                )
                errors.extend(hi_errors)
            if proxy_models_section.get("lo_provider") and proxy_models_section.get("lo_model"):
                lo_errors = _validate_model_for_provider(
                    proxy_models_section["lo_model"],
                    proxy_models_section["lo_provider"],
                    "prompt_learning.proxy_models.lo_model",
                    allow_nano=True,
                )
                errors.extend(lo_errors)

    # Validate verifier config (for GEPA)
    verifier_section = pl_section.get("verifier") or {}
    if verifier_section:
        if not isinstance(verifier_section, dict):
            errors.append(
                f"prompt_learning.verifier must be a table/dict, got {type(verifier_section).__name__}"
            )
        else:
            reward_source = str(verifier_section.get("reward_source", "task_app")).strip().lower()
            enabled = bool(verifier_section.get("enabled"))
            if reward_source and reward_source not in {"task_app", "verifier", "fused"}:
                errors.append(
                    "prompt_learning.verifier.reward_source must be 'task_app', 'verifier', or 'fused'"
                )
            # backend_base, backend_provider, backend_model are extracted but
            # validation is delegated to Pydantic model in configs/prompt_learning.py
            _ = verifier_section.get("backend_base", "")
            _ = verifier_section.get("backend_provider", "")
            _ = verifier_section.get("backend_model", "")
            if enabled:
                pass
            if reward_source == "fused":
                weight_event = verifier_section.get("weight_event", 0.0)
                weight_outcome = verifier_section.get("weight_outcome", 0.0)
                try:
                    weight_event_f = float(weight_event)
                except (TypeError, ValueError):
                    errors.append("prompt_learning.verifier.weight_event must be numeric")
                    weight_event_f = 0.0
                try:
                    weight_outcome_f = float(weight_outcome)
                except (TypeError, ValueError):
                    errors.append("prompt_learning.verifier.weight_outcome must be numeric")
                    weight_outcome_f = 0.0
                if weight_event_f <= 0 and weight_outcome_f <= 0:
                    errors.append(
                        "prompt_learning.verifier.reward_source='fused' requires weight_event > 0 or weight_outcome > 0"
                    )

    # Check for multi-stage/multi-module pipeline config
    initial_prompt = pl_section.get("initial_prompt", {})
    pipeline_modules: list[str | dict[str, Any]] = []
    if isinstance(initial_prompt, dict):
        metadata = initial_prompt.get("metadata", {})
        pipeline_modules = metadata.get("pipeline_modules", [])
        if not isinstance(pipeline_modules, list):
            pipeline_modules = []
    has_multi_stage = isinstance(pipeline_modules, list) and len(pipeline_modules) > 0

    # Validate algorithm-specific config
    if algorithm == "gepa":
        gepa_config = pl_section.get("gepa")
        if not gepa_config or not isinstance(gepa_config, dict):
            errors.append("Missing [prompt_learning.gepa] section for GEPA algorithm")
        else:
            # Multi-stage validation
            modules_config = gepa_config.get("modules")
            if has_multi_stage:
                if (
                    not modules_config
                    or not isinstance(modules_config, list)
                    or len(modules_config) == 0
                ):
                    errors.append(
                        f"GEPA multi-stage pipeline detected (found {len(pipeline_modules)} modules in "
                        f"prompt_learning.initial_prompt.metadata.pipeline_modules), "
                        f"but [prompt_learning.gepa.modules] is missing or empty. "
                        f"Define module configs for each pipeline stage."
                    )
                else:
                    # Validate module IDs match pipeline_modules
                    module_ids = []
                    for m in modules_config:
                        if isinstance(m, dict):
                            module_id = m.get("module_id") or m.get("stage_id")
                            if module_id:
                                module_ids.append(str(module_id).strip())
                        elif hasattr(m, "module_id"):
                            module_ids.append(str(m.module_id).strip())
                        elif hasattr(m, "stage_id"):
                            module_ids.append(str(m.stage_id).strip())

                    # Extract pipeline module names (can be strings or dicts with 'name' field)
                    pipeline_module_names = []
                    for m in pipeline_modules:
                        if isinstance(m, str):
                            pipeline_module_names.append(m.strip())
                        elif isinstance(m, dict):
                            name = m.get("name") or m.get("module_id") or m.get("stage_id")
                            if name:
                                pipeline_module_names.append(str(name).strip())

                    # Check for missing modules
                    missing_modules = set(pipeline_module_names) - set(module_ids)
                    if missing_modules:
                        errors.append(
                            f"Pipeline modules {sorted(missing_modules)} are missing from "
                            f"[prompt_learning.gepa.modules]. Each pipeline module must have a corresponding "
                            f"module config with matching module_id."
                        )

                    # Check for extra modules (warn but don't error)
                    extra_modules = set(module_ids) - set(pipeline_module_names)
                    if extra_modules:
                        # This is a warning, not an error - extra modules are allowed
                        pass

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

            def _pos_int_nested(section: str, name: str) -> None:
                """Check positive int in nested section."""
                section_config = gepa_config.get(section)
                if section_config and isinstance(section_config, dict):
                    val = section_config.get(name)
                    if val is not None:
                        try:
                            ival = int(val)
                            if ival <= 0:
                                errors.append(f"prompt_learning.gepa.{section}.{name} must be > 0")
                        except Exception:
                            errors.append(
                                f"prompt_learning.gepa.{section}.{name} must be an integer"
                            )

            def _non_neg_int(name: str) -> None:
                """Check non-negative int."""
                val = gepa_config.get(name)
                if val is not None:
                    try:
                        ival = int(val)
                        if ival < 0:
                            errors.append(f"prompt_learning.gepa.{name} must be >= 0")
                    except Exception:
                        errors.append(f"prompt_learning.gepa.{name} must be an integer")

            def _rate_float(name: str) -> None:
                """Check float in [0.0, 1.0] range."""
                val = gepa_config.get(name)
                if val is not None:
                    try:
                        fval = float(val)
                        if not (0.0 <= fval <= 1.0):
                            errors.append(
                                f"prompt_learning.gepa.{name} must be between 0.0 and 1.0"
                            )
                    except Exception:
                        errors.append(f"prompt_learning.gepa.{name} must be numeric")

            def _pos_float(name: str) -> None:
                """Check positive float."""
                val = gepa_config.get(name)
                if val is not None:
                    try:
                        fval = float(val)
                        if fval <= 0:
                            errors.append(f"prompt_learning.gepa.{name} must be > 0")
                    except Exception:
                        errors.append(f"prompt_learning.gepa.{name} must be numeric")

            # Required positive integers
            for fld in (
                "initial_population_size",
                "num_generations",
                "children_per_generation",
                "max_concurrent_rollouts",
            ):
                _pos_int(fld)

            # Nested rollout config validation
            _pos_int_nested("rollout", "budget")
            _pos_int_nested("rollout", "max_concurrent")
            _pos_int_nested("rollout", "minibatch_size")

            # Nested population config validation
            _pos_int_nested("population", "initial_size")
            _pos_int_nested("population", "num_generations")
            _pos_int_nested("population", "children_per_generation")
            _rate_float("mutation_rate")  # Can be at top level or in mutation section
            _rate_float("crossover_rate")  # Can be at top level or in population section
            _pos_float("selection_pressure")  # Must be >= 1.0
            selection_pressure = gepa_config.get("selection_pressure")
            if selection_pressure is not None:
                try:
                    sp = float(selection_pressure)
                    if sp < 1.0:
                        errors.append("prompt_learning.gepa.selection_pressure must be >= 1.0")
                except Exception:
                    pass  # Already caught by type check
            _non_neg_int("patience_generations")

            # Nested archive config validation
            _pos_int_nested("archive", "size")
            _pos_int_nested("archive", "pareto_set_size")
            _pos_float("pareto_eps")  # Must be > 0, typically very small
            _rate_float("feedback_fraction")

            # Nested mutation config validation
            mutation_config = gepa_config.get("mutation")
            if mutation_config and isinstance(mutation_config, dict):
                _rate_float("mutation_rate")  # Check in mutation section too
                mutation_model = mutation_config.get("llm_model")
                mutation_provider = mutation_config.get("llm_provider", "").strip()
                if mutation_model:
                    if not mutation_provider:
                        errors.append(
                            "Missing required field: prompt_learning.gepa.mutation.llm_provider\n"
                            "  Required when prompt_learning.gepa.mutation.llm_model is set"
                        )
                    else:
                        errors.extend(
                            _validate_model_for_provider(
                                mutation_model,
                                mutation_provider,
                                "prompt_learning.gepa.mutation.llm_model",
                                allow_nano=False,
                            )
                        )

            # Top-level mutation_rate and crossover_rate (if not in nested sections)
            if not (
                mutation_config and isinstance(mutation_config, dict) and "rate" in mutation_config
            ):
                _rate_float("mutation_rate")
            population_config = gepa_config.get("population")
            if not (
                population_config
                and isinstance(population_config, dict)
                and "crossover_rate" in population_config
            ):
                _rate_float("crossover_rate")

            # Budget cap
            max_spend = gepa_config.get("max_spend_usd")
            if max_spend is not None:
                try:
                    f = float(max_spend)
                    if f <= 0:
                        errors.append(
                            "prompt_learning.gepa.max_spend_usd must be > 0 when provided"
                        )
                except (ValueError, TypeError):
                    errors.append("prompt_learning.gepa.max_spend_usd must be numeric")

            # Rollout budget validation
            rollout_config = gepa_config.get("rollout")
            rollout_budget = None
            if rollout_config and isinstance(rollout_config, dict):
                rollout_budget = rollout_config.get("budget")
            if rollout_budget is None:
                rollout_budget = gepa_config.get("rollout_budget")
            if rollout_budget is not None:
                try:
                    rb = int(rollout_budget)
                    if rb <= 0:
                        errors.append(
                            "prompt_learning.gepa.rollout.budget (or rollout_budget) must be > 0 when provided"
                        )
                except Exception:
                    errors.append(
                        "prompt_learning.gepa.rollout.budget (or rollout_budget) must be an integer"
                    )

            # Minibatch size validation
            minibatch_size = None
            if rollout_config and isinstance(rollout_config, dict):
                minibatch_size = rollout_config.get("minibatch_size")
            if minibatch_size is None:
                minibatch_size = gepa_config.get("minibatch_size")
            if minibatch_size is not None:
                try:
                    mbs = int(minibatch_size)
                    if mbs <= 0:
                        errors.append(
                            "prompt_learning.gepa.rollout.minibatch_size (or minibatch_size) must be > 0"
                        )
                except Exception:
                    errors.append(
                        "prompt_learning.gepa.rollout.minibatch_size (or minibatch_size) must be an integer"
                    )

            # Proposer type validation
            proposer_type = gepa_config.get("proposer_type", "dspy")
            if proposer_type not in ("dspy", "spec", "synth", "gepa-ai"):
                errors.append(
                    f"Invalid proposer_type: '{proposer_type}'\n"
                    f"  Must be one of: 'dspy', 'spec', 'synth', 'gepa-ai'\n"
                    f"  Got: '{proposer_type}'"
                )

            # Proposer effort validation
            proposer_effort = str(gepa_config.get("proposer_effort", "LOW")).upper()
            valid_effort_levels = {"LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"}
            if proposer_effort not in valid_effort_levels:
                errors.append(
                    f"Invalid proposer_effort: '{proposer_effort}'\n"
                    f"  Must be one of: {', '.join(sorted(valid_effort_levels))}\n"
                    f"  Got: '{proposer_effort}'"
                )

            # Proposer output tokens validation
            proposer_output_tokens = str(gepa_config.get("proposer_output_tokens", "FAST")).upper()
            valid_output_tokens = {"RAPID", "FAST", "SLOW"}
            if proposer_output_tokens not in valid_output_tokens:
                errors.append(
                    f"Invalid proposer_output_tokens: '{proposer_output_tokens}'\n"
                    f"  Must be one of: {', '.join(sorted(valid_output_tokens))}\n"
                    f"  Got: '{proposer_output_tokens}'"
                )

            # Note: RAPID can now be used with any proposer_effort level (5000 tokens)

            # Spec validation when proposer_type is "spec"
            if proposer_type == "spec":
                spec_path = gepa_config.get("spec_path")
                if not spec_path:
                    errors.append(
                        "Missing required field: prompt_learning.gepa.spec_path\n"
                        "  Required when proposer_type='spec'\n"
                        "  Example:\n"
                        "    [prompt_learning.gepa]\n"
                        '    proposer_type = "spec"\n'
                        '    spec_path = "examples/task_apps/banking77/banking77_spec.json"'
                    )
                else:
                    # Validate spec_max_tokens if provided
                    spec_max_tokens = gepa_config.get("spec_max_tokens")
                    if spec_max_tokens is not None:
                        try:
                            smt = int(spec_max_tokens)
                            if smt <= 0:
                                errors.append("prompt_learning.gepa.spec_max_tokens must be > 0")
                        except Exception:
                            errors.append("prompt_learning.gepa.spec_max_tokens must be an integer")

                    # Validate spec_priority_threshold if provided
                    spec_priority_threshold = gepa_config.get("spec_priority_threshold")
                    if spec_priority_threshold is not None:
                        try:
                            spt = int(spec_priority_threshold)
                            if spt < 0:
                                errors.append(
                                    "prompt_learning.gepa.spec_priority_threshold must be >= 0"
                                )
                        except Exception:
                            errors.append(
                                "prompt_learning.gepa.spec_priority_threshold must be an integer"
                            )

            # Archive size validation
            archive_config = gepa_config.get("archive")
            archive_size = None
            if archive_config and isinstance(archive_config, dict):
                archive_size = archive_config.get("size")
            if archive_size is None:
                archive_size = gepa_config.get("archive_size")
            if archive_size is not None:
                try:
                    asize = int(archive_size)
                    if asize <= 0:
                        errors.append(
                            "prompt_learning.gepa.archive.size (or archive_size) must be > 0"
                        )
                except Exception:
                    errors.append(
                        "prompt_learning.gepa.archive.size (or archive_size) must be an integer"
                    )

            # CRITICAL: Validate pareto_set_size vs seeds BEFORE submitting to backend
            # This catches config errors immediately instead of after job submission
            eval_config = gepa_config.get("evaluation")
            if eval_config and isinstance(eval_config, dict):
                train_seeds = eval_config.get("seeds") or eval_config.get("train_seeds")
                if train_seeds and isinstance(train_seeds, list) and len(train_seeds) > 0:
                    total_seeds = len(train_seeds)

                    # Get pareto_set_size (can be in archive section or top-level)
                    pareto_set_size = None
                    if archive_config and isinstance(archive_config, dict):
                        pareto_set_size = archive_config.get("pareto_set_size")
                    if pareto_set_size is None:
                        pareto_set_size = gepa_config.get(
                            "pareto_set_size", 64
                        )  # Default from backend

                    try:
                        pareto_count = int(pareto_set_size)
                        feedback_fraction = 0.5  # Default
                        if archive_config and isinstance(archive_config, dict):
                            feedback_fraction = archive_config.get("feedback_fraction", 0.5)
                        if feedback_fraction is None:
                            feedback_fraction = gepa_config.get("feedback_fraction", 0.5)
                        feedback_fraction = float(feedback_fraction)

                        # Calculate split
                        feedback_count = total_seeds - pareto_count

                        # Constants matching backend
                        min_pareto_set_size = 10
                        min_feedback_seeds = 3

                        # Validate pareto_set_size <= total_seeds
                        if pareto_count > total_seeds:
                            errors.append(
                                f"CONFIG ERROR: pareto_set_size={pareto_count} > total_seeds={total_seeds}. "
                                f"Increase [prompt_learning.gepa.evaluation].seeds or decrease "
                                f"[prompt_learning.gepa.archive].pareto_set_size. "
                                f"Seeds: {train_seeds[:10]}{'...' if len(train_seeds) > 10 else ''}"
                            )

                        # Validate pareto_set_size >= min_pareto_set_size
                        if pareto_count < min_pareto_set_size:
                            errors.append(
                                f"CONFIG ERROR: pareto_set_size={pareto_count} < MIN_PARETO_SET_SIZE={min_pareto_set_size}. "
                                f"Increase [prompt_learning.gepa.archive].pareto_set_size to at least {min_pareto_set_size}. "
                                f"Below this threshold, accuracy estimates are too noisy for reliable optimization."
                            )

                        # Validate feedback_count >= min_feedback_seeds
                        if feedback_count < min_feedback_seeds:
                            errors.append(
                                f"CONFIG ERROR: feedback_count={feedback_count} < MIN_FEEDBACK_SEEDS={min_feedback_seeds}. "
                                f"Increase total seeds or decrease pareto_set_size to ensure at least {min_feedback_seeds} feedback seeds. "
                                f"Below this threshold, reflection prompts lack sufficient diversity."
                            )
                    except (ValueError, TypeError):
                        pass  # Type errors already caught by _pos_int_nested above

            # Pareto eps validation
            pareto_eps = None
            if archive_config and isinstance(archive_config, dict):
                pareto_eps = archive_config.get("pareto_eps")
            if pareto_eps is None:
                pareto_eps = gepa_config.get("pareto_eps")
            if pareto_eps is not None:
                try:
                    pe = float(pareto_eps)
                    if pe <= 0:
                        errors.append(
                            "prompt_learning.gepa.archive.pareto_eps (or pareto_eps) must be > 0"
                        )
                    elif pe >= 1.0:
                        errors.append(
                            "prompt_learning.gepa.archive.pareto_eps (or pareto_eps) should be < 1.0 (typically 1e-6)"
                        )
                except Exception:
                    errors.append(
                        "prompt_learning.gepa.archive.pareto_eps (or pareto_eps) must be numeric"
                    )

            # Feedback fraction validation
            feedback_fraction = None
            if archive_config and isinstance(archive_config, dict):
                feedback_fraction = archive_config.get("feedback_fraction")
            if feedback_fraction is None:
                feedback_fraction = gepa_config.get("feedback_fraction")
            if feedback_fraction is not None:
                try:
                    ff = float(feedback_fraction)
                    if not (0.0 <= ff <= 1.0):
                        errors.append(
                            "prompt_learning.gepa.archive.feedback_fraction (or feedback_fraction) must be between 0.0 and 1.0"
                        )
                except Exception:
                    errors.append(
                        "prompt_learning.gepa.archive.feedback_fraction (or feedback_fraction) must be numeric"
                    )

            # Token counting model validation (should be a valid model name)
            token_config = gepa_config.get("token") or gepa_config.get("prompt_budget")
            token_counting_model = None
            if token_config and isinstance(token_config, dict):
                token_counting_model = token_config.get("counting_model")
            if token_counting_model is None:
                token_counting_model = gepa_config.get("token_counting_model")
            if token_counting_model and (
                not isinstance(token_counting_model, str) or not token_counting_model.strip()
            ):
                # Basic validation - should be a non-empty string
                errors.append(
                    "prompt_learning.gepa.token.counting_model (or prompt_budget.counting_model, token_counting_model) must be a non-empty string"
                )

            # Module/stage validation for multi-stage
            if has_multi_stage:
                modules_config = gepa_config.get("modules")
                if modules_config and isinstance(modules_config, list):
                    for idx, module_entry in enumerate(modules_config):
                        if isinstance(module_entry, dict):
                            module_id = (
                                module_entry.get("module_id")
                                or module_entry.get("stage_id")
                                or f"module_{idx}"
                            )
                            max_instruction_slots = module_entry.get("max_instruction_slots")
                            max_tokens = module_entry.get("max_tokens")
                            allowed_tools = module_entry.get("allowed_tools")

                            # Validate max_instruction_slots
                            if max_instruction_slots is not None:
                                try:
                                    mis = int(max_instruction_slots)
                                    if mis < 1:
                                        errors.append(
                                            f"prompt_learning.gepa.modules[{idx}].max_instruction_slots must be >= 1"
                                        )
                                except Exception:
                                    errors.append(
                                        f"prompt_learning.gepa.modules[{idx}].max_instruction_slots must be an integer"
                                    )

                            # Validate max_tokens
                            if max_tokens is not None:
                                try:
                                    mt = int(max_tokens)
                                    if mt <= 0:
                                        errors.append(
                                            f"prompt_learning.gepa.modules[{idx}].max_tokens must be > 0"
                                        )
                                except Exception:
                                    errors.append(
                                        f"prompt_learning.gepa.modules[{idx}].max_tokens must be an integer"
                                    )

                            # Validate allowed_tools
                            if allowed_tools is not None:
                                if not isinstance(allowed_tools, list):
                                    errors.append(
                                        f"prompt_learning.gepa.modules[{idx}].allowed_tools must be a list"
                                    )
                                else:
                                    if len(allowed_tools) == 0:
                                        errors.append(
                                            f"prompt_learning.gepa.modules[{idx}].allowed_tools cannot be empty (use null/omit to allow all tools)"
                                        )
                                    else:
                                        # Check for duplicates
                                        seen_tools = set()
                                        for tool_idx, tool in enumerate(allowed_tools):
                                            if not isinstance(tool, str):
                                                errors.append(
                                                    f"prompt_learning.gepa.modules[{idx}].allowed_tools[{tool_idx}] must be a string"
                                                )
                                            elif not tool.strip():
                                                errors.append(
                                                    f"prompt_learning.gepa.modules[{idx}].allowed_tools[{tool_idx}] cannot be empty"
                                                )
                                            elif tool.strip() in seen_tools:
                                                errors.append(
                                                    f"prompt_learning.gepa.modules[{idx}].allowed_tools contains duplicate '{tool.strip()}'"
                                                )
                                            else:
                                                seen_tools.add(tool.strip())

                            # Validate per-module policy config (REQUIRED)
                            module_policy = module_entry.get("policy")
                            if module_policy is None:
                                errors.append(
                                    f"❌ gepa.modules[{idx}]: [policy] table is REQUIRED. "
                                    f"Each module must have its own policy configuration with 'model' and 'provider' fields."
                                )
                            elif not isinstance(module_policy, dict):
                                errors.append(
                                    f"❌ gepa.modules[{idx}]: [policy] must be a table/dict, got {type(module_policy).__name__}"
                                )
                            else:
                                # Validate required fields in module policy
                                # Note: model is optional - detected from actual LLM calls during rollouts
                                if not module_policy.get("provider"):
                                    errors.append(
                                        f"❌ gepa.modules[{idx}]: [policy].provider is required"
                                    )
                                # Validate model/provider combination
                                module_model = module_policy.get("model")
                                module_provider = module_policy.get("provider")
                                if module_model and module_provider:
                                    errors.extend(
                                        _validate_model_for_provider(
                                            module_model,
                                            module_provider,
                                            f"prompt_learning.gepa.modules[{idx}].policy.model",
                                            allow_nano=True,  # Policy models can be nano
                                        )
                                    )
                                # Reject inference_url in module policy (trainer provides it)
                                if "inference_url" in module_policy:
                                    errors.append(
                                        f"❌ gepa.modules[{idx}]: [policy].inference_url must not be specified. "
                                        f"The trainer provides the inference URL in rollout requests. Remove inference_url from module policy."
                                    )
                                if "api_base" in module_policy:
                                    errors.append(
                                        f"❌ gepa.modules[{idx}]: [policy].api_base must not be specified. "
                                        f"Remove api_base from module policy."
                                    )
                                if "base_url" in module_policy:
                                    errors.append(
                                        f"❌ gepa.modules[{idx}]: [policy].base_url must not be specified. "
                                        f"Remove base_url from module policy."
                                    )

    # Raise all errors at once for better UX
    if errors:
        _raise_validation_errors(errors, config_path)


def _raise_validation_errors(errors: list[str], config_path: Path) -> None:
    """Format and raise validation errors."""
    error_msg = (
        f"\n❌ Invalid prompt learning config: {config_path}\n\nFound {len(errors)} error(s):\n\n"
    )

    for i, error in enumerate(errors, 1):
        # Indent multi-line errors
        indented_error = "\n  ".join(error.split("\n"))
        error_msg += f"{i}. {indented_error}\n\n"

        error_msg += (
            "📖 See example configs:\n"
            "  - cookbooks/dev/blog_posts/gepa/configs/banking77_gepa_local.toml\n"
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
        errors.append("Missing [rl] or [online_rl] section in config")
        _raise_validation_errors(errors, config_path)
        return

    # Validate algorithm
    algorithm = rl_section.get("algorithm")
    if not algorithm:
        errors.append("Missing required field: rl.algorithm\n  Must be one of: 'grpo', 'ppo', etc.")

    # Validate task_url
    task_url = rl_section.get("task_url")
    if not task_url:
        errors.append("Missing required field: rl.task_url")
    elif not isinstance(task_url, str):
        errors.append(f"task_url must be a string, got {type(task_url).__name__}")

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
        errors.append("Missing [sft] section in config")
        _raise_validation_errors(errors, config_path)
        return

    # Validate model
    model = sft_section.get("model")
    if not model:
        errors.append("Missing required field: sft.model")

    if errors:
        _raise_validation_errors(errors, config_path)


def validate_gepa_config_from_file(config_path: Path) -> Tuple[bool, List[str]]:
    """Validate GEPA config from TOML file with comprehensive checks.

    Returns:
        (is_valid, errors) tuple where errors is a list of error messages
    """
    errors = []

    try:
        with open(config_path) as f:
            config_dict = toml.load(f)
    except Exception as e:
        return False, [f"Failed to parse TOML: {e}"]

    pl_section = config_dict.get("prompt_learning", {})
    if not isinstance(pl_section, dict):
        errors.append("❌ [prompt_learning] section is missing or invalid")
        return False, errors

    # Check algorithm
    algorithm = pl_section.get("algorithm")
    if algorithm != "gepa":
        errors.append(f"❌ Expected algorithm='gepa', got '{algorithm}'")

    # Check required top-level fields (env_name is now in gepa section)
    required_top_level = ["task_app_url", "task_app_api_key"]
    for field in required_top_level:
        if not pl_section.get(field):
            errors.append(f"❌ [prompt_learning].{field} is required")

    # Check GEPA section
    gepa_section = pl_section.get("gepa", {})
    if not isinstance(gepa_section, dict):
        errors.append("❌ [prompt_learning.gepa] section is missing or invalid")
        return False, errors

    # Check env_name in gepa section (required)
    if not gepa_section.get("env_name"):
        errors.append("❌ [prompt_learning.gepa].env_name is required")

    # Check required GEPA subsections
    required_sections = ["evaluation", "rollout", "mutation", "population", "archive"]
    missing_sections = [s for s in required_sections if not gepa_section.get(s)]
    if missing_sections:
        errors.append(
            f"❌ Missing required GEPA sections: {', '.join(f'[prompt_learning.gepa.{s}]' for s in missing_sections)}"
        )

    # Validate evaluation section
    eval_section = gepa_section.get("evaluation", {})
    if isinstance(eval_section, dict):
        # Check train_seeds (required, can be in eval section or top-level)
        train_seeds = (
            eval_section.get("train_seeds")
            or eval_section.get("seeds")
            or pl_section.get("train_seeds")
        )
        if not train_seeds:
            errors.append(
                "❌ train_seeds is required. "
                "Must be in [prompt_learning.gepa.evaluation].train_seeds or [prompt_learning].train_seeds"
            )
        elif not isinstance(train_seeds, list):
            errors.append(f"❌ train_seeds must be a list, got {type(train_seeds).__name__}")
        elif len(train_seeds) == 0:
            errors.append("❌ train_seeds cannot be empty")
        elif not all(isinstance(s, int) for s in train_seeds):
            errors.append("❌ train_seeds must contain only integers")

        # Check val_seeds (required)
        val_seeds = eval_section.get("val_seeds") or eval_section.get("validation_seeds")
        if not val_seeds:
            errors.append("❌ val_seeds is required in [prompt_learning.gepa.evaluation].val_seeds")
        elif not isinstance(val_seeds, list):
            errors.append(f"❌ val_seeds must be a list, got {type(val_seeds).__name__}")
        elif len(val_seeds) == 0:
            errors.append("❌ val_seeds cannot be empty")
        elif not all(isinstance(s, int) for s in val_seeds):
            errors.append("❌ val_seeds must contain only integers")

        # Check validation_pool (optional but should be valid if present)
        validation_pool = eval_section.get("validation_pool")
        if validation_pool is not None:
            if not isinstance(validation_pool, str):
                errors.append(
                    f"❌ validation_pool must be a string, got {type(validation_pool).__name__}"
                )
            elif validation_pool not in ("train", "test", "val", "validation"):
                errors.append(
                    f"❌ validation_pool must be one of: train, test, val, validation. Got '{validation_pool}'"
                )

        # Check validation_top_k (optional but should be valid if present)
        validation_top_k = eval_section.get("validation_top_k")
        if validation_top_k is not None:
            if not isinstance(validation_top_k, int):
                errors.append(
                    f"❌ validation_top_k must be an integer, got {type(validation_top_k).__name__}"
                )
            elif validation_top_k <= 0:
                errors.append(f"❌ validation_top_k must be > 0, got {validation_top_k}")

    # Validate rollout section
    rollout_section = gepa_section.get("rollout", {})
    if isinstance(rollout_section, dict):
        budget = rollout_section.get("budget")
        if budget is None:
            errors.append("❌ [prompt_learning.gepa.rollout].budget is required")
        elif not isinstance(budget, int):
            errors.append(f"❌ rollout.budget must be an integer, got {type(budget).__name__}")
        elif budget <= 0:
            errors.append(f"❌ rollout.budget must be > 0, got {budget}")

        max_concurrent = rollout_section.get("max_concurrent")
        if max_concurrent is not None:
            if not isinstance(max_concurrent, int):
                errors.append(
                    f"❌ rollout.max_concurrent must be an integer, got {type(max_concurrent).__name__}"
                )
            elif max_concurrent <= 0:
                errors.append(f"❌ rollout.max_concurrent must be > 0, got {max_concurrent}")

    # Validate mutation section
    mutation_section = gepa_section.get("mutation", {})
    if isinstance(mutation_section, dict):
        required_mutation_fields = ["llm_model", "llm_provider"]
        for field in required_mutation_fields:
            if not mutation_section.get(field):
                errors.append(f"❌ [prompt_learning.gepa.mutation].{field} is required")

        rate = mutation_section.get("rate")
        if rate is not None:
            if not isinstance(rate, int | float):
                errors.append(f"❌ mutation.rate must be a number, got {type(rate).__name__}")
            elif not (0.0 <= rate <= 1.0):
                errors.append(f"❌ mutation.rate must be between 0.0 and 1.0, got {rate}")

    # Validate population section
    population_section = gepa_section.get("population", {})
    if isinstance(population_section, dict):
        initial_size = population_section.get("initial_size")
        if initial_size is not None:
            if not isinstance(initial_size, int):
                errors.append(
                    f"❌ population.initial_size must be an integer, got {type(initial_size).__name__}"
                )
            elif initial_size <= 0:
                errors.append(f"❌ population.initial_size must be > 0, got {initial_size}")

        num_generations = population_section.get("num_generations")
        if num_generations is not None:
            if not isinstance(num_generations, int):
                errors.append(
                    f"❌ population.num_generations must be an integer, got {type(num_generations).__name__}"
                )
            elif num_generations <= 0:
                errors.append(f"❌ population.num_generations must be > 0, got {num_generations}")

    # Validate archive section
    archive_section = gepa_section.get("archive", {})
    if isinstance(archive_section, dict):
        max_size = archive_section.get("max_size")
        if max_size is not None:
            if not isinstance(max_size, int):
                errors.append(
                    f"❌ archive.max_size must be an integer, got {type(max_size).__name__}"
                )
            elif max_size < 0:
                errors.append(f"❌ archive.max_size must be >= 0, got {max_size}")

    # Validate token section
    proposed_prompt_max_tokens = gepa_section.get("proposed_prompt_max_tokens")
    if proposed_prompt_max_tokens is not None:
        if not isinstance(proposed_prompt_max_tokens, int):
            errors.append(
                f"❌ proposed_prompt_max_tokens must be an integer, got {type(proposed_prompt_max_tokens).__name__}"
            )
        elif proposed_prompt_max_tokens <= 0:
            errors.append(
                f"❌ proposed_prompt_max_tokens must be > 0, got {proposed_prompt_max_tokens}"
            )

    token_section = gepa_section.get("token") or gepa_section.get("prompt_budget") or {}
    if isinstance(token_section, dict):
        max_limit = token_section.get("max_limit")
        if max_limit is not None:
            if not isinstance(max_limit, int):
                errors.append(
                    f"❌ token.max_limit must be an integer, got {type(max_limit).__name__}"
                )
            elif max_limit <= 0:
                errors.append(f"❌ token.max_limit must be > 0, got {max_limit}")

    # Check initial_prompt section
    initial_prompt = pl_section.get("initial_prompt", {})
    if not isinstance(initial_prompt, dict):
        errors.append("❌ [prompt_learning.initial_prompt] section is missing or invalid")
    else:
        if not initial_prompt.get("id"):
            errors.append("❌ [prompt_learning.initial_prompt].id is required")
        if not initial_prompt.get("messages"):
            errors.append(
                "❌ [prompt_learning.initial_prompt].messages is required (must be a list)"
            )
        elif not isinstance(initial_prompt.get("messages"), list):
            errors.append("❌ [prompt_learning.initial_prompt].messages must be a list")
        elif len(initial_prompt.get("messages", [])) == 0:
            errors.append("❌ [prompt_learning.initial_prompt].messages cannot be empty")

    # Check policy section
    policy_section = pl_section.get("policy", {})
    if not isinstance(policy_section, dict):
        errors.append("❌ [prompt_learning.policy] section is missing or invalid")
    else:
        # Validate policy section - reject inference_url (backend requirement)
        if "inference_url" in policy_section:
            errors.append(
                "❌ inference_url must not be specified in [prompt_learning.policy]. "
                "The trainer provides the inference URL in rollout requests. "
                "Remove inference_url from your config file."
            )
        if "api_base" in policy_section:
            errors.append(
                "❌ api_base must not be specified in [prompt_learning.policy]. "
                "The trainer provides the inference URL in rollout requests. "
                "Remove api_base from your config file."
            )
        if "base_url" in policy_section:
            errors.append(
                "❌ base_url must not be specified in [prompt_learning.policy]. "
                "The trainer provides the inference URL in rollout requests. "
                "Remove base_url from your config file."
            )

        # Note: model is optional - detected from actual LLM calls during rollouts
        if not policy_section.get("provider"):
            errors.append("❌ [prompt_learning.policy].provider is required")

    # Validate proxy_models section (can be at top-level or gepa-specific)
    proxy_models_section = pl_section.get("proxy_models") or gepa_section.get("proxy_models")
    if proxy_models_section:
        if not isinstance(proxy_models_section, dict):
            errors.append("❌ proxy_models must be a table/dict when provided")
        else:
            required_fields = ["hi_provider", "hi_model", "lo_provider", "lo_model"]
            for field in required_fields:
                if not proxy_models_section.get(field):
                    errors.append(f"❌ proxy_models.{field} is required")
            # Validate numeric fields
            for field, min_val in [
                ("n_min_hi", 0),
                ("r2_thresh", 0.0),
                ("r2_stop", 0.0),
                ("sigma_max", 0.0),
                ("sigma_stop", 0.0),
                ("verify_every", 0),
            ]:
                val = proxy_models_section.get(field)
                if val is not None:
                    try:
                        if field in ("r2_thresh", "r2_stop"):
                            fval = float(val)
                            if not (0.0 <= fval <= 1.0):
                                errors.append(
                                    f"❌ proxy_models.{field} must be between 0.0 and 1.0, got {fval}"
                                )
                        elif field.startswith("sigma"):
                            fval = float(val)
                            if fval < min_val:
                                errors.append(
                                    f"❌ proxy_models.{field} must be >= {min_val}, got {fval}"
                                )
                        else:
                            ival = int(val)
                            if ival < min_val:
                                errors.append(
                                    f"❌ proxy_models.{field} must be >= {min_val}, got {ival}"
                                )
                    except (TypeError, ValueError):
                        errors.append(
                            f"❌ proxy_models.{field} must be numeric, got {type(val).__name__}"
                        )
            # Validate provider/model combinations
            if proxy_models_section.get("hi_provider") and proxy_models_section.get("hi_model"):
                hi_errors = _validate_model_for_provider(
                    proxy_models_section["hi_model"],
                    proxy_models_section["hi_provider"],
                    "proxy_models.hi_model",
                    allow_nano=True,
                )
                errors.extend(hi_errors)
            if proxy_models_section.get("lo_provider") and proxy_models_section.get("lo_model"):
                lo_errors = _validate_model_for_provider(
                    proxy_models_section["lo_model"],
                    proxy_models_section["lo_provider"],
                    "proxy_models.lo_model",
                    allow_nano=True,
                )
                errors.extend(lo_errors)

    # Validate adaptive_pool section (GEPA-specific)
    adaptive_pool_section = gepa_section.get("adaptive_pool")
    if adaptive_pool_section:
        _validate_adaptive_pool_config(adaptive_pool_section, "gepa.adaptive_pool", errors)

    # Validate adaptive_batch section (GEPA-specific)
    adaptive_batch_section = gepa_section.get("adaptive_batch")
    if adaptive_batch_section:
        if not isinstance(adaptive_batch_section, dict):
            errors.append("❌ gepa.adaptive_batch must be a table/dict when provided")
        else:
            level = adaptive_batch_section.get("level")
            if level is not None:
                valid_levels = {"NONE", "LOW", "MODERATE", "HIGH"}
                if str(level).upper() not in valid_levels:
                    errors.append(
                        f"❌ gepa.adaptive_batch.level must be one of {valid_levels}, got '{level}'"
                    )
            # Validate numeric fields
            for field, min_val in [
                ("reflection_minibatch_size", 1),
                ("val_subsample_size", 1),
            ]:
                val = adaptive_batch_section.get(field)
                if val is not None:
                    try:
                        ival = int(val)
                        if ival < min_val:
                            errors.append(
                                f"❌ gepa.adaptive_batch.{field} must be >= {min_val}, got {ival}"
                            )
                    except (TypeError, ValueError):
                        errors.append(
                            f"❌ gepa.adaptive_batch.{field} must be an integer, got {type(val).__name__}"
                        )
            # Validate min_local_improvement
            min_improvement = adaptive_batch_section.get("min_local_improvement")
            if min_improvement is not None:
                try:
                    float(min_improvement)  # Just validate it's numeric
                except (TypeError, ValueError):
                    errors.append(
                        f"❌ gepa.adaptive_batch.min_local_improvement must be numeric, got {type(min_improvement).__name__}"
                    )
            # Validate val_evaluation_mode
            val_mode = adaptive_batch_section.get("val_evaluation_mode")
            if val_mode is not None and val_mode not in ("full", "subsample"):
                errors.append(
                    f"❌ gepa.adaptive_batch.val_evaluation_mode must be 'full' or 'subsample', got '{val_mode}'"
                )
            # Validate candidate_selection_strategy
            selection_strategy = adaptive_batch_section.get("candidate_selection_strategy")
            if selection_strategy is not None and selection_strategy not in ("coverage", "random"):
                errors.append(
                    f"❌ gepa.adaptive_batch.candidate_selection_strategy must be 'coverage' or 'random', got '{selection_strategy}'"
                )
            # Validate val_evaluation_mode="subsample" requires val_subsample_size > 0
            val_mode = adaptive_batch_section.get("val_evaluation_mode")
            if val_mode == "subsample":
                subsample_size = adaptive_batch_section.get("val_subsample_size")
                if subsample_size is None:
                    errors.append(
                        "❌ gepa.adaptive_batch.val_evaluation_mode='subsample' requires val_subsample_size to be set"
                    )
                elif isinstance(subsample_size, int | float) and subsample_size <= 0:
                    errors.append(
                        f"❌ gepa.adaptive_batch.val_subsample_size must be > 0 when val_evaluation_mode='subsample', got {subsample_size}"
                    )

    return len(errors) == 0, errors


def validate_prompt_learning_config_from_file(config_path: Path, algorithm: str) -> None:
    """Validate prompt learning config from TOML file and raise ConfigValidationError if invalid.

    Args:
        config_path: Path to TOML config file
        algorithm: Must be 'gepa'

    Raises:
        ConfigValidationError: If validation fails, with detailed error messages
    """
    ctx: dict[str, Any] = {"config_path": str(config_path), "algorithm": algorithm}
    log_info("validate_prompt_learning_config_from_file invoked", ctx=ctx)
    if algorithm == "gepa":
        is_valid, errors = validate_gepa_config_from_file(config_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'gepa'")

    if not is_valid:
        error_msg = "\n".join(errors)
        raise ConfigValidationError(
            f"\n{'=' * 80}\n"
            f"❌ Config Validation Failed ({algorithm.upper()})\n"
            f"{'=' * 80}\n"
            f"{error_msg}\n"
            f"{'=' * 80}\n"
        )


__all__ = [
    "ConfigValidationError",
    "validate_prompt_learning_config",
    "validate_prompt_learning_config_from_file",
    "validate_gepa_config_from_file",
    "validate_rl_config",
    "validate_sft_config",
]
