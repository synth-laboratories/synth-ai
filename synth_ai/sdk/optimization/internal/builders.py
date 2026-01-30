import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
from pydantic import ValidationError

from synth_ai.sdk.shared.models import normalize_model_identifier


def ensure_allowed_model(model: str) -> None:
    """Validate that a model identifier is allowed.

    Args:
        model: Model identifier to validate

    Raises:
        UnsupportedModelError: If the model is not valid
    """
    normalize_model_identifier(model)  # Will raise if invalid


# SFT module import is optional (moved to research repo)
try:
    _sft_module = cast(Any, importlib.import_module("synth_ai.sdk.learning.sft.config"))
    prepare_sft_job_payload = cast(
        Callable[..., dict[str, Any]], _sft_module.prepare_sft_job_payload
    )
    _SFT_AVAILABLE = True
except Exception:  # pragma: no cover - SFT moved to research repo
    prepare_sft_job_payload = None  # type: ignore[assignment]
    _SFT_AVAILABLE = False

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.builders.") from exc

from synth_ai.core.config.resolver import ConfigResolver  # noqa: E402
from synth_ai.core.utils.urls import is_synthtunnel_url  # noqa: E402
from synth_ai.sdk.localapi.auth import ensure_localapi_auth  # noqa: E402

from .configs import PromptLearningConfig  # noqa: E402
from .utils import ensure_api_base  # noqa: E402


@dataclass(slots=True)
class PromptLearningBuildResult:
    payload: dict[str, Any]
    task_url: str


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        msg = error.get("msg", "invalid value")
        lines.append(f"{loc or '<root>'}: {msg}")
    details = "\n".join(f"  - {line}" for line in lines) or "  - Invalid configuration"
    return f"Config validation failed ({path}):\n{details}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize_mipro_section(
    pl_cfg: PromptLearningConfig,
    config_dict: dict[str, Any],
    *,
    source: str,
    prefer_model: bool,
) -> None:
    _require(
        pl_cfg.mipro is not None,
        f"{source}: MIPRO config missing [prompt_learning.mipro] section.",
    )

    pl_section = config_dict.get("prompt_learning")
    if not isinstance(pl_section, dict):
        pl_section = {}
        config_dict["prompt_learning"] = pl_section

    mipro_section = pl_section.get("mipro", {})
    if hasattr(mipro_section, "model_dump"):
        mipro_section = mipro_section.model_dump(mode="python")
    elif hasattr(mipro_section, "dict"):
        mipro_section = mipro_section.dict()
    if not isinstance(mipro_section, dict):
        mipro_section = {}

    bootstrap_from_model = getattr(pl_cfg.mipro, "bootstrap_train_seeds", None)
    online_from_model = getattr(pl_cfg.mipro, "online_pool", None)
    test_from_model = getattr(pl_cfg.mipro, "test_pool", None)
    reference_from_model = getattr(pl_cfg.mipro, "reference_pool", None)

    def _maybe_set(key: str, value: Any) -> None:
        if value is None:
            return
        if prefer_model or mipro_section.get(key) is None:
            mipro_section[key] = value

    _maybe_set("bootstrap_train_seeds", bootstrap_from_model)
    _maybe_set("online_pool", online_from_model)
    _maybe_set("test_pool", test_from_model)
    _maybe_set("reference_pool", reference_from_model)

    for key in ("bootstrap_train_seeds", "online_pool", "test_pool", "reference_pool"):
        if mipro_section.get(key) is None and pl_section.get(key) is not None:
            mipro_section[key] = pl_section[key]

    mipro_env_name = mipro_section.get("env_name")
    if mipro_env_name and not pl_section.get("env_name") and not pl_section.get("task_app_id"):
        pl_section["env_name"] = mipro_env_name

    pl_section["mipro"] = mipro_section
    config_dict["prompt_learning"] = pl_section

    _require(
        mipro_section.get("bootstrap_train_seeds") is not None,
        f"{source}: bootstrap_train_seeds missing for MIPRO config.",
    )
    _require(
        mipro_section.get("online_pool") is not None,
        f"{source}: online_pool missing for MIPRO config.",
    )


def build_prompt_learning_payload(
    *,
    config_path: Path,
    task_url: str | None,
    overrides: dict[str, Any],
    allow_experimental: bool | None = None,
) -> PromptLearningBuildResult:
    """Build payload for prompt learning job (MIPRO or GEPA)."""
    from pydantic import ValidationError

    from .configs.prompt_learning import load_toml

    # SDK-SIDE VALIDATION: Catch errors BEFORE sending to backend
    from .validators import validate_prompt_learning_config

    raw_config = load_toml(config_path)
    validate_prompt_learning_config(raw_config, config_path)

    try:
        pl_cfg = PromptLearningConfig.from_path(config_path)
    except ValidationError as exc:
        raise click.ClickException(_format_validation_error(config_path, exc)) from exc

    # Early validation: Check required fields for GEPA
    if pl_cfg.algorithm == "gepa":
        if not pl_cfg.gepa:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa] section is required"
            )
        if not pl_cfg.gepa.evaluation:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa.evaluation] section is required"
            )
        train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "seeds", None
        )
        if not train_seeds:
            raise click.ClickException(
                "GEPA config missing train_seeds: [prompt_learning.gepa.evaluation] must have 'train_seeds' or 'seeds' field"
            )
        val_seeds = getattr(pl_cfg.gepa.evaluation, "val_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "validation_seeds", None
        )
        if not val_seeds:
            raise click.ClickException(
                "GEPA config missing val_seeds: [prompt_learning.gepa.evaluation] must have 'val_seeds' or 'validation_seeds' field"
            )

    candidate_task_url = (
        (overrides.get("task_url") or task_url or "").strip()
        or (pl_cfg.task_app_url or "").strip()
        or (os.environ.get("TASK_APP_URL") or "").strip()
    )
    env_api_key: str | None = None
    if not (candidate_task_url and is_synthtunnel_url(candidate_task_url)):
        env_api_key = ensure_localapi_auth()

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()

    if synth_ai_py is None or not hasattr(synth_ai_py, "build_prompt_learning_payload"):
        raise click.ClickException(
            "Rust core payload builder unavailable. synth_ai_py is required; no Python fallback."
        )
    try:
        payload, resolved_task_url = synth_ai_py.build_prompt_learning_payload(
            config_dict, task_url, overrides
        )
        return PromptLearningBuildResult(payload=payload, task_url=resolved_task_url)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    cli_task_url = overrides.get("task_url") or task_url
    env_task_url = os.environ.get("TASK_APP_URL")
    config_task_url = (pl_cfg.task_app_url or "").strip() or None

    # For prompt learning, prefer config value over env if config is explicitly set
    # This allows TOML files to specify task_app_url without env var interference
    # But CLI override always wins
    if cli_task_url:
        # CLI override takes precedence
        final_task_url = ConfigResolver.resolve(
            "task_app_url",
            cli_value=cli_task_url,
            env_value=None,  # Don't check env when CLI is set
            config_value=config_task_url,
            required=True,
        )
    elif config_task_url:
        # Config explicitly set - use it (ignore env var to avoid conflicts)
        final_task_url = config_task_url
    else:
        # No config, fall back to env or error
        final_task_url = ConfigResolver.resolve(
            "task_app_url",
            cli_value=None,
            env_value=env_task_url,
            config_value=None,
            required=True,
        )
    _require(final_task_url is not None, "task_app_url is required")

    # Get task_app_api_key from config or environment
    # Note: task_app_api_key is not a field on PromptLearningConfig, use getattr
    config_api_key = (getattr(pl_cfg, "task_app_api_key", None) or "").strip() or None
    cli_api_key = overrides.get("task_app_api_key")
    skip_task_app_key = os.environ.get(
        "SYNTH_BACKEND_RESOLVES_TASK_APP_KEY", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _task_app_api_key = ConfigResolver.resolve(  # noqa: F841 (validation only)
        "task_app_api_key",
        cli_value=cli_api_key,
        env_value=env_api_key,
        config_value=config_api_key,
        required=not skip_task_app_key,
    )

    # Ensure task_app_url is set (task_app_api_key is resolved by backend from ENVIRONMENT_API_KEY)
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        pl_section["task_app_url"] = final_task_url
        if _task_app_api_key and not skip_task_app_key:
            pl_section["task_app_api_key"] = _task_app_api_key

        # GEPA: Extract train_seeds from nested structure for backwards compatibility
        # Backend checks for train_seeds at top level before parsing nested structure
        if pl_cfg.algorithm == "gepa" and pl_cfg.gepa:
            # Try to get train_seeds directly from the gepa config object first
            train_seeds = None
            if pl_cfg.gepa.evaluation:
                train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
                    pl_cfg.gepa.evaluation, "seeds", None
                )

            # If not found, try from serialized dict
            if not train_seeds:
                gepa_section = pl_section.get("gepa", {})
                # Handle case where gepa_section might still be a Pydantic model
                if hasattr(gepa_section, "model_dump"):
                    gepa_section = gepa_section.model_dump(mode="python")
                elif hasattr(gepa_section, "dict"):
                    gepa_section = gepa_section.dict()

                if isinstance(gepa_section, dict):
                    eval_section = gepa_section.get("evaluation", {})
                    # Handle case where eval_section might still be a Pydantic model
                    if hasattr(eval_section, "model_dump"):
                        eval_section = eval_section.model_dump(mode="python")
                    elif hasattr(eval_section, "dict"):
                        eval_section = eval_section.dict()

                    if isinstance(eval_section, dict):
                        train_seeds = eval_section.get("train_seeds") or eval_section.get("seeds")

                    # Update gepa_section back to pl_section in case we converted it
                    pl_section["gepa"] = gepa_section

            # Add train_seeds to top level for backwards compatibility
            if train_seeds and not pl_section.get("train_seeds"):
                pl_section["train_seeds"] = train_seeds
            if train_seeds and not pl_section.get("evaluation_seeds"):
                pl_section["evaluation_seeds"] = train_seeds

        if pl_cfg.algorithm == "mipro":
            _normalize_mipro_section(pl_cfg, config_dict, source="pre-merge", prefer_model=True)
    else:
        config_dict["prompt_learning"] = {
            "task_app_url": final_task_url,
        }

    # Build payload matching backend API format
    # Extract nested overrides if present, otherwise use flat overrides directly
    # The experiment queue passes flat overrides like {"prompt_learning.policy.model": "..."}
    # But some SDK code passes nested like {"overrides": {"prompt_learning.policy.model": "..."}}
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    # Remove non-override keys (backend, task_url, metadata, auto_start)
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k not in ("backend", "task_url", "metadata", "auto_start")
    }

    # CRITICAL: Merge overrides into config_dict BEFORE sending to backend
    # This ensures early validation in backend sees merged values
    # Use the same deep_update logic used throughout core utilities.
    if config_overrides:
        from synth_ai.core.utils.dict import deep_update as _deep_update

        _deep_update(config_dict, config_overrides)
    if pl_cfg.algorithm == "mipro":
        _normalize_mipro_section(pl_cfg, config_dict, source="post-merge", prefer_model=False)

    # ASSERT: Verify critical overrides are reflected in config_body
    pl_section_in_dict = config_dict.get("prompt_learning", {})
    if config_overrides:
        # Check rollout budget override
        rollout_budget_key = "prompt_learning.gepa.rollout.budget"
        if rollout_budget_key in config_overrides:
            expected_budget = config_overrides[rollout_budget_key]
            gepa_section = pl_section_in_dict.get("gepa", {})
            actual_budget = (
                gepa_section.get("rollout", {}).get("budget")
                if isinstance(gepa_section, dict)
                else None
            )
            if actual_budget is not None:
                _require(
                    actual_budget == expected_budget,
                    f"Rollout budget mismatch: config_body has {actual_budget} but override specifies {expected_budget}. "
                    "This indicates the override wasn't applied correctly.",
                )

        # Check model override
        model_key = "prompt_learning.policy.model"
        if model_key in config_overrides:
            expected_model = config_overrides[model_key]
            policy_section = pl_section_in_dict.get("policy", {})
            actual_model = policy_section.get("model") if isinstance(policy_section, dict) else None
            if actual_model is not None:
                _require(
                    actual_model == expected_model,
                    f"Model mismatch: config_body has {actual_model} but override specifies {expected_model}. "
                    "This indicates the override wasn't applied correctly.",
                )

        # Check provider override
        provider_key = "prompt_learning.policy.provider"
        if provider_key in config_overrides:
            expected_provider = config_overrides[provider_key]
            policy_section = pl_section_in_dict.get("policy", {})
            actual_provider = (
                policy_section.get("provider") if isinstance(policy_section, dict) else None
            )
            if actual_provider is not None:
                _require(
                    actual_provider == expected_provider,
                    f"Provider mismatch: config_body has {actual_provider} but override specifies {expected_provider}. "
                    "This indicates the override wasn't applied correctly.",
                )

    # FINAL CHECK: Ensure config_body has correct structure for backend
    # Backend expects: {"prompt_learning": {...}} (full TOML structure)
    if "prompt_learning" not in config_dict:
        raise ValueError(
            f"config_dict must have 'prompt_learning' key. Found keys: {list(config_dict.keys())}"
        )

    payload: dict[str, Any] = {
        "algorithm": pl_cfg.algorithm,
        "config_body": config_dict,
        "overrides": config_overrides,
        "metadata": overrides.get("metadata", {}),
        "auto_start": overrides.get("auto_start", True),
    }

    backend = overrides.get("backend")
    if backend:
        metadata_default: dict[str, Any] = {}
        metadata = cast(dict[str, Any], payload.setdefault("metadata", metadata_default))
        metadata["backend_base_url"] = ensure_api_base(str(backend))

    return PromptLearningBuildResult(payload=payload, task_url=final_task_url)


def build_prompt_learning_payload_from_mapping(
    *,
    raw_config: dict[str, Any],
    task_url: str | None,
    overrides: dict[str, Any],
    allow_experimental: bool | None = None,
    source_label: str = "programmatic",
) -> PromptLearningBuildResult:
    """Build payload for prompt learning job from a dictionary (programmatic use).

    This is the same as build_prompt_learning_payload but accepts a dict instead of a file path.
    Both functions route through the same PromptLearningConfig Pydantic validation.

    Args:
        raw_config: Configuration dictionary with the same structure as the TOML file.
                   Should have a 'prompt_learning' section.
        task_url: Override for task_app_url
        overrides: Config overrides (merged into config)
        allow_experimental: Allow experimental models
        source_label: Label for logging/error messages (default: "programmatic")

    Returns:
        PromptLearningBuildResult with payload and task_url

    Example:
        >>> result = build_prompt_learning_payload_from_mapping(
        ...     raw_config={
        ...         "prompt_learning": {
        ...             "algorithm": "gepa",
        ...             "task_app_url": "https://tunnel.example.com",
        ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
        ...             "gepa": {...},
        ...         }
        ...     },
        ...     task_url=None,
        ...     overrides={},
        ... )
    """
    from pydantic import ValidationError

    # SDK-SIDE VALIDATION: Catch errors BEFORE sending to backend
    from .validators import validate_prompt_learning_config

    # Use a pseudo-path for error messages (validator expects Path object)
    pseudo_path = Path(f"<{source_label}>")
    validate_prompt_learning_config(raw_config, pseudo_path)

    try:
        pl_cfg = PromptLearningConfig.from_mapping(raw_config)
    except ValidationError as exc:
        # Format validation errors for dict-based config
        lines: list[str] = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ()))
            msg = error.get("msg", "invalid value")
            lines.append(f"{loc or '<root>'}: {msg}")
        details = "\n".join(f"  - {line}" for line in lines) or "  - Invalid configuration"
        raise click.ClickException(
            f"Config validation failed ({source_label}):\n{details}"
        ) from exc

    # Early validation: Check required fields for GEPA
    if pl_cfg.algorithm == "gepa":
        if not pl_cfg.gepa:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa] section is required"
            )
        if not pl_cfg.gepa.evaluation:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa.evaluation] section is required"
            )
        train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "seeds", None
        )
        if not train_seeds:
            raise click.ClickException(
                "GEPA config missing train_seeds: [prompt_learning.gepa.evaluation] must have 'train_seeds' or 'seeds' field"
            )
        val_seeds = getattr(pl_cfg.gepa.evaluation, "val_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "validation_seeds", None
        )
        if not val_seeds:
            raise click.ClickException(
                "GEPA config missing val_seeds: [prompt_learning.gepa.evaluation] must have 'val_seeds' or 'validation_seeds' field"
            )

    candidate_task_url = (
        (overrides.get("task_url") or task_url or "").strip()
        or (pl_cfg.task_app_url or "").strip()
        or (os.environ.get("TASK_APP_URL") or "").strip()
    )
    env_api_key: str | None = None
    if not (candidate_task_url and is_synthtunnel_url(candidate_task_url)):
        env_api_key = ensure_localapi_auth()

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()

    if synth_ai_py is None or not hasattr(synth_ai_py, "build_prompt_learning_payload"):
        raise click.ClickException(
            "Rust core payload builder unavailable. synth_ai_py is required; no Python fallback."
        )
    try:
        payload, resolved_task_url = synth_ai_py.build_prompt_learning_payload(
            config_dict, task_url, overrides
        )
        return PromptLearningBuildResult(payload=payload, task_url=resolved_task_url)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    cli_task_url = overrides.get("task_url") or task_url
    env_task_url = os.environ.get("TASK_APP_URL")
    config_task_url = (pl_cfg.task_app_url or "").strip() or None

    # Resolve task_app_url with same precedence as file-based builder
    if cli_task_url:
        final_task_url = ConfigResolver.resolve(
            "task_app_url",
            cli_value=cli_task_url,
            env_value=None,
            config_value=config_task_url,
            required=True,
        )
    elif config_task_url:
        final_task_url = config_task_url
    else:
        final_task_url = ConfigResolver.resolve(
            "task_app_url",
            cli_value=None,
            env_value=env_task_url,
            config_value=None,
            required=True,
        )
    _require(final_task_url is not None, "task_app_url is required")

    # Get task_app_api_key from config or environment
    # Note: task_app_api_key is not a field on PromptLearningConfig, use getattr
    config_api_key = (getattr(pl_cfg, "task_app_api_key", None) or "").strip() or None
    cli_api_key = overrides.get("task_app_api_key")
    skip_task_app_key = os.environ.get(
        "SYNTH_BACKEND_RESOLVES_TASK_APP_KEY", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _task_app_api_key = ConfigResolver.resolve(  # noqa: F841 (validation only)
        "task_app_api_key",
        cli_value=cli_api_key,
        env_value=env_api_key,
        config_value=config_api_key,
        required=not skip_task_app_key,
    )

    # Ensure task_app_url is set (task_app_api_key is resolved by backend from ENVIRONMENT_API_KEY)
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        pl_section["task_app_url"] = final_task_url
        if _task_app_api_key and not skip_task_app_key:
            pl_section["task_app_api_key"] = _task_app_api_key

        # GEPA: Extract train_seeds from nested structure
        if pl_cfg.algorithm == "gepa" and pl_cfg.gepa:
            train_seeds = None
            if pl_cfg.gepa.evaluation:
                train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
                    pl_cfg.gepa.evaluation, "seeds", None
                )

            if train_seeds and not pl_section.get("train_seeds"):
                pl_section["train_seeds"] = train_seeds
            if train_seeds and not pl_section.get("evaluation_seeds"):
                pl_section["evaluation_seeds"] = train_seeds

        if pl_cfg.algorithm == "mipro":
            _normalize_mipro_section(pl_cfg, config_dict, source="pre-merge", prefer_model=True)
    else:
        config_dict["prompt_learning"] = {
            "task_app_url": final_task_url,
        }

    # Build payload matching backend API format
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k not in ("backend", "task_url", "metadata", "auto_start")
    }

    # Merge overrides into config_dict
    if config_overrides:
        from synth_ai.core.utils.dict import deep_update as _deep_update

        _deep_update(config_dict, config_overrides)

    if pl_cfg.algorithm == "mipro":
        _normalize_mipro_section(pl_cfg, config_dict, source="post-merge", prefer_model=False)

    # Final validation
    if "prompt_learning" not in config_dict:
        raise ValueError(
            f"config_dict must have 'prompt_learning' key. Found keys: {list(config_dict.keys())}"
        )

    payload: dict[str, Any] = {
        "algorithm": pl_cfg.algorithm,
        "config_body": config_dict,
        "overrides": config_overrides,
        "metadata": overrides.get("metadata", {}),
        "auto_start": overrides.get("auto_start", True),
    }

    backend = overrides.get("backend")
    if backend:
        metadata_default: dict[str, Any] = {}
        metadata = cast(dict[str, Any], payload.setdefault("metadata", metadata_default))
        metadata["backend_base_url"] = ensure_api_base(str(backend))

    return PromptLearningBuildResult(payload=payload, task_url=final_task_url)


__all__ = [
    "PromptLearningBuildResult",
    "build_prompt_learning_payload",
    "build_prompt_learning_payload_from_mapping",
]
