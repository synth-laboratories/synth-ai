from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
from pydantic import ValidationError

try:
    _models_module = cast(Any, importlib.import_module("synth_ai.sdk.api.models.supported"))
    UnsupportedModelError = cast(type[Exception], _models_module.UnsupportedModelError)
    ensure_allowed_model = cast(Callable[..., None], _models_module.ensure_allowed_model)
    normalize_model_identifier = cast(
        Callable[[str], str], _models_module.normalize_model_identifier
    )
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load supported model helpers") from exc

try:
    _sft_module = cast(Any, importlib.import_module("synth_ai.sdk.learning.sft.config"))
    prepare_sft_job_payload = cast(
        Callable[..., dict[str, Any]], _sft_module.prepare_sft_job_payload
    )
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load SFT payload helpers") from exc

from synth_ai.core.config.resolver import ConfigResolver
from synth_ai.core.telemetry import log_info

from .configs import PromptLearningConfig, RLConfig, SFTConfig
from .supported_algos import (
    AlgorithmValidationError,
    ensure_model_supported_for_algorithm,
    validate_algorithm_config,
)
from .utils import TrainError, ensure_api_base


def _maybe_expand_minimal_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Expand minimal config to full config if needed.

    Detects if the config uses the simplified minimal format and expands it
    to the full nested format expected by PromptLearningConfig.

    Minimal config indicators:
        - Has 'total_seeds' at prompt_learning level
        - Has 'proposer_effort' at prompt_learning level without 'gepa' section
        - Has 'train_seeds' at prompt_learning level without 'gepa' section

    Args:
        raw_config: Raw config dict, potentially in minimal format.

    Returns:
        Full config dict (original if already full, expanded if minimal).
    """
    from synth_ai.config_expansion import expand_gepa_config, is_minimal_config

    if "prompt_learning" not in raw_config:
        return raw_config

    pl_config = raw_config.get("prompt_learning", {})

    # Check if this looks like a minimal config
    if not is_minimal_config(pl_config):
        return raw_config

    # Only expand GEPA configs for now
    algorithm = pl_config.get("algorithm", "gepa")
    if algorithm != "gepa":
        return raw_config

    # Expand minimal config
    log_info("Expanding minimal config to full config", ctx={"algorithm": algorithm})
    expanded = expand_gepa_config(pl_config)

    # Return in the expected wrapper format
    return {"prompt_learning": expanded}


@dataclass(slots=True)
class RLBuildResult:
    payload: dict[str, Any]
    task_url: str
    idempotency: str | None


@dataclass(slots=True)
class SFTBuildResult:
    payload: dict[str, Any]
    train_file: Path
    validation_file: Path | None


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


def build_rl_payload(
    *,
    config_path: Path,
    task_url: str,
    overrides: dict[str, Any],
    idempotency: str | None,
    allow_experimental: bool | None = None,
) -> RLBuildResult:
    ctx: dict[str, Any] = {"config_path": str(config_path), "task_url": task_url}
    log_info("build_rl_payload invoked", ctx=ctx)
    # Load and validate config with SDK-level checks
    from synth_ai.cli.commands.train.validation import validate_rl_config
    from synth_ai.sdk.api.train.utils import load_toml

    try:
        raw_config = load_toml(config_path)
        validated_config = validate_rl_config(raw_config)  # Adds defaults & validates
        rl_cfg = RLConfig.from_mapping(validated_config)
    except ValidationError as exc:
        raise click.ClickException(_format_validation_error(config_path, exc)) from exc

    data = rl_cfg.to_dict()

    # Remove smoke section - it's CLI-only and should not be sent to the trainer
    if "smoke" in data:
        del data["smoke"]

    # Ensure required [reference] section for backend validators
    try:
        ref_cfg = data.get("reference") if isinstance(data, dict) else None
        if not isinstance(ref_cfg, dict):
            data["reference"] = {"placement": "none"}
        else:
            ref_cfg.setdefault("placement", "none")
    except Exception:
        # Defensive: never fail builder due to optional defaults
        data["reference"] = {"placement": "none"}
    try:
        spec = validate_algorithm_config(rl_cfg.algorithm.model_dump(), expected_family="rl")
    except AlgorithmValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    services = data.get("services") if isinstance(data.get("services"), dict) else {}
    model_cfg = rl_cfg.model

    cli_task_url = overrides.get("task_url")
    env_task_url = task_url or os.environ.get("TASK_APP_URL")
    config_task_url = services.get("task_url") if isinstance(services, dict) else None
    final_task_url = ConfigResolver.resolve(
        "task_app_url",
        cli_value=cli_task_url,
        env_value=env_task_url,
        config_value=config_task_url,
        required=True,
    )
    assert final_task_url is not None  # required=True guarantees non-None

    model_source = (model_cfg.source or "").strip() if model_cfg else ""
    model_base = (model_cfg.base or "").strip() if model_cfg else ""
    override_model = (overrides.get("model") or "").strip()
    if override_model:
        model_source = override_model
        model_base = ""
    if bool(model_source) == bool(model_base):
        details = (
            f"Config: {config_path}\n[model].source={model_source!r} | [model].base={model_base!r}"
        )
        hint = (
            "Set exactly one: [model].base for a base model (e.g. 'Qwen/Qwen3-1.7B') "
            "or [model].source for a fine-tuned model id. Also remove any conflicting "
            "'[policy].model' entries."
        )
        raise click.ClickException(
            "Invalid model config: exactly one of [model].source or [model].base is required.\n"
            + details
            + "\nHint: "
            + hint
        )

    try:
        if model_source:
            model_source = normalize_model_identifier(model_source)
        if model_base:
            model_base = normalize_model_identifier(model_base)
    except UnsupportedModelError as exc:
        raise click.ClickException(str(exc)) from exc

    base_model_for_training: str | None = None
    if model_source:
        base_model_for_training = ensure_allowed_model(
            model_source,
            allow_finetuned_prefixes=True,
            allow_experimental=allow_experimental,
        )
    elif model_base:
        base_model_for_training = ensure_allowed_model(
            model_base,
            allow_finetuned_prefixes=False,
            allow_experimental=allow_experimental,
        )
    if base_model_for_training:
        try:
            ensure_model_supported_for_algorithm(base_model_for_training, spec)
        except AlgorithmValidationError as exc:
            raise click.ClickException(str(exc)) from exc

    # Force TOML services.task_url to the effective endpoint to avoid split URLs
    try:
        if isinstance(data.get("services"), dict):
            data["services"]["task_url"] = final_task_url
        else:
            data["services"] = {"task_url": final_task_url}
    except Exception:
        pass

    payload_data: dict[str, Any] = {
        "endpoint_base_url": final_task_url.rstrip("/"),
        "config": data,
    }
    payload: dict[str, Any] = {
        "job_type": "rl",
        "compute": data.get("compute", {}),
        "data": payload_data,
        "tags": {"source": "train-cli"},
    }
    if model_source:
        payload_data["model"] = model_source
    if model_base:
        payload_data["base_model"] = model_base

    backend = overrides.get("backend")
    if backend:
        metadata_default: dict[str, Any] = {}
        metadata = cast(dict[str, Any], payload.setdefault("metadata", metadata_default))
        metadata["backend_base_url"] = ensure_api_base(str(backend))

    return RLBuildResult(payload=payload, task_url=final_task_url, idempotency=idempotency)


def build_sft_payload(
    *,
    config_path: Path,
    dataset_override: Path | None,
    allow_experimental: bool | None,
) -> SFTBuildResult:
    ctx: dict[str, Any] = {
        "config_path": str(config_path),
        "dataset_override": str(dataset_override) if dataset_override else None,
    }
    log_info("build_sft_payload invoked", ctx=ctx)
    try:
        sft_cfg = SFTConfig.from_path(config_path)
    except ValidationError as exc:
        raise TrainError(_format_validation_error(config_path, exc)) from exc

    data = sft_cfg.to_dict()
    try:
        algo_mapping = sft_cfg.algorithm.model_dump() if sft_cfg.algorithm else None
        spec = validate_algorithm_config(algo_mapping, expected_family="sft")
    except AlgorithmValidationError as exc:
        raise TrainError(str(exc)) from exc
    data_cfg = data.get("data") if isinstance(data.get("data"), dict) else {}
    hp_cfg = data.get("hyperparameters") if isinstance(data.get("hyperparameters"), dict) else {}
    train_cfg = data.get("training") if isinstance(data.get("training"), dict) else {}
    compute_cfg = data.get("compute") if isinstance(data.get("compute"), dict) else {}

    raw_dataset = dataset_override or sft_cfg.job.data or sft_cfg.job.data_path
    if not raw_dataset:
        raise TrainError("Dataset not specified; pass --dataset or set [job].data")
    dataset_path = Path(raw_dataset)
    # Resolve relative paths from current working directory, not config directory
    dataset_path = (
        dataset_path if dataset_path.is_absolute() else (Path.cwd() / dataset_path)
    ).resolve()
    if not dataset_path.exists():
        raise TrainError(f"Dataset not found: {dataset_path}")

    validation_path = (
        data_cfg.get("validation_path")
        if isinstance(data_cfg, dict)
        else None
        if isinstance(data_cfg, dict) and isinstance(data_cfg.get("validation_path"), str)
        else None
    )
    validation_file = None
    if validation_path:
        vpath = Path(validation_path)
        # Resolve relative paths from current working directory, not config directory
        vpath = (vpath if vpath.is_absolute() else (Path.cwd() / vpath)).resolve()
        if not vpath.exists():
            click.echo(f"[WARN] Validation dataset {vpath} missing; continuing without validation")
        else:
            validation_file = vpath

    hp_block: dict[str, Any] = {
        "n_epochs": int(hp_cfg.get("n_epochs", 1) if isinstance(hp_cfg, dict) else 1),
    }
    for key in (
        "batch_size",
        "global_batch",
        "per_device_batch",
        "gradient_accumulation_steps",
        "sequence_length",
        "learning_rate",
        "warmup_ratio",
        "train_kind",
    ):
        if isinstance(hp_cfg, dict) and key in hp_cfg:
            hp_block[key] = hp_cfg[key]
    if isinstance(hp_cfg, dict) and isinstance(hp_cfg.get("parallelism"), dict):
        hp_block["parallelism"] = hp_cfg["parallelism"]

    compute_block = {
        k: compute_cfg[k]
        for k in ("gpu_type", "gpu_count", "nodes")
        if isinstance(compute_cfg, dict) and k in compute_cfg
    }

    effective = {
        "compute": compute_block,
        "data": {
            "topology": data_cfg.get("topology", {})
            if isinstance(data_cfg, dict) and isinstance(data_cfg.get("topology"), dict)
            else {}
        },
        "training": {
            k: v
            for k, v in (train_cfg.items() if isinstance(train_cfg, dict) else [])
            if k in ("mode", "use_qlora")
        },
    }

    validation_cfg = (
        train_cfg.get("validation")
        if isinstance(train_cfg, dict) and isinstance(train_cfg.get("validation"), dict)
        else None
    )
    if isinstance(validation_cfg, dict):
        hp_block.update(
            {
                "evaluation_strategy": validation_cfg.get("evaluation_strategy", "steps"),
                "eval_steps": int(validation_cfg.get("eval_steps", 0) or 0),
                "save_best_model_at_end": bool(validation_cfg.get("save_best_model_at_end", True)),
                "metric_for_best_model": validation_cfg.get("metric_for_best_model", "val.loss"),
                "greater_is_better": bool(validation_cfg.get("greater_is_better", False)),
            }
        )
        effective.setdefault("training", {})["validation"] = {
            "enabled": bool(validation_cfg.get("enabled", True))
        }

    raw_model = (sft_cfg.job.model or "").strip()
    if not raw_model:
        model_block = data.get("model")
        if isinstance(model_block, str):
            raw_model = model_block.strip()
    if not raw_model:
        raise TrainError("Model not specified; set [job].model or [model].base in the config")

    try:
        base_model = ensure_allowed_model(
            raw_model,
            allow_finetuned_prefixes=False,
            allow_experimental=allow_experimental,
        )
    except UnsupportedModelError as exc:
        raise TrainError(str(exc)) from exc

    if base_model:
        try:
            ensure_model_supported_for_algorithm(base_model, spec)
        except AlgorithmValidationError as exc:
            raise TrainError(str(exc)) from exc

    try:
        payload = prepare_sft_job_payload(
            model=raw_model,
            training_file=None,
            hyperparameters=hp_block,
            metadata={"effective_config": effective},
            training_type="sft_offline",
            training_file_field="training_file_id",
            require_training_file=False,
            include_training_file_when_none=True,
            allow_finetuned_prefixes=False,
        )
    except UnsupportedModelError as exc:
        raise TrainError(str(exc)) from exc
    except ValueError as exc:
        raise TrainError(str(exc)) from exc

    return SFTBuildResult(payload=payload, train_file=dataset_path, validation_file=validation_file)


def build_prompt_learning_payload(
    *,
    config_path: Path,
    task_url: str | None,
    overrides: dict[str, Any],
    allow_experimental: bool | None = None,
) -> PromptLearningBuildResult:
    """Build payload for prompt learning job (GEPA).

    Supports both minimal and full config formats in TOML files.
    Minimal configs are auto-expanded before validation.
    """
    ctx: dict[str, Any] = {"config_path": str(config_path), "task_url": task_url}
    log_info("build_prompt_learning_payload invoked", ctx=ctx)
    from pydantic import ValidationError

    from .configs.prompt_learning import load_toml

    # SDK-SIDE VALIDATION: Catch errors BEFORE sending to backend
    from .validators import validate_prompt_learning_config

    raw_config = load_toml(config_path)

    # Expand minimal config to full config if needed
    raw_config = _maybe_expand_minimal_config(raw_config)

    validate_prompt_learning_config(raw_config, config_path)

    try:
        # Use from_mapping to support expanded configs
        pl_cfg = PromptLearningConfig.from_mapping(raw_config)
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
    assert final_task_url is not None  # required=True guarantees non-None

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()

    # Ensure task_app_url is set
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        pl_section["task_app_url"] = final_task_url
        pl_section.pop("task_app_api_key", None)

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

    else:
        config_dict["prompt_learning"] = {"task_app_url": final_task_url}

    # Build payload matching backend API format
    # Extract nested overrides if present, otherwise use flat overrides directly
    # The experiment queue passes flat overrides like {"prompt_learning.policy.model": "..."}
    # But some SDK code passes nested like {"overrides": {"prompt_learning.policy.model": "..."}}
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    # Remove non-override keys (backend, task_url, metadata, auto_start)
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k
        not in (
            "backend",
            "task_url",
            "metadata",
            "auto_start",
            "task_app_api_key",
            "prompt_learning.task_app_api_key",
        )
    }

    # CRITICAL: Merge overrides into config_dict BEFORE sending to backend
    # This ensures early validation in backend sees merged values
    # Use the same _deep_update logic from experiment_queue/config_utils.py
    if config_overrides:
        from synth_ai.cli.local.experiment_queue.config_utils import _deep_update

        _deep_update(config_dict, config_overrides)
        pl_section = config_dict.get("prompt_learning", {})
        if isinstance(pl_section, dict):
            pl_section.pop("task_app_api_key", None)

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
                assert actual_budget == expected_budget, (
                    f"Rollout budget mismatch: config_body has {actual_budget} but override specifies {expected_budget}. "
                    f"This indicates the override wasn't applied correctly."
                )

        # Check model override
        model_key = "prompt_learning.policy.model"
        if model_key in config_overrides:
            expected_model = config_overrides[model_key]
            policy_section = pl_section_in_dict.get("policy", {})
            actual_model = policy_section.get("model") if isinstance(policy_section, dict) else None
            if actual_model is not None:
                assert actual_model == expected_model, (
                    f"Model mismatch: config_body has {actual_model} but override specifies {expected_model}. "
                    f"This indicates the override wasn't applied correctly."
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
                assert actual_provider == expected_provider, (
                    f"Provider mismatch: config_body has {actual_provider} but override specifies {expected_provider}. "
                    f"This indicates the override wasn't applied correctly."
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

    Supports both minimal and full config formats. Minimal configs are auto-expanded:

    Minimal config (4 required fields):
        >>> result = build_prompt_learning_payload_from_mapping(
        ...     raw_config={
        ...         "prompt_learning": {
        ...             "algorithm": "gepa",
        ...             "task_app_url": "https://tunnel.example.com",
        ...             "total_seeds": 200,
        ...             "proposer_effort": "LOW",
        ...             "proposer_output_tokens": "FAST",
        ...         }
        ...     },
        ...     task_url=None,
        ...     overrides={},
        ... )

    Full config (explicit control):
        >>> result = build_prompt_learning_payload_from_mapping(
        ...     raw_config={
        ...         "prompt_learning": {
        ...             "algorithm": "gepa",
        ...             "task_app_url": "https://tunnel.example.com",
        ...             "gepa": {...},
        ...         }
        ...     },
        ...     task_url=None,
        ...     overrides={},
        ... )

    Args:
        raw_config: Configuration dictionary with the same structure as the TOML file.
                   Should have a 'prompt_learning' section.
        task_url: Override for task_app_url
        overrides: Config overrides (merged into config)
        allow_experimental: Allow experimental models
        source_label: Label for logging/error messages (default: "programmatic")

    Returns:
        PromptLearningBuildResult with payload and task_url
    """
    ctx: dict[str, Any] = {"source": source_label}
    log_info("build_prompt_learning_payload_from_mapping invoked", ctx=ctx)

    # Expand minimal config to full config if needed
    raw_config = _maybe_expand_minimal_config(raw_config)
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
    assert final_task_url is not None

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()

    # Ensure task_app_url is set
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        pl_section["task_app_url"] = final_task_url
        pl_section.pop("task_app_api_key", None)

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
    else:
        config_dict["prompt_learning"] = {"task_app_url": final_task_url}

    # Build payload matching backend API format
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k
        not in (
            "backend",
            "task_url",
            "metadata",
            "auto_start",
            "task_app_api_key",
            "prompt_learning.task_app_api_key",
        )
    }

    # Merge overrides into config_dict
    if config_overrides:
        from synth_ai.cli.local.experiment_queue.config_utils import _deep_update

        _deep_update(config_dict, config_overrides)
        pl_section = config_dict.get("prompt_learning", {})
        if isinstance(pl_section, dict):
            pl_section.pop("task_app_api_key", None)

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
    "RLBuildResult",
    "SFTBuildResult",
    "build_prompt_learning_payload",
    "build_prompt_learning_payload_from_mapping",
    "build_rl_payload",
    "build_sft_payload",
]
