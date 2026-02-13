"""Rust-backed validation for optimization configs."""

from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import click

try:
    import tomllib as _toml  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml  # type: ignore[no-redef]

from synth_ai.sdk.optimization.internal.validation.prompt_learning_validation import (
    validate_prompt_learning_config as _validate_unknown_fields,
)

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.validators.") from exc


class ConfigValidationError(Exception):
    """Raised when a training config is invalid."""

    pass


_GEPA_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'proposer_backend' in [prompt_learning.gepa].",
    "Unknown field 'proposer' in [prompt_learning.gepa].",
    "Unknown field 'context_override' in [prompt_learning.gepa].",
    "Unknown field 'proposer_mode' in [prompt_learning.gepa].",
    "Unknown field 'proposal_pipeline' in [prompt_learning.gepa].",
)

_MIPRO_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'mode' in [prompt_learning.mipro].",
    "Unknown field 'val_seeds' in [prompt_learning.mipro].",
    "Unknown field 'proposer' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_mode' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_min_rollouts' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_min_rewards' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_max_candidates' in [prompt_learning.mipro].",
)


def _filter_known_gepa_alias_warnings(warnings_list: list[str]) -> list[str]:
    """Hide unknown-field warnings for accepted GEPA/MIPRO compatibility keys."""
    filtered: list[str] = []
    for warning in warnings_list:
        if any(warning.startswith(prefix) for prefix in _GEPA_ALIAS_WARNING_PREFIXES):
            continue
        if any(warning.startswith(prefix) for prefix in _MIPRO_ALIAS_WARNING_PREFIXES):
            continue
        filtered.append(warning)
    return filtered


def _raise_validation_errors(errors: list[str], config_path: Path) -> None:
    if not errors:
        return
    msg = "\n".join(errors)
    raise click.ClickException(f"{config_path}: {msg}")


def _has_container_id(config_data: dict[str, Any]) -> bool:
    pl = config_data.get("prompt_learning")
    if not isinstance(pl, dict):
        pl = config_data
    container_id = pl.get("container_id") if isinstance(pl, dict) else None
    return isinstance(container_id, str) and bool(container_id.strip())


def _normalize_strict_errors(config_data: dict[str, Any], errors: list[str]) -> list[str]:
    """Filter strict validation errors that conflict with container_id support."""
    if not _has_container_id(config_data):
        return errors
    normalized: list[str] = []
    for err in errors:
        if "Missing required field: prompt_learning.container_url" in err:
            continue
        normalized.append(err)
    return normalized


def _normalize_legacy_mipro_aliases(config_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy MIPRO field aliases before strict validation."""
    normalized = deepcopy(config_data)
    sections: list[dict[str, Any]] = []
    if isinstance(normalized.get("prompt_learning"), dict):
        sections.append(normalized["prompt_learning"])
    if isinstance(normalized.get("policy_optimization"), dict):
        sections.append(normalized["policy_optimization"])

    for section in sections:
        mipro = section.get("mipro")
        if isinstance(mipro, dict):
            if mipro.get("online_pool") is None and mipro.get("online_train_seeds") is not None:
                mipro["online_pool"] = mipro.get("online_train_seeds")
            if section.get("online_pool") is None and mipro.get("online_pool") is not None:
                section["online_pool"] = mipro.get("online_pool")
        if section.get("online_pool") is None and section.get("online_train_seeds") is not None:
            section["online_pool"] = section.get("online_train_seeds")
    return normalized


def _normalize_gepa_aliases(config_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize GEPA proposer alias fields to canonical proposer_type/proposer_mode.

    Legacy/new benchmark configs may send:
    - prompt_learning.gepa.proposer_backend = "prompt" | "rlm" | "agent"
    - prompt_learning.gepa.proposer.prompt.strategy = "synth" | "dspy" | "gepa-ai"
    - prompt_learning.gepa.context_override = {...}

    For proposer_backend='prompt', derives proposer_type/proposer_mode. For
    proposer_backend='rlm' or 'agent', passes through; rust_backend gepa_adapter
    dispatches execution.
    """
    normalized = deepcopy(config_data)
    sections: list[dict[str, Any]] = []
    if isinstance(normalized.get("prompt_learning"), dict):
        sections.append(normalized["prompt_learning"])
    if isinstance(normalized.get("policy_optimization"), dict):
        sections.append(normalized["policy_optimization"])

    def _normalize_prompt_strategy(value: Any) -> str:
        strategy = str(value or "synth").strip().lower()
        if strategy == "gepa_ai":
            strategy = "gepa-ai"
        if strategy == "builtin":
            strategy = "synth"
        return strategy

    for section in sections:
        gepa = section.get("gepa")
        if not isinstance(gepa, dict):
            continue

        # Accept new top-level context_override alias used by benchmark runners.
        if (
            gepa.get("context_override") is not None
            and gepa.get("baseline_context_override") is None
        ):
            gepa["baseline_context_override"] = gepa.get("context_override")

        proposer_backend_raw = gepa.get("proposer_backend")
        if proposer_backend_raw is None:
            continue

        proposer_backend = str(proposer_backend_raw).strip().lower()
        if proposer_backend == "prompt":
            proposer = gepa.get("proposer")
            prompt_strategy = None
            if isinstance(proposer, dict):
                prompt_cfg = proposer.get("prompt")
                if isinstance(prompt_cfg, dict):
                    prompt_strategy = prompt_cfg.get("strategy")
            normalized_strategy = _normalize_prompt_strategy(
                prompt_strategy
                or gepa.get("proposer_type")
                or gepa.get("proposer_mode")
                or gepa.get("proposal_pipeline")
            )
            gepa["proposer_type"] = normalized_strategy
            gepa["proposer_mode"] = normalized_strategy
            gepa["proposal_pipeline"] = normalized_strategy
            continue

        # proposer_backend='rlm' and 'agent' are supported by rust_backend gepa_adapter.
        # Pass through; no normalization needed.
        if proposer_backend in {"rlm", "agent"}:
            continue

        raise click.ClickException(
            f"Invalid prompt_learning.gepa.proposer_backend='{proposer_backend_raw}'. "
            "Expected one of: prompt, rlm, agent."
        )

    return normalized


def validate_prompt_learning_config(config_data: dict[str, Any], config_path: Path) -> None:
    """Validate prompt learning config using Rust core."""
    normalized_for_validation = _normalize_gepa_aliases(
        _normalize_legacy_mipro_aliases(config_data)
    )
    try:
        validation_result = _validate_unknown_fields(
            normalized_for_validation,
            config_path=config_path,
        )
        for warning_msg in _filter_known_gepa_alias_warnings(list(validation_result.warnings)):
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
        for info_msg in validation_result.info:
            warnings.warn(f"Info: {info_msg}", UserWarning, stacklevel=3)
    except Exception:
        pass

    normalized_for_strict = normalized_for_validation
    errors = synth_ai_py.validate_prompt_learning_config_strict(normalized_for_strict)
    errors = _normalize_strict_errors(normalized_for_strict, list(errors))
    if errors:
        _raise_validation_errors(list(errors), config_path)


def validate_prompt_learning_config_from_file(config_path: Path, algorithm: str) -> None:
    """Load a config file and validate prompt learning settings."""
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    with config_path.open("rb") as fh:
        config = _toml.load(fh)
    if algorithm not in ("gepa", "mipro"):
        raise click.ClickException(f"Unknown optimization algorithm: {algorithm}")

    validate_prompt_learning_config(config, config_path)


__all__ = [
    "validate_prompt_learning_config",
    "validate_prompt_learning_config_from_file",
    "ConfigValidationError",
]
