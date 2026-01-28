"""Rust-backed validation for optimization configs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import click
import toml

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


def _raise_validation_errors(errors: list[str], config_path: Path) -> None:
    if not errors:
        return
    msg = "\n".join(errors)
    raise click.ClickException(f"{config_path}: {msg}")


def validate_prompt_learning_config(config_data: dict[str, Any], config_path: Path) -> None:
    """Validate prompt learning config using Rust core."""
    try:
        validation_result = _validate_unknown_fields(config_data, config_path=config_path)
        for warning_msg in validation_result.warnings:
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
        for info_msg in validation_result.info:
            warnings.warn(f"Info: {info_msg}", UserWarning, stacklevel=3)
    except Exception:
        pass

    errors = synth_ai_py.validate_prompt_learning_config_strict(config_data)
    if errors:
        _raise_validation_errors(list(errors), config_path)


def validate_prompt_learning_config_from_file(config_path: Path, algorithm: str) -> None:
    """Load a config file and validate prompt learning settings."""
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    config = toml.load(config_path)
    if algorithm not in ("gepa", "mipro"):
        raise click.ClickException(f"Unknown optimization algorithm: {algorithm}")

    validate_prompt_learning_config(config, config_path)


__all__ = [
    "validate_prompt_learning_config",
    "validate_prompt_learning_config_from_file",
    "ConfigValidationError",
]
