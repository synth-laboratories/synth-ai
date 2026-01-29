"""Rust-backed validation for prompt learning (GEPA/MIPRO) configurations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "synth_ai_py is required for optimization.prompt_learning_validation."
    ) from exc


@dataclass
class PromptLearningValidationResult:
    """Validation results for prompt learning config."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PromptLearningValidationResult:
        return cls(
            errors=list(payload.get("errors") or []),
            warnings=list(payload.get("warnings") or []),
            info=list(payload.get("info") or []),
        )


def validate_prompt_learning_config(
    config: dict[str, Any],
    config_path: Path | None = None,
) -> PromptLearningValidationResult:
    """Validate prompt learning config using Rust core."""
    payload = synth_ai_py.validate_prompt_learning_config(
        config, str(config_path) if config_path else None
    )
    if isinstance(payload, dict):
        return PromptLearningValidationResult.from_payload(payload)
    return PromptLearningValidationResult()


def validate_and_warn(
    config: dict[str, Any],
    config_path: Path | None = None,
    *,
    emit_warnings: bool = True,
) -> PromptLearningValidationResult:
    """Validate config and optionally emit warnings/errors to stderr."""
    result = validate_prompt_learning_config(config, config_path=config_path)

    if emit_warnings:
        for warning in result.warnings:
            warnings.warn(warning, UserWarning, stacklevel=2)
        for info in result.info:
            warnings.warn(f"Info: {info}", UserWarning, stacklevel=2)

    if result.errors:
        error_msg = "\n".join(f"  - {e}" for e in result.errors)
        raise ValueError(f"Config validation failed:\n{error_msg}")

    return result


__all__ = [
    "validate_prompt_learning_config",
    "validate_and_warn",
    "PromptLearningValidationResult",
]
