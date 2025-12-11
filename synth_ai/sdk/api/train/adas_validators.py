"""SDK-side validation for ADAS jobs.

Catch common configuration and dataset issues before calling the backend.
"""

from __future__ import annotations

import difflib
import warnings
from typing import Any, Dict, List, Set

from .adas_models import ADASJobConfig, ADASTaskSet, SUPPORTED_POLICY_MODELS


class ADASValidationError(Exception):
    """Raised when an ADAS job configuration is invalid."""

    def __init__(self, message: str, errors: List[Dict[str, Any]]) -> None:
        self.message = message
        self.errors = errors
        super().__init__(message)


def _find_similar_models(
    model_name: str,
    supported_models: Set[str],
    max_suggestions: int = 3,
) -> List[str]:
    try:
        return difflib.get_close_matches(
            model_name,
            sorted(supported_models),
            n=max_suggestions,
            cutoff=0.4,
        )
    except Exception:
        return []


def validate_adas_job_config(config: ADASJobConfig, dataset: ADASTaskSet) -> None:
    """Validate an ADAS job config + dataset before submission."""
    errors: List[Dict[str, Any]] = []

    # Policy model
    policy_model = (config.policy_model or "").strip()
    if not policy_model:
        errors.append(
            {
                "field": "policy_model",
                "error": "policy_model is required",
                "suggestion": f"Supported models: {sorted(SUPPORTED_POLICY_MODELS)}",
            }
        )
    elif policy_model not in SUPPORTED_POLICY_MODELS:
        errors.append(
            {
                "field": "policy_model",
                "error": f"Unsupported policy model: {policy_model}",
                "suggestion": f"Supported models: {sorted(SUPPORTED_POLICY_MODELS)}",
                "similar": _find_similar_models(policy_model, SUPPORTED_POLICY_MODELS),
            }
        )

    # Proposer effort
    if config.proposer_effort not in ("low", "medium", "high"):
        errors.append(
            {
                "field": "proposer_effort",
                "error": f"Invalid proposer_effort: {config.proposer_effort}",
                "suggestion": "Must be one of: 'low', 'medium', 'high'",
            }
        )

    # Rollout budget bounds (pydantic enforces, but keep for clearer errors)
    if config.rollout_budget < 10:
        errors.append(
            {
                "field": "rollout_budget",
                "error": f"rollout_budget must be >= 10, got {config.rollout_budget}",
            }
        )
    if config.rollout_budget > 10000:
        errors.append(
            {
                "field": "rollout_budget",
                "error": f"rollout_budget must be <= 10000, got {config.rollout_budget}",
            }
        )

    # Dataset tasks
    if not dataset.tasks:
        errors.append(
            {
                "field": "dataset.tasks",
                "error": "Dataset must contain at least one task",
            }
        )
    elif len(dataset.tasks) < 2:
        warnings.warn(
            "ADAS datasets with <2 tasks are unlikely to optimize meaningfully.",
            stacklevel=2,
        )

    if errors:
        raise ADASValidationError("ADAS job configuration validation failed", errors)


__all__ = ["ADASValidationError", "validate_adas_job_config"]

