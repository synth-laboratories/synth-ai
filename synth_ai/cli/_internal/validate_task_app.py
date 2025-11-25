"""Task app validation utilities - imported by task_apps.py"""

# This module provides the validate_task_app function for CLI use
# The actual implementation is imported from the task module

import importlib
from collections.abc import Callable
from typing import Any

_validators_module: Any | None = None
validate_task_app: Callable[..., Any] | None = None

try:
    _validators_module = importlib.import_module("synth_ai.sdk.task.validators")
except Exception:
    _validators_module = None

if _validators_module is not None:
    candidate = getattr(_validators_module, "validate_task_app_endpoint", None)
    if callable(candidate):
        validate_task_app = candidate

if validate_task_app is None:
    def _missing_validate_task_app(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("task validation utilities are unavailable in this environment")

    validate_task_app = _missing_validate_task_app

__all__ = ["validate_task_app"]
