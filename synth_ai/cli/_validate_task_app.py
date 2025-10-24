"""Task app validation utilities - imported by task_apps.py"""

# This module provides the validate_task_app function for CLI use
# The actual implementation is imported from the task module

from synth_ai.task.validators import (
    validate_task_app_endpoint as validate_task_app,  # type: ignore[attr-defined]
)

__all__ = ["validate_task_app"]

