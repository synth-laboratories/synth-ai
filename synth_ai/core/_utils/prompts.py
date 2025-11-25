"""Lazy re-export of prompt utilities to avoid circular imports."""
from __future__ import annotations

from typing import Any

# Functions available via __getattr__:
# - ensure_required_args


def __getattr__(name: str) -> Any:
    if name == "ensure_required_args":
        from synth_ai.cli.lib import prompt_args
        return getattr(prompt_args, name)
    raise AttributeError(f"module 'synth_ai.core._utils.prompts' has no attribute '{name}'")
