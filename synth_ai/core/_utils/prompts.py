"""Lazy re-export of prompt utilities to avoid circular imports."""
from __future__ import annotations

from typing import Any

__all__ = ["ensure_required_args", "get_arg_as_env_var", "optional_env_arg"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from synth_ai.cli.lib import prompt_args
        return getattr(prompt_args, name)
    raise AttributeError(f"module 'synth_ai.core._utils.prompts' has no attribute '{name}'")
