from __future__ import annotations

from .errors import EvalCliError
from .validation import validate_eval_options

__all__ = [
    "command",
    "get_command",
    "EvalCliError",
    "validate_eval_options",
]


def __getattr__(name: str):
    if name in {"command", "get_command"}:
        from .core import command, get_command

        return command if name == "command" else get_command
    raise AttributeError(name)
