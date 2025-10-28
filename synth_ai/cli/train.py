

from typing import Any

from synth_ai.api.train.cli import register as _register
from synth_ai.api.train.cli import train_command as _train_command

__all__ = ["register", "train_command"]


def register(cli: Any) -> None:
    """Compatibility wrapper for the legacy train CLI location."""

    _register(cli)


def train_command(*args: Any, **kwargs: Any) -> Any:
    return _train_command(*args, **kwargs)
