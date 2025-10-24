from __future__ import annotations

from typing import Any

__all__ = ["register", "train_command"]


def register(cli: Any) -> None:
    from synth_ai.cli.train import register as _register  # local import avoids circular dependency

    _register(cli)


def train_command(*args: Any, **kwargs: Any) -> Any:
    from synth_ai.cli.train import train_command as _train_command  # local import avoids cycle

    return _train_command(*args, **kwargs)
