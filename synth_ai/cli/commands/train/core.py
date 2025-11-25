from __future__ import annotations

import click

from synth_ai.sdk.api.train.cli import (
    register as _register_with_cli,
)
from synth_ai.sdk.api.train.cli import (
    train_command as _train_command,
)

__all__ = ["register", "train_command"]


def register(cli: click.Group) -> None:
    """Attach the train command to the root CLI."""
    _register_with_cli(cli)


def train_command(*args, **kwargs):
    """Entrypoint used by the train CLI command."""
    return _train_command(*args, **kwargs)
