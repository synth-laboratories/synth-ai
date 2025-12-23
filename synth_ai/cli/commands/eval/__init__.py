"""Eval command package."""

from __future__ import annotations

def register(cli) -> None:
    from synth_ai.cli.commands.eval.core import eval_command
    cli.add_command(eval_command, name="eval")


__all__ = ["register"]
