from __future__ import annotations

import sys

import click


def forward_to_core(args: list[str]) -> None:
    """Delegate demo subcommands to the demos.core CLI."""
    try:
        from synth_ai.demos.core import cli as demo_cli  # type: ignore
    except Exception as exc:  # pragma: no cover
        click.echo(f"Failed to import demo CLI: {exc}")
        sys.exit(1)
    rc = int(demo_cli.main(args) or 0)
    if rc != 0:
        sys.exit(rc)
