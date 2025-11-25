"""Task app list command."""

from __future__ import annotations

import click

from synth_ai.sdk.task.apps import registry

from .commands import task_app_group


@task_app_group.command("list")
def list_apps() -> None:
    """List registered task apps."""

    entries = registry.list()
    if not entries:
        click.echo("No task apps registered.")
        return

    for entry in entries:
        aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        click.echo(f"- {entry.app_id}{aliases}: {entry.description}")


__all__ = ["list_apps"]
