"""Task app Modal serve command."""

from __future__ import annotations

from collections.abc import Sequence

import click

from synth_ai.cli.lib.task_app_discovery import select_app_choice

from .task_apps import (
    _maybe_use_demo_dir,
    _resolve_modal_choice_env,
    _run_modal_script,
    _run_modal_with_entry,
    task_app_group,
)


@task_app_group.command("modal-serve")
@click.argument("app_id", type=str, required=False)
@click.option("--modal-cli", default="modal", help="Path to Modal CLI executable")
@click.option("--name", "modal_name", default=None, help="Override Modal app name (optional)")
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Env file to load into the container (repeatable)",
)
def modal_serve_command(
    app_id: str | None,
    modal_cli: str,
    modal_name: str | None,
    env_file: Sequence[str],
) -> None:
    """Run ``modal serve`` for a task app."""

    _maybe_use_demo_dir()

    choice = select_app_choice(app_id, purpose="modal-serve")
    env_paths = _resolve_modal_choice_env(choice, env_file)
    click.echo("Using env file(s): " + ", ".join(str(p.resolve()) for p in env_paths))

    if choice.modal_script:
        _run_modal_script(
            choice.modal_script,
            modal_cli,
            "serve",
            env_paths,
            modal_name=modal_name,
        )
        return

    entry = choice.ensure_entry()
    if entry.modal is None:
        raise click.ClickException(
            f"Task app '{entry.app_id}' does not define Modal deployment settings"
        )

    _run_modal_with_entry(
        entry,
        entry.modal,
        modal_cli,
        modal_name,
        env_paths,
        "serve",
        original_path=choice.path,
    )


__all__ = ["modal_serve_command"]
