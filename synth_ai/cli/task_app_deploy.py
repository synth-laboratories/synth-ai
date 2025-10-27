"""Task app deploy command."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import click
from click.core import ParameterSource
from synth_ai.utils.env import resolve_env_var
from synth_ai.utils.cli import print_next_step
from synth_ai.utils.task_app_discovery import AppChoice, select_app_choice
from synth_ai.utils.task_app_state import persist_env_api_key, persist_task_url
from synth_ai.utils.user_config import load_user_env, update_user_config

from .task_apps import _run_modal_script, _run_modal_with_entry, _serve_cli, task_app_group



def _deploy_local_task_app_choice(choice: AppChoice) -> None:
    """Launch a local task app using the shared serve machinery."""

    try:
        choice.ensure_entry()
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Task app '{choice.app_id}' does not support local serving. Select a different app or deploy to Modal."
        ) from exc

    print_next_step(
        "collect rollouts in a separate terminal",
        [
            "open a new terminal",
            "synth-ai rollout",
        ],
    )

    click.echo(f"Launching local task app for [{choice.app_id}]…\n")
    _serve_cli(
        choice.app_id,
        host="0.0.0.0",
        port=None,
        env_file=(),
        reload_flag=False,
        force=False,
        trace_dir=None,
        trace_db=None,
        allow_demo_dir=False,
        choice=choice,
    )


def _deploy_with_legacy_script(script_path: Path, modal_name: str | None) -> str | None:
    env_api_key = os.environ.get("ENVIRONMENT_API_KEY", "").strip()
    if not env_api_key:
        resolve_env_var("ENVIRONMENT_API_KEY")
        env_api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_api_key:
        raise click.ClickException("ENVIRONMENT_API_KEY is required.")

    update_user_config({"ENVIRONMENT_API_KEY": env_api_key})

    try:
        from synth_ai.demos.math.deploy_modal import deploy as modal_deploy
    except Exception as exc:  # pragma: no cover - optional legacy path
        raise click.ClickException(f"Legacy deploy support unavailable: {exc}") from exc

    click.echo(f"Running legacy deploy script: {script_path}")
    url = modal_deploy(script_path=str(script_path), env_api_key=env_api_key)
    click.echo(f"✓ Task app URL: {url}")
    return url


def _persist_modal_deploy(task_app_url: str | None, *, app_path: Path | None, modal_name: str | None) -> None:
    if not task_app_url:
        return

    persist_path: str | None = None
    if app_path is not None:
        try:
            persist_path = str(app_path.resolve())
        except Exception:
            persist_path = str(app_path)

    with contextlib.suppress(Exception):
        persist_task_url(task_app_url, name=modal_name, path=persist_path)


def _choice_supports_modal(choice: AppChoice) -> bool:
    if choice.modal_script:
        return True
    try:
        entry = choice.ensure_entry()
    except Exception:
        return False
    return entry.modal is not None


@task_app_group.command("deploy")
@click.argument("app_id", type=str, required=False)
@click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
@click.option("--app", type=click.Path(), default=None, help="Path to Modal app.py for manual deploy")
@click.option("--script", type=click.Path(), default=None, help="Path to legacy deploy_task_app.sh script")
@click.option("--name", "modal_name", default=None, help="Override Modal app name")
@click.option("--dry-run", is_flag=True, help="Print Modal command without executing")
@click.option("--modal-cli", default="modal", help="Path to Modal CLI executable")
def deploy_command(
    app_id: str | None,
    local: bool,
    app: str | None,
    script: str | None,
    modal_name: str | None,
    dry_run: bool,
    modal_cli: str,
) -> None:
    """Deploy a task app locally or to Modal."""

    load_user_env()

    resolve_env_var("ENVIRONMENT_API_KEY")
    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_key:
        raise click.ClickException("ENVIRONMENT_API_KEY is required.")

    update_user_config(
        {
            "ENVIRONMENT_API_KEY": env_key,
            "DEV_ENVIRONMENT_API_KEY": env_key,
        }
    )
    persist_env_api_key(env_key)

    resolve_env_var("SYNTH_API_KEY")
    synth_key = (os.environ.get("SYNTH_API_KEY") or "").strip()
    if synth_key:
        update_user_config({"SYNTH_API_KEY": synth_key})

    selected_choice: AppChoice | None = None
    if not app and not script:
        selected_choice = select_app_choice(app_id, purpose="any")
        app_id = selected_choice.app_id

    ctx = click.get_current_context()
    local_source = ctx.get_parameter_source("local")

    default_mode = "modal"
    if selected_choice is not None and not _choice_supports_modal(selected_choice):
        default_mode = "local"

    if app or script:
        mode = "modal"
    elif local_source == ParameterSource.COMMANDLINE:
        mode = "local" if local else "modal"
    else:
        mode = click.prompt(
            "Deploy target",
            type=click.Choice(["modal", "local"]),
            default=default_mode,
            show_choices=True,
        )
        click.echo(f"Deploying to {mode.upper()} target.\n")
    local = mode == "local"

    if local:
        if selected_choice is None:
            raise click.ClickException("Select a task app before running local deployment.")
        _deploy_local_task_app_choice(selected_choice)
        return

    if app and script:
        raise click.ClickException("Use either --app or --script, not both.")

    if script:
        script_path = Path(script).expanduser().resolve()
        if not script_path.exists():
            raise click.ClickException(f"Legacy deploy script not found: {script_path}")
        task_app_url = _deploy_with_legacy_script(script_path, modal_name)
        _persist_modal_deploy(task_app_url, app_path=script_path, modal_name=modal_name)
        return

    if app:
        script_path = Path(app).expanduser().resolve()
        if not script_path.is_file():
            raise click.ClickException(f"App file not found: {script_path}")
        task_app_url = _run_modal_script(
            script_path,
            modal_cli,
            "deploy",
            modal_name=modal_name,
            dry_run=dry_run,
        )
        _persist_modal_deploy(task_app_url, app_path=script_path, modal_name=modal_name)
        return

    if selected_choice is None:
        choice = select_app_choice(app_id, purpose="deploy")
    else:
        choice = selected_choice
        if not _choice_supports_modal(choice):
            raise click.ClickException(
                f"Task app '{choice.app_id}' does not define Modal deployment settings. Select a different app or run with --local."
            )

    if choice.modal_script:
        task_app_url = _run_modal_script(
            choice.modal_script,
            modal_cli,
            "deploy",
            modal_name=modal_name,
            dry_run=dry_run,
        )
        persist_name = modal_name
        if persist_name is None:
            try:
                entry = choice.ensure_entry()
                if entry.modal is not None:
                    persist_name = entry.modal.app_name
            except Exception:
                persist_name = None
        _persist_modal_deploy(task_app_url, app_path=choice.path, modal_name=persist_name)
        return

    entry = choice.ensure_entry()
    if entry.modal is None:
        raise click.ClickException(
            f"Task app '{entry.app_id}' does not define Modal deployment settings"
        )

    deploy_name = modal_name or entry.modal.app_name
    task_app_url = _run_modal_with_entry(
        entry,
        entry.modal,
        modal_cli,
        deploy_name,
        "deploy",
        dry_run=dry_run,
        original_path=choice.path,
    )

    _persist_modal_deploy(task_app_url, app_path=choice.path, modal_name=deploy_name)


__all__ = ["deploy_command"]
