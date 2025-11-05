from pathlib import Path
from types import SimpleNamespace
from typing import Literal, TypeAlias, get_args

import click
from synth_ai.task_app_cfgs import LocalTaskAppConfig, ModalTaskAppConfig
from synth_ai.utils.cli import PromptedChoiceOption, PromptedChoiceType, PromptedPathOption
from synth_ai.utils.modal import deploy_modal_app, get_default_modal_bin_path
from synth_ai.utils.uvicorn import deploy_uvicorn_app

RuntimeType: TypeAlias = Literal[
    "local",
    "modal"
]
RUNTIMES = get_args(RuntimeType)

MODAL_RUNTIME_OPTIONS = [
    "task_app_name",
    "cmd_arg",
    "modal_bin_path",
    "dry_run",
    "modal_app_path",
]
LOCAL_RUNTIME_OPTIONS = [
    "trace",
    "host",
    "port"
]

RUNTIME_MSG = SimpleNamespace(
    init="[deploy]",
    local="[deploy --runtime local]",
    modal="[deploy --runtime modal]",
)


@click.command("deploy")
# --- Required options ---
@click.option(
    "--task-app",
    "task_app_path",
    cls=PromptedPathOption,
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path
    ),
    file_type=".py",
    help=f"{RUNTIME_MSG.init} Enter the path to your task app",
)
@click.option(
    "--runtime",
    "runtime",
    cls=PromptedChoiceOption,
    type=PromptedChoiceType(RUNTIMES),
    required=True
)
# --- Local-only options ---
@click.option(
    "--trace/--no-trace",
    "trace",
    default=True,
    help=f"{RUNTIME_MSG.local} Enable or disable trace output"
)
@click.option(
    "--host",
    "host",
    default="127.0.0.1", 
    help=f"{RUNTIME_MSG.local} Host to bind to"
)
@click.option(
    "--port",
    "port",
    default=8000,
    type=int,
    help=f"{RUNTIME_MSG.local} Port to bind to"
)
# --- Modal-only options ---
@click.option(
    "--modal-app",
    "modal_app_path",
    cls=PromptedPathOption,
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path
    ),
    file_type=".py",
    prompt_guard=lambda ctx: (ctx.params.get("runtime") != "local"),
    help=f"{RUNTIME_MSG.modal} Enter the path to your Modal app",
)
@click.option(
    "--name",
    "task_app_name",
    default=None,
    help=f"{RUNTIME_MSG.modal} Override Modal app name"
)
@click.option(
    "--modal-mode",
    "cmd_arg",
    default="deploy",
    help=f"{RUNTIME_MSG.modal} Mode: deploy or serve"
)
@click.option(
    "--modal-cli",
    "modal_bin_path",
    type=click.Path(
        dir_okay=False,
        file_okay=True,
        exists=True,
        path_type=Path
    ),
    default=None,
    help=f"{RUNTIME_MSG.modal} Path to Modal CLI",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    help=f"{RUNTIME_MSG.modal} Print Modal command without executing"
)
@click.option(
    "--env-file",
    "env_file",
    multiple=True,
    type=click.Path(exists=True),
    help="Path to .env file(s) to load"
)
def deploy_cmd(
    task_app_path: Path,
    runtime: RuntimeType,
    env_file: tuple[str, ...],
    **kwargs
) -> None:
    """Deploy a task app to local or Modal runtime."""
    match runtime:
        case "local":
            opts = {k: v for k, v in kwargs.items() if k in LOCAL_RUNTIME_OPTIONS}
            deploy_uvicorn_app(LocalTaskAppConfig(**opts, task_app_path=task_app_path))

        case "modal":
            opts = {k: v for k, v in kwargs.items() if k in MODAL_RUNTIME_OPTIONS}

            if "modal_app_path" not in opts or opts["modal_app_path"] is None:
                raise click.ClickException("Modal app path required")
            
            if opts["cmd_arg"] == "serve" and opts["dry_run"] is True:
                raise click.ClickException("--modal-mode=serve cannot be combined with --dry-run")
            
            modal_bin_path = opts.get("modal_bin_path") or get_default_modal_bin_path()
            if not modal_bin_path:
                raise click.ClickException(
                    "Modal CLI not found. Install the `modal` package or pass --modal-cli with its path."
                )
            if isinstance(modal_bin_path, str):
                modal_bin_path = Path(modal_bin_path)
            opts["modal_bin_path"] = modal_bin_path
            deploy_modal_app(ModalTaskAppConfig(**opts, task_app_path=task_app_path))

__all__ = ["deploy_cmd"]
