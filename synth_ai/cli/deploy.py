import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, TypeAlias, get_args

import click
from synth_ai.cfgs import LocalDeployCfg, ModalDeployCfg
from synth_ai.utils import (
    PromptedChoiceOption,
    PromptedChoiceType,
    PromptedPathOption,
    deploy_modal_app,
    get_default_modal_bin_path,
    read_env_var_from_file,
    validate_task_app,
)
from synth_ai.uvicorn import deploy_app_uvicorn

RuntimeType: TypeAlias = Literal[
    "local",
    "modal"
]
RUNTIMES = get_args(RuntimeType)
RUNTIME_MSG = SimpleNamespace(
    init="[deploy]",
    local="[deploy --runtime local]",
    modal="[deploy --runtime modal]",
)

MODAL_RUNTIME_OPTIONS = [
    "task_app_name",
    "cmd_arg",
    "modal_bin_path",
    "dry_run",
    "modal_app_path",
]


@click.command()
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
# --- Optional option
@click.option(
    "--env",
    "env_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path
    ),
    help="Path to .env file to use"
)
# --- Local runtime-only options ---
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
# --- Modal runtime-only options ---
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
def deploy_cmd(
    task_app_path: Path,
    runtime: RuntimeType,
    **kwargs
) -> None:
    
    env_file_path = kwargs.pop("env_path", None)
    file_env_api_key = None
    if env_file_path is not None:
        file_env_api_key = read_env_var_from_file("ENVIRONMENT_API_KEY", env_file_path)
    env_env_api_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not file_env_api_key and not env_env_api_key:
        raise click.ClickException("ENVIRONMENT_API_KEY not in process environment. Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env")
    
    validate_task_app(task_app_path)

    match runtime:
        case "local":
            deploy_app_uvicorn(LocalDeployCfg.create(
                task_app_path,
                env_api_key=file_env_api_key or env_env_api_key,
                trace = bool(kwargs.get("trace", True)),
                host = str(kwargs.get("host", "127.0.0.1")),
                port = int(kwargs.get("port", 8000))
            ))
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
            deploy_modal_app(ModalDeployCfg(**opts, task_app_path=task_app_path))
 
