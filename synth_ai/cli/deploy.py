import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, TypeAlias, get_args

import click
from synth_ai.cfgs import LocalDeployCfg, ModalDeployCfg
from synth_ai.modal import deploy_app_modal
from synth_ai.utils import (
    PromptedChoiceOption,
    PromptedChoiceType,
    PromptedPathOption,
    flush_logger,
    log_error,
    log_info,
    read_env_var_from_file,
    validate_task_app,
)
from synth_ai.uvicorn import deploy_app_uvicorn

RuntimeType: TypeAlias = Literal[
    "local",
    "modal"
]


RUNTIME_MSG = SimpleNamespace(
    init="[deploy]",
    local="[deploy --runtime local]",
    modal="[deploy --runtime modal]",
)


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
    help="Enter the path to your task app",
)
@click.option(
    "--runtime",
    "runtime",
    cls=PromptedChoiceOption,
    type=PromptedChoiceType(get_args(RuntimeType)),
    required=True
)
# --- Optional option ---
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
    help="Enable or disable trace output"
)
@click.option(
    "--host",
    "host",
    default="127.0.0.1", 
    help="Host to bind to"
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
    help="Enter the path to your Modal app"
)
@click.option(
    "--name",
    "task_app_name",
    default=None,
    help="Override Modal app name"
)
@click.option(
    "--modal-mode",
    "cmd_arg",
    default="deploy",
    help="Mode: deploy or serve"
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
    help="Path to Modal CLI",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    help="Print Modal command without executing"
)
def deploy_cmd(
    task_app_path: Path,
    runtime: RuntimeType,
    **kwargs
) -> None:
    try:
        log_info("deploy command invoked", ctx={"runtime": runtime})
        env_file_path = kwargs.pop("env_path", None)
        file_synth_api_key = None
        file_env_api_key = None
        if env_file_path is not None:
            file_synth_api_key = read_env_var_from_file("SYNTH_API_KEY", env_file_path)
            file_env_api_key = read_env_var_from_file("ENVIRONMENT_API_KEY", env_file_path)
        env_synth_api_key = os.environ.get("SYNTH_API_KEY")
        env_env_api_key = os.environ.get("ENVIRONMENT_API_KEY")
        synth_api_key = file_synth_api_key or env_synth_api_key
        env_api_key = file_env_api_key or env_env_api_key
        if not synth_api_key:
            raise RuntimeError("SYNTH_API_KEY not in process environment. Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env")
        if not env_api_key:
            log_error("ENVIRONMENT_API_KEY missing", ctx={})
            raise RuntimeError("ENVIRONMENT_API_KEY not in process environment. Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env")
        
        validate_task_app(task_app_path)

        match runtime:
            case "local":
                log_info("starting local deploy")
                deploy_app_uvicorn(LocalDeployCfg.create(
                    task_app_path=task_app_path,
                    env_api_key=env_api_key,
                    trace = bool(kwargs.get("trace", True)),
                    host = str(kwargs.get("host", "127.0.0.1")),
                    port = int(kwargs.get("port", 8000))
                ))
            case "modal":
                log_info("starting modal deploy")
                deploy_app_modal(ModalDeployCfg.create_from_kwargs(
                    task_app_path=task_app_path,
                    synth_api_key=synth_api_key,
                    env_api_key=env_api_key,
                    **kwargs
                ))
        log_info("deploy command completed", context={"runtime": runtime})
    except Exception as exc:
        failure_ctx = {
            "runtime": runtime,
            "task_app": str(task_app_path),
            "error": type(exc).__name__,
        }
        log_error("deploy command failed", ctx=failure_ctx)
        click.echo(f"{exc}", err=True)
    finally:
        flush_logger(0.5)
