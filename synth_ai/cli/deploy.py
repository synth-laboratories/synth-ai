import asyncio
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast, get_args

import click

from synth_ai.cli.lib.apps.task_app import find_task_apps_in_cwd
from synth_ai.cli.lib.env import get_synth_and_env_keys
from synth_ai.core.cfgs import CFDeployCfg, LocalDeployCfg, ModalDeployCfg
from synth_ai.core.integrations.cloudflare import deploy_app_tunnel
from synth_ai.core.integrations.modal import deploy_app_modal
from synth_ai.core.paths import print_paths_formatted
from synth_ai.core.telemetry import flush_logger, log_error, log_info
from synth_ai.core.uvicorn import deploy_app_uvicorn

RuntimeType: TypeAlias = Literal[
    "local",
    "modal",
    "tunnel"
]


@click.command()
# --- Arguments ---
@click.argument(
    "runtime",
    type=click.Choice(get_args(RuntimeType)),
    required=False,
    default="local"
)
@click.argument(
    "task_app_path",
    type=click.Path(path_type=Path),
    required=False
)
# --- Universal option(s) ---
@click.option(
    "--env", 
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path
    ),
    required=True,
    help="Path to .env file to use (required)"
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Skip task app validation"
)
# --- Local runtime-only options ---
@click.option(
    "--trace/--no-trace",
    default=True,
    help="Enable or disable trace output"
)
@click.option(
    "--host",
    default="127.0.0.1", 
    help="Host to bind to"
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind to"
)
@click.option(
    "--wait/--no-wait",
    "wait",
    default=False,
    help="Wait for deployment to complete (blocking mode). Default is non-blocking (background)"
)
# --- Tunnel runtime-only options ---
@click.option(
    "--tunnel-mode",
    type=click.Choice(["quick", "managed"], case_sensitive=False),
    default="quick",
    help="Tunnel mode: quick (ephemeral) or managed (stable)"
)
@click.option(
    "--tunnel-subdomain",
    type=str,
    default=None,
    help="Custom subdomain for managed tunnel (e.g., 'my-company')"
)
@click.option(
    "--keep-alive/--background",
    default=False,
    help="(Deprecated: use --wait) Keep tunnel alive (blocking mode). Default is background (non-blocking)"
)
@click.option(
    "--wait/--no-wait",
    "wait",
    default=False,
    help="Wait for deployment to complete (blocking mode). Default is non-blocking (background)"
)
# --- Modal runtime-only options ---
@click.option(
    "--modal-app",
    help="Enter the path to your Modal app"
)
@click.option(
    "--name",
    default=None,
    help="Override Modal app name"
)
@click.option(
    "--modal-mode",
    default="deploy",
    help="Mode: deploy or serve"
)
@click.option(
    "--modal-cli",
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
    is_flag=True,
    help="Print Modal command without executing"
)
def deploy_cmd(
    runtime: RuntimeType,
    task_app_path: Path | None,
    **kwargs
) -> None:
    ctx: dict[str, Any] = {
        "runtime": runtime,
        "task_app_path": str(task_app_path),
        **kwargs
    }
    try:
        log_info("deploy command hit", ctx=ctx)

        if not task_app_path:
            available_task_apps = find_task_apps_in_cwd()
            if len(available_task_apps) == 1:
                task_app_path, _ = available_task_apps[0]
                print("Automatically selected task app at", task_app_path)
            else:
                if len(available_task_apps) == 0:
                    print("No task apps found in cwd.")
                    print("Validate your task app with: synth-ai task-app check [TASK_APP_PATH]")
                else:
                    print("Multiple task apps found. Please specify which one to use:")
                    print_paths_formatted(available_task_apps)
                print("Usage: synth-ai deploy [RUNTIME] [TASK_APP_PATH]")
                return None
        
        env_file = kwargs.get("env")
        synth_api_key, env_api_key = get_synth_and_env_keys(env_file)

        # if not kwargs.get("force", False):
        #     validate_task_app(task_app_path)

        match runtime:
            case "local":
                log_info("starting local deploy", ctx=ctx)
                deploy_app_uvicorn(
                    LocalDeployCfg.create(  # type: ignore[call-arg, arg-type]
                    task_app_path=task_app_path,
                        env_api_key=env_api_key,  # type: ignore[arg-type]
                    trace = bool(kwargs.get("trace", True)),
                    host = str(kwargs.get("host", "127.0.0.1")),
                    port = int(kwargs.get("port", 8000))
                    )
                )
            case "modal":
                log_info("starting modal deploy", ctx=ctx)
                deploy_app_modal(ModalDeployCfg.create_from_kwargs(
                    task_app_path=task_app_path,
                    synth_api_key=synth_api_key,
                    env_api_key=env_api_key,
                    **kwargs
                ))
            case "tunnel":
                log_info("starting tunnel deploy", ctx=ctx)
                tunnel_mode = kwargs.get("tunnel_mode", "managed")
                if tunnel_mode == "managed" and not synth_api_key:
                    raise RuntimeError(
                        "SYNTH_API_KEY required for managed tunnel mode. "
                        "Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env"
                    )
                asyncio.run(deploy_app_tunnel(
                    CFDeployCfg.create(
                        task_app_path=task_app_path,
                        env_api_key=env_api_key,
                        host=str(kwargs.get("host", "127.0.0.1")),
                        port=int(kwargs.get("port", 8000)),
                        mode=cast(Literal["quick", "managed"], tunnel_mode),
                        subdomain=kwargs.get("tunnel_subdomain"),
                        trace=bool(kwargs.get("trace", True))
                    ),
                    env_file,
                    keep_alive=bool(kwargs.get("keep_alive", False))
                ))
        
    except Exception as exc:
        ctx["error"] = type(exc).__name__
        ctx["error_message"] = str(exc)
        log_error("deploy command failed", ctx=ctx)
        click.echo(f"{exc}", err=True)
    
    finally:
        log_info("deploy command completed", ctx=ctx)
        flush_logger(0.5)
