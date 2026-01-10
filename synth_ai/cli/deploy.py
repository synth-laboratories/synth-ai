import asyncio
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast, get_args

import click

from synth_ai.cli.lib.apps.task_app import find_task_apps_in_cwd
from synth_ai.core.cfgs import CFDeployCfg, LocalDeployCfg, ModalDeployCfg
from synth_ai.core.env_utils import get_synth_and_env_keys
from synth_ai.core.integrations.cloudflare import deploy_app_tunnel
from synth_ai.core.integrations.modal import deploy_app_modal
from synth_ai.core.paths import print_paths_formatted
from synth_ai.core.telemetry import flush_logger, log_error, log_info
from synth_ai.core.uvicorn import deploy_app_uvicorn

RuntimeType: TypeAlias = Literal["local", "modal", "tunnel"]


@click.command()
# --- Arguments ---
@click.argument(
    "runtime", type=click.Choice(get_args(RuntimeType)), required=False, default="local"
)
@click.argument("task_app_path", type=click.Path(path_type=Path), required=False)
# --- Universal option(s) ---
@click.option(
    "--env",
    "env_file",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    required=True,
    help="Path to .env file to use (required)",
)
@click.option("--force", is_flag=True, default=False, help="Skip task app validation")
# --- Local runtime-only options ---
@click.option("--trace/--no-trace", default=True, help="Enable or disable trace output")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option(
    "--wait/--no-wait",
    "wait",
    default=False,
    help="Wait for deployment to complete (blocking mode). Default is non-blocking (background)",
)
# --- Tunnel runtime-only options ---
@click.option(
    "--tunnel-mode",
    type=click.Choice(["quick", "managed"], case_sensitive=False),
    default="quick",
    help="Tunnel mode: quick (ephemeral) or managed (stable)",
)
@click.option(
    "--tunnel-subdomain",
    type=str,
    default=None,
    help="Custom subdomain for managed tunnel (e.g., 'my-company')",
)
@click.option(
    "--keep-alive/--background",
    "keep_alive",
    default=False,
    help="(Deprecated: use --wait) Keep tunnel alive (blocking mode). Default is background (non-blocking)",
)
# --- Modal runtime-only options ---
@click.option(
    "--modal-app", type=click.Path(path_type=Path), help="Enter the path to your Modal app"
)
@click.option("--name", default=None, help="Override Modal app name")
@click.option("--modal-mode", default="deploy", help="Mode: deploy or serve")
@click.option(
    "--modal-cli",
    type=click.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    default=None,
    help="Path to Modal CLI",
)
@click.option("--dry-run", is_flag=True, help="Print Modal command without executing")
def deploy(
    runtime: RuntimeType,
    task_app_path: Path | None,
    env_file: Path,
    force: bool,
    trace: bool,
    host: str,
    port: int,
    wait: bool,
    tunnel_mode: str,
    tunnel_subdomain: str | None,
    keep_alive: bool,
    modal_app: Path | None,
    name: str | None,
    modal_mode: str,
    modal_cli: Path | None,
    dry_run: bool,
) -> None:
    ctx: dict[str, Any] = {
        "runtime": runtime,
        "task_app_path": str(task_app_path),
        "env_file": str(env_file),
        "force": force,
        "trace": trace,
        "host": host,
        "port": port,
        "wait": wait,
        "tunnel_mode": tunnel_mode,
        "tunnel_subdomain": tunnel_subdomain,
        "keep_alive": keep_alive,
        "modal_app": str(modal_app) if modal_app else None,
        "name": name,
        "modal_mode": modal_mode,
        "modal_cli": str(modal_cli) if modal_cli else None,
        "dry_run": dry_run,
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

        synth_api_key, env_api_key = get_synth_and_env_keys(env_file)
        if keep_alive:
            wait = True

        # if not kwargs.get("force", False):
        #     validate_task_app(task_app_path)

        match runtime:
            case "local":
                log_info("starting local deploy", ctx=ctx)
                deploy_app_uvicorn(
                    LocalDeployCfg.create(  # type: ignore[call-arg, arg-type]
                        task_app_path=task_app_path,
                        env_api_key=env_api_key,  # type: ignore[arg-type]
                        trace=trace,
                        host=host,
                        port=port,
                    )
                )
            case "modal":
                log_info("starting modal deploy", ctx=ctx)
                deploy_app_modal(
                    ModalDeployCfg.create_from_kwargs(
                        task_app_path=task_app_path,
                        synth_api_key=synth_api_key,
                        env_api_key=env_api_key,
                        modal_app=modal_app,
                        name=name,
                        modal_mode=modal_mode,
                        modal_cli=modal_cli,
                        dry_run=dry_run,
                    )
                )
            case "tunnel":
                log_info("starting tunnel deploy", ctx=ctx)
                if tunnel_mode == "managed" and not synth_api_key:
                    raise RuntimeError(
                        "SYNTH_API_KEY required for managed tunnel mode. "
                        "Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env"
                    )
                asyncio.run(
                    deploy_app_tunnel(
                        CFDeployCfg.create(
                            task_app_path=task_app_path,
                            env_api_key=env_api_key,
                            host=host,
                            port=port,
                            mode=cast(Literal["quick", "managed"], tunnel_mode),
                            subdomain=tunnel_subdomain,
                            trace=trace,
                        ),
                        env_file,
                        keep_alive=keep_alive,
                        wait=wait,
                    )
                )

    except Exception as exc:
        ctx["error"] = type(exc).__name__
        ctx["error_message"] = str(exc)
        log_error("deploy command failed", ctx=ctx)
        click.echo(f"{exc}", err=True)

    finally:
        log_info("deploy command completed", ctx=ctx)
        flush_logger(0.5)
