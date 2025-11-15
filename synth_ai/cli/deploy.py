import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, TypeAlias, cast, get_args

import click
from synth_ai.cfgs import CloudflareTunnelDeployCfg, LocalDeployCfg, ModalDeployCfg
from synth_ai.cloudflare import deploy_app_tunnel
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
    "modal",
    "tunnel"
]


RUNTIME_MSG = SimpleNamespace(
    init="[deploy]",
    local="[deploy --runtime local]",
    modal="[deploy --runtime modal]",
    tunnel="[deploy --runtime tunnel]",
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
# --- Required option ---
@click.option(
    "--env",
    "env_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path
    ),
    required=True,
    help="Path to .env file to use (required)"
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
@click.option(
    "--wait/--no-wait",
    "wait",
    default=False,
    help="Wait for deployment to complete (blocking mode). Default is non-blocking (background)"
)
# --- Tunnel runtime-only options ---
@click.option(
    "--tunnel-mode",
    "tunnel_mode",
    type=click.Choice(["quick", "managed"], case_sensitive=False),
    default="quick",
    help="Tunnel mode: quick (ephemeral) or managed (stable)"
)
@click.option(
    "--tunnel-subdomain",
    "tunnel_subdomain",
    type=str,
    default=None,
    help="Custom subdomain for managed tunnel (e.g., 'my-company')"
)
@click.option(
    "--keep-alive/--background",
    "keep_alive",
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
    "modal_app_path",
    cls=PromptedPathOption,
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path
    ),
    file_type=".py",
    prompt_guard=lambda ctx: (ctx.params.get("runtime") not in ("local", "tunnel")),
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
        
        # .env file is required
        if env_file_path is None:
            raise RuntimeError("--env option is required. Please provide a path to a .env file: synth-ai deploy --env .env ...")
        
        # Check environment variables BEFORE loading provided .env (to see what's explicitly set)
        env_synth_api_key_before = os.environ.get("SYNTH_API_KEY")
        env_env_api_key_before = os.environ.get("ENVIRONMENT_API_KEY")
        
        # Load provided .env file (loads all variables into environment)
        env_file_path_resolved = None
        if env_file_path is not None:
            try:
                from dotenv import load_dotenv
                env_file_path_resolved = Path(env_file_path).expanduser().resolve()
                if env_file_path_resolved.exists():
                    load_dotenv(env_file_path_resolved, override=False)
                    log_info("loaded .env file", ctx={"env_file": str(env_file_path_resolved)})
                else:
                    log_error("env file not found", ctx={"env_file": str(env_file_path_resolved)})
                    raise RuntimeError(f"Environment file not found: {env_file_path_resolved}")
            except ImportError:
                # dotenv not available, fall back to reading specific vars
                pass
        
        # Read API keys directly from the provided .env file
        file_synth_api_key = None
        file_env_api_key = None
        if env_file_path is not None:
            try:
                file_synth_api_key = read_env_var_from_file("SYNTH_API_KEY", env_file_path)
                file_env_api_key = read_env_var_from_file("ENVIRONMENT_API_KEY", env_file_path)
            except Exception:
                pass  # Keys not in file
        
        # Check environment after .env load (in case provided .env set them, or they were already there)
        env_synth_api_key_after = os.environ.get("SYNTH_API_KEY")
        env_env_api_key_after = os.environ.get("ENVIRONMENT_API_KEY")
        
        # For local/tunnel: require keys to be EITHER in provided .env OR explicitly in environment
        # Use keys from provided .env file if present, otherwise from environment
        synth_api_key = file_synth_api_key or env_synth_api_key_after
        env_api_key = file_env_api_key or env_env_api_key_after
        
        # Validate API keys are present (required for local and tunnel deployments)
        # Keys must be EITHER in the provided .env file OR explicitly in the environment
        if runtime in ("local", "tunnel"):
            missing_keys = []
            if not synth_api_key:
                missing_keys.append("SYNTH_API_KEY")
            if not env_api_key:
                missing_keys.append("ENVIRONMENT_API_KEY")
            
            if missing_keys:
                env_file_hint = f" ({env_file_path_resolved})" if env_file_path_resolved else ""
                
                # Determine where keys came from (or didn't come from)
                synth_source = []
                env_source = []
                
                if file_synth_api_key:
                    synth_source.append(f"provided .env file{env_file_hint}")
                if env_synth_api_key_before:
                    synth_source.append("environment (set before deploy)")
                if env_synth_api_key_after and not env_synth_api_key_before and not file_synth_api_key:
                    synth_source.append("environment (auto-loaded from repo root .env)")
                
                if file_env_api_key:
                    env_source.append(f"provided .env file{env_file_hint}")
                if env_env_api_key_before:
                    env_source.append("environment (set before deploy)")
                if env_env_api_key_after and not env_env_api_key_before and not file_env_api_key:
                    env_source.append("environment (auto-loaded from repo root .env)")
                
                status_lines = [
                    f"Missing required API keys for {runtime} deployment: {', '.join(missing_keys)}",
                    "",
                    "For local and tunnel deployments, API keys must be present in EITHER:",
                    "  1. Your environment (export SYNTH_API_KEY=... export ENVIRONMENT_API_KEY=...), OR",
                    f"  2. The .env file provided via --env{env_file_hint}",
                    "",
                    "Current status:",
                ]
                
                if "SYNTH_API_KEY" in missing_keys:
                    status_lines.append("  SYNTH_API_KEY: ✗ missing")
                    if synth_source:
                        status_lines.append(f"    Found in: {', '.join(synth_source)} (but validation failed)")
                    else:
                        status_lines.append("    Not found in provided .env file or environment")
                else:
                    status_lines.append("  SYNTH_API_KEY: ✓ found")
                    if synth_source:
                        status_lines.append(f"    Source: {', '.join(synth_source)}")
                
                if "ENVIRONMENT_API_KEY" in missing_keys:
                    status_lines.append("  ENVIRONMENT_API_KEY: ✗ missing")
                    if env_source:
                        status_lines.append(f"    Found in: {', '.join(env_source)} (but validation failed)")
                    else:
                        status_lines.append("    Not found in provided .env file or environment")
                else:
                    status_lines.append("  ENVIRONMENT_API_KEY: ✓ found")
                    if env_source:
                        status_lines.append(f"    Source: {', '.join(env_source)}")
                
                raise RuntimeError("\n".join(status_lines))
        
        # For modal, SYNTH_API_KEY is required but ENVIRONMENT_API_KEY is optional
        if runtime == "modal":
            if not synth_api_key:
                raise RuntimeError(
                    "SYNTH_API_KEY is required for modal deployment. "
                    "Either set it in your environment or provide it in the .env file via --env .env"
                )
            if not env_api_key:
                log_error("ENVIRONMENT_API_KEY_MISSING", ctx={"runtime": runtime})
                raise RuntimeError(
                    "ENVIRONMENT_API_KEY is required for modal deployment. "
                    "Either set it in your environment or provide it in the .env file via --env .env"
                )
        
        validate_task_app(task_app_path)

        match runtime:
            case "local":
                log_info("starting local deploy")
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
                log_info("starting modal deploy")
                wait = bool(kwargs.get("wait", False))
                deploy_app_modal(ModalDeployCfg.create_from_kwargs(
                    task_app_path=task_app_path,
                    synth_api_key=synth_api_key,
                    env_api_key=env_api_key,
                    **kwargs
                ), wait=wait)
            case "tunnel":
                # For managed tunnels, SYNTH_API_KEY is required
                tunnel_mode = kwargs.get("tunnel_mode", "managed")
                if tunnel_mode == "managed" and not synth_api_key:
                    raise RuntimeError(
                        "SYNTH_API_KEY required for managed tunnel mode. "
                        "Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env"
                    )
                
                cfg = CloudflareTunnelDeployCfg.create(  # type: ignore[call-arg, arg-type]
                    task_app_path=task_app_path,
                    env_api_key=env_api_key,  # type: ignore[arg-type]
                    host=str(kwargs.get("host", "127.0.0.1")),
                    port=int(kwargs.get("port", 8000)),
                    mode=cast(Literal["quick", "managed"], tunnel_mode),
                    subdomain=kwargs.get("tunnel_subdomain"),
                    trace=bool(kwargs.get("trace", True)),
                )
                # Default to background mode (non-blocking), use --wait for blocking
                keep_alive = bool(kwargs.get("keep_alive", False))  # Deprecated, use wait instead
                wait = bool(kwargs.get("wait", False))
                asyncio.run(deploy_app_tunnel(cfg, env_file_path, keep_alive=keep_alive, wait=wait))
                # Note: deploy_app_tunnel prints the URL and status message internally
        log_info("deploy command completed", ctx={"runtime": runtime})
    except Exception as exc:
        log_error("deploy command failed", ctx={
            "runtime": runtime,
            "task_app": str(task_app_path),
            "error": type(exc).__name__,
            "error_message": str(exc)
        })
        click.echo(f"{exc}", err=True)
    finally:
        flush_logger(0.5)
