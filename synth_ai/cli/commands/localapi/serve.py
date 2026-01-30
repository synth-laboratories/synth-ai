"""Serve a LocalAPI with a tunnel."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import click
from synth_ai.core.tunnels import (
    PortConflictBehavior,
    TunnelBackend,
    TunneledLocalAPI,
    acquire_port,
    wait_for_health_check,
)
from synth_ai.sdk.localapi._impl.server import run_server_background
from synth_ai.sdk.localapi.auth import (
    ENVIRONMENT_API_KEY_NAME,
    ensure_localapi_auth,
    mint_environment_api_key,
)


def _load_module(path: Path) -> ModuleType:
    module_name = f"localapi_{path.stem}_{int(time.time())}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_localapi(module: ModuleType, path: Path) -> str | None:
    from fastapi import FastAPI

    if not hasattr(module, "app"):
        return "LocalAPI must define an 'app' variable"

    app = module.app
    if not isinstance(app, FastAPI):
        return f"'app' must be a FastAPI instance, got {type(app).__name__}"

    routes = {route.path for route in app.routes}
    if "/health" not in routes:
        return "LocalAPI missing /health endpoint (use create_local_api() to create your app)"
    if "/rollout" not in routes:
        return "LocalAPI missing /rollout endpoint (use create_local_api() to create your app)"

    if hasattr(module, "get_dataset_size"):
        get_dataset_size = module.get_dataset_size
        if callable(get_dataset_size):
            try:
                result = get_dataset_size()
                if not isinstance(result, int):
                    return f"get_dataset_size() must return an int, got {type(result).__name__}"
                if result <= 0:
                    return f"get_dataset_size() must return a positive number, got {result}"
            except NotImplementedError as exc:
                return f"get_dataset_size() not implemented: {exc}"
            except Exception as exc:
                return f"get_dataset_size() failed: {exc}"

    if hasattr(module, "get_sample"):
        get_sample = module.get_sample
        if callable(get_sample):
            try:
                result = get_sample(0)
                if not isinstance(result, dict):
                    return f"get_sample() must return a dict, got {type(result).__name__}"
            except NotImplementedError as exc:
                return f"get_sample() not implemented: {exc}"
            except Exception as exc:
                return f"get_sample(0) failed: {exc}"

    if hasattr(module, "score_response"):
        score_response = module.score_response
        if callable(score_response):
            sig = inspect.signature(score_response)
            params = list(sig.parameters.keys())
            if len(params) < 2:
                return f"score_response() must accept (response, sample), got {params}"
            try:
                result = score_response("test response", {"input": "test", "expected": "test"})
                if not isinstance(result, (int, float)):
                    return f"score_response() must return a number, got {type(result).__name__}"
            except NotImplementedError as exc:
                return f"score_response() not implemented: {exc}"
            except Exception:
                pass

    return None


def _resolve_env_api_key(
    api_key: str | None,
    backend_url: str | None,
    explicit_env_key: str | None,
) -> str:
    if explicit_env_key:
        os.environ[ENVIRONMENT_API_KEY_NAME] = explicit_env_key
        return explicit_env_key
    if api_key:
        return ensure_localapi_auth(backend_base=backend_url, synth_api_key=api_key)
    env_key = mint_environment_api_key()
    os.environ[ENVIRONMENT_API_KEY_NAME] = env_key
    return env_key


async def _serve_async(
    *,
    localapi_path: Path,
    backend: TunnelBackend,
    port: int,
    host: str,
    port_conflict: PortConflictBehavior,
    api_key: str | None,
    backend_url: str,
    env_api_key: str | None,
    json_output: bool,
) -> None:
    env_key = _resolve_env_api_key(api_key, backend_url, env_api_key)

    sys.path.insert(0, str(localapi_path.parent))
    try:
        module = _load_module(localapi_path)
    finally:
        if sys.path and sys.path[0] == str(localapi_path.parent):
            sys.path.pop(0)

    validation_error = _validate_localapi(module, localapi_path)
    if validation_error:
        raise click.ClickException(validation_error)

    app = module.app
    resolved_port = acquire_port(port, on_conflict=port_conflict)
    proc = run_server_background(app, resolved_port, host=host)

    try:
        await wait_for_health_check("localhost", resolved_port, api_key=env_key, timeout=30.0)
    except Exception as exc:
        proc.terminate()
        raise click.ClickException(f"Health check failed: {exc}") from exc

    tunnel: TunneledLocalAPI | None = None
    url = f"http://localhost:{resolved_port}"
    worker_token: str | None = None

    if backend != TunnelBackend.Localhost:
        tunnel = await TunneledLocalAPI.create(
            local_port=resolved_port,
            backend=backend,
            api_key=api_key,
            env_api_key=env_key,
            backend_url=backend_url,
        )
        url = tunnel.url
        worker_token = tunnel.worker_token

    if json_output:
        click.echo(
            json.dumps(
                {
                    "task_app_url": url,
                    "task_app_worker_token": worker_token,
                    "task_app_api_key": None if worker_token else env_key,
                    "local_port": resolved_port,
                    "backend": backend.value,
                }
            )
        )
    else:
        click.echo("LocalAPI is running.")
        click.echo(f"  Task app URL: {url}")
        if worker_token:
            click.echo(f"  Task app worker token: {worker_token}")
        else:
            click.echo(f"  Task app API key: {env_key}")
        click.echo(f"  Local port: {resolved_port}")
        click.echo(f"  Tunnel backend: {backend.value}")

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        if tunnel:
            tunnel.close()
        proc.terminate()


@click.command()
@click.argument("localapi_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--backend",
    type=click.Choice([b.value for b in TunnelBackend]),
    default=TunnelBackend.SynthTunnel.value,
    show_default=True,
    help="Tunnel backend to use.",
)
@click.option(
    "--port",
    type=int,
    default=8001,
    show_default=True,
    help="Local port for the LocalAPI server.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind the LocalAPI server.",
)
@click.option(
    "--port-conflict",
    type=click.Choice([b.value for b in PortConflictBehavior]),
    default=PortConflictBehavior.FIND_NEW.value,
    show_default=True,
    help="Behavior if the port is already in use.",
)
@click.option(
    "--api-key",
    envvar="SYNTH_API_KEY",
    help="Synth API key (required for managed tunnels).",
)
@click.option(
    "--backend-url",
    envvar="SYNTH_BACKEND_URL",
    default="https://api.usesynth.ai",
    show_default=True,
    help="Synth backend URL.",
)
@click.option(
    "--env-api-key",
    envvar=ENVIRONMENT_API_KEY_NAME,
    help="Environment API key (optional).",
)
@click.option(
    "--json-output/--no-json-output",
    default=False,
    help="Emit machine-readable JSON output.",
)
def serve(
    localapi_path: str,
    backend: str,
    port: int,
    host: str,
    port_conflict: str,
    api_key: str | None,
    backend_url: str,
    env_api_key: str | None,
    json_output: bool,
) -> None:
    """Serve a LocalAPI locally and expose it through a tunnel."""
    backend_enum = TunnelBackend(backend)
    conflict = PortConflictBehavior(port_conflict)

    if (
        backend_enum
        in (
            TunnelBackend.SynthTunnel,
            TunnelBackend.CloudflareManagedLease,
            TunnelBackend.CloudflareManagedTunnel,
        )
        and not api_key
    ):
        raise click.ClickException("SYNTH_API_KEY is required for managed tunnels.")

    asyncio.run(
        _serve_async(
            localapi_path=Path(localapi_path).resolve(),
            backend=backend_enum,
            port=port,
            host=host,
            port_conflict=conflict,
            api_key=api_key,
            backend_url=backend_url,
            env_api_key=env_api_key,
            json_output=json_output,
        )
    )
