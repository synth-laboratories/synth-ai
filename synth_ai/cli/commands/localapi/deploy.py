"""Deploy a LocalAPI as a cloud task app."""

from __future__ import annotations

import json
from typing import Iterable

import click


def _parse_kv_pairs(values: Iterable[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise click.ClickException(f"Invalid env var '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise click.ClickException(f"Invalid env var '{raw}'. KEY cannot be empty.")
        env_vars[key] = value
    return env_vars


def _build_entrypoint(
    entrypoint: str | None,
    app: str | None,
    port: int,
    entrypoint_mode: str,
) -> tuple[str, str]:
    if entrypoint and app:
        raise click.ClickException("Use either --entrypoint or --app, not both.")
    if entrypoint:
        return entrypoint, "stdio" if entrypoint_mode == "command" else entrypoint_mode
    if not app:
        raise click.ClickException("Provide --app or --entrypoint to define how to run the server.")
    if entrypoint_mode not in ("command", "stdio"):
        raise click.ClickException("--app requires --entrypoint-mode=stdio (or command alias).")
    return f"python -m uvicorn {app} --host 0.0.0.0 --port {port}", "stdio"


@click.command()
@click.option(
    "--name",
    "-n",
    required=True,
    help="Deployment name (org-unique).",
)
@click.option(
    "--app",
    help="Uvicorn app path (e.g. my_module:app).",
)
@click.option(
    "--entrypoint",
    "-e",
    help="Entrypoint command to start the LocalAPI server.",
)
@click.option(
    "--entrypoint-mode",
    type=click.Choice(["stdio", "file", "command"]),
    default="stdio",
    show_default=True,
    help="Entrypoint mode for Harbor (use 'stdio' for server processes).",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="Port for the LocalAPI server (used with --app).",
)
@click.option(
    "--dockerfile",
    "-d",
    type=click.Path(exists=True),
    default="./Dockerfile",
    show_default=True,
    help="Path to Dockerfile.",
)
@click.option(
    "--context",
    "-c",
    type=click.Path(exists=True),
    default=".",
    show_default=True,
    help="Build context directory.",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    show_default=True,
    help="Timeout in seconds (30-3600).",
)
@click.option(
    "--cpu",
    type=int,
    default=2,
    show_default=True,
    help="CPU cores (1-8).",
)
@click.option(
    "--memory",
    type=int,
    default=4096,
    show_default=True,
    help="Memory in MB (512-32768).",
)
@click.option(
    "--disk",
    type=int,
    default=10240,
    show_default=True,
    help="Disk in MB (1024-102400).",
)
@click.option(
    "--env",
    multiple=True,
    help="Environment variable to set (KEY=VALUE). Can be provided multiple times.",
)
@click.option(
    "--description",
    help="Optional deployment description.",
)
@click.option(
    "--wait/--no-wait",
    default=False,
    help="Wait for build to complete.",
)
@click.option(
    "--build-timeout",
    type=int,
    default=600,
    show_default=True,
    help="Max time to wait for build completion (seconds).",
)
@click.option(
    "--api-key",
    envvar="SYNTH_API_KEY",
    help="Synth API key.",
)
@click.option(
    "--backend-url",
    envvar="SYNTH_BACKEND_URL",
    default="https://api.usesynth.ai",
    show_default=True,
    help="Synth backend URL.",
)
@click.option(
    "--json-output/--no-json-output",
    default=False,
    help="Emit machine-readable JSON output.",
)
def deploy(
    name: str,
    app: str | None,
    entrypoint: str | None,
    entrypoint_mode: str,
    port: int,
    dockerfile: str,
    context: str,
    timeout: int,
    cpu: int,
    memory: int,
    disk: int,
    env: tuple[str, ...],
    description: str | None,
    wait: bool,
    build_timeout: int,
    api_key: str | None,
    backend_url: str,
    json_output: bool,
) -> None:
    """Deploy a LocalAPI to the cloud via Harbor."""
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key.")

    def _echo(message: str) -> None:
        if not json_output:
            click.echo(message)

    entrypoint_cmd, entrypoint_mode = _build_entrypoint(entrypoint, app, port, entrypoint_mode)
    env_vars = _parse_kv_pairs(env)

    from synth_ai.sdk.harbor import HarborBuildSpec, HarborLimits, upload_harbor_deployment

    _echo(f"Deploying LocalAPI '{name}'...")
    _echo(f"  Dockerfile: {dockerfile}")
    _echo(f"  Context: {context}")
    _echo(f"  Entrypoint: {entrypoint_cmd}")

    try:
        spec = HarborBuildSpec(
            name=name,
            dockerfile_path=dockerfile,
            context_dir=context,
            entrypoint=entrypoint_cmd,
            entrypoint_mode=entrypoint_mode,
            description=description,
            env_vars=env_vars,
            limits=HarborLimits(
                timeout_s=timeout,
                cpu_cores=cpu,
                memory_mb=memory,
                disk_mb=disk,
            ),
            metadata={"localapi": True},
        )

        result = upload_harbor_deployment(
            spec,
            api_key=api_key,
            backend_url=backend_url,
            auto_build=True,
            wait_for_ready=wait,
            build_timeout_s=float(build_timeout),
        )

        deployment_key = result.deployment_name or result.deployment_id
        task_app_url = f"{backend_url.rstrip('/')}/api/harbor/deployments/{deployment_key}"

        if json_output:
            click.echo(
                json.dumps(
                    {
                        "deployment_id": result.deployment_id,
                        "build_id": result.build_id,
                        "status": result.status,
                        "snapshot_id": result.snapshot_id,
                        "task_app_url": task_app_url,
                        "task_app_api_key_env": "SYNTH_API_KEY",
                    }
                )
            )
            return

        click.echo(click.style("Deployment created!", fg="green"))
        click.echo(f"  Deployment ID: {result.deployment_id}")
        if result.deployment_name:
            click.echo(f"  Deployment name: {result.deployment_name}")
        if result.build_id:
            click.echo(f"  Build ID: {result.build_id}")
        click.echo(f"  Status: {result.status}")
        if result.snapshot_id:
            click.echo(f"  Snapshot ID: {result.snapshot_id}")
        click.echo(f"  Task app URL: {task_app_url}")
        click.echo("  Task app API key: SYNTH_API_KEY")

        if not wait and result.status != "ready":
            click.echo("\nTo wait for build completion, run:")
            click.echo(f"  synth harbor status {deployment_key} --wait")
        if result.status == "ready":
            click.echo("\nLocalAPI is ready to accept /rollout traffic.")

    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None
    except Exception as e:
        raise click.ClickException(f"Deploy failed: {e}") from None
