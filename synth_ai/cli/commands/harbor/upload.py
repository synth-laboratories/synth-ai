"""Upload command for Harbor deployments."""

import click


@click.command()
@click.option(
    "--name",
    "-n",
    required=True,
    help="Deployment name (org-unique)",
)
@click.option(
    "--dockerfile",
    "-d",
    type=click.Path(exists=True),
    default="./Dockerfile",
    help="Path to Dockerfile",
)
@click.option(
    "--context",
    "-c",
    type=click.Path(exists=True),
    default=".",
    help="Build context directory",
)
@click.option(
    "--entrypoint",
    "-e",
    default="run_rollout --input /tmp/rollout.json --output /tmp/result.json",
    help="Entrypoint command",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Timeout in seconds (30-3600)",
)
@click.option(
    "--cpu",
    type=int,
    default=2,
    help="CPU cores (1-8)",
)
@click.option(
    "--memory",
    type=int,
    default=4096,
    help="Memory in MB (512-32768)",
)
@click.option(
    "--wait/--no-wait",
    default=False,
    help="Wait for build to complete",
)
@click.option(
    "--api-key",
    envvar="SYNTH_API_KEY",
    help="Synth API key",
)
@click.option(
    "--backend-url",
    envvar="SYNTH_BACKEND_URL",
    default="https://api.usesynth.ai",
    help="Synth backend URL",
)
def upload(
    name: str,
    dockerfile: str,
    context: str,
    entrypoint: str,
    timeout: int,
    cpu: int,
    memory: int,
    wait: bool,
    api_key: str | None,
    backend_url: str,
):
    """Upload a new Harbor deployment.

    Example:
        synth harbor upload --name my-agent -d ./Dockerfile -c . --wait
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    from synth_ai.sdk.harbor import (
        HarborBuildSpec,
        HarborLimits,
        upload_harbor_deployment,
    )

    click.echo(f"Uploading deployment '{name}'...")
    click.echo(f"  Dockerfile: {dockerfile}")
    click.echo(f"  Context: {context}")

    try:
        spec = HarborBuildSpec(
            name=name,
            dockerfile_path=dockerfile,
            context_dir=context,
            entrypoint=entrypoint,
            limits=HarborLimits(
                timeout_s=timeout,
                cpu_cores=cpu,
                memory_mb=memory,
            ),
        )

        result = upload_harbor_deployment(
            spec,
            api_key=api_key,
            backend_url=backend_url,
            auto_build=True,
            wait_for_ready=wait,
        )

        click.echo(click.style("Upload successful!", fg="green"))
        click.echo(f"  Deployment ID: {result.deployment_id}")
        click.echo(f"  Build ID: {result.build_id}")
        click.echo(f"  Status: {result.status}")

        if result.snapshot_id:
            click.echo(f"  Snapshot ID: {result.snapshot_id}")

        if not wait and result.status != "ready":
            click.echo("\nTo wait for build completion, run:")
            click.echo(f"  synth harbor status {result.deployment_id} --wait")

    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None
    except Exception as e:
        raise click.ClickException(f"Upload failed: {e}") from None
