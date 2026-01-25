"""Build command for Harbor deployments."""

import click


@click.command()
@click.argument("deployment_id")
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
def build(
    deployment_id: str,
    wait: bool,
    api_key: str | None,
    backend_url: str,
):
    """Trigger a build for a Harbor deployment.

    Example:
        synth harbor build abc-123-def
        synth harbor build abc-123-def --wait
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    from synth_ai.sdk.harbor import HarborDeploymentUploader

    uploader = HarborDeploymentUploader(
        backend_url=backend_url,
        api_key=api_key,
    )

    try:
        click.echo(f"Triggering build for deployment {deployment_id}...")

        result = uploader.trigger_build(deployment_id)
        build_id = result.get("build_id", "unknown")

        click.echo(click.style("Build triggered!", fg="green"))
        click.echo(f"  Build ID: {build_id}")

        if wait:
            click.echo("\nWaiting for build to complete...")
            status = uploader.wait_for_build(deployment_id)
            click.echo(click.style("Build complete!", fg="green"))
            click.echo(f"  Status: {status.get('status')}")
            if status.get("snapshot_id"):
                click.echo(f"  Snapshot: {status.get('snapshot_id')}")
        else:
            click.echo("\nTo wait for completion, run:")
            click.echo(f"  synth harbor status {deployment_id} --wait")

    except Exception as e:
        if "404" in str(e):
            raise click.ClickException(f"Deployment not found: {deployment_id}") from None
        raise click.ClickException(f"Build failed: {e}") from None
