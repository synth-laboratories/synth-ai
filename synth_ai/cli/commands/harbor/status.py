"""Status command for Harbor deployments."""

import time

import click


@click.command()
@click.argument("deployment_id")
@click.option(
    "--wait/--no-wait",
    default=False,
    help="Wait for build to complete",
)
@click.option(
    "--timeout",
    type=int,
    default=600,
    help="Wait timeout in seconds",
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
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def status(
    deployment_id: str,
    wait: bool,
    timeout: int,
    api_key: str | None,
    backend_url: str,
    output_json: bool,
):
    """Get Harbor deployment status.

    Example:
        synth harbor status abc-123-def
        synth harbor status abc-123-def --wait
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    from synth_ai.sdk.harbor import HarborDeploymentUploader

    uploader = HarborDeploymentUploader(
        backend_url=backend_url,
        api_key=api_key,
    )

    def show_status(data: dict) -> None:
        if output_json:
            import json

            click.echo(json.dumps(data, indent=2, default=str))
            return

        status_color = {
            "ready": "green",
            "building": "yellow",
            "pending": "blue",
            "failed": "red",
        }.get(data.get("status", ""), "white")

        click.echo(f"Deployment: {data.get('name', deployment_id)}")
        click.echo(f"  ID: {data.get('id', deployment_id)}")
        click.echo(f"  Status: {click.style(data.get('status', 'unknown'), fg=status_color)}")

        if data.get("snapshot_id"):
            click.echo(f"  Snapshot: {data.get('snapshot_id')}")

        if data.get("error"):
            click.echo(f"  Error: {click.style(data.get('error'), fg='red')}")

        builds = data.get("builds", [])
        if builds:
            click.echo(f"\n  Recent builds ({len(builds)}):")
            for b in builds[:3]:
                b_status = b.get("status", "unknown")
                b_color = {
                    "completed": "green",
                    "building": "yellow",
                    "pending": "blue",
                    "failed": "red",
                }.get(b_status, "white")
                click.echo(f"    - {b.get('id', '?')[:8]}: {click.style(b_status, fg=b_color)}")

    try:
        if wait:
            click.echo(f"Waiting for deployment {deployment_id} to be ready...")
            start_time = time.time()

            while True:
                data = uploader.get_deployment_status(deployment_id)
                current_status = data.get("status", "unknown")

                if current_status == "ready":
                    click.echo(click.style("\nDeployment ready!", fg="green"))
                    show_status(data)
                    return

                if current_status == "failed":
                    click.echo(click.style("\nDeployment failed!", fg="red"))
                    show_status(data)
                    raise click.ClickException("Build failed")

                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise click.ClickException(
                        f"Timeout after {timeout}s. Status: {current_status}"
                    )

                click.echo(f"  Status: {current_status} ({int(elapsed)}s elapsed)")
                time.sleep(5)
        else:
            data = uploader.get_deployment_status(deployment_id)
            show_status(data)

    except Exception as e:
        if "404" in str(e):
            raise click.ClickException(f"Deployment not found: {deployment_id}") from None
        raise click.ClickException(f"Failed to get status: {e}") from None
