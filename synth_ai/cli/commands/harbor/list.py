"""List command for Harbor deployments."""

import click


@click.command()
@click.option(
    "--status",
    "-s",
    type=click.Choice(["pending", "building", "ready", "failed"]),
    help="Filter by status",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=50,
    help="Maximum results",
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
def list_deployments(
    status: str | None,
    limit: int,
    api_key: str | None,
    backend_url: str,
    output_json: bool,
):
    """List Harbor deployments.

    Example:
        synth harbor list
        synth harbor list --status ready
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    from synth_ai.sdk.harbor import HarborDeploymentUploader

    uploader = HarborDeploymentUploader(
        backend_url=backend_url,
        api_key=api_key,
    )

    try:
        deployments = uploader.list_deployments(status=status, limit=limit)

        if output_json:
            import json

            click.echo(json.dumps(deployments, indent=2, default=str))
            return

        if not deployments:
            click.echo("No deployments found.")
            return

        click.echo(f"Found {len(deployments)} deployment(s):\n")

        for d in deployments:
            status_color = {
                "ready": "green",
                "building": "yellow",
                "pending": "blue",
                "failed": "red",
            }.get(d.get("status", ""), "white")

            click.echo(f"  {d.get('name', 'unnamed')}")
            click.echo(f"    ID: {d.get('id')}")
            click.echo(f"    Status: {click.style(d.get('status', 'unknown'), fg=status_color)}")
            if d.get("snapshot_id"):
                click.echo(f"    Snapshot: {d.get('snapshot_id')}")
            click.echo()

    except Exception as e:
        raise click.ClickException(f"Failed to list deployments: {e}") from None
