"""Instance commands for Harbor deployments."""

import click
import httpx


@click.group()
def instances():
    """Manage Harbor deployment instances.

    Instances allow pre-building snapshots per seed for parallel GEPA execution.
    """
    pass


@instances.command("create")
@click.argument("deployment_id")
@click.option(
    "--count",
    "-c",
    type=int,
    default=100,
    help="Number of instances to create (seeds 0..count-1)",
)
@click.option(
    "--start-seed",
    type=int,
    default=0,
    help="Starting seed value",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build instances after creation",
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
def create_instances(
    deployment_id: str,
    count: int,
    start_seed: int,
    build: bool,
    api_key: str | None,
    backend_url: str,
):
    """Create instances for a deployment.

    Example:
        synth harbor instances create abc-123 --count 100
        synth harbor instances create abc-123 --count 50 --start-seed 100
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        click.echo(f"Creating {count} instances for deployment {deployment_id}...")
        click.echo(f"  Seeds: {start_seed} to {start_seed + count - 1}")

        # Create instances batch
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{backend_url}/api/harbor/deployments/{deployment_id}/instances/batch",
                json={"count": count, "start_seed": start_seed},
                headers=headers,
            )

            if response.status_code >= 400:
                raise click.ClickException(f"API error: {response.text}")

            data = response.json()

            click.echo(click.style("Instances created!", fg="green"))
            click.echo(f"  Total: {data.get('total', count)}")
            click.echo(f"  Ready: {data.get('ready_count', 0)}")
            click.echo(f"  Building: {data.get('building_count', 0)}")
            click.echo(f"  Failed: {data.get('failed_count', 0)}")

            # Build instances if requested
            if build:
                click.echo("\nBuilding all instances...")
                build_response = client.post(
                    f"{backend_url}/api/harbor/deployments/{deployment_id}/instances/build-all",
                    headers=headers,
                )

                if build_response.status_code >= 400:
                    click.echo(click.style("Warning: Build failed", fg="yellow"))
                else:
                    build_data = build_response.json()
                    click.echo(click.style("Build complete!", fg="green"))
                    click.echo(f"  Ready: {build_data.get('ready_count', 0)}")

    except httpx.RequestError as e:
        raise click.ClickException(f"Request failed: {e}") from None


@instances.command("list")
@click.argument("deployment_id")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["pending", "building", "ready", "failed"]),
    help="Filter by status",
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
def list_instances(
    deployment_id: str,
    status: str | None,
    api_key: str | None,
    backend_url: str,
    output_json: bool,
):
    """List instances for a deployment.

    Example:
        synth harbor instances list abc-123
        synth harbor instances list abc-123 --status ready
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    try:
        params = {}
        if status:
            params["status"] = status

        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{backend_url}/api/harbor/deployments/{deployment_id}/instances",
                params=params,
                headers=headers,
            )

            if response.status_code >= 400:
                raise click.ClickException(f"API error: {response.text}")

            data = response.json()

            if output_json:
                import json

                click.echo(json.dumps(data, indent=2, default=str))
                return

            instances = data.get("instances", [])
            click.echo(f"Instances for deployment {deployment_id}:")
            click.echo(f"  Total: {data.get('total', len(instances))}")
            click.echo(f"  Ready: {data.get('ready_count', 0)}")
            click.echo(f"  Building: {data.get('building_count', 0)}")
            click.echo(f"  Failed: {data.get('failed_count', 0)}")

            if instances:
                click.echo("\nInstances:")
                for inst in instances[:20]:  # Show first 20
                    status_color = {
                        "ready": "green",
                        "building": "yellow",
                        "pending": "blue",
                        "failed": "red",
                    }.get(inst.get("status", ""), "white")
                    click.echo(
                        f"  Seed {inst.get('seed', '?'):4d}: "
                        f"{click.style(inst.get('status', 'unknown'), fg=status_color)}"
                    )

                if len(instances) > 20:
                    click.echo(f"  ... and {len(instances) - 20} more")

    except httpx.RequestError as e:
        raise click.ClickException(f"Request failed: {e}") from None


@instances.command("build")
@click.argument("deployment_id")
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
def build_instances(
    deployment_id: str,
    api_key: str | None,
    backend_url: str,
):
    """Build all pending instances for a deployment.

    Example:
        synth harbor instances build abc-123
    """
    if not api_key:
        raise click.ClickException("API key required. Set SYNTH_API_KEY or use --api-key")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    try:
        click.echo(f"Building all instances for deployment {deployment_id}...")

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{backend_url}/api/harbor/deployments/{deployment_id}/instances/build-all",
                headers=headers,
            )

            if response.status_code >= 400:
                raise click.ClickException(f"API error: {response.text}")

            data = response.json()

            click.echo(click.style("Build complete!", fg="green"))
            click.echo(f"  Total instances: {data.get('total_instances', 0)}")
            click.echo(f"  Ready: {data.get('ready_count', 0)}")

    except httpx.RequestError as e:
        raise click.ClickException(f"Request failed: {e}") from None
