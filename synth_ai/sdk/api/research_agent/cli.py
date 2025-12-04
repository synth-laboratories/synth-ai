"""CLI for Research Agent jobs.

Provides the `synth-ai agent` command group for running research agent jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click


def _print_event(event: dict[str, Any]) -> None:
    """Print an event to stdout."""
    event_type = event.get("type", "")
    message = event.get("message", "")
    level = event.get("level", "info")

    # Color based on level/type
    if level == "error" or "failed" in event_type:
        click.secho(f"[{event_type}] {message}", fg="red")
    elif "completed" in event_type or "succeeded" in event_type:
        click.secho(f"[{event_type}] {message}", fg="green")
    elif "warning" in event_type or "budget" in event_type:
        click.secho(f"[{event_type}] {message}", fg="yellow")
    else:
        click.echo(f"[{event_type}] {message}")


@click.group()
def agent_cmd() -> None:
    """Research Agent commands for AI-assisted code analysis and optimization."""
    pass


@agent_cmd.command("run")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to TOML config file",
)
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(["scaffold_tuning", "evaluation", "trace_analysis"]),
    help="Algorithm to run (overrides config)",
)
@click.option(
    "--repo",
    "-r",
    "repo_url",
    help="Repository URL (overrides config)",
)
@click.option(
    "--branch",
    "-b",
    "repo_branch",
    default="main",
    help="Repository branch",
)
@click.option(
    "--backend",
    type=click.Choice(["daytona", "modal", "docker"]),
    default="daytona",
    help="Container backend to use",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o",
    help="Model for the agent to use",
)
@click.option(
    "--poll/--no-poll",
    default=True,
    help="Poll for completion and stream events",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=3600.0,
    help="Timeout in seconds (for --poll)",
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
    help="Backend API URL",
)
def run_cmd(
    config_path: Optional[Path],
    algorithm: Optional[str],
    repo_url: Optional[str],
    repo_branch: str,
    backend: str,
    model: str,
    poll: bool,
    timeout: float,
    api_key: Optional[str],
    backend_url: str,
) -> None:
    """Run a research agent job.

    You can provide configuration via a TOML file or command-line options.

    Examples:

        # From config file
        synth-ai agent run --config my_config.toml --poll

        # Quick scaffold tuning job
        synth-ai agent run \\
            --algorithm scaffold_tuning \\
            --repo https://github.com/your-org/repo \\
            --backend daytona

    """
    from .config import OptimizationTool, ResearchConfig
    from .job import ResearchAgentJob, ResearchAgentJobConfig

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        sys.exit(1)

    # Build config
    if config_path:
        click.echo(f"Loading config from {config_path}...")
        config = ResearchAgentJobConfig.from_toml(config_path)
        # api_key is guaranteed to be str at this point (checked above)
        assert api_key is not None, "api_key should be set by this point"
        config.api_key = api_key
        config.backend_url = backend_url

        # Apply CLI overrides
        if algorithm:
            # Map algorithm string to optimization tool if needed
            # Note: The algorithm parameter is kept for backward compatibility
            # but the actual optimization is controlled by config.research.tools
            pass  # Algorithm is embedded in the research config from TOML
        if repo_url:
            config.repo_url = repo_url
        if repo_branch != "main":
            config.repo_branch = repo_branch
        if backend != "daytona":
            config.backend = backend  # type: ignore
        if model != "gpt-4o":
            config.model = model
    else:
        # Build from CLI options
        if not algorithm:
            click.secho("Error: --algorithm is required when not using --config", fg="red", err=True)
            sys.exit(1)
        if not repo_url:
            click.secho("Error: --repo is required when not using --config", fg="red", err=True)
            sys.exit(1)

        # Create a minimal ResearchConfig
        # The algorithm parameter maps to the optimization tool
        tools = [OptimizationTool.MIPRO]  # Default to MIPRO for CLI usage
        research = ResearchConfig(
            task_description=f"Research job via CLI with {algorithm}",
            tools=tools,
        )

        config = ResearchAgentJobConfig(
            research=research,
            repo_url=repo_url,  # type: ignore[arg-type]
            repo_branch=repo_branch,
            backend=backend,  # type: ignore
            model=model,
            backend_url=backend_url,
            api_key=api_key,  # type: ignore[arg-type]
        )

    # Create and submit job
    job = ResearchAgentJob(config=config)

    click.echo("Submitting research job...")
    click.echo(f"  Repository: {config.repo_url}")
    click.echo(f"  Branch: {config.repo_branch}")
    click.echo(f"  Backend: {config.backend}")
    click.echo(f"  Model: {config.model}")
    click.echo(f"  Tools: {', '.join(t.value for t in config.research.tools)}")

    try:
        job_id = job.submit()
        click.secho(f"Job submitted: {job_id}", fg="green")
    except Exception as e:
        click.secho(f"Failed to submit job: {e}", fg="red", err=True)
        sys.exit(1)

    if not poll:
        click.echo(f"\nTo check status: synth-ai agent status {job_id}")
        return

    # Poll for completion
    click.echo("\nPolling for completion...")
    try:
        result = job.poll_until_complete(
            timeout=timeout,
            poll_interval=5.0,
            on_event=_print_event,
        )
        click.secho(f"\nJob completed: {result.get('status', 'unknown')}", fg="green")

        # Print summary
        if result.get("best_metric_value") is not None:
            click.echo(f"Best metric value: {result['best_metric_value']}")
        if result.get("current_iteration"):
            click.echo(f"Iterations: {result['current_iteration']}")

    except TimeoutError:
        click.secho(f"\nJob timed out after {timeout}s", fg="yellow", err=True)
        click.echo(f"Job ID: {job_id}")
        click.echo("Use 'synth-ai agent status' to check progress")
        sys.exit(1)
    except RuntimeError as e:
        click.secho(f"\nJob failed: {e}", fg="red", err=True)
        sys.exit(1)


@agent_cmd.command("status")
@click.argument("job_id")
@click.option(
    "--api-key",
    envvar="SYNTH_API_KEY",
    help="Synth API key",
)
@click.option(
    "--backend-url",
    envvar="SYNTH_BACKEND_URL",
    default="https://api.usesynth.ai",
    help="Backend API URL",
)
def status_cmd(job_id: str, api_key: Optional[str], backend_url: str) -> None:
    """Get status of a research agent job."""
    from .job import ResearchAgentJob

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        sys.exit(1)

    job = ResearchAgentJob.from_id(job_id, backend_url=backend_url, api_key=api_key)

    try:
        status = job.get_status()
        click.echo(f"Job ID: {job_id}")
        click.echo(f"Status: {status.get('status', 'unknown')}")
        click.echo(f"Algorithm: {status.get('algorithm', 'unknown')}")
        click.echo(f"Backend: {status.get('backend', 'unknown')}")

        if status.get("current_iteration"):
            total = status.get("total_iterations", "?")
            click.echo(f"Progress: {status['current_iteration']}/{total}")

        if status.get("best_metric_value") is not None:
            click.echo(f"Best metric: {status['best_metric_value']}")

        if status.get("error"):
            click.secho(f"Error: {status['error']}", fg="red")

    except Exception as e:
        click.secho(f"Failed to get status: {e}", fg="red", err=True)
        sys.exit(1)


@agent_cmd.command("events")
@click.argument("job_id")
@click.option(
    "--since",
    type=int,
    default=0,
    help="Show events after this sequence number",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    help="Follow events in real-time",
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
    help="Backend API URL",
)
def events_cmd(
    job_id: str,
    since: int,
    follow: bool,
    api_key: Optional[str],
    backend_url: str,
) -> None:
    """Get events from a research agent job."""
    import time

    from .job import ResearchAgentJob

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        sys.exit(1)

    job = ResearchAgentJob.from_id(job_id, backend_url=backend_url, api_key=api_key)
    last_seq = since

    try:
        while True:
            events = job.get_events(since_seq=last_seq)
            for event in events:
                _print_event(event)
                last_seq = max(last_seq, event.get("seq", 0))

            if not follow:
                break

            # Check if job is done
            status = job.get_status()
            if status.get("status") in ("succeeded", "failed", "canceled"):
                break

            time.sleep(2.0)

    except KeyboardInterrupt:
        click.echo("\nStopped following events")
    except Exception as e:
        click.secho(f"Failed to get events: {e}", fg="red", err=True)
        sys.exit(1)


@agent_cmd.command("cancel")
@click.argument("job_id")
@click.option(
    "--api-key",
    envvar="SYNTH_API_KEY",
    help="Synth API key",
)
@click.option(
    "--backend-url",
    envvar="SYNTH_BACKEND_URL",
    default="https://api.usesynth.ai",
    help="Backend API URL",
)
def cancel_cmd(job_id: str, api_key: Optional[str], backend_url: str) -> None:
    """Cancel a running research agent job."""
    from .job import ResearchAgentJob

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        sys.exit(1)

    job = ResearchAgentJob.from_id(job_id, backend_url=backend_url, api_key=api_key)

    if job.cancel():
        click.secho(f"Cancellation requested for job {job_id}", fg="yellow")
    else:
        click.secho(f"Failed to cancel job {job_id}", fg="red", err=True)
        sys.exit(1)


@agent_cmd.command("results")
@click.argument("job_id")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Write results to file (JSON)",
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
    help="Backend API URL",
)
def results_cmd(
    job_id: str,
    output: Optional[Path],
    api_key: Optional[str],
    backend_url: str,
) -> None:
    """Get results from a completed research agent job."""
    import json

    from .job import ResearchAgentJob

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        sys.exit(1)

    job = ResearchAgentJob.from_id(job_id, backend_url=backend_url, api_key=api_key)

    try:
        results = job.get_results()

        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results written to {output}")
        else:
            click.echo(json.dumps(results, indent=2))

    except Exception as e:
        click.secho(f"Failed to get results: {e}", fg="red", err=True)
        sys.exit(1)


def register(cli: Any) -> None:
    """Register the agent command group with the main CLI."""
    cli.add_command(agent_cmd, name="agent")
