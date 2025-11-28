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
    """Research Agent commands for AI-assisted prompt optimization.

    Research Agents run in sandboxed environments and apply MIPRO optimization
    to improve prompt performance on your datasets.
    """
    pass


@agent_cmd.command("run")
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to TOML config file (recommended)",
)
@click.option(
    "--repo",
    "-r",
    "repo_url",
    help="Repository URL (alternative to config file)",
)
@click.option(
    "--branch",
    "-b",
    "repo_branch",
    default="main",
    help="Repository branch",
)
@click.option(
    "--task",
    "-t",
    "task_description",
    help="Task description for the agent",
)
@click.option(
    "--dataset",
    "-d",
    "dataset",
    help="HuggingFace dataset ID (e.g., PolyAI/banking77)",
)
@click.option(
    "--tool",
    type=click.Choice(["mipro"]),
    default="mipro",
    help="Optimization tool to use",
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
    default="gpt-5.1-codex-mini",
    help="Model for the agent to use",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="Reasoning effort level",
)
@click.option(
    "--max-agent-spend",
    type=float,
    default=25.0,
    help="Max spend for agent LLM calls (USD)",
)
@click.option(
    "--max-synth-spend",
    type=float,
    default=150.0,
    help="Max spend for optimization (USD)",
)
@click.option(
    "--iterations",
    "-n",
    type=int,
    default=10,
    help="Number of optimization iterations",
)
@click.option(
    "--poll/--no-poll",
    default=True,
    help="Poll for completion and stream events",
)
@click.option(
    "--timeout",
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
    repo_url: Optional[str],
    repo_branch: str,
    task_description: Optional[str],
    dataset: Optional[str],
    tool: str,
    backend: str,
    model: str,
    reasoning_effort: str,
    max_agent_spend: float,
    max_synth_spend: float,
    iterations: int,
    poll: bool,
    timeout: float,
    api_key: Optional[str],
    backend_url: str,
) -> None:
    """Run a research agent job for prompt optimization.

    Provide configuration via a TOML file (recommended) or command-line options.

    \b
    Examples:
        # From config file (recommended)
        synth-ai agent run --config research.toml --poll

        # Quick job with CLI options
        synth-ai agent run \\
            --repo https://github.com/your-org/repo \\
            --task "Optimize intent classification accuracy" \\
            --dataset PolyAI/banking77 \\
            --iterations 10

    \b
    TOML Config Format:
        [research_agent]
        repo_url = "https://github.com/your-org/repo"
        model = "gpt-5.1-codex-mini"
        max_agent_spend_usd = 25.0

        [research_agent.research]
        task_description = "Optimize classification accuracy"
        tools = ["mipro"]
        num_iterations = 10

        [[research_agent.research.datasets]]
        source_type = "huggingface"
        hf_repo_id = "PolyAI/banking77"
    """
    from .config import DatasetSource, OptimizationTool, ResearchConfig
    from .job import ResearchAgentJob, ResearchAgentJobConfig

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        click.echo("Set via --api-key or SYNTH_API_KEY environment variable")
        sys.exit(1)

    # Build config from TOML or CLI options
    if config_path:
        click.echo(f"Loading config from {config_path}...")
        config = ResearchAgentJobConfig.from_toml(config_path)
        config.api_key = api_key
        config.backend_url = backend_url

        # Apply CLI overrides if provided
        if repo_url:
            config.repo_url = repo_url
        if repo_branch != "main":
            config.repo_branch = repo_branch
        if backend != "daytona":
            config.backend = backend  # type: ignore
        if model != "gpt-5.1-codex-mini":
            config.model = model
        if reasoning_effort != "medium":
            config.reasoning_effort = reasoning_effort  # type: ignore
        if max_agent_spend != 25.0:
            config.max_agent_spend_usd = max_agent_spend
        if max_synth_spend != 150.0:
            config.max_synth_spend_usd = max_synth_spend
    else:
        # Build from CLI options
        if not repo_url and not task_description:
            click.secho(
                "Error: Either --config or (--repo and --task) required",
                fg="red",
                err=True,
            )
            click.echo("\nUse --config to load from a TOML file, or provide:")
            click.echo("  --repo URL      Repository URL")
            click.echo("  --task TEXT     Task description")
            click.echo("  --dataset ID    HuggingFace dataset (optional)")
            sys.exit(1)

        # Build datasets list
        datasets = []
        if dataset:
            datasets.append(
                DatasetSource(
                    source_type="huggingface",
                    hf_repo_id=dataset,
                    hf_split="train",
                )
            )

        # Build research config
        research = ResearchConfig(
            task_description=task_description or "Optimize prompt performance",
            tools=[OptimizationTool.MIPRO] if tool == "mipro" else [],
            datasets=datasets,
            num_iterations=iterations,
        )

        config = ResearchAgentJobConfig(
            research=research,
            repo_url=repo_url or "",
            repo_branch=repo_branch,
            backend=backend,  # type: ignore
            model=model,
            reasoning_effort=reasoning_effort,  # type: ignore
            max_agent_spend_usd=max_agent_spend,
            max_synth_spend_usd=max_synth_spend,
            backend_url=backend_url,
            api_key=api_key,
        )

    # Create and submit job
    job = ResearchAgentJob(config=config)

    click.echo("\n" + "=" * 50)
    click.echo("Research Agent Job")
    click.echo("=" * 50)
    click.echo(f"Repository: {config.repo_url or '(inline files)'}")
    if config.repo_url:
        click.echo(f"Branch: {config.repo_branch}")
    click.echo(f"Model: {config.model}")
    click.echo(f"Backend: {config.backend}")
    click.echo(f"Tool: {', '.join(t.value for t in config.research.tools)}")
    click.echo(f"Iterations: {config.research.num_iterations}")
    click.echo(f"Max Agent Spend: ${config.max_agent_spend_usd:.2f}")
    click.echo(f"Max Synth Spend: ${config.max_synth_spend_usd:.2f}")
    click.echo("=" * 50 + "\n")

    click.echo("Submitting job...")
    try:
        job_id = job.submit()
        click.secho(f"Job submitted: {job_id}", fg="green")
    except NotImplementedError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Failed to submit job: {e}", fg="red", err=True)
        sys.exit(1)

    if not poll:
        click.echo(f"\nTo check status: synth-ai agent status {job_id}")
        click.echo(f"To stream events: synth-ai agent events {job_id} --follow")
        return

    # Poll for completion
    click.echo("\nPolling for completion (this may take 15-30 minutes)...")
    try:
        result = job.poll_until_complete(
            timeout=timeout,
            poll_interval=15.0,
            on_event=_print_event,
        )

        status = result.get("status", "unknown")
        click.echo("\n" + "=" * 50)

        if status == "succeeded":
            click.secho(f"Job completed: {status}", fg="green")
        else:
            click.secho(f"Job completed: {status}", fg="yellow")

        # Print summary
        if result.get("best_metric_value") is not None:
            click.echo(f"Best metric: {result['best_metric_value']}")
        if result.get("baseline_metric_value") is not None:
            improvement = (
                (result["best_metric_value"] - result["baseline_metric_value"])
                / result["baseline_metric_value"]
                * 100
            )
            click.echo(f"Improvement: {improvement:+.1f}%")
        if result.get("current_iteration"):
            click.echo(f"Iterations: {result['current_iteration']}")

        click.echo("=" * 50)
        click.echo(f"\nTo get results: synth-ai agent results {job_id}")

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
    """Get status of a research agent job.

    Example:
        synth-ai agent status ra_abc123def456
    """
    from .job import ResearchAgentJob

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        sys.exit(1)

    job = ResearchAgentJob.from_id(job_id, backend_url=backend_url, api_key=api_key)

    try:
        status = job.get_status()

        click.echo(f"Job: {job_id}")

        job_status = status.get("status", "unknown")
        if job_status == "succeeded":
            click.secho(f"Status: {job_status}", fg="green")
        elif job_status == "failed":
            click.secho(f"Status: {job_status}", fg="red")
        elif job_status == "running":
            click.secho(f"Status: {job_status}", fg="cyan")
        else:
            click.echo(f"Status: {job_status}")

        if status.get("current_iteration"):
            total = status.get("total_iterations", "?")
            click.echo(f"Progress: {status['current_iteration']}/{total}")

        if status.get("best_metric_value") is not None:
            click.echo(f"Best metric: {status['best_metric_value']}")

        if status.get("created_at"):
            click.echo(f"Created: {status['created_at']}")

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
    """Stream events from a research agent job.

    Examples:
        # Show all events
        synth-ai agent events ra_abc123

        # Follow events in real-time
        synth-ai agent events ra_abc123 --follow
    """
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
    """Cancel a running research agent job.

    Example:
        synth-ai agent cancel ra_abc123def456
    """
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
    """Get results from a completed research agent job.

    Examples:
        # Print results to stdout
        synth-ai agent results ra_abc123

        # Save to file
        synth-ai agent results ra_abc123 -o results.json
    """
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


@agent_cmd.command("list")
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Number of jobs to show",
)
@click.option(
    "--status",
    "-s",
    "status_filter",
    type=click.Choice(["queued", "running", "succeeded", "failed"]),
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
    help="Backend API URL",
)
def list_cmd(
    limit: int,
    status_filter: Optional[str],
    api_key: Optional[str],
    backend_url: str,
) -> None:
    """List recent research agent jobs.

    Examples:
        # Show last 10 jobs
        synth-ai agent list

        # Show running jobs
        synth-ai agent list --status running

        # Show last 20 jobs
        synth-ai agent list -n 20
    """
    import httpx

    if not api_key:
        click.secho("Error: SYNTH_API_KEY is required", fg="red", err=True)
        sys.exit(1)

    url = f"{backend_url.rstrip('/')}/api/research-agent/jobs"
    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict[str, Any] = {"limit": limit}
    if status_filter:
        params["status"] = status_filter

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        jobs = data.get("jobs", [])

        if not jobs:
            click.echo("No jobs found")
            return

        click.echo(f"{'Job ID':<25} {'Status':<12} {'Created':<20} {'Model':<20}")
        click.echo("-" * 80)

        for job in jobs:
            job_id = job.get("job_id", "?")[:24]
            status = job.get("status", "?")
            created = job.get("created_at", "?")[:19] if job.get("created_at") else "?"
            model = job.get("model", "?")[:19]

            # Color status
            if status == "succeeded":
                status_str = click.style(status, fg="green")
            elif status == "failed":
                status_str = click.style(status, fg="red")
            elif status == "running":
                status_str = click.style(status, fg="cyan")
            else:
                status_str = status

            click.echo(f"{job_id:<25} {status_str:<21} {created:<20} {model:<20}")

    except httpx.HTTPStatusError as e:
        click.secho(f"Failed to list jobs: HTTP {e.response.status_code}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Failed to list jobs: {e}", fg="red", err=True)
        sys.exit(1)


def register(cli: Any) -> None:
    """Register the agent command group with the main CLI."""
    cli.add_command(agent_cmd, name="agent")
