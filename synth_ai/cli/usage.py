"""CLI command for displaying org usage and limits.

Usage:
    synth-ai usage          # Show usage summary
    synth-ai usage --json   # Output as JSON
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import click


def _format_number(n: int | float) -> str:
    """Format a number for display (e.g., 1000 -> 1K)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    if isinstance(n, float):
        return f"{n:.1f}"
    return str(n)


def _progress_bar(percent: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int(width * min(percent, 100) / 100)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}]"


def _format_metric(name: str, metric, indent: str = "  ") -> str:
    """Format a single usage metric for display."""
    percent = metric.percent_used
    bar = _progress_bar(percent)
    used_str = _format_number(metric.used)
    limit_str = _format_number(metric.limit)
    return f"{indent}{name}: {bar} {used_str} / {limit_str} ({percent:.1f}%)"


@click.command()
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output as JSON",
)
def usage_cmd(as_json: bool) -> None:
    """Display org usage and rate limits.

    Shows current usage across all Synth APIs with progress bars
    indicating how much of each limit has been consumed.
    """
    try:
        from synth_ai.sdk.usage import UsageClient
    except ImportError as e:
        click.echo(f"Error importing usage client: {e}", err=True)
        sys.exit(1)

    try:
        client = UsageClient()
        usage = client.get()
    except Exception as e:
        click.echo(f"Error fetching usage: {e}", err=True)
        sys.exit(1)

    if as_json:
        # Convert to dict and output as JSON
        output = {
            "org_id": usage.org_id,
            "tier": usage.tier,
            "period": {
                "daily_start": usage.period.daily_start.isoformat(),
                "monthly_start": usage.period.monthly_start.isoformat(),
            },
            "apis": {
                "inference": {
                    "requests_per_min": asdict(usage.apis.inference.requests_per_min),
                    "tokens_per_day": asdict(usage.apis.inference.tokens_per_day),
                    "spend_cents_per_month": asdict(usage.apis.inference.spend_cents_per_month),
                },
                "judges": {
                    "evaluations_per_day": asdict(usage.apis.judges.evaluations_per_day),
                },
                "prompt_opt": {
                    "jobs_per_day": asdict(usage.apis.prompt_opt.jobs_per_day),
                    "rollouts_per_day": asdict(usage.apis.prompt_opt.rollouts_per_day),
                    "spend_cents_per_day": asdict(usage.apis.prompt_opt.spend_cents_per_day),
                },
                "rl": {
                    "jobs_per_month": asdict(usage.apis.rl.jobs_per_month),
                    "gpu_hours_per_month": asdict(usage.apis.rl.gpu_hours_per_month),
                },
                "sft": {
                    "jobs_per_month": asdict(usage.apis.sft.jobs_per_month),
                    "gpu_hours_per_month": asdict(usage.apis.sft.gpu_hours_per_month),
                },
                "research": {
                    "jobs_per_month": asdict(usage.apis.research.jobs_per_month),
                    "agent_spend_cents_per_month": asdict(usage.apis.research.agent_spend_cents_per_month),
                },
            },
            "totals": {
                "spend_cents_per_month": asdict(usage.totals.spend_cents_per_month),
            },
        }
        click.echo(json.dumps(output, indent=2))
        return

    # Pretty print format
    click.echo("")
    click.echo(f"Org: {usage.org_id} ({usage.tier} tier)")
    click.echo("")

    # Inference
    click.echo("Inference")
    click.echo(_format_metric("requests/min", usage.apis.inference.requests_per_min))
    click.echo(_format_metric("tokens/day", usage.apis.inference.tokens_per_day))
    click.echo(_format_metric("spend/month", usage.apis.inference.spend_cents_per_month))
    click.echo("")

    # Judges
    click.echo("Judges")
    click.echo(_format_metric("evaluations/day", usage.apis.judges.evaluations_per_day))
    click.echo("")

    # Prompt Optimization
    click.echo("Prompt Optimization")
    click.echo(_format_metric("jobs/day", usage.apis.prompt_opt.jobs_per_day))
    click.echo(_format_metric("rollouts/day", usage.apis.prompt_opt.rollouts_per_day))
    click.echo(_format_metric("spend/day", usage.apis.prompt_opt.spend_cents_per_day))
    click.echo("")

    # RL Training
    click.echo("RL Training")
    click.echo(_format_metric("jobs/month", usage.apis.rl.jobs_per_month))
    click.echo(_format_metric("gpu_hours/month", usage.apis.rl.gpu_hours_per_month))
    click.echo("")

    # SFT Training
    click.echo("SFT Training")
    click.echo(_format_metric("jobs/month", usage.apis.sft.jobs_per_month))
    click.echo(_format_metric("gpu_hours/month", usage.apis.sft.gpu_hours_per_month))
    click.echo("")

    # Research Agents
    click.echo("Research Agents")
    click.echo(_format_metric("jobs/month", usage.apis.research.jobs_per_month))
    click.echo(_format_metric("agent_spend/month", usage.apis.research.agent_spend_cents_per_month))
    click.echo("")

    # Totals
    click.echo("Total Spend")
    click.echo(_format_metric("spend/month", usage.totals.spend_cents_per_month))
    click.echo("")
