"""Discover prompt patterns via backend."""

from __future__ import annotations

import click


@click.command()
@click.option("--job-id", required=True, help="Eval or prompt-learning job ID.")
@click.option("--backend", default="", help="Override backend base URL.")
@click.option("--max-calls", type=int, default=200, help="Limit the total calls analyzed.")
@click.option(
    "--min-support",
    type=int,
    default=3,
    help="Minimum calls per cluster before emitting a pattern.",
)
@click.option(
    "--cluster-threshold",
    type=float,
    default=0.85,
    help="Similarity threshold for clustering calls.",
)
@click.option(
    "--filter-noise/--no-filter-noise",
    default=True,
    help="Enable or disable noise filtering.",
)
@click.option("--max-patterns", type=int, default=5, help="Maximum patterns to return.")
@click.option("--out", "out_path", default="", help="Write TOML snippet to this file.")
@click.option("--debug", "debug_path", default="", help="Write debug JSON to this file.")
def pattern_discover(
    job_id: str,
    backend: str,
    max_calls: int,
    min_support: int,
    cluster_threshold: float,
    filter_noise: bool,
    max_patterns: int,
    out_path: str,
    debug_path: str,
) -> None:
    """Discover prompt patterns through the backend."""
    import asyncio
    import json
    from pathlib import Path

    from synth_ai.sdk.learning.pattern_discovery import (
        PatternDiscoveryClient,
        PatternDiscoveryRequest,
    )

    request = PatternDiscoveryRequest(
        job_id=job_id,
        max_calls=max_calls,
        filter_noise=filter_noise,
        cluster_threshold=cluster_threshold,
        min_support=min_support,
        max_patterns=max_patterns,
    )
    client = PatternDiscoveryClient(base_url=backend or None)
    result = asyncio.run(client.discover(request))

    warnings = result.get("warnings") or []
    if warnings:
        click.echo("Warnings:")
        for warning in warnings:
            click.echo(f"  - {warning}")

    patterns = result.get("patterns") or []
    if patterns:
        click.echo("Patterns:")
        for candidate in patterns:
            name = candidate.get("name") or candidate.get("id") or "pattern"
            support = candidate.get("support_count", "?")
            match_rate = candidate.get("match_rate")
            if isinstance(match_rate, (int, float)):
                match_text = f"{match_rate:.2f}"
            else:
                match_text = "?"
            click.echo(f"  - {name}: support={support} match_rate={match_text}")
    else:
        click.echo("No patterns discovered.")

    toml_snippet = result.get("toml_snippet") or ""
    if out_path:
        if not toml_snippet:
            raise click.ClickException("No TOML snippet available to write.")
        out_file = Path(out_path).expanduser()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(str(toml_snippet), encoding="utf-8")
        click.echo(f"Wrote TOML snippet: {out_file}")
    elif toml_snippet:
        click.echo(toml_snippet)

    if debug_path:
        debug_file = Path(debug_path).expanduser()
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        debug_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        click.echo(f"Wrote debug report: {debug_file}")
