"""List command for baseline discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from synth_ai.sdk.baseline.config import BaselineConfig
from synth_ai.sdk.baseline.discovery import (
    BaselineChoice,
    discover_baseline_files,
    load_baseline_config_from_file,
)


@click.command("list")
@click.option(
    "--tag",
    multiple=True,
    help="Filter baselines by tag (can be specified multiple times)",
)
@click.option(
    "--metadata",
    type=str,
    help="Filter by metadata key-value pair (format: key=value)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed information about each baseline",
)
def list_command(tag: tuple[str, ...], metadata: Optional[str], verbose: bool) -> None:
    """List all available baseline files."""
    search_roots = [Path.cwd()]
    choices = discover_baseline_files(search_roots)
    
    if not choices:
        click.echo("No baseline files found.", err=True)
        click.echo("Create baseline files in examples/baseline/ or */*_baseline.py")
        return
    
    # Load configs for filtering
    configs: list[tuple[BaselineChoice, BaselineConfig]] = []
    for choice in choices:
        try:
            config = load_baseline_config_from_file(choice.baseline_id, choice.path)
            configs.append((choice, config))
        except Exception as e:
            if verbose:
                click.echo(f"Warning: Could not load {choice.baseline_id}: {e}", err=True)
            continue
    
    # Apply filters
    filtered_configs = configs
    
    if tag:
        tag_set = {t.lower() for t in tag}
        filtered_configs = [
            (c, config) for c, config in filtered_configs
            if any(config.matches_tag(t) for t in tag_set)
        ]
    
    if metadata:
        if "=" not in metadata:
            raise click.ClickException("--metadata must be in format key=value")
        key, value = metadata.split("=", 1)
        filtered_configs = [
            (c, config) for c, config in filtered_configs
            if config.matches_metadata(key.strip(), value.strip())
        ]
    
    if not filtered_configs:
        click.echo("No baselines match the specified filters.")
        return
    
    # Display results
    click.echo(f"Found {len(filtered_configs)} baseline(s):\n")
    
    for choice, config in filtered_configs:
        click.echo(f"  {config.baseline_id}")
        click.echo(f"    Name: {config.name}")
        if config.description:
            click.echo(f"    Description: {config.description}")
        if config.tags:
            click.echo(f"    Tags: {', '.join(config.tags)}")
        click.echo(f"    Splits: {', '.join(config.splits.keys())}")
        if verbose:
            click.echo(f"    Path: {choice.path}")
            if config.metadata:
                click.echo(f"    Metadata: {config.metadata}")
        click.echo()

