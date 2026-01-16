"""OpenCode command."""

import click


@click.command()
@click.option("--model", "model_name", type=str, default=None)
@click.option("--force", is_flag=True, help="Prompt for API keys even if cached.")
@click.option("--synth-base-url", type=str, default=None)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to TOML config file.",
)
def opencode(
    model_name: str | None,
    force: bool,
    config_path: str | None,
    synth_base_url: str | None,
) -> None:
    """Launch OpenCode with optional Synth backend routing."""
    from pathlib import Path

    from synth_ai.core.agents.opencode import run_opencode

    config = Path(config_path) if config_path else None
    run_opencode(
        model_name=model_name,
        force=force,
        config_path=config,
        synth_base_url=synth_base_url,
    )
