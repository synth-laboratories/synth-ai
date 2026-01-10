"""OpenCode command."""

import click


@click.command()
@click.option("--model", "model_name", type=str, default=None)
@click.option("--force", is_flag=True, help="Prompt for API keys even if cached.")
@click.option("--url", "override_url", type=str, default=None)
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
    override_url: str | None,
    config_path: str | None,
) -> None:
    """Launch OpenCode with optional Synth backend routing."""
    from pathlib import Path

    from synth_ai.core.agents.opencode import run_opencode

    config = Path(config_path) if config_path else None
    run_opencode(
        model_name=model_name,
        force=force,
        override_url=override_url,
        config_path=config,
    )
