"""Codex command."""

import click


@click.command()
@click.option("--model", "model_name", type=str, default=None)
@click.option("--force", is_flag=True, help="Prompt for API keys even if cached.")
@click.option("--url", "override_url", type=str, default=None)
@click.option(
    "--wire-api",
    type=click.Choice(["chat", "responses"]),
    default=None,
    help="API wire format: chat or responses.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to TOML config file.",
)
def codex(
    model_name: str | None,
    force: bool,
    override_url: str | None,
    wire_api: str | None,
    config_path: str | None,
) -> None:
    """Launch Codex with optional Synth backend routing."""
    from pathlib import Path

    from synth_ai.core.agents.codex import run_codex

    config = Path(config_path) if config_path else None
    run_codex(
        model_name=model_name,
        force=force,
        override_url=override_url,
        wire_api=wire_api,
        config_path=config,
    )
