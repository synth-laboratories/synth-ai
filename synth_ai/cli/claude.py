"""Claude command."""

import click


@click.command()
@click.option(
    "--model",
    "model_name",
    type=str,
    default=None,
    help="Model name for routing through Synth backend.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Prompt for API keys even if cached.",
)
@click.option(
    "--url",
    "override_url",
    type=str,
    default=None,
    help="Custom backend URL.",
)
def claude(model_name: str | None, force: bool, override_url: str | None) -> None:
    """Launch Claude Code with optional Synth backend routing."""
    from synth_ai.core.agents.claude import run_claude

    run_claude(model_name=model_name, force=force, override_url=override_url)
