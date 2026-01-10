"""Setup command."""

import click


@click.command()
@click.option(
    "--source",
    type=click.Choice(["web", "local"]),
    default="web",
    show_default=True,
    help="Source: 'web' for browser, 'local' for env vars.",
)
@click.option(
    "--approve",
    is_flag=True,
    default=False,
    help="Approve automatically opening web browser.",
)
def setup(source: str, approve: bool) -> None:
    """Configure Synth AI credentials."""
    from synth_ai.core.auth import run_setup

    run_setup(source=source, skip_confirm=approve, confirm_callback=click.confirm)
