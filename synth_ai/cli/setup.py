import click
from synth_ai.auth.credentials import fetch_credentials_from_web_browser_session


@click.command("setup")
@click.option(
    "--dev",
    is_flag=True,
    help="Use the development environment instead of production.",
)
def setup(dev: bool) -> None:
    fetch_credentials_from_web_browser_session(prod=not dev)
