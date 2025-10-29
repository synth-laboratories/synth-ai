"""Instructions on Docs → https://usesynth.ai/cli-cmds/setup"""

import click
from synth_ai.auth.credentials import fetch_credentials_from_web_browser_session


@click.command("setup")
@click.option(
    "--local",
    is_flag=True,
    help="Load your credentials from your local machine"
)
@click.option(
    "--dev",
    is_flag=True
)
def setup_cmd(local: bool, dev: bool) -> None:
    fetch_credentials_from_web_browser_session(
        browser=not local,
        prod=not dev
    )
