import os
import typing
from typing import Literal

import click

from synth_ai.cli.lib.env import (
    mask_str,
    resolve_env_var,
    write_env_var_to_dotenv,
    write_env_var_to_json,
)
from synth_ai.core.auth import fetch_credentials_from_web_browser

SourceType = Literal[
    "web",
    "local"
]


@click.command()
@click.option(
    "--source",
    type=click.Choice(
        list(typing.get_args(SourceType)),
        case_sensitive=False
    ),
    default="web",
    show_default=True,
    help="Source for credentials: 'web' for browser authentication, 'local' for environment variables"
)
@click.option(
    "--approve",
    is_flag=True,
    default=False,
    help="Approve automatically opening web browser for authentication"
)
def setup_cmd(
    source: SourceType = "web",
    approve: bool = False
) -> None:
    credentials = {}
    match source:
        case "local":
            credentials["SYNTH_API_KEY"] = resolve_env_var("SYNTH_API_KEY")
            credentials["ENVIRONMENT_API_KEY"] = resolve_env_var("ENVIRONMENT_API_KEY")
        case "web":
            if not approve:
                approve = click.confirm(
                    "This will open your web browser for authentication. Continue?",
                    default=True
                )
                if not approve:
                    return
            credentials = fetch_credentials_from_web_browser()

    required = {
        "SYNTH_API_KEY",
        "ENVIRONMENT_API_KEY"
    }
    missing = [k for k in required if not credentials.get(k)]
    if missing:
        raise ValueError(f"Missing credential values: {', '.join(missing)}")

    for k, v in credentials.items():
        write_env_var_to_json(k, v, "~/.synth-ai/config.json")
        write_env_var_to_dotenv(k, v)
        os.environ[k] = v
        print(f"Loaded {k}={mask_str(v)} to process environment")
