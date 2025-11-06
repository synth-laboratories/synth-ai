import os
import typing
from typing import Literal

import click
from synth_ai.auth.credentials import fetch_credentials_from_web_browser
from synth_ai.utils import (
    resolve_env_var,
    write_env_var_to_dotenv,
    write_env_var_to_json,
)

Sources = Literal[
    "web",
    "local"
]


@click.command()
@click.option(
    "--source",
    type=click.Choice(
        list(typing.get_args(Sources)),
        case_sensitive=False
    ),
    default="web",
    show_default=True
)
def setup_cmd(source: Sources = "web") -> None:
    credentials = {}
    match source:
        case "local":
            credentials["SYNTH_API_KEY"] = resolve_env_var("SYNTH_API_KEY")
            credentials["ENVIRONMENT_API_KEY"] = resolve_env_var("ENVIRONMENT_API_KEY")
        case "web":
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
