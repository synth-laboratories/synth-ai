

import typing
from pathlib import Path

import click

from synth_ai.core.integrations.mcp.claude import ClaudeConfig

Agent = typing.Literal[
    "claude",
]


@click.command()
@click.argument(
    "agent",
    type=click.Choice(
        list(typing.get_args(Agent)),
        case_sensitive=False
    ),
    default = "claude"
)
@click.option(
    "--config-path",
    "config_path",
    type=Path,
    default=None
)
def mcp_cmd(agent: Agent, config_path: Path | None) -> None:
    match agent:
        case "claude":
            target_path = config_path or ClaudeConfig.get_default_config_path()
            config = ClaudeConfig(target_path)
            config.update_mcp_config()