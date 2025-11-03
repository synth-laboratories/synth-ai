import subprocess

import click
from synth_ai.types import ModelName
from synth_ai.utils.agent_launchers import (
    AgentPreparationError,
    prepare_claude_invocation,
)


@click.command("claude")
@click.option(
    "--model",
    "model_name",
    type=str,
    default=None
)
@click.option(
    "--force",
    is_flag=True,
    help="Prompt for API keys even if cached values exist."
)
@click.option(
    "--url",
    "override_url",
    type=str,
    default=None,
)
def claude_cmd(
    model_name: ModelName | None = None,
    force: bool = False,
    override_url: str | None = None
) -> None:

    try:
        invocation = prepare_claude_invocation(
            model_name=model_name,
            force=force,
            override_url=override_url,
        )
    except AgentPreparationError:
        return

    try:
        subprocess.run(invocation.command, check=True, env=invocation.env)
    except subprocess.CalledProcessError:
        print("Failed to launch Claude Code")
