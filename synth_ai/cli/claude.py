import os
import subprocess

import click
from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC
from synth_ai.utils import (
    PromptedChoiceOption,
    PromptedChoiceType,
    find_bin_path,
    resolve_env_var,
)


@click.command("claude")
@click.option(
    "--model",
    "model_name",
    cls=PromptedChoiceOption,
    type=PromptedChoiceType(MODEL_NAMES),
    required=True,
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
    required=False,
)
def claude_cmd(
    model_name: ModelName,
    force: bool = False,
    override_url: str | None = None
) -> None:

    print("Finding your installed Claude Code...")
    bin_path = find_bin_path("claude")
    if not bin_path:
        print("Failed to find Claude Code installation")
        print("Please install from: https://claude.com/claude-code")
        return
    print(f"Found Claude Code at {bin_path}")

    env = os.environ.copy()

    if override_url:
        url = f"{override_url.rstrip('/')}/{model_name}"
        print(f"Using override URL with model: {url}")
    else:
        url = f"{BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC}/{model_name}"

    env["ANTHROPIC_BASE_URL"] = url

    api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
    env["ANTHROPIC_AUTH_TOKEN"] = api_key
    env["SYNTH_API_KEY"] = api_key

    try:
        subprocess.run(["claude"], check=True, env=env)
    except subprocess.CalledProcessError:
        print("Failed to launch Claude Code")
