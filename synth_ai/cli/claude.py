import os
import subprocess

import click
from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC
from synth_ai.utils import find_bin_path, install_bin, resolve_env_var, verify_bin, write_agents_md


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

    while True:
        bin_path = find_bin_path("claude")
        if bin_path:
            break
        if not install_bin(
            "Claude Code",
            ["curl -fsSL https://claude.ai/install.sh | bash"]
        ):
            print("Failed to find your installed Claude Code")
            print("Please install from: https://claude.com/claude-code")
            return
    print(f"Using Claude at {bin_path}")

    if not verify_bin(bin_path):
        print("Failed to verify Claude Code is runnable")
        return

    write_agents_md()
    env = os.environ.copy()

    if model_name is not None:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name={model_name} is invalid. Valid values for model_name: {MODEL_NAMES}")
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
