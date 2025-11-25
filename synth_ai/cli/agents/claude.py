import os
import subprocess

import click

from synth_ai.cli.lib.agents import write_agents_md
from synth_ai.cli.lib.bin import install_bin, verify_bin
from synth_ai.cli.lib.env import resolve_env_var
from synth_ai.core.paths import get_bin_path
from synth_ai.core.urls import BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC


@click.command("claude")
@click.option(
    "--model",
    "model_name",
    type=str,
    default=None,
    help="Model name to use (Claude Code uses Anthropic models directly)"
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
    model_name: str | None = None,
    force: bool = False,
    override_url: str | None = None
) -> None:
    """Launch Claude Code with optional Synth backend routing.
    
    Note: Claude Code uses Anthropic models directly. The --model option
    is for routing through Synth's Anthropic-compatible backend.
    """
    while True:
        bin_path = get_bin_path("claude")
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
        # Route through Synth's Anthropic-compatible backend
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
