"""Claude Code launcher."""

import os
import subprocess

from synth_ai.core.agents.utils import write_agents_md
from synth_ai.core.bin import install_bin, verify_bin
from synth_ai.core.env import resolve_env_var
from synth_ai.core.paths import get_bin_path
from synth_ai.core.urls import synth_research_anthropic_base


def run_claude(model_name=None, force=False, synth_base_url=None):
    """Launch Claude Code with optional Synth backend routing.

    Args:
        model_name: Model name for routing through Synth backend
        force: Prompt for API keys even if cached
        synth_base_url: Backend URL override
    """
    while True:
        bin_path = get_bin_path("claude")
        if bin_path:
            break
        if not install_bin("Claude Code", ["curl -fsSL https://claude.ai/install.sh | bash"]):
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
        url = f"{synth_research_anthropic_base(synth_base_url)}/{model_name}"
        env["ANTHROPIC_BASE_URL"] = url
        synth_user_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        env["ANTHROPIC_AUTH_TOKEN"] = synth_user_key
        env["SYNTH_API_KEY"] = synth_user_key

    try:
        subprocess.run(["claude"], check=True, env=env)
    except subprocess.CalledProcessError:
        print("Failed to launch Claude Code")
