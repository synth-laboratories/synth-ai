"""Codex CLI launcher."""

import os
import subprocess

from synth_ai.core.agents.utils import write_agents_md
from synth_ai.core.bin import install_bin, verify_bin
from synth_ai.core.env import resolve_env_var
from synth_ai.core.paths import get_bin_path
from synth_ai.core.urls import BACKEND_URL_BASE, BACKEND_URL_SYNTH_RESEARCH_OPENAI
from synth_ai.data.enums import SYNTH_MODEL_NAMES


def run_codex(model_name=None, force=False, override_url=None, wire_api=None, config_path=None):
    """Launch Codex with optional Synth backend routing."""
    while True:
        bin_path = get_bin_path("codex")
        if bin_path:
            break
        if not install_bin("Codex", ["brew install codex", "npm install -g @openai/codex"]):
            print("Failed to find your installed Codex")
            print("Please install from: https://developers.openai.com/codex/cli/")
            return
    print(f"Using Codex at {bin_path}")

    if not verify_bin(bin_path):
        print("Failed to verify Codex is runnable")
        return

    write_agents_md()
    env = os.environ.copy()
    override_args = []

    api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
    if override_url:
        base_url = override_url.rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[:-4]
    else:
        base_url = BACKEND_URL_BASE

    if model_name is not None:
        if model_name not in SYNTH_MODEL_NAMES:
            raise ValueError(f"model_name={model_name} is invalid. Valid: {SYNTH_MODEL_NAMES}")
        if override_url:
            url = override_url
            print("Using override URL:", url)
        else:
            url = BACKEND_URL_SYNTH_RESEARCH_OPENAI

        if wire_api is None:
            wire_api = "responses" if "/responses" in url or url.endswith("/responses") else "chat"

        provider_config_parts = [
            'name="Synth"',
            f'base_url="{url}"',
            'env_key="OPENAI_API_KEY"',
            f'wire_api="{wire_api}"',
        ]
        provider_config = "{" + ",".join(provider_config_parts) + "}"

        config_overrides = [
            f"model_providers.synth={provider_config}",
            'model_provider="synth"',
            f'default_model="{model_name}"',
        ]
        override_args = [arg for override in config_overrides for arg in ("-c", override)]
        env["OPENAI_API_KEY"] = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        env["SYNTH_API_KEY"] = env["OPENAI_API_KEY"]
        print(f"Configured with wire_api={wire_api}")

    try:
        cmd = ["codex"]
        if model_name is not None:
            cmd.extend(["-m", model_name])
        cmd.extend(override_args)
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError:
        print("Failed to run Codex")
    finally:
        pass
