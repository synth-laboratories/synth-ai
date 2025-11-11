import os
import subprocess

import click
from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH_OPENAI
from synth_ai.utils import get_bin_path, install_bin, resolve_env_var, verify_bin, write_agents_md


@click.command("codex")
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
@click.option(
    "--wire-api",
    "wire_api",
    type=click.Choice(["chat", "responses"], case_sensitive=False),
    default=None,
    help="API wire format: 'chat' for Chat Completions, 'responses' for Responses API"
)
def codex_cmd(
    model_name: ModelName | None = None,
    force: bool = False,
    override_url: str | None = None,
    wire_api: str | None = None
)-> None:

    while True:
        bin_path = get_bin_path("codex")
        if bin_path:
            break
        if not install_bin(
            "Codex",
            [
                "brew install codex",
                "npm install -g @openai/codex"
            ]
        ):
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
    
    if model_name is not None:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name={model_name} is invalid. Valid values for model_name: {MODEL_NAMES}")
        if override_url:
            url = override_url
            print("Using override URL:", url)
        else:
            url = BACKEND_URL_SYNTH_RESEARCH_OPENAI
        
        # Determine wire_api based on URL or explicit option
        # If URL contains "/responses", default to responses API
        # Otherwise, use explicit wire_api option or default to chat
        if wire_api is None:
            wire_api = "responses" if "/responses" in url or url.endswith("/responses") else "chat"
        
        # Build provider config with wire_api
        provider_config_parts = [
            'name="Synth"',
            f'base_url="{url}"',
            'env_key="OPENAI_API_KEY"',
            f'wire_api="{wire_api}"'
        ]
        provider_config = "{" + ",".join(provider_config_parts) + "}"
        
        config_overrides = [
            f"model_providers.synth={provider_config}",
            'model_provider="synth"',
            f'default_model="{model_name}"'
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
