import subprocess
from pathlib import Path

import click
from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH_BASE
from synth_ai.utils import (
    create_and_write_json,
    find_bin_path,
    install_bin,
    load_json_to_dict,
    resolve_env_var,
    verify_bin,
    write_agents_md,
)

CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
SYNTH_PROVIDER_ID = "synth"


@click.command("opencode")
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
def opencode_cmd(
    model_name: ModelName | None = None,
    force: bool = False,
    override_url: str | None = None
) -> None:

    while True:
        bin_path = find_bin_path("opencode")
        if bin_path:
            break
        if not install_bin(
            "OpenCode",
            [
                "brew install opencode",
                "bun add -g opencode-ai",
                "curl -fsSL https://opencode.ai/install | bash",
                "npm i -g opencode-ai",
                "paru -S opencode"
            ]
        ):
            print("Failed to find your installed OpenCode")
            print("Please install from: https://opencode.ai")
            return
    print(f"Using OpenCode at {bin_path}")

    if not verify_bin(bin_path):
        print("Failed to verify OpenCode is runnable")
        return

    write_agents_md()

    if model_name is not None:
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"model_name={model_name} is invalid. Valid values for model_name: {MODEL_NAMES}"
            )
        synth_api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        data = load_json_to_dict(AUTH_PATH)
        good_entry = {
            "type": "api",
            "key": synth_api_key,
        }
        if data.get(SYNTH_PROVIDER_ID) != good_entry:
            data[SYNTH_PROVIDER_ID] = good_entry
        create_and_write_json(AUTH_PATH, data)
        config = load_json_to_dict(CONFIG_PATH)
        config.setdefault("$schema", "https://opencode.ai/config.json")
        if override_url:
            url = override_url
            print("Using override URL:", url)
        else:
            url = BACKEND_URL_SYNTH_RESEARCH_BASE
        provider_section = config.setdefault("provider", {})
        synth_provider = provider_section.setdefault(SYNTH_PROVIDER_ID, {})
        synth_provider["npm"] = "@ai-sdk/openai-compatible"
        synth_provider.setdefault("name", "Synth")
        models = synth_provider.setdefault("models", {})
        models.setdefault(model_name, {})
        options = synth_provider.setdefault("options", {})
        options["baseURL"] = url
        full_model_name = f"{SYNTH_PROVIDER_ID}/{model_name}"
        config["model"] = full_model_name
        create_and_write_json(CONFIG_PATH, config)

    try:
        subprocess.run([str(bin_path)], check=True)
    except subprocess.CalledProcessError:
        print("Failed to launch OpenCode")
