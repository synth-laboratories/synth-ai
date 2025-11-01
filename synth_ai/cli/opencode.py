import subprocess
from pathlib import Path

import click
from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH
from synth_ai.utils import (
    PromptedChoiceOption,
    PromptedChoiceType,
    create_and_write_json,
    find_bin_path,
    install_opencode,
    load_json_to_dict,
    mask_str,
    resolve_env_var,
    verify_opencode,
)

DIV_START = f"{'-' * 24} OPENCODE CONFIG CHECK START {'-' * 22}"
DIV_END = f"{'-' * 25} OPENCODE CONFIG CHECK END {'-' * 23}"

CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
SCHEMA_URL = "https://opencode.ai/config.json"
SYNTH_PROVIDER_ID = "synth"



def _ensure_synth_provider(config: dict) -> dict:
    provider_section = config.setdefault("provider", {})
    synth_provider = provider_section.setdefault(SYNTH_PROVIDER_ID, {})
    synth_provider["npm"] = "@ai-sdk/openai-compatible"
    synth_provider.setdefault("name", "Synth")

    options = synth_provider.setdefault("options", {})
    options["baseURL"] = BACKEND_URL_SYNTH_RESEARCH

    synth_provider.setdefault("models", {})
    return config


def _ensure_synth_auth(api_key: str) -> bool:
    auth_data = load_json_to_dict(AUTH_PATH)
    desired_entry = {
        "type": "api",
        "key": api_key,
    }
    if auth_data.get(SYNTH_PROVIDER_ID) == desired_entry:
        return False
    auth_data[SYNTH_PROVIDER_ID] = desired_entry
    create_and_write_json(AUTH_PATH, auth_data)
    return True


def _ensure_model_entry(config: dict, model_name: str) -> dict:
    provider_section = config.setdefault("provider", {})
    synth_provider = provider_section.setdefault(SYNTH_PROVIDER_ID, {})
    models = synth_provider.setdefault("models", {})
    models.setdefault(model_name, {})
    return config


@click.command("opencode")
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
def opencode_cmd(model_name: ModelName, force: bool = False) -> None:
    print("\n" + DIV_START)

    print("Finding your installed OpenCode...")
    while True:
        bin_path = find_bin_path("opencode")
        if bin_path:
            break
        if not install_opencode():
            print("Failed to find your installed OpenCode")
            print(DIV_END + "\n")
            return
    print(f"Found your installed OpenCode at {bin_path}")

    print("Verifying your OpenCode is runnable via `opencode --version`...")
    if not verify_opencode(bin_path):
        print("Failed to verify your installed OpenCode is runnable")
        print(DIV_END + "\n")
        return
    print("Verified your installed OpenCode is runnable")

    print("Registering your Synth API key with OpenCode...")
    synth_api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
    if _ensure_synth_auth(synth_api_key):
        print(f"Stored Synth API key ({mask_str(synth_api_key)}) in {AUTH_PATH}")
    else:
        print(f"Synth API key already present in {AUTH_PATH}")

    print("Updating your OpenCode config...")
    config = load_json_to_dict(CONFIG_PATH)
    config.setdefault("$schema", SCHEMA_URL)
    config = _ensure_synth_provider(config)
    config = _ensure_model_entry(config, model_name)
    full_model_name = f"{SYNTH_PROVIDER_ID}/{model_name}"
    config["model"] = full_model_name
    create_and_write_json(CONFIG_PATH, config)
    print(f"Updated your OpenCode config at {CONFIG_PATH}")
    print(DIV_END + "\n")

    print("Launching OpenCode...")
    try:
        subprocess.run([str(bin_path)], check=True)
    except subprocess.CalledProcessError:
        print("Failed to launch OpenCode")
