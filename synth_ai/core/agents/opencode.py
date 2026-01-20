"""OpenCode CLI launcher."""

import subprocess
from pathlib import Path

from synth_ai.core.agents.utils import write_agents_md
from synth_ai.core.bin import install_bin, verify_bin
from synth_ai.core.env import resolve_env_var
from synth_ai.core.json import create_and_write_json, load_json_to_dict
from synth_ai.core.paths import get_bin_path
from synth_ai.core.urls import BACKEND_URL_BASE, BACKEND_URL_SYNTH_RESEARCH_BASE
from synth_ai.data.enums import SYNTH_MODEL_NAMES

CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
SYNTH_PROVIDER_ID = "synth"


def run_opencode(model_name=None, force=False, override_url=None, config_path=None):
    """Launch OpenCode with optional Synth backend routing."""
    while True:
        bin_path = get_bin_path("opencode")
        if bin_path:
            break
        if not install_bin(
            "OpenCode",
            [
                "brew install opencode",
                "bun add -g opencode-ai",
                "curl -fsSL https://opencode.ai/install | bash",
                "npm i -g opencode-ai",
                "paru -S opencode",
            ],
        ):
            print("Failed to find your installed OpenCode")
            print("Please install from: https://opencode.ai")
            return
    print(f"Using OpenCode at {bin_path}")

    if not verify_bin(bin_path):
        print("Failed to verify OpenCode is runnable")
        return

    write_agents_md()

    synth_api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
    if override_url:
        base_url = override_url.rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[:-4]
    else:
        base_url = BACKEND_URL_BASE

    if model_name is not None:
        if model_name not in SYNTH_MODEL_NAMES:
            raise ValueError(f"model_name={model_name} is invalid. Valid: {SYNTH_MODEL_NAMES}")
        synth_api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        data = load_json_to_dict(AUTH_PATH)
        good_entry = {"type": "api", "key": synth_api_key}
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
    finally:
        pass
