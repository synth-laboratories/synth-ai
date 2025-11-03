from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.urls import (
    BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC,
    BACKEND_URL_SYNTH_RESEARCH_BASE,
    BACKEND_URL_SYNTH_RESEARCH_OPENAI,
)
from synth_ai.utils.agents import write_agents_md
from synth_ai.utils.bin import install_bin, verify_bin
from synth_ai.utils.env import resolve_env_var
from synth_ai.utils.json import create_and_write_json, load_json_to_dict
from synth_ai.utils.paths import find_bin_path


class AgentPreparationError(RuntimeError):
    """Raised when we fail to prepare an agent for launch."""


@dataclass(slots=True)
class AgentInvocation:
    agent_id: str
    display_name: str
    command: list[str]
    env: MutableMapping[str, str]
    cwd: Path | None = None


def _ensure_model_name(model_name: ModelName | None) -> ModelName | None:
    if model_name is None:
        return None
    if model_name not in MODEL_NAMES:
        valid = ", ".join(sorted(MODEL_NAMES))
        raise ValueError(
            f"model_name={model_name} is invalid. "
            f"Valid values for model_name: {valid}"
        )
    return model_name


def _ensure_binary(
    agent_id: str,
    display_name: str,
    binary_name: str,
    install_options: Sequence[str],
    install_help_url: str | None = None,
) -> Path:
    while True:
        bin_path = find_bin_path(binary_name)
        if bin_path:
            break
        if not install_bin(display_name, list(install_options)):
            print(f"Failed to find your installed {display_name}")
            if install_help_url:
                print(f"Please install from: {install_help_url}")
            raise AgentPreparationError(f"{display_name} is not installed")
    print(f"Using {display_name} at {bin_path}")
    if not verify_bin(bin_path):
        print(f"Failed to verify {display_name} is runnable")
        raise AgentPreparationError(f"{display_name} binary failed verification")
    return bin_path


def _base_env() -> MutableMapping[str, str]:
    return os.environ.copy()


def prepare_claude_invocation(
    *,
    model_name: ModelName | None = None,
    force: bool = False,
    override_url: str | None = None,
) -> AgentInvocation:
    bin_path = _ensure_binary(
        agent_id="claude",
        display_name="Claude Code",
        binary_name="claude",
        install_options=("curl -fsSL https://claude.ai/install.sh | bash",),
        install_help_url="https://claude.com/claude-code",
    )
    env = _base_env()
    validated_model = _ensure_model_name(model_name)
    if validated_model is not None:
        if override_url:
            url = f"{override_url.rstrip('/')}/{validated_model}"
            print(f"Using override URL with model: {url}")
        else:
            url = f"{BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC}/{validated_model}"
        env["ANTHROPIC_BASE_URL"] = url
        api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        env["ANTHROPIC_AUTH_TOKEN"] = api_key
        env["SYNTH_API_KEY"] = api_key

    write_agents_md()
    command = [str(bin_path)]
    return AgentInvocation(
        agent_id="claude",
        display_name="Claude Code",
        command=command,
        env=env,
    )


def prepare_codex_invocation(
    *,
    model_name: ModelName | None = None,
    force: bool = False,
    override_url: str | None = None,
) -> AgentInvocation:
    bin_path = _ensure_binary(
        agent_id="codex",
        display_name="OpenAI Codex CLI",
        binary_name="codex",
        install_options=("brew install codex", "npm install -g @openai/codex"),
        install_help_url="https://developers.openai.com/codex/cli/",
    )
    env = _base_env()
    command = [str(bin_path)]
    override_args: list[str] = []

    validated_model = _ensure_model_name(model_name)
    if validated_model is not None:
        if override_url:
            url = override_url
            print("Using override URL:", url)
        else:
            url = BACKEND_URL_SYNTH_RESEARCH_OPENAI
        provider_config = (
            f'{{name="Synth",base_url="{url}",env_key="OPENAI_API_KEY"}}'
        )
        config_overrides = [
            f"model_providers.synth={provider_config}",
            'model_provider="synth"',
            f'default_model="{validated_model}"',
        ]
        override_args = [
            arg for override in config_overrides for arg in ("-c", override)
        ]
        env["OPENAI_API_KEY"] = resolve_env_var(
            "SYNTH_API_KEY",
            override_process_env=force,
        )
        env["SYNTH_API_KEY"] = env["OPENAI_API_KEY"]
        command.extend(["-m", validated_model])

    if override_args:
        command.extend(override_args)

    print(" ".join(command))
    write_agents_md()
    return AgentInvocation(
        agent_id="codex",
        display_name="OpenAI Codex CLI",
        command=command,
        env=env,
    )


def prepare_opencode_invocation(
    *,
    model_name: ModelName | None = None,
    force: bool = False,
    override_url: str | None = None,
) -> AgentInvocation:
    bin_path = _ensure_binary(
        agent_id="opencode",
        display_name="OpenCode",
        binary_name="opencode",
        install_options=(
            "brew install opencode",
            "bun add -g opencode-ai",
            "curl -fsSL https://opencode.ai/install | bash",
            "npm i -g opencode-ai",
            "paru -S opencode",
        ),
        install_help_url="https://opencode.ai",
    )

    env = _base_env()
    validated_model = _ensure_model_name(model_name)
    if validated_model is not None:
        synth_api_key = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
        auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
        data = load_json_to_dict(auth_path)
        good_entry = {
            "type": "api",
            "key": synth_api_key,
        }
        if data.get("synth") != good_entry:
            data["synth"] = good_entry
        create_and_write_json(auth_path, data)
        config_path = Path.home() / ".config" / "opencode" / "opencode.json"
        config = load_json_to_dict(config_path)
        config.setdefault("$schema", "https://opencode.ai/config.json")
        if override_url:
            url = override_url
            print("Using override URL:", url)
        else:
            url = BACKEND_URL_SYNTH_RESEARCH_BASE
        provider_section = config.setdefault("provider", {})
        synth_provider = provider_section.setdefault("synth", {})
        synth_provider["npm"] = "@ai-sdk/openai-compatible"
        synth_provider.setdefault("name", "Synth")
        models = synth_provider.setdefault("models", {})
        models.setdefault(validated_model, {})
        options = synth_provider.setdefault("options", {})
        options["baseURL"] = url
        full_model_name = f"synth/{validated_model}"
        config["model"] = full_model_name
        create_and_write_json(config_path, config)

    write_agents_md()
    command = [str(bin_path)]
    return AgentInvocation(
        agent_id="opencode",
        display_name="OpenCode",
        command=command,
        env=env,
    )


AgentBuilder = Callable[..., AgentInvocation]

_AGENT_BUILDERS: Mapping[str, AgentBuilder] = {
    "claude": prepare_claude_invocation,
    "codex": prepare_codex_invocation,
    "opencode": prepare_opencode_invocation,
}

AGENT_IDS = tuple(sorted(_AGENT_BUILDERS))


def list_agent_invocation_builders() -> Mapping[str, AgentBuilder]:
    return _AGENT_BUILDERS


def get_agent_invocation_builder(agent_id: str) -> AgentBuilder:
    try:
        return _AGENT_BUILDERS[agent_id]
    except KeyError as exc:
        raise KeyError(f"Unsupported agent_id: {agent_id}") from exc


def prepare_agent_invocation(
    agent_id: str,
    *,
    model_name: ModelName | None = None,
    force: bool = False,
    override_url: str | None = None,
) -> AgentInvocation:
    builder = get_agent_invocation_builder(agent_id)
    return builder(model_name=model_name, force=force, override_url=override_url)
