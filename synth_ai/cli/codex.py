import os
import subprocess

import click
from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.urls import BACKEND_URL_SYNTH_RESEARCH_OPENAI
from synth_ai.utils import (
    PromptedChoiceOption,
    PromptedChoiceType,
    find_bin_path,
    install_codex,
    resolve_env_var,
    verify_codex,
)

DIV_START = f"{'-' * 24} CODEX CONFIG CHECK START {'-' * 23}"
DIV_END = f"{'-' * 25} CODEX CONFIG CHECK END {'-' * 24}"


@click.command("codex")
@click.option(
    "--model",
    "model_name",
    cls=PromptedChoiceOption,
    type=PromptedChoiceType(MODEL_NAMES),
    required=True
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
def codex_cmd(
    model_name: ModelName,
    force: bool = False,
    override_url: str | None = None
)-> None:
    print('\n' + DIV_START)

    print("Finding your installed Codex...")
    while True:
        bin_path = find_bin_path("codex")
        if bin_path:
            break
        if not install_codex():
            print("Failed to find your installed Codex")
            print(DIV_END + '\n')
            return
    print(f"Found your installed Codex at {bin_path}")

    print("Verifying your Codex is runnable via `codex --version`...")
    if not verify_codex(bin_path):
        print("Failed to verify your installed Codex is runnable")
        print(DIV_END + '\n')
        return
    print("Verified your installed Codex is runnable")

    print(DIV_END + '\n')

    if override_url:
        url = override_url
        print("Using override URL:", url)
    else:
        url = BACKEND_URL_SYNTH_RESEARCH_OPENAI
    provider_config = f'{{name="Synth",base_url="{url}",env_key="OPENAI_API_KEY"}}'
    config_overrides = [
        f"model_providers.synth={provider_config}",
        'model_provider="synth"',
        f'default_model="{model_name}"'
    ]
    override_args = [arg for override in config_overrides for arg in ("-c", override)]

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = resolve_env_var("SYNTH_API_KEY", override_process_env=force)
    env["SYNTH_API_KEY"] = env["OPENAI_API_KEY"]
    
    try:
        cmd = [
            "codex",
            "-m",
            model_name,
            *override_args
        ]
        print("Launching Codex command:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError:
        print("Failed to run Codex")
