import os
import subprocess
import typing
from typing import Literal

import click
from synth_ai.utils import (
    PromptedChoiceOption,
    PromptedChoiceType,
    find_bin_path,
    install_codex,
    verify_codex_is_runnable,
)

BACKEND_URL= "https://agent-learning.onrender.com/api/synth-research"


DIV_START = f"{'-' * 24} CODEX CONFIG CHECK START {'-' * 23}"
DIV_END = f"{'-' * 25} CODEX CONFIG CHECK END {'-' * 24}"


ModelName = Literal[
    "synth-small",
    "synth-medium"
]


@click.command("codex")
@click.option(
    "--model",
    "model_name",
    cls=PromptedChoiceOption,
    type=PromptedChoiceType(list(typing.get_args(ModelName))),
    required=True
)
def codex_cmd(model_name: ModelName) -> None:
    print('\n' + DIV_START)

    print("[1/3] Finding your installed Codex...")
    while True:
        bin_path = find_bin_path("codex")
        if bin_path:
            break
        print("[1/3] Failed to find your installed Codex")
        if not install_codex():
            print(DIV_END + '\n')
            return
    print(f"[1/3] Found your installed Codex at {bin_path}")

    print("[2/3] Verifying your Codex is runnable via `codex --version`...")
    if not verify_codex_is_runnable(bin_path):
        print("[2/3] Failed to verify your installed Codex is runnable")
        print(DIV_END + '\n')
        return
    print("[2/3] Verified your installed Codex is runnable")

    full_model_name = f"synth/{model_name}"
    print("[3/3] Preparing Codex overrides for Synth backend...")
    config_overrides = [
        'provider="synth"',
        f'providers.synth={{"name":"Synth","baseURL":"{BACKEND_URL}","envKey":"OPENAI_API_KEY"}}',
        f'default_model="{full_model_name}"'
    ]
    override_args = [arg for override in config_overrides for arg in ("-c", override)]
    print("[3/3] Prepared Codex overrides")
    print(DIV_END + '\n')

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "dummy-key"
    env["CODEX_LOG_LEVEL"] = "debug"
    env["DEBUG"] = "1"
    try:
        subprocess.run(
            [
                "codex",
                "-m",
                full_model_name,
                *override_args,
                "Tell me about Jacob Roddy Beck"
            ],
            check=True,
            env=env
        )
    except subprocess.CalledProcessError:
        print("Failed to run Codex")
