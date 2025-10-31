import contextlib
import json
import os
import subprocess
from pathlib import Path

import click
from synth_ai.utils import find_bin_path, prompt_choice

BACKEND_URL= "https://agent-learning.onrender.com/api/synth-research"
MODEL_NAME = "synth-qt3.14"

DIV_CODEX_START = f"{'-' * 24} CODEX CONFIG CHECK START {'-' * 23}"
DIV_CODEX_END = f"{'-' * 25} CODEX CONFIG CHECK END {'-' * 24}"
DIV_INSTALL_START = f"{'-' * 29} INSTALL START {'-' * 29}"
DIV_INSTALL_END = f"{'-' * 30} INSTALL END {'-' * 30}"

INSTALL_CMDS = [
    "brew install codex",
    "npm install -g @openai/codex"
]


def install_codex() -> bool:
    cmd = prompt_choice(
        "How would you like to install Codex?",
        INSTALL_CMDS
    )
    try:
        print(f"Installing Codex via {cmd}...")
        print('\n'+ DIV_INSTALL_START)
        subprocess.run(cmd.split(), check=True)
        print(DIV_INSTALL_END + '\n')
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Codex: {e}")
        print(DIV_INSTALL_END + '\n')
        return False
    

def verify_codex_is_runnable(bin_path: Path) -> bool:
    try:
        result = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False


def find_codex_config_path(bin_path: Path) -> Path | None:
    default_path = Path(os.path.expanduser('~')) / ".codex" / "config.json"
    if default_path.exists():
        return default_path
    alt_path = Path(bin_path).parent / ".codex" / "config.json"
    if alt_path.exists():
        return alt_path
    return None


def update_codex_config(config_path: Path) -> None:
    config = {}
    with contextlib.suppress(json.JSONDecodeError):
        config = json.loads(config_path.read_text())
    config.setdefault("providers", {})
    config["providers"]["synth"] = {
        "name": "Synth",
        "baseURL": BACKEND_URL.rstrip('/'),
        "envKey": "OPENAI_API_KEY"
    }
    config["default_model"] = MODEL_NAME
    config_path.write_text(json.dumps(config, indent=2))


@click.command("codex")
def codex_cmd() -> None:
    print('\n' + DIV_CODEX_START)

    print("[1/3] Finding your installed Codex...")
    while True:
        bin_path = find_bin_path("codex")
        if bin_path:
            break
        print("[1/3] Failed to find your installed Codex")
        if not install_codex():
            print(DIV_CODEX_END + '\n')
            return
    print(f"[1/3] Found your installed Codex at {bin_path}")

    print("[2/3] Verifying your Codex is runnable via `codex --version`...")
    if not verify_codex_is_runnable(bin_path):
        print("[2/3] Failed to verify your installed Codex is runnable")
        print(DIV_CODEX_END + '\n')
        return
    print("[2/3] Verified your installed Codex is runnable")

    print("[3/3] Finding your Codex config...")
    codex_config_path = find_codex_config_path(bin_path)
    if not codex_config_path:
        print("[3/3] Failed to find your Codex config")
        print(DIV_CODEX_END + '\n')
        return
    print(f"[3/3] Found your Codex config at {codex_config_path}")
    print(DIV_CODEX_END + '\n')

    os.environ.setdefault("OPENAI_API_KEY", MODEL_NAME)
    update_codex_config(codex_config_path)
    try:
        subprocess.run(
            ["codex", "-m", MODEL_NAME, "Tell me about Jacob Roddy Beck"],
            check=True
        )
    except subprocess.CalledProcessError:
        print("Failed to run Codex")
