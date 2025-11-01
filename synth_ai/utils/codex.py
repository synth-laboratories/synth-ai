import os
import shlex
import subprocess
from pathlib import Path

from .cli import prompt_choice


def install_codex() -> bool:
    cmd = prompt_choice(
        "How would you like to install Codex?",
        [
            "brew install codex",
            "npm install -g @openai/codex"
        ]
    )
    div_start = f"{'-' * 29} INSTALL START {'-' * 29}"
    div_end = f"{'-' * 30} INSTALL END {'-' * 30}"
    try:
        print(f"Installing Codex via {cmd}...")
        print('\n'+ div_start)
        subprocess.run(shlex.split(cmd), check=True)
        print(div_end + '\n')
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Codex: {e}")
        print(div_end + '\n')
        return False


def verify_codex(bin_path: Path) -> bool:
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
