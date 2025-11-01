import shlex
import subprocess
from pathlib import Path

from .cli import prompt_choice


def install_opencode() -> bool:
    cmd = prompt_choice(
        "How would you like to install OpenCode?",
        [
            "brew install sst/tap/opencode",
            "npm install -g opencode-ai"
        ]
    )
    div_start = f"{'-' * 29} INSTALL START {'-' * 29}"
    div_end = f"{'-' * 30} INSTALL END {'-' * 30}"
    try:
        print(f"Installing OpenCode via `{cmd}`...")
        print("\n" + div_start)
        subprocess.run(shlex.split(cmd), check=True)
        print(div_end + "\n")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install OpenCode via `{cmd}`")
        print(div_end + "\n")
        return False
    

def verify_opencode(bin_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(bin_path), "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
