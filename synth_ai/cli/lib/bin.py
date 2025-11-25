import shlex
import subprocess
from pathlib import Path

from .prompts import prompt_choice


def install_bin(name: str, install_options: list[str]) -> bool:
    cmd = prompt_choice(
        f"How would you like to install {name}?",
        install_options
    )
    div_start = f"{'-' * 29} INSTALL START {'-' * 29}"
    div_end = f"{'-' * 30} INSTALL END {'-' * 30}"
    try:
        print(f"Installing {name} via `{cmd}`")
        print('\n' + div_start)
        subprocess.run(shlex.split(cmd), check=True)
        print(div_end + '\n')
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {name}: {e}")
        print(div_end + '\n')
        return False
    

def verify_bin(bin_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(bin_path), "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as e:
        print(e)
        return False
