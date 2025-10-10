import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def warm_uv_environment() -> None:
    """Warm the uv environment once to avoid first-run setup costs during tests."""
    env = os.environ.copy()
    # Keep this minimal; just ensure the env is created and python runs
    subprocess.run(
        ["uv", "run", "python", "-c", "import sys; print('uv-warm:', sys.version)"],
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

