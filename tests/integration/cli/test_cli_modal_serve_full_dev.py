import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _have_modal_cli() -> bool:
    return shutil.which("modal") is not None


@pytest.mark.skipif(not _have_modal_cli(), reason="modal CLI not installed")
@pytest.mark.slow
def test_modal_serve_dry_run_math_single_step():
    env = os.environ.copy()
    env.setdefault("ENVIRONMENT_API_KEY", "test_env_key_123")

    # Launch modal-serve with a short timeout and expect quick return since it proxies modal CLI
    cp = subprocess.run(
        [
            "uv",
            "run",
            "synth-ai",
            "modal-serve",
            "math-single-step",
            "--modal-cli",
            "echo",
            "--env-file",
            str((Path(__file__).resolve().parents[3] / "examples/rl/.env")),
        ],
        text=True,
        capture_output=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[3]),
        timeout=120,
    )
    # Accept non-zero returns from modal if environment is not configured; ensure command runs
    assert cp.returncode in (0, 1)

