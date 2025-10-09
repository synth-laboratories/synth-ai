import os
import subprocess
import pytest


pytestmark = pytest.mark.integration


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, env=env)


def test_train_help_dev():
    env = os.environ.copy()
    env.setdefault("BACKEND_BASE_URL", env.get("DEV_BACKEND_URL", "http://localhost:8000/api"))
    cp = _run(["uv", "run", "synth-ai", "train", "--help"], env=env)
    assert cp.returncode == 0

