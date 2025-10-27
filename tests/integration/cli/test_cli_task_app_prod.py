import os
import subprocess
import pytest


pytestmark = pytest.mark.integration


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, env=env)


@pytest.mark.fast
def test_serve_help_prod():
    env = os.environ.copy()
    for k in ("BACKEND_BASE_URL", "SYNTH_BACKEND_BASE_URL", "SYNTH_BASE_URL"):
        env.pop(k, None)
    cp = _run(["uv", "run", "synth-ai", "serve", "--help"], env=env)
    assert cp.returncode == 0


@pytest.mark.fast
def test_deploy_help_prod():
    env = os.environ.copy()
    for k in ("BACKEND_BASE_URL", "SYNTH_BACKEND_BASE_URL", "SYNTH_BASE_URL"):
        env.pop(k, None)
    cp = _run(["uv", "run", "synth-ai", "deploy", "--help"], env=env)
    assert cp.returncode == 0


@pytest.mark.fast
def test_modal_serve_help_prod():
    env = os.environ.copy()
    for k in ("BACKEND_BASE_URL", "SYNTH_BACKEND_BASE_URL", "SYNTH_BASE_URL"):
        env.pop(k, None)
    cp = _run(["uv", "run", "synth-ai", "modal-serve", "--help"], env=env)
    assert cp.returncode == 0

