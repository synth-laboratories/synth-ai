import os
import subprocess
import pytest


pytestmark = pytest.mark.integration


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, env=env)


@pytest.mark.fast
def test_balance_help():
    cp = _run(["uv", "run", "synth-ai", "balance", "--help"], env=os.environ.copy())
    assert cp.returncode == 0
    assert "Show your remaining credit balance" in cp.stdout


@pytest.mark.fast
def test_balance_dev_backend_help_only():
    env = os.environ.copy()
    # Prefer explicit dev URL if configured; don't require real key for help
    env.setdefault("BACKEND_BASE_URL", env.get("DEV_BACKEND_URL", "http://localhost:8000/api"))
    cp = _run(["uv", "run", "synth-ai", "balance", "--help"], env=env)
    assert cp.returncode == 0


@pytest.mark.fast
def test_balance_prod_help_only():
    env = os.environ.copy()
    # Ensure we don't leak dev overrides; help should still work
    for k in ("BACKEND_BASE_URL", "SYNTH_BACKEND_BASE_URL", "SYNTH_BASE_URL"):
        env.pop(k, None)
    cp = _run(["uv", "run", "synth-ai", "balance", "--help"], env=env)
    assert cp.returncode == 0

