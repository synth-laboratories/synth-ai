import os
import subprocess
import pytest


pytestmark = pytest.mark.integration


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, env=env)


def test_man_runs():
    cp = _run(["uv", "run", "synth-ai", "man"], env=os.environ.copy())
    assert cp.returncode == 0
    assert "Synth AI CLI Manual" in cp.stdout


def test_traces_help():
    cp = _run(["uv", "run", "synth-ai", "traces", "--help"], env=os.environ.copy())
    assert cp.returncode == 0
    assert "Database URL" in cp.stdout


def test_task_app_list_help():
    cp = _run(["uv", "run", "synth-ai", "task-app", "list", "--help"], env=os.environ.copy())
    assert cp.returncode == 0

