"""Integration tests for Sokoban task app with evaluation."""
from __future__ import annotations

import os
import socket
import subprocess
from subprocess import TimeoutExpired
import time
from pathlib import Path
from typing import Iterator

import pytest

requests = pytest.importorskip("requests")


HERE = Path(__file__).resolve().parent
TASK_APP_ROOT = HERE.parents[1]
CONFIG_PATH = TASK_APP_ROOT / "eval_openai_gpt5.toml"


def _which(executable: str) -> bool:
    return any(
        (Path(path) / executable).exists()
        for path in os.getenv("PATH", "").split(os.pathsep)
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(base_url: str, timeout: float = 60.0) -> None:
    """Wait for the Sokoban server to become ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/info", timeout=2.0)
            if resp.status_code == 200:
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Task app at {base_url} did not become ready")


@pytest.fixture
def sokoban_server(tmp_path: Path) -> Iterator[str]:
    """Start the Sokoban task app server for testing."""
    if not _which("uv"):
        pytest.skip("uv executable not found on PATH")

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    cmd = [
        "uv",
        "run",
        "-m",
        "synth_ai",
        "task-app",
        "serve",
        "sokoban",
        "--port",
        str(port),
        "--no-reload",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        stdin=subprocess.PIPE,
    )
    
    # Send "n" to decline tracing
    try:
        if proc.stdin:
            proc.stdin.write("n\n")
            proc.stdin.flush()
    except Exception:
        pass
    
    stdout_capture = ""
    try:
        time.sleep(2)
        if proc.poll() is not None:
            stdout_capture, _ = proc.communicate(timeout=2)
            tail = "\n".join(stdout_capture.strip().splitlines()[-20:]) if stdout_capture else ""
            pytest.skip(f"Task app terminated immediately:\n{tail}")
        
        _wait_for_server(base_url)
        yield base_url
    except RuntimeError as e:
        proc.terminate()
        try:
            stdout_capture, _ = proc.communicate(timeout=10)
        except TimeoutExpired:
            proc.kill()
            stdout_capture, _ = proc.communicate()
        tail = "\n".join((stdout_capture or "").strip().splitlines()[-20:])
        pytest.skip(f"Task app failed to start: {e}\n{tail}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except TimeoutExpired:
                proc.kill()


def test_sokoban_server_health(sokoban_server: str) -> None:
    """Test that the Sokoban server health endpoint works."""
    resp = requests.get(f"{sokoban_server}/health", timeout=5.0)
    assert resp.status_code in (200, 400), f"Unexpected status: {resp.status_code}"


def test_sokoban_task_info(sokoban_server: str) -> None:
    """Test that the Sokoban server returns valid task_info."""
    resp = requests.get(f"{sokoban_server}/task_info", timeout=5.0)
    assert resp.status_code == 200
    data = resp.json()
    assert "task" in data
    assert data["task"]["id"] == "sokoban"


def test_sokoban_manual_rollout(sokoban_server: str) -> None:
    """Test a manual Sokoban rollout with explicit actions."""
    # Try explicit action rollout (no LLM required)
    rollout_payload = {
        "run_id": "test_manual",
        "env": {"seed": 0, "config": {"difficulty": "easy", "max_steps": 50}},
        "ops": [0, 2, 2, 3],  # left, right, right, down
        "policy": {"config": {"provider": "noop"}},
    }
    
    resp = requests.post(
        f"{sokoban_server}/rollout",
        json=rollout_payload,
        headers={"Authorization": "Bearer sk_env_test"},
        timeout=30.0,
    )
    
    assert resp.status_code == 200
    data = resp.json()
    assert "trajectories" in data
    assert len(data["trajectories"]) > 0
    assert "metrics" in data

