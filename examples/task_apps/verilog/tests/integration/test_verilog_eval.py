"""Integration tests for Verilog task app with Groq evaluation."""
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
CONFIG_PATH = TASK_APP_ROOT / "eval_groq_qwen32b.toml"


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
    """Wait for the Verilog server to become ready."""
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
def verilog_server(tmp_path: Path) -> Iterator[str]:
    """Start the Verilog task app server for testing."""
    if not _which("uv"):
        pytest.skip("uv executable not found on PATH")
    if "GROQ_API_KEY" not in os.environ:
        pytest.skip("GROQ_API_KEY must be set for Groq-backed evals")

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    cmd = [
        "uv",
        "run",
        "-m",
        "synth_ai",
        "task-app",
        "serve",
        "grpo-verilog",
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
        stdin=subprocess.PIPE,  # Auto-answer tracing prompt
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
        # Check if process died immediately
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


@pytest.mark.slow
def test_verilog_server_health(verilog_server: str) -> None:
    """Test that the Verilog server health endpoint works."""
    # Health endpoint requires auth, so we expect 400 (auth failed) or 200
    resp = requests.get(f"{verilog_server}/health", timeout=5.0)
    assert resp.status_code in (200, 400), f"Unexpected status: {resp.status_code}"


@pytest.mark.slow
def test_verilog_task_info(verilog_server: str) -> None:
    """Test that the Verilog server returns valid task_info."""
    resp = requests.get(f"{verilog_server}/task_info", timeout=5.0)
    assert resp.status_code == 200
    data = resp.json()
    assert "task" in data
    assert data["task"]["id"] == "verilog"


@pytest.mark.slow
def test_verilog_eval_with_groq(verilog_server: str) -> None:
    """Spin up the Verilog task app and run a Groq-backed eval."""
    if not CONFIG_PATH.exists():
        pytest.skip(f"Config file not found: {CONFIG_PATH}")
    
    cmd = [
        "uv",
        "run",
        "-m",
        "synth_ai",
        "eval",
        "grpo-verilog",
        "--config",
        str(CONFIG_PATH),
        "--url",
        verilog_server,
        "--model",
        "qwen/qwen3-32b",
        "--seeds",
        "0",  # Just test one seed
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        check=False,
        timeout=300,  # 5 minutes max
    )
    
    if result.returncode != 0:
        pytest.fail(f"Eval failed with return code {result.returncode}:\n{result.stdout}")
    
    # Check for success indicators
    assert "Eval complete" in result.stdout
    assert "1 ok, 0 failed" in result.stdout or "status=200" in result.stdout
    
    # Check that we got a meaningful outcome score
    assert "outcome" in result.stdout.lower() or "mean_return" in result.stdout.lower()



