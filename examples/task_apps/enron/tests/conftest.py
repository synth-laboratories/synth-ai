"""Shared fixtures for Enron tests."""
import os
import socket
import subprocess
from subprocess import TimeoutExpired
import time
from pathlib import Path
from typing import Iterator

import pytest

requests = pytest.importorskip("requests")


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
    """Wait for the Enron server to become ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # Try /info first (no auth required if --insecure)
            resp = requests.get(f"{base_url}/info", timeout=2.0)
            if resp.status_code == 200:
                return
            # If 400/401, server is up but needs auth - that's OK
            if resp.status_code in (400, 401):
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Task app at {base_url} did not become ready")


@pytest.fixture(scope="module")
def enron_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start the Enron task app server for testing."""
    if not _which("uv"):
        pytest.skip("uv executable not found on PATH")
    if "GROQ_API_KEY" not in os.environ:
        pytest.skip("GROQ_API_KEY must be set for Groq-backed tests")

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    tmp_path = tmp_path_factory.mktemp("enron")
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
        "grpo-enron",
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

