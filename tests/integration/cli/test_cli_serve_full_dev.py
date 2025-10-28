import os
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import pytest
import requests
from synth_ai.tracing_v3.constants import canonical_trace_db_name


pytestmark = pytest.mark.integration


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ProcLogger:
    def __init__(self, proc: subprocess.Popen[str], *, max_lines: int = 500) -> None:
        self.proc = proc
        self.lines: deque[str] = deque(maxlen=max_lines)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def tail(self, n: int = 50) -> str:
        return "".join(list(self.lines)[-n:])

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self.proc.stdout is None:
                    break
                line = self.proc.stdout.readline()
                if not line:
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                self.lines.append(line)
        except Exception:
            pass


def _wait_for_health(base_url: str, proc: subprocess.Popen[str], logger: _ProcLogger, timeout: float = 180.0) -> dict:
    start = time.time()
    last_err: Exception | None = None
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        # Fail fast if server process exited
        rc = proc.poll()
        if rc is not None:
            print(f"[itest] server exited early with code {rc}")
            print("[itest] recent logs:\n" + logger.tail(120))
            raise AssertionError(f"server exited early rc={rc}")
        try:
            r = requests.get(f"{base_url}/health", timeout=2.5)
            if r.status_code in (200, 400):
                # 400 indicates auth missing but server is up; treat as healthy-enough
                ct = r.headers.get("content-type", "")
                print(f"[itest] health READY (status={r.status_code}) on attempt {attempt}")
                return r.json() if ct.startswith("application/json") else {}
            else:
                print(f"[itest] health status={r.status_code} on attempt {attempt}")
        except Exception as e:
            last_err = e
            if attempt % 10 == 0:
                print(f"[itest] waiting for health… attempt={attempt} err={e}")
        if attempt % 20 == 0:
            # Periodically show recent server logs while waiting
            print("[itest] recent server logs:\n" + logger.tail(80))
        time.sleep(0.5)
    print("[itest] timeout waiting for health; recent logs:\n" + logger.tail(200))
    raise AssertionError(f"server did not become healthy: {last_err}")


@pytest.mark.slow
def test_serve_math_single_step_and_rollout(tmp_path: Path):
    port = _find_free_port()
    base = f"http://127.0.0.1:{port}"

    # Create a tiny local dataset to avoid network downloads
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "train.jsonl").write_text('{"problem": "What is 2+2?", "solution": "4"}\n', encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("ENVIRONMENT_API_KEY", "test_env_key_123")
    # Avoid interactive key prompts in CLI validation
    env.setdefault("GROQ_API_KEY", "dummy")
    env.setdefault("OPENAI_API_KEY", "dummy")
    env.setdefault("MATH_DATASET_LOCAL_DIR", str(ds_dir))

    trace_dir = tmp_path / "traces"
    trace_db = trace_dir / canonical_trace_db_name()
    trace_dir.mkdir(parents=True, exist_ok=True)

    env.setdefault("PYTHONUNBUFFERED", "1")

    print("[itest] starting server…")
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "synth-ai",
            "serve",
            "math-single-step",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-reload",
            "--force",
            "--trace",
            str(trace_dir),
            "--trace-db",
            str(trace_db),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    try:
        logger = _ProcLogger(proc)
        logger.start()
        health = _wait_for_health(base, proc, logger)
        print(f"[itest] health: {health}")
        headers = {"X-API-Key": env["ENVIRONMENT_API_KEY"]}
        # Probe a few internal endpoints to aid debugging
        try:
            print("[itest] GET /task_info")
            ti = requests.get(f"{base}/task_info", timeout=5.0, headers=headers)
            print(f"[itest] /task_info status={ti.status_code}")
        except Exception as e:
            print(f"[itest] /task_info error: {e}")
        try:
            print("[itest] GET /docs")
            docs = requests.get(f"{base}/docs", timeout=5.0)
            print(f"[itest] /docs status={docs.status_code}")
        except Exception as e:
            print(f"[itest] /docs error: {e}")
        info = requests.get(f"{base}/task_info", timeout=10.0, headers=headers).json()
        assert isinstance(info, dict) and any(k in info for k in ("task", "taskset", "dataset", "inference", "capabilities"))

        body = {
            "run_id": "itest-run-1",
            "env": {"config": {"split": "train", "index": 0}, "seed": 0},
            "policy": {"policy_name": "local", "config": {"model": "local"}},
            "ops": [],
        }
        resp = requests.post(f"{base}/rollout", json=body, headers=headers, timeout=60.0)
        assert 200 <= resp.status_code < 300
        js = resp.json()
        assert isinstance(js, dict)
        assert "metrics" in js or "trajectories" in js
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
