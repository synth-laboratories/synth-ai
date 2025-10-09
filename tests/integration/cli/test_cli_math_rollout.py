import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.integration


def _find_free_port() -> int:
    import socket

    for _ in range(10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]
        except PermissionError:
            continue
    for candidate in range(8500, 8900):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", candidate))
                return candidate
        except PermissionError:
            continue
        except OSError:
            continue
    raise RuntimeError("Unable to find free localhost port for math integration test")


class _ProcLogger:
    def __init__(self, proc: subprocess.Popen[str], *, max_lines: int = 500) -> None:
        from collections import deque
        import threading

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


def _wait_for_health(base_url: str, proc: subprocess.Popen[str], logger: _ProcLogger, timeout: float = 120.0) -> dict:
    start = time.time()
    last_err: Exception | None = None
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        rc = proc.poll()
        if rc is not None:
            print(f"[math-itest] server exited early with code {rc}")
            print("[math-itest] recent logs:\n" + logger.tail(120))
            raise AssertionError(f"server exited early rc={rc}")
        try:
            r = requests.get(f"{base_url}/health", timeout=2.5)
            if r.status_code in (200, 400):
                ct = r.headers.get("content-type", "")
                return r.json() if ct.startswith("application/json") else {}
        except Exception as err:
            last_err = err
        time.sleep(0.5)
    print("[math-itest] timeout waiting for health; recent logs:\n" + logger.tail(200))
    raise AssertionError(f"server did not become healthy: {last_err}")


def test_serve_math_rollout_returns_trace(tmp_path: Path):
    try:
        port = _find_free_port()
    except RuntimeError as exc:
        pytest.skip(f"unable to reserve localhost port: {exc}")
    base = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.setdefault("ENVIRONMENT_API_KEY", "test_env_key_789")
    env.setdefault("TASKAPP_TRACING_ENABLED", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OPENAI_API_KEY", "dummy")
    env.setdefault("GROQ_API_KEY", "dummy")
    env.setdefault("SYNTH_FAKE_INFERENCE", "1")

    trace_dir = tmp_path / "traces"
    trace_db = trace_dir / "synth_ai.db"
    trace_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
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
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
    )

    logger = _ProcLogger(proc)
    logger.start()

    try:
        _wait_for_health(base, proc, logger)

        headers = {"X-API-Key": env["ENVIRONMENT_API_KEY"]}
        body = {
            "run_id": "itest-math-trace",
            "env": {
                "env_name": "math_single_step",
                "config": {"difficulty": "easy"},
                "seed": 0,
            },
            "policy": {
                "policy_name": "math-react",
                "config": {
                    "model": "mock-math-model",
                    "inference_url": "http://127.0.0.1:9",
                    "max_tokens": 64,
                },
            },
            "ops": ["agent", "env"],
            "record": {
                "return_trace": True,
                "trace_format": "compact",
            },
            "safety": {
                "max_ops": 4,
                "max_time_s": 60.0,
            },
        }

        resp = requests.post(f"{base}/rollout", json=body, headers=headers, timeout=60.0)
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
        payload = resp.json()
        trace = payload.get("trace")
        assert isinstance(trace, dict) and trace, "Trace payload should be a non-empty dict"
        assert trace.get("session_id"), "Trace missing session_id"
        assert trace.get("metadata"), "Trace metadata missing"

        metrics = payload.get("metrics") or {}
        print(
            "[math-itest] rollout summary: run_id=", payload.get("run_id"),
            " mean_return=", metrics.get("mean_return"),
            " steps=", metrics.get("num_steps"),
            " trace_events=", trace.get("events_count"),
            " decision_rewards=", bool(trace.get("decision_rewards")),
        )

        trajectories = payload.get("trajectories") or []
        assert trajectories, "Expected at least one trajectory in rollout response"
        traj = trajectories[0]
        steps = traj.get("steps") or []
        assert steps, "Trajectory should contain steps"
        print("[math-itest] first step log:\n" + json.dumps(steps[0], indent=2))

        lm_calls = trace.get("lm_calls") or []
        assert lm_calls, "Trace LM calls should not be empty"
        print(
            "[math-itest] first LM call preview:\n"
            + json.dumps(
                {
                    "prompt": (lm_calls[0].get("prompt") or "")[:200],
                    "response": (lm_calls[0].get("response") or "")[:200],
                },
                indent=2,
            )
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logger.stop()
        time.sleep(0.2)
