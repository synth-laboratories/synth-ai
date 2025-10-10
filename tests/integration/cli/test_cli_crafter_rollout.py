import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests


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

    for candidate in range(8100, 8600):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", candidate))
                return candidate
        except PermissionError:
            continue
        except OSError:
            continue
    raise RuntimeError("Unable to find free localhost port for crafter integration test")


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
            print(f"[crafter-itest] server exited early with code {rc}")
            print("[crafter-itest] recent logs:\n" + logger.tail(120))
            raise AssertionError(f"server exited early rc={rc}")
        try:
            r = requests.get(f"{base_url}/health", timeout=2.5)
            if r.status_code in (200, 400):
                ct = r.headers.get("content-type", "")
                return r.json() if ct.startswith("application/json") else {}
        except Exception as err:
            last_err = err
        time.sleep(0.5)
    print("[crafter-itest] timeout waiting for health; recent logs:\n" + logger.tail(200))
    raise AssertionError(f"server did not become healthy: {last_err}")

pytestmark = pytest.mark.integration

pytest.importorskip("crafter", reason="crafter dependency not installed")


def test_serve_crafter_rollout_returns_trace(tmp_path: Path):
    try:
        port = _find_free_port()
    except RuntimeError as exc:
        pytest.skip(f"unable to reserve localhost port: {exc}")
    base = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.setdefault("ENVIRONMENT_API_KEY", "test_env_key_456")
    env.setdefault("TASKAPP_TRACING_ENABLED", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OPENAI_API_KEY", "dummy")
    env.setdefault("GROQ_API_KEY", "dummy")
    env.setdefault("VLLM_BASE_URL", "http://127.0.0.1:9")
    env.setdefault("DEFAULT_MODEL", "mock-crafter-model")
    env.setdefault("SYNTH_FAKE_INFERENCE", "1")

    trace_dir = tmp_path / "traces"
    trace_db = trace_dir / "synth_ai.db"
    trace_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "synth-ai",
        "serve",
        "grpo-crafter",
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
            "run_id": "itest-crafter-trace",
            "env": {
                "env_name": "crafter",
                "config": {},
                "seed": 123,
            },
            "policy": {
                "policy_name": "crafter-react",
                "config": {
                    "model": "mock-crafter-model",
                    "inference_url": "http://127.0.0.1:9",
                    "thinking_mode": "no_think",
                    "max_tokens": 32,
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

        # Emit a quick summary so `pytest -s` shows the rollout outcome
        metrics = payload.get("metrics") or {}
        print(
            "[crafter-itest] rollout summary: run_id=", payload.get("run_id"),
            " mean_return=", metrics.get("mean_return"),
            " steps=", metrics.get("num_steps"),
            " trace_events=", trace.get("events_count"),
            " decision_rewards=", bool(trace.get("decision_rewards")),
        )

        print("[crafter-itest] metrics compact:\n" + json.dumps({
            "episode_returns": metrics.get("episode_returns"),
            "mean_return": metrics.get("mean_return"),
            "num_steps": metrics.get("num_steps"),
            "decision_rewards_count": len(trace.get("decision_rewards") or []),
        }, indent=2))

        trajectories = payload.get("trajectories") or []
        assert trajectories, "Expected at least one trajectory in rollout response"
        traj = trajectories[0]
        steps = traj.get("steps") or []
        assert steps, "Trajectory should contain steps"
        summary = {
            "env_id": traj.get("env_id"),
            "policy_id": traj.get("policy_id"),
            "length": traj.get("length"),
            "first_step_reward": steps[0].get("reward"),
            "first_step_tool_calls": steps[0].get("tool_calls") or [],
        }
        print("[crafter-itest] trajectory summary:\n" + json.dumps(summary, indent=2))

        lm_calls = trace.get("lm_calls") or []
        assert lm_calls, "Trace LM calls should not be empty"
        first_call = {
            "prompt_preview": (lm_calls[0].get("prompt") or "")[:200],
            "response_preview": (lm_calls[0].get("response") or "")[:200],
        }
        print("[crafter-itest] first LM call preview:\n" + json.dumps(first_call, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logger.stop()
        # give process a moment to release resources
        time.sleep(0.2)
