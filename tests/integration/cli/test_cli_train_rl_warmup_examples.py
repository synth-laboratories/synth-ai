import os
import re
import subprocess
from pathlib import Path

import pytest

from synth_ai.jobs.client import JobsClient

pytestmark = pytest.mark.integration


_JOB_ID_PATTERN = re.compile(r"Job created \(id=([^\)]+)\)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _maybe_env() -> None:
    if os.getenv("SYNTH_API_KEY") and (
        os.getenv("PROD_BACKEND_URL") or os.getenv("BACKEND_BASE_URL") or os.getenv("SYNTH_BASE_URL")
    ):
        return
    for candidate in (".env.test.dev", ".env.test", ".env"):
        p = _repo_root() / candidate
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _deploy_or_lookup_task_app(tmp_dir: Path) -> str | None:
    if os.getenv("TASK_APP_URL"):
        return os.getenv("TASK_APP_URL")
    envfile = tmp_dir / "modal.env"
    contents = []
    ak = os.getenv("SYNTH_API_KEY")
    if ak:
        contents.append(f"SYNTH_API_KEY={ak}")
    envk = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("DEV_ENVIRONMENT_API_KEY") or ""
    if envk:
        contents.append(f"ENVIRONMENT_API_KEY={envk}")
    if contents:
        envfile.write_text("\n".join(contents) + "\n", encoding="utf-8")
    repo = _repo_root()
    cmd = ["uv", "run", "synth-ai", "task-app", "deploy", "grpo-crafter", "--name", "grpo-crafter-task-app"]
    if envfile.exists():
        cmd.extend(["--env-file", str(envfile)])
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True, env=env, timeout=600)
    try:
        if envfile.exists():
            for line in envfile.read_text(encoding="utf-8").splitlines():
                if line.startswith("TASK_APP_BASE_URL="):
                    url = line.split("=", 1)[1].strip()
                    if url:
                        os.environ.setdefault("TASK_APP_URL", url)
                        return url
    except Exception:
        pass
    for line in (proc.stdout or "").splitlines():
        if "modal.run" in line:
            candidate = line.strip().split()[-1]
            if candidate.startswith("http"):
                os.environ.setdefault("TASK_APP_URL", candidate)
                return candidate
    return None


async def _wait_for_success(backend_base: str, api_key: str, job_id: str, timeout_s: float, interval_s: float) -> dict:
    async with JobsClient(base_url=backend_base, api_key=api_key, timeout=timeout_s) as client:
        import time

        deadline = time.time() + timeout_s
        last: dict = {}
        while time.time() < deadline:
            last = await client._http.get(f"/api/learning/jobs/{job_id}")
            status = str(last.get("status", "")).lower()
            if status in {"succeeded", "completed"}:
                return last
            if status in {"failed", "errored", "error", "cancelled", "canceled"}:
                raise AssertionError(f"Job {job_id} ended unsuccessfully: {last}")
            await client._http._sleep(max(interval_s, 1.0))
        raise AssertionError(f"Timed out waiting for job {job_id} to succeed; last={last}")


@pytest.mark.fast
def test_cli_train_rl_from_base(tmp_path: Path) -> None:
    _maybe_env()
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
        or os.getenv("DEV_BACKEND_URL")
    )
    api_key = os.getenv("SYNTH_API_KEY")
    task_url = os.getenv("TASK_APP_URL") or _deploy_or_lookup_task_app(tmp_path)
    if not backend or not api_key or not task_url:
        pytest.skip("SYNTH_API_KEY, DEV_BACKEND_URL, and TASK_APP_URL required for RL integration test")

    cfg = _repo_root() / "examples" / "warming_up_to_rl" / "configs" / "rl_from_base_qwen4b.toml"

    poll_timeout = os.getenv("SYNTH_TRAIN_TEST_POLL_TIMEOUT", "300")
    poll_interval = os.getenv("SYNTH_TRAIN_TEST_POLL_INTERVAL", "10")

    cmd = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "rl",
        "--config",
        str(cfg),
        "--backend",
        backend,
        "--task-url",
        task_url,
        "--no-poll",
        "--poll-timeout",
        poll_timeout,
        "--poll-interval",
        poll_interval,
    ]

    env = os.environ.copy()

    proc = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        text=True,
        capture_output=True,
        env=env,
        timeout=int(float(poll_timeout)) + 60,
    )
    if proc.returncode != 0:
        pytest.fail(
            "CLI RL train failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    assert "✓ Job created" in proc.stdout

    m = _JOB_ID_PATTERN.search(proc.stdout)
    assert m, "job id not found in output"
    job_id = m.group(1)

    import asyncio

    final = asyncio.run(
        _wait_for_success(
            backend.rstrip("/"),
            api_key,
            job_id,
            timeout_s=float(poll_timeout),
            interval_s=float(poll_interval),
        )
    )

    status = str(final.get("status", "")).lower()
    assert status in {"succeeded", "completed"}
