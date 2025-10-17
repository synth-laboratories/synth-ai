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
    for candidate in (".env.test.prod", ".env.test", ".env.test.dev", ".env"):
        p = _repo_root() / candidate
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _normalize_backend(base: str | None) -> str | None:
    if not base:
        return None
    url = str(base).strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url if url.endswith("/api") else f"{url}/api"


def _is_local(url: str | None) -> bool:
    if not url:
        return True
    u = url.strip().lower()
    return u.startswith("http://localhost") or u.startswith("http://127.0.0.1") or u.startswith("https://127.0.0.1")


async def _wait_for_success(backend_base: str, api_key: str, job_id: str, timeout_s: float, interval_s: float) -> dict:
    async with JobsClient(base_url=backend_base, api_key=api_key, timeout=timeout_s) as client:
        import time, asyncio

        deadline = time.time() + timeout_s
        last: dict = {}
        while time.time() < deadline:
            last = await client._http.get(f"/api/learning/jobs/{job_id}")
            status = str(last.get("status", "")).lower()
            if status in {"succeeded", "completed"}:
                return last
            if status in {"failed", "errored", "error", "cancelled", "canceled"}:
                raise AssertionError(f"Job {job_id} ended unsuccessfully: {last}")
            await asyncio.sleep(max(interval_s, 1.0))
        raise AssertionError(f"Timed out waiting for job {job_id} to succeed; last={last}")


@pytest.mark.parametrize(
    "config_rel, expect_deepspeed",
    (
        ("tests/artifacts/configs/fft.small.toml", False),
        ("tests/artifacts/configs/lora.small.toml", True),
    ),
)
def test_cli_train_sft_small_configs(tmp_path: Path, config_rel: str, expect_deepspeed: bool) -> None:
    _maybe_env()
    raw_backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
        or os.getenv("DEV_BACKEND_URL")
    )
    backend = _normalize_backend(raw_backend)
    if _is_local(backend):
        # Try to recover a non-local prod URL if any are set
        for key in ("PROD_BACKEND_URL", "BACKEND_BASE_URL", "SYNTH_BASE_URL"):
            cand = _normalize_backend(os.getenv(key))
            if cand and not _is_local(cand):
                backend = cand
                break
    api_key = os.getenv("SYNTH_API_KEY")
    if not backend or not api_key:
        pytest.skip("SYNTH_API_KEY and backend base URL required for SFT CLI integration test")

    config_path = _repo_root() / config_rel

    poll_timeout = os.getenv("SYNTH_TRAIN_TEST_POLL_TIMEOUT", "300")
    poll_interval = os.getenv("SYNTH_TRAIN_TEST_POLL_INTERVAL", "10")

    # Provide an explicit env file to skip interactive .env selection in CLI
    envfile = tmp_path / "sft.env"
    envfile.write_text(f"SYNTH_API_KEY={api_key}\n", encoding="utf-8")

    cmd = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "sft",
        "--config",
        str(config_path),
        "--dataset",
        str(_repo_root() / "tests" / "artifacts" / "datasets" / "crafter_reject_sft.small.jsonl"),
        "--env-file",
        str(envfile),
        "--backend",
        backend or "",
        "--no-poll",
        "--poll-timeout",
        poll_timeout,
        "--poll-interval",
        poll_interval,
    ]

    env = os.environ.copy()
    env.setdefault("SYNTH_GPU_TYPE", "H100-4x")
    # Enable verbose CLI logging (streamed)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("SYNTH_HTTP_DEBUG", "1")
    env.setdefault("SYNTH_VERBOSE", "1")

    print("[itest] Running:", " ".join(cmd))
    # Stream subprocess output live AND capture it for assertions
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    try:
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)
        rc = proc.wait(timeout=int(float(poll_timeout)) + 60)
    finally:
        try:
            proc.kill()
        except Exception:
            pass
    output = "".join(lines)
    if rc != 0:
        pytest.fail(
            "CLI SFT train failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {rc}\n"
            f"OUTPUT (last 4000 chars):\n{output[-4000:]}\n"
        )
    if proc.returncode != 0:
        pytest.fail(
            "CLI SFT train failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    assert "✓ Job created" in output
    if expect_deepspeed:
        assert '"use_deepspeed": true' in output
    else:
        assert '"use_deepspeed": false' in output

    m = _JOB_ID_PATTERN.search(output)
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
