import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from synth_ai.http import HTTPError
from synth_ai.jobs.client import JobsClient


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _load_env_prod_only() -> None:
    """Best-effort load of .env.test.prod for credentialed runs."""
    prod_env = Path(os.getcwd()) / ".env.test.prod"
    if not prod_env.exists():
        return
    try:
        for raw in prod_env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        # Allow tests to proceed even if the env file is malformed.
        pass


def _derive_backend_base_url_prod() -> Optional[str]:
    """Pick the best available prod backend root (no trailing /v1)."""
    direct = os.getenv("SYNTH_BASE_URL")
    if direct:
        direct = direct.rstrip("/")
        return direct[:-3] if direct.endswith("/v1") else direct
    prod_backend = os.getenv("PROD_BACKEND_URL")
    if prod_backend:
        return prod_backend.rstrip("/")
    dev_backend = os.getenv("DEV_BACKEND_URL")
    if dev_backend:
        return dev_backend.rstrip("/")
    return None


async def _poll_rl_job(
    backend_api: str,
    api_key: str,
    job_id: str,
    *,
    interval_s: float,
    timeout_s: float,
) -> dict:
    async with JobsClient(base_url=backend_api.rstrip("/"), api_key=api_key, timeout=timeout_s) as client:
        deadline = time.time() + timeout_s
        last: dict = {}
        while time.time() < deadline:
            last = await client.rl.retrieve(job_id)
            status = str(last.get("status") or "").lower()
            if status in {"succeeded", "completed"}:
                return last
            if status in {"failed", "errored", "error", "cancelled", "canceled"}:
                raise AssertionError(f"RL job {job_id} ended unsuccessfully: {last}")
            await asyncio.sleep(interval_s)
        raise TimeoutError(f"Timed out waiting for RL job {job_id} to succeed; last={last}")


@pytest.mark.parametrize(
    ("config_rel_path", "variant"),
    [
        ("tests/artifacts/configs/rl.lora.small.toml", "lora"),
        ("tests/artifacts/configs/rl.full.small.toml", "full"),
    ],
)
def test_rl_crafter_train_prod(
    monkeypatch: pytest.MonkeyPatch,
    config_rel_path: str,
    variant: str,
) -> None:
    """Trigger the Crafter RL example (LoRA + full) against the hosted agent-learning backend."""
    _load_env_prod_only()

    backend_root = _derive_backend_base_url_prod()
    api_key = os.getenv("SYNTH_API_KEY")
    task_url = os.getenv("TASK_APP_URL")
    if not backend_root or not api_key or not task_url:
        pytest.skip("SYNTH_API_KEY, PROD_BACKEND_URL (or SYNTH_BASE_URL), and TASK_APP_URL required for RL prod test")

    backend_api = backend_root.rstrip("/")
    if not backend_api.endswith("/api"):
        backend_api = f"{backend_api}/api"

    config_path = Path(config_rel_path)
    if not config_path.exists():
        pytest.skip(f"RL config missing: {config_path}")

    require_success = os.getenv("RL_TEST_REQUIRE_SUCCESS", os.getenv("LEARNING_TEST_REQUIRE_SUCCESS", "0")) == "1"
    poll_timeout = float(os.getenv("RL_TEST_TIMEOUT", "900"))
    poll_interval = float(os.getenv("RL_TEST_POLL_INTERVAL", "20"))

    import importlib

    rl_module = importlib.import_module("examples.warming_up_to_rl.run_rl_and_save")

    captured: dict[str, object] = {}

    real_post = rl_module.requests.post

    def _instrumented_post(*args, **kwargs):
        response = real_post(*args, **kwargs)
        captured["response"] = response
        return response

    monkeypatch.setattr(rl_module.requests, "post", _instrumented_post)

    argv = [
        "run_rl_and_save.py",
        "--backend",
        backend_api,
        "--config",
        str(config_path),
        "--task-url",
        task_url,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        rl_module.main()

    exit_code = exit_info.value.code
    if exit_code not in (None, 0):
        pytest.fail(f"RL example ({variant}) exited with code {exit_code}")

    resp = captured.get("response")
    if resp is None:
        pytest.fail(f"RL example ({variant}) did not issue backend POST request")

    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover - best-effort
        pytest.fail(f"Unable to decode RL job response JSON: {exc}")

    job_id = str(data.get("job_id") or data.get("id") or "").strip()
    if not job_id:
        pytest.fail(f"RL job response missing identifier: {data}")

    try:
        final = asyncio.run(
            _poll_rl_job(
                backend_api,
                api_key,
                job_id,
                interval_s=poll_interval,
                timeout_s=poll_timeout,
            )
        )
    except HTTPError as exc:
        if exc.status in (401, 403):
            pytest.skip(f"RL polling requires elevated credentials: {exc}")
        raise
    except TimeoutError as exc:
        if require_success:
            raise
        pytest.xfail(f"{variant} RL job timed out: {exc}")
    else:
        status = str(final.get("status") or "").lower()
        assert status in {"succeeded", "completed"}, f"RL job {job_id} ({variant}) ended with unexpected status: {final}"
