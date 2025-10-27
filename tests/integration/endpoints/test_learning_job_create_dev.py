import os
import time
import uuid
import json
import tempfile
from pathlib import Path
import pytest
import asyncio

from synth_ai.jobs.client import JobsClient
from synth_ai._utils.http import HTTPError


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _load_env_dev_first() -> None:
    env_path = os.path.join(os.getcwd(), ".env.test.dev")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k == "SYNTH_API_KEY":
                    # Always force API key from .env.test.dev
                    os.environ[k] = v
                elif k == "DEV_BACKEND_URL":
                    # Do not override shell-provided DEV_BACKEND_URL
                    os.environ.setdefault(k, v)
                else:
                    # Best-effort: do not clobber explicitly provided envs
                    os.environ.setdefault(k, v)
    except Exception:
        # best-effort env load
        pass


def _derive_backend_base_url() -> str | None:
    dev_backend = os.getenv("DEV_BACKEND_URL")
    if dev_backend:
        return dev_backend.rstrip("/")
    # Dev tests must not run against prod; if no DEV_BACKEND_URL, skip.
    return None


async def _start_and_wait_for_success(http, job_id: str) -> dict:
    """Start a learning job and poll until terminal.

    Respects environment variables:
      - LEARNING_TEST_TIMEOUT: total seconds to wait (default: 120)
      - LEARNING_TEST_POLL_INTERVAL: seconds between polls (default: 5)
      - LEARNING_TEST_REQUIRE_SUCCESS: if "1", fail on timeout; else xfail
    """
    # Start the job
    _ = await http.post_json(f"/api/learning/jobs/{job_id}/start", json={})

    timeout_s = float(os.getenv("LEARNING_TEST_TIMEOUT", "120"))
    poll_interval_s = float(os.getenv("LEARNING_TEST_POLL_INTERVAL", "5"))
    require_success = os.getenv("LEARNING_TEST_REQUIRE_SUCCESS", "0") == "1"

    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        last = await http.get(f"/api/learning/jobs/{job_id}")
        status = str(last.get("status", "")).lower()
        if status in {"succeeded", "completed"}:
            return last
        if status in {"failed", "error", "errored", "cancelled", "canceled"}:
            raise AssertionError(f"Job {job_id} terminal failure: status={status}, last={last}")
        await asyncio.sleep(poll_interval_s)

    # Timed out
    if require_success:
        raise AssertionError(f"Timed out waiting for job {job_id} to succeed; last={last}")
    import pytest as _pytest  # local import to avoid polluting module namespace
    _pytest.xfail(f"Timed out waiting for job {job_id} to succeed; last={last}")
    return last


def _job_metadata(kind: str) -> dict:
    gpu_type = os.getenv("SYNTH_GPU_TYPE") or os.getenv("TRAIN_GPU_TYPE") or "A10G"
    return {
        "sdk": "synth-ai",
        "kind": kind,
        "effective_config": {
            "compute": {
                "gpu_type": gpu_type,
            }
        },
    }


@pytest.mark.asyncio
@pytest.mark.slow
async def test_create_full_finetune_job_dev() -> None:
    _load_env_dev_first()
    base_url = _derive_backend_base_url()
    api_key = os.getenv("SYNTH_API_KEY")
    if not base_url or not api_key:
        pytest.skip("backend base URL and SYNTH_API_KEY required for dev test")

    # Use jobs API to avoid user-context requirement on learning files route
    async with JobsClient(base_url=base_url, api_key=api_key, timeout=60.0) as client:
        # Minimal JSONL training content written to a temp file
        filename = f"sdk-fft-{int(time.time())}.jsonl"
        content = (
            json.dumps(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "response": "ok",
                    "nonce": str(uuid.uuid4()),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / filename
            p.write_text(content, encoding="utf-8")
            try:
                up = await client.files.upload(
                    filename=p.name,
                    content=p.read_bytes(),
                    purpose="sft_training",
                    content_type="application/jsonl",
                    idempotency_key=f"files-{p.name}",
                )
                file_id = up["id"]
            except HTTPError as e:
                if getattr(e, "status", None) == 409:
                    # Backend deduped the content; generate a fresh valid JSONL line and retry once
                    new_line = json.dumps(
                        {
                            "messages": [{"role": "user", "content": "hi"}],
                            "response": "ok",
                            "nonce": str(uuid.uuid4()),
                        },
                        separators=(",", ":"),
                    )
                    p.write_text(new_line + "\n", encoding="utf-8")
                    up = await client.files.upload(
                        filename=p.name,
                        content=p.read_bytes(),
                        purpose="sft_training",
                        content_type="application/jsonl",
                        idempotency_key=f"files-{p.name}",
                    )
                    file_id = up["id"]
                else:
                    raise

        base_model = os.getenv("FFT_BASE_MODEL", "Qwen/Qwen3-0.6B")
        payload = {
            "training_type": "sft_offline",
            "training_file_id": file_id,
            "model": base_model,
            "hyperparameters": {"n_epochs": 1},
            "metadata": _job_metadata("fft"),
        }
        job = await client._http.post_json(
            "/api/learning/jobs",
            json=payload,
            headers={"Idempotency-Key": f"sft-create-{file_id}"},
        )
        assert isinstance(job, dict)
        assert job.get("job_id") and job.get("status")
        # Start and poll for terminal status (xfail on timeout unless LEARNING_TEST_REQUIRE_SUCCESS=1)
        _ = await _start_and_wait_for_success(client._http, job["job_id"])  # noqa: F841


@pytest.mark.asyncio
@pytest.mark.slow
async def test_create_qlora_job_dev() -> None:
    _load_env_dev_first()
    base_url = _derive_backend_base_url()
    api_key = os.getenv("SYNTH_API_KEY")
    if not base_url or not api_key:
        pytest.skip("backend base URL and SYNTH_API_KEY required for dev test")

    async with JobsClient(base_url=base_url, api_key=api_key, timeout=60.0) as client:
        filename = f"sdk-qlora-{int(time.time())}.jsonl"
        content = (
            json.dumps(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "response": "world",
                    "nonce": str(uuid.uuid4()),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / filename
            p.write_text(content, encoding="utf-8")
            try:
                up = await client.files.upload(
                    filename=p.name,
                    content=p.read_bytes(),
                    purpose="sft_training",
                    content_type="application/jsonl",
                    idempotency_key=f"files-{p.name}",
                )
                file_id = up["id"]
            except HTTPError as e:
                if getattr(e, "status", None) == 409:
                    new_line = json.dumps(
                        {
                            "messages": [{"role": "user", "content": "hello"}],
                            "response": "world",
                            "nonce": str(uuid.uuid4()),
                        },
                        separators=(",", ":"),
                    )
                    p.write_text(new_line + "\n", encoding="utf-8")
                    up = await client.files.upload(
                        filename=p.name,
                        content=p.read_bytes(),
                        purpose="sft_training",
                        content_type="application/jsonl",
                        idempotency_key=f"files-{p.name}",
                    )
                    file_id = up["id"]
                else:
                    raise

        base_model = os.getenv("QLORA_BASE_MODEL", "Qwen/Qwen3-0.6B")
        payload = {
            "training_type": "sft_offline",
            "training_file_id": file_id,
            "model": base_model,
            "hyperparameters": {"n_epochs": 1, "use_qlora": True},
            "metadata": _job_metadata("qlora"),
        }
        job = await client._http.post_json(
            "/api/learning/jobs",
            json=payload,
            headers={"Idempotency-Key": f"sft-create-{file_id}"},
        )
        assert isinstance(job, dict)
        assert job.get("job_id") and job.get("status")
        # Start and poll for terminal status (xfail on timeout unless LEARNING_TEST_REQUIRE_SUCCESS=1)
        _ = await _start_and_wait_for_success(client._http, job["job_id"])  # noqa: F841

