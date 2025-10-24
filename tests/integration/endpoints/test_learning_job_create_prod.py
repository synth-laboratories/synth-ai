import os
import time
import uuid
import json
import tempfile
from pathlib import Path
import pytest

from synth_ai.jobs.client import JobsClient
from synth_ai._utils.http import HTTPError


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _load_env_prod_only() -> None:
    prod_env = os.path.join(os.getcwd(), ".env.test.prod")
    if os.path.exists(prod_env):
        try:
            with open(prod_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass


def _derive_backend_base_url_prod() -> str | None:
    base = os.getenv("SYNTH_BASE_URL")
    if base:
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base
    prod_backend = os.getenv("PROD_BACKEND_URL")
    if prod_backend:
        return prod_backend.rstrip("/")
    dev_backend = os.getenv("DEV_BACKEND_URL")
    if dev_backend:
        return dev_backend.rstrip("/")
    return None


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
async def test_create_full_finetune_job_prod() -> None:
    _load_env_prod_only()
    base_url = _derive_backend_base_url_prod()
    api_key = os.getenv("SYNTH_API_KEY")
    if not base_url or not api_key:
        pytest.skip("backend base URL and SYNTH_API_KEY required for prod test")

    async with JobsClient(base_url=base_url, api_key=api_key, timeout=60.0) as client:
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
        job = await client.sft.create(
            training_file=file_id,
            model=base_model,
            hyperparameters={"n_epochs": 1},
            metadata=_job_metadata("fft"),
            idempotency_key=f"sft-create-{file_id}",
        )
        assert isinstance(job, dict)
        assert job.get("id") and job.get("status")


@pytest.mark.asyncio
async def test_create_qlora_job_prod() -> None:
    _load_env_prod_only()
    base_url = _derive_backend_base_url_prod()
    api_key = os.getenv("SYNTH_API_KEY")
    if not base_url or not api_key:
        pytest.skip("backend base URL and SYNTH_API_KEY required for prod test")

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
        job = await client.sft.create(
            training_file=file_id,
            model=base_model,
            hyperparameters={"n_epochs": 1, "use_qlora": True},
            metadata=_job_metadata("qlora"),
            idempotency_key=f"sft-create-{file_id}",
        )
        assert isinstance(job, dict)
        assert job.get("id") and job.get("status")

