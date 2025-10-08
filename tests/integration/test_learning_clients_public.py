from __future__ import annotations

import os
import json
import asyncio
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]
try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

from synth_ai.learning.client import LearningClient
from synth_ai.inference.client import InferenceClient
from synth_ai.config.base_url import get_backend_from_env


pytestmark = [pytest.mark.integration, pytest.mark.public]

# Load environment from .env at repo root
load_dotenv()

# Skip this entire module unless explicitly enabled. These are public-integration tests
# and should not run during unit test runs.
if os.getenv("RUN_PUBLIC_INTEGRATION_TESTS", "").strip() != "1":  # pragma: no cover
    pytest.skip("Skipping public integration tests by default; set RUN_PUBLIC_INTEGRATION_TESTS=1 to enable.", allow_module_level=True)


def _prod_backend_url() -> str:
    url = os.getenv("PROD_BACKEND_URL", "").strip()
    if not url:
        base, _ = get_backend_from_env()
        url = base.rstrip("/")
        if not url:
            pytest.skip("PROD_BACKEND_URL not set and no backend override available")
    return url.rstrip("/")


def _api_key() -> str:
    # Try multiple possible key names, prioritizing TEST_API_KEY which we verified works
    key = (
        os.getenv("TEST_API_KEY", "").strip() or
        os.getenv("TESTING_PROD_SYNTH_API_KEY", "").strip() or
        os.getenv("SYNTH_API_KEY", "").strip()
    )
    if not key:
        pytest.skip("No valid API key found in environment/.env for public integration tests")
    return key


@pytest.mark.asyncio
async def test_chat_completion_public_prod() -> None:
    base = _prod_backend_url()
    api_key = _api_key()

    client = InferenceClient(base_url=base, api_key=api_key, timeout=10.0)
    resp = await client.create_chat_completion(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "Say 'hello world'"}],
        max_tokens=32,
        temperature=0.0,
        stream=False,
    )
    assert isinstance(resp, dict)
    assert "choices" in resp
    assert isinstance(resp["choices"], list)
    assert len(resp["choices"]) >= 1


@pytest.mark.asyncio
async def test_learning_minimal_flow_smoke_prod() -> None:
    base = _prod_backend_url()
    api_key = _api_key()

    # Prepare a tiny JSONL in-memory
    tmp = Path("temp/_int_sft.jsonl")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join([
        json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})
    ]))

    lc = LearningClient(base_url=base, api_key=api_key, timeout=10.0)
    file_id = await lc.upload_training_file(tmp, purpose="fine-tune")
    assert isinstance(file_id, str) and len(file_id) > 0

    job = await lc.create_job(
        training_type="sft_offline",
        model="Qwen/Qwen3-0.6B",
        training_file_id=file_id,
        hyperparameters={"n_epochs": 1, "batch_size": 1},
    )
    job_id = job.get("job_id")
    assert isinstance(job_id, str)

    start = await lc.start_job(job_id)
    assert isinstance(start, dict)

    # Poll briefly just to ensure endpoints respond (do not wait full training)
    status = await lc.get_job(job_id)
    assert status.get("job_id") == job_id

    evs = await lc.get_events(job_id, since_seq=0, limit=5)
    assert isinstance(evs, list)

    # Do not assert terminal; this is a smoke test hitting prod
    # Clean up local file only
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

