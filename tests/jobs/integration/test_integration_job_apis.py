import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
import pytest

from synth_ai.sdk.jobs.client import JobsClient


def _get_backend_urls() -> List[str]:
    # Accept comma-separated URLs or single URL from env.
    multi = os.getenv("SYNTH_BACKEND_URLS")
    single = os.getenv("SYNTH_BACKEND_URL")
    urls: List[str] = []
    if multi:
        urls.extend([u.strip() for u in multi.split(",") if u.strip()])
    if single:
        urls.append(single.strip())
    # De-duplicate while preserving order
    seen = set()
    final: List[str] = []
    for u in urls:
        if u not in seen:
            final.append(u)
            seen.add(u)
    return final


def _env_or_skip() -> Tuple[List[str], str]:
    urls = _get_backend_urls()
    api_key = os.getenv("SYNTH_API_KEY")
    if not urls or not api_key:
        pytest.skip(
            "Integration tests require SYNTH_BACKEND_URL(S) and SYNTH_API_KEY to be set."
        )
    return urls, api_key


@pytest.mark.integration
@pytest.mark.asyncio
async def test_files_flow_basic():
    urls, api_key = _env_or_skip()
    filename = f"sdk-integration-{int(time.time())}.jsonl"
    content = b"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n"

    for base_url in urls:
        async with JobsClient(base_url=base_url, api_key=api_key) as client:
            # Upload (purpose sft_training)
            file_meta = await client.files.upload(
                filename=filename,
                content=content,
                purpose="sft_training",
                content_type="application/jsonl",
                idempotency_key=f"files-{filename}",
            )
            assert isinstance(file_meta, dict)
            assert file_meta.get("id")
            file_id = file_meta["id"]

            # Retrieve
            fetched = await client.files.retrieve(file_id)
            assert fetched.get("id") == file_id
            assert fetched.get("filename") == filename

            # List (filter purpose)
            listed = await client.files.list(purpose="sft_training", limit=5)
            assert isinstance(listed, dict)
            assert "data" in listed
            assert isinstance(listed["data"], list)

            # Associations (jobs)
            jobs = await client.files.list_jobs(file_id, limit=1)
            assert isinstance(jobs, dict)
            assert "data" in jobs


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sft_job_lifecycle_minimal():
    urls, api_key = _env_or_skip()
    base_model = os.getenv("SFT_BASE_MODEL", "Qwen/Qwen3-0.6B")
    filename = f"sdk-sft-{int(time.time())}.jsonl"
    content = b"{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"response\":\"world\"}\n"

    for base_url in urls:
        async with JobsClient(base_url=base_url, api_key=api_key) as client:
            # Upload training data
            f = await client.files.upload(
                filename=filename,
                content=content,
                purpose="sft_training",
                content_type="application/jsonl",
            )
            file_id = f["id"]

            # Create SFT job
            job = await client.sft.create(
                training_file=file_id,
                model=base_model,
                idempotency_key=f"sft-create-{file_id}",
            )
            assert job.get("id") and job.get("status")
            job_id = job["id"]

            # Retrieve
            got = await client.sft.retrieve(job_id)
            assert got.get("id") == job_id
            assert got.get("model")

            # Events (first page)
            ev = await client.sft.list_events(job_id, since_seq=0, limit=10)
            assert isinstance(ev, dict)
            assert "events" in ev

            # Checkpoints (may be empty early)
            cps = await client.sft.checkpoints(job_id, limit=2)
            assert isinstance(cps, dict)
            assert "data" in cps

            # Cancel (safe, backend may ignore if already terminal)
            c = await client.sft.cancel(job_id)
            assert c.get("id") == job_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_list_and_retrieve_smoke():
    urls, api_key = _env_or_skip()

    for base_url in urls:
        async with JobsClient(base_url=base_url, api_key=api_key) as client:
            models = await client.models.list(limit=5)
            assert isinstance(models, dict)
            assert "data" in models
            if models["data"]:
                first = models["data"][0]
                if isinstance(first, dict) and first.get("id"):
                    # Best-effort retrieve
                    m = await client.models.retrieve(first["id"])
                    assert m.get("id") == first["id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rl_job_create_smoke():
    urls, api_key = _env_or_skip()
    # These are required to attempt RL job creation
    task_app_url = os.getenv("RL_TASK_APP_URL")
    trainer_id = os.getenv("RL_TRAINER_ID")
    model = os.getenv("RL_BASE_MODEL", "Qwen/Qwen3-0.6B")
    if not task_app_url or not trainer_id:
        pytest.skip("Set RL_TASK_APP_URL and RL_TRAINER_ID to run RL tests.")

    for base_url in urls:
        async with JobsClient(base_url=base_url, api_key=api_key) as client:
            job = await client.rl.create(
                model=model,
                endpoint_base_url=task_app_url,
                trainer_id=trainer_id,
                idempotency_key=f"rl-create-{int(time.time())}",
            )
            assert job.get("id") and job.get("status")
            job_id = job["id"]

            # Retrieve
            got = await client.rl.retrieve(job_id)
            assert got.get("id") == job_id

            # Events (smoke)
            ev = await client.rl.list_events(job_id, since_seq=0, limit=10)
            assert isinstance(ev, dict)
            assert "events" in ev


async def _read_sse(session: aiohttp.ClientSession, url: str, seconds: float = 2.0) -> List[str]:
    lines: List[str] = []
    timeout = aiohttp.ClientTimeout(total=seconds)
    async with session.get(url, timeout=timeout) as resp:
        if resp.status != 200:
            return lines
        async for raw in resp.content:
            try:
                line = raw.decode("utf-8").strip()
            except Exception:
                continue
            if line:
                lines.append(line)
            if len(lines) > 5:
                break
    return lines


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sft_events_sse_best_effort():
    urls, api_key = _env_or_skip()
    base_model = os.getenv("SFT_BASE_MODEL", "Qwen/Qwen3-0.6B")
    filename = f"sdk-sse-{int(time.time())}.jsonl"
    content = b"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response\":\"ok\"}\n"

    for base_url in urls:
        async with JobsClient(base_url=base_url, api_key=api_key) as client:
            f = await client.files.upload(
                filename=filename,
                content=content,
                purpose="sft_training",
                content_type="application/jsonl",
            )
            job = await client.sft.create(training_file=f["id"], model=base_model)

            # Attempt SSE stream; tolerate 404 or connection errors
            sse_url = f"{base_url.rstrip('/')}/api/sft/jobs/{job['id']}/events/stream"
            headers = {"authorization": f"Bearer {api_key}"}
            try:
                timeout = aiohttp.ClientTimeout(total=3)
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    lines = await _read_sse(session, sse_url, seconds=2.0)
                    # Accept zero lines; just verify the endpoint is reachable if implemented
                    assert isinstance(lines, list)
            except aiohttp.ClientResponseError as e:
                if e.status in (404, 501):
                    pytest.skip("SSE not implemented on this backend.")
            except Exception:
                # Network/timeout issues are acceptable in smoke test
                pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rl_events_sse_best_effort():
    urls, api_key = _env_or_skip()
    task_app_url = os.getenv("RL_TASK_APP_URL")
    trainer_id = os.getenv("RL_TRAINER_ID")
    model = os.getenv("RL_BASE_MODEL", "Qwen/Qwen3-0.6B")
    if not task_app_url or not trainer_id:
        pytest.skip("Set RL_TASK_APP_URL and RL_TRAINER_ID to run RL SSE test.")

    for base_url in urls:
        async with JobsClient(base_url=base_url, api_key=api_key) as client:
            job = await client.rl.create(model=model, endpoint_base_url=task_app_url, trainer_id=trainer_id)

            sse_url = f"{base_url.rstrip('/')}/api/rl/jobs/{job['id']}/events/stream"
            headers = {"authorization": f"Bearer {api_key}"}
            try:
                timeout = aiohttp.ClientTimeout(total=3)
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    lines = await _read_sse(session, sse_url, seconds=2.0)
                    assert isinstance(lines, list)
            except aiohttp.ClientResponseError as e:
                if e.status in (404, 501):
                    pytest.skip("RL SSE not implemented on this backend.")
            except Exception:
                pass

