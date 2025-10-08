#!/usr/bin/env python3
"""
Kick off Qwen 4B SFT against the learning-v2 service using the exact
upload/job/polling flow mirrored from test_qwen3_sft_training_v2.py.

Environment:
- LEARNING_V2_BASE_URL (preferred)
- SYNTH_BASE_URL (fallback if LEARNING_V2_BASE_URL is unset)
- else defaults to http://localhost:8000/api
- SYNTH_API_KEY
- QWEN_BASE_MODEL (optional, defaults to Qwen/Qwen3-4B-Instruct-2507)
- QWEN_TRAINING_JSONL (optional, defaults to ft_data/qwen4b_crafter_sft.jsonl)
"""

import asyncio
import os
import time
import tomllib
from typing import Any

import aiohttp
from synth_ai.config.base_url import get_learning_v2_base_url

API_URL = get_learning_v2_base_url()
API_KEY = os.getenv("SYNTH_API_KEY")

_cfg_path = os.getenv("CRAFTER_CONFIG", "examples/finetuning/synth_qwen/config.toml")
_cfg: dict[str, Any] = {}
if os.path.exists(_cfg_path):
    with open(_cfg_path, "rb") as _f:
        _cfg = tomllib.load(_f)
scfg = _cfg.get("sft", {})

MODEL = os.getenv("QWEN_BASE_MODEL", scfg.get("base_model", "Qwen/Qwen3-4B-Instruct-2507"))
TRAINING_PATH = os.getenv(
    "QWEN_TRAINING_JSONL", scfg.get("training_jsonl", "ft_data/qwen4b_crafter_sft.jsonl")
)


async def upload_file() -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        with open(TRAINING_PATH, "rb") as f:
            form.add_field(
                "file",
                f,
                filename=os.path.basename(TRAINING_PATH),
                content_type="application/jsonl",
            )
            form.add_field("purpose", "fine-tune")
            async with session.post(f"{API_URL}/files", data=form, headers=headers) as resp:
                assert resp.status == 200, await resp.text()
                data = await resp.json()
                return data["id"]


async def create_job(file_id: str) -> str:
    body = {
        "training_file": file_id,
        "model": MODEL,
        "hyperparameters": {
            "training_type": "sft",
            "n_epochs": int(scfg.get("n_epochs", 1)),
            "batch_size": int(scfg.get("batch_size", 4)),
        },
        "upload_to_wasabi": bool(scfg.get("upload_to_wasabi", True)),
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with (
        aiohttp.ClientSession() as session,
        session.post(f"{API_URL}/fine_tuning/jobs", json=body, headers=headers) as resp,
    ):
        assert resp.status == 200, await resp.text()
        data = await resp.json()
        return data["id"]


async def await_success(job_id: str) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with aiohttp.ClientSession() as session:
        check_interval_seconds = 15
        for attempt in range(20):
            async with session.get(f"{API_URL}/fine_tuning/jobs/{job_id}", headers=headers) as resp:
                if resp.status != 200:
                    await asyncio.sleep(check_interval_seconds)
                    continue
                job = await resp.json()
                status = job.get("status")
                print(f"⏳ poll {attempt + 1}/20 – status = {status}")
                if status == "succeeded":
                    return job
                if status in {"failed", "cancelled"}:
                    raise RuntimeError(f"Training failed: {job.get('error')}")
            await asyncio.sleep(check_interval_seconds)
    raise TimeoutError("Training did not finish in time")


async def main() -> None:
    if not API_URL or not API_KEY:
        raise RuntimeError(
            "LEARNING_V2_BASE_URL/SYNTH_BASE_URL and SYNTH_API_KEY must be set or use the default http://localhost:8000/api"
        )
    print("🚀 Starting Qwen 4B SFT")
    fid = await upload_file()
    job_id = await create_job(fid)
    start = time.time()
    job = await await_success(job_id)
    wall = time.time() - start

    ft_model = job["fine_tuned_model"]
    tokens = job.get("trained_tokens")

    print("🟢 Qwen4B SFT fine-tune succeeded →", ft_model)
    print(f"⏱️ wall-clock: {wall:.1f}s | trained_tokens: {tokens}")


if __name__ == "__main__":
    asyncio.run(main())
