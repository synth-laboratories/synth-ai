#!/usr/bin/env python3
"""
Light-weight test that performs a minimal SFT (supervised fine-tuning) job on the
smallest GPT-OSS model currently supported by the Synth backend.

The goal is **not** to exhaustively test quality – only to exercise the end-to-end
pipeline:

1. Upload a tiny JSONL training file.
2. Kick off an SFT job via `/api/fine_tuning/jobs`.
3. Poll the job until it reaches a terminal state.
4. Report the resulting fine-tuned `model_id` so that follow-up tests can run
   inference against it.

The script mirrors the structure of `test_synth_qwen_ft.py` and keeps the runtime
short by using only a handful of training examples and one epoch.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTH_API_URL = os.environ.get("SYNTH_API_URL", "http://localhost:8000")
SYNTH_API_KEY = os.environ.get("SYNTH_API_KEY", "")

# Use the smallest GPT-OSS variant for speed.
BASE_MODEL_ID = "openai/gpt-oss-20b"

# Minimal training data: a few multilingual reasoning examples.
TRAINING_DATA: List[Dict] = [
    {
        "messages": [
            {"role": "system", "content": "reasoning language: English"},
            {"role": "user", "content": "What is 12 × 7?"},
            {"role": "assistant", "thinking": "12 × 7 = 84", "content": "84"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "reasoning language: Spanish"},
            {"role": "user", "content": "¿Cuántos días hay en una semana?"},
            {
                "role": "assistant",
                "thinking": "La semana tiene 7 días.",
                "content": "Hay 7 días en una semana.",
            },
        ]
    },
]

HEADERS = {
    "Authorization": f"Bearer {SYNTH_API_KEY}",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def _upload_training_file(data: List[Dict]) -> Optional[str]:
    """Upload training data as a JSONL file and return the file ID."""
    content = "\n".join(json.dumps(item) for item in data)

    files = {
        "file": ("training.jsonl", content, "application/jsonl"),
        "purpose": (None, "fine-tune"),
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{SYNTH_API_URL}/api/files", files=files, headers=HEADERS
        )
        if response.status_code == 200:
            return response.json()["id"]  # type: ignore[index]

    print(f"✗ File upload failed: {response.status_code} – {response.text[:200]}")
    return None


async def _wait_for_job(job_id: str, timeout: int = 180) -> Dict[str, str]:
    """Poll `/api/fine_tuning/jobs/{job_id}` until the job finishes or times out."""
    start = time.time()
    last_status = ""

    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() - start < timeout:
            resp = await client.get(
                f"{SYNTH_API_URL}/api/fine_tuning/jobs/{job_id}", headers=HEADERS
            )
            if resp.status_code != 200:
                await asyncio.sleep(5)
                continue

            data = resp.json()
            status = data.get("status", "unknown")
            if status != last_status:
                elapsed = int(time.time() - start)
                print(f"  [{elapsed}s] Status: {status}")
                last_status = status

            if status in {"succeeded", "completed", "failed", "cancelled"}:
                return data  # Terminal state reached.

            # Back-off polling over time.
            await asyncio.sleep(5 if time.time() - start < 60 else 10)

    return {"status": "timeout"}


# ---------------------------------------------------------------------------
# Main test logic
# ---------------------------------------------------------------------------

async def run_finetune() -> Optional[str]:
    """Execute the full fine-tuning workflow and return the new `model_id`."""

    if not SYNTH_API_KEY:
        print("❌ SYNTH_API_KEY is not set – aborting fine-tune test")
        return None

    print("\n" + "=" * 60)
    print("GPT-OSS SFT FINE-TUNE TEST")
    print("=" * 60)
    print(f"API URL  : {SYNTH_API_URL}")
    print(f"Base model: {BASE_MODEL_ID}")

    # 1. Upload training data.
    print("\n📁 Uploading training data…")
    file_id = await _upload_training_file(TRAINING_DATA)
    if not file_id:
        print("❌ Training file upload failed – cannot continue")
        return None
    print(f"✅ Uploaded file ID: {file_id}")

    # 2. Kick off the fine-tuning job.
    job_payload = {
        "model": BASE_MODEL_ID,
        "training_file": file_id,
        "hyperparameters": {
            "training_type": "sft",
            "n_epochs": 1,
            "batch_size": 4,
            "learning_rate_multiplier": 0.4,
            "warmup_ratio": 0.03,
            "gradient_accumulation_steps": 4,
            "lr_scheduler_type": "cosine_with_min_lr",
        },
    }

    print("\n🚀 Creating fine-tuning job…")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{SYNTH_API_URL}/api/fine_tuning/jobs",
            json=job_payload,
            headers={**HEADERS, "Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            print(f"❌ Job creation failed: {resp.status_code} – {resp.text[:200]}")
            return None
        job_id = resp.json()["id"]  # type: ignore[index]
    print(f"✅ Fine-tuning job created: {job_id}")

    # 3. Wait for job completion.
    print("\n⏳ Waiting for job to finish…")
    result = await _wait_for_job(job_id, timeout=600)
    status = result.get("status", "unknown")
    if status not in {"succeeded", "completed"}:
        print(f"❌ Job did not complete successfully – status = {status}")
        return None

    model_id = result["fine_tuned_model"]  # type: ignore[index]
    print(f"\n🎉 Fine-tuning succeeded! New model ID: {model_id}")
    return model_id


async def main() -> None:
    model_id = await run_finetune()
    if model_id:
        # Print in a parse-friendly way so that the next test can grab it.
        print(f"\n::GPT_OSS_FINE_TUNED_MODEL::{model_id}")


if __name__ == "__main__":
    asyncio.run(main())
