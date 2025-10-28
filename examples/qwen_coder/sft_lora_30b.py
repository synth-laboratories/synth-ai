#!/usr/bin/env python3
"""Submit a LoRA SFT job for Qwen/Qwen3-Coder-30B-A3B-Instruct via Synth API.

Steps:
  - Generate a tiny coder dataset if missing
  - Upload the JSONL
  - Create the job with coder LoRA hyperparameters
  - Start and poll until terminal, then print the resulting model id

Env:
  SYNTH_API_KEY (required)
  BACKEND_BASE_URL (defaults to https://agent-learning.onrender.com/api)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from examples.qwen_coder._shared import (
    ensure_tiny_dataset,
    optional_validation_dataset,
    resolve_output_path,
)
from synth_ai.learning.client import LearningClient


def _backend() -> str:
    raw = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api").strip()
    return raw if raw.endswith("/api") else (raw + "/api")


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SYNTH_API_KEY required in env")

    backend = _backend()
    client = LearningClient(base_url=backend, api_key=api_key, timeout=60.0)

    # Ensure dataset exists
    data_path = ensure_tiny_dataset()

    # Upload training file
    file_id = await client.upload_training_file(str(data_path))

    # Optional validation file if present alongside training set
    val_path = optional_validation_dataset()
    validation_file_id: str | None = None
    if val_path and val_path.exists():
        validation_file_id = await client.upload_training_file(str(val_path))

    # Minimal hyperparameters for LoRA SFT (aligned with coder_lora_30b.toml)
    hyper: dict[str, Any] = {
        "n_epochs": 1,
        "per_device_batch": 1,
        "gradient_accumulation_steps": 64,
        "sequence_length": 4096,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.03,
        "train_kind": "peft",
    }

    # Create job
    job = await client.create_job(
        training_type="sft_offline",
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        training_file_id=file_id,
        hyperparameters=hyper,
        metadata={
            "example": "qwen_coder_lora_30b",
            # Include effective compute hints for backend routing/validation
            "effective_config": {
                "compute": {"gpu_type": "H100", "gpu_count": 4, "nodes": 1}
            },
        },
        validation_file=validation_file_id,
    )
    job_id = str(job.get("id"))
    if not job_id:
        raise SystemExit(f"Invalid create_job response: {job}")

    # Start
    await client.start_job(job_id)

    # Poll until terminal
    job_final = await client.poll_until_terminal(job_id, interval_seconds=5.0, max_seconds=7200)
    status = str(job_final.get("status"))
    print(f"Job status: {status}")
    # Print resulting model id if available and write to ft_data/ft_model_id.txt
    result_model = (
        job_final.get("result", {}).get("model_id")
        if isinstance(job_final.get("result"), dict)
        else None
    )
    print(f"Model ID: {result_model}")
    try:
        out_file = resolve_output_path("ft_model_id.txt")
        text = (result_model or "").strip()
        if text:
            out_file.write_text(text + "\n", encoding="utf-8")
            print(f"Wrote {out_file} with ft model id")
    except Exception as exc:
        # Best-effort write; don't crash if filesystem issues
        print(f"Warning: failed to write ft_model_id.txt: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
