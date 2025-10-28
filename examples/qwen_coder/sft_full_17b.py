#!/usr/bin/env python3
"""Submit a full-parameter SFT job for Qwen/Qwen3-1.7B via Synth API."""

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

    data_path = ensure_tiny_dataset()
    file_id = await client.upload_training_file(str(data_path))

    validation_file_id: str | None = None
    val_path = optional_validation_dataset()
    if val_path and val_path.exists():
        validation_file_id = await client.upload_training_file(str(val_path))

    hyper: dict[str, Any] = {
        "n_epochs": int(os.getenv("QWEN_CODER_FULL_EPOCHS", "1")),
        "per_device_batch": int(os.getenv("QWEN_CODER_FULL_PER_DEVICE", "1")),
        "gradient_accumulation_steps": int(os.getenv("QWEN_CODER_FULL_ACCUM", "8")),
        "sequence_length": int(os.getenv("QWEN_CODER_FULL_SEQ_LEN", "4096")),
        "learning_rate": float(os.getenv("QWEN_CODER_FULL_LR", "2e-5")),
        "warmup_ratio": float(os.getenv("QWEN_CODER_FULL_WARMUP", "0.05")),
        "train_kind": os.getenv("QWEN_CODER_FULL_TRAIN_KIND", "full"),
    }

    metadata = {
        "example": "qwen_coder_full_17b",
        "effective_config": {
            "compute": {
                "gpu_type": os.getenv("SYNTH_GPU_TYPE", "H100"),
                "gpu_count": int(os.getenv("SYNTH_GPU_COUNT", "4")),
                "nodes": int(os.getenv("SYNTH_GPU_NODES", "1")),
            }
        },
    }

    job = await client.create_job(
        training_type="sft_offline",
        model=os.getenv("QWEN_CODER_FULL_MODEL", "Qwen/Qwen3-1.7B"),
        training_file_id=file_id,
        hyperparameters=hyper,
        metadata=metadata,
        validation_file=validation_file_id,
    )
    job_id = str(job.get("id") or job.get("job_id") or "").strip()
    if not job_id:
        raise SystemExit(f"Invalid create_job response: {job}")

    await client.start_job(job_id)

    timeout_seconds = float(os.getenv("SYNTH_TIMEOUT", "7200"))
    poll_interval = float(os.getenv("QWEN_CODER_FULL_POLL_INTERVAL", "10"))

    job_final = await client.poll_until_terminal(
        job_id,
        interval_seconds=poll_interval,
        max_seconds=timeout_seconds,
    )

    status = str(job_final.get("status"))
    print(f"Job status: {status}")
    result_model = (
        job_final.get("result", {}).get("model_id")
        if isinstance(job_final.get("result"), dict)
        else None
    )
    print(f"Model ID: {result_model}")
    try:
        out_file = resolve_output_path("ft_model_id_full.txt")
        text = (result_model or "").strip()
        if text:
            out_file.write_text(text + "\n", encoding="utf-8")
            print(f"Wrote {out_file} with ft model id")
    except Exception as exc:
        print(f"Warning: failed to write ft_model_id_full.txt: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

