#!/usr/bin/env python3
"""
End-to-end SFT workflow for Qwen (single script).

Steps performed:
 1) Ensure/validate training JSONL (creates a minimal one if missing)
 2) Upload training file → save file_id to state.json
 3) Create SFT job (Qwen/Qwen3-0.6B by default) and start → save job_id
 4) Poll until terminal → save fine_tuned_model when available
 5) (Optional) Quick inference with the fine-tuned model (or base if absent)

Usage:
  uv run python examples/finetuning/synth_qwen_v1/run_ft_job.py --mode dev

Options:
  --mode {local,dev,prod}     Backend mode/environment (default: env override or prod)
  --data PATH                 Path to training JSONL (default: ./data/training.jsonl)
  --model NAME                Base model for SFT (default: Qwen/Qwen3-0.6B)
  --epochs N                  Epochs (default: 1)
  --batch-size N              Batch size (default: 4)
  --no-infer                  Skip the post-training inference check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make repo root importable when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from synth_ai.config.base_url import get_backend_from_env
from synth_ai.learning import FtClient, JobHandle, validate_training_jsonl  # type: ignore
from synth_ai.inference import InferenceClient  # type: ignore
from examples.finetuning.synth_qwen_v1.util import load_env, load_state, save_state  # type: ignore

try:
    from examples.common.backend import resolve_backend_url as _resolve_backend_default  # type: ignore
except Exception:  # pragma: no cover - fallback for direct execution

    def _resolve_backend_default() -> str:
        base, _ = get_backend_from_env()
        base = base.rstrip("/")
        return base if base.endswith("/api") else f"{base}/api"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Qwen SFT end-to-end")
    p.add_argument("--mode", choices=["prod", "dev", "local"], default=None)
    p.add_argument("--data", default=str(Path(__file__).parent / "data" / "training_crafter.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    p.add_argument("--no-infer", action="store_true")
    return p.parse_args()


def ensure_training_jsonl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Minimal JSONL with a single example
        lines: list[str] = [
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Write a short greeting."},
                        {"role": "assistant", "content": "Hello there!"},
                    ]
                }
            )
        ]
        path.write_text("\n".join(lines) + "\n")
    # Validate using shared SDK validator
    validate_training_jsonl(path)
    return path


async def run(args: argparse.Namespace) -> None:
    # Resolve backend and key
    base_url, api_key = load_env(args.mode)
    # Force canonical prod base when prod mode (or override) is selected
    try:
        if (args.mode == "prod") or (
            os.getenv("SYNTH_BACKEND_URL_OVERRIDE", "").strip().lower() == "prod"
        ):
            base_url = _resolve_backend_default()
            # Also export for any downstream helpers that read env
            os.environ["PROD_BACKEND_URL"] = base_url
    except Exception:
        pass

    # Ensure/validate training JSONL
    data_path = ensure_training_jsonl(Path(args.data))
    print(f"Training JSONL: {data_path}")

    # Upload file
    ft = FtClient(base_url=base_url, api_key=api_key)
    file_id = await ft.upload_training_file(data_path, purpose="fine-tune")
    if not file_id:
        raise RuntimeError("upload_training_file returned empty file_id")
    print(f"file_id={file_id}")
    save_state({"file_id": file_id})

    # Create job
    hyperparameters: dict[str, Any] = {
        "n_epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
    }
    # Include explicit compute topology for billing/inference resolution.
    # Default: 1x A10G (can be surfaced via CLI later if needed).
    metadata = {
        "upload_to_wasabi": True,
        # Normalized effective config consumed by the backend SFT workflow
        "effective_config": {
            "compute": {
                "gpu_type": "A10G",
                "gpu_count": 1,
                "nodes": 1,
            },
            "data": {
                "topology": {
                    "gpu_type": "A10G",
                    "container_count": 1,
                }
            },
        },
    }

    create_resp = await ft.create_sft_job(
        model=str(args.model),
        training_file_id=file_id,
        hyperparameters=hyperparameters,
        metadata=metadata,
    )
    job_id = (create_resp or {}).get("job_id")
    if not job_id:
        raise RuntimeError(f"create_sft_job missing job_id: {create_resp}")
    print(f"job_id={job_id}")
    save_state({"job_id": job_id})

    # Start job
    start_resp = await ft.start_job(job_id)
    print(f"start={start_resp}")

    # Poll until terminal with streaming event/metric logs
    def _on_event(e: dict[str, Any]) -> None:
        try:
            seq = e.get("seq")
            etype = e.get("type") or e.get("event_type")
            msg = e.get("message")
            print(f"event seq={seq} type={etype} msg={msg}")
        except Exception:
            pass

    def _on_metric(p: dict[str, Any]) -> None:
        try:
            name = str(p.get("name") or "")
            step = p.get("step")
            epoch = p.get("epoch")
            val = p.get("value")
            print(f"metric {name} step={step} epoch={epoch} value={val}")
        except Exception:
            pass

    handle = JobHandle(base_url, api_key, job_id, strict=True)
    final = await handle.poll_until_terminal(
        interval_seconds=2.0,
        max_seconds=1800,
        on_event=_on_event,
        on_metric=_on_metric,
    )
    status = (final or {}).get("status")
    print(f"final_status={status}")
    ft_model = (final or {}).get("fine_tuned_model")
    if ft_model:
        print(f"fine_tuned_model={ft_model}")
        save_state({"fine_tuned_model": ft_model})

    # Optional: quick inference check
    if not args.no_infer:
        model_for_infer = ft_model or str(args.model)
        try:
            ic = InferenceClient(base_url=base_url, api_key=api_key, timeout=600.0)
            print(f"\nInference sanity check (model={model_for_infer})…")
            resp = await ic.create_chat_completion(
                model=model_for_infer,
                messages=[{"role": "user", "content": "Give me a cheerful two-line greeting."}],
                max_tokens=128,
                temperature=0.7,
                stream=False,
            )
            print(resp)
        except Exception as e:
            # Always print full error details and traceback
            import traceback

            try:
                from synth_ai.http import HTTPError  # type: ignore
            except Exception:  # pragma: no cover - fallback if import shape changes
                HTTPError = tuple()  # type: ignore
            print("\n===== Inference Error =====")
            print(f"Type: {type(e).__name__}")
            print(f"Repr: {repr(e)}")
            tb = traceback.format_exc()
            if tb:
                print("Traceback:")
                print(tb)
            # If HTTP error from backend, surface structured fields
            if "HTTPError" in str(type(e)) or (isinstance((), tuple) and False):
                pass
            try:
                if HTTPError and isinstance(e, HTTPError):  # type: ignore[arg-type]
                    print("HTTPError details:")
                    print(f"  status={e.status}")
                    print(f"  url={e.url}")
                    print(f"  message={e.message}")
                    if getattr(e, "detail", None) is not None:
                        print(f"  detail={e.detail}")
                    if getattr(e, "body_snippet", None):
                        print(f"  body_snippet={e.body_snippet}")
            except Exception:
                pass
            print("===== End Inference Error =====\n")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
