#!/usr/bin/env python3
"""
Crafter → SFT end-to-end runner (single script).

Pipeline:
 1) Read v3 traces DB (sqld/Turso) and filter sessions (achievements >= min)
 2) Export OpenAI-format JSONL
 3) Upload file, create/start SFT job, poll to terminal
 4) (Optional) quick inference with the resulting model

Usage:
  uv run python examples/finetuning/synth_qwen_v1/run_crafter_sft_job.py --mode dev \
    --db /Users/joshpurtell/Documents/GitHub/synth-ai/traces/v3/synth_ai.db/dbs/default/data \
    --min-achievements 2 --output examples/finetuning/synth_qwen_v1/data/training_crafter.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path for local runs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from synth_ai.learning import FtClient, JobHandle, validate_training_jsonl  # type: ignore
from synth_ai.inference import InferenceClient  # type: ignore
from examples.finetuning.synth_qwen_v1.util import load_env, save_state  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crafter traces → SFT JSONL → FT job runner")
    p.add_argument("--mode", choices=["local", "dev", "prod"], default=None)
    p.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parents[3] / "traces/v3/synth_ai.db/dbs/default/data"),
        help="Path to sqld internal data file or sqlite+aiosqlite URL",
    )
    p.add_argument(
        "--output", default=str(Path(__file__).parent / "data" / "training_crafter.jsonl")
    )
    p.add_argument("--min-achievements", type=int, default=2)
    p.add_argument("--max-cost", type=float, default=10.0)
    p.add_argument("--max-tokens", type=int, default=100000)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--models", nargs="*", help="Optional model name filter (any match)")
    return p.parse_args()


def _normalize_db_url(raw: str) -> str:
    if raw.endswith(".db") and not raw.startswith("sqlite"):
        return f"sqlite+aiosqlite:///{raw}"
    if raw.startswith("sqlite+aiosqlite:///"):
        return raw
    if raw.startswith("sqlite:///") and raw.endswith(".db"):
        return raw.replace("sqlite:///", "sqlite+aiosqlite:///")
    return raw


async def extract_jsonl_from_traces(db_url: str, output_path: str, cfg: dict[str, Any]) -> int:
    # Import extractor with robust fallbacks across dist variants
    Extractor = None
    try:
        from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft.filter_traces_sft_turso import (  # type: ignore
            FinetuningDataExtractorV3 as _Ex,
        )

        Extractor = _Ex
    except Exception:
        try:
            from synth_ai.environments.examples.crafter_classic.agent_demos.crafter_openai_ft.filter_traces_sft_turso import (  # type: ignore
                FinetuningDataExtractorV3 as _Ex,
            )

            Extractor = _Ex
        except Exception as e:
            raise ImportError("FinetuningDataExtractorV3 not available in current build") from e

    filters: dict[str, Any] = cfg.get("filters", {})
    min_ach = int(filters.get("min_achievements", 2))
    max_cost = float(filters.get("max_cost", 10.0))
    max_tokens = int(filters.get("max_tokens", 100000))
    models: list[str] = list(filters.get("models", []) or [])

    kept: list[str] = []
    async with Extractor(db_url) as ex:
        sessions = await ex.get_all_sessions()
        for _, row in sessions.iterrows():
            sid = row["session_id"]
            metrics = await ex.get_session_metrics(sid)
            if float(metrics.get("total_cost", 0.0)) > max_cost:
                continue
            if int(metrics.get("total_tokens", 0) or 0) > max_tokens:
                continue
            # Optional model filter
            if models:
                model_df = await ex.db_manager.query_traces(
                    """
                    SELECT DISTINCT model_name
                    FROM events
                    WHERE session_id = :session_id
                      AND event_type = 'cais'
                      AND model_name IS NOT NULL
                    """,
                    {"session_id": sid},
                )
                session_models = (
                    model_df["model_name"].tolist()
                    if model_df is not None and not model_df.empty
                    else []
                )
                if not any(m in session_models for m in models):
                    continue
            ach = await ex.get_session_achievements(sid) or []
            if len([a for a in ach if a]) >= min_ach:
                kept.append(sid)

        data = await ex.extract_openai_format(kept)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for exm in data:
                f.write(json.dumps(exm) + "\n")
        return len(data)


async def run(args: argparse.Namespace) -> None:
    base_url, api_key = load_env(args.mode)

    # 1) Filter and export JSONL from v3 traces
    db_url = _normalize_db_url(args.db)
    cfg = {
        "mode": "trajectory",
        "filters": {
            "min_achievements": int(args.min_achievements),
            "max_cost": float(args.max_cost),
            "max_tokens": int(args.max_tokens),
            "models": args.models or [],
        },
    }
    out_path = str(Path(args.output))
    print("Extracting SFT data from traces…")
    n = await extract_jsonl_from_traces(db_url, out_path, cfg)
    print(f"✅ Wrote {n} examples → {out_path}")

    # 2) Validate JSONL
    validate_training_jsonl(out_path)

    # 3) Upload and create FT job
    client = FtClient(base_url=base_url, api_key=api_key)
    file_id = await client.upload_training_file(Path(out_path), purpose="fine-tune")
    print(f"file_id={file_id}")
    save_state({"file_id": file_id})

    create = await client.create_sft_job(
        model=str(args.model),
        training_file_id=file_id,
        hyperparameters={"n_epochs": int(args.epochs), "batch_size": int(args.batch_size)},
        metadata={"upload_to_wasabi": True},
    )
    job_id = (create or {}).get("job_id")
    if not job_id:
        raise RuntimeError(f"create_sft_job missing job_id: {create}")
    print(f"job_id={job_id}")
    save_state({"job_id": job_id})

    start = await client.start_job(job_id)
    print(f"start={start}")

    # 4) Poll to terminal
    handle = JobHandle(base_url, api_key, job_id, strict=True)
    final = await handle.poll_until_terminal(interval_seconds=2.0, max_seconds=1800)
    status = (final or {}).get("status")
    print(f"final_status={status}")
    ft_model = (final or {}).get("fine_tuned_model")
    if ft_model:
        save_state({"fine_tuned_model": ft_model})
        print(f"fine_tuned_model={ft_model}")

    # 5) Optional inference check
    if not args.no_infer:
        try:
            ic = InferenceClient(base_url=base_url, api_key=api_key)
            model_for_infer = ft_model or str(args.model)
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
            print(f"(inference skipped due to error: {e})")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
