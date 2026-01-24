#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from synth_ai.sdk import run_in_process_job
from synth_ai.sdk.localapi.auth import mint_environment_api_key, setup_environment_api_key
CONFIG_PATH = Path(__file__).resolve().parent / "gepa_mtg_image_config.toml"
TASK_APP_PATH = Path(__file__).resolve().parent / "mtg_image_task_app.py"

DEFAULT_SYNTH_API_BASE = "https://api.usesynth.ai"


def _ensure_api_key() -> str:
    api_key = os.getenv("SYNTH_API_KEY", "").strip()
    if api_key:
        return api_key

    resp = httpx.post(
        f"{os.environ.get('SYNTH_API_BASE', DEFAULT_SYNTH_API_BASE)}/api/demo/keys",
        json={"ttl_hours": 4},
        timeout=30,
    )
    resp.raise_for_status()
    api_key = resp.json()["api_key"]
    os.environ["SYNTH_API_KEY"] = api_key
    return api_key


def _load_verifier_graph_id(artist_key: str, artifact_path: str | None) -> str:
    if artifact_path:
        path = Path(artifact_path)
    else:
        path = Path(__file__).resolve().parent / "artifacts" / artist_key / "verifier_opt.json"

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph_id = payload.get("graph_id")
        if graph_id:
            return str(graph_id)

    return "zero_shot_verifier_rubric_single"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTG artist style GEPA job.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--artist", type=str, default="seb_mckinnon")
    parser.add_argument("--verifier-graph-id", type=str, default=None)
    parser.add_argument("--verifier-artifact", type=str, default=None)
    parser.add_argument("--local", action="store_true", help="Use localhost backend")
    args = parser.parse_args()

    if args.local:
        os.environ["SYNTH_API_BASE"] = "http://127.0.0.1:8000"
    synth_api_base = os.getenv("SYNTH_API_BASE", DEFAULT_SYNTH_API_BASE)

    api_key = _ensure_api_key()

    env_key = mint_environment_api_key()
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    setup_environment_api_key(synth_api_base, api_key, token=env_key)

    os.environ["MTG_ARTIST_KEY"] = args.artist

    verifier_graph_id = args.verifier_graph_id or _load_verifier_graph_id(
        artist_key=args.artist, artifact_path=args.verifier_artifact
    )

    overrides = {
        "prompt_learning.env_config.artist_key": args.artist,
        "prompt_learning.verifier.verifier_graph_id": verifier_graph_id,
        "prompt_learning.verifier.backend_base": synth_api_base,
    }

    print(f"Using verifier graph: {verifier_graph_id}")
    print(f"Starting MTG GEPA job (artist={args.artist})...")

    result = await run_in_process_job(
        job_type="prompt_learning",
        config_path=args.config,
        task_app_path=TASK_APP_PATH,
        backend_url=synth_api_base,
        api_key=api_key,
        poll=True,
        poll_interval=5.0,
        timeout=1800.0,
        overrides=overrides,
        port=8119,
    )

    status = result.status
    if isinstance(status, dict):
        print("Job status:", status.get("status"))
        if status.get("best_score") is not None:
            print("Best score:", status.get("best_score"))
        if status.get("error"):
            print("Error:", status.get("error"))
    else:
        print("Job status:", status)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
