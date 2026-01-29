#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend
from synth_ai.core.utils.paths import REPO_ROOT, temporary_import_paths
from synth_ai.sdk import find_available_port, is_port_available, kill_port
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.localapi.auth import mint_environment_api_key, setup_environment_api_key
from synth_ai.sdk.localapi._impl.apps_common import get_asgi_app, load_module
from synth_ai.sdk.localapi._impl.server import run_server_background
CONFIG_PATH = Path(__file__).resolve().parent / "gepa_mtg_image_config.toml"
TASK_APP_PATH = Path(__file__).resolve().parent / "mtg_image_task_app.py"
ROOT_DIR = Path(__file__).resolve().parents[2]

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


def _load_task_app():
    with temporary_import_paths(TASK_APP_PATH, REPO_ROOT):
        module = load_module(TASK_APP_PATH, f"_mtg_task_app_{os.getpid()}")
    return get_asgi_app(module)


def _task_app_healthcheck(host: str, port: int, api_key: str | None) -> bool:
    url = f"http://{host}:{port}/health"
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = httpx.get(url, headers=headers, timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _wait_for_task_app(host: str, port: int, api_key: str | None, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _task_app_healthcheck(host, port, api_key):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Task app health check failed after {timeout:.0f}s")


async def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
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

    gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not gemini_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is required for gemini-2.5-flash-image."
        )
    os.environ.setdefault("GEMINI_API_KEY", gemini_key)

    api_key = _ensure_api_key()

    env_key = mint_environment_api_key()
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    setup_environment_api_key(synth_api_base, api_key, token=env_key)

    os.environ["MTG_ARTIST_KEY"] = args.artist
    os.environ["MTG_POLICY_PROVIDER"] = "google"
    os.environ["MTG_POLICY_MODEL"] = "gemini-2.5-flash-image"

    verifier_graph_id = args.verifier_graph_id or _load_verifier_graph_id(
        artist_key=args.artist, artifact_path=args.verifier_artifact
    )

    overrides = {
        "prompt_learning.env_config.artist_key": args.artist,
        "prompt_learning.verifier.verifier_graph_id": verifier_graph_id,
        "prompt_learning.verifier.backend_base": synth_api_base,
        "prompt_learning.policy.provider": "google",
        "prompt_learning.policy.model": "gemini-2.5-flash-image",
        "prompt_learning.policy.api_key": gemini_key,
    }

    print(f"Using verifier graph: {verifier_graph_id}")
    print(f"Starting MTG GEPA job (artist={args.artist})...")
    app = _load_task_app()
    host = "127.0.0.1"
    port = 8119

    kill_port(port)
    if not is_port_available(port):
        port = find_available_port(port + 1)
        print(f"Port in use; switched to {port}")

    proc = run_server_background(app, port, host=host)
    _wait_for_task_app(host, port, env_key, timeout=90.0)

    tunnel = None
    try:
        if args.local:
            task_app_url = f"http://{host}:{port}"
        else:
            backend = TunnelBackend.CloudflareManagedLease
            print("Provisioning Cloudflare tunnel...")
            tunnel = await TunneledLocalAPI.create(
                local_port=port,
                backend=backend,
                api_key=api_key,
                env_api_key=env_key,
                backend_url=synth_api_base,
                progress=True,
            )
            task_app_url = tunnel.url
            print(f"Tunnel URL: {task_app_url}")

        overrides["prompt_learning.task_app_url"] = task_app_url
        overrides["task_app_url"] = task_app_url
        overrides["task_url"] = task_app_url

        job = PromptLearningJob.from_config(
            config_path=args.config,
            backend_url=synth_api_base,
            api_key=api_key,
            task_app_api_key=env_key,
            overrides=overrides,
        )

        job_id = job.submit()
        print(f"Job submitted: {job_id}")

        result = job.poll_until_complete(timeout=1800.0, interval=5.0, progress=True)
        status = result.status.value if hasattr(result.status, "value") else result.status
        print("Job status:", status)
        if result.succeeded:
            if result.best_score is not None:
                print("Best score:", result.best_score)
        if result.failed:
            print("Error:", result.error)
    finally:
        if tunnel is not None:
            tunnel.close()
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
