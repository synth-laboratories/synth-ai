#!/usr/bin/env python3
"""
Run a GEPA job for EngineBench using the public synth_ai SDK.

This script:
- starts the EngineBench task app locally
- submits a GEPA prompt-learning job to the backend
- polls until completion
"""

import argparse
import asyncio
import time
from pathlib import Path

import httpx

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from localapi_engine_bench import INSTANCE_IDS, app
from synth_ai.core.urls import synth_health_url
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.auth import get_or_mint_synth_api_key
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    from synth_ai.sdk.task import run_server_background

SYNTH_USER_KEY = get_or_mint_synth_api_key()


def _wait_for_health(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, headers=headers, timeout=5.0)
            if r.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {url}")


def _load_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run EngineBench GEPA job (SDK)")
    parser.add_argument("--port", type=int, default=8020, help="Task app port")
    parser.add_argument(
        "--config",
        type=str,
        default="enginebench_gepa_quick.toml",
        help="Path to GEPA config file",
    )
    parser.add_argument("--budget", type=int, help="Override rollout budget")
    parser.add_argument("--generations", type=int, help="Override number of generations")
    args = parser.parse_args()

    print(f"Instances available: {len(INSTANCE_IDS)}")

    # Check backend health
    async with httpx.AsyncClient() as client:
        r = await client.get(synth_health_url(), timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Backend not healthy: {r.status_code}")

    print(f"API Key: {SYNTH_USER_KEY[:20]}...")

    port = acquire_port(args.port, on_conflict=PortConflictBehavior.FIND_NEW)
    localapi_url = f"http://localhost:{port}"
    print(f"Localapi URL: {localapi_url}")

    config_path = Path(__file__).parent / args.config
    config_dict = _load_config(config_path)
    prompt_cfg = config_dict.get("prompt_learning")
    if not isinstance(prompt_cfg, dict):
        raise RuntimeError(f"Config {config_path} must contain a [prompt_learning] section")
    prompt_cfg["localapi_url"] = localapi_url

    if args.budget is not None:
        prompt_cfg.setdefault("gepa", {}).setdefault("rollout", {})["budget"] = args.budget
    if args.generations is not None:
        prompt_cfg.setdefault("gepa", {}).setdefault("population", {})["num_generations"] = (
            args.generations
        )

    print("\nSubmitting GEPA job...")
    job = PromptLearningJob.from_dict(
        config_dict=config_dict,
        synth_user_key=SYNTH_USER_KEY,
    )

    run_server_background(app, port)
    _wait_for_health("localhost", port, job.localapi_key)
    print(f"Localapi ready: {localapi_url}")

    job_id = job.submit()
    print(f"Job ID: {job_id}")

    print("Polling for results...")
    result = job.poll_until_complete(timeout=7200.0, interval=15.0, progress=True)
    print(f"Status: {result.status}")
    if result.failed:
        print(f"Job failed: {result.error}")
        return 1

    if result.best_score is not None:
        print(f"Best score: {result.best_score:.4f}")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
