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
import os
import threading
import time
from pathlib import Path

import httpx

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from localapi_engine_bench import INSTANCE_IDS, app
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.core.tunnels.tunneled_api import TunneledLocalAPI, TunnelBackend
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    from synth_ai.sdk.task import run_server_background


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
    parser.add_argument("--local", action="store_true", help="Use localhost backend")
    parser.add_argument("--local-host", type=str, default="localhost")
    parser.add_argument("--no-tunnel", action="store_true", help="Skip tunnel, use localhost for task app (requires local Temporal worker)")
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

    backend_url = f"http://{args.local_host}:8000" if args.local else BACKEND_URL_BASE
    print(f"Backend: {backend_url}")
    print(f"Instances available: {len(INSTANCE_IDS)}")

    # Check backend
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{backend_url}/health", timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Backend not healthy: {r.status_code}")

    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("No SYNTH_API_KEY, minting demo key...")
        api_key = mint_demo_api_key(backend_url=backend_url)
        os.environ["SYNTH_API_KEY"] = api_key
    print(f"API Key: {api_key[:20]}...")

    env_key = ensure_localapi_auth(
        backend_base=backend_url,
        synth_api_key=api_key,
    )
    print(f"Environment key: {env_key[:12]}...")

    port = acquire_port(args.port, on_conflict=PortConflictBehavior.FIND_NEW)
    run_server_background(app, port)
    _wait_for_health(args.local_host, port, env_key)

    # Use managed tunnel when talking to remote backend
    tunnel = None
    if args.local or args.no_tunnel:
        task_url = f"http://{args.local_host}:{port}"
    else:
        print("Setting up managed tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            backend=TunnelBackend.CloudflareManagedLease,
            api_key=api_key,
            env_api_key=env_key,
            progress=True,
        )
        task_url = tunnel.url
    print(f"Task app ready: {task_url}")

    # Start keepalive thread to prevent tunnel from going stale
    keepalive_stop = threading.Event()
    def _keepalive_loop(url: str, stop: threading.Event) -> None:
        while not stop.wait(15.0):
            try:
                httpx.get(f"{url}/health", timeout=5.0)
            except Exception:
                pass
    if tunnel is not None:
        keepalive_thread = threading.Thread(
            target=_keepalive_loop, args=(task_url, keepalive_stop), daemon=True
        )
        keepalive_thread.start()

    config_path = Path(__file__).parent / args.config
    config_dict = _load_config(config_path)
    prompt_cfg = config_dict.get("prompt_learning")
    if not isinstance(prompt_cfg, dict):
        raise RuntimeError(f"Config {config_path} must contain a [prompt_learning] section")
    prompt_cfg["task_app_url"] = task_url

    if args.budget is not None:
        prompt_cfg.setdefault("gepa", {}).setdefault("rollout", {})["budget"] = args.budget
    if args.generations is not None:
        prompt_cfg.setdefault("gepa", {}).setdefault("population", {})["num_generations"] = (
            args.generations
        )

    print("\nSubmitting GEPA job...")
    job = PromptLearningJob.from_dict(
        config_dict=config_dict,
        backend_url=backend_url,
        api_key=api_key,
        task_app_api_key=env_key,
        skip_health_check=True,
    )
    job_id = job.submit()
    print(f"Job ID: {job_id}")

    print("Polling for results...")
    result = job.poll_until_complete(timeout=7200.0, interval=15.0, progress=True)
    keepalive_stop.set()
    print(f"Status: {result.status}")
    if result.failed:
        print(f"Job failed: {result.error}")
        if tunnel is not None:
            tunnel.close()
        return 1

    if result.best_score is not None:
        print(f"Best score: {result.best_score:.4f}")
    if tunnel is not None:
        tunnel.close()
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
