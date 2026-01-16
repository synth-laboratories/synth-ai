#!/usr/bin/env python3
"""
Run a minimal GEPA job for hello_world_bench.

Usage:
    uv run python demos/hello_world_bench/run_gepa_minimal.py --model gpt-5-nano --generations 2
"""

import argparse
import asyncio
import time
from pathlib import Path

import httpx
from localapi_hello_world_bench import app
from synth_ai.core.urls import synth_health_url
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.auth import get_or_mint_synth_user_key
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

SYNTH_USER_KEY = get_or_mint_synth_user_key()

parser = argparse.ArgumentParser(description="Run minimal GEPA for hello_world_bench")
parser.add_argument("--port", type=int, default=8030)
parser.add_argument("--model", type=str, default="gpt-5-nano")
parser.add_argument("--timeout", type=int, default=120, help="Agent timeout per rollout")
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to GEPA config file (relative to this directory). Auto-selected based on --agent if not specified.",
)
parser.add_argument("--budget", type=int, help="Override rollout budget")
parser.add_argument("--generations", type=int, help="Override number of generations")
parser.add_argument(
    "--agent", type=str, default="opencode", choices=["opencode", "codex"], help="Agent to use"
)
args = parser.parse_args()

# Auto-select config based on agent if not specified
if args.config is None:
    if args.agent == "codex":
        args.config = "hello_world_gepa_codex.toml"
    else:
        args.config = "hello_world_gepa_minimal.toml"


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


async def main() -> None:
    async with httpx.AsyncClient() as client:
        r = await client.get(synth_health_url(), timeout=10)
        print(f"Synth health: {r.status_code}")

    print(f"API Key: {SYNTH_USER_KEY[:20]}...")

    # Acquire port and prepare localapi URL
    port = acquire_port(args.port, on_conflict=PortConflictBehavior.FIND_NEW)
    localapi_url = f"http://localhost:{port}"

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")

    import tomllib

    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)

    # Apply overrides
    config_dict["prompt_learning"]["localapi_url"] = localapi_url
    config_dict["prompt_learning"]["policy"]["model"] = args.model
    if (
        "policy" in config_dict["prompt_learning"]
        and "config" in config_dict["prompt_learning"]["policy"]
    ):
        config_dict["prompt_learning"]["policy"]["config"]["timeout"] = args.timeout

    # Ensure policy config includes the timeout used by the task app
    config_dict.setdefault("prompt_learning", {}).setdefault("policy", {})
    config_dict["prompt_learning"]["policy"].setdefault("config", {})
    config_dict["prompt_learning"]["policy"]["config"]["timeout"] = args.timeout
    config_dict["prompt_learning"]["policy"]["config"]["agent"] = args.agent

    if args.budget is not None:
        config_dict.setdefault("prompt_learning", {}).setdefault("gepa", {}).setdefault(
            "rollout", {}
        )
        config_dict["prompt_learning"]["gepa"]["rollout"]["budget"] = int(args.budget)

    if args.generations is not None:
        config_dict.setdefault("prompt_learning", {}).setdefault("gepa", {}).setdefault(
            "population", {}
        )
        config_dict["prompt_learning"]["gepa"]["population"]["num_generations"] = int(
            args.generations
        )

    print("Submitting GEPA job...")
    job = PromptLearningJob.from_dict(
        config_dict=config_dict,
        synth_user_key=SYNTH_USER_KEY,
    )

    # Start localapi server
    run_server_background(app, port)
    _wait_for_health("localhost", port, job.localapi_key)
    print(f"Localapi ready: {localapi_url}")

    job_id = job.submit()
    print(f"Job ID: {job_id}")

    result = job.poll_until_complete(timeout=7200.0, interval=15.0, progress=True)
    print(f"Status: {result.status}")

    if result.failed:
        print(f"Job failed: {result.error}")
        if result.raw and result.raw.get("error"):
            print(f"Error details: {result.raw['error']}")
    else:
        if result.best_prompt:
            print("Best prompt (preview):")
            if isinstance(result.best_prompt, str):
                print(result.best_prompt[:500] + ("..." if len(result.best_prompt) > 500 else ""))
            elif isinstance(result.best_prompt, dict):
                messages = result.best_prompt.get("messages", [])
                for msg in messages:
                    role = msg.get("role", "unknown")
                    pattern = msg.get("pattern", "")
                    print(f"\n[{role.upper()}]")
                    print(pattern[:400] + ("..." if len(pattern) > 400 else ""))


if __name__ == "__main__":
    asyncio.run(main())
