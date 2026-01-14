#!/usr/bin/env python3
"""Run GEPA prompt optimization for the PTCG ReAct agent using an optimized verifier."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from localapi_ptcg import INSTANCE_IDS, PTCG_REACT_SYSTEM_PROMPT, app
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

parser = argparse.ArgumentParser(description="Run GEPA prompt optimization for PTCG")
parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
parser.add_argument("--local-host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8017, help="Port for task app")
parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Policy model")
parser.add_argument(
    "--verifier-path",
    type=str,
    default="demos/gepa_ptcg/artifacts/verifier_opt.json",
    help="Path to verifier optimization artifact JSON",
)
parser.add_argument("--budget", type=int, default=30, help="Rollout budget per candidate")
parser.add_argument("--generations", type=int, default=3, help="Number of GEPA generations")
parser.add_argument("--train-seeds", type=int, default=6, help="Number of training seeds")
parser.add_argument("--val-seeds", type=int, default=3, help="Number of validation seeds")
parser.add_argument(
    "--out",
    type=str,
    default="demos/gepa_ptcg/artifacts/prompt_opt.json",
    help="Path to write prompt optimization artifacts",
)
args = parser.parse_args()


def wait_for_health(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
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
    print("=" * 60)
    print("POKEMON TCG - GEPA PROMPT OPTIMIZATION (ReAct)")
    print("=" * 60)

    if args.local:
        backend_url = f"http://{args.local_host}:8000"
        print(f"LOCAL MODE - {backend_url}")
    else:
        backend_url = PROD_BASE_URL
        print(f"PROD MODE - {backend_url}")

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{backend_url}/health", timeout=10)
            print(f"Backend health: {r.status_code}")
        except Exception as e:
            print(f"Backend check failed: {e}")
            return

    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("No SYNTH_API_KEY, minting demo key...")
        api_key = mint_demo_api_key(backend_url=backend_url)
        print(f"API Key: {api_key[:20]}...")

    env_key = ensure_localapi_auth(backend_base=backend_url, synth_api_key=api_key)

    port = acquire_port(args.port, on_conflict=PortConflictBehavior.FIND_NEW)
    run_server_background(app, port)
    wait_for_health(args.local_host, port, env_key)
    print(f"Task app ready on port {port}")

    task_url = f"http://{args.local_host}:{port}"

    verifier_path = Path(args.verifier_path)
    if not verifier_path.exists():
        raise FileNotFoundError(f"Verifier artifact not found: {verifier_path}")
    verifier_data = json.loads(verifier_path.read_text(encoding="utf-8"))
    verifier_graph_id = verifier_data.get("graph_id")
    if not verifier_graph_id:
        raise RuntimeError("verifier artifact missing graph_id")

    train_seeds = list(range(args.train_seeds))
    val_start = args.train_seeds
    validation_seeds = list(range(val_start, val_start + args.val_seeds))

    config_dict = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": task_url,
            "task_app_id": "ptcg",
            "policy": {
                "model": args.model,
                "provider": "openai",
                "temperature": 0.3,
                "max_completion_tokens": 512,
            },
            "initial_prompt": {
                "id": "ptcg_react_baseline",
                "name": "PTCG ReAct Baseline",
                "messages": [
                    {
                        "role": "system",
                        "pattern": PTCG_REACT_SYSTEM_PROMPT,
                        "order": 0,
                    }
                ],
                "wildcards": {},
            },
            "gepa": {
                "env_name": "ptcg",
                "rollout": {
                    "budget": args.budget,
                    "max_concurrent": 3,
                    "minibatch_size": 3,
                },
                "evaluation": {
                    "seeds": train_seeds,
                    "validation_seeds": validation_seeds,
                    "validation_top_k": 3,
                },
                "population": {
                    "initial_size": 4,
                    "num_generations": args.generations,
                    "children_per_generation": 3,
                },
                "mutation": {"rate": 0.3},
                "archive": {"size": 10, "pareto_set_size": 10},
                "token": {"max_limit": 4000, "counting_model": "gpt-4", "max_spend_usd": 50.0},
            },
            "verifier": {
                "enabled": True,
                "reward_source": "fused",
                "backend_base": backend_url,
                "backend_provider": "openai",
                "backend_model": args.model,
                "verifier_graph_id": verifier_graph_id,
                "backend_outcome_enabled": True,
                "backend_event_enabled": True,
                "concurrency": 1,
                "timeout": 240.0,
                "weight_env": 0.6,
                "weight_event": 0.2,
                "weight_outcome": 0.2,
            },
        }
    }

    job = PromptLearningJob.from_dict(
        config_dict=config_dict,
        backend_url=backend_url,
        api_key=api_key,
        task_app_api_key=env_key,
    )

    job_id = job.submit()
    print(f"Job ID: {job_id}")

    result = job.poll_until_complete(timeout=3600.0, interval=10.0, progress=True)
    print(f"Job status: {result.status}")
    if result.failed:
        print(f"Job failed: {result.error}")
        return

    pl_client = PromptLearningClient(backend_url, api_key)
    prompt_results = await pl_client.get_prompts(job_id)
    optimized_prompt = None
    if prompt_results.best_prompt:
        for msg in prompt_results.best_prompt.get("messages", []):
            if msg.get("role") == "system":
                optimized_prompt = msg.get("pattern") or msg.get("content")
                break

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_value = getattr(result.status, "value", str(result.status))
    payload = {
        "job_id": job_id,
        "status": status_value,
        "verifier_graph_id": verifier_graph_id,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "baseline_prompt": PTCG_REACT_SYSTEM_PROMPT,
        "optimized_prompt": optimized_prompt,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if optimized_prompt:
        prompt_path = output_path.parent / "optimized_prompt.txt"
        prompt_path.write_text(optimized_prompt, encoding="utf-8")
        print(f"Optimized prompt saved to {prompt_path}")

    print(f"Wrote artifacts to {output_path}")
    print(f"Available instances: {len(INSTANCE_IDS)}")


if __name__ == "__main__":
    asyncio.run(main())
