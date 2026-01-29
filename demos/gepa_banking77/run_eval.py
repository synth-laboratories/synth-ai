#!/usr/bin/env python3
"""Minimal eval job runner for debugging inference_url issues.

Usage:
    uv run python demos/gepa_banking77/run_eval.py --local
"""

import argparse
import asyncio
import json
import os
import time

import httpx
from datasets import load_dataset
from fastapi import Request
from openai import AsyncOpenAI
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi._impl.http_pool import get_shared_http_client
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port, TunneledLocalAPI, TunnelBackend

# Parse args early
parser = argparse.ArgumentParser(description="Run Banking77 eval job")
parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
parser.add_argument("--local-host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8016, help="Port for task app")
parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to eval")
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host
PORT = args.port
NUM_SEEDS = args.seeds

# Backend config
if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    print("=" * 60)
    print("LOCAL MODE - using localhost:8000 backend")
    print("=" * 60)
else:
    SYNTH_API_BASE = BACKEND_URL_BASE
    print(f"PROD MODE - using {SYNTH_API_BASE}")

os.environ["SYNTH_API_BASE"] = SYNTH_API_BASE

# Check backend health
r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
print(f"Backend health: {r.status_code}")

# API Key
API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("No SYNTH_API_KEY, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=SYNTH_API_BASE)
os.environ["SYNTH_API_KEY"] = API_KEY
print(f"API Key: {API_KEY[:20]}...")

# Environment Key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key: {ENVIRONMENT_API_KEY[:12]}...")

# Tool schema
TOOL_NAME = "banking77_classify"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the predicted banking77 intent label.",
        "parameters": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
}

BASELINE_SYSTEM_PROMPT = """You are a banking customer service intent classifier.
Given a customer query, classify it into one of the available banking intents.
Be precise and choose the most specific matching intent."""


class Banking77Dataset:
    def __init__(self):
        self._cache = {}
        self._label_names = None

    def _load_split(self, split: str):
        if split not in self._cache:
            ds = load_dataset("banking77", split=split, trust_remote_code=False)
            self._cache[split] = ds
            if self._label_names is None and hasattr(ds.features.get("label"), "names"):
                self._label_names = ds.features["label"].names
        return self._cache[split]

    def ensure_ready(self, splits):
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        return len(self._load_split(split))

    def sample(self, *, split: str, index: int) -> dict:
        ds = self._load_split(split)
        idx = index % len(ds)
        row = ds[idx]
        label_idx = int(row.get("label", 0))
        label_text = (
            self._label_names[label_idx]
            if self._label_names and label_idx < len(self._label_names)
            else f"label_{label_idx}"
        )
        return {"index": idx, "split": split, "text": str(row.get("text", "")), "label": label_text}

    @property
    def label_names(self) -> list:
        if self._label_names is None:
            self._load_split("train")
        return self._label_names or []


def format_available_intents(label_names: list) -> str:
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(label_names))


async def classify_banking77_query(
    query: str,
    system_prompt: str,
    available_intents: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    inference_url: str | None = None,
) -> str:
    user_msg = (
        f"Customer Query: {query}\n\n"
        f"Available Intents:\n{available_intents}\n\n"
        f"Classify this query into one of the above banking intents using the tool call."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    if inference_url:
        print(f"\n{'=' * 60}")
        print("[DEBUG] classify_banking77_query using OpenAI SDK with base_url:")
        print(f"  base_url: {inference_url}")
        print(f"  Model: {model}")
        print(f"{'=' * 60}\n")

        # Use OpenAI SDK with custom base_url - SDK will append /chat/completions
        # Pass Synth API key via X-API-Key header (interceptor auth), not Authorization
        # (Authorization: Bearer with Synth key would be rejected by OpenAI passthrough)
        default_headers = {"X-API-Key": api_key} if api_key else {}
        client = AsyncOpenAI(
            base_url=inference_url,
            api_key="synth-interceptor",  # Dummy - interceptor uses its own key
            default_headers=default_headers,
            http_client=get_shared_http_client(),
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args_raw = tool_call.function.arguments
    else:
        print("[DEBUG] Using OpenAI SDK directly (no inference_url)")
        client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args_raw = tool_call.function.arguments

    if not args_raw:
        raise RuntimeError("No tool call arguments returned from model")

    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    return args["intent"]


def create_banking77_local_api(system_prompt: str):
    dataset = Banking77Dataset()
    dataset.ensure_ready(["train", "test"])

    async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        split = request.env.config.get("split", "train")
        seed = request.env.seed

        sample = dataset.sample(split=split, index=seed)

        policy_config = request.policy.config or {}

        # DEBUG: Print the entire policy_config received
        print(f"\n{'#' * 60}")
        print(f"[TASK APP] run_rollout called for seed={seed}")
        print(f"[TASK APP] policy_config keys: {list(policy_config.keys())}")
        print(f"[TASK APP] inference_url: {policy_config.get('inference_url')}")
        print(f"[TASK APP] api_base: {policy_config.get('api_base')}")
        print(f"[TASK APP] base_url: {policy_config.get('base_url')}")
        print(f"[TASK APP] model: {policy_config.get('model')}")
        print(f"{'#' * 60}\n")

        inference_url = policy_config.get("inference_url")
        api_key = policy_config.get("api_key")

        start = time.perf_counter()
        predicted_intent = await classify_banking77_query(
            query=sample["text"],
            system_prompt=system_prompt,
            available_intents=format_available_intents(dataset.label_names),
            model=policy_config.get("model", "gpt-4o-mini"),
            api_key=api_key,
            inference_url=inference_url,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        expected_intent = sample["label"]
        is_correct = (
            predicted_intent.lower().strip() == expected_intent.lower().strip()
            or expected_intent.lower().strip() in predicted_intent.lower().strip()
        )

        # Extract trace_correlation_id from policy_config (required for response)
        trace_correlation_id = (
            policy_config.get("trace_correlation_id") or request.trace_correlation_id
        )

        reward = 1.0 if is_correct else 0.0
        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                outcome_objectives={"reward": reward, "latency_ms": latency_ms},
                instance_objectives=[{"reward": reward, "latency_ms": latency_ms}],
                details={
                    "seed": seed,
                    "split": split,
                    "query": sample["text"],
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "is_correct": is_correct,
                    "latency_ms": latency_ms,
                },
            ),
            trace=None,
            trace_correlation_id=trace_correlation_id,
            inference_url=inference_url or "",
            success_status=SuccessStatus.SUCCESS,
        )

    def provide_taskset_description():
        return {
            "splits": ["train", "test"],
            "sizes": {"train": dataset.size("train"), "test": dataset.size("test")},
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = dataset.sample(split="train", index=seed)
            yield TaskInfo(
                task={"id": "banking77", "name": "Banking77 Intent Classification"},
                dataset={"id": "banking77", "split": sample["split"], "index": sample["index"]},
                inference={"tool": TOOL_NAME},
                limits={"max_turns": 1},
                task_metadata={"query": sample["text"], "expected_intent": sample["label"]},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id="banking77",
            name="Banking77 Intent Classification",
            description="Banking77 intent classification task app",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, headers=headers, timeout=5.0)
            if response.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {health_url}")


async def main():
    print("\n" + "=" * 60)
    print("STARTING EVAL DEBUG")
    print("=" * 60)

    # Start task app
    app = create_banking77_local_api(BASELINE_SYSTEM_PROMPT)
    port = acquire_port(PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != PORT:
        print(f"Port {PORT} in use, using {port} instead")

    run_server_background(app, port)
    wait_for_health_check_sync("localhost", port, ENVIRONMENT_API_KEY, timeout=30.0)
    print(f"Task app ready on port {port}")

    # Provision tunnel if not in local mode
    tunnel = None
    if LOCAL_MODE:
        task_app_url = f"http://{LOCAL_HOST}:{port}"
    else:
        print("\nProvisioning Cloudflare tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            backend=TunnelBackend.CloudflareManagedTunnel,
            backend_url=SYNTH_API_BASE,
            progress=True,
        )
        task_app_url = tunnel.url
    print(f"Task app URL: {task_app_url}")

    # Create eval job
    seeds = list(range(100, 100 + NUM_SEEDS))
    print(f"\nSubmitting eval job with seeds: {seeds}")

    config = EvalJobConfig(
        local_api_url=task_app_url,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
        env_name="banking77",
        seeds=seeds,
        policy_config={
            "model": "gpt-4.1-nano",
            "provider": "openai",
            "inference_mode": "synth_hosted",
            "api_key": API_KEY,
        },
        env_config={"split": "test"},
        concurrency=5,
    )

    job = EvalJob(config)

    try:
        job_id = job.submit()
        print(f"Job submitted: {job_id}")

        result = job.poll_until_complete(timeout=120.0, interval=2.0, progress=True)

        print("\n" + "=" * 60)
        print("EVAL RESULT")
        print("=" * 60)
        print(f"Status: {result.status}")
        print(f"Mean reward: {result.mean_reward}")
        print(f"Error: {result.error}")
        if result.seed_results:
            print(f"Seed results: {len(result.seed_results)}")
            for sr in result.seed_results[:3]:
                print(f"  - {sr}")
    except Exception as e:
        print(f"\nEval job failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup tunnel
        if tunnel:
            print("\nClosing tunnel...")
            tunnel.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
