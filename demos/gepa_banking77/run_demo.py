#!/usr/bin/env python3
"""Run the Banking77 GEPA demo end-to-end.

Usage:
    uv run python demos/gepa_banking77/run_demo.py           # Production mode (Cloudflare tunnels)
    uv run python demos/gepa_banking77/run_demo.py --local   # Local mode (localhost, no tunnels)
"""

import argparse
import asyncio
import json
import os
import time
from copy import deepcopy
from typing import Any

import httpx
from datasets import load_dataset
from fastapi import Request
from openai import AsyncOpenAI
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE as PROD_BASE_URL
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig, EvalResult
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task import normalize_inference_url, run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.sdk.tunnels import (
    PortConflictBehavior,
    TunnelBackend,
    TunneledLocalAPI,
    acquire_port,
    cleanup_all,
)

# Parse args
parser = argparse.ArgumentParser(description="Run Banking77 GEPA demo")
parser.add_argument(
    "--local",
    action="store_true",
    help="Run in local mode: use localhost:8000 backend and skip Cloudflare tunnels",
)
parser.add_argument(
    "--local-host",
    type=str,
    default="localhost",
    help="Hostname for local API URLs (use 'host.docker.internal' if backend runs in Docker)",
)
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for a local API health check using sync httpx (avoids Python 3.14 sniffio issues)."""
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

    raise RuntimeError(
        f"Health check failed: {health_url} not ready after {timeout}s. "
        "Make sure your task app has a /health endpoint."
    )


# Backend configuration
if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    TUNNEL_BACKEND = TunnelBackend.Localhost
    LOCAL_API_PORT = 8013
    OPTIMIZED_LOCAL_API_PORT = 8014
    print("=" * 60)
    print("RUNNING IN LOCAL MODE")
    print("=" * 60)
else:
    # Use dev backend for testing
    SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", "https://api-dev.usesynth.ai")
    TUNNEL_BACKEND = TunnelBackend.CloudflareManagedTunnel
    LOCAL_API_PORT = 8001
    OPTIMIZED_LOCAL_API_PORT = 8002

print(f"Backend: {SYNTH_API_BASE}")
print(f"Tunnel backend: {TUNNEL_BACKEND.value}")
print(f"Local API Ports: {LOCAL_API_PORT}, {OPTIMIZED_LOCAL_API_PORT}")

# Check backend health
r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
if r.status_code == 200:
    print(f"Backend health: {r.json()}")
else:
    print(f"WARNING: Backend returned status {r.status_code}")
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")


# Cell 3: Get API Key
API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("No SYNTH_API_KEY found, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=SYNTH_API_BASE)
    print(f"Demo API Key: {API_KEY[:25]}...")
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")


# Set API key in environment for SDK to use
os.environ["SYNTH_API_KEY"] = API_KEY

# Cell 4: Ensure Environment Key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key ready: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")


# Cell 5: Define Banking77 Local API
APP_ID = "banking77"
APP_NAME = "Banking77 Intent Classification"

BANKING77_LABELS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

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
        # Use OpenAI SDK with custom base_url - SDK will append /chat/completions
        # Pass Synth API key via X-API-Key header (interceptor auth), not Authorization
        # CRITICAL: Override User-Agent to bypass Cloudflare WAF blocking OpenAI SDK requests
        default_headers = {
            "X-API-Key": api_key,
            "User-Agent": "synth-ai/1.0",  # Cloudflare blocks "OpenAI/Python" User-Agent
        } if api_key else {"User-Agent": "synth-ai/1.0"}
        client = AsyncOpenAI(
            base_url=inference_url,
            api_key="synth-interceptor",  # Dummy - interceptor uses its own key
            default_headers=default_headers,
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


def create_banking77_local_api(system_prompt: str):
    dataset = Banking77Dataset()
    dataset.ensure_ready(["train", "test"])

    async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        split = request.env.config.get("split", "train")
        seed = request.env.seed

        sample = dataset.sample(split=split, index=seed)

        policy_config = request.policy.config or {}
        inference_url = policy_config.get("inference_url")
        os.environ["OPENAI_BASE_URL"] = inference_url
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
            predicted_intent.lower().replace("_", " ").strip()
            == expected_intent.lower().replace("_", " ").strip()
        )
        reward = 1.0 if is_correct else 0.0

        policy_cfg_for_trace = {
            key: value
            for key, value in policy_config.items()
            if key not in {"trace_correlation_id", "trace"}
        }
        trace_correlation_id = extract_trace_correlation_id(
            policy_config=policy_cfg_for_trace,
            inference_url=str(inference_url or ""),
        )

        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                outcome_objectives={"reward": reward, "latency_ms": latency_ms},
                instance_objectives=[{"reward": reward, "latency_ms": latency_ms}],
                details={"latency_ms": latency_ms},
            ),
            trace=None,
            trace_correlation_id=trace_correlation_id,
            inference_url=str(inference_url or ""),
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
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": sample["split"], "index": sample["index"]},
                inference={"tool": TOOL_NAME},
                limits={"max_turns": 1},
                task_metadata={"query": sample["text"], "expected_intent": sample["label"]},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description=f"{APP_NAME} local API for classifying customer queries into banking intents.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


print("Banking77 local API defined")


# Main async function
async def main():
    baseline_system_prompt = "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."
    user_prompt = "Customer Query: {query}\n\nAvailable Intents:\n{available_intents}\n\nClassify this query into one of the above banking intents using the tool call."

    # Timing helper
    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"

    timings: dict[str, float] = {}
    total_start = time.time()

    # Cell 7: Start Baseline Local API with Cloudflare Tunnel
    baseline_app = create_banking77_local_api(baseline_system_prompt)

    # Acquire port - find new one if requested port is in use
    baseline_port = acquire_port(LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if baseline_port != LOCAL_API_PORT:
        print(f"Port {LOCAL_API_PORT} in use, using port {baseline_port} instead")

    run_server_background(baseline_app, baseline_port)

    print(f"Waiting for baseline local API on port {baseline_port}...")
    wait_for_health_check_sync("localhost", baseline_port, ENVIRONMENT_API_KEY, timeout=30.0)
    print("Baseline local API ready!")

    if LOCAL_MODE:
        print(f"\nUsing {LOCAL_HOST} (no tunnel)...")
        baseline_local_api_url = f"http://{LOCAL_HOST}:{baseline_port}"
        baseline_tunnel = None
    else:
        print("\nProvisioning Cloudflare tunnel for baseline...")
        tunnel_start = time.time()
        baseline_tunnel = await TunneledLocalAPI.create(
            local_port=baseline_port,
            backend=TUNNEL_BACKEND,
            backend_url=SYNTH_API_BASE,
            progress=True,
        )
        baseline_local_api_url = baseline_tunnel.url
        timings["baseline_tunnel"] = time.time() - tunnel_start
    print(
        f"Baseline local API URL: {baseline_local_api_url}"
        + (
            f" ({format_duration(timings['baseline_tunnel'])})"
            if "baseline_tunnel" in timings
            else ""
        )
    )

    # Cell 8: Run GEPA Optimization
    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": "banking77",
            "task_app_url": baseline_local_api_url,
            "initial_prompt": {
                "id": "banking77_pattern",
                "name": "Banking77 Classification",
                "messages": [
                    {"role": "system", "order": 0, "pattern": baseline_system_prompt},
                    {"role": "user", "order": 1, "pattern": user_prompt},
                ],
                "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
            },
            "policy": {
                "model": "gpt-4.1-nano",
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "env_config": {"split": "train"},
            "gepa": {
                "env_name": "banking77",
                "evaluation": {"seeds": list(range(30)), "validation_seeds": list(range(50, 70))},
                "rollout": {"budget": 200, "max_concurrent": 20, "minibatch_size": 5},
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 3,
                    "num_generations": 4,
                    "children_per_generation": 2,
                },
                "archive": {"pareto_set_size": 20},
                "token": {"counting_model": "gpt-4"},
            },
        },
    }

    # NOTE: PromptLearningJob.submit() validates/builds the payload and may mutate the provided mapping
    # (e.g., nested dicts can be parsed into Pydantic models like GEPAConfig). Anything that expects
    # dict-like `.get(...)` access should be derived BEFORE submission or from a copy.
    total_pareto_seeds = int(
        (config_body.get("prompt_learning", {}) or {})
        .get("gepa", {})
        .get("archive", {})
        .get("pareto_set_size", 0)
        or 0
    )

    print(f"Creating GEPA job (local_api_url={baseline_local_api_url})...")

    pl_job = PromptLearningJob.from_dict(
        config_dict=deepcopy(config_body),
        backend_url=SYNTH_API_BASE,
    )

    job_id = pl_job.submit()
    print(f"Job ID: {job_id}")

    # Track events we've already displayed to avoid duplicates
    last_event_seq = 0

    def on_status_update(status_data: dict[str, Any]) -> None:
        """Callback to fetch and display events during polling."""
        nonlocal last_event_seq
        try:
            # Use sync httpx to fetch events (avoids nested event loop issues)
            response = httpx.get(
                f"{SYNTH_API_BASE}/api/prompt-learning/online/jobs/{job_id}/events",
                params={"since_seq": last_event_seq, "limit": 100},
                headers={"X-API-Key": API_KEY},
                timeout=30.0,
            )
            events = response.json().get("events", []) if response.status_code == 200 else []

            for event in events:
                event_type = event.get("type", "")
                event_seq = event.get("seq", 0)
                if event_seq > last_event_seq:
                    last_event_seq = event_seq

                data = event.get("data", {})
                message = event.get("message", "")

                def format_pareto_growth(growth: Any) -> str:
                    if not isinstance(growth, dict):
                        return ""
                    all_time = growth.get("all_time")
                    last_1 = growth.get("last_1")
                    last_5 = growth.get("last_5")
                    last_20 = growth.get("last_20")
                    parts = []
                    if all_time is not None:
                        parts.append(f"all={all_time:.2f}")
                    if last_1 is not None:
                        parts.append(f"last1={last_1:.2f}")
                    if last_5 is not None:
                        parts.append(f"last5={last_5:.2f}")
                    if last_20 is not None:
                        parts.append(f"last20={last_20:.2f}")
                    return " ".join(parts)

                def format_seeds_outstanding(total_solved: Any) -> str:
                    if total_pareto_seeds <= 0 or total_solved is None:
                        return ""
                    try:
                        outstanding = max(0, total_pareto_seeds - int(total_solved))
                    except (TypeError, ValueError):
                        return ""
                    return f"outstanding={outstanding}/{total_pareto_seeds}"

                # GEPA-specific events (from backend logs)
                if event_type == "prompt.learning.gepa.rollouts_limit_progress":
                    # Extract rollout count from message like "Rollout progress: 20 rollouts executed"
                    if "rollouts executed" in message:
                        print(f"\n  {message}")

                elif event_type == "prompt.learning.gepa.candidate.evaluated":
                    # Message format: "Candidate trans_00001 evaluated (accepted=True) acc=0.500"
                    if "evaluated" in message:
                        # Extract candidate ID and accuracy
                        version_id = data.get("version_id", "")
                        accuracy = data.get("accuracy") or data.get("acc")
                        accepted = data.get("accepted", True)

                        # Parse accuracy from message if not in data (format: "acc=0.500")
                        if accuracy is None and "acc=" in message:
                            try:
                                acc_str = message.split("acc=")[1].split()[0]
                                accuracy = float(acc_str)
                            except (ValueError, IndexError):
                                pass

                        # Parse accepted status from message if not in data
                        if "accepted=True" in message:
                            accepted = True
                        elif "accepted=False" in message:
                            accepted = False

                        if accuracy is not None:
                            status = "✓" if accepted else "✗"
                            print(
                                f"  {status} Candidate {version_id}: mean reward = {accuracy:.2f}"
                            )
                        program_candidate = (
                            data.get("program_candidate", {}) if isinstance(data, dict) else {}
                        )
                        if isinstance(program_candidate, dict):
                            prompt_summary = program_candidate.get("prompt_summary")
                            if isinstance(prompt_summary, str) and prompt_summary.strip():
                                print(f"    prompt_summary: {prompt_summary.strip()[:200]}")
                            objectives = program_candidate.get("objectives")
                            if isinstance(objectives, dict):
                                print(f"    objectives: {objectives}")

                elif event_type == "prompt.learning.gepa.proposal.completed":
                    # Message format: "Proposal generated in 11.23s (evaluation will start next)"
                    if "Proposal generated" in message:
                        print(f"  {message}")

                elif event_type == "prompt.learning.gepa.generation.start":
                    # Message format: "Generation 1/2 starting"
                    if "Generation" in message:
                        print(f"\n  {message}")

                elif event_type == "prompt.learning.gepa.progress":
                    frontier_density = data.get("frontier_density")
                    frontier_size = data.get("frontier_size") or data.get("archive_size")
                    total_seeds_solved = data.get("total_seeds_solved")
                    pareto_growth = format_pareto_growth(data.get("pareto_growth"))
                    seeds_outstanding = format_seeds_outstanding(total_seeds_solved)
                    best_reward = data.get("best_reward")
                    details = []
                    if frontier_density is not None:
                        details.append(f"density={frontier_density:.3f}")
                    if frontier_size is not None:
                        details.append(f"frontier={frontier_size}")
                    if total_seeds_solved is not None:
                        details.append(f"total_seeds={total_seeds_solved}")
                    if seeds_outstanding:
                        details.append(seeds_outstanding)
                    if pareto_growth:
                        details.append(f"growth[{pareto_growth}]")
                    if best_reward is not None:
                        details.append(f"best={best_reward:.3f}")
                    if details:
                        print(f"\n  GEPA progress: {' | '.join(details)}")
                    else:
                        print(f"\n  GEPA progress (raw): {data}")

                elif event_type == "prompt.learning.gepa.archive.frontier_improved":
                    frontier_density = data.get("frontier_density")
                    frontier_size = data.get("archive_size")
                    total_seeds_solved = data.get("total_seeds_solved")
                    pareto_growth = format_pareto_growth(data.get("pareto_growth"))
                    seeds_outstanding = format_seeds_outstanding(total_seeds_solved)
                    best_reward = data.get("best_reward")
                    details = []
                    if best_reward is not None:
                        details.append(f"best={best_reward:.3f}")
                    if frontier_density is not None:
                        details.append(f"density={frontier_density:.3f}")
                    if frontier_size is not None:
                        details.append(f"frontier={frontier_size}")
                    if total_seeds_solved is not None:
                        details.append(f"total_seeds={total_seeds_solved}")
                    if seeds_outstanding:
                        details.append(seeds_outstanding)
                    if pareto_growth:
                        details.append(f"growth[{pareto_growth}]")
                    if details:
                        print(f"\n  Frontier improved: {' | '.join(details)}")
                    else:
                        print(f"\n  Frontier improved (raw): {data}")

                elif event_type == "prompt.learning.gepa.generation.complete":
                    frontier_density = data.get("frontier_density")
                    frontier_size = data.get("archive_size")
                    total_seeds_solved = data.get("total_seeds_solved")
                    pareto_growth = format_pareto_growth(data.get("pareto_growth"))
                    seeds_outstanding = format_seeds_outstanding(total_seeds_solved)
                    best_reward = data.get("best_reward")
                    details = []
                    if best_reward is not None:
                        details.append(f"best={best_reward:.3f}")
                    if frontier_density is not None:
                        details.append(f"density={frontier_density:.3f}")
                    if frontier_size is not None:
                        details.append(f"frontier={frontier_size}")
                    if total_seeds_solved is not None:
                        details.append(f"total_seeds={total_seeds_solved}")
                    if seeds_outstanding:
                        details.append(seeds_outstanding)
                    if pareto_growth:
                        details.append(f"growth[{pareto_growth}]")
                    if details:
                        print(f"\n  Generation complete metrics: {' | '.join(details)}")
                    else:
                        print(f"\n  Generation complete metrics (raw): {data}")

                elif event_type == "prompt.learning.candidate.evaluation.started":
                    # Message format: "Evaluating candidate trans_00004... (10 seeds)"
                    if "Evaluating candidate" in message:
                        print(f"  {message}")

                # Legacy/fallback event types
                elif event_type == "prompt.learning.progress":
                    rollouts = data.get("rollouts_completed", 0)
                    total = data.get("rollouts_total", 0)
                    if rollouts > 0:
                        print(f"\n  Progress: {rollouts}/{total} rollouts completed")

        except Exception:
            # Silently ignore event fetching errors to avoid polluting output
            pass

    optimization_start = time.time()
    gepa_result = pl_job.poll_until_complete(
        timeout=3600.0,
        interval=3.0,
        progress=False,  # Disable basic progress output since we're showing detailed events
        on_status=on_status_update,
    )
    timings["optimization"] = time.time() - optimization_start

    print(f"\nFINAL: {gepa_result.status.value} ({format_duration(timings['optimization'])})")

    if gepa_result.succeeded:
        best_reward = None
        if isinstance(gepa_result.raw, dict):
            best_reward = (
                gepa_result.raw.get("best_reward")
                or gepa_result.raw.get("best_avg_reward")
                or gepa_result.raw.get("best_train_reward")
            )
        if isinstance(best_reward, (int, float)):
            print(f"BEST REWARD: {float(best_reward):.1%}")
        else:
            print("BEST REWARD: N/A")
    elif gepa_result.failed:
        print(f"ERROR: {gepa_result.error}")
        # Print full raw response for debugging
        if gepa_result.raw:
            print("\n--- Full error details from status ---")
            for key in [
                "error",
                "error_message",
                "error_details",
                "traceback",
                "failure_reason",
                "message",
            ]:
                if key in gepa_result.raw and gepa_result.raw[key]:
                    print(f"{key}: {gepa_result.raw[key]}")

        # Fetch events for more detailed error info
        try:
            print("\n--- Fetching job events for error details ---")
            pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)
            events = await pl_client.get_events(gepa_result.job_id, limit=100)
            error_events = [
                e
                for e in events
                if "error" in e.get("type", "").lower()
                or "fail" in e.get("type", "").lower()
                or e.get("data", {}).get("error")
            ]
            if error_events:
                for event in error_events[-3:]:  # Last 3 error events
                    print(f"\n[{event.get('type')}] {event.get('message', '')}")
                    data = event.get("data", {})
                    if data.get("error"):
                        print(f"  error: {data['error']}")
                    if data.get("traceback"):
                        print(f"  traceback: {data['traceback'][:500]}...")
            else:
                # Print last few events regardless
                print("No error events found. Last events:")
                for event in events[-5:]:
                    print(f"  [{event.get('type')}] {event.get('message', '')[:100]}")
        except Exception as e:
            print(f"Could not fetch events: {e}")

    # Cell 9: Evaluation
    eval_seeds = list(range(100, 120))

    def run_eval_job(local_api_url: str, seeds: list[int], mode: str) -> EvalResult:
        config = EvalJobConfig(
            local_api_url=local_api_url,
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
            concurrency=10,
        )
        job = EvalJob(config)
        job_id = job.submit()
        print(f"  {mode} eval job: {job_id}")
        return job.poll_until_complete(timeout=600.0, interval=2.0, progress=True)

    def extract_system_prompt(prompt_results) -> str:
        """Extract system prompt from prompt results, handling multiple formats."""
        # First try to get from top_prompts
        if prompt_results.top_prompts:
            top = prompt_results.top_prompts[0]

            # Check for full_text first (most common format)
            if "full_text" in top and top["full_text"]:
                return top["full_text"]

            # Check for template with sections
            if "template" in top and top["template"]:
                template = top["template"]
                if "sections" in template:
                    for section in template["sections"]:
                        if section.get("role") == "system":
                            return section.get("content", "")
                # Template might have full_text directly
                if "full_text" in template:
                    return template["full_text"]

            # Other possible formats
            if "system_prompt" in top:
                return top["system_prompt"]
            if "prompt" in top:
                return top["prompt"]

        # Try best_prompt from results
        if prompt_results.best_prompt:
            if isinstance(prompt_results.best_prompt, str):
                return prompt_results.best_prompt
            elif isinstance(prompt_results.best_prompt, dict):
                # Could be a dict with 'full_text' or 'content'
                if "full_text" in prompt_results.best_prompt:
                    return prompt_results.best_prompt["full_text"]
                if "content" in prompt_results.best_prompt:
                    return prompt_results.best_prompt["content"]
                # Extract from messages array (OpenAI format)
                if "messages" in prompt_results.best_prompt:
                    messages = prompt_results.best_prompt["messages"]
                    if messages and isinstance(messages, list):
                        # Find system message
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "system":
                                return msg.get("content") or msg.get("pattern", "")
                        # Fallback to first message
                        if messages[0]:
                            return messages[0].get("content") or messages[0].get("pattern", "")

        # Last resort: return debug info
        if prompt_results.top_prompts:
            return f"[Could not extract prompt. Keys available: {list(prompt_results.top_prompts[0].keys())}]"
        return "[No prompts found in results]"

    if gepa_result.succeeded:
        print("GEPA Job Succeeded!\n")

        try:
            pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)
            prompt_results = await pl_client.get_prompts(gepa_result.job_id)

            # Try to get the optimized prompt
            optimized_system = extract_system_prompt(prompt_results)

            # If extraction failed, show what's available and try alternatives
            if optimized_system.startswith("[Could not extract") or optimized_system.startswith(
                "[No prompts"
            ):
                print(
                    f"Debug: top_prompts[0] = {prompt_results.top_prompts[0] if prompt_results.top_prompts else 'empty'}"
                )
                print(f"Debug: best_prompt type = {type(prompt_results.best_prompt)}", flush=True)
                print(
                    f"Debug: best_prompt = {str(prompt_results.best_prompt)[:200] if prompt_results.best_prompt else 'None'}",
                    flush=True,
                )
                print(
                    f"Debug: optimized_candidates count = {len(prompt_results.optimized_candidates)}",
                    flush=True,
                )

                # Try to extract from optimized_candidates
                if prompt_results.optimized_candidates:
                    cand = prompt_results.optimized_candidates[0]
                    if isinstance(cand, dict):
                        print(
                            f"Debug: optimized_candidates[0] keys = {list(cand.keys())}", flush=True
                        )
                        # Try common keys
                        for key in ["full_text", "prompt", "template", "content", "system_prompt"]:
                            if key in cand and cand[key]:
                                val = cand[key]
                                if isinstance(val, str) and len(val) > 20:
                                    optimized_system = val
                                    print(
                                        f"Extracted prompt from optimized_candidates[0]['{key}']",
                                        flush=True,
                                    )
                                    break
                                elif isinstance(val, dict):
                                    if "full_text" in val:
                                        optimized_system = val["full_text"]
                                        print(
                                            f"Extracted from optimized_candidates[0]['{key}']['full_text']",
                                            flush=True,
                                        )
                                        break
                                    elif "sections" in val:
                                        for sec in val["sections"]:
                                            if sec.get("role") == "system":
                                                optimized_system = sec.get("content", "")
                                                print(
                                                    "Extracted from template sections", flush=True
                                                )
                                                break

                # If still failed, fall back to baseline
                if optimized_system.startswith("["):
                    print(
                        "\nWARNING: Could not extract optimized prompt. Using baseline for comparison.",
                        flush=True,
                    )
                    optimized_system = baseline_system_prompt

            best_train_reward = 0.0
            if isinstance(gepa_result.raw, dict):
                raw_best = gepa_result.raw.get("best_reward") or gepa_result.raw.get(
                    "best_avg_reward"
                )
                if isinstance(raw_best, (int, float)):
                    best_train_reward = float(raw_best)

        except Exception as e:
            print(f"\nERROR extracting prompts: {e}", flush=True)
            import traceback

            traceback.print_exc()
            optimized_system = baseline_system_prompt
            best_train_reward = 0.0
            if isinstance(gepa_result.raw, dict):
                raw_best = gepa_result.raw.get("best_reward") or gepa_result.raw.get(
                    "best_avg_reward"
                )
                if isinstance(raw_best, (int, float)):
                    best_train_reward = float(raw_best)

        print("=" * 60)
        print("BASELINE SYSTEM PROMPT")
        print("=" * 60)
        print(baseline_system_prompt)

        print("\n" + "=" * 60)
        print("OPTIMIZED SYSTEM PROMPT (from GEPA)")
        print("=" * 60)
        print(optimized_system[:800] + "..." if len(optimized_system) > 800 else optimized_system)

        print("\n" + "=" * 60)
        print("GEPA TRAINING RESULTS")
        print("=" * 60)
        print(
            f"Best Train Reward: {best_train_reward:.1%}"
            if best_train_reward
            else "Best Train Reward: N/A"
        )

        print("\n" + "=" * 60)
        print(f"FORMAL EVAL JOBS (test split, seeds {eval_seeds[0]}-{eval_seeds[-1]})")
        print("=" * 60)

        print(f"\nStarting optimized local API on port {OPTIMIZED_LOCAL_API_PORT}...")
        optimized_app = create_banking77_local_api(optimized_system)

        # Acquire port - find new one if requested port is in use
        optimized_port = acquire_port(
            OPTIMIZED_LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW
        )
        if optimized_port != OPTIMIZED_LOCAL_API_PORT:
            print(f"Port {OPTIMIZED_LOCAL_API_PORT} in use, using port {optimized_port} instead")

        run_server_background(optimized_app, optimized_port)
        wait_for_health_check_sync("localhost", optimized_port, ENVIRONMENT_API_KEY, timeout=30.0)
        print("Optimized local API ready!")

        if LOCAL_MODE:
            print(f"\nUsing {LOCAL_HOST} for optimized (no tunnel)...")
            optimized_local_api_url = f"http://{LOCAL_HOST}:{optimized_port}"
            optimized_tunnel = None
        else:
            print("\nProvisioning Cloudflare tunnel for optimized...")
            tunnel_start = time.time()
            optimized_tunnel = await TunneledLocalAPI.create(
                local_port=optimized_port,
                backend=TUNNEL_BACKEND,
                backend_url=SYNTH_API_BASE,
                progress=True,
            )
            optimized_local_api_url = optimized_tunnel.url
            timings["optimized_tunnel"] = time.time() - tunnel_start
            print(f"Optimized tunnel ready ({format_duration(timings['optimized_tunnel'])})")

        print("\nRunning BASELINE eval job...")
        eval_start = time.time()
        baseline_result = run_eval_job(
            local_api_url=baseline_local_api_url, seeds=eval_seeds, mode="baseline"
        )
        timings["baseline_eval"] = time.time() - eval_start

        if baseline_result.succeeded:
            # Prefer typed mean_reward; otherwise fall back to summary/seed results.
            mean_reward = getattr(baseline_result, "mean_reward", None)
            if mean_reward is None:
                mean_reward = getattr(baseline_result, "mean_score", None)
            if mean_reward is None:
                summary = baseline_result.raw.get("summary", {})
                mean_reward = summary.get("mean_reward")
            if mean_reward is None and baseline_result.seed_results:
                rewards = [
                    r.get("outcome_reward") or r.get("reward_mean") or r.get("reward")
                    for r in baseline_result.seed_results
                    if isinstance(r, dict)
                    and (r.get("outcome_reward") or r.get("reward_mean") or r.get("reward"))
                    is not None
                ]
                if rewards:
                    mean_reward = sum(rewards) / len(rewards)

            if mean_reward is not None:
                print(
                    f"  Baseline eval reward: {mean_reward:.1%} ({format_duration(timings['baseline_eval'])})"
                )
            else:
                print(
                    f"  Baseline eval completed but no reward available ({format_duration(timings['baseline_eval'])})"
                )
        else:
            print(f"  Baseline eval failed: {baseline_result.error}")

        print("\nRunning OPTIMIZED eval job...")
        eval_start = time.time()
        optimized_result = run_eval_job(
            local_api_url=optimized_local_api_url, seeds=eval_seeds, mode="optimized"
        )
        timings["optimized_eval"] = time.time() - eval_start

        if optimized_result.succeeded:
            # Prefer typed mean_reward; otherwise fall back to summary/seed results.
            mean_reward = getattr(optimized_result, "mean_reward", None)
            if mean_reward is None:
                mean_reward = getattr(optimized_result, "mean_score", None)
            if mean_reward is None:
                summary = optimized_result.raw.get("summary", {})
                mean_reward = summary.get("mean_reward")
            if mean_reward is None and optimized_result.seed_results:
                rewards = [
                    r.get("outcome_reward") or r.get("reward_mean") or r.get("reward")
                    for r in optimized_result.seed_results
                    if isinstance(r, dict)
                    and (r.get("outcome_reward") or r.get("reward_mean") or r.get("reward"))
                    is not None
                ]
                if rewards:
                    mean_reward = sum(rewards) / len(rewards)

            if mean_reward is not None:
                print(
                    f"  Optimized eval reward: {mean_reward:.1%} ({format_duration(timings['optimized_eval'])})"
                )
            else:
                print(
                    f"  Optimized eval completed but no reward available ({format_duration(timings['optimized_eval'])})"
                )
        else:
            print(f"  Optimized eval failed: {optimized_result.error}")

        if baseline_result.succeeded and optimized_result.succeeded:

            def extract_mean_reward(result: EvalResult) -> float | None:
                mean_reward = getattr(result, "mean_reward", None)
                if mean_reward is None:
                    mean_reward = getattr(result, "mean_score", None)
                if mean_reward is None:
                    summary = result.raw.get("summary", {})
                    mean_reward = summary.get("mean_reward")
                if mean_reward is None and result.seed_results:
                    rewards = [
                        r.get("outcome_reward") or r.get("reward_mean") or r.get("reward")
                        for r in result.seed_results
                        if isinstance(r, dict)
                        and (r.get("outcome_reward") or r.get("reward_mean") or r.get("reward"))
                        is not None
                    ]
                    if rewards:
                        mean_reward = sum(rewards) / len(rewards)
                return mean_reward

            baseline_reward = extract_mean_reward(baseline_result)
            optimized_reward = extract_mean_reward(optimized_result)

            if baseline_reward is not None and optimized_reward is not None:
                print("\n" + "=" * 60)
                print("FINAL COMPARISON")
                print("=" * 60)
                print("Training:")
                print(f"  Best Train Reward: {best_train_reward:.1%}")

                print(f"\nEval (seeds {eval_seeds[0]}-{eval_seeds[-1]}, held-out):")
                print(f"  Baseline Reward:  {baseline_reward:.1%}")
                print(f"  Optimized Reward: {optimized_reward:.1%}")

                eval_lift = optimized_reward - baseline_reward
                print(f"  Lift:             {eval_lift:+.1%}")

                if eval_lift > 0:
                    print("\n>>> OPTIMIZATION GENERALIZES TO HELD-OUT DATA!")
                elif eval_lift == 0:
                    print("\n=== Same performance on held-out data")
                else:
                    print("\n<<< Baseline better on held-out (possible overfitting)")
            else:
                print("\n" + "=" * 60)
                print("FINAL COMPARISON")
                print("=" * 60)
                print("Eval jobs completed but rewards not available for comparison")
                if baseline_reward is None:
                    print("  Baseline: reward not available")
                if optimized_reward is None:
                    print("  Optimized: reward not available")
    else:
        print(f"Job did not succeed: {gepa_result.status.value}")
        # Error details already printed above for failed jobs

    # Cell 10: Cleanup and Timing Summary
    if not LOCAL_MODE:
        print("\nCleaning up cloudflared processes...")
        cleanup_all()

    # Print timing summary
    timings["total"] = time.time() - total_start
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    if "baseline_tunnel" in timings:
        print(f"  Baseline tunnel:    {format_duration(timings['baseline_tunnel'])}")
    if "optimization" in timings:
        print(f"  GEPA optimization:  {format_duration(timings['optimization'])}")
    if "optimized_tunnel" in timings:
        print(f"  Optimized tunnel:   {format_duration(timings['optimized_tunnel'])}")
    if "baseline_eval" in timings:
        print(f"  Baseline eval:      {format_duration(timings['baseline_eval'])}")
    if "optimized_eval" in timings:
        print(f"  Optimized eval:     {format_duration(timings['optimized_eval'])}")
    print("  ─────────────────────────")
    print(f"  Total:              {format_duration(timings['total'])}")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
