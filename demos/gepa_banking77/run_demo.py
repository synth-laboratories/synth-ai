#!/usr/bin/env python3
"""Run the Banking77 GEPA demo end-to-end."""

import os
import sys
import asyncio
from pathlib import Path

# Add synth-ai to path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

# Load .env file from repo root
from dotenv import load_dotenv
env_path = Path(repo_root) / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)  # override=True ensures .env values take precedence
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"No .env file found at {env_path}, using system environment variables")

import nest_asyncio
nest_asyncio.apply()

# Cell 1: Imports, Config, and Backend Health Check
import json
import threading
import time
from typing import Any, Optional

import httpx
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

# Synth SDK imports
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig, EvalResult
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task import normalize_inference_url, run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.sdk.tunnels import TunnelBackend, TunneledLocalAPI, cleanup_all, wait_for_health_check, acquire_port, PortConflictBehavior
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key

# Production backend
SYNTH_API_BASE = PROD_BASE_URL
LOCAL_API_PORT = 8001
OPTIMIZED_LOCAL_API_PORT = 8002

print(f'Backend: {SYNTH_API_BASE}')
print(f'Local API Ports: {LOCAL_API_PORT}, {OPTIMIZED_LOCAL_API_PORT}')

# Check backend health
r = httpx.get(f'{SYNTH_API_BASE}/health', timeout=30)
if r.status_code == 200:
    print(f'Backend health: {r.json()}')
else:
    print(f'WARNING: Backend returned status {r.status_code}')
    raise RuntimeError(f'Backend not healthy: status {r.status_code}')


# Cell 3: Get API Key
API_KEY = os.environ.get('SYNTH_API_KEY', '')
if not API_KEY:
    print('No SYNTH_API_KEY found, minting demo key...')
    API_KEY = mint_demo_api_key()
    print(f'Demo API Key: {API_KEY[:25]}...')
else:
    print(f'Using SYNTH_API_KEY: {API_KEY[:20]}...')


# Set API key in environment for SDK to use
os.environ['SYNTH_API_KEY'] = API_KEY

# Cell 4: Ensure Environment Key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f'Env key ready: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}')


# Cell 5: Define Banking77 Local API
APP_ID = "banking77"
APP_NAME = "Banking77 Intent Classification"

BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
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
    return "\n".join(f"{i+1}. {l}" for i, l in enumerate(label_names))


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
        url = normalize_inference_url(inference_url)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        payload = {
            "model": model,
            "messages": messages,
            "tools": [TOOL_SCHEMA],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = str(error_json.get("error", {}).get("message", error_json))
            except Exception:
                error_msg = response.text[:500]
            raise RuntimeError(f"Proxy error ({response.status_code}): {error_msg}")

        data = response.json()
        tool_call = (data.get("choices") or [])[0].get("message", {}).get("tool_calls", [])[0]
        args_raw = tool_call.get("function", {}).get("arguments")
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
        label_text = self._label_names[label_idx] if self._label_names and label_idx < len(self._label_names) else f"label_{label_idx}"
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

        predicted_intent = await classify_banking77_query(
            query=sample["text"],
            system_prompt=system_prompt,
            available_intents=format_available_intents(dataset.label_names),
            model=policy_config.get("model", "gpt-4o-mini"),
            api_key=api_key,
            inference_url=inference_url,
        )

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
            mode=request.mode,
        )

        return RolloutResponse(
            run_id=request.run_id,
            metrics=RolloutMetrics(outcome_reward=reward),
            trace=None,
            trace_correlation_id=trace_correlation_id,
            inference_url=str(inference_url or ""),
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

    return create_local_api(LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description=f"{APP_NAME} local API for classifying customer queries into banking intents.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    ))


print('Banking77 local API defined')


# Main async function
async def main():
    BASELINE_SYSTEM_PROMPT = "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."
    USER_PROMPT = "Customer Query: {query}\n\nAvailable Intents:\n{available_intents}\n\nClassify this query into one of the above banking intents using the tool call."

    # Timing helper
    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"

    timings: dict[str, float] = {}
    total_start = time.time()

    # Cell 7: Start Baseline Local API with Cloudflare Tunnel
    baseline_app = create_banking77_local_api(BASELINE_SYSTEM_PROMPT)

    # Acquire port - find new one if requested port is in use
    baseline_port = acquire_port(LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if baseline_port != LOCAL_API_PORT:
        print(f'Port {LOCAL_API_PORT} in use, using port {baseline_port} instead')

    run_server_background(baseline_app, baseline_port)

    print(f'Waiting for baseline local API on port {baseline_port}...')
    await wait_for_health_check("localhost", baseline_port, ENVIRONMENT_API_KEY, timeout=30.0)
    print('Baseline local API ready!')

    print('\nProvisioning Cloudflare tunnel for baseline...')
    tunnel_start = time.time()
    baseline_tunnel = await TunneledLocalAPI.create(
        local_port=LOCAL_API_PORT,
        backend=TunnelBackend.CloudflareQuickTunnel,
        progress=True,
    )
    BASELINE_LOCAL_API_URL = baseline_tunnel.url
    timings['baseline_tunnel'] = time.time() - tunnel_start
    print(f'\nBaseline local API URL: {BASELINE_LOCAL_API_URL} ({format_duration(timings["baseline_tunnel"])})')

    # Cell 8: Run GEPA Optimization
    config_body = {
        'prompt_learning': {
            'algorithm': 'gepa',
            'task_app_id': 'banking77',
            'task_app_url': BASELINE_LOCAL_API_URL,
            'initial_prompt': {
                'id': 'banking77_pattern',
                'name': 'Banking77 Classification',
                'messages': [
                    {'role': 'system', 'order': 0, 'pattern': BASELINE_SYSTEM_PROMPT},
                    {'role': 'user', 'order': 1, 'pattern': USER_PROMPT},
                ],
                'wildcards': {'query': 'REQUIRED', 'available_intents': 'OPTIONAL'},
            },
            'policy': {
                'model': 'gpt-4.1-nano',
                'provider': 'openai',
                'inference_mode': 'synth_hosted',
                'temperature': 0.0,
                'max_completion_tokens': 256,
            },
            'env_config': {'split': 'train'},
            'gepa': {
                'env_name': 'banking77',
                'evaluation': {'seeds': list(range(30)), 'validation_seeds': list(range(50, 70))},
                'rollout': {'budget': 50, 'max_concurrent': 5, 'minibatch_size': 5},
                'mutation': {'rate': 0.3},
                'population': {'initial_size': 3, 'num_generations': 2, 'children_per_generation': 2},
                'archive': {'pareto_set_size': 10},
            },
        },
    }

    print(f'Creating GEPA job (local_api_url={BASELINE_LOCAL_API_URL})...')

    pl_job = PromptLearningJob.from_dict(
        config_dict=config_body,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
    )

    job_id = pl_job.submit()
    print(f'Job ID: {job_id}')

    optimization_start = time.time()
    gepa_result = pl_job.poll_until_complete(timeout=3600.0, interval=3.0, progress=True)
    timings['optimization'] = time.time() - optimization_start

    print(f'\nFINAL: {gepa_result.status.value} ({format_duration(timings["optimization"])})')

    if gepa_result.succeeded:
        print(f'BEST SCORE: {gepa_result.best_score}')
    elif gepa_result.failed:
        print(f'ERROR: {gepa_result.error}')

    # Cell 9: Evaluation
    EVAL_SEEDS = list(range(100, 120))

    def run_eval_job(local_api_url: str, seeds: list[int], mode: str) -> EvalResult:
        config = EvalJobConfig(
            local_api_url=local_api_url,
            backend_url=SYNTH_API_BASE,
            api_key=API_KEY,
            env_name='banking77',
            seeds=seeds,
            policy_config={
                'model': 'gpt-4.1-nano',
                'provider': 'openai',
                'inference_mode': 'synth_hosted',
                'api_key': API_KEY,
            },
            env_config={'split': 'test'},
            concurrency=10,
        )
        job = EvalJob(config)
        job_id = job.submit()
        print(f'  {mode} eval job: {job_id}')
        return job.poll_until_complete(timeout=600.0, interval=2.0, progress=True)

    def extract_system_prompt(prompt_results) -> str:
        """Extract system prompt from prompt results, handling multiple formats."""
        # First try to get from top_prompts
        if prompt_results.top_prompts:
            top = prompt_results.top_prompts[0]

            # Check for full_text first (most common format)
            if 'full_text' in top and top['full_text']:
                return top['full_text']

            # Check for template with sections
            if 'template' in top and top['template']:
                template = top['template']
                if 'sections' in template:
                    for section in template['sections']:
                        if section.get('role') == 'system':
                            return section.get('content', '')
                # Template might have full_text directly
                if 'full_text' in template:
                    return template['full_text']

            # Other possible formats
            if 'system_prompt' in top:
                return top['system_prompt']
            if 'prompt' in top:
                return top['prompt']

        # Try best_prompt from results
        if prompt_results.best_prompt:
            if isinstance(prompt_results.best_prompt, str):
                return prompt_results.best_prompt
            elif isinstance(prompt_results.best_prompt, dict):
                # Could be a dict with 'full_text' or 'content'
                if 'full_text' in prompt_results.best_prompt:
                    return prompt_results.best_prompt['full_text']
                if 'content' in prompt_results.best_prompt:
                    return prompt_results.best_prompt['content']

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
            if optimized_system.startswith("[Could not extract") or optimized_system.startswith("[No prompts"):
                print(f"Debug: top_prompts[0] = {prompt_results.top_prompts[0] if prompt_results.top_prompts else 'empty'}")
                print(f"Debug: best_prompt type = {type(prompt_results.best_prompt)}", flush=True)
                print(f"Debug: best_prompt = {str(prompt_results.best_prompt)[:200] if prompt_results.best_prompt else 'None'}", flush=True)
                print(f"Debug: optimized_candidates count = {len(prompt_results.optimized_candidates)}", flush=True)

                # Try to extract from optimized_candidates
                if prompt_results.optimized_candidates:
                    cand = prompt_results.optimized_candidates[0]
                    if isinstance(cand, dict):
                        print(f"Debug: optimized_candidates[0] keys = {list(cand.keys())}", flush=True)
                        # Try common keys
                        for key in ['full_text', 'prompt', 'template', 'content', 'system_prompt']:
                            if key in cand and cand[key]:
                                val = cand[key]
                                if isinstance(val, str) and len(val) > 20:
                                    optimized_system = val
                                    print(f"Extracted prompt from optimized_candidates[0]['{key}']", flush=True)
                                    break
                                elif isinstance(val, dict):
                                    if 'full_text' in val:
                                        optimized_system = val['full_text']
                                        print(f"Extracted from optimized_candidates[0]['{key}']['full_text']", flush=True)
                                        break
                                    elif 'sections' in val:
                                        for sec in val['sections']:
                                            if sec.get('role') == 'system':
                                                optimized_system = sec.get('content', '')
                                                print(f"Extracted from template sections", flush=True)
                                                break

                # If still failed, fall back to baseline
                if optimized_system.startswith("["):
                    print("\nWARNING: Could not extract optimized prompt. Using baseline for comparison.", flush=True)
                    optimized_system = BASELINE_SYSTEM_PROMPT

            best_train_reward = prompt_results.best_score or gepa_result.best_score or 0.0

        except Exception as e:
            print(f"\nERROR extracting prompts: {e}", flush=True)
            import traceback
            traceback.print_exc()
            optimized_system = BASELINE_SYSTEM_PROMPT
            best_train_reward = gepa_result.best_score or 0.0

        print('=' * 60)
        print('BASELINE SYSTEM PROMPT')
        print('=' * 60)
        print(BASELINE_SYSTEM_PROMPT)

        print('\n' + '=' * 60)
        print('OPTIMIZED SYSTEM PROMPT (from GEPA)')
        print('=' * 60)
        print(optimized_system[:800] + "..." if len(optimized_system) > 800 else optimized_system)

        print('\n' + '=' * 60)
        print('GEPA TRAINING RESULTS')
        print('=' * 60)
        print(f"Best Train Reward: {best_train_reward:.1%}" if best_train_reward else "Best Train Reward: N/A")

        print('\n' + '=' * 60)
        print(f'FORMAL EVAL JOBS (test split, seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]})')
        print('=' * 60)

        print(f'\nStarting optimized local API on port {OPTIMIZED_LOCAL_API_PORT}...')
        optimized_app = create_banking77_local_api(optimized_system)

        # Acquire port - find new one if requested port is in use
        optimized_port = acquire_port(OPTIMIZED_LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)
        if optimized_port != OPTIMIZED_LOCAL_API_PORT:
            print(f'Port {OPTIMIZED_LOCAL_API_PORT} in use, using port {optimized_port} instead')

        run_server_background(optimized_app, optimized_port)
        await wait_for_health_check("localhost", optimized_port, ENVIRONMENT_API_KEY, timeout=30.0)
        print('Optimized local API ready!')

        print('\nProvisioning Cloudflare tunnel for optimized...')
        tunnel_start = time.time()
        optimized_tunnel = await TunneledLocalAPI.create(
            local_port=OPTIMIZED_LOCAL_API_PORT,
            backend=TunnelBackend.CloudflareQuickTunnel,
            progress=True,
        )
        OPTIMIZED_LOCAL_API_URL = optimized_tunnel.url
        timings['optimized_tunnel'] = time.time() - tunnel_start
        print(f'Optimized tunnel ready ({format_duration(timings["optimized_tunnel"])})')

        print('\nRunning BASELINE eval job...')
        eval_start = time.time()
        baseline_result = run_eval_job(
            local_api_url=BASELINE_LOCAL_API_URL,
            seeds=EVAL_SEEDS,
            mode='baseline'
        )
        timings['baseline_eval'] = time.time() - eval_start

        if baseline_result.succeeded:
            print(f'  Baseline eval reward: {baseline_result.mean_score:.1%} ({format_duration(timings["baseline_eval"])})')
        else:
            print(f'  Baseline eval failed: {baseline_result.error}')

        print('\nRunning OPTIMIZED eval job...')
        eval_start = time.time()
        optimized_result = run_eval_job(
            local_api_url=OPTIMIZED_LOCAL_API_URL,
            seeds=EVAL_SEEDS,
            mode='optimized'
        )
        timings['optimized_eval'] = time.time() - eval_start

        if optimized_result.succeeded:
            print(f'  Optimized eval reward: {optimized_result.mean_score:.1%} ({format_duration(timings["optimized_eval"])})')
        else:
            print(f'  Optimized eval failed: {optimized_result.error}')

        if baseline_result.succeeded and optimized_result.succeeded:
            print('\n' + '=' * 60)
            print('FINAL COMPARISON')
            print('=' * 60)
            print(f"Training:")
            print(f"  Best Train Reward: {best_train_reward:.1%}")

            print(f"\nEval (seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}, held-out):")
            print(f"  Baseline Reward:  {baseline_result.mean_score:.1%}")
            print(f"  Optimized Reward: {optimized_result.mean_score:.1%}")

            eval_lift = optimized_result.mean_score - baseline_result.mean_score
            print(f"  Lift:             {eval_lift:+.1%}")

            if eval_lift > 0:
                print("\n>>> OPTIMIZATION GENERALIZES TO HELD-OUT DATA!")
            elif eval_lift == 0:
                print("\n=== Same performance on held-out data")
            else:
                print("\n<<< Baseline better on held-out (possible overfitting)")
    else:
        print(f"Job failed: {gepa_result.status.value}")
        if gepa_result.error:
            print(f"Error: {gepa_result.error}")

    # Cell 10: Cleanup and Timing Summary
    print('\nCleaning up cloudflared processes...')
    cleanup_all()

    # Print timing summary
    timings['total'] = time.time() - total_start
    print('\n' + '=' * 60)
    print('TIMING SUMMARY')
    print('=' * 60)
    if 'baseline_tunnel' in timings:
        print(f"  Baseline tunnel:    {format_duration(timings['baseline_tunnel'])}")
    if 'optimization' in timings:
        print(f"  GEPA optimization:  {format_duration(timings['optimization'])}")
    if 'optimized_tunnel' in timings:
        print(f"  Optimized tunnel:   {format_duration(timings['optimized_tunnel'])}")
    if 'baseline_eval' in timings:
        print(f"  Baseline eval:      {format_duration(timings['baseline_eval'])}")
    if 'optimized_eval' in timings:
        print(f"  Optimized eval:     {format_duration(timings['optimized_eval'])}")
    print(f"  ─────────────────────────")
    print(f"  Total:              {format_duration(timings['total'])}")

    print('\nDemo complete!')


if __name__ == "__main__":
    asyncio.run(main())
