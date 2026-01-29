#!/usr/bin/env python3
"""
Online GEPA Demo - Banking77 Classification

This script demonstrates GEPA in ONLINE mode - learning in real-time as requests come in.
Unlike offline GEPA which optimizes in batches, online GEPA:

1. Sets up a proxy URL for your LLM requests
2. Routes requests through the proxy (which selects candidates)
3. Learns from rewards submitted after each response
4. Automatically proposes new prompt candidates as data accumulates

The system uses evolutionary search with Pareto-style selection to find
better-performing prompt candidates over time.

Usage:
    python run_online_demo.py
    python run_online_demo.py --queries 100 --model gpt-4.1-nano

Requirements:
    pip install synth-ai datasets openai httpx
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

# Add synth-ai to path
synth_ai_path = Path(__file__).resolve().parents[2]
if str(synth_ai_path) not in sys.path:
    sys.path.insert(0, str(synth_ai_path))

try:
    from datasets import load_dataset
    from synth_ai.core.utils.env import mint_demo_api_key
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    print("Install with: pip install synth-ai datasets openai httpx")
    sys.exit(1)


# Banking77 intent labels
BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
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

SYSTEM_PROMPT = (
    "You are an expert banking assistant that classifies customer queries into banking intents. "
    "Given a customer message, respond with exactly one intent label from the provided list "
    "using the `banking77_classify` tool."
)


def format_available_intents(label_names: list) -> str:
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(label_names))


class Banking77Dataset:
    """Simple loader for Banking77 dataset."""

    _DATA_URLS = {
        "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    }

    def __init__(self):
        self._cache = {}
        self._label_names = None

    def _load_split(self, split: str):
        if split not in self._cache:
            url = self._DATA_URLS.get(split)
            if not url:
                raise ValueError(f"Unknown split: {split}")
            ds = load_dataset("csv", data_files=url, split="train")
            self._cache[split] = ds
            if self._label_names is None:
                self._label_names = sorted(set(ds["category"]))
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
        return {
            "index": idx,
            "split": split,
            "text": str(row.get("text", "")),
            "label": row.get("category", "unknown"),
        }

    @property
    def label_names(self) -> list:
        if self._label_names is None:
            self._load_split("train")
        return self._label_names or []


async def init_online_gepa_system(
    system_id: str,
    config: dict,
    api_key: str,
    infra_api_base: str,
) -> dict:
    """Initialize an online GEPA system and get the proxy URL."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{infra_api_base}/api/gepa/v1/init",
            json={
                "system_id": system_id,
                "org_id": "demo",
                "config": config,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
        )
        if response.status_code == 200:
            return response.json()

        print(f"  Init returned {response.status_code}: {response.text[:200]}")
        raise RuntimeError(f"Failed to init GEPA system: {response.status_code}")


async def submit_reward(
    system_id: str,
    rollout_id: str,
    candidate_id: str,
    reward: float,
    api_key: str,
    infra_api_base: str,
) -> dict:
    """Submit a reward for a completed rollout to trigger learning."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{infra_api_base}/api/gepa/v1/{system_id}/status",
            json={
                "status": "done",
                "rollout_id": rollout_id,
                "candidate_id": candidate_id,
                "reward": reward,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
        )
        if response.status_code != 200:
            return {}
        return response.json()


async def query_via_proxy(
    proxy_url: str,
    query: str,
    available_intents: str,
    model: str,
    api_key: str,
) -> tuple[str, str | None, str | None]:
    """
    Send a classification query through the GEPA proxy.
    Returns (predicted_intent, rollout_id, candidate_id).
    """
    user_msg = (
        f"Customer Query: {query}\n\n"
        f"Available Intents:\n{available_intents}\n\n"
        f"Classify this query into one of the above banking intents using the tool call."
    )

    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "tools": [TOOL_SCHEMA],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{proxy_url}/chat/completions",
            json=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"Proxy returned {response.status_code}: {response.text[:200]}")

        rollout_id = response.headers.get("x-gepa-rollout-id")
        candidate_id = response.headers.get("x-gepa-candidate-id")
        data = response.json()

    # Parse tool call response
    try:
        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        args_raw = tool_call["function"]["arguments"]
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        return args["intent"], rollout_id, candidate_id
    except (KeyError, IndexError, json.JSONDecodeError):
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip(), rollout_id, candidate_id


async def get_system_state(system_id: str, api_key: str, infra_api_base: str) -> dict | None:
    """Fetch the current GEPA system state."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{infra_api_base}/api/gepa/v1/{system_id}/state",
                headers={"X-API-Key": api_key},
            )
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return None


def print_system_state(state: dict | None):
    """Pretty print the GEPA system state."""
    if not state:
        print("Could not fetch system state")
        return

    print(f"System ID: {state.get('system_id')}")
    print(f"Rollout count: {state.get('rollout_count', 0)}")
    print(f"Reward count: {state.get('reward_count', 0)}")
    print(f"Proposals triggered: {state.get('proposal_seq', 0)}")
    print(f"Archived candidates: {len(state.get('archived_candidates', []))}")

    candidates = state.get("candidates", {})
    print(f"\nActive Candidates ({len(candidates)}):")
    for cid, candidate in candidates.items():
        avg_r = candidate.get("avg_reward")
        avg_str = f"{avg_r:.2%}" if avg_r is not None else "N/A"
        rollouts = candidate.get("rollout_count", 0)
        parent = candidate.get("parent_id") or "baseline"
        print(f"  - {cid[:40]}: accuracy={avg_str}, rollouts={rollouts}, parent={parent[:20]}")


async def main():
    parser = argparse.ArgumentParser(description="Online GEPA Demo - Banking77")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    # Configuration
    synth_api_base = os.environ.get("SYNTH_API_URL", "https://api-dev.usesynth.ai")
    infra_api_base = os.environ.get("INFRA_API_URL", "https://infra-api-dev.usesynth.ai")

    print("=" * 60)
    print("Online GEPA Demo - Banking77 Classification")
    print("=" * 60)
    print(f"Backend: {synth_api_base}")
    print(f"Infra API: {infra_api_base}")
    print(f"Model: {args.model}")
    print(f"Queries: {args.queries}")

    # Check backend health
    r = httpx.get(f"{synth_api_base}/health", timeout=30)
    if r.status_code != 200:
        print(f"ERROR: Backend not healthy (status {r.status_code})")
        sys.exit(1)
    print(f"Backend: healthy")

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=synth_api_base)
    print(f"API Key: {api_key[:20]}...")

    # Load dataset
    print("\nLoading Banking77 dataset...")
    dataset = Banking77Dataset()
    dataset.ensure_ready(["train", "test"])
    available_intents = format_available_intents(dataset.label_names)
    print(f"Loaded: {dataset.size('train')} train, {dataset.size('test')} test samples")

    # Initialize online GEPA system
    system_id = f"banking77_online_{uuid.uuid4().hex[:8]}"
    print(f"\nInitializing online GEPA system: {system_id}")

    config = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": "banking77",
            "initial_prompt": {
                "id": "banking77_pattern",
                "name": "Banking77 Classification",
                "messages": [{"role": "system", "order": 0, "pattern": SYSTEM_PROMPT}],
            },
            "policy": {
                "model": args.model,
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "gepa": {
                "mode": "online",
                "env_name": "banking77",
                "online_proposer_min_rollouts": 10,
                "online_prune_threshold": 0.3,
                "online_prune_min_rollouts": 20,
                "proposer_url": "https://api.openai.com/v1/chat/completions",
                "proposer_model": "gpt-4.1-mini",
                "proposer_api_key": "",  # Uses server's OPENAI_API_KEY
            },
        },
    }

    try:
        init_result = await init_online_gepa_system(system_id, config, api_key, infra_api_base)
        print(f"System initialized: {init_result.get('status', 'ok')}")
        proxy_url = f"{infra_api_base}/api/gepa/v1/{system_id}"
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)

    # Run queries
    print("\n" + "=" * 60)
    print(f"Running {args.queries} queries through online GEPA")
    print("=" * 60)

    results = []
    correct = 0
    total = 0
    start_time = time.time()

    test_size = dataset.size("test")
    seeds = random.sample(range(test_size), min(args.queries, test_size))

    for i, seed in enumerate(seeds):
        sample = dataset.sample(split="test", index=seed)
        query = sample["text"]
        expected = sample["label"]

        try:
            predicted, rollout_id, candidate_id = await query_via_proxy(
                proxy_url=proxy_url,
                query=query,
                available_intents=available_intents,
                model=args.model,
                api_key=api_key,
            )

            is_correct = (
                predicted.lower().replace("_", " ").strip()
                == expected.lower().replace("_", " ").strip()
            )
            reward = 1.0 if is_correct else 0.0

            if is_correct:
                correct += 1
            total += 1

            if rollout_id and candidate_id:
                await submit_reward(
                    system_id, rollout_id, candidate_id, reward, api_key, infra_api_base
                )

            results.append({
                "seed": seed,
                "query": query[:100],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "rollout_id": rollout_id,
                "candidate_id": candidate_id,
            })

            if (i + 1) % 10 == 0:
                accuracy = correct / total if total > 0 else 0
                print(f"  [{i+1}/{args.queries}] Accuracy: {accuracy:.1%} ({correct}/{total})")

        except Exception as e:
            print(f"  Error on query {i+1}: {e}")
            results.append({"seed": seed, "query": query[:100], "error": str(e)})

    elapsed = time.time() - start_time
    final_accuracy = correct / total if total > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total queries: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {final_accuracy:.1%}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Throughput: {total/elapsed:.2f} queries/sec")

    # Save results
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"online_gepa_{timestamp}.json"

    output_data = {
        "system_id": system_id,
        "model": args.model,
        "total_queries": total,
        "correct": correct,
        "accuracy": final_accuracy,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print system state
    print("\n" + "=" * 60)
    print("GEPA SYSTEM STATE")
    print("=" * 60)
    state = await get_system_state(system_id, api_key, infra_api_base)
    print_system_state(state)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
