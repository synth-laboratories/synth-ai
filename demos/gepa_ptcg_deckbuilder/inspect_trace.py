#!/usr/bin/env python3
"""
Inspect trace data for a specific seed to see the actual rollout response.
"""

import argparse
import json
import os

import httpx
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key

parser = argparse.ArgumentParser(description="Inspect trace data for eval job")
parser.add_argument("job_id", help="Eval job ID")
parser.add_argument("seed", type=int, help="Seed to inspect")
parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
args = parser.parse_args()


def main():
    # Backend setup
    if args.local:
        backend_url = "http://localhost:8000"
    else:
        backend_url = PROD_BASE_URL

    # API key
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("No SYNTH_API_KEY, minting demo key...")
        api_key = mint_demo_api_key(backend_url=backend_url)

    # Get trace data
    print(f"\nFetching trace for seed {args.seed}...")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{backend_url}/api/eval/jobs/{args.job_id}/traces/{args.seed}",
            params={"format": "json"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code != 200:
            print(f"Failed to get trace: {resp.status_code} {resp.text}")
            return
        trace_data = resp.json()

    trace = trace_data.get("trace", {})
    if not trace:
        print("No trace data found")
        return

    print("\n" + "=" * 60)
    print(f"TRACE DATA FOR SEED {args.seed}")
    print("=" * 60)

    # Look for rollout response in trace
    rollout_response = None
    if isinstance(trace, dict):
        # Check common locations for rollout response
        if "rollout_response" in trace:
            rollout_response = trace["rollout_response"]
        elif "response" in trace:
            rollout_response = trace["response"]
        elif "data" in trace:
            rollout_response = trace["data"]
        else:
            # The trace might be the rollout response itself
            rollout_response = trace

    if rollout_response:
        print("\nRollout Response Structure:")
        print(
            f"  Keys: {list(rollout_response.keys()) if isinstance(rollout_response, dict) else 'not a dict'}"
        )

        metrics = rollout_response.get("metrics", {}) if isinstance(rollout_response, dict) else {}
        if metrics:
            print("\nMetrics:")
            print(f"  outcome_reward: {metrics.get('outcome_reward')}")
            print(f"  outcome_score: {metrics.get('outcome_score')}")
            print(f"  reward_mean: {metrics.get('reward_mean')}")
            print(f"  Keys: {list(metrics.keys())}")

            details = metrics.get("details", {})
            if details:
                print("\nDetails:")
                print(f"  Keys: {list(details.keys())}")
                instance_id = details.get("instance_id")
                deck_size = details.get("deck_size")
                error = details.get("error")
                constraint_score = details.get("constraint_score")
                win_rate = details.get("win_rate")
                constraint_results = details.get("constraint_results", [])
                failed_constraints = details.get("failed_constraints", [])
                deck = details.get("deck", [])

                print(f"  Instance ID: {instance_id}")
                print(f"  Deck size: {deck_size}")
                print(f"  Constraint score: {constraint_score}")
                print(f"  Win rate: {win_rate}")
                if error:
                    print(f"  Error: {error}")
                if constraint_results:
                    print(f"\n  Constraint Results ({len(constraint_results)}):")
                    for cr in constraint_results:
                        satisfied = cr.get("satisfied", False)
                        ctype = cr.get("type", "?")
                        explanation = cr.get("explanation", "")
                        status = "✓" if satisfied else "✗"
                        print(f"    [{status}] {ctype}: {explanation}")
                if failed_constraints:
                    print(f"\n  Failed Constraints ({len(failed_constraints)}):")
                    for fc in failed_constraints[:10]:  # Show first 10
                        print(f"    - {fc.get('type', '?')}: {fc.get('explanation', '')}")
                if deck:
                    print(f"\n  Deck (first 20 cards): {deck[:20]}")
        else:
            print("\nNo metrics found in rollout response")
            print(
                f"Full response (first 2000 chars):\n{json.dumps(rollout_response, indent=2, default=str)[:2000]}"
            )
    else:
        print("\nNo rollout response found in trace")
        print(f"Trace keys: {list(trace.keys()) if isinstance(trace, dict) else 'not a dict'}")
        print(
            f"Trace sample (first 2000 chars):\n{json.dumps(trace, indent=2, default=str)[:2000]}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
