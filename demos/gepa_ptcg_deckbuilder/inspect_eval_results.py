#!/usr/bin/env python3
"""
Inspect detailed eval job results to understand why seeds failed.
"""

import argparse
import os

import httpx
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key

parser = argparse.ArgumentParser(description="Inspect eval job results")
parser.add_argument("job_id", help="Eval job ID to inspect")
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
        print(f"API Key: {api_key[:20]}...")

    # Get job status
    print(f"\nFetching job status for: {args.job_id}")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{backend_url}/api/eval/jobs/{args.job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code != 200:
            print(f"Failed to get job: {resp.status_code} {resp.text}")
            return
        job_data = resp.json()
        print(f"Status: {job_data.get('status')}")
        print(f"Error: {job_data.get('error')}")

    # Get detailed results
    print("\nFetching detailed results...")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{backend_url}/api/eval/jobs/{args.job_id}/results",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code != 200:
            print(f"Failed to get results: {resp.status_code} {resp.text}")
            return
        results_data = resp.json()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = results_data.get("summary", {})
    print(f"Mean reward: {summary.get('mean_reward', 'N/A')}")
    print(f"Total tokens: {summary.get('total_tokens', 'N/A')}")
    print(f"Total cost: ${summary.get('total_cost_usd', 'N/A')}")
    print(f"Completed: {summary.get('completed', 'N/A')}/{summary.get('total', 'N/A')}")

    print("\n" + "=" * 60)
    print("PER-SEED RESULTS")
    print("=" * 60)
    results = results_data.get("results", [])
    for row in results:
        seed = row.get("seed", "?")
        outcome_reward = (
            row.get("outcome_reward") or row.get("outcome_score") or row.get("reward_mean")
        )
        error = row.get("error")
        tokens = row.get("tokens")
        cost = row.get("cost_usd")
        correlation_id = row.get("correlation_id")
        trace_id = row.get("trace_id")

        print(f"\nSeed {seed}:")
        print(f"  Outcome reward: {outcome_reward}")
        print(f"  Tokens: {tokens}")
        print(f"  Cost: ${cost}" if cost else "  Cost: N/A")
        print(f"  Correlation ID: {correlation_id}")
        print(f"  Trace ID: {trace_id}")
        if error:
            print(f"  Error: {error}")

    # Check trajectory data in results (contains full rollout response)
    print("\n" + "=" * 60)
    print("TRAJECTORY DATA (full rollout response)")
    print("=" * 60)
    for row in results:
        seed = row.get("seed", "?")
        trajectory = row.get("trajectory")
        if trajectory:
            print(f"\nSeed {seed} trajectory:")
            metrics = trajectory.get("metrics", {})
            details = metrics.get("details", {})

            outcome_reward = metrics.get("outcome_reward")
            print(f"  Metrics outcome_reward: {outcome_reward}")

            if details:
                print(f"  Details keys: {list(details.keys())}")
                instance_id = details.get("instance_id")
                deck_size = details.get("deck_size")
                constraint_results = details.get("constraint_results", [])
                failed_constraints = details.get("failed_constraints", [])
                deck = details.get("deck", [])
                error = details.get("error")

                print(f"  Instance ID: {instance_id}")
                print(f"  Deck size: {deck_size}")
                if error:
                    print(f"  Error: {error}")
                if constraint_results:
                    print(f"  Constraint results: {len(constraint_results)}")
                    print("  Constraint breakdown:")
                    for cr in constraint_results:
                        satisfied = cr.get("satisfied", False)
                        ctype = cr.get("type", "?")
                        explanation = cr.get("explanation", "")
                        status = "✓" if satisfied else "✗"
                        print(f"    [{status}] {ctype}: {explanation}")
                if failed_constraints:
                    print(f"  Failed constraints: {len(failed_constraints)}")
                    for fc in failed_constraints[:5]:  # Show first 5
                        print(f"    - {fc.get('type', '?')}: {fc.get('explanation', '')}")
                if deck:
                    print(f"  Deck sample (first 15): {deck[:15]}")
                if details.get("constraint_score") is not None:
                    print(f"  Constraint score: {details.get('constraint_score')}")
                if details.get("win_rate") is not None:
                    print(f"  Win rate: {details.get('win_rate')}")
            else:
                print("  No details found in trajectory")
                print(f"  Trajectory keys: {list(trajectory.keys())}")
                print(
                    f"  Metrics keys: {list(metrics.keys()) if isinstance(metrics, dict) else 'not a dict'}"
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
