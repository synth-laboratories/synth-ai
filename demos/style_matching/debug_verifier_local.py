import argparse
import json
import os
from typing import Any, Dict, List

import httpx
from synth_ai.core.urls import BACKEND_URL_BASE, join_url


def extract_verifier_score(result: Dict[str, Any]) -> float:
    output = result.get("output", result)
    if isinstance(output, dict):
        outcome_review = output.get("outcome_review")
        if isinstance(outcome_review, dict) and isinstance(
            outcome_review.get("total"), (int, float)
        ):
            return float(outcome_review["total"])
        event_reviews = output.get("event_reviews")
        if isinstance(event_reviews, list) and event_reviews:
            totals = [rev.get("total") for rev in event_reviews if isinstance(rev, dict)]
            totals = [t for t in totals if isinstance(t, (int, float))]
            if totals:
                return float(sum(totals) / len(totals))
        event_totals = output.get("event_totals")
        if isinstance(event_totals, list) and event_totals:
            totals = [t for t in event_totals if isinstance(t, (int, float))]
            if totals:
                return float(sum(totals) / len(totals))
        if isinstance(output.get("total"), (int, float)):
            return float(output["total"])
    return 0.0


def make_session_trace() -> Dict[str, Any]:
    return {
        "session_id": "debug-trace",
        "session_time_steps": [
            {
                "step_id": "1",
                "step_index": 0,
                "events": [
                    {
                        "event_type": "runtime",
                        "event_id": 1,
                        "type": "user_message",
                        "content": "Write about focus.",
                    },
                    {
                        "event_type": "runtime",
                        "event_id": 2,
                        "type": "assistant_message",
                        "content": "Optionality feels safe, but it dilutes learning. Clarity is leverage.",
                    },
                ],
            }
        ],
    }


def make_gold_examples(session_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "summary": "Short, direct example",
            "gold_score": 0.95,
            "gold_reasoning": "Direct stance, concrete advice, crisp closing line.",
            "trace": session_trace,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug verifier scoring against backend.")
    parser.add_argument(
        "--verifier-job-id",
        default="zero_shot_verifier_contrastive_single",
        help="Verifier graph job id",
    )
    parser.add_argument("--model", default="gpt-4.1-nano", help="Verifier model override")
    args = parser.parse_args()

    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        raise SystemExit("SYNTH_API_KEY not found in environment")

    session_trace = make_session_trace()
    gold_examples = make_gold_examples(session_trace)

    payload = {
        "job_id": args.verifier_job_id,
        "input": {
            "trace": session_trace,
            "gold_examples": gold_examples,
            "candidate_score": 0.5,
            "candidate_reasoning": "Auto-evaluated from debug script",
            "options": {"model": args.model},
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = join_url(BACKEND_URL_BASE, "/api/graphs/completions")
    print(f"POST {url}")
    print(f"job_id: {args.verifier_job_id}")

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, headers=headers, json=payload)

    print(f"status: {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text[:2000])
        return

    data = resp.json()
    print("response keys:", sorted(data.keys()))
    score = extract_verifier_score(data)
    print(f"parsed score: {score}")
    print("raw output snippet:")
    output = data.get("output", data)
    print(json.dumps(output, indent=2)[:2000])


if __name__ == "__main__":
    main()
