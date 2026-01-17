#!/usr/bin/env python3
"""Check GEPA job status and events."""

import argparse
import os

import httpx
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key

parser = argparse.ArgumentParser(description="Check GEPA job status")
parser.add_argument("job_id", help="GEPA job ID")
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
        api_key = mint_demo_api_key(backend_url=backend_url)

    # Get job status
    print(f"\nFetching job status for: {args.job_id}")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{backend_url}/api/prompt-learning/online/jobs/{args.job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code != 200:
            print(f"Failed to get job: {resp.status_code} {resp.text}")
            return
        job_data = resp.json()

    print(f"\nStatus: {job_data.get('status')}")
    print(f"Error: {job_data.get('error')}")

    # Get events
    print("\nFetching events...")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{backend_url}/api/prompt-learning/online/jobs/{args.job_id}/events",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": 50},
        )
        if resp.status_code == 200:
            events_data = resp.json()
            # Events might be a list or a dict with 'events' key
            if isinstance(events_data, list):
                events = events_data
            elif isinstance(events_data, dict) and "events" in events_data:
                events = events_data["events"]
            else:
                events = []

            print(f"\nFound {len(events)} events:")
            for event in events[-10:] if len(events) > 0 else []:  # Show last 10 events
                print(f"\n  [{event.get('seq', '?')}] {event.get('type', '?')}")
                print(f"     Message: {event.get('message', '')}")
                if event.get("data"):
                    data = event.get("data", {})
                    if isinstance(data, dict):
                        error = data.get("error")
                        if error:
                            print(f"     Error: {error}")
                        # Show other important fields
                        for key in ["candidate_id", "score", "generation", "version_id"]:
                            if key in data:
                                print(f"     {key}: {data[key]}")
        else:
            print(f"Failed to get events: {resp.status_code} {resp.text}")

    print("\nDone!")


if __name__ == "__main__":
    main()
