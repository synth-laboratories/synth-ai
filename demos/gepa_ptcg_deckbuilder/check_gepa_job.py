#!/usr/bin/env python3
"""Check GEPA job status and events."""

import argparse

import httpx
from synth_ai.core.urls import synth_prompt_learning_events_url, synth_prompt_learning_job_url
from synth_ai.sdk.auth import get_or_mint_synth_user_key

SYNTH_USER_KEY = get_or_mint_synth_user_key()

parser = argparse.ArgumentParser(description="Check GEPA job status")
parser.add_argument("job_id", help="GEPA job ID")
args = parser.parse_args()


def main():
    # Get job status
    print(f"\nFetching job status for: {args.job_id}")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            synth_prompt_learning_job_url(args.job_id),
            headers={"Authorization": f"Bearer {SYNTH_USER_KEY}"},
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
            synth_prompt_learning_events_url(args.job_id),
            headers={"Authorization": f"Bearer {SYNTH_USER_KEY}"},
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
