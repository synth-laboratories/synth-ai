#!/usr/bin/env python3
"""
Quick test to check fine-tuning job status.
"""

import httpx
import asyncio
import os

SYNTH_API_KEY = os.getenv("SYNTH_API_KEY", "sk_live_9592524d-be1b-48b2-aff7-976b277eac95")
SYNTH_API_URL = os.getenv("SYNTH_API_URL", "http://localhost:8000")

async def check_job_status(job_id: str = "ftjob-9383913972144b06"):
    """Check status of a fine-tuning job."""
    headers = {
        "Authorization": f"Bearer {SYNTH_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=10) as client:
        # Check specific job if provided
        if job_id:
            print(f"Checking job: {job_id}")
            response = await client.get(
                f"{SYNTH_API_URL}/api/fine_tuning/jobs/{job_id}",
                headers=headers
            )
            
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"\nJob details:")
                    for key, value in data.items():
                        if key not in ["hyperparameters"]:
                            print(f"  {key}: {value}")
                    
                    if "hyperparameters" in data:
                        print("  Hyperparameters:")
                        for k, v in data["hyperparameters"].items():
                            print(f"    {k}: {v}")
                except Exception as e:
                    print(f"Failed to parse response: {e}")
                    print(f"Raw response: {response.text[:500]}")
            else:
                print(f"Failed to get job: {response.text[:500]}")
        
        # Try listing with POST
        print("\n\nTrying to list jobs with POST...")
        list_response = await client.post(
            f"{SYNTH_API_URL}/api/fine_tuning/jobs",
            headers=headers,
            json={}
        )
        print(f"POST list status: {list_response.status_code}")
        if list_response.status_code != 405:
            print(f"Response: {list_response.text[:200]}")

if __name__ == "__main__":
    asyncio.run(check_job_status())