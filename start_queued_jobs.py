#!/usr/bin/env python3
"""Manually start queued prompt learning jobs."""
import os
import sys
from pathlib import Path

import requests

# Get API key from .env
repo_root = Path(__file__).parent
env_file = repo_root / ".env"

api_key = None
if env_file.exists():
    from synth_ai.api.train.utils import read_env_file
    env_data = read_env_file(env_file)
    api_key = env_data.get("SYNTH_API_KEY")

if not api_key:
    api_key = os.getenv("SYNTH_API_KEY")

if not api_key:
    print("ERROR: No API key found")
    sys.exit(1)

headers = {"Authorization": f"Bearer {api_key}"}
base_url = "http://localhost:8000/api"

# Get queued jobs
print("Fetching queued jobs...")
resp = requests.get(f"{base_url}/prompt-learning/online/jobs?limit=20", headers=headers, timeout=5)
if resp.status_code != 200:
    print(f"ERROR: Failed to fetch jobs: {resp.status_code}")
    sys.exit(1)

jobs = resp.json()
queued_jobs = [j for j in jobs if j.get("status") == "queued"]

if not queued_jobs:
    print("✅ No queued jobs found")
    sys.exit(0)

print(f"\nFound {len(queued_jobs)} queued job(s):")
for job in queued_jobs:
    job_id = job.get("job_id", "N/A")
    created = job.get("created_at", "")[:19] if job.get("created_at") else "N/A"
    print(f"  {job_id}: Created at {created}")

print(f"\nStarting {len(queued_jobs)} job(s)...")
for job in queued_jobs:
    job_id = job.get("job_id")
    if not job_id:
        continue
    
    print(f"\nStarting {job_id}...")
    start_url = f"{base_url}/prompt-learning/online/jobs/{job_id}/start"
    start_resp = requests.post(start_url, headers=headers, json={}, timeout=30)
    
    if start_resp.status_code in (200, 201):
        print("  ✅ Started successfully")
    else:
        try:
            error = start_resp.json()
        except Exception:
            error = start_resp.text[:200]
        print(f"  ❌ Failed: {start_resp.status_code} - {error}")

print("\n✅ Done")


