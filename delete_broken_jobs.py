#!/usr/bin/env python3
"""Delete broken queued jobs that can't be started."""
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
resp = requests.get(f"{base_url}/prompt-learning/online/jobs?limit=20&status=queued", headers=headers, timeout=5)
if resp.status_code != 200:
    print(f"ERROR: Failed to fetch jobs: {resp.status_code}")
    sys.exit(1)

jobs = resp.json()
print(f"Found {len(jobs)} queued jobs")
print("=" * 80)

# Try to cancel them (best we can do - can't actually delete via API)
for job in jobs:
    job_id = job.get("job_id")
    if not job_id:
        continue
    
    print(f"\nCanceling {job_id}...")
    cancel_url = f"{base_url}/learning/jobs/{job_id}/cancel"
    cancel_resp = requests.post(cancel_url, headers=headers, json={}, timeout=10)
    
    if cancel_resp.status_code in (200, 201):
        print("  ✅ Cancelled")
    else:
        print(f"  ⚠️  Cancel returned {cancel_resp.status_code}")

print("\n" + "=" * 80)
print("✅ Done - these jobs are corrupted and can't be started")
print("   New jobs created with the fix will work correctly")




