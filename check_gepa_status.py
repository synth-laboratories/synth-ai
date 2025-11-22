#!/usr/bin/env python3
"""Quick script to check GEPA job status."""
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

print("=" * 80)
print("GEPA Job Status Check")
print("=" * 80)

# Get recent jobs
resp = requests.get(f"{base_url}/prompt-learning/online/jobs?limit=10", headers=headers, timeout=5)
if resp.status_code != 200:
    print(f"ERROR: Failed to fetch jobs: {resp.status_code}")
    sys.exit(1)

jobs = resp.json()
print("\n📊 Recent Jobs (last 10):")
print("-" * 80)
for job in jobs[:10]:
    job_id = job.get("job_id", "N/A")
    status = job.get("status", "N/A")
    created = job.get("created_at", "")[:19] if job.get("created_at") else "N/A"
    started = job.get("started_at", "")[:19] if job.get("started_at") else "N/A"
    finished = job.get("finished_at", "")[:19] if job.get("finished_at") else "N/A"
    
    status_emoji = {
        "queued": "⏳",
        "running": "🔄",
        "succeeded": "✅",
        "failed": "❌",
    }.get(status, "❓")
    
    print(f"{status_emoji} {job_id}: {status:10} | Created: {created} | Started: {started} | Finished: {finished}")

# Check for queued jobs
queued = [j for j in jobs if j.get("status") == "queued"]
if queued:
    print(f"\n⏳ Queued Jobs ({len(queued)}):")
    print("-" * 80)
    for job in queued:
        job_id = job.get("job_id", "N/A")
        created = job.get("created_at", "")[:19] if job.get("created_at") else "N/A"
        print(f"  {job_id}: Queued since {created}")
else:
    print("\n✅ No queued jobs")

# Check for running jobs
running = [j for j in jobs if j.get("status") == "running"]
if running:
    print(f"\n🔄 Running Jobs ({len(running)}):")
    print("-" * 80)
    for job in running:
        job_id = job.get("job_id", "N/A")
        started = job.get("started_at", "")[:19] if job.get("started_at") else "N/A"
        print(f"  {job_id}: Running since {started}")
else:
    print("\n✅ No running jobs")

# Check for recent succeeded jobs
succeeded = [j for j in jobs if j.get("status") == "succeeded"]
if succeeded:
    print(f"\n✅ Recent Completed Jobs ({len(succeeded)}):")
    print("-" * 80)
    for job in succeeded[:5]:
        job_id = job.get("job_id", "N/A")
        finished = job.get("finished_at", "")[:19] if job.get("finished_at") else "N/A"
        print(f"  {job_id}: Completed at {finished}")

print("\n" + "=" * 80)



