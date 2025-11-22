#!/usr/bin/env python3
"""Inspect job metadata to understand the data structure."""
import json
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

# Get a queued job to inspect
resp = requests.get(f"{base_url}/prompt-learning/online/jobs?limit=1&status=queued", headers=headers, timeout=5)
if resp.status_code != 200:
    print(f"ERROR: Failed to fetch jobs: {resp.status_code}")
    sys.exit(1)

jobs = resp.json()
if not jobs:
    print("No queued jobs found")
    sys.exit(0)

job = jobs[0]
job_id = job.get("job_id")

print(f"Inspecting job: {job_id}")
print("=" * 80)

# Get full job details
detail_resp = requests.get(f"{base_url}/prompt-learning/online/jobs/{job_id}", headers=headers, timeout=5)
if detail_resp.status_code != 200:
    print(f"ERROR: Failed to get job details: {detail_resp.status_code}")
    sys.exit(1)

job_detail = detail_resp.json()
print("\nFull job data:")
print(json.dumps(job_detail, indent=2, default=str))

# Check job_metadata structure
job_metadata = job_detail.get("job_metadata") or {}
print("\n" + "=" * 80)
print("job_metadata structure:")
print(json.dumps(job_metadata, indent=2, default=str))

# Check if algorithm is anywhere
print("\n" + "=" * 80)
print("Looking for algorithm...")
print(f"  job_detail.get('algorithm'): {job_detail.get('algorithm')}")
print(f"  job_detail.get('prompt_algorithm'): {job_detail.get('prompt_algorithm')}")
if isinstance(job_metadata, dict):
    print(f"  job_metadata.get('algorithm'): {job_metadata.get('algorithm')}")
    
    # Check nested structures
    request_metadata = job_metadata.get("request_metadata", {})
    if isinstance(request_metadata, dict):
        print(f"  request_metadata.get('algorithm'): {request_metadata.get('algorithm')}")
    
    prompt_initial_snapshot = job_metadata.get("prompt_initial_snapshot", {})
    if isinstance(prompt_initial_snapshot, dict):
        raw_config = prompt_initial_snapshot.get("raw_config", {})
        if isinstance(raw_config, dict):
            pl_section = raw_config.get("prompt_learning", {})
            if isinstance(pl_section, dict):
                gepa = pl_section.get("gepa")
                mipro = pl_section.get("mipro")
                print(f"  raw_config has gepa: {gepa is not None}")
                print(f"  raw_config has mipro: {mipro is not None}")



