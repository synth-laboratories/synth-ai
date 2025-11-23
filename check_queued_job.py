#!/usr/bin/env python3
"""Check a specific queued job's structure."""
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

job_id = sys.argv[1] if len(sys.argv) > 1 else "pl_037651b3c89b4430"

print(f"Checking job: {job_id}")
print("=" * 80)

resp = requests.get(f"{base_url}/prompt-learning/online/jobs/{job_id}", headers=headers, timeout=5)
if resp.status_code != 200:
    print(f"ERROR: {resp.status_code} - {resp.text[:200]}")
    sys.exit(1)

job = resp.json()

print("\njob_metadata:")
print(json.dumps(job.get("job_metadata", {}), indent=2, default=str)[:500])

print("\n\nprompt_initial_snapshot keys:")
snapshot = job.get("prompt_initial_snapshot", {})
if isinstance(snapshot, dict):
    print(list(snapshot.keys())[:10])
    
    raw_config = snapshot.get("raw_config", {})
    if isinstance(raw_config, dict):
        pl = raw_config.get("prompt_learning", {})
        if isinstance(pl, dict):
            print("\nraw_config.prompt_learning:")
            print(f"  has gepa: {pl.get('gepa') is not None}")
            print(f"  has mipro: {pl.get('mipro') is not None}")
            print(f"  algorithm field: {pl.get('algorithm')}")
            print(f"  gepa keys: {list(pl.get('gepa', {}).keys())[:5] if isinstance(pl.get('gepa'), dict) else 'N/A'}")
            print(f"  mipro keys: {list(pl.get('mipro', {}).keys())[:5] if isinstance(pl.get('mipro'), dict) else 'N/A'}")




