#!/usr/bin/env python3
"""Cancel stuck backend jobs that have been polling with no events for hours."""

import os
import sys
import requests
from pathlib import Path

# Try to load from .env files
try:
    from dotenv import load_dotenv
    # Try multiple possible .env locations
    for env_file in [Path(".env"), Path(".env.dev"), Path("monorepo/backend/.env.dev")]:
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
api_key = os.getenv("API_KEY") or os.getenv("ENVIRONMENT_API_KEY")

if not api_key:
    print("❌ No API_KEY found. Please set BACKEND_URL and API_KEY environment variables.")
    print("   Example: export API_KEY=your_key && python cancel_stuck_jobs.py")
    sys.exit(1)

stuck_job_ids = [
    "pl_c1b6750e160f4ef6",
    "pl_5b4075a0e0094371",
    "pl_8962be637d7a4812",
    "pl_049e3665072a4174",
]

print("=" * 80)
print("CHECKING STUCK JOBS STATUS")
print("=" * 80)
print(f"Backend URL: {backend_url}")
print()

# First check status
for job_id in stuck_job_ids:
    url = f"{backend_url}/api/prompt-learning/online/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            print(f"📊 {job_id}: Status = {status}")
        else:
            print(f"❌ {job_id}: Failed to get status ({response.status_code})")
    except Exception as e:
        print(f"❌ {job_id}: Error checking status - {e}")

print()
print("=" * 80)
print("CANCELING STUCK JOBS")
print("=" * 80)

# Cancel each job
for job_id in stuck_job_ids:
    url = f"{backend_url}/api/learning/jobs/{job_id}/cancel"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json={}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            message = data.get("message", "Cancellation requested")
            print(f"✅ {job_id}: {message} (status: {status})")
        else:
            print(f"❌ {job_id}: Failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ {job_id}: Error - {e}")

print()
print("=" * 80)

