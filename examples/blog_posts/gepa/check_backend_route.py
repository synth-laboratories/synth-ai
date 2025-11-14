#!/usr/bin/env python3
"""Check if the backend prompt learning route is registered."""

import httpx
import sys

backend_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

print(f"Checking backend at: {backend_url}")
print()

# Check health
try:
    resp = httpx.get(f"{backend_url}/api/health", timeout=2.0)
    print(f"✓ Health check: {resp.status_code}")
except Exception as e:
    print(f"✗ Health check failed: {e}")
    sys.exit(1)

# Check if route exists
try:
    resp = httpx.post(
        f"{backend_url}/api/prompt-learning/online/jobs",
        headers={"Authorization": "Bearer test"},
        json={"config_body": {"prompt_learning": {"algorithm": "gepa"}}},
        timeout=2.0,
    )
    if resp.status_code == 404:
        print("✗ Route /api/prompt-learning/online/jobs returns 404 (not registered)")
        print()
        print("SOLUTION: Restart the backend server:")
        print("  1. Stop the current backend process")
        print("  2. Restart: cd monorepo/backend && uv run uvicorn app.routes.main:app --host 127.0.0.1 --port 8000")
        print("  3. Check for warnings: 'Failed to import prompt_learning_online_router'")
        sys.exit(1)
    elif resp.status_code in (200, 201, 400, 401, 403):
        print(f"✓ Route exists (status: {resp.status_code})")
    else:
        print(f"? Unexpected status: {resp.status_code}")
except Exception as e:
    print(f"✗ Error checking route: {e}")
    sys.exit(1)

