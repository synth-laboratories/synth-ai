#!/usr/bin/env python3
"""Test the local API TaskInfo endpoint."""

import asyncio
from pathlib import Path

demo_dir = Path(__file__).parent
repo_root = demo_dir.parent.parent

# Load .env
try:
    from dotenv import load_dotenv

    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded {env_file}")
except ImportError:
    pass

import os

import httpx

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    from synth_ai.sdk.task import run_server_background
# Import the app creation function
from run_demo import create_web_design_local_api
from synth_ai.core.env import PROD_BASE_URL
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

# Get API key
API_KEY = os.environ.get("SYNTH_API_KEY", "")
SYNTH_API_BASE = PROD_BASE_URL

# Get environment key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")


async def main():
    BASELINE_STYLE_PROMPT = """You are generating a professional startup website screenshot.

VISUAL STYLE GUIDELINES:
- Use a clean, modern, minimalist design aesthetic
- Color Scheme: Light backgrounds with high contrast dark text
- Typography: Large, bold headings with clear hierarchy
- Layout: Spacious with generous padding and margins
- Branding: Professional, tech-forward visual identity

Create a webpage that feels polished, modern, and trustworthy."""

    # Start local API
    print("Starting local API...")
    app = create_web_design_local_api(BASELINE_STYLE_PROMPT)
    port = acquire_port(8002, on_conflict=PortConflictBehavior.FIND_NEW)

    run_server_background(app, port)
    print(f"Local API running on port {port}")

    # Wait for health check
    await asyncio.sleep(3)

    # Test TaskInfo endpoint
    print("\nTesting TaskInfo endpoint...")
    url = f"http://localhost:{port}/task_info?seed=0"
    print(f"GET {url}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {ENVIRONMENT_API_KEY}"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response length: {len(response.text)} bytes")
                print(f"Number of task infos: {len(data)}")
                # Don't print the full response as it contains large base64 images
                print("TaskInfo fetch successful!")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback

        traceback.print_exc()

    print("\nKeeping server running for 60 seconds...")
    await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
