#!/usr/bin/env python3
"""Test the local API TaskInfo endpoint."""

import asyncio
import importlib
import os
import sys
from pathlib import Path

import httpx
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    from synth_ai.sdk.task import run_server_background

demo_dir = Path(__file__).parent
repo_root = demo_dir.parent.parent

# Import local module dynamically
sys.path.insert(0, str(demo_dir))
_run_demo = importlib.import_module("run_demo")
create_web_design_local_api = _run_demo.create_web_design_local_api

# Get API key
API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("No SYNTH_API_KEY, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=BACKEND_URL_BASE)
    os.environ["SYNTH_API_KEY"] = API_KEY

# Get environment key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=BACKEND_URL_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")


async def main():
    baseline_style_prompt = """You are generating a professional startup website screenshot.

VISUAL STYLE GUIDELINES:
- Use a clean, modern, minimalist design aesthetic
- Color Scheme: Light backgrounds with high contrast dark text
- Typography: Large, bold headings with clear hierarchy
- Layout: Spacious with generous padding and margins
- Branding: Professional, tech-forward visual identity

Create a webpage that feels polished, modern, and trustworthy."""

    # Start local API
    print("Starting local API...")
    app = create_web_design_local_api(baseline_style_prompt)
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
