#!/usr/bin/env python3
"""
Capture OpenCode's actual request messages by running it through the interceptor.
Simple, minimal, correct.
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

import httpx


async def main():
    backend = "http://localhost:8000"
    correlation_id = f"capture_{uuid.uuid4().hex[:8]}"

    # The interceptor baseURL that OpenCode will call
    # OpenCode appends /responses to this, so we give it the base
    interceptor_base = f"{backend}/api/interceptor/v1/{correlation_id}"

    print(f"Correlation ID: {correlation_id}")
    print(f"Interceptor base for OpenCode: {interceptor_base}")

    # Check backend
    r = httpx.get(f"{backend}/health", timeout=10)
    if r.status_code != 200:
        print(f"Backend not healthy: {r.status_code}")
        return
    print("Backend healthy")

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return
    print(f"API key: {api_key[:20]}...")

    # Create sandbox
    with tempfile.TemporaryDirectory(prefix="capture_") as td:
        sandbox = Path(td)

        # Write opencode.json config pointing to interceptor
        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": "openai/gpt-5-nano",
            "provider": {
                "openai": {
                    "options": {
                        "apiKey": api_key,
                        "baseURL": interceptor_base,
                    }
                }
            },
            "permission": {"*": "allow"},
        }
        (sandbox / "opencode.json").write_text(json.dumps(config, indent=2))

        # Write test file
        (sandbox / "output.txt").write_text("placeholder\n")
        (sandbox / "AGENTS.md").write_text("Follow the task prompt.\n")

        # The prompt we send to OpenCode
        prompt = """Edit output.txt to contain exactly: Hello, world!"""

        print(f"\nRunning OpenCode with prompt: {prompt}")
        print(f"Sandbox: {sandbox}")
        print(f"Config baseURL: {interceptor_base}")

        # Run OpenCode
        opencode_bin = str(Path.home() / ".opencode" / "bin" / "opencode")
        cmd = [opencode_bin, "run", "--format", "json", "--model", "openai/gpt-5-nano", prompt]

        env = os.environ.copy()
        env["OPENAI_API_KEY"] = api_key

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(sandbox),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

        print(f"\nOpenCode exit code: {proc.returncode}")
        if stderr:
            print(f"Stderr: {stderr.decode()[:500]}")

        # Check output
        output_file = sandbox / "output.txt"
        if output_file.exists():
            print(f"output.txt: {output_file.read_text()!r}")

    # Now fetch the trace
    print(f"\n{'=' * 60}")
    print("FETCHING TRACE")
    print(f"{'=' * 60}")

    await asyncio.sleep(2)  # Let trace settle

    trace_url = f"{backend}/api/interceptor/v1/trace/by-correlation/{correlation_id}"
    print(f"GET {trace_url}")

    r = httpx.get(trace_url, timeout=10)
    print(f"Status: {r.status_code}")

    if r.status_code != 200:
        print(f"Error: {r.text}")
        return

    data = r.json()
    matches = data.get("matches", [])
    print(f"Matches: {len(matches)}")

    if not matches:
        print("No trace found!")
        # Debug: list what traces exist
        lease_r = httpx.post(
            f"{backend}/api/interceptor/v1/trace/lease-any",
            json={"limit": 10, "ttl_seconds": 30},
            timeout=10,
        )
        if lease_r.status_code == 200:
            leased = lease_r.json().get("granted", {})
            print(f"Available traces: {list(leased.keys())}")
        return

    # Extract request_messages
    match = matches[0]
    metadata = match.get("metadata", {})
    conversation = metadata.get("conversation", {})
    request_messages = conversation.get("request_messages", [])

    print(f"\n{'=' * 60}")
    print("REQUEST MESSAGES")
    print(f"{'=' * 60}")

    for i, msg in enumerate(request_messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        print(f"\n[{i}] ROLE: {role}")
        print(f"    LENGTH: {len(content)} chars")
        print("    CONTENT:")
        print("-" * 40)
        # Print full content, not truncated
        print(content)
        print("-" * 40)

    # Save to file
    out_file = Path(__file__).parent / "captured_messages.json"
    with open(out_file, "w") as f:
        json.dump({"correlation_id": correlation_id, "messages": request_messages}, f, indent=2)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
