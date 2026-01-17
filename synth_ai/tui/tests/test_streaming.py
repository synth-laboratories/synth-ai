"""
Integration test that verifies OpenCode streaming actually works.

This test:
1. Connects to a running OpenCode server
2. Subscribes to SSE events
3. Sends a prompt
4. Verifies that text-delta events arrive at DIFFERENT timestamps (proving streaming)
"""

import asyncio
import contextlib
import json
import os
import time
from typing import Optional

import httpx
import pytest

pytestmark = pytest.mark.asyncio


async def find_working_opencode() -> Optional[tuple[int, str]]:
    """Find an OpenCode server that has synth provider configured."""
    # OpenCode typically uses ports in the 49xxx-55xxx range
    # Check from high to low to prefer newer servers
    candidate_ports = list(range(55999, 49000, -1)) + list(range(4096, 4100))

    # Allow explicit port override
    env_port = os.environ.get("OPENCODE_PORT")
    if env_port:
        candidate_ports = [int(env_port)] + candidate_ports

    found_ports = []

    async with httpx.AsyncClient(timeout=1.0) as client:
        for port in candidate_ports:
            try:
                resp = await client.get(f"http://127.0.0.1:{port}/session")
                if resp.status_code == 200:
                    found_ports.append(port)
                    # Check if this server has synth provider
                    try:
                        prov_resp = await client.get(f"http://127.0.0.1:{port}/provider")
                        if prov_resp.status_code == 200:
                            providers = prov_resp.json()
                            if providers.get("all"):
                                provider_ids = [p.get("id") for p in providers["all"]]
                                if "synth" in provider_ids:
                                    print(f"  Found OpenCode with synth provider on port {port}")
                                    return (port, f"http://127.0.0.1:{port}")
                    except Exception:
                        pass
            except Exception:
                pass

        # If we found some ports but none with synth, use the first one anyway
        if found_ports:
            port = found_ports[0]
            print(f"  Using OpenCode on port {port} (synth provider not confirmed)")
            return (port, f"http://127.0.0.1:{port}")

    return None


async def test_streaming_events():
    """
    Verify that OpenCode emits streaming events at different timestamps.

    This proves streaming is working end-to-end:
    - Backend streams chunks
    - OpenCode processes them incrementally
    - SSE events are emitted in real-time
    """
    # Find running OpenCode server by probing ports
    result = await find_working_opencode()
    if not result:
        pytest.skip("No OpenCode server found - start TUI first")

    port, base_url = result
    print(f"\nTesting against OpenCode at {base_url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Get or create session
        sessions_resp = await client.get(f"{base_url}/session")
        sessions_resp.raise_for_status()
        sessions = sessions_resp.json()

        if sessions:
            session_id = sessions[0]["id"]
        else:
            create_resp = await client.post(f"{base_url}/session")
            create_resp.raise_for_status()
            session_id = create_resp.json()["id"]

        print(f"Using session: {session_id}")

        # 2. Track events with timestamps
        events_received: list[tuple[float, str, str]] = []  # (timestamp, type, delta)
        start_time = time.time()

        # 3. Subscribe to SSE in background
        async def collect_events():
            try:
                async with client.stream("GET", f"{base_url}/event") as response:
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            if event_str.startswith("data:"):
                                data_str = event_str[5:].strip()
                                try:
                                    event = json.loads(data_str)
                                    elapsed = time.time() - start_time

                                    if event.get("type") == "message.part.updated":
                                        part = event.get("properties", {}).get("part", {})
                                        delta = event.get("properties", {}).get("delta", "")
                                        if part.get("type") == "text":
                                            events_received.append(
                                                (elapsed, "text-delta", delta or "(no delta)")
                                            )
                                            print(f"  [{elapsed:.2f}s] TEXT: {repr(delta)[:30]}")
                                except json.JSONDecodeError:
                                    pass
            except httpx.ReadError:
                pass  # Connection closed

        # Start event collection
        event_task = asyncio.create_task(collect_events())

        # Wait for SSE connection to establish
        await asyncio.sleep(0.5)

        # 4. Send prompt (API endpoint is /message, not /prompt)
        print("Sending prompt: 'Count from 1 to 5'")
        prompt_start = time.time()

        prompt_resp = await client.post(
            f"{base_url}/session/{session_id}/message",
            json={"parts": [{"type": "text", "text": "Count from 1 to 5, one number per line"}]},
        )
        prompt_resp.raise_for_status()

        prompt_duration = time.time() - prompt_start
        print(f"Prompt completed in {prompt_duration:.2f}s")

        # Wait for events to arrive
        await asyncio.sleep(1.0)

        # Cancel event collection
        event_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await event_task

        # 5. VERIFY: Events should arrive at DIFFERENT timestamps
        print(f"\nReceived {len(events_received)} text-delta events")

        if len(events_received) < 3:
            pytest.fail(f"Expected at least 3 text-delta events, got {len(events_received)}")

        # Extract timestamps
        timestamps = [t for t, _, _ in events_received]

        # Check that timestamps are NOT all the same (which would indicate no streaming)
        unique_timestamps = {round(t, 1) for t in timestamps}  # Round to 0.1s precision

        print(f"Unique timestamp buckets (0.1s precision): {len(unique_timestamps)}")
        print(f"Timestamp range: {min(timestamps):.2f}s to {max(timestamps):.2f}s")

        # CRITICAL ASSERTION: If streaming works, events arrive at different times
        # If NOT streaming, all events would arrive at nearly the same timestamp
        timestamp_spread = max(timestamps) - min(timestamps)

        if timestamp_spread < 0.1:
            pytest.fail(
                f"STREAMING NOT WORKING: All {len(events_received)} events arrived within {timestamp_spread:.3f}s. "
                f"Expected events to be spread over time (streaming). "
                f"Timestamps: {[f'{t:.2f}' for t in timestamps]}"
            )

        print(f"\n✅ STREAMING VERIFIED: Events spread over {timestamp_spread:.2f}s")
        print(f"   First event: {timestamps[0]:.2f}s, Last event: {timestamps[-1]:.2f}s")

        # Additional check: there should be multiple distinct time buckets
        assert len(unique_timestamps) >= 2, (
            f"Expected events at multiple distinct times, got timestamps: {timestamps}"
        )


if __name__ == "__main__":
    asyncio.run(test_streaming_events())
