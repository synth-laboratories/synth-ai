#!/usr/bin/env python3
"""Quick script to capture and inspect trace structure."""

import json
from pathlib import Path

# Add instrumentation to capture traces
import httpx

# Monkey-patch httpx to capture the verifier request
original_post = httpx.AsyncClient.post
captured_payload = None


async def capturing_post(self, url, **kwargs):
    global captured_payload
    if "verifiers/completions" in str(url):
        captured_payload = kwargs.get("json", {})
        print(f"\n{'=' * 80}")
        print("CAPTURED VERIFIER REQUEST")
        print(f"{'=' * 80}")

        trace = captured_payload.get("trace", {})
        event_history = trace.get("event_history", [])

        print(f"\nTrace has {len(event_history)} events")

        for i, event in enumerate(event_history):
            print(f"\n--- Event {i} ---")
            print(f"Event keys: {list(event.keys())}")

            # Check llm_request
            llm_req = event.get("llm_request", {})
            if llm_req:
                print(f"  llm_request keys: {list(llm_req.keys())}")
                messages = llm_req.get("messages", [])
                print(f"  llm_request has {len(messages)} messages")
                for j, msg in enumerate(messages):
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        content_type = type(content).__name__
                        if isinstance(content, str):
                            print(
                                f"    msg[{j}]: role={msg.get('role')}, content type=str, len={len(content)}"
                            )
                            if "data:image" in content[:200]:
                                print("      ⚠️  STRING CONTAINS BASE64 IMAGE!")
                        elif isinstance(content, list):
                            print(
                                f"    msg[{j}]: role={msg.get('role')}, content type=list, len={len(content)}"
                            )
                            for k, part in enumerate(content):
                                if isinstance(part, dict):
                                    part_type = part.get("type", "unknown")
                                    print(f"      part[{k}]: type={part_type}")
                                    if part_type == "image_url":
                                        img_url = part.get("image_url", {})
                                        if isinstance(img_url, dict):
                                            url = img_url.get("url", "")
                                        else:
                                            url = str(img_url)
                                        print(f"        image_url len: {len(url) if url else 0}")

            # Check llm_response
            llm_resp = event.get("llm_response", {})
            if llm_resp:
                print(f"  llm_response keys: {list(llm_resp.keys())}")
                choices = llm_resp.get("choices", [])
                print(f"  llm_response has {len(choices)} choices")
                for j, choice in enumerate(choices):
                    if isinstance(choice, dict):
                        message = choice.get("message", {})
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str):
                                print(f"    choice[{j}]: content type=str, len={len(content)}")
                                if "data:image" in content[:200]:
                                    print("      ⚠️  STRING CONTAINS BASE64 IMAGE!")
                            elif isinstance(content, list):
                                print(f"    choice[{j}]: content type=list, len={len(content)}")
                                for k, part in enumerate(content):
                                    if isinstance(part, dict):
                                        part_type = part.get("type", "unknown")
                                        print(f"      part[{k}]: type={part_type}")
                                        if part_type == "image_url":
                                            img_url = part.get("image_url", {})
                                            if isinstance(img_url, dict):
                                                url = img_url.get("url", "")
                                            else:
                                                url = str(img_url)
                                            print(
                                                f"        image_url len: {len(url) if url else 0}"
                                            )

        # Calculate total size
        trace_str = json.dumps(trace)
        print(f"\n{'=' * 80}")
        print(f"TOTAL TRACE SIZE: {len(trace_str):,} bytes ({len(trace_str) / 1024 / 1024:.2f} MB)")
        print(f"{'=' * 80}\n")

        # Save to file
        debug_file = Path("/tmp/captured_trace.json")
        debug_file.write_text(json.dumps(captured_payload, indent=2))
        print(f"Saved full payload to: {debug_file}")

    return await original_post(self, url, **kwargs)


httpx.AsyncClient.post = capturing_post

# Now run the demo
if __name__ == "__main__":
    import subprocess

    subprocess.run(["uv", "run", "python", "demos/web-design/run_demo.py", "--local"])
