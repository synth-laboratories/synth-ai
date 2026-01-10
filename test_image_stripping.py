#!/usr/bin/env python3
"""Test that image stripping works correctly on actual trace data."""

import json
from pathlib import Path


def strip_message_content(content):
    """Strip images from message content (list of parts)."""
    if not isinstance(content, list):
        return content, 0

    new_content = []
    stripped_count = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            img_url = part.get("image_url", {})
            if isinstance(img_url, dict):
                url = img_url.get("url", "")
            else:
                url = str(img_url)
            url_len = len(url) if url else 0
            if url_len > 1000:
                print(f"[STRIP] Stripping image ({url_len:,} bytes)")
            new_content.append({"type": "text", "text": "[IMAGE]"})
            stripped_count += 1
        else:
            new_content.append(part)
    return new_content, stripped_count


def strip_images_from_trace(trace_dict: dict) -> dict:
    """Remove base64 image content from trace."""
    images_stripped = 0

    # Strip from session_time_steps (V3 format)
    session_time_steps = trace_dict.get("session_time_steps", [])
    if isinstance(session_time_steps, list):
        print(f"[STRIP] Processing {len(session_time_steps)} session_time_steps")
        for step_idx, step in enumerate(session_time_steps):
            if not isinstance(step, dict):
                continue
            events = step.get("events", [])
            if not isinstance(events, list):
                continue
            for event_idx, event in enumerate(events):
                if not isinstance(event, dict):
                    continue

                # Strip from llm_request
                llm_req = event.get("llm_request", {})
                if isinstance(llm_req, dict):
                    messages = llm_req.get("messages", [])
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, dict):
                                content = msg.get("content")
                                new_content, count = strip_message_content(content)
                                if count > 0:
                                    msg["content"] = new_content
                                    images_stripped += count

                # Strip from llm_response.message
                llm_resp = event.get("llm_response", {})
                if isinstance(llm_resp, dict):
                    message = llm_resp.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content")
                        new_content, count = strip_message_content(content)
                        if count > 0:
                            message["content"] = new_content
                            images_stripped += count

    # Strip from event_history (legacy format)
    event_history = trace_dict.get("event_history", [])
    if isinstance(event_history, list):
        print(f"[STRIP] Processing {len(event_history)} event_history events")
        for event in event_history:
            if not isinstance(event, dict):
                continue

            # Strip from llm_request
            llm_req = event.get("llm_request", {})
            if isinstance(llm_req, dict):
                messages = llm_req.get("messages", [])
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            new_content, count = strip_message_content(content)
                            if count > 0:
                                msg["content"] = new_content
                                images_stripped += count

            # Strip from llm_response (direct message)
            llm_resp = event.get("llm_response", {})
            if isinstance(llm_resp, dict):
                if "message" in llm_resp:
                    message = llm_resp.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content")
                        new_content, count = strip_message_content(content)
                        if count > 0:
                            message["content"] = new_content
                            images_stripped += count

    print(f"[STRIP] Total images stripped: {images_stripped}")
    return trace_dict


if __name__ == "__main__":
    trace_file = Path("/tmp/captured_trace.json")
    if not trace_file.exists():
        print(f"ERROR: No trace file at {trace_file}")
        exit(1)

    # Load and analyze
    print(f"\n{'=' * 80}")
    print("LOADING TRACE")
    print(f"{'=' * 80}")

    data = json.loads(trace_file.read_text())
    trace = data.get("trace", {})

    # Size before stripping
    trace_before = json.dumps(trace)
    size_before = len(trace_before)
    print(f"Trace size BEFORE stripping: {size_before:,} bytes (~{size_before // 4:,} tokens)")

    # Strip images
    print(f"\n{'=' * 80}")
    print("STRIPPING IMAGES")
    print(f"{'=' * 80}")
    trace_stripped = strip_images_from_trace(trace)

    # Size after stripping
    trace_after = json.dumps(trace_stripped)
    size_after = len(trace_after)
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Trace size AFTER stripping:  {size_after:,} bytes (~{size_after // 4:,} tokens)")
    print(
        f"Reduction: {size_before - size_after:,} bytes ({(1 - size_after / size_before) * 100:.1f}%)"
    )
    print("Gemini limit: 1,048,576 tokens")

    if size_after // 4 > 1_048_576:
        print(f"⚠️  STILL EXCEEDS LIMIT by {size_after // 4 - 1_048_576:,} tokens")
    else:
        print(f"✓ Within limit ({1_048_576 - size_after // 4:,} tokens remaining)")
