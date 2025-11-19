from typing import Dict, List, Tuple

import requests


def collect_sse_events(resp: requests.Response) -> Tuple[List[Dict[str, str]], str]:
    """Read entire SSE response into a list of events."""
    chunks: List[bytes] = []
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            chunks.append(chunk)
    payload = b"".join(chunks).decode("utf-8", errors="ignore")

    events: List[Dict[str, str]] = []
    current_event: Dict[str, str] = {"event": "message", "data": ""}

    for line in payload.splitlines():
        if not line:
            if current_event.get("data"):
                events.append(current_event.copy())
            current_event = {"event": "message", "data": ""}
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event: "):
            current_event["event"] = line[7:].strip()
        elif line.startswith("data: "):
            data_line = line[6:]
            if current_event["data"]:
                current_event["data"] += "\n" + data_line
            else:
                current_event["data"] = data_line

    if current_event.get("data"):
        events.append(current_event.copy())

    return events, payload


def assert_usage_info(usage: Dict[str, int], context: str) -> None:
    assert isinstance(usage, dict), f"Usage must be dict for {context}, got {type(usage)}"
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert key in usage, f"Usage missing '{key}' for {context}"
