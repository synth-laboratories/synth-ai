#!/usr/bin/env python3
"""Smoke test: Qwen3 Coder inference via the Synth prod proxy endpoint.

No CLI args. Reads SYNTH_API_KEY from env. Optional overrides via env:
  - BACKEND_BASE_URL (defaults to https://agent-learning.onrender.com/api)
  - MODEL (defaults to Qwen/Qwen3-Coder-30B-A3B-Instruct)
  - PROMPT (defaults to a simple coding prompt)

Run:
  SYNTH_API_KEY=sk_... uv run python examples/qwen_coder/infer_prod_proxy.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx


def _backend_root() -> str:
    raw = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api").strip()
    if raw.endswith("/api"):
        raw = raw[:-4]
    return raw.rstrip("/")


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SYNTH_API_KEY required in environment")

    model = os.getenv("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    prompt = os.getenv(
        "PROMPT",
        "Write a Python function to reverse a string, then show an example call.",
    )

    # Prod proxy endpoint
    url = f"{_backend_root()}/api/inference/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 256,
        "thinking_budget": 256,
    }

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        # Print assistant content (compact)
        try:
            msg = data.get("choices", [{}])[0].get("message", {})
            print(msg.get("content") or data)
        except Exception:
            print(data)


if __name__ == "__main__":
    asyncio.run(main())


