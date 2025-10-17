#!/usr/bin/env python3
"""One-shot inference for Qwen3 (and Coder) models via the Synth backend proxy.

Usage examples:

  SYNTH_API_KEY=sk_... BACKEND_BASE_URL=https://agent-learning.onrender.com/api \
  uv run python examples/qwen_coder/infer_via_synth.py \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --prompt "Write a Python function to reverse a string." \
    --max-tokens 128 --temperature 0.2

Optionally you can point to a specific inference host (e.g., your vLLM or task-app proxy):

  ... infer_via_synth.py --inference-url https://your-host/api/inference

The script defaults the backend base URL to the hosted service if BACKEND_BASE_URL is not set.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from synth_ai.inference.client import InferenceClient


def _default_backend() -> str:
    raw = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api").strip()
    return raw if raw.endswith("/api") else (raw + "/api")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=os.getenv("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        help="Base or ft:<id> model identifier",
    )
    p.add_argument(
        "--prompt",
        default="Write a Python function to reverse a string.",
        help="User prompt text",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--inference-url",
        default=os.getenv("INFERENCE_URL"),
        help="Optional backend inference base (e.g., https://host/api/inference)",
    )
    p.add_argument(
        "--timeout", type=float, default=60.0, help="HTTP timeout seconds for backend calls"
    )
    args = p.parse_args()

    backend = _default_backend()
    api_key = os.getenv("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SYNTH_API_KEY required (export it or pass via env-file to uvx)")

    client = InferenceClient(base_url=backend, api_key=api_key, timeout=args.timeout)

    body: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }
    if args.inference_url:
        # Backend supports forwarding to a specific host when provided
        body["inference_url"] = str(args.inference_url)

    resp = await client.create_chat_completion(**body)
    try:
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content")
        print(content or resp)
    except Exception:
        print(resp)


if __name__ == "__main__":
    asyncio.run(main())


