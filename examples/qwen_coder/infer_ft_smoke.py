#!/usr/bin/env python3
"""Post-SFT smoke test: read ft_model_id.txt and run a short inference.

Env:
  SYNTH_API_KEY (required)
  BACKEND_BASE_URL (defaults to https://agent-learning.onrender.com/api)

Writes:
  examples/qwen_coder/ft_data/ft_infer_smoke.txt
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from examples.qwen_coder._shared import resolve_infer_output_path, resolve_model_id_path
from synth_ai.inference.client import InferenceClient


def _backend() -> str:
    raw = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api").strip()
    return raw if raw.endswith("/api") else (raw + "/api")


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SYNTH_API_KEY required in environment")

    ft_id_path = resolve_model_id_path(os.getenv("QWEN_CODER_FT_FILENAME", "ft_model_id.txt"))
    if not ft_id_path.exists():
        raise SystemExit(f"Missing {ft_id_path}; run SFT first")
    model_id = ft_id_path.read_text(encoding="utf-8").strip()
    if not model_id:
        raise SystemExit("ft_model_id.txt is empty")

    client = InferenceClient(base_url=_backend(), api_key=api_key, timeout=60.0)

    prompt = os.getenv(
        "PROMPT",
        "Write a Python function to check if a string is a palindrome, then test it.",
    )
    resp: dict[str, Any] = await client.create_chat_completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
        thinking_budget=256,
    )

    # Extract assistant content
    content: str = (
        resp.get("choices", [{}])[0].get("message", {}).get("content") or str(resp)
    )
    out_path = resolve_infer_output_path(os.getenv("QWEN_CODER_FT_INFER_FILENAME", "ft_infer_smoke.txt"))
    out_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (len={len(content)})")


if __name__ == "__main__":
    asyncio.run(main())

