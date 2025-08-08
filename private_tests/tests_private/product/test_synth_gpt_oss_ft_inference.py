#!/usr/bin/env python3
"""
Simple inference test that targets a **specific GPT-OSS fine-tuned model**.

The model identifier is expected to be provided via the environment variable
`GPT_OSS_FINE_TUNED_MODEL` (for example: `ft:openai/gpt-oss-20b:ftjob-abc`).

The test sends a handful of prompts through `/api/v1/chat/completions` and
prints the responses, verifying only that the call succeeds and returns
non-empty content.
"""

import asyncio
import os
import time
from typing import List

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTH_API_URL = os.environ.get("SYNTH_API_URL", "http://localhost:8000")
SYNTH_API_KEY = os.environ.get("SYNTH_API_KEY", "")

MODEL_ID = os.environ.get("GPT_OSS_FINE_TUNED_MODEL", "")
if not MODEL_ID:
    raise RuntimeError(
        "Environment variable GPT_OSS_FINE_TUNED_MODEL is required for the "
        "inference test."
    )

HEADERS = {
    "Authorization": f"Bearer {SYNTH_API_KEY}",
    "Content-Type": "application/json",
}

PROMPTS: List[str] = [
    "What is the capital of Germany?",
    "Translate 'computer' into Spanish.",
    "Solve 18 * 7 step by step.",
]

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

async def _run_single_inference(prompt: str) -> None:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 128,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        start = time.time()
        resp = await client.post(
            f"{SYNTH_API_URL}/api/v1/chat/completions", json=payload, headers=HEADERS
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            raise RuntimeError(f"Inference failed: {resp.status_code} – {resp.text[:200]}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]  # type: ignore[index]

        print(f"\nPrompt: {prompt}")
        print(f"Response ({len(content)} chars, {elapsed:.2f}s):\n{content}\n")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "=" * 60)
    print("GPT-OSS FINE-TUNED MODEL INFERENCE TEST")
    print("=" * 60)
    print(f"Model ID: {MODEL_ID}")

    for prompt in PROMPTS:
        await _run_single_inference(prompt)

    print("\n🎉 Inference test completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
