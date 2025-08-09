#!/usr/bin/env python3
"""
Qwen3 tokens/sec probe across 4B, 8B, 14B on Synth backend.

Env:
- SYNTH_API_KEY
- SYNTH_BASE_URL (should include /api) or SYNTH_API_URL (we will append /api)

Run:
- uv run python synth-ai/tests/qwen3_toksec.py
"""

import asyncio
import os
import time
from statistics import mean

import httpx
from openai import AsyncOpenAI


def _resolve_base_url() -> str:
    base = os.getenv("SYNTH_BASE_URL")
    if not base:
        api = os.getenv("SYNTH_API_URL", "")
        if not api:
            raise RuntimeError("SYNTH_BASE_URL or SYNTH_API_URL must be set")
        base = api.rstrip("/")
        if not base.endswith("/api"):
            base = base + "/api"
    return base


async def run_once(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int = 512, gpu_pref: str = "h100") -> float | None:
    start = time.time()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.5,
        extra_headers={"X-GPU-Preference": gpu_pref},
    )
    elapsed = max(1e-6, time.time() - start)
    usage = getattr(resp, "usage", None)
    comp = getattr(usage, "completion_tokens", None) if usage else None
    if comp is None:
        return None
    tps = float(comp) / elapsed
    print(f"model={model} tokens={comp} input={getattr(usage, 'prompt_tokens', None)} tps={tps:.2f}")
    return tps


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY must be set")
    base_url = _resolve_base_url()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers={"X-GPU-Preference": "h100"})

    models = [
        os.getenv("QWEN3_0.6B_MODEL", "Qwen/Qwen3-0.6B"),
        os.getenv("QWEN3_8B_MODEL", "Qwen/Qwen3-8B"),
        os.getenv("QWEN3_14B_MODEL", "Qwen/Qwen3-14B"),
    ]

    print("\n🔧 Qwen3 tokens/sec probe (Synth)")
    print("=" * 40)
    print(f"Base URL: {base_url}")
    print()

    prompt = (
        "Write a short 6-8 sentence overview of the benefits of unit testing in Python. "
        "Avoid bullet points."
    )

    async def warmup(model_id: str, gpu: str = "h100") -> None:
        url = f"{base_url}/warmup/{model_id}?gpu={gpu}"
        headers = {"authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=120.0) as http:
                r = await http.post(url, headers=headers)
            if r.status_code == 200:
                print(f"🔥 warmed {model_id} on {gpu}")
            else:
                print(f"⚠️ warmup failed {model_id} on {gpu}: {r.status_code}")
        except Exception as e:
            print(f"⚠️ warmup error {model_id} on {gpu}: {e}")

    for model in models:
        print(f"\n🧪 {model}")
        # Warmup model on H100 to avoid cold start
        await warmup(model, gpu="h100")
        per_model_tps: list[float] = []
        # Run 3 trials to smooth variance
        for _ in range(3):
            tps = await run_once(client, model, prompt, max_tokens=512, gpu_pref="h100")
            if tps is not None:
                per_model_tps.append(tps)
        if per_model_tps:
            per_model_tps.sort()
            avg = mean(per_model_tps)
            p90 = per_model_tps[max(0, min(len(per_model_tps) - 1, int(0.9 * len(per_model_tps)) - 1))]
            print(
                f"summary model={model} count={len(per_model_tps)} avg={avg:.2f} "
                f"p90={p90:.2f} min={per_model_tps[0]:.2f} max={per_model_tps[-1]:.2f}"
            )
        else:
            print(f"summary model={model} no-usage")


if __name__ == "__main__":
    asyncio.run(main())

