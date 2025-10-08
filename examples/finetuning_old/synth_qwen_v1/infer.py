from __future__ import annotations

import asyncio

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from synth_ai.inference import InferenceClient
from examples.finetuning.synth_qwen_v1.util import load_env, load_state, parse_args


async def _run(mode: str | None) -> None:
    base, key = load_env(mode)
    client = InferenceClient(base_url=base, api_key=key)
    st = load_state()
    model = st.get("fine_tuned_model") or "Qwen/Qwen2.5-0.5B"
    print(f"model={model}")
    resp = await client.create_chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Give me a cheerful two-line greeting."}],
        max_tokens=128,
        temperature=0.7,
        stream=False,
    )
    print(resp)


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args.mode))


if __name__ == "__main__":
    main()
