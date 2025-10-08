#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import sys

# Ensure repo root on path
ROOT = Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.finetuning.synth_qwen_v1.util import load_env, load_state  # type: ignore
from synth_ai.inference import InferenceClient  # type: ignore


async def main() -> None:
    base_url, api_key = load_env(os.getenv("MODE") or os.getenv("ENV") or "local")
    st_path = Path(__file__).parent / "state.json"
    if not st_path.exists():
        raise FileNotFoundError(f"state.json not found at {st_path}")
    state = json.loads(st_path.read_text())
    model = state.get("fine_tuned_model")
    if not model:
        raise RuntimeError("fine_tuned_model missing in state.json")

    print(f"Backend: {base_url}")
    print(f"Model: {model}")

    ic = InferenceClient(base_url=base_url, api_key=api_key)
    try:
        resp = await ic.create_chat_completion(
            model=model,
            messages=[{"role": "user", "content": "Hello world!"}],
            max_tokens=64,
            temperature=0.2,
            stream=False,
        )
        print("\n===== Response =====")
        print(json.dumps(resp, indent=2))
        print("===== End Response =====\n")
    except Exception as e:  # always print full failure context
        import traceback

        print("\n===== Inference Error =====")
        print(f"Type: {type(e).__name__}")
        print(f"Repr: {repr(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        try:
            from synth_ai.http import HTTPError  # type: ignore

            if isinstance(e, HTTPError):
                print("HTTPError details:")
                print(f"  status={e.status}")
                print(f"  url={e.url}")
                print(f"  message={e.message}")
                if getattr(e, "detail", None) is not None:
                    print(f"  detail={e.detail}")
                if getattr(e, "body_snippet", None):
                    print(f"  body_snippet={e.body_snippet}")
        except Exception:
            pass
        print("===== End Inference Error =====\n")


if __name__ == "__main__":
    asyncio.run(main())
