from __future__ import annotations

import asyncio
from typing import Dict

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from synth_ai.learning import JobHandle
from examples.finetuning.synth_qwen_v1.util import load_env, load_state, save_state, parse_args


def _print_event(e: Dict) -> None:
    try:
        print(f"event seq={e.get('seq')} type={e.get('type')} msg={e.get('message')}")
    except Exception:
        print(str(e))


async def _run(mode: str | None) -> None:
    base, key = load_env(mode)
    st = load_state()
    job_id = st.get("job_id")
    if not job_id:
        raise RuntimeError("state.json missing job_id; run finetune.py first")

    # Use shared JobHandle poller abstraction (strict=True for FT)
    handle = JobHandle(base, key, job_id, strict=True)
    final = await handle.poll_until_terminal(
        interval_seconds=2.0, max_seconds=1800, on_event=_print_event
    )
    print(f"final_status={final.get('status')}")
    ft = final.get("fine_tuned_model")
    if ft:
        save_state({"fine_tuned_model": ft})


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args.mode))


if __name__ == "__main__":
    main()
