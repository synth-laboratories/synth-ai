from __future__ import annotations

from pathlib import Path

import asyncio

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from synth_ai.learning import FtClient, validate_training_jsonl
from examples.finetuning.synth_qwen_v1.util import load_env, save_state, parse_args


async def _run(mode: str | None) -> None:
    base, key = load_env(mode)
    client = FtClient(base_url=base, api_key=key)

    p = Path(__file__).parent / "data" / "training.jsonl"
    # Use shared validator from synth_ai.learning.validators
    validate_training_jsonl(p)
    file_id = await client.upload_training_file(p, purpose="fine-tune")
    print(f"file_id={file_id}")
    save_state({"file_id": file_id})


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args.mode))


if __name__ == "__main__":
    main()
