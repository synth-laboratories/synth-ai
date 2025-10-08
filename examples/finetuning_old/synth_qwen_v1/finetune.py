from __future__ import annotations

import asyncio

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from synth_ai.learning import FtClient
from examples.finetuning.synth_qwen_v1.util import load_env, load_state, save_state, parse_args


async def _run(mode: str | None) -> None:
    base, key = load_env(mode)
    client = FtClient(base_url=base, api_key=key)

    st = load_state()
    file_id = st.get("file_id")
    if not file_id:
        raise RuntimeError("state.json missing file_id; run upload_data.py first")

    # Qwen3 0.6B default
    resp = await client.create_sft_job(
        model="Qwen/Qwen3-0.6B",
        training_file_id=file_id,
        hyperparameters={"n_epochs": 1, "batch_size": 4},
        metadata={"upload_to_wasabi": True},
    )
    job_id = resp.get("job_id")
    if not job_id:
        raise RuntimeError(f"create_job missing job_id: {resp}")
    print(f"job_id={job_id}")
    save_state({"job_id": job_id})

    start = await client.start_job(job_id)
    print(f"start={start}")


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args.mode))


if __name__ == "__main__":
    main()
