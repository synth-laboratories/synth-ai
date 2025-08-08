### Quickstart (Crafter): generate → filter → finetune → run

1) Generate traces (v3 tracing)
```bash
CRAFTER_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
CRAFTER_EPISODES=10 \
CRAFTER_MAX_STEPS=30 \
CRAFTER_DIFFICULTY=easy \
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

2) Filter traces → SFT JSONL
```bash
# Defaults shown; override as needed
CRAFTER_DB_URL="sqlite:///traces/v3/lm_synth/traces.db" \
OUTPUT_JSONL="ft_data/qwen4b_crafter_sft.jsonl" \
MIN_TOTAL_REWARD=1.0 \
MIN_ACHIEVEMENTS=0 \
MAX_COST=10.0 \
MAX_TOKENS=100000 \
MODELS="Qwen/Qwen3-4B-Instruct-2507" \
uvpm examples.finetuning.synth_qwen.filter_traces
```

3) Finetune via learning service (SFT)
```bash
# Prefer hitting the backend proxy (e.g., http://localhost:8000/api or your SYNTH_BASE_URL)
LEARNING_V2_BASE_URL="http://localhost:8000/api" \
SYNTH_API_KEY="sk_live_xxx" \
QWEN_BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
QWEN_TRAINING_JSONL="ft_data/qwen4b_crafter_sft.jsonl" \
uvpm examples.finetuning.synth_qwen.sft_kickoff
# Note the printed fine_tuned_model id (e.g., ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-abc123)
```

4) Run the finetuned model in Crafter
```bash
CRAFTER_MODEL="ft:...your-returned-id..." \
CRAFTER_EPISODES=10 \
CRAFTER_MAX_STEPS=30 \
CRAFTER_DIFFICULTY=easy \
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

---

## Qwen 4B finetuning (Crafter)

This guide shows how to:
- use the Qwen 4B model string from the Crafter runner,
- prepare SFT data from v3 traces,
- kick off an SFT job via the learning service backend APIs,
- and evaluate the resulting adapter.

### Prereqs
- Set base URLs and API key once (prefer the backend proxy over the direct Modal URL):

```bash
# copy and edit locally (do not commit secrets)
cp examples/finetuning/synth_qwen/env.example examples/finetuning/synth_qwen/.env
# then source it in your shell
set -a; source examples/finetuning/synth_qwen/.env; set +a
```

- Recommended directories inside repo root:
  - training output: `ft_data/`
- v3 trace DB: `traces/v3/lm_synth/traces.db`

### 1) Generate trajectories with Qwen 4B
Use the maintained wrapper that forwards into the canonical Crafter runner with v3 tracing.

```bash
# Episodes/steps are small for smoke tests; increase as needed
CRAFTER_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
CRAFTER_EPISODES=10 \
CRAFTER_MAX_STEPS=30 \
CRAFTER_DIFFICULTY=easy \
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```

### 2) Filter v3 traces → SFT JSONL
Use the v3 Turso/SQLite filter to produce OpenAI/Synth-compatible SFT data.

```bash
# Example: filter only Qwen-4B trajectories
uvpm synth_ai.environments.examples.crafter_classic.agent_demos.crafter_modal_ft.filter_traces_sft_turso \
  -d sqlite:///traces/v3/lm_synth/traces.db \
  -o ft_data/qwen4b_crafter_sft.jsonl \
  -c synth_ai/environments/examples/crafter_classic/agent_demos/crafter_modal_ft/filter_config_modal.toml \
  --models "Qwen/Qwen3-4B-Instruct-2507"
```

### 2.5) Filter helper (no CLI)
Alternatively, use the helper in this directory that wraps the same v3 filter logic with env vars:

```bash
# Defaults shown; override as needed
CRAFTER_DB_URL="sqlite:///traces/v3/lm_synth/traces.db" \
OUTPUT_JSONL="ft_data/qwen4b_crafter_sft.jsonl" \
MIN_TOTAL_REWARD=1.0 \
MIN_ACHIEVEMENTS=0 \
MAX_COST=10.0 \
MAX_TOKENS=100000 \
MODELS="Qwen/Qwen3-4B-Instruct-2507" \
uvpm examples.finetuning.synth_qwen.filter_traces
```

### 3) Kick off SFT via learning service (recommended)

```python
# save as scripts/qwen4b_sft.py (example), then run with: uvpm scripts.qwen4b_sft
import asyncio, os, time
from typing import Dict
import aiohttp

API_URL = os.getenv("LEARNING_V2_BASE_URL")
API_KEY = os.getenv("SYNTH_API_KEY")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # model string from the Crafter wrapper
TRAINING_PATH = "ft_data/qwen4b_crafter_sft.jsonl"

async def upload_file() -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with aiohttp.ClientSession() as s:
        data = aiohttp.FormData()
        data.add_field("file", open(TRAINING_PATH, "rb"), filename=os.path.basename(TRAINING_PATH), content_type="application/jsonl")
        data.add_field("purpose", "fine-tune")
        async with s.post(f"{API_URL}/files", data=data, headers=headers) as r:
            assert r.status == 200, await r.text()
            return (await r.json())["id"]

async def create_job(file_id: str) -> str:
    body = {
        "training_file": file_id,
        "model": MODEL,
        "hyperparameters": {"training_type": "sft", "n_epochs": 1, "batch_size": 4},
        "upload_to_wasabi": True,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{API_URL}/fine_tuning/jobs", json=body, headers=headers) as r:
            assert r.status == 200, await r.text()
            return (await r.json())["id"]

async def await_success(job_id: str) -> Dict[str, object]:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with aiohttp.ClientSession() as s:
        check_interval = 15
        for attempt in range(20):
            async with s.get(f"{API_URL}/fine_tuning/jobs/{job_id}", headers=headers) as r:
                if r.status != 200:
                    await asyncio.sleep(check_interval); continue
                j = await r.json()
                status = j.get("status")
                if status == "succeeded":
                    return j
                if status in {"failed", "cancelled"}:
                    raise RuntimeError(f"Training failed: {j.get('error')}")
            await asyncio.sleep(check_interval)
    raise TimeoutError("Training did not finish in time")

async def main() -> None:
    fid = await upload_file()
    job_id = await create_job(fid)
    start = time.time()
    job = await await_success(job_id)
    wall = time.time() - start
    ft_model = job["fine_tuned_model"]
    tokens = job.get("trained_tokens")
    output_dir = job.get("output_dir")
    wasabi_path = job.get("wasabi_path")

    print("🟢 Qwen4B SFT fine-tune succeeded →", ft_model)
    print(f"⏱️ wall-clock: {wall:.1f}s | trained_tokens: {tokens}")
    if output_dir:
        print("📂 adapter saved at:", output_dir)
    print("🪣 adapter uploaded to:", wasabi_path)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uvpm scripts.qwen4b_sft
```

### 4) Evaluate the fine-tuned adapter on Crafter

```bash
CRAFTER_MODEL="ft:...your-returned-id..." \
CRAFTER_EPISODES=10 \
CRAFTER_MAX_STEPS=30 \
CRAFTER_DIFFICULTY=easy \
uvpm examples.finetuning.synth_qwen.run_crafter_qwen4b
```


### Shortcuts (helpers in this directory)

- Kickoff SFT (uses env + JSONL path):
  ```bash
  QWEN_TRAINING_JSONL=ft_data/qwen4b_crafter_sft.jsonl \
  QWEN_BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
  uvpm examples.finetuning.synth_qwen.sft_kickoff
  ```
