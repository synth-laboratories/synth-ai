Qwen3 Coder – SFT with LoRA (all linear)

This example mirrors the SFT LoRA flow under `examples/sft/` but targets the smallest Qwen3 Coder family model supported downstream. It configures LoRA on all linear projections ("all-linear") to match our RL LoRA recipes.

Quick start

1) Generate a tiny synthetic dataset (or export your own)

```
uv run python examples/qwen_coder/generate_dataset.py \
  --output examples/qwen_coder/ft_data/coder_sft.small.jsonl \
  --n 50 --seed 42 --lang python
```

2) Run training via the CLI:

```
uvx synth-ai train \
  --type sft \
  --config examples/qwen_coder/configs/coder_lora_small.toml \
  --dataset examples/qwen_coder/ft_data/coder_sft.small.jsonl \
  --env-file /path/to/.env
```

3) Inference via Synth API (pre/post SFT)

Use the SDK’s OpenAI-compatible chat client routed through Synth. Export your env with SYNTH_API_KEY (and optional BACKEND_BASE_URL) or pass an env file to CLI helpers.

Minimal one-shot inference:

```bash
python - <<'PY'
import os, asyncio
from synth_ai.v0.lm.core import main_v3 as lm

async def run():
    model = os.getenv("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    resp = await lm.chat_async(
        model,
        messages=[{"role":"user","content":"Write a Python function to reverse a string."}],
        max_tokens=128,
        temperature=0.2,
    )
    print(resp["choices"][0]["message"]["content"]) 
asyncio.run(run())
PY
```

After training, set `MODEL=ft:...` to query the finetuned adapter.

4) 30B LoRA variant

```bash
uvx synth-ai train \
  --type sft \
  --config examples/qwen_coder/configs/coder_lora_30b.toml \
  --dataset examples/qwen_coder/ft_data/coder_sft.small.jsonl \
  --env-file /path/to/.env
```

5) Faster iteration: 4B LoRA config

```bash
uvx synth-ai train \
  --type sft \
  --config examples/qwen_coder/configs/coder_lora_4b.toml \
  --dataset examples/qwen_coder/ft_data/coder_sft.small.jsonl \
  --env-file /path/to/.env
```

Environment variables

- `SYNTH_API_KEY`: required for training/inference through the hosted backend
- `BACKEND_BASE_URL`: defaults to `https://agent-learning.onrender.com/api`

Post‑SFT smoke

- The training helper `sft_lora_30b.py` writes the resulting `ft:<id>` to `examples/qwen_coder/ft_data/ft_model_id.txt`.
- Validate inference with your finetuned adapter:

```bash
uv run python examples/qwen_coder/infer_ft_smoke.py
```

Dataset utilities

- `examples/qwen_coder/validate_jsonl.py`: sanity‑check first N lines for chat structure
- `examples/qwen_coder/subset_jsonl.py`: create a capped subset for quick tests

Optional CLI wrappers

- `examples/qwen_coder/scripts/train_coder_30b.sh [/path/to/.env]`
- `examples/qwen_coder/scripts/infer_coder.sh [/path/to/.env]`

Notes

- LoRA is enabled with `training.mode = "lora"` and `hyperparameters.train_kind = "peft"`.
- The config sets an `all-linear` target to apply adapters broadly across attention and MLP projections.
- Adjust `gradient_accumulation_steps`, `per_device_batch`, and `sequence_length` based on available GPU memory.
- Use the Synth API client (above) for inference to ensure requests route via the hosted backend.


