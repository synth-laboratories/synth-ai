### Supervised Fine-Tuning for Crafter

This folder provides a minimal, reusable SFT workflow that pulls out the SFT step from `examples/warming_up_to_rl/` and focuses it on LoRA/QLoRA. We've also added guidance for running full finetuning (FFT) so you can compare adapters against end-to-end weight updates.

It supports distilling Groq (or other vendor) rollouts into JSONL using tracing and then training a small base model like `Qwen/Qwen3-0.6B`.

---

### 0) Load environment from .env.dev (recommended)

Use your dev env file so keys/URLs are sourced consistently:

```bash
# Example path; update to your actual dev env
set -a && source /Users/joshpurtell/Documents/GitHub/monorepo/backend/.env.dev && set +a
```

This ensures `ENVIRONMENT_API_KEY`, `GROQ_API_KEY`, and (optionally) `BACKEND_BASE_URL` are available to the steps below.

---

### 1) Collect traces and export SFT JSONL

You can generate traces with the Crafter task app and then export them to SFT JSONL using the existing exporter:

```bash
# Serve the task app locally with tracing enabled (example)
uvx synth-ai serve grpo-crafter \
  --trace traces/v3 \
  --trace-db traces/v3/synth_ai.db \
  --port 8001

# Or run traced local rollouts to accumulate data
uv run python examples/warming_up_to_rl/run_local_rollout_traced.py \
  --episodes 50 --max-turns 10

# Export SFT dataset from the trace DB
uv run python examples/warming_up_to_rl/export_trace_sft.py \
  --db traces/v3/synth_ai.db \
  --min-unique 0 \
  --output examples/sft/ft_data/crafter_traces.jsonl
```

Notes:
- The exporter uses achievements and event rewards to filter high-signal steps. Combine `--min-unique`, `--min-outcome-reward`, `--event-reward`, and `--require-achievement` to control data quality.
- You can restrict to sessions from certain providers/models with `--provider`/`--model`.
- Use `--limit` while debugging to reduce dataset size quickly.

---

### 2a) Train LoRA (QLoRA) on Qwen/Qwen3-0.6B

Use the standard CLI. Do not use a custom Python finetuning script. Point the CLI at your `.env.dev` so it picks up keys automatically:

```bash
uvx synth-ai train \
  --type sft \
  --config examples/sft/configs/crafter_lora_qwen0p6b.toml \
  --dataset examples/sft/ft_data/crafter_traces.jsonl \
  --env-file /Users/joshpurtell/Documents/GitHub/monorepo/backend/.env.dev
```

The config sets `training.use_qlora = true` and `hyperparameters.train_kind = "peft"` to request LoRA adapters.

Experiment tips:
- The backend currently defaults to a LoRA rank of 16. If you need other ranks, generate the payload with `--dry-run`, add `"lora_rank": <value>` (and optional `"lora_alpha"`, `"lora_dropout"`) under `hyperparameters`, and submit it via the API until the CLI exposes these knobs directly.
- Duplicate the TOML and adjust `hyperparameters.warmup_ratio`, `learning_rate`, or `gradient_accumulation_steps` to keep the global batch size comparable across datasets.

---

### 2b) Train Full Finetune (FFT) on Qwen/Qwen3-0.6B

Full finetuning updates all weights and uses a near-identical CLI flow with the LoRA toggle disabled. The helper config lives alongside the LoRA sample:

```bash
uvx synth-ai train \
  --type sft \
  --config examples/sft/configs/crafter_fft_qwen0p6b.toml \
  --dataset examples/sft/ft_data/crafter_traces.jsonl \
  --env-file /Users/joshpurtell/Documents/GitHub/monorepo/backend/.env.dev
```

Key differences vs LoRA:
- `training.use_qlora = false` and `hyperparameters.train_kind = "fft"` request a full-weight update.
- `per_device_batch` defaults to 1 to keep memory use comfortable on a single H100; raise gradually as you confirm headroom.
- FFT runs slower per step. Consider trimming the dataset with `--examples` or the exporter filters for quick baselines.

If you want the 4B Crafter FFT baseline from the RL examples, reuse `examples/warming_up_to_rl/configs/crafter_fft_4b.toml` with the same CLI command.

---

### 3) Evaluate the fine-tuned models

After the job completes, list your fine-tuned models and evaluate them in the Crafter loop:

```bash
# List models
uv run python - <<'PY'
import asyncio
import os
from synth_ai.learning.client import LearningClient

backend = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api")
api_key = os.getenv("SYNTH_API_KEY", "")
async def main():
    client = LearningClient(backend, api_key)
    models = await client.list_fine_tuned_models()
    for m in models:
        print(m)
asyncio.run(main())
PY

# Evaluate in the Crafter eval loop (example via warming_up_to_rl)
TASK_APP_URL=http://localhost:8001 \
uv run python examples/warming_up_to_rl/run_eval.py \
  --toml examples/warming_up_to_rl/configs/eval_local_vllm.toml \
  --model ft:YOUR_FT_MODEL_ID \
  --use-rollout
```

---

### 4) Plan comparison runs

Keep runs comparable by adjusting one axis at a time and logging the settings in your experiment tracker (spreadsheet, weights & biases, etc.).

- **LoRA rank sweeps:** start from `crafter_lora_qwen0p6b.toml`, clone it per rank (e.g., `r=4,8,16,64`). For now add the desired `lora_rank` in the job payload manually (see note above) and include it in the run name.
- **Dataset size:** duplicate the exported JSONL and slice with `head -n`, or pass `--examples N` to the CLI for quick subsamples. Track the effective token count using the exporter logs.
- **Data quality:** increase `--min-unique`, require specific achievements, or exclude low-reward sessions with `export_trace_sft.py`. Capture the filter tuple in your run metadata so evaluations stay reproducible.
- **FFT vs LoRA:** run both configs on the same dataset/cardinality so differences reflect the training method rather than the data.

For each sweep, use consistent evaluation seeds and write down throughput (tokens/sec) so you can weigh quality vs cost.

---

### Files
- `configs/crafter_lora_qwen0p6b.toml`: LoRA/QLoRA SFT config for `Qwen/Qwen3-0.6B`.
- `configs/crafter_fft_qwen0p6b.toml`: Full-finetune SFT config for `Qwen/Qwen3-0.6B`.
- `ft_data/`: place your exported JSONL here (ignored by VCS).
