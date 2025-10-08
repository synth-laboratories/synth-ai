# Warming Up to RL (Crafter)

The Crafter example demonstrates the full Synth AI workflow: task app serving, Groq rollouts, tracing, SFT dataset export, FFT training, evaluation of fine-tuned models, and RL training.

## Quick Reference Commands

- Serve task app locally with tracing:
  ```bash
  uvx synth-ai serve --port 8001 --env-file examples/warming_up_to_rl/.env --trace traces/v3
  ```
- Deploy to Modal:
  ```bash
  uvx synth-ai deploy grpo-crafter --name grpo-crafter-task-app
  ```
- Groq rollout (server-side):
  ```bash
  uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_groq_qwen32b.toml --use-rollout
  ```
- Export SFT data from traced runs:
  ```bash
  python examples/warming_up_to_rl/export_trace_sft.py --db traces/v3/synth_ai.db --output ft_data/crafter_traces.jsonl
  ```
- FFT via CLI:
  ```bash
  uvx synth-ai train --type sft --config examples/warming_up_to_rl/configs/crafter_fft.toml --dataset /absolute/path/to/data.jsonl
  ```
- Evaluate FFT checkpoint:
  ```bash
  uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_fft_qwen4b.toml --use-rollout
  ```
- RL via CLI (FFT-first):
  ```bash
  uvx synth-ai train --type rl --config examples/warming_up_to_rl/configs/rl_from_ft.toml
  ```

---

## 1. Prerequisites

- Python 3.11+
- `uv`/`uvx` available (or install Synth in a virtualenv)
- Modal CLI (`modal token new`) if you plan to deploy the task app
- `.env` in this directory with at least:
  - `SYNTH_API_KEY`
  - `ENVIRONMENT_API_KEY`
  - `TASK_APP_URL` (when running against a hosted task app)
  - Optional: `GROQ_API_KEY`, `OPENAI_API_KEY` for proxy endpoints

`uvx synth-ai setup` can populate the `.env` by guiding you through the dashboard handshake.

> All commands below assume you are running from the repository root unless noted.

## 2. Task App Operations

### Local development

```bash
uvx synth-ai serve --port 8001 --env-file examples/warming_up_to_rl/.env --trace traces/v3 --trace-db traces/v3/synth_ai.db
```

- `--trace` and `--trace-db` enable tracing v3 and SFT JSONL dumps.
- Add `--reload` for uvicorn auto-reload while editing code.

### Modal deploy / serve

```bash
uvx synth-ai deploy grpo-crafter --name grpo-crafter-task-app --env-file examples/warming_up_to_rl/.env
uvx synth-ai modal-serve grpo-crafter --name grpo-crafter-task-app --env-file examples/warming_up_to_rl/.env
```

Both commands preflight the environment key with the backend when `SYNTH_API_KEY` is present.

## 3. Baseline Evaluations (Groq and Synth vLLM)

Evaluation scripts auto-load `.env` values. Update TOMLs under `configs/` with the correct `task_app_url` and provider-specific model names.

- Groq Qwen3-32B:
  ```bash
  uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_groq_qwen32b.toml --use-rollout
  ```
- Synth vLLM Qwen3-4B (Modal-hosted inference URL specified in TOML):
  ```bash
  uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_modal_qwen4b.toml --use-rollout
  ```

`--use-rollout` drives the task app’s `/rollout` endpoint so achievements and metrics are captured. Without it the script issues per-step `initialize/step/terminate` calls.

## 4. Tracing and SFT Dataset Export

1. Serve the task app with tracing enabled (see Section 2). Optionally, run the traced rollout helper against the running server:
   ```bash
   uv run python examples/warming_up_to_rl/run_local_rollout_traced.py \
     --base-url http://localhost:8001 \
     --api-key "$ENVIRONMENT_API_KEY" \
     --inference-api-key "$GROQ_API_KEY" \
     --model qwen/qwen3-32b \
     --inference-url https://api.groq.com/openai \
     --max-llm-calls 3 \
     --run-id local-trace
   ```
2. Inspect local trace databases:
   ```bash
   uvx synth-ai traces --limit 10
   ```
3. Export JSONL suitable for SFT:
   ```bash
   python examples/warming_up_to_rl/export_trace_sft.py \
     --db traces/v3/synth_ai.db \
     --min-achievements 3 \
     --output ft_data/crafter_traces.jsonl
   ```

The exporter enriches each example with achievements unlocked, model metadata, and reward summaries.

## 5. SFT / FFT Training

### Preferred: `uvx synth-ai train`

```bash
uvx synth-ai train \
  --type sft \
  --config examples/warming_up_to_rl/configs/crafter_fft.toml \
  --dataset /absolute/path/to/crafter_traces.jsonl
```

The CLI will:
- Prompt for `.env` selection (or use `--env-file`).
- Upload training (and optional validation) data to `/learning/files`.
- Submit the job and poll until completion unless `--no-poll` is set.

### Legacy script

```bash
uv run python examples/warming_up_to_rl/run_fft_and_save.py \
  --toml examples/warming_up_to_rl/configs/crafter_fft.toml \
  --data /absolute/path/to/crafter_traces.jsonl \
  --poll-seconds 1800
```

The script writes the resulting model ID to `ft_model_id.txt`. Use that ID in evaluation and RL configs (e.g., `model = "ft:abc123"`).

## 6. Evaluate the Fine-tuned Model

After FFT completes, update `configs/eval_fft_qwen4b.toml` so `model = "ft:<model_id>"`, then rerun the evaluation:

```bash
uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_fft_qwen4b.toml --use-rollout
```

This reuses the same Groq/vLLM pipeline but exercises the finetuned checkpoint.

## 7. RL Training

### Preferred: `uvx synth-ai train --type rl`

```bash
uvx synth-ai train \
  --type rl \
  --config examples/warming_up_to_rl/configs/rl_from_base_qwen4b.toml
```

During the interactive setup the CLI ensures `SYNTH_API_KEY`, `ENVIRONMENT_API_KEY`, and `TASK_APP_URL` are present, health-checks the task app, and submits the RL job to `/rl/jobs`.

### Legacy script

```bash
uv run python examples/warming_up_to_rl/run_rl_and_save.py \
  --config examples/warming_up_to_rl/configs/rl_from_ft.toml
```

To start directly from a base model, switch the config to `rl_from_base_qwen4b.toml` and ensure `[model].base` is populated.

## 8. Additional Utilities

- `manage_secrets.py` – convenience helpers for Modal secret management.
- `run_local_rollout.py`, `run_local_rollout_parallel.py`, `run_rollout_remote.py` – alternative rollout launchers for benchmarking.
- `analyze_trace_db.py` – inspect trace quality/achievements before exporting.

Refer to `docs/workflows/` for end-to-end guidance that mirrors these commands.
