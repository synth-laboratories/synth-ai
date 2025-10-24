# Warming Up to RL (Crafter)

This folder contains an end-to-end Crafter workflow: stand up the task app, collect Groq-powered rollouts, export tracing data for supervised fine-tuning, run FFT/RL jobs, and evaluate checkpoints. Commands assume the repository root as the working directory unless stated otherwise.

## 1. Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) / `uvx` (or install `synth-ai` inside a virtualenv)
- Modal CLI (`modal token new`) if you plan to deploy the task app remotely
- API keys:
  - `SYNTH_API_KEY` and `ENVIRONMENT_API_KEY` are required for CLI flows
  - `GROQ_API_KEY` (used by the Groq policy) and optional `OPENAI_API_KEY`
- Run `uvx synth-ai setup` once to pair with the Synth dashboard and populate `~/.synth-ai/user_config.json`

## 2. Task App

### Local serve (FastAPI)

```bash
uvx synth-ai serve \
  --env-file examples/warming_up_to_rl/.env \
  --host 127.0.0.1 --port 8001 \
  --trace traces/v3
```

- `--trace` creates/uses `traces/v3/task_app_traces_<timestamp>.db` for the lifetime of the server. All rollouts append to this file.
- Add `--trace-db` to override the SQLite path (one DB per server instance).
- Pass `--reload` during development for auto-reload.

### Modal deploy / serve

```bash
uvx synth-ai deploy grpo-crafter --name grpo-crafter-task-app
uvx synth-ai modal-serve grpo-crafter --name grpo-crafter-task-app
```

Both commands reuse the same tracing defaults; the backend persists rollouts into the configured SQLite/Turso store.

## 3. Collect rollouts

Hit the running task app with the local helper to gather a traced rollout (Groq policy shown below):

```bash
python examples/warming_up_to_rl/run_local_rollout_traced.py \
  --base-url http://localhost:8001 \
  --api-key "$ENVIRONMENT_API_KEY" \
  --inference-api-key "$GROQ_API_KEY" \
  --model qwen/qwen3-32b \
  --inference-url https://api.groq.com/openai \
  --max-llm-calls 3 \
  --run-id local-trace
```

Artifacts produced per rollout:
- `traces/v3/task_app_traces_<timestamp>.db`: the task app’s append-only database (one per server lifetime; new rollouts append rows).
- `local-trace_trace.json`: single-run JSON snapshot for inspection.

## 4. Export SFT-ready data

```bash
python examples/warming_up_to_rl/export_trace_sft.py
```

- When run without `--in`, the script lists every `task_app_traces*.db` under the current directory (and subdirectories), sorted by recency, and prompts you to pick one (the newest is marked `← most recent`).
- The exporter validates the trace data, filters sessions, and writes JSONL to `ft_data/crafter_sft.jsonl` by default (override with `--out`).

## 5. FFT / SFT Training

Recommended via CLI:

```bash
uvx synth-ai train \
  --type sft \
  --config examples/warming_up_to_rl/configs/crafter_fft.toml \
  --dataset /absolute/path/to/crafter_sft.jsonl
```

The CLI uploads training data, submits the job to the Synth backend, and polls for completion. A legacy helper (`run_fft_and_save.py`) is still provided for ad-hoc usage.

## 6. Evaluate checkpoints

Update the relevant TOML with the model identifier (e.g., `model = "ft:<model_id>"`) and run:

```bash
uv run python examples/warming_up_to_rl/run_eval.py \
  --toml examples/warming_up_to_rl/configs/eval_fft_qwen4b.toml \
  --use-rollout
```

`--use-rollout` exercises the `/rollout` endpoint so achievements/rewards are surfaced in traces.

## 7. RL Training

```bash
uvx synth-ai train \
  --type rl \
  --config examples/warming_up_to_rl/configs/rl_from_base_qwen4b.toml
```

Start from `rl_from_ft.toml` if you want to bootstrap from a previously fine-tuned checkpoint.

---

### Notes on tracing

- **One SQLite DB per server:** every task app instance maintains a single `task_app_traces_<timestamp>.db` and appends each new rollout. If you want a fresh file, start another `synth-ai serve` with a different `--trace-db` path.
- **JSON snapshots per run:** `run_local_rollout_traced.py` writes `<run_id>_trace.json` so you can inspect or hand-edit individual runs.
- **Exporter discovery:** the SFT exporter recursively catalogs all `task_app_traces*.db` files beneath the task app directory, allowing you to select any historical snapshot when exporting training data.

These conventions keep tracing predictable: continuous history per server, easy selection of historical DBs, and one-off JSON exports for quick analysis.
