# Reinforcement Learning Workflow (FFT-first)

The RL pipeline assumes you start from a finetuned checkpoint (FFT or QLoRA) and continue training with Synth’s RL backend. The reference scripts live alongside the Crafter example.

## Legacy script

```bash
BACKEND_BASE_URL=https://<backend>/api \
SYNTH_API_KEY=<key> \
TASK_APP_URL=https://<task-app>.modal.run \
uv run python examples/warming_up_to_rl/run_rl_and_save.py \
  --config examples/warming_up_to_rl/configs/rl_from_ft.toml
```

Key behaviour (`run_rl_and_save.py`):
- Loads TOML config and validates `[model]` (exactly one of `source` or `base`).
- Resolves the task app URL from CLI, env var, or `[services].task_url`.
- Submits an RL job to `/rl/jobs` with a payload that embeds the entire TOML under `data.config` for reproducibility.

## `uvx synth-ai train --type rl`

The train CLI modernises the flow with guardrails (`synth_ai/api/train/cli.py:95`).

```bash
uvx synth-ai train --type rl --config examples/warming_up_to_rl/configs/rl_from_base_qwen4b.toml
```

The CLI will:
1. Prompt for an `.env` that contains `SYNTH_API_KEY`, `ENVIRONMENT_API_KEY`, and `TASK_APP_URL` (or guide you to discover them via Modal).
2. Health-check the task app (`GET /health`, `/task_info`) using the resolved environment API key.
3. Build the RL job payload (`build_rl_payload`) with overrides for `--task-url`, `--model`, and backend metadata when provided.
4. Submit the job and poll status (`RLJobPoller`) until terminal.

### Config highlights

- `[services].task_url` – default task app endpoint (overridden by CLI).
- `[model].source = "ft:…"` – resume from an FFT model ID.
- `[model].base = "Qwen/Qwen3-4B"` – start from a base model.
- `[training]` – RL-specific parameters (batch sizes, rollout configs embedded in the job payload).
- `[tags]` – annotate jobs for later retrieval (optional).

### Seeds & rollouts

- The task app controls seeds through its rollout executor; update `[data].seeds` in your config if you need deterministic runs.
- Ensure the task app advertises the correct rubric and dataset metadata so RL metrics (outcome/events scores) are meaningful.

## Roadmap

- **Math FFT-first RL**: migrate the public RL example to train on the Math task app using FFT outputs instead of Crafter. The docs will be updated with math-specific configs once they land in `examples/rl`.
- **Unified CLI**: once legacy scripts are deprecated, cross-link only the `train` command to avoid duplicate instructions.

