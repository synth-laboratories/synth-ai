# Math RL Demo (Single Step)

This example trains a reinforcement learning policy on single-step math problems sourced from the [EleutherAI/math](https://huggingface.co/datasets/EleutherAI/math) dataset. Episodes consist of a single tool call: the model must emit a `math_submit` function call whose `answer` field contains the final solution. Missing or malformed tool calls receive negative reward; correct answers earn positive reward.

## Quick Commands

```bash
# Serve locally with tracing
uvx synth-ai serve math-single-step --port 8101 --env-file examples/rl/.env --trace traces/math

# Modal deployment
uvx synth-ai deploy --name synth-math-single-step --env-file examples/rl/.env

# Evaluate base Qwen policy (validation split)
uv run python examples/rl/run_eval.py --toml examples/rl/configs/eval_base_qwen.toml

# Launch RL job from base model
uvx synth-ai train --type rl --config examples/rl/configs/rl_from_base_qwen.toml

# Evaluate RL checkpoint on held-out test split
uv run python examples/rl/run_eval.py --toml examples/rl/configs/eval_rl_qwen.toml
```

## 1. Prerequisites

- Python 3.11+
- `uv`/`uvx`
- Modal CLI (`modal token new`) for deployment
- `.env` at `examples/rl/.env` containing at least:
  - `SYNTH_API_KEY`
  - `ENVIRONMENT_API_KEY`
  - Optional: `TASK_APP_URL` (Modal URL), `GROQ_API_KEY`, `OPENAI_API_KEY`

Run `uvx synth-ai setup` to populate the `.env` if you have not paired the SDK before.

## 2. Task App

The task app is defined in `synth_ai/task/apps/math_single_step.py` and registered as `math-single-step`. It loads problems from the Hugging Face dataset (configurable via `MATH_DATASET_*` env vars) and manages per-episode state with an in-memory environment manager.

- **Observation**: single math problem (string) plus dataset metadata.
- **Actions**: exactly one `math_submit` tool call with an `answer` string.
- **Rewards**:
  - `+1.0` for correct answer
  - `0.0` for incorrect answer
  - `-0.5` if the tool call omits an answer or uses the wrong tool
  - `-1.0` when no tool call is provided

Serve locally with tracing to capture trajectories:

```bash
uvx synth-ai serve math-single-step \
  --port 8101 \
  --env-file examples/rl/.env \
  --trace traces/math \
  --trace-db traces/math/synth_ai.db
```

Deploy or serve on Modal using the same env file; the registration includes a `ModalDeploymentConfig` that installs the `datasets` package automatically.

## 3. Evaluation

`examples/rl/run_eval.py` evaluates a policy by sampling deterministic seeds from the dataset splits. TOML configuration controls the model, split, and number of episodes. Example config (`eval_base_qwen.toml`):

```toml
provider = "synth"
task_app_url = "http://localhost:8101"
model = "Qwen/Qwen3-4B"
split = "validation"
num_episodes = 50
seed_start = 0

[policy]
inference_url = "http://localhost:8000/api/inference"
max_tokens = 128
temperature = 0.0
# Optional: override headers for inference requests
# [policy.extra_headers]
# Authorization = "Bearer ..."
```

The `[policy]` table maps directly to the inference payload; add `[policy.headers]` if you need to forward custom HTTP headers (e.g., `Authorization`). If `SYNTH_API_KEY` is present, the evaluator automatically sends `Authorization: Bearer <key>`.

Set `--use-rollout` to exercise the server-side rollout endpoint instead of the per-step API.

The script reports accuracy and a breakdown of failure modes (`missing_tool_call`, `blank_answer`, etc.).

## 4. RL Training

Example RL config (`configs/rl_from_base_qwen.toml`):

```toml
[services]
task_url = "https://your-app.modal.run"

[model]
base = "Qwen/Qwen3-4B"

[data]
split = "train"
seed_start = 0
episodes_per_iteration = 2048

[training]
max_turns = 1
ops = ["agent", "env"]
batch_size = 128
group_size = 1024
reward_positive = 1.0
reward_negative_no_tool = -1.0
reward_negative_no_answer = -0.5

[policy]
model = "Qwen/Qwen3-4B"
inference_url = "https://your-inference-host"
max_tokens = 128
temperature = 0.0

[tags]
experiment = "math_single_step"
```

Submit jobs interactively with:

```bash
uvx synth-ai train --type rl --config examples/rl/configs/rl_from_base_qwen.toml
```

The CLI ensures the task app is reachable (`/health`, `/task_info`), prompts for missing secrets, and polls job status until completion. For scripted automation, use `run_rl_and_save.py`:

```bash
uv run python examples/rl/run_rl_and_save.py \
  --config examples/rl/configs/rl_from_base_qwen.toml \
  --backend https://backend.synth.ai/api
```

## 5. Evaluating RL Outputs

After training completes, set `model = "rl:<job_or_model_id>"` in `configs/eval_rl_qwen.toml` (and update `split = "test"` for a held-out set). Re-run `run_eval.py` to compare:

```bash
uv run python examples/rl/run_eval.py --toml examples/rl/configs/eval_rl_qwen.toml
```

Record both validation (pre-RL) and test (post-RL) accuracy to quantify improvements.

## 6. Dataset Notes

- By default the task app loads the [Hendrycks MATH benchmark](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark). Override via `MATH_DATASET_NAME` / `MATH_DATASET_CONFIG` env vars if you want a different variant. The dataset is public and automatically downloaded when the task app starts; the server will fail fast with a clear error if it cannot be fetched.
- For offline use, run `uv run python examples/rl/download_dataset.py --output-dir examples/rl/data --dataset nlile/hendrycks-MATH-benchmark --config algebra --limit 2000`. Then start the task app with `MATH_DATASET_LOCAL_DIR=examples/rl/data` (or set `MATH_DATASET_LOCAL_<SPLIT>_FILE`).
- Hugging Face downloads occur at runtime; pre-fetch locally or mount a Modal volume if you need offline access.
- Hugging Face downloads occur at runtime; pre-fetch locally or mount a Modal volume if you need offline access.
- Seeds map directly to dataset indices. Use `seed_start` to control determinism in configs and evaluations.

## 7. Additional Utilities

- `examples/rl/task_app/math_task_app.py` – legacy runner (`python .../math_task_app.py --reload`).
- `examples/rl/run_eval.py` – CLI evaluation helper (supports proxying Groq or hitting arbitrary inference URLs).
- `examples/rl/run_rl_and_save.py` – thin wrapper around the Synth `/rl/jobs` API.

For broader background on Synth task apps, CLI commands, and tracing, see the new documentation under `docs/`.



uv run python examples/rl/run_eval.py --toml examples/rl/configs/eval_base_qwen.toml
uvx synth-ai serve math-single-step \
    --port 8101 \
    --env-file examples/rl/.env \
    --trace traces/math \
    --force
