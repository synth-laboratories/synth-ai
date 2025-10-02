# Warming Up to RL - Examples

This folder contains a minimal Crafter-like task app and scripts for baseline evals, finetuning, RL submission, and evaluations.

## Prereqs
- Python 3.11+
- `.env` populated at `examples/warming_up_to_rl/.env` (copy from your research repo)
- Modal CLI configured if deploying the task app

## Deploy the task app to Modal

```
cd examples/warming_up_to_rl/task_app
modal deploy grpo_crafter_task_app.py
```

App name uses suffix `_warming_up_ex`. Ensure your Modal secret(s) include:
- `ENVIRONMENT_API_KEY` (required)
- `OPENAI_API_KEY` (optional; for OpenAI proxy)
- `GROQ_API_KEY` (optional; for Groq proxy)

## Baseline evaluations (TOML-driven)

- Groq Qwen3-32B (server-side rollout):
```
TASK_APP_URL=https://your-app.modal.run \
uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_groq_qwen32b.toml --use-rollout
```

- Synth vLLM Qwen/Qwen3-4B (server-side rollout; explicit inference_url in TOML):
```
TASK_APP_URL=https://your-app.modal.run \
uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_modal_qwen4b.toml --use-rollout
```

- Finetuned Qwen/Qwen3-4B (FFT) via Modal Learning v2 (server-side rollout):
```
TASK_APP_URL=https://your-app.modal.run \
uv run python examples/warming_up_to_rl/run_eval.py --toml examples/warming_up_to_rl/configs/eval_fft_qwen4b.toml --use-rollout
```

The TOML controls:
- `task_app_url`, `model`
- `inference_url`, `max_tokens`, `thinking_mode`, `thinking_budget`
- `num_episodes`, `max_turns`, `concurrency`, and optional `difficulty`

## Finetuning (FFT)

Submit full finetune job (saves model id to `ft_model_id.txt`):
```
export BACKEND_BASE_URL=... SYNTH_API_KEY=...
python examples/warming_up_to_rl/run_fft_and_save.py \
  --toml examples/warming_up_to_rl/configs/crafter_fft.toml \
  --data /absolute/path/to/your_sft.jsonl
```

## RL (TOML-only model selection)

- RL from finetuned model id (set `[model].source = "ft:..."` in TOML):
```
export BACKEND_BASE_URL=... SYNTH_API_KEY=... TASK_APP_URL=https://your-app.modal.run
python examples/warming_up_to_rl/run_rl_and_save.py \
  --config examples/warming_up_to_rl/configs/rl_from_ft.toml
```

- RL from base Qwen/Qwen3-4B (set `[model].base = "Qwen/Qwen3-4B"` in TOML):
```
export BACKEND_BASE_URL=... SYNTH_API_KEY=... TASK_APP_URL=https://your-app.modal.run
python examples/warming_up_to_rl/run_rl_and_save.py \
  --config examples/warming_up_to_rl/configs/rl_from_base_qwen4b.toml
```

See `experiment_plan.txt` for detailed specs and acceptance criteria.





uv run /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/run_rl_and_save.py \
  --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/configs/rl_from_base_qwen4b.toml


BACKEND_BASE_URL=https://synth-backend-dev-docker.onrender.com/api SYNTH_API_KEY=$(grep -E "^SYNTH_API_KEY=" /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/.env | head -n1 | cut -d= -f2- | tr -d '"') uv run python examples/warming_up_to_rl/run_fft_and_save.py --toml examples/warming_up_to_rl/configs/crafter_fft_4b.toml --data /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/ft_data/qwen3_32b_ach_ge3_raw_filtered.tokens_1000000_seed_123.jsonl --poll-seconds 3600

BACKEND_BASE_URL=http://localhost:8000/api \
SYNTH_API_KEY=$(grep -E "^SYNTH_API_KEY=" /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/.env | head -n1 | cut -d= -f2- | tr -d '"') \
uv run python /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/run_fft_and_save.py \
  --toml /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/configs/crafter_fft_4b.toml \
  --data /Users/joshpurtell/Documents/GitHub/synth-ai/examples/warming_up_to_rl/ft_data/qwen3_32b_ach_ge3_raw_filtered.tokens_1000000_seed_123.jsonl \
  --poll-seconds 3600